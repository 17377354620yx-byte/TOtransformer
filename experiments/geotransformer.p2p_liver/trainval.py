"""Train the RTOR+A3 model with the P2P liver protocol.

For real DePoLL evaluation of these checkpoints, use test_depoll.py or
scripts/test_depoll.sh; see DEPOLL_EVALUATION.md. Training validation RMSE
is not DePoLL's millimetre clips/balls TRE.
"""

import argparse
import json
import os
import os.path as osp
import sys
import time
from collections import OrderedDict


_ROOT_DIR = osp.realpath(osp.join(osp.dirname(__file__), "..", ".."))
if _ROOT_DIR in sys.path:
    sys.path.remove(_ROOT_DIR)
sys.path.insert(0, _ROOT_DIR)

import torch
import torch.optim as optim

from geotransformer.engine import EpochBasedTrainer

from config import ABLATION_PROFILES, make_cfg
from dataset import train_valid_data_loader
from loss import Evaluator, OverallLoss
from model import create_model
from geotransformer.modules.liver.cooperative_matching import predicted_ratio_at_epoch


class Trainer(EpochBasedTrainer):
    def __init__(self, cfg, parser=None):
        super().__init__(cfg, max_epoch=cfg.optim.max_epoch, parser=parser)
        self.logger.info(f"Architecture: {cfg.ablation.architecture}")
        self.cfg = cfg
        self.best_validation_error = float('inf')
        start = time.time()
        train_loader, val_loader, limits = train_valid_data_loader(cfg, self.distributed)
        self.logger.info(f"Data loader created: {time.time() - start:.3f}s collapsed.")
        self.logger.info(f"Calibrate neighbors: {limits}.")
        protocol = dict(architecture=cfg.ablation.architecture,
                        dual_encoder=cfg.model.dual_encoder,
                        registration_profile=cfg.model.registration_profile,
                        interaction_profile=cfg.model.interaction_profile,
                        ablation_profile=cfg.ablation.profile,
                        ablation_switches={
                            key: bool(cfg.ablation[key])
                            for key in (
                                'predicted_proposal_exposure',
                                'overlap_soft_weight',
                                'rtor_poincare',
                                'a3_geometry_bias',
                                'rtor_descriptor_update',
                                'topology_attention',
                                'overlap_cross_attention',
                                'legacy_rtor_post_refine',
                                'overlap_supervision_enabled',
                                'coarse_ranking_enabled',
                                'coarse_overlap_prior_enabled',
                                'coarse_topology_compatibility_enabled',
                                'ranking_loss_enabled',
                            )
                        },
                        neighbor_limits=[int(x) for x in limits], seed=int(cfg.seed),
                        validation='disjoint deformation groups')
        self.save_state('p2p_protocol', protocol)
        with open(osp.join(cfg.output_dir, 'run_manifest.json'), 'w') as handle:
            json.dump(dict(protocol=protocol, command=sys.argv, config=cfg), handle, indent=2)
        self.register_loader(train_loader, val_loader)
        model = self.register_model(create_model(cfg).cuda())
        optimizer = optim.Adam(
            model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
        )
        self.register_optimizer(optimizer)
        self.register_scheduler(
            optim.lr_scheduler.StepLR(
                optimizer, cfg.optim.lr_decay_steps, gamma=cfg.optim.lr_decay
            )
        )
        self.loss_func = OverallLoss(cfg).cuda()
        self.evaluator = Evaluator(cfg).cuda()

    def load_snapshot(self, snapshot, fix_prefix=True):
        """Resume RTOR+A3 or warm-start only its model parameters."""
        state_dict = torch.load(snapshot, map_location=torch.device("cpu"))
        model_dict = state_dict["model"]
        if not self.args.warm_start:
            saved_profile = state_dict.get('metadata', {}).get('p2p_protocol', {}).get('interaction_profile')
            if saved_profile is not None and saved_profile != self.cfg.model.interaction_profile:
                raise ValueError('Resume must use the checkpoint interaction profile; use --warm_start for an intentional new run')
            saved_ablation = state_dict.get('metadata', {}).get(
                'p2p_protocol', {}
            ).get('ablation_profile', 'none')
            if saved_ablation != self.cfg.ablation.profile:
                raise ValueError(
                    'Resume must use the checkpoint ablation profile; '
                    'use --warm_start for an intentional new run'
                )
            self.best_validation_error = state_dict.get('metadata', {}).get('best_validation_error', float('inf'))
            strict_dict = (OrderedDict(('module.' + k, v) for k, v in model_dict.items())
                           if fix_prefix and self.distributed else model_dict)
            self.model.load_state_dict(strict_dict, strict=True)
            return super().load_snapshot(snapshot, fix_prefix=fix_prefix)

        self.logger.info(
            f'Warm-starting model weights only from "{snapshot}".'
        )
        if fix_prefix and self.distributed:
            model_dict = OrderedDict(
                [("module." + key, value) for key, value in model_dict.items()]
            )
        incompatibility = self.model.load_state_dict(model_dict, strict=False)
        allowed_prefixes = (
            'module.topology_overlap_refiner.',
            'module.fine_local_refiner.',
            'module.transformer.conditioner.',
            'module.coarse_ranker.',
        ) if self.distributed else (
            'topology_overlap_refiner.',
            'fine_local_refiner.',
            'transformer.conditioner.',
            'coarse_ranker.',
        )
        missing = list(incompatibility.missing_keys)
        required_missing = [
            key for key in missing if not key.startswith(allowed_prefixes)
        ]
        unexpected = list(incompatibility.unexpected_keys)
        legacy_prefix = (
            'module.topology_overlap_refiner.'
            if self.distributed else 'topology_overlap_refiner.'
        )
        required_unexpected = [
            key for key in unexpected
            if not (
                self.cfg.ablation.get('topology_attention', False)
                and key.startswith(legacy_prefix)
            )
        ]
        if required_unexpected:
            raise RuntimeError(
                f"Unexpected keys in warm start: {sorted(required_unexpected)}"
            )
        if required_missing:
            raise RuntimeError(
                f"Required keys missing from warm start: {sorted(required_missing)}"
            )
        self.logger.info(
            f"Loaded weights; initialized {len(missing)} RTOR/A3 tensors. "
            "Epoch, optimizer and scheduler start fresh."
        )

    def _step(self, data):
        output = self.model(data)
        result = self.loss_func(output, data)
        result.update(self.evaluator(output, data))
        return output, result

    def train_step(self, epoch, iteration, data):
        model = self.model.module if self.distributed else self.model
        model.predicted_coarse_ratio = predicted_ratio_at_epoch(
            epoch, self.cfg.coarse_matching.exposure_start_epoch,
            self.cfg.coarse_matching.exposure_end_epoch,
            self.cfg.coarse_matching.predicted_ratio_max)
        return self._step(data)

    def val_step(self, epoch, iteration, data):
        output, result = self._step(data)
        self.validation_errors.append(float(result['RMSE']))
        return output, result

    def before_val_epoch(self, epoch):
        self.validation_errors = []

    def after_val_epoch(self, epoch):
        if self.validation_errors:
            error = sum(self.validation_errors) / len(self.validation_errors)
            if error < self.best_validation_error:
                self.best_validation_error = error
                self.save_state('best_validation_error', error)
                self.save_snapshot('best.pth.tar')
                self.logger.info(f'Best training-side validation displacement: {error:.6f}, epoch {epoch}')


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--architecture', choices=['geotransformer', 'rtor_only', 'a3_only', 'rtor_a3'], default='rtor_a3')
    parser.add_argument('--registration_profile', choices=['legacy', 'tight', 'robust'], default='legacy')
    parser.add_argument('--dual_encoder', action='store_true')
    parser.add_argument(
        '--interaction_profile',
        choices=[
            'legacy', 'cooperative', 'soft_overlap',
            'togg_phase1', 'togg_phase2',
            'togg_phase3', 'togg_full',
        ],
        default='legacy',
    )
    parser.add_argument(
        '--ablation_profile',
        choices=['none', *ABLATION_PROFILES],
        default='none',
    )
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--validation_size", type=int, default=None)
    parser.add_argument("--max_epoch", type=int, default=None)
    parser.add_argument(
        "--warm_start",
        action="store_true",
        help="Load --snapshot model weights only; start epoch/optimizer fresh.",
    )
    known, _ = parser.parse_known_args()
    if known.train_limit > 0:
        os.environ["P2P_TRAIN_LIMIT"] = str(known.train_limit)
    cfg = make_cfg(
        known.architecture,
        known.registration_profile,
        known.dual_encoder,
        known.interaction_profile,
        known.ablation_profile,
    )
    if known.lr is not None:
        if known.lr <= 0:
            parser.error('--lr must be positive')
        cfg.optim.lr = known.lr
    if known.validation_size is not None:
        if known.validation_size <= 0:
            parser.error("--validation_size must be positive")
        cfg.data.validation_size = known.validation_size
    if known.max_epoch is not None:
        if known.max_epoch <= 0:
            parser.error("--max_epoch must be positive")
        cfg.optim.max_epoch = known.max_epoch
    Trainer(cfg, parser=parser).run()


if __name__ == "__main__":
    main()
