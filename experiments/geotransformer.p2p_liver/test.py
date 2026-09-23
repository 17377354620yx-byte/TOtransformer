"""Evaluate the single-hypothesis GeoTransformer + RTOR liver model."""

import argparse
import csv
import json
import os
import os.path as osp
import sys
import time


_ROOT_DIR = osp.realpath(osp.join(osp.dirname(__file__), '..', '..'))
if _ROOT_DIR in sys.path:
    sys.path.remove(_ROOT_DIR)
sys.path.insert(0, _ROOT_DIR)

import numpy as np
import torch

from geotransformer.engine import SingleTester
from geotransformer.modules.ops.transformation import apply_transform
from geotransformer.modules.registration.metrics import isotropic_transform_error
from geotransformer.utils.common import get_log_string
from geotransformer.utils.torch import release_cuda

from config import ABLATION_PROFILES, make_cfg
from dataset import test_data_loader
from model import create_model
from loss import Evaluator


VISIBILITY_BINS = np.arange(0.2, 1.01, 0.1)


def make_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', choices=['in_silico', 'in_vitro'], default='in_silico')
    parser.add_argument('--noise', choices=['none', '2', '4'], default='none')
    parser.add_argument('--test_limit', type=int, default=0)
    parser.add_argument('--output', default=None)
    parser.add_argument('--save_predictions', action='store_true')
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
    return parser


def _visibility_summary(values, visibility):
    rows = []
    for index in range(8):
        lower, upper = (index + 2) / 10, (index + 3) / 10
        mask = (visibility >= lower) & ((visibility <= upper) if index == 7 else (visibility < upper))
        selected = values[mask]
        rows.append(
            {
                'range': f'[{lower:.1f}, {upper:.1f}' + (']' if index == 7 else ')'),
                'count': int(mask.sum()),
                'mean_rms_tre_mm': float(selected.mean()) if len(selected) else None,
                'std_rms_tre_mm': float(selected.std()) if len(selected) else None,
                'success_rate_20mm': float((selected < 20.0).mean()) if len(selected) else None,
                'success_rate_5mm': float((selected < 5.0).mean()) if len(selected) else None,
            }
        )
    return rows


def _method_summary(values, visibility, deformation):
    values = np.asarray(values)
    low_visibility = (visibility >= 0.2) & (visibility <= 0.3)
    deformation_rows = []
    for label, lower, upper, include_upper in (
        ('[0, 6)', 0.0, 6.0, False),
        ('[6, 12]', 6.0, 12.0, True),
    ):
        deform_mask = (deformation >= lower) & (
            (deformation <= upper) if include_upper else (deformation < upper)
        )
        selected = values[low_visibility & deform_mask]
        deformation_rows.append(
            {
                'range_mm': label,
                'count': int(len(selected)),
                'mean_rms_tre_mm': float(selected.mean()) if len(selected) else None,
                'std_rms_tre_mm': float(selected.std()) if len(selected) else None,
                'success_rate_20mm': float((selected < 20.0).mean()) if len(selected) else None,
                'success_rate_5mm': float((selected < 5.0).mean()) if len(selected) else None,
            }
        )
    return {
        'count': int(len(values)),
        'mean_rms_tre_mm': float(values.mean()),
        'std_rms_tre_mm': float(values.std()),
        'success_rate_20mm': float((values < 20.0).mean()),
        'success_rate_5mm': float((values < 5.0).mean()),
        'visibility_bins': _visibility_summary(values, visibility),
        'low_visibility_deformation_bins': deformation_rows,
    }


class Tester(SingleTester):
    def __init__(self, cfg, parser):
        super().__init__(cfg, parser=parser)
        self._checkpoint = torch.load(self.args.snapshot, map_location='cpu', weights_only=False)
        protocol = self._checkpoint.get('metadata', {}).get('p2p_protocol', {})
        if protocol.get('interaction_profile', cfg.model.interaction_profile) != cfg.model.interaction_profile:
            raise ValueError('Checkpoint interaction profile differs; supply its --interaction_profile explicitly')
        checkpoint_ablation = protocol.get('ablation_profile', 'none')
        if checkpoint_ablation != cfg.ablation.profile:
            raise ValueError(
                'Checkpoint ablation profile differs; supply its '
                '--ablation_profile explicitly'
            )
        if 'neighbor_limits' in protocol:
            cfg.data.neighbor_limits = protocol['neighbor_limits']
        noise_mm = None if self.args.noise == 'none' else int(self.args.noise)
        if self.args.dataset == 'in_vitro' and noise_mm is not None:
            parser.error('--dataset in_vitro only supports --noise none')
        start = time.time()
        loader, limits = test_data_loader(
            cfg, noise_mm=noise_mm, dataset_name=self.args.dataset
        )
        self.logger.info(f'Data loader created: {time.time() - start:.3f}s collapsed.')
        self.logger.info(f'Calibrate neighbors: {limits}.')
        self.register_loader(loader)
        self.register_model(create_model(cfg).cuda())
        self.cfg = cfg
        self.evaluator = Evaluator(cfg).cuda()
        self.records = []
        self.tre_values = []
        self.output_path = self.args.output or osp.join(
            cfg.output_dir,
            f'test_{self.args.dataset}_noise_{self.args.noise}_summary.json',
        )
        statistics_path = (
            cfg.data.in_vitro_statistics
            if self.args.dataset == 'in_vitro'
            else cfg.data.statistics
        )
        statistics = np.load(statistics_path)
        self.visibility = statistics['vis']
        self.deformation = statistics['deform']

    def load_snapshot(self, snapshot):
        """Strictly load an RTOR+A3 checkpoint."""
        self.logger.info(f'Loading from "{snapshot}".')
        state_dict = self._checkpoint
        self.model.load_state_dict(state_dict['model'], strict=True)
        del self._checkpoint
        self.logger.info(f'Checkpoint architecture: {self.cfg.ablation.architecture} (strict match).')

    def test_step(self, iteration, data):
        return self.model(data)

    def eval_step(self, iteration, data, output):
        predicted_markers = apply_transform(
            data['source_markers'], output['estimated_transform']
        )
        distances = torch.linalg.norm(
            predicted_markers - data['target_markers'], dim=1
        )
        rms_tre = torch.sqrt(torch.mean(distances.square())) * float(
            data['physical_scale']
        )
        rre, rte = isotropic_transform_error(
            data['transform'], output['estimated_transform']
        )
        predicted_pairs = torch.stack(
            [output['ref_node_corr_indices'], output['src_node_corr_indices']], dim=1
        )
        gt_pairs = output['gt_node_corr_indices'][
            output['gt_node_corr_overlaps'] > self.cfg.coarse_matching.overlap_threshold
        ]
        if predicted_pairs.numel() and gt_pairs.numel():
            pair_hits = (
                predicted_pairs[:, None, :] == gt_pairs[None, :, :]
            ).all(dim=2)
            coarse_candidate_correct = pair_hits.any()
        else:
            coarse_candidate_correct = rms_tre.new_tensor(False, dtype=torch.bool)
        result = self.evaluator(output, data)
        return {
            **result,
            'RMS_TRE_mm': rms_tre,
            'SR_20mm': (rms_tre < 20.0).float(),
            'SR_5mm': (rms_tre < 5.0).float(),
            'RRE': rre,
            'RTE': rte,
            'coarse_candidate_correct': coarse_candidate_correct.float(),
            'lgr_failed_given_correct_coarse': (
                coarse_candidate_correct & (rms_tre >= 20.0)
            ).float(),
        }

    def summary_string(self, iteration, data, output, result):
        focused_ref = int(output['num_focused_ref_nodes'])
        focused_src = int(output['num_focused_src_nodes'])
        valid_ref = int(output['ref_node_masks'].sum())
        valid_src = int(output['src_node_masks'].sum())
        return (
            f"{data['sample_name']}, "
            + get_log_string(result_dict=result)
            + f", nCorr: {output['corr_scores'].shape[0]}"
            + f", focus: {focused_ref}/{valid_ref}|{focused_src}/{valid_src}"
        )

    def after_test_step(self, iteration, data, output, result):
        index = int(data['index'])
        rms_tre = float(result['RMS_TRE_mm'].detach().cpu())
        focused_ref = int(output['num_focused_ref_nodes'])
        focused_src = int(output['num_focused_src_nodes'])
        valid_ref = int(output['ref_node_masks'].sum())
        valid_src = int(output['src_node_masks'].sum())
        self.tre_values.append(rms_tre)
        self.records.append(
            {
                'index': index,
                'sample': data['sample_name'],
                'visibility': float(self.visibility[index]),
                'deformation_mm': float(self.deformation[index]),
                'rms_tre_mm': rms_tre,
                'success_20mm': int(rms_tre < 20.0),
                'success_5mm': int(rms_tre < 5.0),
                'rre_deg': float(result['RRE'].detach().cpu()),
                'rte': float(result['RTE'].detach().cpu()),
                'pir': float(result['PIR']),
                'candidate_recall_at_k': float(result['CR@K']),
                'gt_mrr': (
                    float(result['GT_MRR']) if 'GT_MRR' in result else None
                ),
                'ir': float(result['IR']),
                'mean_point_displacement': float(result['RMSE']),
                'rr': float(result['RR']),
                'coarse_candidate_correct': int(result['coarse_candidate_correct']),
                'lgr_failed_given_correct_coarse': int(
                    result['lgr_failed_given_correct_coarse']
                ),
                'fine_corr_count': int(output['corr_scores'].numel()),
                'focused_ref_nodes': focused_ref,
                'valid_ref_nodes': valid_ref,
                'focused_src_nodes': focused_src,
                'valid_src_nodes': valid_src,
            }
        )
        if self.args.save_predictions:
            os.makedirs(self.cfg.registration_dir, exist_ok=True)
            np.savez_compressed(
                osp.join(self.cfg.registration_dir, f'{index:05d}.npz'),
                sample=data['sample_name'],
                estimated_transform=release_cuda(output['estimated_transform']),
                rms_tre_mm=rms_tre,
            )

    def after_test_epoch(self):
        if not self.records:
            return
        indices = np.asarray([row['index'] for row in self.records], dtype=np.int64)
        values = np.asarray(self.tre_values)
        summary = _method_summary(
            values, self.visibility[indices], self.deformation[indices]
        )
        summary.update(
            {
                'mean_rre_deg': float(np.mean([row['rre_deg'] for row in self.records])),
                'mean_rte': float(np.mean([row['rte'] for row in self.records])),
                'rr': float(np.mean([row['rr'] for row in self.records])),
                'mean_pir': float(np.mean([row['pir'] for row in self.records])),
                'mean_candidate_recall_at_k': float(np.mean([
                    row['candidate_recall_at_k'] for row in self.records
                ])),
                'mean_ir': float(np.mean([row['ir'] for row in self.records])),
                'mean_point_displacement': float(np.mean([row['mean_point_displacement'] for row in self.records])),
                'coarse_candidate_recall': float(
                    np.mean([row['coarse_candidate_correct'] for row in self.records])
                ),
                'lgr_failure_given_correct_coarse_count': int(
                    np.sum([
                        row['lgr_failed_given_correct_coarse'] for row in self.records
                    ])
                ),
                'coarse_wrong_failure_count': int(
                    np.sum([
                        (not row['success_20mm']) and (not row['coarse_candidate_correct'])
                        for row in self.records
                    ])
                ),
            }
        )
        gt_mrr_values = [
            row['gt_mrr'] for row in self.records if row['gt_mrr'] is not None
        ]
        if gt_mrr_values:
            summary['mean_gt_mrr'] = float(np.mean(gt_mrr_values))
        payload = {
            'metric_schema_version': 4,
            'metric_note': 'RR uses mean normalized source displacement < eval.rmse_threshold; SR@5mm and SR@20mm use volumetric RMS-TRE. coarse_candidate_recall is the legacy any-hit rate; mean_candidate_recall_at_k is GT-pair recall within the fixed Top-K budget.',
            'protocol': f"P2P paper {self.args.dataset.replace('_', '-')} rigid registration",
            'metric': 'RMS-TRE on paired volumetric fiducials, millimetres',
            'checkpoint': osp.abspath(self.args.snapshot),
            'architecture': self.cfg.ablation.architecture,
            'dual_encoder': self.cfg.model.dual_encoder,
            'registration_profile': self.cfg.model.registration_profile,
            'interaction_profile': self.cfg.model.interaction_profile,
            'ablation_profile': self.cfg.ablation.profile,
            'ablation_switches': {
                key: bool(self.cfg.ablation[key])
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
            'fine_matching': dict(self.cfg.fine_matching),
            'noise_mm': None if self.args.noise == 'none' else int(self.args.noise),
            'rtor': {
                'num_neighbors': int(self.cfg.topology_overlap.num_neighbors),
                'score_floor': float(self.cfg.topology_overlap.score_floor),
                'focus': 'complete source only',
                'focus_threshold': float(self.cfg.overlap_selection.threshold),
                'focus_max_ratio': float(self.cfg.overlap_selection.max_ratio),
                'fine_refiner': dict(self.cfg.fine_refiner),
            },
            'evaluated_samples': int(len(values)),
            'method': summary,
            'samples': self.records,
        }
        output_path = osp.abspath(self.output_path)
        os.makedirs(osp.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        csv_path = osp.splitext(output_path)[0] + '.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.records[0]))
            writer.writeheader()
            writer.writerows(self.records)
        np.save(osp.splitext(output_path)[0] + '_rms_tre_mm.npy', values)
        self.logger.critical(
            f'RTOR metrics: RMS-TRE={values.mean():.4f}+/-{values.std():.4f} mm, '
            f'SR@5mm={(values < 5).mean():.4f}, '
            f'SR@20mm={(values < 20).mean():.4f}'
        )
        self.logger.info(f'Results written to {output_path} and {csv_path}')


def main():
    parser = make_parser()
    known, _ = parser.parse_known_args()
    if known.test_limit > 0:
        os.environ['P2P_TEST_LIMIT'] = str(known.test_limit)
    cfg = make_cfg(
        known.architecture,
        known.registration_profile,
        known.dual_encoder,
        known.interaction_profile,
        known.ablation_profile,
    )
    Tester(cfg, parser).run()


if __name__ == '__main__':
    main()
