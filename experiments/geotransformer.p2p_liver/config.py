"""Configuration for rigid complete-to-partial liver registration."""

import copy
import os
import os.path as osp
import sys


_ROOT_DIR = osp.realpath(osp.join(osp.dirname(__file__), '..', '..'))
if _ROOT_DIR in sys.path:
    sys.path.remove(_ROOT_DIR)
sys.path.insert(0, _ROOT_DIR)

from geotransformer.config import make_rtor_a3_cfg
from geotransformer.utils.common import ensure_dir

_C = make_rtor_a3_cfg()
_C.working_dir = osp.dirname(osp.realpath(__file__))
_C.root_dir = _ROOT_DIR
_RUN_NAME = os.environ.get('P2P_RUN_NAME', 'rtor_a3').strip()
if not _RUN_NAME:
    raise ValueError('P2P_RUN_NAME must not be empty')
_C.exp_name = f'geotransformer.p2p_liver.{_RUN_NAME}'
_C.output_dir = osp.join(_C.root_dir, 'output', _C.exp_name)
_C.snapshot_dir = osp.join(_C.output_dir, 'snapshots')
_C.log_dir = osp.join(_C.output_dir, 'logs')
_C.event_dir = osp.join(_C.output_dir, 'events')
_C.feature_dir = osp.join(_C.output_dir, 'features')
_C.registration_dir = osp.join(_C.output_dir, 'registration')

_DATA_ROOT = os.environ.get('P2P_DATA_ROOT', '/mnt/data3/yangx/P2P/Dataset')
_C.data.train_root = osp.join(_DATA_ROOT, 'Deform_mesh_npz')
_C.data.train_list = osp.join(_C.data.train_root, 'dict.json')
_C.data.test_root = osp.join(_DATA_ROOT, 'Deform_mesh_npz_test', 'Test')
_C.data.test_list = osp.join(_DATA_ROOT, 'Deform_mesh_npz_test', 'list.npz')
_C.data.statistics = osp.join(
    _DATA_ROOT,
    'Deform_mesh_npz_test',
    'stat_svd.npz',
)
_C.data.in_vitro_root = os.environ.get(
    'P2P_IN_VITRO_ROOT',
    '/mnt/data3/yangx/P2P/in_vitro',
)
_C.data.in_vitro_list = osp.join(_C.data.in_vitro_root, 'rigid_list.npy')
_C.data.in_vitro_statistics = osp.join(_C.data.in_vitro_root, 'stat.npz')
_C.data.voxel_size = 0.04
_C.data.min_visibility = 0.18
_C.data.max_visibility = 1.0
_C.data.max_noise_mm = 5.0
_C.data.validation_size = 100
_C.data.validation_fraction = 0.1
_C.data.neighbor_calibration_samples = 2000
_C.data.neighbor_limits = None

_C.backbone.init_voxel_size = 0.02
_C.backbone.init_radius = _C.backbone.base_radius * _C.backbone.init_voxel_size
_C.backbone.init_sigma = _C.backbone.base_sigma * _C.backbone.init_voxel_size
_C.model.ground_truth_matching_radius = 0.04
_C.fine_loss.positive_radius = 0.04


ABLATION_PROFILES = {
    'abl1_no_proposal': (False, True, True, True, True),
    'abl2_no_soft_weight': (False, False, True, True, True),
    'abl3_no_poincare': (False, False, False, True, True),
    'abl4_no_a3_geometry': (False, False, False, False, True),
    'abl5_no_rtor_descriptor': (False, False, False, False, False),
    # Compact candidate selected after the cumulative ablation study. Keep
    # overlap weighting and descriptor update; remove proposal exposure and
    # the two geometry branches.
    'compact_no_proposal_poincare_geometry': (False, True, False, False, True),
}


def _configure_ablation(cfg, ablation_profile):
    if ablation_profile == 'none':
        switches = (True, True, True, True, True)
    else:
        try:
            switches = ABLATION_PROFILES[ablation_profile]
        except KeyError as error:
            choices = ', '.join(('none', *ABLATION_PROFILES))
            raise ValueError(
                f'Unknown ablation profile: {ablation_profile}; choose from {choices}'
            ) from error
    cfg.ablation.profile = ablation_profile
    (
        cfg.ablation.predicted_proposal_exposure,
        cfg.ablation.overlap_soft_weight,
        cfg.ablation.rtor_poincare,
        cfg.ablation.a3_geometry_bias,
        cfg.ablation.rtor_descriptor_update,
    ) = switches


def make_cfg(
    architecture='rtor_a3',
    registration_profile='legacy',
    dual_encoder=False,
    interaction_profile='legacy',
    ablation_profile='none',
):
    """Return a fresh P2P RTOR+A3 configuration."""
    cfg = copy.deepcopy(_C)
    choices = {'geotransformer': (False, False), 'rtor_only': (True, False),
               'a3_only': (False, True), 'rtor_a3': (True, True)}
    if architecture not in choices:
        raise ValueError(f'Unknown architecture: {architecture}')
    cfg.ablation.architecture = architecture
    cfg.ablation.rtor_enabled, cfg.ablation.a3_enabled = choices[architecture]
    if registration_profile not in ('legacy', 'tight', 'robust'):
        raise ValueError(f'Unknown registration profile: {registration_profile}')
    cfg.model.registration_profile = registration_profile
    cfg.model.dual_encoder = bool(dual_encoder)
    if interaction_profile not in (
        'legacy',
        'cooperative',
        'soft_overlap',
        'togg_phase1',
        'togg_phase2',
        'togg_phase3',
        'togg_full',
    ):
        raise ValueError(f'Unknown interaction profile: {interaction_profile}')
    _configure_ablation(cfg, ablation_profile)
    if ablation_profile != 'none' and (
        architecture != 'rtor_a3'
        or interaction_profile != 'cooperative'
        or dual_encoder
        or registration_profile != 'legacy'
    ):
        raise ValueError(
            'Ablation profiles require single-encoder RTOR+A3 with '
            'interaction_profile=cooperative and registration_profile=legacy'
        )
    if interaction_profile == 'cooperative' and (architecture != 'rtor_a3' or dual_encoder or registration_profile != 'legacy'):
        raise ValueError('Cooperative profile uses single-encoder RTOR+A3 and the original LGR settings')
    cfg.model.interaction_profile = interaction_profile
    conditioned_profile = interaction_profile in (
        'togg_phase1', 'togg_phase2', 'togg_phase3', 'togg_full'
    )
    if conditioned_profile and not cfg.ablation.rtor_enabled:
        raise ValueError(
            'TOGGT profiles require architecture=rtor_only or rtor_a3 so the '
            'master RTOR switch controls all added behavior'
        )
    cfg.ablation.topology_attention = conditioned_profile
    cfg.ablation.overlap_cross_attention = interaction_profile in (
        'togg_phase2', 'togg_phase3', 'togg_full'
    )
    cfg.ablation.coarse_ranking_enabled = interaction_profile in (
        'togg_phase3', 'togg_full'
    )
    cfg.ablation.coarse_overlap_prior_enabled = (
        cfg.ablation.coarse_ranking_enabled
    )
    cfg.ablation.coarse_topology_compatibility_enabled = (
        cfg.ablation.coarse_ranking_enabled
    )
    cfg.ablation.ranking_loss_enabled = interaction_profile == 'togg_full'
    cfg.ablation.legacy_rtor_post_refine = (
        cfg.ablation.rtor_enabled and not conditioned_profile
    )
    cfg.ablation.overlap_supervision_enabled = (
        cfg.ablation.rtor_enabled and interaction_profile != 'togg_phase1'
    )
    cfg.coarse_matching.predicted_ratio_max = (
        .25
        if interaction_profile == 'cooperative'
        and cfg.ablation.predicted_proposal_exposure
        else 0.
    )
    cfg.coarse_matching.exposure_start_epoch = 5
    cfg.coarse_matching.exposure_end_epoch = 20
    # Keep historical loss weights by default: checkpoint probes did not show
    # sustained gradient conflict or an over-dominant overlap objective.
    if interaction_profile in (
        'cooperative', 'soft_overlap', 'togg_phase1', 'togg_phase2',
        'togg_phase3', 'togg_full'
    ):
        cfg.overlap_selection.enabled = False
    if registration_profile == 'tight':
        cfg.fine_matching.acceptance_radius = 0.06
    elif registration_profile == 'robust':
        cfg.fine_matching.robust_refinement_radius = 0.04
    default_name = architecture + ('_dual' if dual_encoder else '')
    if interaction_profile != 'legacy':
        default_name += '_' + interaction_profile
    if ablation_profile != 'none':
        default_name += '_' + ablation_profile
    if registration_profile != 'legacy':
        default_name += '_' + registration_profile
    run_name = os.environ.get('P2P_RUN_NAME', default_name).strip()
    if not run_name or '/' in run_name or '\\' in run_name or run_name in ('.', '..'):
        raise ValueError('P2P_RUN_NAME must be a nonempty directory name')
    cfg.exp_name = f'geotransformer.p2p_liver.{run_name}'
    cfg.output_dir = osp.join(cfg.root_dir, 'output', cfg.exp_name)
    for key, folder in (('snapshot_dir', 'snapshots'), ('log_dir', 'logs'),
                        ('event_dir', 'events'), ('feature_dir', 'features'),
                        ('registration_dir', 'registration')):
        cfg[key] = osp.join(cfg.output_dir, folder)
    for path in (
        cfg.output_dir,
        cfg.snapshot_dir,
        cfg.log_dir,
        cfg.event_dir,
        cfg.feature_dir,
        cfg.registration_dir,
    ):
        ensure_dir(path)
    return cfg


__all__ = ['ABLATION_PROFILES', 'make_cfg']
