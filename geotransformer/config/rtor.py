"""Base configuration for the fixed GeoTransformer RTOR+A3 model."""

import copy
import os.path as osp

from easydict import EasyDict as edict


_C = edict()
_C.seed = 7351
_C.ablation = edict(
    rtor_enabled=True,
    a3_enabled=True,
    architecture='rtor_a3',
    topology_attention=False,
    overlap_cross_attention=False,
    legacy_rtor_post_refine=True,
    overlap_supervision_enabled=True,
    coarse_ranking_enabled=False,
    coarse_overlap_prior_enabled=False,
    coarse_topology_compatibility_enabled=False,
    ranking_loss_enabled=False,
)

_C.working_dir = osp.dirname(osp.realpath(__file__))
_C.root_dir = osp.dirname(osp.dirname(_C.working_dir))
_C.exp_name = osp.basename(_C.working_dir)
_C.output_dir = osp.join(_C.root_dir, 'output', _C.exp_name)
_C.snapshot_dir = osp.join(_C.output_dir, 'snapshots')
_C.log_dir = osp.join(_C.output_dir, 'logs')
_C.event_dir = osp.join(_C.output_dir, 'events')
_C.feature_dir = osp.join(_C.output_dir, 'features')
_C.registration_dir = osp.join(_C.output_dir, 'registration')

_C.data = edict()

_C.train = edict()
_C.train.batch_size = 1
_C.train.num_workers = 0
_C.train.point_limit = None
_C.train.use_augmentation = False
_C.train.augmentation_noise = 0.005
_C.train.augmentation_rotation = 1.0

_C.test = edict()
_C.test.batch_size = 1
_C.test.num_workers = 0
_C.test.point_limit = None

_C.eval = edict()
_C.eval.acceptance_overlap = 0.0
_C.eval.acceptance_radius = 0.1
_C.eval.inlier_ratio_threshold = 0.05
_C.eval.rmse_threshold = 0.2
_C.eval.rre_threshold = 15.0
_C.eval.rte_threshold = 0.3

_C.optim = edict()
_C.optim.lr = 1e-4
_C.optim.lr_decay = 0.95
_C.optim.lr_decay_steps = 1
_C.optim.weight_decay = 1e-6
_C.optim.max_epoch = 150
_C.optim.grad_acc_steps = 1

_C.backbone = edict()
_C.backbone.num_stages = 4
_C.backbone.init_voxel_size = 0.025
_C.backbone.kernel_size = 15
_C.backbone.base_radius = 2.5
_C.backbone.base_sigma = 2.0
_C.backbone.init_radius = _C.backbone.base_radius * _C.backbone.init_voxel_size
_C.backbone.init_sigma = _C.backbone.base_sigma * _C.backbone.init_voxel_size
_C.backbone.group_norm = 32
_C.backbone.input_dim = 1
_C.backbone.init_dim = 64
_C.backbone.output_dim = 256

_C.model = edict()
_C.model.dual_encoder = False
_C.model.ground_truth_matching_radius = 0.05
_C.model.num_points_in_patch = 64
_C.model.num_sinkhorn_iterations = 100

_C.coarse_matching = edict()
_C.coarse_matching.num_targets = 128
_C.coarse_matching.overlap_threshold = 0.1
_C.coarse_matching.num_correspondences = 256
_C.coarse_matching.dual_normalization = True

_C.coarse_ranking = edict()
_C.coarse_ranking.hidden_dim = 64
_C.coarse_ranking.margin = 0.1
_C.coarse_ranking.temperature = 0.1
_C.coarse_ranking.boundary_window = 16
_C.coarse_ranking.weight_loss = 0.5

_C.geotransformer = edict()
_C.geotransformer.input_dim = 1024
_C.geotransformer.hidden_dim = 256
_C.geotransformer.output_dim = 256
_C.geotransformer.num_heads = 4
_C.geotransformer.blocks = ['self', 'cross', 'self', 'cross', 'self', 'cross']
_C.geotransformer.sigma_d = 0.2
_C.geotransformer.sigma_a = 15
_C.geotransformer.angle_k = 3
_C.geotransformer.reduction_a = 'max'

_C.topology_overlap = edict()
_C.topology_overlap.hidden_dim = 128
_C.topology_overlap.num_neighbors = 8
_C.topology_overlap.poincare_curvature = 1.0
_C.topology_overlap.dropout = 0.1
_C.topology_overlap.score_floor = 0.05
_C.topology_overlap.score_power = 1.0
_C.topology_overlap.positive_overlap = 0.1
_C.topology_overlap.focal_gamma = 2.0
_C.topology_overlap.weight_loss = 0.5
_C.topology_overlap.attention_hidden_dim = 64

_C.overlap_selection = edict()
_C.overlap_selection.enabled = True
_C.overlap_selection.threshold = 0.5
_C.overlap_selection.min_superpoints = 16
_C.overlap_selection.max_ratio = 0.5
_C.overlap_selection.min_spread = 0.05

_C.fine_refiner = edict()
_C.fine_refiner.num_heads = 4
_C.fine_refiner.dropout = 0.0
_C.fine_refiner.geometry_sigma = 0.25
_C.fine_refiner.geometry_weight = 1.0
_C.fine_refiner.residual_init = 0.0

_C.fine_matching = edict()
_C.fine_matching.topk = 3
_C.fine_matching.acceptance_radius = 0.1
_C.fine_matching.mutual = True
_C.fine_matching.confidence_threshold = 0.05
_C.fine_matching.use_dustbin = False
_C.fine_matching.use_global_score = True
_C.fine_matching.correspondence_threshold = 3
_C.fine_matching.correspondence_limit = None
_C.fine_matching.num_refinement_steps = 5
_C.fine_matching.robust_refinement_radius = None

_C.coarse_loss = edict()
_C.coarse_loss.positive_margin = 0.1
_C.coarse_loss.negative_margin = 1.4
_C.coarse_loss.positive_optimal = 0.1
_C.coarse_loss.negative_optimal = 1.4
_C.coarse_loss.log_scale = 24
_C.coarse_loss.positive_overlap = 0.1

_C.fine_loss = edict()
_C.fine_loss.positive_radius = 0.05

_C.loss = edict()
_C.loss.weight_coarse_loss = 1.0
_C.loss.weight_fine_loss = 1.0


def _validate(cfg) -> None:
    if cfg.topology_overlap.hidden_dim <= 0:
        raise ValueError('topology_overlap.hidden_dim must be positive')
    if cfg.topology_overlap.num_neighbors <= 0:
        raise ValueError('topology_overlap.num_neighbors must be positive')
    if not 0.0 <= cfg.topology_overlap.score_floor < 1.0:
        raise ValueError('topology_overlap.score_floor must be in [0, 1)')
    if cfg.topology_overlap.score_power <= 0:
        raise ValueError('topology_overlap.score_power must be positive')
    if not 0.0 <= cfg.overlap_selection.threshold <= 1.0:
        raise ValueError('overlap_selection.threshold must be in [0, 1]')
    if cfg.overlap_selection.min_superpoints <= 0:
        raise ValueError('overlap_selection.min_superpoints must be positive')
    if not 0.0 < cfg.overlap_selection.max_ratio <= 1.0:
        raise ValueError('overlap_selection.max_ratio must be in (0, 1]')
    if not 0.0 <= cfg.overlap_selection.min_spread <= 1.0:
        raise ValueError('overlap_selection.min_spread must be in [0, 1]')
    if cfg.fine_refiner.geometry_sigma <= 0:
        raise ValueError('fine_refiner.geometry_sigma must be positive')
    if cfg.coarse_ranking.hidden_dim <= 0:
        raise ValueError('coarse_ranking.hidden_dim must be positive')
    if cfg.coarse_ranking.temperature <= 0:
        raise ValueError('coarse_ranking.temperature must be positive')
    if cfg.coarse_ranking.boundary_window <= 0:
        raise ValueError('coarse_ranking.boundary_window must be positive')
    if cfg.coarse_ranking.weight_loss < 0:
        raise ValueError('coarse_ranking.weight_loss must be non-negative')


def make_cfg():
    """Return an isolated RTOR+A3 base configuration."""
    cfg = copy.deepcopy(_C)
    _validate(cfg)
    return cfg
