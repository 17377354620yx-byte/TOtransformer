import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

from geotransformer.modules.geotransformer import SuperPointMatching
from geotransformer.modules.liver.coarse_ranker import (
    TopologyOverlapCoarseRanker,
)
from geotransformer.modules.ops import pairwise_distance


ROOT = Path(__file__).resolve().parents[1]


def _load_experiment_module(name, filename):
    path = ROOT / 'experiments' / 'geotransformer.p2p_liver' / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ranker(**overrides):
    options = dict(
        feature_dim=8,
        hidden_dim=12,
        num_neighbors=3,
        dual_normalization=True,
        overlap_score_floor=0.05,
        overlap_score_power=1.0,
    )
    options.update(overrides)
    return TopologyOverlapCoarseRanker(**options)


def test_zero_topology_head_exactly_reproduces_phase2_soft_ranking():
    torch.manual_seed(59)
    ranker = _ranker().eval()
    ref_features = F.normalize(torch.randn(7, 8), dim=1)
    src_features = F.normalize(torch.randn(9, 8), dim=1)
    ref_probability = torch.rand(7)
    src_probability = torch.rand(9)
    scores, diagnostics = ranker(
        torch.randn(7, 3),
        torch.randn(9, 3),
        ref_features,
        src_features,
        ref_probability,
        src_probability,
    )

    base = torch.exp(
        -pairwise_distance(ref_features, src_features, normalized=True)
    )
    base = (
        base / base.sum(dim=1, keepdim=True)
        * base / base.sum(dim=0, keepdim=True)
    )
    ref_weight = 0.05 + 0.95 * ref_probability
    src_weight = 0.05 + 0.95 * src_probability
    expected = SuperPointMatching._calibrate_scores(
        base, ref_weight, src_weight
    )
    torch.testing.assert_close(scores, expected, atol=1e-7, rtol=2e-6)
    torch.testing.assert_close(
        diagnostics['coarse_topology_compatibility'],
        torch.zeros_like(scores),
    )


def test_precomputed_scores_control_topk_without_filtering_nodes():
    matcher = SuperPointMatching(num_correspondences=3, dual_normalization=True)
    ref_features = F.normalize(torch.randn(2, 4), dim=1)
    src_features = F.normalize(torch.randn(3, 4), dim=1)
    scores = torch.tensor([[0.1, 0.2, 0.9], [0.8, 0.3, 0.4]])
    ref_indices, src_indices, selected = matcher(
        ref_features, src_features, precomputed_scores=scores
    )
    pairs = set(zip(ref_indices.tolist(), src_indices.tolist()))
    assert pairs == {(0, 2), (1, 0), (1, 2)}
    torch.testing.assert_close(selected, torch.tensor([0.9, 0.8, 0.4]))


def test_budget_ranking_loss_penalizes_missed_positive_and_backpropagates():
    config = _load_experiment_module('ranking_cfg_test', 'config.py')
    losses = _load_experiment_module('ranking_loss_test', 'loss.py')
    cfg = config.make_cfg(
        architecture='rtor_a3', interaction_profile='togg_full'
    )
    cfg.coarse_matching.num_correspondences = 2
    cfg.coarse_ranking.boundary_window = 1
    criterion = losses.CandidateRankingLoss(cfg)

    logits = torch.tensor(
        [[-2.0, 2.0, 1.5], [1.0, 0.5, 0.0]], requires_grad=True
    )
    output = {
        'coarse_ranking_logits': logits,
        'coarse_valid_pair_masks': torch.ones(2, 3, dtype=torch.bool),
        'gt_node_corr_indices': torch.tensor([[0, 0]]),
        'gt_node_corr_overlaps': torch.tensor([0.8]),
    }
    missed_loss = criterion(output)
    promoted_logits = logits.detach().clone()
    promoted_logits[0, 0] = 3.0
    output['coarse_ranking_logits'] = promoted_logits
    promoted_loss = criterion(output)
    assert missed_loss > promoted_loss
    missed_loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0] < 0


def test_full_profile_enables_phase3_and_phase4_independently():
    config = _load_experiment_module('ranking_profile_test', 'config.py')
    phase3 = config.make_cfg(
        architecture='rtor_a3', interaction_profile='togg_phase3'
    )
    full = config.make_cfg(
        architecture='rtor_a3', interaction_profile='togg_full'
    )
    assert phase3.ablation.coarse_ranking_enabled
    assert not phase3.ablation.ranking_loss_enabled
    assert full.ablation.coarse_ranking_enabled
    assert full.ablation.coarse_overlap_prior_enabled
    assert full.ablation.coarse_topology_compatibility_enabled
    assert full.ablation.ranking_loss_enabled
    assert not full.overlap_selection.enabled


def test_candidate_recall_and_mrr_use_all_gt_pairs_not_any_hit():
    config = _load_experiment_module('ranking_metric_cfg_test', 'config.py')
    losses = _load_experiment_module('ranking_metric_loss_test', 'loss.py')
    cfg = config.make_cfg(
        architecture='rtor_a3', interaction_profile='togg_full'
    )
    evaluator = losses.Evaluator(cfg)
    output = {
        'ref_points_c': torch.randn(2, 3),
        'src_points_c': torch.randn(2, 3),
        'gt_node_corr_indices': torch.tensor([[0, 0], [0, 1], [1, 1]]),
        'gt_node_corr_overlaps': torch.tensor([0.8, 0.7, 0.9]),
        'ref_node_corr_indices': torch.tensor([0, 1]),
        'src_node_corr_indices': torch.tensor([0, 1]),
        'coarse_ranking_logits': torch.tensor([[4.0, 3.0], [2.0, 1.0]]),
        'coarse_valid_pair_masks': torch.ones(2, 2, dtype=torch.bool),
    }
    torch.testing.assert_close(
        evaluator.evaluate_candidate_recall(output), torch.tensor(2.0 / 3.0)
    )
    # GT ranks are 1, 2 and 4: mean reciprocal rank = 7/12.
    torch.testing.assert_close(
        evaluator.evaluate_gt_mrr(output), torch.tensor(7.0 / 12.0)
    )
