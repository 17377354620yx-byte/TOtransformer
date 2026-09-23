"""Training losses and validation metrics for RTOR+A3."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from geotransformer.modules.loss import WeightedCircleLoss
from geotransformer.modules.ops import pairwise_distance
from geotransformer.modules.ops.transformation import apply_transform
from geotransformer.modules.registration.metrics import isotropic_transform_error


class CoarseMatchingLoss(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            cfg.coarse_loss.positive_margin,
            cfg.coarse_loss.negative_margin,
            cfg.coarse_loss.positive_optimal,
            cfg.coarse_loss.negative_optimal,
            cfg.coarse_loss.log_scale,
        )
        self.positive_overlap = float(cfg.coarse_loss.positive_overlap)

    def forward(self, output_dict):
        feature_distances = torch.sqrt(
            pairwise_distance(
                output_dict['ref_feats_c'],
                output_dict['src_feats_c'],
                normalized=True,
            ).clamp_min(1e-12)
        )
        overlaps = torch.zeros_like(feature_distances)
        corr_indices = output_dict['gt_node_corr_indices']
        overlaps[corr_indices[:, 0], corr_indices[:, 1]] = output_dict[
            'gt_node_corr_overlaps'
        ]
        positive_masks = overlaps > self.positive_overlap
        negative_masks = overlaps == 0
        positive_scales = torch.sqrt(overlaps * positive_masks.float())
        return self.weighted_circle_loss(
            positive_masks,
            negative_masks,
            feature_distances,
            positive_scales,
        )


class FineMatchingLoss(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.positive_radius = float(cfg.fine_loss.positive_radius)

    def forward(self, output_dict, data_dict):
        ref_points = output_dict['ref_node_corr_knn_points']
        src_points = apply_transform(
            output_dict['src_node_corr_knn_points'],
            data_dict['transform'],
        )
        ref_masks = output_dict['ref_node_corr_knn_masks']
        src_masks = output_dict['src_node_corr_knn_masks']
        matching_scores = output_dict['matching_scores']

        distances = pairwise_distance(ref_points, src_points)
        valid_pairs = ref_masks.unsqueeze(2) & src_masks.unsqueeze(1)
        gt_corr_map = (distances < self.positive_radius**2) & valid_pairs
        slack_row_labels = (gt_corr_map.sum(2) == 0) & ref_masks
        slack_col_labels = (gt_corr_map.sum(1) == 0) & src_masks

        labels = torch.zeros_like(matching_scores, dtype=torch.bool)
        labels[:, :-1, :-1] = gt_corr_map
        labels[:, :-1, -1] = slack_row_labels
        labels[:, -1, :-1] = slack_col_labels
        selected = matching_scores[labels]
        return -selected.mean() if selected.numel() else selected.sum()


class TopologyOverlapLoss(nn.Module):
    """Balanced focal supervision for bilateral superpoint overlap logits."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.positive_overlap = float(cfg.topology_overlap.positive_overlap)
        self.gamma = float(cfg.topology_overlap.focal_gamma)

    @staticmethod
    def _node_targets(count, indices, overlaps, device, dtype):
        targets = torch.zeros(count, device=device, dtype=dtype)
        if indices.numel():
            targets.scatter_reduce_(
                0,
                indices.to(device=device),
                overlaps.to(device=device, dtype=dtype),
                reduce='amax',
            )
        return targets

    def _focal_loss(self, logits, targets, masks):
        logits = logits[masks]
        targets = targets[masks]
        if logits.numel() == 0:
            return logits.sum()
        labels = (targets > self.positive_overlap).to(logits.dtype)
        probabilities = torch.sigmoid(logits)
        pt = torch.where(labels > 0, probabilities, 1.0 - probabilities)
        negative_count = labels.numel() - labels.sum()
        positive_weight = (negative_count / max(labels.numel(), 1)).clamp(
            0.05, 0.95
        )
        alpha = torch.where(labels > 0, positive_weight, 1.0 - positive_weight)
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        return (alpha * (1.0 - pt).pow(self.gamma) * bce).mean()

    def forward(self, output_dict):
        corr_indices = output_dict['gt_node_corr_indices'].detach()
        corr_overlaps = output_dict['gt_node_corr_overlaps'].detach()
        ref_logits = output_dict['ref_overlap_logits']
        src_logits = output_dict['src_overlap_logits']
        ref_targets = self._node_targets(
            len(ref_logits),
            corr_indices[:, 0],
            corr_overlaps,
            ref_logits.device,
            ref_logits.dtype,
        )
        src_targets = self._node_targets(
            len(src_logits),
            corr_indices[:, 1],
            corr_overlaps,
            src_logits.device,
            src_logits.dtype,
        )
        ref_loss = self._focal_loss(
            ref_logits,
            ref_targets,
            output_dict['ref_node_masks'],
        )
        src_loss = self._focal_loss(
            src_logits,
            src_targets,
            output_dict['src_node_masks'],
        )
        return ref_loss + src_loss


class CandidateRankingLoss(nn.Module):
    """Budget-aware surrogate for placing GT pairs inside global Top-K."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.positive_overlap = float(cfg.coarse_loss.positive_overlap)
        self.num_correspondences = int(cfg.coarse_matching.num_correspondences)
        self.margin = float(cfg.coarse_ranking.margin)
        self.temperature = float(cfg.coarse_ranking.temperature)
        self.boundary_window = int(cfg.coarse_ranking.boundary_window)
        if self.temperature <= 0:
            raise ValueError('coarse_ranking.temperature must be positive')
        if self.boundary_window <= 0:
            raise ValueError('coarse_ranking.boundary_window must be positive')

    def forward(self, output_dict):
        logits = output_dict['coarse_ranking_logits']
        valid_pairs = output_dict['coarse_valid_pair_masks']
        overlaps = torch.zeros_like(logits)
        corr_indices = output_dict['gt_node_corr_indices'].detach()
        corr_overlaps = output_dict['gt_node_corr_overlaps'].detach().to(logits)
        if corr_indices.numel():
            overlaps[corr_indices[:, 0], corr_indices[:, 1]] = corr_overlaps

        positive_mask = (overlaps > self.positive_overlap) & valid_pairs
        negative_mask = (overlaps == 0) & valid_pairs
        positive_scores = logits[positive_mask]
        positive_overlaps = overlaps[positive_mask]
        negative_scores = logits[negative_mask]
        if positive_scores.numel() == 0 or negative_scores.numel() == 0:
            return logits[valid_pairs].sum() * 0.0

        target_count = min(self.num_correspondences, positive_scores.numel())
        if positive_scores.numel() > target_count:
            target_indices = positive_overlaps.topk(target_count).indices
            positive_scores = positive_scores[target_indices]
            positive_overlaps = positive_overlaps[target_indices]

        # If P positives must enter a K-sized list, each should outrank the
        # (K-P+1)-th negative. Average a small boundary window to distribute
        # gradients among hard negatives without changing the target budget.
        negative_slots = max(1, self.num_correspondences - target_count + 1)
        negative_slots = min(negative_slots, negative_scores.numel())
        boundary_values = negative_scores.topk(negative_slots).values
        window = min(self.boundary_window, boundary_values.numel())
        negative_boundary = boundary_values[-window:].mean()

        violations = F.softplus(
            (
                negative_boundary
                + self.margin
                - positive_scores
            )
            / self.temperature
        ) * self.temperature
        weights = positive_overlaps.clamp_min(1e-12).sqrt()
        return (violations * weights).sum() / weights.sum().clamp_min(1e-12)


class OverallLoss(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.coarse_loss = CoarseMatchingLoss(cfg)
        self.fine_loss = FineMatchingLoss(cfg)
        self.overlap_loss = TopologyOverlapLoss(cfg)
        self.ranking_loss = CandidateRankingLoss(cfg)
        self.weight_coarse = float(cfg.loss.weight_coarse_loss)
        self.weight_fine = float(cfg.loss.weight_fine_loss)
        self.weight_overlap = float(cfg.topology_overlap.weight_loss)
        self.rtor_enabled = bool(
            cfg.ablation.get('overlap_supervision_enabled', cfg.ablation.rtor_enabled)
        )
        self.ranking_enabled = bool(cfg.ablation.get('ranking_loss_enabled', False))
        self.weight_ranking = float(cfg.coarse_ranking.weight_loss)

    def forward(self, output_dict, data_dict):
        coarse_loss = self.coarse_loss(output_dict)
        fine_loss = self.fine_loss(output_dict, data_dict)
        overlap_loss = (self.overlap_loss(output_dict)
                        if self.rtor_enabled
                        else coarse_loss.new_zeros(()))
        ranking_loss = (
            self.ranking_loss(output_dict)
            if self.ranking_enabled
            else coarse_loss.new_zeros(())
        )
        total_loss = (
            self.weight_coarse * coarse_loss
            + self.weight_fine * fine_loss
            + self.weight_overlap * overlap_loss
            + self.weight_ranking * ranking_loss
        )
        return {
            'loss': total_loss,
            'c_loss': coarse_loss,
            'f_loss': fine_loss,
            'o_loss': overlap_loss,
            'r_loss': ranking_loss,
        }


class Evaluator(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.acceptance_overlap = float(cfg.eval.acceptance_overlap)
        self.acceptance_radius = float(cfg.eval.acceptance_radius)
        self.acceptance_rmse = float(cfg.eval.rmse_threshold)
        self.ranking_positive_overlap = float(cfg.coarse_loss.positive_overlap)

    @torch.no_grad()
    def evaluate_coarse(self, output_dict):
        ref_count = output_dict['ref_points_c'].shape[0]
        src_count = output_dict['src_points_c'].shape[0]
        gt_mask = output_dict['gt_node_corr_overlaps'] > self.acceptance_overlap
        gt_indices = output_dict['gt_node_corr_indices'][gt_mask]
        gt_map = torch.zeros(
            ref_count,
            src_count,
            device=output_dict['ref_points_c'].device,
        )
        gt_map[gt_indices[:, 0], gt_indices[:, 1]] = 1.0
        return gt_map[
            output_dict['ref_node_corr_indices'],
            output_dict['src_node_corr_indices'],
        ].mean()

    @torch.no_grad()
    def evaluate_candidate_recall(self, output_dict):
        gt_mask = (
            output_dict['gt_node_corr_overlaps']
            > self.ranking_positive_overlap
        )
        gt_indices = output_dict['gt_node_corr_indices'][gt_mask]
        if gt_indices.numel() == 0:
            return output_dict['ref_points_c'].new_zeros(())
        predicted = torch.stack(
            [
                output_dict['ref_node_corr_indices'],
                output_dict['src_node_corr_indices'],
            ],
            dim=1,
        )
        if predicted.numel() == 0:
            return output_dict['ref_points_c'].new_zeros(())
        hits = (gt_indices[:, None, :] == predicted[None, :, :]).all(dim=2)
        return hits.any(dim=1).float().mean()

    @torch.no_grad()
    def evaluate_gt_mrr(self, output_dict):
        logits = output_dict['coarse_ranking_logits']
        valid_pairs = output_dict['coarse_valid_pair_masks']
        gt_mask = (
            output_dict['gt_node_corr_overlaps']
            > self.ranking_positive_overlap
        )
        gt_indices = output_dict['gt_node_corr_indices'][gt_mask]
        if gt_indices.numel() == 0:
            return logits.new_zeros(())
        gt_pair_valid = valid_pairs[gt_indices[:, 0], gt_indices[:, 1]]
        gt_indices = gt_indices[gt_pair_valid]
        if gt_indices.numel() == 0:
            return logits.new_zeros(())
        valid_linear_indices = torch.nonzero(
            valid_pairs.reshape(-1), as_tuple=True
        )[0]
        valid_scores = logits.reshape(-1)[valid_linear_indices]
        order = valid_scores.argsort(descending=True)
        rank_by_linear_index = torch.zeros(
            logits.numel(), dtype=torch.long, device=logits.device
        )
        rank_by_linear_index[valid_linear_indices[order]] = torch.arange(
            1, valid_scores.numel() + 1, device=logits.device
        )
        gt_linear_indices = (
            gt_indices[:, 0] * logits.shape[1] + gt_indices[:, 1]
        )
        ranks = rank_by_linear_index[gt_linear_indices]
        return ranks.float().reciprocal().mean()

    @torch.no_grad()
    def evaluate_fine(self, output_dict, data_dict):
        src_corr_points = apply_transform(
            output_dict['src_corr_points'],
            data_dict['transform'],
        )
        distances = torch.linalg.norm(
            output_dict['ref_corr_points'] - src_corr_points,
            dim=1,
        )
        return (distances < self.acceptance_radius).float().mean()

    @torch.no_grad()
    def evaluate_registration(self, output_dict, data_dict):
        transform = data_dict['transform']
        estimated_transform = output_dict['estimated_transform']
        rre, rte = isotropic_transform_error(transform, estimated_transform)
        realignment = torch.inverse(transform) @ estimated_transform
        realigned_source = apply_transform(output_dict['src_points'], realignment)
        rmse = torch.linalg.norm(
            realigned_source - output_dict['src_points'], dim=1
        ).mean()
        recall = (rmse < self.acceptance_rmse).float()
        return rre, rte, rmse, recall

    def forward(self, output_dict, data_dict):
        rre, rte, rmse, recall = self.evaluate_registration(
            output_dict, data_dict
        )
        results = {
            'PIR': self.evaluate_coarse(output_dict),
            'CR@K': self.evaluate_candidate_recall(output_dict),
            'IR': self.evaluate_fine(output_dict, data_dict),
            'RRE': rre,
            'RTE': rte,
            'RMSE': rmse,
            'RR': recall,
        }
        if 'coarse_ranking_logits' in output_dict:
            results['GT_MRR'] = self.evaluate_gt_mrr(output_dict)
        return results


__all__ = [
    'CandidateRankingLoss', 'Evaluator', 'OverallLoss', 'TopologyOverlapLoss'
]
