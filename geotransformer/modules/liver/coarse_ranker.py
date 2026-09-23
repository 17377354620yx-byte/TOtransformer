"""Differentiable topology-overlap coarse correspondence ranking."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from geotransformer.modules.ops import pairwise_distance


def _batchless_gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return values[indices.reshape(-1)].reshape(*indices.shape, values.shape[-1])


class TopologyOverlapCoarseRanker(nn.Module):
    """Rank every valid superpoint pair without deleting candidates.

    The descriptor term exactly follows ``SuperPointMatching``.  The learned
    overlap gate starts at one so a Phase-2 checkpoint retains its existing
    soft-overlap ranking.  The topology compatibility head is zero initialized,
    making a Phase-2 warm start output-equivalent before optimization.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        num_neighbors: int = 8,
        poincare_curvature: float = 1.0,
        dual_normalization: bool = True,
        overlap_score_floor: float = 0.05,
        overlap_score_power: float = 1.0,
        use_overlap_prior: bool = True,
        use_topology_compatibility: bool = True,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError('feature_dim and hidden_dim must be positive')
        if num_neighbors <= 0:
            raise ValueError('num_neighbors must be positive')
        if poincare_curvature <= 0:
            raise ValueError('poincare_curvature must be positive')
        if not 0.0 <= overlap_score_floor < 1.0:
            raise ValueError('overlap_score_floor must be in [0, 1)')
        if overlap_score_power <= 0:
            raise ValueError('overlap_score_power must be positive')

        self.feature_dim = int(feature_dim)
        self.num_neighbors = int(num_neighbors)
        self.poincare_curvature = float(poincare_curvature)
        self.dual_normalization = bool(dual_normalization)
        self.overlap_score_floor = float(overlap_score_floor)
        self.overlap_score_power = float(overlap_score_power)
        self.use_overlap_prior = bool(use_overlap_prior)
        self.use_topology_compatibility = bool(use_topology_compatibility)

        if self.use_overlap_prior:
            # One reproduces Phase-2 soft overlap calibration exactly.
            self.overlap_gate = nn.Parameter(torch.ones(()))
        else:
            self.register_parameter('overlap_gate', None)

        if self.use_topology_compatibility:
            # |local topology_i - local topology_j| (4) + Poincare distance (1)
            self.topology_head = nn.Sequential(
                nn.Linear(5, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.topology_head[-1].weight)
            nn.init.zeros_(self.topology_head[-1].bias)
        else:
            self.topology_head = None

    @staticmethod
    def _valid_masks(features, masks):
        if masks is None:
            return torch.ones(
                features.shape[0], dtype=torch.bool, device=features.device
            )
        return masks

    @torch.no_grad()
    def _local_topology(self, points: torch.Tensor) -> torch.Tensor:
        count = points.shape[0]
        if count == 0:
            raise ValueError('coarse ranking requires at least one point')
        if count == 1:
            return points.new_zeros((1, 4))
        k = min(self.num_neighbors, count - 1)
        with torch.autocast(device_type=points.device.type, enabled=False):
            float_points = points.float()
            distances = torch.cdist(float_points, float_points)
            indices = distances.topk(k=k + 1, dim=1, largest=False).indices[:, 1:]
            neighbour_distances = distances.gather(1, indices)
            neighbours = _batchless_gather(float_points, indices)
            offsets = neighbours - float_points[:, None, :]
            covariance = offsets.transpose(1, 2) @ offsets / float(k)
            eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
            l0, l1, l2 = eigenvalues.unbind(dim=1)
            trace = (l0 + l1 + l2).clamp_min(1e-12)
            geometry = torch.stack(
                [
                    l0 / trace,
                    (l2 - l1) / l2.clamp_min(1e-12),
                    (l1 - l0) / l2.clamp_min(1e-12),
                ],
                dim=1,
            )
            scale = neighbour_distances[:, 0].median().clamp_min(1e-12)
            mean_distance = (neighbour_distances / scale).mean(dim=1, keepdim=True)
        return torch.cat([mean_distance, geometry], dim=1)

    def _poincare_map(self, features: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(features.float(), p=2, dim=-1)
        curvature = normalized.new_tensor(self.poincare_curvature)
        sqrt_c = curvature.sqrt()
        norm = torch.linalg.norm(normalized, dim=-1, keepdim=True).clamp_min(1e-12)
        mapped = torch.tanh(sqrt_c * norm) * normalized / (sqrt_c * norm)
        max_norm = (1.0 - 1e-4) / sqrt_c
        mapped_norm = torch.linalg.norm(mapped, dim=-1, keepdim=True).clamp_min(1e-12)
        return mapped * (max_norm / mapped_norm).clamp_max(1.0)

    def _poincare_pair_distance(
        self, ref_features: torch.Tensor, src_features: torch.Tensor
    ) -> torch.Tensor:
        ref = self._poincare_map(ref_features)
        src = self._poincare_map(src_features)
        curvature = ref.new_tensor(self.poincare_curvature)
        delta2 = (ref[:, None, :] - src[None, :, :]).square().sum(dim=-1)
        ref2 = ref.square().sum(dim=-1, keepdim=True)
        src2 = src.square().sum(dim=-1)[None, :]
        denominator = (
            (1.0 - curvature * ref2) * (1.0 - curvature * src2)
        ).clamp_min(1e-8)
        argument = (1.0 + 2.0 * curvature * delta2 / denominator).clamp_min(
            1.0 + 1e-7
        )
        return torch.acosh(argument) / curvature.sqrt()

    def _descriptor_scores(
        self,
        ref_features: torch.Tensor,
        src_features: torch.Tensor,
        pair_masks: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.exp(
            -pairwise_distance(ref_features, src_features, normalized=True)
        )
        scores = scores * pair_masks.to(scores.dtype)
        if self.dual_normalization:
            ref_scores = scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-12)
            src_scores = scores / scores.sum(dim=0, keepdim=True).clamp_min(1e-12)
            scores = ref_scores * src_scores
        return scores

    def forward(
        self,
        ref_points: torch.Tensor,
        src_points: torch.Tensor,
        ref_features: torch.Tensor,
        src_features: torch.Tensor,
        ref_overlap_probability: torch.Tensor,
        src_overlap_probability: torch.Tensor,
        ref_masks: torch.Tensor = None,
        src_masks: torch.Tensor = None,
    ):
        ref_masks = self._valid_masks(ref_features, ref_masks)
        src_masks = self._valid_masks(src_features, src_masks)
        pair_masks = ref_masks[:, None] & src_masks[None, :]
        descriptor_scores = self._descriptor_scores(
            ref_features, src_features, pair_masks
        )
        ranking_logits = torch.log(descriptor_scores.clamp_min(1e-12))

        if self.use_overlap_prior:
            ref_weight = self.overlap_score_floor + (
                1.0 - self.overlap_score_floor
            ) * ref_overlap_probability.pow(self.overlap_score_power)
            src_weight = self.overlap_score_floor + (
                1.0 - self.overlap_score_floor
            ) * src_overlap_probability.pow(self.overlap_score_power)
            overlap_prior = (
                torch.log(ref_weight.clamp_min(1e-12))[:, None]
                + torch.log(src_weight.clamp_min(1e-12))[None, :]
            )
            ranking_logits = ranking_logits + self.overlap_gate * overlap_prior

        if self.use_topology_compatibility:
            ref_topology = self._local_topology(ref_points)
            src_topology = self._local_topology(src_points)
            topology_difference = (
                ref_topology[:, None, :] - src_topology[None, :, :]
            ).abs()
            poincare_distance = self._poincare_pair_distance(
                ref_features, src_features
            )
            topology_features = torch.cat(
                [topology_difference, poincare_distance[..., None]], dim=-1
            )
            topology_bias = self.topology_head(topology_features).squeeze(-1)
            ranking_logits = ranking_logits + topology_bias
        else:
            topology_bias = torch.zeros_like(ranking_logits)

        ranking_logits = ranking_logits.masked_fill(~pair_masks, float('-inf'))
        ranking_scores = torch.zeros_like(descriptor_scores)
        valid_logits = ranking_logits[pair_masks]
        if valid_logits.numel():
            original_mass = descriptor_scores.sum()
            ranking_scores[pair_masks] = (
                torch.softmax(valid_logits, dim=0) * original_mass
            )
        diagnostics = {
            'coarse_descriptor_scores': descriptor_scores,
            'coarse_ranking_logits': ranking_logits,
            'coarse_topology_compatibility': topology_bias,
            'coarse_valid_pair_masks': pair_masks,
        }
        return ranking_scores, diagnostics


__all__ = ['TopologyOverlapCoarseRanker']
