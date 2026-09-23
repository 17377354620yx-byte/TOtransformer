"""Topology and visibility conditioning for geometric transformer attention."""

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _batch_gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather ``(B, M, C)`` values with ``(B, N, K)`` indices."""
    batch = torch.arange(values.shape[0], device=values.device)[:, None, None]
    return values[batch, indices]


class TopologyOverlapConditioner(nn.Module):
    """Build additive topology and overlap biases without altering descriptors.

    The topology branch augments only geometric self-attention.  The overlap
    branch predicts bilateral visibility before every cross-attention block and
    adds a key-side log-prior.  Zero-initialized output projections and overlap
    gates make the complete conditioner neutral at initialization.
    """

    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        num_self_blocks: int,
        num_cross_blocks: int,
        num_neighbors: int = 8,
        hidden_dim: int = 64,
        poincare_curvature: float = 1.0,
        overlap_attention: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError('feature_dim and hidden_dim must be positive')
        if feature_dim % num_heads != 0:
            raise ValueError('feature_dim must be divisible by num_heads')
        if num_neighbors <= 0:
            raise ValueError('num_neighbors must be positive')
        if poincare_curvature <= 0:
            raise ValueError('poincare_curvature must be positive')

        self.feature_dim = int(feature_dim)
        self.num_heads = int(num_heads)
        self.num_neighbors = int(num_neighbors)
        self.poincare_curvature = float(poincare_curvature)
        self.overlap_attention = bool(overlap_attention)
        self.eps = float(eps)

        # normalized edge distance, curvature, linearity, planarity,
        # and dynamic Poincare feature distance.
        self.topology_bias = nn.ModuleList()
        for _ in range(num_self_blocks):
            projection = nn.Sequential(
                nn.Linear(5, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, num_heads),
            )
            nn.init.zeros_(projection[-1].weight)
            nn.init.zeros_(projection[-1].bias)
            self.topology_bias.append(projection)

        if self.overlap_attention:
            # feature + mean normalized NN distance + three spectral geometry
            # values + cross-support maximum and normalized entropy.
            self.overlap_head = nn.Sequential(
                nn.Linear(feature_dim + 6, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.overlap_head[-1].weight)
            nn.init.zeros_(self.overlap_head[-1].bias)
            self.overlap_beta = nn.Parameter(
                torch.zeros(num_cross_blocks, num_heads)
            )
        else:
            self.overlap_head = None
            self.register_parameter('overlap_beta', None)

    @torch.no_grad()
    def prepare(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Pre-compute rigid-invariant local topology for one point cloud."""
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError('points must have shape (B, N, 3)')
        batch_size, count, _ = points.shape
        if count == 0:
            raise ValueError('topology attention requires at least one point')
        if count == 1:
            indices = torch.zeros(
                (batch_size, 1, 1), dtype=torch.long, device=points.device
            )
            distances = points.new_zeros((batch_size, 1, 1))
            geometry = points.new_zeros((batch_size, 1, 3))
        else:
            k = min(self.num_neighbors, count - 1)
            # CUDA eigvalsh does not implement BF16.  Keep only the rigid
            # geometry statistics in FP32 while the surrounding Transformer
            # remains eligible for autocast.
            with torch.autocast(device_type=points.device.type, enabled=False):
                float_points = points.float()
                distance_map = torch.cdist(float_points, float_points)
                indices = distance_map.topk(
                    k=k + 1, dim=2, largest=False
                ).indices[:, :, 1:]
                distances = distance_map.gather(2, indices)
                neighbours = _batch_gather(float_points, indices)
                offsets = neighbours - float_points[:, :, None, :]
                covariance = offsets.transpose(2, 3) @ offsets / float(k)
                eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
                l0, l1, l2 = eigenvalues.unbind(dim=2)
                trace = (l0 + l1 + l2).clamp_min(1e-12)
                curvature = l0 / trace
                linearity = (l2 - l1) / l2.clamp_min(1e-12)
                planarity = (l1 - l0) / l2.clamp_min(1e-12)
                geometry = torch.stack(
                    [curvature, linearity, planarity], dim=2
                )
                median_nn = distances[:, :, 0].reshape(
                    batch_size, -1
                ).median(dim=1).values.clamp_min(1e-12)
                distances = distances / median_nn[:, None, None]

        node_topology = torch.cat(
            [distances.mean(dim=2, keepdim=True), geometry], dim=2
        )
        return {
            'indices': indices,
            'distances': distances,
            'geometry': geometry,
            'node_topology': node_topology,
        }

    def _poincare_map(self, features: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features.float(), p=2, dim=-1)
        curvature = features.new_tensor(self.poincare_curvature)
        sqrt_c = curvature.sqrt()
        norm = torch.linalg.norm(features, dim=-1, keepdim=True).clamp_min(1e-12)
        mapped = torch.tanh(sqrt_c * norm) * features / (sqrt_c * norm)
        max_norm = (1.0 - 1e-4) / sqrt_c
        mapped_norm = torch.linalg.norm(mapped, dim=-1, keepdim=True).clamp_min(1e-12)
        return mapped * (max_norm / mapped_norm).clamp_max(1.0)

    def _poincare_neighbor_distance(
        self, features: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        mapped = self._poincare_map(features)
        neighbours = _batch_gather(mapped, indices)
        center = mapped[:, :, None, :]
        curvature = mapped.new_tensor(self.poincare_curvature)
        delta2 = (center - neighbours).square().sum(dim=-1)
        center2 = center.square().sum(dim=-1)
        neighbour2 = neighbours.square().sum(dim=-1)
        denominator = (
            (1.0 - curvature * center2)
            * (1.0 - curvature * neighbour2)
        ).clamp_min(1e-8)
        argument = (1.0 + 2.0 * curvature * delta2 / denominator).clamp_min(
            1.0 + 1e-7
        )
        return torch.acosh(argument) / curvature.sqrt()

    def self_attention_bias(
        self,
        features: torch.Tensor,
        context: Dict[str, torch.Tensor],
        block_index: int,
    ) -> torch.Tensor:
        """Return sparse local additive bias with shape ``(B, H, N, N)``."""
        indices = context['indices']
        hyperbolic_distance = self._poincare_neighbor_distance(features, indices)
        edge_features = torch.cat(
            [
                context['distances'][..., None],
                context['geometry'][:, :, None, :].expand(
                    -1, -1, indices.shape[2], -1
                ),
                hyperbolic_distance[..., None],
            ],
            dim=-1,
        )
        local_bias = self.topology_bias[block_index](edge_features)
        local_bias = local_bias.permute(0, 3, 1, 2)
        batch_size, _, count, _ = local_bias.shape
        bias = local_bias.new_zeros(
            (batch_size, self.num_heads, count, count)
        )
        scatter_indices = indices[:, None].expand(-1, self.num_heads, -1, -1)
        bias.scatter_(3, scatter_indices, local_bias)
        return bias.to(dtype=features.dtype)

    @staticmethod
    def _cross_statistics(
        ref_features: torch.Tensor, src_features: torch.Tensor
    ):
        ref_normalized = F.normalize(ref_features.float(), p=2, dim=-1)
        src_normalized = F.normalize(src_features.float(), p=2, dim=-1)
        similarities = ref_normalized @ src_normalized.transpose(1, 2)

        ref_probabilities = torch.softmax(similarities, dim=2)
        src_probabilities = torch.softmax(similarities.transpose(1, 2), dim=2)
        ref_max = ref_probabilities.max(dim=2).values
        src_max = src_probabilities.max(dim=2).values
        ref_entropy = -(
            ref_probabilities * ref_probabilities.clamp_min(1e-12).log()
        ).sum(dim=2)
        src_entropy = -(
            src_probabilities * src_probabilities.clamp_min(1e-12).log()
        ).sum(dim=2)
        if src_features.shape[1] > 1:
            ref_entropy = ref_entropy / math.log(src_features.shape[1])
        else:
            ref_entropy = torch.zeros_like(ref_entropy)
        if ref_features.shape[1] > 1:
            src_entropy = src_entropy / math.log(ref_features.shape[1])
        else:
            src_entropy = torch.zeros_like(src_entropy)
        return ref_max, ref_entropy, src_max, src_entropy

    def overlap_logits(
        self,
        ref_features: torch.Tensor,
        src_features: torch.Tensor,
        ref_context: Dict[str, torch.Tensor],
        src_context: Dict[str, torch.Tensor],
    ):
        if self.overlap_head is None:
            raise RuntimeError('overlap attention is disabled')
        ref_max, ref_entropy, src_max, src_entropy = self._cross_statistics(
            ref_features, src_features
        )
        ref_input = torch.cat(
            [
                ref_features.float(),
                ref_context['node_topology'],
                ref_max[..., None],
                ref_entropy[..., None],
            ],
            dim=-1,
        )
        src_input = torch.cat(
            [
                src_features.float(),
                src_context['node_topology'],
                src_max[..., None],
                src_entropy[..., None],
            ],
            dim=-1,
        )
        return (
            self.overlap_head(ref_input).squeeze(-1),
            self.overlap_head(src_input).squeeze(-1),
        )

    def cross_attention_bias(
        self,
        ref_logits: torch.Tensor,
        src_logits: torch.Tensor,
        block_index: int,
    ):
        """Build key-side ``beta * log(p + eps)`` biases for both directions."""
        beta = self.overlap_beta[block_index]
        ref_log_probability = torch.log(torch.sigmoid(ref_logits) + self.eps)
        src_log_probability = torch.log(torch.sigmoid(src_logits) + self.eps)
        ref_to_src = (
            beta[None, :, None, None]
            * src_log_probability[:, None, None, :]
        )
        src_to_ref = (
            beta[None, :, None, None]
            * ref_log_probability[:, None, None, :]
        )
        return ref_to_src, src_to_ref


__all__ = ['TopologyOverlapConditioner']
