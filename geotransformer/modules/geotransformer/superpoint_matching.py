import torch
import torch.nn as nn

from geotransformer.modules.ops import pairwise_distance


class SuperPointMatching(nn.Module):
    """Global coarse matching with optional bilateral RTOR calibration."""

    def __init__(self, num_correspondences, dual_normalization=True):
        super(SuperPointMatching, self).__init__()
        self.num_correspondences = num_correspondences
        self.dual_normalization = dual_normalization

    @staticmethod
    def _calibrate_scores(scores, ref_weights=None, src_weights=None):
        if ref_weights is None and src_weights is None:
            return scores
        original_mass = scores.sum()
        calibrated = scores
        if ref_weights is not None:
            calibrated = calibrated * ref_weights[:, None]
        if src_weights is not None:
            calibrated = calibrated * src_weights[None, :]
        return calibrated * (original_mass / calibrated.sum().clamp_min(1e-12))

    def forward(
        self,
        ref_feats,
        src_feats,
        ref_masks=None,
        src_masks=None,
        ref_weights=None,
        src_weights=None,
        precomputed_scores=None,
    ):
        r"""Extract global Top-K superpoint correspondences."""
        if ref_masks is None:
            ref_masks = torch.ones(
                ref_feats.shape[0], dtype=torch.bool, device=ref_feats.device
            )
        if src_masks is None:
            src_masks = torch.ones(
                src_feats.shape[0], dtype=torch.bool, device=src_feats.device
            )

        ref_indices = torch.nonzero(ref_masks, as_tuple=True)[0]
        src_indices = torch.nonzero(src_masks, as_tuple=True)[0]
        ref_feats = ref_feats[ref_indices]
        src_feats = src_feats[src_indices]
        if ref_weights is not None:
            ref_weights = ref_weights[ref_indices].clamp_min(1e-6)
        if src_weights is not None:
            src_weights = src_weights[src_indices].clamp_min(1e-6)

        if precomputed_scores is None:
            matching_scores = torch.exp(
                -pairwise_distance(ref_feats, src_feats, normalized=True)
            )
            if self.dual_normalization:
                ref_scores = matching_scores / matching_scores.sum(
                    dim=1, keepdim=True
                ).clamp_min(1e-12)
                src_scores = matching_scores / matching_scores.sum(
                    dim=0, keepdim=True
                ).clamp_min(1e-12)
                matching_scores = ref_scores * src_scores
        else:
            expected_shape = (ref_masks.shape[0], src_masks.shape[0])
            if precomputed_scores.shape != expected_shape:
                raise ValueError(
                    'precomputed_scores must have shape '
                    f'{expected_shape}, got {tuple(precomputed_scores.shape)}'
                )
            matching_scores = precomputed_scores[ref_indices][:, src_indices]
        matching_scores = self._calibrate_scores(
            matching_scores, ref_weights, src_weights
        )

        num_correspondences = min(self.num_correspondences, matching_scores.numel())
        corr_scores, corr_indices = matching_scores.reshape(-1).topk(
            k=num_correspondences, largest=True
        )
        ref_sel_indices = corr_indices // matching_scores.shape[1]
        src_sel_indices = corr_indices % matching_scores.shape[1]
        ref_corr_indices = ref_indices[ref_sel_indices]
        src_corr_indices = src_indices[src_sel_indices]
        return ref_corr_indices, src_corr_indices, corr_scores
