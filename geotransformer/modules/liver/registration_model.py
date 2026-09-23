"""Rigid complete-to-partial registration with RTOR and A3."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from geotransformer.modules.geotransformer import (
    GeometricTransformer,
    LocalGlobalRegistration,
    SuperPointMatching,
    SuperPointTargetGenerator,
)
from geotransformer.modules.kpconv.backbone import KPConvFPN
from geotransformer.modules.liver.fine_local_refiner import (
    GeometryAwareFineRefiner,
)
from geotransformer.modules.liver.overlap_selection import select_overlap_region
from geotransformer.modules.liver.topology_overlap import TopologyOverlapRefiner
from geotransformer.modules.liver.cooperative_matching import mix_coarse_proposals
from geotransformer.modules.liver.coarse_ranker import TopologyOverlapCoarseRanker
from geotransformer.modules.ops import index_select, point_to_node_partition
from geotransformer.modules.registration import get_node_correspondences
from geotransformer.modules.sinkhorn import LearnableLogOptimalTransport


class GeoTransformer(nn.Module):
    """GeoTransformer specialized to the fixed RTOR+A3 liver pipeline.

    The model always produces one rigid transformation. RTOR refines coarse
    features, calibrates coarse similarity, and focuses only the complete
    source cloud during inference. A3 refines paired fine patches before
    Sinkhorn and Local-to-Global Registration (LGR).
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.num_points_in_patch = int(cfg.model.num_points_in_patch)
        self.matching_radius = float(cfg.model.ground_truth_matching_radius)
        self.rtor_enabled = bool(cfg.ablation.rtor_enabled)
        self.a3_enabled = bool(cfg.ablation.a3_enabled)
        self.topology_attention_enabled = bool(
            cfg.ablation.get('topology_attention', False)
        )
        self.overlap_cross_attention_enabled = bool(
            cfg.ablation.get('overlap_cross_attention', False)
        )
        self.legacy_rtor_post_refine = bool(
            cfg.ablation.get('legacy_rtor_post_refine', self.rtor_enabled)
        )
        self.coarse_ranking_enabled = bool(
            cfg.ablation.get('coarse_ranking_enabled', False)
        )
        self.overlap_soft_weight = bool(
            cfg.ablation.get('overlap_soft_weight', True)
        )
        precision_cfg = cfg.get('precision', {})
        self.selective_bf16 = bool(precision_cfg.get('selective_bf16', False))
        self.predicted_coarse_ratio = 0.0

        self.backbone = KPConvFPN(
            cfg.backbone.input_dim,
            cfg.backbone.output_dim,
            cfg.backbone.init_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.init_radius,
            cfg.backbone.init_sigma,
            cfg.backbone.group_norm,
        )
        self.fine_backbone = None
        if cfg.model.get('dual_encoder', False):
            import copy
            self.fine_backbone = copy.deepcopy(self.backbone)
        self.transformer = GeometricTransformer(
            cfg.geotransformer.input_dim,
            cfg.geotransformer.output_dim,
            cfg.geotransformer.hidden_dim,
            cfg.geotransformer.num_heads,
            cfg.geotransformer.blocks,
            cfg.geotransformer.sigma_d,
            cfg.geotransformer.sigma_a,
            cfg.geotransformer.angle_k,
            reduction_a=cfg.geotransformer.reduction_a,
            topology_attention=self.topology_attention_enabled,
            overlap_attention=self.overlap_cross_attention_enabled,
            topology_num_neighbors=cfg.topology_overlap.num_neighbors,
            topology_hidden_dim=cfg.topology_overlap.attention_hidden_dim,
            poincare_curvature=cfg.topology_overlap.poincare_curvature,
        )
        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets,
            cfg.coarse_matching.overlap_threshold,
        )
        self.coarse_matching = SuperPointMatching(
            cfg.coarse_matching.num_correspondences,
            cfg.coarse_matching.dual_normalization,
        )
        self.coarse_ranker = (
            TopologyOverlapCoarseRanker(
                feature_dim=cfg.geotransformer.output_dim,
                hidden_dim=cfg.coarse_ranking.hidden_dim,
                num_neighbors=cfg.topology_overlap.num_neighbors,
                poincare_curvature=cfg.topology_overlap.poincare_curvature,
                dual_normalization=cfg.coarse_matching.dual_normalization,
                overlap_score_floor=cfg.topology_overlap.score_floor,
                overlap_score_power=cfg.topology_overlap.score_power,
                use_overlap_prior=cfg.ablation.get(
                    'coarse_overlap_prior_enabled', True
                ),
                use_topology_compatibility=cfg.ablation.get(
                    'coarse_topology_compatibility_enabled', True
                ),
            )
            if self.coarse_ranking_enabled
            else None
        )
        self.fine_matching = LocalGlobalRegistration(
            cfg.fine_matching.topk,
            cfg.fine_matching.acceptance_radius,
            mutual=cfg.fine_matching.mutual,
            confidence_threshold=cfg.fine_matching.confidence_threshold,
            use_dustbin=cfg.fine_matching.use_dustbin,
            use_global_score=cfg.fine_matching.use_global_score,
            correspondence_threshold=cfg.fine_matching.correspondence_threshold,
            correspondence_limit=cfg.fine_matching.correspondence_limit,
            num_refinement_steps=cfg.fine_matching.num_refinement_steps,
        )
        self.optimal_transport = LearnableLogOptimalTransport(
            cfg.model.num_sinkhorn_iterations
        )
        self.fine_matching.robust_refinement_radius = cfg.fine_matching.get('robust_refinement_radius')
        self.fine_local_refiner = GeometryAwareFineRefiner(
            feature_dim=cfg.backbone.output_dim,
            num_heads=cfg.fine_refiner.num_heads,
            dropout=cfg.fine_refiner.dropout,
            geometry_sigma=cfg.fine_refiner.geometry_sigma,
            geometry_weight=cfg.fine_refiner.geometry_weight,
            residual_init=cfg.fine_refiner.residual_init,
            use_geometry_bias=cfg.ablation.get('a3_geometry_bias', True),
        ) if self.a3_enabled else None
        self.topology_overlap_refiner = TopologyOverlapRefiner(
            feature_dim=cfg.geotransformer.output_dim,
            hidden_dim=cfg.topology_overlap.hidden_dim,
            num_neighbors=cfg.topology_overlap.num_neighbors,
            poincare_curvature=cfg.topology_overlap.poincare_curvature,
            dropout=cfg.topology_overlap.dropout,
            use_poincare=cfg.ablation.get('rtor_poincare', True),
            refine_descriptors=cfg.ablation.get(
                'rtor_descriptor_update', True
            ),
        ) if self.rtor_enabled and self.legacy_rtor_post_refine else None

        self.overlap_score_floor = float(cfg.topology_overlap.score_floor)
        self.rtor_residual_scale = 1.0
        self.overlap_score_power = float(cfg.topology_overlap.score_power)
        self.focus_threshold = float(cfg.overlap_selection.threshold)
        self.focus_enabled = bool(cfg.overlap_selection.get('enabled', True))
        self.focus_min_superpoints = int(cfg.overlap_selection.min_superpoints)
        self.focus_max_ratio = float(cfg.overlap_selection.max_ratio)
        self.focus_min_spread = float(cfg.overlap_selection.min_spread)

    def _encode_coarse_features(
        self,
        ref_points_c,
        src_points_c,
        ref_feats_c,
        src_feats_c,
    ):
        device_type = ref_feats_c.device.type
        autocast_enabled = self.selective_bf16 and device_type in ('cuda', 'cpu')
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            encoded = self.transformer(
                ref_points_c.unsqueeze(0),
                src_points_c.unsqueeze(0),
                ref_feats_c.unsqueeze(0),
                src_feats_c.unsqueeze(0),
            )
        if self.topology_attention_enabled:
            ref_feats_c, src_feats_c, diagnostics = encoded
            diagnostics = {
                key: value.float().squeeze(0)
                for key, value in diagnostics.items()
            }
        else:
            ref_feats_c, src_feats_c = encoded
            diagnostics = None
        # Keep all RTOR, matching, loss and pose-estimation operations in FP32.
        ref_feats_c = ref_feats_c.float()
        src_feats_c = src_feats_c.float()
        ref_feats_c = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        src_feats_c = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)
        if self.topology_attention_enabled:
            return ref_feats_c, src_feats_c, diagnostics
        return ref_feats_c, src_feats_c

    def _apply_rtor(
        self,
        ref_points_c,
        src_points_c,
        ref_feats_c,
        src_feats_c,
        conditioned_diagnostics=None,
    ):
        if not self.rtor_enabled:
            ref_probability = ref_feats_c.new_ones(len(ref_feats_c))
            src_probability = src_feats_c.new_ones(len(src_feats_c))
            diagnostics = dict(ref_overlap_logits=torch.zeros_like(ref_probability),
                               src_overlap_logits=torch.zeros_like(src_probability))
            return (ref_feats_c, src_feats_c, ref_probability, src_probability,
                    ref_probability, src_probability, diagnostics)
        if not self.legacy_rtor_post_refine:
            diagnostics = conditioned_diagnostics or {}
            diagnostics.setdefault(
                'ref_overlap_logits', ref_feats_c.new_zeros(len(ref_feats_c))
            )
            diagnostics.setdefault(
                'src_overlap_logits', src_feats_c.new_zeros(len(src_feats_c))
            )
            ref_probability = torch.sigmoid(diagnostics['ref_overlap_logits'])
            src_probability = torch.sigmoid(diagnostics['src_overlap_logits'])
            if not self.overlap_cross_attention_enabled:
                ref_probability = torch.ones_like(ref_probability)
                src_probability = torch.ones_like(src_probability)
            if self.overlap_soft_weight and self.overlap_cross_attention_enabled:
                ref_weight = self.overlap_score_floor + (
                    1.0 - self.overlap_score_floor
                ) * ref_probability.pow(self.overlap_score_power)
                src_weight = self.overlap_score_floor + (
                    1.0 - self.overlap_score_floor
                ) * src_probability.pow(self.overlap_score_power)
            else:
                ref_weight = torch.ones_like(ref_probability)
                src_weight = torch.ones_like(src_probability)
            return (
                ref_feats_c,
                src_feats_c,
                ref_probability,
                src_probability,
                ref_weight,
                src_weight,
                diagnostics,
            )
        original_ref, original_src = ref_feats_c, src_feats_c
        ref_feats_c, src_feats_c, diagnostics = self.topology_overlap_refiner(
            ref_points_c,
            src_points_c,
            ref_feats_c,
            src_feats_c,
        )
        if self.rtor_residual_scale != 1.0:
            ref_feats_c = original_ref + self.rtor_residual_scale * (ref_feats_c - original_ref)
            src_feats_c = original_src + self.rtor_residual_scale * (src_feats_c - original_src)
        ref_feats_c = F.normalize(ref_feats_c, p=2, dim=1)
        src_feats_c = F.normalize(src_feats_c, p=2, dim=1)
        ref_probability = torch.sigmoid(diagnostics['ref_overlap_logits'])
        src_probability = torch.sigmoid(diagnostics['src_overlap_logits'])
        if self.overlap_soft_weight:
            ref_weight = self.overlap_score_floor + (
                1.0 - self.overlap_score_floor
            ) * ref_probability.pow(self.overlap_score_power)
            src_weight = self.overlap_score_floor + (
                1.0 - self.overlap_score_floor
            ) * src_probability.pow(self.overlap_score_power)
        else:
            ref_weight = torch.ones_like(ref_probability)
            src_weight = torch.ones_like(src_probability)
        return (
            ref_feats_c,
            src_feats_c,
            ref_probability,
            src_probability,
            ref_weight,
            src_weight,
            diagnostics,
        )

    def forward(self, data_dict):
        output_dict = {}
        features = data_dict['features'].detach()
        transform = data_dict.get('transform')
        if transform is not None:
            transform = transform.detach()
        elif self.training:
            raise ValueError('Training requires a supervision transform')

        ref_length_c = int(data_dict['lengths'][-1][0])
        ref_length_f = int(data_dict['lengths'][1][0])
        ref_length = int(data_dict['lengths'][0][0])
        points_c = data_dict['points'][-1].detach()
        points_f = data_dict['points'][1].detach()
        points = data_dict['points'][0].detach()

        ref_points_c, src_points_c = points_c[:ref_length_c], points_c[ref_length_c:]
        ref_points_f, src_points_f = points_f[:ref_length_f], points_f[ref_length_f:]
        ref_points, src_points = points[:ref_length], points[ref_length:]
        output_dict.update(
            ref_points_c=ref_points_c,
            src_points_c=src_points_c,
            ref_points_f=ref_points_f,
            src_points_f=src_points_f,
            ref_points=ref_points,
            src_points=src_points,
        )

        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = (
            point_to_node_partition(
                ref_points_f,
                ref_points_c,
                self.num_points_in_patch,
            )
        )
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = (
            point_to_node_partition(
                src_points_f,
                src_points_c,
                self.num_points_in_patch,
            )
        )
        ref_padded_points_f = torch.cat(
            [ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0
        )
        src_padded_points_f = torch.cat(
            [src_points_f, torch.zeros_like(src_points_f[:1])], dim=0
        )
        ref_node_knn_points = index_select(
            ref_padded_points_f, ref_node_knn_indices, dim=0
        )
        src_node_knn_points = index_select(
            src_padded_points_f, src_node_knn_indices, dim=0
        )

        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
            ref_points_c,
            src_points_c,
            ref_node_knn_points,
            src_node_knn_points,
            transform,
            self.matching_radius,
            ref_masks=ref_node_masks,
            src_masks=src_node_masks,
            ref_knn_masks=ref_node_knn_masks,
            src_knn_masks=src_node_knn_masks,
        ) if transform is not None else (
            torch.empty((0, 2), dtype=torch.long, device=points.device),
            points.new_empty((0,)),
        )
        output_dict.update(
            gt_node_corr_indices=gt_node_corr_indices,
            gt_node_corr_overlaps=gt_node_corr_overlaps,
            ref_node_masks=ref_node_masks,
            src_node_masks=src_node_masks,
        )

        feature_pyramid = self.backbone(features, data_dict)
        fine_features = feature_pyramid[0]
        if self.fine_backbone is not None:
            fine_features = self.fine_backbone(features, data_dict)[0]
        coarse_features = feature_pyramid[-1]
        ref_feats_f = fine_features[:ref_length_f]
        src_feats_f = fine_features[ref_length_f:]
        ref_feats_c = coarse_features[:ref_length_c]
        src_feats_c = coarse_features[ref_length_c:]
        encoded_coarse = self._encode_coarse_features(
            ref_points_c,
            src_points_c,
            ref_feats_c,
            src_feats_c,
        )
        if self.topology_attention_enabled:
            ref_feats_c, src_feats_c, conditioned_diagnostics = encoded_coarse
        else:
            ref_feats_c, src_feats_c = encoded_coarse
            conditioned_diagnostics = None
        (
            ref_feats_c,
            src_feats_c,
            ref_probability,
            src_probability,
            ref_weight,
            src_weight,
            rtor_diagnostics,
        ) = self._apply_rtor(
            ref_points_c,
            src_points_c,
            ref_feats_c,
            src_feats_c,
            conditioned_diagnostics,
        )

        # The partial reference remains intact. Only the complete source is
        # focused at inference; training retains all nodes for GT supervision.
        focused_ref_node_masks = ref_node_masks
        focused_src_node_masks = src_node_masks
        if not self.training and self.focus_enabled and self.rtor_enabled:
            focused_src_node_masks = select_overlap_region(
                src_probability,
                src_node_masks,
                self.focus_threshold,
                self.focus_min_superpoints,
                self.focus_max_ratio,
                self.focus_min_spread,
            )

        output_dict.update(rtor_diagnostics)
        output_dict.update(
            ref_overlap_probability=ref_probability,
            src_overlap_probability=src_probability,
            ref_overlap_weight=ref_weight,
            src_overlap_weight=src_weight,
            focused_ref_node_masks=focused_ref_node_masks,
            focused_src_node_masks=focused_src_node_masks,
            num_focused_ref_nodes=focused_ref_node_masks.sum(),
            num_focused_src_nodes=focused_src_node_masks.sum(),
            ref_feats_c=ref_feats_c,
            src_feats_c=src_feats_c,
            ref_feats_f=ref_feats_f,
            src_feats_f=src_feats_f,
        )

        ranking_scores = None
        if self.coarse_ranker is not None:
            ranking_scores, ranking_diagnostics = self.coarse_ranker(
                ref_points_c,
                src_points_c,
                ref_feats_c,
                src_feats_c,
                ref_probability,
                src_probability,
                focused_ref_node_masks,
                focused_src_node_masks,
            )
            output_dict.update(ranking_diagnostics)

        with torch.no_grad():
            matching_ref_weights = ref_weight if ranking_scores is None else None
            matching_src_weights = src_weight if ranking_scores is None else None
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = (
                self.coarse_matching(
                    ref_feats_c,
                    src_feats_c,
                    focused_ref_node_masks,
                    focused_src_node_masks,
                    ref_weights=matching_ref_weights,
                    src_weights=matching_src_weights,
                    precomputed_scores=ranking_scores,
                )
            )
            output_dict.update(
                ref_node_corr_indices=ref_node_corr_indices,
                src_node_corr_indices=src_node_corr_indices,
                node_corr_scores=node_corr_scores,
                node_overlap_scores=torch.sqrt(
                    ref_probability[ref_node_corr_indices]
                    * src_probability[src_node_corr_indices]
                ),
            )
            if self.training:
                targets = self.coarse_target(gt_node_corr_indices, gt_node_corr_overlaps)
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = mix_coarse_proposals(
                    *targets, ref_node_corr_indices, src_node_corr_indices, node_corr_scores,
                    self.predicted_coarse_ratio,
                )
        output_dict['fine_ref_node_corr_indices'] = ref_node_corr_indices
        output_dict['fine_src_node_corr_indices'] = src_node_corr_indices
        output_dict['predicted_coarse_ratio'] = features.new_tensor(self.predicted_coarse_ratio)

        ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]
        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]
        ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]
        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]
        ref_padded_feats_f = torch.cat(
            [ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0
        )
        src_padded_feats_f = torch.cat(
            [src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0
        )
        ref_node_corr_knn_feats = index_select(
            ref_padded_feats_f,
            ref_node_corr_knn_indices,
            dim=0,
        )
        src_node_corr_knn_feats = index_select(
            src_padded_feats_f,
            src_node_corr_knn_indices,
            dim=0,
        )
        output_dict.update(
            ref_node_corr_knn_points=ref_node_corr_knn_points,
            src_node_corr_knn_points=src_node_corr_knn_points,
            ref_node_corr_knn_masks=ref_node_corr_knn_masks,
            src_node_corr_knn_masks=src_node_corr_knn_masks,
        )

        if self.a3_enabled:
            ref_node_corr_knn_feats, src_node_corr_knn_feats = self.fine_local_refiner(
                ref_node_corr_knn_feats,
                src_node_corr_knn_feats,
                ref_node_corr_knn_points,
                src_node_corr_knn_points,
                ref_node_corr_knn_masks,
                src_node_corr_knn_masks,
            )
        matching_scores = torch.einsum(
            'bnd,bmd->bnm',
            ref_node_corr_knn_feats,
            src_node_corr_knn_feats,
        )
        matching_scores = matching_scores / fine_features.shape[1] ** 0.5
        matching_scores = self.optimal_transport(
            matching_scores,
            ref_node_corr_knn_masks,
            src_node_corr_knn_masks,
        )
        output_dict['matching_scores'] = matching_scores

        with torch.no_grad():
            lgr_scores = matching_scores
            if not self.fine_matching.use_dustbin:
                lgr_scores = lgr_scores[:, :-1, :-1]
            ref_corr_points, src_corr_points, corr_scores, estimated_transform = (
                self.fine_matching(
                    ref_node_corr_knn_points,
                    src_node_corr_knn_points,
                    ref_node_corr_knn_masks,
                    src_node_corr_knn_masks,
                    lgr_scores,
                    node_corr_scores,
                )
            )
        output_dict.update(
            ref_corr_points=ref_corr_points,
            src_corr_points=src_corr_points,
            corr_scores=corr_scores,
            estimated_transform=estimated_transform,
        )
        return output_dict


def create_model(config):
    return GeoTransformer(config)


__all__ = ['GeoTransformer', 'create_model']
