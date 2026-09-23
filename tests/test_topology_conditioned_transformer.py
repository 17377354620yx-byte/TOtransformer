import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

from geotransformer.modules.geotransformer.geotransformer import GeometricTransformer
from geotransformer.modules.liver.registration_model import GeoTransformer


ROOT = Path(__file__).resolve().parents[1]


def _make_transformer(**overrides):
    options = dict(
        input_dim=16,
        output_dim=8,
        hidden_dim=16,
        num_heads=4,
        blocks=['self', 'cross', 'self', 'cross'],
        sigma_d=0.2,
        sigma_a=15,
        angle_k=3,
        dropout=None,
    )
    options.update(overrides)
    return GeometricTransformer(**options)


def _inputs():
    return (
        torch.randn(1, 8, 3),
        torch.randn(1, 11, 3),
        torch.randn(1, 8, 16),
        torch.randn(1, 11, 16),
    )


def _load_config_module():
    path = ROOT / 'experiments' / 'geotransformer.p2p_liver' / 'config.py'
    spec = importlib.util.spec_from_file_location('togg_config_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_zero_initialized_conditioner_is_exact_baseline():
    torch.manual_seed(41)
    baseline = _make_transformer().eval()
    conditioned = _make_transformer(
        topology_attention=True,
        overlap_attention=True,
        topology_num_neighbors=3,
    ).eval()
    incompatibility = conditioned.load_state_dict(
        baseline.state_dict(), strict=False
    )
    assert incompatibility.unexpected_keys == []
    assert incompatibility.missing_keys
    assert all(key.startswith('conditioner.') for key in incompatibility.missing_keys)

    inputs = _inputs()
    with torch.no_grad():
        expected = baseline(*inputs)
        actual = conditioned(*inputs)
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
    torch.testing.assert_close(
        actual[2]['ref_overlap_logits'], torch.zeros(1, 8)
    )
    torch.testing.assert_close(
        actual[2]['src_overlap_logits'], torch.zeros(1, 11)
    )


def test_conditioned_transformer_is_rigid_invariant():
    torch.manual_seed(43)
    model = _make_transformer(
        topology_attention=True,
        overlap_attention=True,
        topology_num_neighbors=3,
    ).eval()
    # Make the new branches observable while retaining deterministic behavior.
    for projection in model.conditioner.topology_bias:
        torch.nn.init.normal_(projection[-1].weight, std=0.02)
    torch.nn.init.normal_(model.conditioner.overlap_head[-1].weight, std=0.02)
    model.conditioner.overlap_beta.data.fill_(0.1)

    ref_points, src_points, ref_features, src_features = _inputs()
    rotation = torch.linalg.qr(torch.randn(3, 3)).Q
    if torch.det(rotation) < 0:
        rotation[:, 0] *= -1
    translation = torch.randn(3)
    transformed = (
        ref_points @ rotation.T + translation,
        src_points @ rotation.T + translation,
        ref_features,
        src_features,
    )
    with torch.no_grad():
        first = model(ref_points, src_points, ref_features, src_features)
        second = model(*transformed)
    for first_value, second_value in zip(first[:2], second[:2]):
        torch.testing.assert_close(first_value, second_value, atol=3e-5, rtol=3e-5)
    for key in first[2]:
        torch.testing.assert_close(
            first[2][key], second[2][key], atol=3e-5, rtol=3e-5
        )


def test_topology_and_overlap_predictor_receive_gradients():
    torch.manual_seed(47)
    model = _make_transformer(
        topology_attention=True,
        overlap_attention=True,
        topology_num_neighbors=3,
    ).train()
    ref_output, src_output, diagnostics = model(*_inputs())
    feature_weights_ref = torch.randn_like(ref_output)
    feature_weights_src = torch.randn_like(src_output)
    feature_loss = (
        (ref_output * feature_weights_ref).sum()
        + (src_output * feature_weights_src).sum()
    )
    overlap_loss = F.binary_cross_entropy_with_logits(
        diagnostics['ref_overlap_logits'],
        torch.randint(0, 2, diagnostics['ref_overlap_logits'].shape).float(),
    ) + F.binary_cross_entropy_with_logits(
        diagnostics['src_overlap_logits'],
        torch.randint(0, 2, diagnostics['src_overlap_logits'].shape).float(),
    )
    (feature_loss + overlap_loss).backward()

    topology_grad = model.conditioner.topology_bias[0][-1].weight.grad
    overlap_grad = model.conditioner.overlap_head[-1].weight.grad
    assert topology_grad is not None and torch.isfinite(topology_grad).all()
    assert overlap_grad is not None and torch.isfinite(overlap_grad).all()
    assert torch.count_nonzero(topology_grad) > 0
    assert torch.count_nonzero(overlap_grad) > 0


def test_nonuniform_overlap_prior_trains_zero_initialized_beta():
    torch.manual_seed(49)
    model = _make_transformer(
        topology_attention=True,
        overlap_attention=True,
        topology_num_neighbors=3,
    ).train()
    # Emulate the state after overlap supervision has made probabilities
    # non-uniform; beta itself remains at its required zero initialization.
    torch.nn.init.normal_(model.conditioner.overlap_head[-1].weight, std=0.1)
    ref_output, src_output, _ = model(*_inputs())
    loss = (
        ref_output * torch.randn_like(ref_output)
    ).sum() + (
        src_output * torch.randn_like(src_output)
    ).sum()
    loss.backward()
    beta_grad = model.conditioner.overlap_beta.grad
    assert beta_grad is not None and torch.isfinite(beta_grad).all()
    assert torch.count_nonzero(beta_grad) > 0


def test_togg_profiles_are_explicit_and_do_not_build_legacy_rtor():
    config = _load_config_module()
    phase1_cfg = config.make_cfg(
        architecture='rtor_a3', interaction_profile='togg_phase1'
    )
    phase2_cfg = config.make_cfg(
        architecture='rtor_a3', interaction_profile='togg_phase2'
    )
    assert phase1_cfg.ablation.topology_attention
    assert not phase1_cfg.ablation.overlap_cross_attention
    assert not phase1_cfg.ablation.overlap_supervision_enabled
    assert phase2_cfg.ablation.topology_attention
    assert phase2_cfg.ablation.overlap_cross_attention
    assert phase2_cfg.ablation.overlap_supervision_enabled
    assert not phase2_cfg.overlap_selection.enabled

    phase2_model = GeoTransformer(phase2_cfg)
    assert phase2_model.topology_overlap_refiner is None
    assert phase2_model.transformer.conditioner is not None


def test_rtor_off_keeps_original_parameter_tree():
    config = _load_config_module()
    cfg = config.make_cfg(
        architecture='geotransformer', interaction_profile='legacy'
    )
    model = GeoTransformer(cfg)
    assert model.transformer.conditioner is None
    assert model.topology_overlap_refiner is None
    assert not any('conditioner.' in key for key in model.state_dict())


def test_legacy_rtor_checkpoint_has_only_expected_phase2_incompatibilities():
    config = _load_config_module()
    legacy = GeoTransformer(
        config.make_cfg(architecture='rtor_a3', interaction_profile='legacy')
    )
    phase2 = GeoTransformer(
        config.make_cfg(architecture='rtor_a3', interaction_profile='togg_phase2')
    )
    incompatibility = phase2.load_state_dict(legacy.state_dict(), strict=False)
    assert incompatibility.missing_keys
    assert all(
        key.startswith('transformer.conditioner.')
        for key in incompatibility.missing_keys
    )
    assert incompatibility.unexpected_keys
    assert all(
        key.startswith('topology_overlap_refiner.')
        for key in incompatibility.unexpected_keys
    )
