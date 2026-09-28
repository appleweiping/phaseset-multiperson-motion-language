"""A6 data-free relation-field and model-seam checks, not trained retrieval."""

from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from phaseset_core.continuous_retrieval import ContinuousRetrievalSystem
from phaseset_core.dct_calibration import DCT_FLOOR_SCHEMA, DctFloorReceipt, dct_config_sha256
from phaseset_core.dct_relations import DctRelationField, local_dct_pair_chunk
from phaseset_core.directional_phase import (
    LocalPhaseConfig,
    continuous_directional_phase_field,
    true_mean_difference_dct_features,
)
from phaseset_core.temporal_coordination import TemporalIncidenceEncoder
from phaseset_core.training import BaseRetrievalSystem
from test_continuous_retrieval import _MockB2
from test_continuous_training_input import _input
from test_phaseset_frozen_clip_text import _commitment, _fixture


def _dct_view(tmp_path):
    source, _, _, _ = _input(tmp_path)
    view = source.view(allow_shared_yaw=False)
    floors = _receipt(view.phase_field.config)
    return view, DctRelationField.from_capture(
        view.capture, view.phase_field, dct_floor_receipt=floors
    ), floors


def _receipt(config, values=None):
    return DctFloorReceipt(
        DCT_FLOOR_SCHEMA,
        "main",
        ("C01",),
        "a" * 64,
        dct_config_sha256(config),
        np.zeros(6, dtype=np.float64) if values is None else values,
    )


def test_a6_receipt_rejects_morlet_or_test_population_and_config_drift():
    config = LocalPhaseConfig()
    receipt = _receipt(config)
    receipt.require_population("main", ("C01",))
    assert receipt.sha256 == _receipt(config).sha256
    with pytest.raises(ValueError, match="population"):
        receipt.require_population("fold_0", ("C01",))
    with pytest.raises(ValueError, match="config"):
        receipt.require_config(LocalPhaseConfig(hop_frames=10))
    with pytest.raises(ValueError, match="population"):
        DctFloorReceipt(
            DCT_FLOOR_SCHEMA,
            "main",
            ("C09",),
            "a" * 64,
            dct_config_sha256(config),
            np.zeros(6, np.float64),
        )
    with pytest.raises(ValueError, match="schema"):
        DctFloorReceipt(
            "phaseset-morlet-floor-v1",
            "main",
            ("C01",),
            "a" * 64,
            dct_config_sha256(config),
            np.zeros(6, np.float64),
        )


def test_a6_edge_features_are_real_signal_dct_not_cached_morlet(tmp_path):
    view, field, floors = _dct_view(tmp_path)
    chunk = local_dct_pair_chunk(field, 0, field.pair_count)
    first = chunk.endpoints[0]
    start, stop = field.intervals[0]
    direct, observed = true_mean_difference_dct_features(
        field.velocities[first[0], start:stop],
        field.velocities[first[1], start:stop],
        field.velocity_mask[first[0], start:stop],
        field.velocity_mask[first[1], start:stop],
    )
    np.testing.assert_array_equal(chunk.features[0, 0, :, :6], direct[:, :6])
    np.testing.assert_array_equal(chunk.support_mask[0, 0], observed)
    np.testing.assert_array_equal(chunk.phase_mask[0, 0], chunk.features[0, 0, :, 6] > 0)
    assert bool(chunk.phase_mask.any())
    for name in ("actor_features", "actor_patch_mask", "root_positions", "root_patch_mask"):
        np.testing.assert_array_equal(getattr(field, name), getattr(view.phase_field, name))

    # The phase response arrays can be absent and A6 still computes the same
    # edge features from admitted signed velocities plus shared actor context.
    without_morlet = replace(view.phase_field, responses=(), response_masks=(), response_centers=())
    independent = DctRelationField.from_capture(
        view.capture, without_morlet, dct_floor_receipt=floors
    )
    np.testing.assert_array_equal(local_dct_pair_chunk(independent, 0, 3).features, chunk.features)
    forged_context = replace(
        view.phase_field,
        actor_features=np.zeros_like(view.phase_field.actor_features),
        root_positions=np.zeros_like(view.phase_field.root_positions),
    )
    recomputed = DctRelationField.from_capture(
        view.capture, forged_context, dct_floor_receipt=floors
    )
    np.testing.assert_array_equal(recomputed.actor_features, field.actor_features)
    np.testing.assert_array_equal(recomputed.root_positions, field.root_positions)


def test_a6_reverse_keeps_difference_power_and_swaps_only_endpoint_powers(tmp_path):
    _, field, _ = _dct_view(tmp_path)
    chunk = local_dct_pair_chunk(field, 0, 1)
    reversed_values = chunk.reverse_features()
    np.testing.assert_array_equal(reversed_values[..., :3], chunk.features[..., :3])
    np.testing.assert_array_equal(reversed_values[..., 3], chunk.features[..., 4])
    np.testing.assert_array_equal(reversed_values[..., 4], chunk.features[..., 3])
    np.testing.assert_array_equal(reversed_values[..., 5:], chunk.features[..., 5:])


def test_a6_zero_energy_has_coverage_but_no_calibrated_evidence(tmp_path):
    view, _, floors = _dct_view(tmp_path)
    stationary = replace(view.capture, skeletons=np.zeros_like(view.capture.skeletons))
    matching = continuous_directional_phase_field(
        stationary, energy_floors=view.phase_field.energy_floors
    )
    field = DctRelationField.from_capture(
        stationary, matching, dct_floor_receipt=floors
    )
    chunk = local_dct_pair_chunk(field, 0, 1)
    assert bool(chunk.support_mask.any())
    assert not bool(chunk.phase_mask.any())
    assert not bool(np.count_nonzero(chunk.features[..., 6]))
    model = TemporalIncidenceEncoder(width=8)
    result = model.score_text(field, torch.randn(2, 8))
    assert not bool(result.periodic_support)
    assert not bool(torch.count_nonzero(result.cosine))


def test_a6_checkpointed_score_and_gradient_are_finite(tmp_path):
    _, field, _ = _dct_view(tmp_path)
    torch.manual_seed(1729)
    model = TemporalIncidenceEncoder(width=8)
    text = torch.randn(3, 8, requires_grad=True)
    result = model.score_text(field, text)
    assert bool(result.periodic_support)
    assert result.cosine.shape == (3,)
    result.cosine.sum().backward()
    assert bool(torch.isfinite(text.grad).all())
    assert any(
        parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
        for parameter in model.parameters()
    )


def test_a6_complete_retrieval_seam_uses_dct_field_and_frozen_text(tmp_path):
    (tmp_path / "input").mkdir()
    (tmp_path / "clip").mkdir()
    view, _, receipt = _dct_view(tmp_path / "input")
    clip, _, _, _ = _fixture(tmp_path / "clip")
    text = clip.encode(
        ("people move together", "people move apart"),
        caption_commitments=(_commitment("one"), _commitment("two")),
        batch_size=2,
    )
    model = ContinuousRetrievalSystem(
        BaseRetrievalSystem(_MockB2(), embedding_dim=512),
        relation_kind="true_mean_difference_dct",
        dct_floor_receipt=receipt,
    )
    score = model.score((view,), text)
    assert score.scores.shape == (1, 2)
    assert bool(torch.isfinite(score.scores).all())
    assert bool(score.periodic_support[0])
    assert score.text_receipt_sha256 == text.receipt.sha256


def test_a6_retrieval_seam_requires_independent_floors_and_keeps_head_capacity():
    base = BaseRetrievalSystem(_MockB2(), embedding_dim=512)
    with pytest.raises(ValueError, match="typed independent DCT floor receipt"):
        ContinuousRetrievalSystem(base, relation_kind="true_mean_difference_dct")
    with pytest.raises(ValueError, match="must not enter"):
        ContinuousRetrievalSystem(base, dct_floor_receipt=_receipt(LocalPhaseConfig()))
    phase = ContinuousRetrievalSystem(base)
    floors = np.full(6, 1e-3, np.float64)
    receipt = _receipt(LocalPhaseConfig(), floors)
    dct = ContinuousRetrievalSystem(
        base,
        relation_kind="true_mean_difference_dct",
        dct_floor_receipt=receipt,
    )
    floors[:] = 1e6
    assert torch.equal(
        dct.state_dict()["dct_energy_floors"], torch.full((6,), 1e-3, dtype=torch.float64)
    )
    assert "dct_energy_floors" not in phase.state_dict()
    state_bytes = bytes(dct.state_dict()["_extra_state"].tolist())
    assert json.loads(state_bytes)["dct_floor_receipt_sha256"] == receipt.sha256
    assert all(type(value) is torch.Tensor for value in phase.state_dict().values())
    assert all(type(value) is torch.Tensor for value in dct.state_dict().values())
    ContinuousRetrievalSystem(base).load_state_dict(phase.state_dict())
    ContinuousRetrievalSystem(
        base, relation_kind="true_mean_difference_dct", dct_floor_receipt=receipt
    ).load_state_dict(dct.state_dict())
    with pytest.raises(ValueError, match="checkpoint resume"):
        dct.set_extra_state(dct.get_extra_state().to(torch.int64))
    mutated_state = dct.get_extra_state().clone()
    mutated_state[0] ^= 1
    with pytest.raises(ValueError, match="checkpoint resume"):
        dct.set_extra_state(mutated_state)
    with pytest.raises(ValueError, match="checkpoint resume"):
        phase.set_extra_state(dct.get_extra_state())
    changed = ContinuousRetrievalSystem(
        base,
        relation_kind="true_mean_difference_dct",
        dct_floor_receipt=_receipt(LocalPhaseConfig(), np.full(6, 0.01, np.float64)),
    )
    with pytest.raises(ValueError, match="checkpoint resume"):
        changed.load_state_dict(dct.state_dict())
    phase_count = sum(p.numel() for p in phase.parameters() if p.requires_grad)
    dct_count = sum(p.numel() for p in dct.parameters() if p.requires_grad)
    assert phase_count == dct_count


def test_a6_refuses_patch_longer_than_frozen_dct_contract(tmp_path):
    view, _, floors = _dct_view(tmp_path)
    for invalid in (1, 41):
        mismatched = replace(view.phase_field, config=LocalPhaseConfig(patch_frames=invalid))
        with pytest.raises(ValueError, match="binding"):
            DctRelationField.from_capture(
                view.capture, mismatched, dct_floor_receipt=floors
            )


def test_a6_refuses_same_shape_capture_field_lineage_mismatch(tmp_path):
    view, _, floors = _dct_view(tmp_path)
    assert view.phase_field.source_sha256 == view.capture.source_sha256
    assert view.phase_field.actor_commitments == view.capture.actor_commitments
    for wrong in (
        replace(view.phase_field, source_sha256="b" * 64),
        replace(view.phase_field, actor_commitments=view.phase_field.actor_commitments[::-1]),
        replace(view.phase_field, physical_view_sha256="b" * 64),
    ):
        with pytest.raises(ValueError, match="binding"):
            DctRelationField.from_capture(view.capture, wrong, dct_floor_receipt=floors)
    rotated = replace(view.capture, augmentation_yaw=0.5)
    with pytest.raises(ValueError, match="binding"):
        DctRelationField.from_capture(rotated, view.phase_field, dct_floor_receipt=floors)
    changed = replace(view.capture, skeletons=np.zeros_like(view.capture.skeletons))
    with pytest.raises(ValueError, match="binding"):
        DctRelationField.from_capture(changed, view.phase_field, dct_floor_receipt=floors)
