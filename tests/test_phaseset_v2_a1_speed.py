"""A1 speed-only physical input and cache separation, not research scores."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from phaseset_core.continuous_capture import prepare_continuous_capture
from phaseset_core.continuous_phase_cache import (
    load_continuous_phase_cache,
    write_continuous_phase_cache,
)
from phaseset_core.contracts import PreparedGroupBatch, group_commitment
from phaseset_core.directional_calibration import (
    DirectionalCalibrationCapture,
    SPEED_CALIBRATION_METHOD,
    fit_directional_energy_floors,
)
from phaseset_core.directional_phase import (
    LocalPhaseConfig,
    continuous_directional_phase_field,
    directional_phase_fields,
    local_pair_chunk,
    signed_joint_velocity,
    speed_only_joint_velocity,
)
from phaseset_core.temporal_coordination import TemporalIncidenceEncoder
from phaseset_core.pipeline import PrivateCaptureArrays
from phaseset_core.speed_calibration import SpeedFloorReceipt


def _opposite_batch() -> PreparedGroupBatch:
    frames = 200
    skeletons = np.zeros((1, 2, frames, 22, 3), dtype=np.float32)
    signal = np.sin(2 * np.pi * 1.125 * np.arange(frames) / 20).astype(np.float32)
    skeletons[0, 0, :, 1:, 0] = signal[:, None]
    skeletons[0, 1, :, 1:, 0] = -signal[:, None]
    keys = (b"a" * 32, b"b" * 32)
    return PreparedGroupBatch(
        skeletons,
        np.ones((1, 2), dtype=np.bool_),
        np.ones((1, frames), dtype=np.bool_),
        np.ones((1, 2, frames, 22), dtype=np.bool_),
        (keys,),
        (group_commitment(keys),),
    )


def test_a1_speed_removes_direction_before_morlet_but_preserves_vector_energy() -> None:
    batch = _opposite_batch()
    signed, signed_mask = signed_joint_velocity(batch)
    speed, speed_mask = speed_only_joint_velocity(batch)
    np.testing.assert_array_equal(signed_mask, speed_mask)
    np.testing.assert_array_equal(speed[0, 0], speed[0, 1])
    np.testing.assert_allclose(
        np.sum(speed**2, axis=-1), np.sum(signed**2, axis=-1), rtol=1e-14, atol=1e-14
    )
    assert np.any(signed[0, 0] != signed[0, 1])
    floors = np.zeros(6, dtype=np.float64)
    full = directional_phase_fields(batch, energy_floors=floors)[0]
    a1 = directional_phase_fields(batch, energy_floors=floors, velocity_mode="speed_only")[0]
    assert (full.velocity_mode, a1.velocity_mode) == ("signed_vector", "speed_only")
    assert any(np.any(row[0] != row[1]) for row in full.responses)
    for row in a1.responses:
        np.testing.assert_array_equal(row[0], row[1])
    np.testing.assert_array_equal(full.root_positions, a1.root_positions)


def test_a1_actor_permutation_holey_padding_and_pair_reversal_are_exact() -> None:
    reference = _opposite_batch()
    skeletons = np.zeros((1, 5, 200, 22, 3), dtype=np.float32)
    track = np.zeros((1, 5, 200, 22), dtype=np.bool_)
    skeletons[0, 1] = reference.skeletons[0, 1]
    skeletons[0, 3] = reference.skeletons[0, 0]
    track[0, 1] = reference.track_mask[0, 1]
    track[0, 3] = reference.track_mask[0, 0]
    candidate = PreparedGroupBatch(
        skeletons,
        np.array([[False, True, False, True, False]], dtype=np.bool_),
        np.ones((1, 200), dtype=np.bool_),
        track,
        ((None, b"b" * 32, None, b"a" * 32, None),),
        reference.group_commitments,
    )
    floors = np.zeros(6, dtype=np.float64)
    left = directional_phase_fields(reference, energy_floors=floors, velocity_mode="speed_only")[0]
    right = directional_phase_fields(candidate, energy_floors=floors, velocity_mode="speed_only")[0]
    for a, b in zip(left.responses, right.responses, strict=True):
        np.testing.assert_array_equal(a, b)
    pair = local_pair_chunk(left, 0, 1)
    np.testing.assert_array_equal(pair.features, local_pair_chunk(right, 0, 1).features)
    reverse = pair.reverse_features()
    np.testing.assert_array_equal(reverse[..., 0], pair.features[..., 0])
    np.testing.assert_array_equal(reverse[..., 1], -pair.features[..., 1])
    np.testing.assert_array_equal(reverse[..., 3], pair.features[..., 4])
    np.testing.assert_array_equal(reverse[..., 4], pair.features[..., 3])


def test_a1_k2_topology_gradient_is_positive_zero_and_pair_only_equal() -> None:
    torch.manual_seed(1729)
    full = TemporalIncidenceEncoder(32)
    pair = copy.deepcopy(full)
    pair.use_topology = False
    source = directional_phase_fields(
        _opposite_batch(), energy_floors=np.zeros(6, np.float64), velocity_mode="speed_only"
    )[0]
    result, control = full(source), pair(source)
    assert torch.equal(result.embedding, control.embedding)
    assert torch.count_nonzero(result.topology_nodes) == 0
    assert not bool(torch.signbit(result.topology_nodes).any())
    result.embedding.sum().backward()
    topology = list(full.node_mlp.parameters()) + list(full.node_temporal.parameters())
    topology += list(full.edge_delta.parameters()) + [full.topology_scale]
    for parameter in topology:
        assert parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0
        assert not bool(torch.signbit(parameter.grad).any())


def _continuous_capture():
    frames = 600
    joints = np.zeros((frames, 2, 22, 3), dtype=np.float32)
    signal = np.sin(2 * np.pi * 1.125 * np.arange(frames) / 30).astype(np.float32)
    joints[:, 0, 1:, 0] = signal[:, None]
    joints[:, 1, 1:, 0] = -signal[:, None]
    return prepare_continuous_capture(
        PrivateCaptureArrays(
            joints,
            np.ones(joints.shape[:-1], dtype=np.bool_),
            (b"a" * 32, b"b" * 32),
            "a" * 64,
        )
    )


def test_a1_has_separate_complete_cache_and_training_only_floor_method(tmp_path) -> None:
    capture = _continuous_capture()
    floors = np.zeros(6, dtype=np.float64)
    ram = continuous_directional_phase_field(
        capture, energy_floors=floors, velocity_mode="speed_only"
    )
    cache = tmp_path / "a1-physical"
    raw = write_continuous_phase_cache(capture, cache, velocity_mode="speed_only")
    mapped = load_continuous_phase_cache(
        cache,
        expected_source_sha256=capture.source_sha256,
        energy_floors=floors,
        expected_velocity_mode="speed_only",
    )
    assert raw.velocity_mode == mapped.velocity_mode == "speed_only"
    for expected, actual in zip(ram.responses, mapped.responses, strict=True):
        np.testing.assert_array_equal(expected.view(np.uint64), actual.view(np.uint64))
    with pytest.raises(ValueError, match="schema"):
        load_continuous_phase_cache(
            cache, expected_source_sha256=capture.source_sha256, energy_floors=floors
        )
    fit = fit_directional_energy_floors(
        (DirectionalCalibrationCapture(capture.source_sha256, "C01", raw),),
        training_components=("C01",),
        expected_capture_ids=(capture.source_sha256,),
    )
    assert fit.method == SPEED_CALIBRATION_METHOD
    receipt = SpeedFloorReceipt.from_fit(
        fit,
        population_name="pilot",
        training_source_manifest_sha256="b" * 64,
        config=raw.config,
    )
    assert receipt.population_name == "pilot" and len(receipt.sha256) == 64
    with pytest.raises(ValueError, match="actual training fit"):
        SpeedFloorReceipt.from_fit(
            fit,
            population_name="pilot",
            training_source_manifest_sha256="b" * 64,
            config=LocalPhaseConfig(patch_frames=20),
        )
    with pytest.raises(ValueError, match="frontends"):
        fit_directional_energy_floors(
            (
                DirectionalCalibrationCapture(capture.source_sha256, "C01", raw),
                DirectionalCalibrationCapture("other", "C01", continuous_directional_phase_field(capture, energy_floors=floors)),
            ),
            training_components=("C01",),
            expected_capture_ids=(capture.source_sha256, "other"),
        )
    with pytest.raises(ValueError, match="physical configurations"):
        fit_directional_energy_floors(
            (
                DirectionalCalibrationCapture(capture.source_sha256, "C01", raw),
                DirectionalCalibrationCapture(
                    "other", "C01",
                    continuous_directional_phase_field(
                        capture,
                        energy_floors=floors,
                        velocity_mode="speed_only",
                        config=LocalPhaseConfig(patch_frames=20),
                    ),
                ),
            ),
            training_components=("C01",),
            expected_capture_ids=(capture.source_sha256, "other"),
        )
