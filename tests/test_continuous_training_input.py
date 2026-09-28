from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math

import numpy as np
import pytest
import torch

from phaseset_core.continuous_capture import prepare_continuous_capture
from phaseset_core.continuous_phase_cache import write_continuous_phase_cache
from phaseset_core.continuous_training_input import ContinuousTrainingInput, shared_yaw_capture
from phaseset_core.directional_phase import LocalPhaseConfig, continuous_directional_phase_field
from phaseset_core.pipeline import PrivateCaptureArrays
from phaseset_core.temporal_coordination import TemporalIncidenceEncoder


def _input(tmp_path):
    frames, actors = 900, 3
    t = np.arange(frames) / 30
    joints = np.zeros((frames, actors, 22, 3), dtype=np.float32)
    for actor in range(actors):
        joints[:, actor, :, 0] = (actor + np.sin(2 * np.pi * 1.125 * t + actor))[:, None]
        joints[:, actor, :, 2] = (actor * 0.3 + np.cos(2 * np.pi * 0.75 * t))[:, None]
        joints[:, actor, 1:, 0] += np.sin(2 * np.pi * 2.53125 * t)[:, None]
        joints[:, actor, 1:, 2] += 0.2 * np.cos(2 * np.pi * 2.53125 * t + 0.6)[:, None]
    tracked = np.ones(joints.shape[:-1], dtype=np.bool_)
    tracked[300:360] = False  # Keep the rejected middle interval on the timeline.
    capture = prepare_continuous_capture(
        PrivateCaptureArrays(
            joints, tracked, tuple(bytes([i + 1]) * 32 for i in range(actors)), "a" * 64
        )
    )
    body, physical = tmp_path / "body", tmp_path / "physical"
    body.mkdir()
    arrays = {
        "skeletons.npy": capture.skeletons,
        "track_mask.npy": capture.track_mask,
        "actor_commitments.npy": np.array([list(x) for x in capture.actor_commitments], np.uint8),
    }
    for name, value in arrays.items():
        np.save(body / name, value, allow_pickle=False)
    prepared = {
        "shape": list(capture.skeletons.shape),
        "source_sha256": capture.source_sha256,
        "group_commitment": capture.group_commitment.hex(),
        "artifact_sha256s": {
            name: hashlib.sha256((body / name).read_bytes()).hexdigest() for name in arrays
        },
        "window_decisions": [asdict(x) for x in capture.decisions],
        "trailing_source_frames": capture.trailing_source_frames,
        "reference_frame_20fps": capture.reference_frame,
        "group_center": capture.group_center.tolist(),
        "augmentation_yaw": 0.0,
    }
    (body / "metadata.json").write_text(json.dumps(prepared))
    field = write_continuous_phase_cache(capture, physical)
    record = {
        "source_sha256": capture.source_sha256,
        "patches": field.patch_count,
        "prepared_record": prepared,
        "artifact_sha256s": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in physical.iterdir()
        },
    }
    source = ContinuousTrainingInput.load(
        body, physical, cache_record=record, energy_floors=np.full(6, 1e-10, np.float64)
    )
    return source, body, physical, record


def test_zero_yaw_reuses_readonly_cache_and_source_capture_positive(tmp_path):
    source, _, _, _ = _input(tmp_path)
    view = source.view(allow_shared_yaw=False)
    assert view.capture is source.capture and view.phase_field is source.cached_field
    assert view.reused_physical_cache
    assert view.positive_capture_key == source.capture.source_sha256
    assert view.positive_capture_key != source.capture.group_commitment.hex()
    assert not view.phase_field.actor_features.flags.writeable


def test_shared_yaw_rotates_whole_capture_without_recenter_or_timeline_reset(tmp_path):
    source, _, _, _ = _input(tmp_path)
    view = source.view(yaw_delta=math.pi / 2, allow_shared_yaw=True)
    assert not view.reused_physical_cache
    original, rotated = source.capture, view.capture
    np.testing.assert_allclose(rotated.skeletons[..., 0], original.skeletons[..., 2], atol=1e-7)
    np.testing.assert_allclose(rotated.skeletons[..., 2], -original.skeletons[..., 0], atol=1e-7)
    np.testing.assert_array_equal(rotated.track_mask, original.track_mask)
    np.testing.assert_array_equal(rotated.group_center, original.group_center)
    assert rotated.decisions == original.decisions and not rotated.decisions[1].accepted
    assert rotated.reference_frame == original.reference_frame and rotated.frame_count == 600
    assert rotated.actor_commitments == original.actor_commitments
    assert rotated.source_sha256 == original.source_sha256
    assert not rotated.skeletons[~rotated.track_mask].view(np.uint32).any()
    assert not rotated.skeletons.flags.writeable


def test_yaw_view_recomputes_rms_roots_and_physics_bitwise(tmp_path):
    source, _, _, _ = _input(tmp_path)
    view = source.view(yaw_delta=0.731, allow_shared_yaw=True)
    reference = continuous_directional_phase_field(
        view.capture, energy_floors=source.cached_field.energy_floors
    )
    for name in ("actor_features", "actor_patch_mask", "root_positions", "root_patch_mask"):
        np.testing.assert_array_equal(getattr(view.phase_field, name), getattr(reference, name))
    for name in ("responses", "response_masks", "response_centers"):
        for actual, expected in zip(
            getattr(view.phase_field, name), getattr(reference, name), strict=True
        ):
            np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(view.phase_field.actor_features, source.cached_field.actor_features)
    # RMS(x cos(yaw)+z sin(yaw)) cannot be obtained by rotating the RMS vector.
    old_rms = source.cached_field.actor_features[..., 66:].reshape(3, -1, 22, 3)
    naive = math.cos(0.731) * old_rms[..., 0] + math.sin(0.731) * old_rms[..., 2]
    new_rms = view.phase_field.actor_features[..., 66:].reshape(3, -1, 22, 3)[..., 0]
    assert float(np.abs(new_rms - naive).max()) > 0.01
    np.testing.assert_array_equal(view.phase_field.energy_floors, source.cached_field.energy_floors)


def test_yaw_full_trajectory_checkpointed_backward_equals_direct_field(tmp_path):
    source, _, _, _ = _input(tmp_path)
    view = source.view(yaw_delta=-0.4, allow_shared_yaw=True)
    reference = continuous_directional_phase_field(
        view.capture, energy_floors=view.phase_field.energy_floors
    )
    torch.manual_seed(1729)
    model = TemporalIncidenceEncoder(width=8)
    results, gradients = [], []
    for field in (view.phase_field, reference):
        model.zero_grad(set_to_none=True)
        output = model(field)
        results.append(output.embedding.detach().clone())
        output.embedding[0].backward()
        gradients.append(
            [p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()]
        )
    assert torch.equal(*results)
    for a, b in zip(*gradients, strict=True):
        assert (a is None and b is None) or torch.equal(a, b)


@pytest.mark.parametrize("angle", [True, float("nan"), float("inf"), "1"])
def test_invalid_yaw_is_rejected(tmp_path, angle):
    source, _, _, _ = _input(tmp_path)
    with pytest.raises(ValueError, match="finite"):
        shared_yaw_capture(source.capture, yaw_delta=angle)


def test_unapproved_yaw_or_response_budget_never_falls_back_to_stale_cache(tmp_path):
    source, _, _, _ = _input(tmp_path)
    with pytest.raises(ValueError, match="not admitted"):
        source.view(yaw_delta=0.5, allow_shared_yaw=False)
    with pytest.raises(MemoryError, match="RESOURCE_LIMIT"):
        source.view(
            yaw_delta=0.5, allow_shared_yaw=True, config=LocalPhaseConfig(max_response_bytes=1)
        )
    with pytest.raises(ValueError, match="cannot change"):
        source.view(allow_shared_yaw=True, config=LocalPhaseConfig(patch_frames=20))
    with pytest.raises(ValueError, match="frozen physical"):
        source.view(
            yaw_delta=0.5,
            allow_shared_yaw=True,
            config=LocalPhaseConfig(patch_frames=20),
        )


def test_cache_drift_and_wrong_prepared_binding_are_rejected(tmp_path):
    _, body, physical, record = _input(tmp_path)
    wrong = {**record, "source_sha256": "b" * 64}
    with pytest.raises(ValueError, match="different captures"):
        ContinuousTrainingInput.load(body, physical, cache_record=wrong, energy_floors=np.zeros(6))
    with (physical / "actor-features.npy").open("ab") as stream:
        stream.write(b"drift")
    with pytest.raises(ValueError, match="artifact differs"):
        ContinuousTrainingInput.load(body, physical, cache_record=record, energy_floors=np.zeros(6))


def test_nonzero_cached_preparation_is_not_assumed_unaugmented(tmp_path):
    _, body, physical, record = _input(tmp_path)
    wrong = {**record, "prepared_record": {**record["prepared_record"], "augmentation_yaw": 1.0}}
    with pytest.raises(ValueError, match="unaugmented"):
        ContinuousTrainingInput.load(body, physical, cache_record=wrong, energy_floors=np.zeros(6))
