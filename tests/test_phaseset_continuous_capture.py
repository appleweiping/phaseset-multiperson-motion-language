from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from phaseset_core.continuous_capture import prepare_continuous_capture
from phaseset_core.directional_phase import LocalPhaseConfig, continuous_directional_phase_field
from phaseset_core.pipeline import (
    PrivateCaptureArrays,
    PreparationResourceLimit,
    prepare_window,
)
from phaseset_core.preprocessing import PreprocessingError


def _capture(frames: int = 600, actors: int = 3) -> PrivateCaptureArrays:
    joints = np.zeros((frames, actors, 22, 3), dtype=np.float32)
    t = np.arange(frames, dtype=np.float64) / 30
    for actor in range(actors):
        joints[:, actor, :, 0] = (t + actor * 2)[:, None]
        joints[:, actor, :, 1] = np.arange(22) * 0.01
        joints[:, actor, :, 2] = np.sin(2 * np.pi * 1.6875 * t + actor * 0.5)[:, None]
    return PrivateCaptureArrays(
        joints,
        np.ones(joints.shape[:-1], dtype=np.bool_),
        tuple(bytes([actor + 1]) * 32 for actor in range(actors)),
        "a" * 64,
    )


def test_origin_and_base_window_views_do_not_reset_at_ten_seconds() -> None:
    prepared = prepare_continuous_capture(_capture())
    assert prepared.skeletons.shape == (3, 400, 22, 3)
    assert prepared.reference_frame == 0
    assert np.allclose(prepared.skeletons[:, 0, 0].mean(axis=0), 0, atol=2e-7)
    # A translating group retains ten seconds of travel instead of resetting
    # its centroid to zero at the second window's first frame.
    assert prepared.skeletons[:, 200, 0, 0].mean() > 9.9
    views = prepared.windows()
    assert [sample.source_start_frame for sample in views] == [0, 300]
    np.testing.assert_array_equal(views[1].skeletons, prepared.skeletons[:, 200:])
    assert views[1].skeletons[:, 0, 0, 0].mean() > 9.9


@pytest.mark.parametrize("chunk", [1, 7, 31, 300, 1024])
def test_fir_chunk_halos_preserve_exact_full_capture_outputs(chunk: int) -> None:
    capture = _capture()
    reference = prepare_continuous_capture(capture, resample_chunk_frames=5000)
    actual = prepare_continuous_capture(capture, resample_chunk_frames=chunk)
    np.testing.assert_array_equal(actual.skeletons.view(np.uint32), reference.skeletons.view(np.uint32))
    np.testing.assert_array_equal(actual.track_mask, reference.track_mask)


def test_single_complete_window_retains_legacy_numerical_convention() -> None:
    capture = _capture(300)
    legacy = prepare_window(capture, source_start_frame=0)
    actual = prepare_continuous_capture(capture).windows()[0]
    np.testing.assert_allclose(actual.skeletons, legacy.skeletons, atol=2e-6, rtol=2e-6)
    np.testing.assert_array_equal(actual.track_mask, legacy.track_mask)
    assert actual.window_sha256 == legacy.window_sha256


def test_actor_permutations_are_bitwise_identical_after_canonical_preparation() -> None:
    capture = _capture()
    permutation = [2, 0, 1]
    changed = PrivateCaptureArrays(
        capture.joints[:, permutation],
        capture.track_mask[:, permutation],
        tuple(capture.actor_commitments[index] for index in permutation),
        capture.source_sha256,
    )
    original, permuted = prepare_continuous_capture(capture), prepare_continuous_capture(changed)
    np.testing.assert_array_equal(original.skeletons.view(np.uint32), permuted.skeletons.view(np.uint32))
    assert original.actor_commitments == permuted.actor_commitments


def test_short_gaps_are_interpolated_but_original_observation_mask_is_retained() -> None:
    capture = _capture()
    mask, values = capture.track_mask.copy(), capture.joints.copy()
    mask[100:107, 1] = False
    values[100:107, 1] = np.nan
    prepared = prepare_continuous_capture(replace(capture, joints=values, track_mask=mask))
    assert all(decision.accepted for decision in prepared.decisions)
    assert prepared.decisions[0].missing_actor_frames == 7
    assert not prepared.track_mask[1, 67:72].all()
    assert not np.any(prepared.skeletons[~prepared.track_mask].view(np.uint32))


def test_rejected_intervals_keep_absolute_time_and_do_not_pollute_fir_neighbors() -> None:
    capture = _capture(900)
    mask, values = capture.track_mask.copy(), capture.joints.copy()
    mask[400:408, 1] = False
    values[400:408, 1] = np.nan
    prepared = prepare_continuous_capture(replace(capture, joints=values, track_mask=mask))
    assert [decision.accepted for decision in prepared.decisions] == [True, False, True]
    assert prepared.frame_count == 600
    assert not prepared.track_mask[:, 200:400].any()
    assert not prepared.track_mask[:, 199].any()
    assert not prepared.track_mask[:, 400].any()
    assert prepared.track_mask[:, 410].all()
    assert [window.source_start_frame for window in prepared.windows()] == [0, 600]
    assert not prepared.skeletons[:, 200:400].view(np.uint32).any()


def test_rejected_leading_window_does_not_become_an_artificial_origin() -> None:
    capture = _capture()
    mask = capture.track_mask.copy()
    mask[:8, 0] = False
    prepared = prepare_continuous_capture(replace(capture, track_mask=mask))
    assert prepared.reference_frame >= 210
    assert [sample.source_start_frame for sample in prepared.windows()] == [300]


def test_yaw_is_shared_for_the_whole_capture_and_never_actor_specific() -> None:
    capture = _capture()
    original = prepare_continuous_capture(capture)
    rotated = prepare_continuous_capture(capture, augmentation_yaw=math.pi / 2)
    np.testing.assert_allclose(rotated.skeletons[..., 0], original.skeletons[..., 2], atol=1e-6)
    np.testing.assert_allclose(rotated.skeletons[..., 2], -original.skeletons[..., 0], atol=1e-6)


def test_short_tail_is_declared_not_mixed_into_complete_windows() -> None:
    prepared = prepare_continuous_capture(_capture(637))
    assert prepared.frame_count == 400
    assert prepared.trailing_source_frames == 37


def test_resource_limits_and_no_eligible_windows_are_explicit() -> None:
    with pytest.raises(PreparationResourceLimit, match="RESOURCE_LIMIT"):
        prepare_continuous_capture(_capture(), max_preparation_bytes=1)
    capture = _capture()
    mask = capture.track_mask.copy()
    mask[::2, 0] = False
    with pytest.raises(PreprocessingError, match="no accepted"):
        prepare_continuous_capture(replace(capture, track_mask=mask))


def test_continuous_phase_has_real_support_across_ten_second_seams() -> None:
    prepared = prepare_continuous_capture(_capture())
    field = continuous_directional_phase_field(prepared, energy_floors=np.zeros(6))
    assert field.actor_count == 3 and field.intervals[-1][1] == 400
    assert (180, 220) in field.intervals
    for centers, mask in zip(field.response_centers, field.response_masks, strict=True):
        at_boundary = (centers >= 195) & (centers <= 205)
        assert at_boundary.any() and mask[:, at_boundary].all()
    with pytest.raises(MemoryError, match="RESOURCE_LIMIT"):
        continuous_directional_phase_field(
            prepared, energy_floors=np.zeros(6), config=LocalPhaseConfig(max_response_bytes=1)
        )
