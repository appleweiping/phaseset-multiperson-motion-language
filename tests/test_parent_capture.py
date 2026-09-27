"""Native-episode timeline assembly; all fixtures are data-free world body22."""

from dataclasses import replace

import numpy as np
import pytest

from phaseset_core.continuous_capture import prepare_continuous_capture
from phaseset_core.directional_phase import continuous_directional_phase_field, local_pair_chunk
from phaseset_core.parent_capture import ParentCaptureEpisode, assemble_parent_capture
from phaseset_core.pipeline import PrivateCaptureArrays, PreparationResourceLimit
from phaseset_core.preprocessing import PreprocessingError
from test_phaseset_continuous_capture import _capture


def _episodes(gap=300, tail=0):
    full = _capture(600 + gap + tail)
    episodes = tuple(
        ParentCaptureEpisode(
            PrivateCaptureArrays(
                full.joints[start:stop],
                full.track_mask[start:stop],
                full.actor_commitments,
                digest * 64,
            ),
            1000 + start,
            1000 + stop,
        )
        for start, stop, digest in ((0, 300, "a"), (300 + gap, 600 + gap + tail, "b"))
    )
    return full, episodes


def test_real_gap_retains_raw_world_coordinates_and_shared_parent_preprocessing():
    full, episodes = _episodes()
    assembly = assemble_parent_capture(episodes)
    raw = assembly.capture
    assert raw.frame_count == 900 and assembly.missing_intervals == ((1300, 1600),)
    assert assembly.episode_intervals == ((1000, 1300), (1600, 1900))
    np.testing.assert_array_equal(raw.joints[600:], full.joints[600:])
    assert not raw.track_mask[300:600].any()
    assert not raw.joints[300:600].view(np.uint32).any()
    prepared = prepare_continuous_capture(raw)
    assert prepared.frame_count == 600
    assert [d.accepted for d in prepared.decisions] == [True, False, True]
    assert not prepared.track_mask[:, 200:400].any()
    assert not prepared.skeletons[:, 200:400].view(np.uint32).any()
    assert prepared.skeletons[:, 410, 0, 0].mean() > 20
    child = prepare_continuous_capture(episodes[1].capture)
    assert abs(float(child.skeletons[:, 10, 0, 0].mean())) < 1
    # Independent reference: mark the same hole on one true world trajectory,
    # then use the existing, audited full-capture preprocessing once.
    mask = full.track_mask.copy()
    mask[300:600] = False
    oracle = prepare_continuous_capture(replace(full, track_mask=mask))
    np.testing.assert_array_equal(
        prepared.skeletons.view(np.uint32), oracle.skeletons.view(np.uint32)
    )
    np.testing.assert_array_equal(prepared.track_mask, oracle.track_mask)
    assert [assembly.source_start_frame + w.source_start_frame for w in prepared.windows()] == [
        1000,
        1600,
    ]


def test_episode_order_and_each_episodes_actor_permutation_are_bitwise_identical():
    _, episodes = _episodes()
    shuffled = []
    for episode, order in zip(episodes, ([2, 0, 1], [1, 2, 0]), strict=True):
        raw = episode.capture
        shuffled.append(
            replace(
                episode,
                capture=PrivateCaptureArrays(
                    raw.joints[:, order],
                    raw.track_mask[:, order],
                    tuple(raw.actor_commitments[i] for i in order),
                    raw.source_sha256,
                ),
            )
        )
    a, b = assemble_parent_capture(episodes), assemble_parent_capture(tuple(reversed(shuffled)))
    assert a.capture.source_sha256 == b.capture.source_sha256
    assert a.episode_source_sha256s == b.episode_source_sha256s
    np.testing.assert_array_equal(
        a.capture.joints.view(np.uint32), b.capture.joints.view(np.uint32)
    )
    np.testing.assert_array_equal(a.capture.track_mask, b.capture.track_mask)


def test_fifteen_second_and_150_second_gaps_do_not_create_physical_phase_evidence():
    for gap in (450, 4500):
        _, episodes = _episodes(gap=gap)
        assembled = assemble_parent_capture(episodes)
        prepared = prepare_continuous_capture(assembled.capture)
        assert prepared.frame_count == (600 + gap) // 300 * 200
        field = continuous_directional_phase_field(prepared, energy_floors=np.zeros(6))
        interior = [
            p
            for p, (start, stop) in enumerate(field.intervals)
            if start >= 220 and stop <= (300 + gap) * 2 // 3 - 20
        ]
        assert interior
        assert not field.actor_patch_mask[:, interior].any()
        pair = local_pair_chunk(field, 0, field.pair_count)
        assert not pair.support_mask[:, interior].any()
        assert not pair.phase_mask[:, interior].any()


def test_native_missing_values_are_masked_not_promoted_to_observed_zeros():
    _, episodes = _episodes()
    raw = episodes[0].capture
    values, mask = raw.joints.copy(), raw.track_mask.copy()
    values[100:107, 1] = np.nan
    mask[100:107, 1] = False
    changed = (
        replace(episodes[0], capture=replace(raw, joints=values, track_mask=mask)),
        episodes[1],
    )
    assembled = assemble_parent_capture(changed)
    assert not assembled.capture.track_mask[100:107, 1].any()
    assert not assembled.capture.joints[100:107, 1].view(np.uint32).any()
    prepared = prepare_continuous_capture(assembled.capture)
    assert prepared.decisions[0].accepted and prepared.decisions[0].missing_actor_frames == 7


def test_absolute_frame_lineage_and_declared_partial_tail_are_preserved():
    _, episodes = _episodes(tail=13)
    a = assemble_parent_capture(episodes)
    shifted = tuple(
        replace(
            e,
            source_start_frame=e.source_start_frame + 300,
            source_stop_frame=e.source_stop_frame + 300,
        )
        for e in episodes
    )
    b = assemble_parent_capture(shifted)
    assert a.capture.source_sha256 != b.capture.source_sha256
    np.testing.assert_array_equal(a.capture.joints, b.capture.joints)
    assert a.capture.frame_count == 913 and a.source_stop_frame == 1913
    assert prepare_continuous_capture(a.capture).trailing_source_frames == 13


def test_overlap_actor_change_and_resource_limits_require_explicit_resolution():
    _, episodes = _episodes()
    with pytest.raises(PreprocessingError, match="overlapping"):
        assemble_parent_capture(
            (episodes[0], replace(episodes[1], source_start_frame=1200, source_stop_frame=1500))
        )
    raw = episodes[1].capture
    changed = replace(raw, actor_commitments=(bytes([9]) * 32,) + raw.actor_commitments[1:])
    with pytest.raises(PreprocessingError, match="membership"):
        assemble_parent_capture((episodes[0], replace(episodes[1], capture=changed)))
    with pytest.raises(PreparationResourceLimit, match="RESOURCE_LIMIT"):
        assemble_parent_capture(episodes, max_assembly_bytes=1)
    with pytest.raises(TypeError, match="raw"):
        ParentCaptureEpisode(prepare_continuous_capture(raw), 1600, 1900)
    with pytest.raises(PreprocessingError, match="real raw frames"):
        replace(episodes[0], source_stop_frame=1299)
    with pytest.raises(PreprocessingError, match="native source provenance"):
        replace(episodes[0], capture=replace(episodes[0].capture, source_sha256=None))
