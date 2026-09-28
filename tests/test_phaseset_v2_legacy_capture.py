"""Data-free full-capture pooling of the unchanged old six-token encoder."""

from __future__ import annotations

import copy
from unittest.mock import patch

import numpy as np
import pytest
import torch

from phaseset_core.capture_pooling import CaptureWindowPlan, pool_capture_windows
from phaseset_core.continuous_capture import prepare_continuous_capture
from phaseset_core.controls import PhaseSetSystem
from phaseset_core.legacy_capture import encode_legacy_capture_tokens
from phaseset_core.models import GroupTokenOutput
from phaseset_core.pipeline import PrivateCaptureArrays, collate_group_samples


def _capture(
    order: tuple[int, ...] = (0, 1, 2), *, windows: int = 3,
    reject_middle: bool = True,
):
    frames, actors = 300 * windows, 3
    time = np.arange(frames, dtype=np.float64) / 30.0
    motion = np.zeros((frames, actors, 22, 3), dtype=np.float32)
    for actor in range(actors):
        motion[:, actor, :, 0] = (
            actor + np.sin(2 * np.pi * 1.125 * time + actor)
        )[:, None]
        motion[:, actor, :, 2] = (
            0.25 * actor + np.cos(2 * np.pi * 0.75 * time + actor)
        )[:, None]
        motion[:, actor, 1:, 0] += 0.3 * np.sin(2 * np.pi * 2.5 * time)[:, None]
    tracked = np.ones(motion.shape[:-1], dtype=np.bool_)
    if reject_middle:
        tracked[300:360] = False
    commitments = tuple(bytes([actor + 1]) * 32 for actor in range(actors))
    return prepare_continuous_capture(
        PrivateCaptureArrays(
            motion[:, order].copy(),
            tracked[:, order].copy(),
            tuple(commitments[actor] for actor in order),
            "a" * 64,
        )
    )


def _encoder(width: int = 8) -> PhaseSetSystem:
    return PhaseSetSystem(
        "08",
        embedding_dim=width,
        hidden_dim=8,
        energy_floors=np.full((6,), 1e-10, dtype=np.float64),
        edge_budget=64,
    ).eval()


def test_legacy_complete_capture_consumes_every_accepted_window_and_backpropagates() -> None:
    torch.manual_seed(1729)
    capture = _capture()
    encoder = _encoder()
    output = encode_legacy_capture_tokens(encoder, capture, edge_chunk_size=64)
    assert output.timeline_window_count == 3
    assert output.accepted_window_count == 2
    windows = capture.windows()
    assert len(windows) == 2
    direct = [
        encoder(collate_group_samples((window,)), edge_chunk_size=64)
        for window in windows
    ]
    tokens = torch.stack([row.tokens[0].to(torch.float64) for row in direct])
    masks = torch.stack([row.band_mask[0] for row in direct])
    counts = masks.sum(0)
    expected = torch.where(
        counts[:, None] > 0,
        torch.where(masks[:, :, None], tokens, 0.0).sum(0) / counts.clamp_min(1)[:, None],
        0.0,
    ).to(torch.float32)
    assert torch.equal(output.band_mask[0], counts > 0)
    assert torch.equal(output.supporting_window_count, counts)
    assert torch.equal(output.tokens[0], expected)
    weights = torch.arange(1, 49, dtype=torch.float32).reshape(1, 6, 8) / 48
    (output.tokens * weights).sum().backward()
    assert encoder.encoder is not None
    for module in (
        encoder.encoder.half_edge_encoder,
        encoder.encoder.pair_encoder,
        encoder.encoder.topology_encoder,
        encoder.encoder.common_postprocess,
    ):
        assert any(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            and bool(torch.count_nonzero(parameter.grad) > 0)
            for parameter in module.parameters()
        )


def test_legacy_complete_capture_is_actor_permutation_invariant() -> None:
    torch.manual_seed(2718)
    encoder = _encoder()
    original = encode_legacy_capture_tokens(encoder, _capture())
    permuted = encode_legacy_capture_tokens(encoder, _capture((2, 0, 1)))
    assert torch.equal(original.tokens, permuted.tokens)
    assert torch.equal(original.band_mask, permuted.band_mask)
    assert torch.equal(original.supporting_window_count, permuted.supporting_window_count)


def test_legacy_capture_partial_support_matches_frozen_tree_pooling() -> None:
    capture = _capture(windows=5, reject_middle=False)
    encoder = _encoder(width=512)
    mask = np.ascontiguousarray(np.array([
        [1, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 1],
        [1, 1, 0, 0, 0, 1],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 0, 0, 1, 0],
    ], dtype=np.bool_))
    tokens = np.zeros((5, 6, 512), dtype=np.float32)
    tokens[:, 0, 0] = np.array([1e20, 1, -1e20, 1, 1], dtype=np.float32)
    for index in range(5):
        for band in range(1, 6):
            if mask[index, band]:
                tokens[index, band, 0] = np.float32((index + 1) * (band + 1))
    outputs = [
        GroupTokenOutput(
            torch.from_numpy(tokens[index : index + 1]),
            torch.from_numpy(mask[index : index + 1]),
            torch.zeros((1, 6, 512)),
            torch.zeros((1, 6, 512)),
            torch.zeros((1, 6), dtype=torch.int64),
            torch.zeros((1, 6), dtype=torch.int64),
        )
        for index in range(5)
    ]
    with patch.object(encoder, "forward", side_effect=outputs) as mocked:
        actual = encode_legacy_capture_tokens(
            encoder, capture, checkpoint_windows=False,
        )
    assert mocked.call_count == 5
    windows = capture.windows()
    commitments = tuple(bytes.fromhex(window.window_sha256) for window in windows)
    plan = CaptureWindowPlan(
        capture.group_commitment,
        commitments,
        tuple(window.source_start_frame for window in windows),
    )
    expected = pool_capture_windows(
        plan,
        window_commitments=commitments,
        base_embeddings=np.zeros((5, 512), dtype=np.float32),
        tokens=np.ascontiguousarray(tokens),
        band_mask=mask,
    )
    np.testing.assert_array_equal(actual.tokens[0].detach().numpy(), expected.tokens)
    np.testing.assert_array_equal(actual.band_mask[0].numpy(), expected.band_mask)
    np.testing.assert_array_equal(
        actual.supporting_window_count.numpy(), expected.valid_window_count
    )
    assert actual.supporting_window_count.tolist() == [5, 4, 3, 0, 1, 2]
    assert not bool(actual.tokens[0, 3].view(torch.int32).any())


def test_legacy_window_checkpoint_matches_direct_forward_and_backward() -> None:
    torch.manual_seed(31415)
    capture = _capture()
    checkpointed = _encoder()
    direct = copy.deepcopy(checkpointed)
    a = encode_legacy_capture_tokens(checkpointed, capture, checkpoint_windows=True)
    b = encode_legacy_capture_tokens(direct, capture, checkpoint_windows=False)
    assert torch.equal(a.tokens, b.tokens)
    assert torch.equal(a.band_mask, b.band_mask)
    weights = torch.arange(1, 49, dtype=torch.float32).reshape(1, 6, 8) / 48
    (a.tokens * weights).sum().backward()
    (b.tokens * weights).sum().backward()
    for left, right in zip(checkpointed.parameters(), direct.parameters(), strict=True):
        if left.grad is None or right.grad is None:
            assert left.grad is None and right.grad is None
        else:
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


def test_legacy_capture_rejects_nonlegacy_system_and_invalid_chunk() -> None:
    capture = _capture()
    with pytest.raises(TypeError, match="system 08"):
        encode_legacy_capture_tokens(PhaseSetSystem("00", embedding_dim=8), capture)
    with pytest.raises(ValueError, match="chunk_size"):
        encode_legacy_capture_tokens(_encoder(), capture, edge_chunk_size=0)
