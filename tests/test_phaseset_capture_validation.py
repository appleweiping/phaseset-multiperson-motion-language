"""Server-only capture-gallery contracts; analytic fixtures are not results."""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib

import numpy as np
import pytest
import torch
from torch import nn

from phaseset_core import capture_validation as cv
from phaseset_core import frozen_clip_text as clip
from phaseset_core.capture_pooling import CaptureWindowPlan
from phaseset_core.contracts import PreparedGroupBatch, group_commitment
from phaseset_core.models import ActorMeanBase, GroupBaseOutput, GroupTokenOutput, PhaseSetEncoder
from phaseset_core.objectives import PhaseSetRetrievalHead
from phaseset_core.training import BaseRetrievalSystem, ResidualRetrievalSystem, TrainingConfig


def _id(label: str) -> bytes:
    return hashlib.sha256(label.encode("ascii")).digest()


def _groups(x: float, y: float, *, pad: int = 0, reverse: bool = False) -> PreparedGroupBatch:
    actors = tuple(_id(f"shared/actor/{index}") for index in range(3))
    values = np.zeros((1, 3 + pad, 200, 22, 3), dtype=np.float32)
    values[0, :3, :, :, 0] = x
    values[0, :3, :, :, 1] = y
    values[0, :3, :, :, 2] = np.arange(200, dtype=np.float32)[None, :, None] * 0.01
    mask = np.zeros((1, 3 + pad), dtype=np.bool_)
    mask[:, :3] = True
    track = np.zeros((1, 3 + pad, 200, 22), dtype=np.bool_)
    track[:, :3] = True
    keys = actors + (None,) * pad
    if reverse:
        order = tuple(reversed(range(3 + pad)))
        values = np.ascontiguousarray(values[:, order])
        mask = np.ascontiguousarray(mask[:, order])
        track = np.ascontiguousarray(track[:, order])
        keys = tuple(keys[index] for index in order)
    return PreparedGroupBatch(values, mask, np.ones((1, 200), dtype=np.bool_), track,
                              (keys,), (group_commitment(actors),))


def _bytes(value: torch.Tensor) -> bytes:
    return value.detach().contiguous().view(torch.uint8).numpy().tobytes()


def _text(label: str, axes: tuple[int, ...]) -> clip.FrozenClipTextBatch:
    values = torch.zeros((len(axes), 512), dtype=torch.float32)
    for row, axis in enumerate(axes):
        values[row, axis] = 1.0
    keys = tuple(_id(f"{label}/caption/{row}") for row in range(len(axes)))
    rows = tuple((row, key.hex(), _id(f"{label}/text/{row}").hex(), 3, 3, False,
                  _id(f"tokens/{row}").hex(), _id(f"mask/{row}").hex(),
                  hashlib.sha256(_bytes(values[row:row + 1])).hexdigest())
                 for row, key in enumerate(keys))
    raw = _bytes(values)
    receipt = clip.FrozenClipTextReceipt(
        batch_size=len(axes), caption_rows=rows, chunk_ranges=((0, len(axes)),),
        frozen_embedding_cache_key_sha256="1" * 64, live_model_manifest_sha256="2" * 64,
        output_bytes=len(raw), output_sha256=hashlib.sha256(raw).hexdigest(),
        output_shape=tuple(values.shape), output_stride=tuple(values.stride()),
        runtime_identity=(("fixture", "server-only-not-official-text"),),
        runtime_manifest_sha256="3" * 64, snapshot_files=(), snapshot_manifest_sha256="4" * 64,
        source_files=(), source_manifest_sha256="5" * 64,
    )
    return clip.FrozenClipTextBatch(values, keys, receipt, _seal=clip._CONSTRUCTION_SEAL)


def _capture(label: str, coordinates: tuple[tuple[float, float], ...],
             axes: tuple[int, ...], *, reverse: bool = False, pad: int = 0):
    keys = tuple(_id(f"{label}/window/{row}") for row in range(len(coordinates)))
    windows = tuple(cv.CaptureValidationWindow(key, _groups(*xy, pad=pad, reverse=reverse))
                    for key, xy in zip(keys, coordinates, strict=True))
    plan = CaptureWindowPlan(_id(label), keys, tuple(300 * row for row in range(len(keys))))
    return cv.CaptureValidationCapture(plan, windows[::-1] if reverse else windows,
                                       _text(label, axes), "C00")


def _source(*, reverse: bool = False, pad: int = 0):
    captures = (_capture("A", ((10.0, 0.0), (0.0, 2.0)), (0, 0), reverse=reverse, pad=pad),
                _capture("B", ((0.0, 2.0),), (1,), reverse=reverse, pad=pad))
    return cv.CaptureValidationSource("val", "a" * 64, captures[::-1] if reverse else captures)


class _AnalyticBase(nn.Module):
    """Controllable encoder output, only for an independent pooling/scoring oracle."""
    system_id = "B0"

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.calls = 0
        self.fail = False
        self.dropout = nn.Dropout(0.5)

    def forward(self, groups):
        self.calls += 1
        if self.fail:
            raise RuntimeError("injected encoder failure")
        assert not self.training
        assert not torch.backends.mha.get_fastpath_enabled()
        value = torch.zeros((1, 512), dtype=torch.float32, device=self.anchor.device)
        value[0, :2] = torch.tensor(groups.skeletons[0, 0, 0, 0, :2].copy(), device=value.device)
        actors = value[:, None, :].expand(1, groups.actor_counts[0], 512).contiguous()
        return GroupBaseOutput(value, actors, torch.ones(actors.shape[:2], dtype=torch.bool))


class _RecordingBaseSystem(BaseRetrievalSystem):
    def __init__(self):
        super().__init__(_AnalyticBase(), embedding_dim=512)
        self.scored = []

    def scores(self, motion, text):
        self.scored.append((motion.clone(), text.clone()))
        assert not torch.backends.mha.get_fastpath_enabled()
        return super().scores(motion, text)


class _AnalyticPeriodic(nn.Module):
    system_id = "08"
    embedding_dim = 512

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.bad_mask = False

    def forward(self, groups):
        x = float(groups.skeletons[0, 0, 0, 0, 0])
        tokens = torch.zeros((1, 6, 512), dtype=torch.float32)
        mask = torch.zeros((1, 6), dtype=torch.bool)
        if x > 0:
            tokens[0, 0, 0] = x
            mask[0, 0] = True
        if self.bad_mask:
            mask = mask.float()
        counts = torch.zeros((1, 6), dtype=torch.int64)
        return GroupTokenOutput(tokens, mask, tokens.clone(), torch.zeros_like(tokens), counts, counts)


class _RecordingResidual(ResidualRetrievalSystem):
    def scores(self, tokens, mask, base, text):
        self.scored_tokens = tokens.clone()
        self.scored_mask = mask.clone()
        return super().scores(tokens, mask, base, text)


def _config(stage="base", **kwargs):
    return TrainingConfig(stage=stage, seed=1729, **kwargs)


def test_pool_before_normalization_one_capture_gallery_and_variable_caption_counts():
    system = _RecordingBaseSystem()
    source = _source()
    result = cv.run_capture_validation(system, _config(), source)
    assert system.group_base.calls == 3
    assert len(system.scored) == 1
    motion, text = system.scored[0]
    row_a = tuple(c.capture_commitment for c in source.captures).index(_id("A"))
    row_b = 1 - row_a
    assert torch.equal(motion[row_a, :2], torch.tensor([5.0, 1.0]))
    assert torch.equal(motion[row_b, :2], torch.tensor([0.0, 2.0]))
    assert text.shape == (3, 512)
    assert result.dataset.scores.shape == (2, 3)
    assert len(set(result.dataset.motion_commitments)) == 2
    assert source.captures[0].group_commitment == source.captures[1].group_commitment
    assert result.text_to_motion_capture_r1 == result.motion_to_text_capture_r1 == Fraction(1)
    assert result.primary_capture_r1 == Fraction(1)
    assert result.encoded_bytes == 3 * 512 * 4
    assert result.source_census_sha256 == source.census_sha256
    assert np.isfinite(result.loss)


def test_capture_window_actor_order_and_padding_preserve_scores():
    system = _RecordingBaseSystem()
    left = cv.run_capture_validation(system, _config(), _source())
    right = cv.run_capture_validation(system, _config(), _source(reverse=True, pad=5))
    assert left.dataset.motion_commitments == right.dataset.motion_commitments
    assert left.dataset.caption_commitments == right.dataset.caption_commitments
    assert left.scores_float64_sha256 == right.scores_float64_sha256


def test_modes_flags_rng_and_state_restored_after_success_and_failure():
    system = _RecordingBaseSystem().train()
    system.group_base.dropout.eval()
    modes = tuple(module.training for module in system.modules())
    state = {key: value.clone() for key, value in system.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    old_flag = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(True)
        cv.run_capture_validation(system, _config(), _source())
        assert torch.backends.mha.get_fastpath_enabled() is True
        assert modes == tuple(module.training for module in system.modules())
        system.group_base.fail = True
        with pytest.raises(RuntimeError, match="injected encoder failure"):
            cv.run_capture_validation(system, _config(), _source())
        assert torch.backends.mha.get_fastpath_enabled() is True
        assert modes == tuple(module.training for module in system.modules())
    finally:
        torch.backends.mha.set_fastpath_enabled(old_flag)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(value, system.state_dict()[key]) for key, value in state.items())
    assert all(parameter.grad is None for parameter in system.parameters())


@pytest.mark.parametrize("split", ["test", "train"])
def test_source_rejects_other_splits(split):
    with pytest.raises(cv.CaptureValidationError, match="split='val'"):
        replace(_source(), split=split)


def test_complete_plan_required_and_duplicate_ids_rejected():
    source = _source()
    capture = next(c for c in source.captures if len(c.windows) == 2)
    with pytest.raises(cv.CaptureValidationError, match="complete accepted-window plan"):
        replace(capture, windows=capture.windows[:1])
    with pytest.raises(cv.CaptureValidationError, match="globally unique"):
        replace(source, captures=(source.captures[0], source.captures[0]))
    with pytest.raises(cv.CaptureValidationError, match="globally unique"):
        replace(source, captures=(source.captures[0], replace(source.captures[1],
                       holistic_text=source.captures[0].holistic_text)))


def test_rejects_short_window_and_changed_actor_group():
    groups = _groups(1.0, 2.0)
    short = PreparedGroupBatch(np.ascontiguousarray(groups.skeletons[:, :, :100]),
        groups.actor_mask, np.ascontiguousarray(groups.frame_mask[:, :100]),
        np.ascontiguousarray(groups.track_mask[:, :, :100]), groups.actor_commitments,
        groups.group_commitments)
    with pytest.raises(cv.CaptureValidationError, match="200-frame"):
        cv.CaptureValidationWindow(_id("short"), short)
    source = _source()
    capture = next(c for c in source.captures if len(c.windows) == 2)
    keys = tuple(_id(f"other/{i}") for i in range(3))
    other = PreparedGroupBatch(groups.skeletons, groups.actor_mask, groups.frame_mask,
                               groups.track_mask, (keys,), (group_commitment(keys),))
    with pytest.raises(cv.CaptureValidationError, match="valid actor set"):
        replace(capture, windows=(capture.windows[0],
                                  replace(capture.windows[1], groups=other)))


@pytest.mark.parametrize("limit", [1, 6143])
def test_encoded_budget_rejects_before_any_forward(limit):
    system = _RecordingBaseSystem()
    with pytest.raises(cv.CaptureValidationResourceLimit, match="max_encoded_bytes"):
        cv.run_capture_validation(system, _config(), _source(), max_encoded_bytes=limit)
    assert system.group_base.calls == 0


def test_edge_budget_rejects_all_edges_before_any_forward():
    system = _RecordingBaseSystem()
    with pytest.raises(cv.CaptureValidationResourceLimit, match="edge_budget"):
        cv.run_capture_validation(system, _config(edge_budget=2), _source())
    assert system.group_base.calls == 0


def test_concrete_device_resolves_cuda_alias_and_cpu_index(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    assert cv._concrete_device("cuda") == torch.device("cuda:3")
    assert cv._concrete_device("cuda:1") == torch.device("cuda:1")
    assert cv._concrete_device("cpu:0") == torch.device("cpu")


def test_encoder_dtype_is_not_silently_cast():
    with pytest.raises(cv.CaptureValidationError, match="float32 tensor"):
        cv._finite_cpu_float32(torch.ones((1, 512), dtype=torch.float64), (1, 512), "fixture")


def test_residual_masked_pooling_and_bool_mask_contract():
    with torch.random.fork_rng():
        torch.manual_seed(1729)
        system = _RecordingResidual(_AnalyticBase(), _AnalyticPeriodic(),
             PhaseSetRetrievalHead(embedding_dim=512, text_hidden_dim=512), embedding_dim=512)
    base = cv.run_capture_validation(_RecordingBaseSystem(), _config(), _source())
    result = cv.run_capture_validation(system, _config("residual"), _source())
    assert np.array_equal(result.dataset.scores, base.dataset.scores)
    assert result.primary_capture_r1 == Fraction(1)
    assert result.encoded_bytes == 3 * (512 * 4 + 6 * 512 * 4 + 6)
    row_a = result.dataset.motion_commitments.index(_id("A"))
    assert system.scored_tokens[row_a, 0, 0].item() == 10.0
    assert system.scored_mask.sum().item() == 1
    invalid = system.scored_tokens[~system.scored_mask]
    assert torch.count_nonzero(invalid).item() == 0
    assert not torch.signbit(invalid).any().item()
    system.periodic_encoder.bad_mask = True
    with pytest.raises(cv.CaptureValidationError, match="bool"):
        cv.run_capture_validation(system, _config("residual"), _source())


@pytest.mark.parametrize("stage", ["base", "residual"])
def test_actual_width_native_encoder_capture_validation(stage):
    with torch.random.fork_rng():
        torch.manual_seed(1729)
        base = ActorMeanBase()
        if stage == "base":
            system = BaseRetrievalSystem(base, embedding_dim=512)
        else:
            system = ResidualRetrievalSystem(base, PhaseSetEncoder(),
                PhaseSetRetrievalHead(embedding_dim=512, text_hidden_dim=512), embedding_dim=512)
    source = cv.CaptureValidationSource("val", "c" * 64,
        (_capture("native-A", ((1.0, 2.0), (0.0, 1.0)), (0,)),
         _capture("native-B", ((2.0, 1.0),), (1,))))
    result = cv.run_capture_validation(system, _config(stage), source)
    repeat = cv.run_capture_validation(system, _config(stage), source)
    assert result.dataset.scores.shape == (2, 2)
    assert np.isfinite(result.dataset.scores).all()
    assert result.scores_float64_sha256 == repeat.scores_float64_sha256
    assert result.primary_capture_r1 == repeat.primary_capture_r1
    assert all(parameter.grad is None for parameter in system.parameters())


def test_capture_macro_fraction_does_not_weight_captions_as_independent_captures():
    from phaseset_core.evaluation import RetrievalDataset
    dataset = RetrievalDataset(
        scores=np.array([[2.0, 0.0, 0.0], [1.0, 3.0, 4.0]], dtype=np.float64),
        motion_commitments=(_id("a"), _id("b")),
        caption_commitments=(_id("a0"), _id("a1"), _id("b0")),
        positive_motion_indices=((0,), (0,), (1,)),
        group_sizes=np.array([3, 3], dtype=np.int64), component_labels=("C00", "C00"))
    # Capture A text success=1/2, B=1; both motion queries retrieve a positive.
    assert cv._capture_fractions(dataset) == (Fraction(3, 4), Fraction(1), Fraction(7, 8))
