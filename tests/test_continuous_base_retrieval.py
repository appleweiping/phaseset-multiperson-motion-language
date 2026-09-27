"""Real registered architecture checks plus explicit analytic host fixtures.

No native learning result, fitted base, pilot or formal stage is claimed.
"""

from __future__ import annotations

import copy
from dataclasses import replace
import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from phaseset_core.continuous_base_retrieval import ContinuousBaseRetrievalSystem
from phaseset_core.continuous_capture import prepare_continuous_capture
from phaseset_core.continuous_parent_host import (
    ContinuousParentTrainingHost,
    ParentHostBindings,
    ParentHostConfig,
)
from phaseset_core.continuous_parent_training import (
    ParentCounterfactualRows,
    backward_loaded_parent_batch,
)
from phaseset_core.models import GroupBaseOutput, build_group_base
from phaseset_core.objectives import variable_positive_symmetric_infonce
from phaseset_core.parent_retrieval_task import ParentCaptionRecord, ParentRetrievalTask
from phaseset_core.pipeline import PrivateCaptureArrays, collate_group_samples
from phaseset_core.training import (
    BaseRetrievalSystem,
    TrainingCheckpointError,
    _capture_rng,
    _load_torch_checkpoint,
    _restore_rng,
    _stable_hash,
)
from test_continuous_parent_training import _empty_cf, _key
from test_phaseset_frozen_clip_text import _fixture


def _capture():
    # Analytic three-actor input: two distinct accepted windows separated by
    # a rejected interval. It is not a native capture or augmented benchmark.
    t = np.arange(900, dtype=np.float32) / 30
    joints = np.zeros((900, 3, 22, 3), np.float32)
    for actor in range(3):
        joints[:, actor, :, 0] = (actor + 0.02 * t + np.sin(t + actor))[:, None]
        joints[:, actor, :, 2] = (np.cos(2 * t + actor) + 0.03 * t)[:, None]
    track = np.ones(joints.shape[:-1], np.bool_)
    track[300:360] = False
    return prepare_continuous_capture(
        PrivateCaptureArrays(
            joints, track, tuple(bytes([i + 1]) * 32 for i in range(3)), _key("analytic")
        )
    )


class _AnalyticWindowBase(nn.Module):
    """Only the host/RNG fixture, never substituted for a registered model."""

    system_id = "B2"

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 512)
        self.dropout = nn.Dropout(0.2)

    def forward(self, batch):
        signal = torch.from_numpy(batch.skeletons.copy()).mean((1, 2, 3, 4))[:, None]
        embedding = self.dropout(self.linear(signal))
        return GroupBaseOutput(
            embedding, embedding[:, None], torch.ones(len(signal), 1, dtype=torch.bool)
        )


class _BaseSource:
    def __init__(self, captures, clip):
        self.captures = {capture.source_sha256: capture for capture in captures}
        self.clip = clip
        self.loaded = []

    def view(self, parent, *, seed, epoch, training):
        self.loaded.append((parent.source_sha256, epoch, training))
        return self.captures[parent.source_sha256]

    def text(self, labels, *, training):
        return self.clip.encode(
            labels.captions, caption_commitments=labels.caption_commitments, batch_size=3
        ), _empty_cf()


def _setup(tmp_path):
    (tmp_path / "clip").mkdir()
    clip, _, _, _ = _fixture(tmp_path / "clip")
    capture = _capture()
    captures, parents = [], []
    for index in range(6):
        key = _key(f"analytic-source/{index}")
        captures.append(replace(capture, source_sha256=key))
        parents.append(
            ParentCaptionRecord(
                key,
                _key(f"analytic-family/{index}"),
                "C01" if index < 2 else "C02" if index < 4 else "C03",
                "train" if index < 4 else "validation",
                (f"human {index}", f"extra {index}") if index == 0 else (f"human {index}",),
            )
        )
    task = ParentRetrievalTask(
        tuple(parents), expected_family_keys=tuple(row.annotation_family_sha256 for row in parents)
    )
    torch.manual_seed(1729)
    model = ContinuousBaseRetrievalSystem(
        BaseRetrievalSystem(_AnalyticWindowBase(), embedding_dim=512)
    )
    config = ParentHostConfig.for_base(
        1729, ("C01", "C02"), ("C03",), epochs=2, parent_batch_size=2
    )
    bindings = ParentHostBindings(_key("inputs"), _key("code"), _key("runtime"), None)
    return model, task, _BaseSource(captures, clip), config, bindings


@pytest.mark.parametrize("system_id", ["B0", "B1", "B2"])
def test_real_registered_bases_all_windows_checkpoint_scores_and_gradients(system_id, tmp_path):
    capture = _capture()
    windows = capture.windows()
    assert len(windows) == 2 and windows[1].source_start_frame == 600
    (tmp_path / "clip").mkdir()
    clip, _, _, _ = _fixture(tmp_path / "clip")
    text = clip.encode(
        ("a group moves", "people interact"),
        caption_commitments=(b"a" * 32, b"b" * 32),
        batch_size=2,
    )
    torch.manual_seed(1729)
    model = ContinuousBaseRetrievalSystem(
        BaseRetrievalSystem(build_group_base(system_id), embedding_dim=512)
    )
    direct = copy.deepcopy(model)
    direct.checkpoint_windows = False
    # Training dropout is active: checkpoint must replay its exact RNG and
    # each bound window, not merely agree in evaluation mode.
    state = _capture_rng()
    actual = model.score((capture,), text)
    actual.scores.sum().backward()
    after = _capture_rng()
    _restore_rng(state)
    expected = direct.score((capture,), text)
    expected.scores.sum().backward()
    assert torch.equal(actual.scores, expected.scores)
    assert _stable_hash(_capture_rng()) == _stable_hash(after)
    count = 0
    for (name, p), (other, q) in zip(
        model.named_parameters(), direct.named_parameters(), strict=True
    ):
        assert name == other and p.grad is not None and q.grad is not None
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
        count += int(bool(p.grad.abs().sum() > 0))
    assert count > 1
    model.eval()
    with torch.no_grad():
        rows = torch.cat(
            [model.base.encode_trainable(collate_group_samples((window,))) for window in windows]
        )
        oracle = F.normalize(rows.double().mean(0).float(), dim=0)
        assert torch.equal(model.encode_capture(capture), oracle)
        # Readout is exactly the frozen residual global anchor's readout;
        # constructing a residual is not needed just to obtain this oracle.
        assert not torch.equal(oracle, F.normalize(rows[:1].double().mean(0).float(), dim=0))
    assert actual.motion_embeddings.shape == (1, 512)
    assert actual.capture_keys == (capture.source_sha256,)


def test_base_full_gallery_replay_matches_dense_all_parameter_gradients(tmp_path):
    model, task, source, _, _ = _setup(tmp_path)
    labels = task.batch(tuple(row.annotation_family_sha256 for row in task.parents[:4]))
    text, cf = source.text(labels, training=True)
    captures = tuple(source.captures[key] for key in labels.motion_source_keys)
    dense = copy.deepcopy(model)
    state = _capture_rng()
    scores = dense.score(captures, text).scores
    loss = variable_positive_symmetric_infonce(scores, labels.positive_mask(device=scores.device))[
        2
    ]
    loss.backward()
    after = _capture_rng()
    _restore_rng(state)
    result = backward_loaded_parent_batch(
        model,
        labels,
        text,
        load_view=lambda key: source.captures[key],
        counterfactuals=cf,
        cf_weight=0,
        margin=0,
    )
    # Dense B-row GEMM and single-row replay can differ by float32 rounding;
    # cache vs its own replay remains bitwise inside the production helper.
    torch.testing.assert_close(scores, result.scores, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(torch.tensor(result.loss), loss.detach(), rtol=2e-6, atol=2e-7)
    assert result.parent_count == 4 and result.retrieval_caption_count == 5
    assert result.verified_counterfactual_count == 0 and result.replayed_captures == 4
    assert _stable_hash(_capture_rng()) == _stable_hash(after)
    for p, q in zip(model.parameters(), dense.parameters(), strict=True):
        assert p.grad is not None and q.grad is not None
        # The dense full-batch graph accumulates contributions in reverse
        # graph order; replay accumulates canonical parent rows in forward
        # order. Only this independent accumulation oracle has a tolerance.
        torch.testing.assert_close(p.grad, q.grad, rtol=3e-5, atol=3e-6)


def test_every_window_checkpoint_receives_explicit_model_device_tensor(monkeypatch):
    import phaseset_core.continuous_base_retrieval as module

    model = ContinuousBaseRetrievalSystem(
        BaseRetrievalSystem(_AnalyticWindowBase(), embedding_dim=512)
    )
    actual_checkpoint, witnessed = module.checkpoint, []

    def checked(function, *args, **settings):
        # CPU qualification verifies the actual invocation contract, not CUDA
        # numerical correctness. A later admitted CUDA all-grad check remains
        # required; parameters hidden solely in a closure do not stash its RNG.
        assert args and type(args[0]) is nn.Parameter
        assert args[0] is model.base.logit_scale
        assert args[0].device == next(model.parameters()).device
        assert settings == {"use_reentrant": False, "preserve_rng_state": True}
        witnessed.append(args[0].device)
        return actual_checkpoint(function, *args, **settings)

    monkeypatch.setattr(module, "checkpoint", checked)
    model.encode_capture(_capture()).square().sum().backward()
    assert len(witnessed) == 2


@pytest.mark.parametrize("stop", [0, 1, 2, 4])
def test_base_host_actual_updates_and_resume_are_bitwise(tmp_path, stop):
    model, task, source, config, bindings = _setup(tmp_path)
    initial = copy.deepcopy(model)
    full = ContinuousParentTrainingHost(model, task, source, config, bindings).fit(
        tmp_path / "full"
    )
    interrupted = ContinuousParentTrainingHost(
        copy.deepcopy(initial), task, source, config, bindings
    ).fit(tmp_path / "interrupted", stop_after_steps=stop)
    resumed_model = copy.deepcopy(initial)
    resumed = ContinuousParentTrainingHost(resumed_model, task, source, config, bindings).fit(
        tmp_path / "resumed",
        resume_checkpoint=interrupted.latest_checkpoint.path,
        resume_sha256=interrupted.latest_checkpoint.sha256,
    )
    assert full.outcome == resumed.outcome == "COMPLETED"
    assert (
        full.parents_seen == resumed.parents_seen == 8
        and full.global_step == resumed.global_step == 4
    )
    assert full.validation_history == resumed.validation_history
    assert any(
        not torch.equal(p, initial.state_dict()[name]) for name, p in model.state_dict().items()
    )
    expected, _ = _load_torch_checkpoint(full.latest_checkpoint.path)
    actual, _ = _load_torch_checkpoint(resumed.latest_checkpoint.path)
    for key in ("model", "optimizer", "scheduler", "rng"):
        assert _stable_hash(actual[key]) == _stable_hash(expected[key])
    assert expected["manifest"]["frozen_base_state_sha256"] is None
    assert expected["manifest"]["config"]["stage"] == "base"
    assert (
        json.loads((tmp_path / "full" / "terminal.json").read_text())["formal_authority"] is False
    )


def test_base_host_stage_checkpoint_and_cf_contracts_are_explicit(tmp_path):
    model, task, source, config, bindings = _setup(tmp_path)
    defaults = ParentHostConfig.for_base(1729, ("C01",), ("C03",))
    assert defaults.epochs == 30 and defaults.learning_rate == 2e-4 and defaults.cf_weight == 0
    for bad_config, bad_bindings in (
        (replace(config, stage="residual"), bindings),
        (config, replace(bindings, frozen_base_checkpoint_sha256=_key("unexpected"))),
    ):
        with pytest.raises(ValueError):
            ContinuousParentTrainingHost(model, task, source, bad_config, bad_bindings)
    labels = task.batch(tuple(row.annotation_family_sha256 for row in task.parents[:2]))
    text, _ = source.text(labels, training=True)
    cf = ParentCounterfactualRows((labels.motion_source_keys[0],), (0,), (2,), (False,))
    for pair, weight in ((_empty_cf(), 0.2), (cf, 0)):
        with pytest.raises(ValueError, match="no CF"):
            backward_loaded_parent_batch(
                model,
                labels,
                text,
                load_view=lambda key: source.captures[key],
                counterfactuals=pair,
                cf_weight=weight,
                margin=0,
            )
    initial = copy.deepcopy(model)
    result = ContinuousParentTrainingHost(model, task, source, config, bindings).fit(
        tmp_path / "first", stop_after_steps=0
    )
    initial.checkpoint_windows = False
    with pytest.raises(TrainingCheckpointError, match="drifted"):
        ContinuousParentTrainingHost(initial, task, source, config, bindings).fit(
            tmp_path / "drift",
            resume_checkpoint=result.latest_checkpoint.path,
            resume_sha256=result.latest_checkpoint.sha256,
        )
