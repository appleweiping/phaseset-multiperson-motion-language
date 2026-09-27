"""Actual host-loop qualification on analytic data, not native training results."""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import math

import pytest
import torch

from phaseset_core.continuous_parent_host import (
    ContinuousParentTrainingHost,
    ParentHostBindings,
    ParentHostConfig,
)
from phaseset_core.continuous_parent_training import backward_loaded_parent_batch
from phaseset_core.continuous_parent_training import ParentBackwardResult
from phaseset_core.parent_retrieval_task import ParentCaptionRecord, ParentRetrievalTask
from phaseset_core.training import (
    TrainingCheckpointError,
    _atomic_torch_save,
    _capture_macro_bidirectional_r1,
    _load_torch_checkpoint,
    _stable_hash,
)
from test_continuous_parent_training import _case, _empty_cf, _key, _population


class _Source:
    def __init__(self, views, clip):
        self.views = {view.positive_capture_key: view for view in views}
        self.clip = clip
        self.loaded = []
        self.text_loaded = []

    def view(self, parent, *, seed, epoch, training):
        self.loaded.append((parent.source_sha256, epoch, training))
        return self.views[parent.source_sha256]

    def text(self, labels, *, training):
        self.text_loaded.append((labels.motion_source_keys, training))
        return self.clip.encode(
            labels.captions, caption_commitments=labels.caption_commitments, batch_size=3
        ), _empty_cf()


def _setup(tmp_path, *, batch_size=4):
    system, views, labels, _, clip = _case(tmp_path)
    # All duplicated motions/identities here are analytic fixtures, explicitly
    # not separate real captures or an empirical validation population.
    first = views[0]
    learning = list(labels.parents)
    all_views = list(views)
    for index in range(2, 4):
        view = replace(first, capture=replace(first.capture, source_sha256=_key(f"train/{index}")))
        learning.append(
            ParentCaptionRecord(
                view.positive_capture_key,
                _key(f"trainfamily/{index}"),
                "C01",
                "train",
                (f"learning {index}",),
            )
        )
        all_views.append(view)
    for index in range(2):
        view = replace(first, capture=replace(first.capture, source_sha256=_key(f"val/{index}")))
        learning.append(
            ParentCaptionRecord(
                view.positive_capture_key,
                _key(f"valfamily/{index}"),
                "C03",
                "validation",
                (f"validation {index}",),
            )
        )
        all_views.append(view)
    task = ParentRetrievalTask(
        tuple(learning),
        expected_family_keys=tuple(row.annotation_family_sha256 for row in learning),
    )
    config = ParentHostConfig(
        1729, ("C01", "C02"), ("C03",), epochs=2, parent_batch_size=batch_size
    )
    bindings = ParentHostBindings(*(_key(name) for name in ("inputs", "code", "runtime", "base")))
    return system, task, _Source(all_views, clip), config, bindings


def _events(path):
    return [json.loads(item.read_text()) for item in sorted(path.glob("event-*.json"))]


def test_host_runs_optimizer_selects_validation_and_preserves_full_parent_census(tmp_path):
    system, task, source, config, bindings = _setup(tmp_path)
    initial_base = copy.deepcopy(system.frozen_b2.state_dict())
    initial = copy.deepcopy(system.state_dict())
    host = ContinuousParentTrainingHost(system, task, source, config, bindings)
    result = host.fit(tmp_path / "run")
    assert result.outcome == "COMPLETED" and result.global_step == 2
    assert result.completed_epochs == 2 and result.parents_seen == 8
    assert result.best_checkpoint is not None and len(result.validation_history) == 2
    assert result.best_validation_r1 == max(value for _, value in result.validation_history)
    assert any(not torch.equal(value, initial[name]) for name, value in system.state_dict().items())
    for name, value in system.frozen_b2.state_dict().items():
        assert torch.equal(value, initial_base[name])
    # Cache + replay per training parent; exactly one full view per validation
    # parent/epoch. No physical input is skipped because its caption count differs.
    assert len(source.loaded) == 20
    assert len([row for row in source.loaded if row[2]]) == 16
    events = _events(tmp_path / "run")
    updates = [row for row in events if row["phase"] == "update"]
    assert [row["human_rows"] for row in updates] == [5, 5]
    assert all(row["verified_cf"] == 0 for row in updates)
    assert [row["learning_rate"] for row in updates] == [3e-4, 3e-4]
    assert json.loads((tmp_path / "run" / "terminal.json").read_text())["outcome"] == "COMPLETED"
    best_payload, _ = _load_torch_checkpoint(
        result.best_checkpoint.path, expected_sha256=result.best_checkpoint.sha256
    )
    assert best_payload["validation_history"][-1][1] == result.best_validation_r1
    # Identical fixture motions yield a provable 1/2 endpoint regardless of
    # which caption is maximal; both epochs tie and the first must remain best.
    assert result.validation_history == ((0, 0.5), (1, 0.5))
    assert best_payload["validation_history"] == [(0, 0.5)]
    with pytest.raises(RuntimeError, match="only one attempt"):
        host.fit(tmp_path / "second")


@pytest.mark.parametrize("stop", [0, 1, 2, 4])
def test_host_interruption_resume_matches_uninterrupted_bitwise(tmp_path, stop):
    system, task, source, config, bindings = _setup(tmp_path, batch_size=2)
    # Dropout tests replay RNG and validation's save/restore, not a new model.
    system.text_adapter[1] = torch.nn.Dropout(0.2)
    initial = copy.deepcopy(system)
    full = ContinuousParentTrainingHost(system, task, source, config, bindings).fit(
        tmp_path / "full"
    )
    interrupted_model = copy.deepcopy(initial)
    interrupted = ContinuousParentTrainingHost(
        interrupted_model, task, source, config, bindings
    ).fit(tmp_path / "interrupted", stop_after_steps=stop)
    assert interrupted.global_step == stop
    # Construct from the same seed-bound initial model, not from post-training weights.
    resumed_model = copy.deepcopy(initial)
    resumed = ContinuousParentTrainingHost(resumed_model, task, source, config, bindings).fit(
        tmp_path / "resumed",
        resume_checkpoint=interrupted.latest_checkpoint.path,
        resume_sha256=interrupted.latest_checkpoint.sha256,
    )
    assert resumed.outcome == full.outcome == "COMPLETED"
    assert resumed.validation_history == full.validation_history
    assert resumed.parents_seen == full.parents_seen == 8
    for name, value in system.state_dict().items():
        assert torch.equal(value, resumed_model.state_dict()[name])
    expected, _ = _load_torch_checkpoint(full.latest_checkpoint.path)
    actual, _ = _load_torch_checkpoint(resumed.latest_checkpoint.path)
    for key in ("model", "optimizer", "scheduler", "rng"):
        assert _stable_hash(actual[key]) == _stable_hash(expected[key])


@pytest.mark.parametrize("drift", ["inputs", "code", "runtime", "base", "config", "text", "phase"])
def test_resume_rejects_actual_execution_input_drift(tmp_path, drift):
    system, task, source, config, bindings = _setup(tmp_path)
    initial = copy.deepcopy(system)
    result = ContinuousParentTrainingHost(system, task, source, config, bindings).fit(
        tmp_path / "first", stop_after_steps=0
    )
    if drift in ("inputs", "code", "runtime", "base"):
        field = {
            "inputs": "input_manifest_sha256",
            "code": "code_manifest_sha256",
            "runtime": "runtime_sha256",
            "base": "frozen_base_checkpoint_sha256",
        }[drift]
        bindings = replace(bindings, **{field: _key("drift")})
    elif drift == "config":
        config = replace(config, learning_rate=2e-4)
    elif drift == "phase":
        initial.coordination.strip_phase = True
    else:
        changed = list(task.parents)
        changed[0] = replace(changed[0], captions=("changed exact human row",))
        task = ParentRetrievalTask(
            tuple(changed),
            expected_family_keys=tuple(row.annotation_family_sha256 for row in changed),
        )
    with pytest.raises(TrainingCheckpointError, match="drifted"):
        ContinuousParentTrainingHost(initial, task, source, config, bindings).fit(
            tmp_path / "drift",
            resume_checkpoint=result.latest_checkpoint.path,
            resume_sha256=result.latest_checkpoint.sha256,
        )
    assert json.loads((tmp_path / "drift" / "terminal.json").read_text())["outcome"] == "FAILED"


def test_truncated_checkpoint_and_rehashed_wrong_cursor_fail_with_terminal_evidence(tmp_path):
    system, task, source, config, bindings = _setup(tmp_path)
    initial = copy.deepcopy(system)
    result = ContinuousParentTrainingHost(system, task, source, config, bindings).fit(
        tmp_path / "first", stop_after_steps=0
    )
    raw = result.latest_checkpoint.path.read_bytes()[:80]
    truncated = tmp_path / "truncated.pt"
    truncated.write_bytes(raw)
    with pytest.raises(TrainingCheckpointError, match="truncated"):
        ContinuousParentTrainingHost(copy.deepcopy(initial), task, source, config, bindings).fit(
            tmp_path / "truncated-run",
            resume_checkpoint=truncated,
            resume_sha256=hashlib.sha256(raw).hexdigest(),
        )
    payload, _ = _load_torch_checkpoint(result.latest_checkpoint.path)
    payload.pop("state_sha256")
    payload["parent_batch_offset"] = 1
    payload["state_sha256"] = _stable_hash(payload)
    wrong = _atomic_torch_save(tmp_path / "wrong-cursor.pt", payload)
    with pytest.raises(TrainingCheckpointError, match="cursor"):
        ContinuousParentTrainingHost(copy.deepcopy(initial), task, source, config, bindings).fit(
            tmp_path / "wrong-run", resume_checkpoint=wrong.path, resume_sha256=wrong.sha256
        )


def test_full_validation_metric_uses_any_positive_capture_macro_and_lineage_ties():
    # Three captions for capture A, one for B: caption-weighted T2M would be
    # 3/4, but capture-macro T2M is 1/2. M2T=1, endpoint=(1+1/2)/2=3/4.
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0], [1.0, 0.0, -1.0, 1.0]])
    positives = torch.tensor([[True, True, True, False], [False, False, False, True]])
    a, b = bytes(32), bytes([1]) * 32
    value = _capture_macro_bidirectional_r1(
        logits, positives, (a, b), (a, a, a, b), (a, b), tuple(bytes([i]) * 32 for i in range(4))
    )
    # Motion B has an exact tie for A/B; lineage selects caption0 => miss.
    assert value == 0.5
    logits[1, 3] = 1.5
    assert (
        _capture_macro_bidirectional_r1(
            logits,
            positives,
            (a, b),
            (a, a, a, b),
            (a, b),
            tuple(bytes([i]) * 32 for i in range(4)),
        )
        == 0.75
    )


def test_no_test_or_held_out_rows_loaded_when_outer_fold_learns_old_main_validation(tmp_path):
    system, _, _, _, clip = _case(tmp_path)
    task = _population(4, component="C00", split="validation")
    val = _population(2, component="C03", split="train")
    # Give fixtures distinct physical/family identities; no native data claim.
    val_rows = tuple(
        replace(
            row, source_sha256=_key(f"valsrc/{i}"), annotation_family_sha256=_key(f"valfamily/{i}")
        )
        for i, row in enumerate(val.parents)
    )
    test = ParentCaptionRecord(_key("testsource"), _key("testfamily"), "C09", "test", ("sealed",))
    rows = task.parents + val_rows + (test,)
    full = ParentRetrievalTask(
        rows, expected_family_keys=tuple(row.annotation_family_sha256 for row in rows)
    )
    config = ParentHostConfig(1729, ("C00",), ("C03",), epochs=1, parent_batch_size=4)
    source = _Source((), clip)
    host = ContinuousParentTrainingHost(
        system, full, source, config, ParentHostBindings(*([_key("binding")] * 4))
    )
    assert host._steps_per_epoch == 1
    assert {row.component for row in host._gallery.parents} == {"C03"}
    assert source.loaded == source.text_loaded == []
    assert all(row["component"] in {"C00", "C03"} for row in host._manifest["task"])
    host.fit(tmp_path / "dev-only", stop_after_steps=0)
    assert "sealed" not in (tmp_path / "dev-only" / "run.json").read_text()
    assert source.loaded == source.text_loaded == []
    for wrong in (
        replace(config, train_components=("C09",)),
        replace(config, validation_components=("C09",)),
    ):
        with pytest.raises(ValueError):
            ContinuousParentTrainingHost(system, full, source, wrong, host.bindings)


def test_loader_replay_rejects_wrong_native_source_before_scoring(tmp_path):
    system, views, labels, text, _ = _case(tmp_path)
    with pytest.raises(ValueError, match="different physical source"):
        backward_loaded_parent_batch(
            system,
            labels,
            text,
            load_view=lambda _: views[1],
            counterfactuals=_empty_cf(),
            cf_weight=0.2,
            margin=0.2,
        )


def test_nontrivial_warmup_cosine_schedule_has_independent_scalar_oracle(tmp_path):
    system, task, source, config, bindings = _setup(tmp_path)
    host = ContinuousParentTrainingHost(system, task, source, replace(config, epochs=40), bindings)
    # No motion/CLIP or training run here: isolate the scheduler's 40-step
    # schedule, including two warmup steps and interior/final cosine points.
    for step in range(41):
        expected = (
            (step + 1) / 2
            if step < 2
            else 0.5 * (1 + math.cos(math.pi * min(1.0, (step - 2) / 38)))
        )
        assert host._optimizer.param_groups[0]["lr"] == pytest.approx(3e-4 * expected, abs=1e-15)
        if step < 40:
            host._optimizer.step()
            host._scheduler.step()


def test_actual_host_clips_before_adamw_with_fixed_decay_independent_scalar_oracle(
    tmp_path, monkeypatch
):
    system, task, source, config, bindings = _setup(tmp_path)
    config = replace(config, epochs=1)
    old = float(system.calibration.log_scale.detach())

    def forced_gradient(model, labels, text, **kwargs):
        # This unit fixture isolates optimizer ordering; the actual full
        # loss/backward is exercised independently by the preceding tests.
        model.zero_grad(set_to_none=True)
        model.calibration.log_scale.grad = torch.tensor(10.0)
        return ParentBackwardResult(
            1.0,
            torch.zeros(len(labels.parents), len(labels.captions)),
            len(labels.parents),
            len(labels.captions),
            0,
            0,
        )

    monkeypatch.setattr(
        "phaseset_core.continuous_parent_host.backward_loaded_parent_batch", forced_gradient
    )
    ContinuousParentTrainingHost(system, task, source, config, bindings).fit(tmp_path / "clip-run")
    assert 0.99999 <= float(system.calibration.log_scale.grad) <= 1.0
    # First-step AdamW: bias-corrected moments make the normalized positive
    # gradient ~1, with epsilon 1e-8; decoupled WD acts on the old parameter.
    gradient = 10 / (10 + 1e-6)
    expected = old * (1 - 3e-4 * 0.01) - 3e-4 * gradient / (gradient + 1e-8)
    assert float(system.calibration.log_scale.detach()) == pytest.approx(expected, abs=5e-7)
    update = next(row for row in _events(tmp_path / "clip-run") if row["phase"] == "update")
    assert update["gradient_norm"] == 10.0
