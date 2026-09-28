"""Optimizer settings and resume identity, not native baseline convergence."""

import copy
from dataclasses import replace

import pytest
import torch

from phaseset_core.continuous_parent_host import ContinuousParentTrainingHost, ParentHostConfig
from phaseset_core.continuous_parent_training import ParentBackwardResult
from phaseset_core.literature_parent_host import LiteratureParentTrainingHost
from phaseset_core.training import TrainingCheckpointError, _load_torch_checkpoint, _stable_hash
from test_continuous_parent_host import _setup
from test_literature_parent_host import setup as literature_setup


def test_internal_default_adamw_decay_and_manifest_are_unchanged(tmp_path):
    model, task, source, cfg, bindings = _setup(tmp_path)
    host = ContinuousParentTrainingHost(model, task, source, cfg, bindings)
    assert type(host._optimizer) is torch.optim.AdamW
    assert host._optimizer.param_groups[0]["weight_decay"] == 0.01
    assert host._manifest["config"]["optimizer"] == "adamw"
    assert host._manifest["config"]["weight_decay"] == 0.01
    assert source.loaded == source.text_loaded == []


@pytest.mark.parametrize(
    "settings",
    [
        {"optimizer": "sgd"},
        {"weight_decay": -1.0},
        {"weight_decay": float("nan")},
        {"weight_decay": float("inf")},
        {"weight_decay": True},
    ],
)
def test_invalid_optimizer_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        ParentHostConfig.for_base(1729, ("C01",), ("C03",), **settings)


@pytest.mark.parametrize("optimizer", ["adam", "adamw"])
def test_clip_then_first_update_has_independent_coupled_decay_oracle(
    optimizer, tmp_path, monkeypatch
):
    model, task, source, cfg, bindings = _setup(tmp_path)
    cfg = replace(cfg, epochs=1, optimizer=optimizer, weight_decay=0.1)
    old = float(model.calibration.log_scale.detach())

    def forced_gradient(system, labels, text, **kwargs):
        system.zero_grad(set_to_none=True)
        system.calibration.log_scale.grad = torch.tensor(10.0)
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
    host = ContinuousParentTrainingHost(model, task, source, cfg, bindings)
    host.fit(tmp_path / "run")
    clipped = 10 / (10 + 1e-6)
    if optimizer == "adam":
        effective = clipped + cfg.weight_decay * old
        expected = old - cfg.learning_rate * effective / (abs(effective) + 1e-8)
    else:
        expected = old * (1 - cfg.learning_rate * cfg.weight_decay)
        expected -= cfg.learning_rate * clipped / (clipped + 1e-8)
    assert float(model.calibration.log_scale.detach()) == pytest.approx(expected, abs=5e-7)


@pytest.mark.parametrize(
    "kind,optimizer,decay",
    [("tmr", "adamw", 0.01), ("wamo", "adam", 0.0), ("mime", "adamw", 1e-4)],
)
def test_literature_host_uses_explicit_paper_optimizer_settings(
    kind, optimizer, decay, tmp_path
):
    model, task, source, _, bindings = literature_setup(kind)
    cfg = ParentHostConfig.for_literature(
        {"tmr": "TMR-Set", "wamo": "WaMo-Set", "mime": "MIME-Set"}[kind],
        1729,
        ("C01",),
        ("C03",),
        epochs=2,
        parent_batch_size=2,
    )
    host = LiteratureParentTrainingHost(model, task, source, cfg, bindings)
    expected_type = torch.optim.Adam if optimizer == "adam" else torch.optim.AdamW
    assert type(host._optimizer) is expected_type
    assert host._optimizer.param_groups[0]["weight_decay"] == decay
    assert host._optimizer.param_groups[0]["lr"] == 1e-4
    assert host._manifest["config"]["optimizer"] == optimizer
    assert host._manifest["config"]["weight_decay"] == decay
    assert source.loaded == source.texts == []
    initial = copy.deepcopy(model.state_dict())
    result = host.fit(tmp_path / "run")
    assert result.outcome == "COMPLETED" and result.global_step == 2
    assert result.best_checkpoint is not None
    assert len(result.validation_history) == 2
    assert any(not torch.equal(initial[n], p) for n, p in model.state_dict().items())


def test_paper_optimizer_helper_rejects_unknown_method_and_binds_explicit_pilot_override():
    with pytest.raises(ValueError, match="registered literature"):
        ParentHostConfig.for_literature("Other", 1729, ("C01",), ("C03",))
    cfg = ParentHostConfig.for_literature(
        "WaMo-Set", 1729, ("C01",), ("C03",), learning_rate=2e-4
    )
    assert cfg.optimizer == "adam" and cfg.weight_decay == 0.0
    assert cfg.learning_rate == 2e-4


@pytest.mark.parametrize("change", [{"optimizer": "adam"}, {"weight_decay": 1e-4}])
def test_optimizer_drift_rejected_before_resume_loads_training_inputs(change, tmp_path):
    model, task, source, cfg, bindings = _setup(tmp_path)
    initial = copy.deepcopy(model)
    previous = ContinuousParentTrainingHost(model, task, source, cfg, bindings).fit(
        tmp_path / "previous", stop_after_steps=0
    )
    source.loaded.clear()
    source.text_loaded.clear()
    changed = ContinuousParentTrainingHost(initial, task, source, replace(cfg, **change), bindings)
    with pytest.raises(TrainingCheckpointError, match="execution inputs drifted"):
        changed.fit(
            tmp_path / "changed",
            resume_checkpoint=previous.latest_checkpoint.path,
            resume_sha256=previous.latest_checkpoint.sha256,
        )
    assert source.loaded == source.text_loaded == []


@pytest.mark.parametrize("optimizer,decay", [("adam", 0.0), ("adamw", 1e-4)])
def test_literature_decay_dropout_interruption_resume_matches_uninterrupted(
    optimizer, decay, tmp_path
):
    model, task, source, cfg, bindings = _setup(tmp_path, batch_size=2)
    model.text_adapter[1] = torch.nn.Dropout(0.2)
    cfg = replace(cfg, optimizer=optimizer, weight_decay=decay)
    initial = copy.deepcopy(model)
    complete = ContinuousParentTrainingHost(model, task, source, cfg, bindings).fit(tmp_path / "full")
    interrupted = ContinuousParentTrainingHost(
        copy.deepcopy(initial), task, source, cfg, bindings
    ).fit(tmp_path / "interrupted", stop_after_steps=1)
    resumed_model = copy.deepcopy(initial)
    resumed = ContinuousParentTrainingHost(resumed_model, task, source, cfg, bindings).fit(
        tmp_path / "resumed",
        resume_checkpoint=interrupted.latest_checkpoint.path,
        resume_sha256=interrupted.latest_checkpoint.sha256,
    )
    assert resumed.global_step == complete.global_step == 4
    assert resumed.validation_history == complete.validation_history
    for name, value in model.state_dict().items():
        assert torch.equal(value, resumed_model.state_dict()[name])
    expected, _ = _load_torch_checkpoint(complete.latest_checkpoint.path)
    actual, _ = _load_torch_checkpoint(resumed.latest_checkpoint.path)
    for key in ("model", "optimizer", "scheduler", "rng"):
        assert _stable_hash(actual[key]) == _stable_hash(expected[key])
