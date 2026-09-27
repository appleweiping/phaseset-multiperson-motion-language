"""Execution qualifications; analytic checkpoints are not empirical runs."""

import copy
import errno
import hashlib
import json
from pathlib import Path
import time

import pytest
import torch

from phaseset_core.continuous_parent_host import ContinuousParentTrainingHost
from phaseset_core.host import HostConfigurationError, _AttemptLease
from phaseset_core.parent_run_monitor import ParentRunMonitor, parent_failure_code
from phaseset_core.training import (
    CheckpointArtifact,
    TrainingCheckpointError,
    _load_torch_checkpoint,
    _stable_hash,
)


def _payload(root, index):
    path = root / f"checkpoint-{index:012d}-{index + 1:06d}.pt"
    raw = f"analytic payload {index}".encode()
    path.write_bytes(raw)
    return CheckpointArtifact(path, hashlib.sha256(raw).hexdigest(), index)


def test_long_operation_heartbeat_is_chained_and_lease_is_process_released(tmp_path, monkeypatch):
    monkeypatch.setattr(ParentRunMonitor, "HEARTBEAT_SECONDS", 0.01)
    root = tmp_path / "attempt"
    root.mkdir()
    monitor = ParentRunMonitor(root)
    monitor.start()
    try:
        with pytest.raises(HostConfigurationError, match="live process"):
            _AttemptLease.acquire(root)
        monitor.observe(
            {
                "phase": "native_backward",
                "global_step": 3,
                "epoch": 0,
                "observed_at_utc": "2000-01-01T00:00:00+00:00",
            }
        )
        # No event or update occurs in this interval: the worker is independent.
        deadline = time.monotonic() + 5
        while not (root / "heartbeats" / "heartbeat-00000003.json").exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
    finally:
        monitor.finish(outcome="INTERRUPTED", failure=None)
        with pytest.raises(HostConfigurationError, match="live process"):
            _AttemptLease.acquire(root)
        monitor.release()
    with _AttemptLease.acquire(root):
        pass
    previous, elapsed = None, -1.0
    heartbeats = sorted((root / "heartbeats").glob("*.json"))
    for index, path in enumerate(heartbeats):
        raw = path.read_bytes()
        row = json.loads(raw)
        assert row["sequence"] == index and row["previous_heartbeat_sha256"] == previous
        assert row["process_elapsed_seconds"] >= elapsed
        assert row["formal_authority"] is False
        assert not row["observed_at_utc"].startswith("2000-")
        previous, elapsed = hashlib.sha256(raw).hexdigest(), row["process_elapsed_seconds"]
    assert (
        sum(json.loads(path.read_text())["phase"] == "native_backward" for path in heartbeats) >= 2
    )
    assert json.loads(heartbeats[-1].read_text())["outcome"] == "INTERRUPTED"
    assert monitor.summary()["latest_heartbeat_sha256"] == previous


@pytest.mark.parametrize("bad", [0, 1, True, 2.0])
def test_retention_requires_two_rollback_candidates(bad, tmp_path):
    with pytest.raises(ValueError, match="at least two"):
        ParentRunMonitor(tmp_path, keep_recent=bad)


@pytest.mark.parametrize("phase", ["mkdir", "heartbeat", "thread"])
def test_initialization_failure_keeps_lease_until_terminal_owner_releases(
    tmp_path, monkeypatch, phase
):
    import phaseset_core.parent_run_monitor as module

    root = tmp_path / "attempt"
    root.mkdir()
    monitor = ParentRunMonitor(root)
    error = OSError("actual monitor initialization failure")
    original_mkdir = Path.mkdir
    original_record = module._record

    def failing_mkdir(path, *args, **kwargs):
        if path.name == "checkpoint-receipts":
            raise error
        return original_mkdir(path, *args, **kwargs)

    def failing_record(*args, **kwargs):
        raise error

    def failing_thread(*args, **kwargs):
        raise error

    if phase == "mkdir":
        monkeypatch.setattr(Path, "mkdir", failing_mkdir)
    elif phase == "heartbeat":
        monkeypatch.setattr(module, "_record", failing_record)
    else:
        monkeypatch.setattr(module.threading.Thread, "start", failing_thread)
    try:
        with pytest.raises(OSError, match="initialization failure") as raised:
            monitor.start()
        assert raised.value is error
        with pytest.raises(HostConfigurationError, match="live process"):
            _AttemptLease.acquire(root)
    finally:
        monkeypatch.setattr(module, "_record", original_record)
        monitor.release()
    with _AttemptLease.acquire(root):
        pass


def test_retention_protects_recent_best_dependencies_and_external_predecessor(tmp_path):
    root = tmp_path / "attempt"
    root.mkdir()
    predecessor = _payload(tmp_path, 99)
    monitor = ParentRunMonitor(root, keep_recent=2)
    monitor.start()
    try:
        first, second, third, fourth = [_payload(root, index) for index in range(4)]
        monitor.checkpoint(first, best=first)
        monitor.checkpoint(second, best=first)
        monitor.checkpoint(third, best=third)
        assert first.path.exists()  # The second/rollback candidate still names it.
        monitor.checkpoint(fourth, best=third)
        assert not first.path.exists() and not second.path.exists()
        assert third.path.exists() and fourth.path.exists()
        fifth = _payload(root, 4)
        monitor.checkpoint(fifth, best=predecessor)
        assert third.path.exists()  # The fourth still names it as best.
        assert predecessor.path.exists()
        with pytest.raises(ValueError, match="own checkpoint"):
            monitor.checkpoint(predecessor, best=None)
    finally:
        monitor.finish(outcome="COMPLETED", failure=None)
        monitor.release()
    assert len(list((root / "checkpoint-receipts").glob("*.json"))) == 5
    assert len(list((root / "retention").glob("retired-*.json"))) == 2
    assert monitor.summary()["retired_payload_count"] == 2


def test_changed_payload_is_preserved_before_retirement(tmp_path):
    root = tmp_path / "attempt"
    root.mkdir()
    monitor = ParentRunMonitor(root, keep_recent=2)
    monitor.start()
    first, second, third = [_payload(root, index) for index in range(3)]
    try:
        monitor.checkpoint(first, best=None)
        monitor.checkpoint(second, best=None)
        first.path.write_bytes(b"actual corruption")
        with pytest.raises(TrainingCheckpointError, match="digest changed"):
            monitor.checkpoint(third, best=None)
        assert first.path.read_bytes() == b"actual corruption"
        assert not list((root / "retention").glob("*.json"))
    finally:
        monitor.finish(outcome="FAILED", failure=TrainingCheckpointError("corruption"))
        monitor.release()


def test_keep_all_and_close_never_prune_failed_or_predecessor_attempt(tmp_path):
    root = tmp_path / "attempt"
    root.mkdir()
    monitor = ParentRunMonitor(root)
    monitor.start()
    artifacts = [_payload(root, index) for index in range(4)]
    for item in artifacts:
        monitor.checkpoint(item, best=artifacts[0])
    monitor.finish(outcome="FAILED", failure=OSError(errno.ENOSPC, "full"))
    monitor.release()
    assert all(item.path.exists() for item in artifacts)
    assert parent_failure_code(OSError(errno.ENOSPC, "full")) == "DISK_FULL"
    assert parent_failure_code(TrainingCheckpointError("drift")) == "CHECKPOINT_INVALID"
    assert parent_failure_code(KeyboardInterrupt()) == "PROCESS_INTERRUPTED"
    assert parent_failure_code(MemoryError()) == "RESOURCE_LIMIT"


def test_worker_io_failure_surfaces_without_false_completion(tmp_path, monkeypatch):
    import phaseset_core.parent_run_monitor as module

    monkeypatch.setattr(ParentRunMonitor, "HEARTBEAT_SECONDS", 0.01)
    root = tmp_path / "attempt"
    root.mkdir()
    original = module._record
    error = OSError(errno.ENOSPC, "actual heartbeat disk full")

    def disk_full(path, payload):
        if payload.get("sequence") == 1:
            raise error
        return original(path, payload)

    monkeypatch.setattr(module, "_record", disk_full)
    monitor = ParentRunMonitor(root)
    monitor.start()
    try:
        assert monitor._stop.wait(5)
        with pytest.raises(OSError, match="heartbeat disk full") as raised:
            monitor.check()
        assert raised.value is error
        with pytest.raises(OSError, match="heartbeat disk full"):
            monitor.finish(outcome="COMPLETED", failure=None)
    finally:
        monitor.release()
    with _AttemptLease.acquire(root):
        pass
    assert len(list((root / "heartbeats").glob("*.json"))) == 1
    assert monitor.summary()["monitor_failure_class"] == "OSError"


@pytest.mark.parametrize("terminal_write_bad", [False, True])
def test_monitor_failure_releases_lease_and_preserves_primary_training_error(
    tmp_path, monkeypatch, terminal_write_bad
):
    from test_continuous_base_retrieval import _setup

    model, task, source, config, bindings = _setup(tmp_path)
    error = ValueError("actual source failure")

    def broken_source(*args, **kwargs):
        raise error

    source.text = broken_source
    original_finish = ParentRunMonitor.finish

    def broken_finish(monitor, **kwargs):
        original_finish(monitor, **kwargs)
        raise OSError("monitor close failed")

    monkeypatch.setattr(ParentRunMonitor, "finish", broken_finish)
    original_release = ParentRunMonitor.release

    def broken_release(monitor):
        original_release(monitor)
        raise OSError("lease release failed")

    monkeypatch.setattr(ParentRunMonitor, "release", broken_release)
    if terminal_write_bad:
        import phaseset_core.continuous_parent_host as host_module

        original_write = host_module._write_json_once

        def broken_terminal(path, payload):
            if path.name == "terminal.json":
                raise OSError(errno.ENOSPC, "terminal disk full")
            return original_write(path, payload)

        monkeypatch.setattr(host_module, "_write_json_once", broken_terminal)
    with pytest.raises(ValueError, match="actual source failure") as raised:
        ContinuousParentTrainingHost(model, task, source, config, bindings).fit(tmp_path / "failed")
    assert raised.value is error and any("monitor close" in note for note in error.__notes__)
    assert any("lease release" in note for note in error.__notes__)
    if terminal_write_bad:
        assert any("terminal evidence write" in note for note in error.__notes__)
        assert not (tmp_path / "failed" / "terminal.json").exists()
    else:
        row = json.loads((tmp_path / "failed" / "terminal.json").read_text())
        assert row["outcome"] == "FAILED" and row["failure_code"] == "INPUT_OR_CONTRACT"
    with _AttemptLease.acquire(tmp_path / "failed"):
        pass


@pytest.mark.parametrize("kind", ["base", "residual"])
def test_actual_host_retention_resume_is_bitwise_and_keeps_best(tmp_path, kind):
    if kind == "base":
        from test_continuous_base_retrieval import _setup
    else:
        from test_continuous_parent_host import _setup as residual_setup

        def _setup(path):
            return residual_setup(path, batch_size=2)

    model, task, source, config, bindings = _setup(tmp_path)
    initial = copy.deepcopy(model)
    full = ContinuousParentTrainingHost(model, task, source, config, bindings).fit(
        tmp_path / "full", checkpoint_keep_recent=2
    )
    stopped = ContinuousParentTrainingHost(
        copy.deepcopy(initial), task, source, config, bindings
    ).fit(tmp_path / "stopped", stop_after_steps=2, checkpoint_keep_recent=2)
    resumed_model = copy.deepcopy(initial)
    resumed = ContinuousParentTrainingHost(resumed_model, task, source, config, bindings).fit(
        tmp_path / "resumed",
        resume_checkpoint=stopped.latest_checkpoint.path,
        resume_sha256=stopped.latest_checkpoint.sha256,
        checkpoint_keep_recent=2,
    )
    assert full.outcome == resumed.outcome == "COMPLETED"
    assert full.validation_history == resumed.validation_history
    assert full.best_checkpoint.path.exists() and resumed.best_checkpoint.path.exists()
    assert stopped.latest_checkpoint.path.exists() and stopped.best_checkpoint.path.exists()
    for name, value in model.state_dict().items():
        assert torch.equal(value, resumed_model.state_dict()[name])
    expected, _ = _load_torch_checkpoint(full.latest_checkpoint.path)
    actual, _ = _load_torch_checkpoint(resumed.latest_checkpoint.path)
    for key in ("model", "optimizer", "scheduler", "rng"):
        assert _stable_hash(actual[key]) == _stable_hash(expected[key])
    terminal = json.loads((tmp_path / "full" / "terminal.json").read_text())
    assert terminal["monitor"]["retired_payload_count"] >= 1
    assert terminal["monitor"]["budget_or_study_authority"] is False
    assert len(list((tmp_path / "full").glob("checkpoint-*.pt"))) <= 4
    manifest = json.loads((tmp_path / "resumed" / "run.json").read_text())
    assert Path(manifest["predecessor_checkpoint"]) == stopped.latest_checkpoint.path
