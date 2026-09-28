"""Real Linux stdlib children; no model, native training or GPU is launched."""

import errno
import json
import os
from pathlib import Path
import sys
import signal
import subprocess
import threading
import time

import pytest

from phaseset_core.host import _AttemptLease, HostConfigurationError
import phaseset_core.study_process as module
from phaseset_core.study_process import StudyProcessError, StudyResourceHold, run_budgeted_process
from phaseset_core.study_storage import StudyStorageHold, StudyStorageProjection
from test_study_budget import create as create_budget


linux = pytest.mark.skipif(
    sys.platform != "linux", reason="registered production controller is Linux-only"
)


def inputs(tmp_path, *, timeout=2, body=None):
    work = tmp_path / "work"
    work.mkdir()
    terminal = work / "host-terminal.json"
    body = (
        body
        or "terminal.write_text(json.dumps({'global_step':0,'outcome':'COMPLETED'})); print('STDLIB_CHILD_EXECUTED')"
    )
    code = (
        "import json,time,signal,subprocess,sys; from pathlib import Path; terminal=Path(sys.argv[1]); "
        + body
    )
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONHASHSEED="0",
        PYTHONDONTWRITEBYTECODE="1",
    )
    return {
        "process_directory": work / "controller",
        "working_directory": work,
        "argv": (sys.executable, "-B", "-c", code, str(terminal)),
        "environment": env,
        "reservation": {
            "attempt_id": "analytic-process-1",
            "run_id": "analytic-process-profile-1",
            "purpose": "profile",
            "seed": 1729,
            "gpu_count": 0,
            "wall_time_limit_seconds": timeout + 5,
            "planned_steps": 0,
            "evidence": "stdlib process qualification only; no training/GPU",
        },
        "gpu_uuids": (),
        "timeout_seconds": timeout,
        "kill_grace_seconds": 1,
        "cleanup_seconds": 3,
        "host_terminal_path": terminal,
        "bindings": {"stage": "analytic stdlib child; not native optimizer evidence"},
    }


def report(args):
    return json.loads((args["process_directory"] / "process-terminal.json").read_text())


def simulate_card(args, monkeypatch):
    # Pure accounting/gate injection. The stdlib child never imports a CUDA library.
    args["gpu_uuids"] = ("GPU-analytic-not-real-hardware",)
    args["environment"]["CUDA_VISIBLE_DEVICES"] = args["gpu_uuids"][0]
    args["reservation"]["gpu_count"] = 1
    monkeypatch.setattr(module, "check_gpu_free", lambda uuids: None)
    monkeypatch.setattr(module, "_owned_gpu_clear", lambda *args, **kwargs: True)


def test_non_linux_rejects_before_any_launch(tmp_path, monkeypatch):
    budget = create_budget(tmp_path)
    args = inputs(tmp_path)
    monkeypatch.setattr(module.sys, "platform", "unsupported-qualification-platform")
    with pytest.raises(StudyProcessError, match="requires Linux"):
        run_budgeted_process(budget, **args)
    assert not args["process_directory"].exists()
    assert budget.summary()["unsettled_attempt_ids"] == []


@linux
def test_training_requires_storage_bound_before_attempt_or_budget(tmp_path):
    budget = create_budget(tmp_path)
    args = inputs(tmp_path)
    args["reservation"]["purpose"] = "pilot"
    with pytest.raises(StudyStorageHold, match="family-specific"):
        run_budgeted_process(budget, **args)
    assert not args["process_directory"].exists()
    assert budget.summary()["usage"]["pilot_reservations"] == 0


@linux
def test_formal_requires_cumulative_forecast_before_attempt(tmp_path):
    budget = create_budget(tmp_path)
    args = inputs(tmp_path)
    args["reservation"]["purpose"] = "formal"
    args["environment"]["TMPDIR"] = str(tmp_path)
    args["storage_projection"] = StudyStorageProjection(1, 0, free_floor_bytes=0)
    with pytest.raises(StudyStorageHold, match="cumulative"):
        run_budgeted_process(budget, **args)
    assert not args["process_directory"].exists()


@linux
def test_profile_storage_snapshot_is_bound_into_launch(tmp_path):
    budget = create_budget(tmp_path)
    args = inputs(tmp_path)
    args["environment"]["TMPDIR"] = str(tmp_path)
    args["storage_projection"] = StudyStorageProjection(1, 0, free_floor_bytes=0)
    result = run_budgeted_process(budget, **args)
    assert result["outcome"] == "COMPLETED"
    launch = json.loads((args["process_directory"] / "launch.json").read_text())
    assert launch["storage_admission"]["required_free_bytes"] == 6
    assert launch["storage_admission"]["available_bytes"] >= 6


@linux
def test_storage_admitted_process_is_serialized_before_attempt(tmp_path):
    budget = create_budget(tmp_path)
    args = inputs(tmp_path)
    args["environment"]["TMPDIR"] = str(tmp_path)
    args["storage_projection"] = StudyStorageProjection(1, 0, free_floor_bytes=0)
    lease_root = budget.root / "storage-lease"
    lease_root.mkdir()
    with _AttemptLease.acquire(lease_root):
        with pytest.raises(HostConfigurationError):
            run_budgeted_process(budget, **args)
    assert not args["process_directory"].exists()


@linux
def test_storage_monitor_stops_only_owned_child_and_records_failure(tmp_path, monkeypatch):
    budget = create_budget(tmp_path)
    args = inputs(
        tmp_path,
        body="time.sleep(0.4); terminal.write_text(json.dumps({'global_step':0,'outcome':'COMPLETED'}))",
    )
    args["environment"]["TMPDIR"] = str(tmp_path)
    args["storage_projection"] = StudyStorageProjection(1, 0, free_floor_bytes=100)
    monkeypatch.setattr(module, "STORAGE_MONITOR_SECONDS", 0.02)
    monkeypatch.setattr(module, "current_free_bytes", lambda _path: 0)
    with pytest.raises(StudyStorageHold, match="storage monitor floor"):
        run_budgeted_process(budget, **args)
    result = report(args)
    assert result["outcome"] == "FAILED"
    assert "storage monitor floor" in result["primary_error"]
    assert result["storage_monitor_observations"] >= 1
    assert result["storage_min_available_bytes"] == 0
    assert result["cleanup"]["owned_group_and_cuda_exit_verified"]
    assert budget.summary()["unsettled_attempt_ids"] == []


@linux
def test_real_completed_child_logs_terminal_and_settlement_are_distinct(tmp_path):
    budget = create_budget(tmp_path)
    args = inputs(tmp_path)
    result = run_budgeted_process(budget, **args)
    assert result["outcome"] == "COMPLETED" and result["global_step"] == 0
    assert result["returncode"] == 0 and result["actual_wall_seconds"] > 0
    assert (
        result["actual_gpu_seconds"] == 0
        and result["cleanup"]["owned_group_and_cuda_exit_verified"]
    )
    assert "STDLIB_CHILD_EXECUTED" in (args["process_directory"] / "stdout.log").read_text()
    assert (args["process_directory"] / "stderr.log").stat().st_size == 0
    assert json.loads((args["process_directory"] / "settlement.json").read_text())[
        "actual_settlement_completed"
    ]
    assert budget.summary()["usage"]["gpu_seconds"] == 34
    assert budget.summary()["unsettled_attempt_ids"] == []
    with pytest.raises(ValueError, match="fresh host terminal"):
        run_budgeted_process(budget, **args)


@linux
def test_real_failed_child_keeps_logs_and_simulated_allocated_card_wall_cost(tmp_path, monkeypatch):
    budget = create_budget(tmp_path)
    args = inputs(
        tmp_path,
        body="terminal.write_text(json.dumps({'global_step':0,'outcome':'FAILED'})); print('actual child failure',file=sys.stderr); sys.exit(7)",
    )
    simulate_card(args, monkeypatch)
    result = run_budgeted_process(budget, **args)
    assert result["outcome"] == "FAILED" and result["returncode"] == 7
    assert result["actual_gpu_seconds"] == result["actual_wall_seconds"] > 0
    assert budget.summary()["usage"]["gpu_seconds"] == 34 + result["actual_wall_seconds"]
    assert "actual child failure" in (args["process_directory"] / "stderr.log").read_text()


@linux
def test_timeout_closes_real_child_and_does_not_invent_optimizer_cursor(tmp_path):
    budget = create_budget(tmp_path)
    args = inputs(tmp_path, timeout=1, body="time.sleep(60)")
    result = run_budgeted_process(budget, **args)
    assert result["timed_out"] and result["outcome"] == "INTERRUPTED"
    assert result["global_step"] is None and result["cleanup"]["owned_group_and_cuda_exit_verified"]
    assert budget.summary()["unknown_optimizer_cursor_attempt_ids"] == ["analytic-process-1"]
    pid = json.loads((args["process_directory"] / "process.json").read_text())["pid"]
    assert module._live_group(pid) == set()


@linux
def test_ignored_term_is_forced_killed_and_exit_is_actually_verified(tmp_path):
    budget = create_budget(tmp_path)
    args = inputs(
        tmp_path, timeout=1, body="signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"
    )
    result = run_budgeted_process(budget, **args)
    assert result["timed_out"] and (
        result["cleanup"]["forced_kill"] or result["watchdog_forced_kill"]
    )
    assert result["returncode"] == -9


@linux
def test_exited_leader_does_not_leave_own_descendant_or_claim_completed(tmp_path, monkeypatch):
    budget = create_budget(tmp_path)
    args = inputs(
        tmp_path,
        body="subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); terminal.write_text(json.dumps({'global_step':0,'outcome':'COMPLETED'}))",
    )
    original_kill = module.os.killpg
    observed = []

    def checked_kill(group, sig):
        raw = (Path("/proc") / str(group) / "stat").read_text()
        state = raw[raw.rfind(")") + 2 :].split()[0]
        assert state == "Z"  # Leader PID retained, not available for foreign reuse.
        observed.append(group)
        return original_kill(group, sig)

    monkeypatch.setattr(module.os, "killpg", checked_kill)
    result = run_budgeted_process(budget, **args)
    assert result["returncode"] == 0 and result["outcome"] == "FAILED"
    assert result["cleanup"]["descendants_after_leader_exit"]
    assert len(result["cleanup"]["observed_owned_pids"]) == 2
    assert observed
    assert module._live_group(result["cleanup"]["observed_owned_pids"][0]) == set()


@linux
@pytest.mark.parametrize("body", ["pass", "terminal.write_text('{')"])
def test_missing_or_truncated_host_terminal_is_not_success_or_zero_updates(tmp_path, body):
    budget = create_budget(tmp_path)
    args = inputs(tmp_path, body=body)
    result = run_budgeted_process(budget, **args)
    assert result["returncode"] == 0 and result["outcome"] == "FAILED"
    assert result["global_step"] is None and result["cursor_error"]
    assert budget.summary()["unsettled_attempt_ids"] == []


@linux
def test_busy_card_holds_before_creating_attempt_or_reserving(tmp_path, monkeypatch):
    budget = create_budget(tmp_path)
    args = inputs(tmp_path)
    simulate_card(args, monkeypatch)

    def busy(uuids):
        raise StudyResourceHold("injected other compute, no foreign process stopped")

    monkeypatch.setattr(module, "check_gpu_free", busy)
    with pytest.raises(StudyResourceHold, match="other compute"):
        run_budgeted_process(budget, **args)
    assert not args["process_directory"].exists()
    assert budget.summary()["unsettled_attempt_ids"] == []


@linux
def test_card_busy_after_reservation_settles_only_verified_unstarted_zero(tmp_path, monkeypatch):
    budget = create_budget(tmp_path)
    args = inputs(tmp_path)
    simulate_card(args, monkeypatch)
    calls = []

    def race(uuids):
        calls.append(uuids)
        if len(calls) == 3:
            raise StudyResourceHold("actual pre-spawn gate fixture")

    monkeypatch.setattr(module, "check_gpu_free", race)
    with pytest.raises(StudyResourceHold, match="pre-spawn"):
        run_budgeted_process(budget, **args)
    assert report(args)["outcome"] == "NOT_STARTED" and report(args)["actual_wall_seconds"] == 0
    assert budget.summary()["unsettled_attempt_ids"] == []
    assert not (args["process_directory"] / "process.json").exists()


@linux
def test_known_exec_failure_is_recorded_without_refunding_reservation_identity(
    tmp_path, monkeypatch
):
    budget = create_budget(tmp_path)
    args = inputs(tmp_path)

    def missing_watchdog(*args, **kwargs):
        raise FileNotFoundError(errno.ENOENT, "injected missing watchdog executable")

    monkeypatch.setattr(module.subprocess, "Popen", missing_watchdog)
    with pytest.raises(FileNotFoundError):
        run_budgeted_process(budget, **args)
    assert report(args)["outcome"] == "NOT_STARTED" and report(args)["global_step"] == 0
    assert budget.summary()["unsettled_attempt_ids"] == []
    assert len(list(budget.root.glob("event-*.json"))) == 3


@linux
def test_interrupted_spawn_without_returned_handle_does_not_invent_unstarted_refund(
    tmp_path, monkeypatch
):
    budget = create_budget(tmp_path)
    args = inputs(tmp_path)

    def ambiguous(*args, **kwargs):
        raise KeyboardInterrupt("injected Popen handle ambiguity; no real child in this case")

    monkeypatch.setattr(module.subprocess, "Popen", ambiguous)
    with pytest.raises(KeyboardInterrupt, match="handle ambiguity"):
        run_budgeted_process(budget, **args)
    assert budget.summary()["unsettled_attempt_ids"] == ["analytic-process-1"]
    assert (
        json.loads((args["process_directory"] / "cleanup-unverified.json").read_text())["pid"]
        is None
    )
    assert not (args["process_directory"] / "settlement.json").exists()


@linux
def test_own_controller_sigterm_closes_child_settles_and_restores_handler(tmp_path):
    budget = create_budget(tmp_path)
    args = inputs(tmp_path, timeout=10, body="time.sleep(60)")
    old_handler = signal.getsignal(signal.SIGTERM)
    emitted = []

    def send_own_signal():
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if (args["process_directory"] / "process.json").exists():
                emitted.append(True)
                os.kill(os.getpid(), signal.SIGTERM)
                return
            time.sleep(0.02)

    sender = threading.Thread(target=send_own_signal)
    sender.start()
    try:
        with pytest.raises(KeyboardInterrupt, match="termination signal"):
            run_budgeted_process(budget, **args)
    finally:
        sender.join(timeout=6)
    assert emitted and not sender.is_alive()
    assert signal.getsignal(signal.SIGTERM) == old_handler
    assert report(args)["outcome"] == "INTERRUPTED"
    assert report(args)["received_termination_signals"] == [signal.SIGTERM]
    assert budget.summary()["unsettled_attempt_ids"] == []


@linux
def test_autonomous_watchdog_stops_child_after_controller_sigkill(tmp_path):
    args = inputs(tmp_path, timeout=1, body="time.sleep(60)")
    encoded = json.dumps(
        {key: str(value) if isinstance(value, Path) else value for key, value in args.items()}
    )
    code = (
        "import json,sys; from pathlib import Path; from test_study_budget import create; "
        "from phaseset_core.study_process import run_budgeted_process; "
        "args=json.loads(sys.argv[1]); "
        "args.update({k:Path(args[k]) for k in ('process_directory','working_directory','host_terminal_path')}); "
        "args['argv']=tuple(args['argv']); args['gpu_uuids']=tuple(args['gpu_uuids']); "
        "run_budgeted_process(create(Path(sys.argv[2])),**args)"
    )
    controller = subprocess.Popen(
        (sys.executable, "-B", "-c", code, encoded, str(tmp_path)),
        env=args["environment"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    watchdog_pid, start_ticks = None, None
    try:
        deadline = time.monotonic() + 20
        pid_path = args["process_directory"] / "process.json"
        while not pid_path.exists() and time.monotonic() < deadline:
            assert controller.poll() is None
            time.sleep(0.02)
        assert pid_path.exists()
        watchdog_pid = json.loads(pid_path.read_text())["pid"]
        raw = (Path("/proc") / str(watchdog_pid) / "stat").read_text()
        start_ticks = raw[raw.rfind(")") + 2 :].split()[19]
        controller.kill()
        assert controller.wait(timeout=5) == -9
        deadline = time.monotonic() + 5
        while module._live_group(watchdog_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert module._live_group(watchdog_pid) == set()
        from phaseset_core.study_budget import StudyBudget

        assert StudyBudget(tmp_path / "budget").summary()["unsettled_attempt_ids"] == [
            "analytic-process-1"
        ]
        assert not (args["process_directory"] / "settlement.json").exists()
    finally:
        if controller.poll() is None:
            controller.kill()
            controller.wait(timeout=5)
        if watchdog_pid is not None and module._live_group(watchdog_pid):
            path = Path("/proc") / str(watchdog_pid) / "stat"
            if path.exists():
                raw = path.read_text()
                if raw[raw.rfind(")") + 2 :].split()[19] == start_ticks:
                    os.killpg(watchdog_pid, signal.SIGKILL)


@linux
def test_pid_receipt_write_failure_still_closes_real_owned_child_and_preserves_primary(
    tmp_path, monkeypatch
):
    budget = create_budget(tmp_path)
    args = inputs(tmp_path, body="time.sleep(60)")
    original = module._write_once_json

    def broken(path, value):
        if path.name == "process.json":
            raise OSError(errno.ENOSPC, "primary PID write disk fixture")
        return original(path, value)

    monkeypatch.setattr(module, "_write_once_json", broken)
    with pytest.raises(OSError, match="primary PID write"):
        run_budgeted_process(budget, **args)
    assert report(args)["cleanup"]["owned_group_and_cuda_exit_verified"]
    assert budget.summary()["unsettled_attempt_ids"] == []


@linux
def test_terminal_write_failure_preserves_reservation_after_child_exit(tmp_path, monkeypatch):
    budget = create_budget(tmp_path)
    args = inputs(tmp_path)
    original = module._write_once_json

    def broken(path, value):
        if path.name == "process-terminal.json":
            raise OSError(errno.ENOSPC, "terminal disk fixture")
        return original(path, value)

    monkeypatch.setattr(module, "_write_once_json", broken)
    with pytest.raises(OSError, match="terminal disk"):
        run_budgeted_process(budget, **args)
    pid = json.loads((args["process_directory"] / "process.json").read_text())["pid"]
    assert module._live_group(pid) == set()
    assert budget.summary()["unsettled_attempt_ids"] == ["analytic-process-1"]
    assert not (args["process_directory"] / "settlement.json").exists()


@linux
def test_settlement_write_failure_is_not_falsely_reported_as_settled(tmp_path, monkeypatch):
    budget = create_budget(tmp_path)
    args = inputs(tmp_path)

    def broken(*args, **kwargs):
        raise OSError(errno.ENOSPC, "budget event disk fixture")

    monkeypatch.setattr(budget, "settle", broken)
    with pytest.raises(OSError, match="budget event disk"):
        run_budgeted_process(budget, **args)
    assert (args["process_directory"] / "process-terminal.json").exists()
    assert not (args["process_directory"] / "settlement.json").exists()
    assert budget.summary()["unsettled_attempt_ids"] == ["analytic-process-1"]


@linux
def test_cleanup_uncertainty_never_refunds_even_if_test_child_really_exited(tmp_path, monkeypatch):
    budget = create_budget(tmp_path)
    args = inputs(tmp_path)
    original = module._close_owned_group

    def uncertain(*args, **kwargs):
        original(*args, **kwargs)
        raise StudyProcessError("injected uncertain exit proof")

    monkeypatch.setattr(module, "_close_owned_group", uncertain)
    with pytest.raises(StudyProcessError, match="uncertain exit"):
        run_budgeted_process(budget, **args)
    assert json.loads((args["process_directory"] / "cleanup-unverified.json").read_text())[
        "reservation_retained"
    ]
    assert budget.summary()["unsettled_attempt_ids"] == ["analytic-process-1"]


@linux
def test_cooperating_controller_gpu_lease_collision_cannot_double_launch(tmp_path, monkeypatch):
    budget = create_budget(tmp_path)
    args = inputs(tmp_path)
    simulate_card(args, monkeypatch)
    gpu_root = budget.root / "gpu-leases" / args["gpu_uuids"][0]
    gpu_root.mkdir(parents=True)
    with _AttemptLease.acquire(gpu_root):
        with pytest.raises(HostConfigurationError, match="held"):
            run_budgeted_process(budget, **args)
    assert not args["process_directory"].exists()
    assert budget.summary()["unsettled_attempt_ids"] == []
