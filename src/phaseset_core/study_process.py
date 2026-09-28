"""Linux owned-process timeout/cleanup and persistent study-cost settlement.

No SSH, shell evaluation, model or automatic retry. Scientific admission stays
with the existing private study operator; accounting never grants it.
"""

from __future__ import annotations

from contextlib import contextmanager
import errno
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

from .host import _AttemptLease, _write_once_json
from .study_budget import StudyBudget
from .study_storage import (
    StudyStorageHold,
    StudyStorageProjection,
    check_study_storage,
    current_free_bytes,
)


class StudyProcessError(RuntimeError):
    """Execution/cleanup evidence could not close; never retry automatically."""


class StudyResourceHold(StudyProcessError):
    """A requested card is not actually idle; do not preempt others."""


STORAGE_MONITOR_SECONDS = 30.0


@contextmanager
def _storage_lease(budget, enabled):
    """Serialize storage-admitted training across this study's cards."""
    if not enabled:
        yield
        return
    root = budget.root / "storage-lease"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    lease = _AttemptLease.acquire(root)
    try:
        yield
    finally:
        lease.release()


@contextmanager
def _termination_guard():
    received = []
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def record(signum, frame):
        received.append(signum)

    try:
        for sig in previous:
            signal.signal(sig, record)
        yield received
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def check_gpu_free(uuids: tuple[str, ...]):
    if not uuids:
        return
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    ).splitlines()
    memory = {row.split(",")[0].strip(): int(row.split(",")[1]) for row in rows}
    compute = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    ).splitlines()
    if any(uuid not in memory or memory[uuid] >= 500 for uuid in uuids) or any(
        row.split(",")[0].strip() in uuids for row in compute if row.strip()
    ):
        raise StudyResourceHold("requested GPU is not idle; no foreign process is stopped")


@contextmanager
def _gpu_leases(budget, uuids):
    leases = []
    try:
        for uuid in sorted(uuids):
            root = budget.root / "gpu-leases" / uuid
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
            leases.append(_AttemptLease.acquire(root))
        yield
    finally:
        primary = sys.exception()
        first_error = None
        for lease in reversed(leases):
            try:
                lease.release()
            except BaseException as error:
                if primary is not None:
                    primary.add_note(f"GPU lease release also failed: {error!r}")
                elif first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


def _live_group(group_id):
    live = set()
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            raw = path.read_text()
        except (FileNotFoundError, ProcessLookupError):
            continue
        fields = raw[raw.rfind(")") + 2 :].split()
        if int(fields[2]) == group_id and fields[0] != "Z":
            live.add(int(path.parent.name))
    return live


def _owned_gpu_clear(uuids, seen, *, timeout):
    if not uuids:
        return True
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
        timeout=timeout,
    ).splitlines()
    return not any(
        row.split(",")[0].strip() in uuids and int(row.split(",")[1]) in seen
        for row in rows
        if row.strip()
    )


def _leader_exited(process):
    # Keep the leader unreaped until group cleanup is verified: its PID/PGID
    # cannot be recycled into somebody else's process group while we signal it.
    return os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None


def _close_owned_group(process, uuids, grace, cleanup_seconds):
    """TERM then KILL this fresh session only; verify group and CUDA exit."""
    started = time.monotonic()
    force_at = started + grace
    deadline = force_at + cleanup_seconds
    seen = {process.pid}
    live = _live_group(process.pid)
    seen.update(live)
    descendants_after_leader_exit = _leader_exited(process) and bool(live)
    if live:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    forced = False
    while time.monotonic() < deadline:
        leader_exited = _leader_exited(process)
        live = _live_group(process.pid)
        seen.update(live)
        if not live and leader_exited:
            remaining = deadline - time.monotonic()
            if remaining > 0 and _owned_gpu_clear(uuids, seen, timeout=min(2, remaining)):
                process.wait(timeout=0)
                return {
                    "observed_owned_pids": sorted(seen),
                    "owned_group_and_cuda_exit_verified": True,
                    "forced_kill": forced,
                    "descendants_after_leader_exit": descendants_after_leader_exit,
                }
        if live and not forced and time.monotonic() >= force_at:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            forced = True
        time.sleep(0.05)
    raise StudyProcessError("owned group/CUDA exit unverified; keep the full budget reservation")


def _cursor(terminal_path):
    try:
        value = json.loads(terminal_path.read_bytes())
        step, outcome = value["global_step"], value["outcome"]
        if (
            type(step) is not int
            or step < 0
            or outcome not in ("COMPLETED", "FAILED", "INTERRUPTED")
        ):
            raise ValueError("invalid host terminal cursor/outcome")
        return step, outcome, None
    except (OSError, ValueError, KeyError, TypeError) as error:
        return None, None, f"{type(error).__name__}: {error}"


def run_budgeted_process(
    budget: StudyBudget,
    *,
    process_directory: Path,
    argv: tuple[str, ...],
    working_directory: Path,
    environment: dict[str, str],
    reservation: dict,
    gpu_uuids: tuple[str, ...],
    timeout_seconds: int,
    kill_grace_seconds: int,
    cleanup_seconds: int,
    host_terminal_path: Path,
    bindings: dict,
    storage_projection: StudyStorageProjection | None = None,
):
    """One immutable process attempt, never a retry or a scientific grant.

    Registered children must remain in their fresh POSIX session (no daemon/
    setsid escape). ContinuousParentTrainingHost and its literature subclass
    produce the outcome/global_step terminal cursor consumed here, not the
    superseded legacy Host terminal schema. Missing/truncated evidence means
    FAILED and unknown cursor,
    not zero updates or successful training; observed cost is still charged.
    """
    if sys.platform != "linux":
        raise StudyProcessError("production owned-group controller requires Linux /proc")
    if threading.current_thread() is not threading.main_thread():
        raise StudyProcessError("each production controller runs in its own main-thread process")
    for value in (timeout_seconds, kill_grace_seconds, cleanup_seconds):
        if type(value) is not int or value < 1:
            raise ValueError(
                "timeout, TERM grace and exit-verification envelope are positive integers"
            )
    if (
        type(argv) is not tuple
        or not argv
        or any(type(arg) is not str or not arg for arg in argv)
        or not Path(argv[0]).is_absolute()
        or type(gpu_uuids) is not tuple
        or len(set(gpu_uuids)) != len(gpu_uuids)
        or any(
            type(uuid) is not str or not re.fullmatch(r"GPU-[A-Za-z0-9-]+", uuid)
            for uuid in gpu_uuids
        )
        or reservation.get("gpu_count") != len(gpu_uuids)
        or reservation.get("wall_time_limit_seconds")
        != timeout_seconds + 2 * kill_grace_seconds + cleanup_seconds
    ):
        raise ValueError("exact argv/card-count/full timeout envelope required")
    if (
        environment.get("CUDA_VISIBLE_DEVICES") != ",".join(gpu_uuids)
        or any(
            environment.get(key) != "1"
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        )
        or environment.get("PYTHONHASHSEED") != "0"
    ):
        raise ValueError("fixed visible cards/CPU1/PYTHONHASHSEED=0 required")
    root = Path(process_directory).resolve()
    cwd = Path(working_directory).resolve()
    terminal = Path(host_terminal_path).resolve()
    if not cwd.is_dir() or terminal.exists():
        raise ValueError("existing working directory and fresh host terminal required")
    purpose = reservation.get("purpose")
    if purpose in ("pilot", "formal") and storage_projection is None:
        raise StudyStorageHold("pilot/formal requires a family-specific storage projection")
    if purpose == "formal" and (
        storage_projection is None or storage_projection.cumulative_remaining_bytes is None
    ):
        raise StudyStorageHold("formal requires a reviewed cumulative remaining-byte forecast")
    if storage_projection is not None and not environment.get("TMPDIR"):
        raise StudyStorageHold("storage-admitted attempt needs an explicit TMPDIR")
    storage = (
        None
        if storage_projection is None
        else check_study_storage(root.parent, environment["TMPDIR"], storage_projection)
    )
    check_gpu_free(gpu_uuids)
    with (
        _storage_lease(budget, storage_projection is not None),
        _gpu_leases(budget, gpu_uuids),
        _termination_guard() as received_signals,
    ):
        check_gpu_free(gpu_uuids)
        if storage_projection is not None:
            storage = check_study_storage(root.parent, environment["TMPDIR"], storage_projection)
        root.mkdir(mode=0o700, exist_ok=False)
        lease = _AttemptLease.acquire(root)
        primary = None
        try:
            _write_once_json(
                root / "launch.json",
                {
                    "argv": argv,
                    "working_directory": str(cwd),
                    "reservation": reservation,
                    "gpu_uuids": gpu_uuids,
                    "bindings": bindings,
                    "host_terminal_path": str(terminal),
                    "storage_admission": storage,
                    "scientific_or_launch_authority": False,
                },
            )
            with (
                (root / "stdout.log").open("xb") as stdout,
                (root / "stderr.log").open("xb") as stderr,
            ):
                budget.reserve(**reservation)
                process, started, timed_out, cleanup = None, None, False, None
                storage_observations = 0
                storage_min_available = None if storage is None else storage["available_bytes"]
                try:
                    check_gpu_free(gpu_uuids)
                    if received_signals:
                        raise KeyboardInterrupt("controller terminated before child spawn")
                    started = time.monotonic()
                    next_storage_sample = started + STORAGE_MONITOR_SECONDS
                    process = subprocess.Popen(
                        (
                            "/usr/bin/timeout",
                            f"--kill-after={kill_grace_seconds}s",
                            f"{timeout_seconds}s",
                            *argv,
                        ),
                        cwd=cwd,
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout,
                        stderr=stderr,
                        start_new_session=True,
                        shell=False,
                    )
                    _write_once_json(
                        root / "process.json", {"pid": process.pid, "group_id": process.pid}
                    )
                    while not _leader_exited(process):
                        if received_signals:
                            raise KeyboardInterrupt("controller received termination signal")
                        if storage_projection is not None and time.monotonic() >= next_storage_sample:
                            available = current_free_bytes(root.parent)
                            storage_observations += 1
                            storage_min_available = min(storage_min_available, available)
                            if available < storage_projection.monitor_floor_bytes:
                                raise StudyStorageHold(
                                    "RESOURCE_LIMIT: training fell below storage monitor floor"
                                )
                            next_storage_sample = time.monotonic() + STORAGE_MONITOR_SECONDS
                        if time.monotonic() - started >= timeout_seconds + kill_grace_seconds:
                            timed_out = True
                            break
                        time.sleep(0.05)
                except BaseException as error:
                    primary = error
                if process is not None:
                    try:
                        cleanup = _close_owned_group(
                            process, gpu_uuids, kill_grace_seconds, cleanup_seconds
                        )
                    except BaseException as error:
                        if primary is not None:
                            primary.add_note(f"owned cleanup also failed: {error!r}")
                        else:
                            primary = error
                        try:
                            _write_once_json(
                                root / "cleanup-unverified.json",
                                {
                                    "pid": process.pid,
                                    "reservation_retained": True,
                                    "error": repr(primary),
                                    "scientific_or_launch_authority": False,
                                },
                            )
                        except BaseException as record_error:
                            primary.add_note(
                                f"cleanup evidence write also failed: {record_error!r}"
                            )
                        raise primary
                elif started is not None and not (
                    isinstance(primary, OSError)
                    and primary.errno in (errno.ENOENT, errno.EACCES, errno.ENOEXEC, errno.ENOTDIR)
                ):
                    # Interrupted Popen may have created a child without returning
                    # its handle. Do not invent zero allocation or kill guessed PIDs.
                    try:
                        _write_once_json(
                            root / "cleanup-unverified.json",
                            {
                                "pid": None,
                                "reservation_retained": True,
                                "error": repr(primary),
                                "scientific_or_launch_authority": False,
                            },
                        )
                    except BaseException as record_error:
                        primary.add_note(f"startup evidence write also failed: {record_error!r}")
                    raise primary
                wall = 0.0 if process is None else time.monotonic() - started
                if received_signals and primary is None:
                    primary = KeyboardInterrupt("controller received termination signal")
                watchdog_forced_kill = (
                    process is not None
                    and process.returncode in (-9, 137)
                    and wall >= timeout_seconds
                )
                timed_out = (
                    timed_out
                    or (process is not None and process.returncode == 124)
                    or watchdog_forced_kill
                )
                step, host_outcome, cursor_error = (
                    (0, None, None) if process is None else _cursor(terminal)
                )
                outcome = (
                    "NOT_STARTED"
                    if process is None
                    else (
                        "INTERRUPTED"
                        if timed_out or isinstance(primary, (KeyboardInterrupt, SystemExit))
                        else "FAILED"
                        if primary is not None
                        or process.returncode != 0
                        or cursor_error is not None
                        or cleanup["descendants_after_leader_exit"]
                        or step > reservation["planned_steps"]
                        else host_outcome
                    )
                )
                report = {
                    "attempt_id": reservation["attempt_id"],
                    "outcome": outcome,
                    "returncode": None if process is None else process.returncode,
                    "timed_out": timed_out,
                    "watchdog_forced_kill": watchdog_forced_kill,
                    "received_termination_signals": received_signals,
                    "global_step": step,
                    "cursor_error": cursor_error,
                    "actual_wall_seconds": wall,
                    "actual_gpu_seconds": wall * reservation["gpu_count"],
                    "storage_monitor_observations": storage_observations,
                    "storage_min_available_bytes": storage_min_available,
                    "cleanup": cleanup,
                    "primary_error": None if primary is None else repr(primary),
                    "scientific_or_launch_authority": False,
                }
                try:
                    _write_once_json(root / "process-terminal.json", report)
                    budget.settle(
                        reservation["attempt_id"],
                        actual_wall_seconds=wall,
                        completed_steps=step,
                        outcome=outcome,
                        evidence=str(root / "process-terminal.json"),
                    )
                    _write_once_json(
                        root / "settlement.json",
                        {
                            "actual_settlement_completed": True,
                            "summary": budget.summary(),
                        },
                    )
                except BaseException as record_error:
                    if primary is not None:
                        primary.add_note(
                            f"terminal/budget evidence write also failed: {record_error!r}"
                        )
                        raise primary
                    raise
                if primary is not None:
                    raise primary
                return report
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                lease.release()
            except BaseException as error:
                if primary is not None:
                    primary.add_note(f"process lease release also failed: {error!r}")
                else:
                    raise
