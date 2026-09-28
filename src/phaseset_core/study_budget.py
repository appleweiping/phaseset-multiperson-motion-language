"""Persistent V2 study-wide reservations and observed resource accounting.

Accounting is not scientific/rights/hardware admission. Existing attempt
receipts prove those separately. No model, optimizer or SSH is launched here.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from datetime import datetime, timezone

from .execution import _write_once
from .host import _AttemptLease
from .study_storage import StudyStorageHold, StudyStorageProjection, check_study_storage


PILOT_ALLOCATION = {"TMR-Set": 3, "WaMo-Set": 3, "MIME-Set": 3, "B2": 1, "PhaseSet-V2-head": 2}
LIMITS = {
    "gpu_seconds": 300 * 3600,
    "formal_reservations": 90,
    "pilot_reservations": 12,
    "formal_restarts": 3,
    "architecture_reworks": 2,
}


class StudyBudgetError(ValueError):
    """Invalid accounting input, exhausted allowance or unresolved predecessor."""


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise StudyBudgetError(f"{name} must be an integer >= {minimum}")
    return value


def _seconds(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise StudyBudgetError(f"{name} must be finite nonnegative seconds")
    return float(value)


def _text(value, name):
    if type(value) is not str or not value.strip():
        raise StudyBudgetError(f"{name} must be explicit nonempty text")
    return value


def _fsync_directory(path):
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class StudyBudget:
    """Append-only transactions under the already qualified lifetime lease.

    Reserve before spawning; the full wall-time envelope remains charged until
    verified settlement. Reservation counts are conservative upper bounds on
    actual starts: failed spawning does NOT refund a pilot/formal slot. All
    allocated GPUs multiply wall time, including failed and interrupted work.
    Missing processes or old heartbeats never silently refund reservations.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        if not (self.root / "event-00000000.json").is_file():
            raise StudyBudgetError("budget absent; import verified history explicitly, never reset")

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        matrix: dict,
        historical_usage: dict,
        historical_evidence: tuple[str, ...],
        pilot_formal_steps: dict[str, int],
    ):
        """One explicit bootstrap; no default-zero history or overwrite path."""
        budget = matrix.get("budget", {})
        expected = {
            "gpu_hours": 300,
            "formal_stages": 87,
            "max_formal_starts": 90,
            "max_pilot_stages": 12,
            "max_root_cause_restarts": 3,
            "max_architecture_revisions": 2,
            "pilot_max_formal_step_fraction": 0.2,
        }
        if any(budget.get(key) != value for key, value in expected.items()):
            raise StudyBudgetError("the fixed V2 study budget cannot be raised")
        runs = matrix.get("runs", [])
        run_seeds = {row["id"]: row["seed"] for row in runs}
        if (
            len(runs) != 87
            or set(run_seeds) != {f"V2-{i:03d}" for i in range(1, 88)}
            or any(
                type(seed) is not int or seed not in (1729, 2718, 31415)
                for seed in run_seeds.values()
            )
            or matrix.get("seeds") != [1729, 2718, 31415]
        ):
            raise StudyBudgetError("the complete 87-stage/three-seed matrix is required")
        if set(historical_usage) != set(LIMITS) | {"pilot_families"}:
            raise StudyBudgetError("all historical counters must be imported explicitly")
        _seconds(historical_usage["gpu_seconds"], "historical GPU time")
        for key in set(LIMITS) - {"gpu_seconds"}:
            _integer(historical_usage[key], key)
        families = historical_usage["pilot_families"]
        if (
            type(families) is not dict
            or set(families) != set(PILOT_ALLOCATION)
            or sum(_integer(value, "historical family count") for value in families.values())
            != historical_usage["pilot_reservations"]
        ):
            raise StudyBudgetError("historical pilot family census must match the total")
        if any(
            historical_usage[key]
            for key in ("formal_reservations", "pilot_reservations", "formal_restarts")
        ):
            raise StudyBudgetError(
                "historical training needs its original detailed journal, not a new summary bootstrap"
            )
        if not historical_evidence or type(historical_evidence) is not tuple:
            raise StudyBudgetError("explicit historical evidence is required")
        for reference in historical_evidence:
            _text(reference, "historical evidence")
        if set(pilot_formal_steps) != set(PILOT_ALLOCATION):
            raise StudyBudgetError("each pilot family needs its explicit nominal formal step cap")
        for value in pilot_formal_steps.values():
            _integer(value, "nominal formal steps", minimum=1)
        path = Path(root).resolve()
        path.mkdir(mode=0o700, exist_ok=False)
        with _AttemptLease.acquire(path):
            cls._append(
                path,
                0,
                "IMPORT_HISTORY",
                {
                    "historical_usage": historical_usage,
                    "historical_evidence": historical_evidence,
                    "run_seeds": run_seeds,
                    "pilot_formal_steps": pilot_formal_steps,
                    "matrix_status_at_import": matrix.get("status"),
                },
            )
            _fsync_directory(path.parent)
        return cls(path)

    @staticmethod
    def _append(root, sequence, kind, fields):
        raw = (
            json.dumps(
                {
                    "schema": "phaseset-v2-study-budget-v1",
                    "sequence": sequence,
                    "observed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "kind": kind,
                    **fields,
                    "scientific_or_launch_authority": False,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
            + b"\n"
        )
        _write_once(root / f"event-{sequence:08d}.json", raw)
        _fsync_directory(root)

    def _state(self):
        events = []
        for index, path in enumerate(sorted(self.root.glob("event-*.json"))):
            if path.name != f"event-{index:08d}.json":
                raise StudyBudgetError("budget journal sequence has a gap; preserve and inspect")
            row = json.loads(path.read_bytes())
            if row.get("sequence") != index or row.get("schema") != "phaseset-v2-study-budget-v1":
                raise StudyBudgetError("budget journal metadata is inconsistent")
            events.append(row)
        if not events or events[0]["kind"] != "IMPORT_HISTORY":
            raise StudyBudgetError("budget historical import is missing")
        history = events[0]
        usage = dict(history["historical_usage"])
        usage["pilot_families"] = dict(usage["pilot_families"])
        attempts, settled, step_limits = {}, {}, None
        for row in events[1:]:
            kind = row["kind"]
            if kind == "RESERVE":
                if row["attempt_id"] in attempts:
                    raise StudyBudgetError("duplicate attempt in budget journal")
                attempts[row["attempt_id"]] = row
                if row["purpose"] != "profile":
                    usage[row["purpose"] + "_reservations"] += 1
                if row["purpose"] == "pilot":
                    usage["pilot_families"][row["family"]] += 1
                if row["purpose"] == "formal" and row["predecessor_attempt_id"] is not None:
                    usage["formal_restarts"] += 1
            elif kind == "SETTLE":
                if row["attempt_id"] not in attempts or row["attempt_id"] in settled:
                    raise StudyBudgetError("settlement is missing its unique reservation")
                settled[row["attempt_id"]] = row
                usage["gpu_seconds"] += row["actual_gpu_seconds"]
            elif kind == "REWORK":
                usage["architecture_reworks"] += 1
            elif kind == "FORMAL_STEP_LIMITS":
                if step_limits is not None:
                    raise StudyBudgetError("formal step limits were rewritten")
                step_limits = row["run_max_steps"]
            else:
                raise StudyBudgetError("unknown budget journal event; do not reset it")
        pending = {key: value for key, value in attempts.items() if key not in settled}
        reserved = sum(row["reserved_gpu_seconds"] for row in pending.values())
        return {
            "events": events,
            "history": history,
            "usage": usage,
            "attempts": attempts,
            "settled": settled,
            "pending": pending,
            "reserved_gpu_seconds": reserved,
            "step_limits": step_limits,
        }

    def summary(self):
        with _AttemptLease.acquire(self.root):
            state = self._state()
            return {
                "usage": state["usage"],
                "reserved_gpu_seconds": state["reserved_gpu_seconds"],
                "available_gpu_seconds": max(
                    0,
                    LIMITS["gpu_seconds"]
                    - state["usage"]["gpu_seconds"]
                    - state["reserved_gpu_seconds"],
                ),
                "unsettled_attempt_ids": sorted(state["pending"]),
                "formal_step_limits_recorded": state["step_limits"] is not None,
                "optimizer_step_cap_exceeded_attempt_ids": sorted(
                    key
                    for key, row in state["settled"].items()
                    if row["optimizer_step_cap_exceeded"]
                ),
                "unknown_optimizer_cursor_attempt_ids": sorted(
                    key for key, row in state["settled"].items() if row["completed_steps"] is None
                ),
                "scientific_or_launch_authority": False,
            }

    def record_formal_step_limits(
        self,
        run_max_steps: dict[str, int],
        *,
        evidence: str,
        storage_projection: StudyStorageProjection,
        storage_directory: str | Path,
        temporary_directory: str | Path,
    ):
        """Record the already frozen external schedule, not freeze the science."""
        _text(evidence, "dated external freeze evidence")
        if (
            not isinstance(storage_projection, StudyStorageProjection)
            or storage_projection.cumulative_remaining_bytes is None
        ):
            raise StudyStorageHold("formal freeze needs a cumulative storage forecast")
        with _AttemptLease.acquire(self.root):
            state = self._state()
            if state["step_limits"] is not None:
                raise StudyBudgetError("formal step limits are immutable once recorded")
            if any(row["purpose"] == "pilot" for row in state["pending"].values()):
                raise StudyBudgetError("cannot freeze formal limits while a pilot is unsettled")
            if set(run_max_steps) != set(state["history"]["run_seeds"]):
                raise StudyBudgetError("formal step limits must cover all 87 registered stages")
            for value in run_max_steps.values():
                _integer(value, "formal max steps", minimum=1)
            storage_admission = check_study_storage(
                storage_directory, temporary_directory, storage_projection
            )
            self._append(
                self.root,
                len(state["events"]),
                "FORMAL_STEP_LIMITS",
                {
                    "run_max_steps": run_max_steps,
                    "evidence": evidence,
                    "storage_admission": storage_admission,
                },
            )

    def reserve(
        self,
        *,
        attempt_id: str,
        run_id: str,
        purpose: str,
        seed: int,
        gpu_count: int,
        wall_time_limit_seconds: int,
        planned_steps: int,
        evidence: str,
        family: str | None = None,
        predecessor_attempt_id: str | None = None,
        root_cause: str | None = None,
        future_budget_floor_gpu_seconds: float | None = None,
    ):
        """Reserve the full timeout+kill-grace envelope BEFORE child spawning."""
        for value, name in (
            (attempt_id, "attempt ID"),
            (run_id, "run ID"),
            (evidence, "execution evidence"),
        ):
            _text(value, name)
        if (
            purpose not in ("profile", "pilot", "formal")
            or type(seed) is not int
            or seed not in (1729, 2718, 31415)
        ):
            raise StudyBudgetError("purpose and one fixed seed must be explicit")
        _integer(gpu_count, "allocated GPU count")
        _integer(wall_time_limit_seconds, "wall envelope", minimum=1)
        _integer(planned_steps, "planned optimizer steps")
        if future_budget_floor_gpu_seconds is not None:
            if purpose != "pilot":
                raise StudyBudgetError("future study floor applies only to pilot admission")
            _seconds(future_budget_floor_gpu_seconds, "future study GPU floor")
        if purpose == "profile" and planned_steps != 0:
            raise StudyBudgetError("optimizer training cannot be hidden as an uncounted profile")
        if purpose != "profile" and (gpu_count < 1 or planned_steps < 1):
            raise StudyBudgetError("training reserves actual GPU allocation and positive step caps")
        cost = gpu_count * wall_time_limit_seconds
        with _AttemptLease.acquire(self.root):
            state = self._state()
            if attempt_id in state["attempts"]:
                raise StudyBudgetError("attempt IDs are immutable, including failed attempts")
            usage = state["usage"]
            if usage["gpu_seconds"] + state["reserved_gpu_seconds"] + cost > LIMITS["gpu_seconds"]:
                raise StudyBudgetError("study GPU budget exhausted or fully reserved")
            if (
                future_budget_floor_gpu_seconds is not None
                and usage["gpu_seconds"]
                + state["reserved_gpu_seconds"]
                + cost
                + future_budget_floor_gpu_seconds
                > LIMITS["gpu_seconds"]
            ):
                raise StudyBudgetError("pilot plus future study floor exceeds GPU budget")
            if purpose != "formal" and (
                predecessor_attempt_id is not None or root_cause is not None
            ):
                raise StudyBudgetError("only a formal rooted restart has a predecessor")
            if purpose != "pilot" and family is not None:
                raise StudyBudgetError("pilot family must not disguise another purpose")
            same_run = [row for row in state["attempts"].values() if row["run_id"] == run_id]
            if purpose != "formal" and run_id in state["history"]["run_seeds"]:
                raise StudyBudgetError(
                    "profile/pilot IDs must not occupy registered formal run IDs"
                )
            if purpose == "pilot":
                if (
                    state["step_limits"] is not None
                    or usage["formal_reservations"]
                    or state["pending"]
                    or any(row["completed_steps"] is None for row in state["settled"].values())
                ):
                    raise StudyBudgetError(
                        "pilot requires a settled known-cursor pre-formal journal"
                    )
                if same_run or family not in PILOT_ALLOCATION:
                    raise StudyBudgetError(
                        "each pilot stage is unique and uses a registered family"
                    )
                if (
                    usage["pilot_reservations"] >= 12
                    or usage["pilot_families"][family] >= PILOT_ALLOCATION[family]
                ):
                    raise StudyBudgetError("fixed pilot allocation exhausted")
                if 5 * planned_steps > state["history"]["pilot_formal_steps"][family]:
                    raise StudyBudgetError(
                        "pilot exceeds 20% of its registered nominal formal steps"
                    )
            if purpose == "formal":
                if state["step_limits"] is None:
                    raise StudyBudgetError(
                        "dated external formal step freeze has not been recorded"
                    )
                if state["history"]["run_seeds"].get(run_id) != seed or planned_steps != state[
                    "step_limits"
                ].get(run_id):
                    raise StudyBudgetError(
                        "formal run/seed/step cap differs from the registered schedule"
                    )
                if usage["formal_reservations"] >= 90:
                    raise StudyBudgetError("formal start reservation cap exhausted")
                if usage["formal_restarts"] >= 3:
                    raise StudyBudgetError("global three rooted formal restart allowance exhausted")
                if same_run:
                    previous = same_run[-1]
                    terminal = state["settled"].get(previous["attempt_id"])
                    if (
                        previous["purpose"] != "formal"
                        or predecessor_attempt_id != previous["attempt_id"]
                        or terminal is None
                        or terminal["outcome"] not in ("FAILED", "INTERRUPTED", "NOT_STARTED")
                    ):
                        raise StudyBudgetError(
                            "restart needs the latest failed/interrupted, settled predecessor"
                        )
                    _text(root_cause, "root cause from original failure evidence")
                elif predecessor_attempt_id is not None or root_cause is not None:
                    raise StudyBudgetError("a first formal start has no invented predecessor")
            self._append(
                self.root,
                len(state["events"]),
                "RESERVE",
                {
                    "attempt_id": attempt_id,
                    "run_id": run_id,
                    "purpose": purpose,
                    "seed": seed,
                    "gpu_count": gpu_count,
                    "wall_time_limit_seconds": wall_time_limit_seconds,
                    "reserved_gpu_seconds": cost,
                    "planned_steps": planned_steps,
                    "family": family,
                    "predecessor_attempt_id": predecessor_attempt_id,
                    "root_cause": root_cause,
                    "evidence": evidence,
                },
            )

    def settle(
        self,
        attempt_id: str,
        *,
        actual_wall_seconds: float,
        completed_steps: int | None,
        outcome: str,
        evidence: str,
    ):
        """Observed child termination; overspend is recorded, never suppressed."""
        _seconds(actual_wall_seconds, "actual allocated wall time")
        if completed_steps is not None:
            _integer(completed_steps, "completed optimizer cursor")
        _text(evidence, "actual process/terminal evidence")
        if outcome not in ("COMPLETED", "FAILED", "INTERRUPTED", "NOT_STARTED"):
            raise StudyBudgetError("explicit terminal outcome is required")
        if completed_steps is None and outcome in ("COMPLETED", "NOT_STARTED"):
            raise StudyBudgetError("success or unstarted needs a known optimizer cursor")
        with _AttemptLease.acquire(self.root):
            state = self._state()
            reservation = state["pending"].get(attempt_id)
            if reservation is None:
                raise StudyBudgetError(
                    "settlement needs one unresolved reservation, not a duplicate"
                )
            if outcome == "NOT_STARTED" and (actual_wall_seconds != 0 or completed_steps != 0):
                raise StudyBudgetError(
                    "unstarted requires verified zero allocation and zero updates"
                )
            self._append(
                self.root,
                len(state["events"]),
                "SETTLE",
                {
                    "attempt_id": attempt_id,
                    "outcome": outcome,
                    "optimizer_step_cap_exceeded": completed_steps is not None
                    and completed_steps > reservation["planned_steps"],
                    "completed_steps": completed_steps,
                    "actual_wall_seconds": actual_wall_seconds,
                    "actual_gpu_seconds": actual_wall_seconds * reservation["gpu_count"],
                    "evidence": evidence,
                },
            )

    def record_rework(self, *, reason: str, evidence: str):
        _text(reason, "structural rework reason")
        _text(evidence, "structural rework evidence")
        with _AttemptLease.acquire(self.root):
            state = self._state()
            if state["usage"]["architecture_reworks"] >= 2:
                raise StudyBudgetError("two structural rework cap exhausted")
            self._append(
                self.root, len(state["events"]), "REWORK", {"reason": reason, "evidence": evidence}
            )
