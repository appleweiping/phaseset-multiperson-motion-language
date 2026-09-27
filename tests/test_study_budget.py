"""Synthetic bookkeeping only; none of these reservations launches a model."""

import copy
import json
import os
from pathlib import Path
import stat
import threading

import pytest

from phaseset_core.host import HostConfigurationError
from phaseset_core.study_budget import PILOT_ALLOCATION, StudyBudget, StudyBudgetError


def matrix():
    return json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs" / "phaseset_v2_experiment_matrix.json"
        ).read_text()
    )


def history():
    return {
        "gpu_seconds": 34.0,
        "formal_reservations": 0,
        "pilot_reservations": 0,
        "formal_restarts": 0,
        "architecture_reworks": 0,
        "pilot_families": {name: 0 for name in PILOT_ALLOCATION},
    }


def create(tmp_path, **kwargs):
    return StudyBudget.create(
        tmp_path / "budget",
        **{
            "matrix": matrix(),
            "historical_usage": history(),
            "historical_evidence": ("analytic import fixture, not actual GPU accounting",),
            "pilot_formal_steps": {
                name: 40 if name == "PhaseSet-V2-head" else 60 for name in PILOT_ALLOCATION
            },
            **kwargs,
        },
    )


def reserve(budget, attempt="analytic-1", **kwargs):
    return budget.reserve(
        **{
            "attempt_id": attempt,
            "run_id": "analytic-profile-1",
            "purpose": "profile",
            "seed": 1729,
            "gpu_count": 2,
            "wall_time_limit_seconds": 40,
            "planned_steps": 0,
            "evidence": "analytic metadata only; no child/model launched",
            **kwargs,
        }
    )


def settle(budget, attempt="analytic-1", **kwargs):
    return budget.settle(
        attempt,
        **{
            "actual_wall_seconds": 10.5,
            "completed_steps": 0,
            "outcome": "FAILED",
            "evidence": "analytic terminal fixture only; no actual job result",
            **kwargs,
        },
    )


def formal_steps():
    return {row["id"]: 60 for row in matrix()["runs"]}


def freeze(budget):
    budget.record_formal_step_limits(
        formal_steps(), evidence="synthetic schedule; NOT scientific freeze"
    )


def formal(budget, attempt="analytic-formal-1", run="V2-001", **kwargs):
    seed = next(row["seed"] for row in matrix()["runs"] if row["id"] == run)
    return reserve(
        budget,
        attempt,
        **{
            "run_id": run,
            "purpose": "formal",
            "seed": seed,
            "gpu_count": 1,
            "planned_steps": 60,
            **kwargs,
        },
    )


def test_history_is_explicit_immutable_and_not_reset_on_reopen(tmp_path):
    initial = history()
    budget = create(tmp_path, historical_usage=initial)
    initial["gpu_seconds"] = 0
    assert budget.summary()["usage"]["gpu_seconds"] == 34
    assert StudyBudget(budget.root).summary() == budget.summary()
    assert budget.summary()["scientific_or_launch_authority"] is False
    with pytest.raises(FileExistsError):
        create(tmp_path)
    with pytest.raises(StudyBudgetError, match="budget absent"):
        StudyBudget(tmp_path / "new-root")


@pytest.mark.parametrize("fail_parent_sync", [False, True])
def test_bootstrap_syncs_event_then_journal_then_parent_and_preserves_failure(
    tmp_path, monkeypatch, fail_parent_sync
):
    original = os.fsync
    calls = []

    def observed_sync(descriptor):
        observed = os.fstat(descriptor)
        if stat.S_ISDIR(observed.st_mode):
            parent = tmp_path.stat()
            label = (
                "parent"
                if (observed.st_dev, observed.st_ino) == (parent.st_dev, parent.st_ino)
                else "journal"
            )
        else:
            label = "event"
        calls.append(label)
        if label == "parent" and fail_parent_sync:
            raise OSError("parent directory sync failure fixture")
        return original(descriptor)

    monkeypatch.setattr(os, "fsync", observed_sync)
    if os.name == "posix" and fail_parent_sync:
        with pytest.raises(OSError, match="parent directory sync"):
            create(tmp_path)
        assert (tmp_path / "budget" / "event-00000000.json").is_file()
        with pytest.raises(FileExistsError):
            create(tmp_path)
        assert StudyBudget(tmp_path / "budget").summary()["usage"]["gpu_seconds"] == 34
    else:
        assert create(tmp_path).summary()["usage"]["gpu_seconds"] == 34
    assert calls == (["event", "journal", "parent"] if os.name == "posix" else ["event"])


@pytest.mark.parametrize(
    "missing", ["gpu_seconds", "formal_reservations", "pilot_families", "architecture_reworks"]
)
def test_missing_historical_counter_is_not_assumed_zero(tmp_path, missing):
    incomplete = history()
    del incomplete[missing]
    with pytest.raises(StudyBudgetError, match="explicitly"):
        create(tmp_path, historical_usage=incomplete)
    assert not (tmp_path / "budget").exists()


def test_history_evidence_and_fixed_caps_cannot_be_omitted_or_raised(tmp_path):
    with pytest.raises(StudyBudgetError, match="evidence"):
        create(tmp_path, historical_evidence=())
    changed = matrix()
    changed["budget"]["gpu_hours"] = 301
    with pytest.raises(StudyBudgetError, match="cannot be raised"):
        create(tmp_path, matrix=changed)
    bad = history()
    bad["pilot_reservations"] = 1
    with pytest.raises(StudyBudgetError, match="census"):
        create(tmp_path, historical_usage=bad)
    trained = history()
    trained["formal_reservations"] = 1
    with pytest.raises(StudyBudgetError, match="original detailed journal"):
        create(tmp_path, historical_usage=trained)


def test_multigpu_failed_wall_time_and_pending_envelope_are_persistent(tmp_path):
    budget = create(tmp_path)
    reserve(budget)
    row = StudyBudget(budget.root).summary()
    assert row["usage"]["gpu_seconds"] == 34 and row["reserved_gpu_seconds"] == 80
    assert row["available_gpu_seconds"] == 300 * 3600 - 114
    assert row["unsettled_attempt_ids"] == ["analytic-1"]
    settle(budget)
    final = StudyBudget(budget.root).summary()
    assert final["usage"]["gpu_seconds"] == 55
    assert final["reserved_gpu_seconds"] == 0 and final["unsettled_attempt_ids"] == []
    assert final["usage"]["formal_reservations"] == final["usage"]["pilot_reservations"] == 0
    with pytest.raises(StudyBudgetError, match="duplicate"):
        settle(budget)
    with pytest.raises(StudyBudgetError, match="immutable"):
        reserve(budget)


def test_missing_process_does_not_release_reservation_and_overspend_is_recorded(tmp_path):
    budget = create(tmp_path)
    reserve(budget, gpu_count=1, wall_time_limit_seconds=300 * 3600 - 34)
    with pytest.raises(StudyBudgetError, match="budget exhausted"):
        reserve(StudyBudget(budget.root), "analytic-2", gpu_count=1, wall_time_limit_seconds=1)
    # Actual allocated wall-time overshoot is evidence, not rejected input.
    settle(budget, actual_wall_seconds=300 * 3600, outcome="INTERRUPTED")
    assert budget.summary()["usage"]["gpu_seconds"] == 300 * 3600 + 34
    assert budget.summary()["available_gpu_seconds"] == 0
    with pytest.raises(StudyBudgetError, match="budget exhausted"):
        reserve(budget, "analytic-3", gpu_count=1, wall_time_limit_seconds=1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", 42),
        ("gpu_count", True),
        ("gpu_count", -1),
        ("wall_time_limit_seconds", 0),
        ("planned_steps", 1),
        ("purpose", "extra_training"),
        ("run_id", "V2-001"),
    ],
)
def test_profiles_cannot_hide_optimizer_steps_or_formal_ids(tmp_path, field, value):
    budget = create(tmp_path)
    with pytest.raises(StudyBudgetError):
        reserve(budget, **{field: value})
    assert budget.summary()["unsettled_attempt_ids"] == []


def test_all_twelve_pilot_slots_and_family_allocation_include_spawn_failures(tmp_path):
    budget = create(tmp_path)
    for family, allowance in PILOT_ALLOCATION.items():
        for index in range(allowance):
            attempt = f"analytic-{family}-{index}"
            reserve(
                budget,
                attempt,
                run_id=attempt,
                purpose="pilot",
                family=family,
                gpu_count=1,
                planned_steps=8 if family == "PhaseSet-V2-head" else 12,
            )
            settle(budget, attempt, actual_wall_seconds=0, outcome="NOT_STARTED")
    counts = budget.summary()["usage"]
    assert counts["pilot_reservations"] == 12 and counts["pilot_families"] == PILOT_ALLOCATION
    assert counts["gpu_seconds"] == 34  # Simulated spawning failures had no allocation.
    with pytest.raises(StudyBudgetError, match="allocation exhausted"):
        reserve(
            budget,
            "analytic-thirteenth",
            run_id="analytic-thirteenth",
            purpose="pilot",
            family="TMR-Set",
            gpu_count=1,
            planned_steps=12,
        )


def test_pilot_step_fraction_is_exact_and_unsupported_family_is_rejected(tmp_path):
    budget = create(tmp_path)
    with pytest.raises(StudyBudgetError, match="20%"):
        reserve(budget, purpose="pilot", family="TMR-Set", planned_steps=13)
    with pytest.raises(StudyBudgetError, match="registered family"):
        reserve(budget, purpose="pilot", family="extra-family", planned_steps=1)
    assert budget.summary()["usage"]["pilot_reservations"] == 0


def test_formal_needs_all_87_step_limits_once_and_exact_seed_schedule(tmp_path):
    budget = create(tmp_path)
    with pytest.raises(StudyBudgetError, match="freeze"):
        formal(budget)
    incomplete = formal_steps()
    incomplete.pop("V2-087")
    with pytest.raises(StudyBudgetError, match="all 87"):
        budget.record_formal_step_limits(incomplete, evidence="analytic incomplete schedule")
    freeze(budget)
    with pytest.raises(StudyBudgetError, match="immutable"):
        freeze(budget)
    for change in ({"seed": 2718}, {"planned_steps": 61}):
        with pytest.raises(StudyBudgetError, match="differs"):
            formal(budget, **change)
    formal(budget)
    settle(budget, "analytic-formal-1", completed_steps=60, outcome="COMPLETED")
    with pytest.raises(StudyBudgetError, match="predecessor"):
        formal(
            budget,
            "analytic-formal-2",
            predecessor_attempt_id="analytic-formal-1",
            root_cause="cannot rerun completed seed",
        )


def test_rooted_restarts_need_latest_settled_failure_and_stop_at_global_three(tmp_path):
    budget = create(tmp_path)
    freeze(budget)
    formal(budget)
    with pytest.raises(StudyBudgetError, match="settled predecessor"):
        formal(
            budget,
            "analytic-early",
            predecessor_attempt_id="analytic-formal-1",
            root_cause="not actually terminal",
        )
    settle(budget, "analytic-formal-1")
    with pytest.raises(StudyBudgetError, match="root cause"):
        formal(budget, "analytic-unrooted", predecessor_attempt_id="analytic-formal-1")
    previous = "analytic-formal-1"
    for index in range(3):
        attempt = f"analytic-restart-{index}"
        formal(
            budget,
            attempt,
            predecessor_attempt_id=previous,
            root_cause="synthetic infrastructure failure fixture",
        )
        settle(budget, attempt)
        previous = attempt
        if index == 0:
            with pytest.raises(StudyBudgetError, match="latest"):
                formal(
                    budget,
                    "analytic-fork-old",
                    predecessor_attempt_id="analytic-formal-1",
                    root_cause="old branch fork forbidden",
                )
    assert budget.summary()["usage"]["formal_reservations"] == 4
    assert budget.summary()["usage"]["formal_restarts"] == 3
    with pytest.raises(StudyBudgetError, match="three rooted"):
        formal(
            budget,
            "analytic-fourth-restart",
            predecessor_attempt_id=previous,
            root_cause="cap must hold",
        )
    with pytest.raises(StudyBudgetError, match="three rooted"):
        formal(budget, "analytic-new-after-cap", "V2-004")


def test_complete_87_matrix_plus_three_restarts_cannot_launch_91st(tmp_path):
    budget = create(tmp_path)
    freeze(budget)
    for row in matrix()["runs"]:
        attempt = "analytic-" + row["id"]
        formal(budget, attempt, row["id"])
        settle(budget, attempt, actual_wall_seconds=1)
    for index in range(1, 4):
        run_id = f"V2-{index:03d}"
        attempt = "analytic-restart-" + run_id
        formal(
            budget,
            attempt,
            run_id,
            predecessor_attempt_id="analytic-" + run_id,
            root_cause="synthetic fixture only",
        )
        settle(budget, attempt, actual_wall_seconds=1)
    assert budget.summary()["usage"]["formal_reservations"] == 90
    with pytest.raises(StudyBudgetError, match="start reservation cap"):
        formal(
            budget,
            "analytic-91",
            "V2-004",
            predecessor_attempt_id="analytic-V2-004",
            root_cause="must not add run",
        )


def test_structural_reworks_do_not_reset_pilots_or_allow_a_third(tmp_path):
    budget = create(tmp_path)
    for index in range(2):
        budget.record_rework(reason=f"analytic rework {index}", evidence="synthetic only")
    with pytest.raises(StudyBudgetError, match="two structural"):
        StudyBudget(budget.root).record_rework(reason="third", evidence="must reject")
    assert budget.summary()["usage"]["architecture_reworks"] == 2
    assert budget.summary()["usage"]["pilot_reservations"] == 0


def test_step_cap_violation_cannot_hide_actual_gpu_charge(tmp_path):
    budget = create(tmp_path)
    reserve(budget)
    settle(budget, actual_wall_seconds=41, completed_steps=1)
    summary = budget.summary()
    assert summary["usage"]["gpu_seconds"] == 116
    assert summary["optimizer_step_cap_exceeded_attempt_ids"] == ["analytic-1"]


def test_missing_optimizer_cursor_does_not_invent_zero_or_hide_failed_cost(tmp_path):
    budget = create(tmp_path)
    reserve(budget)
    settle(budget, completed_steps=None)
    summary = budget.summary()
    assert summary["usage"]["gpu_seconds"] == 55
    assert summary["unknown_optimizer_cursor_attempt_ids"] == ["analytic-1"]
    assert summary["unsettled_attempt_ids"] == []
    row = json.loads((budget.root / "event-00000002.json").read_text())
    assert row["completed_steps"] is None


def test_unknown_cursor_cannot_claim_completed_or_unstarted(tmp_path):
    budget = create(tmp_path)
    reserve(budget)
    for outcome in ("COMPLETED", "NOT_STARTED"):
        with pytest.raises(StudyBudgetError, match="known optimizer cursor"):
            settle(budget, completed_steps=None, outcome=outcome, actual_wall_seconds=0)
    assert budget.summary()["unsettled_attempt_ids"] == ["analytic-1"]


def test_concurrent_reservations_cannot_double_spend_last_envelope(tmp_path):
    budget = create(tmp_path)
    barrier = threading.Barrier(3)
    outcomes = []

    def worker(index):
        barrier.wait()
        try:
            reserve(
                StudyBudget(budget.root),
                f"analytic-{index}",
                gpu_count=1,
                wall_time_limit_seconds=300 * 3600 - 34,
            )
            outcomes.append("reserved")
        except (StudyBudgetError, HostConfigurationError):
            outcomes.append("held")

    workers = [threading.Thread(target=worker, args=(index,)) for index in range(2)]
    for item in workers:
        item.start()
    barrier.wait()
    for item in workers:
        item.join(timeout=5)
        assert not item.is_alive()
    assert sorted(outcomes) == ["held", "reserved"]
    assert len(budget.summary()["unsettled_attempt_ids"]) == 1
    assert budget.summary()["available_gpu_seconds"] == 0


def test_partial_disk_write_is_preserved_and_not_replaced_by_fresh_history(tmp_path, monkeypatch):
    import phaseset_core.study_budget as module

    budget = create(tmp_path)
    snapshot = copy.deepcopy(budget.summary())

    def partial_write(path, raw):
        path.write_bytes(raw[:10])
        raise OSError("actual partial disk write fixture")

    monkeypatch.setattr(module, "_write_once", partial_write)
    with pytest.raises(OSError, match="partial disk"):
        reserve(budget)
    assert (budget.root / "event-00000001.json").read_bytes()
    with pytest.raises(json.JSONDecodeError):
        budget.summary()
    with pytest.raises(FileExistsError):
        create(tmp_path)
    assert snapshot["usage"]["gpu_seconds"] == 34
