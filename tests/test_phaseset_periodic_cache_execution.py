"""Focused software contracts for the private periodic-cache builder.

These fixtures are analytic and carry no empirical result.  Numerical execution
is reserved for the registered server test lane.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from phaseset_core import (
    capture_validation,
    execution,
    experiments,
    periodic_cache_execution as cache_execution,
    training,
)
from phaseset_core.periodic_cache_execution import (
    CACHEABLE_SYSTEM_STREAMS,
    PeriodicCacheExecutionBounds,
    PeriodicCacheExecutionError,
    build_periodic_descriptor_cache_for_seed,
    consume_periodic_descriptor_cache_build,
    load_qualified_energy_floor_input,
)
from test_periodic_capture_training_cache import _write_capture_source
from test_periodic_training_cache import _write_sources


_SEED = 1729
_FLOORS = np.ascontiguousarray(
    np.asarray([(index + 1) * 1.0e-12 for index in range(6)], dtype=np.float64)
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest(label: str) -> str:
    return _sha(label.encode("ascii"))


def _qualification(
    checkpoint_sha256s: dict[int, str],
) -> tuple[experiments.BaseQualification, str]:
    scores = {"B0": (60, 62, 58), "B1": (70, 71, 69), "B2": (65, 66, 64)}
    rows: list[experiments.BaseScore] = []
    for index, run_id in enumerate(experiments.base_run_ids(), start=1):
        _, seed, system_id = experiments.parse_run_id(run_id)
        checkpoint = (
            checkpoint_sha256s[seed] if system_id == "B1" else _digest(f"checkpoint/{run_id}")
        )
        rows.append(
            experiments.BaseScore(
                run_id=run_id,
                bidirectional_r1_numerator=scores[system_id][experiments.SEEDS.index(seed)],
                bidirectional_r1_denominator=100,
                parameter_count={"B0": 10_000, "B1": 11_000, "B2": 12_000}[system_id],
                frozen_runtime_latency_ns={
                    "B0": 1_000,
                    "B1": 1_100,
                    "B2": 1_200,
                }[system_id],
                terminal_sha256=_digest(f"terminal/{index}"),
                selected_checkpoint_sha256=checkpoint,
                split="validation",
                validation_manifest_sha256=_digest("capture-validation-manifest"),
                query_census_sha256=_digest("capture-validation-census"),
                evaluator_sha256=_digest("capture-evaluator"),
                score_artifact_sha256=_digest(f"score/{index}"),
            )
        )
    qualification = experiments.qualify_base(tuple(rows))
    raw = experiments.canonical_base_qualification_bytes(qualification)
    return qualification, _sha(raw)


def _floor_payload(*, extra_key: bool = False) -> bytes:
    stream = io.BytesIO()
    if extra_key:
        np.savez(stream, energy_floors=_FLOORS, ignored=np.zeros(1, dtype=np.float64))
    else:
        np.savez(stream, energy_floors=_FLOORS)
    return stream.getvalue()


def _write_floor_root(
    root: Path,
    *,
    qualification: experiments.BaseQualification,
    qualification_sha256: str,
    payload: bytes | None = None,
) -> Path:
    root.mkdir()
    raw_payload = _floor_payload() if payload is None else payload
    record = execution.PeriodicCacheRecord(
        seed=_SEED,
        qualified_base_system_id=qualification.winner_system_id,
        source_run_id=(
            f"phaseset-run-v1/BASE_QUALIFICATION/{_SEED}/{qualification.winner_system_id}"
        ),
        created_at_utc="2026-09-08T00:00:00Z",
        base_terminal_sha256=qualification.winner_terminal_sha256s[experiments.SEEDS.index(_SEED)],
        qualification_sha256=qualification_sha256,
        cache_content_sha256=_sha(raw_payload),
    )
    (root / "record.json").write_bytes(execution.canonical_periodic_cache_bytes(record))
    (root / "cache-content.npz").write_bytes(raw_payload)
    return root


def _bundle(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / "winner-checkpoint.pt"
    checkpoint.write_bytes(b"analytic selected checkpoint fixture\n")
    qualification, qualification_sha256 = _qualification(
        {
            1729: _sha(checkpoint.read_bytes()),
            2718: _digest("checkpoint/winner/2718"),
            31415: _digest("checkpoint/winner/31415"),
        }
    )
    floor_root = _write_floor_root(
        tmp_path / "qualified-floors",
        qualification=qualification,
        qualification_sha256=qualification_sha256,
    )
    floors = load_qualified_energy_floor_input(
        floor_root,
        seed=_SEED,
        qualification=qualification,
        qualification_sha256=qualification_sha256,
    )
    train_source, auxiliary_val = _write_sources(tmp_path / "prepared")
    capture_source = _write_capture_source(tmp_path / "capture")
    return (
        checkpoint,
        qualification,
        qualification_sha256,
        floors,
        train_source,
        auxiliary_val,
        capture_source,
        training.TrainingConfig(stage="residual", seed=_SEED),
    )


def _arguments(tmp_path: Path) -> dict[str, object]:
    (
        checkpoint,
        qualification,
        qualification_sha256,
        floors,
        train_source,
        _auxiliary_val,
        capture_source,
        config,
    ) = _bundle(tmp_path)
    return {
        "seed": _SEED,
        "config": config,
        "qualification": qualification,
        "qualification_sha256": qualification_sha256,
        "winner_checkpoint": checkpoint,
        "floors": floors,
        "train_source": train_source,
        "capture_source": capture_source,
        "source_tree_sha256": _digest("registered-source-tree"),
        "environment_sha256": _digest("registered-environment"),
    }


def test_build_and_consume_complete_twenty_epoch_and_capture_plan(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    root = tmp_path / "periodic-descriptor-build"
    built = build_periodic_descriptor_cache_for_seed(
        output_root=root,
        **arguments,
    )

    assert built.root == root.resolve()
    assert built.manifest_sha256 == _sha(built.manifest_json_bytes)
    assert [plan.system_id for plan in built.plans] == [
        system_id for system_id, _stream in CACHEABLE_SYSTEM_STREAMS
    ]
    assert [plan.stream_kind for plan in built.plans] == [
        stream for _system_id, stream in CACHEABLE_SYSTEM_STREAMS
    ]
    assert {entry.name for entry in root.iterdir()} == {
        "build.json",
        "train",
        "capture-validation",
        "plans",
    }
    value = json.loads(built.manifest_json_bytes)
    assert value["authority"] == 0
    assert value["production"] is False
    assert value["result_claimed"] is False
    assert value["external_authentication_asserted"] is False
    assert value["train_cache"]["shard_count"] == training.RESIDUAL_EPOCHS
    capture_source = arguments["capture_source"]
    assert type(capture_source) is capture_validation.CaptureValidationSource
    expected_windows = sum(len(capture.windows) for capture in capture_source.captures)
    assert value["capture_validation_cache"]["shard_count"] == expected_windows
    assert value["capture_source"]["window_count"] == expected_windows
    assert str(tmp_path).encode() not in built.manifest_json_bytes

    consumed = consume_periodic_descriptor_cache_build(
        root,
        expected_manifest_sha256=built.manifest_sha256,
        **arguments,
    )
    assert consumed.manifest_json_bytes == built.manifest_json_bytes
    assert tuple(plan.sha256 for plan in consumed.plans) == tuple(
        plan.sha256 for plan in built.plans
    )
    assert consumed.train_cache.cache_key_census == built.train_cache.cache_key_census
    assert (
        consumed.capture_validation_cache.cache_key_census
        == built.capture_validation_cache.cache_key_census
    )


def test_floor_loader_requires_exact_two_file_tree_and_exact_npz_key(
    tmp_path: Path,
) -> None:
    checkpoint = b"floor loader checkpoint"
    qualification, qualification_sha256 = _qualification(
        {
            1729: _sha(checkpoint),
            2718: _digest("floor/2718"),
            31415: _digest("floor/31415"),
        }
    )
    extra_payload = _floor_payload(extra_key=True)
    root = _write_floor_root(
        tmp_path / "floors-extra-key",
        qualification=qualification,
        qualification_sha256=qualification_sha256,
        payload=extra_payload,
    )
    with pytest.raises(PeriodicCacheExecutionError, match="member census"):
        load_qualified_energy_floor_input(
            root,
            seed=_SEED,
            qualification=qualification,
            qualification_sha256=qualification_sha256,
        )

    root = _write_floor_root(
        tmp_path / "floors-extra-file",
        qualification=qualification,
        qualification_sha256=qualification_sha256,
    )
    (root / "unregistered.txt").write_text("unregistered", encoding="ascii")
    with pytest.raises(PeriodicCacheExecutionError, match="member census"):
        load_qualified_energy_floor_input(
            root,
            seed=_SEED,
            qualification=qualification,
            qualification_sha256=qualification_sha256,
        )


def test_preflight_rejects_auxiliary_val_checkpoint_drift_and_reused_root(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path)
    auxiliary_val = _bundle(tmp_path / "other")[5]
    invalid = dict(arguments)
    invalid["train_source"] = auxiliary_val
    with pytest.raises(PeriodicCacheExecutionError, match="train-only"):
        build_periodic_descriptor_cache_for_seed(
            output_root=tmp_path / "must-not-exist",
            **invalid,
        )
    assert not (tmp_path / "must-not-exist").exists()

    checkpoint = arguments["winner_checkpoint"]
    assert isinstance(checkpoint, Path)
    checkpoint.write_bytes(b"changed after qualification\n")
    with pytest.raises(PeriodicCacheExecutionError, match="checkpoint bytes differ"):
        build_periodic_descriptor_cache_for_seed(
            output_root=tmp_path / "must-not-exist-either",
            **arguments,
        )
    assert not (tmp_path / "must-not-exist-either").exists()

    checkpoint.write_bytes(b"analytic selected checkpoint fixture\n")
    reused = tmp_path / "already-present"
    reused.mkdir()
    with pytest.raises(PeriodicCacheExecutionError, match="output root must be new"):
        build_periodic_descriptor_cache_for_seed(output_root=reused, **arguments)
    assert list(reused.iterdir()) == []


def test_consumption_rejects_plan_mutation_and_extra_tree_member(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    built = build_periodic_descriptor_cache_for_seed(
        output_root=tmp_path / "built",
        **arguments,
    )
    plan_mutant = tmp_path / "plan-mutant"
    shutil.copytree(built.root, plan_mutant)
    plan_path = plan_mutant / "plans" / "02.json"
    plan_path.write_bytes(plan_path.read_bytes() + b" ")
    with pytest.raises(PeriodicCacheExecutionError, match="differs from complete admission"):
        consume_periodic_descriptor_cache_build(
            plan_mutant,
            expected_manifest_sha256=built.manifest_sha256,
            **arguments,
        )

    extra_member = tmp_path / "extra-member"
    shutil.copytree(built.root, extra_member)
    (extra_member / "unexpected").write_bytes(b"unexpected")
    with pytest.raises(PeriodicCacheExecutionError, match="member census"):
        consume_periodic_descriptor_cache_build(
            extra_member,
            expected_manifest_sha256=built.manifest_sha256,
            **arguments,
        )

    bool_mutant = tmp_path / "bool-mutant"
    shutil.copytree(built.root, bool_mutant)
    manifest_path = bool_mutant / "build.json"
    value = json.loads(manifest_path.read_bytes())
    value["authority"] = False
    raw = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    manifest_path.write_bytes(raw)
    with pytest.raises(PeriodicCacheExecutionError, match="live consumed evidence"):
        consume_periodic_descriptor_cache_build(
            bool_mutant,
            expected_manifest_sha256=_sha(raw),
            **arguments,
        )


@pytest.mark.parametrize(
    ("max_shards", "max_bytes"),
    [(5, 1 << 40), (524_288, 1)],
)
def test_combined_bounds_stop_writer_before_exceeding_disk_census(
    tmp_path: Path,
    max_shards: int,
    max_bytes: int,
) -> None:
    arguments = _arguments(tmp_path / "inputs")
    root = tmp_path / "bounded-build"
    bounds = PeriodicCacheExecutionBounds(
        max_combined_shards=max_shards,
        max_combined_shard_bytes=max_bytes,
    )
    with pytest.raises(PeriodicCacheExecutionError, match="construction failed"):
        build_periodic_descriptor_cache_for_seed(
            output_root=root,
            bounds=bounds,
            **arguments,
        )
    shards = tuple(root.rglob("*.pdc2"))
    assert len(shards) <= max_shards
    assert sum(path.stat().st_size for path in shards) <= max_bytes
    assert not (root / "build.json").exists()


def test_cross_interface_identity_excludes_only_nonportable_metadata() -> None:
    common = {
        "st_mode": 0o100600,
        "st_dev": 7,
        "st_ino": 11,
        "st_size": 13,
        "st_mtime_ns": 17,
        "st_nlink": 1,
    }
    path_stat = SimpleNamespace(**common, st_ctime_ns=19)
    descriptor_stat = SimpleNamespace(**common, st_ctime_ns=23)
    assert cache_execution._cross_interface_identity(  # noqa: SLF001
        path_stat
    ) == cache_execution._cross_interface_identity(descriptor_stat)  # noqa: SLF001
    assert cache_execution._stable_stat_identity(  # noqa: SLF001
        path_stat
    ) != cache_execution._stable_stat_identity(descriptor_stat)  # noqa: SLF001


def test_floor_bytes_and_result_objects_are_defensive_snapshots(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    floors = arguments["floors"]
    original = floors.energy_floors.copy()
    with pytest.raises(ValueError):
        floors.energy_floors[0] = 1.0
    assert np.array_equal(floors.energy_floors, original)

    built = build_periodic_descriptor_cache_for_seed(
        output_root=tmp_path / "built",
        **arguments,
    )
    with pytest.raises(PeriodicCacheExecutionError, match="record bytes changed"):
        replace(floors, record_sha256="f" * 64)
    with pytest.raises(PeriodicCacheExecutionError, match="manifest bytes changed"):
        replace(built, manifest_sha256="f" * 64)
