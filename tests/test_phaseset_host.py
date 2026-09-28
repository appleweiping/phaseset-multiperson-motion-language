from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pytest
import torch
from phaseset_core import cli, host
from phaseset_core.contracts import group_commitment
from phaseset_core.execution import AttemptRecord, AttemptStore, parse_terminal_bytes
from phaseset_core.training import CheckpointArtifact

PUBLIC_ROOT = Path(cli.__file__).resolve().parents[2]
PUBLIC_CONFIG = PUBLIC_ROOT / "configs" / "phaseset" / "training.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _commitment(label: str) -> bytes:
    return hashlib.sha256(label.encode("ascii")).digest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )


def _write_batch(path: Path, label: str) -> None:
    actor_ids = (_commitment(f"{label}-actor-0"), _commitment(f"{label}-actor-1"))
    family = _commitment(f"{label}-family")
    text_id = _commitment(f"{label}-text")
    np.savez(
        path,
        skeletons=np.zeros((1, 2, 200, 22, 3), dtype=np.float32),
        actor_mask=np.ones((1, 2), dtype=np.bool_),
        frame_mask=np.ones((1, 200), dtype=np.bool_),
        track_mask=np.ones((1, 2, 200, 22), dtype=np.bool_),
        actor_commitments=np.asarray(
            [[list(actor_ids[0]), list(actor_ids[1])]], dtype=np.uint8
        ),
        group_commitments=np.asarray(
            [list(group_commitment(actor_ids))], dtype=np.uint8
        ),
        text_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        motion_positive_ids=np.asarray([list(family)], dtype=np.uint8),
        text_positive_ids=np.asarray([list(family)], dtype=np.uint8),
        text_commitments=np.asarray([list(text_id)], dtype=np.uint8),
    )


def _write_split(root: Path, split: str, labels: tuple[str, ...]) -> Path:
    batches = []
    for label in labels:
        path = root / f"{split}-{label}.npz"
        _write_batch(path, f"{split}-{label}")
        batches.append({"path": path.name, "sha256": _sha(path)})
    manifest = root / f"{split}.json"
    _write_json(
        manifest,
        {"batches": batches, "schema": host.PREPARED_SPLIT_SCHEMA, "split": split},
    )
    return manifest


def _prepared_tree(root: Path) -> tuple[Path, Path, Path]:
    train = _write_split(root, "train", ("a", "b"))
    val = _write_split(root, "val", ("a",))
    index = root / "prepared-index.json"
    _write_json(
        index,
        {
            "schema": host.PREPARED_INDEX_SCHEMA,
            "train": {"path": train.name, "sha256": _sha(train)},
            "val": {"path": val.name, "sha256": _sha(val)},
        },
    )
    return train, val, index


def _host_config_with_missing_rights(root: Path) -> Path:
    _, _, index = _prepared_tree(root)
    artifacts: dict[str, str] = {}
    digests: dict[str, str] = {}
    for field in host.RECEIPT_DIGEST_FIELDS:
        path = (
            index
            if field == "prepared_data_manifest_sha256"
            else root / f"{field}.json"
        )
        if (
            field != "prepared_data_manifest_sha256"
            and field != "rights_assertion_sha256"
        ):
            path.write_text(f"{field}\n", encoding="ascii")
        artifacts[field] = path.name
        digests[field] = _sha(path) if path.exists() else "0" * 64
    receipt = root / "receipts.json"
    _write_json(
        receipt,
        {
            "admitted_at_utc": "2026-09-07T12:00:00Z",
            # This deliberately incomplete fixture must never mint authority.
            "authority": 0,
            **digests,
        },
    )
    config = root / "host.json"
    _write_json(
        config,
        {
            "prepared_index": index.name,
            "receipts": {"artifacts": artifacts, "record": receipt.name},
            "runtime": {
                "bf16_runtime_qualified": False,
                "checkpoint_every_updates": 1,
                "corrective_change_sha256": hashlib.sha256(b"none").hexdigest(),
                "device": "cpu",
                "edge_budget": 32768,
                "max_batch_bytes": 10_000_000,
                "request_bf16": False,
                "resume_retry_class": "INFRA_TRANSIENT",
            },
            "schema": host.HOST_CONFIG_SCHEMA,
            "source_tree_sha256": hashlib.sha256(b"source-tree").hexdigest(),
        },
    )
    return config


def _invoke(arguments: list[str], config: Path) -> tuple[int, dict[str, object], str]:
    stdout = StringIO()
    stderr = StringIO()
    code = cli.main(
        arguments,
        stdout=stdout,
        stderr=stderr,
        runtime_adapter_factory=host.RuntimeAdapterFactory(config),
    )
    raw = stdout.getvalue() or stderr.getvalue()
    return code, json.loads(raw), stderr.getvalue()


def test_prepared_source_loads_strict_batches_and_orders_deterministically(
    tmp_path: Path,
) -> None:
    train, _, _ = _prepared_tree(tmp_path)
    source = host.PrivatePreparedDataSource(train, max_batch_bytes=10_000_000)
    first = tuple(source.iter_epoch(epoch=3, seed=1729))
    second = tuple(source.iter_epoch(epoch=3, seed=1729))
    assert source.split == "train"
    assert len(first) == 2
    assert [row.groups.group_commitments for row in first] == [
        row.groups.group_commitments for row in second
    ]
    assert all(row.split == "train" for row in first)
    assert all(row.text_embeddings.dtype.is_floating_point for row in first)


def test_prepared_source_rejects_mutation_after_construction(tmp_path: Path) -> None:
    train, _, _ = _prepared_tree(tmp_path)
    source = host.PrivatePreparedDataSource(train, max_batch_bytes=10_000_000)
    batch = tmp_path / "train-a.npz"
    batch.write_bytes(batch.read_bytes() + b"changed")
    try:
        tuple(source.iter_epoch(epoch=0, seed=1729))
    except host.HostConfigurationError as error:
        assert "changed" in str(error)
    else:
        raise AssertionError("mutated prepared batch was accepted")


def test_prepared_source_parses_the_exact_buffer_it_hashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = _write_split(tmp_path, "train", ("original",))
    source = host.PrivatePreparedDataSource(train, max_batch_bytes=10_000_000)
    batch = tmp_path / "train-original.npz"
    replacement = tmp_path / "replacement.npz"
    _write_batch(replacement, "train-replacement")
    replacement_raw = replacement.read_bytes()
    original_load = host.np.load

    def swapping_load(value: object, *args: object, **kwargs: object) -> object:
        batch.write_bytes(replacement_raw)
        assert isinstance(value, host.io.BytesIO)
        return original_load(value, *args, **kwargs)

    monkeypatch.setattr(host.np, "load", swapping_load)
    loaded = tuple(source.iter_epoch(epoch=0, seed=1729))
    expected_actor_ids = (
        _commitment("train-original-actor-0"),
        _commitment("train-original-actor-1"),
    )
    assert loaded[0].groups.group_commitments == (
        group_commitment(expected_actor_ids),
    )


def test_preflight_holds_when_real_rights_artifact_is_absent(tmp_path: Path) -> None:
    config = _host_config_with_missing_rights(tmp_path)
    code, payload, stderr = _invoke(
        ["preflight", "--config", str(PUBLIC_CONFIG)], config
    )
    assert code == cli.EXIT_HOLD
    assert stderr == ""
    assert payload["authority"] == 0
    assert payload["production"] is False
    assert payload["execution_authorized"] is False
    assert payload["hold_codes"] == [
        "HOLD_RIGHTS_GRANT_ABSENT",
        "HOLD_EXTERNAL_RECEIPTS_UNVERIFIED",
    ]


def test_run_base_and_resume_do_not_reach_backend_without_rights(
    tmp_path: Path,
) -> None:
    config = _host_config_with_missing_rights(tmp_path)
    attempt = tmp_path / "attempts" / "base-a"
    run_id = "phaseset-run-v1/BASE_QUALIFICATION/1729/B0"
    run_code, run_payload, _ = _invoke(
        [
            "run-base",
            "--run-id",
            run_id,
            "--checkpoint-dir",
            str(attempt / "model-checkpoints"),
            "--config",
            str(PUBLIC_CONFIG),
        ],
        config,
    )
    resume_code, resume_payload, _ = _invoke(
        [
            "resume",
            "--attempt-dir",
            str(tmp_path / "attempts" / "base-b"),
            "--checkpoint",
            str(attempt / "model-checkpoints" / "checkpoint.pt"),
            "--config",
            str(PUBLIC_CONFIG),
        ],
        config,
    )
    assert run_code == resume_code == cli.EXIT_HOLD
    assert run_payload["authority"] == resume_payload["authority"] == 0
    assert not attempt.exists()


def test_live_ledger_defers_last_update_and_recovers_crash(tmp_path: Path) -> None:
    run_id = "phaseset-run-v1/BASE_QUALIFICATION/1729/B0"
    attempt = AttemptRecord(
        attempt_id="live-a",
        run_id=run_id,
        created_at_utc="2026-09-07T12:00:00Z",
        plan_sha256="1" * 64,
        matrix_sha256="2" * 64,
        training_config_sha256="3" * 64,
        source_tree_sha256="4" * 64,
    )
    store = AttemptStore.create(tmp_path, attempt)
    ledger = host._LiveAttemptLedger(store, attempt)
    ledger.start()
    checkpoint = store.path / "model-checkpoints" / "checkpoint-validation.pt"
    checkpoint.parent.mkdir()
    torch.save(
        {
            "model": {},
            "optimizer": {},
            "rng": {},
            "epoch": 1,
            "update_index": 1,
            "updates_per_epoch": 1,
            "global_step": 1,
            "train_manifest_sha256": "5" * 64,
            "val_manifest_sha256": "6" * 64,
            "best_validation_metric": 0.5,
            "best_checkpoint_name": checkpoint.name,
            "best_checkpoint_sha256": None,
        },
        checkpoint,
    )
    artifact = CheckpointArtifact(checkpoint, _sha(checkpoint), 1)
    ledger.observe_checkpoint(artifact, "update")
    assert ledger.latest_checkpoint_receipt_sha256 is None
    ledger.observe_checkpoint(artifact, "validation")
    ledger.stop()
    assert ledger.latest_checkpoint_payload_sha256 == artifact.sha256
    assert ledger.latest_checkpoint_receipt_sha256 is not None
    assert ledger.latest_heartbeat_sha256 is not None

    host.PhaseSetHostBackend._recover_crashed_attempt(store.path)
    terminal = parse_terminal_bytes((store.path / "terminal.json").read_bytes())
    assert terminal.outcome == "FAILED"
    assert terminal.failure_code == "HOST_PROCESS_LOST"
    assert (
        terminal.latest_checkpoint_receipt_sha256
        == ledger.latest_checkpoint_receipt_sha256
    )


def test_live_attempt_lease_refuses_false_crash_terminal(tmp_path: Path) -> None:
    attempt = AttemptRecord(
        attempt_id="still-live",
        run_id="phaseset-run-v1/BASE_QUALIFICATION/1729/B0",
        created_at_utc="2026-09-07T12:00:00Z",
        plan_sha256="1" * 64,
        matrix_sha256="2" * 64,
        training_config_sha256="3" * 64,
        source_tree_sha256="4" * 64,
    )
    store = AttemptStore.create(tmp_path, attempt)
    lease = host._AttemptLease.acquire(store.path)
    try:
        with pytest.raises(host.HostConfigurationError, match="live process"):
            host.PhaseSetHostBackend._recover_crashed_attempt(store.path)
        assert not (store.path / "terminal.json").exists()
    finally:
        lease.release()


def test_attempt_lease_releases_only_after_owned_child_exits(tmp_path: Path) -> None:
    attempt = AttemptRecord(
        attempt_id="child-live",
        run_id="phaseset-run-v1/BASE_QUALIFICATION/1729/B0",
        created_at_utc="2026-09-07T12:00:00Z",
        plan_sha256="1" * 64,
        matrix_sha256="2" * 64,
        training_config_sha256="3" * 64,
        source_tree_sha256="4" * 64,
    )
    store = AttemptStore.create(tmp_path, attempt)
    ready = tmp_path / "child-ready"
    child_code = """
import sys
import time
from pathlib import Path

from phaseset_core import host

lease = host._AttemptLease.acquire(Path(sys.argv[1]))
Path(sys.argv[2]).write_text("ready\\n", encoding="ascii")
try:
    time.sleep(30.0)
finally:
    lease.release()
"""
    child = subprocess.Popen(
        [sys.executable, "-c", child_code, str(store.path), str(ready)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        # A cold Torch import on a shared Windows runner can exceed 10s. This
        # bounds process startup, not the live-lease exclusion check below.
        deadline = time.monotonic() + 30.0
        while not ready.is_file() and time.monotonic() < deadline:
            if child.poll() is not None:
                _, stderr = child.communicate(timeout=1.0)
                pytest.fail(f"lease child exited before ready: {stderr}")
            time.sleep(0.02)
        assert ready.is_file(), "lease child did not signal ready before timeout"
        with pytest.raises(host.HostConfigurationError, match="live process"):
            host._AttemptLease.acquire(store.path)
        assert not (store.path / "terminal.json").exists()

        child.kill()
        child.wait(timeout=10.0)
        lease = host._AttemptLease.acquire(store.path)
        lease.release()
        assert not (store.path / "terminal.json").exists()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10.0)


def test_ensure_checkpoint_receipts_no_progress_report_checkpoint(
    tmp_path: Path,
) -> None:
    attempt = AttemptRecord(
        attempt_id="no-progress",
        run_id="phaseset-run-v1/BASE_QUALIFICATION/1729/B0",
        created_at_utc="2026-09-07T12:00:00Z",
        plan_sha256="1" * 64,
        matrix_sha256="2" * 64,
        training_config_sha256="3" * 64,
        source_tree_sha256="4" * 64,
    )
    store = AttemptStore.create(tmp_path, attempt)
    checkpoint = store.path / "model-checkpoints" / "restored.pt"
    checkpoint.parent.mkdir()
    torch.save(
        {
            "model": {},
            "optimizer": {},
            "rng": {},
            "epoch": 1,
            "update_index": 1,
            "updates_per_epoch": 1,
            "global_step": 1,
            "train_manifest_sha256": "5" * 64,
            "val_manifest_sha256": "6" * 64,
            "best_validation_metric": 0.5,
            "best_checkpoint_name": checkpoint.name,
            "best_checkpoint_sha256": None,
        },
        checkpoint,
    )
    ledger = host._LiveAttemptLedger(store, attempt)
    ledger.start()
    artifact = CheckpointArtifact(checkpoint, _sha(checkpoint), 1)
    ledger.ensure_checkpoint(artifact)
    first_receipt = ledger.latest_checkpoint_receipt_sha256
    ledger.ensure_checkpoint(artifact)
    ledger.stop()
    assert first_receipt is not None
    assert ledger.latest_checkpoint_receipt_sha256 == first_receipt


def test_failed_execution_returns_only_after_real_artifacts_persist(
    tmp_path: Path,
) -> None:
    attempt = AttemptRecord(
        attempt_id="failed",
        run_id="phaseset-run-v1/BASE_QUALIFICATION/1729/B0",
        created_at_utc="2026-09-07T12:00:00Z",
        plan_sha256="1" * 64,
        matrix_sha256="2" * 64,
        training_config_sha256="3" * 64,
        source_tree_sha256="4" * 64,
    )
    store = AttemptStore.create(tmp_path, attempt)
    ledger = host._LiveAttemptLedger(store, attempt)
    ledger.start()
    ledger.stop()
    execution = host.PhaseSetHostBackend._persist_failed_execution(
        store, attempt, ledger, RuntimeError("not serialized")
    )
    assert execution.outcome == "FAILED"
    assert all(path.is_file() for path in execution.artifacts)
    failure = json.loads((store.path / "failure.json").read_text(encoding="ascii"))
    terminal = parse_terminal_bytes((store.path / "terminal.json").read_bytes())
    assert failure["error_type"] == "RuntimeError"
    assert "not serialized" not in (store.path / "failure.json").read_text(
        encoding="ascii"
    )
    assert terminal.failure_code == "HOST_EXECUTION_FAILED"


def test_failed_terminal_follows_durable_failure_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = AttemptRecord(
        attempt_id="durable-failure",
        run_id="phaseset-run-v1/BASE_QUALIFICATION/1729/B0",
        created_at_utc="2026-09-07T12:00:00Z",
        plan_sha256="1" * 64,
        matrix_sha256="2" * 64,
        training_config_sha256="3" * 64,
        source_tree_sha256="4" * 64,
    )
    store = AttemptStore.create(tmp_path, attempt)
    ledger = host._LiveAttemptLedger(store, attempt)
    ledger.start()
    ledger.stop()
    events: list[str] = []
    real_fsync = host.os.fsync
    real_write_terminal = AttemptStore.write_terminal

    def tracking_fsync(descriptor: int) -> None:
        mode = os.fstat(descriptor).st_mode
        events.append("directory-fsync" if stat.S_ISDIR(mode) else "file-fsync")
        real_fsync(descriptor)

    def tracking_terminal(self: AttemptStore, record: object) -> str:
        events.append("terminal")
        return real_write_terminal(self, record)

    monkeypatch.setattr(host.os, "fsync", tracking_fsync)
    monkeypatch.setattr(AttemptStore, "write_terminal", tracking_terminal)
    host.PhaseSetHostBackend._persist_failed_execution(
        store, attempt, ledger, RuntimeError("failure")
    )
    assert events[0] == "file-fsync"
    if os.name == "posix":
        assert events[1] == "directory-fsync"
        assert events[2] == "terminal"
    else:
        assert events[1] == "terminal"


def test_fsync_failure_prevents_failed_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = AttemptRecord(
        attempt_id="nondurable-failure",
        run_id="phaseset-run-v1/BASE_QUALIFICATION/1729/B0",
        created_at_utc="2026-09-07T12:00:00Z",
        plan_sha256="1" * 64,
        matrix_sha256="2" * 64,
        training_config_sha256="3" * 64,
        source_tree_sha256="4" * 64,
    )
    store = AttemptStore.create(tmp_path, attempt)
    ledger = host._LiveAttemptLedger(store, attempt)
    ledger.start()
    ledger.stop()

    def fail_fsync(_: int) -> None:
        raise OSError("injected durability failure")

    monkeypatch.setattr(host.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="injected durability failure"):
        host.PhaseSetHostBackend._persist_failed_execution(
            store, attempt, ledger, RuntimeError("failure")
        )
    assert (store.path / "failure.json").exists()
    assert not (store.path / "terminal.json").exists()
