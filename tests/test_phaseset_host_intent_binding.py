from __future__ import annotations

import hashlib
import json
from io import StringIO
from pathlib import Path

import numpy as np
import pytest
from phaseset_core import cli, execution, experiments, host
from phaseset_core.contracts import group_commitment


# The focused test must be copied into ``<public checkout>/tests`` and run with
# the candidate host overlaid at ``src/phaseset_core/host.py``.  Do not infer a
# repository root from an installed wheel's package location: wheels need not
# contain the repository-level frozen config fixture.
PUBLIC_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_CONFIG = PUBLIC_ROOT / "configs" / "phaseset" / "training.json"


class _StoppedBeforeModel(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _commitment(label: str) -> bytes:
    return hashlib.sha256(label.encode("ascii")).digest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="ascii",
    )


def _write_batch(path: Path, label: str) -> None:
    actors = (_commitment(f"{label}-actor-0"), _commitment(f"{label}-actor-1"))
    family = _commitment(f"{label}-family")
    np.savez(
        path,
        skeletons=np.zeros((1, 2, 200, 22, 3), dtype=np.float32),
        actor_mask=np.ones((1, 2), dtype=np.bool_),
        frame_mask=np.ones((1, 200), dtype=np.bool_),
        track_mask=np.ones((1, 2, 200, 22), dtype=np.bool_),
        actor_commitments=np.asarray([[list(actors[0]), list(actors[1])]], dtype=np.uint8),
        group_commitments=np.asarray([list(group_commitment(actors))], dtype=np.uint8),
        text_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        motion_positive_ids=np.asarray([list(family)], dtype=np.uint8),
        text_positive_ids=np.asarray([list(family)], dtype=np.uint8),
        text_commitments=np.asarray([list(_commitment(f"{label}-text"))], dtype=np.uint8),
    )


def _valid_host_config(root: Path) -> Path:
    split_rows: dict[str, dict[str, str]] = {}
    for split in ("train", "val"):
        batch = root / f"{split}.npz"
        _write_batch(batch, split)
        manifest = root / f"{split}.json"
        _write_json(
            manifest,
            {
                "batches": [{"path": batch.name, "sha256": _sha(batch)}],
                "schema": host.PREPARED_SPLIT_SCHEMA,
                "split": split,
            },
        )
        split_rows[split] = {"path": manifest.name, "sha256": _sha(manifest)}
    index = root / "prepared-index.json"
    _write_json(index, {"schema": host.PREPARED_INDEX_SCHEMA, **split_rows})

    artifact_paths: dict[str, str] = {}
    artifact_digests: dict[str, str] = {}
    for field in host.RECEIPT_DIGEST_FIELDS:
        artifact = index if field == "prepared_data_manifest_sha256" else root / field
        if artifact != index:
            artifact.write_text(f"{field}\n", encoding="ascii")
        artifact_paths[field] = artifact.name
        artifact_digests[field] = _sha(artifact)
    receipts = root / "receipts.json"
    _write_json(
        receipts,
        {
            "admitted_at_utc": "2026-09-08T12:00:00Z",
            "authority": 7,
            **artifact_digests,
        },
    )
    config = root / "host.json"
    _write_json(
        config,
        {
            "prepared_index": index.name,
            "receipts": {"artifacts": artifact_paths, "record": receipts.name},
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


@pytest.mark.parametrize("command", ["run-base", "run-residual"])
def test_real_cli_intent_uses_authenticated_admission_bindings_before_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    config = _valid_host_config(tmp_path)
    attempts = tmp_path / "attempts"
    attempts.mkdir()
    attempt = attempts / command
    checkpoint_directory = attempt / "model-checkpoints"
    captured_intents: list[execution.CommandIntent] = []
    original_execute = host.PhaseSetHostBackend.execute

    def capture_execute(
        self: host.PhaseSetHostBackend,
        intent: execution.CommandIntent,
    ) -> object:
        assert type(intent) is execution.CommandIntent
        captured_intents.append(intent)
        return original_execute(self, intent)

    def stop_before_base_model(self: object, **kwargs: object) -> object:
        del self, kwargs
        raise _StoppedBeforeModel("test stopped before base model construction")

    def stop_before_residual_model(self: object, **kwargs: object) -> object:
        del self, kwargs
        raise _StoppedBeforeModel("test stopped before residual model construction")

    monkeypatch.setattr(host.PhaseSetHostBackend, "execute", capture_execute)
    monkeypatch.setattr(
        host.PhaseSetHostBackend,
        "_execute_base_runtime",
        stop_before_base_model,
    )
    monkeypatch.setattr(
        host.PhaseSetHostBackend,
        "_execute_residual_runtime",
        stop_before_residual_model,
    )

    if command == "run-base":
        run_id = "phaseset-run-v1/BASE_QUALIFICATION/1729/B0"
        arguments = [
            "run-base",
            "--run-id",
            run_id,
            "--checkpoint-dir",
            str(checkpoint_directory),
            "--config",
            str(PUBLIC_CONFIG),
        ]
    else:
        run_id = "phaseset-run-v1/RESIDUAL_TRAIN/1729/01"
        base_checkpoint = tmp_path / "base-selected.pt"
        base_checkpoint.write_bytes(b"test stops before decoding this checkpoint\n")
        periodic_cache = tmp_path / "periodic-cache"
        periodic_cache.mkdir()
        arguments = [
            "run-residual",
            "--run-id",
            run_id,
            "--checkpoint-dir",
            str(checkpoint_directory),
            "--base-checkpoint",
            str(base_checkpoint),
            "--periodic-cache",
            str(periodic_cache),
            "--config",
            str(PUBLIC_CONFIG),
        ]

    code, payload, stderr = _invoke(arguments, config)
    assert code == cli.EXIT_EXECUTION_FAILED
    assert stderr == ""
    assert payload["command"] == command
    assert payload["outcome"] == "FAILED"
    assert len(captured_intents) == 1
    intent = captured_intents[0]
    assert not hasattr(intent, "matrix_sha256")
    assert not hasattr(intent, "training_config_sha256")

    record = execution.parse_attempt_bytes((attempt / "attempt.json").read_bytes())
    plan = experiments.build_experiment_plan()
    assert record.run_id == run_id
    assert record.plan_sha256 == experiments.experiment_plan_sha256(plan)
    assert record.matrix_sha256 == plan.matrix_sha256
    assert record.training_config_sha256 == execution.artifact_sha256(
        execution.canonical_public_training_config_bytes()
    )
    terminal = execution.parse_terminal_bytes((attempt / "terminal.json").read_bytes())
    assert terminal.outcome == "FAILED"
    assert terminal.failure_code == "HOST_EXECUTION_FAILED"
    assert not checkpoint_directory.exists()


def test_backend_rejects_noncanonical_admissions_without_poisoning_snapshot(
    tmp_path: Path,
) -> None:
    config_path = _valid_host_config(tmp_path)
    config = host.HostConfig.load(config_path)
    request = cli.CLICommandRequest(command="run-base")
    backend = host.PhaseSetHostBackend(config, request)
    receipts = config.load_receipts()
    plan = experiments.build_experiment_plan()
    canonical = execution.RuntimeAdmissionRequest(
        plan_sha256=experiments.experiment_plan_sha256(plan),
        matrix_sha256=plan.matrix_sha256,
        training_config_sha256=execution.artifact_sha256(
            execution.canonical_public_training_config_bytes()
        ),
        handler_manifest_sha256=execution.runtime_handler_manifest_sha256(
            execution.COMMANDS
        ),
    )
    changed_matrix = execution.RuntimeAdmissionRequest(
        plan_sha256=canonical.plan_sha256,
        matrix_sha256="f" * 64,
        training_config_sha256=canonical.training_config_sha256,
        handler_manifest_sha256=canonical.handler_manifest_sha256,
    )
    changed_commands = execution.RuntimeAdmissionRequest(
        plan_sha256=canonical.plan_sha256,
        matrix_sha256=canonical.matrix_sha256,
        training_config_sha256=canonical.training_config_sha256,
        handler_manifest_sha256=canonical.handler_manifest_sha256,
        commands=tuple(reversed(execution.COMMANDS)),
    )
    changed_handler = execution.RuntimeAdmissionRequest(
        plan_sha256=canonical.plan_sha256,
        matrix_sha256=canonical.matrix_sha256,
        training_config_sha256=canonical.training_config_sha256,
        handler_manifest_sha256="e" * 64,
    )
    for changed in (changed_matrix, changed_commands, changed_handler):
        assert backend.authenticate(changed, receipts, "0" * 64) is False
        with pytest.raises(host.HostConfigurationError, match="admission"):
            backend._attempt_plan_bindings()
    assert backend.authenticate(canonical, receipts, "0" * 64) is True
    assert backend.authenticate(canonical, receipts, "0" * 64) is True
    assert backend._attempt_plan_bindings() == (
        canonical.plan_sha256,
        canonical.matrix_sha256,
        canonical.training_config_sha256,
    )


def test_request_specific_backend_rejects_other_real_intent_command(
    tmp_path: Path,
) -> None:
    config = host.HostConfig.load(_valid_host_config(tmp_path))
    backend = host.PhaseSetHostBackend(
        config,
        cli.CLICommandRequest(command="run-base"),
    )
    intent = execution.CommandIntent(
        command="resume",
        run_id=None,
        seed=None,
        split=None,
        plan_sha256="0" * 64,
    )
    with pytest.raises(host.HostConfigurationError, match="request-specific"):
        backend.execute(intent)
