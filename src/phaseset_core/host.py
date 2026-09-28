"""Private-host harness for the public PhaseSet runtime.

This module is deliberately an in-process composition root.  It does not read
an endpoint, key, or adapter import path from the public PhaseSet CLI.  A host
configuration names existing private receipt artifacts and prepared numeric
microbatches; missing or mismatched artifacts stop at the repository's
existing ``ExecutionHold`` boundary.

The implemented command surface is intentionally small: ``preflight``,
``run-base``, ``qualify-base``, ``run-residual``, and base/residual ``resume``.
Other commands raise before a ``BackendExecution`` can be manufactured.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import io
import json
import os
import stat
import sys
import threading
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
from phaseset_core import cli
from phaseset_core.base_qualification_host import (
    CONFIG_SCHEMA as BASE_QUALIFICATION_CONTROLLER_SCHEMA,
    HostBaseQualificationAuthentication,
    HostBaseQualificationConfig,
    HostBaseQualificationError,
    load_base_attempt_registry,
    run_host_base_qualification,
)
from phaseset_core.base_cohort_resolver import BaseCohortAdmission
from phaseset_core.capture_prepared_storage import (
    CapturePreparedStorageError,
    CaptureStorageLimits,
    load_capture_validation_source,
)
from phaseset_core.capture_validation import CaptureValidationSource
from phaseset_core.contracts import PreparedGroupBatch
from phaseset_core.execution import (
    AttemptRecord,
    AttemptStore,
    CheckpointRecord,
    CommandIntent,
    ExecutionHold,
    HeartbeatRecord,
    ResumeRecord,
    RuntimeAdmissionRequest,
    TerminalRecord,
    artifact_sha256,
    canonical_attempt_bytes,
    canonical_public_training_config_bytes,
    parse_attempt_bytes,
    parse_checkpoint_bytes,
    parse_periodic_cache_bytes,
    parse_terminal_bytes,
)
from phaseset_core.experiments import (
    SEEDS,
    BaseQualification,
    BaseScore,
    build_experiment_plan,
    canonical_base_qualification_bytes,
    experiment_plan_sha256,
    parse_run_id,
)
from phaseset_core.prepared_data_v2 import (
    PREPARED_INDEX_V2_SCHEMA,
    PreparedDataV2Error,
    load_prepared_training_sources_v2,
)
from phaseset_core.production import (
    BackendExecution,
    PrivateBaseQualificationAuthorization,
    PrivateReceiptAssertions,
    ProductionRuntimeAdapter,
)
from phaseset_core.training import (
    CheckpointArtifact,
    PhaseSetTrainingRuntime,
    ResidualCapacityAudit,
    RetrievalTrainingBatch,
    TrainingConfig,
    TrainingDataSource,
    TrainingReport,
    TrainingRuntimeError,
    construct_registered_base_seed_bound_system,
    construct_registered_residual_seed_bound_system,
    load_qualified_frozen_base,
)

HOST_CONFIG_SCHEMA = "phaseset-private-host-v1"
HOST_CONFIG_QUALIFICATION_SCHEMA = "phaseset-private-host-v2"
PREPARED_INDEX_SCHEMA = "phaseset-prepared-index-v1"
PREPARED_SPLIT_SCHEMA = "phaseset-prepared-training-split-v1"
CAPTURE_VALIDATION_INDEX_SCHEMA = "phaseset-host-capture-validation-index-v1"
CAPTURE_VALIDATION_KIND = "capture-validation-storage-v1"
PREPARED_TRAIN_V1_KIND = "prepared-training-index-v1"
PREPARED_TRAIN_V2_KIND = "prepared-training-index-v2"
PERIODIC_CACHE_MAX_BYTES = 1024 * 1024
PREPARED_BATCH_KEYS = frozenset(
    {
        "skeletons",
        "actor_mask",
        "frame_mask",
        "track_mask",
        "actor_commitments",
        "group_commitments",
        "text_embeddings",
        "motion_positive_ids",
        "text_positive_ids",
        "text_commitments",
    }
)
RECEIPT_DIGEST_FIELDS = (
    "data_manifest_sha256",
    "prepared_data_manifest_sha256",
    "split_audit_sha256",
    "rights_assertion_sha256",
    "caption_manifest_sha256",
    "score_execution_census_sha256",
    "validation_evaluation_census_sha256",
    "hard_gallery_freeze_binding_sha256",
    "validation_hard_gallery_collection_sha256",
    "runtime_assertion_sha256",
    "execution_assertion_sha256",
)


class HostConfigurationError(ValueError):
    """The private host configuration or prepared seam is not closed."""


def _qualify_base_request_is_closed(request: cli.CLICommandRequest) -> bool:
    """Whether qualify-base carries exactly its one factory-only path."""

    return request.terminal_root is not None and all(
        value is None
        for value in (
            request.run_id,
            request.seed,
            request.split,
            request.system_id,
            request.source_manifest,
            request.prepared_root,
            request.split_manifest,
            request.base_checkpoint,
            request.periodic_cache,
            request.cache_root,
            request.checkpoint_directory,
            request.resume_checkpoint,
            request.stop_after_global_step,
            request.evaluation_manifest,
            request.statistical_report,
            request.attempt_directory,
        )
    )


def _qualify_base_runtime_hold(*, prepared_data: bool) -> ExecutionHold:
    """Map only post-admission qualify-base runtime drift to the public HOLD seam."""

    if type(prepared_data) is not bool:
        raise TypeError("prepared_data must be exact bool")
    holds = (
        (
            "HOLD_PREPARED_DATA_MANIFEST_ABSENT",
            "HOLD_EXTERNAL_RECEIPTS_UNVERIFIED",
        )
        if prepared_data
        else ("HOLD_EXTERNAL_RECEIPTS_UNVERIFIED",)
    )
    return ExecutionHold("qualify-base", holds)


class _AttemptLease:
    """Process-released exclusive lease held for one attempt's full lifetime."""

    FILE_NAME = ".lifetime.lock"

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._released = False

    @classmethod
    def acquire(cls, attempt_directory: Path) -> _AttemptLease:
        _reject_symlink_components(attempt_directory, "attempt directory")
        if not attempt_directory.is_dir():
            raise HostConfigurationError("attempt directory is absent")
        path = attempt_directory / cls.FILE_NAME
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as error:
            raise HostConfigurationError("attempt lifetime lease cannot be opened") from error
        stream = os.fdopen(descriptor, "r+b", buffering=0)
        try:
            if os.name == "posix":
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            elif os.name == "nt":
                import msvcrt

                stream.seek(0, os.SEEK_END)
                if stream.tell() == 0:
                    stream.write(b"\x00")
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                raise HostConfigurationError(
                    "attempt lifetime leases support POSIX and Windows hosts only"
                )
        except OSError as error:
            stream.close()
            contention_errors = {errno.EACCES, errno.EAGAIN, errno.EDEADLK}
            if error.errno in contention_errors:
                raise HostConfigurationError(
                    "attempt lifetime lease is held by a live process"
                ) from error
            raise HostConfigurationError(
                "attempt lifetime lease cannot be acquired"
            ) from error
        except BaseException:
            stream.close()
            raise
        return cls(stream)

    def release(self) -> None:
        if self._released:
            return
        try:
            if os.name == "posix":
                import fcntl

                fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
            elif os.name == "nt":
                import msvcrt

                self._stream.seek(0)
                msvcrt.locking(self._stream.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            self._released = True
            self._stream.close()

    def __enter__(self) -> _AttemptLease:
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


@dataclass(frozen=True, slots=True)
class ResidualArtifacts:
    qualification: Path
    capacity_audits: Mapping[int, Path]


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _lower_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise HostConfigurationError(f"{label} must be lowercase SHA-256 hex")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_bounded_bytes(path: Path, *, max_bytes: int, label: str) -> bytes:
    if type(max_bytes) is not int or max_bytes < 1:
        raise HostConfigurationError(f"{label} byte limit must be positive")
    _reject_symlink_components(path, label)
    if not path.is_file():
        raise HostConfigurationError(f"{label} must be a regular existing file")
    with path.open("rb") as stream:
        raw = stream.read(max_bytes + 1)
    if not raw or len(raw) > max_bytes:
        raise HostConfigurationError(f"{label} size is outside the host limit")
    return raw


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json_object(path: Path, label: str) -> tuple[bytes, dict[str, object]]:
    if path.is_symlink() or not path.is_file():
        raise HostConfigurationError(f"{label} must be a regular non-symlink file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HostConfigurationError(f"{label} is not valid JSON") from error
    if type(value) is not dict:
        raise HostConfigurationError(f"{label} must contain one JSON object")
    return raw, value


def _require_canonical_json_object(
    raw: bytes,
    value: Mapping[str, object],
    label: str,
) -> None:
    try:
        canonical = _canonical_json_bytes(value)
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise HostConfigurationError(f"{label} is not canonical JSON") from error
    if raw != canonical:
        raise HostConfigurationError(f"{label} bytes are not canonical")


def _closed_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise HostConfigurationError(f"{label} keys are not closed")


def _load_base_qualification(path: Path) -> tuple[BaseQualification, str]:
    raw, value = _load_json_object(path, "base qualification")
    expected_keys = {
        "authority",
        "completion_sha256s",
        "evaluator_sha256",
        "production",
        "query_census_sha256",
        "result_claimed",
        "schema",
        "score_artifact_sha256s",
        "score_rows",
        "score_rows_sha256",
        "selected_checkpoint_sha256s",
        "status",
        "system_mean_rows",
        "system_resource_rows",
        "validation_manifest_sha256",
        "winner_checkpoint_sha256s",
        "winner_frozen_runtime_latency_ns",
        "winner_parameter_count",
        "winner_system_id",
        "winner_system_name",
        "winner_terminal_sha256s",
    }
    _closed_keys(value, expected_keys, "base qualification")
    score_values = value["score_rows"]
    mean_values = value["system_mean_rows"]
    resource_values = value["system_resource_rows"]
    if not all(
        type(rows) is list for rows in (score_values, mean_values, resource_values)
    ):
        raise HostConfigurationError(
            "base qualification row collections must be arrays"
        )
    try:
        score_rows = tuple(BaseScore(**row) for row in score_values)
        mean_rows = tuple(
            (row["system_id"], row["mean_numerator"], row["mean_denominator"])
            for row in mean_values
        )
        resource_rows = tuple(
            (
                row["system_id"],
                row["parameter_count"],
                row["frozen_runtime_latency_ns"],
            )
            for row in resource_values
        )
        qualification = BaseQualification(
            score_rows=score_rows,
            winner_system_id=value["winner_system_id"],
            winner_system_name=value["winner_system_name"],
            winner_parameter_count=value["winner_parameter_count"],
            winner_frozen_runtime_latency_ns=value["winner_frozen_runtime_latency_ns"],
            system_mean_rows=mean_rows,
            system_resource_rows=resource_rows,
            completion_sha256s=tuple(value["completion_sha256s"]),
            selected_checkpoint_sha256s=tuple(value["selected_checkpoint_sha256s"]),
            winner_terminal_sha256s=tuple(value["winner_terminal_sha256s"]),
            winner_checkpoint_sha256s=tuple(value["winner_checkpoint_sha256s"]),
            validation_manifest_sha256=value["validation_manifest_sha256"],
            query_census_sha256=value["query_census_sha256"],
            evaluator_sha256=value["evaluator_sha256"],
            score_artifact_sha256s=tuple(value["score_artifact_sha256s"]),
            score_rows_sha256=value["score_rows_sha256"],
            schema=value["schema"],
            status=value["status"],
            authority=value["authority"],
            production=value["production"],
            result_claimed=value["result_claimed"],
        )
    except (KeyError, TypeError) as error:
        raise HostConfigurationError("base qualification rows are malformed") from error
    digest = _sha256_bytes(raw)
    if canonical_base_qualification_bytes(qualification, expected_sha256=digest) != raw:
        raise HostConfigurationError("base qualification bytes are not canonical")
    return qualification, digest


def _load_capacity_audit(path: Path) -> tuple[ResidualCapacityAudit, str]:
    raw, value = _load_json_object(path, "residual capacity audit")
    _closed_keys(
        value,
        {
            "code_artifact_sha256",
            "edge_budget",
            "energy_floors_sha256",
            "environment_sha256",
            "frozen_base_checkpoint_sha256",
            "frozen_base_state_sha256",
            "full_system_id",
            "qualified_base_selection_sha256",
            "rows",
            "schema",
            "seed",
            "tolerance_decimal",
        },
        "residual capacity audit",
    )
    if value["full_system_id"] != "08" or value["tolerance_decimal"] != "0.01":
        raise HostConfigurationError("residual capacity audit constants changed")
    rows = value["rows"]
    if type(rows) is not list:
        raise HostConfigurationError("residual capacity rows must be an array")
    try:
        audit = ResidualCapacityAudit(
            rows=tuple(
                (
                    row["system_id"],
                    row["trainable_parameters"],
                    row["behavior_sha256"],
                )
                for row in rows
            ),
            seed=value["seed"],
            edge_budget=value["edge_budget"],
            energy_floors_sha256=value["energy_floors_sha256"],
            frozen_base_checkpoint_sha256=value["frozen_base_checkpoint_sha256"],
            frozen_base_state_sha256=value["frozen_base_state_sha256"],
            qualified_base_selection_sha256=value["qualified_base_selection_sha256"],
            code_artifact_sha256=value["code_artifact_sha256"],
            environment_sha256=value["environment_sha256"],
            schema=value["schema"],
        )
    except (KeyError, TypeError) as error:
        raise HostConfigurationError("residual capacity rows are malformed") from error
    if audit.canonical_bytes() != raw:
        raise HostConfigurationError("residual capacity audit bytes are not canonical")
    return audit, _sha256_bytes(raw)


def _load_periodic_cache(
    root: Path,
    *,
    seed: int,
    qualification: BaseQualification,
    qualification_sha256: str,
) -> np.ndarray:
    _reject_symlink_components(root, "periodic cache")
    directory = root.resolve()
    if not directory.is_dir():
        raise HostConfigurationError("periodic cache must be a non-symlink directory")
    record_path = directory / "record.json"
    payload_path = directory / "cache-content.npz"
    if any(
        path.is_symlink() or not path.is_file() for path in (record_path, payload_path)
    ):
        raise HostConfigurationError("periodic cache record or payload is absent")
    record = parse_periodic_cache_bytes(record_path.read_bytes())
    payload_raw = _read_bounded_bytes(
        payload_path,
        max_bytes=PERIODIC_CACHE_MAX_BYTES,
        label="periodic cache payload",
    )
    expected_run_id = (
        f"phaseset-run-v1/BASE_QUALIFICATION/{seed}/{qualification.winner_system_id}"
    )
    expected_terminal = qualification.winner_terminal_sha256s[SEEDS.index(seed)]
    if (
        record.seed != seed
        or record.qualified_base_system_id != qualification.winner_system_id
        or record.source_run_id != expected_run_id
        or record.base_terminal_sha256 != expected_terminal
        or record.qualification_sha256 != qualification_sha256
        or record.cache_content_sha256 != _sha256_bytes(payload_raw)
    ):
        raise HostConfigurationError(
            "periodic cache is not bound to the selected seed/base/qualification"
        )
    try:
        with np.load(io.BytesIO(payload_raw), allow_pickle=False) as archive:
            if "energy_floors" not in archive.files:
                raise HostConfigurationError("periodic cache lacks energy_floors")
            floors = np.array(archive["energy_floors"], copy=True)
    except (OSError, ValueError) as error:
        raise HostConfigurationError(
            "periodic cache payload is not safe numeric NPZ"
        ) from error
    if floors.dtype != np.float64 or floors.shape != (6,):
        raise HostConfigurationError("periodic cache energy_floors must be float64 [6]")
    return np.ascontiguousarray(floors)


def _reject_symlink_components(path: Path, label: str) -> None:
    for component in (path, *path.parents):
        if component.is_symlink():
            raise HostConfigurationError(f"{label} cannot traverse a symlink")


def _resolve_existing_file(base: Path, raw: object, label: str) -> Path:
    if type(raw) is not str or not raw or "\x00" in raw:
        raise HostConfigurationError(f"{label} must be a nonempty path string")
    unresolved = base / raw if not Path(raw).is_absolute() else Path(raw)
    _reject_symlink_components(unresolved, label)
    candidate = unresolved.resolve()
    if not candidate.is_file():
        raise HostConfigurationError(f"{label} must be a regular existing file")
    return candidate


def _resolve_config_path(base: Path, raw: object, label: str) -> Path:
    if type(raw) is not str or not raw or "\x00" in raw:
        raise HostConfigurationError(f"{label} must be a nonempty path string")
    unresolved = base / raw if not Path(raw).is_absolute() else Path(raw)
    _reject_symlink_components(unresolved, label)
    return unresolved.resolve()


def _resolve_under(root: Path, raw: object, label: str) -> Path:
    if type(raw) is not str or not raw or "\x00" in raw or Path(raw).is_absolute():
        raise HostConfigurationError(f"{label} must be a relative path")
    unresolved = root / raw
    _reject_symlink_components(unresolved, label)
    candidate = unresolved.resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise HostConfigurationError(
            f"{label} escapes its prepared-data root"
        ) from error
    if not candidate.is_file():
        raise HostConfigurationError(f"{label} must be a regular existing file")
    return candidate


def _bytes32_rows(value: np.ndarray, label: str) -> tuple[bytes, ...]:
    if value.dtype != np.uint8 or value.ndim != 2 or value.shape[1] != 32:
        raise HostConfigurationError(f"{label} must be uint8 [N,32]")
    return tuple(bytes(row) for row in np.ascontiguousarray(value))


def _actor_commitment_rows(
    values: np.ndarray, actor_mask: np.ndarray
) -> tuple[tuple[bytes | None, ...], ...]:
    if (
        values.dtype != np.uint8
        or values.ndim != 3
        or values.shape[:2] != actor_mask.shape
        or values.shape[2] != 32
    ):
        raise HostConfigurationError("actor_commitments must be uint8 [B,K,32]")
    rows: list[tuple[bytes | None, ...]] = []
    for batch_index in range(values.shape[0]):
        row: list[bytes | None] = []
        for actor_index in range(values.shape[1]):
            if bool(actor_mask[batch_index, actor_index]):
                row.append(bytes(values[batch_index, actor_index]))
            else:
                row.append(None)
        rows.append(tuple(row))
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class _BatchEntry:
    path: Path
    sha256: str


class PrivatePreparedDataSource:
    """Strict ``allow_pickle=False`` prepared-microbatch source.

    A split manifest lists immutable NPZ files.  Epoch ordering is a stable
    digest sort over the frozen split-manifest identity, seed, epoch, and row;
    no global RNG state or filesystem enumeration enters sampling.
    """

    def __init__(self, manifest: str | Path, *, max_batch_bytes: int = 2**31) -> None:
        path = Path(manifest).resolve()
        raw, value = _load_json_object(path, "prepared split manifest")
        _closed_keys(value, {"batches", "schema", "split"}, "prepared split manifest")
        if value["schema"] != PREPARED_SPLIT_SCHEMA:
            raise HostConfigurationError("prepared split schema is not registered")
        split = value["split"]
        if split not in {"train", "val"}:
            raise HostConfigurationError("prepared split must be train or val")
        rows = value["batches"]
        if type(rows) is not list or not rows:
            raise HostConfigurationError("prepared split must list at least one batch")
        if type(max_batch_bytes) is not int or max_batch_bytes < 1:
            raise HostConfigurationError("max_batch_bytes must be positive")
        entries: list[_BatchEntry] = []
        seen_paths: set[Path] = set()
        for index, item in enumerate(rows):
            if type(item) is not dict:
                raise HostConfigurationError(f"batch row {index} must be an object")
            _closed_keys(item, {"path", "sha256"}, f"batch row {index}")
            batch_path = _resolve_under(
                path.parent, item["path"], f"batch row {index} path"
            )
            digest = item["sha256"]
            if type(digest) is not str or len(digest) != 64 or digest.lower() != digest:
                raise HostConfigurationError(f"batch row {index} sha256 is malformed")
            if batch_path in seen_paths:
                raise HostConfigurationError("prepared split repeats a batch path")
            batch_raw = _read_bounded_bytes(
                batch_path,
                max_bytes=max_batch_bytes,
                label=f"batch row {index}",
            )
            if _sha256_bytes(batch_raw) != digest:
                raise HostConfigurationError("prepared batch digest mismatch")
            seen_paths.add(batch_path)
            entries.append(_BatchEntry(batch_path, digest))
        self.split = split
        self.manifest_sha256 = _sha256_bytes(raw)
        self._entries = tuple(entries)
        self._max_batch_bytes = max_batch_bytes

    def _load_batch(self, entry: _BatchEntry) -> RetrievalTrainingBatch:
        raw = _read_bounded_bytes(
            entry.path,
            max_bytes=self._max_batch_bytes,
            label="prepared batch",
        )
        if _sha256_bytes(raw) != entry.sha256:
            raise HostConfigurationError(
                "prepared batch changed after source construction"
            )
        try:
            with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
                if set(archive.files) != PREPARED_BATCH_KEYS:
                    raise HostConfigurationError(
                        "prepared batch NPZ keys are not closed"
                    )
                arrays = {
                    name: np.array(archive[name], copy=True) for name in archive.files
                }
        except (OSError, ValueError) as error:
            raise HostConfigurationError(
                "prepared batch is not a safe numeric NPZ"
            ) from error
        exact_dtypes = {
            "skeletons": np.dtype(np.float32),
            "actor_mask": np.dtype(np.bool_),
            "frame_mask": np.dtype(np.bool_),
            "track_mask": np.dtype(np.bool_),
            "actor_commitments": np.dtype(np.uint8),
            "group_commitments": np.dtype(np.uint8),
            "text_embeddings": np.dtype(np.float32),
            "motion_positive_ids": np.dtype(np.uint8),
            "text_positive_ids": np.dtype(np.uint8),
            "text_commitments": np.dtype(np.uint8),
        }
        for name, dtype in exact_dtypes.items():
            if arrays[name].dtype != dtype:
                raise HostConfigurationError(
                    f"prepared batch {name} has the wrong dtype"
                )
        actor_mask = np.ascontiguousarray(arrays["actor_mask"])
        groups = PreparedGroupBatch(
            skeletons=np.ascontiguousarray(arrays["skeletons"]),
            actor_mask=actor_mask,
            frame_mask=np.ascontiguousarray(arrays["frame_mask"]),
            track_mask=np.ascontiguousarray(arrays["track_mask"]),
            actor_commitments=_actor_commitment_rows(
                arrays["actor_commitments"], actor_mask
            ),
            group_commitments=_bytes32_rows(
                arrays["group_commitments"], "group_commitments"
            ),
        )
        text = np.ascontiguousarray(arrays["text_embeddings"])
        return RetrievalTrainingBatch(
            groups=groups,
            text_embeddings=torch.from_numpy(text),
            motion_positive_ids=_bytes32_rows(
                arrays["motion_positive_ids"], "motion_positive_ids"
            ),
            text_positive_ids=_bytes32_rows(
                arrays["text_positive_ids"], "text_positive_ids"
            ),
            text_commitments=_bytes32_rows(
                arrays["text_commitments"], "text_commitments"
            ),
            split=self.split,
        )

    def iter_epoch(self, *, epoch: int, seed: int) -> Iterable[RetrievalTrainingBatch]:
        if (
            type(epoch) is not int
            or epoch < 0
            or epoch >= 2**64
            or type(seed) is not int
            or not -(2**127) <= seed < 2**127
        ):
            raise HostConfigurationError(
                "epoch and seed must be exact nonnegative/int values"
            )
        prefix = bytes.fromhex(self.manifest_sha256) + seed.to_bytes(
            16, "big", signed=True
        )
        ordered = sorted(
            enumerate(self._entries),
            key=lambda row: hashlib.sha256(
                prefix + epoch.to_bytes(8, "big") + row[0].to_bytes(8, "big")
            ).digest(),
        )
        for _, entry in ordered:
            yield self._load_batch(entry)


@dataclass(frozen=True, slots=True)
class HostConfig:
    path: Path
    prepared_index: Path
    receipt_record: Path
    receipt_artifacts: Mapping[str, Path]
    source_tree_sha256: str
    device: str
    request_bf16: bool
    bf16_runtime_qualified: bool
    edge_budget: int
    checkpoint_every_updates: int
    max_batch_bytes: int
    resume_retry_class: str
    corrective_change_sha256: str
    residual_artifacts: ResidualArtifacts | None
    base_qualification: HostBaseQualificationConfig | None = None

    @classmethod
    def load(cls, path: str | Path) -> HostConfig:
        config_path = Path(path).resolve()
        _, value = _load_json_object(config_path, "host configuration")
        required_keys = {
            "prepared_index",
            "receipts",
            "runtime",
            "schema",
            "source_tree_sha256",
        }
        optional_keys = frozenset({"residual", "base_qualification"})
        if not required_keys.issubset(value) or not set(value).issubset(
            required_keys | optional_keys
        ):
            raise HostConfigurationError("host configuration keys are not closed")
        if value["schema"] not in {
            HOST_CONFIG_SCHEMA,
            HOST_CONFIG_QUALIFICATION_SCHEMA,
        }:
            raise HostConfigurationError("host configuration schema is not registered")
        if (value["schema"] == HOST_CONFIG_SCHEMA) != (
            "base_qualification" not in value
        ):
            raise HostConfigurationError(
                "base qualification configuration requires private host schema v2"
            )
        receipts = value["receipts"]
        runtime = value["runtime"]
        if type(receipts) is not dict or type(runtime) is not dict:
            raise HostConfigurationError("receipts and runtime must be objects")
        _closed_keys(receipts, {"artifacts", "record"}, "receipt configuration")
        _closed_keys(
            runtime,
            {
                "bf16_runtime_qualified",
                "checkpoint_every_updates",
                "corrective_change_sha256",
                "device",
                "edge_budget",
                "max_batch_bytes",
                "request_bf16",
                "resume_retry_class",
            },
            "runtime configuration",
        )
        artifacts = receipts["artifacts"]
        if type(artifacts) is not dict or set(artifacts) != set(RECEIPT_DIGEST_FIELDS):
            raise HostConfigurationError("receipt artifact keys are not closed")
        source_tree = value["source_tree_sha256"]
        corrective = runtime["corrective_change_sha256"]
        for digest, label in (
            (source_tree, "source tree"),
            (corrective, "corrective change"),
        ):
            if type(digest) is not str or len(digest) != 64 or digest.lower() != digest:
                raise HostConfigurationError(f"{label} sha256 is malformed")
        residual_value = value.get("residual")
        residual_artifacts: ResidualArtifacts | None = None
        if residual_value is not None:
            if type(residual_value) is not dict:
                raise HostConfigurationError("residual configuration must be an object")
            _closed_keys(
                residual_value,
                {"capacity_audits", "qualification"},
                "residual configuration",
            )
            capacity_values = residual_value["capacity_audits"]
            if type(capacity_values) is not dict or set(capacity_values) != {
                str(seed) for seed in SEEDS
            }:
                raise HostConfigurationError(
                    "residual capacity audits must cover the three registered seeds"
                )
            residual_artifacts = ResidualArtifacts(
                qualification=_resolve_config_path(
                    config_path.parent,
                    residual_value["qualification"],
                    "base qualification artifact",
                ),
                capacity_audits={
                    seed: _resolve_config_path(
                        config_path.parent,
                        capacity_values[str(seed)],
                        f"capacity audit seed {seed}",
                    )
                    for seed in SEEDS
                },
            )
        qualification_value = value.get("base_qualification")
        base_qualification: HostBaseQualificationConfig | None = None
        if qualification_value is not None:
            if type(qualification_value) is not dict:
                raise HostConfigurationError(
                    "base qualification configuration must be an object"
                )
            _closed_keys(
                qualification_value,
                {
                    "journal_parent",
                    "latency_protocol_sha256",
                    "launcher_sha256",
                    "output_root",
                    "registered_cuda_uuid",
                    "registry_path",
                    "registry_sha256",
                    "schema",
                    "session_id",
                    "wall_timeout_seconds",
                },
                "base qualification configuration",
            )
            if qualification_value["schema"] != BASE_QUALIFICATION_CONTROLLER_SCHEMA:
                raise HostConfigurationError(
                    "base qualification controller schema is not registered"
                )
            try:
                base_qualification = HostBaseQualificationConfig(
                    registry_path=_resolve_config_path(
                        config_path.parent,
                        qualification_value["registry_path"],
                        "base attempt registry",
                    ),
                    registry_sha256=qualification_value["registry_sha256"],
                    output_root=_resolve_config_path(
                        config_path.parent,
                        qualification_value["output_root"],
                        "base qualification output root",
                    ),
                    journal_parent=_resolve_config_path(
                        config_path.parent,
                        qualification_value["journal_parent"],
                        "base latency journal parent",
                    ),
                    session_id=qualification_value["session_id"],
                    registered_cuda_uuid=qualification_value["registered_cuda_uuid"],
                    latency_protocol_sha256=qualification_value[
                        "latency_protocol_sha256"
                    ],
                    launcher_sha256=qualification_value["launcher_sha256"],
                    wall_timeout_seconds=qualification_value[
                        "wall_timeout_seconds"
                    ],
                )
            except (HostBaseQualificationError, TypeError, ValueError) as error:
                raise HostConfigurationError(
                    "base qualification configuration is invalid"
                ) from error
        return cls(
            path=config_path,
            prepared_index=_resolve_config_path(
                config_path.parent, value["prepared_index"], "prepared index"
            ),
            receipt_record=_resolve_config_path(
                config_path.parent, receipts["record"], "receipt record"
            ),
            receipt_artifacts={
                name: _resolve_config_path(
                    config_path.parent, artifacts[name], f"receipt artifact {name}"
                )
                for name in RECEIPT_DIGEST_FIELDS
            },
            source_tree_sha256=source_tree,
            device=runtime["device"],
            request_bf16=runtime["request_bf16"],
            bf16_runtime_qualified=runtime["bf16_runtime_qualified"],
            edge_budget=runtime["edge_budget"],
            checkpoint_every_updates=runtime["checkpoint_every_updates"],
            max_batch_bytes=runtime["max_batch_bytes"],
            resume_retry_class=runtime["resume_retry_class"],
            corrective_change_sha256=corrective,
            residual_artifacts=residual_artifacts,
            base_qualification=base_qualification,
        )

    def load_receipts(self) -> PrivateReceiptAssertions:
        _, value = _load_json_object(self.receipt_record, "private receipt record")
        expected = {"admitted_at_utc", "authority", *RECEIPT_DIGEST_FIELDS}
        _closed_keys(value, expected, "private receipt record")
        for name in RECEIPT_DIGEST_FIELDS:
            if _sha256_file(self.receipt_artifacts[name]) != value[name]:
                raise HostConfigurationError(
                    f"private receipt artifact mismatch: {name}"
                )
        return PrivateReceiptAssertions(**value)

    def validate_runtime(self) -> None:
        """Validate host runtime knobs against the formal public contract."""

        TrainingConfig(
            stage="base",
            seed=1729,
            device=self.device,
            request_bf16=self.request_bf16,
            bf16_runtime_qualified=self.bf16_runtime_qualified,
            edge_budget=self.edge_budget,
            checkpoint_every_updates=self.checkpoint_every_updates,
            synthetic_contract=False,
        )
        if type(self.max_batch_bytes) is not int or self.max_batch_bytes < 1:
            raise HostConfigurationError("runtime max_batch_bytes must be positive")
        if self.resume_retry_class not in {
            "INFRA_TRANSIENT",
            "RESOURCE",
            "IMPLEMENTATION",
            "DATA",
        }:
            raise HostConfigurationError("runtime resume_retry_class is not registered")

    def validate_base_qualification_request(
        self,
        request: cli.CLICommandRequest,
    ) -> None:
        """Close the qualify-base private inputs without running a model."""

        if request.command != "qualify-base":
            return
        qualification = self.base_qualification
        if qualification is None:
            raise HostConfigurationError(
                "qualify-base requires a base qualification controller configuration"
            )
        if not _qualify_base_request_is_closed(request):
            raise HostConfigurationError(
                "qualify-base accepts only one --terminal-root input"
            )
        if request.terminal_root is None:  # narrowed by the closed-input predicate
            raise AssertionError("closed qualify-base request lost terminal_root")
        for value, label in (
            (request.terminal_root, "base terminal root"),
            (qualification.output_root, "base qualification output root"),
            (qualification.journal_parent, "base latency journal parent"),
        ):
            _reject_symlink_components(value, label)
            if not value.is_dir():
                raise HostConfigurationError(f"{label} must be an existing directory")
        if request.terminal_root.resolve() in {
            qualification.output_root.resolve(),
            qualification.journal_parent.resolve(),
        }:
            raise HostConfigurationError(
                "attempt, qualification-output, and journal roots must be distinct"
            )
        if qualification.output_root.resolve() == qualification.journal_parent.resolve():
            raise HostConfigurationError(
                "qualification output and journal roots must be distinct"
            )
        try:
            load_base_attempt_registry(
                qualification.registry_path,
                expected_sha256=qualification.registry_sha256,
            )
        except (HostBaseQualificationError, OSError, TypeError, ValueError) as error:
            raise HostConfigurationError(
                "base attempt registry manifest is invalid"
            ) from error
        output = qualification.output_root / qualification.session_id
        journal = qualification.journal_parent / qualification.session_id
        if output.exists() or output.is_symlink() or journal.exists() or journal.is_symlink():
            raise HostConfigurationError(
                "base qualification session output or journal already exists"
            )

    def load_sources(
        self,
    ) -> tuple[TrainingDataSource, TrainingDataSource | CaptureValidationSource]:
        raw, value = _load_json_object(self.prepared_index, "prepared index")
        _closed_keys(value, {"schema", "train", "val"}, "prepared index")
        schema = value["schema"]
        if type(schema) is not str or schema not in {
            PREPARED_INDEX_SCHEMA,
            PREPARED_INDEX_V2_SCHEMA,
            CAPTURE_VALIDATION_INDEX_SCHEMA,
        }:
            raise HostConfigurationError("prepared index schema is not registered")
        receipts = self.load_receipts()
        if _sha256_bytes(raw) != receipts.prepared_data_manifest_sha256:
            raise HostConfigurationError(
                "prepared index differs from authenticated receipt"
            )
        if schema == PREPARED_INDEX_V2_SCHEMA:
            try:
                return load_prepared_training_sources_v2(
                    self.prepared_index,
                    expected_index_sha256=receipts.prepared_data_manifest_sha256,
                    max_batch_bytes=self.max_batch_bytes,
                )
            except (PreparedDataV2Error, TrainingRuntimeError) as error:
                raise HostConfigurationError(
                    "prepared v2 source is invalid"
                ) from error
        if schema == CAPTURE_VALIDATION_INDEX_SCHEMA:
            _require_canonical_json_object(raw, value, "capture-validation index")
            return self._load_capture_validation_sources(value)
        sources: list[PrivatePreparedDataSource] = []
        for split in ("train", "val"):
            item = value[split]
            if type(item) is not dict:
                raise HostConfigurationError(
                    f"prepared index {split} row must be an object"
                )
            _closed_keys(item, {"path", "sha256"}, f"prepared index {split} row")
            path = _resolve_under(
                self.prepared_index.parent, item["path"], f"{split} manifest"
            )
            source = PrivatePreparedDataSource(
                path, max_batch_bytes=self.max_batch_bytes
            )
            if source.manifest_sha256 != item["sha256"]:
                raise HostConfigurationError(f"{split} manifest digest mismatch")
            if source.split != split:
                raise HostConfigurationError(
                    f"{split} manifest declares the wrong split"
                )
            sources.append(source)
        return sources[0], sources[1]

    def _load_capture_validation_sources(
        self,
        value: Mapping[str, object],
    ) -> tuple[TrainingDataSource, CaptureValidationSource]:
        train_row = value["train"]
        val_row = value["val"]
        if type(train_row) is not dict or type(val_row) is not dict:
            raise HostConfigurationError(
                "capture-validation index train and val rows must be objects"
            )
        _closed_keys(
            train_row,
            {"kind", "path", "sha256"},
            "capture-validation training row",
        )
        _closed_keys(
            val_row,
            {
                "kind",
                "path",
                "sha256",
                "source_census_sha256",
                "upstream_manifest_sha256",
            },
            "capture-validation val row",
        )
        train_kind = train_row["kind"]
        if type(train_kind) is not str or train_kind not in {
            PREPARED_TRAIN_V1_KIND,
            PREPARED_TRAIN_V2_KIND,
        }:
            raise HostConfigurationError(
                "capture-validation training kind is not registered"
            )
        if type(val_row["kind"]) is not str or val_row["kind"] != CAPTURE_VALIDATION_KIND:
            raise HostConfigurationError(
                "capture-validation val kind is not registered"
            )
        train_digest = _lower_sha256(
            train_row["sha256"],
            "capture-validation training index SHA-256",
        )
        val_digest = _lower_sha256(
            val_row["sha256"],
            "capture-validation storage manifest SHA-256",
        )
        expected_census = _lower_sha256(
            val_row["source_census_sha256"],
            "capture-validation source census SHA-256",
        )
        expected_upstream = _lower_sha256(
            val_row["upstream_manifest_sha256"],
            "capture-validation upstream manifest SHA-256",
        )
        train_path = _resolve_under(
            self.prepared_index.parent,
            train_row["path"],
            "capture-validation training index",
        )
        val_path = _resolve_under(
            self.prepared_index.parent,
            val_row["path"],
            "capture-validation storage manifest",
        )
        if train_path == val_path:
            raise HostConfigurationError(
                "capture-validation train and val artifacts must be distinct"
            )
        train = self._load_registered_training_index(
            train_path,
            expected_sha256=train_digest,
            expected_kind=train_kind,
        )
        try:
            val = load_capture_validation_source(
                val_path,
                expected_manifest_sha256=val_digest,
                limits=CaptureStorageLimits(
                    max_file_bytes=self.max_batch_bytes,
                    max_decoded_file_bytes=self.max_batch_bytes,
                ),
            )
        except CapturePreparedStorageError as error:
            raise HostConfigurationError(
                "capture-validation val storage is invalid"
            ) from error
        if type(val) is not CaptureValidationSource or val.split != "val":
            raise HostConfigurationError(
                "capture-validation loader did not return an exact val source"
            )
        if val.census_sha256 != expected_census:
            raise HostConfigurationError(
                "capture-validation source census differs from its index"
            )
        if val.manifest_sha256 != expected_upstream:
            raise HostConfigurationError(
                "capture-validation upstream manifest differs from its index"
            )
        if not isinstance(train, TrainingDataSource) or train.split != "train":
            raise HostConfigurationError(
                "capture-validation training index did not produce a train source"
            )
        self._reject_capture_training_identity_overlap(
            train,
            val,
            compare_window_identities=(train_kind == PREPARED_TRAIN_V2_KIND),
        )
        return train, val

    @staticmethod
    def _reject_capture_training_identity_overlap(
        train: TrainingDataSource,
        val: CaptureValidationSource,
        *,
        compare_window_identities: bool,
    ) -> None:
        """Read every concrete train batch and reject exact val identity reuse.

        Actor commitments identify prepared tracks, not participant identities.
        Participant-level disjointness remains the authenticated split audit's
        responsibility. Prepared-data v2, unlike v1, freezes each motion positive
        identity as the source window SHA-256, so only v2 admits the window check.
        """

        train_manifest = _lower_sha256(
            train.manifest_sha256,
            "capture-validation train manifest SHA-256",
        )
        val_census = val.census_sha256
        val_actors = {
            actor
            for capture in val.captures
            for window in capture.windows
            for actor in window.groups.actor_commitments[0]
            if actor is not None
        }
        val_captions = {
            commitment
            for capture in val.captures
            for commitment in capture.holistic_text.caption_commitments
        }
        val_windows = {
            commitment
            for capture in val.captures
            for commitment in capture.plan.window_commitments
        }
        batch_count = 0
        try:
            # Registered v1/v2 sources expose their complete immutable artifact
            # census through iter_epoch. Seed/epoch zero only fixes deterministic
            # order and v2 yaw; neither path consumes a global RNG.
            for batch in train.iter_epoch(epoch=0, seed=0):
                if type(batch) is not RetrievalTrainingBatch or batch.split != "train":
                    raise HostConfigurationError(
                        "capture-validation train identity scan returned an invalid batch"
                    )
                batch_count += 1
                train_actors = {
                    actor
                    for row in batch.groups.actor_commitments
                    for actor in row
                    if actor is not None
                }
                if train_actors & val_actors:
                    raise HostConfigurationError(
                        "capture-validation train and val reuse an actor/track identity"
                    )
                if set(batch.text_commitments) & val_captions:
                    raise HostConfigurationError(
                        "capture-validation train and val reuse a caption identity"
                    )
                if (
                    compare_window_identities
                    and set(batch.motion_positive_ids) & val_windows
                ):
                    raise HostConfigurationError(
                        "capture-validation train and val reuse a window identity"
                    )
        except HostConfigurationError:
            raise
        except Exception as error:
            raise HostConfigurationError(
                "capture-validation train identity scan failed"
            ) from error
        if batch_count == 0:
            raise HostConfigurationError(
                "capture-validation train identity scan returned no batches"
            )
        if train.manifest_sha256 != train_manifest:
            raise HostConfigurationError(
                "capture-validation train source changed during identity scan"
            )
        if val.census_sha256 != val_census:
            raise HostConfigurationError(
                "capture-validation val source changed during identity scan"
            )

    def _load_registered_training_index(
        self,
        path: Path,
        *,
        expected_sha256: str,
        expected_kind: str,
    ) -> TrainingDataSource:
        raw, value = _load_json_object(path, "referenced training index")
        if _sha256_bytes(raw) != expected_sha256:
            raise HostConfigurationError(
                "referenced training index differs from its exact digest"
            )
        _require_canonical_json_object(raw, value, "referenced training index")
        _closed_keys(
            value,
            {"schema", "train", "val"},
            "referenced training index",
        )
        expected_schema = {
            PREPARED_TRAIN_V1_KIND: PREPARED_INDEX_SCHEMA,
            PREPARED_TRAIN_V2_KIND: PREPARED_INDEX_V2_SCHEMA,
        }[expected_kind]
        if value["schema"] != expected_schema:
            raise HostConfigurationError(
                "referenced training index schema differs from its kind"
            )
        if expected_schema == PREPARED_INDEX_V2_SCHEMA:
            try:
                train, _auxiliary_window_val = load_prepared_training_sources_v2(
                    path,
                    expected_index_sha256=expected_sha256,
                    max_batch_bytes=self.max_batch_bytes,
                )
            except (PreparedDataV2Error, TrainingRuntimeError) as error:
                raise HostConfigurationError(
                    "referenced prepared v2 training source is invalid"
                ) from error
            return train
        sources: list[PrivatePreparedDataSource] = []
        for split in ("train", "val"):
            row = value[split]
            if type(row) is not dict:
                raise HostConfigurationError(
                    f"referenced prepared v1 {split} row must be an object"
                )
            _closed_keys(
                row,
                {"path", "sha256"},
                f"referenced prepared v1 {split} row",
            )
            split_manifest = _resolve_under(
                path.parent,
                row["path"],
                f"referenced prepared v1 {split} manifest",
            )
            source = PrivatePreparedDataSource(
                split_manifest,
                max_batch_bytes=self.max_batch_bytes,
            )
            if source.manifest_sha256 != row["sha256"] or source.split != split:
                raise HostConfigurationError(
                    f"referenced prepared v1 {split} source differs from its index"
                )
            sources.append(source)
        return sources[0]


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_once_json(path: Path, value: object) -> Path:
    raw = _canonical_json_bytes(_jsonable(value))
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if os.name == "posix":
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        directory_descriptor = os.open(path.parent, flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    return path


def _copy_verified_checkpoint(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
) -> tuple[Path, bytes]:
    """Materialize already-verified checkpoint bytes under a new attempt."""

    _reject_symlink_components(source, "resume checkpoint source")
    _reject_symlink_components(destination.parent, "resume checkpoint destination")
    if (
        not source.is_file()
        or not destination.parent.is_dir()
        or destination.exists()
        or destination.is_symlink()
        or destination.name != source.name
    ):
        raise HostConfigurationError("resume checkpoint materialization path is invalid")
    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    destination_flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        destination_flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        destination_flags |= os.O_NOFOLLOW
    source_descriptor: int | None = None
    destination_descriptor: int | None = None
    digest = hashlib.sha256()
    owned_raw = b""
    try:
        source_descriptor = os.open(source, flags)
        source_metadata = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_metadata.st_mode):
            raise HostConfigurationError(
                "resume checkpoint source descriptor is not a regular file"
            )
        destination_descriptor = os.open(destination, destination_flags, 0o600)
        destination_metadata = os.fstat(destination_descriptor)
        if not stat.S_ISREG(destination_metadata.st_mode):
            raise HostConfigurationError(
                "resume checkpoint destination descriptor is not a regular file"
            )
        with os.fdopen(source_descriptor, "rb", closefd=False) as reader:
            with os.fdopen(destination_descriptor, "wb", closefd=False) as writer:
                while chunk := reader.read(1024 * 1024):
                    digest.update(chunk)
                    writer.write(chunk)
                writer.flush()
                os.fsync(writer.fileno())
        if digest.hexdigest() != expected_sha256:
            raise HostConfigurationError(
                "resume checkpoint materialization digest mismatch"
            )
        os.lseek(destination_descriptor, 0, os.SEEK_SET)
        owned_chunks: list[bytes] = []
        owned_digest = hashlib.sha256()
        while chunk := os.read(destination_descriptor, 1024 * 1024):
            owned_chunks.append(chunk)
            owned_digest.update(chunk)
        if owned_digest.hexdigest() != expected_sha256:
            raise HostConfigurationError(
                "resume checkpoint materialized bytes failed verification"
            )
        owned_raw = b"".join(owned_chunks)
        if os.name == "posix":
            directory_flags = os.O_RDONLY
            if hasattr(os, "O_DIRECTORY"):
                directory_flags |= os.O_DIRECTORY
            directory_descriptor = os.open(destination.parent, directory_flags)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    except Exception as error:
        if source_descriptor is not None:
            with suppress(OSError):
                os.close(source_descriptor)
            source_descriptor = None
        if destination_descriptor is not None:
            with suppress(OSError):
                os.close(destination_descriptor)
            destination_descriptor = None
        if isinstance(error, HostConfigurationError):
            raise
        raise HostConfigurationError(
            "resume checkpoint materialization failed"
        ) from error
    except BaseException:
        if source_descriptor is not None:
            with suppress(OSError):
                os.close(source_descriptor)
            source_descriptor = None
        if destination_descriptor is not None:
            with suppress(OSError):
                os.close(destination_descriptor)
            destination_descriptor = None
        raise
    finally:
        if source_descriptor is not None:
            os.close(source_descriptor)
        if destination_descriptor is not None:
            os.close(destination_descriptor)
    return destination, owned_raw


def _materialize_resume_checkpoints(
    checkpoint: Path,
    checkpoint_receipt: CheckpointRecord,
    destination_directory: Path,
) -> Path:
    """Copy only ledger-bound latest/best checkpoint bytes into a new attempt."""

    _reject_symlink_components(
        destination_directory, "resume checkpoint destination directory"
    )
    destination_directory.mkdir(mode=0o700, exist_ok=False)
    latest, latest_raw = _copy_verified_checkpoint(
        checkpoint,
        destination_directory / checkpoint.name,
        expected_sha256=checkpoint_receipt.checkpoint_payload_sha256,
    )
    payload = _decode_runtime_checkpoint_bytes(latest_raw)
    best_name = payload.get("best_checkpoint_name")
    best_sha256 = payload.get("best_checkpoint_sha256")
    if best_name is not None and (
        type(best_name) is not str
        or Path(best_name).name != best_name
        or not best_name.startswith("checkpoint-")
        or not best_name.endswith(".pt")
    ):
        raise HostConfigurationError("resume checkpoint best-checkpoint name is invalid")
    if best_sha256 is not None and (
        type(best_sha256) is not str
        or len(best_sha256) != 64
        or any(character not in "0123456789abcdef" for character in best_sha256)
    ):
        raise HostConfigurationError("resume checkpoint best-checkpoint digest is invalid")
    if best_name is None and best_sha256 is not None:
        raise HostConfigurationError("resume checkpoint best digest has no name")
    if best_name is None or best_name == checkpoint.name:
        if best_sha256 not in (None, checkpoint_receipt.checkpoint_payload_sha256):
            raise HostConfigurationError(
                "resume checkpoint self-selected best digest is inconsistent"
            )
        return latest
    if best_sha256 is None:
        raise HostConfigurationError("resume checkpoint prior best lacks a digest")
    best_source = checkpoint.parent / best_name
    _copy_verified_checkpoint(
        best_source,
        destination_directory / best_name,
        expected_sha256=best_sha256,
    )
    return latest


def _torch_value_sha256(value: object) -> str:
    stream = io.BytesIO()
    torch.save(value, stream)
    return _sha256_bytes(stream.getvalue())


def _decode_runtime_checkpoint_bytes(raw: bytes) -> dict[str, object]:
    try:
        payload = torch.load(
            io.BytesIO(raw),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise HostConfigurationError(
            "training checkpoint is truncated or unreadable"
        ) from error
    if type(payload) is not dict:
        raise HostConfigurationError("training checkpoint payload is not an object")
    return payload


def _load_runtime_checkpoint(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise HostConfigurationError(
            "training checkpoint must be a regular non-symlink file"
        )
    raw = path.read_bytes()
    if (
        expected_sha256 is not None
        and _sha256_bytes(raw) != expected_sha256
    ):
        raise HostConfigurationError("training checkpoint artifact digest mismatch")
    return _decode_runtime_checkpoint_bytes(raw)


def _checkpoint_record(
    checkpoint_path: Path,
    checkpoint_sha256: str,
    *,
    attempt_id: str,
    run_id: str,
    written_at_utc: str,
    parent_checkpoint_receipt_sha256: str | None,
) -> CheckpointRecord:
    payload = _load_runtime_checkpoint(checkpoint_path)
    required = {
        "model",
        "optimizer",
        "rng",
        "epoch",
        "update_index",
        "global_step",
        "train_manifest_sha256",
        "val_manifest_sha256",
        "best_validation_metric",
        "best_checkpoint_name",
        "best_checkpoint_sha256",
    }
    if not required.issubset(payload):
        raise HostConfigurationError("training checkpoint lacks attempt-ledger state")
    cursor = {
        "epoch": payload["epoch"],
        "global_step": payload["global_step"],
        "update_index": payload["update_index"],
    }
    data_state = {
        "train_manifest_sha256": payload["train_manifest_sha256"],
        "val_manifest_sha256": payload["val_manifest_sha256"],
    }
    validation = {
        "best_validation_metric": payload["best_validation_metric"],
        "best_checkpoint_name": payload["best_checkpoint_name"],
        "best_checkpoint_sha256": payload["best_checkpoint_sha256"],
    }
    rng_digest = _torch_value_sha256(payload["rng"])
    return CheckpointRecord(
        attempt_id=attempt_id,
        run_id=run_id,
        epoch_index=int(payload["epoch"]),
        global_step=int(payload["global_step"]),
        written_at_utc=written_at_utc,
        model_state_sha256=_torch_value_sha256(payload["model"]),
        optimizer_state_sha256=_torch_value_sha256(payload["optimizer"]),
        rng_state_sha256=rng_digest,
        sampler_state_sha256=_sha256_bytes(_canonical_json_bytes(cursor)),
        dataloader_state_sha256=_sha256_bytes(_canonical_json_bytes(data_state)),
        dropout_state_sha256=rng_digest,
        validation_state_sha256=_sha256_bytes(_canonical_json_bytes(validation)),
        checkpoint_payload_sha256=checkpoint_sha256,
        parent_checkpoint_receipt_sha256=parent_checkpoint_receipt_sha256,
    )


class _LiveAttemptLedger:
    """Write live heartbeats and checkpoint receipts during a blocking fit."""

    HEARTBEAT_SECONDS = 30.0

    def __init__(self, store: AttemptStore, attempt: AttemptRecord) -> None:
        self.store = store
        self.attempt = attempt
        self.latest_heartbeat_sha256: str | None = None
        self.latest_checkpoint_receipt_sha256: str | None = None
        self.latest_checkpoint_payload_sha256: str | None = None
        self.latest_global_step = 0
        self._heartbeat_sequence = 0
        self._error: BaseException | None = None
        self._started = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"phaseset-heartbeat-{attempt.attempt_id}",
            daemon=True,
        )

    def start(self) -> None:
        with self._lock:
            self._write_heartbeat("STARTING")
        self._thread.start()
        self._started = True

    def _write_heartbeat(self, phase: str) -> None:
        self.latest_heartbeat_sha256 = self.store.write_heartbeat(
            HeartbeatRecord(
                attempt_id=self.attempt.attempt_id,
                run_id=self.attempt.run_id,
                sequence=self._heartbeat_sequence,
                global_step=self.latest_global_step,
                observed_at_utc=_utc_now(),
                phase=phase,
                previous_heartbeat_sha256=self.latest_heartbeat_sha256,
            )
        )
        self._heartbeat_sequence += 1

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.HEARTBEAT_SECONDS):
            try:
                with self._lock:
                    self._write_heartbeat("RUNNING")
            except BaseException as error:
                self._error = error
                self._stop.set()
                return

    def observe_checkpoint(self, artifact: CheckpointArtifact, reason: str) -> None:
        """Training callback invoked only after the atomic checkpoint commit."""

        if type(artifact) is not CheckpointArtifact or reason not in {
            "update",
            "validation",
        }:
            raise HostConfigurationError(
                "live checkpoint callback payload is malformed"
            )
        payload = self._validated_checkpoint_payload(artifact)
        # The runtime writes an update checkpoint at the last update and then a
        # validation checkpoint at the same global step. AttemptStore requires
        # strictly increasing checkpoint steps, so wait for the validation file.
        if reason == "update" and payload.get("update_index") == payload.get(
            "updates_per_epoch"
        ):
            return
        self._record_checkpoint(artifact)

    def ensure_checkpoint(self, artifact: CheckpointArtifact) -> None:
        """Receipt the report checkpoint even when a restored fit made no progress."""

        self._validated_checkpoint_payload(artifact)
        with self._lock:
            if (
                self.latest_checkpoint_payload_sha256 == artifact.sha256
                and self.latest_global_step == artifact.global_step
            ):
                return
        self._record_checkpoint(artifact)

    @staticmethod
    def _validated_checkpoint_payload(
        artifact: CheckpointArtifact,
    ) -> Mapping[str, object]:
        if type(artifact) is not CheckpointArtifact:
            raise HostConfigurationError("runtime checkpoint artifact is malformed")
        payload = _load_runtime_checkpoint(artifact.path)
        if _sha256_file(artifact.path) != artifact.sha256:
            raise HostConfigurationError("live checkpoint changed after atomic write")
        return payload

    def _record_checkpoint(self, artifact: CheckpointArtifact) -> None:
        with self._lock:
            if artifact.global_step <= self.latest_global_step and (
                self.latest_checkpoint_receipt_sha256 is not None
            ):
                raise HostConfigurationError(
                    "live checkpoint callback did not advance global_step"
                )
            receipt = _checkpoint_record(
                artifact.path,
                artifact.sha256,
                attempt_id=self.attempt.attempt_id,
                run_id=self.attempt.run_id,
                written_at_utc=_utc_now(),
                parent_checkpoint_receipt_sha256=(
                    self.latest_checkpoint_receipt_sha256
                ),
            )
            self.latest_checkpoint_receipt_sha256 = self.store.write_checkpoint(receipt)
            self.latest_checkpoint_payload_sha256 = artifact.sha256
            self.latest_global_step = artifact.global_step
            self._write_heartbeat("CHECKPOINTING")

    def stop(self) -> None:
        self._stop.set()
        if self._started:
            self._thread.join(timeout=self.HEARTBEAT_SECONDS + 5.0)
        if self._thread.is_alive():
            raise HostConfigurationError("heartbeat thread did not stop")
        if self._error is not None:
            raise HostConfigurationError(
                "live heartbeat writer failed"
            ) from self._error


class PhaseSetHostBackend:
    """Request-bound backend for real base/residual training and resume."""

    def __init__(self, config: HostConfig, request: cli.CLICommandRequest) -> None:
        self._config = config
        self._request = request
        self._admission_request: RuntimeAdmissionRequest | None = None
        self._authenticated_adapter_sha256: str | None = None
        self._qualification_adapter: ProductionRuntimeAdapter | None = None
        self._qualification_authentication: (
            HostBaseQualificationAuthentication | None
        ) = None
        self._qualification_intent: CommandIntent | None = None
        self.backend_sha256 = _sha256_bytes(Path(__file__).read_bytes())

    def authenticate(
        self,
        request: RuntimeAdmissionRequest,
        receipts: PrivateReceiptAssertions,
        adapter_sha256: str,
    ) -> bool:
        if receipts != self._config.load_receipts():
            return False
        plan = build_experiment_plan()
        if (
            request.plan_sha256 != experiment_plan_sha256(plan)
            or request.matrix_sha256 != plan.matrix_sha256
            or request.training_config_sha256
            != artifact_sha256(canonical_public_training_config_bytes())
            or request.commands != ProductionRuntimeAdapter.commands
            or request.handler_manifest_sha256
            != ProductionRuntimeAdapter.handler_manifest_sha256
        ):
            return False
        # RuntimeAdmissionRequest deliberately has no command field.  The backend
        # is request-specific and execute() binds the later CommandIntent to that
        # parsed CLI command.  Retain the exact runner-created admission request so
        # attempt ledgers never read fields that CommandIntent does not define.
        if self._admission_request is not None:
            return (
                request == self._admission_request
                and adapter_sha256 == self._authenticated_adapter_sha256
            )
        self._admission_request = request
        self._authenticated_adapter_sha256 = _lower_sha256(
            adapter_sha256,
            "production adapter SHA-256",
        )
        return True

    def bind_qualification_adapter(self, adapter: ProductionRuntimeAdapter) -> None:
        """Bind the one adapter whose private slot receives actual derived evidence."""

        if self._request.command != "qualify-base":
            raise HostConfigurationError(
                "base qualification adapter binding is command-specific"
            )
        if type(adapter) is not ProductionRuntimeAdapter:
            raise TypeError("qualification adapter must be exact ProductionRuntimeAdapter")
        if getattr(adapter, "_backend", None) is not self:
            raise HostConfigurationError("qualification adapter belongs to another backend")
        if self._qualification_adapter is not None:
            raise HostConfigurationError("qualification adapter was already bound")
        self._qualification_adapter = adapter

    def authenticate_base_qualification(
        self,
        intent: CommandIntent,
        authorization: PrivateBaseQualificationAuthorization,
        qualification_sha256: str,
        adapter_sha256: str,
    ) -> bool:
        """Match production recomputation to the controller's retained real bytes."""

        expected = self._qualification_authentication
        return (
            intent.command == "qualify-base"
            and intent == self._qualification_intent
            and type(expected) is HostBaseQualificationAuthentication
            and expected.verifies(
                authorization,
                qualification_sha256=qualification_sha256,
                adapter_sha256=adapter_sha256,
            )
        )

    def _attempt_plan_bindings(self) -> tuple[str, str, str]:
        request = self._admission_request
        if request is None:
            raise HostConfigurationError(
                "runtime admission must complete before an attempt is created"
            )
        return (
            request.plan_sha256,
            request.matrix_sha256,
            request.training_config_sha256,
        )

    def execute(self, intent: CommandIntent) -> BackendExecution:
        if intent.command != self._request.command:
            raise HostConfigurationError(
                "command intent differs from the request-specific host backend"
            )
        if intent.command == "run-base":
            return self._run_base(intent)
        if intent.command == "run-residual":
            return self._run_residual(intent)
        if intent.command == "resume":
            return self._resume_base(intent)
        if intent.command == "qualify-base":
            return self._qualify_base(intent)
        raise HostConfigurationError(
            f"host harness does not implement command {intent.command}; no result was produced"
        )

    def _training_config(self, *, stage: str, seed: int) -> TrainingConfig:
        return TrainingConfig(
            stage=stage,
            seed=seed,
            device=self._config.device,
            request_bf16=self._config.request_bf16,
            bf16_runtime_qualified=self._config.bf16_runtime_qualified,
            edge_budget=self._config.edge_budget,
            checkpoint_every_updates=self._config.checkpoint_every_updates,
            synthetic_contract=False,
        )

    def _qualify_base(self, intent: CommandIntent) -> BackendExecution:
        request = self._request
        qualification = self._config.base_qualification
        adapter = self._qualification_adapter
        admission_request = self._admission_request
        if (
            qualification is None
            or adapter is None
            or admission_request is None
            or self._authenticated_adapter_sha256 is None
        ):
            raise HostConfigurationError(
                "qualify-base controller is not fully admitted and bound"
            )
        if (
            not _qualify_base_request_is_closed(request)
            or intent.run_id is not None
            or intent.seed is not None
            or intent.split is not None
        ):
            raise HostConfigurationError(
                "qualify-base accepts only the registered terminal root"
            )
        if request.terminal_root is None:  # narrowed by the closed-input predicate
            raise AssertionError("closed qualify-base request lost terminal_root")
        try:
            train, val = self._config.load_sources()
            if type(val) is not CaptureValidationSource or val.split != "val":
                raise HostConfigurationError(
                    "qualify-base requires the complete holistic capture val source"
                )
            cohort_admission = BaseCohortAdmission(
                plan_sha256=admission_request.plan_sha256,
                matrix_sha256=admission_request.matrix_sha256,
                training_config_sha256=admission_request.training_config_sha256,
                source_tree_sha256=self._config.source_tree_sha256,
                train_manifest_sha256=train.manifest_sha256,
                val_manifest_sha256=val.census_sha256,
                device=self._config.device,
                request_bf16=self._config.request_bf16,
                bf16_runtime_qualified=self._config.bf16_runtime_qualified,
                edge_budget=self._config.edge_budget,
                checkpoint_every_updates=self._config.checkpoint_every_updates,
            )
        except (
            AttributeError,
            HostConfigurationError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            raise _qualify_base_runtime_hold(prepared_data=True) from error
        output_directory = qualification.output_root / qualification.session_id
        try:
            output_directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        except OSError as error:
            raise _qualify_base_runtime_hold(prepared_data=False) from error
        try:
            lease = _AttemptLease.acquire(output_directory)
        except (HostConfigurationError, OSError, RuntimeError) as error:
            raise _qualify_base_runtime_hold(prepared_data=False) from error
        primary_error: BaseException | None = None
        try:
            result = run_host_base_qualification(
                request.terminal_root.resolve(),
                config=qualification,
                admission=cohort_admission,
                validation_source=val,
                output_directory=output_directory,
                adapter_sha256=self._authenticated_adapter_sha256,
            )
            if result.execution.outcome == "COMPLETED":
                authentication = result.authentication
                if type(authentication) is not HostBaseQualificationAuthentication:
                    raise HostConfigurationError(
                        "completed qualification lost controller authentication"
                    )
                if getattr(adapter, "_base_qualification_authorization", None) is not None:
                    raise HostConfigurationError(
                        "production adapter already contains qualification authorization"
                    )
                self._qualification_authentication = authentication
                self._qualification_intent = intent
                # ProductionRuntimeAdapter intentionally validates this exact
                # object after independently rebuilding the semantic artifact.
                # The private host derives it here from actual controller output;
                # no CLI/user digest is accepted as authorization.
                object.__setattr__(
                    adapter,
                    "_base_qualification_authorization",
                    authentication.authorization,
                )
            return result.execution
        except BaseException as error:
            primary_error = error
            raise
        finally:
            try:
                lease.release()
            except Exception as error:
                if primary_error is None:
                    raise _qualify_base_runtime_hold(prepared_data=False) from error

    def _execute_base_runtime(
        self,
        *,
        run_id: str,
        system_id: str,
        seed: int,
        checkpoint_directory: Path,
        live_ledger: _LiveAttemptLedger,
        resume_checkpoint: Path | None = None,
        resume_attempt_root: Path | None = None,
        resume_record: ResumeRecord | None = None,
        stop_after_global_step: int | None = None,
    ) -> TrainingReport:
        train, val = self._config.load_sources()
        config = self._training_config(stage="base", seed=seed)
        system, binding = construct_registered_base_seed_bound_system(system_id, config)
        runtime = PhaseSetTrainingRuntime(
            system,
            config,
            checkpoint_directory,
            initialization_binding=binding,
            checkpoint_observer=live_ledger.observe_checkpoint,
        )
        return runtime.fit(
            train,
            val,
            resume_checkpoint=resume_checkpoint,
            resume_attempt_root=resume_attempt_root,
            resume_record=resume_record,
            stop_after_global_step=stop_after_global_step,
        )

    def _execute_residual_runtime(
        self,
        *,
        system_id: str,
        seed: int,
        checkpoint_directory: Path,
        live_ledger: _LiveAttemptLedger,
        base_checkpoint: Path,
        periodic_cache: Path,
        stop_after_global_step: int | None,
        resume_checkpoint: Path | None = None,
        resume_attempt_root: Path | None = None,
        resume_record: ResumeRecord | None = None,
    ) -> TrainingReport:
        artifacts = self._config.residual_artifacts
        if artifacts is None:
            raise HostConfigurationError(
                "run-residual requires qualification and seed capacity-audit artifacts"
            )
        if (
            artifacts.qualification.is_symlink()
            or not artifacts.qualification.is_file()
        ):
            raise HostConfigurationError("base qualification artifact is absent")
        audit_path = artifacts.capacity_audits[seed]
        if audit_path.is_symlink() or not audit_path.is_file():
            raise HostConfigurationError(
                f"residual capacity-audit artifact is absent for seed {seed}"
            )
        qualification, qualification_sha256 = _load_base_qualification(
            artifacts.qualification
        )
        qualified_base = load_qualified_frozen_base(
            base_checkpoint,
            qualification=qualification,
            seed=seed,
            expected_qualification_sha256=qualification_sha256,
        )
        energy_floors = _load_periodic_cache(
            periodic_cache,
            seed=seed,
            qualification=qualification,
            qualification_sha256=qualification_sha256,
        )
        capacity_audit, capacity_sha256 = _load_capacity_audit(audit_path)
        train, val = self._config.load_sources()
        config = self._training_config(stage="residual", seed=seed)
        system, binding = construct_registered_residual_seed_bound_system(
            system_id,
            qualified_base,
            config,
            energy_floors=energy_floors,
            residual_capacity_audit=capacity_audit,
            expected_qualification_sha256=qualification_sha256,
            expected_capacity_audit_sha256=capacity_sha256,
        )
        runtime = PhaseSetTrainingRuntime(
            system,
            config,
            checkpoint_directory,
            initialization_binding=binding,
            checkpoint_observer=live_ledger.observe_checkpoint,
        )
        return runtime.fit(
            train,
            val,
            resume_checkpoint=resume_checkpoint,
            resume_attempt_root=resume_attempt_root,
            resume_record=resume_record,
            stop_after_global_step=stop_after_global_step,
        )

    def _run_base(self, intent: CommandIntent) -> BackendExecution:
        request = self._request
        if request.run_id is None or request.checkpoint_directory is None:
            raise HostConfigurationError(
                "run-base requires run_id and --checkpoint-dir"
            )
        role, seed, system_id = parse_run_id(request.run_id)
        if role != "BASE_QUALIFICATION" or intent.run_id != request.run_id:
            raise HostConfigurationError("run-base identity mismatch")
        _reject_symlink_components(request.checkpoint_directory, "checkpoint directory")
        checkpoint_directory = request.checkpoint_directory.resolve()
        if checkpoint_directory.name != "model-checkpoints":
            raise HostConfigurationError(
                "checkpoint directory must end in model-checkpoints"
            )
        attempt_directory = checkpoint_directory.parent
        if attempt_directory.exists() or attempt_directory.is_symlink():
            raise HostConfigurationError("new attempt directory already exists")
        plan_sha256, matrix_sha256, training_config_sha256 = (
            self._attempt_plan_bindings()
        )
        created_at = _utc_now()
        attempt = AttemptRecord(
            attempt_id=attempt_directory.name,
            run_id=request.run_id,
            created_at_utc=created_at,
            plan_sha256=plan_sha256,
            matrix_sha256=matrix_sha256,
            training_config_sha256=training_config_sha256,
            source_tree_sha256=self._config.source_tree_sha256,
        )
        store = AttemptStore.create(attempt_directory.parent, attempt)
        lease = _AttemptLease.acquire(store.path)
        try:
            live_ledger = _LiveAttemptLedger(store, attempt)
            try:
                live_ledger.start()
                report = self._execute_base_runtime(
                    run_id=request.run_id,
                    system_id=system_id,
                    seed=seed,
                    checkpoint_directory=checkpoint_directory,
                    live_ledger=live_ledger,
                    stop_after_global_step=request.stop_after_global_step,
                )
                live_ledger.ensure_checkpoint(report.latest_checkpoint)
                live_ledger.stop()
                return self._close_attempt(store, attempt, report, live_ledger)
            except Exception as error:
                with suppress(Exception):
                    live_ledger.stop()
                return self._persist_failed_execution(
                    store, attempt, live_ledger, error
                )
            except BaseException as error:
                with suppress(Exception):
                    live_ledger.stop()
                with suppress(Exception):
                    self._persist_failed_execution(
                        store, attempt, live_ledger, error
                    )
                raise
        finally:
            lease.release()

    def _run_residual(self, intent: CommandIntent) -> BackendExecution:
        request = self._request
        if (
            request.run_id is None
            or request.checkpoint_directory is None
            or request.base_checkpoint is None
            or request.periodic_cache is None
        ):
            raise HostConfigurationError(
                "run-residual requires run_id, checkpoint-dir, base-checkpoint, "
                "and periodic-cache"
            )
        role, seed, system_id = parse_run_id(request.run_id)
        if role != "RESIDUAL_TRAIN" or intent.run_id != request.run_id:
            raise HostConfigurationError("run-residual identity mismatch")
        _reject_symlink_components(request.checkpoint_directory, "checkpoint directory")
        checkpoint_directory = request.checkpoint_directory.resolve()
        if checkpoint_directory.name != "model-checkpoints":
            raise HostConfigurationError(
                "checkpoint directory must end in model-checkpoints"
            )
        attempt_directory = checkpoint_directory.parent
        if attempt_directory.exists() or attempt_directory.is_symlink():
            raise HostConfigurationError("new attempt directory already exists")
        plan_sha256, matrix_sha256, training_config_sha256 = (
            self._attempt_plan_bindings()
        )
        _reject_symlink_components(request.base_checkpoint, "qualified base checkpoint")
        base_checkpoint = request.base_checkpoint.resolve()
        if not base_checkpoint.is_file():
            raise HostConfigurationError("qualified base checkpoint is absent")
        _reject_symlink_components(request.periodic_cache, "periodic cache")
        periodic_cache = request.periodic_cache.resolve()
        created_at = _utc_now()
        attempt = AttemptRecord(
            attempt_id=attempt_directory.name,
            run_id=request.run_id,
            created_at_utc=created_at,
            plan_sha256=plan_sha256,
            matrix_sha256=matrix_sha256,
            training_config_sha256=training_config_sha256,
            source_tree_sha256=self._config.source_tree_sha256,
        )
        store = AttemptStore.create(attempt_directory.parent, attempt)
        lease = _AttemptLease.acquire(store.path)
        try:
            live_ledger = _LiveAttemptLedger(store, attempt)
            try:
                live_ledger.start()
                report = self._execute_residual_runtime(
                    system_id=system_id,
                    seed=seed,
                    checkpoint_directory=checkpoint_directory,
                    live_ledger=live_ledger,
                    base_checkpoint=base_checkpoint,
                    periodic_cache=periodic_cache,
                    stop_after_global_step=request.stop_after_global_step,
                )
                live_ledger.ensure_checkpoint(report.latest_checkpoint)
                live_ledger.stop()
                return self._close_attempt(store, attempt, report, live_ledger)
            except Exception as error:
                with suppress(Exception):
                    live_ledger.stop()
                return self._persist_failed_execution(
                    store, attempt, live_ledger, error
                )
            except BaseException as error:
                with suppress(Exception):
                    live_ledger.stop()
                with suppress(Exception):
                    self._persist_failed_execution(store, attempt, live_ledger, error)
                raise
        finally:
            lease.release()

    def _resume_base(self, intent: CommandIntent) -> BackendExecution:
        request = self._request
        if request.attempt_directory is None or request.resume_checkpoint is None:
            raise HostConfigurationError(
                "resume requires --attempt-dir and --checkpoint"
            )
        if request.stop_after_global_step is not None:
            raise HostConfigurationError("resume does not accept a stop parameter")
        _reject_symlink_components(
            request.attempt_directory, "resume attempt directory"
        )
        _reject_symlink_components(request.resume_checkpoint, "resume checkpoint")
        new_directory = request.attempt_directory.resolve()
        checkpoint = request.resume_checkpoint.resolve()
        if not checkpoint.is_file():
            raise HostConfigurationError(
                "resume checkpoint must be a regular existing file"
            )
        if checkpoint.parent.name != "model-checkpoints":
            raise HostConfigurationError(
                "resume checkpoint is outside model-checkpoints"
            )
        predecessor_directory = checkpoint.parent.parent
        if predecessor_directory.parent != new_directory.parent:
            raise HostConfigurationError("resume attempts must share one attempt root")
        self._recover_crashed_attempt(predecessor_directory)
        predecessor_attempt = parse_attempt_bytes(
            (predecessor_directory / "attempt.json").read_bytes()
        )
        predecessor_terminal_raw = (
            predecessor_directory / "terminal.json"
        ).read_bytes()
        predecessor_terminal = parse_terminal_bytes(predecessor_terminal_raw)
        checkpoint_receipt_path = sorted(
            (predecessor_directory / "checkpoints").glob("checkpoint-*.json")
        )[-1]
        checkpoint_receipt_raw = checkpoint_receipt_path.read_bytes()
        checkpoint_receipt = parse_checkpoint_bytes(checkpoint_receipt_raw)
        if checkpoint_receipt.checkpoint_payload_sha256 != _sha256_file(checkpoint):
            raise HostConfigurationError(
                "resume checkpoint differs from predecessor receipt"
            )
        role, seed, system_id = parse_run_id(predecessor_attempt.run_id)
        base_checkpoint: Path | None = None
        periodic_cache: Path | None = None
        if role == "BASE_QUALIFICATION":
            if request.base_checkpoint is not None or request.periodic_cache is not None:
                raise HostConfigurationError(
                    "base resume rejects base-checkpoint and periodic-cache"
                )
        elif role == "RESIDUAL_TRAIN":
            if request.base_checkpoint is None or request.periodic_cache is None:
                raise HostConfigurationError(
                    "residual resume requires --base-checkpoint and --periodic-cache"
                )
            _reject_symlink_components(
                request.base_checkpoint, "qualified base checkpoint"
            )
            base_checkpoint = request.base_checkpoint.resolve()
            if not base_checkpoint.is_file():
                raise HostConfigurationError(
                    "qualified base checkpoint must be a regular existing file"
                )
            _reject_symlink_components(request.periodic_cache, "periodic cache")
            periodic_cache = request.periodic_cache.resolve()
            if not periodic_cache.is_dir():
                raise HostConfigurationError(
                    "periodic cache must be a regular existing directory"
                )
        else:
            raise HostConfigurationError(
                "host resume supports only base and residual attempts"
            )
        if request.system_id is not None and request.system_id != system_id:
            raise HostConfigurationError("resume system differs from predecessor run")
        created_at = _utc_now()
        resumed_attempt = AttemptRecord(
            attempt_id=new_directory.name,
            run_id=predecessor_attempt.run_id,
            created_at_utc=created_at,
            plan_sha256=predecessor_attempt.plan_sha256,
            matrix_sha256=predecessor_attempt.matrix_sha256,
            training_config_sha256=predecessor_attempt.training_config_sha256,
            source_tree_sha256=predecessor_attempt.source_tree_sha256,
        )
        resume = ResumeRecord(
            new_attempt_id=resumed_attempt.attempt_id,
            predecessor_attempt_id=predecessor_attempt.attempt_id,
            run_id=predecessor_attempt.run_id,
            created_at_utc=created_at,
            predecessor_terminal_sha256=artifact_sha256(predecessor_terminal_raw),
            checkpoint_receipt_sha256=artifact_sha256(checkpoint_receipt_raw),
            corrective_change_sha256=self._config.corrective_change_sha256,
            retry_class=self._config.resume_retry_class,
        )
        if (
            predecessor_terminal.latest_checkpoint_receipt_sha256
            != resume.checkpoint_receipt_sha256
        ):
            raise HostConfigurationError(
                "resume does not select the predecessor terminal checkpoint"
            )
        store = AttemptStore.create_resumed(
            new_directory.parent, resumed_attempt, resume
        )
        lease = _AttemptLease.acquire(store.path)
        try:
            live_ledger = _LiveAttemptLedger(store, resumed_attempt)
            try:
                live_ledger.start()
                materialized_checkpoint = _materialize_resume_checkpoints(
                    checkpoint,
                    checkpoint_receipt,
                    new_directory / "model-checkpoints",
                )
                if role == "BASE_QUALIFICATION":
                    report = self._execute_base_runtime(
                        run_id=predecessor_attempt.run_id,
                        system_id=system_id,
                        seed=seed,
                        checkpoint_directory=new_directory / "model-checkpoints",
                        live_ledger=live_ledger,
                        resume_checkpoint=materialized_checkpoint,
                        resume_attempt_root=new_directory.parent,
                        resume_record=resume,
                        stop_after_global_step=None,
                    )
                else:
                    if base_checkpoint is None or periodic_cache is None:
                        raise HostConfigurationError(
                            "residual resume artifacts were not resolved"
                        )
                    report = self._execute_residual_runtime(
                        system_id=system_id,
                        seed=seed,
                        checkpoint_directory=new_directory / "model-checkpoints",
                        live_ledger=live_ledger,
                        base_checkpoint=base_checkpoint,
                        periodic_cache=periodic_cache,
                        stop_after_global_step=None,
                        resume_checkpoint=materialized_checkpoint,
                        resume_attempt_root=new_directory.parent,
                        resume_record=resume,
                    )
                live_ledger.ensure_checkpoint(report.latest_checkpoint)
                live_ledger.stop()
                return self._close_attempt(
                    store, resumed_attempt, report, live_ledger
                )
            except Exception as error:
                with suppress(Exception):
                    live_ledger.stop()
                return self._persist_failed_execution(
                    store, resumed_attempt, live_ledger, error
                )
            except BaseException as error:
                with suppress(Exception):
                    live_ledger.stop()
                with suppress(Exception):
                    self._persist_failed_execution(
                        store, resumed_attempt, live_ledger, error
                    )
                raise
        finally:
            lease.release()

    @staticmethod
    def _recover_crashed_attempt(attempt_directory: Path) -> None:
        """Close an orphaned attempt at its latest already-receipted checkpoint."""

        terminal_path = attempt_directory / "terminal.json"
        if terminal_path.exists() or terminal_path.is_symlink():
            return
        with _AttemptLease.acquire(attempt_directory):
            # A predecessor may have terminalized between the first observation
            # and lease acquisition. Re-read everything only while holding the
            # same process-released lease used by active runs.
            if terminal_path.exists() or terminal_path.is_symlink():
                return
            attempt_path = attempt_directory / "attempt.json"
            if attempt_path.is_symlink() or not attempt_path.is_file():
                raise HostConfigurationError(
                    "crashed predecessor attempt record is absent"
                )
            attempt = parse_attempt_bytes(attempt_path.read_bytes())
            store = AttemptStore.open(attempt_directory.parent, attempt.attempt_id)
            checkpoint_paths = sorted(
                (attempt_directory / "checkpoints").glob("checkpoint-*.json")
            )
            if not checkpoint_paths:
                raise HostConfigurationError(
                    "crashed predecessor has no live checkpoint receipt to resume"
                )
            checkpoint_raw = checkpoint_paths[-1].read_bytes()
            parse_checkpoint_bytes(checkpoint_raw)
            heartbeat_paths = sorted(
                (attempt_directory / "heartbeats").glob("heartbeat-*.json")
            )
            heartbeat_sha256 = (
                artifact_sha256(heartbeat_paths[-1].read_bytes())
                if heartbeat_paths
                else None
            )
            store.write_terminal(
                TerminalRecord(
                    attempt_id=attempt.attempt_id,
                    run_id=attempt.run_id,
                    completed_at_utc=_utc_now(),
                    outcome="FAILED",
                    attempt_receipt_sha256=artifact_sha256(
                        canonical_attempt_bytes(attempt)
                    ),
                    latest_heartbeat_sha256=heartbeat_sha256,
                    latest_checkpoint_receipt_sha256=artifact_sha256(checkpoint_raw),
                    failure_code="HOST_PROCESS_LOST",
                )
            )

    @staticmethod
    def _persist_failed_execution(
        store: AttemptStore,
        attempt: AttemptRecord,
        live_ledger: _LiveAttemptLedger,
        error: BaseException,
    ) -> BackendExecution:
        """Persist a real failure artifact and terminal before returning FAILED."""

        completed_at = _utc_now()
        failure_path = _write_once_json(
            store.path / "failure.json",
            {
                "attempt_id": attempt.attempt_id,
                "authority": 0,
                "completed_at_utc": completed_at,
                "error_type": type(error).__qualname__,
                "failure_code": "HOST_EXECUTION_FAILED",
                "latest_checkpoint_receipt_sha256": (
                    live_ledger.latest_checkpoint_receipt_sha256
                ),
                "latest_heartbeat_sha256": live_ledger.latest_heartbeat_sha256,
                "production": False,
                "result_claimed": False,
                "run_id": attempt.run_id,
                "schema": "phaseset-host-failure-v1",
            },
        )
        terminal = TerminalRecord(
            attempt_id=attempt.attempt_id,
            run_id=attempt.run_id,
            completed_at_utc=completed_at,
            outcome="FAILED",
            attempt_receipt_sha256=artifact_sha256(
                canonical_attempt_bytes(attempt)
            ),
            latest_heartbeat_sha256=live_ledger.latest_heartbeat_sha256,
            latest_checkpoint_receipt_sha256=(
                live_ledger.latest_checkpoint_receipt_sha256
            ),
            failure_code="HOST_EXECUTION_FAILED",
        )
        store.write_terminal(terminal)
        return BackendExecution(
            outcome="FAILED",
            completed_at_utc=completed_at,
            artifacts=(failure_path, store.path / "terminal.json"),
        )

    def _close_attempt(
        self,
        store: AttemptStore,
        attempt: AttemptRecord,
        report: TrainingReport,
        live_ledger: _LiveAttemptLedger,
    ) -> BackendExecution:
        if (
            live_ledger.latest_checkpoint_receipt_sha256 is None
            or live_ledger.latest_checkpoint_payload_sha256
            != report.latest_checkpoint.sha256
            or live_ledger.latest_global_step != report.global_step
        ):
            raise HostConfigurationError(
                "terminal training checkpoint lacks a live attempt-ledger receipt"
            )
        succeeded = report.status == "COMPLETED"
        terminal = TerminalRecord(
            attempt_id=attempt.attempt_id,
            run_id=attempt.run_id,
            completed_at_utc=_utc_now(),
            outcome="SUCCEEDED" if succeeded else "FAILED",
            attempt_receipt_sha256=artifact_sha256(canonical_attempt_bytes(attempt)),
            latest_heartbeat_sha256=live_ledger.latest_heartbeat_sha256,
            latest_checkpoint_receipt_sha256=(
                live_ledger.latest_checkpoint_receipt_sha256
            ),
            failure_code=None if succeeded else "CONTROLLED_INTERRUPTION",
        )
        report_path = _write_once_json(
            store.path / "training-report.json", asdict(report)
        )
        store.write_terminal(terminal)
        terminal_path = store.path / "terminal.json"
        return BackendExecution(
            outcome="COMPLETED" if succeeded else "FAILED",
            completed_at_utc=terminal.completed_at_utc,
            artifacts=(report_path, terminal_path),
        )


class RuntimeAdapterFactory:
    """Callable passed directly to ``phaseset_core.cli.main``."""

    def __init__(self, config_path: str | Path) -> None:
        self._config_path = Path(config_path)

    def __call__(self, request: cli.CLICommandRequest) -> ProductionRuntimeAdapter:
        try:
            config = HostConfig.load(self._config_path)
            receipts = config.load_receipts()
            config.validate_runtime()
            config.validate_base_qualification_request(request)
            # Preflight verifies the authenticated prepared tree, materializes the
            # bounded capture source, and streams every concrete train batch once.
            config.load_sources()
        except (HostConfigurationError, OSError, TypeError, ValueError) as error:
            text = str(error).lower()
            if "rights" in text:
                holds = (
                    "HOLD_RIGHTS_GRANT_ABSENT",
                    "HOLD_EXTERNAL_RECEIPTS_UNVERIFIED",
                )
            elif "runtime" in text:
                holds = (
                    "HOLD_RUNTIME_RECEIPT_ABSENT",
                    "HOLD_EXTERNAL_RECEIPTS_UNVERIFIED",
                )
            elif "execution" in text:
                holds = (
                    "HOLD_EXECUTION_GRANT_ABSENT",
                    "HOLD_EXTERNAL_RECEIPTS_UNVERIFIED",
                )
            elif "prepared" in text or "batch" in text or "manifest" in text:
                holds = (
                    "HOLD_PREPARED_DATA_MANIFEST_ABSENT",
                    "HOLD_EXTERNAL_RECEIPTS_UNVERIFIED",
                )
            else:
                holds = ("HOLD_EXTERNAL_RECEIPTS_UNVERIFIED",)
            raise ExecutionHold(request.command, holds) from error
        backend = PhaseSetHostBackend(config, request)
        adapter = ProductionRuntimeAdapter(backend, receipts)
        if request.command == "qualify-base":
            backend.bind_qualification_adapter(adapter)
        return adapter


def main(argv: Sequence[str] | None = None, *, config_path: str | Path) -> int:
    """Run the public CLI with one request-aware private factory."""

    return cli.main(argv, runtime_adapter_factory=RuntimeAdapterFactory(config_path))


def entrypoint(argv: Sequence[str] | None = None) -> int:
    """Private executable wrapper; arguments after ``--`` go to PhaseSet."""

    parser = argparse.ArgumentParser(prog="phaseset-private-host")
    parser.add_argument("--host-config", type=Path, required=True)
    parser.add_argument("phaseset_arguments", nargs=argparse.REMAINDER)
    namespace = parser.parse_args(argv)
    arguments = list(namespace.phaseset_arguments)
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if not arguments:
        parser.error("a PhaseSet command is required after --")
    return main(arguments, config_path=namespace.host_config)


if __name__ == "__main__":
    raise SystemExit(entrypoint(sys.argv[1:]))


__all__ = [
    "HostConfig",
    "HostConfigurationError",
    "HOST_CONFIG_QUALIFICATION_SCHEMA",
    "PrivatePreparedDataSource",
    "RuntimeAdapterFactory",
    "entrypoint",
    "main",
]
