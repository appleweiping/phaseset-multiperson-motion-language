"""Resolve the completed registered base cohort without scoring it.

This private controller helper consumes an already authenticated attempt-registry
snapshot.  It validates the exact nine successful terminal chains through
``execution.verified_completed_attempt_evidence``, retains their failed resume
ancestry, and returns owned bytes for each validation-selected checkpoint.

The result is deliberately authority zero.  It is neither a score artifact nor
a qualification, and a registry row or digest supplied to this API is never
treated as proof that an external registry is authentic.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
from typing import Literal, Mapping

import torch
from torch import nn

from phaseset_core import execution as execution_module
from phaseset_core import training as training_module
from phaseset_core.experiments import (
    base_run_ids,
    build_experiment_plan,
    experiment_plan_sha256,
    parse_run_id,
)


AUTHORITY = 0
PRODUCTION = False
RESULT_CLAIMED = False
STATUS = "BASE_COHORT_CHECKPOINTS_RESOLVED_NO_SCORE_NO_QUALIFICATION"
MAX_CHECKPOINT_BYTES = 2 * 1024**3
MAX_SELECTED_COHORT_BYTES = 4 * 1024**3
MAX_LEDGER_ARTIFACT_BYTES = 1024**2
MAX_CHECKPOINT_DIRECTORY_ENTRIES = 100_000
MAX_ATTEMPTS_PER_RUN = 64

_ATTEMPT_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_CHECKPOINT_NAME = re.compile(
    r"checkpoint-(?P<step>[0-9]{12})-(?P<sequence>[0-9]{6})-"
    r"(?P<reason>update|validation)\.pt\Z"
)
_CHECKPOINT_KEYS = frozenset(
    {
        "authority",
        "behavior_sha256",
        "best_checkpoint_name",
        "best_checkpoint_sha256",
        "best_validation_metric",
        "checkpoint_sequence",
        "code_artifact_sha256",
        "config_sha256",
        "environment_sha256",
        "epoch",
        "external_receipt_verified",
        "factory_sha256",
        "frozen_base_checkpoint_sha256",
        "frozen_base_state_sha256",
        "global_step",
        "initial_optimizable_state_sha256",
        "initialization_binding_sha256",
        "last_train_loss",
        "max_microbatch_edges",
        "model",
        "optimizer",
        "precision_mode",
        "production",
        "qualified_base_selection_sha256",
        "residual_capacity_audit_sha256",
        "result_claimed",
        "rng",
        "scheduler",
        "schema",
        "seed",
        "stage",
        "state_digest",
        "system_id",
        "total_steps",
        "train_edges_seen",
        "train_manifest_sha256",
        "update_index",
        "updates_per_epoch",
        "val_manifest_sha256",
        "validation_edges_seen",
    }
)


class BaseCohortResolutionError(ValueError):
    """The registry, terminal cohort, or selected checkpoint is inconsistent."""


@dataclass(frozen=True, slots=True)
class BaseCohortAdmission:
    """Expected bindings retained by the trusted private controller.

    These values are comparison inputs only.  Constructing this object does not
    authenticate their origin and does not confer execution or result authority.
    """

    plan_sha256: str
    matrix_sha256: str
    training_config_sha256: str
    source_tree_sha256: str
    train_manifest_sha256: str
    val_manifest_sha256: str
    device: str
    request_bf16: bool
    bf16_runtime_qualified: bool
    edge_budget: int
    checkpoint_every_updates: int


@dataclass(frozen=True, slots=True)
class BaseAttemptRegistryRow:
    """One row from an externally authenticated cohort registry snapshot.

    Rows for one run are ordered oldest-to-newest and must describe exactly one
    resume chain.  A digest here is checked against actual terminal bytes but is
    not accepted as evidence that the registry itself was authenticated.
    """

    attempt_id: str
    run_id: str
    terminal_outcome: Literal["FAILED", "HELD", "SUCCEEDED"]
    terminal_sha256: str


@dataclass(frozen=True, slots=True, repr=False)
class FailedBaseAttempt:
    attempt_id: str
    terminal_sha256: str
    terminal_outcome: Literal["FAILED", "HELD"]
    failure_code: str


@dataclass(frozen=True, slots=True, repr=False)
class SelectedBaseCheckpoint:
    """Owned immutable bytes for one fully checked validation-best payload."""

    name: str
    sha256: str
    raw: bytes
    epoch: int
    global_step: int
    state_digest: str


@dataclass(frozen=True, slots=True, repr=False)
class ResolvedBaseRun:
    run_id: str
    attempt_id: str
    system_id: str
    seed: int
    terminal_sha256: str
    latest_checkpoint_sha256: str
    latest_checkpoint_name: str
    selected_checkpoint: SelectedBaseCheckpoint
    failed_predecessors: tuple[FailedBaseAttempt, ...]


@dataclass(frozen=True, slots=True, repr=False)
class ResolvedBaseCohort:
    """Exact registered row order with no score, latency, or winner."""

    rows: tuple[ResolvedBaseRun, ...]
    plan_sha256: str
    matrix_sha256: str
    training_config_sha256: str
    source_tree_sha256: str
    train_manifest_sha256: str
    val_manifest_sha256: str
    authority: int = AUTHORITY
    production: bool = PRODUCTION
    result_claimed: bool = RESULT_CLAIMED
    status: str = STATUS


@dataclass(frozen=True, slots=True, repr=False)
class _CheckedCheckpoint:
    raw: bytes
    sha256: str
    name: str
    epoch: int
    update_index: int
    global_step: int
    checkpoint_sequence: int
    updates_per_epoch: int
    total_steps: int
    train_edges_seen: int
    validation_edges_seen: int
    max_microbatch_edges: int
    best_validation_metric: float | None
    state_digest: str
    best_name: str
    best_sha256: str


def _lower_sha256(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise BaseCohortResolutionError(f"{label} must be lowercase SHA-256 hex")
    return value


def _attempt_id(value: object) -> str:
    if (
        type(value) is not str
        or _ATTEMPT_ID.fullmatch(value) is None
        or value in {".", ".."}
    ):
        raise BaseCohortResolutionError("attempt_id is not one safe path component")
    return value


def _exact_positive_int(value: object, label: str) -> int:
    if type(value) is not int or value < 1 or value > (1 << 63) - 1:
        raise BaseCohortResolutionError(f"{label} must be an exact positive int")
    return value


def _checked_admission(value: object) -> BaseCohortAdmission:
    if type(value) is not BaseCohortAdmission:
        raise TypeError("admission must be exact BaseCohortAdmission")
    plan = build_experiment_plan()
    expected_plan = experiment_plan_sha256(plan)
    if value.plan_sha256 != expected_plan or value.matrix_sha256 != plan.matrix_sha256:
        raise BaseCohortResolutionError("admission differs from the installed experiment plan")
    for label in (
        "training_config_sha256",
        "source_tree_sha256",
        "train_manifest_sha256",
        "val_manifest_sha256",
    ):
        _lower_sha256(getattr(value, label), f"admission.{label}")
    if type(value.device) is not str or not value.device:
        raise BaseCohortResolutionError("admission.device must be a nonempty exact string")
    if (
        type(value.request_bf16) is not bool
        or type(value.bf16_runtime_qualified) is not bool
    ):
        raise TypeError("admission precision switches must be exact bool values")
    _exact_positive_int(value.edge_budget, "admission.edge_budget")
    _exact_positive_int(
        value.checkpoint_every_updates,
        "admission.checkpoint_every_updates",
    )
    try:
        # This closes all formal runtime knobs not represented as fields above.
        training_module.TrainingConfig(
            stage="base",
            seed=1729,
            device=value.device,
            request_bf16=value.request_bf16,
            bf16_runtime_qualified=value.bf16_runtime_qualified,
            edge_budget=value.edge_budget,
            checkpoint_every_updates=value.checkpoint_every_updates,
            synthetic_contract=False,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise BaseCohortResolutionError("admission training runtime is invalid") from error
    return value


def _checked_registry_rows(value: object) -> tuple[BaseAttemptRegistryRow, ...]:
    if type(value) is not tuple:
        raise TypeError("registry_rows must be an exact tuple")
    if not value:
        raise BaseCohortResolutionError("attempt registry snapshot is empty")
    registered = frozenset(base_run_ids())
    if len(value) > MAX_ATTEMPTS_PER_RUN * len(registered):
        raise BaseCohortResolutionError("attempt registry exceeds its closed row bound")
    seen_attempts: set[str] = set()
    checked: list[BaseAttemptRegistryRow] = []
    for ordinal, row in enumerate(value):
        if type(row) is not BaseAttemptRegistryRow:
            raise TypeError(f"registry_rows[{ordinal}] has the wrong exact type")
        attempt = _attempt_id(row.attempt_id)
        if attempt in seen_attempts:
            raise BaseCohortResolutionError("attempt registry contains a duplicate attempt_id")
        seen_attempts.add(attempt)
        if type(row.run_id) is not str or row.run_id not in registered:
            raise BaseCohortResolutionError("attempt registry contains an unregistered base run")
        role, _seed, _system = parse_run_id(row.run_id)
        if role != "BASE_QUALIFICATION":
            raise BaseCohortResolutionError("attempt registry row is not a base run")
        if row.terminal_outcome not in ("FAILED", "HELD", "SUCCEEDED"):
            raise BaseCohortResolutionError("registry terminal outcome is outside the vocabulary")
        _lower_sha256(row.terminal_sha256, "registry terminal_sha256")
        checked.append(row)
    successes = [row for row in checked if row.terminal_outcome == "SUCCEEDED"]
    expected = base_run_ids()
    if len(successes) != len(expected):
        raise BaseCohortResolutionError("registry must contain exactly nine successful terminals")
    for run_id in expected:
        if sum(row.run_id == run_id for row in successes) != 1:
            raise BaseCohortResolutionError(
                "registry must contain exactly one successful terminal per base run"
            )
        if sum(row.run_id == run_id for row in checked) > MAX_ATTEMPTS_PER_RUN:
            raise BaseCohortResolutionError(
                "registry run history exceeds its closed attempt bound"
            )
    return tuple(checked)


def _checked_directory(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise BaseCohortResolutionError(f"{label} is absent") from error
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
        raise BaseCohortResolutionError(f"{label} must be a non-symlink directory")
    return path


def _read_owned_file(
    path: Path,
    *,
    label: str,
    maximum_bytes: int,
    expected_sha256: str | None = None,
) -> tuple[bytes, str]:
    """Read one stable regular leaf from one descriptor without following it."""

    flags = os.O_RDONLY
    for name in ("O_BINARY", "O_CLOEXEC", "O_NOFOLLOW"):
        flags |= int(getattr(os, name, 0))
    before_link = None
    if not hasattr(os, "O_NOFOLLOW"):
        try:
            before_link = path.lstat()
        except OSError as error:
            raise BaseCohortResolutionError(f"{label} is absent") from error
        if stat.S_ISLNK(before_link.st_mode):
            raise BaseCohortResolutionError(f"{label} cannot be a symlink")
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise BaseCohortResolutionError(f"{label} could not be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise BaseCohortResolutionError(f"{label} must be a regular file")
        if before.st_size < 1 or before.st_size > maximum_bytes:
            raise BaseCohortResolutionError(f"{label} exceeds its closed byte bound")
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise BaseCohortResolutionError(f"{label} ended before its stated size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise BaseCohortResolutionError(f"{label} grew during its bounded read")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise BaseCohortResolutionError(f"{label} changed during its bounded read")
    if before_link is not None:
        try:
            after_link = path.lstat()
        except OSError as error:
            raise BaseCohortResolutionError(f"{label} changed after its read") from error
        if stat.S_ISLNK(after_link.st_mode) or any(
            getattr(before_link, field) != getattr(after_link, field)
            for field in stable_fields
        ):
            raise BaseCohortResolutionError(f"{label} changed around its fallback read")
    raw = b"".join(chunks)
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != _lower_sha256(
        expected_sha256,
        f"expected {label} sha256",
    ):
        raise BaseCohortResolutionError(f"{label} digest mismatch")
    return raw, digest


def _decode_checkpoint(raw: bytes, label: str) -> dict[str, object]:
    try:
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as error:
        raise BaseCohortResolutionError(f"{label} is not a restricted checkpoint") from error
    if type(payload) is not dict or frozenset(payload) != _CHECKPOINT_KEYS:
        raise BaseCohortResolutionError(f"{label} checkpoint keys are not closed")
    return payload


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


def _torch_value_sha256(value: object) -> str:
    stream = io.BytesIO()
    torch.save(value, stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def _exact_payload_int(
    payload: Mapping[str, object],
    name: str,
    *,
    minimum: int = 0,
) -> int:
    value = payload.get(name)
    if type(value) is not int or value < minimum or value > (1 << 63) - 1:
        raise BaseCohortResolutionError(f"checkpoint {name} is not an exact bounded int")
    return value


def _validate_rng_without_consuming(payload: Mapping[str, object]) -> None:
    rng = payload.get("rng")
    if not isinstance(rng, Mapping):
        raise BaseCohortResolutionError("checkpoint RNG state is malformed")
    ambient = training_module._capture_rng()
    try:
        training_module._restore_rng(rng)
    except training_module.TrainingCheckpointError as error:
        raise BaseCohortResolutionError("checkpoint RNG state is malformed") from error
    finally:
        training_module._restore_rng(ambient)


def _validate_model_and_optimizer(
    payload: Mapping[str, object],
    *,
    config: training_module.TrainingConfig,
    system_id: str,
    global_step: int,
    total_steps: int,
) -> None:
    try:
        system, binding = training_module.construct_registered_base_seed_bound_system(
            system_id,
            config,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise BaseCohortResolutionError("registered base reconstruction failed") from error
    expected_bindings = {
        "behavior_sha256": binding.behavior_sha256,
        "code_artifact_sha256": binding.code_artifact_sha256,
        "environment_sha256": binding.environment_sha256,
        "factory_sha256": binding.factory_sha256,
        "initial_optimizable_state_sha256": binding.initial_optimizable_state_sha256,
        "initialization_binding_sha256": binding.sha256,
    }
    for name, expected in expected_bindings.items():
        if payload.get(name) != expected:
            raise BaseCohortResolutionError(f"checkpoint {name} binding mismatch")

    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise BaseCohortResolutionError("checkpoint model state is malformed")
    parameters = [parameter for parameter in system.parameters() if parameter.requires_grad]
    if not parameters or any(not isinstance(parameter, nn.Parameter) for parameter in parameters):
        raise BaseCohortResolutionError("registered base parameter census is malformed")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=training_module.WEIGHT_DECAY,
    )

    def multiplier(step: int) -> float:
        return training_module._learning_rate_multiplier(step, total_steps)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    try:
        training_module._validate_restored_optimizer_scheduler(
            payload,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            global_step=global_step,
            total_steps=total_steps,
        )
        system.load_state_dict(model, strict=True)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise BaseCohortResolutionError("checkpoint model/optimizer state is malformed") from error
    if training_module.training_system_state_sha256(system) != training_module._stable_hash(
        model
    ):
        raise BaseCohortResolutionError("loaded registered base differs from checkpoint bytes")
    if training_module.training_system_behavior_sha256(system) != binding.behavior_sha256:
        raise BaseCohortResolutionError("loaded registered base behavior changed")
    _validate_rng_without_consuming(payload)


def _validate_ledger_payload_binding(
    payload: Mapping[str, object],
    record: execution_module.CheckpointRecord,
) -> None:
    epoch = _exact_payload_int(payload, "epoch")
    global_step = _exact_payload_int(payload, "global_step")
    if record.epoch_index != epoch or record.global_step != global_step:
        raise BaseCohortResolutionError("ledger/runtime checkpoint cursor mismatch")
    required = {
        "model_state_sha256": _torch_value_sha256(payload["model"]),
        "optimizer_state_sha256": _torch_value_sha256(payload["optimizer"]),
        "rng_state_sha256": _torch_value_sha256(payload["rng"]),
        "dropout_state_sha256": _torch_value_sha256(payload["rng"]),
        "sampler_state_sha256": hashlib.sha256(
            _canonical_json_bytes(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "update_index": _exact_payload_int(payload, "update_index"),
                }
            )
        ).hexdigest(),
        "dataloader_state_sha256": hashlib.sha256(
            _canonical_json_bytes(
                {
                    "train_manifest_sha256": payload["train_manifest_sha256"],
                    "val_manifest_sha256": payload["val_manifest_sha256"],
                }
            )
        ).hexdigest(),
        "validation_state_sha256": hashlib.sha256(
            _canonical_json_bytes(
                {
                    "best_checkpoint_name": payload["best_checkpoint_name"],
                    "best_checkpoint_sha256": payload["best_checkpoint_sha256"],
                    "best_validation_metric": payload["best_validation_metric"],
                }
            )
        ).hexdigest(),
    }
    for name, expected in required.items():
        if getattr(record, name) != expected:
            raise BaseCohortResolutionError(f"ledger/runtime checkpoint {name} mismatch")


def _validate_checkpoint_payload(
    raw: bytes,
    *,
    sha256: str,
    name: str,
    run_id: str,
    admission: BaseCohortAdmission,
    ledger_record: execution_module.CheckpointRecord | None,
    require_completed_latest: bool,
    require_selected_best: bool,
) -> _CheckedCheckpoint:
    match = _CHECKPOINT_NAME.fullmatch(name)
    if match is None:
        raise BaseCohortResolutionError("checkpoint filename is outside the runtime grammar")
    role, seed, system_id = parse_run_id(run_id)
    if role != "BASE_QUALIFICATION":
        raise BaseCohortResolutionError("checkpoint run is not a registered base run")
    payload = _decode_checkpoint(raw, name)
    state_digest = payload.get("state_digest")
    without_digest = dict(payload)
    without_digest.pop("state_digest")
    if (
        type(state_digest) is not str
        or state_digest != training_module._stable_hash(without_digest)
    ):
        raise BaseCohortResolutionError("checkpoint state digest mismatch")
    if (
        type(payload["authority"]) is not int
        or payload["authority"] != 0
        or payload["production"] is not False
        or payload["result_claimed"] is not False
        or payload["external_receipt_verified"] is not False
    ):
        raise BaseCohortResolutionError("checkpoint authority/result header is malformed")
    if any(
        payload[name_] is not None
        for name_ in (
            "frozen_base_checkpoint_sha256",
            "frozen_base_state_sha256",
            "qualified_base_selection_sha256",
            "residual_capacity_audit_sha256",
        )
    ):
        raise BaseCohortResolutionError("base checkpoint contains residual-only bindings")
    config = training_module.TrainingConfig(
        stage="base",
        seed=seed,
        device=admission.device,
        request_bf16=admission.request_bf16,
        bf16_runtime_qualified=admission.bf16_runtime_qualified,
        edge_budget=admission.edge_budget,
        checkpoint_every_updates=admission.checkpoint_every_updates,
        synthetic_contract=False,
    )
    expected = {
        "config_sha256": config.sha256,
        "precision_mode": training_module.resolve_precision(config).mode,
        "schema": training_module.CHECKPOINT_SCHEMA,
        "seed": seed,
        "stage": "base",
        "system_id": system_id,
        "train_manifest_sha256": admission.train_manifest_sha256,
        "val_manifest_sha256": admission.val_manifest_sha256,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise BaseCohortResolutionError(f"checkpoint {field} binding mismatch")
    updates_per_epoch = _exact_payload_int(payload, "updates_per_epoch", minimum=1)
    total_steps = _exact_payload_int(payload, "total_steps", minimum=1)
    if total_steps != training_module.BASE_EPOCHS * updates_per_epoch:
        raise BaseCohortResolutionError("checkpoint total schedule is not the formal base schedule")
    try:
        cursor = training_module._validate_checkpoint_progress(
            payload,
            epochs=training_module.BASE_EPOCHS,
            updates_per_epoch=updates_per_epoch,
            total_steps=total_steps,
            artifact_path=Path(name),
            artifact_sha256=sha256,
        )
    except training_module.TrainingCheckpointError as error:
        raise BaseCohortResolutionError("checkpoint cursor/state is malformed") from error
    if int(match.group("step")) != cursor.global_step:
        raise BaseCohortResolutionError("checkpoint filename step differs from payload")
    if int(match.group("sequence")) != cursor.checkpoint_sequence:
        raise BaseCohortResolutionError("checkpoint filename sequence differs from payload")
    if require_completed_latest and (
        match.group("reason") != "validation"
        or cursor.epoch != training_module.BASE_EPOCHS
        or cursor.update_index != 0
        or cursor.global_step != total_steps
        or cursor.best_validation_metric is None
    ):
        raise BaseCohortResolutionError("successful terminal does not point to a completed base checkpoint")
    if cursor.best_checkpoint_name is None or cursor.best_checkpoint_sha256 is None:
        raise BaseCohortResolutionError("checkpoint has no immutable validation-best pointer")
    if require_selected_best and (
        match.group("reason") != "validation"
        or cursor.epoch < 1
        or cursor.update_index != 0
        or cursor.best_validation_metric is None
        or cursor.best_checkpoint_name != name
        or cursor.best_checkpoint_sha256 != sha256
    ):
        raise BaseCohortResolutionError("selected payload is not its best-at-write checkpoint")
    _validate_model_and_optimizer(
        payload,
        config=config,
        system_id=system_id,
        global_step=cursor.global_step,
        total_steps=total_steps,
    )
    if ledger_record is not None:
        _validate_ledger_payload_binding(payload, ledger_record)
    return _CheckedCheckpoint(
        raw=raw,
        sha256=sha256,
        name=name,
        epoch=cursor.epoch,
        update_index=cursor.update_index,
        global_step=cursor.global_step,
        checkpoint_sequence=cursor.checkpoint_sequence,
        updates_per_epoch=updates_per_epoch,
        total_steps=total_steps,
        train_edges_seen=cursor.train_edges_seen,
        validation_edges_seen=cursor.validation_edges_seen,
        max_microbatch_edges=cursor.max_microbatch_edges,
        best_validation_metric=cursor.best_validation_metric,
        state_digest=state_digest,
        best_name=cursor.best_checkpoint_name,
        best_sha256=cursor.best_checkpoint_sha256,
    )


def _validate_latest_selected_link(
    latest: _CheckedCheckpoint,
    selected: _CheckedCheckpoint,
) -> None:
    """Close invariants shared by the terminal and its validation-best payload."""

    if latest.best_name != selected.name or latest.best_sha256 != selected.sha256:
        raise BaseCohortResolutionError("latest checkpoint best pointer changed")
    if (
        latest.updates_per_epoch != selected.updates_per_epoch
        or latest.total_steps != selected.total_steps
    ):
        raise BaseCohortResolutionError("latest/selected checkpoint schedule mismatch")
    if (
        latest.best_validation_metric is None
        or selected.best_validation_metric is None
        or latest.best_validation_metric != selected.best_validation_metric
    ):
        raise BaseCohortResolutionError("latest/selected validation-best metric mismatch")
    if (
        selected.epoch > latest.epoch
        or selected.global_step > latest.global_step
        or selected.checkpoint_sequence > latest.checkpoint_sequence
    ):
        raise BaseCohortResolutionError("selected checkpoint is later than terminal latest")
    if (
        selected.train_edges_seen > latest.train_edges_seen
        or selected.validation_edges_seen > latest.validation_edges_seen
        or selected.max_microbatch_edges > latest.max_microbatch_edges
    ):
        raise BaseCohortResolutionError("selected checkpoint counters exceed terminal latest")


def _latest_checkpoint_file(
    checkpoint_directory: Path,
    *,
    global_step: int,
) -> Path:
    candidates: list[Path] = []
    try:
        with os.scandir(checkpoint_directory) as entries:
            for ordinal, entry in enumerate(entries, start=1):
                if ordinal > MAX_CHECKPOINT_DIRECTORY_ENTRIES:
                    raise BaseCohortResolutionError(
                        "checkpoint directory exceeds its closed entry bound"
                    )
                match = _CHECKPOINT_NAME.fullmatch(entry.name)
                if (
                    match is not None
                    and int(match.group("step")) == global_step
                    and match.group("reason") == "validation"
                ):
                    candidates.append(checkpoint_directory / entry.name)
    except BaseCohortResolutionError:
        raise
    except OSError as error:
        raise BaseCohortResolutionError("checkpoint directory could not be enumerated") from error
    if len(candidates) != 1:
        raise BaseCohortResolutionError(
            "terminal checkpoint step must have exactly one validation payload"
        )
    return candidates[0]


def _owned_failed_predecessors(
    attempt_root: Path,
    evidence: execution_module.CompletedAttemptEvidence,
    *,
    admission: BaseCohortAdmission,
) -> tuple[FailedBaseAttempt, ...]:
    rows: list[FailedBaseAttempt] = []
    resume = evidence.resume
    visited = {evidence.attempt.attempt_id}
    while resume is not None:
        if len(rows) >= MAX_ATTEMPTS_PER_RUN - 1:
            raise BaseCohortResolutionError("resume chain exceeds its closed attempt bound")
        predecessor_id = _attempt_id(resume.predecessor_attempt_id)
        if predecessor_id in visited:
            raise BaseCohortResolutionError("resume chain contains a cycle")
        visited.add(predecessor_id)
        directory = _checked_directory(attempt_root / predecessor_id, "predecessor attempt")
        attempt_raw, _ = _read_owned_file(
            directory / "attempt.json",
            label="predecessor attempt receipt",
            maximum_bytes=MAX_LEDGER_ARTIFACT_BYTES,
        )
        terminal_raw, terminal_sha = _read_owned_file(
            directory / "terminal.json",
            label="predecessor terminal",
            maximum_bytes=MAX_LEDGER_ARTIFACT_BYTES,
            expected_sha256=resume.predecessor_terminal_sha256,
        )
        try:
            attempt = execution_module.parse_attempt_bytes(attempt_raw)
            terminal = execution_module.parse_terminal_bytes(terminal_raw)
        except (TypeError, ValueError) as error:
            raise BaseCohortResolutionError("predecessor ledger artifact is malformed") from error
        if (
            attempt.attempt_id != predecessor_id
            or attempt.run_id != evidence.attempt.run_id
            or attempt.plan_sha256 != admission.plan_sha256
            or attempt.matrix_sha256 != admission.matrix_sha256
            or attempt.training_config_sha256 != admission.training_config_sha256
            or attempt.source_tree_sha256 != admission.source_tree_sha256
            or terminal.attempt_id != predecessor_id
            or terminal.run_id != evidence.attempt.run_id
            or terminal.outcome == "SUCCEEDED"
            or terminal.failure_code is None
            or terminal.attempt_receipt_sha256 != hashlib.sha256(attempt_raw).hexdigest()
            or terminal.latest_checkpoint_receipt_sha256 != resume.checkpoint_receipt_sha256
        ):
            raise BaseCohortResolutionError("failed predecessor identity/binding is inconsistent")
        rows.append(
            FailedBaseAttempt(
                attempt_id=predecessor_id,
                terminal_sha256=terminal_sha,
                terminal_outcome=terminal.outcome,
                failure_code=terminal.failure_code,
            )
        )
        resume_path = directory / "resume.json"
        if resume_path.exists() or resume_path.is_symlink():
            resume_raw, _ = _read_owned_file(
                resume_path,
                label="predecessor resume receipt",
                maximum_bytes=MAX_LEDGER_ARTIFACT_BYTES,
            )
            try:
                predecessor_resume = execution_module.parse_resume_bytes(resume_raw)
            except (TypeError, ValueError) as error:
                raise BaseCohortResolutionError("predecessor resume receipt is malformed") from error
            if (
                predecessor_resume.new_attempt_id != predecessor_id
                or predecessor_resume.run_id != evidence.attempt.run_id
                or predecessor_resume.created_at_utc != attempt.created_at_utc
            ):
                raise BaseCohortResolutionError("predecessor resume identity is inconsistent")
            resume = predecessor_resume
        else:
            resume = None
    rows.reverse()
    return tuple(rows)


def _resolve_registered_run(
    attempt_root: Path,
    run_id: str,
    registry_history: tuple[BaseAttemptRegistryRow, ...],
    admission: BaseCohortAdmission,
    *,
    selected_maximum_bytes: int = MAX_CHECKPOINT_BYTES,
) -> ResolvedBaseRun:
    success = [row for row in registry_history if row.terminal_outcome == "SUCCEEDED"]
    if len(success) != 1 or registry_history[-1] != success[0]:
        raise BaseCohortResolutionError("successful registry row must close its run history")
    success_row = success[0]
    try:
        evidence = execution_module.verified_completed_attempt_evidence(
            attempt_root,
            success_row.attempt_id,
        )
    except (TypeError, ValueError) as error:
        raise BaseCohortResolutionError("successful attempt chain did not verify") from error
    attempt = evidence.attempt
    if (
        attempt.run_id != run_id
        or attempt.plan_sha256 != admission.plan_sha256
        or attempt.matrix_sha256 != admission.matrix_sha256
        or attempt.training_config_sha256 != admission.training_config_sha256
        or attempt.source_tree_sha256 != admission.source_tree_sha256
        or evidence.terminal_sha256 != success_row.terminal_sha256
    ):
        raise BaseCohortResolutionError("successful attempt differs from cohort admission/registry")
    failed = _owned_failed_predecessors(attempt_root, evidence, admission=admission)
    actual_history = tuple(
        (row.attempt_id, row.terminal_outcome, row.terminal_sha256) for row in failed
    ) + ((success_row.attempt_id, "SUCCEEDED", evidence.terminal_sha256),)
    expected_history = tuple(
        (row.attempt_id, row.terminal_outcome, row.terminal_sha256)
        for row in registry_history
    )
    if actual_history != expected_history:
        raise BaseCohortResolutionError("registry history differs from the verified resume chain")

    checkpoint_directory = _checked_directory(
        attempt_root / attempt.attempt_id / "model-checkpoints",
        "successful checkpoint directory",
    )
    latest_path = _latest_checkpoint_file(
        checkpoint_directory,
        global_step=evidence.latest_checkpoint.global_step,
    )
    latest_raw, latest_sha = _read_owned_file(
        latest_path,
        label="terminal-bound latest checkpoint",
        maximum_bytes=MAX_CHECKPOINT_BYTES,
        expected_sha256=evidence.latest_checkpoint_payload_sha256,
    )
    latest = _validate_checkpoint_payload(
        latest_raw,
        sha256=latest_sha,
        name=latest_path.name,
        run_id=run_id,
        admission=admission,
        ledger_record=evidence.latest_checkpoint,
        require_completed_latest=True,
        require_selected_best=False,
    )
    selected_path = checkpoint_directory / latest.best_name
    selected_raw, selected_sha = _read_owned_file(
        selected_path,
        label="validation-selected checkpoint",
        maximum_bytes=min(MAX_CHECKPOINT_BYTES, selected_maximum_bytes),
        expected_sha256=latest.best_sha256,
    )
    selected = _validate_checkpoint_payload(
        selected_raw,
        sha256=selected_sha,
        name=selected_path.name,
        run_id=run_id,
        admission=admission,
        ledger_record=None,
        require_completed_latest=False,
        require_selected_best=True,
    )
    _validate_latest_selected_link(latest, selected)
    _role, seed, system_id = parse_run_id(run_id)
    return ResolvedBaseRun(
        run_id=run_id,
        attempt_id=attempt.attempt_id,
        system_id=system_id,
        seed=seed,
        terminal_sha256=evidence.terminal_sha256,
        latest_checkpoint_sha256=latest.sha256,
        latest_checkpoint_name=latest.name,
        selected_checkpoint=SelectedBaseCheckpoint(
            name=selected.name,
            sha256=selected.sha256,
            raw=selected.raw,
            epoch=selected.epoch,
            global_step=selected.global_step,
            state_digest=selected.state_digest,
        ),
        failed_predecessors=failed,
    )


def resolve_completed_base_cohort(
    attempt_root: str | Path,
    *,
    registry_rows: tuple[BaseAttemptRegistryRow, ...],
    admission: BaseCohortAdmission,
) -> ResolvedBaseCohort:
    """Resolve exactly nine terminal chains and their validation-best payloads.

    ``registry_rows`` must come from the controller's separately authenticated,
    stable private registry snapshot.  This function validates the snapshot's
    claims against ledger/checkpoint bytes, but cannot authenticate the source
    of the snapshot and never treats its hashes as a qualification grant.
    """

    checked_admission = _checked_admission(admission)
    checked_registry = _checked_registry_rows(registry_rows)
    root = _checked_directory(Path(attempt_root), "attempt root")
    rows: list[ResolvedBaseRun] = []
    selected_bytes = 0
    for run_id in base_run_ids():
        history = tuple(row for row in checked_registry if row.run_id == run_id)
        remaining = MAX_SELECTED_COHORT_BYTES - selected_bytes
        if remaining < 1:
            raise BaseCohortResolutionError(
                "selected checkpoint cohort exceeds its closed byte bound"
            )
        resolved = _resolve_registered_run(
            root,
            run_id,
            history,
            checked_admission,
            selected_maximum_bytes=remaining,
        )
        selected_bytes += len(resolved.selected_checkpoint.raw)
        if selected_bytes > MAX_SELECTED_COHORT_BYTES:
            raise AssertionError("selected checkpoint byte accounting exceeded its bound")
        rows.append(resolved)
    canonical = tuple(rows)
    if tuple(row.run_id for row in canonical) != base_run_ids():
        raise AssertionError("resolved base cohort lost the registered order")
    return ResolvedBaseCohort(
        rows=canonical,
        plan_sha256=checked_admission.plan_sha256,
        matrix_sha256=checked_admission.matrix_sha256,
        training_config_sha256=checked_admission.training_config_sha256,
        source_tree_sha256=checked_admission.source_tree_sha256,
        train_manifest_sha256=checked_admission.train_manifest_sha256,
        val_manifest_sha256=checked_admission.val_manifest_sha256,
    )


__all__ = [
    "AUTHORITY",
    "BaseAttemptRegistryRow",
    "BaseCohortAdmission",
    "BaseCohortResolutionError",
    "FailedBaseAttempt",
    "MAX_CHECKPOINT_BYTES",
    "MAX_SELECTED_COHORT_BYTES",
    "PRODUCTION",
    "RESULT_CLAIMED",
    "ResolvedBaseCohort",
    "ResolvedBaseRun",
    "STATUS",
    "SelectedBaseCheckpoint",
    "resolve_completed_base_cohort",
]
