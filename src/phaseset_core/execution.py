"""Fail-closed PhaseSet execution contracts and local write-once ledgers.

This module never connects to a server, loads a dataset, starts a GPU, or mints
an external grant.  It provides immutable artifact schemas, a crash-auditable
local attempt ledger, and an authority-zero preflight used by the public CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Protocol

from phaseset_core.experiments import (
    BASE_SYSTEMS,
    ExperimentPlan,
    build_experiment_plan,
    parse_run_id,
)


AUTHORITY = 0
PRODUCTION = False
EXTERNAL_RECEIPT_VERIFIED = False
RESULT_CLAIMED = False
STATUS = "AUTHORITY0_EXTERNAL_RECEIPTS_ABSENT_EXECUTION_HELD"
ATTEMPT_SCHEMA = "phaseset-attempt-v1"
HEARTBEAT_SCHEMA = "phaseset-heartbeat-v1"
CHECKPOINT_SCHEMA = "phaseset-checkpoint-v2"
TERMINAL_SCHEMA = "phaseset-terminal-v1"
RESUME_SCHEMA = "phaseset-resume-v2"
PERIODIC_CACHE_SCHEMA = "phaseset-periodic-cache-v1"
SEALED_TEST_SCHEMA = "phaseset-sealed-test-consumption-v4"
SEALED_TEST_EVALUATION_SCHEMA = "phaseset-sealed-test-evaluation-binding-v1"
TRAINING_CONFIG_SCHEMA = "phaseset-training-config-v1"
RUNTIME_ADMISSION_SCHEMA = "phaseset-runtime-admission-v1"
COMMAND_RESULT_SCHEMA = "phaseset-command-result-v2"
MAX_VERIFIED_ATTEMPT_CHAIN_LENGTH = 64

COMMANDS = (
    "preflight",
    "prepare-data",
    "audit-split",
    "run-base",
    "qualify-base",
    "build-periodic-cache",
    "run-residual",
    "evaluate",
    "bootstrap",
    "render-paper",
    "resume",
)

_ATTEMPT_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
_FAILURE_CODE = re.compile(r"[A-Z][A-Z0-9_]{2,95}\Z")
_HEARTBEAT_FILE = re.compile(r"heartbeat-(\d{8,})\.json\Z")
_CHECKPOINT_FILE = re.compile(r"checkpoint-(\d{12,})\.json\Z")


class ExecutionContractError(ValueError):
    """An immutable execution artifact or transition is invalid."""


class ExecutionHold(RuntimeError):
    """Execution stopped because external prerequisites are absent."""

    def __init__(self, command: str, hold_codes: tuple[str, ...]) -> None:
        self.command = command
        self.hold_codes = hold_codes
        super().__init__(f"{command} held: {','.join(hold_codes)}")


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    attempt_id: str
    run_id: str
    created_at_utc: str
    plan_sha256: str
    matrix_sha256: str
    training_config_sha256: str
    source_tree_sha256: str
    schema: str = ATTEMPT_SCHEMA
    authority: int = AUTHORITY
    production: bool = PRODUCTION
    external_receipt_verified: bool = EXTERNAL_RECEIPT_VERIFIED
    result_claimed: bool = RESULT_CLAIMED


@dataclass(frozen=True, slots=True)
class HeartbeatRecord:
    attempt_id: str
    run_id: str
    sequence: int
    global_step: int
    observed_at_utc: str
    phase: str
    previous_heartbeat_sha256: str | None
    schema: str = HEARTBEAT_SCHEMA
    authority: int = AUTHORITY
    production: bool = PRODUCTION
    result_claimed: bool = RESULT_CLAIMED


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    attempt_id: str
    run_id: str
    epoch_index: int
    global_step: int
    written_at_utc: str
    model_state_sha256: str
    optimizer_state_sha256: str
    rng_state_sha256: str
    sampler_state_sha256: str
    dataloader_state_sha256: str
    dropout_state_sha256: str
    validation_state_sha256: str
    checkpoint_payload_sha256: str
    parent_checkpoint_receipt_sha256: str | None
    schema: str = CHECKPOINT_SCHEMA
    authority: int = AUTHORITY
    production: bool = PRODUCTION
    result_claimed: bool = RESULT_CLAIMED


@dataclass(frozen=True, slots=True)
class TerminalRecord:
    attempt_id: str
    run_id: str
    completed_at_utc: str
    outcome: str
    attempt_receipt_sha256: str
    latest_heartbeat_sha256: str | None
    latest_checkpoint_receipt_sha256: str | None
    failure_code: str | None
    schema: str = TERMINAL_SCHEMA
    authority: int = AUTHORITY
    production: bool = PRODUCTION
    external_receipt_verified: bool = EXTERNAL_RECEIPT_VERIFIED
    result_claimed: bool = RESULT_CLAIMED


@dataclass(frozen=True, slots=True)
class ResumeRecord:
    """A new-attempt link to one verified predecessor checkpoint."""

    new_attempt_id: str
    predecessor_attempt_id: str
    run_id: str
    created_at_utc: str
    predecessor_terminal_sha256: str
    checkpoint_receipt_sha256: str
    corrective_change_sha256: str
    retry_class: str
    schema: str = RESUME_SCHEMA
    authority: int = AUTHORITY
    production: bool = PRODUCTION
    external_receipt_verified: bool = EXTERNAL_RECEIPT_VERIFIED
    result_claimed: bool = RESULT_CLAIMED


@dataclass(frozen=True, slots=True)
class PeriodicCacheRecord:
    """Metadata binding one seed cache to qualification and base terminal."""

    seed: int
    qualified_base_system_id: str
    source_run_id: str
    created_at_utc: str
    base_terminal_sha256: str
    qualification_sha256: str
    cache_content_sha256: str
    schema: str = PERIODIC_CACHE_SCHEMA
    authority: int = AUTHORITY
    production: bool = PRODUCTION
    external_receipt_verified: bool = EXTERNAL_RECEIPT_VERIFIED
    result_claimed: bool = RESULT_CLAIMED


@dataclass(frozen=True, slots=True)
class PreflightReport:
    ready: bool
    hold_codes: tuple[str, ...]
    checked_commands: tuple[str, ...] = COMMANDS
    authority: int = AUTHORITY
    production: bool = PRODUCTION
    external_receipt_verified: bool = EXTERNAL_RECEIPT_VERIFIED
    result_claimed: bool = RESULT_CLAIMED
    status: str = STATUS


@dataclass(frozen=True, slots=True)
class CommandIntent:
    """A validated request identity, not an execution capability."""

    command: str
    run_id: str | None
    seed: int | None
    split: str | None
    plan_sha256: str
    sealed_test_consumption_sha256: str | None = None
    runtime_mode: str = "PUBLIC_HOLD"
    admission_sha256: str | None = None
    authority: int = AUTHORITY
    execution_authorized: bool = False
    production: bool = PRODUCTION
    result_claimed: bool = RESULT_CLAIMED


@dataclass(frozen=True, slots=True)
class RuntimeAdmissionRequest:
    """Public binding request passed to a host-injected runtime adapter."""

    plan_sha256: str
    matrix_sha256: str
    training_config_sha256: str
    handler_manifest_sha256: str
    commands: tuple[str, ...] = COMMANDS


@dataclass(frozen=True, slots=True)
class RuntimeAdmission:
    """Digest-only admission asserted by a host-injected runtime adapter.

    The public package validates the shape and bindings. It does not verify the
    authenticity of private receipts and never upgrades the assertion into a
    public claim of external verification.
    """

    runtime_mode: str
    admitted_at_utc: str
    adapter_sha256: str
    handler_manifest_sha256: str
    plan_sha256: str
    matrix_sha256: str
    training_config_sha256: str
    data_manifest_sha256: str
    prepared_data_manifest_sha256: str
    split_audit_sha256: str
    rights_assertion_sha256: str
    runtime_assertion_sha256: str
    execution_assertion_sha256: str
    authority: int
    execution_authorized: bool
    production: bool
    external_authentication_asserted: bool
    public_verification_performed: bool
    result_claimed: bool
    status: str
    schema: str = RUNTIME_ADMISSION_SCHEMA

    @property
    def ready(self) -> bool:
        """The injected mode is admitted for dispatch, not scientifically validated."""

        return True

    @property
    def hold_codes(self) -> tuple[str, ...]:
        return ()

    @property
    def external_receipt_verified(self) -> bool:
        """Public code never upgrades an adapter assertion into verification."""

        return False


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Closed, digest-only result returned by one injected command handler."""

    command: str
    outcome: str
    completed_at_utc: str
    run_id: str | None
    seed: int | None
    split: str | None
    artifact_sha256s: tuple[str, ...]
    admission_sha256: str
    runtime_mode: str
    authority: int
    production: bool
    sealed_test_consumption_sha256: str | None = None
    result_claimed: bool = RESULT_CLAIMED
    schema: str = COMMAND_RESULT_SCHEMA


class RuntimeAdapter(Protocol):
    """Host-injected adapter boundary; never loaded from a CLI path or endpoint."""

    commands: tuple[str, ...]
    adapter_sha256: str
    handler_manifest_sha256: str

    def admit(self, request: RuntimeAdmissionRequest) -> RuntimeAdmission:
        """Assert a digest-bound synthetic or private runtime admission."""

    def handle(self, intent: CommandIntent) -> CommandResult:
        """Execute one already-admitted command intent."""

    def consume_sealed_test(self, intent: CommandIntent) -> str:
        """Authenticate and durably consume the sole test grant."""


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


def _sha256(value: object, label: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if type(value) is not str:
        raise TypeError(f"{label} must be exact built-in str")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ExecutionContractError(f"{label} must be lowercase SHA-256 hex")
    return value


def _timestamp(value: object, label: str) -> str:
    if type(value) is not str or _UTC.fullmatch(value) is None:
        raise ExecutionContractError(f"{label} must be exact second-resolution UTC")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ExecutionContractError(f"{label} is not a real UTC timestamp") from exc
    return value


def _attempt_id(value: object) -> str:
    if type(value) is not str or _ATTEMPT_ID.fullmatch(value) is None:
        raise ExecutionContractError("attempt_id must match [a-z0-9][a-z0-9._-]{0,63}")
    if value in {".", ".."}:
        raise ExecutionContractError("attempt_id cannot be a path component alias")
    return value


def _exact_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0 or value > (1 << 63) - 1:
        raise ExecutionContractError(f"{label} must be an exact nonnegative 63-bit int")
    return value


def _artifact_header(record: object, schema: str) -> None:
    if getattr(record, "schema", None) != schema:
        raise ExecutionContractError(f"artifact schema must be {schema}")
    if (
        getattr(record, "authority", None) != 0
        or getattr(record, "production", None) is not False
        or getattr(record, "result_claimed", None) is not False
    ):
        raise ExecutionContractError("public artifacts must remain authority zero/no-result")


def runtime_handler_manifest_sha256(commands: object = COMMANDS) -> str:
    """Hash the exact closed command census implemented by an adapter."""

    if type(commands) is not tuple or commands != COMMANDS:
        raise ExecutionContractError("runtime adapter must cover the exact eleven commands")
    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "commands": list(commands),
                "schema": "phaseset-runtime-handler-manifest-v1",
            }
        )
    ).hexdigest()


def canonical_runtime_admission_bytes(record: object) -> bytes:
    """Validate a host assertion without claiming public receipt verification."""

    if type(record) is not RuntimeAdmission:
        raise TypeError("record must be exact RuntimeAdmission")
    if record.schema != RUNTIME_ADMISSION_SCHEMA:
        raise ExecutionContractError("runtime admission schema is invalid")
    _timestamp(record.admitted_at_utc, "admitted_at_utc")
    digest_fields = {
        "adapter_sha256": record.adapter_sha256,
        "data_manifest_sha256": record.data_manifest_sha256,
        "execution_assertion_sha256": record.execution_assertion_sha256,
        "handler_manifest_sha256": record.handler_manifest_sha256,
        "matrix_sha256": record.matrix_sha256,
        "plan_sha256": record.plan_sha256,
        "prepared_data_manifest_sha256": record.prepared_data_manifest_sha256,
        "rights_assertion_sha256": record.rights_assertion_sha256,
        "runtime_assertion_sha256": record.runtime_assertion_sha256,
        "split_audit_sha256": record.split_audit_sha256,
        "training_config_sha256": record.training_config_sha256,
    }
    for label, digest in digest_fields.items():
        _sha256(digest, label)
    if record.execution_authorized is not True or record.result_claimed is not False:
        raise ExecutionContractError("admission must authorize execution without claiming a result")
    if record.public_verification_performed is not False:
        raise ExecutionContractError("public code cannot claim private receipt verification")

    if record.runtime_mode == "SYNTHETIC_DATA_FREE":
        if (
            type(record.authority) is not int
            or record.authority != 0
            or record.production is not False
            or record.external_authentication_asserted is not False
            or record.status != "SYNTHETIC_DATA_FREE_NO_SCIENTIFIC_RESULT"
        ):
            raise ExecutionContractError("synthetic admission must remain authority zero/no-result")
        notice = "SYNTHETIC / NO SCIENTIFIC RESULT"
    elif record.runtime_mode == "PRIVATE_AUTHORIZED":
        if (
            type(record.authority) is not int
            or record.authority <= 0
            or record.production is not True
            or record.external_authentication_asserted is not True
            or record.status != "PRIVATE_ADAPTER_ASSERTED_AUTHORIZATION_NOT_PUBLICLY_VERIFIED"
        ):
            raise ExecutionContractError("private admission assertion is internally inconsistent")
        notice = "PRIVATE ADAPTER ASSERTION / NOT PUBLICLY VERIFIED / NO SCIENTIFIC RESULT YET"
    else:
        raise ExecutionContractError("runtime mode must be synthetic or private-authorized")

    return _canonical_json_bytes(
        {
            "adapter_sha256": record.adapter_sha256,
            "admitted_at_utc": record.admitted_at_utc,
            "authority": record.authority,
            "data_manifest_sha256": record.data_manifest_sha256,
            "execution_assertion_sha256": record.execution_assertion_sha256,
            "execution_authorized": record.execution_authorized,
            "external_authentication_asserted": (record.external_authentication_asserted),
            "handler_manifest_sha256": record.handler_manifest_sha256,
            "matrix_sha256": record.matrix_sha256,
            "notice": notice,
            "plan_sha256": record.plan_sha256,
            "prepared_data_manifest_sha256": record.prepared_data_manifest_sha256,
            "production": record.production,
            "public_verification_performed": record.public_verification_performed,
            "result_claimed": record.result_claimed,
            "rights_assertion_sha256": record.rights_assertion_sha256,
            "runtime_assertion_sha256": record.runtime_assertion_sha256,
            "runtime_mode": record.runtime_mode,
            "schema": record.schema,
            "split_audit_sha256": record.split_audit_sha256,
            "status": record.status,
            "training_config_sha256": record.training_config_sha256,
        }
    )


def runtime_admission_sha256(record: object) -> str:
    return hashlib.sha256(canonical_runtime_admission_bytes(record)).hexdigest()


def _validate_command_identity(
    command: object,
    run_id: object,
    seed: object,
    split: object,
) -> tuple[str, str | None, int | None, str | None]:
    if type(command) is not str or command not in COMMANDS:
        raise ExecutionContractError("command is outside the closed CLI census")
    checked_run: str | None = None
    checked_seed: int | None = None
    checked_split: str | None = None
    if command == "run-base":
        role, _, _ = parse_run_id(run_id)
        if role != "BASE_QUALIFICATION":
            raise ExecutionContractError("run-base requires a base run_id")
        checked_run = run_id
    elif command == "run-residual":
        role, _, _ = parse_run_id(run_id)
        if role != "RESIDUAL_TRAIN":
            raise ExecutionContractError("run-residual requires a residual run_id")
        checked_run = run_id
    elif command == "build-periodic-cache":
        if type(seed) is not int or seed not in {1729, 2718, 31415}:
            raise ExecutionContractError("periodic cache seed is outside the census")
        checked_seed = seed
    elif command == "evaluate":
        if type(split) is not str or split not in {"validation", "test"}:
            raise ExecutionContractError("evaluate split must be validation or test")
        checked_split = split
    if run_id != checked_run or seed != checked_seed or split != checked_split:
        raise ExecutionContractError("command carries fields outside its registered identity")
    return command, checked_run, checked_seed, checked_split


def canonical_command_result_bytes(record: object) -> bytes:
    """Validate and serialize a digest-only adapter result."""

    if type(record) is not CommandResult:
        raise TypeError("record must be exact CommandResult")
    if record.schema != COMMAND_RESULT_SCHEMA:
        raise ExecutionContractError("command result schema is invalid")
    _validate_command_identity(record.command, record.run_id, record.seed, record.split)
    if record.outcome not in {"COMPLETED", "FAILED", "HELD"}:
        raise ExecutionContractError("command result outcome is invalid")
    _timestamp(record.completed_at_utc, "completed_at_utc")
    if type(record.artifact_sha256s) is not tuple or not record.artifact_sha256s:
        raise ExecutionContractError("command result must bind at least one artifact")
    for digest in record.artifact_sha256s:
        _sha256(digest, "artifact_sha256")
    if len(set(record.artifact_sha256s)) != len(record.artifact_sha256s):
        raise ExecutionContractError("command result artifact digests must be unique")
    _sha256(record.admission_sha256, "admission_sha256")
    if record.command == "evaluate" and record.split == "test":
        _sha256(
            record.sealed_test_consumption_sha256,
            "sealed_test_consumption_sha256",
        )
    elif record.sealed_test_consumption_sha256 is not None:
        raise ExecutionContractError(
            "only sealed-test evaluation may bind a test consumption receipt"
        )
    if record.result_claimed is not False:
        raise ExecutionContractError("command result cannot claim a scientific result")

    if record.runtime_mode == "SYNTHETIC_DATA_FREE":
        if (
            type(record.authority) is not int
            or record.authority != 0
            or record.production is not False
        ):
            raise ExecutionContractError("synthetic result must remain authority zero")
        notice = "SYNTHETIC / NO SCIENTIFIC RESULT"
    elif record.runtime_mode == "PRIVATE_AUTHORIZED":
        if (
            type(record.authority) is not int
            or record.authority <= 0
            or record.production is not True
        ):
            raise ExecutionContractError("private result authority binding is invalid")
        notice = "PRIVATE ADAPTER OUTPUT / NOT PUBLICLY VERIFIED / NO SCIENTIFIC RESULT YET"
    else:
        raise ExecutionContractError("command result runtime mode is invalid")

    return _canonical_json_bytes(
        {
            "admission_sha256": record.admission_sha256,
            "artifact_sha256s": list(record.artifact_sha256s),
            "authority": record.authority,
            "command": record.command,
            "completed_at_utc": record.completed_at_utc,
            "notice": notice,
            "outcome": record.outcome,
            "production": record.production,
            "result_claimed": record.result_claimed,
            "run_id": record.run_id,
            "runtime_mode": record.runtime_mode,
            "schema": record.schema,
            "sealed_test_consumption_sha256": record.sealed_test_consumption_sha256,
            "seed": record.seed,
            "split": record.split,
        }
    )


def canonical_attempt_bytes(record: object) -> bytes:
    if type(record) is not AttemptRecord:
        raise TypeError("record must be exact AttemptRecord")
    _artifact_header(record, ATTEMPT_SCHEMA)
    attempt_id = _attempt_id(record.attempt_id)
    parse_run_id(record.run_id)
    _timestamp(record.created_at_utc, "created_at_utc")
    if record.external_receipt_verified is not False:
        raise ExecutionContractError("public code cannot assert an external receipt")
    payload = {
        "attempt_id": attempt_id,
        "authority": record.authority,
        "created_at_utc": record.created_at_utc,
        "external_receipt_verified": record.external_receipt_verified,
        "matrix_sha256": _sha256(record.matrix_sha256, "matrix_sha256"),
        "plan_sha256": _sha256(record.plan_sha256, "plan_sha256"),
        "production": record.production,
        "result_claimed": record.result_claimed,
        "run_id": record.run_id,
        "schema": record.schema,
        "source_tree_sha256": _sha256(record.source_tree_sha256, "source_tree_sha256"),
        "training_config_sha256": _sha256(record.training_config_sha256, "training_config_sha256"),
    }
    return _canonical_json_bytes(payload)


def canonical_heartbeat_bytes(record: object) -> bytes:
    if type(record) is not HeartbeatRecord:
        raise TypeError("record must be exact HeartbeatRecord")
    _artifact_header(record, HEARTBEAT_SCHEMA)
    _attempt_id(record.attempt_id)
    parse_run_id(record.run_id)
    _exact_nonnegative_int(record.sequence, "sequence")
    _exact_nonnegative_int(record.global_step, "global_step")
    _timestamp(record.observed_at_utc, "observed_at_utc")
    if record.phase not in {"STARTING", "RUNNING", "CHECKPOINTING", "EVALUATING"}:
        raise ExecutionContractError("heartbeat phase is outside the closed census")
    if record.sequence == 0 and record.previous_heartbeat_sha256 is not None:
        raise ExecutionContractError("first heartbeat cannot bind a predecessor")
    if record.sequence > 0 and record.previous_heartbeat_sha256 is None:
        raise ExecutionContractError("non-first heartbeat must bind its predecessor")
    return _canonical_json_bytes(
        {
            "attempt_id": record.attempt_id,
            "authority": record.authority,
            "global_step": record.global_step,
            "observed_at_utc": record.observed_at_utc,
            "phase": record.phase,
            "previous_heartbeat_sha256": _sha256(
                record.previous_heartbeat_sha256,
                "previous_heartbeat_sha256",
                optional=True,
            ),
            "production": record.production,
            "result_claimed": record.result_claimed,
            "run_id": record.run_id,
            "schema": record.schema,
            "sequence": record.sequence,
        }
    )


def canonical_checkpoint_bytes(record: object) -> bytes:
    if type(record) is not CheckpointRecord:
        raise TypeError("record must be exact CheckpointRecord")
    _artifact_header(record, CHECKPOINT_SCHEMA)
    _attempt_id(record.attempt_id)
    parse_run_id(record.run_id)
    _exact_nonnegative_int(record.epoch_index, "epoch_index")
    _exact_nonnegative_int(record.global_step, "global_step")
    _timestamp(record.written_at_utc, "written_at_utc")
    return _canonical_json_bytes(
        {
            "attempt_id": record.attempt_id,
            "authority": record.authority,
            "checkpoint_payload_sha256": _sha256(
                record.checkpoint_payload_sha256, "checkpoint_payload_sha256"
            ),
            "dataloader_state_sha256": _sha256(
                record.dataloader_state_sha256, "dataloader_state_sha256"
            ),
            "dropout_state_sha256": _sha256(record.dropout_state_sha256, "dropout_state_sha256"),
            "epoch_index": record.epoch_index,
            "global_step": record.global_step,
            "model_state_sha256": _sha256(record.model_state_sha256, "model_state_sha256"),
            "optimizer_state_sha256": _sha256(
                record.optimizer_state_sha256, "optimizer_state_sha256"
            ),
            "parent_checkpoint_receipt_sha256": _sha256(
                record.parent_checkpoint_receipt_sha256,
                "parent_checkpoint_receipt_sha256",
                optional=True,
            ),
            "production": record.production,
            "result_claimed": record.result_claimed,
            "rng_state_sha256": _sha256(record.rng_state_sha256, "rng_state_sha256"),
            "run_id": record.run_id,
            "sampler_state_sha256": _sha256(record.sampler_state_sha256, "sampler_state_sha256"),
            "schema": record.schema,
            "validation_state_sha256": _sha256(
                record.validation_state_sha256, "validation_state_sha256"
            ),
            "written_at_utc": record.written_at_utc,
        }
    )


def canonical_terminal_bytes(record: object) -> bytes:
    if type(record) is not TerminalRecord:
        raise TypeError("record must be exact TerminalRecord")
    _artifact_header(record, TERMINAL_SCHEMA)
    _attempt_id(record.attempt_id)
    parse_run_id(record.run_id)
    _timestamp(record.completed_at_utc, "completed_at_utc")
    if record.external_receipt_verified is not False:
        raise ExecutionContractError("public code cannot assert an external receipt")
    if record.outcome not in {"SUCCEEDED", "FAILED", "HELD"}:
        raise ExecutionContractError("terminal outcome is outside the closed census")
    if record.outcome == "SUCCEEDED":
        if record.failure_code is not None or record.latest_checkpoint_receipt_sha256 is None:
            raise ExecutionContractError("success requires a checkpoint and no failure code")
    else:
        if (
            type(record.failure_code) is not str
            or _FAILURE_CODE.fullmatch(record.failure_code) is None
        ):
            raise ExecutionContractError("failed/held terminal requires a canonical failure code")
    return _canonical_json_bytes(
        {
            "attempt_id": record.attempt_id,
            "attempt_receipt_sha256": _sha256(
                record.attempt_receipt_sha256, "attempt_receipt_sha256"
            ),
            "authority": record.authority,
            "completed_at_utc": record.completed_at_utc,
            "external_receipt_verified": record.external_receipt_verified,
            "failure_code": record.failure_code,
            "latest_checkpoint_receipt_sha256": _sha256(
                record.latest_checkpoint_receipt_sha256,
                "latest_checkpoint_receipt_sha256",
                optional=True,
            ),
            "latest_heartbeat_sha256": _sha256(
                record.latest_heartbeat_sha256,
                "latest_heartbeat_sha256",
                optional=True,
            ),
            "outcome": record.outcome,
            "production": record.production,
            "result_claimed": record.result_claimed,
            "run_id": record.run_id,
            "schema": record.schema,
        }
    )


def canonical_resume_bytes(record: object) -> bytes:
    if type(record) is not ResumeRecord:
        raise TypeError("record must be exact ResumeRecord")
    _artifact_header(record, RESUME_SCHEMA)
    new_attempt_id = _attempt_id(record.new_attempt_id)
    predecessor_attempt_id = _attempt_id(record.predecessor_attempt_id)
    if new_attempt_id == predecessor_attempt_id:
        raise ExecutionContractError("resume must create a distinct attempt_id")
    parse_run_id(record.run_id)
    _timestamp(record.created_at_utc, "created_at_utc")
    if record.external_receipt_verified is not False:
        raise ExecutionContractError("public code cannot assert an external receipt")
    if record.retry_class not in {
        "INFRA_TRANSIENT",
        "RESOURCE",
        "IMPLEMENTATION",
        "DATA",
    }:
        raise ExecutionContractError("retry_class is not eligible for automatic resume")
    return _canonical_json_bytes(
        {
            "authority": record.authority,
            "checkpoint_receipt_sha256": _sha256(
                record.checkpoint_receipt_sha256, "checkpoint_receipt_sha256"
            ),
            "corrective_change_sha256": _sha256(
                record.corrective_change_sha256, "corrective_change_sha256"
            ),
            "created_at_utc": record.created_at_utc,
            "external_receipt_verified": record.external_receipt_verified,
            "new_attempt_id": new_attempt_id,
            "predecessor_attempt_id": predecessor_attempt_id,
            "predecessor_terminal_sha256": _sha256(
                record.predecessor_terminal_sha256, "predecessor_terminal_sha256"
            ),
            "production": record.production,
            "result_claimed": record.result_claimed,
            "retry_class": record.retry_class,
            "run_id": record.run_id,
            "schema": record.schema,
        }
    )


def canonical_periodic_cache_bytes(record: object) -> bytes:
    if type(record) is not PeriodicCacheRecord:
        raise TypeError("record must be exact PeriodicCacheRecord")
    _artifact_header(record, PERIODIC_CACHE_SCHEMA)
    if type(record.seed) is not int or record.seed not in {1729, 2718, 31415}:
        raise ExecutionContractError("periodic cache seed is outside the census")
    if type(record.qualified_base_system_id) is not str or record.qualified_base_system_id not in {
        system_id for system_id, _ in BASE_SYSTEMS
    }:
        raise ExecutionContractError("periodic cache base winner is outside B0-B2")
    expected_run_id = (
        f"phaseset-run-v1/BASE_QUALIFICATION/{record.seed}/{record.qualified_base_system_id}"
    )
    if record.source_run_id != expected_run_id:
        raise ExecutionContractError("periodic cache source is not its seed-specific base")
    _timestamp(record.created_at_utc, "created_at_utc")
    if record.external_receipt_verified is not False:
        raise ExecutionContractError("public code cannot assert an external receipt")
    return _canonical_json_bytes(
        {
            "authority": record.authority,
            "base_terminal_sha256": _sha256(record.base_terminal_sha256, "base_terminal_sha256"),
            "cache_content_sha256": _sha256(record.cache_content_sha256, "cache_content_sha256"),
            "created_at_utc": record.created_at_utc,
            "external_receipt_verified": record.external_receipt_verified,
            "production": record.production,
            "qualified_base_system_id": record.qualified_base_system_id,
            "qualification_sha256": _sha256(record.qualification_sha256, "qualification_sha256"),
            "result_claimed": record.result_claimed,
            "schema": record.schema,
            "seed": record.seed,
            "source_run_id": record.source_run_id,
        }
    )


def _pairs_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ExecutionContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_float(value: str) -> object:
    raise ExecutionContractError(f"floating-point JSON is forbidden: {value}")


def _parse_canonical(raw: object, label: str) -> dict[str, object]:
    if type(raw) is not bytes or not raw.endswith(b"\n") or b"\r" in raw:
        raise ExecutionContractError(f"{label} must be LF-terminated canonical bytes")
    try:
        parsed = json.loads(
            raw.decode("ascii", "strict"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_float=_reject_float,
            parse_constant=_reject_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExecutionContractError(f"{label} is not strict ASCII JSON") from exc
    if type(parsed) is not dict or _canonical_json_bytes(parsed) != raw:
        raise ExecutionContractError(f"{label} bytes are not canonical")
    return parsed


def parse_attempt_bytes(raw: object) -> AttemptRecord:
    value = _parse_canonical(raw, "attempt")
    try:
        record = AttemptRecord(**value)
    except TypeError as exc:
        raise ExecutionContractError("attempt keys are not closed") from exc
    if canonical_attempt_bytes(record) != raw:
        raise ExecutionContractError("attempt semantics changed after parsing")
    return record


def parse_heartbeat_bytes(raw: object) -> HeartbeatRecord:
    value = _parse_canonical(raw, "heartbeat")
    try:
        record = HeartbeatRecord(**value)
    except TypeError as exc:
        raise ExecutionContractError("heartbeat keys are not closed") from exc
    if canonical_heartbeat_bytes(record) != raw:
        raise ExecutionContractError("heartbeat semantics changed after parsing")
    return record


def parse_checkpoint_bytes(raw: object) -> CheckpointRecord:
    value = _parse_canonical(raw, "checkpoint")
    try:
        record = CheckpointRecord(**value)
    except TypeError as exc:
        raise ExecutionContractError("checkpoint keys are not closed") from exc
    if canonical_checkpoint_bytes(record) != raw:
        raise ExecutionContractError("checkpoint semantics changed after parsing")
    return record


def parse_terminal_bytes(raw: object) -> TerminalRecord:
    value = _parse_canonical(raw, "terminal")
    try:
        record = TerminalRecord(**value)
    except TypeError as exc:
        raise ExecutionContractError("terminal keys are not closed") from exc
    if canonical_terminal_bytes(record) != raw:
        raise ExecutionContractError("terminal semantics changed after parsing")
    return record


def parse_resume_bytes(raw: object) -> ResumeRecord:
    value = _parse_canonical(raw, "resume")
    try:
        record = ResumeRecord(**value)
    except TypeError as exc:
        raise ExecutionContractError("resume keys are not closed") from exc
    if canonical_resume_bytes(record) != raw:
        raise ExecutionContractError("resume semantics changed after parsing")
    return record


def parse_periodic_cache_bytes(raw: object) -> PeriodicCacheRecord:
    value = _parse_canonical(raw, "periodic cache")
    try:
        record = PeriodicCacheRecord(**value)
    except TypeError as exc:
        raise ExecutionContractError("periodic cache keys are not closed") from exc
    if canonical_periodic_cache_bytes(record) != raw:
        raise ExecutionContractError("periodic cache semantics changed after parsing")
    return record


def parse_runtime_admission_bytes(raw: object) -> RuntimeAdmission:
    value = _parse_canonical(raw, "runtime admission")
    value.pop("notice", None)
    try:
        record = RuntimeAdmission(**value)
    except TypeError as exc:
        raise ExecutionContractError("runtime admission keys are not closed") from exc
    if canonical_runtime_admission_bytes(record) != raw:
        raise ExecutionContractError("runtime admission semantics changed after parsing")
    return record


def parse_command_result_bytes(raw: object) -> CommandResult:
    value = _parse_canonical(raw, "command result")
    value.pop("notice", None)
    if type(value.get("artifact_sha256s")) is list:
        value["artifact_sha256s"] = tuple(value["artifact_sha256s"])
    try:
        record = CommandResult(**value)
    except TypeError as exc:
        raise ExecutionContractError("command result keys are not closed") from exc
    if canonical_command_result_bytes(record) != raw:
        raise ExecutionContractError("command result semantics changed after parsing")
    return record


def artifact_sha256(raw: object) -> str:
    if type(raw) is not bytes:
        raise TypeError("raw must be exact built-in bytes")
    return hashlib.sha256(raw).hexdigest()


def _write_once(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise ExecutionContractError(f"write-once artifact already exists: {path.name}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _read_regular(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ExecutionContractError(f"{label} must be an existing non-symlink file")
    return path.read_bytes()


def _heartbeat_artifacts(
    directory: Path,
) -> tuple[tuple[HeartbeatRecord, Path, bytes], ...]:
    if directory.is_symlink() or not directory.is_dir():
        raise ExecutionContractError("heartbeat ledger must be a non-symlink directory")
    rows: list[tuple[HeartbeatRecord, Path, bytes]] = []
    for path in directory.iterdir():
        match = _HEARTBEAT_FILE.fullmatch(path.name)
        if match is None:
            raise ExecutionContractError("heartbeat directory contains an unexpected entry")
        raw = _read_regular(path, "heartbeat artifact")
        record = parse_heartbeat_bytes(raw)
        if int(match.group(1)) != record.sequence or path.name != (
            f"heartbeat-{record.sequence:08d}.json"
        ):
            raise ExecutionContractError("heartbeat filename does not bind its sequence")
        rows.append((record, path, raw))
    rows.sort(key=lambda item: item[0].sequence)
    if tuple(row[0].sequence for row in rows) != tuple(range(len(rows))):
        raise ExecutionContractError("heartbeat ledger is not contiguous")
    for ordinal, (record, _, raw) in enumerate(rows):
        if ordinal == 0:
            if record.previous_heartbeat_sha256 is not None:
                raise ExecutionContractError("first heartbeat cannot bind a predecessor")
            continue
        previous_record, _, previous_raw = rows[ordinal - 1]
        if record.global_step < previous_record.global_step:
            raise ExecutionContractError("heartbeat global_step cannot move backward")
        if record.previous_heartbeat_sha256 != artifact_sha256(previous_raw):
            raise ExecutionContractError("heartbeat predecessor digest mismatch")
        if artifact_sha256(raw) == artifact_sha256(previous_raw):
            raise ExecutionContractError("heartbeat ledger contains duplicate artifacts")
    return tuple(rows)


def _checkpoint_artifacts(
    directory: Path,
) -> tuple[tuple[CheckpointRecord, Path, bytes], ...]:
    if directory.is_symlink() or not directory.is_dir():
        raise ExecutionContractError("checkpoint ledger must be a non-symlink directory")
    rows: list[tuple[CheckpointRecord, Path, bytes]] = []
    for path in directory.iterdir():
        match = _CHECKPOINT_FILE.fullmatch(path.name)
        if match is None:
            raise ExecutionContractError("checkpoint directory contains an unexpected entry")
        raw = _read_regular(path, "checkpoint artifact")
        record = parse_checkpoint_bytes(raw)
        if int(match.group(1)) != record.global_step or path.name != (
            f"checkpoint-{record.global_step:012d}.json"
        ):
            raise ExecutionContractError("checkpoint filename does not bind global_step")
        rows.append((record, path, raw))
    rows.sort(key=lambda item: item[0].global_step)
    if len({row[0].global_step for row in rows}) != len(rows):
        raise ExecutionContractError("checkpoint ledger contains a duplicate global_step")
    for ordinal, (record, _, _) in enumerate(rows):
        if ordinal == 0:
            if record.parent_checkpoint_receipt_sha256 is not None:
                raise ExecutionContractError("first checkpoint cannot bind a parent")
            continue
        previous_raw = rows[ordinal - 1][2]
        if record.parent_checkpoint_receipt_sha256 != artifact_sha256(previous_raw):
            raise ExecutionContractError("checkpoint parent digest mismatch")
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class _VerifiedAttemptLedger:
    attempt: AttemptRecord
    attempt_raw: bytes
    heartbeats: tuple[tuple[HeartbeatRecord, Path, bytes], ...]
    checkpoints: tuple[tuple[CheckpointRecord, Path, bytes], ...]
    terminal: TerminalRecord | None
    terminal_raw: bytes | None
    resume: ResumeRecord | None
    resume_raw: bytes | None


@dataclass(frozen=True, slots=True, repr=False)
class CompletedAttemptEvidence:
    """Owned canonical evidence for one fully verified successful attempt.

    The records and byte strings come from the same complete attempt-chain
    verification pass. ``latest_checkpoint`` is the receipt bound by the
    terminal; it is not a validation-best selection. A consumer that needs the
    validation-best payload must load the payload identified by
    ``latest_checkpoint.checkpoint_payload_sha256``, validate that payload's
    best-checkpoint name/digest pointer, and authenticate the separately
    materialized best payload against that digest.
    """

    attempt: AttemptRecord
    attempt_raw: bytes
    latest_checkpoint: CheckpointRecord
    latest_checkpoint_raw: bytes
    terminal: TerminalRecord
    terminal_raw: bytes
    resume: ResumeRecord | None
    resume_raw: bytes | None

    def __post_init__(self) -> None:
        if (
            type(self.attempt) is not AttemptRecord
            or type(self.latest_checkpoint) is not CheckpointRecord
            or type(self.terminal) is not TerminalRecord
            or type(self.attempt_raw) is not bytes
            or type(self.latest_checkpoint_raw) is not bytes
            or type(self.terminal_raw) is not bytes
        ):
            raise TypeError("completed attempt evidence fields have the wrong exact type")
        if (self.resume is None) != (self.resume_raw is None):
            raise ExecutionContractError("completed attempt resume record/bytes are incomplete")
        if self.resume is not None and (
            type(self.resume) is not ResumeRecord or type(self.resume_raw) is not bytes
        ):
            raise TypeError("completed attempt resume fields have the wrong exact type")
        if (
            parse_attempt_bytes(self.attempt_raw) != self.attempt
            or parse_checkpoint_bytes(self.latest_checkpoint_raw)
            != self.latest_checkpoint
            or parse_terminal_bytes(self.terminal_raw) != self.terminal
            or (
                self.resume is not None
                and parse_resume_bytes(self.resume_raw) != self.resume
            )
        ):
            raise ExecutionContractError("completed attempt records differ from owned bytes")
        identity = (self.attempt.attempt_id, self.attempt.run_id)
        if (
            (self.latest_checkpoint.attempt_id, self.latest_checkpoint.run_id)
            != identity
            or (self.terminal.attempt_id, self.terminal.run_id) != identity
            or self.terminal.attempt_receipt_sha256
            != artifact_sha256(self.attempt_raw)
            or self.terminal.outcome != "SUCCEEDED"
            or self.terminal.failure_code is not None
            or self.terminal.latest_checkpoint_receipt_sha256
            != artifact_sha256(self.latest_checkpoint_raw)
            or (
                self.resume is not None
                and (
                    (self.resume.new_attempt_id, self.resume.run_id) != identity
                    or self.resume.created_at_utc != self.attempt.created_at_utc
                )
            )
        ):
            raise ExecutionContractError("completed attempt evidence identity is inconsistent")

    @property
    def terminal_sha256(self) -> str:
        return artifact_sha256(self.terminal_raw)

    @property
    def latest_checkpoint_receipt_sha256(self) -> str:
        return artifact_sha256(self.latest_checkpoint_raw)

    @property
    def latest_checkpoint_payload_sha256(self) -> str:
        return self.latest_checkpoint.checkpoint_payload_sha256


def _attempt_root(root: str | Path) -> Path:
    checked = Path(root)
    if checked.is_symlink() or not checked.is_dir():
        raise ExecutionContractError("attempt root must be an existing non-symlink directory")
    return checked


def _verify_local_attempt(path: Path, expected_attempt_id: str) -> _VerifiedAttemptLedger:
    if path.is_symlink() or not path.is_dir():
        raise ExecutionContractError("attempt directory is absent or a symlink")
    attempt_raw = _read_regular(path / "attempt.json", "attempt artifact")
    attempt = parse_attempt_bytes(attempt_raw)
    if attempt.attempt_id != expected_attempt_id:
        raise ExecutionContractError("attempt directory does not match its receipt")

    heartbeats = _heartbeat_artifacts(path / "heartbeats")
    checkpoints = _checkpoint_artifacts(path / "checkpoints")
    for record, _, _ in heartbeats:
        if (record.attempt_id, record.run_id) != (attempt.attempt_id, attempt.run_id):
            raise ExecutionContractError("heartbeat identity does not match attempt")
    for record, _, _ in checkpoints:
        if (record.attempt_id, record.run_id) != (attempt.attempt_id, attempt.run_id):
            raise ExecutionContractError("checkpoint identity does not match attempt")

    terminal_path = path / "terminal.json"
    terminal: TerminalRecord | None = None
    terminal_raw: bytes | None = None
    if terminal_path.exists() or terminal_path.is_symlink():
        terminal_raw = _read_regular(terminal_path, "terminal artifact")
        terminal = parse_terminal_bytes(terminal_raw)
        if (terminal.attempt_id, terminal.run_id) != (attempt.attempt_id, attempt.run_id):
            raise ExecutionContractError("terminal identity does not match attempt")
        if terminal.attempt_receipt_sha256 != artifact_sha256(attempt_raw):
            raise ExecutionContractError("terminal attempt digest mismatch")
        expected_heartbeat = artifact_sha256(heartbeats[-1][2]) if heartbeats else None
        if terminal.latest_heartbeat_sha256 != expected_heartbeat:
            raise ExecutionContractError("terminal latest heartbeat digest mismatch")
        expected_checkpoint = artifact_sha256(checkpoints[-1][2]) if checkpoints else None
        if terminal.latest_checkpoint_receipt_sha256 != expected_checkpoint:
            raise ExecutionContractError("terminal latest checkpoint digest mismatch")

    resume_path = path / "resume.json"
    resume: ResumeRecord | None = None
    resume_raw: bytes | None = None
    if resume_path.exists() or resume_path.is_symlink():
        resume_raw = _read_regular(resume_path, "resume artifact")
        resume = parse_resume_bytes(resume_raw)
        if (
            resume.new_attempt_id != attempt.attempt_id
            or resume.run_id != attempt.run_id
            or resume.created_at_utc != attempt.created_at_utc
        ):
            raise ExecutionContractError("resume identity does not match its new attempt")

    return _VerifiedAttemptLedger(
        attempt=attempt,
        attempt_raw=attempt_raw,
        heartbeats=heartbeats,
        checkpoints=checkpoints,
        terminal=terminal,
        terminal_raw=terminal_raw,
        resume=resume,
        resume_raw=resume_raw,
    )


def _validate_resume_source(
    resume: ResumeRecord,
    predecessor: _VerifiedAttemptLedger,
) -> CheckpointRecord:
    canonical_resume_bytes(resume)
    if predecessor.attempt.attempt_id != resume.predecessor_attempt_id:
        raise ExecutionContractError("resume predecessor identity mismatch")
    terminal = predecessor.terminal
    terminal_raw = predecessor.terminal_raw
    if terminal is None or terminal_raw is None:
        raise ExecutionContractError("resume predecessor has no immutable terminal")
    if terminal.outcome == "SUCCEEDED":
        raise ExecutionContractError("successful attempts cannot be resumed")
    if terminal.run_id != resume.run_id:
        raise ExecutionContractError("resume run_id differs from predecessor")
    if artifact_sha256(terminal_raw) != resume.predecessor_terminal_sha256:
        raise ExecutionContractError("resume predecessor terminal digest mismatch")
    if terminal.latest_checkpoint_receipt_sha256 != resume.checkpoint_receipt_sha256:
        raise ExecutionContractError("resume checkpoint does not match predecessor terminal")
    if not predecessor.checkpoints:
        raise ExecutionContractError("resume predecessor has no verified checkpoint")
    checkpoint, _, checkpoint_raw = predecessor.checkpoints[-1]
    if artifact_sha256(checkpoint_raw) != resume.checkpoint_receipt_sha256:
        raise ExecutionContractError("resume checkpoint receipt is absent from predecessor")
    return checkpoint


def _verify_attempt_chain(
    root: Path,
    attempt_id: str,
    visited: frozenset[str] = frozenset(),
    *,
    remaining_attempts: int = MAX_VERIFIED_ATTEMPT_CHAIN_LENGTH,
) -> _VerifiedAttemptLedger:
    if (
        type(remaining_attempts) is not int
        or remaining_attempts < 1
        or remaining_attempts > MAX_VERIFIED_ATTEMPT_CHAIN_LENGTH
    ):
        raise ExecutionContractError("resume predecessor chain exceeds its closed attempt bound")
    if attempt_id in visited:
        raise ExecutionContractError("resume predecessor chain contains a cycle")
    ledger = _verify_local_attempt(root / attempt_id, attempt_id)
    if ledger.resume is not None:
        predecessor = _verify_attempt_chain(
            root,
            ledger.resume.predecessor_attempt_id,
            visited | {attempt_id},
            remaining_attempts=remaining_attempts - 1,
        )
        _validate_resume_source(ledger.resume, predecessor)
        if (
            ledger.attempt.plan_sha256 != predecessor.attempt.plan_sha256
            or ledger.attempt.matrix_sha256 != predecessor.attempt.matrix_sha256
            or ledger.attempt.training_config_sha256
            != predecessor.attempt.training_config_sha256
        ):
            raise ExecutionContractError("resumed attempt changed a frozen plan binding")
    return ledger


def verified_completed_attempt_evidence(
    root: str | Path,
    attempt_id: object,
) -> CompletedAttemptEvidence:
    """Return owned records/bytes after one complete successful-chain check.

    No ledger path is reread after ``_verify_attempt_chain`` returns. The
    returned latest checkpoint is only the terminal-bound resume/progress
    checkpoint. Runtime-specific best-checkpoint pointers live inside that
    checkpoint payload and remain the responsibility of the consumer.
    """

    checked_id = _attempt_id(attempt_id)
    checked_root = _attempt_root(root)
    ledger = _verify_attempt_chain(checked_root, checked_id)
    terminal = ledger.terminal
    terminal_raw = ledger.terminal_raw
    if terminal is None or terminal_raw is None:
        raise ExecutionContractError("completed attempt has no immutable terminal")
    if terminal.outcome != "SUCCEEDED":
        raise ExecutionContractError("completed attempt terminal is not SUCCEEDED")
    if terminal.failure_code is not None:
        raise ExecutionContractError("successful completed attempt cannot carry a failure code")
    if not ledger.checkpoints:
        raise ExecutionContractError("successful completed attempt has no checkpoint receipt")
    latest_checkpoint, _, latest_checkpoint_raw = ledger.checkpoints[-1]
    return CompletedAttemptEvidence(
        attempt=ledger.attempt,
        attempt_raw=ledger.attempt_raw,
        latest_checkpoint=latest_checkpoint,
        latest_checkpoint_raw=latest_checkpoint_raw,
        terminal=terminal,
        terminal_raw=terminal_raw,
        resume=ledger.resume,
        resume_raw=ledger.resume_raw,
    )


def verified_resume_checkpoint_payload_sha256(
    root: str | Path,
    resume: object,
) -> str:
    """Return the payload digest only after revalidating the complete resume chain."""

    if type(resume) is not ResumeRecord:
        raise TypeError("resume must be exact ResumeRecord")
    resume_raw = canonical_resume_bytes(resume)
    checked_root = _attempt_root(root)
    current = _verify_attempt_chain(checked_root, resume.new_attempt_id)
    if current.resume is None or current.resume_raw != resume_raw:
        raise ExecutionContractError("stored resume record differs from the requested binding")
    predecessor = _verify_attempt_chain(checked_root, resume.predecessor_attempt_id)
    return _validate_resume_source(resume, predecessor).checkpoint_payload_sha256


class AttemptStore:
    """Write-once, hash-chained local metadata for one attempt directory."""

    __slots__ = ("_attempt", "_path")

    def __init__(self, path: Path, attempt: AttemptRecord) -> None:
        self._path = path
        self._attempt = attempt

    @classmethod
    def create(cls, root: str | Path, record: object) -> AttemptStore:
        if type(record) is not AttemptRecord:
            raise TypeError("record must be exact AttemptRecord")
        raw = canonical_attempt_bytes(record)
        checked_root = _attempt_root(root)
        path = checked_root / record.attempt_id
        path.mkdir(mode=0o700, exist_ok=False)
        (path / "heartbeats").mkdir(mode=0o700)
        (path / "checkpoints").mkdir(mode=0o700)
        _write_once(path / "attempt.json", raw)
        return cls(path, record)

    @classmethod
    def open(cls, root: str | Path, attempt_id: object) -> AttemptStore:
        checked_id = _attempt_id(attempt_id)
        checked_root = _attempt_root(root)
        ledger = _verify_attempt_chain(checked_root, checked_id)
        return cls(checked_root / checked_id, ledger.attempt)

    @classmethod
    def create_resumed(
        cls,
        root: str | Path,
        new_attempt: object,
        resume: object,
    ) -> AttemptStore:
        """Create a distinct attempt after verifying an immutable predecessor."""

        if type(new_attempt) is not AttemptRecord:
            raise TypeError("new_attempt must be exact AttemptRecord")
        if type(resume) is not ResumeRecord:
            raise TypeError("resume must be exact ResumeRecord")
        resume_raw = canonical_resume_bytes(resume)
        if (
            new_attempt.attempt_id != resume.new_attempt_id
            or new_attempt.run_id != resume.run_id
            or new_attempt.created_at_utc != resume.created_at_utc
        ):
            raise ExecutionContractError("new attempt identity does not match resume record")
        checked_root = _attempt_root(root)
        predecessor = _verify_attempt_chain(
            checked_root,
            resume.predecessor_attempt_id,
        )
        _validate_resume_source(resume, predecessor)
        if (
            new_attempt.plan_sha256 != predecessor.attempt.plan_sha256
            or new_attempt.matrix_sha256 != predecessor.attempt.matrix_sha256
            or new_attempt.training_config_sha256
            != predecessor.attempt.training_config_sha256
        ):
            raise ExecutionContractError("resumed attempt changed a frozen plan binding")
        created = cls.create(checked_root, new_attempt)
        _write_once(created.path / "resume.json", resume_raw)
        verified = _verify_attempt_chain(checked_root, new_attempt.attempt_id)
        return cls(created.path, verified.attempt)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def attempt(self) -> AttemptRecord:
        return self._attempt

    def _require_open(self) -> None:
        terminal_path = self._path / "terminal.json"
        if terminal_path.exists() or terminal_path.is_symlink():
            raise ExecutionContractError("attempt is terminal and cannot be mutated")

    def write_heartbeat(self, record: object) -> str:
        if type(record) is not HeartbeatRecord:
            raise TypeError("record must be exact HeartbeatRecord")
        self._require_open()
        if (record.attempt_id, record.run_id) != (
            self._attempt.attempt_id,
            self._attempt.run_id,
        ):
            raise ExecutionContractError("heartbeat identity does not match attempt")
        directory = self._path / "heartbeats"
        if directory.is_symlink():
            raise ExecutionContractError("heartbeat directory cannot be a symlink")
        existing = _heartbeat_artifacts(directory)
        if record.sequence != len(existing):
            raise ExecutionContractError("heartbeat sequence must be contiguous")
        if existing:
            previous, _, previous_raw = existing[-1]
            if record.global_step < previous.global_step:
                raise ExecutionContractError("heartbeat global_step cannot move backward")
            if record.previous_heartbeat_sha256 != artifact_sha256(previous_raw):
                raise ExecutionContractError("heartbeat predecessor digest mismatch")
        elif record.previous_heartbeat_sha256 is not None:
            raise ExecutionContractError("first heartbeat cannot bind a predecessor")
        raw = canonical_heartbeat_bytes(record)
        path = directory / f"heartbeat-{record.sequence:08d}.json"
        _write_once(path, raw)
        return artifact_sha256(raw)

    def write_checkpoint(self, record: object) -> str:
        if type(record) is not CheckpointRecord:
            raise TypeError("record must be exact CheckpointRecord")
        self._require_open()
        if (record.attempt_id, record.run_id) != (
            self._attempt.attempt_id,
            self._attempt.run_id,
        ):
            raise ExecutionContractError("checkpoint identity does not match attempt")
        directory = self._path / "checkpoints"
        if directory.is_symlink():
            raise ExecutionContractError("checkpoint directory cannot be a symlink")
        existing = _checkpoint_artifacts(directory)
        if existing:
            previous, _, previous_raw = existing[-1]
            if record.global_step <= previous.global_step:
                raise ExecutionContractError("checkpoint global_step must strictly increase")
            if record.parent_checkpoint_receipt_sha256 != artifact_sha256(previous_raw):
                raise ExecutionContractError("checkpoint parent digest mismatch")
        elif record.parent_checkpoint_receipt_sha256 is not None:
            raise ExecutionContractError("first checkpoint cannot bind a parent")
        raw = canonical_checkpoint_bytes(record)
        path = directory / f"checkpoint-{record.global_step:012d}.json"
        _write_once(path, raw)
        return artifact_sha256(raw)

    def write_terminal(self, record: object) -> str:
        if type(record) is not TerminalRecord:
            raise TypeError("record must be exact TerminalRecord")
        self._require_open()
        if (record.attempt_id, record.run_id) != (
            self._attempt.attempt_id,
            self._attempt.run_id,
        ):
            raise ExecutionContractError("terminal identity does not match attempt")
        attempt_raw = _read_regular(self._path / "attempt.json", "attempt artifact")
        if record.attempt_receipt_sha256 != artifact_sha256(attempt_raw):
            raise ExecutionContractError("terminal attempt digest mismatch")

        heartbeats = _heartbeat_artifacts(self._path / "heartbeats")
        expected_heartbeat = artifact_sha256(heartbeats[-1][2]) if heartbeats else None
        if record.latest_heartbeat_sha256 != expected_heartbeat:
            raise ExecutionContractError("terminal latest heartbeat digest mismatch")

        checkpoints = _checkpoint_artifacts(self._path / "checkpoints")
        expected_checkpoint = artifact_sha256(checkpoints[-1][2]) if checkpoints else None
        if record.latest_checkpoint_receipt_sha256 != expected_checkpoint:
            raise ExecutionContractError("terminal latest checkpoint digest mismatch")
        raw = canonical_terminal_bytes(record)
        _write_once(self._path / "terminal.json", raw)
        return artifact_sha256(raw)


class SealedTestGate:
    """Persist one authority-zero consumption; it does not issue a test grant."""

    __slots__ = ("_root",)

    def __init__(self, ledger_root: str | Path) -> None:
        root = Path(ledger_root)
        if not root.is_dir() or root.is_symlink():
            raise ExecutionContractError("sealed-test ledger root must be a non-symlink directory")
        self._root = root

    @property
    def consumed(self) -> bool:
        path = self._root / "sealed-test-consumption.json"
        return path.exists() or path.is_symlink()

    @property
    def evaluation_bound(self) -> bool:
        path = self._root / "sealed-test-evaluation.json"
        return path.exists() or path.is_symlink()

    @staticmethod
    def _consumption_bytes(
        *,
        external_grant_sha256: object,
        test_manifest_sha256: object,
        caption_manifest_sha256: object,
        evaluator_sha256: object,
        validation_freeze_sha256: object,
        aggregate_code_sha256: object,
        evaluation_census_sha256: object,
        hard_gallery_freeze_binding_sha256: object,
        hard_gallery_collection_sha256: object,
        consumed_at_utc: object,
    ) -> bytes:
        return _canonical_json_bytes(
            {
                "aggregate_code_sha256": _sha256(
                    aggregate_code_sha256, "aggregate_code_sha256"
                ),
                "authority": AUTHORITY,
                "caption_manifest_sha256": _sha256(
                    caption_manifest_sha256, "caption_manifest_sha256"
                ),
                "consumed_at_utc": _timestamp(consumed_at_utc, "consumed_at_utc"),
                "evaluator_sha256": _sha256(evaluator_sha256, "evaluator_sha256"),
                "external_grant_sha256": _sha256(
                    external_grant_sha256, "external_grant_sha256"
                ),
                "external_receipt_verified": False,
                "evaluation_census_sha256": _sha256(
                    evaluation_census_sha256,
                    "evaluation_census_sha256",
                ),
                "hard_gallery_collection_sha256": _sha256(
                    hard_gallery_collection_sha256,
                    "hard_gallery_collection_sha256",
                ),
                "hard_gallery_freeze_binding_sha256": _sha256(
                    hard_gallery_freeze_binding_sha256,
                    "hard_gallery_freeze_binding_sha256",
                ),
                "production": PRODUCTION,
                "result_claimed": RESULT_CLAIMED,
                "schema": SEALED_TEST_SCHEMA,
                "status": "LOCAL_SINGLE_CONSUMPTION_ONLY_NOT_AN_AUTHORITY_GRANT",
                "test_manifest_sha256": _sha256(
                    test_manifest_sha256, "test_manifest_sha256"
                ),
                "validation_freeze_sha256": _sha256(
                    validation_freeze_sha256, "validation_freeze_sha256"
                ),
            }
        )

    def consume(
        self,
        *,
        external_grant_sha256: object,
        test_manifest_sha256: object,
        caption_manifest_sha256: object,
        evaluator_sha256: object,
        validation_freeze_sha256: object,
        aggregate_code_sha256: object,
        evaluation_census_sha256: object,
        hard_gallery_freeze_binding_sha256: object,
        hard_gallery_collection_sha256: object,
        consumed_at_utc: object,
    ) -> str:
        """Record one use; all inputs remain unverified external assertions."""

        raw = self._consumption_bytes(
            external_grant_sha256=external_grant_sha256,
            test_manifest_sha256=test_manifest_sha256,
            caption_manifest_sha256=caption_manifest_sha256,
            evaluator_sha256=evaluator_sha256,
            validation_freeze_sha256=validation_freeze_sha256,
            aggregate_code_sha256=aggregate_code_sha256,
            evaluation_census_sha256=evaluation_census_sha256,
            hard_gallery_freeze_binding_sha256=hard_gallery_freeze_binding_sha256,
            hard_gallery_collection_sha256=hard_gallery_collection_sha256,
            consumed_at_utc=consumed_at_utc,
        )
        _write_once(self._root / "sealed-test-consumption.json", raw)
        return artifact_sha256(raw)

    def verify_consumption(
        self,
        *,
        external_grant_sha256: object,
        test_manifest_sha256: object,
        caption_manifest_sha256: object,
        evaluator_sha256: object,
        validation_freeze_sha256: object,
        aggregate_code_sha256: object,
        evaluation_census_sha256: object,
        hard_gallery_freeze_binding_sha256: object,
        hard_gallery_collection_sha256: object,
        consumed_at_utc: object,
    ) -> str:
        expected = self._consumption_bytes(
            external_grant_sha256=external_grant_sha256,
            test_manifest_sha256=test_manifest_sha256,
            caption_manifest_sha256=caption_manifest_sha256,
            evaluator_sha256=evaluator_sha256,
            validation_freeze_sha256=validation_freeze_sha256,
            aggregate_code_sha256=aggregate_code_sha256,
            evaluation_census_sha256=evaluation_census_sha256,
            hard_gallery_freeze_binding_sha256=hard_gallery_freeze_binding_sha256,
            hard_gallery_collection_sha256=hard_gallery_collection_sha256,
            consumed_at_utc=consumed_at_utc,
        )
        observed = _read_regular(
            self._root / "sealed-test-consumption.json",
            "sealed-test consumption artifact",
        )
        if observed != expected:
            raise ExecutionContractError(
                "sealed-test consumption differs from authenticated authorization"
            )
        return artifact_sha256(observed)

    def bind_evaluation(
        self,
        *,
        consumption_sha256: object,
        evaluation_aggregate_sha256: object,
        evaluator_sha256: object,
        aggregate_code_sha256: object,
        completed_at_utc: object,
    ) -> str:
        consumption_raw = _read_regular(
            self._root / "sealed-test-consumption.json",
            "sealed-test consumption artifact",
        )
        checked_consumption = _sha256(
            consumption_sha256, "sealed_test_consumption_sha256"
        )
        if artifact_sha256(consumption_raw) != checked_consumption:
            raise ExecutionContractError("sealed-test evaluation consumption digest mismatch")
        raw = _canonical_json_bytes(
            {
                "aggregate_code_sha256": _sha256(
                    aggregate_code_sha256, "aggregate_code_sha256"
                ),
                "authority": AUTHORITY,
                "completed_at_utc": _timestamp(completed_at_utc, "completed_at_utc"),
                "evaluation_aggregate_sha256": _sha256(
                    evaluation_aggregate_sha256,
                    "evaluation_aggregate_sha256",
                ),
                "evaluator_sha256": _sha256(evaluator_sha256, "evaluator_sha256"),
                "production": PRODUCTION,
                "result_claimed": RESULT_CLAIMED,
                "schema": SEALED_TEST_EVALUATION_SCHEMA,
                "sealed_test_consumption_sha256": checked_consumption,
                "status": "LOCAL_SEALED_TEST_EVALUATION_BOUND_NOT_EXTERNAL_AUTHORITY",
            }
        )
        _write_once(self._root / "sealed-test-evaluation.json", raw)
        return artifact_sha256(raw)

    def verified_evaluation_binding(
        self,
        *,
        consumption_sha256: object,
        evaluator_sha256: object,
        aggregate_code_sha256: object,
    ) -> tuple[str, str]:
        consumption_raw = _read_regular(
            self._root / "sealed-test-consumption.json",
            "sealed-test consumption artifact",
        )
        checked_consumption = _sha256(
            consumption_sha256, "sealed_test_consumption_sha256"
        )
        if artifact_sha256(consumption_raw) != checked_consumption:
            raise ExecutionContractError("sealed-test consumption digest mismatch")
        raw = _read_regular(
            self._root / "sealed-test-evaluation.json",
            "sealed-test evaluation artifact",
        )
        value = _parse_canonical(raw, "sealed-test evaluation binding")
        expected_keys = {
            "aggregate_code_sha256",
            "authority",
            "completed_at_utc",
            "evaluation_aggregate_sha256",
            "evaluator_sha256",
            "production",
            "result_claimed",
            "schema",
            "sealed_test_consumption_sha256",
            "status",
        }
        if set(value) != expected_keys:
            raise ExecutionContractError("sealed-test evaluation binding keys are not closed")
        _timestamp(value["completed_at_utc"], "completed_at_utc")
        aggregate_sha256 = _sha256(
            value["evaluation_aggregate_sha256"],
            "evaluation_aggregate_sha256",
        )
        if (
            value["aggregate_code_sha256"]
            != _sha256(aggregate_code_sha256, "aggregate_code_sha256")
            or value["evaluator_sha256"]
            != _sha256(evaluator_sha256, "evaluator_sha256")
            or value["sealed_test_consumption_sha256"] != checked_consumption
            or value["authority"] != AUTHORITY
            or value["production"] is not PRODUCTION
            or value["result_claimed"] is not RESULT_CLAIMED
            or value["schema"] != SEALED_TEST_EVALUATION_SCHEMA
            or value["status"]
            != "LOCAL_SEALED_TEST_EVALUATION_BOUND_NOT_EXTERNAL_AUTHORITY"
        ):
            raise ExecutionContractError("sealed-test evaluation binding is invalid")
        if _canonical_json_bytes(value) != raw:
            raise ExecutionContractError("sealed-test evaluation binding is not canonical")
        return aggregate_sha256, artifact_sha256(raw)


def _expected_training_config() -> dict[str, object]:
    return {
        "authority": 0,
        "base_training": {
            "effective_global_batch": 128,
            "epochs": 30,
            "gradient_clip_decimal": "1.0",
            "learning_rate_decimal": "0.0002",
            "optimizer": "AdamW",
            "schedule": "linear-warmup-cosine-decay",
            "warmup_fraction_decimal": "0.05",
            "weight_decay_decimal": "0.01",
        },
        "checkpoint_policy": {
            "atomic_write_once": True,
            "freeze_exact_cadence_before_base_runs": True,
            "required_state": [
                "model",
                "optimizer",
                "cpu_rng",
                "cuda_rng",
                "sampler_cursor",
                "dataloader_generator",
                "dropout_global_step",
                "validation_best",
            ],
        },
        "control_parameter_matching": {
            "control_system_ids": [f"{index:02d}" for index in range(1, 8)],
            "full_system_id": "08",
            "relative_tolerance_decimal": "0.01",
        },
        "data_manifest_sha256": None,
        "dataset_root": None,
        "execution_authorized": False,
        "execution_grant_sha256": None,
        "gpu_execution_allowed": False,
        "precision_policy": {
            "bf16_requires_target_runtime_qualification": True,
            "comparable_systems_share_precision": True,
            "default": "FP32",
        },
        "prepared_data_manifest_sha256": None,
        "production": False,
        "residual_training": {
            "base_frozen": True,
            "effective_global_batch": 128,
            "epochs": 20,
            "gradient_clip_decimal": "1.0",
            "learning_rate_decimal": "0.0003",
            "optimizer": "AdamW",
            "schedule": "linear-warmup-cosine-decay",
            "warmup_fraction_decimal": "0.05",
            "weight_decay_decimal": "0.01",
        },
        "result_claimed": False,
        "rights_grant_sha256": None,
        "runtime_receipt_sha256": None,
        "schema": TRAINING_CONFIG_SCHEMA,
        "server_endpoint": None,
        "server_inventory_sha256": None,
        "status": "HOLD_EXTERNAL_DATA_SERVER_RIGHTS_RUNTIME_AND_EXECUTION_GRANT",
    }


def canonical_public_training_config_bytes() -> bytes:
    """Return the built-in authority-zero default used by installed wheels."""

    return _canonical_json_bytes(_expected_training_config())


def validate_training_config_bytes(raw: object) -> bytes:
    value = _parse_canonical(raw, "training config")
    if value != _expected_training_config():
        raise ExecutionContractError("training config is not the public fail-closed v1 config")
    return bytes(raw)


def load_training_config(path: str | Path) -> bytes:
    return validate_training_config_bytes(Path(path).read_bytes())


def public_preflight(training_config_raw: object) -> PreflightReport:
    """Return all explicit holds; public JSON can never verify external grants."""

    validate_training_config_bytes(training_config_raw)
    holds = (
        "HOLD_SERVER_ENDPOINT_ABSENT",
        "HOLD_SERVER_INVENTORY_RECEIPT_ABSENT",
        "HOLD_DATASET_ROOT_ABSENT",
        "HOLD_DATA_MANIFEST_ABSENT",
        "HOLD_PREPARED_DATA_MANIFEST_ABSENT",
        "HOLD_RIGHTS_GRANT_ABSENT",
        "HOLD_RUNTIME_RECEIPT_ABSENT",
        "HOLD_EXECUTION_GRANT_ABSENT",
        "HOLD_GPU_EXECUTION_DISABLED",
        "HOLD_EXTERNAL_RECEIPTS_UNVERIFIED",
    )
    return PreflightReport(ready=False, hold_codes=holds)


def canonical_preflight_bytes(report: object) -> bytes:
    if type(report) is not PreflightReport:
        raise TypeError("report must be exact PreflightReport")
    if report != public_preflight(canonical_public_training_config_bytes()):
        raise ExecutionContractError("preflight report is not the public fail-closed result")
    return _canonical_json_bytes(
        {
            "authority": report.authority,
            "checked_commands": list(report.checked_commands),
            "external_receipt_verified": report.external_receipt_verified,
            "hold_codes": list(report.hold_codes),
            "production": report.production,
            "ready": report.ready,
            "result_claimed": report.result_claimed,
            "status": report.status,
        }
    )


class DataFreeRunner:
    """Fail closed by default or dispatch through one host-injected adapter."""

    __slots__ = (
        "_adapter",
        "_adapter_sha256",
        "_admission",
        "_config",
        "_handler_manifest_sha256",
        "_plan",
        "_plan_sha256",
        "_training_config_sha256",
    )

    def __init__(
        self,
        training_config_raw: object,
        plan: object | None = None,
        *,
        runtime_adapter: RuntimeAdapter | None = None,
    ) -> None:
        self._config = validate_training_config_bytes(training_config_raw)
        self._plan = build_experiment_plan() if plan is None else plan
        if type(self._plan) is not ExperimentPlan or self._plan != build_experiment_plan():
            raise ExecutionContractError("runner requires the exact frozen experiment plan")
        from phaseset_core.experiments import experiment_plan_sha256

        self._plan_sha256 = experiment_plan_sha256(self._plan)
        self._training_config_sha256 = artifact_sha256(self._config)
        self._adapter = runtime_adapter
        self._admission: RuntimeAdmission | None = None
        self._adapter_sha256: str | None = None
        self._handler_manifest_sha256: str | None = None
        if runtime_adapter is not None:
            commands = getattr(runtime_adapter, "commands", None)
            if type(commands) is not tuple or commands != COMMANDS:
                raise ExecutionContractError(
                    "runtime adapter must declare the exact eleven-command census"
                )
            adapter_sha256 = getattr(runtime_adapter, "adapter_sha256", None)
            handler_manifest_sha256 = getattr(runtime_adapter, "handler_manifest_sha256", None)
            self._adapter_sha256 = _sha256(adapter_sha256, "adapter_sha256")
            self._handler_manifest_sha256 = _sha256(
                handler_manifest_sha256, "handler_manifest_sha256"
            )
            if self._handler_manifest_sha256 != runtime_handler_manifest_sha256(commands):
                raise ExecutionContractError("runtime adapter handler manifest is not frozen")
            if not callable(getattr(runtime_adapter, "admit", None)) or not callable(
                getattr(runtime_adapter, "handle", None)
            ):
                raise ExecutionContractError("runtime adapter surface is incomplete")

    def _runtime_admission(self) -> RuntimeAdmission:
        if self._adapter is None:
            raise ExecutionContractError("no runtime adapter was injected")
        if self._admission is not None:
            return self._admission
        if self._adapter_sha256 is None or self._handler_manifest_sha256 is None:
            raise ExecutionContractError("runtime adapter digest bindings are absent")
        request = RuntimeAdmissionRequest(
            plan_sha256=self._plan_sha256,
            matrix_sha256=self._plan.matrix_sha256,
            training_config_sha256=self._training_config_sha256,
            handler_manifest_sha256=self._handler_manifest_sha256,
        )
        admission = self._adapter.admit(request)
        canonical_runtime_admission_bytes(admission)
        if (
            admission.adapter_sha256 != self._adapter_sha256
            or admission.handler_manifest_sha256 != request.handler_manifest_sha256
            or admission.plan_sha256 != request.plan_sha256
            or admission.matrix_sha256 != request.matrix_sha256
            or admission.training_config_sha256 != request.training_config_sha256
        ):
            raise ExecutionContractError("runtime admission digest bindings do not match")
        self._admission = admission
        return admission

    def preflight(self) -> PreflightReport | RuntimeAdmission:
        if self._adapter is None:
            return public_preflight(self._config)
        return self._runtime_admission()

    def command_intent(
        self,
        command: object,
        *,
        run_id: object = None,
        seed: object = None,
        split: object = None,
    ) -> CommandIntent:
        checked_command, checked_run, checked_seed, checked_split = _validate_command_identity(
            command, run_id, seed, split
        )
        if self._adapter is None:
            report = public_preflight(self._config)
            raise ExecutionHold(checked_command, report.hold_codes)
        admission = self._runtime_admission()
        intent = CommandIntent(
            command=checked_command,
            run_id=checked_run,
            seed=checked_seed,
            split=checked_split,
            plan_sha256=self._plan_sha256,
            runtime_mode=admission.runtime_mode,
            admission_sha256=runtime_admission_sha256(admission),
            authority=admission.authority,
            execution_authorized=True,
            production=admission.production,
        )
        if checked_command == "evaluate" and checked_split == "test":
            consume = getattr(self._adapter, "consume_sealed_test", None)
            if not callable(consume):
                raise ExecutionContractError(
                    "sealed-test evaluation requires an authenticated persistent gate"
                )
            consumption_sha256 = consume(intent)
            _sha256(consumption_sha256, "sealed_test_consumption_sha256")
            intent = replace(
                intent,
                sealed_test_consumption_sha256=consumption_sha256,
            )
        return intent

    def execute_command(
        self,
        command: object,
        *,
        run_id: object = None,
        seed: object = None,
        split: object = None,
    ) -> CommandResult:
        """Dispatch one admitted command without exposing paths or endpoints."""

        intent = self.command_intent(
            command,
            run_id=run_id,
            seed=seed,
            split=split,
        )
        if self._adapter is None:
            raise ExecutionContractError("no runtime adapter was injected")
        result = self._adapter.handle(intent)
        canonical_command_result_bytes(result)
        if (
            result.command != intent.command
            or result.run_id != intent.run_id
            or result.seed != intent.seed
            or result.split != intent.split
            or result.runtime_mode != intent.runtime_mode
            or result.admission_sha256 != intent.admission_sha256
            or result.authority != intent.authority
            or result.production is not intent.production
            or result.sealed_test_consumption_sha256
            != intent.sealed_test_consumption_sha256
        ):
            raise ExecutionContractError("command result is not bound to its admitted intent")
        return result


__all__ = [
    "ATTEMPT_SCHEMA",
    "AUTHORITY",
    "AttemptRecord",
    "AttemptStore",
    "CHECKPOINT_SCHEMA",
    "COMMAND_RESULT_SCHEMA",
    "COMMANDS",
    "CheckpointRecord",
    "CommandIntent",
    "CommandResult",
    "CompletedAttemptEvidence",
    "DataFreeRunner",
    "ExecutionContractError",
    "ExecutionHold",
    "HEARTBEAT_SCHEMA",
    "MAX_VERIFIED_ATTEMPT_CHAIN_LENGTH",
    "HeartbeatRecord",
    "PERIODIC_CACHE_SCHEMA",
    "PeriodicCacheRecord",
    "PreflightReport",
    "SEALED_TEST_SCHEMA",
    "RESUME_SCHEMA",
    "RUNTIME_ADMISSION_SCHEMA",
    "ResumeRecord",
    "RuntimeAdapter",
    "RuntimeAdmission",
    "RuntimeAdmissionRequest",
    "STATUS",
    "SealedTestGate",
    "TERMINAL_SCHEMA",
    "TRAINING_CONFIG_SCHEMA",
    "TerminalRecord",
    "artifact_sha256",
    "canonical_attempt_bytes",
    "canonical_checkpoint_bytes",
    "canonical_command_result_bytes",
    "canonical_heartbeat_bytes",
    "canonical_periodic_cache_bytes",
    "canonical_preflight_bytes",
    "canonical_public_training_config_bytes",
    "canonical_runtime_admission_bytes",
    "canonical_resume_bytes",
    "canonical_terminal_bytes",
    "load_training_config",
    "parse_attempt_bytes",
    "parse_checkpoint_bytes",
    "parse_command_result_bytes",
    "parse_heartbeat_bytes",
    "parse_periodic_cache_bytes",
    "parse_resume_bytes",
    "parse_runtime_admission_bytes",
    "parse_terminal_bytes",
    "public_preflight",
    "runtime_admission_sha256",
    "runtime_handler_manifest_sha256",
    "validate_training_config_bytes",
    "verified_completed_attempt_evidence",
    "verified_resume_checkpoint_payload_sha256",
]
