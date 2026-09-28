"""Run the frozen nine-row PhaseSet base latency session on one CUDA device.

The public entry point derives every timed sample itself.  Resolver-owned
selected-best checkpoint bytes and the separately recomputed capture-score
cohort are comparison inputs; neither a latency nor a parameter count can be
submitted by a caller.  Results remain authority zero and stop before
``BaseScore`` construction, winner selection, or production authorization.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
import struct
import subprocess
import time
from typing import Final, Protocol, runtime_checkable

import numpy as np
import torch

from . import base_cohort_resolver as resolver_module
from . import base_cohort_validation as scoring_module
from . import training as training_module
from .contracts import PreparedGroupBatch, group_commitment
from .experiments import base_run_ids, parse_run_id


AUTHORITY: Final = 0
PRODUCTION: Final = False
RESULT_CLAIMED: Final = False
SESSION_SCHEMA: Final = "phaseset-base-cohort-latency-session-v1"
VISIT_SCHEMA: Final = "phaseset-base-cohort-latency-visit-v1"
ROW_SCHEMA: Final = "phaseset-base-cohort-latency-row-v1"
WORKLOAD_SCHEMA: Final = "phaseset-base-cohort-latency-workload-v1"
RUNTIME_SCHEMA: Final = "phaseset-base-cohort-latency-runtime-v1"
SNAPSHOT_SCHEMA: Final = "phaseset-shared-cuda-observation-v1"
ACTUAL_COMPLETE_STATUS: Final = "ACTUAL_NINE_ROW_LATENCY_SESSION_COMPLETE_NO_QUALIFICATION"
FAKE_COMPLETE_STATUS: Final = "ENGINEERING_FAKE_RUNTIME_COMPLETE_NOT_FORMAL"
HOLD_STATUS: Final = "LATENCY_SESSION_HOLD_NO_FORMAL_ROW_LATENCY"

ROUND_COUNT: Final = 9
ROWS_PER_ROUND: Final = 9
WARMUPS_PER_VISIT: Final = 5
TIMED_PER_VISIT: Final = 11
SAMPLES_PER_ROW: Final = ROUND_COUNT * TIMED_PER_VISIT
FIXTURE_BATCH: Final = 1
FIXTURE_ACTORS: Final = 32
FIXTURE_FRAMES: Final = 200
FIXTURE_JOINTS: Final = 22
FIXTURE_COORDINATES: Final = 3
FIXTURE_EMBEDDING_DIM: Final = 512
MINIMUM_FREE_MIB: Final = 8192
CUDA_ALLOCATOR_LIMIT_BYTES: Final = 2 * 1024**3
_ACTOR_DOMAIN: Final = b"phaseset-base-latency-actor-v1\n"
_CUDA_UUID_PAYLOAD: Final = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)


class BaseCohortLatencyError(ValueError):
    """The cohort, score binding, runtime, or measured sample is invalid."""


class BaseCohortLatencyHold(RuntimeError):
    """A started session failed; ``receipt`` retains every observed sample."""

    def __init__(self, failure_code: str, receipt: BaseCohortLatencySession) -> None:
        super().__init__(failure_code)
        self.failure_code = failure_code
        self.receipt = receipt


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _lower_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BaseCohortLatencyError(f"{label} must be lowercase SHA-256 hex")
    return value


def _prefixed_cuda_uuid(value: object, label: str) -> str:
    if (
        type(value) is not str
        or not value.startswith("GPU-")
        or _CUDA_UUID_PAYLOAD.fullmatch(value[4:]) is None
    ):
        raise BaseCohortLatencyError(
            f"{label} must be GPU- plus one canonical lowercase CUDA UUID"
        )
    return value


def _torch_cuda_uuid(value: object) -> str:
    try:
        rendered = str(value)
    except Exception as error:
        raise BaseCohortLatencyError(
            "Torch device UUID cannot be rendered as a canonical CUDA UUID"
        ) from error
    payload = rendered[4:] if rendered.startswith("GPU-") else rendered
    if _CUDA_UUID_PAYLOAD.fullmatch(payload) is None:
        raise BaseCohortLatencyError(
            "Torch device UUID is not a canonical lowercase CUDA UUID"
        )
    return f"GPU-{payload}"


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or not 1 <= value <= (1 << 63) - 1:
        raise BaseCohortLatencyError(f"{label} must be an exact positive int")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or not 0 <= value <= (1 << 63) - 1:
        raise BaseCohortLatencyError(f"{label} must be an exact nonnegative int")
    return value


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


def _canonical_digest(value: object) -> str:
    return _sha256(_canonical_json_bytes(value))


@dataclass(frozen=True, slots=True)
class SharedCudaObservation:
    captured_unix_ns: int
    device_uuid: str
    memory_total_mib: int
    memory_free_mib: int
    memory_used_mib: int
    gpu_utilization_percent: int
    memory_utilization_percent: int
    compute_mode: str
    driver_version: str
    schema: str = SNAPSHOT_SCHEMA

    def __post_init__(self) -> None:
        _positive_int(self.captured_unix_ns, "captured_unix_ns")
        _prefixed_cuda_uuid(self.device_uuid, "CUDA observation UUID")
        _positive_int(self.memory_total_mib, "memory_total_mib")
        _nonnegative_int(self.memory_free_mib, "memory_free_mib")
        _nonnegative_int(self.memory_used_mib, "memory_used_mib")
        if (
            self.memory_free_mib > self.memory_total_mib
            or self.memory_used_mib > self.memory_total_mib
        ):
            raise BaseCohortLatencyError("CUDA memory observation exceeds physical total")
        for name in ("gpu_utilization_percent", "memory_utilization_percent"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= 100:
                raise BaseCohortLatencyError(f"{name} must be an exact percent")
        if type(self.compute_mode) is not str or not self.compute_mode:
            raise BaseCohortLatencyError("CUDA compute mode is absent")
        if type(self.driver_version) is not str or not self.driver_version:
            raise BaseCohortLatencyError("CUDA driver version is absent")
        if self.schema != SNAPSHOT_SCHEMA:
            raise BaseCohortLatencyError("CUDA observation schema changed")

    def payload(self) -> dict[str, object]:
        return {
            "captured_unix_ns": self.captured_unix_ns,
            "compute_mode": self.compute_mode,
            "device_uuid": self.device_uuid,
            "driver_version": self.driver_version,
            "gpu_utilization_percent": self.gpu_utilization_percent,
            "memory_free_mib": self.memory_free_mib,
            "memory_total_mib": self.memory_total_mib,
            "memory_used_mib": self.memory_used_mib,
            "memory_utilization_percent": self.memory_utilization_percent,
            "schema": self.schema,
        }


@dataclass(frozen=True, slots=True)
class BaseLatencyRuntimeIdentity:
    device: str
    device_uuid: str
    device_name: str
    torch_version: str
    cuda_runtime_version: str
    cudnn_version: int
    driver_version: str
    nvidia_total_memory_mib: int
    compute_mode: str
    torch_total_memory_bytes: int
    initial_allocated_bytes: int
    initial_reserved_bytes: int
    environment_sha256: str
    precision_mode: str
    allocator_limit_bytes: int = CUDA_ALLOCATOR_LIMIT_BYTES
    schema: str = RUNTIME_SCHEMA

    def __post_init__(self) -> None:
        try:
            device = torch.device(self.device)
        except (RuntimeError, ValueError) as error:
            raise BaseCohortLatencyError("latency runtime device is invalid") from error
        if device.type != "cuda":
            raise BaseCohortLatencyError("formal latency runtime must be CUDA")
        _prefixed_cuda_uuid(self.device_uuid, "latency runtime UUID")
        for name in (
            "device_name",
            "torch_version",
            "cuda_runtime_version",
            "driver_version",
        ):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise BaseCohortLatencyError(f"latency runtime {name} is absent")
        _positive_int(self.cudnn_version, "cudnn_version")
        _positive_int(self.nvidia_total_memory_mib, "nvidia_total_memory_mib")
        if type(self.compute_mode) is not str or not self.compute_mode:
            raise BaseCohortLatencyError("latency runtime compute mode is absent")
        _positive_int(self.torch_total_memory_bytes, "torch_total_memory_bytes")
        _nonnegative_int(self.initial_allocated_bytes, "initial_allocated_bytes")
        _nonnegative_int(self.initial_reserved_bytes, "initial_reserved_bytes")
        if self.initial_allocated_bytes != 0 or self.initial_reserved_bytes != 0:
            raise BaseCohortLatencyError("latency process has preexisting CUDA allocations")
        _lower_sha256(self.environment_sha256, "environment_sha256")
        if self.precision_mode not in ("FP32", "BF16"):
            raise BaseCohortLatencyError("latency precision mode is invalid")
        if self.allocator_limit_bytes != CUDA_ALLOCATOR_LIMIT_BYTES:
            raise BaseCohortLatencyError("CUDA allocator limit must remain exact 2 GiB")
        if self.schema != RUNTIME_SCHEMA:
            raise BaseCohortLatencyError("latency runtime schema changed")

    @property
    def sha256(self) -> str:
        return _canonical_digest(
            {
                "allocator_limit_bytes": self.allocator_limit_bytes,
                "cuda_runtime_version": self.cuda_runtime_version,
                "cudnn_version": self.cudnn_version,
                "compute_mode": self.compute_mode,
                "device": self.device,
                "device_name": self.device_name,
                "device_uuid": self.device_uuid,
                "driver_version": self.driver_version,
                "environment_sha256": self.environment_sha256,
                "initial_allocated_bytes": self.initial_allocated_bytes,
                "initial_reserved_bytes": self.initial_reserved_bytes,
                "nvidia_total_memory_mib": self.nvidia_total_memory_mib,
                "precision_mode": self.precision_mode,
                "schema": self.schema,
                "torch_total_memory_bytes": self.torch_total_memory_bytes,
                "torch_version": self.torch_version,
            }
        )


@dataclass(frozen=True, slots=True)
class BaseLatencyWorkloadReceipt:
    array_rows: tuple[tuple[str, str, tuple[int, ...], str], ...]
    raw_skeletons_sha256: str
    actor_commitments: tuple[bytes, ...] = field(repr=False)
    group_commitment: bytes = field(repr=False)
    schema: str = WORKLOAD_SCHEMA

    def __post_init__(self) -> None:
        expected_rows = (
            (
                "skeletons",
                (
                    FIXTURE_BATCH,
                    FIXTURE_ACTORS,
                    FIXTURE_FRAMES,
                    FIXTURE_JOINTS,
                    FIXTURE_COORDINATES,
                ),
                "<f4",
            ),
            ("actor_mask", (FIXTURE_BATCH, FIXTURE_ACTORS), "|b1"),
            ("frame_mask", (FIXTURE_BATCH, FIXTURE_FRAMES), "|b1"),
            ("track_mask", (FIXTURE_BATCH, FIXTURE_ACTORS, FIXTURE_FRAMES, FIXTURE_JOINTS), "|b1"),
        )
        if tuple((row[0], row[2], row[3]) for row in self.array_rows) != expected_rows:
            raise BaseCohortLatencyError("latency workload array census changed")
        for name, digest, shape, dtype in self.array_rows:
            if type(name) is not str or type(shape) is not tuple or type(dtype) is not str:
                raise BaseCohortLatencyError("latency workload array row is malformed")
            _lower_sha256(digest, f"{name} SHA-256")
        _lower_sha256(self.raw_skeletons_sha256, "raw_skeletons_sha256")
        if (
            type(self.actor_commitments) is not tuple
            or len(self.actor_commitments) != FIXTURE_ACTORS
        ):
            raise BaseCohortLatencyError("latency actor census must contain exact K=32")
        if any(type(value) is not bytes or len(value) != 32 for value in self.actor_commitments):
            raise BaseCohortLatencyError("latency actor commitments are malformed")
        if len(set(self.actor_commitments)) != FIXTURE_ACTORS:
            raise BaseCohortLatencyError("latency actor commitments must be unique")
        if type(self.group_commitment) is not bytes or len(self.group_commitment) != 32:
            raise BaseCohortLatencyError("latency group commitment is malformed")
        if group_commitment(self.actor_commitments) != self.group_commitment:
            raise BaseCohortLatencyError("latency group commitment differs from actors")
        if self.schema != WORKLOAD_SCHEMA:
            raise BaseCohortLatencyError("latency workload schema changed")

    @property
    def sha256(self) -> str:
        return _canonical_digest(
            {
                "actor_commitments": [value.hex() for value in self.actor_commitments],
                "array_rows": [
                    {
                        "dtype": dtype,
                        "name": name,
                        "sha256": digest,
                        "shape": list(shape),
                    }
                    for name, digest, shape, dtype in self.array_rows
                ],
                "group_commitment": self.group_commitment.hex(),
                "raw_skeletons_sha256": self.raw_skeletons_sha256,
                "schema": self.schema,
            }
        )


@dataclass(frozen=True, slots=True)
class BaseLatencyVisit:
    round_index: int
    ordinal_in_round: int
    run_id: str
    before: SharedCudaObservation
    after: SharedCudaObservation | None
    samples_ns: tuple[int, ...]
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    completed: bool
    failure_code: str | None
    warmups_completed: int = WARMUPS_PER_VISIT
    schema: str = VISIT_SCHEMA

    def __post_init__(self) -> None:
        if type(self.round_index) is not int or not 0 <= self.round_index < ROUND_COUNT:
            raise BaseCohortLatencyError("visit round index is invalid")
        if (
            type(self.ordinal_in_round) is not int
            or not 0 <= self.ordinal_in_round < ROWS_PER_ROUND
        ):
            raise BaseCohortLatencyError("visit ordinal is invalid")
        if self.run_id not in base_run_ids():
            raise BaseCohortLatencyError("visit run ID is unregistered")
        if type(self.before) is not SharedCudaObservation:
            raise BaseCohortLatencyError("visit before observation is missing")
        if type(self.samples_ns) is not tuple or len(self.samples_ns) > TIMED_PER_VISIT:
            raise BaseCohortLatencyError("visit sample census exceeds eleven")
        for value in self.samples_ns:
            _positive_int(value, "latency sample")
        if (
            type(self.warmups_completed) is not int
            or not 0 <= self.warmups_completed <= WARMUPS_PER_VISIT
            or self.schema != VISIT_SCHEMA
        ):
            raise BaseCohortLatencyError("visit protocol header changed")
        if self.completed:
            if (
                type(self.after) is not SharedCudaObservation
                or len(self.samples_ns) != TIMED_PER_VISIT
                or type(self.peak_allocated_bytes) is not int
                or type(self.peak_reserved_bytes) is not int
                or self.warmups_completed != WARMUPS_PER_VISIT
                or self.failure_code is not None
            ):
                raise BaseCohortLatencyError("completed visit evidence is incomplete")
        elif self.failure_code is None:
            raise BaseCohortLatencyError("incomplete visit requires a failure code")
        for name in ("peak_allocated_bytes", "peak_reserved_bytes"):
            value = getattr(self, name)
            if value is not None:
                _nonnegative_int(value, name)

    @property
    def sha256(self) -> str:
        return _canonical_digest(
            {
                "after": None if self.after is None else self.after.payload(),
                "before": self.before.payload(),
                "completed": self.completed,
                "failure_code": self.failure_code,
                "ordinal_in_round": self.ordinal_in_round,
                "peak_allocated_bytes": self.peak_allocated_bytes,
                "peak_reserved_bytes": self.peak_reserved_bytes,
                "round_index": self.round_index,
                "run_id": self.run_id,
                "samples_ns": list(self.samples_ns),
                "schema": self.schema,
                "warmups_completed": self.warmups_completed,
            }
        )


def _nearest_rank(sorted_values: tuple[int, ...], percentile: int) -> int:
    rank = math.ceil(percentile * len(sorted_values) / 100)
    return sorted_values[max(1, rank) - 1]


@dataclass(frozen=True, slots=True)
class BaseLatencyRowObservation:
    run_id: str
    system_id: str
    seed: int
    selected_checkpoint_sha256: str
    selected_checkpoint_state_digest: str
    parameter_census_sha256: str
    parameter_count: int
    runtime_sha256: str
    workload_sha256: str
    samples_ns: tuple[int, ...]
    median_ns: int
    median_absolute_deviation_ns: int
    minimum_ns: int
    maximum_ns: int
    p05_nearest_rank_ns: int
    p95_nearest_rank_ns: int
    schema: str = ROW_SCHEMA

    def __post_init__(self) -> None:
        role, seed, system_id = parse_run_id(self.run_id)
        if role != "BASE_QUALIFICATION" or self.seed != seed or self.system_id != system_id:
            raise BaseCohortLatencyError("latency row identity is inconsistent")
        for name in (
            "selected_checkpoint_sha256",
            "selected_checkpoint_state_digest",
            "parameter_census_sha256",
            "runtime_sha256",
            "workload_sha256",
        ):
            _lower_sha256(getattr(self, name), name)
        _positive_int(self.parameter_count, "parameter_count")
        if type(self.samples_ns) is not tuple or len(self.samples_ns) != SAMPLES_PER_ROW:
            raise BaseCohortLatencyError("latency row must retain exact 99 samples")
        for value in self.samples_ns:
            _positive_int(value, "latency sample")
        sorted_values = tuple(sorted(self.samples_ns))
        median = sorted_values[len(sorted_values) // 2]
        deviations = tuple(sorted(abs(value - median) for value in self.samples_ns))
        expected = (
            median,
            deviations[len(deviations) // 2],
            sorted_values[0],
            sorted_values[-1],
            _nearest_rank(sorted_values, 5),
            _nearest_rank(sorted_values, 95),
        )
        observed = (
            self.median_ns,
            self.median_absolute_deviation_ns,
            self.minimum_ns,
            self.maximum_ns,
            self.p05_nearest_rank_ns,
            self.p95_nearest_rank_ns,
        )
        if observed != expected or self.schema != ROW_SCHEMA:
            raise BaseCohortLatencyError("latency row statistics differ from raw samples")

    @property
    def frozen_runtime_latency_ns(self) -> int:
        return self.median_ns

    @property
    def sha256(self) -> str:
        return _canonical_digest(
            {
                "maximum_ns": self.maximum_ns,
                "median_absolute_deviation_ns": self.median_absolute_deviation_ns,
                "median_ns": self.median_ns,
                "minimum_ns": self.minimum_ns,
                "p05_nearest_rank_ns": self.p05_nearest_rank_ns,
                "p95_nearest_rank_ns": self.p95_nearest_rank_ns,
                "parameter_census_sha256": self.parameter_census_sha256,
                "parameter_count": self.parameter_count,
                "run_id": self.run_id,
                "runtime_sha256": self.runtime_sha256,
                "samples_ns": list(self.samples_ns),
                "schema": self.schema,
                "seed": self.seed,
                "selected_checkpoint_sha256": self.selected_checkpoint_sha256,
                "selected_checkpoint_state_digest": self.selected_checkpoint_state_digest,
                "system_id": self.system_id,
                "workload_sha256": self.workload_sha256,
            }
        )


@dataclass(frozen=True, slots=True, repr=False)
class BaseCohortLatencySession:
    registered_cuda_uuid: str
    resolved_cohort_sha256: str
    scored_cohort_sha256: str
    workload_receipt: BaseLatencyWorkloadReceipt
    admission_observation: SharedCudaObservation
    post_context_observation: SharedCudaObservation | None
    runtime_identity: BaseLatencyRuntimeIdentity | None
    visits: tuple[BaseLatencyVisit, ...]
    rows: tuple[BaseLatencyRowObservation, ...]
    actual_runtime: bool
    complete: bool
    formal_samples_complete: bool
    failure_code: str | None
    authority: int = AUTHORITY
    production: bool = PRODUCTION
    result_claimed: bool = RESULT_CLAIMED
    schema: str = SESSION_SCHEMA
    status: str = HOLD_STATUS

    def __post_init__(self) -> None:
        _prefixed_cuda_uuid(self.registered_cuda_uuid, "session CUDA UUID")
        _lower_sha256(self.resolved_cohort_sha256, "resolved_cohort_sha256")
        _lower_sha256(self.scored_cohort_sha256, "scored_cohort_sha256")
        if type(self.workload_receipt) is not BaseLatencyWorkloadReceipt:
            raise BaseCohortLatencyError("session workload receipt is invalid")
        if type(self.admission_observation) is not SharedCudaObservation:
            raise BaseCohortLatencyError("session admission observation is invalid")
        if type(self.visits) is not tuple or any(
            type(value) is not BaseLatencyVisit for value in self.visits
        ):
            raise BaseCohortLatencyError("session visit census is malformed")
        if type(self.rows) is not tuple or any(
            type(value) is not BaseLatencyRowObservation for value in self.rows
        ):
            raise BaseCohortLatencyError("session row census is malformed")
        if type(self.actual_runtime) is not bool or type(self.complete) is not bool:
            raise BaseCohortLatencyError("session runtime/completion flags are malformed")
        if type(self.formal_samples_complete) is not bool:
            raise BaseCohortLatencyError("session formal sample flag is malformed")
        if self.complete:
            expected_visits = tuple(
                (round_index, ordinal, run_id)
                for round_index, round_rows in enumerate(cyclic_visit_schedule())
                for ordinal, run_id in enumerate(round_rows)
            )
            observed_visits = tuple(
                (visit.round_index, visit.ordinal_in_round, visit.run_id) for visit in self.visits
            )
            if (
                observed_visits != expected_visits
                or any(not visit.completed for visit in self.visits)
                or tuple(row.run_id for row in self.rows) != base_run_ids()
                or type(self.runtime_identity) is not BaseLatencyRuntimeIdentity
                or type(self.post_context_observation) is not SharedCudaObservation
                or self.failure_code is not None
            ):
                raise BaseCohortLatencyError("completed latency session census is incomplete")
            if any(
                visit.before.device_uuid != self.registered_cuda_uuid
                or visit.after is None
                or visit.after.device_uuid != self.registered_cuda_uuid
                for visit in self.visits
            ):
                raise BaseCohortLatencyError("completed session visit changed CUDA UUID")
            if any(
                row.runtime_sha256 != self.runtime_identity.sha256
                or row.workload_sha256 != self.workload_receipt.sha256
                for row in self.rows
            ):
                raise BaseCohortLatencyError("completed latency rows lost runtime/workload binding")
            expected_formal = self.actual_runtime
            expected_status = (
                ACTUAL_COMPLETE_STATUS if self.actual_runtime else FAKE_COMPLETE_STATUS
            )
            if (
                self.formal_samples_complete is not expected_formal
                or self.status != expected_status
            ):
                raise BaseCohortLatencyError("completed latency session status is inconsistent")
        elif (
            self.rows
            or self.formal_samples_complete
            or self.failure_code is None
            or self.status != HOLD_STATUS
        ):
            raise BaseCohortLatencyError("held latency session must not expose formal rows")
        if (
            type(self.authority) is not int
            or self.authority != AUTHORITY
            or self.production is not PRODUCTION
            or self.result_claimed is not RESULT_CLAIMED
            or self.schema != SESSION_SCHEMA
        ):
            raise BaseCohortLatencyError("latency session result header is invalid")

    @property
    def sha256(self) -> str:
        return _canonical_digest(
            {
                "actual_runtime": self.actual_runtime,
                "admission_observation": self.admission_observation.payload(),
                "authority": self.authority,
                "complete": self.complete,
                "failure_code": self.failure_code,
                "formal_samples_complete": self.formal_samples_complete,
                "post_context_observation": (
                    None
                    if self.post_context_observation is None
                    else self.post_context_observation.payload()
                ),
                "production": self.production,
                "registered_cuda_uuid": self.registered_cuda_uuid,
                "resolved_cohort_sha256": self.resolved_cohort_sha256,
                "result_claimed": self.result_claimed,
                "row_sha256s": [row.sha256 for row in self.rows],
                "runtime_sha256": None
                if self.runtime_identity is None
                else self.runtime_identity.sha256,
                "schema": self.schema,
                "scored_cohort_sha256": self.scored_cohort_sha256,
                "status": self.status,
                "visit_sha256s": [visit.sha256 for visit in self.visits],
                "workload_sha256": self.workload_receipt.sha256,
            }
        )

    def __repr__(self) -> str:
        return (
            "BaseCohortLatencySession("
            f"complete={self.complete}, actual_runtime={self.actual_runtime}, "
            f"visits={len(self.visits)}, sha256={self.sha256!r})"
        )


class _LatencyRuntime(Protocol):
    def snapshot(self) -> SharedCudaObservation: ...

    def initialize(
        self,
        *,
        device: torch.device,
        precision_mode: str,
        admission_observation: SharedCudaObservation,
    ) -> BaseLatencyRuntimeIdentity: ...

    def synchronize(self, device: torch.device) -> None: ...

    def perf_counter_ns(self) -> int: ...

    def reset_peak_memory(self, device: torch.device) -> None: ...

    def peak_memory(self, device: torch.device) -> tuple[int, int]: ...

    def place(self, system: training_module.BaseRetrievalSystem, device: torch.device) -> None: ...

    def encode(
        self,
        system: training_module.BaseRetrievalSystem,
        groups: PreparedGroupBatch,
    ) -> object: ...

    def validate_output(self, output: object, device: torch.device) -> None: ...

    def release(
        self, system: training_module.BaseRetrievalSystem, device: torch.device
    ) -> None: ...


@runtime_checkable
class LatencySessionObserver(Protocol):
    """Typed post-operation observer for the private latency controller.

    Implementations receive only values already derived by this module.  They
    cannot submit samples, metrics, checkpoints, or completion state.
    """

    def session_start(
        self,
        *,
        admission_observation_sha256: str,
        started_unix_ns: int,
    ) -> object: ...

    def context_ready(
        self,
        *,
        runtime_sha256: str,
        post_context_observation_sha256: str,
    ) -> object: ...

    def visit_started(
        self,
        *,
        round_index: int,
        ordinal_in_round: int,
        run_id: str,
        before_observation_sha256: str,
    ) -> object: ...

    def warmup_completed(
        self,
        *,
        round_index: int,
        ordinal_in_round: int,
        run_id: str,
        warmup_index: int,
    ) -> object: ...

    def timed_sample_completed(
        self,
        *,
        round_index: int,
        ordinal_in_round: int,
        run_id: str,
        sample_index: int,
        sample_ns: int,
    ) -> object: ...

    def visit_ended(
        self,
        *,
        round_index: int,
        ordinal_in_round: int,
        run_id: str,
        outcome: str,
        after_observation_sha256: str | None,
        peak_allocated_bytes: int | None,
        peak_reserved_bytes: int | None,
        failure_code: str | None,
    ) -> object: ...

    def terminal(
        self,
        *,
        outcome: str,
        latency_session_sha256: str,
        failure_code: str | None,
    ) -> object: ...


class _LatencyObserverFailure(RuntimeError):
    """An observer callback failed; no later callback may be attempted."""


class _NvidiaSmiTorchRuntime:
    __slots__ = ("_uuid",)

    def __init__(self, registered_cuda_uuid: str) -> None:
        self._uuid = _prefixed_cuda_uuid(
            registered_cuda_uuid,
            "registered CUDA UUID",
        )

    def snapshot(self) -> SharedCudaObservation:
        query = (
            "uuid,memory.total,memory.free,memory.used,utilization.gpu,"
            "utilization.memory,compute_mode,driver_version"
        )
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self._uuid}",
                    f"--query-gpu={query}",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise BaseCohortLatencyError("NVIDIA device observation failed") from error
        rows = tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())
        if len(rows) != 1:
            raise BaseCohortLatencyError("NVIDIA device observation is not one exact row")
        fields = tuple(value.strip() for value in rows[0].split(","))
        if len(fields) != 8:
            raise BaseCohortLatencyError("NVIDIA device observation field count changed")
        try:
            return SharedCudaObservation(
                captured_unix_ns=time.time_ns(),
                device_uuid=fields[0],
                memory_total_mib=int(fields[1]),
                memory_free_mib=int(fields[2]),
                memory_used_mib=int(fields[3]),
                gpu_utilization_percent=int(fields[4]),
                memory_utilization_percent=int(fields[5]),
                compute_mode=fields[6],
                driver_version=fields[7],
            )
        except (TypeError, ValueError) as error:
            raise BaseCohortLatencyError("NVIDIA device observation is malformed") from error

    def initialize(
        self,
        *,
        device: torch.device,
        precision_mode: str,
        admission_observation: SharedCudaObservation,
    ) -> BaseLatencyRuntimeIdentity:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise BaseCohortLatencyError("formal latency CUDA device is unavailable")
        index = torch.cuda.current_device() if device.index is None else device.index
        properties = torch.cuda.get_device_properties(index)
        observed_uuid = _torch_cuda_uuid(getattr(properties, "uuid", None))
        _prefixed_cuda_uuid(admission_observation.device_uuid, "admission CUDA UUID")
        if observed_uuid != self._uuid or admission_observation.device_uuid != self._uuid:
            raise BaseCohortLatencyError("Torch and NVIDIA device UUIDs differ")
        fraction = CUDA_ALLOCATOR_LIMIT_BYTES / int(properties.total_memory)
        if not 0.0 < fraction <= 1.0:
            raise BaseCohortLatencyError("CUDA allocator limit is incompatible with device")
        torch.cuda.set_per_process_memory_fraction(fraction, device=index)
        probe = torch.empty((1,), dtype=torch.float32, device=device)
        del probe
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        initial_allocated = int(torch.cuda.memory_allocated(index))
        initial_reserved = int(torch.cuda.memory_reserved(index))
        if initial_allocated != 0 or initial_reserved != 0:
            raise BaseCohortLatencyError("latency process has preexisting CUDA allocations")
        cudnn = torch.backends.cudnn.version()
        if type(cudnn) is not int or cudnn < 1 or type(torch.version.cuda) is not str:
            raise BaseCohortLatencyError("Torch CUDA runtime identity is incomplete")
        return BaseLatencyRuntimeIdentity(
            device=str(torch.device("cuda", index)),
            device_uuid=self._uuid,
            device_name=str(properties.name),
            torch_version=str(torch.__version__),
            cuda_runtime_version=torch.version.cuda,
            cudnn_version=cudnn,
            driver_version=admission_observation.driver_version,
            nvidia_total_memory_mib=admission_observation.memory_total_mib,
            compute_mode=admission_observation.compute_mode,
            torch_total_memory_bytes=int(properties.total_memory),
            initial_allocated_bytes=initial_allocated,
            initial_reserved_bytes=initial_reserved,
            environment_sha256=training_module.training_environment_sha256(),
            precision_mode=precision_mode,
        )

    def synchronize(self, device: torch.device) -> None:
        torch.cuda.synchronize(device)

    def perf_counter_ns(self) -> int:
        return time.perf_counter_ns()

    def reset_peak_memory(self, device: torch.device) -> None:
        torch.cuda.reset_peak_memory_stats(device)

    def peak_memory(self, device: torch.device) -> tuple[int, int]:
        return (
            int(torch.cuda.max_memory_allocated(device)),
            int(torch.cuda.max_memory_reserved(device)),
        )

    def place(self, system: training_module.BaseRetrievalSystem, device: torch.device) -> None:
        system.to(device)

    def encode(
        self,
        system: training_module.BaseRetrievalSystem,
        groups: PreparedGroupBatch,
    ) -> object:
        return system.encode_trainable(groups)

    def validate_output(self, output: object, device: torch.device) -> None:
        _validate_output(output, device)

    def release(self, system: training_module.BaseRetrievalSystem, device: torch.device) -> None:
        system.to(torch.device("cpu"))
        torch.cuda.synchronize(device)
        try:
            clear_cublas_workspaces = torch._C._cuda_clearCublasWorkspaces
        except AttributeError as error:
            raise BaseCohortLatencyError(
                "Torch CUDA cuBLAS workspace clear API is unavailable"
            ) from error
        if not callable(clear_cublas_workspaces):
            raise BaseCohortLatencyError(
                "Torch CUDA cuBLAS workspace clear API is unavailable"
            )
        try:
            clear_cublas_workspaces()
        except Exception as error:
            raise BaseCohortLatencyError(
                "Torch CUDA cuBLAS workspace clear failed"
            ) from error
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        if torch.cuda.memory_allocated(device) != 0 or torch.cuda.memory_reserved(device) != 0:
            raise BaseCohortLatencyError(
                "measured model CUDA allocation remained after visit release"
            )


def cyclic_visit_schedule() -> tuple[tuple[str, ...], ...]:
    rows = base_run_ids()
    if len(rows) != ROWS_PER_ROUND:
        raise AssertionError("registered base row count changed")
    return tuple(
        tuple(rows[(round_index + ordinal) % len(rows)] for ordinal in range(len(rows)))
        for round_index in range(ROUND_COUNT)
    )


def _array_row(name: str, value: np.ndarray) -> tuple[str, str, tuple[int, ...], str]:
    return (
        name,
        _sha256(value.tobytes(order="C")),
        tuple(int(axis) for axis in value.shape),
        value.dtype.str,
    )


def _source_actor_commitments() -> tuple[bytes, ...]:
    return tuple(
        hashlib.sha256(_ACTOR_DOMAIN + struct.pack(">I", actor)).digest()
        for actor in range(FIXTURE_ACTORS)
    )


def build_frozen_latency_workload() -> tuple[PreparedGroupBatch, BaseLatencyWorkloadReceipt]:
    actors = np.arange(FIXTURE_ACTORS, dtype=np.int64).reshape(1, FIXTURE_ACTORS, 1, 1, 1)
    frames = np.arange(FIXTURE_FRAMES, dtype=np.int64).reshape(1, 1, FIXTURE_FRAMES, 1, 1)
    joints = np.arange(FIXTURE_JOINTS, dtype=np.int64).reshape(1, 1, 1, FIXTURE_JOINTS, 1)
    coordinates = np.arange(FIXTURE_COORDINATES, dtype=np.int64).reshape(
        1, 1, 1, 1, FIXTURE_COORDINATES
    )
    raw = ((actors * 131 + frames * 17 + joints * 7 + coordinates * 3) % 2048) - 1024
    skeletons = np.ascontiguousarray(raw.astype(np.float32) * np.float32(2**-10))
    actor_mask = np.ones((FIXTURE_BATCH, FIXTURE_ACTORS), dtype=np.bool_)
    frame_mask = np.ones((FIXTURE_BATCH, FIXTURE_FRAMES), dtype=np.bool_)
    track_mask = np.ones(
        (FIXTURE_BATCH, FIXTURE_ACTORS, FIXTURE_FRAMES, FIXTURE_JOINTS),
        dtype=np.bool_,
    )
    commitments = _source_actor_commitments()
    group = group_commitment(commitments)
    batch = PreparedGroupBatch(
        skeletons,
        actor_mask,
        frame_mask,
        track_mask,
        (commitments,),
        (group,),
    )
    receipt = BaseLatencyWorkloadReceipt(
        array_rows=tuple(
            _array_row(name, value)
            for name, value in (
                ("skeletons", batch.skeletons),
                ("actor_mask", batch.actor_mask),
                ("frame_mask", batch.frame_mask),
                ("track_mask", batch.track_mask),
            )
        ),
        raw_skeletons_sha256=_sha256(skeletons.tobytes(order="C")),
        actor_commitments=tuple(value for value in batch.actor_commitments[0] if value is not None),
        group_commitment=batch.group_commitments[0],
    )
    return batch, receipt


def _workload_receipt(groups: PreparedGroupBatch) -> BaseLatencyWorkloadReceipt:
    commitments = tuple(value for value in groups.actor_commitments[0] if value is not None)
    source_commitments = _source_actor_commitments()
    if set(commitments) != set(source_commitments):
        raise BaseCohortLatencyError("latency workload actor census changed")
    source_order = tuple(commitments.index(value) for value in source_commitments)
    raw_skeletons = np.ascontiguousarray(groups.skeletons[:, source_order])
    return BaseLatencyWorkloadReceipt(
        array_rows=tuple(
            _array_row(name, value)
            for name, value in (
                ("skeletons", groups.skeletons),
                ("actor_mask", groups.actor_mask),
                ("frame_mask", groups.frame_mask),
                ("track_mask", groups.track_mask),
            )
        ),
        raw_skeletons_sha256=_sha256(raw_skeletons.tobytes(order="C")),
        actor_commitments=commitments,
        group_commitment=groups.group_commitments[0],
    )


def _resolved_cohort_sha256(cohort: resolver_module.ResolvedBaseCohort) -> str:
    return _canonical_digest(
        {
            "matrix_sha256": cohort.matrix_sha256,
            "plan_sha256": cohort.plan_sha256,
            "rows": [
                {
                    "attempt_id": row.attempt_id,
                    "failed_predecessors": [
                        {
                            "attempt_id": failed.attempt_id,
                            "failure_code": failed.failure_code,
                            "terminal_outcome": failed.terminal_outcome,
                            "terminal_sha256": failed.terminal_sha256,
                        }
                        for failed in row.failed_predecessors
                    ],
                    "latest_checkpoint_name": row.latest_checkpoint_name,
                    "latest_checkpoint_sha256": row.latest_checkpoint_sha256,
                    "run_id": row.run_id,
                    "selected_checkpoint_epoch": row.selected_checkpoint.epoch,
                    "selected_checkpoint_global_step": row.selected_checkpoint.global_step,
                    "selected_checkpoint_name": row.selected_checkpoint.name,
                    "selected_checkpoint_sha256": row.selected_checkpoint.sha256,
                    "selected_checkpoint_state_digest": row.selected_checkpoint.state_digest,
                    "terminal_sha256": row.terminal_sha256,
                }
                for row in cohort.rows
            ],
            "source_tree_sha256": cohort.source_tree_sha256,
            "training_config_sha256": cohort.training_config_sha256,
            "train_manifest_sha256": cohort.train_manifest_sha256,
            "val_manifest_sha256": cohort.val_manifest_sha256,
        }
    )


def _checked_inputs(
    cohort: object,
    admission: object,
    scored_cohort: object,
) -> tuple[
    resolver_module.ResolvedBaseCohort,
    resolver_module.BaseCohortAdmission,
    scoring_module.BaseCohortValidationResult,
]:
    try:
        checked_admission = scoring_module._checked_admission(admission)
        checked_cohort = scoring_module._checked_cohort(cohort, checked_admission)
    except (TypeError, ValueError, RuntimeError) as error:
        raise BaseCohortLatencyError("resolved base cohort admission is invalid") from error
    if type(scored_cohort) is not scoring_module.BaseCohortValidationResult:
        raise BaseCohortLatencyError("score input must be exact BaseCohortValidationResult")
    if (
        type(scored_cohort.authority) is not int
        or scored_cohort.authority != scoring_module.AUTHORITY
        or scored_cohort.production is not scoring_module.PRODUCTION
        or scored_cohort.result_claimed is not scoring_module.RESULT_CLAIMED
        or scored_cohort.status != scoring_module.COHORT_STATUS
    ):
        raise BaseCohortLatencyError("score cohort header is invalid")
    expected = {
        "plan_sha256": checked_cohort.plan_sha256,
        "matrix_sha256": checked_cohort.matrix_sha256,
        "training_config_sha256": checked_cohort.training_config_sha256,
        "source_tree_sha256": checked_cohort.source_tree_sha256,
        "train_manifest_sha256": checked_cohort.train_manifest_sha256,
        "query_census_sha256": checked_cohort.val_manifest_sha256,
    }
    if any(getattr(scored_cohort, name) != value for name, value in expected.items()):
        raise BaseCohortLatencyError("score cohort differs from resolved cohort admission")
    if tuple(row.run_id for row in scored_cohort.observations) != base_run_ids():
        raise BaseCohortLatencyError("score cohort lost the registered nine-row order")
    for resolved, scored in zip(checked_cohort.rows, scored_cohort.observations, strict=True):
        if (
            scored.run_id != resolved.run_id
            or scored.attempt_id != resolved.attempt_id
            or scored.system_id != resolved.system_id
            or scored.seed != resolved.seed
            or scored.terminal_sha256 != resolved.terminal_sha256
            or scored.selected_checkpoint_sha256 != resolved.selected_checkpoint.sha256
            or scored.selected_checkpoint_state_digest != resolved.selected_checkpoint.state_digest
        ):
            raise BaseCohortLatencyError("score row differs from resolved selected-best row")
    return checked_cohort, checked_admission, scored_cohort


def _checked_scorer_runtime_device(
    *,
    config_device: str,
    score_device: str,
    runtime_device: torch.device,
) -> None:
    try:
        configured_device = torch.device(config_device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise BaseCohortLatencyError("configured latency device is invalid") from error
    if configured_device.type == "cuda" and configured_device.index is None:
        configured_device = torch.device("cuda", torch.cuda.current_device())
    if (
        configured_device.type != "cuda"
        or runtime_device.type != "cuda"
        or configured_device != runtime_device
        or type(score_device) is not str
        or score_device != str(configured_device)
    ):
        raise BaseCohortLatencyError(
            "score device differs from the resolved physical latency device"
        )


def _load_selected_system(
    row: resolver_module.ResolvedBaseRun,
    score: scoring_module.BaseValidationScoreObservation,
    admission: resolver_module.BaseCohortAdmission,
    *,
    runtime_device: torch.device,
) -> tuple[
    training_module.BaseRetrievalSystem,
    training_module.TrainingConfig,
    training_module.PrecisionDecision,
    scoring_module.BaseParameterCensus,
]:
    payload = scoring_module._validate_selected_payload(row, admission)
    config = training_module.TrainingConfig(
        stage="base",
        seed=row.seed,
        device=admission.device,
        request_bf16=admission.request_bf16,
        bf16_runtime_qualified=admission.bf16_runtime_qualified,
        edge_budget=admission.edge_budget,
        checkpoint_every_updates=admission.checkpoint_every_updates,
        synthetic_contract=False,
    )
    try:
        system, binding = training_module.construct_registered_base_seed_bound_system(
            row.system_id,
            config,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise BaseCohortLatencyError("registered base reconstruction failed") from error
    expected_bindings = {
        "behavior_sha256": binding.behavior_sha256,
        "code_artifact_sha256": binding.code_artifact_sha256,
        "environment_sha256": binding.environment_sha256,
        "factory_sha256": binding.factory_sha256,
        "initial_optimizable_state_sha256": binding.initial_optimizable_state_sha256,
        "initialization_binding_sha256": binding.sha256,
    }
    precision = training_module.resolve_precision(config)
    _checked_scorer_runtime_device(
        config_device=config.device,
        score_device=score.device,
        runtime_device=runtime_device,
    )
    if any(payload.get(name) != expected for name, expected in expected_bindings.items()):
        raise BaseCohortLatencyError("selected checkpoint initialization binding changed")
    if (
        score.initialization_binding_sha256 != binding.sha256
        or score.factory_sha256 != binding.factory_sha256
        or score.code_artifact_sha256 != binding.code_artifact_sha256
        or score.environment_sha256 != binding.environment_sha256
        or score.behavior_sha256 != binding.behavior_sha256
        or score.runtime_config_sha256 != config.sha256
        or score.precision_mode != precision.mode
    ):
        raise BaseCohortLatencyError("score runtime binding differs from latency reconstruction")
    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise BaseCohortLatencyError("selected checkpoint model is malformed")
    try:
        system.load_state_dict(model, strict=True)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise BaseCohortLatencyError("selected model state could not be loaded") from error
    if (
        training_module.training_system_state_sha256(system) != training_module._stable_hash(model)
        or training_module.training_system_behavior_sha256(system) != binding.behavior_sha256
    ):
        raise BaseCohortLatencyError("loaded selected model state or behavior changed")
    census = scoring_module._parameter_census(system)
    if census != score.parameter_census:
        raise BaseCohortLatencyError("latency live parameter census differs from score census")
    return system, config, precision, census


def _validate_snapshot(
    snapshot: SharedCudaObservation,
    *,
    registered_uuid: str,
    runtime: BaseLatencyRuntimeIdentity | None,
) -> None:
    if type(snapshot) is not SharedCudaObservation or snapshot.device_uuid != registered_uuid:
        raise BaseCohortLatencyError("shared CUDA observation changed device UUID")
    if runtime is not None and (
        snapshot.driver_version != runtime.driver_version
        or snapshot.memory_total_mib != runtime.nvidia_total_memory_mib
        or snapshot.compute_mode != runtime.compute_mode
    ):
        raise BaseCohortLatencyError("shared CUDA runtime identity changed")


def _validate_output(output: object, device: torch.device) -> torch.Tensor:
    if (
        type(output) is not torch.Tensor
        or output.shape != (FIXTURE_BATCH, FIXTURE_EMBEDDING_DIM)
        or output.dtype != torch.float32
        or output.device != device
        or not output.is_contiguous()
        or output.requires_grad
        or output.grad_fn is not None
        or not bool(torch.isfinite(output).all().item())
    ):
        raise BaseCohortLatencyError("timed base embedding is malformed or nonfinite")
    return output


def _forward_once(
    system: training_module.BaseRetrievalSystem,
    groups: PreparedGroupBatch,
    *,
    precision: training_module.PrecisionDecision,
    device: torch.device,
    runtime: _LatencyRuntime,
    timed: bool,
) -> int | None:
    runtime.synchronize(device)
    start = runtime.perf_counter_ns() if timed else None
    with training_module._autocast_context(precision, device):
        output = runtime.encode(system, groups)
        runtime.validate_output(output, device)
    runtime.synchronize(device)
    stop = runtime.perf_counter_ns() if timed else None
    del output
    if start is None or stop is None:
        return None
    if type(start) is not int or type(stop) is not int or stop <= start:
        raise BaseCohortLatencyError("host monotonic latency timer is invalid")
    return stop - start


def _row_observations(
    visits: tuple[BaseLatencyVisit, ...],
    cohort: resolver_module.ResolvedBaseCohort,
    scored: scoring_module.BaseCohortValidationResult,
    runtime: BaseLatencyRuntimeIdentity,
    workload: BaseLatencyWorkloadReceipt,
) -> tuple[BaseLatencyRowObservation, ...]:
    rows: list[BaseLatencyRowObservation] = []
    for resolved, score in zip(cohort.rows, scored.observations, strict=True):
        samples = tuple(
            sample
            for visit in visits
            if visit.run_id == resolved.run_id
            for sample in visit.samples_ns
        )
        ordered = tuple(sorted(samples))
        median = ordered[len(ordered) // 2]
        deviations = tuple(sorted(abs(value - median) for value in samples))
        rows.append(
            BaseLatencyRowObservation(
                run_id=resolved.run_id,
                system_id=resolved.system_id,
                seed=resolved.seed,
                selected_checkpoint_sha256=resolved.selected_checkpoint.sha256,
                selected_checkpoint_state_digest=resolved.selected_checkpoint.state_digest,
                parameter_census_sha256=score.parameter_census.sha256,
                parameter_count=score.parameter_census.total,
                runtime_sha256=runtime.sha256,
                workload_sha256=workload.sha256,
                samples_ns=samples,
                median_ns=median,
                median_absolute_deviation_ns=deviations[len(deviations) // 2],
                minimum_ns=ordered[0],
                maximum_ns=ordered[-1],
                p05_nearest_rank_ns=_nearest_rank(ordered, 5),
                p95_nearest_rank_ns=_nearest_rank(ordered, 95),
            )
        )
    return tuple(rows)


def _failure_code(error: BaseException) -> str:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__
    if any(isinstance(item, torch.cuda.OutOfMemoryError) for item in chain):
        return "CUDA_OUT_OF_MEMORY"
    if any(isinstance(item, (TimeoutError, subprocess.TimeoutExpired)) for item in chain):
        return "SESSION_TIMEOUT"
    if isinstance(error, BaseCohortLatencyError):
        message = str(error)
        if "UUID" in message or "runtime identity" in message or "precision" in message:
            return "RUNTIME_DRIFT"
        if "allocator" in message or "memory" in message or "free" in message:
            return "RESOURCE_LIMIT"
        if "checkpoint" in message or "selected" in message or "score" in message:
            return "CHECKPOINT_OR_SCORE_DRIFT"
        if "embedding" in message or "timer" in message:
            return "MALFORMED_MEASUREMENT"
    return "UNEXPECTED_SESSION_FAILURE"


def run_base_cohort_latency_session(
    cohort: resolver_module.ResolvedBaseCohort,
    *,
    admission: resolver_module.BaseCohortAdmission,
    scored_cohort: scoring_module.BaseCohortValidationResult,
    registered_cuda_uuid: str,
    observer: LatencySessionObserver | None = None,
    _runtime: _LatencyRuntime | None = None,
) -> BaseCohortLatencySession:
    """Measure all nine selected-best rows; no latency value is caller supplied.

    ``_runtime`` exists solely for deterministic software contract tests.  Any
    injected runtime is permanently marked non-formal in the returned object.
    """

    checked_cohort, checked_admission, checked_scores = _checked_inputs(
        cohort,
        admission,
        scored_cohort,
    )
    _prefixed_cuda_uuid(registered_cuda_uuid, "registered_cuda_uuid")
    if observer is not None and not isinstance(observer, LatencySessionObserver):
        raise BaseCohortLatencyError("latency observer does not implement the typed interface")
    device = torch.device(checked_admission.device)
    if device.type != "cuda":
        raise BaseCohortLatencyError("formal base latency requires the admitted CUDA device")
    decisions = tuple(
        training_module.resolve_precision(
            training_module.TrainingConfig(
                stage="base",
                seed=row.seed,
                device=checked_admission.device,
                request_bf16=checked_admission.request_bf16,
                bf16_runtime_qualified=checked_admission.bf16_runtime_qualified,
                edge_budget=checked_admission.edge_budget,
                checkpoint_every_updates=checked_admission.checkpoint_every_updates,
                synthetic_contract=False,
            )
        )
        for row in checked_cohort.rows
    )
    if len({decision.mode for decision in decisions}) != 1:
        raise BaseCohortLatencyError("latency precision must be session-wide")
    precision_mode = decisions[0].mode
    groups, workload = build_frozen_latency_workload()
    if checked_admission.edge_budget < FIXTURE_ACTORS * (FIXTURE_ACTORS - 1) // 2:
        raise BaseCohortLatencyError("admitted edge budget cannot contain the K=32 workload")
    actual_runtime = _runtime is None
    runtime: _LatencyRuntime = (
        _NvidiaSmiTorchRuntime(registered_cuda_uuid) if _runtime is None else _runtime
    )
    resolved_sha = _resolved_cohort_sha256(checked_cohort)
    scored_sha = checked_scores.sha256
    admission_observation = runtime.snapshot()
    if type(admission_observation) is not SharedCudaObservation:
        raise BaseCohortLatencyError("pre-session CUDA observation is malformed")
    visits: list[BaseLatencyVisit] = []
    runtime_identity: BaseLatencyRuntimeIdentity | None = None
    post_context: SharedCudaObservation | None = None
    observer_failed = False

    def observe(method_name: str, **payload: object) -> None:
        nonlocal observer_failed
        if observer is None:
            return
        try:
            getattr(observer, method_name)(**payload)
        except Exception as error:
            observer_failed = True
            raise _LatencyObserverFailure(
                f"latency observer failed during {method_name}"
            ) from error

    def held(error: BaseException) -> BaseCohortLatencyHold:
        code = _failure_code(error)
        receipt = BaseCohortLatencySession(
            registered_cuda_uuid=registered_cuda_uuid,
            resolved_cohort_sha256=resolved_sha,
            scored_cohort_sha256=scored_sha,
            workload_receipt=workload,
            admission_observation=admission_observation,
            post_context_observation=post_context,
            runtime_identity=runtime_identity,
            visits=tuple(visits),
            rows=(),
            actual_runtime=actual_runtime,
            complete=False,
            formal_samples_complete=False,
            failure_code=code,
            status=HOLD_STATUS,
        )
        return BaseCohortLatencyHold(code, receipt)

    def observed_hold(error: BaseException) -> BaseCohortLatencyHold:
        session_hold = held(error)
        if not observer_failed:
            try:
                observe(
                    "terminal",
                    outcome="HELD",
                    latency_session_sha256=session_hold.receipt.sha256,
                    failure_code=session_hold.failure_code,
                )
            except _LatencyObserverFailure as observer_error:
                raise session_hold from observer_error
        return session_hold

    try:
        observe(
            "session_start",
            admission_observation_sha256=_canonical_digest(admission_observation.payload()),
            started_unix_ns=time.time_ns(),
        )
        _validate_snapshot(
            admission_observation,
            registered_uuid=registered_cuda_uuid,
            runtime=None,
        )
        if admission_observation.memory_free_mib < MINIMUM_FREE_MIB:
            raise BaseCohortLatencyError("shared CUDA admission has less than 8 GiB free")
        runtime_identity = runtime.initialize(
            device=device,
            precision_mode=precision_mode,
            admission_observation=admission_observation,
        )
        if type(runtime_identity) is not BaseLatencyRuntimeIdentity:
            raise BaseCohortLatencyError("latency runtime initialization receipt is invalid")
        device = torch.device(runtime_identity.device)
        post_context = runtime.snapshot()
        _validate_snapshot(
            post_context,
            registered_uuid=registered_cuda_uuid,
            runtime=runtime_identity,
        )
        if post_context.memory_free_mib < MINIMUM_FREE_MIB:
            raise BaseCohortLatencyError("shared CUDA context has less than 8 GiB free")
        if runtime_identity.precision_mode != precision_mode:
            raise BaseCohortLatencyError("latency runtime precision changed")
        score_environments = {score.environment_sha256 for score in checked_scores.observations}
        if score_environments != {runtime_identity.environment_sha256}:
            raise BaseCohortLatencyError("score cohort and latency runtime environment differ")
        observe(
            "context_ready",
            runtime_sha256=runtime_identity.sha256,
            post_context_observation_sha256=_canonical_digest(post_context.payload()),
        )
        with training_module._frozen_numerical_runtime(device), torch.no_grad():
            for round_index, round_rows in enumerate(cyclic_visit_schedule()):
                for ordinal, run_id in enumerate(round_rows):
                    row_index = base_run_ids().index(run_id)
                    row = checked_cohort.rows[row_index]
                    score = checked_scores.observations[row_index]
                    before = runtime.snapshot()
                    _validate_snapshot(
                        before,
                        registered_uuid=registered_cuda_uuid,
                        runtime=runtime_identity,
                    )
                    observe(
                        "visit_started",
                        round_index=round_index,
                        ordinal_in_round=ordinal,
                        run_id=run_id,
                        before_observation_sha256=_canonical_digest(before.payload()),
                    )
                    samples: list[int] = []
                    warmups_completed = 0
                    peaks: tuple[int, int] | None = None
                    after: SharedCudaObservation | None = None
                    system: training_module.BaseRetrievalSystem | None = None
                    visit_error: BaseException | None = None
                    try:
                        if (
                            _resolved_cohort_sha256(checked_cohort) != resolved_sha
                            or checked_scores.sha256 != scored_sha
                            or _workload_receipt(groups) != workload
                            or _sha256(row.selected_checkpoint.raw)
                            != row.selected_checkpoint.sha256
                        ):
                            raise BaseCohortLatencyError("session input changed between visits")
                        system, _config, precision, census = _load_selected_system(
                            row,
                            score,
                            checked_admission,
                            runtime_device=device,
                        )
                        if precision.mode != precision_mode or census != score.parameter_census:
                            raise BaseCohortLatencyError(
                                "visit precision or parameter census changed"
                            )
                        system.eval()
                        for parameter in system.parameters():
                            parameter.grad = None
                            parameter.requires_grad_(False)
                        if any(parameter.requires_grad for parameter in system.parameters()):
                            raise BaseCohortLatencyError("measured model was not completely frozen")
                        runtime.reset_peak_memory(device)
                        runtime.place(system, device)
                        for warmup_index in range(WARMUPS_PER_VISIT):
                            _forward_once(
                                system,
                                groups,
                                precision=precision,
                                device=device,
                                runtime=runtime,
                                timed=False,
                            )
                            warmups_completed += 1
                            observe(
                                "warmup_completed",
                                round_index=round_index,
                                ordinal_in_round=ordinal,
                                run_id=run_id,
                                warmup_index=warmup_index,
                            )
                        for sample_index in range(TIMED_PER_VISIT):
                            sample = _forward_once(
                                system,
                                groups,
                                precision=precision,
                                device=device,
                                runtime=runtime,
                                timed=True,
                            )
                            if type(sample) is not int:
                                raise AssertionError("timed forward did not produce an integer")
                            samples.append(sample)
                            observe(
                                "timed_sample_completed",
                                round_index=round_index,
                                ordinal_in_round=ordinal,
                                run_id=run_id,
                                sample_index=sample_index,
                                sample_ns=sample,
                            )
                        peaks = runtime.peak_memory(device)
                        if (
                            type(peaks) is not tuple
                            or len(peaks) != 2
                            or any(type(value) is not int or value < 0 for value in peaks)
                            or max(peaks) > CUDA_ALLOCATOR_LIMIT_BYTES
                        ):
                            raise BaseCohortLatencyError("CUDA allocator peak exceeded 2 GiB")
                        if any(parameter.grad is not None for parameter in system.parameters()):
                            raise BaseCohortLatencyError("latency measurement created a gradient")
                        if _sha256(row.selected_checkpoint.raw) != row.selected_checkpoint.sha256:
                            raise BaseCohortLatencyError("selected checkpoint changed during visit")
                    except Exception as error:
                        visit_error = error
                    finally:
                        if system is not None:
                            if peaks is None:
                                try:
                                    peaks = runtime.peak_memory(device)
                                except Exception as error:
                                    if visit_error is None:
                                        visit_error = error
                            try:
                                runtime.release(system, device)
                            except Exception as error:
                                if visit_error is None:
                                    visit_error = error
                        try:
                            after = runtime.snapshot()
                        except Exception as error:
                            if visit_error is None:
                                visit_error = error
                        else:
                            try:
                                _validate_snapshot(
                                    after,
                                    registered_uuid=registered_cuda_uuid,
                                    runtime=runtime_identity,
                                )
                            except Exception as error:
                                if visit_error is None:
                                    visit_error = error
                    if visit_error is not None:
                        code = _failure_code(visit_error)
                        valid_peaks = (
                            type(peaks) is tuple
                            and len(peaks) == 2
                            and all(type(value) is int and value >= 0 for value in peaks)
                        )
                        failed_visit = BaseLatencyVisit(
                            round_index=round_index,
                            ordinal_in_round=ordinal,
                            run_id=run_id,
                            before=before,
                            after=after,
                            samples_ns=tuple(samples),
                            peak_allocated_bytes=peaks[0] if valid_peaks else None,
                            peak_reserved_bytes=peaks[1] if valid_peaks else None,
                            completed=False,
                            failure_code=code,
                            warmups_completed=warmups_completed,
                        )
                        visits.append(failed_visit)
                        if not observer_failed:
                            observe(
                                "visit_ended",
                                round_index=round_index,
                                ordinal_in_round=ordinal,
                                run_id=run_id,
                                outcome="HELD",
                                after_observation_sha256=(
                                    None if after is None else _canonical_digest(after.payload())
                                ),
                                peak_allocated_bytes=failed_visit.peak_allocated_bytes,
                                peak_reserved_bytes=failed_visit.peak_reserved_bytes,
                                failure_code=code,
                            )
                        raise visit_error
                    completed_visit = BaseLatencyVisit(
                        round_index=round_index,
                        ordinal_in_round=ordinal,
                        run_id=run_id,
                        before=before,
                        after=after,
                        samples_ns=tuple(samples),
                        peak_allocated_bytes=peaks[0],
                        peak_reserved_bytes=peaks[1],
                        completed=True,
                        failure_code=None,
                        warmups_completed=warmups_completed,
                    )
                    visits.append(completed_visit)
                    observe(
                        "visit_ended",
                        round_index=round_index,
                        ordinal_in_round=ordinal,
                        run_id=run_id,
                        outcome="COMPLETED",
                        after_observation_sha256=_canonical_digest(after.payload()),
                        peak_allocated_bytes=completed_visit.peak_allocated_bytes,
                        peak_reserved_bytes=completed_visit.peak_reserved_bytes,
                        failure_code=None,
                    )
    except BaseCohortLatencyHold:
        raise
    except Exception as error:
        raise observed_hold(error) from error
    if runtime_identity is None or post_context is None:
        raise AssertionError("completed session lost its runtime admission")
    visit_tuple = tuple(visits)
    try:
        rows = _row_observations(
            visit_tuple,
            checked_cohort,
            checked_scores,
            runtime_identity,
            workload,
        )
        if (
            _resolved_cohort_sha256(checked_cohort) != resolved_sha
            or checked_scores.sha256 != scored_sha
            or _workload_receipt(groups) != workload
        ):
            raise BaseCohortLatencyError("session input changed after measurement")
    except Exception as error:
        raise observed_hold(error) from error
    status = ACTUAL_COMPLETE_STATUS if actual_runtime else FAKE_COMPLETE_STATUS
    result = BaseCohortLatencySession(
        registered_cuda_uuid=registered_cuda_uuid,
        resolved_cohort_sha256=resolved_sha,
        scored_cohort_sha256=scored_sha,
        workload_receipt=workload,
        admission_observation=admission_observation,
        post_context_observation=post_context,
        runtime_identity=runtime_identity,
        visits=visit_tuple,
        rows=rows,
        actual_runtime=actual_runtime,
        complete=True,
        formal_samples_complete=actual_runtime,
        failure_code=None,
        status=status,
    )
    try:
        observe(
            "terminal",
            outcome="COMPLETED",
            latency_session_sha256=result.sha256,
            failure_code=None,
        )
    except _LatencyObserverFailure as error:
        raise held(error) from error
    return result


__all__ = [
    "ACTUAL_COMPLETE_STATUS",
    "AUTHORITY",
    "BaseCohortLatencyError",
    "BaseCohortLatencyHold",
    "BaseCohortLatencySession",
    "BaseLatencyRowObservation",
    "BaseLatencyRuntimeIdentity",
    "BaseLatencyVisit",
    "BaseLatencyWorkloadReceipt",
    "CUDA_ALLOCATOR_LIMIT_BYTES",
    "FAKE_COMPLETE_STATUS",
    "HOLD_STATUS",
    "MINIMUM_FREE_MIB",
    "LatencySessionObserver",
    "ROUND_COUNT",
    "SAMPLES_PER_ROW",
    "SharedCudaObservation",
    "TIMED_PER_VISIT",
    "WARMUPS_PER_VISIT",
    "build_frozen_latency_workload",
    "cyclic_visit_schedule",
    "run_base_cohort_latency_session",
]
