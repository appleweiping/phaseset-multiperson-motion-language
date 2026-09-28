"""Private host controller for one actual nine-row base qualification.

The public CLI supplies only an attempt root.  This module consumes a frozen
private registry, resolves and scores the actual selected checkpoints, runs
one complete latency session with a durable observer, and invokes the existing
qualification assembler.  It never accepts submitted scores, latencies,
parameter counts, or a winner.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import threading
from typing import Final, Iterator

from . import base_cohort_latency as latency_module
from . import base_cohort_qualification as assembly_module
from . import base_cohort_resolver as resolver_module
from . import base_cohort_validation as scoring_module
from . import latency_progress_journal as journal_module
from . import production as production_module
from . import training as training_module
from .capture_validation import CaptureValidationResourceLimit, CaptureValidationSource
from .experiments import base_run_ids


CONFIG_SCHEMA: Final = "phaseset-private-base-qualification-controller-v2"
REGISTRY_SCHEMA: Final = "phaseset-private-base-attempt-registry-v1"
START_SCHEMA: Final = "phaseset-private-base-qualification-start-v1"
FAILURE_SCHEMA: Final = "phaseset-private-base-qualification-failure-v2"
GRANT_BASIS_SCHEMA: Final = "phaseset-private-base-qualification-grant-basis-v2"
AUTHORIZATION_SCHEMA: Final = "phaseset-private-base-qualification-authorization-v1"
COMPLETION_SCHEMA: Final = "phaseset-private-base-qualification-completion-v2"
SCORING_CUDA_SCHEMA: Final = "phaseset-private-base-scoring-cuda-v1"
MAX_REGISTRY_BYTES: Final = 1024 * 1024
MAX_OUTPUT_BYTES: Final = 8 * 1024 * 1024
MAX_WALL_TIMEOUT_SECONDS: Final = 7 * 24 * 60 * 60
_SESSION_ID: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
_CUDA_UUID: Final = re.compile(
    r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)


class HostBaseQualificationError(ValueError):
    """The private controller configuration or retained evidence is invalid."""


class _NonformalRuntimeHold(RuntimeError):
    """An explicitly injected engineering runtime reached its closed boundary."""


class _ScoringCudaHold(RuntimeError):
    """The actual scorer CUDA admission or zero-allocation boundary held."""

    def __init__(self, failure_code: str, message: str) -> None:
        super().__init__(message)
        self.failure_code = failure_code


class _FormalLatencyTimerHold(RuntimeError):
    """The required process-local latency timer is unavailable or drifted."""


@dataclass(frozen=True, slots=True)
class _ScoringCudaState:
    runtime: object
    admission_observation: latency_module.SharedCudaObservation
    runtime_identity: latency_module.BaseLatencyRuntimeIdentity
    post_context_observation: latency_module.SharedCudaObservation


@dataclass(frozen=True, slots=True)
class HostBaseQualificationConfig:
    registry_path: Path
    registry_sha256: str
    output_root: Path
    journal_parent: Path
    session_id: str
    registered_cuda_uuid: str
    latency_protocol_sha256: str
    launcher_sha256: str
    wall_timeout_seconds: int
    schema: str = CONFIG_SCHEMA

    def __post_init__(self) -> None:
        for name in ("registry_path", "output_root", "journal_parent"):
            if not isinstance(getattr(self, name), Path):
                raise TypeError(f"{name} must be pathlib.Path")
        for name in (
            "registry_sha256",
            "latency_protocol_sha256",
            "launcher_sha256",
        ):
            _lower_sha256(getattr(self, name), name)
        if type(self.session_id) is not str or _SESSION_ID.fullmatch(self.session_id) is None:
            raise HostBaseQualificationError("qualification session_id is not one safe component")
        if (
            type(self.registered_cuda_uuid) is not str
            or _CUDA_UUID.fullmatch(self.registered_cuda_uuid) is None
        ):
            raise HostBaseQualificationError("registered CUDA UUID is invalid")
        if (
            type(self.wall_timeout_seconds) is not int
            or not 1 <= self.wall_timeout_seconds <= MAX_WALL_TIMEOUT_SECONDS
        ):
            raise HostBaseQualificationError("formal latency wall timeout is outside its bound")
        if self.schema != CONFIG_SCHEMA:
            raise HostBaseQualificationError("qualification controller schema changed")


@dataclass(frozen=True, slots=True, repr=False)
class HostBaseQualificationAuthentication:
    authorization: production_module.PrivateBaseQualificationAuthorization
    qualification_sha256: str
    adapter_sha256: str
    grant_path: Path
    qualification_path: Path
    grant_raw: bytes = field(repr=False)
    qualification_raw: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.authorization) is not production_module.PrivateBaseQualificationAuthorization:
            raise HostBaseQualificationError("qualification authorization type is invalid")
        for name in ("qualification_sha256", "adapter_sha256"):
            _lower_sha256(getattr(self, name), name)
        if not isinstance(self.grant_path, Path) or not isinstance(
            self.qualification_path, Path
        ):
            raise TypeError("qualification authentication paths must be pathlib.Path")
        if type(self.grant_raw) is not bytes or type(self.qualification_raw) is not bytes:
            raise TypeError("qualification authentication bytes must be exact bytes")
        if _sha256(self.grant_raw) != self.authorization.external_grant_sha256:
            raise HostBaseQualificationError("authorization does not bind the grant bytes")
        if _sha256(self.qualification_raw) != self.qualification_sha256:
            raise HostBaseQualificationError("authentication does not bind qualification bytes")

    def verifies(
        self,
        authorization: object,
        *,
        qualification_sha256: object,
        adapter_sha256: object,
    ) -> bool:
        if (
            authorization != self.authorization
            or qualification_sha256 != self.qualification_sha256
            or adapter_sha256 != self.adapter_sha256
        ):
            return False
        try:
            grant, grant_sha = resolver_module._read_owned_file(
                self.grant_path,
                label="qualification grant",
                maximum_bytes=MAX_OUTPUT_BYTES,
                expected_sha256=self.authorization.external_grant_sha256,
            )
            qualification, qualification_sha = resolver_module._read_owned_file(
                self.qualification_path,
                label="qualification artifact",
                maximum_bytes=MAX_OUTPUT_BYTES,
                expected_sha256=self.qualification_sha256,
            )
        except (OSError, TypeError, ValueError, RuntimeError):
            return False
        return (
            grant == self.grant_raw
            and grant_sha == self.authorization.external_grant_sha256
            and qualification == self.qualification_raw
            and qualification_sha == self.qualification_sha256
        )


@dataclass(frozen=True, slots=True, repr=False)
class HostBaseQualificationRun:
    execution: production_module.BackendExecution
    authentication: HostBaseQualificationAuthentication | None
    journal_path: Path | None
    registry_sha256: str
    formal: bool

    def __post_init__(self) -> None:
        if type(self.execution) is not production_module.BackendExecution:
            raise HostBaseQualificationError("controller execution type is invalid")
        _lower_sha256(self.registry_sha256, "registry_sha256")
        if type(self.formal) is not bool:
            raise TypeError("formal must be an exact bool")
        if self.execution.outcome == "COMPLETED":
            if not self.formal or type(self.authentication) is not HostBaseQualificationAuthentication:
                raise HostBaseQualificationError("only a formal run may return completion")
        elif self.authentication is not None or self.execution.semantic_evidence is not None:
            raise HostBaseQualificationError("non-completion cannot expose qualification evidence")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _lower_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise HostBaseQualificationError(f"{label} must be lowercase SHA-256 hex")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    try:
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
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise HostBaseQualificationError("controller value is not canonical JSON") from error


def _without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise HostBaseQualificationError("registry JSON repeats a key")
        result[key] = value
    return result


def _parse_canonical_object(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_without_duplicates,
            parse_float=lambda _value: (_ for _ in ()).throw(
                HostBaseQualificationError(f"{label} cannot contain floats")
            ),
            parse_constant=lambda _value: (_ for _ in ()).throw(
                HostBaseQualificationError(f"{label} cannot contain constants")
            ),
        )
    except HostBaseQualificationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise HostBaseQualificationError(f"{label} is not valid bounded JSON") from error
    if type(value) is not dict or raw != _canonical_json_bytes(value):
        raise HostBaseQualificationError(f"{label} is not one canonical object")
    return value


def _write_once(path: Path, raw: bytes) -> Path:
    if type(raw) is not bytes or not raw or len(raw) > MAX_OUTPUT_BYTES:
        raise HostBaseQualificationError("controller output bytes exceed their bound")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    for name in ("O_BINARY", "O_CLOEXEC", "O_NOFOLLOW"):
        flags |= int(getattr(os, name, 0))
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as error:
        raise HostBaseQualificationError("controller output already exists or cannot be created") from error
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(fd, raw[offset:])
            if written <= 0:
                raise OSError("short controller artifact write")
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)
    if os.name == "posix":
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return path


def _write_json_once(path: Path, value: object) -> tuple[Path, bytes]:
    raw = _canonical_json_bytes(value)
    return _write_once(path, raw), raw


def _checked_directory(path: Path, label: str) -> Path:
    if not isinstance(path, Path):
        raise TypeError(f"{label} must be pathlib.Path")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise HostBaseQualificationError(f"{label} is absent") from error
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
        raise HostBaseQualificationError(f"{label} must be a non-symlink directory")
    return path


def load_base_attempt_registry(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[tuple[resolver_module.BaseAttemptRegistryRow, ...], bytes, str]:
    """Read one canonical private registry into the resolver's exact rows."""

    expected = _lower_sha256(expected_sha256, "registry expected SHA-256")
    try:
        raw, digest = resolver_module._read_owned_file(
            path,
            label="base attempt registry",
            maximum_bytes=MAX_REGISTRY_BYTES,
            expected_sha256=expected,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise HostBaseQualificationError("base attempt registry bytes are invalid") from error
    value = _parse_canonical_object(raw, "base attempt registry")
    if set(value) != {
        "authority",
        "production",
        "result_claimed",
        "rows",
        "schema",
    }:
        raise HostBaseQualificationError("base attempt registry keys are not closed")
    if (
        type(value["authority"]) is not int
        or value["authority"] != 0
        or type(value["production"]) is not bool
        or value["production"] is not False
        or type(value["result_claimed"]) is not bool
        or value["result_claimed"] is not False
        or value["schema"] != REGISTRY_SCHEMA
        or type(value["rows"]) is not list
    ):
        raise HostBaseQualificationError("base attempt registry header is invalid")
    rows: list[resolver_module.BaseAttemptRegistryRow] = []
    for index, item in enumerate(value["rows"]):
        if type(item) is not dict or set(item) != {
            "attempt_id",
            "run_id",
            "terminal_outcome",
            "terminal_sha256",
        }:
            raise HostBaseQualificationError(f"base attempt registry row {index} is invalid")
        try:
            rows.append(resolver_module.BaseAttemptRegistryRow(**item))
        except (TypeError, ValueError, RuntimeError) as error:
            raise HostBaseQualificationError(
                f"base attempt registry row {index} is invalid"
            ) from error
    try:
        checked = resolver_module._checked_registry_rows(tuple(rows))
    except (TypeError, ValueError, RuntimeError) as error:
        raise HostBaseQualificationError("base attempt registry census is invalid") from error
    run_order = {run_id: index for index, run_id in enumerate(base_run_ids())}
    if tuple(run_order[row.run_id] for row in checked) != tuple(
        sorted(run_order[row.run_id] for row in checked)
    ):
        raise HostBaseQualificationError("base attempt registry rows changed registered run order")
    for run_id in base_run_ids():
        history = tuple(row for row in checked if row.run_id == run_id)
        if history[-1].terminal_outcome != "SUCCEEDED" or any(
            row.terminal_outcome == "SUCCEEDED" for row in history[:-1]
        ):
            raise HostBaseQualificationError(
                "each base registry history must end in its only successful attempt"
            )
    return checked, raw, digest


def _validate_formal_latency_timer_support(
    *,
    _os_name: str | None = None,
    _signal_api: object | None = None,
    _threading_api: object | None = None,
) -> None:
    """Check the real process timer boundary without mutating timer state."""

    os_name = os.name if _os_name is None else _os_name
    signal_api = signal if _signal_api is None else _signal_api
    threading_api = threading if _threading_api is None else _threading_api
    if os_name != "posix" or not all(
        hasattr(signal_api, name)
        for name in (
            "SIGALRM",
            "ITIMER_REAL",
            "getitimer",
            "getsignal",
            "setitimer",
            "signal",
        )
    ):
        raise _FormalLatencyTimerHold(
            "formal latency timeout requires POSIX interval timers"
        )
    if not all(
        callable(getattr(threading_api, name, None))
        for name in ("current_thread", "main_thread")
    ):
        raise _FormalLatencyTimerHold("formal latency timer thread API is unavailable")
    try:
        if threading_api.current_thread() is not threading_api.main_thread():
            raise _FormalLatencyTimerHold(
                "formal latency timeout requires the Python main thread"
            )
        prior_timer = signal_api.getitimer(signal_api.ITIMER_REAL)
        signal_api.getsignal(signal_api.SIGALRM)
    except _FormalLatencyTimerHold:
        raise
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise _FormalLatencyTimerHold(
            "formal latency timer capability check failed"
        ) from error
    if prior_timer != (0.0, 0.0):
        raise _FormalLatencyTimerHold(
            "formal latency refuses an inherited process timer"
        )


@contextmanager
def _formal_latency_timeout(
    seconds: int,
    *,
    _os_name: str | None = None,
    _signal_api: object | None = None,
    _threading_api: object | None = None,
) -> Iterator[None]:
    """Install the process-local formal timeout; the outer hard-kill is separate."""

    signal_api = signal if _signal_api is None else _signal_api
    if type(seconds) is not int or not 1 <= seconds <= MAX_WALL_TIMEOUT_SECONDS:
        raise HostBaseQualificationError("formal latency timeout is outside its bound")
    _validate_formal_latency_timer_support(
        _os_name=_os_name,
        _signal_api=signal_api,
        _threading_api=_threading_api,
    )
    try:
        prior_handler = signal_api.getsignal(signal_api.SIGALRM)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise _FormalLatencyTimerHold(
            "formal latency timer handler could not be read"
        ) from error

    def timeout_handler(_signum: int, _frame: object) -> None:
        raise TimeoutError("formal base latency wall timeout expired")

    primary_error: BaseException | None = None
    try:
        try:
            signal_api.signal(signal_api.SIGALRM, timeout_handler)
            signal_api.setitimer(signal_api.ITIMER_REAL, float(seconds), 0.0)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise _FormalLatencyTimerHold(
                "formal latency timer could not be installed"
            ) from error
        yield
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup_error: BaseException | None = None
        try:
            signal_api.setitimer(signal_api.ITIMER_REAL, 0.0, 0.0)
        except BaseException as error:
            cleanup_error = error
        try:
            signal_api.signal(signal_api.SIGALRM, prior_handler)
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
        if primary_error is None and cleanup_error is not None:
            if isinstance(cleanup_error, Exception):
                raise _FormalLatencyTimerHold(
                    "formal latency timer could not be restored"
                ) from cleanup_error
            raise cleanup_error


def _journal_terminal_payload(
    read: journal_module.LatencyJournalReadResult,
) -> dict[str, object]:
    if not read.records:
        raise HostBaseQualificationError("latency journal has no committed records")
    value = _parse_canonical_object(read.records[-1].raw, "latency terminal event")
    if value.get("event_type") != "TERMINAL" or type(value.get("payload")) is not dict:
        raise HostBaseQualificationError("latency journal did not end in a terminal event")
    return value["payload"]


def _scoring_precision(admission: resolver_module.BaseCohortAdmission) -> str:
    try:
        decision = training_module.resolve_precision(
            training_module.TrainingConfig(
                stage="base",
                seed=1729,
                device=admission.device,
                request_bf16=admission.request_bf16,
                bf16_runtime_qualified=admission.bf16_runtime_qualified,
                edge_budget=admission.edge_budget,
                checkpoint_every_updates=admission.checkpoint_every_updates,
                synthetic_contract=False,
            )
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise HostBaseQualificationError(
            "base scoring CUDA precision could not be resolved"
        ) from error
    return decision.mode


def _admit_scoring_cuda(
    config: HostBaseQualificationConfig,
    admission: resolver_module.BaseCohortAdmission,
    *,
    _runtime: object | None = None,
) -> _ScoringCudaState:
    """Apply the latency UUID/free/cap admission before any capture scoring."""

    runtime = (
        latency_module._NvidiaSmiTorchRuntime(config.registered_cuda_uuid)
        if _runtime is None
        else _runtime
    )
    try:
        device = latency_module.torch.device(admission.device)
        if device.type != "cuda":
            raise HostBaseQualificationError("base capture scoring requires CUDA")
        observed = runtime.snapshot()
        if type(observed) is not latency_module.SharedCudaObservation:
            raise _ScoringCudaHold(
                "RUNTIME_DRIFT",
                "base scoring admission snapshot is malformed",
            )
        latency_module._validate_snapshot(
            observed,
            registered_uuid=config.registered_cuda_uuid,
            runtime=None,
        )
        if observed.memory_free_mib < latency_module.MINIMUM_FREE_MIB:
            raise _ScoringCudaHold(
                "RESOURCE_LIMIT",
                "base scoring admission has less than 8 GiB free"
            )
        identity = runtime.initialize(
            device=device,
            precision_mode=_scoring_precision(admission),
            admission_observation=observed,
        )
        if type(identity) is not latency_module.BaseLatencyRuntimeIdentity:
            raise _ScoringCudaHold(
                "RUNTIME_DRIFT",
                "base scoring CUDA identity is malformed",
            )
        post_context = runtime.snapshot()
        if type(post_context) is not latency_module.SharedCudaObservation:
            raise _ScoringCudaHold(
                "RUNTIME_DRIFT",
                "base scoring post-context snapshot is malformed",
            )
        latency_module._validate_snapshot(
            post_context,
            registered_uuid=config.registered_cuda_uuid,
            runtime=identity,
        )
        if post_context.memory_free_mib < latency_module.MINIMUM_FREE_MIB:
            raise _ScoringCudaHold(
                "RESOURCE_LIMIT",
                "base scoring CUDA context has less than 8 GiB free"
            )
    except (HostBaseQualificationError, _ScoringCudaHold):
        raise
    except (TypeError, ValueError, RuntimeError) as error:
        raise _ScoringCudaHold(
            latency_module._failure_code(error),
            "base scoring CUDA admission failed",
        ) from error
    return _ScoringCudaState(
        runtime=runtime,
        admission_observation=observed,
        runtime_identity=identity,
        post_context_observation=post_context,
    )


def _strip_exception_tracebacks(error: BaseException) -> None:
    """Release scorer frames while retaining exact exception objects and chaining."""

    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        current.__traceback__ = None
        for linked in (current.__cause__, current.__context__):
            if linked is not None:
                pending.append(linked)


def _release_scoring_cuda(
    state: _ScoringCudaState,
    *,
    _torch_api: object | None = None,
) -> latency_module.SharedCudaObservation:
    """Clear scorer objects/workspaces, then retain the exact-zero latency gate."""

    torch_api = latency_module.torch if _torch_api is None else _torch_api
    gc.collect()
    try:
        try:
            device = torch_api.device(state.runtime_identity.device)
        except (AttributeError, TypeError, ValueError, RuntimeError) as error:
            raise _ScoringCudaHold(
                "RUNTIME_DRIFT",
                "base scoring CUDA release device is malformed",
            ) from error
        state.runtime.synchronize(device)
        clear_workspaces = getattr(
            getattr(torch_api, "_C", None),
            "_cuda_clearCublasWorkspaces",
            None,
        )
        if not callable(clear_workspaces):
            raise _ScoringCudaHold(
                "RUNTIME_DRIFT",
                "Torch CUDA cuBLAS workspace clear API is unavailable after scoring"
            )
        clear_workspaces()
        state.runtime.synchronize(device)
        torch_api.cuda.empty_cache()
        allocated = int(torch_api.cuda.memory_allocated(device))
        reserved = int(torch_api.cuda.memory_reserved(device))
        if allocated != 0 or reserved != 0:
            raise _ScoringCudaHold(
                "RESOURCE_LIMIT",
                "base scoring CUDA allocation remained before latency"
            )
        after = state.runtime.snapshot()
        if type(after) is not latency_module.SharedCudaObservation:
            raise _ScoringCudaHold(
                "RUNTIME_DRIFT",
                "base scoring release snapshot is malformed",
            )
        latency_module._validate_snapshot(
            after,
            registered_uuid=state.runtime_identity.device_uuid,
            runtime=state.runtime_identity,
        )
        if after.memory_free_mib < latency_module.MINIMUM_FREE_MIB:
            raise _ScoringCudaHold(
                "RESOURCE_LIMIT",
                "base scoring release has less than 8 GiB free"
            )
    except (HostBaseQualificationError, _ScoringCudaHold):
        raise
    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
        raise _ScoringCudaHold(
            latency_module._failure_code(error),
            "base scoring CUDA release failed",
        ) from error
    return after


def _failure_code(error: BaseException, stage: str) -> tuple[str, str]:
    if isinstance(error, _NonformalRuntimeHold):
        return "HELD", "NONFORMAL_TEST_RUNTIME"
    if isinstance(error, _ScoringCudaHold):
        return "HELD", f"BASE_SCORING_CUDA_{error.failure_code}"
    if isinstance(error, latency_module.BaseCohortLatencyHold):
        return "HELD", f"LATENCY_{error.failure_code}"
    if isinstance(error, _FormalLatencyTimerHold):
        return "HELD", "LATENCY_RUNTIME_DRIFT"
    if isinstance(error, (TimeoutError, journal_module.LatencyJournalWriteError)):
        return "HELD", "LATENCY_OBSERVATION_INCOMPLETE"
    if isinstance(error, resolver_module.BaseCohortResolutionError):
        return "HELD", "BASE_COHORT_INPUT_INVALID"
    if isinstance(error, HostBaseQualificationError) and stage in {
        "REGISTRY",
        "RESOLVE",
    }:
        return "HELD", "BASE_COHORT_INPUT_INVALID"
    chain: list[BaseException] = []
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__
    if any(
        isinstance(item, (CaptureValidationResourceLimit, latency_module.torch.cuda.OutOfMemoryError))
        for item in chain
    ):
        return "HELD", "BASE_CAPTURE_SCORE_RESOURCE_LIMIT"
    if isinstance(error, scoring_module.BaseCohortValidationError):
        return "FAILED", "BASE_CAPTURE_SCORE_FAILED"
    if isinstance(error, assembly_module.BaseCohortQualificationError):
        return "FAILED", "BASE_QUALIFICATION_ASSEMBLY_FAILED"
    return "FAILED", "BASE_QUALIFICATION_CONTROLLER_FAILED"


def _persist_failure(
    output_directory: Path,
    *,
    error: BaseException,
    stage: str,
    registry_sha256: str,
    journal_path: Path | None,
    journal_sha256: str | None,
    secondary_cleanup_error_type: str | None,
    completed_at_utc: str,
) -> tuple[production_module.BackendExecution, Path]:
    outcome, failure_code = _failure_code(error, stage)
    path, _raw = _write_json_once(
        output_directory / "failure.json",
        {
            "authority": 0,
            "completed_at_utc": completed_at_utc,
            "error_type": type(error).__qualname__,
            "failure_code": failure_code,
            "formal_qualification_completed": False,
            "journal_present": journal_path is not None,
            "journal_read_sha256": journal_sha256,
            "production": False,
            "registry_sha256": registry_sha256,
            "result_claimed": False,
            "schema": FAILURE_SCHEMA,
            "secondary_cleanup_error_type": secondary_cleanup_error_type,
            "stage": stage,
            "status": outcome,
        },
    )
    return (
        production_module.BackendExecution(
            outcome=outcome,
            completed_at_utc=completed_at_utc,
            artifacts=(path,),
        ),
        path,
    )


def run_host_base_qualification(
    attempt_root: Path,
    *,
    config: HostBaseQualificationConfig,
    admission: resolver_module.BaseCohortAdmission,
    validation_source: CaptureValidationSource,
    output_directory: Path,
    adapter_sha256: str,
    _test_runtime: object | None = None,
) -> HostBaseQualificationRun:
    """Execute the actual chain; an injected runtime can only return nonformal HOLD."""

    if type(config) is not HostBaseQualificationConfig:
        raise TypeError("config must be exact HostBaseQualificationConfig")
    if type(admission) is not resolver_module.BaseCohortAdmission:
        raise TypeError("admission must be exact BaseCohortAdmission")
    if type(validation_source) is not CaptureValidationSource:
        raise TypeError("validation_source must be exact CaptureValidationSource")
    _lower_sha256(adapter_sha256, "adapter_sha256")
    root = _checked_directory(attempt_root, "base attempt root")
    output = _checked_directory(output_directory, "qualification output directory")
    output_root = _checked_directory(
        config.output_root,
        "base qualification output root",
    )
    journal_parent = _checked_directory(
        config.journal_parent,
        "latency journal parent",
    )
    if len({root.resolve(), output_root.resolve(), journal_parent.resolve()}) != 3:
        raise HostBaseQualificationError(
            "attempt, qualification-output, and journal roots must be distinct"
        )
    if output.resolve() != (output_root / config.session_id).resolve():
        raise HostBaseQualificationError(
            "qualification output directory differs from its frozen session path"
        )
    formal = _test_runtime is None
    stage = "LATENCY_PREFLIGHT" if formal else "REGISTRY"
    registry_digest = config.registry_sha256
    journal_path: Path | None = None
    journal_sha256: str | None = None
    scoring_cuda_path: Path | None = None
    scoring_cuda_raw: bytes | None = None
    secondary_cleanup_error_type: str | None = None
    start_path: Path | None = None
    try:
        if formal:
            _validate_formal_latency_timer_support()
            stage = "REGISTRY"
        registry_rows, registry_raw, registry_digest = load_base_attempt_registry(
            config.registry_path,
            expected_sha256=config.registry_sha256,
        )
        start_path, _ = _write_json_once(
            output / "controller-start.json",
            {
                "adapter_sha256": adapter_sha256,
                "authority": 0,
                "formal_execution_requested": formal,
                "journal_binding": {
                    "latency_protocol_sha256": config.latency_protocol_sha256,
                    "launcher_sha256": config.launcher_sha256,
                    "registered_cuda_uuid": config.registered_cuda_uuid,
                },
                "production": False,
                "registry_sha256": registry_digest,
                "result_claimed": False,
                "schema": START_SCHEMA,
                "session_id": config.session_id,
                "started_at_utc": _utc_now(),
                "source_tree_sha256": admission.source_tree_sha256,
                "train_manifest_sha256": admission.train_manifest_sha256,
                "val_census_sha256": admission.val_manifest_sha256,
                "wall_timeout_seconds": config.wall_timeout_seconds,
            },
        )
        stage = "RESOLVE"
        cohort = resolver_module.resolve_completed_base_cohort(
            root,
            registry_rows=registry_rows,
            admission=admission,
        )
        stage = "SCORE"
        if formal:
            scoring_state = _admit_scoring_cuda(config, admission)
            score_error: Exception | None = None
            scored = None
            try:
                scored = scoring_module.score_resolved_base_cohort(
                    cohort,
                    admission=admission,
                    validation_source=validation_source,
                )
            except Exception as error:
                score_error = error
                _strip_exception_tracebacks(error)
            release_error: Exception | None = None
            try:
                post_score_observation = _release_scoring_cuda(scoring_state)
            except Exception as error:
                release_error = error
                _strip_exception_tracebacks(error)
            if score_error is not None:
                if release_error is not None:
                    secondary_cleanup_error_type = type(release_error).__qualname__
                raise score_error
            if release_error is not None:
                raise release_error
            if type(scored) is not scoring_module.BaseCohortValidationResult:
                raise HostBaseQualificationError(
                    "base scorer returned an invalid cohort result"
                )
            if any(
                row.device != scoring_state.runtime_identity.device
                or row.precision_mode != scoring_state.runtime_identity.precision_mode
                for row in scored.observations
            ):
                raise HostBaseQualificationError(
                    "base scorer device or precision differs from CUDA admission"
                )
            scoring_cuda_path, scoring_cuda_raw = _write_json_once(
                output / "scoring-cuda.json",
                {
                    "admission_observation": scoring_state.admission_observation.payload(),
                    "allocator_limit_bytes": latency_module.CUDA_ALLOCATOR_LIMIT_BYTES,
                    "authority": 0,
                    "post_context_observation": (
                        scoring_state.post_context_observation.payload()
                    ),
                    "post_score_observation": post_score_observation.payload(),
                    "production": False,
                    "registered_cuda_uuid": config.registered_cuda_uuid,
                    "result_claimed": False,
                    "runtime_identity": asdict(scoring_state.runtime_identity),
                    "schema": SCORING_CUDA_SCHEMA,
                    "scored_cohort_sha256": scored.sha256,
                    "status": "RESOURCE_BOUND_SCORING_COMPLETED",
                },
            )
        else:
            scored = scoring_module.score_resolved_base_cohort(
                cohort,
                admission=admission,
                validation_source=validation_source,
            )
        bindings = journal_module.LatencyJournalBindings(
            resolved_cohort_sha256=latency_module._resolved_cohort_sha256(cohort),
            scored_cohort_sha256=scored.sha256,
            source_tree_sha256=cohort.source_tree_sha256,
            latency_protocol_sha256=config.latency_protocol_sha256,
            wrapper_sha256=config.launcher_sha256,
            registered_cuda_uuid=config.registered_cuda_uuid,
        )
        stage = "LATENCY"
        with journal_module.create_latency_progress_journal(
            config.journal_parent,
            session_id=config.session_id,
            bindings=bindings,
        ) as writer:
            journal_path = writer.path
            with _formal_latency_timeout(config.wall_timeout_seconds):
                if _test_runtime is None:
                    latency = latency_module.run_base_cohort_latency_session(
                        cohort,
                        admission=admission,
                        scored_cohort=scored,
                        registered_cuda_uuid=config.registered_cuda_uuid,
                        observer=writer,
                    )
                else:
                    latency = latency_module.run_base_cohort_latency_session(
                        cohort,
                        admission=admission,
                        scored_cohort=scored,
                        registered_cuda_uuid=config.registered_cuda_uuid,
                        observer=writer,
                        _runtime=_test_runtime,
                    )
        read = journal_module.read_latency_progress_journal(
            journal_path,
            expected_bindings=bindings,
        )
        journal_sha256 = read.sha256
        terminal_payload = _journal_terminal_payload(read)
        if (
            read.status != journal_module.COMPLETED_STATUS
            or read.terminal_outcome != "COMPLETED"
            or terminal_payload.get("outcome") != "COMPLETED"
            or terminal_payload.get("failure_code") is not None
            or terminal_payload.get("latency_session_sha256") != latency.sha256
        ):
            raise HostBaseQualificationError(
                "latency call result and durable terminal do not form one completion"
            )
        if not formal:
            raise _NonformalRuntimeHold(
                "injected latency runtime is an engineering observation, not qualification"
            )
        if (
            type(latency) is not latency_module.BaseCohortLatencySession
            or latency.actual_runtime is not True
            or latency.complete is not True
            or latency.formal_samples_complete is not True
            or latency.failure_code is not None
            or latency.status != latency_module.ACTUAL_COMPLETE_STATUS
        ):
            raise HostBaseQualificationError("formal latency session is incomplete")
        stage = "ASSEMBLY"
        assembled = assembly_module.assemble_base_cohort_qualification(
            cohort,
            admission=admission,
            scored_cohort=scored,
            latency_session=latency,
        )
        if _sha256(registry_raw) != registry_digest:
            raise HostBaseQualificationError("owned registry bytes changed in memory")
        registry_after, registry_after_digest = resolver_module._read_owned_file(
            config.registry_path,
            label="base attempt registry",
            maximum_bytes=MAX_REGISTRY_BYTES,
            expected_sha256=registry_digest,
        )
        if registry_after != registry_raw or registry_after_digest != registry_digest:
            raise HostBaseQualificationError("base attempt registry changed during qualification")
        if (
            validation_source.census_sha256 != admission.val_manifest_sha256
            or validation_source.manifest_sha256 != scored.source_manifest_sha256
            or latency.sha256 != assembled.dependency_binding.latency_session_sha256
        ):
            raise HostBaseQualificationError("qualification source or latency changed after assembly")
        stage = "PERSIST"
        if scoring_cuda_path is None or scoring_cuda_raw is None:
            raise AssertionError("formal qualification lost scoring CUDA evidence")
        qualification_path = _write_once(output / "base-qualification.json", assembled.artifact)
        dependency_path, dependency_raw = _write_json_once(
            output / "dependency-binding.json",
            asdict(assembled.dependency_binding),
        )
        grant_path, grant_raw = _write_json_once(
            output / "grant-basis.json",
            {
                "adapter_sha256": adapter_sha256,
                "assembly_sha256": assembled.sha256,
                "authority": 0,
                "dependency_binding_sha256": assembled.dependency_binding.sha256,
                "journal_read_sha256": journal_sha256,
                "latency_session_sha256": latency.sha256,
                "production": False,
                "qualification_sha256": _sha256(assembled.artifact),
                "registry_sha256": registry_digest,
                "result_claimed": False,
                "row_binding_sha256s": [row.sha256 for row in assembled.row_bindings],
                "schema": GRANT_BASIS_SCHEMA,
                "scoring_cuda_sha256": _sha256(scoring_cuda_raw),
            },
        )
        authorization = production_module.PrivateBaseQualificationAuthorization(
            authorized_at_utc=_utc_now(),
            external_grant_sha256=_sha256(grant_raw),
            validation_manifest_sha256=assembled.qualification.validation_manifest_sha256,
            query_census_sha256=assembled.qualification.query_census_sha256,
            evaluator_sha256=assembled.qualification.evaluator_sha256,
            score_rows_sha256=assembled.qualification.score_rows_sha256,
        )
        authorization_path, authorization_raw = _write_json_once(
            output / "authorization.json",
            {
                **asdict(authorization),
                "authority": 0,
                "production": False,
                "result_claimed": False,
                "schema": AUTHORIZATION_SCHEMA,
            },
        )
        qualification_sha256 = _sha256(assembled.artifact)
        completed_at_utc = _utc_now()
        completion_path, _completion_raw = _write_json_once(
            output / "completion.json",
            {
                "artifact_sha256s": {
                    "authorization": _sha256(authorization_raw),
                    "dependency_binding": _sha256(dependency_raw),
                    "grant_basis": _sha256(grant_raw),
                    "qualification": qualification_sha256,
                    "scoring_cuda": _sha256(scoring_cuda_raw),
                },
                "assembly_sha256": assembled.sha256,
                "authority": 0,
                "completed_at_utc": completed_at_utc,
                "journal_read_sha256": journal_sha256,
                "production": False,
                "registry_sha256": registry_digest,
                "result_claimed": False,
                "schema": COMPLETION_SCHEMA,
                "status": "COMPLETED_PENDING_EXISTING_ADAPTER_RECOMPUTATION",
            },
        )
        authentication = HostBaseQualificationAuthentication(
            authorization=authorization,
            qualification_sha256=qualification_sha256,
            adapter_sha256=adapter_sha256,
            grant_path=grant_path,
            qualification_path=qualification_path,
            grant_raw=grant_raw,
            qualification_raw=assembled.artifact,
        )
        execution = production_module.BackendExecution(
            outcome="COMPLETED",
            completed_at_utc=completed_at_utc,
            artifacts=(
                qualification_path,
                scoring_cuda_path,
                dependency_path,
                grant_path,
                authorization_path,
                completion_path,
            ),
            semantic_evidence=assembled.execution_evidence,
        )
        return HostBaseQualificationRun(
            execution=execution,
            authentication=authentication,
            journal_path=journal_path,
            registry_sha256=registry_digest,
            formal=True,
        )
    except Exception as error:
        if journal_path is not None and journal_path.exists():
            try:
                # A structurally complete terminal remains nonformal unless it
                # agrees with the actual call outcome consumed above.
                if "bindings" in locals():
                    observed = journal_module.read_latency_progress_journal(
                        journal_path,
                        expected_bindings=bindings,
                    )
                    journal_sha256 = observed.sha256
            except (OSError, TypeError, ValueError, RuntimeError):
                journal_sha256 = None
        execution, _failure_path = _persist_failure(
            output,
            error=error,
            stage=stage,
            registry_sha256=registry_digest,
            journal_path=journal_path,
            journal_sha256=journal_sha256,
            secondary_cleanup_error_type=secondary_cleanup_error_type,
            completed_at_utc=_utc_now(),
        )
        prefix = tuple(
            path for path in (start_path, scoring_cuda_path) if path is not None
        )
        artifacts = (*prefix, *execution.artifacts)
        return HostBaseQualificationRun(
            execution=production_module.BackendExecution(
                outcome=execution.outcome,
                completed_at_utc=execution.completed_at_utc,
                artifacts=artifacts,
            ),
            authentication=None,
            journal_path=journal_path,
            registry_sha256=registry_digest,
            formal=formal,
        )


__all__ = [
    "CONFIG_SCHEMA",
    "HostBaseQualificationAuthentication",
    "HostBaseQualificationConfig",
    "HostBaseQualificationError",
    "HostBaseQualificationRun",
    "MAX_WALL_TIMEOUT_SECONDS",
    "REGISTRY_SCHEMA",
    "load_base_attempt_registry",
    "run_host_base_qualification",
]
