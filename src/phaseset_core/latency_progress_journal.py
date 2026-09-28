"""POSIX write-once progress journal for the formal base-latency controller.

The journal preserves a durable event prefix across ordinary failures or a
hard process kill.  It is diagnostic evidence only: neither a terminal event
nor a complete prefix is accepted as formal latency or qualification evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
from typing import Callable, Final

from . import base_cohort_latency as latency_module


AUTHORITY: Final = 0
PRODUCTION: Final = False
RESULT_CLAIMED: Final = False
EVENT_SCHEMA: Final = "phaseset-latency-progress-event-v1"
MARKER_SCHEMA: Final = "phaseset-latency-progress-completion-marker-v1"
BINDING_SCHEMA: Final = "phaseset-latency-progress-binding-v1"
READ_SCHEMA: Final = "phaseset-latency-progress-read-v1"
INCOMPLETE_STATUS: Final = "INCOMPLETE_PREFIX_NOT_FORMAL"
COMPLETED_STATUS: Final = "TERMINAL_COMPLETED_PREFIX_NOT_FORMAL"
HELD_STATUS: Final = "TERMINAL_HELD_PREFIX_NOT_FORMAL"
MAX_EVENTS: Final = 4096
MAX_EVENT_BYTES: Final = 64 * 1024
MAX_MARKER_BYTES: Final = 1024
ZERO_SHA256: Final = "0" * 64

_SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_EVENT_NAME = re.compile(r"event-([0-9]{8})\.json")
_MARKER_NAME = re.compile(r"event-([0-9]{8})\.complete")
_EVENT_TYPES = frozenset(
    {
        "SESSION_START",
        "CONTEXT_READY",
        "VISIT_START",
        "WARMUP_COMPLETED",
        "TIMED_SAMPLE_COMPLETED",
        "VISIT_END",
        "TERMINAL",
    }
)
_FAILURE_CODES = frozenset(
    {
        "CUDA_OUT_OF_MEMORY",
        "SESSION_TIMEOUT",
        "RUNTIME_DRIFT",
        "RESOURCE_LIMIT",
        "CHECKPOINT_OR_SCORE_DRIFT",
        "MALFORMED_MEASUREMENT",
        "UNEXPECTED_SESSION_FAILURE",
    }
)


class LatencyJournalError(ValueError):
    """The journal request, durable record, or on-disk tree is invalid."""


class LatencyJournalWriteError(RuntimeError):
    """A write failed; the session directory retains its physical prefix."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _lower_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LatencyJournalError(f"{label} must be lowercase SHA-256 hex")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or not 1 <= value <= (1 << 63) - 1:
        raise LatencyJournalError(f"{label} must be an exact positive int")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or not 0 <= value <= (1 << 63) - 1:
        raise LatencyJournalError(f"{label} must be an exact nonnegative int")
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


def _reject_float(value: str) -> object:
    raise LatencyJournalError(f"floating-point JSON is forbidden: {value}")


def _reject_constant(value: str) -> object:
    raise LatencyJournalError(f"non-finite JSON is forbidden: {value}")


def _object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise LatencyJournalError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_canonical_object(raw: bytes, label: str) -> dict[str, object]:
    if type(raw) is not bytes:
        raise LatencyJournalError(f"{label} must be exact bytes")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_object_without_duplicates,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, RecursionError, ValueError) as error:
        raise LatencyJournalError(f"{label} is not strict canonical JSON") from error
    if type(value) is not dict or _canonical_json_bytes(value) != raw:
        raise LatencyJournalError(f"{label} differs from canonical JSON bytes")
    return value


def _exact_keys(value: dict[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise LatencyJournalError(f"{label} keys changed")


def _safe_session_id(value: object) -> str:
    if type(value) is not str or _SESSION_ID.fullmatch(value) is None or value in {".", ".."}:
        raise LatencyJournalError("session_id must be one safe ASCII component")
    return value


def _require_posix() -> None:
    if os.name != "posix" or not all(
        hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC")
    ):
        raise LatencyJournalError("latency journal v1 requires POSIX nofollow support")


@dataclass(frozen=True, slots=True)
class LatencyJournalBindings:
    resolved_cohort_sha256: str
    scored_cohort_sha256: str
    source_tree_sha256: str
    latency_protocol_sha256: str
    wrapper_sha256: str
    registered_cuda_uuid: str
    schema: str = BINDING_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "resolved_cohort_sha256",
            "scored_cohort_sha256",
            "source_tree_sha256",
            "latency_protocol_sha256",
            "wrapper_sha256",
        ):
            _lower_sha256(getattr(self, name), name)
        if (
            type(self.registered_cuda_uuid) is not str
            or not self.registered_cuda_uuid.startswith("GPU-")
            or len(self.registered_cuda_uuid) > 80
        ):
            raise LatencyJournalError("registered CUDA UUID is invalid")
        if self.schema != BINDING_SCHEMA:
            raise LatencyJournalError("latency journal binding schema changed")

    def payload(self) -> dict[str, object]:
        return {
            "latency_protocol_sha256": self.latency_protocol_sha256,
            "registered_cuda_uuid": self.registered_cuda_uuid,
            "resolved_cohort_sha256": self.resolved_cohort_sha256,
            "schema": self.schema,
            "scored_cohort_sha256": self.scored_cohort_sha256,
            "source_tree_sha256": self.source_tree_sha256,
            "wrapper_sha256": self.wrapper_sha256,
        }

    @property
    def sha256(self) -> str:
        return _sha256(_canonical_json_bytes(self.payload()))


@dataclass(frozen=True, slots=True, repr=False)
class LatencyJournalRecord:
    sequence: int
    event_type: str
    previous_event_sha256: str
    binding_sha256: str
    raw: bytes
    marker_raw: bytes

    def __post_init__(self) -> None:
        _nonnegative_int(self.sequence, "event sequence")
        if self.event_type not in _EVENT_TYPES:
            raise LatencyJournalError("event type is outside the closed vocabulary")
        _lower_sha256(self.previous_event_sha256, "previous_event_sha256")
        _lower_sha256(self.binding_sha256, "binding_sha256")
        if type(self.raw) is not bytes or not self.raw or len(self.raw) > MAX_EVENT_BYTES:
            raise LatencyJournalError("journal record raw bytes are invalid")
        if (
            type(self.marker_raw) is not bytes
            or not self.marker_raw
            or len(self.marker_raw) > MAX_MARKER_BYTES
        ):
            raise LatencyJournalError("journal marker raw bytes are invalid")

    @property
    def sha256(self) -> str:
        return _sha256(self.raw)


@dataclass(frozen=True, slots=True, repr=False)
class LatencyJournalReadResult:
    session_id: str
    bindings: LatencyJournalBindings
    records: tuple[LatencyJournalRecord, ...]
    committed_event_raw: tuple[bytes, ...]
    committed_marker_raw: tuple[bytes, ...]
    uncommitted_tail_files: tuple[tuple[str, bytes], ...]
    status: str
    terminal_outcome: str | None
    formal_latency: bool = False
    authority: int = AUTHORITY
    production: bool = PRODUCTION
    result_claimed: bool = RESULT_CLAIMED
    schema: str = READ_SCHEMA

    def __post_init__(self) -> None:
        _safe_session_id(self.session_id)
        if type(self.bindings) is not LatencyJournalBindings:
            raise LatencyJournalError("journal read bindings are invalid")
        if type(self.records) is not tuple or any(
            type(record) is not LatencyJournalRecord for record in self.records
        ):
            raise LatencyJournalError("journal record census is invalid")
        if self.committed_event_raw != tuple(record.raw for record in self.records):
            raise LatencyJournalError("journal committed event bytes changed")
        if self.committed_marker_raw != tuple(record.marker_raw for record in self.records):
            raise LatencyJournalError("journal committed marker bytes changed")
        if type(self.uncommitted_tail_files) is not tuple or len(self.uncommitted_tail_files) > 2:
            raise LatencyJournalError("journal uncommitted tail census is invalid")
        for name, raw in self.uncommitted_tail_files:
            if type(name) is not str or type(raw) is not bytes:
                raise LatencyJournalError("journal uncommitted tail is malformed")
            limit = MAX_MARKER_BYTES if name.endswith(".complete") else MAX_EVENT_BYTES
            if len(raw) > limit:
                raise LatencyJournalError("journal uncommitted tail exceeds its byte bound")
        expected = {
            INCOMPLETE_STATUS: None,
            COMPLETED_STATUS: "COMPLETED",
            HELD_STATUS: "HELD",
        }
        if self.status not in expected or self.terminal_outcome != expected[self.status]:
            raise LatencyJournalError("journal read terminal status is inconsistent")
        if self.status != INCOMPLETE_STATUS and self.uncommitted_tail_files:
            raise LatencyJournalError("terminal journal cannot contain an extra tail")
        if (
            self.formal_latency is not False
            or type(self.authority) is not int
            or self.authority != AUTHORITY
            or self.production is not PRODUCTION
            or self.result_claimed is not RESULT_CLAIMED
            or self.schema != READ_SCHEMA
        ):
            raise LatencyJournalError("journal read result cannot claim formal authority")

    @property
    def sha256(self) -> str:
        return _sha256(
            _canonical_json_bytes(
                {
                    "authority": self.authority,
                    "binding_sha256": self.bindings.sha256,
                    "event_sha256s": [record.sha256 for record in self.records],
                    "formal_latency": self.formal_latency,
                    "production": self.production,
                    "result_claimed": self.result_claimed,
                    "schema": self.schema,
                    "session_id": self.session_id,
                    "status": self.status,
                    "tail_files": [
                        {"name": name, "sha256": _sha256(raw)}
                        for name, raw in self.uncommitted_tail_files
                    ],
                    "terminal_outcome": self.terminal_outcome,
                }
            )
        )


@dataclass(frozen=True, slots=True)
class _JournalState:
    stage: str = "EMPTY"
    visit_index: int = 0
    warmups_completed: int = 0
    timed_completed: int = 0
    held_failure_code: str | None = None
    terminal_outcome: str | None = None

    def transition(self, event_type: str, payload: dict[str, object]) -> _JournalState:
        if self.terminal_outcome is not None:
            raise LatencyJournalError("journal contains an event after terminal")
        if event_type == "SESSION_START":
            _exact_keys(
                payload,
                {"admission_observation_sha256", "bindings", "started_unix_ns"},
                "SESSION_START payload",
            )
            _lower_sha256(
                payload["admission_observation_sha256"],
                "admission_observation_sha256",
            )
            _positive_int(payload["started_unix_ns"], "started_unix_ns")
            if self.stage != "EMPTY":
                raise LatencyJournalError("SESSION_START must be the first event")
            return replace(self, stage="STARTED")
        if event_type == "CONTEXT_READY":
            _exact_keys(
                payload,
                {"post_context_observation_sha256", "runtime_sha256"},
                "CONTEXT_READY payload",
            )
            _lower_sha256(payload["runtime_sha256"], "runtime_sha256")
            _lower_sha256(
                payload["post_context_observation_sha256"],
                "post_context_observation_sha256",
            )
            if self.stage != "STARTED":
                raise LatencyJournalError("CONTEXT_READY is out of order")
            return replace(self, stage="BETWEEN_VISITS")
        if event_type == "VISIT_START":
            self._validate_visit_identity(payload, {"before_observation_sha256"})
            _lower_sha256(payload["before_observation_sha256"], "before_observation_sha256")
            if self.stage != "BETWEEN_VISITS" or self.visit_index >= 81:
                raise LatencyJournalError("VISIT_START is out of order")
            return replace(
                self,
                stage="ACTIVE_VISIT",
                warmups_completed=0,
                timed_completed=0,
            )
        if event_type == "WARMUP_COMPLETED":
            self._validate_visit_identity(payload, {"warmup_index"})
            index = _nonnegative_int(payload["warmup_index"], "warmup_index")
            if (
                self.stage != "ACTIVE_VISIT"
                or self.timed_completed != 0
                or index != self.warmups_completed
                or index >= latency_module.WARMUPS_PER_VISIT
            ):
                raise LatencyJournalError("WARMUP_COMPLETED is out of order")
            return replace(self, warmups_completed=self.warmups_completed + 1)
        if event_type == "TIMED_SAMPLE_COMPLETED":
            self._validate_visit_identity(payload, {"sample_index", "sample_ns"})
            index = _nonnegative_int(payload["sample_index"], "sample_index")
            _positive_int(payload["sample_ns"], "sample_ns")
            if (
                self.stage != "ACTIVE_VISIT"
                or self.warmups_completed != latency_module.WARMUPS_PER_VISIT
                or index != self.timed_completed
                or index >= latency_module.TIMED_PER_VISIT
            ):
                raise LatencyJournalError("TIMED_SAMPLE_COMPLETED is out of order")
            return replace(self, timed_completed=self.timed_completed + 1)
        if event_type == "VISIT_END":
            self._validate_visit_identity(
                payload,
                {
                    "after_observation_sha256",
                    "failure_code",
                    "outcome",
                    "peak_allocated_bytes",
                    "peak_reserved_bytes",
                },
            )
            if self.stage != "ACTIVE_VISIT":
                raise LatencyJournalError("VISIT_END is out of order")
            outcome = payload["outcome"]
            if outcome == "COMPLETED":
                if (
                    self.warmups_completed != latency_module.WARMUPS_PER_VISIT
                    or self.timed_completed != latency_module.TIMED_PER_VISIT
                    or payload["failure_code"] is not None
                ):
                    raise LatencyJournalError("completed visit has an incomplete event census")
                _lower_sha256(
                    payload["after_observation_sha256"],
                    "after_observation_sha256",
                )
                _nonnegative_int(payload["peak_allocated_bytes"], "peak_allocated_bytes")
                _nonnegative_int(payload["peak_reserved_bytes"], "peak_reserved_bytes")
                return replace(
                    self,
                    stage="BETWEEN_VISITS",
                    visit_index=self.visit_index + 1,
                    warmups_completed=0,
                    timed_completed=0,
                )
            if outcome != "HELD":
                raise LatencyJournalError("visit outcome is outside the closed vocabulary")
            failure = payload["failure_code"]
            if type(failure) is not str or failure not in _FAILURE_CODES:
                raise LatencyJournalError("held visit failure_code is invalid")
            for name in (
                "after_observation_sha256",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
            ):
                value = payload[name]
                if value is not None:
                    if name.endswith("sha256"):
                        _lower_sha256(value, name)
                    else:
                        _nonnegative_int(value, name)
            if (payload["peak_allocated_bytes"] is None) != (
                payload["peak_reserved_bytes"] is None
            ):
                raise LatencyJournalError("held visit peak pair is incomplete")
            return replace(self, stage="HELD_VISIT", held_failure_code=failure)
        if event_type == "TERMINAL":
            _exact_keys(
                payload,
                {"failure_code", "latency_session_sha256", "outcome"},
                "TERMINAL payload",
            )
            _lower_sha256(payload["latency_session_sha256"], "latency_session_sha256")
            outcome = payload["outcome"]
            if outcome == "COMPLETED":
                if (
                    self.stage != "BETWEEN_VISITS"
                    or self.visit_index != 81
                    or payload["failure_code"] is not None
                ):
                    raise LatencyJournalError("completed terminal has an incomplete visit census")
            elif outcome == "HELD":
                failure = payload["failure_code"]
                if type(failure) is not str or failure not in _FAILURE_CODES:
                    raise LatencyJournalError("held terminal failure_code is invalid")
                if self.stage not in {"STARTED", "BETWEEN_VISITS", "HELD_VISIT"}:
                    raise LatencyJournalError("held terminal cannot skip an active visit end")
                if self.stage == "HELD_VISIT" and failure != self.held_failure_code:
                    raise LatencyJournalError(
                        "held terminal failure_code differs from held visit"
                    )
            else:
                raise LatencyJournalError("terminal outcome is outside the closed vocabulary")
            return replace(self, stage="TERMINAL", terminal_outcome=outcome)
        raise LatencyJournalError("event type is outside the closed vocabulary")

    def _validate_visit_identity(
        self,
        payload: dict[str, object],
        additional_keys: set[str],
    ) -> None:
        _exact_keys(
            payload,
            {"ordinal_in_round", "round_index", "run_id"} | additional_keys,
            "visit event payload",
        )
        round_index = _nonnegative_int(payload["round_index"], "round_index")
        ordinal = _nonnegative_int(payload["ordinal_in_round"], "ordinal_in_round")
        if round_index >= latency_module.ROUND_COUNT or ordinal >= 9:
            raise LatencyJournalError("visit coordinate is outside the schedule")
        expected_index = round_index * 9 + ordinal
        schedule = latency_module.cyclic_visit_schedule()
        if (
            expected_index != self.visit_index
            or type(payload["run_id"]) is not str
            or payload["run_id"] != schedule[round_index][ordinal]
        ):
            raise LatencyJournalError("visit identity differs from the cyclic schedule")


def _event_payload(
    *,
    sequence: int,
    session_id: str,
    event_type: str,
    previous_event_sha256: str,
    binding_sha256: str,
    recorded_unix_ns: int,
    payload: dict[str, object],
) -> dict[str, object]:
    if event_type not in _EVENT_TYPES:
        raise LatencyJournalError("event type is outside the closed vocabulary")
    return {
        "binding_sha256": binding_sha256,
        "event_type": event_type,
        "payload": payload,
        "previous_event_sha256": previous_event_sha256,
        "recorded_unix_ns": recorded_unix_ns,
        "schema": EVENT_SCHEMA,
        "sequence": sequence,
        "session_id": session_id,
    }


def _event_names(sequence: int) -> tuple[str, str]:
    if type(sequence) is not int or not 0 <= sequence < MAX_EVENTS:
        raise LatencyJournalError("event sequence exceeds the fixed journal bound")
    return f"event-{sequence:08d}.json", f"event-{sequence:08d}.complete"


def _completion_marker_raw(sequence: int, event_sha256: str) -> bytes:
    _lower_sha256(event_sha256, "completion event_sha256")
    return _canonical_json_bytes(
        {
            "event_sha256": event_sha256,
            "schema": MARKER_SCHEMA,
            "sequence": sequence,
        }
    )


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _directory_identity(value: os.stat_result) -> tuple[int, int, int]:
    return value.st_dev, value.st_ino, value.st_mode


def _open_directory(path: Path, *, private: bool) -> tuple[int, os.stat_result]:
    _require_posix()
    try:
        before = path.lstat()
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as error:
        raise LatencyJournalError("journal directory could not be opened safely") from error
    try:
        opened = os.fstat(fd)
        if (
            not stat.S_ISDIR(before.st_mode)
            or not stat.S_ISDIR(opened.st_mode)
            or _stat_identity(before) != _stat_identity(opened)
            or (private and stat.S_IMODE(opened.st_mode) & 0o077)
        ):
            raise LatencyJournalError("journal directory identity or mode is unsafe")
    except Exception:
        os.close(fd)
        raise
    return fd, opened


def _check_directory_stable(path: Path, fd: int, opened: os.stat_result) -> None:
    try:
        current_fd = os.fstat(fd)
        current_path = path.lstat()
    except OSError as error:
        raise LatencyJournalError("journal directory disappeared") from error
    if _directory_identity(current_fd) != _directory_identity(opened) or _directory_identity(
        current_path
    ) != _directory_identity(opened):
        raise LatencyJournalError("journal directory identity changed")


def _check_directory_unchanged(path: Path, fd: int, expected: os.stat_result) -> None:
    try:
        current_fd = os.fstat(fd)
        current_path = path.lstat()
    except OSError as error:
        raise LatencyJournalError("journal directory disappeared during read") from error
    if _stat_identity(current_fd) != _stat_identity(expected) or _stat_identity(
        current_path
    ) != _stat_identity(expected):
        raise LatencyJournalError("journal directory changed during read")


def _write_exclusive_at(directory_fd: int, name: str, raw: bytes, limit: int) -> None:
    if type(raw) is not bytes or not raw or len(raw) > limit:
        raise LatencyJournalError("journal file exceeds its fixed byte bound")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    fd: int | None = None
    try:
        fd = os.open(name, flags, 0o600, dir_fd=directory_fd)
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise LatencyJournalError("new journal file is not one regular link")
        view = memoryview(raw)
        written = 0
        while written < len(view):
            count = os.write(fd, view[written:])
            if count <= 0:
                raise OSError(errno.ENOSPC, "journal write made no progress")
            written += count
        os.fsync(fd)
    finally:
        if fd is not None:
            os.close(fd)


def _read_regular_at(directory_fd: int, name: str, limit: int) -> bytes:
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        fd = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=directory_fd,
        )
    except OSError as error:
        raise LatencyJournalError("journal leaf could not be opened safely") from error
    try:
        opened = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or before.st_nlink != 1
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) & 0o077
            or _stat_identity(before) != _stat_identity(opened)
            or opened.st_size > limit
        ):
            raise LatencyJournalError("journal leaf identity, type, or size is invalid")
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(fd, min(remaining, 8192))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after_fd = os.fstat(fd)
        after_path = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            len(raw) > limit
            or len(raw) != opened.st_size
            or _stat_identity(after_fd) != _stat_identity(opened)
            or _stat_identity(after_path) != _stat_identity(opened)
        ):
            raise LatencyJournalError("journal leaf changed during its bounded read")
        return raw
    finally:
        os.close(fd)


def _bounded_directory_names(directory_fd: int) -> tuple[str, ...]:
    names: list[str] = []
    try:
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                if len(names) >= 2 * MAX_EVENTS:
                    raise LatencyJournalError(
                        "journal directory exceeds its fixed event bound"
                    )
                names.append(entry.name)
    except LatencyJournalError:
        raise
    except OSError as error:
        raise LatencyJournalError("journal directory could not be enumerated") from error
    return tuple(names)


class LatencyProgressJournalWriter:
    """Typed, single-process writer for one freshly created journal directory."""

    __slots__ = (
        "_bindings",
        "_clock_ns",
        "_closed",
        "_directory_fd",
        "_directory_stat",
        "_failed",
        "_next_sequence",
        "_path",
        "_previous_sha256",
        "_session_id",
        "_state",
    )

    def __init__(
        self,
        path: Path,
        directory_fd: int,
        directory_stat: os.stat_result,
        session_id: str,
        bindings: LatencyJournalBindings,
        clock_ns: Callable[[], int],
    ) -> None:
        self._path = path
        self._directory_fd = directory_fd
        self._directory_stat = directory_stat
        self._session_id = session_id
        self._bindings = bindings
        self._clock_ns = clock_ns
        self._next_sequence = 0
        self._previous_sha256 = ZERO_SHA256
        self._state = _JournalState()
        self._failed = False
        self._closed = False

    @property
    def path(self) -> Path:
        return self._path

    def _append(self, event_type: str, payload: dict[str, object]) -> LatencyJournalRecord:
        if self._closed or self._failed:
            raise LatencyJournalWriteError("latency journal writer is closed or failed")
        if self._next_sequence >= MAX_EVENTS:
            self._failed = True
            raise LatencyJournalWriteError("latency journal event bound is exhausted")
        recorded = _positive_int(self._clock_ns(), "recorded_unix_ns")
        if event_type == "SESSION_START":
            bindings_payload = payload.get("bindings")
            if bindings_payload != self._bindings.payload():
                raise LatencyJournalError("SESSION_START bindings differ from writer binding")
        next_state = self._state.transition(event_type, payload)
        event = _event_payload(
            sequence=self._next_sequence,
            session_id=self._session_id,
            event_type=event_type,
            previous_event_sha256=self._previous_sha256,
            binding_sha256=self._bindings.sha256,
            recorded_unix_ns=recorded,
            payload=payload,
        )
        raw = _canonical_json_bytes(event)
        if len(raw) > MAX_EVENT_BYTES:
            raise LatencyJournalError("journal event exceeds 64 KiB")
        event_sha256 = _sha256(raw)
        marker_raw = _completion_marker_raw(self._next_sequence, event_sha256)
        event_name, marker_name = _event_names(self._next_sequence)
        try:
            _check_directory_stable(self._path, self._directory_fd, self._directory_stat)
            _write_exclusive_at(
                self._directory_fd,
                event_name,
                raw,
                MAX_EVENT_BYTES,
            )
            _write_exclusive_at(
                self._directory_fd,
                marker_name,
                marker_raw,
                MAX_MARKER_BYTES,
            )
            os.fsync(self._directory_fd)
            _check_directory_stable(self._path, self._directory_fd, self._directory_stat)
        except (OSError, LatencyJournalError) as error:
            self._failed = True
            raise LatencyJournalWriteError(
                f"latency journal failed while writing sequence {self._next_sequence}"
            ) from error
        record = LatencyJournalRecord(
            sequence=self._next_sequence,
            event_type=event_type,
            previous_event_sha256=self._previous_sha256,
            binding_sha256=self._bindings.sha256,
            raw=raw,
            marker_raw=marker_raw,
        )
        self._state = next_state
        self._previous_sha256 = event_sha256
        self._next_sequence += 1
        return record

    def session_start(
        self,
        *,
        admission_observation_sha256: str,
        started_unix_ns: int,
    ) -> LatencyJournalRecord:
        return self._append(
            "SESSION_START",
            {
                "admission_observation_sha256": _lower_sha256(
                    admission_observation_sha256,
                    "admission_observation_sha256",
                ),
                "bindings": self._bindings.payload(),
                "started_unix_ns": _positive_int(started_unix_ns, "started_unix_ns"),
            },
        )

    def context_ready(
        self,
        *,
        runtime_sha256: str,
        post_context_observation_sha256: str,
    ) -> LatencyJournalRecord:
        return self._append(
            "CONTEXT_READY",
            {
                "post_context_observation_sha256": _lower_sha256(
                    post_context_observation_sha256,
                    "post_context_observation_sha256",
                ),
                "runtime_sha256": _lower_sha256(runtime_sha256, "runtime_sha256"),
            },
        )

    def visit_started(
        self,
        *,
        round_index: int,
        ordinal_in_round: int,
        run_id: str,
        before_observation_sha256: str,
    ) -> LatencyJournalRecord:
        return self._append(
            "VISIT_START",
            {
                "before_observation_sha256": _lower_sha256(
                    before_observation_sha256,
                    "before_observation_sha256",
                ),
                "ordinal_in_round": ordinal_in_round,
                "round_index": round_index,
                "run_id": run_id,
            },
        )

    def warmup_completed(
        self,
        *,
        round_index: int,
        ordinal_in_round: int,
        run_id: str,
        warmup_index: int,
    ) -> LatencyJournalRecord:
        return self._append(
            "WARMUP_COMPLETED",
            {
                "ordinal_in_round": ordinal_in_round,
                "round_index": round_index,
                "run_id": run_id,
                "warmup_index": warmup_index,
            },
        )

    def timed_sample_completed(
        self,
        *,
        round_index: int,
        ordinal_in_round: int,
        run_id: str,
        sample_index: int,
        sample_ns: int,
    ) -> LatencyJournalRecord:
        """Persist one sample only after the latency function returned its stop delta."""

        return self._append(
            "TIMED_SAMPLE_COMPLETED",
            {
                "ordinal_in_round": ordinal_in_round,
                "round_index": round_index,
                "run_id": run_id,
                "sample_index": sample_index,
                "sample_ns": sample_ns,
            },
        )

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
    ) -> LatencyJournalRecord:
        return self._append(
            "VISIT_END",
            {
                "after_observation_sha256": after_observation_sha256,
                "failure_code": failure_code,
                "ordinal_in_round": ordinal_in_round,
                "outcome": outcome,
                "peak_allocated_bytes": peak_allocated_bytes,
                "peak_reserved_bytes": peak_reserved_bytes,
                "round_index": round_index,
                "run_id": run_id,
            },
        )

    def terminal(
        self,
        *,
        outcome: str,
        latency_session_sha256: str,
        failure_code: str | None,
    ) -> LatencyJournalRecord:
        return self._append(
            "TERMINAL",
            {
                "failure_code": failure_code,
                "latency_session_sha256": _lower_sha256(
                    latency_session_sha256,
                    "latency_session_sha256",
                ),
                "outcome": outcome,
            },
        )

    def close(self) -> None:
        if not self._closed:
            os.close(self._directory_fd)
            self._closed = True

    def __enter__(self) -> LatencyProgressJournalWriter:
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


def create_latency_progress_journal(
    parent: Path,
    *,
    session_id: str,
    bindings: LatencyJournalBindings,
    _clock_ns: Callable[[], int] = time.time_ns,
) -> LatencyProgressJournalWriter:
    """Create one new private session directory; existing paths are rejected."""

    _require_posix()
    if not isinstance(parent, Path):
        raise TypeError("parent must be pathlib.Path")
    checked_session_id = _safe_session_id(session_id)
    if type(bindings) is not LatencyJournalBindings:
        raise TypeError("bindings must be exact LatencyJournalBindings")
    if not callable(_clock_ns):
        raise TypeError("_clock_ns must be callable")
    parent_fd, parent_stat = _open_directory(parent, private=False)
    try:
        try:
            os.mkdir(checked_session_id, mode=0o700, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except OSError as error:
            raise LatencyJournalWriteError(
                "latency journal session directory already exists or cannot be created"
            ) from error
        _check_directory_stable(parent, parent_fd, parent_stat)
    finally:
        os.close(parent_fd)
    path = parent / checked_session_id
    directory_fd, directory_stat = _open_directory(path, private=True)
    return LatencyProgressJournalWriter(
        path,
        directory_fd,
        directory_stat,
        checked_session_id,
        bindings,
        _clock_ns,
    )


def _parse_event(
    raw: bytes,
    *,
    expected_sequence: int,
    expected_session_id: str,
    expected_previous_sha256: str,
    expected_bindings: LatencyJournalBindings,
) -> tuple[str, dict[str, object]]:
    value = _parse_canonical_object(raw, "journal event")
    _exact_keys(
        value,
        {
            "binding_sha256",
            "event_type",
            "payload",
            "previous_event_sha256",
            "recorded_unix_ns",
            "schema",
            "sequence",
            "session_id",
        },
        "journal event",
    )
    if (
        value["schema"] != EVENT_SCHEMA
        or value["sequence"] != expected_sequence
        or value["session_id"] != expected_session_id
        or value["previous_event_sha256"] != expected_previous_sha256
        or value["binding_sha256"] != expected_bindings.sha256
        or type(value["event_type"]) is not str
        or value["event_type"] not in _EVENT_TYPES
        or type(value["payload"]) is not dict
    ):
        raise LatencyJournalError("journal event header or chain is invalid")
    _positive_int(value["recorded_unix_ns"], "recorded_unix_ns")
    return value["event_type"], value["payload"]


def _read_result(
    *,
    session_id: str,
    bindings: LatencyJournalBindings,
    records: list[LatencyJournalRecord],
    tail: tuple[tuple[str, bytes], ...],
    terminal_outcome: str | None,
) -> LatencyJournalReadResult:
    status = {
        None: INCOMPLETE_STATUS,
        "COMPLETED": COMPLETED_STATUS,
        "HELD": HELD_STATUS,
    }[terminal_outcome]
    return LatencyJournalReadResult(
        session_id=session_id,
        bindings=bindings,
        records=tuple(records),
        committed_event_raw=tuple(record.raw for record in records),
        committed_marker_raw=tuple(record.marker_raw for record in records),
        uncommitted_tail_files=tail,
        status=status,
        terminal_outcome=terminal_outcome,
    )


def read_latency_progress_journal(
    path: Path,
    *,
    expected_bindings: LatencyJournalBindings,
) -> LatencyJournalReadResult:
    """Read one stable committed prefix; never upgrade it to formal latency."""

    _require_posix()
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if type(expected_bindings) is not LatencyJournalBindings:
        raise TypeError("expected_bindings must be exact LatencyJournalBindings")
    session_id = _safe_session_id(path.name)
    directory_fd, opened = _open_directory(path, private=True)
    try:
        try:
            before_list = os.fstat(directory_fd)
            names = _bounded_directory_names(directory_fd)
            after_list = os.fstat(directory_fd)
        except OSError as error:
            raise LatencyJournalError("journal directory could not be enumerated") from error
        if _stat_identity(before_list) != _stat_identity(after_list):
            raise LatencyJournalError("journal directory changed during enumeration")
        event_names: dict[int, str] = {}
        marker_names: dict[int, str] = {}
        for name in names:
            if type(name) is not str:
                raise LatencyJournalError("journal leaf name is not text")
            event_match = _EVENT_NAME.fullmatch(name)
            marker_match = _MARKER_NAME.fullmatch(name)
            if event_match is None and marker_match is None:
                raise LatencyJournalError("journal directory contains an unexpected leaf")
            sequence = int((event_match or marker_match).group(1))
            if sequence >= MAX_EVENTS:
                raise LatencyJournalError("journal leaf sequence exceeds the fixed bound")
            target = event_names if event_match is not None else marker_names
            if sequence in target:
                raise LatencyJournalError("journal directory repeats a sequence leaf")
            target[sequence] = name
        sequences = sorted(set(event_names) | set(marker_names))
        if sequences and sequences != list(range(sequences[-1] + 1)):
            raise LatencyJournalError("journal sequence has a gap")
        state = _JournalState()
        records: list[LatencyJournalRecord] = []
        previous_sha256 = ZERO_SHA256
        highest = sequences[-1] if sequences else None
        for sequence in sequences:
            if sequence not in event_names:
                raise LatencyJournalError("completion marker has no event file")
            event_name = event_names[sequence]
            event_raw = _read_regular_at(
                directory_fd,
                event_name,
                MAX_EVENT_BYTES,
            )
            if sequence not in marker_names:
                if sequence != highest or state.terminal_outcome is not None:
                    raise LatencyJournalError("uncommitted event is not the sole final tail")
                _check_directory_unchanged(path, directory_fd, after_list)
                return _read_result(
                    session_id=session_id,
                    bindings=expected_bindings,
                    records=records,
                    tail=((event_name, event_raw),),
                    terminal_outcome=None,
                )
            marker_name = marker_names[sequence]
            marker_raw = _read_regular_at(
                directory_fd,
                marker_name,
                MAX_MARKER_BYTES,
            )
            expected_marker_raw = _completion_marker_raw(
                sequence,
                _sha256(event_raw),
            )
            if marker_raw != expected_marker_raw:
                is_truncated = len(marker_raw) < len(
                    expected_marker_raw
                ) and expected_marker_raw.startswith(marker_raw)
                if not is_truncated:
                    raise LatencyJournalError("completion marker differs from event bytes")
                if sequence != highest or state.terminal_outcome is not None:
                    raise LatencyJournalError(
                        "truncated completion marker is not the sole final tail"
                    )
                _check_directory_unchanged(path, directory_fd, after_list)
                return _read_result(
                    session_id=session_id,
                    bindings=expected_bindings,
                    records=records,
                    tail=((event_name, event_raw), (marker_name, marker_raw)),
                    terminal_outcome=None,
                )
            event_type, payload = _parse_event(
                event_raw,
                expected_sequence=sequence,
                expected_session_id=session_id,
                expected_previous_sha256=previous_sha256,
                expected_bindings=expected_bindings,
            )
            if sequence == 0 and payload.get("bindings") != expected_bindings.payload():
                raise LatencyJournalError("SESSION_START does not contain expected bindings")
            state = state.transition(event_type, payload)
            records.append(
                LatencyJournalRecord(
                    sequence=sequence,
                    event_type=event_type,
                    previous_event_sha256=previous_sha256,
                    binding_sha256=expected_bindings.sha256,
                    raw=event_raw,
                    marker_raw=marker_raw,
                )
            )
            previous_sha256 = _sha256(event_raw)
        _check_directory_stable(path, directory_fd, opened)
        _check_directory_unchanged(path, directory_fd, after_list)
        return _read_result(
            session_id=session_id,
            bindings=expected_bindings,
            records=records,
            tail=(),
            terminal_outcome=state.terminal_outcome,
        )
    finally:
        os.close(directory_fd)


__all__ = [
    "AUTHORITY",
    "COMPLETED_STATUS",
    "HELD_STATUS",
    "INCOMPLETE_STATUS",
    "LatencyJournalBindings",
    "LatencyJournalError",
    "LatencyJournalReadResult",
    "LatencyJournalRecord",
    "LatencyJournalWriteError",
    "LatencyProgressJournalWriter",
    "MAX_EVENT_BYTES",
    "MAX_EVENTS",
    "PRODUCTION",
    "RESULT_CLAIMED",
    "create_latency_progress_journal",
    "read_latency_progress_journal",
]
