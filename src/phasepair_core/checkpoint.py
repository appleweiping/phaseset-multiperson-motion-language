"""Canonical, atomic CPU-synthetic checkpoint envelope.

The envelope is deliberately authority-zero.  It supplies the byte and I/O
foundation needed by the trainer: exact dependency bindings, deterministic
section framing, stable-read validation, no overwrite, and atomic replacement.
The live trainer remains responsible for deriving the section bytes from its
own parameter/optimizer state; arbitrary caller bytes are never promoted to a
production checkpoint by this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import threading
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from weakref import WeakKeyDictionary

from phasepair_core._identity_registry import make_identity_weak_registry


STATUS: Final = "CPU_SYNTHETIC_CHECKPOINT_ENVELOPE_VALIDATED_AUTHORITY0_NO_RESULT"
SCHEMA: Final = "phasepair-cpu-synthetic-checkpoint-envelope-v1"
MAGIC: Final = b"PHASEPAIR-CPU-SYNTHETIC-CHECKPOINT-V1\x00"
REQUIRED_SECTIONS: Final = (
    "dropout_state",
    "model_state",
    "optimizer_state",
    "rng_state",
    "sampler_state",
    "validation_state",
)
_MAX_FILE_BYTES: Final = 64 * 1024 * 1024 * 1024
_PATH_TYPE: Final = type(Path.cwd())


class CheckpointContractError(ValueError):
    """Raised when checkpoint bytes or bindings violate the closed schema."""


class CheckpointPathError(CheckpointContractError):
    """Raised when the target/source path is not a safe ordinary file path."""


def _exact_uint(value: object, label: str, *, bits: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact built-in int")
    if value < 0 or value >= 1 << bits:
        raise CheckpointContractError(f"{label} is outside uint{bits}")
    return value


def _ascii(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{label} must be a nonempty exact built-in str")
    try:
        raw = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise CheckpointContractError(f"{label} must be ASCII") from exc
    if raw.decode("ascii") != value or any(byte < 0x21 or byte > 0x7E for byte in raw):
        raise CheckpointContractError(f"{label} contains forbidden characters")
    return value


def _raw32(value: object, label: str) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        raise TypeError(f"{label} must be exact bytes32")
    return value


def _optional_raw32(value: object, label: str) -> bytes | None:
    if value is None:
        return None
    return _raw32(value, label)


@dataclass(frozen=True, slots=True)
class CheckpointBindings:
    run_id: str
    attempt_id: str
    system_id: str
    seed: int
    epoch_index: int
    global_step: int
    source_manifest_sha256: bytes
    data_manifest_sha256: bytes
    split_manifest_sha256: bytes
    environment_manifest_sha256: bytes
    code_manifest_sha256: bytes
    model_manifest_sha256: bytes
    text_runtime_manifest_sha256: bytes
    optimizer_manifest_sha256: bytes
    sampler_state_sha256: bytes
    dropout_state_sha256: bytes
    validation_state_sha256: bytes
    parent_checkpoint_sha256: bytes | None = None

    def __post_init__(self) -> None:
        _validate_bindings(self)


_BINDING_FIELDS = (
    "run_id",
    "attempt_id",
    "system_id",
    "seed",
    "epoch_index",
    "global_step",
    "source_manifest_sha256",
    "data_manifest_sha256",
    "split_manifest_sha256",
    "environment_manifest_sha256",
    "code_manifest_sha256",
    "model_manifest_sha256",
    "text_runtime_manifest_sha256",
    "optimizer_manifest_sha256",
    "sampler_state_sha256",
    "dropout_state_sha256",
    "validation_state_sha256",
    "parent_checkpoint_sha256",
)
_BINDING_READERS = tuple(
    CheckpointBindings.__dict__[name].__get__ for name in _BINDING_FIELDS
)


def _bindings_key(value: CheckpointBindings) -> tuple[object, ...]:
    if type(value) is not CheckpointBindings:
        raise TypeError("bindings must be exactly CheckpointBindings")
    result = tuple(
        reader(value, CheckpointBindings) for reader in _BINDING_READERS
    )
    (
        run_id,
        attempt_id,
        system_id,
        seed,
        epoch_index,
        global_step,
        *digests,
        parent_checkpoint_sha256,
    ) = result
    _ascii(run_id, "run_id")
    _ascii(attempt_id, "attempt_id")
    _ascii(system_id, "system_id")
    _exact_uint(seed, "seed", bits=64)
    _exact_uint(epoch_index, "epoch_index", bits=32)
    if epoch_index >= 30:
        raise CheckpointContractError("epoch_index must be in [0,29]")
    _exact_uint(global_step, "global_step", bits=32)
    for label, digest in zip(_BINDING_FIELDS[6:-1], digests, strict=True):
        _raw32(digest, label)
    _optional_raw32(parent_checkpoint_sha256, "parent_checkpoint_sha256")
    return result


def _validate_bindings(value: CheckpointBindings) -> CheckpointBindings:
    _bindings_key(value)
    return value


def _bindings_object(value: CheckpointBindings) -> dict[str, object]:
    fields = dict(zip(_BINDING_FIELDS, _bindings_key(value), strict=True))
    return {
        "attempt_id": fields["attempt_id"],
        "code_manifest_sha256": fields["code_manifest_sha256"].hex(),
        "data_manifest_sha256": fields["data_manifest_sha256"].hex(),
        "dropout_state_sha256": fields["dropout_state_sha256"].hex(),
        "environment_manifest_sha256": fields["environment_manifest_sha256"].hex(),
        "epoch_index": fields["epoch_index"],
        "global_step": fields["global_step"],
        "model_manifest_sha256": fields["model_manifest_sha256"].hex(),
        "optimizer_manifest_sha256": fields["optimizer_manifest_sha256"].hex(),
        "parent_checkpoint_sha256": (
            None
            if fields["parent_checkpoint_sha256"] is None
            else fields["parent_checkpoint_sha256"].hex()
        ),
        "run_id": fields["run_id"],
        "sampler_state_sha256": fields["sampler_state_sha256"].hex(),
        "seed": fields["seed"],
        "source_manifest_sha256": fields["source_manifest_sha256"].hex(),
        "split_manifest_sha256": fields["split_manifest_sha256"].hex(),
        "system_id": fields["system_id"],
        "text_runtime_manifest_sha256": fields["text_runtime_manifest_sha256"].hex(),
        "validation_state_sha256": fields["validation_state_sha256"].hex(),
    }


def _bindings_from_object(value: object) -> CheckpointBindings:
    expected_keys = {
        "attempt_id",
        "code_manifest_sha256",
        "data_manifest_sha256",
        "dropout_state_sha256",
        "environment_manifest_sha256",
        "epoch_index",
        "global_step",
        "model_manifest_sha256",
        "optimizer_manifest_sha256",
        "parent_checkpoint_sha256",
        "run_id",
        "sampler_state_sha256",
        "seed",
        "source_manifest_sha256",
        "split_manifest_sha256",
        "system_id",
        "text_runtime_manifest_sha256",
        "validation_state_sha256",
    }
    if type(value) is not dict or set(value) != expected_keys:
        raise CheckpointContractError("checkpoint binding key census mismatch")

    def digest(name: str, *, optional: bool = False) -> bytes | None:
        item = value[name]
        if optional and item is None:
            return None
        if type(item) is not str or len(item) != 64:
            raise CheckpointContractError(f"{name} must be lowercase hex sha256")
        try:
            raw = bytes.fromhex(item)
        except ValueError as exc:
            raise CheckpointContractError(f"{name} is not hex") from exc
        if raw.hex() != item:
            raise CheckpointContractError(f"{name} is not canonical lowercase hex")
        return _raw32(raw, name)

    return CheckpointBindings(
        run_id=value["run_id"],
        attempt_id=value["attempt_id"],
        system_id=value["system_id"],
        seed=value["seed"],
        epoch_index=value["epoch_index"],
        global_step=value["global_step"],
        source_manifest_sha256=digest("source_manifest_sha256"),
        data_manifest_sha256=digest("data_manifest_sha256"),
        split_manifest_sha256=digest("split_manifest_sha256"),
        environment_manifest_sha256=digest("environment_manifest_sha256"),
        code_manifest_sha256=digest("code_manifest_sha256"),
        model_manifest_sha256=digest("model_manifest_sha256"),
        text_runtime_manifest_sha256=digest("text_runtime_manifest_sha256"),
        optimizer_manifest_sha256=digest("optimizer_manifest_sha256"),
        sampler_state_sha256=digest("sampler_state_sha256"),
        dropout_state_sha256=digest("dropout_state_sha256"),
        validation_state_sha256=digest("validation_state_sha256"),
        parent_checkpoint_sha256=digest("parent_checkpoint_sha256", optional=True),
    )


def _sections(value: object) -> tuple[tuple[str, bytes], ...]:
    if type(value) is not tuple:
        raise TypeError("sections must be an exact built-in tuple")
    rows: list[tuple[str, bytes]] = []
    for index, row in enumerate(value):
        if type(row) is not tuple or len(row) != 2:
            raise TypeError(f"sections[{index}] must be an exact pair tuple")
        name, raw = row
        checked_name = _ascii(name, f"sections[{index}].name")
        if type(raw) is not bytes:
            raise TypeError(f"sections[{index}].raw must be exact bytes")
        rows.append((checked_name, raw))
    result = tuple(rows)
    names = tuple(name for name, _ in result)
    required = (
        "dropout_state",
        "model_state",
        "optimizer_state",
        "rng_state",
        "sampler_state",
        "validation_state",
    )
    if names != required:
        raise CheckpointContractError(
            f"section census/order must be exactly {required!r}"
        )
    return result


class ValidatedCheckpoint:
    """Opaque validated checkpoint envelope."""

    __slots__ = ("__weakref__",)

    def __init__(self, *_: object, **__: object) -> None:
        raise CheckpointContractError("validated checkpoints are minted internally")


@dataclass(frozen=True, slots=True)
class _CheckpointRecord:
    raw: bytes
    bindings: CheckpointBindings
    sections: tuple[tuple[str, bytes], ...]


_ISSUED_LOCK = threading.RLock()
_ISSUED_SET, _ISSUED_GET, _, _, _ = make_identity_weak_registry(_ISSUED_LOCK)
_ISSUED: WeakKeyDictionary[ValidatedCheckpoint, _CheckpointRecord] = WeakKeyDictionary()


def _manifest_bytes(
    bindings: CheckpointBindings,
    sections: tuple[tuple[str, bytes], ...],
) -> bytes:
    offset = 0
    section_rows: list[dict[str, object]] = []
    for name, raw in sections:
        section_rows.append(
            {
                "bytes": len(raw),
                "name": name,
                "offset": offset,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        offset += len(raw)
    payload = {
        "authority": 0,
        "bindings": _bindings_object(bindings),
        "production": False,
        "result_claimed": False,
        "schema": "phasepair-cpu-synthetic-checkpoint-envelope-v1",
        "sections": section_rows,
        "status": "CPU_SYNTHETIC_CHECKPOINT_ENVELOPE_VALIDATED_AUTHORITY0_NO_RESULT",
        "total_section_bytes": offset,
    }
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def build_cpu_synthetic_checkpoint(
    bindings: CheckpointBindings,
    sections: tuple[tuple[str, bytes], ...],
) -> ValidatedCheckpoint:
    checked_bindings = _validate_bindings(bindings)
    checked_sections = _sections(sections)
    _crossbind_state_sections(checked_bindings, checked_sections)
    manifest = _manifest_bytes(checked_bindings, checked_sections)
    payload = b"".join(raw for _, raw in checked_sections)
    raw = (
        b"PHASEPAIR-CPU-SYNTHETIC-CHECKPOINT-V1\x00"
        + len(manifest).to_bytes(8, "big")
        + manifest
        + payload
    )
    return _mint_validated(raw, checked_bindings, checked_sections)


def _mint_validated(
    raw: bytes,
    bindings: CheckpointBindings,
    sections: tuple[tuple[str, bytes], ...],
) -> ValidatedCheckpoint:
    value = object.__new__(ValidatedCheckpoint)
    frozen_bindings = _bindings_from_object(_bindings_object(bindings))
    frozen_sections = tuple((str(name), bytes(section_raw)) for name, section_raw in sections)
    _ISSUED_SET(
        value,
        _CheckpointRecord(bytes(raw), frozen_bindings, frozen_sections),
    )
    return value


def _crossbind_state_sections(
    bindings: CheckpointBindings,
    sections: tuple[tuple[str, bytes], ...],
) -> None:
    by_name = dict(sections)
    expected = {
        "dropout_state": bindings.dropout_state_sha256,
        "sampler_state": bindings.sampler_state_sha256,
        "validation_state": bindings.validation_state_sha256,
    }
    for name, digest in expected.items():
        if hashlib.sha256(by_name[name]).digest() != digest:
            raise CheckpointContractError(f"{name} does not match its binding digest")


def _parse(raw: bytes) -> tuple[CheckpointBindings, tuple[tuple[str, bytes], ...]]:
    if type(raw) is not bytes:
        raise TypeError("checkpoint raw must be exact bytes")
    magic = b"PHASEPAIR-CPU-SYNTHETIC-CHECKPOINT-V1\x00"
    required = (
        "dropout_state",
        "model_state",
        "optimizer_state",
        "rng_state",
        "sampler_state",
        "validation_state",
    )
    if len(raw) > 64 * 1024 * 1024 * 1024:
        raise CheckpointContractError("checkpoint exceeds the fixed maximum")
    prefix = len(magic) + 8
    if len(raw) < prefix or not raw.startswith(magic):
        raise CheckpointContractError("checkpoint magic/truncation mismatch")
    manifest_length = int.from_bytes(raw[len(magic) : prefix], "big")
    if manifest_length < 2 or prefix + manifest_length > len(raw):
        raise CheckpointContractError("checkpoint manifest length mismatch")
    manifest = raw[prefix : prefix + manifest_length]
    if not manifest.endswith(b"\n") or b"\r" in manifest:
        raise CheckpointContractError("checkpoint manifest must be canonical LF JSON")
    try:
        text = manifest.decode("ascii")
        parsed = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointContractError("checkpoint manifest parse failure") from exc
    if type(parsed) is not dict or set(parsed) != {
        "authority",
        "bindings",
        "production",
        "result_claimed",
        "schema",
        "sections",
        "status",
        "total_section_bytes",
    }:
        raise CheckpointContractError("checkpoint manifest key census mismatch")
    if (
        parsed["authority"] != 0
        or type(parsed["authority"]) is not int
        or parsed["production"] is not False
        or parsed["result_claimed"] is not False
        or parsed["schema"] != "phasepair-cpu-synthetic-checkpoint-envelope-v1"
        or parsed["status"]
        != "CPU_SYNTHETIC_CHECKPOINT_ENVELOPE_VALIDATED_AUTHORITY0_NO_RESULT"
    ):
        raise CheckpointContractError("checkpoint authority/status invariant mismatch")
    bindings = _bindings_from_object(parsed["bindings"])
    section_rows = parsed["sections"]
    if type(section_rows) is not list or len(section_rows) != len(required):
        raise CheckpointContractError("checkpoint section manifest census mismatch")
    payload = raw[prefix + manifest_length :]
    sections: list[tuple[str, bytes]] = []
    expected_offset = 0
    for index, row in enumerate(section_rows):
        if type(row) is not dict or set(row) != {"bytes", "name", "offset", "sha256"}:
            raise CheckpointContractError(f"section manifest row {index} key mismatch")
        name = _ascii(row["name"], f"sections[{index}].name")
        if name != required[index]:
            raise CheckpointContractError("section manifest order/name mismatch")
        size = _exact_uint(row["bytes"], f"sections[{index}].bytes", bits=64)
        offset = _exact_uint(row["offset"], f"sections[{index}].offset", bits=64)
        if offset != expected_offset or offset + size > len(payload):
            raise CheckpointContractError("section offset/length mismatch")
        digest = row["sha256"]
        if type(digest) is not str or len(digest) != 64:
            raise CheckpointContractError("section sha256 shape mismatch")
        section_raw = payload[offset : offset + size]
        if hashlib.sha256(section_raw).hexdigest() != digest:
            raise CheckpointContractError("section sha256 mismatch")
        sections.append((name, section_raw))
        expected_offset += size
    if expected_offset != len(payload) or parsed["total_section_bytes"] != len(payload):
        raise CheckpointContractError("checkpoint payload has truncation or trailing bytes")
    checked_sections = _sections(tuple(sections))
    _crossbind_state_sections(bindings, checked_sections)
    if _manifest_bytes(bindings, checked_sections) != manifest:
        raise CheckpointContractError("checkpoint manifest is not exact canonical JSON")
    return bindings, checked_sections


def validate_cpu_synthetic_checkpoint_bytes(
    raw: bytes,
    *,
    expected_bindings: CheckpointBindings,
) -> ValidatedCheckpoint:
    bindings, sections = _parse(raw)
    expected = _validate_bindings(expected_bindings)
    if _bindings_key(bindings) != _bindings_key(expected):
        raise CheckpointContractError("checkpoint dependency binding mismatch")
    return _mint_validated(raw, bindings, sections)


def canonical_checkpoint_bytes(value: ValidatedCheckpoint) -> bytes:
    if type(value) is not ValidatedCheckpoint:
        raise TypeError("value must be exactly ValidatedCheckpoint")
    record = _ISSUED_GET(value)
    if record is None:
        raise CheckpointContractError("checkpoint was not issued by this module")
    bindings, sections = _parse(record.raw)
    if (
        _bindings_key(bindings) != _bindings_key(record.bindings)
        or sections != record.sections
    ):
        raise CheckpointContractError("issued checkpoint record drift")
    return record.raw


def checkpoint_sha256(value: ValidatedCheckpoint) -> bytes:
    return hashlib.sha256(canonical_checkpoint_bytes(value)).digest()


def checkpoint_sections(value: ValidatedCheckpoint) -> tuple[tuple[str, bytes], ...]:
    if type(value) is not ValidatedCheckpoint:
        raise TypeError("value must be exactly ValidatedCheckpoint")
    record = _ISSUED_GET(value)
    if record is None:
        raise CheckpointContractError("checkpoint was not issued by this module")
    bindings, sections = _parse(record.raw)
    if (
        _bindings_key(bindings) != _bindings_key(record.bindings)
        or sections != record.sections
    ):
        raise CheckpointContractError("issued checkpoint record drift")
    return tuple((name, bytes(raw)) for name, raw in sections)


def checkpoint_bindings(value: ValidatedCheckpoint) -> CheckpointBindings:
    if type(value) is not ValidatedCheckpoint:
        raise TypeError("value must be exactly ValidatedCheckpoint")
    record = _ISSUED_GET(value)
    if record is None:
        raise CheckpointContractError("checkpoint was not issued by this module")
    bindings, sections = _parse(record.raw)
    if (
        _bindings_key(bindings) != _bindings_key(record.bindings)
        or sections != record.sections
    ):
        raise CheckpointContractError("issued checkpoint record drift")
    payload = _bindings_object(bindings)
    return _bindings_from_object(payload)


def _metadata_is_reparse(metadata: os.stat_result) -> bool:
    attributes = getattr(metadata, "st_file_attributes", 0)
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & flag)


def _path_entry_exists(path: Path, unavailable_message: str) -> bool:
    """Probe a directory entry without version-dependent pathlib swallowing.

    ``Path.exists``/``Path.is_symlink`` changed their ``OSError`` behaviour
    across the supported CPython lanes.  Atomic checkpoint decisions must not
    depend on that difference, and a filesystem observation failure must never
    escape as an unclassified raw ``OSError``.
    """

    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise CheckpointPathError(unavailable_message) from exc
    return True


def _safe_parent(path: Path) -> Path:
    if type(path) is not _PATH_TYPE or not path.is_absolute():
        raise CheckpointPathError("checkpoint path must be an exact absolute Path")
    if path.name in {"", ".", ".."}:
        raise CheckpointPathError("checkpoint filename is invalid")
    parent = path.parent
    try:
        lexical_parent = Path(os.path.abspath(os.fspath(parent)))
        resolved_parent = parent.resolve(strict=True)
        metadata = parent.lstat()
    except OSError as exc:
        raise CheckpointPathError("checkpoint parent is unavailable") from exc
    if lexical_parent != resolved_parent or parent.is_symlink() or _metadata_is_reparse(metadata):
        raise CheckpointPathError("checkpoint parent alias/reparse is forbidden")
    if not stat.S_ISDIR(metadata.st_mode):
        raise CheckpointPathError("checkpoint parent is not an ordinary directory")
    return resolved_parent


def write_checkpoint_atomic(value: ValidatedCheckpoint, path: Path) -> bytes:
    """Write once via same-directory temp + fsync + atomic replace, then re-read."""

    raw = canonical_checkpoint_bytes(value)
    issued_record = _ISSUED_GET(value)
    if issued_record is None:
        raise CheckpointContractError("checkpoint was not issued by this module")
    parent = _safe_parent(path)
    target = parent / path.name
    if _path_entry_exists(target, "checkpoint target availability is unavailable"):
        raise CheckpointPathError("checkpoint target already exists")
    descriptor = -1
    ownership_descriptor = -1
    temp_name: str | None = None
    committed = False
    succeeded = False
    committed_identity: tuple[int, int] | None = None
    retirement_path: Path | None = None

    def remove_committed_target_if_owned() -> None:
        if committed_identity is None or retirement_path is None:
            return
        try:
            os.rename(target, retirement_path)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise CheckpointPathError(
                "checkpoint committed-target cleanup failed"
            ) from exc
        try:
            moved = retirement_path.lstat()
            if (moved.st_dev, moved.st_ino) == committed_identity:
                retirement_path.unlink()
                return
            if _path_entry_exists(
                target,
                "checkpoint cleanup replacement identity is unavailable",
            ):
                raise CheckpointPathError(
                    "checkpoint cleanup found a replacement and cannot restore it"
                )
            os.rename(retirement_path, target)
        except BaseException as exc:
            raise CheckpointPathError(
                "checkpoint committed-target cleanup failed"
            ) from exc

    try:
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=parent,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            written = handle.write(raw)
            if written != len(raw):
                raise CheckpointPathError("checkpoint temp short write")
            handle.flush()
            os.fsync(handle.fileno())
            if os.name == "posix":
                # Keep the original inode alive even if every pathname is
                # concurrently removed. Otherwise POSIX may immediately reuse
                # its (device,inode) for a replacement owned by another writer,
                # and failure cleanup could mistake that replacement for ours.
                ownership_descriptor = os.dup(handle.fileno())
        temp_path = Path(temp_name)
        retirement_path = Path(temp_name + ".committed-retirement")
        if _path_entry_exists(
            retirement_path,
            "checkpoint retirement path availability is unavailable",
        ):
            raise CheckpointPathError("checkpoint retirement path collision")
        metadata = temp_path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or _metadata_is_reparse(metadata)
            or metadata.st_size != len(raw)
            or temp_path.read_bytes() != raw
        ):
            raise CheckpointPathError("checkpoint temp verification failed")
        if _path_entry_exists(
            target,
            "checkpoint target availability is unavailable before commit",
        ):
            raise CheckpointPathError("checkpoint target appeared before commit")
        try:
            os.link(temp_path, target, follow_symlinks=False)
            committed = True
            committed_identity = (
                metadata.st_dev,
                metadata.st_ino,
            )
        except (FileExistsError, OSError) as exc:
            if _path_entry_exists(
                target,
                "checkpoint target identity is unavailable after failed atomic link",
            ):
                raise CheckpointPathError(
                    "checkpoint target appeared before atomic commit"
                ) from exc
            raise CheckpointPathError("checkpoint atomic link commit failed") from exc
        try:
            committed_metadata = target.lstat()
        except OSError as exc:
            raise CheckpointPathError(
                "checkpoint committed target identity is unavailable"
            ) from exc
        if (
            committed_metadata.st_dev != metadata.st_dev
            or committed_metadata.st_ino != metadata.st_ino
            or not stat.S_ISREG(committed_metadata.st_mode)
            or _metadata_is_reparse(committed_metadata)
        ):
            raise CheckpointPathError("checkpoint committed target identity mismatch")
        try:
            temp_path.unlink()
        except OSError as exc:
            remove_committed_target_if_owned()
            raise CheckpointPathError("checkpoint temp retirement failed") from exc
        temp_name = None
        loaded = read_checkpoint(
            target,
            expected_bindings=issued_record.bindings,
        )
        if canonical_checkpoint_bytes(loaded) != raw:
            raise CheckpointPathError("checkpoint post-commit verification failed")
        succeeded = True
        return hashlib.sha256(raw).digest()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temp_name is not None:
            try:
                Path(temp_name).unlink(missing_ok=True)
            except OSError:
                pass
        try:
            if committed and not succeeded:
                remove_committed_target_if_owned()
        finally:
            if ownership_descriptor >= 0:
                os.close(ownership_descriptor)


def _stable_read(path: Path) -> bytes:
    parent = _safe_parent(path)
    target = parent / path.name
    def identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            getattr(value, "st_file_attributes", 0),
        )

    def read_once() -> tuple[
        bytes,
        tuple[tuple[int, int, int, int, int, int, int], ...],
    ]:
        try:
            before = target.lstat()
        except OSError as exc:
            raise CheckpointPathError("checkpoint file is unavailable") from exc
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or target.is_symlink()
            or _metadata_is_reparse(before)
            or before.st_size > 64 * 1024 * 1024 * 1024
        ):
            raise CheckpointPathError("checkpoint source is not a safe ordinary file")
        try:
            with target.open("rb") as handle:
                first_fd = os.fstat(handle.fileno())
                raw = handle.read(before.st_size)
                trailing = handle.read(1)
                second_fd = os.fstat(handle.fileno())
            after = target.lstat()
        except OSError as exc:
            raise CheckpointPathError("checkpoint stable read failed") from exc
        if (
            trailing
            or len(raw) > 64 * 1024 * 1024 * 1024
            or len(raw) != before.st_size
        ):
            raise CheckpointPathError("checkpoint read length drift")
        metadata_rows = (before, first_fd, second_fd, after)
        if any(
            not stat.S_ISREG(row.st_mode)
            or row.st_nlink != 1
            or _metadata_is_reparse(row)
            for row in metadata_rows
        ):
            raise CheckpointPathError("checkpoint metadata changed to an unsafe file")
        if identity(first_fd) != identity(second_fd) or identity(before) != identity(after):
            raise CheckpointPathError("checkpoint identity changed during read")
        if (
            first_fd.st_dev != before.st_dev
            or first_fd.st_ino != before.st_ino
            or first_fd.st_mode != before.st_mode
            or first_fd.st_nlink != before.st_nlink
            or first_fd.st_size != before.st_size
        ):
            raise CheckpointPathError("checkpoint path/fd identity mismatch")
        return raw, tuple(identity(row) for row in metadata_rows)

    first_raw, first_identity = read_once()
    second_raw, second_identity = read_once()
    if first_raw != second_raw or first_identity[-1] != second_identity[0]:
        raise CheckpointPathError("checkpoint changed between stable-read passes")
    return first_raw


def read_checkpoint(
    path: Path,
    *,
    expected_bindings: CheckpointBindings,
) -> ValidatedCheckpoint:
    raw = _stable_read(path)
    return validate_cpu_synthetic_checkpoint_bytes(
        raw,
        expected_bindings=expected_bindings,
    )


def _install_sealed_api() -> None:
    """Clone the checkpoint implementation into a private dependency snapshot."""

    function_names = (
        "_exact_uint",
        "_ascii",
        "_raw32",
        "_optional_raw32",
        "_bindings_key",
        "_validate_bindings",
        "_bindings_object",
        "_bindings_from_object",
        "_sections",
        "_manifest_bytes",
        "build_cpu_synthetic_checkpoint",
        "_mint_validated",
        "_crossbind_state_sections",
        "_parse",
        "validate_cpu_synthetic_checkpoint_bytes",
        "canonical_checkpoint_bytes",
        "checkpoint_sha256",
        "checkpoint_sections",
        "checkpoint_bindings",
        "_metadata_is_reparse",
        "_path_entry_exists",
        "_safe_parent",
        "write_checkpoint_atomic",
        "_stable_read",
        "read_checkpoint",
    )
    originals = {name: globals()[name] for name in function_names}

    issued_lock = threading.RLock()
    issued_set, issued_get, _, _, _ = make_identity_weak_registry(issued_lock)
    hashlib_proxy = types.SimpleNamespace(sha256=hashlib.sha256)
    json_proxy = types.SimpleNamespace(
        JSONDecodeError=json.JSONDecodeError,
        dumps=json.dumps,
        loads=json.loads,
    )
    os_proxy = types.SimpleNamespace(
        close=os.close,
        dup=os.dup,
        fdopen=os.fdopen,
        fspath=os.fspath,
        fstat=os.fstat,
        fsync=os.fsync,
        link=os.link,
        name=os.name,
        rename=os.rename,
        path=types.SimpleNamespace(abspath=os.path.abspath),
    )
    stat_proxy = types.SimpleNamespace(
        FILE_ATTRIBUTE_REPARSE_POINT=getattr(
            stat,
            "FILE_ATTRIBUTE_REPARSE_POINT",
            0x400,
        ),
        S_ISDIR=stat.S_ISDIR,
        S_ISREG=stat.S_ISREG,
    )
    tempfile_proxy = types.SimpleNamespace(mkstemp=tempfile.mkstemp)
    sealed_globals = dict(globals())
    sealed_globals.update(
        {
            "_ISSUED_SET": issued_set,
            "_ISSUED_GET": issued_get,
            "_BINDING_FIELDS": _BINDING_FIELDS,
            "_BINDING_READERS": _BINDING_READERS,
            "hashlib": hashlib_proxy,
            "json": json_proxy,
            "os": os_proxy,
            "stat": stat_proxy,
            "tempfile": tempfile_proxy,
        }
    )

    clones: dict[str, object] = {}
    for name in function_names:
        original = originals[name]
        clone = types.FunctionType(
            original.__code__,
            sealed_globals,
            original.__name__,
            original.__defaults__,
            original.__closure__,
        )
        clone.__kwdefaults__ = original.__kwdefaults__
        clone.__annotations__ = dict(original.__annotations__)
        clone.__dict__.update(original.__dict__)
        clone.__doc__ = original.__doc__
        clone.__module__ = original.__module__
        clone.__qualname__ = original.__qualname__
        clones[name] = clone
    sealed_globals.update(clones)
    globals().update(clones)

    validate_bindings = clones["_validate_bindings"]
    contract_error = CheckpointContractError

    def sealed_bindings_post_init(self: CheckpointBindings) -> None:
        validate_bindings(self)  # type: ignore[operator]

    def sealed_checkpoint_init(self: ValidatedCheckpoint, *_: object, **__: object) -> None:
        raise contract_error("validated checkpoints are minted internally")

    CheckpointBindings.__post_init__ = sealed_bindings_post_init
    ValidatedCheckpoint.__init__ = sealed_checkpoint_init


_install_sealed_api()
del _install_sealed_api


__all__ = [
    "MAGIC",
    "REQUIRED_SECTIONS",
    "SCHEMA",
    "STATUS",
    "CheckpointBindings",
    "CheckpointContractError",
    "CheckpointPathError",
    "ValidatedCheckpoint",
    "build_cpu_synthetic_checkpoint",
    "canonical_checkpoint_bytes",
    "checkpoint_sections",
    "checkpoint_bindings",
    "checkpoint_sha256",
    "read_checkpoint",
    "validate_cpu_synthetic_checkpoint_bytes",
    "write_checkpoint_atomic",
]
