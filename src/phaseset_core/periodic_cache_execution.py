"""Build and consume one immutable seed-specific periodic descriptor cache.

This private module composes the existing prepared-data source, capture source,
descriptor writer/reader, and complete six-system plan admission.  It does not
estimate energy floors, execute a model, authorize data, or claim a result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import json
import os
from pathlib import Path
import stat
from typing import Final
import zipfile

import numpy as np

from .capture_validation import (
    CaptureDescriptorWindowSource,
    CaptureValidationSource,
)
from .execution import (
    PeriodicCacheRecord,
    parse_periodic_cache_bytes,
)
from .experiments import (
    SEEDS,
    BaseQualification,
    canonical_base_qualification_bytes,
)
from .periodic import validate_energy_floors
from .periodic_capture_training_cache import (
    PeriodicDescriptorCaptureTrainingPlan,
    admit_periodic_descriptor_capture_training_plan,
)
from .periodic_descriptor_cache_v2 import (
    DescriptorCacheArtifact,
    DescriptorCacheBounds,
    DescriptorCacheSourceBinding,
    PeriodicDescriptorCacheV2,
    PeriodicDescriptorCacheWriterV2,
    energy_floors_sha256,
)
from .prepared_data_v2 import PreparedTrainingDataSourceV2
from .training import (
    OFFICIAL_SEEDS,
    RESIDUAL_EPOCHS,
    RetrievalTrainingBatch,
    TrainingConfig,
    training_code_artifact_sha256,
)


AUTHORITY: Final = 0
PRODUCTION: Final = False
RESULT_CLAIMED: Final = False
EXTERNAL_AUTHENTICATION_ASSERTED: Final = False
BUILD_SCHEMA: Final = "phaseset-periodic-descriptor-cache-build-v1"
BUILD_STATUS: Final = "COMPLETE_PRIVATE_CACHE_BUILD_AUTHORITY0"
FLOOR_INPUT_SCHEMA: Final = "phaseset-qualified-energy-floor-input-v1"
FLOOR_RECORD_NAME: Final = "record.json"
FLOOR_PAYLOAD_NAME: Final = "cache-content.npz"
BUILD_MANIFEST_NAME: Final = "build.json"
TRAIN_CACHE_NAME: Final = "train"
CAPTURE_CACHE_NAME: Final = "capture-validation"
PLAN_DIRECTORY_NAME: Final = "plans"
MAX_FLOOR_RECORD_BYTES: Final = 64 * 1024
MAX_FLOOR_PAYLOAD_BYTES: Final = 1024 * 1024
MAX_FLOOR_NPY_BYTES: Final = 4096

CACHEABLE_SYSTEM_STREAMS: Final[tuple[tuple[str, str], ...]] = (
    ("02", "MARGINAL_POWER"),
    ("03", "MEAN_DIFFERENCE_DCT"),
    ("04", "FULL_RELATION"),
    ("06", "FULL_RELATION"),
    ("07", "FULL_RELATION"),
    ("08", "FULL_RELATION"),
)

_FLOOR_TOKEN: Final = object()
_RESULT_TOKEN: Final = object()


class PeriodicCacheExecutionError(ValueError):
    """The seed-specific cache build or its consumed bytes are invalid."""


@dataclass(frozen=True, slots=True)
class PeriodicCacheExecutionBounds:
    """Explicit bounds above the two existing descriptor-cache bounds."""

    descriptor: DescriptorCacheBounds = DescriptorCacheBounds()
    max_checkpoint_bytes: int = 8 * 1024 * 1024 * 1024
    max_combined_shards: int = 524_288
    max_combined_shard_bytes: int = 2 * (1 << 40)
    max_build_manifest_bytes: int = 4 * 1024 * 1024
    max_plan_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        if type(self.descriptor) is not DescriptorCacheBounds:
            raise TypeError("descriptor must be exact DescriptorCacheBounds")
        for name in (
            "max_checkpoint_bytes",
            "max_combined_shards",
            "max_combined_shard_bytes",
            "max_build_manifest_bytes",
            "max_plan_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise TypeError(f"{name} must be an exact positive int")


@dataclass(frozen=True, slots=True, repr=False)
class QualifiedEnergyFloorInput:
    """Exact consumed v1 floor bytes bound to a canonical base selection.

    The type proves byte and selection linkage only.  It does not invent the
    still-missing scientific estimator that creates training-final floors.
    """

    record: PeriodicCacheRecord
    energy_floors: np.ndarray = field(repr=False)
    record_json_bytes: bytes = field(repr=False)
    payload_npz_bytes: bytes = field(repr=False)
    record_sha256: str
    payload_sha256: str
    qualification_sha256: str
    energy_floors_sha256: str
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _FLOOR_TOKEN:
            raise PeriodicCacheExecutionError(
                "qualified floor input must come from the same-byte v1 loader"
            )
        if type(self.record) is not PeriodicCacheRecord:
            raise TypeError("record must be exact PeriodicCacheRecord")
        if type(self.record_json_bytes) is not bytes or type(self.payload_npz_bytes) is not bytes:
            raise TypeError("floor artifacts must be exact built-in bytes")
        for name in (
            "record_sha256",
            "payload_sha256",
            "qualification_sha256",
            "energy_floors_sha256",
        ):
            _lower_sha256(getattr(self, name), name)
        if _sha256(self.record_json_bytes) != self.record_sha256:
            raise PeriodicCacheExecutionError("floor record bytes changed")
        if _sha256(self.payload_npz_bytes) != self.payload_sha256:
            raise PeriodicCacheExecutionError("floor payload bytes changed")
        floors = _frozen_floors(self.energy_floors)
        if energy_floors_sha256(floors) != self.energy_floors_sha256:
            raise PeriodicCacheExecutionError("floor vector bytes changed")
        object.__setattr__(self, "energy_floors", floors)

    def __repr__(self) -> str:
        return (
            "QualifiedEnergyFloorInput(seed="
            f"{self.record.seed}, record_sha256={self.record_sha256!r}, "
            "<floor bytes hidden>)"
        )


@dataclass(frozen=True, slots=True, repr=False)
class PeriodicDescriptorCacheBuildResult:
    """Canonical manifest plus live readers and six sealed admitted plans."""

    root: Path = field(repr=False)
    seed: int
    config_sha256: str
    qualification_sha256: str
    manifest_json_bytes: bytes = field(repr=False)
    manifest_sha256: str
    train_cache: PeriodicDescriptorCacheV2 = field(repr=False, compare=False)
    capture_validation_cache: PeriodicDescriptorCacheV2 = field(
        repr=False,
        compare=False,
    )
    plans: tuple[PeriodicDescriptorCaptureTrainingPlan, ...] = field(
        repr=False,
        compare=False,
    )
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _RESULT_TOKEN:
            raise PeriodicCacheExecutionError(
                "cache result must come from build or complete consumption verification"
            )
        if not isinstance(self.root, Path):
            raise TypeError("result root must be a pathlib.Path")
        if type(self.seed) is not int or self.seed not in OFFICIAL_SEEDS:
            raise PeriodicCacheExecutionError("result seed is outside the census")
        for name in ("config_sha256", "qualification_sha256", "manifest_sha256"):
            _lower_sha256(getattr(self, name), name)
        if type(self.manifest_json_bytes) is not bytes:
            raise TypeError("manifest_json_bytes must be exact built-in bytes")
        if _sha256(self.manifest_json_bytes) != self.manifest_sha256:
            raise PeriodicCacheExecutionError("result manifest bytes changed")
        if (
            type(self.train_cache) is not PeriodicDescriptorCacheV2
            or type(self.capture_validation_cache) is not PeriodicDescriptorCacheV2
            or type(self.plans) is not tuple
            or tuple(plan.system_id for plan in self.plans)
            != tuple(system_id for system_id, _ in CACHEABLE_SYSTEM_STREAMS)
            or any(type(plan) is not PeriodicDescriptorCaptureTrainingPlan for plan in self.plans)
        ):
            raise PeriodicCacheExecutionError("result readers or plan census changed")

    def __repr__(self) -> str:
        return (
            f"PeriodicDescriptorCacheBuildResult(seed={self.seed}, "
            f"manifest_sha256={self.manifest_sha256!r}, <private tree hidden>)"
        )


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _lower_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TypeError(f"{label} must be one lowercase SHA-256 hex digest")
    return value


def _canonical_json(value: object) -> bytes:
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


def _pairs_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise PeriodicCacheExecutionError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_float(value: str) -> object:
    raise PeriodicCacheExecutionError(f"JSON floating point is forbidden: {value}")


def _parse_canonical_json(raw: bytes, label: str) -> dict[str, object]:
    if type(raw) is not bytes:
        raise TypeError(f"{label} must be exact built-in bytes")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_float=_reject_float,
            parse_constant=_reject_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise PeriodicCacheExecutionError(f"{label} is not bounded canonical JSON") from error
    if type(value) is not dict or _canonical_json(value) != raw:
        raise PeriodicCacheExecutionError(f"{label} is not exact canonical JSON")
    return value


def _closed(value: object, keys: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise PeriodicCacheExecutionError(f"{label} keys are not closed")
    return value


def _reject_symlink_components(path: Path, label: str) -> None:
    for component in (path, *path.parents):
        if component.is_symlink():
            raise PeriodicCacheExecutionError(f"{label} cannot traverse a symlink")


def _stable_stat_identity(
    value: os.stat_result,
) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_mode,
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_nlink,
    )


def _cross_interface_identity(
    value: os.stat_result,
) -> tuple[int, int, int, int, int]:
    # Windows path and descriptor APIs can expose different ctime semantics.
    # Compare ctime and nlink only before/after within the same API.
    return (
        value.st_mode,
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )


def _open_owned(
    path: Path,
    label: str,
) -> tuple[int, os.stat_result, os.stat_result]:
    _reject_symlink_components(path, label)
    try:
        before = path.lstat()
    except OSError as error:
        raise PeriodicCacheExecutionError(f"{label} is absent") from error
    if not stat.S_ISREG(before.st_mode):
        raise PeriodicCacheExecutionError(f"{label} must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise PeriodicCacheExecutionError(f"{label} cannot be opened safely") from error
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode) or _cross_interface_identity(
        opened
    ) != _cross_interface_identity(before):
        os.close(descriptor)
        raise PeriodicCacheExecutionError(f"{label} identity changed before consumption")
    return descriptor, before, opened


def _verify_after_read(
    descriptor: int,
    path: Path,
    path_before: os.stat_result,
    opened_before: os.stat_result,
    label: str,
) -> None:
    try:
        opened_after = os.fstat(descriptor)
        path_after = path.lstat()
    except OSError as error:
        raise PeriodicCacheExecutionError(f"{label} changed during consumption") from error
    if (
        not stat.S_ISREG(opened_after.st_mode)
        or not stat.S_ISREG(path_after.st_mode)
        or _stable_stat_identity(opened_before) != _stable_stat_identity(opened_after)
        or _stable_stat_identity(path_before) != _stable_stat_identity(path_after)
        or _cross_interface_identity(opened_after) != _cross_interface_identity(path_after)
    ):
        raise PeriodicCacheExecutionError(f"{label} changed during consumption")


def _read_owned(path: Path, *, maximum: int, label: str) -> bytes:
    if type(maximum) is not int or maximum < 1:
        raise TypeError("maximum must be an exact positive int")
    descriptor, path_before, opened = _open_owned(path, label)
    try:
        if opened.st_size > maximum:
            raise PeriodicCacheExecutionError(f"{label} exceeds its byte bound")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise PeriodicCacheExecutionError(f"{label} exceeds its byte bound")
        raw = b"".join(chunks)
        if len(raw) != opened.st_size:
            raise PeriodicCacheExecutionError(f"{label} size changed during consumption")
        _verify_after_read(descriptor, path, path_before, opened, label)
    finally:
        os.close(descriptor)
    return raw


def _hash_owned(path: Path, *, maximum: int, label: str) -> tuple[str, int]:
    if type(maximum) is not int or maximum < 1:
        raise TypeError("maximum must be an exact positive int")
    descriptor, path_before, opened = _open_owned(path, label)
    digest = hashlib.sha256()
    total = 0
    try:
        if opened.st_size > maximum:
            raise PeriodicCacheExecutionError(f"{label} exceeds its byte bound")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
            if total > maximum:
                raise PeriodicCacheExecutionError(f"{label} exceeds its byte bound")
        if total != opened.st_size:
            raise PeriodicCacheExecutionError(f"{label} size changed during consumption")
        _verify_after_read(descriptor, path, path_before, opened, label)
    finally:
        os.close(descriptor)
    return digest.hexdigest(), total


def _existing_directory(path: Path, label: str) -> Path:
    _reject_symlink_components(path, label)
    try:
        value = path.resolve(strict=True)
        metadata = value.lstat()
    except OSError as error:
        raise PeriodicCacheExecutionError(f"{label} is absent") from error
    if not stat.S_ISDIR(metadata.st_mode) or value.is_symlink():
        raise PeriodicCacheExecutionError(f"{label} must be a non-symlink directory")
    return value


def _exact_names(root: Path, expected: set[str], label: str) -> None:
    try:
        names = {entry.name for entry in os.scandir(root)}
    except OSError as error:
        raise PeriodicCacheExecutionError(f"{label} cannot be enumerated") from error
    if names != expected:
        raise PeriodicCacheExecutionError(f"{label} member census changed")


def _write_exclusive(path: Path, raw: bytes, label: str) -> None:
    if type(raw) is not bytes:
        raise TypeError("published artifacts must be exact built-in bytes")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
        try:
            view = memoryview(raw)
            while view:
                written = os.write(descriptor, view)
                if written < 1:
                    raise OSError("short write")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise PeriodicCacheExecutionError(f"{label} exclusive publish failed") from error
    if _read_owned(path, maximum=max(1, len(raw)), label=label) != raw:
        raise PeriodicCacheExecutionError(f"{label} post-publish bytes changed")


def _frozen_floors(value: object) -> np.ndarray:
    try:
        checked = validate_energy_floors(value)
    except (TypeError, ValueError) as error:
        raise PeriodicCacheExecutionError("energy floors are invalid") from error
    frozen = np.array(checked, dtype=np.float64, copy=True, order="C")
    frozen.setflags(write=False)
    return frozen


def _qualification_facts(
    qualification: object,
    expected_sha256: object,
    seed: object,
) -> tuple[BaseQualification, str, str, str, str]:
    if type(seed) is not int or seed not in OFFICIAL_SEEDS or tuple(SEEDS) != OFFICIAL_SEEDS:
        raise PeriodicCacheExecutionError("cache seed is outside the frozen census")
    try:
        expected = _lower_sha256(expected_sha256, "qualification_sha256")
        raw = canonical_base_qualification_bytes(
            qualification,
            expected_sha256=expected,
        )
    except (TypeError, ValueError) as error:
        raise PeriodicCacheExecutionError("base qualification is not canonical") from error
    if type(qualification) is not BaseQualification or _sha256(raw) != expected:
        raise PeriodicCacheExecutionError("base qualification digest mismatch")
    position = OFFICIAL_SEEDS.index(seed)
    system_id = qualification.winner_system_id
    run_id = f"phaseset-run-v1/BASE_QUALIFICATION/{seed}/{system_id}"
    return (
        qualification,
        expected,
        run_id,
        qualification.winner_terminal_sha256s[position],
        qualification.winner_checkpoint_sha256s[position],
    )


def _decode_floor_payload(raw: bytes) -> np.ndarray:
    try:
        with zipfile.ZipFile(io.BytesIO(raw), mode="r") as archive:
            rows = archive.infolist()
            if (
                len(rows) != 1
                or rows[0].filename != "energy_floors.npy"
                or rows[0].is_dir()
                or rows[0].flag_bits & 0x1
                or rows[0].file_size > MAX_FLOOR_NPY_BYTES
            ):
                raise PeriodicCacheExecutionError(
                    "floor payload member census or decoded bound changed"
                )
        with np.load(io.BytesIO(raw), allow_pickle=False) as values:
            if values.files != ["energy_floors"]:
                raise PeriodicCacheExecutionError("floor payload keys are not exact")
            floors = np.array(values["energy_floors"], copy=True, order="C")
    except PeriodicCacheExecutionError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as error:
        raise PeriodicCacheExecutionError("floor payload is not safe numeric NPZ") from error
    return _frozen_floors(floors)


def load_qualified_energy_floor_input(
    root: str | Path,
    *,
    seed: int,
    qualification: BaseQualification,
    qualification_sha256: str,
) -> QualifiedEnergyFloorInput:
    """Load the two exact legacy v1 files and retain their consumed bytes."""

    checked_qualification, qualification_digest, run_id, terminal, _checkpoint = (
        _qualification_facts(qualification, qualification_sha256, seed)
    )
    directory = _existing_directory(Path(root), "qualified energy-floor root")
    _exact_names(
        directory,
        {FLOOR_RECORD_NAME, FLOOR_PAYLOAD_NAME},
        "qualified energy-floor root",
    )
    record_raw = _read_owned(
        directory / FLOOR_RECORD_NAME,
        maximum=MAX_FLOOR_RECORD_BYTES,
        label="qualified energy-floor record",
    )
    payload_raw = _read_owned(
        directory / FLOOR_PAYLOAD_NAME,
        maximum=MAX_FLOOR_PAYLOAD_BYTES,
        label="qualified energy-floor payload",
    )
    try:
        record = parse_periodic_cache_bytes(record_raw)
    except (TypeError, ValueError) as error:
        raise PeriodicCacheExecutionError("qualified energy-floor record is invalid") from error
    if (
        record.seed != seed
        or record.qualified_base_system_id != checked_qualification.winner_system_id
        or record.source_run_id != run_id
        or record.base_terminal_sha256 != terminal
        or record.qualification_sha256 != qualification_digest
        or record.cache_content_sha256 != _sha256(payload_raw)
    ):
        raise PeriodicCacheExecutionError("qualified energy-floor bytes differ from base selection")
    floors = _decode_floor_payload(payload_raw)
    return QualifiedEnergyFloorInput(
        record=record,
        energy_floors=floors,
        record_json_bytes=record_raw,
        payload_npz_bytes=payload_raw,
        record_sha256=_sha256(record_raw),
        payload_sha256=_sha256(payload_raw),
        qualification_sha256=qualification_digest,
        energy_floors_sha256=energy_floors_sha256(floors),
        _seal=_FLOOR_TOKEN,
    )


def _validate_floor_input(
    value: object,
    *,
    seed: int,
    qualification: BaseQualification,
    qualification_sha256: str,
) -> QualifiedEnergyFloorInput:
    if type(value) is not QualifiedEnergyFloorInput or value._seal is not _FLOOR_TOKEN:
        raise TypeError("floors must be exact QualifiedEnergyFloorInput")
    checked = QualifiedEnergyFloorInput(
        record=value.record,
        energy_floors=value.energy_floors,
        record_json_bytes=value.record_json_bytes,
        payload_npz_bytes=value.payload_npz_bytes,
        record_sha256=value.record_sha256,
        payload_sha256=value.payload_sha256,
        qualification_sha256=value.qualification_sha256,
        energy_floors_sha256=value.energy_floors_sha256,
        _seal=_FLOOR_TOKEN,
    )
    _qualification, expected, run_id, terminal, _checkpoint = _qualification_facts(
        qualification,
        qualification_sha256,
        seed,
    )
    record = checked.record
    if (
        checked.qualification_sha256 != expected
        or record.seed != seed
        or record.qualified_base_system_id != qualification.winner_system_id
        or record.source_run_id != run_id
        or record.base_terminal_sha256 != terminal
        or record.qualification_sha256 != expected
        or record.cache_content_sha256 != checked.payload_sha256
        or parse_periodic_cache_bytes(checked.record_json_bytes) != record
        or not np.array_equal(
            _decode_floor_payload(checked.payload_npz_bytes),
            checked.energy_floors,
        )
    ):
        raise PeriodicCacheExecutionError("qualified energy-floor input binding changed")
    return checked


def _capture_snapshot_and_facts(
    source: object,
) -> tuple[CaptureValidationSource, str, int, int, int]:
    if type(source) is not CaptureValidationSource:
        raise TypeError("capture_source must be exact CaptureValidationSource")
    checked = CaptureValidationSource(source.split, source.manifest_sha256, source.captures)
    descriptors = tuple(
        window.descriptor_source for capture in checked.captures for window in capture.windows
    )
    if not descriptors or any(
        type(row) is not CaptureDescriptorWindowSource for row in descriptors
    ):
        raise PeriodicCacheExecutionError(
            "capture source requires complete authenticated v2 window lineage"
        )
    storage_manifest = descriptors[0].storage_manifest_sha256
    if any(row.storage_manifest_sha256 != storage_manifest for row in descriptors):
        raise PeriodicCacheExecutionError("capture storage manifest changed across windows")
    window_count = len(descriptors)
    caption_count = sum(
        len(capture.holistic_text.caption_commitments) for capture in checked.captures
    )
    return checked, storage_manifest, len(checked.captures), window_count, caption_count


def _validate_inputs(
    *,
    seed: object,
    config: object,
    qualification: object,
    qualification_sha256: object,
    winner_checkpoint: object,
    floors: object,
    train_source: object,
    capture_source: object,
    source_tree_sha256: object,
    environment_sha256: object,
    bounds: object,
) -> tuple[
    TrainingConfig,
    BaseQualification,
    str,
    str,
    str,
    QualifiedEnergyFloorInput,
    PreparedTrainingDataSourceV2,
    CaptureValidationSource,
    str,
    int,
    int,
    int,
    str,
    str,
    PeriodicCacheExecutionBounds,
    Path,
]:
    if type(config) is not TrainingConfig:
        raise TypeError("config must be exact TrainingConfig")
    if (
        type(seed) is not int
        or config.stage != "residual"
        or config.seed != seed
        or config.synthetic_contract
        or config.epochs != RESIDUAL_EPOCHS
    ):
        raise PeriodicCacheExecutionError(
            "cache build requires the exact formal 20-epoch residual config for its seed"
        )
    if type(bounds) is not PeriodicCacheExecutionBounds:
        raise TypeError("bounds must be exact PeriodicCacheExecutionBounds")
    if bounds.descriptor.max_edges_per_shard > config.edge_budget:
        raise PeriodicCacheExecutionError(
            "descriptor shard edge bound exceeds the formal runtime edge budget"
        )
    checked_qualification, qualification_digest, run_id, terminal, checkpoint = (
        _qualification_facts(qualification, qualification_sha256, seed)
    )
    checked_floors = _validate_floor_input(
        floors,
        seed=seed,
        qualification=checked_qualification,
        qualification_sha256=qualification_digest,
    )
    if type(train_source) is not PreparedTrainingDataSourceV2:
        raise TypeError("train_source must be exact PreparedTrainingDataSourceV2")
    if train_source.split != "train":
        raise PeriodicCacheExecutionError("descriptor cache source must be train-only")
    checked_capture, storage_manifest, capture_count, window_count, caption_count = (
        _capture_snapshot_and_facts(capture_source)
    )
    source_digest = _lower_sha256(source_tree_sha256, "source_tree_sha256")
    environment_digest = _lower_sha256(environment_sha256, "environment_sha256")
    if not isinstance(winner_checkpoint, (str, Path)):
        raise TypeError("winner_checkpoint must be a path")
    checkpoint_path = Path(winner_checkpoint)
    actual_checkpoint, _size = _hash_owned(
        checkpoint_path,
        maximum=bounds.max_checkpoint_bytes,
        label="selected winner checkpoint",
    )
    if actual_checkpoint != checkpoint:
        raise PeriodicCacheExecutionError(
            "selected winner checkpoint bytes differ from base qualification"
        )
    return (
        config,
        checked_qualification,
        qualification_digest,
        run_id,
        terminal,
        checked_floors,
        train_source,
        checked_capture,
        storage_manifest,
        capture_count,
        window_count,
        caption_count,
        source_digest,
        environment_digest,
        bounds,
        checkpoint_path,
    )


def _binding_value(binding: DescriptorCacheSourceBinding) -> dict[str, str]:
    return {
        "environment_sha256": binding.environment_sha256,
        "prepared_manifest_sha256": binding.prepared_manifest_sha256,
        "source_tree_sha256": binding.source_tree_sha256,
    }


def _bounds_value(bounds: PeriodicCacheExecutionBounds) -> dict[str, object]:
    descriptor = bounds.descriptor
    return {
        "descriptor": {
            "max_decoded_shard_bytes": descriptor.max_decoded_shard_bytes,
            "max_edges_per_shard": descriptor.max_edges_per_shard,
            "max_index_bytes": descriptor.max_index_bytes,
            "max_shard_bytes": descriptor.max_shard_bytes,
            "max_shards": descriptor.max_shards,
            "max_total_shard_bytes": descriptor.max_total_shard_bytes,
        },
        "max_build_manifest_bytes": bounds.max_build_manifest_bytes,
        "max_checkpoint_bytes": bounds.max_checkpoint_bytes,
        "max_combined_shard_bytes": bounds.max_combined_shard_bytes,
        "max_combined_shards": bounds.max_combined_shards,
        "max_plan_bytes": bounds.max_plan_bytes,
    }


def _plan_value(plan: PeriodicDescriptorCaptureTrainingPlan) -> dict[str, object]:
    return {
        "path": f"{PLAN_DIRECTORY_NAME}/{plan.system_id}.json",
        "sha256": plan.sha256,
        "stream_kind": plan.stream_kind,
        "system_id": plan.system_id,
    }


def _cache_value(
    artifact: DescriptorCacheArtifact,
    binding: DescriptorCacheSourceBinding,
    relative_index: str,
) -> dict[str, object]:
    return {
        "index_path": relative_index,
        "index_sha256": artifact.index_sha256,
        "shard_count": artifact.shard_count,
        "source_binding": _binding_value(binding),
        "total_shard_bytes": artifact.total_shard_bytes,
    }


def _manifest_value(
    *,
    seed: int,
    config: TrainingConfig,
    qualification: BaseQualification,
    qualification_sha256: str,
    run_id: str,
    terminal_sha256: str,
    checkpoint_sha256: str,
    floors: QualifiedEnergyFloorInput,
    source_tree_sha256: str,
    environment_sha256: str,
    training_code_sha256: str,
    train_source: PreparedTrainingDataSourceV2,
    capture_source: CaptureValidationSource,
    storage_manifest_sha256: str,
    capture_count: int,
    window_count: int,
    caption_count: int,
    train_artifact: DescriptorCacheArtifact,
    capture_artifact: DescriptorCacheArtifact,
    train_binding: DescriptorCacheSourceBinding,
    capture_binding: DescriptorCacheSourceBinding,
    plans: tuple[PeriodicDescriptorCaptureTrainingPlan, ...],
    bounds: PeriodicCacheExecutionBounds,
) -> dict[str, object]:
    return {
        "authority": AUTHORITY,
        "bounds": _bounds_value(bounds),
        "capture_source": {
            "caption_count": caption_count,
            "capture_count": capture_count,
            "census_sha256": capture_source.census_sha256,
            "manifest_sha256": capture_source.manifest_sha256,
            "storage_manifest_sha256": storage_manifest_sha256,
            "window_count": window_count,
        },
        "capture_validation_cache": _cache_value(
            capture_artifact,
            capture_binding,
            f"{CAPTURE_CACHE_NAME}/index.json",
        ),
        "config_sha256": config.sha256,
        "energy_floor_input": {
            "energy_floors_sha256": floors.energy_floors_sha256,
            "payload_sha256": floors.payload_sha256,
            "record_sha256": floors.record_sha256,
            "schema": FLOOR_INPUT_SCHEMA,
        },
        "environment_sha256": environment_sha256,
        "external_authentication_asserted": EXTERNAL_AUTHENTICATION_ASSERTED,
        "plans": [_plan_value(plan) for plan in plans],
        "production": PRODUCTION,
        "qualification": {
            "sha256": qualification_sha256,
            "winner_checkpoint_sha256": checkpoint_sha256,
            "winner_system_id": qualification.winner_system_id,
            "winner_terminal_sha256": terminal_sha256,
            "winner_seed_run_id": run_id,
        },
        "result_claimed": RESULT_CLAIMED,
        "schema": BUILD_SCHEMA,
        "seed": seed,
        "source_tree_sha256": source_tree_sha256,
        "status": BUILD_STATUS,
        "train_cache": _cache_value(
            train_artifact,
            train_binding,
            f"{TRAIN_CACHE_NAME}/index.json",
        ),
        "train_source_manifest_sha256": train_source.manifest_sha256,
        "training_code_artifact_sha256": training_code_sha256,
    }


def _check_combined_bounds(
    train: DescriptorCacheArtifact,
    capture: DescriptorCacheArtifact,
    bounds: PeriodicCacheExecutionBounds,
) -> None:
    if train.shard_count + capture.shard_count > bounds.max_combined_shards:
        raise PeriodicCacheExecutionError("combined cache exceeds its shard-count bound")
    if train.total_shard_bytes + capture.total_shard_bytes > bounds.max_combined_shard_bytes:
        raise PeriodicCacheExecutionError("combined cache exceeds its shard-byte bound")


def _bounded_writer_limits(
    descriptor: DescriptorCacheBounds,
    *,
    remaining_shards: int,
    remaining_bytes: int,
    label: str,
) -> DescriptorCacheBounds:
    if type(remaining_shards) is not int or type(remaining_bytes) is not int:
        raise TypeError("remaining aggregate cache bounds must be exact ints")
    if remaining_shards < 1 or remaining_bytes < 1:
        raise PeriodicCacheExecutionError(f"combined cache has no remaining capacity for {label}")
    return DescriptorCacheBounds(
        max_index_bytes=descriptor.max_index_bytes,
        max_shard_bytes=descriptor.max_shard_bytes,
        max_decoded_shard_bytes=descriptor.max_decoded_shard_bytes,
        max_shards=min(descriptor.max_shards, remaining_shards),
        max_total_shard_bytes=min(
            descriptor.max_total_shard_bytes,
            remaining_bytes,
        ),
        max_edges_per_shard=descriptor.max_edges_per_shard,
    )


def _admit_all_plans(
    *,
    config: TrainingConfig,
    train_source: PreparedTrainingDataSourceV2,
    capture_source: CaptureValidationSource,
    train_cache: PeriodicDescriptorCacheV2,
    capture_cache: PeriodicDescriptorCacheV2,
) -> tuple[PeriodicDescriptorCaptureTrainingPlan, ...]:
    plans = tuple(
        admit_periodic_descriptor_capture_training_plan(
            system_id=system_id,
            config=config,
            train_source=train_source,
            capture_validation_source=capture_source,
            train_cache=train_cache,
            capture_validation_cache=capture_cache,
        )
        for system_id, _stream in CACHEABLE_SYSTEM_STREAMS
    )
    if tuple((plan.system_id, plan.stream_kind) for plan in plans) != CACHEABLE_SYSTEM_STREAMS:
        raise PeriodicCacheExecutionError("six-system descriptor plan mapping changed")
    if len({plan.sha256 for plan in plans}) != len(plans):
        raise PeriodicCacheExecutionError("six-system descriptor plans are not distinct")
    return plans


def _claim_output_root(path: Path) -> Path:
    _reject_symlink_components(path, "periodic descriptor output root")
    try:
        path.mkdir(parents=False, exist_ok=False)
    except OSError as error:
        raise PeriodicCacheExecutionError("periodic descriptor output root must be new") from error
    return path.resolve()


def _result(
    *,
    root: Path,
    seed: int,
    config_sha256: str,
    qualification_sha256: str,
    manifest: bytes,
    train_cache: PeriodicDescriptorCacheV2,
    capture_cache: PeriodicDescriptorCacheV2,
    plans: tuple[PeriodicDescriptorCaptureTrainingPlan, ...],
) -> PeriodicDescriptorCacheBuildResult:
    return PeriodicDescriptorCacheBuildResult(
        root=root,
        seed=seed,
        config_sha256=config_sha256,
        qualification_sha256=qualification_sha256,
        manifest_json_bytes=manifest,
        manifest_sha256=_sha256(manifest),
        train_cache=train_cache,
        capture_validation_cache=capture_cache,
        plans=plans,
        _seal=_RESULT_TOKEN,
    )


def build_periodic_descriptor_cache_for_seed(
    *,
    seed: int,
    config: TrainingConfig,
    qualification: BaseQualification,
    qualification_sha256: str,
    winner_checkpoint: str | Path,
    floors: QualifiedEnergyFloorInput,
    train_source: PreparedTrainingDataSourceV2,
    capture_source: CaptureValidationSource,
    output_root: str | Path,
    source_tree_sha256: str,
    environment_sha256: str,
    bounds: PeriodicCacheExecutionBounds = PeriodicCacheExecutionBounds(),
) -> PeriodicDescriptorCacheBuildResult:
    """Build two complete descriptor stores and admit all six cached systems."""

    try:
        (
            checked_config,
            checked_qualification,
            qualification_digest,
            run_id,
            terminal,
            checked_floors,
            checked_train,
            checked_capture,
            storage_manifest,
            capture_count,
            window_count,
            caption_count,
            source_digest,
            environment_digest,
            checked_bounds,
            checkpoint_path,
        ) = _validate_inputs(
            seed=seed,
            config=config,
            qualification=qualification,
            qualification_sha256=qualification_sha256,
            winner_checkpoint=winner_checkpoint,
            floors=floors,
            train_source=train_source,
            capture_source=capture_source,
            source_tree_sha256=source_tree_sha256,
            environment_sha256=environment_sha256,
            bounds=bounds,
        )
        checkpoint_sha256 = checked_qualification.winner_checkpoint_sha256s[
            OFFICIAL_SEEDS.index(seed)
        ]
        training_code_sha256 = training_code_artifact_sha256()
        train_binding = DescriptorCacheSourceBinding(
            prepared_manifest_sha256=checked_train.manifest_sha256,
            source_tree_sha256=source_digest,
            environment_sha256=environment_digest,
        )
        capture_binding = DescriptorCacheSourceBinding(
            prepared_manifest_sha256=storage_manifest,
            source_tree_sha256=source_digest,
            environment_sha256=environment_digest,
        )
        root = _claim_output_root(Path(output_root))
        train_writer_bounds = _bounded_writer_limits(
            checked_bounds.descriptor,
            remaining_shards=checked_bounds.max_combined_shards,
            remaining_bytes=checked_bounds.max_combined_shard_bytes,
            label="training cache",
        )
        train_writer = PeriodicDescriptorCacheWriterV2(
            root / TRAIN_CACHE_NAME,
            source_binding=train_binding,
            energy_floors=checked_floors.energy_floors,
            bounds=train_writer_bounds,
        )
        for epoch in range(RESIDUAL_EPOCHS):
            for batch in checked_train.iter_epoch(epoch=epoch, seed=seed):
                if type(batch) is not RetrievalTrainingBatch or batch.split != "train":
                    raise PeriodicCacheExecutionError(
                        "prepared train iterator emitted a non-training batch"
                    )
                contexts = batch.descriptor_contexts
                if type(contexts) is not tuple or len(contexts) != batch.motion_count:
                    raise PeriodicCacheExecutionError(
                        "prepared train batch lacks complete descriptor contexts"
                    )
                train_writer.add_batch(batch.groups, contexts)
        train_artifact = train_writer.finalize()

        capture_writer_bounds = _bounded_writer_limits(
            checked_bounds.descriptor,
            remaining_shards=(checked_bounds.max_combined_shards - train_artifact.shard_count),
            remaining_bytes=(
                checked_bounds.max_combined_shard_bytes - train_artifact.total_shard_bytes
            ),
            label="capture validation cache",
        )
        capture_writer = PeriodicDescriptorCacheWriterV2(
            root / CAPTURE_CACHE_NAME,
            source_binding=capture_binding,
            energy_floors=checked_floors.energy_floors,
            bounds=capture_writer_bounds,
        )
        for capture in checked_capture.captures:
            for window in capture.windows:
                capture_writer.add_batch(
                    window.groups,
                    (window.descriptor_context(seed=seed),),
                )
        capture_artifact = capture_writer.finalize()
        _check_combined_bounds(train_artifact, capture_artifact, checked_bounds)

        train_cache = PeriodicDescriptorCacheV2(
            train_artifact.root,
            expected_index_sha256=train_artifact.index_sha256,
            expected_source_binding=train_binding,
            energy_floors=checked_floors.energy_floors,
            bounds=checked_bounds.descriptor,
        )
        capture_cache = PeriodicDescriptorCacheV2(
            capture_artifact.root,
            expected_index_sha256=capture_artifact.index_sha256,
            expected_source_binding=capture_binding,
            energy_floors=checked_floors.energy_floors,
            bounds=checked_bounds.descriptor,
        )
        plans = _admit_all_plans(
            config=checked_config,
            train_source=checked_train,
            capture_source=checked_capture,
            train_cache=train_cache,
            capture_cache=capture_cache,
        )
        plan_root = root / PLAN_DIRECTORY_NAME
        plan_root.mkdir(parents=False, exist_ok=False)
        for plan in plans:
            raw = plan.canonical_bytes()
            if len(raw) > checked_bounds.max_plan_bytes:
                raise PeriodicCacheExecutionError("descriptor plan exceeds its byte bound")
            _write_exclusive(
                plan_root / f"{plan.system_id}.json",
                raw,
                f"descriptor plan {plan.system_id}",
            )

        actual_checkpoint, _size = _hash_owned(
            checkpoint_path,
            maximum=checked_bounds.max_checkpoint_bytes,
            label="selected winner checkpoint",
        )
        if (
            actual_checkpoint != checkpoint_sha256
            or checked_train.manifest_sha256 != train_source.manifest_sha256
            or checked_capture.census_sha256 != capture_source.census_sha256
            or checked_capture.manifest_sha256 != capture_source.manifest_sha256
            or checked_config.sha256 != config.sha256
            or training_code_artifact_sha256() != training_code_sha256
        ):
            raise PeriodicCacheExecutionError("cache inputs changed during construction")
        checked_floors = _validate_floor_input(
            checked_floors,
            seed=seed,
            qualification=checked_qualification,
            qualification_sha256=qualification_digest,
        )
        manifest_value = _manifest_value(
            seed=seed,
            config=checked_config,
            qualification=checked_qualification,
            qualification_sha256=qualification_digest,
            run_id=run_id,
            terminal_sha256=terminal,
            checkpoint_sha256=checkpoint_sha256,
            floors=checked_floors,
            source_tree_sha256=source_digest,
            environment_sha256=environment_digest,
            training_code_sha256=training_code_sha256,
            train_source=checked_train,
            capture_source=checked_capture,
            storage_manifest_sha256=storage_manifest,
            capture_count=capture_count,
            window_count=window_count,
            caption_count=caption_count,
            train_artifact=train_artifact,
            capture_artifact=capture_artifact,
            train_binding=train_binding,
            capture_binding=capture_binding,
            plans=plans,
            bounds=checked_bounds,
        )
        manifest = _canonical_json(manifest_value)
        if len(manifest) > checked_bounds.max_build_manifest_bytes:
            raise PeriodicCacheExecutionError("build manifest exceeds its byte bound")
        _write_exclusive(root / BUILD_MANIFEST_NAME, manifest, "descriptor build manifest")
        _exact_names(
            root,
            {
                BUILD_MANIFEST_NAME,
                TRAIN_CACHE_NAME,
                CAPTURE_CACHE_NAME,
                PLAN_DIRECTORY_NAME,
            },
            "periodic descriptor output root",
        )
        return _result(
            root=root,
            seed=seed,
            config_sha256=checked_config.sha256,
            qualification_sha256=qualification_digest,
            manifest=manifest,
            train_cache=train_cache,
            capture_cache=capture_cache,
            plans=plans,
        )
    except PeriodicCacheExecutionError:
        raise
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise PeriodicCacheExecutionError(
            "periodic descriptor cache construction failed"
        ) from error


def _manifest_cache_row(value: object, label: str) -> dict[str, object]:
    row = _closed(
        value,
        {
            "index_path",
            "index_sha256",
            "shard_count",
            "source_binding",
            "total_shard_bytes",
        },
        label,
    )
    _lower_sha256(row["index_sha256"], f"{label} index_sha256")
    if type(row["shard_count"]) is not int or row["shard_count"] < 1:
        raise PeriodicCacheExecutionError(f"{label} shard_count is invalid")
    if type(row["total_shard_bytes"]) is not int or row["total_shard_bytes"] < 1:
        raise PeriodicCacheExecutionError(f"{label} total_shard_bytes is invalid")
    return row


def _index_counts(
    path: Path,
    *,
    expected_sha256: str,
    bounds: DescriptorCacheBounds,
) -> tuple[int, int]:
    raw = _read_owned(path, maximum=bounds.max_index_bytes, label="descriptor cache index")
    if _sha256(raw) != expected_sha256:
        raise PeriodicCacheExecutionError("descriptor cache index digest changed")
    value = _parse_canonical_json(raw, "descriptor cache index")
    rows = value.get("shards")
    total = value.get("total_shard_bytes")
    if type(rows) is not list or type(total) is not int or total < 1:
        raise PeriodicCacheExecutionError("descriptor cache index census is invalid")
    return len(rows), total


def consume_periodic_descriptor_cache_build(
    root: str | Path,
    *,
    expected_manifest_sha256: str,
    seed: int,
    config: TrainingConfig,
    qualification: BaseQualification,
    qualification_sha256: str,
    winner_checkpoint: str | Path,
    floors: QualifiedEnergyFloorInput,
    train_source: PreparedTrainingDataSourceV2,
    capture_source: CaptureValidationSource,
    source_tree_sha256: str,
    environment_sha256: str,
    bounds: PeriodicCacheExecutionBounds = PeriodicCacheExecutionBounds(),
) -> PeriodicDescriptorCacheBuildResult:
    """Reopen exact build bytes and repeat reader plus six-plan admission."""

    try:
        (
            checked_config,
            checked_qualification,
            qualification_digest,
            run_id,
            terminal,
            checked_floors,
            checked_train,
            checked_capture,
            storage_manifest,
            capture_count,
            window_count,
            caption_count,
            source_digest,
            environment_digest,
            checked_bounds,
            checkpoint_path,
        ) = _validate_inputs(
            seed=seed,
            config=config,
            qualification=qualification,
            qualification_sha256=qualification_sha256,
            winner_checkpoint=winner_checkpoint,
            floors=floors,
            train_source=train_source,
            capture_source=capture_source,
            source_tree_sha256=source_tree_sha256,
            environment_sha256=environment_sha256,
            bounds=bounds,
        )
        expected_manifest = _lower_sha256(
            expected_manifest_sha256,
            "expected_manifest_sha256",
        )
        directory = _existing_directory(Path(root), "periodic descriptor build root")
        _exact_names(
            directory,
            {
                BUILD_MANIFEST_NAME,
                TRAIN_CACHE_NAME,
                CAPTURE_CACHE_NAME,
                PLAN_DIRECTORY_NAME,
            },
            "periodic descriptor build root",
        )
        manifest = _read_owned(
            directory / BUILD_MANIFEST_NAME,
            maximum=checked_bounds.max_build_manifest_bytes,
            label="descriptor build manifest",
        )
        if _sha256(manifest) != expected_manifest:
            raise PeriodicCacheExecutionError("descriptor build manifest digest mismatch")
        value = _closed(
            _parse_canonical_json(manifest, "descriptor build manifest"),
            {
                "authority",
                "bounds",
                "capture_source",
                "capture_validation_cache",
                "config_sha256",
                "energy_floor_input",
                "environment_sha256",
                "external_authentication_asserted",
                "plans",
                "production",
                "qualification",
                "result_claimed",
                "schema",
                "seed",
                "source_tree_sha256",
                "status",
                "train_cache",
                "train_source_manifest_sha256",
                "training_code_artifact_sha256",
            },
            "descriptor build manifest",
        )
        train_row = _manifest_cache_row(value["train_cache"], "train cache")
        capture_row = _manifest_cache_row(
            value["capture_validation_cache"],
            "capture validation cache",
        )
        if (
            train_row["index_path"] != f"{TRAIN_CACHE_NAME}/index.json"
            or capture_row["index_path"] != f"{CAPTURE_CACHE_NAME}/index.json"
        ):
            raise PeriodicCacheExecutionError("descriptor index relative paths changed")
        train_binding = DescriptorCacheSourceBinding(
            prepared_manifest_sha256=checked_train.manifest_sha256,
            source_tree_sha256=source_digest,
            environment_sha256=environment_digest,
        )
        capture_binding = DescriptorCacheSourceBinding(
            prepared_manifest_sha256=storage_manifest,
            source_tree_sha256=source_digest,
            environment_sha256=environment_digest,
        )
        train_cache = PeriodicDescriptorCacheV2(
            directory / TRAIN_CACHE_NAME,
            expected_index_sha256=train_row["index_sha256"],  # type: ignore[arg-type]
            expected_source_binding=train_binding,
            energy_floors=checked_floors.energy_floors,
            bounds=checked_bounds.descriptor,
        )
        capture_cache = PeriodicDescriptorCacheV2(
            directory / CAPTURE_CACHE_NAME,
            expected_index_sha256=capture_row["index_sha256"],  # type: ignore[arg-type]
            expected_source_binding=capture_binding,
            energy_floors=checked_floors.energy_floors,
            bounds=checked_bounds.descriptor,
        )
        train_count, train_bytes = _index_counts(
            directory / TRAIN_CACHE_NAME / "index.json",
            expected_sha256=train_cache.index_sha256,
            bounds=checked_bounds.descriptor,
        )
        capture_shards, capture_bytes = _index_counts(
            directory / CAPTURE_CACHE_NAME / "index.json",
            expected_sha256=capture_cache.index_sha256,
            bounds=checked_bounds.descriptor,
        )
        train_artifact = DescriptorCacheArtifact(
            root=directory / TRAIN_CACHE_NAME,
            index_path=directory / TRAIN_CACHE_NAME / "index.json",
            index_sha256=train_cache.index_sha256,
            shard_count=train_count,
            total_shard_bytes=train_bytes,
        )
        capture_artifact = DescriptorCacheArtifact(
            root=directory / CAPTURE_CACHE_NAME,
            index_path=directory / CAPTURE_CACHE_NAME / "index.json",
            index_sha256=capture_cache.index_sha256,
            shard_count=capture_shards,
            total_shard_bytes=capture_bytes,
        )
        _check_combined_bounds(train_artifact, capture_artifact, checked_bounds)
        plans = _admit_all_plans(
            config=checked_config,
            train_source=checked_train,
            capture_source=checked_capture,
            train_cache=train_cache,
            capture_cache=capture_cache,
        )
        plan_root = _existing_directory(directory / PLAN_DIRECTORY_NAME, "descriptor plan root")
        _exact_names(
            plan_root,
            {f"{system_id}.json" for system_id, _stream in CACHEABLE_SYSTEM_STREAMS},
            "descriptor plan root",
        )
        for plan in plans:
            raw = _read_owned(
                plan_root / f"{plan.system_id}.json",
                maximum=checked_bounds.max_plan_bytes,
                label=f"descriptor plan {plan.system_id}",
            )
            if raw != plan.canonical_bytes():
                raise PeriodicCacheExecutionError(
                    f"descriptor plan {plan.system_id} differs from complete admission"
                )
        training_code_sha256 = training_code_artifact_sha256()
        checkpoint_sha256, _size = _hash_owned(
            checkpoint_path,
            maximum=checked_bounds.max_checkpoint_bytes,
            label="selected winner checkpoint",
        )
        expected_value = _manifest_value(
            seed=seed,
            config=checked_config,
            qualification=checked_qualification,
            qualification_sha256=qualification_digest,
            run_id=run_id,
            terminal_sha256=terminal,
            checkpoint_sha256=checkpoint_sha256,
            floors=checked_floors,
            source_tree_sha256=source_digest,
            environment_sha256=environment_digest,
            training_code_sha256=training_code_sha256,
            train_source=checked_train,
            capture_source=checked_capture,
            storage_manifest_sha256=storage_manifest,
            capture_count=capture_count,
            window_count=window_count,
            caption_count=caption_count,
            train_artifact=train_artifact,
            capture_artifact=capture_artifact,
            train_binding=train_binding,
            capture_binding=capture_binding,
            plans=plans,
            bounds=checked_bounds,
        )
        if value != expected_value or manifest != _canonical_json(expected_value):
            raise PeriodicCacheExecutionError(
                "descriptor build manifest differs from live consumed evidence"
            )
        return _result(
            root=directory,
            seed=seed,
            config_sha256=checked_config.sha256,
            qualification_sha256=qualification_digest,
            manifest=manifest,
            train_cache=train_cache,
            capture_cache=capture_cache,
            plans=plans,
        )
    except PeriodicCacheExecutionError:
        raise
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise PeriodicCacheExecutionError(
            "periodic descriptor cache consumption verification failed"
        ) from error


__all__ = [
    "AUTHORITY",
    "BUILD_SCHEMA",
    "BUILD_STATUS",
    "CACHEABLE_SYSTEM_STREAMS",
    "EXTERNAL_AUTHENTICATION_ASSERTED",
    "FLOOR_INPUT_SCHEMA",
    "PRODUCTION",
    "PeriodicCacheExecutionBounds",
    "PeriodicCacheExecutionError",
    "PeriodicDescriptorCacheBuildResult",
    "QualifiedEnergyFloorInput",
    "RESULT_CLAIMED",
    "build_periodic_descriptor_cache_for_seed",
    "consume_periodic_descriptor_cache_build",
    "load_qualified_energy_floor_input",
]
