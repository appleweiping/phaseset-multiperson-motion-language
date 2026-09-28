"""Fail-closed disk admission for one whole-parent training attempt.

This is a resource check, not scientific, dataset, or GPU launch authority.
The caller supplies a family-specific checkpoint bound from an independently
reviewed real-width serialization dry run; this module does not infer one from
the current free space or a synthetic test fixture.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from collections.abc import Mapping


class StudyStorageHold(RuntimeError):
    """The attempt lacks a defensible byte bound or sufficient free space."""


V2_MATRIX_SHA256 = "482a4432ac95cfbd21d5b6363893377ab61f6c84373bcf07b7441c4bfa8891dd"


def forecast_v2_cumulative_storage(
    matrix_bytes: bytes,
    *,
    checkpoint_bounds_by_run: Mapping[str, int],
    physical_test_cache_bytes: int,
    sealed_output_bytes: int,
    metadata_bytes_per_run: int,
    uncertainty_bytes: int,
) -> dict[str, int | str | bool]:
    """Advisory full 87-stage storage arithmetic, never launch admission.

    This does not authenticate per-run checkpoint specimens or prove cache,
    output and TMPDIR mount identity. No caller may treat its numeric result
    as a qualified ``StudyStorageProjection`` or GPU/formal-launch authority.
    The production gate must separately verify specimen provenance, bound
    every remaining family, check all target mounts and read free space under
    the study lock. Missing run bounds fail closed even for this planning view.
    """
    if type(matrix_bytes) is not bytes or hashlib.sha256(matrix_bytes).hexdigest() != V2_MATRIX_SHA256:
        raise StudyStorageHold("exact frozen V2 matrix bytes required")
    try:
        matrix = json.loads(matrix_bytes)
    except (ValueError, UnicodeDecodeError) as exc:
        raise StudyStorageHold("invalid V2 matrix JSON") from exc
    runs = matrix.get("runs") if isinstance(matrix, dict) else None
    expected_ids = [f"V2-{index:03d}" for index in range(1, 88)]
    if (
        not isinstance(matrix, dict)
        or matrix.get("schema") != "phaseset-v2-training-matrix-v1"
        or matrix.get("status") != "PLANNED_NOT_LAUNCHABLE"
        or not isinstance(runs, list)
        or len(runs) != 87
        or [row.get("id") for row in runs if isinstance(row, dict)] != expected_ids
        or matrix.get("budget", {}).get("formal_stages") != 87
        or matrix.get("budget", {}).get("max_formal_starts") != 90
        or matrix.get("budget", {}).get("max_root_cause_restarts") != 3
    ):
        raise StudyStorageHold("complete frozen 87-stage V2 matrix required")
    if not isinstance(checkpoint_bounds_by_run, Mapping) or set(checkpoint_bounds_by_run) != set(expected_ids):
        raise StudyStorageHold("one explicit checkpoint bound per formal run required")
    for run_id, bound in checkpoint_bounds_by_run.items():
        if type(bound) is not int or bound < 1:
            raise StudyStorageHold(f"invalid checkpoint bound for {run_id}")
    for name, value in (
        ("physical_test_cache_bytes", physical_test_cache_bytes),
        ("sealed_output_bytes", sealed_output_bytes),
        ("metadata_bytes_per_run", metadata_bytes_per_run),
        ("uncertainty_bytes", uncertainty_bytes),
    ):
        if type(value) is not int or value < 1:
            raise StudyStorageHold(f"{name} needs a positive measured or reviewed byte bound")

    retained_stage_bytes = 4 * sum(checkpoint_bounds_by_run.values())
    maximum_payload_bytes = max(checkpoint_bounds_by_run.values())
    restart_bytes = 3 * 4 * maximum_payload_bytes
    atomic_transient_bytes = 2 * maximum_payload_bytes
    metadata_bytes = 90 * metadata_bytes_per_run
    free_floor_bytes = 16 * 1024**3
    remaining = (
        retained_stage_bytes
        + restart_bytes
        + atomic_transient_bytes
        + physical_test_cache_bytes
        + sealed_output_bytes
        + metadata_bytes
        + uncertainty_bytes
    )
    return {
        "status": "ADVISORY_ARITHMETIC_ONLY_NOT_ADMISSION",
        "admission_ready": False,
        "scientific_or_launch_authority": False,
        "bound_provenance_verified": False,
        "target_mounts_verified": False,
        "matrix_sha256": V2_MATRIX_SHA256,
        "formal_stages": 87,
        "formal_restart_slots": 3,
        "retained_stage_bytes": retained_stage_bytes,
        "restart_bytes": restart_bytes,
        "atomic_transient_bytes": atomic_transient_bytes,
        "physical_test_cache_bytes": physical_test_cache_bytes,
        "sealed_output_bytes": sealed_output_bytes,
        "metadata_bytes": metadata_bytes,
        "uncertainty_bytes": uncertainty_bytes,
        "remaining_bytes_excluding_floor": remaining,
        "free_floor_bytes": free_floor_bytes,
        "required_free_bytes": remaining + free_floor_bytes,
    }


def _device(path: Path) -> int:
    return path.stat().st_dev


@dataclass(frozen=True)
class StudyStorageProjection:
    checkpoint_bound_bytes: int
    additional_write_bytes: int
    free_floor_bytes: int = 16 * 1024**3
    cumulative_remaining_bytes: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_bound_bytes",
            "additional_write_bytes",
            "free_floor_bytes",
        ):
            value = getattr(self, name)
            minimum = 1 if name == "checkpoint_bound_bytes" else 0
            if type(value) is not int or value < minimum:
                raise StudyStorageHold(f"{name} needs an exact nonnegative byte bound")
        if self.cumulative_remaining_bytes is not None and (
            type(self.cumulative_remaining_bytes) is not int
            or self.cumulative_remaining_bytes < 1
        ):
            raise StudyStorageHold("cumulative remaining bytes must be a positive integer")

    @property
    def required_free_bytes(self) -> int:
        # Two recent, up to two selected-best dependencies, the next payload,
        # and the in-progress atomic-save temporary payload.
        attempt = 6 * self.checkpoint_bound_bytes + self.additional_write_bytes
        if self.cumulative_remaining_bytes is not None:
            attempt = max(attempt, self.cumulative_remaining_bytes)
        return attempt + self.free_floor_bytes

    @property
    def monitor_floor_bytes(self) -> int:
        # Admission already reserves all six payloads.  During a valid write,
        # free bytes may fall to the fixed floor without violating that bound.
        return self.free_floor_bytes


def current_free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def check_study_storage(
    output_parent: str | Path,
    temporary_directory: str | Path,
    projection: StudyStorageProjection,
) -> dict[str, int]:
    """Read-only admission on one filesystem; recheck just before reservation.

    ``output_parent`` is the already-existing parent of the fresh attempt.
    Temporary files must remain on that same device, so a full system root
    cannot be hidden by free data-volume space. Other users can change free
    bytes after this snapshot; the host still handles ENOSPC as a failed run.
    """
    if not isinstance(projection, StudyStorageProjection):
        raise StudyStorageHold("family-specific storage projection is required")
    output = Path(output_parent).resolve(strict=True)
    temporary = Path(temporary_directory).resolve(strict=True)
    if not output.is_dir() or not temporary.is_dir():
        raise StudyStorageHold("existing output parent and TMPDIR directories required")
    if _device(output) != _device(temporary):
        raise StudyStorageHold("attempt output and TMPDIR must share the same filesystem")
    available = current_free_bytes(output)
    if available < projection.required_free_bytes:
        raise StudyStorageHold("RESOURCE_LIMIT: insufficient same-filesystem free bytes")
    return {
        "available_bytes": available,
        "required_free_bytes": projection.required_free_bytes,
        "checkpoint_bound_bytes": projection.checkpoint_bound_bytes,
        "additional_write_bytes": projection.additional_write_bytes,
        "free_floor_bytes": projection.free_floor_bytes,
        "cumulative_remaining_bytes": projection.cumulative_remaining_bytes,
    }
