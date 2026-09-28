"""Fail-closed disk admission for one whole-parent training attempt.

This is a resource check, not scientific, dataset, or GPU launch authority.
The caller supplies a family-specific checkpoint bound from an independently
reviewed real-width serialization dry run; this module does not infer one from
the current free space or a synthetic test fixture.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil


class StudyStorageHold(RuntimeError):
    """The attempt lacks a defensible byte bound or sufficient free space."""


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
