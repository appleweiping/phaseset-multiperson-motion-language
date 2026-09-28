"""Disk-backed, text-independent signed-vector response cache for one capture.

Complete Morlet responses use NPY memory maps and the unchanged full-halo
arithmetic. No actor, edge, time position or rejected interval is sampled away.
Linear velocity/features and neural trajectories still use RAM; this is not
a completed training host or a constant-resident-memory guarantee.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil

import numpy as np

from .continuous_capture import PreparedContinuousCapture, physical_view_sha256
from .directional_phase import (
    ACTOR_FEATURE_DIM,
    DirectionalPhaseField,
    LocalPhaseConfig,
    VELOCITY_MODES,
    _directional_fields_from_arrays,
)
from .morlet import morlet_kernel_bank
from .periodic import validate_energy_floors


CACHE_SCHEMA = "phaseset-continuous-signed-response-cache-v1"
SPEED_CACHE_SCHEMA = "phaseset-continuous-speed-response-cache-v1"


def _cache_schema(velocity_mode: str) -> str:
    if velocity_mode not in VELOCITY_MODES:
        raise ValueError("unknown periodic velocity frontend")
    return CACHE_SCHEMA if velocity_mode == "signed_vector" else SPEED_CACHE_SCHEMA


def write_continuous_phase_cache(
    capture: PreparedContinuousCapture,
    directory: str | Path,
    *,
    config: LocalPhaseConfig = LocalPhaseConfig(),
    max_disk_bytes: int = 16 * 2**30,
    velocity_mode: str = "signed_vector",
) -> DirectionalPhaseField:
    """Write a fresh private physical cache; zero floors are collection-only.

    Calibration reads marginal powers independently of these placeholder
    floors. Reload with that split's fitted floors before computing relations.
    metadata.json is written last; an interrupted directory is not complete.
    """
    if type(capture) is not PreparedContinuousCapture:
        raise TypeError("capture must be exactly PreparedContinuousCapture")
    if type(max_disk_bytes) is not int or max_disk_bytes < 1:
        raise ValueError("disk byte budget must be a positive integer")
    schema = _cache_schema(velocity_mode)
    destination = Path(directory)
    if destination.exists():
        raise FileExistsError("a physical cache must not overwrite an existing attempt")
    response_counts = tuple(
        max(0, capture.frame_count - band.length + 1) for band in morlet_kernel_bank()
    )
    patches = (
        capture.frame_count - min(capture.frame_count, config.patch_frames)
    ) // config.hop_frames + 2
    # Six complex128+bool responses, centers, actor/root features and masks.
    projected = (
        sum(response_counts) * (capture.actor_count * 66 * 17 + 8)
        + capture.actor_count * patches * ((ACTOR_FEATURE_DIM + 3) * 4 + 2)
        + 65536
    )
    if projected > max_disk_bytes or shutil.disk_usage(destination.parent).free < projected + 2**27:
        raise MemoryError("RESOURCE_LIMIT: complete physical cache exceeds disk budget/free space")
    destination.mkdir(mode=0o700)
    field = _directional_fields_from_arrays(
        capture.skeletons[None],
        capture.track_mask[None],
        np.ones((1, capture.frame_count), dtype=np.bool_),
        np.ones((1, capture.actor_count), dtype=np.bool_),
        (capture.actor_count,),
        (capture.frame_count,),
        validate_energy_floors(np.zeros(6, dtype=np.float64)),
        config,
        response_directory=destination,
        velocity_mode=velocity_mode,
    )[0]
    small_arrays = {
        "actor-features": field.actor_features,
        "actor-patch-mask": field.actor_patch_mask,
        "root-positions": field.root_positions,
        "root-patch-mask": field.root_patch_mask,
    }
    for index, (response, mask, centers) in enumerate(
        zip(field.responses, field.response_masks, field.response_centers, strict=True), start=1
    ):
        # Empty bands have no mapping and are saved as ordinary empty NPYs.
        if not (destination / f"response-{index}.npy").exists():
            small_arrays[f"response-{index}"] = response
            small_arrays[f"response-mask-{index}"] = mask
        small_arrays[f"centers-{index}"] = centers
    for name, value in small_arrays.items():
        with (destination / f"{name}.npy").open("xb") as stream:
            np.save(stream, value, allow_pickle=False)
    metadata = {
        "schema": schema,
        "source_sha256": capture.source_sha256,
        "actor_count": capture.actor_count,
        "frame_count": capture.frame_count,
        "intervals": field.intervals,
        "config": asdict(config),
        "energy_floors": "not_fitted_in_physical_cache",
    }
    with (destination / "metadata.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, sort_keys=True)
        stream.write("\n")
    return replace(
        field,
        source_sha256=capture.source_sha256,
        actor_commitments=capture.actor_commitments,
        physical_view_sha256=physical_view_sha256(capture),
    )


def load_continuous_phase_cache(
    directory: str | Path,
    *,
    expected_source_sha256: str,
    energy_floors: np.ndarray,
    expected_velocity_mode: str = "signed_vector",
) -> DirectionalPhaseField:
    """Read-only maps with declared capture lineage and explicit fitted floors.

    Source admission/file checks belong to the existing execution manifest;
    this loader is not an independent data-rights or content-integrity gate.
    It checks cache schema and array headers without loading every response.
    """
    root = Path(directory)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if (
        metadata["schema"] != _cache_schema(expected_velocity_mode)
        or metadata["source_sha256"] != expected_source_sha256
    ):
        raise ValueError("physical cache schema or capture source lineage differs")
    config = LocalPhaseConfig(**metadata["config"])
    intervals = tuple(tuple(row) for row in metadata["intervals"])
    actors, patches = metadata["actor_count"], len(intervals)

    def mapped(name, dtype, shape):
        value = np.load(root / f"{name}.npy", allow_pickle=False, mmap_mode="r")
        if value.dtype != np.dtype(dtype) or value.shape != shape or not value.flags.c_contiguous:
            raise ValueError(f"physical cache array header differs: {name}")
        return value

    counts = tuple(
        max(0, metadata["frame_count"] - band.length + 1) for band in morlet_kernel_bank()
    )
    responses = tuple(
        mapped(f"response-{index}", np.complex128, (actors, count, 22, 3))
        for index, count in enumerate(counts, start=1)
    )
    masks = tuple(
        mapped(f"response-mask-{index}", np.bool_, response.shape)
        for index, response in enumerate(responses, start=1)
    )
    centers = tuple(
        mapped(f"centers-{index}", np.float64, (count,))
        for index, count in enumerate(counts, start=1)
    )
    return DirectionalPhaseField(
        responses,
        masks,
        centers,
        intervals,
        mapped("actor-features", np.float32, (actors, patches, ACTOR_FEATURE_DIM)),
        mapped("actor-patch-mask", np.bool_, (actors, patches)),
        mapped("root-positions", np.float32, (actors, patches, 3)),
        mapped("root-patch-mask", np.bool_, (actors, patches)),
        validate_energy_floors(energy_floors),
        config,
        source_sha256=metadata["source_sha256"],
        velocity_mode=expected_velocity_mode,
    )
