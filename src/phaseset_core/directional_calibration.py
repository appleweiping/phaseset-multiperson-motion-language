"""Train-only energy calibration for the signed-vector, capture-wide frontend.

The population is one observation per actor/local patch/band, not per pair.
Observed zero power is included; absent kernel support is excluded and counted.
The inherited fifth-percentile estimator is fixed before scores are produced.
It must be refitted on each pilot/main/fold's complete training population.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .directional_phase import DirectionalPhaseField, LocalPhaseConfig
from .periodic import validate_energy_floors


SEALED_EMBODY_COMPONENTS = frozenset(("C09", "C11", "C15"))
CALIBRATION_METHOD = "signed-vector-actor-patch-power/linear-fifth-percentile-v1"
SPEED_CALIBRATION_METHOD = "speed-only-actor-patch-power/linear-fifth-percentile-v1"


@dataclass(frozen=True)
class DirectionalCalibrationCapture:
    capture_id: str  # private capture identity, never an actor-set positive ID
    component: str
    field: DirectionalPhaseField


@dataclass(frozen=True)
class DirectionalEnergyCalibration:
    energy_floors: np.ndarray
    capture_count: int
    training_components: tuple[str, ...]
    observed_counts: tuple[int, ...]
    missing_counts: tuple[int, ...]
    exact_zero_counts: tuple[int, ...]
    method: str = CALIBRATION_METHOD
    config: LocalPhaseConfig = LocalPhaseConfig()


def actor_patch_power_observations(
    field: DirectionalPhaseField,
) -> tuple[np.ndarray, np.ndarray]:
    """Return float64[K,P,6] powers and support, with no pair enumeration.

    Units agree with ``local_pair_chunk``: sum of squared complex vector
    components divided by the number of supported scalar components. Actor
    support is marginal; a particular pair can have a smaller intersection.
    No energy threshold or caption is consulted during collection.
    """
    powers = np.zeros((field.actor_count, field.patch_count, 6), dtype=np.float64)
    supported = np.zeros(powers.shape, dtype=np.bool_)
    for band, (response, mask, centers) in enumerate(
        zip(field.responses, field.response_masks, field.response_centers, strict=True)
    ):
        for patch, (start, stop) in enumerate(field.intervals):
            selected = (centers >= start) & (centers < stop)
            for actor in range(field.actor_count):
                valid = mask[actor, selected]
                count = int(valid.sum())
                if count:
                    values = response[actor, selected][valid]
                    power = float(np.vdot(values, values).real) / count
                    if not np.isfinite(power) or power < 0:
                        raise ValueError("observed signed-vector power must be finite/nonnegative")
                    powers[actor, patch, band] = power if power else 0.0
                    supported[actor, patch, band] = True
    powers.setflags(write=False)
    supported.setflags(write=False)
    return powers, supported


def _linear_fifth_percentile(values: np.ndarray) -> float:
    ordered = np.sort(values)
    if ordered.size == 0:
        raise ValueError("every band requires at least one training observation")
    # q=1/20, h=q*(n-1); no rounding-sensitive floating point rank decision.
    lower, remainder = divmod(ordered.size - 1, 20)
    first = float(ordered[lower])
    result = (
        first if not remainder else (first + (float(ordered[lower + 1]) - first) * (remainder / 20))
    )
    return result if result else 0.0


def fit_directional_energy_floors(
    captures: Iterable[DirectionalCalibrationCapture],
    *,
    training_components: tuple[str, ...],
    expected_capture_ids: tuple[str, ...],
    max_actor_patch_observations: int = 1_000_000,
) -> DirectionalEnergyCalibration:
    """Consume exactly the declared training census, or fail without a fit.

    Callers derive component/capture admission from the frozen split manifest.
    C00 is allowed in the training side of the appropriate external folds;
    main validation, pilot validation and each held-out fold are excluded by
    that invocation's admission list. Final sealed components are never fit.
    Runtime limits cannot sample observations or silently shorten captures.
    """
    if (
        not training_components
        or any(not isinstance(value, str) or not value for value in training_components)
        or len(set(training_components)) != len(training_components)
        or set(training_components) & SEALED_EMBODY_COMPONENTS
    ):
        raise ValueError("training components must be nonempty, unique and exclude sealed test")
    if (
        not expected_capture_ids
        or any(not isinstance(value, str) or not value for value in expected_capture_ids)
        or len(set(expected_capture_ids)) != len(expected_capture_ids)
    ):
        raise ValueError("expected training capture census must be nonempty and unique")
    if type(max_actor_patch_observations) is not int or max_actor_patch_observations < 1:
        raise ValueError("observation budget must be a positive integer")
    allowed, expected = set(training_components), set(expected_capture_ids)
    seen: set[str] = set()
    samples: list[list[np.ndarray]] = [[] for _ in range(6)]
    missing = np.zeros(6, dtype=np.int64)
    total = 0
    observed_components: set[str] = set()
    velocity_mode: str | None = None
    physical_config: LocalPhaseConfig | None = None
    for capture in captures:
        # Admission precedes physical collection, including for validation rows.
        if capture.component not in allowed:
            raise ValueError("calibration capture belongs to a non-training component")
        if capture.capture_id not in expected or capture.capture_id in seen:
            raise ValueError("calibration capture is unexpected or repeated")
        field = capture.field
        if velocity_mode is None:
            velocity_mode = field.velocity_mode
        elif field.velocity_mode != velocity_mode:
            raise ValueError("training calibration may not mix periodic velocity frontends")
        if physical_config is None:
            physical_config = field.config
        elif field.config != physical_config:
            raise ValueError("training calibration may not mix physical configurations")
        total += field.actor_count * field.patch_count
        if total > max_actor_patch_observations:
            raise MemoryError("RESOURCE_LIMIT: complete actor-patch population exceeds budget")
        powers, support = actor_patch_power_observations(field)
        for band in range(6):
            mask = support[..., band]
            samples[band].append(powers[..., band][mask])
            missing[band] += int(mask.size - mask.sum())
        seen.add(capture.capture_id)
        observed_components.add(capture.component)
    if seen != expected or observed_components != allowed:
        raise ValueError("consumed calibration stream differs from the complete training census")
    values = tuple(np.concatenate(rows) for rows in samples)
    floors = validate_energy_floors(
        np.array([_linear_fifth_percentile(row) for row in values], dtype=np.float64)
    ).copy()
    floors.setflags(write=False)
    assert physical_config is not None
    return DirectionalEnergyCalibration(
        energy_floors=floors,
        capture_count=len(seen),
        training_components=tuple(sorted(allowed)),
        observed_counts=tuple(int(row.size) for row in values),
        missing_counts=tuple(int(count) for count in missing),
        exact_zero_counts=tuple(int(np.count_nonzero(row == 0.0)) for row in values),
        method=CALIBRATION_METHOD if velocity_mode == "signed_vector" else SPEED_CALIBRATION_METHOD,
        config=physical_config,
    )
