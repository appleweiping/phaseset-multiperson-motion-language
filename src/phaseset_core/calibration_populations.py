"""Training populations for V2 main, pilot and independent outer folds.

Read the user-supplied matrix rather than inventing component allocations.
This does not make that planned training matrix launchable or fit any data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .directional_calibration import SEALED_EMBODY_COMPONENTS


@dataclass(frozen=True)
class CalibrationPopulation:
    name: str
    training_components: tuple[str, ...]
    validation_components: tuple[str, ...]
    held_out_components: tuple[str, ...]


def calibration_populations(matrix: dict[str, Any]) -> tuple[CalibrationPopulation, ...]:
    """Derive five exact populations, rejecting inconsistent held-out routes.

    The complete eligible capture census must additionally be established
    before fitting; this function only translates the registered components.
    Main validation C00 is training data in folds 1/2, never in main/pilot.
    """
    if matrix["schema"] != "phaseset-v2-training-matrix-v1":
        raise ValueError("V2 matrix schema differs")
    groups = matrix["pilot_and_group_fold_components"]

    def components(values):
        if (
            not values
            or any(not isinstance(value, str) or not value for value in values)
            or len(set(values)) != len(values)
            or set(values) & SEALED_EMBODY_COMPONENTS
        ):
            raise ValueError("development components must be unique and exclude sealed test")
        return set(values)

    development = components(groups["development"])
    main_validation = components(matrix["primary_split"]["validation_component_labels"])
    pilot_train = components(groups["pilot_train"])
    pilot_validation = components(groups["pilot_validation"])
    fold_validation = components(groups["fold_validation"])
    if (
        not (main_validation | pilot_train | pilot_validation | fold_validation) <= development
        or pilot_train & pilot_validation
        or fold_validation != pilot_validation
        or groups["reuse_main_checkpoints"] is not False
    ):
        raise ValueError("pilot/main/fold development routes differ")
    populations = [
        CalibrationPopulation(
            "main", tuple(sorted(development - main_validation)), tuple(sorted(main_validation)), ()
        ),
        CalibrationPopulation(
            "pilot", tuple(sorted(pilot_train)), tuple(sorted(pilot_validation)), ()
        ),
    ]
    held_union: set[str] = set()
    folds = groups["folds"]
    if len(folds) != 3 or {fold["id"] for fold in folds} != {"fold_0", "fold_1", "fold_2"}:
        raise ValueError("exactly three distinct registered outer folds are required")
    for fold in sorted(folds, key=lambda value: value["id"]):
        held, training = components(fold["held_out"]), components(fold["train"])
        if (
            not held <= development
            or held & (fold_validation | pilot_train | held_union)
            or training != development - held - fold_validation
        ):
            raise ValueError(
                "outer fold training overlaps or differs from the registered complement"
            )
        held_union |= held
        populations.append(
            CalibrationPopulation(
                fold["id"],
                tuple(sorted(training)),
                tuple(sorted(fold_validation)),
                tuple(sorted(held)),
            )
        )
    if held_union != development - pilot_train - fold_validation:
        raise ValueError(
            "outer held-out components do not cover the registered evaluation population"
        )
    return tuple(populations)
