from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from phaseset_core.calibration_populations import calibration_populations


def _matrix():
    return json.loads(
        (Path(__file__).parents[1] / "configs/phaseset_v2_experiment_matrix.json").read_text()
    )


def test_main_pilot_and_fold_floors_have_distinct_exact_training_routes():
    matrix = _matrix()
    routes = {row.name: row for row in calibration_populations(matrix)}
    assert set(routes) == {"main", "pilot", "fold_0", "fold_1", "fold_2"}
    assert len(routes["main"].training_components) == 12
    assert routes["pilot"].training_components == ("C01", "C02")
    assert routes["pilot"].validation_components == ("C03",)
    assert "C03" in routes["main"].training_components
    assert "C00" not in routes["main"].training_components
    assert "C00" not in routes["fold_0"].training_components
    assert "C00" in routes["fold_1"].training_components
    assert "C00" in routes["fold_2"].training_components
    for route in routes.values():
        assert not set(route.training_components) & set(route.validation_components)
        assert not set(route.training_components) & set(route.held_out_components)
        assert not {"C09", "C11", "C15"} & set(route.training_components)
    assert matrix["status"] == "PLANNED_NOT_LAUNCHABLE"


@pytest.mark.parametrize("mutation", ["sealed", "heldout", "validation", "missing", "reuse"])
def test_bad_population_routes_cannot_admit_floors(mutation):
    matrix = deepcopy(_matrix())
    groups = matrix["pilot_and_group_fold_components"]
    if mutation == "sealed":
        groups["development"].append("C15")
    elif mutation == "heldout":
        groups["folds"][0]["train"].append("C00")
    elif mutation == "validation":
        groups["folds"][0]["train"].append("C03")
    elif mutation == "missing":
        groups["folds"][0]["train"].remove("C01")
    else:
        groups["reuse_main_checkpoints"] = True
    with pytest.raises(ValueError):
        calibration_populations(matrix)
