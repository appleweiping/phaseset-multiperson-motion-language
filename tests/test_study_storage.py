"""Pure disk-admission arithmetic; no model, test gallery, or GPU is opened."""

from types import SimpleNamespace
from pathlib import Path

import pytest

import phaseset_core.study_storage as storage
from phaseset_core.study_storage import (
    StudyStorageHold,
    StudyStorageProjection,
    check_study_storage,
    forecast_v2_cumulative_storage,
)


_V2_MATRIX = (Path(__file__).parents[1] / "configs" / "phaseset_v2_experiment_matrix.json").read_bytes()


def test_complete_v2_cumulative_forecast_keeps_all_stages_and_restart_envelope():
    bounds = {f"V2-{index:03d}": 10 for index in range(1, 88)}
    result = forecast_v2_cumulative_storage(
        _V2_MATRIX,
        checkpoint_bounds_by_run=bounds,
        physical_test_cache_bytes=100,
        sealed_output_bytes=200,
        metadata_bytes_per_run=3,
        uncertainty_bytes=400,
    )
    assert result["status"] == "ADVISORY_ARITHMETIC_ONLY_NOT_ADMISSION"
    assert result["admission_ready"] is False
    assert result["scientific_or_launch_authority"] is False
    assert result["bound_provenance_verified"] is False
    assert result["target_mounts_verified"] is False
    assert result["formal_stages"] == 87
    assert result["formal_restart_slots"] == 3
    assert result["retained_stage_bytes"] == 3480
    assert result["restart_bytes"] == 120
    assert result["atomic_transient_bytes"] == 20
    assert result["metadata_bytes"] == 270
    assert result["remaining_bytes_excluding_floor"] == 4590
    assert result["free_floor_bytes"] == 16 * 1024**3
    assert result["required_free_bytes"] == 4590 + 16 * 1024**3


def test_cumulative_forecast_fails_closed_on_drift_missing_bound_or_unmeasured_inputs():
    bounds = {f"V2-{index:03d}": 10 for index in range(1, 88)}
    inputs = dict(
        checkpoint_bounds_by_run=bounds,
        physical_test_cache_bytes=100,
        sealed_output_bytes=200,
        metadata_bytes_per_run=3,
        uncertainty_bytes=400,
    )
    with pytest.raises(StudyStorageHold, match="exact frozen"):
        forecast_v2_cumulative_storage(_V2_MATRIX + b"\n", **inputs)
    with pytest.raises(StudyStorageHold, match="per formal run"):
        forecast_v2_cumulative_storage(
            _V2_MATRIX, **{**inputs, "checkpoint_bounds_by_run": {"V2-001": 10}}
        )
    with pytest.raises(StudyStorageHold, match="V2-087"):
        forecast_v2_cumulative_storage(
            _V2_MATRIX,
            **{**inputs, "checkpoint_bounds_by_run": {**bounds, "V2-087": True}},
        )
    with pytest.raises(StudyStorageHold, match="physical_test_cache_bytes"):
        forecast_v2_cumulative_storage(
            _V2_MATRIX, **{**inputs, "physical_test_cache_bytes": 0}
        )


def test_projection_requires_explicit_real_family_bound():
    for invalid in (0, -1, 1.0, True, None):
        with pytest.raises(StudyStorageHold):
            StudyStorageProjection(invalid, 0)
    with pytest.raises(StudyStorageHold):
        StudyStorageProjection(123, -1)
    projection = StudyStorageProjection(200, 17, free_floor_bytes=1000)
    assert projection.required_free_bytes == 6 * 200 + 17 + 1000
    assert projection.monitor_floor_bytes == 1000
    assert StudyStorageProjection(
        200, 17, free_floor_bytes=1000, cumulative_remaining_bytes=3000
    ).required_free_bytes == 4000
    with pytest.raises(StudyStorageHold):
        StudyStorageProjection(200, 0, cumulative_remaining_bytes=0)


def test_exact_floor_admits_and_one_byte_less_holds(tmp_path, monkeypatch):
    temporary = tmp_path / "tmp"
    temporary.mkdir()
    projection = StudyStorageProjection(200, 17, free_floor_bytes=1000)
    needed = projection.required_free_bytes
    monkeypatch.setattr(
        storage.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=needed),
    )
    row = check_study_storage(tmp_path, temporary, projection)
    assert row == {
        "available_bytes": needed,
        "required_free_bytes": needed,
        "checkpoint_bound_bytes": 200,
        "additional_write_bytes": 17,
        "free_floor_bytes": 1000,
        "cumulative_remaining_bytes": None,
    }
    monkeypatch.setattr(
        storage.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=needed - 1),
    )
    with pytest.raises(StudyStorageHold, match="RESOURCE_LIMIT"):
        check_study_storage(tmp_path, temporary, projection)


def test_different_tmp_device_or_missing_bound_holds_without_output(tmp_path, monkeypatch):
    temporary = tmp_path / "tmp"
    temporary.mkdir()
    with pytest.raises(StudyStorageHold, match="projection"):
        check_study_storage(tmp_path, temporary, None)
    with pytest.raises(FileNotFoundError):
        check_study_storage(tmp_path / "not-created", temporary, StudyStorageProjection(1, 0))
    original = storage._device
    monkeypatch.setattr(
        storage,
        "_device",
        lambda path: original(path) + (1 if path == temporary else 0),
    )
    with pytest.raises(StudyStorageHold, match="same"):
        check_study_storage(tmp_path, temporary, StudyStorageProjection(1, 0))
    assert not (tmp_path / "attempt").exists()
