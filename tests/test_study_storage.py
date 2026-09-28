"""Pure disk-admission arithmetic; no model, test gallery, or GPU is opened."""

from types import SimpleNamespace

import pytest

import phaseset_core.study_storage as storage
from phaseset_core.study_storage import (
    StudyStorageHold,
    StudyStorageProjection,
    check_study_storage,
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
