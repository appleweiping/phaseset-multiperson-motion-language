"""Data-free contract tests for old five-channel scalar Morlet floors."""

from dataclasses import replace

import numpy as np
import pytest

from phaseset_core.contracts import skeleton_to_activity
from phaseset_core.legacy_scalar_calibration import (
    LEGACY_SCALAR_CALIBRATION_METHOD,
    LegacyScalarCalibrationCapture,
    LegacyScalarFloorReceipt,
    fit_legacy_scalar_energy_floors,
    legacy_scalar_config_sha256,
)
from phaseset_core.periodic import _actor_marginal_powers, _precompute_actor_responses
from phaseset_core.pipeline import collate_group_samples
from test_phaseset_v2_legacy_capture import _capture


def _row():
    capture = _capture(windows=1, reject_middle=False)
    return LegacyScalarCalibrationCapture(capture.source_sha256, "C01", capture)


def _fit(rows, *, expected=("a" * 64,), components=("C01",)):
    return fit_legacy_scalar_energy_floors(
        rows, training_components=components, expected_capture_ids=expected,
    )


def test_old_scalar_floor_is_fifth_percentile_of_every_actor_window() -> None:
    row = _row()
    fit = _fit((row,))
    window = row.capture.windows()[0]
    activity = skeleton_to_activity(collate_group_samples((window,)))
    responses, _, length_mask = _precompute_actor_responses(
        activity.activities[0, :3], activity.activity_mask[0, :3], 200,
    )
    direct = np.stack([_actor_marginal_powers(item, length_mask) for item in responses])
    ordered = np.sort(direct, axis=0)
    expected = ordered[0] + (ordered[1] - ordered[0]) * 0.1
    np.testing.assert_array_equal(fit.floors, expected)
    assert fit.method == LEGACY_SCALAR_CALIBRATION_METHOD
    assert fit.config_sha256 == legacy_scalar_config_sha256()
    assert fit.capture_count == fit.accepted_window_count == 1
    assert fit.actor_window_count == 3
    assert fit.observed_counts == (3,) * 6
    assert fit.missing_counts == (0,) * 6
    assert not fit.floors.flags.writeable
    receipt = LegacyScalarFloorReceipt.from_fit(
        fit, training_source_manifest_sha256="f" * 64,
    )
    assert receipt.capture_count == 1
    assert receipt.actor_window_count == 3
    np.testing.assert_array_equal(receipt.floors, fit.floors)
    assert len(receipt.sha256) == 64
    receipt.require_population(("C01",))
    with pytest.raises(ValueError, match="population"):
        receipt.require_population(("C02",))


def test_old_scalar_floor_refuses_incomplete_duplicate_sealed_or_augmented_sources() -> None:
    row = _row()
    with pytest.raises(ValueError, match="complete training census"):
        _fit((row,), expected=("a" * 64, "b" * 64))
    with pytest.raises(ValueError, match="repeated"):
        _fit((row, row))
    with pytest.raises(ValueError, match="training components"):
        _fit((row,), components=("C15",))
    with pytest.raises(ValueError, match="unaugmented"):
        _fit((replace(row, capture=replace(row.capture, augmentation_yaw=0.1)),))
    with pytest.raises(MemoryError, match="RESOURCE_LIMIT"):
        fit_legacy_scalar_energy_floors(
            (row,), training_components=("C01",), expected_capture_ids=("a" * 64,),
            max_actor_window_observations=2,
        )


def test_old_scalar_observed_zero_power_is_not_treated_as_missing() -> None:
    row = _row()
    stationary = replace(row.capture, skeletons=np.zeros_like(row.capture.skeletons))
    fit = _fit((replace(row, capture=stationary),))
    assert fit.observed_counts == (3,) * 6
    assert fit.missing_counts == (0,) * 6
    assert fit.exact_zero_counts == (3,) * 6
    assert fit.floors.tobytes() == np.zeros((6,), dtype=np.float64).tobytes()


def test_old_scalar_receipt_rejects_fake_coverage_and_wrong_manifest() -> None:
    fit = _fit((_row(),))
    with pytest.raises(ValueError, match="training source"):
        LegacyScalarFloorReceipt.from_fit(fit, training_source_manifest_sha256="bad")
    receipt = LegacyScalarFloorReceipt.from_fit(
        fit, training_source_manifest_sha256="f" * 64,
    )
    with pytest.raises(ValueError, match="diagnostics"):
        replace(receipt, observed_counts=(2,) * 6)
