from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from phaseset_core.contracts import PreparedGroupBatch, group_commitment
from phaseset_core.directional_calibration import (
    DirectionalCalibrationCapture,
    _linear_fifth_percentile,
    actor_patch_power_observations,
    fit_directional_energy_floors,
)
from phaseset_core.directional_phase import directional_phase_fields, local_pair_chunk


def _field(amplitude=1.0, actors=3):
    length = 200
    values = np.zeros((1, actors, length, 22, 3), dtype=np.float32)
    time = np.arange(length) / 20
    for actor in range(actors):
        values[0, actor, :, :, 1] = actor
        values[0, actor, :, 1:, 0] = (amplitude * np.sin(2 * np.pi * 1.125 * time + actor * 0.7))[
            :, None
        ]
    keys = tuple(bytes([actor + 1]) * 32 for actor in range(actors))
    batch = PreparedGroupBatch(
        values,
        np.ones((1, actors), dtype=np.bool_),
        np.ones((1, length), dtype=np.bool_),
        np.ones(values.shape[:-1], dtype=np.bool_),
        (keys,),
        (group_commitment(keys),),
    )
    return directional_phase_fields(batch, energy_floors=np.zeros(6, dtype=np.float64))[0]


def _fit(rows, **kwargs):
    return fit_directional_energy_floors(
        rows,
        training_components=kwargs.pop("training_components", ("C01",)),
        expected_capture_ids=kwargs.pop("expected_capture_ids", ("first",)),
        **kwargs,
    )


def test_actor_observations_use_pair_power_units_without_enumerating_edges(monkeypatch):
    field = _field()
    pair = local_pair_chunk(field, 0, 1)
    import phaseset_core.directional_phase as physical

    def no_pairs(*args, **kwargs):
        raise AssertionError("calibration must not enumerate pairs")

    monkeypatch.setattr(physical, "canonical_edge_pairs", no_pairs)
    powers, support = actor_patch_power_observations(field)
    assert powers.shape == support.shape == (3, field.patch_count, 6)
    assert not powers.flags.writeable and not support.flags.writeable
    for actor, slot in ((0, 3), (1, 4)):
        selected = pair.support_mask[0]
        np.testing.assert_allclose(
            np.log1p(powers[actor])[selected],
            pair.features[0, ..., slot][selected],
            atol=1e-15,
            rtol=1e-15,
        )


def test_fit_matches_literal_fifth_percentile_and_counts_missing_support():
    field = _field()
    powers, support = actor_patch_power_observations(field)
    result = _fit((DirectionalCalibrationCapture("first", "C01", field),))
    for band in range(6):
        observed = powers[..., band][support[..., band]]
        assert result.energy_floors[band] == pytest.approx(np.quantile(observed, 0.05))
        assert result.observed_counts[band] == observed.size
        assert result.missing_counts[band] == 3 * field.patch_count - observed.size
    assert result.capture_count == 1 and result.training_components == ("C01",)
    assert not result.energy_floors.flags.writeable


def test_actor_marginal_support_is_not_a_particular_pair_intersection():
    field = _field()
    responses = tuple(np.ones_like(value) for value in field.responses)
    masks = tuple(value.copy() for value in field.response_masks)
    centers = field.response_centers[5]
    responses[5][0, centers <= 20] = 7.0
    masks[5][1, centers <= 20] = False
    changed = replace(field, responses=responses, response_masks=masks)
    powers, support = actor_patch_power_observations(changed)
    pair = local_pair_chunk(changed, 0, 1)
    assert support[0, 0, 5] and support[1, 0, 5]
    assert powers[0, 0, 5] > 1.0
    assert powers[1, 0, 5] == 1.0
    assert pair.features[0, 0, 5, 3] == pytest.approx(np.log1p(1.0))


def test_observed_zero_power_is_included_and_has_exact_positive_zero_floor():
    field = _field(0.0)
    result = _fit((DirectionalCalibrationCapture("first", "C01", field),))
    assert not result.energy_floors.view(np.uint64).any()
    assert result.observed_counts == result.exact_zero_counts
    assert min(result.observed_counts) > 0


def test_no_support_is_not_counted_as_observed_zero():
    field = _field()
    absent = replace(
        field, response_masks=tuple(np.zeros_like(mask) for mask in field.response_masks)
    )
    powers, support = actor_patch_power_observations(absent)
    assert not powers.any() and not support.any()
    with pytest.raises(ValueError, match="at least one training observation"):
        _fit((DirectionalCalibrationCapture("first", "C01", absent),))


def test_calibration_is_collection_order_invariant_and_ignores_input_thresholds():
    first, second = _field(), _field(0.25)
    rows = (
        DirectionalCalibrationCapture("first", "C01", first),
        DirectionalCalibrationCapture("second", "C02", second),
    )
    options = dict(training_components=("C01", "C02"), expected_capture_ids=("first", "second"))
    original = _fit(rows, **options)
    changed = _fit(tuple(reversed(rows)), **options)
    np.testing.assert_array_equal(
        original.energy_floors.view(np.uint64), changed.energy_floors.view(np.uint64)
    )
    same = _fit((replace(rows[0], field=replace(first, energy_floors=np.full(6, 1e9))),))
    reference = _fit((rows[0],))
    np.testing.assert_array_equal(
        same.energy_floors.view(np.uint64), reference.energy_floors.view(np.uint64)
    )


@pytest.mark.parametrize("component", ["C00", "C03", "C09", "C11", "C15"])
def test_nontraining_capture_is_rejected_before_observations(component, monkeypatch):
    import phaseset_core.directional_calibration as calibration

    def never_collect(*args):
        raise AssertionError("non-training motion must not be observed")

    monkeypatch.setattr(calibration, "actor_patch_power_observations", never_collect)
    with pytest.raises(ValueError, match="non-training component"):
        _fit((DirectionalCalibrationCapture("first", component, _field()),))


def test_c00_can_be_training_in_an_external_fold_but_sealed_test_never_can():
    row = DirectionalCalibrationCapture("first", "C00", _field())
    assert _fit((row,), training_components=("C00",)).capture_count == 1
    for component in ("C09", "C11", "C15"):
        with pytest.raises(ValueError, match="exclude sealed test"):
            _fit((replace(row, component=component),), training_components=(component,))


def test_incomplete_duplicate_and_unexpected_capture_censuses_fail():
    row = DirectionalCalibrationCapture("first", "C01", _field())
    with pytest.raises(ValueError, match="complete training census"):
        _fit((row,), expected_capture_ids=("first", "second"))
    with pytest.raises(ValueError, match="unexpected or repeated"):
        _fit((row, row))
    with pytest.raises(ValueError, match="unexpected or repeated"):
        _fit((replace(row, capture_id="other"),))
    with pytest.raises(ValueError, match="complete training census"):
        _fit((row,), training_components=("C01", "C02"))


def test_resource_limit_never_samples_the_training_population():
    row = DirectionalCalibrationCapture("first", "C01", _field())
    with pytest.raises(MemoryError, match="RESOURCE_LIMIT"):
        _fit((row,), max_actor_patch_observations=1)


@pytest.mark.parametrize("size", [1, 2, 20, 21, 22, 103])
def test_fixed_quantile_rank_and_interpolation(size):
    values = np.arange(size, dtype=np.float64)
    assert _linear_fifth_percentile(values[::-1]) == pytest.approx((size - 1) / 20)
