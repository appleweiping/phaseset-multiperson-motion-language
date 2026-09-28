"""A6 physical-control probes; no dataset, text, GPU, or fitted floor is used."""

import numpy as np
import pytest

from phaseset_core.directional_phase import (
    mean_difference_signal_dct,
    true_mean_difference_dct_features,
)


def _band_signal() -> np.ndarray:
    length = 40
    samples = np.arange(length, dtype=np.float64) + 0.5
    return np.cos(np.pi * 3 * samples / length)[:, None]


def test_paired_signal_dct_preserves_cross_term_with_identical_endpoint_power():
    signal = _band_signal()
    mask = np.ones_like(signal, dtype=np.bool_)
    same, same_support = true_mean_difference_dct_features(signal, signal, mask, mask)
    opposite, opposite_support = true_mean_difference_dct_features(
        signal, -signal, mask, mask
    )
    np.testing.assert_array_equal(same_support, opposite_support)
    assert bool(same_support.all())
    np.testing.assert_allclose(same[:, 3:5], opposite[:, 3:5], atol=1e-14)
    assert same[0, 0] > 0 and same[0, 1] == 0.0
    assert opposite[0, 0] == 0.0 and opposite[0, 1] > 0
    assert same[0, 2] == pytest.approx(1.0, abs=1e-14)
    assert opposite[0, 2] == pytest.approx(-1.0, abs=1e-14)


def test_endpoint_swap_only_exchanges_endpoint_power_not_symmetric_dct_fields():
    left = _band_signal()
    right = 0.4 * left
    mask = np.ones_like(left, dtype=np.bool_)
    forward, support = true_mean_difference_dct_features(left, right, mask, mask)
    reverse, reverse_support = true_mean_difference_dct_features(right, left, mask, mask)
    np.testing.assert_array_equal(support, reverse_support)
    np.testing.assert_allclose(forward[:, :3], reverse[:, :3], atol=1e-14)
    np.testing.assert_allclose(forward[:, 3], reverse[:, 4], atol=1e-14)
    np.testing.assert_allclose(forward[:, 4], reverse[:, 3], atol=1e-14)
    np.testing.assert_array_equal(forward[:, 5:], reverse[:, 5:])


def test_control_aggregates_complete_frequency_bands_not_only_center_bins():
    length = 40
    samples = np.arange(length, dtype=np.float64) + 0.5
    # Bin 1 is below the first 0.75 Hz center but belongs to its low band.
    signal = np.cos(np.pi * samples / length)[:, None]
    mask = np.ones_like(signal, dtype=np.bool_)
    features, support = true_mean_difference_dct_features(signal, signal, mask, mask)
    assert bool(support[0])
    assert features[0, 0] > 0
    assert features[0, 2] == pytest.approx(1.0, abs=1e-14)
    np.testing.assert_allclose(features[1:, 0], 0.0, atol=1e-27)


def test_shared_mask_excludes_invalid_values_and_empty_support_is_positive_zero():
    signal = _band_signal()
    mask = np.ones_like(signal, dtype=np.bool_)
    mask[:5] = False
    corrupted = signal.copy()
    corrupted[:5] = 10000.0
    baseline, support = true_mean_difference_dct_features(signal, signal, mask, mask)
    changed, changed_support = true_mean_difference_dct_features(
        corrupted, signal, mask, mask
    )
    np.testing.assert_array_equal(changed, baseline)
    np.testing.assert_array_equal(changed_support, support)
    assert baseline[0, 5] == pytest.approx(35 / 40)

    one_sample = np.zeros_like(mask)
    one_sample[10] = True
    empty, empty_support = true_mean_difference_dct_features(
        signal, signal, one_sample, one_sample
    )
    assert not bool(empty_support.any())
    assert not bool(np.count_nonzero(empty))
    assert not bool(np.signbit(empty).any())
    assert not empty.flags.writeable and not empty_support.flags.writeable


def test_coverage_counts_missing_channels_not_only_spectrally_eligible_channels():
    signal = np.concatenate((_band_signal(), _band_signal()), axis=1)
    mask = np.ones_like(signal, dtype=np.bool_)
    mask[:, 1] = False
    features, support = true_mean_difference_dct_features(signal, signal, mask, mask)
    assert bool(support[0])
    assert features[0, 5] == pytest.approx(0.5)


def test_multichannel_asymmetric_mask_band_sum_matches_full_orthonormal_dct():
    rng = np.random.default_rng(1729)
    left = rng.normal(size=(40, 4))
    right = rng.normal(size=(40, 4))
    left_mask = np.ones((40, 4), dtype=np.bool_)
    right_mask = np.ones((40, 4), dtype=np.bool_)
    left_mask[::4, 1] = False
    right_mask[::5, 1] = False
    right_mask[:, 3] = False
    common = left_mask & right_mask
    eligible = common.sum(axis=0) >= 2
    masked_left = np.where(common, left, 0.0)
    masked_right = np.where(common, right, 0.0)
    full_mean, full_difference = mean_difference_signal_dct(masked_left, masked_right)
    expected_mean = np.square(full_mean[1:, eligible]).sum(axis=0).mean()
    expected_difference = np.square(full_difference[1:, eligible]).sum(axis=0).mean()
    features, support = true_mean_difference_dct_features(
        left, right, left_mask, right_mask
    )
    assert bool(support.all())
    assert np.expm1(features[:, 0]).sum() == pytest.approx(expected_mean, rel=1e-12)
    assert np.expm1(features[:, 1]).sum() == pytest.approx(expected_difference, rel=1e-12)
    assert features[0, 5] == pytest.approx(common.sum() / common.size)


def test_dct_control_rejects_bad_shapes_masks_and_nonfinite_values():
    signal = _band_signal()
    mask = np.ones_like(signal, dtype=np.bool_)
    with pytest.raises(ValueError):
        true_mean_difference_dct_features(signal, signal[:-1], mask, mask)
    with pytest.raises(ValueError):
        true_mean_difference_dct_features(signal, signal, mask.astype(np.int8), mask)
    bad = signal.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        true_mean_difference_dct_features(bad, signal, mask, mask)
    with pytest.raises(ValueError):
        true_mean_difference_dct_features(1j * signal, signal, mask, mask)
    huge = np.full_like(signal, 1e200)
    with pytest.raises(ValueError, match="nonfinite"):
        true_mean_difference_dct_features(huge, huge, mask, mask)
