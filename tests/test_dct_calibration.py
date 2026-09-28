"""A6 train-only DCT floor arithmetic/census probes on synthetic body22."""

from dataclasses import replace

import numpy as np
import pytest

import phaseset_core.dct_calibration as calibration
from phaseset_core.dct_calibration import (
    DCT_CALIBRATION_METHOD,
    DctCalibrationCapture,
    actor_patch_dct_power_observations,
    fit_dct_energy_floors,
)
from phaseset_core.directional_phase import LocalPhaseConfig
from test_continuous_training_input import _input


@pytest.fixture(scope="module")
def synthetic_capture(tmp_path_factory):
    source, _, _, _ = _input(tmp_path_factory.mktemp("a6-dct-fit"))
    return source.capture


def test_a6_actor_patch_power_is_finite_and_preserves_observed_zeros(synthetic_capture):
    powers, support = actor_patch_dct_power_observations(
        synthetic_capture, LocalPhaseConfig()
    )
    assert powers.shape == support.shape
    assert powers.shape[0] == synthetic_capture.actor_count
    assert powers.shape[-1] == 6
    assert bool(np.isfinite(powers).all()) and bool((powers >= 0).all())
    assert bool(support.any()) and bool((~support).any())
    assert not powers.flags.writeable and not support.flags.writeable
    assert not bool(np.count_nonzero(powers[~support]))

    stationary = replace(
        synthetic_capture, skeletons=np.zeros_like(synthetic_capture.skeletons)
    )
    zeros, stationary_support = actor_patch_dct_power_observations(
        stationary, LocalPhaseConfig()
    )
    assert bool(stationary_support.any())
    assert not bool(np.count_nonzero(zeros))
    assert not bool(np.signbit(zeros).any())


def test_a6_fifth_percentile_uses_exact_complete_parent_population(synthetic_capture):
    other = replace(synthetic_capture, source_sha256="b" * 64)
    records = (
        DctCalibrationCapture("a" * 64, "C01", synthetic_capture),
        DctCalibrationCapture("b" * 64, "C02", other),
    )
    powers, support = actor_patch_dct_power_observations(
        synthetic_capture, LocalPhaseConfig()
    )
    fitted = fit_dct_energy_floors(
        records,
        training_components=("C01", "C02"),
        expected_capture_ids=("a" * 64, "b" * 64),
        config=LocalPhaseConfig(),
    )
    assert fitted.capture_count == 2
    assert fitted.training_components == ("C01", "C02")
    assert fitted.method == DCT_CALIBRATION_METHOD
    assert not fitted.floors.flags.writeable
    for band in range(6):
        observed = powers[..., band][support[..., band]]
        assert fitted.floors[band] == pytest.approx(
            np.quantile(observed, 0.05, method="linear"), rel=1e-12, abs=1e-12
        )
        assert fitted.observed_counts[band] == 2 * len(observed)
        assert fitted.missing_counts[band] == 2 * (support[..., band].size - len(observed))
        assert fitted.exact_zero_counts[band] == 2 * int(np.count_nonzero(observed == 0))


def test_a6_floor_fitter_rejects_test_missing_repeated_and_resource_shortcuts(
    synthetic_capture,
):
    first = DctCalibrationCapture("a" * 64, "C01", synthetic_capture)
    kwargs = dict(
        training_components=("C01",),
        expected_capture_ids=("a" * 64,),
        config=LocalPhaseConfig(),
    )
    with pytest.raises(ValueError, match="contract"):
        fit_dct_energy_floors((first,), **(kwargs | {"training_components": ("C09",)}))
    with pytest.raises(ValueError, match="stream"):
        fit_dct_energy_floors((), **kwargs)
    with pytest.raises(ValueError, match="stream"):
        fit_dct_energy_floors((first, first), **kwargs)
    with pytest.raises(ValueError, match="stream"):
        fit_dct_energy_floors(
            (DctCalibrationCapture("a" * 64, "C00", synthetic_capture),), **kwargs
        )
    with pytest.raises(MemoryError, match="RESOURCE_LIMIT"):
        fit_dct_energy_floors((first,), **(kwargs | {"max_actor_patch_observations": 1}))
    with pytest.raises(MemoryError, match="working set"):
        fit_dct_energy_floors(
            (first,), **(kwargs | {"max_working_bytes": 1, "config": LocalPhaseConfig(hop_frames=10**9)})
        )
    with pytest.raises(MemoryError, match="scalar budget"):
        fit_dct_energy_floors((first,), **(kwargs | {"max_total_velocity_scalars": 1}))
    with pytest.raises(ValueError, match="patch length"):
        actor_patch_dct_power_observations(
            synthetic_capture, LocalPhaseConfig(patch_frames=1)
        )


def test_a6_cumulative_timeline_budget_rejects_second_parent_before_collection(
    synthetic_capture, monkeypatch
):
    other = replace(synthetic_capture, source_sha256="b" * 64)
    records = (
        DctCalibrationCapture("a" * 64, "C01", synthetic_capture),
        DctCalibrationCapture("b" * 64, "C02", other),
    )
    calls = 0
    original = calibration.actor_patch_dct_power_observations

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(calibration, "actor_patch_dct_power_observations", counted)
    one_parent_scalars = synthetic_capture.actor_count * synthetic_capture.frame_count * 66
    with pytest.raises(MemoryError, match="scalar budget"):
        fit_dct_energy_floors(
            records,
            training_components=("C01", "C02"),
            expected_capture_ids=("a" * 64, "b" * 64),
            config=LocalPhaseConfig(hop_frames=10**9),
            max_total_velocity_scalars=one_parent_scalars,
        )
    assert calls == 1
