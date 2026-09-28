"""Independent 20-Hz Morlet schema and physical-frequency probes."""

from __future__ import annotations

import math

import numpy as np

from phasepair_core import signal as legacy_signal
from phaseset_core import morlet
from phaseset_core.contracts import (
    ACTIVITY_SAMPLE_RATE_HZ,
    PreparedActivityBatch,
    group_commitment,
)
from phaseset_core.periodic import iter_unordered_pair_chunks


def _known_tone_pair(frequency_hz: float, delay_samples: int) -> PreparedActivityBatch:
    valid_length = 200
    padded_time = 201
    time = np.arange(valid_length, dtype=np.float64)
    left = 2.0 + np.sin(2.0 * math.pi * frequency_hz * time / 20.0)
    right = 2.0 + np.sin(
        2.0 * math.pi * frequency_hz * (time - delay_samples) / 20.0
    )
    activities = np.zeros((1, 2, padded_time, 5), dtype=np.float32)
    activities[0, 0, :valid_length] = left[:, None]
    activities[0, 1, :valid_length] = right[:, None]
    activity_mask = np.zeros_like(activities, dtype=np.bool_)
    activity_mask[:, :, :valid_length] = True
    actor_mask = np.ones((1, 2), dtype=np.bool_)
    frame_mask = np.zeros((1, padded_time), dtype=np.bool_)
    frame_mask[:, :valid_length] = True
    keys = (bytes(32), bytes([1]) * 32)
    return PreparedActivityBatch(
        activities,
        actor_mask,
        frame_mask,
        activity_mask,
        (keys,),
        (group_commitment(keys),),
    )


def test_phaseset_20hz_bank_has_independent_schema_digest_and_legacy_is_unchanged() -> None:
    bands = morlet.morlet_kernel_bank()
    assert morlet.PHASESET_MORLET_SCHEMA == "phaseset-morlet-bank-v1/fs20/three-cycle"
    assert morlet.SAMPLE_RATE_HZ == 20.0
    assert morlet.SAMPLE_RATE_HZ == ACTIVITY_SAMPLE_RATE_HZ
    assert morlet.MORLET_FREQUENCIES_HZ == legacy_signal.MORLET_FREQUENCIES_HZ
    assert tuple(band.length for band in bands) == (80, 54, 36, 24, 16, 11)
    assert morlet.morlet_bank_sha256(bands) == morlet.MORLET_ORACLE_SHA256
    assert morlet.MORLET_ORACLE_SHA256 != legacy_signal.MORLET_ORACLE_SHA256
    assert legacy_signal.morlet_bank_sha256(
        legacy_signal.morlet_kernel_bank()
    ) == "4f75ab687d333af8b436feea371dbaee56c55256e91c42c5e1ce45935beed5d7"
    assert legacy_signal.MORLET_LENGTHS == (120, 80, 54, 36, 24, 16)


def test_each_20hz_kernel_is_zero_dc_unit_energy_and_peaks_at_registered_hz() -> None:
    zero_dc_errors = []
    unit_energy_errors = []
    frequency_grid = np.linspace(0.05, 9.95, 9901, dtype=np.float64)
    for band in morlet.morlet_kernel_bank():
        zero_dc_error = float(
            np.abs(np.sum(band.kernel, dtype=np.complex128))
        )
        zero_dc_errors.append(zero_dc_error)
        assert zero_dc_error <= morlet.MORLET_ZERO_DC_BOUND
        energy = np.sum(
            np.square(np.abs(band.kernel), dtype=np.float64),
            dtype=np.float64,
        )
        unit_energy_error = float(np.abs(energy - np.float64(1.0)))
        unit_energy_errors.append(unit_energy_error)
        assert unit_energy_error <= morlet.MORLET_UNIT_ENERGY_BOUND
        samples = np.arange(band.length, dtype=np.float64)
        responses = np.abs(
            np.exp(
                1j
                * 2.0
                * math.pi
                * frequency_grid[:, None]
                * samples[None, :]
                / morlet.SAMPLE_RATE_HZ
            )
            @ band.kernel
        )
        peak_hz = float(frequency_grid[int(np.argmax(responses))])
        assert abs(peak_hz - band.frequency_hz) <= 0.001

        registered = abs(
            np.exp(
                1j
                * 2.0
                * math.pi
                * band.frequency_hz
                * samples
                / morlet.SAMPLE_RATE_HZ
            )
            @ band.kernel
        )
        silently_shifted = abs(
            np.exp(
                1j
                * 2.0
                * math.pi
                * (2.0 * band.frequency_hz / 3.0)
                * samples
                / morlet.SAMPLE_RATE_HZ
            )
            @ band.kernel
        )
        assert registered > 1.7 * silently_shifted

    assert tuple(zero_dc_errors) == morlet.MORLET_EXPECTED_ZERO_DC_ERRORS
    assert tuple(unit_energy_errors) == morlet.MORLET_EXPECTED_UNIT_ENERGY_ERRORS


def test_known_frequency_and_one_sample_delay_recover_band_phase_and_seconds() -> None:
    for band_index, frequency in enumerate(morlet.MORLET_FREQUENCIES_HZ):
        chunk = next(
            iter_unordered_pair_chunks(
                _known_tone_pair(frequency, delay_samples=1),
                energy_floors=np.zeros((6,), dtype=np.float64),
                edge_chunk_size=64,
            )
        )
        forward = chunk.tokens_ij[0]
        reverse = chunk.tokens_ji[0]
        assert int(np.argmax(forward[:, 0])) == band_index
        assert float(forward[band_index, 2]) > 0.99999

        observed_phase = math.atan2(
            float(forward[band_index, 4]),
            float(forward[band_index, 3]),
        )
        expected_phase = -2.0 * math.pi * frequency / morlet.SAMPLE_RATE_HZ
        observed_delay = float(forward[band_index, 5]) / (2.0 * frequency)
        assert abs(observed_phase - expected_phase) < 5e-6
        assert abs(observed_delay - 1.0 / morlet.SAMPLE_RATE_HZ) < 1e-6
        assert reverse[band_index, 5] == -forward[band_index, 5]
