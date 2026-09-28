from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from phaseset_core.continuous_capture import prepare_continuous_capture
from phaseset_core.continuous_phase_cache import (
    load_continuous_phase_cache,
    write_continuous_phase_cache,
)
from phaseset_core.directional_calibration import (
    DirectionalCalibrationCapture,
    fit_directional_energy_floors,
)
from phaseset_core.directional_phase import (
    LocalPhaseConfig,
    continuous_directional_phase_field,
    local_pair_chunk,
)
from phaseset_core.pipeline import PrivateCaptureArrays
from phaseset_core.temporal_coordination import TemporalIncidenceEncoder


def _capture():
    frames, actors = 600, 3
    joints = np.zeros((frames, actors, 22, 3), dtype=np.float32)
    for actor in range(actors):
        joints[:, actor, :, 1] = actor
        joints[:, actor, 1:, 0] = np.sin(2 * np.pi * 1.125 * np.arange(frames) / 30 + actor * 0.3)[
            :, None
        ]
    return prepare_continuous_capture(
        PrivateCaptureArrays(
            joints,
            np.ones(joints.shape[:-1], dtype=np.bool_),
            tuple(bytes([actor + 1]) * 32 for actor in range(actors)),
            "a" * 64,
        )
    )


@pytest.mark.parametrize("chunk", [1, 31, 1024])
def test_memory_maps_are_bitwise_equal_to_full_halo_ram_backend(chunk, tmp_path):
    capture = _capture()
    floors = np.zeros(6, dtype=np.float64)
    ram = continuous_directional_phase_field(capture, energy_floors=floors)
    cache = tmp_path / "physical"
    write_continuous_phase_cache(
        capture,
        cache,
        config=LocalPhaseConfig(convolution_chunk_frames=chunk, max_response_bytes=1),
    )
    disk = load_continuous_phase_cache(
        cache, expected_source_sha256=capture.source_sha256, energy_floors=floors
    )
    for reference, actual in zip(ram.responses, disk.responses, strict=True):
        assert isinstance(actual, np.memmap) and not actual.flags.writeable
        np.testing.assert_array_equal(reference.view(np.uint64), actual.view(np.uint64))
    for name in ("response_masks", "response_centers"):
        for reference, actual in zip(getattr(ram, name), getattr(disk, name), strict=True):
            np.testing.assert_array_equal(reference, actual)
    assert ram.intervals == disk.intervals
    for name in ("actor_features", "actor_patch_mask", "root_positions", "root_patch_mask"):
        np.testing.assert_array_equal(getattr(ram, name), getattr(disk, name))
    np.testing.assert_array_equal(
        local_pair_chunk(ram, 0, 3).features, local_pair_chunk(disk, 0, 3).features
    )


def test_calibration_reuses_cache_and_applies_floors_without_reconvolution(tmp_path):
    capture = _capture()
    raw = write_continuous_phase_cache(capture, tmp_path / "physical")
    assert not raw.energy_floors.flags.writeable
    result = fit_directional_energy_floors(
        (DirectionalCalibrationCapture("capture", "C01", raw),),
        training_components=("C01",),
        expected_capture_ids=("capture",),
    )
    disk = load_continuous_phase_cache(
        tmp_path / "physical",
        expected_source_sha256=capture.source_sha256,
        energy_floors=result.energy_floors,
    )
    np.testing.assert_array_equal(disk.energy_floors, result.energy_floors)
    assert not disk.energy_floors.flags.writeable
    expected = replace(raw, energy_floors=result.energy_floors)
    np.testing.assert_array_equal(
        local_pair_chunk(expected, 0, 3).features, local_pair_chunk(disk, 0, 3).features
    )


def test_cached_encoder_forward_backward_matches_ram_source(tmp_path):
    capture = _capture()
    raw = continuous_directional_phase_field(capture, energy_floors=np.zeros(6, dtype=np.float64))
    write_continuous_phase_cache(capture, tmp_path / "physical")
    disk = load_continuous_phase_cache(
        tmp_path / "physical",
        expected_source_sha256=capture.source_sha256,
        energy_floors=np.zeros(6, dtype=np.float64),
    )
    torch.manual_seed(1729)
    model = TemporalIncidenceEncoder(width=8)
    results, gradients = [], []
    for source in (raw, disk):
        model.zero_grad(set_to_none=True)
        output = model(source)
        results.append(output.embedding.detach().clone())
        output.embedding[0].backward()
        gradients.append(
            tuple(
                None if parameter.grad is None else parameter.grad.detach().clone()
                for parameter in model.parameters()
            )
        )
    assert torch.equal(*results)
    for first, second in zip(*gradients, strict=True):
        assert (first is None and second is None) or torch.equal(first, second)


def test_limits_and_existing_attempts_do_not_overwrite_data(tmp_path):
    capture = _capture()
    with pytest.raises(MemoryError, match="RESOURCE_LIMIT"):
        write_continuous_phase_cache(capture, tmp_path / "limited", max_disk_bytes=1)
    assert not (tmp_path / "limited").exists()
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("user")
    with pytest.raises(FileExistsError):
        write_continuous_phase_cache(capture, output)
    assert (output / "keep.txt").read_text() == "user"


def test_incomplete_lineage_and_wrong_array_headers_are_rejected(tmp_path):
    capture = _capture()
    root = tmp_path / "physical"
    write_continuous_phase_cache(capture, root)
    with pytest.raises(ValueError, match="lineage"):
        load_continuous_phase_cache(
            root, expected_source_sha256="b" * 64, energy_floors=np.zeros(6, dtype=np.float64)
        )
    metadata = json.loads((root / "metadata.json").read_text())
    metadata["actor_count"] += 1
    (root / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="header differs"):
        load_continuous_phase_cache(
            root,
            expected_source_sha256=capture.source_sha256,
            energy_floors=np.zeros(6, dtype=np.float64),
        )
