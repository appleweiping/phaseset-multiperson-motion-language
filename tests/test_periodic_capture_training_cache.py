"""Focused analytic contracts for the train-plus-capture descriptor plan."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from phaseset_core import capture_prepared_storage as capture_storage
from phaseset_core import capture_validation, prepared_data_v2, training
from phaseset_core.periodic import ResourceLimitError
from phaseset_core.periodic_capture_training_cache import (
    CAPTURE_VALIDATION_SEMANTICS,
    PLAN_SCHEMA,
    PeriodicCaptureTrainingCacheError,
    admit_periodic_descriptor_capture_training_plan,
)
from phaseset_core.periodic_descriptor_cache_v2 import (
    CachedPairChunkStream,
    DescriptorCacheSourceBinding,
    PeriodicDescriptorCacheV2,
    PeriodicDescriptorCacheWriterV2,
)
from test_periodic_training_cache import _block, _write_cache, _write_sources
from test_phaseset_capture_prepared_storage import _frozen_text, _source


_MAX_BYTES = 32 * 1024 * 1024
_FLOORS = np.ascontiguousarray(
    np.asarray([(index + 1) * 1.0e-12 for index in range(6)], dtype=np.float64)
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _capture_binding(storage_manifest_sha256: str) -> DescriptorCacheSourceBinding:
    return DescriptorCacheSourceBinding(
        prepared_manifest_sha256=storage_manifest_sha256,
        source_tree_sha256=_digest("capture/source-tree"),
        environment_sha256=_digest("capture/environment"),
    )


def _write_capture_source(root: Path) -> capture_validation.CaptureValidationSource:
    result = capture_storage.write_capture_validation_source_v2(root, _source())
    return capture_storage.load_capture_validation_source(
        result.manifest_path,
        expected_manifest_sha256=result.manifest_sha256,
    )


def _write_capture_cache(
    root: Path,
    source: capture_validation.CaptureValidationSource,
    *,
    seeds: tuple[int, ...] = (1729,),
    window_limit: int | None = None,
) -> PeriodicDescriptorCacheV2:
    descriptors = [
        window.descriptor_source for capture in source.captures for window in capture.windows
    ]
    assert descriptors and all(
        type(value) is capture_validation.CaptureDescriptorWindowSource for value in descriptors
    )
    storage_manifest = descriptors[0].storage_manifest_sha256
    binding = _capture_binding(storage_manifest)
    writer = PeriodicDescriptorCacheWriterV2(
        root,
        source_binding=binding,
        energy_floors=_FLOORS,
    )
    written = 0
    for seed in seeds:
        for capture in source.captures:
            for window in capture.windows:
                if window_limit is not None and written >= window_limit:
                    break
                writer.add_batch(window.groups, (window.descriptor_context(seed=seed),))
                written += 1
            if window_limit is not None and written >= window_limit:
                break
        if window_limit is not None and written >= window_limit:
            break
    artifact = writer.finalize()
    return PeriodicDescriptorCacheV2(
        root,
        expected_index_sha256=artifact.index_sha256,
        expected_source_binding=binding,
        energy_floors=_FLOORS,
    )


@dataclass(frozen=True)
class _Bundle:
    train_source: prepared_data_v2.PreparedTrainingDataSourceV2
    capture_source: capture_validation.CaptureValidationSource
    train_cache: PeriodicDescriptorCacheV2
    capture_cache: PeriodicDescriptorCacheV2
    config: training.TrainingConfig


def _bundle(tmp_path: Path, *, seed: int = 1729) -> _Bundle:
    train_source, _unused_window_val = _write_sources(tmp_path / "prepared")
    capture_source = _write_capture_source(tmp_path / "capture-source")
    train_cache = _write_cache(
        tmp_path / "train-cache",
        train_source,
        seed=seed,
        epochs=tuple(range(training.RESIDUAL_EPOCHS)),
        floors=_FLOORS,
    )
    capture_cache = _write_capture_cache(
        tmp_path / "capture-cache",
        capture_source,
        seeds=(seed,),
    )
    return _Bundle(
        train_source,
        capture_source,
        train_cache,
        capture_cache,
        training.TrainingConfig(stage="residual", seed=seed),
    )


def _admit(bundle: _Bundle, *, system_id: str = "08"):
    return admit_periodic_descriptor_capture_training_plan(
        system_id=system_id,
        config=bundle.config,
        train_source=bundle.train_source,
        capture_validation_source=bundle.capture_source,
        train_cache=bundle.train_cache,
        capture_validation_cache=bundle.capture_cache,
    )


def _rng_snapshot():
    state = np.random.get_state()
    return (
        random.getstate(),
        (state[0], np.array(state[1], copy=True), *state[2:]),
        torch.random.get_rng_state().clone(),
    )


def _assert_rng_equal(before) -> None:
    python_before, numpy_before, torch_before = before
    current = np.random.get_state()
    assert random.getstate() == python_before
    assert current[0] == numpy_before[0]
    assert np.array_equal(current[1], numpy_before[1])
    assert current[2:] == numpy_before[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_before)


def test_complete_plan_binds_twenty_epoch_train_and_every_capture_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    original = PeriodicDescriptorCacheV2.open_batch
    calls = 0

    def counted(self, batch, contexts):
        nonlocal calls
        calls += 1
        return original(self, batch, contexts)

    monkeypatch.setattr(PeriodicDescriptorCacheV2, "open_batch", counted)
    before = _rng_snapshot()
    cuda_initialized_before = torch.cuda.is_initialized()
    plan = _admit(bundle, system_id="06")
    _assert_rng_equal(before)
    assert torch.cuda.is_initialized() is cuda_initialized_before
    window_count = sum(len(capture.windows) for capture in bundle.capture_source.captures)
    caption_count = sum(
        len(capture.holistic_text.caption_commitments) for capture in bundle.capture_source.captures
    )
    assert calls == training.RESIDUAL_EPOCHS + window_count
    assert plan.authority == 0
    assert plan.production is False
    assert plan.result_claimed is False
    assert plan.stream_kind == "FULL_RELATION"
    assert plan.capture_count == len(bundle.capture_source.captures)
    assert plan.window_count == window_count
    assert plan.caption_count == caption_count
    assert plan.capture_source_manifest_sha256 == bundle.capture_source.manifest_sha256
    assert plan.capture_source_census_sha256 == bundle.capture_source.census_sha256
    assert (
        plan.capture_storage_manifest_sha256
        == bundle.capture_source.captures[0].windows[0].descriptor_source.storage_manifest_sha256
    )
    assert plan.capture_storage_manifest_sha256 != plan.capture_source_manifest_sha256
    assert len(plan.train_rows) == training.RESIDUAL_EPOCHS
    assert {row.epoch for row in plan.train_rows} == set(range(training.RESIDUAL_EPOCHS))
    assert len(plan.capture_rows) == window_count
    assert tuple(row.execution_position for row in plan.capture_rows) == tuple(
        range(training.RESIDUAL_EPOCHS, training.RESIDUAL_EPOCHS + window_count)
    )
    assert tuple(row.window_ordinal for row in plan.capture_rows) == tuple(range(window_count))
    assert tuple(sorted(row.cache_key_sha256 for row in plan.capture_rows)) == (
        bundle.capture_cache.cache_key_census
    )
    expected = tuple(
        (
            capture.capture_commitment.hex(),
            capture.plan.sha256,
            window_position,
            capture.plan.source_start_frames[window_position],
            window.window_commitment.hex(),
            window.descriptor_source.source_window_npz_sha256,
            window.descriptor_source.window_ordinal,
        )
        for capture in bundle.capture_source.captures
        for window_position, window in enumerate(capture.windows)
    )
    observed = tuple(
        (
            row.capture_commitment,
            row.capture_plan_sha256,
            row.window_position,
            row.source_start_frame,
            row.window_commitment,
            row.source_window_npz_sha256,
            row.window_ordinal,
        )
        for row in plan.capture_rows
    )
    assert observed == expected
    raw = plan.canonical_bytes()
    assert raw.endswith(b"\n")
    assert PLAN_SCHEMA.encode("ascii") in raw
    assert CAPTURE_VALIDATION_SEMANTICS.encode("ascii") in raw
    assert len(plan.sha256) == 64

    checked = plan.validate_capture_validation_source(bundle.capture_source)
    assert checked.census_sha256 == bundle.capture_source.census_sha256
    assert calls == training.RESIDUAL_EPOCHS + window_count
    train_batch = next(bundle.train_source.iter_epoch(epoch=7, seed=1729))
    assert type(plan.open_training_batch(train_batch)) is CachedPairChunkStream
    for capture in checked.captures:
        for window_position in range(len(capture.windows)):
            stream = plan.open_capture_validation_window(capture, window_position)
            assert type(stream) is CachedPairChunkStream
            assert stream.stream_kind == "FULL_RELATION"
    assert calls == training.RESIDUAL_EPOCHS + 2 * window_count + 1
    _assert_rng_equal(before)
    assert torch.cuda.is_initialized() is cuda_initialized_before


def test_registered_capture_plan_stream_mapping(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    expected = {
        "02": "MARGINAL_POWER",
        "03": "MEAN_DIFFERENCE_DCT",
        "04": "FULL_RELATION",
        "06": "FULL_RELATION",
        "07": "FULL_RELATION",
        "08": "FULL_RELATION",
    }
    for system_id, stream_kind in expected.items():
        assert _admit(bundle, system_id=system_id).stream_kind == stream_kind


def test_full_scientific_source_census_including_text_is_rechecked(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    plan = _admit(bundle)
    first = bundle.capture_source.captures[0]
    replacement_axes = tuple(
        17 + index for index in range(len(first.holistic_text.caption_commitments))
    )
    changed_capture = replace(
        first,
        holistic_text=_frozen_text("changed-holistic-gallery", replacement_axes),
    )
    changed = capture_validation.CaptureValidationSource(
        "val",
        bundle.capture_source.manifest_sha256,
        (changed_capture, *bundle.capture_source.captures[1:]),
    )
    assert changed.captures[0].plan == bundle.capture_source.captures[0].plan
    assert tuple(window.window_commitment for window in changed.captures[0].windows) == tuple(
        window.window_commitment for window in bundle.capture_source.captures[0].windows
    )
    assert (
        sum(len(capture.holistic_text.caption_commitments) for capture in changed.captures)
        == plan.caption_count
    )
    assert changed.census_sha256 != bundle.capture_source.census_sha256
    with pytest.raises(
        PeriodicCaptureTrainingCacheError,
        match="complete admitted census",
    ):
        plan.validate_capture_validation_source(changed)

    changed_manifest = capture_validation.CaptureValidationSource(
        "val",
        _digest("changed-upstream-manifest"),
        bundle.capture_source.captures,
    )
    with pytest.raises(
        PeriodicCaptureTrainingCacheError,
        match="complete admitted census",
    ):
        plan.validate_capture_validation_source(changed_manifest)


def test_legacy_capture_source_and_prepared_window_val_never_enter_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    calls = 0
    original = PeriodicDescriptorCacheV2.open_batch

    def counted(self, batch, contexts):
        nonlocal calls
        calls += 1
        return original(self, batch, contexts)

    monkeypatch.setattr(PeriodicDescriptorCacheV2, "open_batch", counted)
    with pytest.raises(
        PeriodicCaptureTrainingCacheError,
        match="v2 descriptor lineage",
    ):
        admit_periodic_descriptor_capture_training_plan(
            system_id="08",
            config=bundle.config,
            train_source=bundle.train_source,
            capture_validation_source=_source(),
            train_cache=bundle.train_cache,
            capture_validation_cache=bundle.capture_cache,
        )
    assert calls == 0

    _train, prepared_window_val = _write_sources(tmp_path / "other-prepared")
    with pytest.raises(TypeError, match="exact CaptureValidationSource"):
        admit_periodic_descriptor_capture_training_plan(
            system_id="08",
            config=bundle.config,
            train_source=bundle.train_source,
            capture_validation_source=prepared_window_val,
            train_cache=bundle.train_cache,
            capture_validation_cache=bundle.capture_cache,
        )
    assert calls == 0


def test_capture_selector_rejects_stale_npz_lineage_before_reader_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    plan = _admit(bundle)
    first = bundle.capture_source.captures[0]
    window = first.windows[0]
    descriptor = window.descriptor_source
    assert type(descriptor) is capture_validation.CaptureDescriptorWindowSource
    changed_descriptor = replace(descriptor, source_window_npz_sha256=_digest("stale-npz"))
    changed_window = replace(window, descriptor_source=changed_descriptor)
    changed_capture = replace(first, windows=(changed_window, *first.windows[1:]))
    calls = 0

    def forbidden(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("stale selector must not open a shard")

    monkeypatch.setattr(PeriodicDescriptorCacheV2, "open_batch", forbidden)
    with pytest.raises(PeriodicCaptureTrainingCacheError, match="not an exact admitted"):
        plan.open_capture_validation_window(changed_capture, 0)
    assert calls == 0


def test_uncached_and_base_systems_reject_before_any_shard_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    calls = 0

    def forbidden(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("cache must not open")

    monkeypatch.setattr(PeriodicDescriptorCacheV2, "open_batch", forbidden)
    for system_id in ("01", "05", "B0", "00"):
        with pytest.raises(PeriodicCaptureTrainingCacheError):
            _admit(bundle, system_id=system_id)
    assert calls == 0


@pytest.mark.parametrize("mode", ["missing", "extra"])
def test_capture_reader_census_must_equal_all_capture_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    bundle = _bundle(tmp_path)
    cache = _write_capture_cache(
        tmp_path / f"capture-cache-{mode}",
        bundle.capture_source,
        seeds=(1729,) if mode == "missing" else (1729, 2718),
        window_limit=2 if mode == "missing" else None,
    )
    original = PeriodicDescriptorCacheV2.open_batch
    capture_opens = 0

    def counted(self, batch, contexts):
        nonlocal capture_opens
        if self is cache:
            capture_opens += 1
        return original(self, batch, contexts)

    monkeypatch.setattr(PeriodicDescriptorCacheV2, "open_batch", counted)
    expected = "exceeds its closed cache census" if mode == "missing" else "required key set"
    with pytest.raises(PeriodicCaptureTrainingCacheError, match=expected):
        admit_periodic_descriptor_capture_training_plan(
            system_id="08",
            config=bundle.config,
            train_source=bundle.train_source,
            capture_validation_source=bundle.capture_source,
            train_cache=bundle.train_cache,
            capture_validation_cache=cache,
        )
    all_windows = sum(len(capture.windows) for capture in bundle.capture_source.captures)
    assert capture_opens == (2 if mode == "missing" else all_windows)


def _two_actor_train_source(
    root: Path,
) -> prepared_data_v2.PreparedTrainingDataSourceV2:
    result = prepared_data_v2.write_prepared_training_tree_v2(
        root,
        train_blocks=(_block("budget/train", actors=2, ordinal=10),),
        val_blocks=(_block("budget/val", actors=2, ordinal=20),),
        max_total_edges=1,
        max_batch_size=1,
        max_batch_bytes=_MAX_BYTES,
        max_decoded_batch_bytes=_MAX_BYTES,
    )
    train_source, _val_source = prepared_data_v2.load_prepared_training_sources_v2(
        result.index_path,
        expected_index_sha256=result.index_sha256,
        max_batch_bytes=_MAX_BYTES,
        max_decoded_batch_bytes=_MAX_BYTES,
    )
    return train_source


def test_capture_edge_budget_fails_before_capture_reader_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_source = _two_actor_train_source(tmp_path / "budget-prepared")
    train_cache = _write_cache(
        tmp_path / "budget-train-cache",
        train_source,
        seed=1729,
        epochs=tuple(range(training.RESIDUAL_EPOCHS)),
        floors=_FLOORS,
    )
    capture_source = _write_capture_source(tmp_path / "budget-capture-source")
    capture_cache = _write_capture_cache(
        tmp_path / "budget-capture-cache",
        capture_source,
    )
    capture_opens = 0
    original = PeriodicDescriptorCacheV2.open_batch

    def counted(self, batch, contexts):
        nonlocal capture_opens
        if self is capture_cache:
            capture_opens += 1
        return original(self, batch, contexts)

    monkeypatch.setattr(PeriodicDescriptorCacheV2, "open_batch", counted)
    with pytest.raises(ResourceLimitError):
        admit_periodic_descriptor_capture_training_plan(
            system_id="08",
            config=training.TrainingConfig(stage="residual", seed=1729, edge_budget=1),
            train_source=train_source,
            capture_validation_source=capture_source,
            train_cache=train_cache,
            capture_validation_cache=capture_cache,
        )
    assert capture_opens == 0
