"""Server-only contracts for cached holistic capture validation."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from phaseset_core import capture_validation, frozen_clip_text, training
from phaseset_core.periodic_capture_training_cache import (
    PeriodicCaptureTrainingCacheError,
    PeriodicDescriptorCaptureTrainingPlan,
)
from phaseset_core.periodic_descriptor_cache_v2 import (
    DescriptorCacheV2Error,
    PeriodicDescriptorCacheV2,
)
from test_periodic_capture_training_cache import (
    _FLOORS,
    _admit,
    _assert_rng_equal,
    _bundle,
    _rng_snapshot,
)


def _system(system_id: str = "08") -> training.ResidualRetrievalSystem:
    with torch.random.fork_rng():
        torch.manual_seed(1729)
        frozen_base = training.build_registered_base_training_system("B0").group_base
        return training.build_registered_residual_training_system(
            system_id,
            frozen_base,
            seed=1729,
            energy_floors=_FLOORS,
            edge_budget=32_768,
        )


def _assert_same_result(left, right) -> None:
    assert np.array_equal(left.dataset.scores, right.dataset.scores)
    assert left.dataset.motion_commitments == right.dataset.motion_commitments
    assert left.dataset.caption_commitments == right.dataset.caption_commitments
    assert left.dataset.positive_motion_indices == right.dataset.positive_motion_indices
    assert np.array_equal(left.dataset.group_sizes, right.dataset.group_sizes)
    assert left.dataset.component_labels == right.dataset.component_labels
    assert left.text_to_motion_capture_r1 == right.text_to_motion_capture_r1
    assert left.motion_to_text_capture_r1 == right.motion_to_text_capture_r1
    assert left.primary_capture_r1 == right.primary_capture_r1
    assert left.loss == right.loss
    assert left.source_census_sha256 == right.source_census_sha256
    assert left.scores_float64_sha256 == right.scores_float64_sha256
    assert left.encoded_bytes == right.encoded_bytes


def test_reader_backed_cached_runner_matches_uncached_full_capture_gallery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    plan = _admit(bundle, system_id="08")
    uncached_system = _system().train()
    cached_system = copy.deepcopy(uncached_system)
    uncached_modes = tuple(module.training for module in uncached_system.modules())
    cached_modes = tuple(module.training for module in cached_system.modules())
    states = {key: value.detach().clone() for key, value in cached_system.state_dict().items()}
    original_open = PeriodicDescriptorCacheV2.open_batch
    capture_opens = 0

    def counted_open(self, groups, contexts):
        nonlocal capture_opens
        if self is bundle.capture_cache:
            capture_opens += 1
        return original_open(self, groups, contexts)

    def forbidden_text_encode(*_args, **_kwargs):
        raise AssertionError("persisted holistic text must not be re-encoded")

    monkeypatch.setattr(PeriodicDescriptorCacheV2, "open_batch", counted_open)
    monkeypatch.setattr(
        frozen_clip_text.FrozenClipTextAdapter,
        "encode",
        forbidden_text_encode,
    )
    before = _rng_snapshot()
    uncached = capture_validation.run_capture_validation(
        uncached_system,
        bundle.config,
        bundle.capture_source,
    )
    cached = capture_validation.run_capture_validation_cached(
        cached_system,
        bundle.config,
        bundle.capture_source,
        plan,
    )
    _assert_rng_equal(before)
    _assert_same_result(uncached, cached)
    assert capture_opens == sum(len(capture.windows) for capture in bundle.capture_source.captures)
    assert uncached_modes == tuple(module.training for module in uncached_system.modules())
    assert cached_modes == tuple(module.training for module in cached_system.modules())
    assert all(torch.equal(value, cached_system.state_dict()[key]) for key, value in states.items())
    assert all(parameter.grad is None for parameter in cached_system.parameters())


def test_complete_source_drift_and_missing_shard_fail_without_scores_or_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    plan = _admit(bundle)
    system = _system()
    calls = {"cached": 0, "uncached": 0, "scores": 0}
    original_cached = training.ResidualRetrievalSystem.encode_trainable_cached
    original_scores = training.ResidualRetrievalSystem.scores

    def counted_cached(self, groups, stream):
        calls["cached"] += 1
        return original_cached(self, groups, stream)

    def forbidden_uncached(*_args, **_kwargs):
        calls["uncached"] += 1
        raise AssertionError("cached runner must not fall back")

    def counted_scores(self, *args, **kwargs):
        calls["scores"] += 1
        return original_scores(self, *args, **kwargs)

    monkeypatch.setattr(
        training.ResidualRetrievalSystem,
        "encode_trainable_cached",
        counted_cached,
    )
    monkeypatch.setattr(
        training.ResidualRetrievalSystem,
        "encode_trainable",
        forbidden_uncached,
    )
    monkeypatch.setattr(training.ResidualRetrievalSystem, "scores", counted_scores)
    changed = replace(bundle.capture_source, manifest_sha256="f" * 64)
    with pytest.raises(PeriodicCaptureTrainingCacheError, match="complete admitted census"):
        capture_validation.run_capture_validation_cached(
            system,
            bundle.config,
            changed,
            plan,
        )
    assert calls == {"cached": 0, "uncached": 0, "scores": 0}

    first_key = plan.capture_rows[0].cache_key_sha256
    shard = bundle.capture_cache.root / "shards" / f"{first_key}.pdc2"
    # The fixture deliberately removes an immutable shard; Windows requires
    # clearing its read-only attribute before the missing-file fault injection.
    shard.chmod(0o600)
    shard.unlink()
    with pytest.raises(DescriptorCacheV2Error):
        capture_validation.run_capture_validation_cached(
            system,
            bundle.config,
            bundle.capture_source,
            plan,
        )
    assert calls == {"cached": 0, "uncached": 0, "scores": 0}


def test_wrong_system_or_descriptor_family_is_rejected_without_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    plan = _admit(bundle, system_id="08")
    system = _system("08")
    other = _system("06")
    with pytest.raises(capture_validation.CaptureValidationError, match="differs"):
        capture_validation.run_capture_validation_cached(
            other,
            bundle.config,
            bundle.capture_source,
            plan,
        )

    original_open = PeriodicDescriptorCaptureTrainingPlan.open_capture_validation_window
    uncached_calls = 0

    def wrong_family(self, capture, window_position):
        stream = original_open(self, capture, window_position)
        return stream.batch.stream("MARGINAL_POWER")

    def forbidden_uncached(*_args, **_kwargs):
        nonlocal uncached_calls
        uncached_calls += 1
        raise AssertionError("wrong cached family must not trigger uncached fallback")

    monkeypatch.setattr(
        PeriodicDescriptorCaptureTrainingPlan,
        "open_capture_validation_window",
        wrong_family,
    )
    monkeypatch.setattr(
        training.ResidualRetrievalSystem,
        "encode_trainable",
        forbidden_uncached,
    )
    with pytest.raises((DescriptorCacheV2Error, RuntimeError, ValueError)):
        capture_validation.run_capture_validation_cached(
            system,
            bundle.config,
            bundle.capture_source,
            plan,
        )
    assert uncached_calls == 0


def test_public_uncached_api_and_runtime_binding_order_remain_compatible() -> None:
    assert capture_validation.__all__[-2:] == [
        "run_capture_validation",
        "run_capture_validation_cached",
    ]
    source_type, public_runner = capture_validation._verified_capture_validation_bindings()
    assert source_type is capture_validation.CaptureValidationSource
    assert public_runner is capture_validation.run_capture_validation
