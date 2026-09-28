"""Focused tests for the private periodic training-cache plan candidate.

All inputs are analytic fixtures.  These tests do not load CLIP, construct a
training model, run optimization, or claim a formal dataset/cache result.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import random
import struct

import numpy as np
import pytest
import torch

from phaseset_core import frozen_clip_text as clip
from phaseset_core import prepared_data_v2 as prepared
from phaseset_core import training
from phaseset_core.contracts import PreparedGroupBatch, group_commitment
from phaseset_core.periodic import ResourceLimitError
from phaseset_core.periodic_descriptor_cache_v2 import (
    CachedPairChunkStream,
    DescriptorCacheSourceBinding,
    DescriptorWindowContext,
    PeriodicDescriptorCacheV2,
    PeriodicDescriptorCacheWriterV2,
)
from phaseset_core.periodic_training_cache import (
    PLAN_SCHEMA,
    PeriodicTrainingCacheError,
    admit_periodic_descriptor_training_plan,
)
from phaseset_core.pipeline import PreparedGroupSample, deterministic_group_yaw
from phaseset_core.preprocessing import WindowDecision


_MAX_BYTES = 32 * 1024 * 1024
_FLOORS = np.ascontiguousarray(
    np.asarray([(index + 1) * 1.0e-12 for index in range(6)], dtype=np.float64)
)


def _digest(label: str) -> bytes:
    return hashlib.sha256(label.encode("ascii")).digest()


def _sample(label: str, *, actors: int) -> PreparedGroupSample:
    commitments = tuple(sorted(_digest(f"{label}/actor/{index}") for index in range(actors)))
    skeletons = np.zeros((actors, 200, 22, 3), dtype=np.float32)
    frame = np.arange(200, dtype=np.float32)
    for actor in range(actors):
        for joint in range(22):
            skeletons[actor, :, joint, 0] = frame * np.float32(0.001) + actor
            skeletons[actor, :, joint, 1] = np.float32(joint * 0.002)
            skeletons[actor, :, joint, 2] = frame * np.float32(-0.003) + joint
    return PreparedGroupSample(
        skeletons=np.ascontiguousarray(skeletons),
        track_mask=np.ones((actors, 200, 22), dtype=np.bool_),
        actor_commitments=commitments,
        group_commitment=group_commitment(commitments),
        source_sha256=hashlib.sha256(f"{label}/source".encode("ascii")).hexdigest(),
        window_sha256=hashlib.sha256(f"{label}/window".encode("ascii")).hexdigest(),
        source_start_frame=0,
        augmentation_yaw=0.0,
        decision=WindowDecision(True, "ACCEPT", 300, actors, 0, 1.0, 0, False),
    )


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().contiguous().view(torch.uint8).numpy().tobytes()


def _text_batch(label: str) -> clip.FrozenClipTextBatch:
    values = torch.arange(512, dtype=torch.float32).reshape(1, 512)
    embeddings = (torch.sin(values * 0.001) + len(label)).contiguous()
    commitment = _digest(f"{label}/caption")
    output = _tensor_bytes(embeddings)
    receipt = clip.FrozenClipTextReceipt(
        batch_size=1,
        caption_rows=(
            (
                0,
                commitment.hex(),
                hashlib.sha256(f"{label}/text".encode("ascii")).hexdigest(),
                3,
                3,
                False,
                hashlib.sha256(f"{label}/tokens".encode("ascii")).hexdigest(),
                hashlib.sha256(f"{label}/mask".encode("ascii")).hexdigest(),
                hashlib.sha256(output).hexdigest(),
            ),
        ),
        chunk_ranges=((0, 1),),
        frozen_embedding_cache_key_sha256="1" * 64,
        live_model_manifest_sha256="2" * 64,
        output_bytes=len(output),
        output_sha256=hashlib.sha256(output).hexdigest(),
        output_shape=tuple(embeddings.shape),
        output_stride=tuple(embeddings.stride()),
        runtime_identity=(("fixture", "server-only"),),
        runtime_manifest_sha256="3" * 64,
        snapshot_files=(),
        snapshot_manifest_sha256="4" * 64,
        source_files=(),
        source_manifest_sha256="5" * 64,
    )
    return clip.FrozenClipTextBatch(
        embeddings,
        (commitment,),
        receipt,
        _seal=clip._CONSTRUCTION_SEAL,
    )


def _block(label: str, *, actors: int, ordinal: int) -> prepared.PreparedMotionTextBlock:
    return prepared.PreparedMotionTextBlock(
        samples=(_sample(label, actors=actors),),
        text_batch=_text_batch(label),
        text_counts=(1,),
        window_ordinals=(ordinal,),
    )


def _write_sources(root: Path, label: str = "canonical"):
    root.parent.mkdir(parents=True, exist_ok=True)
    result = prepared.write_prepared_training_tree_v2(
        root,
        train_blocks=(_block(f"{label}/train", actors=3, ordinal=10),),
        val_blocks=(_block(f"{label}/val", actors=2, ordinal=20),),
        max_total_edges=3,
        max_batch_size=1,
        max_batch_bytes=_MAX_BYTES,
        max_decoded_batch_bytes=_MAX_BYTES,
    )
    train_source, val_source = prepared.load_prepared_training_sources_v2(
        result.index_path,
        expected_index_sha256=result.index_sha256,
        max_batch_bytes=_MAX_BYTES,
        max_decoded_batch_bytes=_MAX_BYTES,
    )
    return train_source, val_source


def _binding(source: prepared.PreparedTrainingDataSourceV2, label: str):
    return DescriptorCacheSourceBinding(
        prepared_manifest_sha256=source.manifest_sha256,
        source_tree_sha256=hashlib.sha256(f"{label}/tree".encode("ascii")).hexdigest(),
        environment_sha256=hashlib.sha256(f"{label}/env".encode("ascii")).hexdigest(),
    )


def _write_cache(
    root: Path,
    source: prepared.PreparedTrainingDataSourceV2,
    *,
    seed: int,
    epochs: tuple[int, ...],
    floors: np.ndarray = _FLOORS,
    extras: tuple[tuple[int, int], ...] = (),
) -> PeriodicDescriptorCacheV2:
    binding = _binding(source, source.split)
    writer = PeriodicDescriptorCacheWriterV2(
        root,
        source_binding=binding,
        energy_floors=floors,
    )
    for epoch in epochs:
        for batch in source.iter_epoch(epoch=epoch, seed=seed):
            assert batch.descriptor_contexts is not None
            writer.add_batch(batch.groups, batch.descriptor_contexts)
    for extra_seed, extra_epoch in extras:
        for batch in source.iter_epoch(epoch=extra_epoch, seed=extra_seed):
            assert batch.descriptor_contexts is not None
            writer.add_batch(batch.groups, batch.descriptor_contexts)
    artifact = writer.finalize()
    return PeriodicDescriptorCacheV2(
        root,
        expected_index_sha256=artifact.index_sha256,
        expected_source_binding=binding,
        energy_floors=floors,
    )


@dataclass(frozen=True)
class _Bundle:
    train_source: prepared.PreparedTrainingDataSourceV2
    val_source: prepared.PreparedTrainingDataSourceV2
    train_cache: PeriodicDescriptorCacheV2
    val_cache: PeriodicDescriptorCacheV2
    config: training.TrainingConfig


def _bundle(tmp_path: Path, *, seed: int = 1729) -> _Bundle:
    train_source, val_source = _write_sources(tmp_path / "prepared")
    train_cache = _write_cache(
        tmp_path / "train-cache",
        train_source,
        seed=seed,
        epochs=tuple(range(training.RESIDUAL_EPOCHS)),
    )
    val_cache = _write_cache(
        tmp_path / "val-cache",
        val_source,
        seed=seed,
        epochs=(0,),
    )
    return _Bundle(
        train_source=train_source,
        val_source=val_source,
        train_cache=train_cache,
        val_cache=val_cache,
        config=training.TrainingConfig(stage="residual", seed=seed),
    )


def _admit(bundle: _Bundle, *, system_id: str = "08"):
    return admit_periodic_descriptor_training_plan(
        system_id=system_id,
        config=bundle.config,
        train_source=bundle.train_source,
        val_source=bundle.val_source,
        train_cache=bundle.train_cache,
        val_cache=bundle.val_cache,
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


def test_complete_plan_is_seed_specific_exact_and_reopens_each_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    plan = _admit(bundle, system_id="06")
    _assert_rng_equal(before)
    assert calls == 21
    assert plan.authority == 0
    assert plan.production is False
    assert plan.result_claimed is False
    assert plan.canonical_bytes().endswith(b"\n")
    assert PLAN_SCHEMA.encode("ascii") in plan.canonical_bytes()
    assert len(plan.sha256) == 64
    assert plan.stream_kind == "FULL_RELATION"
    assert len(plan.train_rows) == 20
    assert {row.epoch for row in plan.train_rows} == set(range(20))
    assert len(plan.validation_rows) == 1
    assert plan.validation_rows[0].epoch == 0
    assert tuple(row.execution_position for row in plan.train_rows) == tuple(range(20))
    assert plan.validation_rows[0].execution_position == 20
    assert tuple(sorted(row.cache_key_sha256 for row in plan.train_rows)) == (
        bundle.train_cache.cache_key_census
    )

    train_batch = next(bundle.train_source.iter_epoch(epoch=7, seed=1729))
    first = plan.open_training_batch(train_batch)
    second = plan.open_training_batch(train_batch)
    val_batch = next(bundle.val_source.iter_epoch(epoch=0, seed=1729))
    validation = plan.open_validation_batch(val_batch)
    assert calls == 24
    assert type(first) is CachedPairChunkStream
    assert first.stream_kind == second.stream_kind == validation.stream_kind == "FULL_RELATION"


def test_registered_system_stream_mapping_and_uncached_system_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    opened = 0
    original = PeriodicDescriptorCacheV2.open_batch

    def counted(self, batch, contexts):
        nonlocal opened
        opened += 1
        return original(self, batch, contexts)

    monkeypatch.setattr(PeriodicDescriptorCacheV2, "open_batch", counted)
    for system_id in ("01", "05", "B0", "00"):
        with pytest.raises(PeriodicTrainingCacheError):
            _admit(bundle, system_id=system_id)
    base_config = replace(bundle.config, stage="base")
    with pytest.raises(PeriodicTrainingCacheError):
        admit_periodic_descriptor_training_plan(
            system_id="08",
            config=base_config,
            train_source=bundle.train_source,
            val_source=bundle.val_source,
            train_cache=bundle.train_cache,
            val_cache=bundle.val_cache,
        )
    assert opened == 0


@pytest.mark.parametrize("seed", training.OFFICIAL_SEEDS)
def test_each_official_seed_is_admitted_as_a_separate_twenty_epoch_plan(
    tmp_path: Path, seed: int
) -> None:
    bundle = _bundle(tmp_path, seed=seed)
    plan = _admit(bundle)
    assert plan.seed == seed
    assert len(plan.train_rows) == training.RESIDUAL_EPOCHS
    assert {row.seed for row in plan.train_rows + plan.validation_rows} == {seed}
    assert {row.epoch for row in plan.train_rows} == set(range(training.RESIDUAL_EPOCHS))


def test_cache_key_census_is_a_read_only_closed_tuple(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    census = bundle.train_cache.cache_key_census
    assert type(census) is tuple
    assert census == tuple(sorted(census))
    assert len(census) == 20
    with pytest.raises(AttributeError):
        bundle.train_cache.cache_key_census = ()  # type: ignore[misc]


def test_missing_and_extra_cache_keys_never_admit_a_subset(tmp_path: Path) -> None:
    train_source, val_source = _write_sources(tmp_path / "prepared")
    val_cache = _write_cache(tmp_path / "val-cache", val_source, seed=1729, epochs=(0,))
    config = training.TrainingConfig(stage="residual", seed=1729)
    missing = _write_cache(
        tmp_path / "missing-train-cache",
        train_source,
        seed=1729,
        epochs=tuple(range(19)),
    )
    with pytest.raises(PeriodicTrainingCacheError, match="exceeds its closed"):
        admit_periodic_descriptor_training_plan(
            system_id="08",
            config=config,
            train_source=train_source,
            val_source=val_source,
            train_cache=missing,
            val_cache=val_cache,
        )
    extra = _write_cache(
        tmp_path / "extra-train-cache",
        train_source,
        seed=1729,
        epochs=tuple(range(20)),
        extras=((2718, 0),),
    )
    with pytest.raises(PeriodicTrainingCacheError, match="differs from the closed"):
        admit_periodic_descriptor_training_plan(
            system_id="08",
            config=config,
            train_source=train_source,
            val_source=val_source,
            train_cache=extra,
            val_cache=val_cache,
        )


def test_source_binding_and_energy_floor_mismatch_fail_closed(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "one")
    other_train, _ = _write_sources(tmp_path / "two" / "prepared", "other")
    with pytest.raises(PeriodicTrainingCacheError, match="source binding"):
        admit_periodic_descriptor_training_plan(
            system_id="08",
            config=bundle.config,
            train_source=other_train,
            val_source=bundle.val_source,
            train_cache=bundle.train_cache,
            val_cache=bundle.val_cache,
        )

    changed_floors = np.ascontiguousarray(_FLOORS + np.float64(1.0e-13))
    wrong_val = _write_cache(
        tmp_path / "wrong-val-cache",
        bundle.val_source,
        seed=1729,
        epochs=(0,),
        floors=changed_floors,
    )
    with pytest.raises(PeriodicTrainingCacheError, match="energy floors"):
        admit_periodic_descriptor_training_plan(
            system_id="08",
            config=bundle.config,
            train_source=bundle.train_source,
            val_source=bundle.val_source,
            train_cache=bundle.train_cache,
            val_cache=wrong_val,
        )


def test_resource_limit_precedes_any_shard_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path)
    opened = 0
    original = PeriodicDescriptorCacheV2.open_batch

    def counted(self, batch, contexts):
        nonlocal opened
        opened += 1
        return original(self, batch, contexts)

    monkeypatch.setattr(PeriodicDescriptorCacheV2, "open_batch", counted)
    with pytest.raises(ResourceLimitError) as captured:
        admit_periodic_descriptor_training_plan(
            system_id="08",
            config=replace(bundle.config, edge_budget=2),
            train_source=bundle.train_source,
            val_source=bundle.val_source,
            train_cache=bundle.train_cache,
            val_cache=bundle.val_cache,
        )
    assert captured.value.required_edges == 3
    assert captured.value.edge_budget == 2
    assert opened == 0


def _numeric_mutant(batch: training.RetrievalTrainingBatch):
    groups = batch.groups
    skeletons = np.array(groups.skeletons, copy=True)
    skeletons[0, 0, 0, 0, 0] += np.float32(0.125)
    changed = PreparedGroupBatch(
        skeletons=np.ascontiguousarray(skeletons),
        actor_mask=np.ascontiguousarray(groups.actor_mask),
        frame_mask=np.ascontiguousarray(groups.frame_mask),
        track_mask=np.ascontiguousarray(groups.track_mask),
        actor_commitments=groups.actor_commitments,
        group_commitments=groups.group_commitments,
    )
    return replace(batch, groups=changed)


def _padded_mutant(batch: training.RetrievalTrainingBatch):
    groups = batch.groups
    batch_size, actors, frames, joints, axes = groups.skeletons.shape
    skeletons = np.zeros((batch_size, actors + 1, frames, joints, axes), dtype=np.float32)
    skeletons[:, :actors] = groups.skeletons
    actor_mask = np.zeros((batch_size, actors + 1), dtype=np.bool_)
    actor_mask[:, :actors] = groups.actor_mask
    track_mask = np.zeros((batch_size, actors + 1, frames, joints), dtype=np.bool_)
    track_mask[:, :actors] = groups.track_mask
    changed = PreparedGroupBatch(
        skeletons=np.ascontiguousarray(skeletons),
        actor_mask=np.ascontiguousarray(actor_mask),
        frame_mask=np.ascontiguousarray(groups.frame_mask),
        track_mask=np.ascontiguousarray(track_mask),
        actor_commitments=tuple(row + (None,) for row in groups.actor_commitments),
        group_commitments=groups.group_commitments,
    )
    return replace(batch, groups=changed)


def _source_digest_mutant(batch: training.RetrievalTrainingBatch):
    assert batch.descriptor_contexts is not None
    contexts = tuple(
        replace(context, source_batch_sha256="f" * 64) for context in batch.descriptor_contexts
    )
    return replace(batch, descriptor_contexts=contexts)


def _epoch_mutant(batch: training.RetrievalTrainingBatch):
    assert batch.descriptor_contexts is not None
    contexts = tuple(
        DescriptorWindowContext.from_yaw(
            prepared_manifest_sha256=context.prepared_manifest_sha256,
            source_batch_sha256=context.source_batch_sha256,
            window_sha256=context.window_sha256,
            window_ordinal=context.window_ordinal,
            split="train",
            seed=context.seed,
            epoch=context.epoch + 1,
            augmentation_yaw=deterministic_group_yaw(
                seed=context.seed,
                epoch=context.epoch + 1,
                window_ordinal=context.window_ordinal,
            ),
        )
        for context in batch.descriptor_contexts
    )
    return replace(batch, descriptor_contexts=contexts)


def test_selector_rejects_numeric_padding_source_and_context_mutants_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path)
    plan = _admit(bundle)
    batch = next(bundle.train_source.iter_epoch(epoch=2, seed=1729))
    opened = 0
    original = PeriodicDescriptorCacheV2.open_batch

    def counted(self, groups, contexts):
        nonlocal opened
        opened += 1
        return original(self, groups, contexts)

    monkeypatch.setattr(PeriodicDescriptorCacheV2, "open_batch", counted)
    for mutant in (
        _numeric_mutant(batch),
        _padded_mutant(batch),
        _source_digest_mutant(batch),
        _epoch_mutant(batch),
    ):
        with pytest.raises(PeriodicTrainingCacheError, match="not an exact admitted"):
            plan.open_training_batch(mutant)
    assert opened == 0
    bundle.train_cache.index_sha256 = "f" * 64
    with pytest.raises(PeriodicTrainingCacheError, match="metadata changed"):
        plan.open_training_batch(batch)
    assert opened == 0


def test_rng_consuming_source_is_detected_and_caller_states_are_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path)
    original = prepared.PreparedTrainingDataSourceV2.iter_epoch

    def consuming(self, *, epoch: int, seed: int):
        random.random()
        np.random.random()
        torch.rand(1)
        yield from original(self, epoch=epoch, seed=seed)

    monkeypatch.setattr(prepared.PreparedTrainingDataSourceV2, "iter_epoch", consuming)
    before = _rng_snapshot()
    with pytest.raises(PeriodicTrainingCacheError, match="changed caller RNG"):
        _admit(bundle)
    _assert_rng_equal(before)


def test_validation_selector_requires_canonical_epoch_zero_context_and_same_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path)
    plan = _admit(bundle)
    opened = 0
    original = PeriodicDescriptorCacheV2.open_batch

    def counted(self, groups, contexts):
        nonlocal opened
        opened += 1
        return original(self, groups, contexts)

    monkeypatch.setattr(PeriodicDescriptorCacheV2, "open_batch", counted)
    wrong_seed = next(bundle.val_source.iter_epoch(epoch=0, seed=2718))
    with pytest.raises(PeriodicTrainingCacheError, match="differs from the plan"):
        plan.open_validation_batch(wrong_seed)
    canonical = next(bundle.val_source.iter_epoch(epoch=0, seed=1729))
    assert canonical.descriptor_contexts is not None
    assert all(context.epoch == 0 for context in canonical.descriptor_contexts)
    assert all(
        context.augmentation_yaw_be_hex == struct.pack(">d", 0.0).hex()
        for context in canonical.descriptor_contexts
    )
    assert plan.open_validation_batch(canonical).stream_kind == "FULL_RELATION"
    assert opened == 1


def test_plan_has_no_training_checkpoint_or_capture_validation_authority(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    plan = _admit(bundle)
    payload = plan.canonical_bytes()
    assert b'"authority":0' in payload
    assert b'"production":false' in payload
    assert b'"result_claimed":false' in payload
    assert b"checkpoint" not in payload
    assert b"capture" in payload
    assert b"holistic-capture-validation" in payload
    assert not hasattr(plan, "fit")
    assert not hasattr(plan, "resume")
    validation = next(bundle.val_source.iter_epoch(epoch=0, seed=1729))
    assert validation.descriptor_contexts is not None
    assert struct.pack(">d", 0.0).hex() in {
        context.augmentation_yaw_be_hex for context in validation.descriptor_contexts
    }
