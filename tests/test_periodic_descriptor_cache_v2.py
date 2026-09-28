from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import errno
import hashlib
import json
import os
from pathlib import Path
import random
import stat
import struct

import numpy as np
import pytest
import torch

from phaseset_core import periodic_descriptor_cache_v2 as cache_module
from phaseset_core.contracts import (
    PreparedGroupBatch,
    PreparedActivityBatch,
    group_commitment,
)
from phaseset_core.models import PhaseSetEncoder
from phaseset_core.periodic import PairChunk
from phaseset_core.periodic_descriptor_cache_v2 import (
    STREAM_KINDS,
    CachedDescriptorBatch,
    DescriptorCacheBounds,
    DescriptorCacheSourceBinding,
    DescriptorCacheV2Error,
    DescriptorCacheV2Miss,
    DescriptorWindowContext,
    PeriodicDescriptorCacheV2,
    PeriodicDescriptorCacheWriterV2,
    iter_uncached_pair_chunks,
    prepared_group_batch_sha256,
)
from phaseset_core.pipeline import deterministic_group_yaw


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


MANIFEST_SHA = _digest("prepared-manifest")
SOURCE_BATCH_SHA = _digest("prepared-source-batch")
SOURCE_TREE_SHA = _digest("source-tree")
ENVIRONMENT_SHA = _digest("environment")
WINDOW_SHAS = (_digest("window-a"), _digest("window-b"))


def _binding() -> DescriptorCacheSourceBinding:
    return DescriptorCacheSourceBinding(
        prepared_manifest_sha256=MANIFEST_SHA,
        source_tree_sha256=SOURCE_TREE_SHA,
        environment_sha256=ENVIRONMENT_SHA,
    )


def _floors(offset: float = 0.0) -> np.ndarray:
    return np.ascontiguousarray(
        np.asarray(
            [offset + (index + 1) * 1.0e-12 for index in range(6)], dtype=np.float64
        )
    )


def _actor_motion(actor_identity: int, valid_length: int) -> np.ndarray:
    time = np.arange(valid_length, dtype=np.float64)[:, None]
    joint = np.arange(22, dtype=np.float64)[None, :]
    value = np.zeros((valid_length, 22, 3), dtype=np.float64)
    value[..., 0] = 0.013 * actor_identity + 0.0011 * time * (joint + 1.0)
    value[..., 1] = 0.007 * actor_identity + 0.0003 * time * (23.0 - joint)
    value[..., 2] = -0.017 * actor_identity + np.sin((time + joint) / 31.0)
    return np.ascontiguousarray(value, dtype=np.float32)


def _batch(
    *,
    permuted: bool = False,
    padded_actors: int = 4,
    identities: tuple[tuple[int, ...], tuple[int, ...]] = ((1, 3), (2, 4, 6)),
) -> PreparedGroupBatch:
    if padded_actors < 3:
        raise ValueError("fixture requires at least three padded actors")
    if any(len(row) > padded_actors for row in identities):
        raise ValueError("fixture identities exceed padded actor count")
    batch_size = 2
    time = 200
    skeletons = np.zeros((batch_size, padded_actors, time, 22, 3), dtype=np.float32)
    actor_mask = np.zeros((batch_size, padded_actors), dtype=np.bool_)
    frame_mask = np.zeros((batch_size, time), dtype=np.bool_)
    track_mask = np.zeros((batch_size, padded_actors, time, 22), dtype=np.bool_)
    valid_lengths = (181, 200)
    actor_rows: list[tuple[bytes | None, ...]] = []
    groups: list[bytes] = []
    for row, (row_ids, valid_length) in enumerate(
        zip(identities, valid_lengths, strict=True)
    ):
        frame_mask[row, :valid_length] = True
        if permuted:
            positions = tuple(range(len(row_ids)))
            source_ids = tuple(reversed(row_ids))
        else:
            positions = tuple(range(len(row_ids)))
            source_ids = row_ids
        commitments: list[bytes | None] = [None] * padded_actors
        for position, identity in zip(positions, source_ids, strict=True):
            commitment = bytes([identity]) * 32
            commitments[position] = commitment
            actor_mask[row, position] = True
            track_mask[row, position, :valid_length] = True
            skeletons[row, position, :valid_length] = _actor_motion(
                identity, valid_length
            )
        valid_commitments = tuple(bytes([identity]) * 32 for identity in row_ids)
        actor_rows.append(tuple(commitments))
        groups.append(group_commitment(valid_commitments))
    return PreparedGroupBatch(
        skeletons=np.ascontiguousarray(skeletons),
        actor_mask=np.ascontiguousarray(actor_mask),
        frame_mask=np.ascontiguousarray(frame_mask),
        track_mask=np.ascontiguousarray(track_mask),
        actor_commitments=tuple(actor_rows),
        group_commitments=tuple(groups),
    )


def _boundary_batch() -> PreparedGroupBatch:
    return _batch(
        padded_actors=16,
        identities=(tuple(range(1, 13)), tuple(range(33, 49))),
    )


def _numeric_mutant(batch: PreparedGroupBatch) -> PreparedGroupBatch:
    skeletons = np.array(batch.skeletons, copy=True)
    skeletons[0, 0, 17, 3, 0] += np.float32(0.125)
    return PreparedGroupBatch(
        skeletons=np.ascontiguousarray(skeletons),
        actor_mask=np.ascontiguousarray(batch.actor_mask),
        frame_mask=np.ascontiguousarray(batch.frame_mask),
        track_mask=np.ascontiguousarray(batch.track_mask),
        actor_commitments=batch.actor_commitments,
        group_commitments=batch.group_commitments,
    )


def _mask_mutant(batch: PreparedGroupBatch) -> PreparedGroupBatch:
    skeletons = np.array(batch.skeletons, copy=True)
    track_mask = np.array(batch.track_mask, copy=True)
    track_mask[0, 0, 17, 3] = False
    skeletons[0, 0, 17, 3] = np.float32(0.0)
    return PreparedGroupBatch(
        skeletons=np.ascontiguousarray(skeletons),
        actor_mask=np.ascontiguousarray(batch.actor_mask),
        frame_mask=np.ascontiguousarray(batch.frame_mask),
        track_mask=np.ascontiguousarray(track_mask),
        actor_commitments=batch.actor_commitments,
        group_commitments=batch.group_commitments,
    )


def _commitment_mutant(batch: PreparedGroupBatch) -> PreparedGroupBatch:
    rows = [list(row) for row in batch.actor_commitments]
    rows[0][0] = bytes([8]) * 32
    commitments = tuple(tuple(row) for row in rows)
    groups = tuple(
        group_commitment(tuple(value for value in row if value is not None))
        for row in commitments
    )
    return PreparedGroupBatch(
        skeletons=np.ascontiguousarray(batch.skeletons),
        actor_mask=np.ascontiguousarray(batch.actor_mask),
        frame_mask=np.ascontiguousarray(batch.frame_mask),
        track_mask=np.ascontiguousarray(batch.track_mask),
        actor_commitments=commitments,
        group_commitments=groups,
    )


def _contexts(
    *,
    split: str = "val",
    seed: int = 1729,
    epoch: int = 0,
    source_batch_sha256: str = SOURCE_BATCH_SHA,
    window_shas: tuple[str, str] = WINDOW_SHAS,
) -> tuple[DescriptorWindowContext, ...]:
    values = []
    for window_sha, ordinal in zip(window_shas, (101, 205), strict=True):
        yaw = (
            deterministic_group_yaw(seed=seed, epoch=epoch, window_ordinal=ordinal)
            if split == "train"
            else 0.0
        )
        values.append(
            DescriptorWindowContext.from_yaw(
                prepared_manifest_sha256=MANIFEST_SHA,
                source_batch_sha256=source_batch_sha256,
                window_sha256=window_sha,
                window_ordinal=ordinal,
                split=split,  # type: ignore[arg-type]
                seed=seed,
                epoch=epoch,
                augmentation_yaw=yaw,
            )
        )
    return tuple(values)


def _create_cache(
    root: Path,
    batch: PreparedGroupBatch,
    contexts: tuple[DescriptorWindowContext, ...],
    floors: np.ndarray,
) -> tuple[PeriodicDescriptorCacheV2, CachedDescriptorBatch]:
    writer = PeriodicDescriptorCacheWriterV2(
        root,
        source_binding=_binding(),
        energy_floors=floors,
    )
    writer.add_batch(batch, contexts)
    artifact = writer.finalize()
    cache = PeriodicDescriptorCacheV2(
        root,
        expected_index_sha256=artifact.index_sha256,
        expected_source_binding=_binding(),
        energy_floors=floors,
    )
    return cache, cache.open_batch(batch, contexts)


def _flatten(chunks: object) -> dict[str, np.ndarray]:
    selected = tuple(chunks)  # type: ignore[arg-type]
    assert selected and all(type(chunk) is PairChunk for chunk in selected)
    return {
        "batch_indices": np.concatenate([chunk.batch_indices for chunk in selected]),
        "actor_i": np.concatenate([chunk.actor_i for chunk in selected]),
        "actor_j": np.concatenate([chunk.actor_j for chunk in selected]),
        "tokens_ij": np.concatenate([chunk.tokens_ij for chunk in selected]),
        "tokens_ji": np.concatenate([chunk.tokens_ji for chunk in selected]),
        "support_mask": np.concatenate([chunk.support_mask for chunk in selected]),
    }


def _assert_exact_arrays(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray]
) -> None:
    assert left.keys() == right.keys()
    for name in left:
        assert left[name].dtype == right[name].dtype
        assert left[name].shape == right[name].shape
        assert left[name].tobytes(order="C") == right[name].tobytes(order="C")


def _assert_numpy_rng_equal(
    left: tuple[object, np.ndarray, int, int, float],
    right: tuple[object, np.ndarray, int, int, float],
) -> None:
    assert left[0] == right[0]
    assert np.array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )


def _reauthenticate_single_shard(
    root: Path,
    mutate: Callable[[bytes], bytes],
) -> str:
    index_path = root / "index.json"
    value = json.loads(index_path.read_text(encoding="ascii"))
    assert type(value) is dict and type(value["shards"]) is list
    assert len(value["shards"]) == 1
    row = value["shards"][0]
    assert type(row) is dict and type(row["path"]) is str
    shard_path = root / row["path"]
    changed = mutate(shard_path.read_bytes())
    os.chmod(shard_path, stat.S_IREAD | stat.S_IWRITE)
    shard_path.write_bytes(changed)
    row["bytes"] = len(changed)
    row["sha256"] = hashlib.sha256(changed).hexdigest()
    value["total_shard_bytes"] = sum(item["bytes"] for item in value["shards"])
    index_raw = _canonical_json_bytes(value)
    os.chmod(index_path, stat.S_IREAD | stat.S_IWRITE)
    index_path.write_bytes(index_raw)
    return hashlib.sha256(index_raw).hexdigest()


def _bad_magic(raw: bytes) -> bytes:
    return b"X" + raw[1:]


def _bad_header_length(raw: bytes) -> bytes:
    offset = len(cache_module.MAGIC)
    return raw[:offset] + struct.pack(">Q", len(raw)) + raw[offset + 8 :]


def _bad_array_digest(raw: bytes) -> bytes:
    changed = bytearray(raw)
    changed[-1] ^= 1
    return bytes(changed)


def test_cached_stream_is_exact_uncached_descriptor_and_reiterable(
    tmp_path: Path,
) -> None:
    batch = _boundary_batch()
    floors = _floors()
    _, cached = _create_cache(tmp_path / "cache", batch, _contexts(), floors)

    assert cached.edge_count == 66 + 120 == 186
    for stream_kind in STREAM_KINDS:
        expected = _flatten(
            iter_uncached_pair_chunks(
                batch,
                stream_kind,
                energy_floors=floors,
                edge_chunk_size=64,
            )
        )
        for chunk_size, expected_lengths in (
            (64, (64, 64, 58)),
            (128, (128, 58)),
            (256, (186,)),
        ):
            chunks = tuple(
                cached.stream(stream_kind).iter_chunks(edge_chunk_size=chunk_size)
            )
            assert (
                tuple(len(chunk.batch_indices) for chunk in chunks) == expected_lengths
            )
            if chunk_size == 128:
                assert set(chunks[0].batch_indices.tolist()) == {0, 1}
            _assert_exact_arrays(_flatten(chunks), expected)
    for _, array in cached._arrays:  # noqa: SLF001 - deliberate immutable-byte audit
        assert not array.flags.writeable
    with pytest.raises(ValueError):
        cached._arrays[0][1][0] = 99  # noqa: SLF001 - deliberate mutation probe


@pytest.mark.parametrize(
    "bounds",
    (
        DescriptorCacheBounds(max_edges_per_shard=3),
        DescriptorCacheBounds(max_decoded_shard_bytes=4 * 1914 - 1),
        DescriptorCacheBounds(
            max_decoded_shard_bytes=4 * 1914,
            max_shard_bytes=len(cache_module.MAGIC) + 8 + 4 * 1914 - 1,
        ),
        DescriptorCacheBounds(
            max_total_shard_bytes=len(cache_module.MAGIC) + 8 + 4 * 1914 - 1,
        ),
    ),
)
def test_writer_rejects_resource_bounds_before_descriptor_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bounds: DescriptorCacheBounds,
) -> None:
    assert cache_module._DECODED_BYTES_PER_EDGE == 1914
    calls = 0

    def forbidden_collect(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("descriptor collection ran before its resource preflight")

    monkeypatch.setattr(cache_module, "_collect_all_arrays", forbidden_collect)
    writer = PeriodicDescriptorCacheWriterV2(
        tmp_path / "cache",
        source_binding=_binding(),
        energy_floors=_floors(),
        bounds=bounds,
    )
    with pytest.raises(DescriptorCacheV2Error, match="bound"):
        writer.add_batch(_batch(), _contexts())
    assert calls == 0


def test_context_enforces_native_train_yaw_and_positive_zero_validation() -> None:
    ordinal = 101
    yaw = deterministic_group_yaw(seed=1729, epoch=7, window_ordinal=ordinal)
    value = DescriptorWindowContext.from_yaw(
        prepared_manifest_sha256=MANIFEST_SHA,
        source_batch_sha256=SOURCE_BATCH_SHA,
        window_sha256=WINDOW_SHAS[0],
        window_ordinal=ordinal,
        split="train",
        seed=1729,
        epoch=7,
        augmentation_yaw=yaw,
    )
    assert value.augmentation_yaw_be_hex == struct.pack(">d", yaw).hex()
    with pytest.raises(DescriptorCacheV2Error, match="deterministic_group_yaw"):
        DescriptorWindowContext.from_yaw(
            prepared_manifest_sha256=MANIFEST_SHA,
            source_batch_sha256=SOURCE_BATCH_SHA,
            window_sha256=WINDOW_SHAS[0],
            window_ordinal=ordinal,
            split="train",
            seed=1729,
            epoch=7,
            augmentation_yaw=0.0,
        )
    with pytest.raises(DescriptorCacheV2Error, match="validation context"):
        DescriptorWindowContext.from_yaw(
            prepared_manifest_sha256=MANIFEST_SHA,
            source_batch_sha256=SOURCE_BATCH_SHA,
            window_sha256=WINDOW_SHAS[0],
            window_ordinal=ordinal,
            split="val",
            seed=1729,
            epoch=0,
            augmentation_yaw=-0.0,
        )


def test_actor_permutation_canonicalizes_but_dynamic_repacking_is_rejected(
    tmp_path: Path,
) -> None:
    canonical = _batch()
    permuted = _batch(permuted=True)
    repacked = _batch(padded_actors=5)
    assert prepared_group_batch_sha256(canonical) == prepared_group_batch_sha256(
        permuted
    )
    assert prepared_group_batch_sha256(canonical) != prepared_group_batch_sha256(
        repacked
    )
    contexts = _contexts()
    floors = _floors()
    cache, expected = _create_cache(tmp_path / "cache", canonical, contexts, floors)

    actual = cache.open_batch(permuted, contexts)
    assert actual.cache_key_sha256 == expected.cache_key_sha256
    with pytest.raises(DescriptorCacheV2Miss, match="absent"):
        cache.open_batch(repacked, contexts)


@pytest.mark.parametrize(
    "mutant_factory",
    (_numeric_mutant, _mask_mutant, _commitment_mutant),
)
def test_complete_numeric_mask_and_commitment_digest_rejects_mutants(
    tmp_path: Path,
    mutant_factory: Callable[[PreparedGroupBatch], PreparedGroupBatch],
) -> None:
    batch = _batch()
    contexts = _contexts()
    floors = _floors()
    cache, _ = _create_cache(tmp_path / "cache", batch, contexts, floors)
    mutant = mutant_factory(batch)
    assert prepared_group_batch_sha256(mutant) != prepared_group_batch_sha256(batch)
    with pytest.raises(DescriptorCacheV2Miss, match="absent"):
        cache.open_batch(mutant, contexts)


def test_wrong_source_floor_window_and_missing_entry_fail_closed(
    tmp_path: Path,
) -> None:
    batch = _batch()
    contexts = _contexts()
    floors = _floors()
    writer = PeriodicDescriptorCacheWriterV2(
        tmp_path / "cache",
        source_binding=_binding(),
        energy_floors=floors,
    )
    writer.add_batch(batch, contexts)
    artifact = writer.finalize()

    wrong_binding = replace(_binding(), source_tree_sha256=_digest("wrong-source"))
    with pytest.raises(DescriptorCacheV2Error, match="binding mismatch"):
        PeriodicDescriptorCacheV2(
            artifact.root,
            expected_index_sha256=artifact.index_sha256,
            expected_source_binding=wrong_binding,
            energy_floors=floors,
        )
    with pytest.raises(DescriptorCacheV2Error, match="binding mismatch"):
        PeriodicDescriptorCacheV2(
            artifact.root,
            expected_index_sha256=artifact.index_sha256,
            expected_source_binding=_binding(),
            energy_floors=_floors(1.0e-8),
        )
    cache = PeriodicDescriptorCacheV2(
        artifact.root,
        expected_index_sha256=artifact.index_sha256,
        expected_source_binding=_binding(),
        energy_floors=floors,
    )
    missing = (
        replace(contexts[0], window_sha256=_digest("other-window")),
        contexts[1],
    )
    with pytest.raises(DescriptorCacheV2Miss, match="absent"):
        cache.open_batch(batch, missing)
    wrong_batch = (
        replace(contexts[0], source_batch_sha256=_digest("other-batch")),
        replace(contexts[1], source_batch_sha256=_digest("other-batch")),
    )
    with pytest.raises(DescriptorCacheV2Miss, match="absent"):
        cache.open_batch(batch, wrong_batch)


def test_shard_tamper_and_extra_file_are_rejected_before_stream(tmp_path: Path) -> None:
    batch = _batch()
    contexts = _contexts()
    floors = _floors()
    cache, _ = _create_cache(tmp_path / "cache", batch, contexts, floors)
    shard = next((cache.root / "shards").iterdir())
    os.chmod(shard, stat.S_IREAD | stat.S_IWRITE)
    raw = bytearray(shard.read_bytes())
    raw[-1] ^= 1
    shard.write_bytes(raw)
    with pytest.raises(DescriptorCacheV2Error, match="bytes differ"):
        cache.open_batch(batch, contexts)

    other_root = tmp_path / "other-cache"
    other_cache, _ = _create_cache(other_root, batch, contexts, floors)
    (other_root / "shards" / "extra.pdc2").write_bytes(b"extra")
    with pytest.raises(DescriptorCacheV2Error, match="census"):
        other_cache.open_batch(batch, contexts)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (_bad_magic, "magic"),
        (_bad_header_length, "header length"),
        (_bad_array_digest, "digest mismatch"),
    ),
)
def test_reauthenticated_malformed_shard_reaches_closed_decoder(
    tmp_path: Path,
    mutate: Callable[[bytes], bytes],
    message: str,
) -> None:
    batch = _batch()
    contexts = _contexts()
    floors = _floors()
    original, _ = _create_cache(tmp_path / "cache", batch, contexts, floors)
    changed_index_sha256 = _reauthenticate_single_shard(original.root, mutate)
    changed = PeriodicDescriptorCacheV2(
        original.root,
        expected_index_sha256=changed_index_sha256,
        expected_source_binding=_binding(),
        energy_floors=floors,
    )

    with pytest.raises(DescriptorCacheV2Error, match=message):
        changed.open_batch(batch, contexts)


def test_owned_shard_read_rejects_hardlink_before_payload_read(tmp_path: Path) -> None:
    batch = _batch()
    contexts = _contexts()
    floors = _floors()
    hardlink_cache, _ = _create_cache(
        tmp_path / "hardlink-cache", batch, contexts, floors
    )
    hardlink_shard = next((hardlink_cache.root / "shards").iterdir())
    try:
        os.link(hardlink_shard, tmp_path / "outside-hardlink.pdc2")
    except OSError as error:
        capability_errnos = {
            errno.EACCES,
            errno.EPERM,
            errno.ENOSYS,
            getattr(errno, "ENOTSUP", -1),
            getattr(errno, "EOPNOTSUPP", -1),
        }
        capability_winerrors = {1, 5, 50, 1314}
        if (
            error.errno in capability_errnos
            or getattr(error, "winerror", None) in capability_winerrors
        ):
            pytest.skip("temporary filesystem does not permit hard-link creation")
        raise
    with pytest.raises(DescriptorCacheV2Error, match="single-link"):
        hardlink_cache.open_batch(batch, contexts)


def test_owned_shard_read_rejects_oversize_before_payload_read(tmp_path: Path) -> None:
    batch = _batch()
    contexts = _contexts()
    floors = _floors()
    oversize_cache, _ = _create_cache(
        tmp_path / "oversize-cache", batch, contexts, floors
    )
    oversize_shard = next((oversize_cache.root / "shards").iterdir())
    original_size = oversize_shard.stat().st_size
    bounds = DescriptorCacheBounds(
        max_shard_bytes=original_size,
        max_decoded_shard_bytes=original_size,
    )
    os.chmod(oversize_shard, stat.S_IREAD | stat.S_IWRITE)
    with oversize_shard.open("ab") as stream:
        stream.write(b"x")
    bounded = PeriodicDescriptorCacheV2(
        oversize_cache.root,
        expected_index_sha256=oversize_cache.index_sha256,
        expected_source_binding=_binding(),
        energy_floors=floors,
        bounds=bounds,
    )
    with pytest.raises(DescriptorCacheV2Error, match="stored byte bound"):
        bounded.open_batch(batch, contexts)


def test_writer_and_reader_do_not_consume_any_rng_stream(tmp_path: Path) -> None:
    batch = _batch()
    contexts = _contexts()
    floors = _floors()
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()

    _, cached = _create_cache(tmp_path / "cache", batch, contexts, floors)
    tuple(cached.stream("FULL_RELATION").iter_chunks(edge_chunk_size=64))

    assert random.getstate() == python_before
    _assert_numpy_rng_equal(np.random.get_state(), numpy_before)
    assert torch.equal(torch.random.get_rng_state(), torch_before)


class _CachedFullEncoder(PhaseSetEncoder):
    def __init__(self, cached: CachedDescriptorBatch, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._cached_descriptor_batch = cached

    def _iter_pair_chunks(
        self,
        batch: PreparedActivityBatch,
        *,
        edge_chunk_size: int,
    ):
        del batch
        return self._cached_descriptor_batch.stream("FULL_RELATION").iter_chunks(
            edge_chunk_size=edge_chunk_size
        )


def test_cached_stream_preserves_actual_edge_gradients(tmp_path: Path) -> None:
    batch = _boundary_batch()
    contexts = _contexts()
    floors = _floors()
    _, cached = _create_cache(tmp_path / "cache", batch, contexts, floors)
    torch.manual_seed(81)
    uncached = PhaseSetEncoder(
        embedding_dim=16,
        hidden_dim=11,
        energy_floors=floors,
        edge_budget=32_768,
    )
    expected = uncached(batch, edge_chunk_size=64)
    expected.tokens.square().sum().backward()
    expected_parameters = dict(uncached.named_parameters())
    for chunk_size in (64, 128, 256):
        cached_encoder = _CachedFullEncoder(
            cached,
            embedding_dim=16,
            hidden_dim=11,
            energy_floors=floors,
            edge_budget=32_768,
        )
        cached_encoder.load_state_dict(uncached.state_dict(), strict=True)
        actual = cached_encoder(batch, edge_chunk_size=chunk_size)
        for name in (
            "tokens",
            "band_mask",
            "pair_component",
            "topology_delta",
            "valid_pair_count",
            "topology_node_count",
        ):
            assert torch.equal(getattr(actual, name), getattr(expected, name))

        actual.tokens.square().sum().backward()
        actual_parameters = dict(cached_encoder.named_parameters())
        assert expected_parameters.keys() == actual_parameters.keys()
        for name in expected_parameters:
            expected_gradient = expected_parameters[name].grad
            actual_gradient = actual_parameters[name].grad
            assert expected_gradient is not None
            assert actual_gradient is not None
            assert torch.equal(actual_gradient, expected_gradient), (chunk_size, name)
