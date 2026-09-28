from __future__ import annotations

import hashlib
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from phaseset_core.contracts import PreparedGroupBatch, group_commitment
from phaseset_core.controls import PhaseSetControlError, build_phaseset_system
from phaseset_core.models import PhaseSetEncoder, PhaseSetModelError
from phaseset_core.periodic import ResourceLimitError
from phaseset_core.periodic_descriptor_cache_v2 import (
    CachedDescriptorBatch,
    DescriptorCacheSourceBinding,
    DescriptorWindowContext,
    PeriodicDescriptorCacheV2,
    PeriodicDescriptorCacheWriterV2,
    energy_floors_sha256,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


MANIFEST_SHA = _digest("prepared-manifest")
SOURCE_BATCH_SHA = _digest("prepared-source-batch")
SOURCE_TREE_SHA = _digest("source-tree")
ENVIRONMENT_SHA = _digest("environment")


def _binding() -> DescriptorCacheSourceBinding:
    return DescriptorCacheSourceBinding(
        prepared_manifest_sha256=MANIFEST_SHA,
        source_tree_sha256=SOURCE_TREE_SHA,
        environment_sha256=ENVIRONMENT_SHA,
    )


def _floors(offset: float = 0.0) -> np.ndarray:
    return np.ascontiguousarray(
        np.asarray(
            [offset + (index + 1) * 1.0e-12 for index in range(6)],
            dtype=np.float64,
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
    padded_actors: int = 4,
    identities: tuple[tuple[int, ...], tuple[int, ...]] = ((1, 3), (2, 4, 6)),
    permuted_storage: bool = False,
) -> PreparedGroupBatch:
    batch_size = 2
    time_steps = 200
    skeletons = np.zeros((batch_size, padded_actors, time_steps, 22, 3), dtype=np.float32)
    actor_mask = np.zeros((batch_size, padded_actors), dtype=np.bool_)
    frame_mask = np.zeros((batch_size, time_steps), dtype=np.bool_)
    track_mask = np.zeros((batch_size, padded_actors, time_steps, 22), dtype=np.bool_)
    commitments: list[tuple[bytes | None, ...]] = []
    groups: list[bytes] = []
    for row, (actor_ids, valid_length) in enumerate(zip(identities, (181, 200), strict=True)):
        frame_mask[row, :valid_length] = True
        actor_commitments: list[bytes | None] = [None] * padded_actors
        stored_actor_ids = tuple(reversed(actor_ids)) if permuted_storage else actor_ids
        for position, actor_identity in enumerate(stored_actor_ids):
            actor_mask[row, position] = True
            track_mask[row, position, :valid_length] = True
            skeletons[row, position, :valid_length] = _actor_motion(actor_identity, valid_length)
            actor_commitments[position] = bytes([actor_identity]) * 32
        valid_commitments = tuple(bytes([value]) * 32 for value in actor_ids)
        commitments.append(tuple(actor_commitments))
        groups.append(group_commitment(valid_commitments))
    return PreparedGroupBatch(
        skeletons=np.ascontiguousarray(skeletons),
        actor_mask=np.ascontiguousarray(actor_mask),
        frame_mask=np.ascontiguousarray(frame_mask),
        track_mask=np.ascontiguousarray(track_mask),
        actor_commitments=tuple(commitments),
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


def _contexts() -> tuple[DescriptorWindowContext, ...]:
    return tuple(
        DescriptorWindowContext.from_yaw(
            prepared_manifest_sha256=MANIFEST_SHA,
            source_batch_sha256=SOURCE_BATCH_SHA,
            window_sha256=_digest(f"window-{row}"),
            window_ordinal=ordinal,
            split="val",
            seed=1729,
            epoch=0,
            augmentation_yaw=0.0,
        )
        for row, ordinal in enumerate((101, 205))
    )


def _create_cached_batch(
    root: Path,
    batch: PreparedGroupBatch,
    floors: np.ndarray,
) -> CachedDescriptorBatch:
    contexts = _contexts()
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
    return cache.open_batch(batch, contexts)


def _assert_outputs_equal(left: object, right: object) -> None:
    for name in (
        "tokens",
        "band_mask",
        "pair_component",
        "topology_delta",
        "valid_pair_count",
        "topology_node_count",
    ):
        assert torch.equal(getattr(left, name), getattr(right, name)), name


def _parameter_gradients(
    model: torch.nn.Module,
) -> dict[str, torch.Tensor | None]:
    values: dict[str, torch.Tensor | None] = {}
    for name, parameter in model.named_parameters():
        values[name] = None if parameter.grad is None else parameter.grad.detach().clone()
    return values


def _assert_numpy_rng_equal(
    left: tuple[object, np.ndarray, int, int, float],
    right: tuple[object, np.ndarray, int, int, float],
) -> None:
    assert left[0] == right[0]
    assert np.array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_production_cached_entry_replays_full_stream_for_exact_vjp(
    tmp_path: Path,
) -> None:
    batch = _boundary_batch()
    floors = _floors()
    cached = _create_cached_batch(tmp_path / "cache", batch, floors)
    assert cached.edge_count == 66 + 120 == 186
    assert cached.energy_floors_sha256 == energy_floors_sha256(floors)
    assert cached.contexts == _contexts()

    torch.manual_seed(81)
    reference = PhaseSetEncoder(
        embedding_dim=16,
        hidden_dim=11,
        energy_floors=floors,
        edge_budget=32_768,
    )
    expected = reference(batch, edge_chunk_size=64)
    expected.tokens.square().sum().backward()
    expected_gradients = _parameter_gradients(reference)

    for chunk_size in (64, 128, 256):
        candidate = PhaseSetEncoder(
            embedding_dim=16,
            hidden_dim=11,
            energy_floors=floors,
            edge_budget=32_768,
        )
        candidate.load_state_dict(reference.state_dict(), strict=True)
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.random.get_rng_state().clone()

        actual = candidate.forward_cached(
            batch,
            descriptor_stream=cached.stream("FULL_RELATION"),
            edge_chunk_size=chunk_size,
        )
        _assert_outputs_equal(actual, expected)
        actual.tokens.square().sum().backward()
        actual_gradients = _parameter_gradients(candidate)
        assert actual_gradients.keys() == expected_gradients.keys()
        for name in expected_gradients:
            expected_gradient = expected_gradients[name]
            actual_gradient = actual_gradients[name]
            assert expected_gradient is not None, name
            assert actual_gradient is not None, name
            assert torch.equal(actual_gradient, expected_gradient), (chunk_size, name)

        assert random.getstate() == python_before
        _assert_numpy_rng_equal(np.random.get_state(), numpy_before)
        assert torch.equal(torch.random.get_rng_state(), torch_before)


def test_cached_production_path_never_calls_uncached_descriptor_iterator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _batch()
    floors = _floors()
    cached = _create_cached_batch(tmp_path / "cache", batch, floors)
    encoder = PhaseSetEncoder(
        embedding_dim=16,
        hidden_dim=11,
        energy_floors=floors,
    )
    calls = 0

    def forbidden_uncached(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("configured cached execution used the uncached iterator")

    monkeypatch.setattr(encoder, "_iter_pair_chunks", forbidden_uncached)
    output = encoder.forward_cached(
        batch,
        descriptor_stream=cached.stream("FULL_RELATION"),
        edge_chunk_size=128,
    )
    output.tokens.square().sum().backward()
    assert calls == 0


def test_system06_cached_incidence_and_vjp_never_call_uncached_iterator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _batch()
    floors = _floors()
    cached = _create_cached_batch(tmp_path / "cache", batch, floors)
    system = build_phaseset_system(
        "06",
        embedding_dim=16,
        hidden_dim=11,
        energy_floors=floors,
        edge_budget=32_768,
    )
    assert system.encoder is not None
    calls = 0

    def forbidden_uncached(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("system 06 cached incidence used the uncached iterator")

    monkeypatch.setattr(system.encoder, "_iter_pair_chunks", forbidden_uncached)
    output = system.forward_cached(
        batch,
        descriptor_stream=cached.stream("FULL_RELATION"),
        edge_chunk_size=128,
    )
    output.tokens.square().sum().backward()
    assert calls == 0


def test_equivalent_actor_storage_permutation_preserves_cached_output_and_gradients(
    tmp_path: Path,
) -> None:
    canonical = _batch()
    permuted = _batch(permuted_storage=True)
    floors = _floors()
    cached = _create_cached_batch(tmp_path / "cache", canonical, floors)
    torch.manual_seed(211)
    reference = PhaseSetEncoder(
        embedding_dim=16,
        hidden_dim=11,
        energy_floors=floors,
        edge_budget=32_768,
    )
    expected = reference(canonical, edge_chunk_size=64)
    expected.tokens.square().sum().backward()
    expected_gradients = _parameter_gradients(reference)

    candidate = PhaseSetEncoder(
        embedding_dim=16,
        hidden_dim=11,
        energy_floors=floors,
        edge_budget=32_768,
    )
    candidate.load_state_dict(reference.state_dict(), strict=True)
    actual = candidate.forward_cached(
        permuted,
        descriptor_stream=cached.stream("FULL_RELATION"),
        edge_chunk_size=128,
    )
    _assert_outputs_equal(actual, expected)
    actual.tokens.square().sum().backward()
    actual_gradients = _parameter_gradients(candidate)
    assert actual_gradients.keys() == expected_gradients.keys()
    for name in expected_gradients:
        expected_gradient = expected_gradients[name]
        actual_gradient = actual_gradients[name]
        assert expected_gradient is not None, name
        assert actual_gradient is not None, name
        assert torch.equal(actual_gradient, expected_gradient), name


def test_cached_and_uncached_paths_share_resource_limit_before_stream_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _boundary_batch()
    floors = _floors()
    cached = _create_cached_batch(tmp_path / "cache", batch, floors)
    stream = cached.stream("FULL_RELATION")
    calls = 0

    def forbidden_cached_stream(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("over-budget cached execution consumed descriptors")

    monkeypatch.setattr(type(stream), "iter_chunks", forbidden_cached_stream)
    encoder = PhaseSetEncoder(
        embedding_dim=16,
        hidden_dim=11,
        energy_floors=floors,
        edge_budget=185,
    )
    for execute in (
        lambda: encoder(batch, edge_chunk_size=128),
        lambda: encoder.forward_cached(
            batch,
            descriptor_stream=stream,
            edge_chunk_size=128,
        ),
    ):
        with pytest.raises(ResourceLimitError) as caught:
            execute()
        assert caught.value.code == "RESOURCE_LIMIT"
        assert caught.value.required_edges == 186
        assert caught.value.edge_budget == 185
    assert calls == 0


def test_cached_and_uncached_paths_accept_exact_edge_budget_boundary(
    tmp_path: Path,
) -> None:
    batch = _boundary_batch()
    floors = _floors()
    cached = _create_cached_batch(tmp_path / "cache", batch, floors)
    encoder = PhaseSetEncoder(
        embedding_dim=16,
        hidden_dim=11,
        energy_floors=floors,
        edge_budget=186,
    )
    with torch.no_grad():
        expected = encoder(batch, edge_chunk_size=128)
        actual = encoder.forward_cached(
            batch,
            descriptor_stream=cached.stream("FULL_RELATION"),
            edge_chunk_size=128,
        )
    _assert_outputs_equal(actual, expected)


@pytest.mark.parametrize(
    ("system_id", "stream_kind"),
    (
        ("02", "MARGINAL_POWER"),
        ("03", "MEAN_DIFFERENCE_DCT"),
        ("04", "FULL_RELATION"),
        ("06", "FULL_RELATION"),
        ("07", "FULL_RELATION"),
        ("08", "FULL_RELATION"),
    ),
)
def test_registered_systems_consume_only_their_cached_descriptor_family(
    tmp_path: Path,
    system_id: str,
    stream_kind: str,
) -> None:
    batch = _batch()
    floors = _floors()
    cached = _create_cached_batch(tmp_path / system_id, batch, floors)
    torch.manual_seed(109)
    system = build_phaseset_system(
        system_id,
        embedding_dim=16,
        hidden_dim=11,
        energy_floors=floors,
        edge_budget=32_768,
    )

    expected = system(batch, edge_chunk_size=64)
    expected.tokens.square().sum().backward()
    expected_gradients = _parameter_gradients(system)
    system.zero_grad(set_to_none=True)

    actual = system.forward_cached(
        batch,
        descriptor_stream=cached.stream(stream_kind),  # type: ignore[arg-type]
        edge_chunk_size=128,
    )
    _assert_outputs_equal(actual, expected)
    actual.tokens.square().sum().backward()
    actual_gradients = _parameter_gradients(system)
    for name in expected_gradients:
        expected_gradient = expected_gradients[name]
        actual_gradient = actual_gradients[name]
        if expected_gradient is None:
            assert actual_gradient is None, name
        else:
            assert actual_gradient is not None, name
            assert torch.equal(actual_gradient, expected_gradient), name


def test_cached_entry_rejects_wrong_batch_floors_stream_and_missing_verified_stream(
    tmp_path: Path,
) -> None:
    batch = _batch()
    floors = _floors()
    cached = _create_cached_batch(tmp_path / "cache", batch, floors)
    encoder = PhaseSetEncoder(
        embedding_dim=16,
        hidden_dim=11,
        energy_floors=floors,
    )

    with pytest.raises(PhaseSetModelError, match="actual model batch"):
        encoder.forward_cached(
            _numeric_mutant(batch),
            descriptor_stream=cached.stream("FULL_RELATION"),
        )
    wrong_floors = PhaseSetEncoder(
        embedding_dim=16,
        hidden_dim=11,
        energy_floors=_floors(1.0e-8),
    )
    with pytest.raises(PhaseSetModelError, match="energy floors"):
        wrong_floors.forward_cached(
            batch,
            descriptor_stream=cached.stream("FULL_RELATION"),
        )
    with pytest.raises(PhaseSetModelError, match="stream kind"):
        encoder.forward_cached(
            batch,
            descriptor_stream=cached.stream("MARGINAL_POWER"),
        )
    with pytest.raises(PhaseSetModelError, match="exact verified"):
        encoder.forward_cached(batch, descriptor_stream=None)  # type: ignore[arg-type]

    marginal = build_phaseset_system(
        "02",
        embedding_dim=16,
        hidden_dim=11,
        energy_floors=floors,
    )
    with pytest.raises(PhaseSetModelError, match="stream kind"):
        marginal.forward_cached(
            batch,
            descriptor_stream=cached.stream("FULL_RELATION"),
        )


@pytest.mark.parametrize("system_id", ("01", "05"))
def test_uncacheable_registered_controls_fail_closed_but_legacy_forward_remains(
    tmp_path: Path,
    system_id: str,
) -> None:
    batch = _batch()
    floors = _floors()
    cached = _create_cached_batch(tmp_path / system_id, batch, floors)
    system = build_phaseset_system(
        system_id,
        embedding_dim=16,
        hidden_dim=11,
        energy_floors=floors,
    )
    assert system(batch).tokens.shape == (2, 6, 16)
    with pytest.raises(PhaseSetModelError, match="does not admit"):
        system.forward_cached(
            batch,
            descriptor_stream=cached.stream("FULL_RELATION"),
        )


def test_zero_residual_system_rejects_cached_execution(tmp_path: Path) -> None:
    batch = _batch()
    floors = _floors()
    cached = _create_cached_batch(tmp_path / "cache", batch, floors)
    system = build_phaseset_system("00", embedding_dim=16, energy_floors=floors)
    assert not bool(system(batch).band_mask.any().item())
    with pytest.raises(PhaseSetControlError, match="system 00"):
        system.forward_cached(
            batch,
            descriptor_stream=cached.stream("FULL_RELATION"),
        )
