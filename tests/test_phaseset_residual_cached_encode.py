from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from phaseset_core import training
from phaseset_core.contracts import PreparedGroupBatch, group_commitment
from phaseset_core.models import PhaseSetModelError
from phaseset_core.objectives import PhaseSetRetrievalHead
from phaseset_core.periodic import ResourceLimitError
from phaseset_core.periodic_descriptor_cache_v2 import (
    CachedPairChunkStream,
    DescriptorCacheSourceBinding,
    DescriptorWindowContext,
    PeriodicDescriptorCacheV2,
    PeriodicDescriptorCacheWriterV2,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _floors(offset: float = 0.0) -> np.ndarray:
    return np.ascontiguousarray(
        np.asarray(
            [offset + (index + 1) * 1.0e-12 for index in range(6)],
            dtype=np.float64,
        )
    )


def _batch(*, numeric_offset: float = 0.0) -> PreparedGroupBatch:
    time_steps = 200
    actor_count = 3
    time = np.arange(time_steps, dtype=np.float64)[:, None]
    joint = np.arange(22, dtype=np.float64)[None, :]
    skeletons = np.zeros((1, actor_count, time_steps, 22, 3), dtype=np.float32)
    for actor in range(actor_count):
        skeletons[0, actor, :, :, 0] = 0.013 * (actor + 1) + 0.0007 * time * (joint + 1.0)
        skeletons[0, actor, :, :, 1] = 0.009 * (actor + 1) + 0.0003 * time * (23.0 - joint)
        skeletons[0, actor, :, :, 2] = np.sin((time + joint + 5.0 * actor) / 29.0)
    skeletons[0, 0, 17, 3, 0] += np.float32(numeric_offset)
    actors = tuple(bytes([index + 1]) * 32 for index in range(actor_count))
    return PreparedGroupBatch(
        skeletons=np.ascontiguousarray(skeletons),
        actor_mask=np.ones((1, actor_count), dtype=np.bool_),
        frame_mask=np.ones((1, time_steps), dtype=np.bool_),
        track_mask=np.ones((1, actor_count, time_steps, 22), dtype=np.bool_),
        actor_commitments=(actors,),
        group_commitments=(group_commitment(actors),),
    )


def _context() -> tuple[DescriptorWindowContext, ...]:
    return (
        DescriptorWindowContext.from_yaw(
            prepared_manifest_sha256=_digest("prepared-manifest"),
            source_batch_sha256=_digest("source-npz"),
            window_sha256=_digest("window"),
            window_ordinal=7,
            split="val",
            seed=1729,
            epoch=0,
            augmentation_yaw=0.0,
        ),
    )


def _binding() -> DescriptorCacheSourceBinding:
    return DescriptorCacheSourceBinding(
        prepared_manifest_sha256=_digest("prepared-manifest"),
        source_tree_sha256=_digest("source-tree"),
        environment_sha256=_digest("environment"),
    )


def _cached_streams(
    root: Path,
    batch: PreparedGroupBatch,
    floors: np.ndarray,
) -> dict[str, CachedPairChunkStream]:
    writer = PeriodicDescriptorCacheWriterV2(
        root,
        source_binding=_binding(),
        energy_floors=floors,
    )
    writer.add_batch(batch, _context())
    artifact = writer.finalize()
    reader = PeriodicDescriptorCacheV2(
        root,
        expected_index_sha256=artifact.index_sha256,
        expected_source_binding=_binding(),
        energy_floors=floors,
    )
    opened = reader.open_batch(batch, _context())
    return {
        kind: opened.stream(kind)
        for kind in ("FULL_RELATION", "MARGINAL_POWER", "MEAN_DIFFERENCE_DCT")
    }


def _system(
    system_id: str,
    floors: np.ndarray,
    *,
    edge_budget: int = 32,
) -> training.ResidualRetrievalSystem:
    frozen_base = training.build_registered_base_training_system("B0").group_base
    return training.build_registered_residual_training_system(
        system_id,
        frozen_base,
        seed=1729,
        energy_floors=floors,
        edge_budget=edge_budget,
    )


def _gradients(module: torch.nn.Module) -> dict[str, torch.Tensor | None]:
    return {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in module.named_parameters()
    }


def _assert_gradients_equal(
    left: dict[str, torch.Tensor | None],
    right: dict[str, torch.Tensor | None],
) -> None:
    assert left.keys() == right.keys()
    for name in left:
        if left[name] is None:
            assert right[name] is None, name
        else:
            assert right[name] is not None, name
            assert torch.equal(left[name], right[name]), name


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
def test_registered_residual_cached_method_matches_uncached_tokens_and_mask(
    tmp_path: Path,
    system_id: str,
    stream_kind: str,
) -> None:
    batch = _batch()
    floors = _floors()
    streams = _cached_streams(tmp_path / "cache", batch, floors)
    reference = _system(system_id, floors).eval()
    candidate = copy.deepcopy(reference).eval()

    expected = reference.encode_trainable(batch)
    actual = candidate.encode_trainable_cached(batch, streams[stream_kind])

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert actual[0].dtype == torch.float32
    assert actual[0].is_contiguous()
    assert actual[1].is_contiguous()


def test_cached_stream_is_reiterable_and_preserves_rng_and_vjp(tmp_path: Path) -> None:
    batch = _batch()
    floors = _floors()
    stream = _cached_streams(tmp_path / "cache", batch, floors)["FULL_RELATION"]
    reference = _system("06", floors).eval()
    first = copy.deepcopy(reference).eval()
    second = copy.deepcopy(reference).eval()

    expected_tokens, expected_mask = reference.encode_trainable(batch)
    expected_tokens.square().sum().backward()
    expected_gradients = _gradients(reference)

    random.seed(19)
    np.random.seed(23)
    torch.manual_seed(29)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    first_tokens, first_mask = first.encode_trainable_cached(batch, stream)
    first_tokens.square().sum().backward()
    first_gradients = _gradients(first)
    second_tokens, second_mask = second.encode_trainable_cached(batch, stream)
    second_tokens.square().sum().backward()
    second_gradients = _gradients(second)

    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert torch.equal(first_tokens, expected_tokens)
    assert torch.equal(second_tokens, expected_tokens)
    assert torch.equal(first_mask, expected_mask)
    assert torch.equal(second_mask, expected_mask)
    _assert_gradients_equal(first_gradients, expected_gradients)
    _assert_gradients_equal(second_gradients, expected_gradients)


def test_cached_method_rejects_wrong_batch_stream_floors_and_resource_limit(
    tmp_path: Path,
) -> None:
    batch = _batch()
    floors = _floors()
    streams = _cached_streams(tmp_path / "cache", batch, floors)
    system = _system("08", floors).eval()

    with pytest.raises(PhaseSetModelError, match="actual model batch"):
        system.encode_trainable_cached(_batch(numeric_offset=0.125), streams["FULL_RELATION"])
    with pytest.raises(PhaseSetModelError, match="stream kind"):
        system.encode_trainable_cached(batch, streams["MARGINAL_POWER"])
    with pytest.raises(PhaseSetModelError, match="energy floors"):
        _system("08", _floors(1.0e-8)).eval().encode_trainable_cached(
            batch,
            streams["FULL_RELATION"],
        )
    with pytest.raises(ResourceLimitError):
        _system("08", floors, edge_budget=2).eval().encode_trainable_cached(
            batch,
            streams["FULL_RELATION"],
        )
    with pytest.raises(TypeError, match="exact CachedPairChunkStream"):
        system.encode_trainable_cached(batch, None)  # type: ignore[arg-type]


@pytest.mark.parametrize("system_id", ("01", "05"))
def test_noncacheable_registered_residuals_are_rejected(
    tmp_path: Path,
    system_id: str,
) -> None:
    batch = _batch()
    floors = _floors()
    stream = _cached_streams(tmp_path / "cache", batch, floors)["FULL_RELATION"]
    with pytest.raises(training.TrainingRuntimeError, match="does not admit"):
        _system(system_id, floors).encode_trainable_cached(batch, stream)


def test_nonregistered_encoder_and_method_drift_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _batch()
    floors = _floors()
    stream = _cached_streams(tmp_path / "cache", batch, floors)["FULL_RELATION"]

    class UnregisteredPeriodic(torch.nn.Module):
        system_id = "08"
        embedding_dim = 512

        def forward(self, _groups: PreparedGroupBatch) -> object:
            pytest.fail("unregistered periodic encoder was executed")

        def forward_cached(
            self,
            _groups: PreparedGroupBatch,
            *,
            descriptor_stream: CachedPairChunkStream,
        ) -> object:
            del descriptor_stream
            pytest.fail("unregistered cached encoder was executed")

    frozen_base = training.build_registered_base_training_system("B0").group_base
    unregistered = training.ResidualRetrievalSystem(
        frozen_base,
        UnregisteredPeriodic(),
        PhaseSetRetrievalHead(512, text_hidden_dim=512, residual_lambda_init=0.0),
        embedding_dim=512,
    )
    with pytest.raises(training.TrainingRuntimeError, match="exact registered"):
        unregistered.encode_trainable_cached(batch, stream)

    registered = _system("08", floors)
    config = training.TrainingConfig(
        stage="residual",
        seed=1729,
        device="cpu",
        request_bf16=False,
        bf16_runtime_qualified=False,
        edge_budget=32,
        checkpoint_every_updates=1,
        synthetic_contract=False,
    )
    monkeypatch.setattr(
        training.ResidualRetrievalSystem,
        "encode_trainable_cached",
        lambda *_args, **_kwargs: pytest.fail("drifted cached method executed"),
    )
    with pytest.raises(training.TrainingRuntimeError, match="method identity changed"):
        training._validate_registered_formal_system(registered, config)
