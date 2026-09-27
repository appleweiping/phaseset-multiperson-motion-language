"""Oracle and activation-scaling tests for PhaseSet streaming autograd."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest
import torch

from phaseset_core import models as models_module
from phaseset_core.contracts import PreparedActivityBatch, group_commitment
from phaseset_core.controls import RESIDUAL_SYSTEM_IDS, build_phaseset_system
from phaseset_core.models import PhaseSetEncoder
from phaseset_core.streaming_autograd import (
    eager_edge_summaries,
    streaming_edge_summaries,
)


def _key(actor: int) -> bytes:
    return hashlib.sha256(f"streaming-autograd/{actor}".encode()).digest()


def _activity_batch(actor_count: int, *, time_steps: int = 100) -> PreparedActivityBatch:
    time = np.arange(time_steps, dtype=np.float64) / 20.0
    activities = np.zeros((1, actor_count, time_steps, 5), dtype=np.float32)
    for actor in range(actor_count):
        for channel in range(5):
            frequency = (0.75, 1.125, 1.6875)[(actor + channel) % 3]
            activities[0, actor, :, channel] = np.asarray(
                (1.0 + 0.07 * actor)
                * np.sin(
                    2.0 * math.pi * frequency * time
                    + 0.23 * actor
                    + 0.11 * channel
                ),
                dtype=np.float32,
            )
    actor_mask = np.ones((1, actor_count), dtype=np.bool_)
    frame_mask = np.ones((1, time_steps), dtype=np.bool_)
    activity_mask = np.ones_like(activities, dtype=np.bool_)
    keys = tuple(_key(actor) for actor in range(actor_count))
    return PreparedActivityBatch(
        np.ascontiguousarray(activities),
        np.ascontiguousarray(actor_mask),
        np.ascontiguousarray(frame_mask),
        np.ascontiguousarray(activity_mask),
        (keys,),
        (group_commitment(keys),),
    )


def _small_system(system_id: str):
    torch.manual_seed(20260825)
    return build_phaseset_system(
        system_id,
        embedding_dim=8,
        hidden_dim=12,
        incidence_seed=31415,
    )


@pytest.mark.parametrize("system_id", RESIDUAL_SYSTEM_IDS)
def test_streaming_vjp_matches_eager_oracle_for_every_residual_control(
    system_id: str,
) -> None:
    """This includes the differentiable incidence permutation in system 06."""

    batch = _activity_batch(4)
    oracle_system = _small_system(system_id)
    streamed_system = _small_system(system_id)
    streamed_system.load_state_dict(oracle_system.state_dict())
    assert oracle_system.encoder is not None
    assert streamed_system.encoder is not None

    oracle = eager_edge_summaries(
        oracle_system.encoder,
        batch,
        edge_chunk_size=64,
    )
    streamed = streaming_edge_summaries(
        streamed_system.encoder,
        batch,
        edge_chunk_size=64,
    )
    assert torch.equal(oracle.pair_component, streamed.pair_component)
    assert torch.equal(oracle.node_statistics, streamed.node_statistics)
    assert torch.equal(oracle.valid_pair_count, streamed.valid_pair_count)
    assert torch.equal(oracle.topology_node_mask, streamed.topology_node_mask)

    generator = torch.Generator().manual_seed(90100 + int(system_id))
    pair_upstream = torch.randn(
        oracle.pair_component.shape,
        dtype=oracle.pair_component.dtype,
        generator=generator,
    )
    statistic_upstream = torch.randn(
        oracle.node_statistics.shape,
        dtype=oracle.node_statistics.dtype,
        generator=generator,
    )
    torch.autograd.backward(
        (oracle.pair_component, oracle.node_statistics),
        (pair_upstream, statistic_upstream),
    )
    torch.autograd.backward(
        (streamed.pair_component, streamed.node_statistics),
        (pair_upstream, statistic_upstream),
    )

    oracle_parameters = dict(oracle_system.encoder.named_parameters())
    streamed_parameters = dict(streamed_system.encoder.named_parameters())
    edge_names = tuple(
        name
        for name in oracle_parameters
        if name.startswith(("half_edge_encoder.", "pair_encoder."))
    )
    assert edge_names
    for name in edge_names:
        expected = oracle_parameters[name].grad
        actual = streamed_parameters[name].grad
        assert expected is not None and actual is not None
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)


class _CountingEncoder(PhaseSetEncoder):
    def __init__(self) -> None:
        # Widths deliberately differ from K=12 so the dense-allocation assertion
        # cannot confuse a legitimate square model weight with an actor matrix.
        super().__init__(embedding_dim=7, hidden_dim=11)
        self.edge_forward_calls = 0

    def _edge_forward(
        self,
        relation_ij: torch.Tensor,
        relation_ji: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.edge_forward_calls += 1
        return super()._edge_forward(relation_ij, relation_ji)


def test_backward_replays_every_microblock_twice_and_k2_zero_contract_survives() -> None:
    model = _CountingEncoder().train()
    batch = _activity_batch(12, time_steps=17)
    saved_shapes: list[tuple[int, ...]] = []

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved_shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = model.forward_activity(batch, edge_chunk_size=64)
        # C(12,2)=66: one full and one partial canonical microblock.
        assert model.edge_forward_calls == 2
        output.tokens.square().sum().backward()
    # Backward first reconstructs final moments without a graph, then performs
    # one immediately-released edge-MLP VJP per microblock.
    assert model.edge_forward_calls == 6
    assert not any(
        len(shape) >= 2 and shape[0] == 12 and shape[1] == 12
        for shape in saved_shapes
    )

    k2_model = PhaseSetEncoder(embedding_dim=8, hidden_dim=12).train()
    k2 = k2_model.forward_activity(_activity_batch(2, time_steps=17))
    assert torch.equal(k2.topology_delta, torch.zeros_like(k2.topology_delta))
    assert not torch.signbit(k2.topology_delta).any()
    k2.tokens.square().sum().backward()
    for parameter in (
        k2_model.topology_lambda,
        *k2_model.topology_encoder.parameters(),
    ):
        assert parameter.grad is not None
        assert torch.equal(parameter.grad, torch.zeros_like(parameter.grad))
        assert not torch.signbit(parameter.grad).any()


def test_public_full_forward_and_all_parameter_gradients_match_eager_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _activity_batch(12, time_steps=17)
    torch.manual_seed(2718)
    oracle_model = PhaseSetEncoder(embedding_dim=8, hidden_dim=12).train()
    streamed_model = PhaseSetEncoder(embedding_dim=8, hidden_dim=12).train()
    streamed_model.load_state_dict(oracle_model.state_dict())

    production_summary = models_module.streaming_edge_summaries
    monkeypatch.setattr(models_module, "streaming_edge_summaries", eager_edge_summaries)
    oracle = oracle_model.forward_activity(batch, edge_chunk_size=64)
    monkeypatch.setattr(models_module, "streaming_edge_summaries", production_summary)
    streamed = streamed_model.forward_activity(batch, edge_chunk_size=64)
    for field in (
        "tokens",
        "band_mask",
        "pair_component",
        "topology_delta",
        "valid_pair_count",
        "topology_node_count",
    ):
        assert torch.equal(getattr(oracle, field), getattr(streamed, field))

    generator = torch.Generator().manual_seed(31415)
    token_upstream = torch.randn(
        oracle.tokens.shape,
        dtype=oracle.tokens.dtype,
        generator=generator,
    )
    pair_upstream = torch.randn(
        oracle.pair_component.shape,
        dtype=oracle.pair_component.dtype,
        generator=generator,
    )
    topology_upstream = torch.randn(
        oracle.topology_delta.shape,
        dtype=oracle.topology_delta.dtype,
        generator=generator,
    )
    torch.autograd.backward(
        (oracle.tokens, oracle.pair_component, oracle.topology_delta),
        (token_upstream, pair_upstream, topology_upstream),
    )
    torch.autograd.backward(
        (streamed.tokens, streamed.pair_component, streamed.topology_delta),
        (token_upstream, pair_upstream, topology_upstream),
    )
    for (expected_name, expected), (actual_name, actual) in zip(
        oracle_model.named_parameters(),
        streamed_model.named_parameters(),
        strict=True,
    ):
        assert actual_name == expected_name
        assert expected.grad is not None and actual.grad is not None
        torch.testing.assert_close(actual.grad, expected.grad, rtol=5e-5, atol=5e-6)


@torch.backends.mkldnn.flags(enabled=False)
def test_cpu_bfloat16_autocast_is_replayed_during_streaming_backward() -> None:
    # Exercise autocast replay with portable ATen BF16 kernels, not an ISA-specific
    # oneDNN primitive. The scoped decorator restores the backend even on failure;
    # other tests and the server's native BF16 qualification keep their backend.
    assert not torch.backends.mkldnn.enabled
    batch = _activity_batch(4)
    torch.manual_seed(1729)
    oracle_model = PhaseSetEncoder(embedding_dim=8, hidden_dim=12).train()
    streamed_model = PhaseSetEncoder(embedding_dim=8, hidden_dim=12).train()
    streamed_model.load_state_dict(oracle_model.state_dict())
    observed_dtypes: list[torch.dtype] = []

    def observe_bf16(
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        observed_dtypes.append(output.dtype)
        assert output.dtype == torch.bfloat16

    for model in (oracle_model, streamed_model):
        model.half_edge_encoder.register_forward_hook(observe_bf16)
        model.pair_encoder.register_forward_hook(observe_bf16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        oracle = eager_edge_summaries(
            oracle_model,
            batch,
            edge_chunk_size=64,
        )
        streamed = streaming_edge_summaries(
            streamed_model,
            batch,
            edge_chunk_size=64,
        )
        statistic_upstream = torch.randn_like(oracle.node_statistics)
        pair_upstream = torch.randn_like(oracle.pair_component)
        oracle_loss = (oracle.node_statistics * statistic_upstream).sum() + (
            oracle.pair_component * pair_upstream
        ).sum()
        streamed_loss = (streamed.node_statistics * statistic_upstream).sum() + (
            streamed.pair_component * pair_upstream
        ).sum()
    assert torch.equal(oracle.node_statistics, streamed.node_statistics)
    oracle_loss.backward()
    streamed_loss.backward()
    assert len(observed_dtypes) >= 12  # Both forwards and both streaming replays.
    for (expected_name, expected), (actual_name, actual) in zip(
        oracle_model.named_parameters(),
        streamed_model.named_parameters(),
        strict=True,
    ):
        assert actual_name == expected_name
        if expected_name.startswith(("half_edge_encoder.", "pair_encoder.")):
            assert expected.grad is not None and actual.grad is not None
            torch.testing.assert_close(actual.grad, expected.grad, rtol=3e-5, atol=3e-6)


def _forward_saved_tensor_profile(actor_count: int) -> tuple[int, int, tuple[tuple[int, ...], ...]]:
    torch.manual_seed(7000 + actor_count)
    model = PhaseSetEncoder(embedding_dim=7, hidden_dim=11).train()
    saved: list[tuple[tuple[int, ...], int, torch.dtype]] = []

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved.append((tuple(tensor.shape), tensor.numel(), tensor.dtype))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = model.forward_activity(
            _activity_batch(actor_count, time_steps=17),
            edge_chunk_size=64,
        )
        loss = output.tokens.square().sum()
    assert loss.requires_grad
    assert not any(
        shape == (model.embedding_dim,) and dtype == torch.float64
        for shape, _, dtype in saved
    )
    assert not any(
        len(shape) >= 2 and shape[0] == actor_count and shape[1] == actor_count
        for shape, _, _ in saved
    )
    return len(saved), sum(numel for _, numel, _ in saved), tuple(
        shape for shape, _, _ in saved
    )


def test_saved_tensor_growth_is_linear_in_actors_not_quadratic_in_edges() -> None:
    profiles = {
        actor_count: _forward_saved_tensor_profile(actor_count)
        for actor_count in (8, 16, 24)
    }
    # The custom Function saves a constant set of edge parameters; topology-head
    # tensors grow with K.  No per-edge Chan vector is retained.
    assert len({profiles[count][0] for count in profiles}) == 1
    first_slope = profiles[16][1] - profiles[8][1]
    second_slope = profiles[24][1] - profiles[16][1]
    assert first_slope > 0
    assert second_slope == first_slope
    # Edge-count increments are 92 then 156, so equal saved-tensor increments
    # rule out an O(C(K,2)*D) forward activation graph.
    assert 16 * 15 // 2 - 8 * 7 // 2 != 24 * 23 // 2 - 16 * 15 // 2
