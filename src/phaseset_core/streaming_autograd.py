"""Memory-bounded autograd for streamed PhaseSet edge summaries.

The public encoder reduces every directed half-edge to per-node mean/M2 and
every symmetric pair token to a per-band mean.  Building those reductions with
ordinary autograd retains one width-D Chan intermediate per edge.  This module
keeps the exact forward reduction order, but exposes only the O(B*K*D)
summaries to autograd.  Its backward pass first reconstructs the final moments
without a graph, then replays one canonical microblock at a time and immediately
accumulates the edge-MLP vector-Jacobian product.

Motion descriptors remain model inputs rather than differentiable tensors, so
the custom backward only has to return gradients for the shared half-edge and
pair encoders.  Topology and common-postprocess heads stay in ordinary PyTorch
autograd downstream of these summaries.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Protocol, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.autograd.function import once_differentiable

from .contracts import PreparedActivityBatch, validate_prepared_activity_batch
from .periodic import BAND_COUNT, CANONICAL_MICROBLOCK_SIZE, PairChunk


class _StreamingEdgeModel(Protocol):
    """Protected model seam required by the streaming summary operator."""

    embedding_dim: int
    half_edge_encoder: nn.Module
    pair_encoder: nn.Module

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: ...

    def _iter_pair_chunks(
        self,
        batch: PreparedActivityBatch,
        *,
        edge_chunk_size: int,
    ) -> Iterator[PairChunk]: ...

    def _edge_forward(
        self,
        relation_ij: Tensor,
        relation_ji: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]: ...

    def _transform_edge_outputs(
        self,
        half_ij: Tensor,
        half_ji: Tensor,
        pair_tokens: Tensor,
        support_mask: np.ndarray,
        *,
        batch_indices: np.ndarray,
        actor_i: np.ndarray,
        actor_j: np.ndarray,
    ) -> tuple[Tensor, Tensor, Tensor]: ...


class _PairChunkStream(Protocol):
    """Re-iterable descriptor input retained by the custom autograd context."""

    def iter_chunks(self, *, edge_chunk_size: int) -> Iterator[PairChunk]: ...


@dataclass(frozen=True, slots=True)
class StreamingEdgeSummaries:
    """Differentiable summaries plus structural, non-differentiable masks."""

    pair_component: Tensor
    node_statistics: Tensor
    valid_pair_count: Tensor
    topology_node_mask: Tensor


@dataclass(frozen=True, slots=True)
class _MomentState:
    """O(B*K*bands*D) state needed by the analytical streaming VJP."""

    node_mean: Tensor
    node_m2: Tensor
    node_degree: Tensor
    pair_count: Tensor


@dataclass(frozen=True, slots=True)
class _GlobalIncidenceRoutingPlan:
    """O(E)-integer routing metadata for system 06's group-global bijection."""

    target_actors: tuple[tuple[np.ndarray, ...], ...]
    offsets: tuple[tuple[int, ...], ...]
    inverse_steps: tuple[tuple[int, ...], ...]

    def target_rank(self, group: int, band: int, source_rank: int) -> int:
        length = int(self.target_actors[group][band].shape[0])
        if not 0 <= source_rank < length:
            raise RuntimeError("global incidence source rank is outside its census")
        if length <= 1:
            return source_rank
        return (
            (source_rank - self.offsets[group][band])
            * self.inverse_steps[group][band]
        ) % length

    def target_actor(self, group: int, band: int, source_rank: int) -> int:
        return int(
            self.target_actors[group][band][
                self.target_rank(group, band, source_rank)
            ]
        )


def _fixed_binary_tree_sum(values: list[Tensor]) -> Tensor:
    if not values:
        raise ValueError("fixed-tree reduction requires at least one value")
    level = values
    while len(level) > 1:
        next_level: list[Tensor] = []
        for offset in range(0, len(level) - 1, 2):
            next_level.append(level[offset] + level[offset + 1])
        if len(level) % 2:
            next_level.append(level[-1])
        level = next_level
    return level[0]


def _compensated_add(
    total: Tensor,
    correction: Tensor,
    value: Tensor,
) -> tuple[Tensor, Tensor]:
    updated = total + value
    use_total = torch.abs(total) >= torch.abs(value)
    increment = torch.where(
        use_total,
        (total - updated) + value,
        (value - updated) + total,
    )
    return updated, correction + increment


def _chan_merge(
    left_count: int,
    left_mean: Tensor,
    left_m2: Tensor,
    right_count: int,
    right_mean: Tensor,
    right_m2: Tensor,
) -> tuple[int, Tensor, Tensor]:
    if left_count == 0:
        return right_count, right_mean, right_m2
    if right_count == 0:
        return left_count, left_mean, left_m2
    count = left_count + right_count
    delta = right_mean - left_mean
    mean = left_mean + delta * (right_count / count)
    m2 = (
        left_m2
        + right_m2
        + delta * delta * (left_count * right_count / count)
    )
    return count, mean, m2


def _fixed_tree_chan(values: list[Tensor]) -> tuple[int, Tensor, Tensor]:
    if not values:
        raise ValueError("Chan reduction requires at least one value")
    states = [(1, value, torch.zeros_like(value)) for value in values]
    while len(states) > 1:
        next_states: list[tuple[int, Tensor, Tensor]] = []
        for offset in range(0, len(states) - 1, 2):
            next_states.append(_chan_merge(*states[offset], *states[offset + 1]))
        if len(states) % 2:
            next_states.append(states[-1])
        states = next_states
    return states[0]


def _build_group_microblock(
    batch_indices: list[int],
    actor_i: list[int],
    actor_j: list[int],
    tokens_ij: list[np.ndarray],
    tokens_ji: list[np.ndarray],
    support_masks: list[np.ndarray],
) -> PairChunk:
    """Materialize one bounded, single-group canonical edge microblock."""

    if not batch_indices or len(batch_indices) > CANONICAL_MICROBLOCK_SIZE:
        raise RuntimeError("invalid group-local edge microblock size")
    if len(set(batch_indices)) != 1:
        raise RuntimeError("group-local edge microblock mixed batch rows")
    return PairChunk(
        np.ascontiguousarray(batch_indices, dtype=np.int64),
        np.ascontiguousarray(actor_i, dtype=np.int64),
        np.ascontiguousarray(actor_j, dtype=np.int64),
        np.ascontiguousarray(np.stack(tokens_ij), dtype=np.float32),
        np.ascontiguousarray(np.stack(tokens_ji), dtype=np.float32),
        np.ascontiguousarray(np.stack(support_masks), dtype=np.bool_),
    )


def _iter_group_microblocks(
    model: _StreamingEdgeModel,
    batch: PreparedActivityBatch,
    *,
    edge_chunk_size: int,
    pair_chunk_stream: _PairChunkStream | None = None,
) -> Iterator[PairChunk]:
    """Reblock a runtime stream by group, independent of batch/chunk packing.

    Runtime chunks are an I/O and scheduling choice.  Numerical reductions and
    incidence controls instead use blocks numbered from zero separately inside
    every canonical group.  The pending state never exceeds 64 edges.
    """

    groups: list[int] = []
    left_actors: list[int] = []
    right_actors: list[int] = []
    forward_tokens: list[np.ndarray] = []
    reverse_tokens: list[np.ndarray] = []
    supports: list[np.ndarray] = []
    pending_group: int | None = None

    def clear() -> None:
        groups.clear()
        left_actors.clear()
        right_actors.clear()
        forward_tokens.clear()
        reverse_tokens.clear()
        supports.clear()

    runtime_chunks = (
        model._iter_pair_chunks(batch, edge_chunk_size=edge_chunk_size)
        if pair_chunk_stream is None
        else pair_chunk_stream.iter_chunks(edge_chunk_size=edge_chunk_size)
    )
    for runtime_chunk in runtime_chunks:
        for edge in range(int(runtime_chunk.batch_indices.shape[0])):
            group = int(runtime_chunk.batch_indices[edge])
            if pending_group is not None and group != pending_group:
                yield _build_group_microblock(
                    groups,
                    left_actors,
                    right_actors,
                    forward_tokens,
                    reverse_tokens,
                    supports,
                )
                clear()
                pending_group = None
            if pending_group is None:
                pending_group = group
            groups.append(group)
            left_actors.append(int(runtime_chunk.actor_i[edge]))
            right_actors.append(int(runtime_chunk.actor_j[edge]))
            forward_tokens.append(runtime_chunk.tokens_ij[edge])
            reverse_tokens.append(runtime_chunk.tokens_ji[edge])
            supports.append(runtime_chunk.support_mask[edge])
            if len(groups) == CANONICAL_MICROBLOCK_SIZE:
                yield _build_group_microblock(
                    groups,
                    left_actors,
                    right_actors,
                    forward_tokens,
                    reverse_tokens,
                    supports,
                )
                clear()
                pending_group = None
    if groups:
        yield _build_group_microblock(
            groups,
            left_actors,
            right_actors,
            forward_tokens,
            reverse_tokens,
            supports,
        )


def _global_edge_identity_digest(actor_count: int) -> bytes:
    payload = bytearray(b"phaseset-incidence-global-edge-census-v1\x00")
    for left in range(actor_count):
        for right in range(left + 1, actor_count):
            payload.extend(left.to_bytes(8, "big", signed=False))
            payload.extend(right.to_bytes(8, "big", signed=False))
    return hashlib.sha256(payload).digest()


def _global_incidence_coefficients(
    length: int,
    *,
    seed: int,
    band: int,
    edge_identity_digest: bytes,
) -> tuple[int, int]:
    if length <= 1:
        return 0, 1
    digest = hashlib.sha256(
        f"phaseset-incidence-global-v1/{seed}/{band}/{length}/".encode()
        + edge_identity_digest
    ).digest()
    offset = int.from_bytes(digest[:8], "big") % length
    step = int.from_bytes(digest[8:16], "big") % length
    if step == 0:
        step = 1
    while math.gcd(step, length) != 1:
        step = (step + 1) % length
        if step == 0:
            step = 1
    if step == 1 and offset == 0:
        offset = 1
    return offset, pow(step, -1, length)


def _global_incidence_seed(model: _StreamingEdgeModel) -> int | None:
    control_spec = getattr(model, "control_spec", None)
    if getattr(control_spec, "feature_policy", None) != "SHUFFLED_INCIDENCE":
        return None
    seed = getattr(model, "incidence_seed", None)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise RuntimeError("global incidence control has an invalid seed")
    return seed


def _build_global_incidence_routing_plan(
    model: _StreamingEdgeModel,
    batch: PreparedActivityBatch,
    *,
    edge_chunk_size: int,
    pair_chunk_stream: _PairChunkStream | None = None,
) -> _GlobalIncidenceRoutingPlan | None:
    """Census all valid endpoint slots without retaining any width-D activation."""

    seed = _global_incidence_seed(model)
    if seed is None:
        return None
    checked = validate_prepared_activity_batch(batch)
    actor_lists: list[list[list[int]]] = [
        [[] for _ in range(BAND_COUNT)] for _ in range(checked.batch_size)
    ]
    for chunk in _iter_group_microblocks(
        model,
        checked,
        edge_chunk_size=edge_chunk_size,
        pair_chunk_stream=pair_chunk_stream,
    ):
        for edge in range(int(chunk.batch_indices.shape[0])):
            group = int(chunk.batch_indices[edge])
            left = int(chunk.actor_i[edge])
            right = int(chunk.actor_j[edge])
            for band in range(BAND_COUNT):
                if bool(chunk.support_mask[edge, band]):
                    actor_lists[group][band].extend((left, right))
    targets: list[tuple[np.ndarray, ...]] = []
    offsets: list[tuple[int, ...]] = []
    inverse_steps: list[tuple[int, ...]] = []
    for group, actor_count in enumerate(checked.actor_counts):
        digest = _global_edge_identity_digest(actor_count)
        group_targets: list[np.ndarray] = []
        group_offsets: list[int] = []
        group_inverses: list[int] = []
        for band in range(BAND_COUNT):
            values = np.ascontiguousarray(actor_lists[group][band], dtype=np.int64)
            offset, inverse = _global_incidence_coefficients(
                int(values.shape[0]),
                seed=seed,
                band=band,
                edge_identity_digest=digest,
            )
            group_targets.append(values)
            group_offsets.append(offset)
            group_inverses.append(inverse)
        targets.append(tuple(group_targets))
        offsets.append(tuple(group_offsets))
        inverse_steps.append(tuple(group_inverses))
    return _GlobalIncidenceRoutingPlan(
        tuple(targets),
        tuple(offsets),
        tuple(inverse_steps),
    )


def _incidence_cursors(batch: PreparedActivityBatch) -> list[list[int]]:
    return [[0 for _ in range(BAND_COUNT)] for _ in range(batch.batch_size)]


def _routed_half_edge_actors(
    plan: _GlobalIncidenceRoutingPlan | None,
    cursors: list[list[int]],
    *,
    group: int,
    band: int,
    left: int,
    right: int,
) -> tuple[int, int]:
    if plan is None:
        return left, right
    source_rank = cursors[group][band]
    cursors[group][band] += 2
    return (
        plan.target_actor(group, band, source_rank),
        plan.target_actor(group, band, source_rank + 1),
    )


def _assert_incidence_census_consumed(
    plan: _GlobalIncidenceRoutingPlan | None,
    cursors: list[list[int]],
) -> None:
    if plan is None:
        return
    for group, bands in enumerate(plan.target_actors):
        for band, actors in enumerate(bands):
            if cursors[group][band] != int(actors.shape[0]):
                raise RuntimeError("global incidence routing did not consume its census")


def _edge_parameters(model: _StreamingEdgeModel) -> tuple[nn.Parameter, ...]:
    parameters = tuple(model.half_edge_encoder.parameters()) + tuple(
        model.pair_encoder.parameters()
    )
    if not parameters or len({id(parameter) for parameter in parameters}) != len(
        parameters
    ):
        raise RuntimeError("edge encoders must expose distinct registered parameters")
    return parameters


def _compute_edge_summaries(
    model: _StreamingEdgeModel,
    batch: PreparedActivityBatch,
    *,
    edge_chunk_size: int,
    pair_chunk_stream: _PairChunkStream | None = None,
) -> tuple[StreamingEdgeSummaries, _MomentState]:
    """Run the exact canonical forward reductions under the caller's grad mode."""

    checked = validate_prepared_activity_batch(batch)
    device = next(model.parameters()).device
    batch_size, padded_actors = checked.actor_mask.shape
    width = model.embedding_dim
    incidence_plan = _build_global_incidence_routing_plan(
        model,
        checked,
        edge_chunk_size=edge_chunk_size,
        pair_chunk_stream=pair_chunk_stream,
    )
    incidence_cursors = _incidence_cursors(checked)

    pair_sums = [
        [
            torch.zeros((width,), dtype=torch.float64, device=device)
            for _ in range(BAND_COUNT)
        ]
        for _ in range(batch_size)
    ]
    pair_corrections = [
        [
            torch.zeros((width,), dtype=torch.float64, device=device)
            for _ in range(BAND_COUNT)
        ]
        for _ in range(batch_size)
    ]
    pair_counts = [[0 for _ in range(BAND_COUNT)] for _ in range(batch_size)]
    node_means = [
        [
            [
                torch.zeros((width,), dtype=torch.float64, device=device)
                for _ in range(BAND_COUNT)
            ]
            for _ in range(padded_actors)
        ]
        for _ in range(batch_size)
    ]
    node_m2 = [
        [
            [
                torch.zeros((width,), dtype=torch.float64, device=device)
                for _ in range(BAND_COUNT)
            ]
            for _ in range(padded_actors)
        ]
        for _ in range(batch_size)
    ]
    node_degrees = [
        [[0 for _ in range(BAND_COUNT)] for _ in range(padded_actors)]
        for _ in range(batch_size)
    ]

    for chunk in _iter_group_microblocks(
        model,
        checked,
        edge_chunk_size=edge_chunk_size,
        pair_chunk_stream=pair_chunk_stream,
    ):
        edge_count = int(chunk.batch_indices.shape[0])
        for micro_start in range(0, edge_count, CANONICAL_MICROBLOCK_SIZE):
            micro_stop = min(
                micro_start + CANONICAL_MICROBLOCK_SIZE,
                edge_count,
            )
            relation_ij = torch.tensor(
                chunk.tokens_ij[micro_start:micro_stop],
                dtype=torch.float32,
                device=device,
            )
            relation_ji = torch.tensor(
                chunk.tokens_ji[micro_start:micro_stop],
                dtype=torch.float32,
                device=device,
            )
            half_ij, half_ji, pair_tokens = model._edge_forward(
                relation_ij,
                relation_ji,
            )
            half_ij, half_ji, pair_tokens = model._transform_edge_outputs(
                half_ij,
                half_ji,
                pair_tokens,
                chunk.support_mask[micro_start:micro_stop],
                batch_indices=chunk.batch_indices[micro_start:micro_stop],
                actor_i=chunk.actor_i[micro_start:micro_stop],
                actor_j=chunk.actor_j[micro_start:micro_stop],
            )

            pair_block: dict[tuple[int, int], list[Tensor]] = {}
            node_block: dict[tuple[int, int, int], list[Tensor]] = {}
            for local_edge, edge in enumerate(range(micro_start, micro_stop)):
                group = int(chunk.batch_indices[edge])
                left = int(chunk.actor_i[edge])
                right = int(chunk.actor_j[edge])
                for band in range(BAND_COUNT):
                    if not bool(chunk.support_mask[edge, band]):
                        continue
                    pair_block.setdefault((group, band), []).append(
                        pair_tokens[local_edge, band].to(dtype=torch.float64)
                    )
                    routed_left, routed_right = _routed_half_edge_actors(
                        incidence_plan,
                        incidence_cursors,
                        group=group,
                        band=band,
                        left=left,
                        right=right,
                    )
                    node_block.setdefault((group, routed_left, band), []).append(
                        half_ij[local_edge, band].to(dtype=torch.float64)
                    )
                    node_block.setdefault((group, routed_right, band), []).append(
                        half_ji[local_edge, band].to(dtype=torch.float64)
                    )

            for group, band in sorted(pair_block):
                values = pair_block[(group, band)]
                block_sum = _fixed_binary_tree_sum(values)
                pair_sums[group][band], pair_corrections[group][band] = (
                    _compensated_add(
                        pair_sums[group][band],
                        pair_corrections[group][band],
                        block_sum,
                    )
                )
                pair_counts[group][band] += len(values)

            for group, actor, band in sorted(node_block):
                block_state = _fixed_tree_chan(node_block[(group, actor, band)])
                merged = _chan_merge(
                    node_degrees[group][actor][band],
                    node_means[group][actor][band],
                    node_m2[group][actor][band],
                    *block_state,
                )
                node_degrees[group][actor][band] = merged[0]
                node_means[group][actor][band] = merged[1]
                node_m2[group][actor][band] = merged[2]

    _assert_incidence_census_consumed(incidence_plan, incidence_cursors)

    pair_rows: list[Tensor] = []
    for group in range(batch_size):
        bands: list[Tensor] = []
        for band in range(BAND_COUNT):
            count = pair_counts[group][band]
            value = (
                (pair_sums[group][band] + pair_corrections[group][band]) / count
                if count
                else torch.zeros((width,), dtype=torch.float64, device=device)
            )
            bands.append(value.to(dtype=torch.float32))
        pair_rows.append(torch.stack(bands))
    pair_component = torch.stack(pair_rows)
    pair_count_tensor = torch.tensor(
        pair_counts,
        dtype=torch.int64,
        device=device,
    )

    global_halfedge_means: list[list[Tensor]] = []
    for group, actor_count in enumerate(checked.actor_counts):
        group_means: list[Tensor] = []
        for band in range(BAND_COUNT):
            count = 0
            mean = torch.zeros((width,), dtype=torch.float64, device=device)
            m2 = torch.zeros((width,), dtype=torch.float64, device=device)
            for actor in range(actor_count):
                count, mean, m2 = _chan_merge(
                    count,
                    mean,
                    m2,
                    node_degrees[group][actor][band],
                    node_means[group][actor][band],
                    node_m2[group][actor][band],
                )
            group_means.append(mean)
        global_halfedge_means.append(group_means)

    statistic_groups: list[Tensor] = []
    topology_masks: list[list[list[bool]]] = []
    for group, actor_count in enumerate(checked.actor_counts):
        actor_statistics: list[Tensor] = []
        actor_topology_masks: list[list[bool]] = []
        possible_neighbors = actor_count - 1
        for actor in range(padded_actors):
            band_statistics: list[Tensor] = []
            band_topology_masks: list[bool] = []
            for band in range(BAND_COUNT):
                degree = node_degrees[group][actor][band] if actor < actor_count else 0
                if degree:
                    mean = node_means[group][actor][band]
                    moment = torch.clamp(
                        node_m2[group][actor][band] / degree,
                        min=0.0,
                    )
                else:
                    mean = torch.zeros(
                        (width,), dtype=torch.float64, device=device
                    )
                    moment = torch.zeros(
                        (width,), dtype=torch.float64, device=device
                    )
                delta_mean = mean - global_halfedge_means[group][band]
                coverage = degree / possible_neighbors if actor < actor_count else 0.0
                scalars = torch.tensor(
                    (coverage, 1.0 - coverage),
                    dtype=torch.float64,
                    device=device,
                )
                band_statistics.append(torch.cat((delta_mean, moment, scalars)))
                band_topology_masks.append(actor < actor_count and degree >= 2)
            actor_statistics.append(torch.stack(band_statistics))
            actor_topology_masks.append(band_topology_masks)
        statistic_groups.append(torch.stack(actor_statistics))
        topology_masks.append(actor_topology_masks)
    statistics = torch.stack(statistic_groups).to(dtype=torch.float32)
    topology_mask = torch.tensor(topology_masks, dtype=torch.bool, device=device)

    moment_state = _MomentState(
        node_mean=torch.stack(
            [
                torch.stack(
                    [torch.stack(actor_bands) for actor_bands in group_actors]
                )
                for group_actors in node_means
            ]
        ),
        node_m2=torch.stack(
            [
                torch.stack(
                    [torch.stack(actor_bands) for actor_bands in group_actors]
                )
                for group_actors in node_m2
            ]
        ),
        node_degree=torch.tensor(node_degrees, dtype=torch.int64, device=device),
        pair_count=pair_count_tensor,
    )
    return (
        StreamingEdgeSummaries(
            pair_component=pair_component.contiguous(),
            node_statistics=statistics.contiguous(),
            valid_pair_count=pair_count_tensor.contiguous(),
            topology_node_mask=topology_mask.contiguous(),
        ),
        moment_state,
    )


def eager_edge_summaries(
    model: _StreamingEdgeModel,
    batch: PreparedActivityBatch,
    *,
    edge_chunk_size: int,
    pair_chunk_stream: _PairChunkStream | None = None,
) -> StreamingEdgeSummaries:
    """Autograd-heavy reference path used to audit the analytical VJP."""

    summaries, _ = _compute_edge_summaries(
        model,
        batch,
        edge_chunk_size=edge_chunk_size,
        pair_chunk_stream=pair_chunk_stream,
    )
    return summaries


def _stream_parameter_vjp(
    model: _StreamingEdgeModel,
    batch: PreparedActivityBatch,
    *,
    edge_chunk_size: int,
    pair_chunk_stream: _PairChunkStream | None,
    state: _MomentState,
    grad_pair_component: Tensor | None,
    grad_node_statistics: Tensor | None,
    parameters: tuple[Tensor, ...],
) -> tuple[Tensor, ...]:
    """Replay edges and free each microblock graph immediately after its VJP."""

    checked = validate_prepared_activity_batch(batch)
    device = next(model.parameters()).device
    width = model.embedding_dim
    incidence_plan = _build_global_incidence_routing_plan(
        model,
        checked,
        edge_chunk_size=edge_chunk_size,
        pair_chunk_stream=pair_chunk_stream,
    )
    incidence_cursors = _incidence_cursors(checked)
    if grad_pair_component is None:
        grad_pair64 = torch.zeros(
            (checked.batch_size, BAND_COUNT, width),
            dtype=torch.float64,
            device=device,
        )
    else:
        grad_pair64 = grad_pair_component.to(device=device, dtype=torch.float64)
    if grad_node_statistics is None:
        grad_statistics64 = torch.zeros(
            (
                checked.batch_size,
                checked.padded_actor_count,
                BAND_COUNT,
                2 * width + 2,
            ),
            dtype=torch.float64,
            device=device,
        )
    else:
        grad_statistics64 = grad_node_statistics.to(
            device=device,
            dtype=torch.float64,
        )
    grad_delta = grad_statistics64[..., :width]
    grad_moment = grad_statistics64[..., width : 2 * width]
    node_degrees = state.node_degree.tolist()
    pair_counts = state.pair_count.tolist()

    global_coefficients = torch.zeros(
        (checked.batch_size, BAND_COUNT, width),
        dtype=torch.float64,
        device=device,
    )
    for group in range(checked.batch_size):
        for band in range(BAND_COUNT):
            total_degree = sum(
                int(node_degrees[group][actor][band])
                for actor in range(checked.padded_actor_count)
            )
            if total_degree:
                global_coefficients[group, band] = (
                    -grad_delta[group, :, band].sum(dim=0) / total_degree
                )

    accumulated = [torch.zeros_like(parameter) for parameter in parameters]
    for chunk in _iter_group_microblocks(
        model,
        checked,
        edge_chunk_size=edge_chunk_size,
        pair_chunk_stream=pair_chunk_stream,
    ):
        edge_count = int(chunk.batch_indices.shape[0])
        for micro_start in range(0, edge_count, CANONICAL_MICROBLOCK_SIZE):
            micro_stop = min(
                micro_start + CANONICAL_MICROBLOCK_SIZE,
                edge_count,
            )
            with torch.enable_grad():
                relation_ij = torch.tensor(
                    chunk.tokens_ij[micro_start:micro_stop],
                    dtype=torch.float32,
                    device=device,
                )
                relation_ji = torch.tensor(
                    chunk.tokens_ji[micro_start:micro_stop],
                    dtype=torch.float32,
                    device=device,
                )
                half_ij, half_ji, pair_tokens = model._edge_forward(
                    relation_ij,
                    relation_ji,
                )
                half_ij, half_ji, pair_tokens = model._transform_edge_outputs(
                    half_ij,
                    half_ji,
                    pair_tokens,
                    chunk.support_mask[micro_start:micro_stop],
                    batch_indices=chunk.batch_indices[micro_start:micro_stop],
                    actor_i=chunk.actor_i[micro_start:micro_stop],
                    actor_j=chunk.actor_j[micro_start:micro_stop],
                )

                objective = pair_tokens.sum() * 0.0
                for local_edge, edge in enumerate(range(micro_start, micro_stop)):
                    group = int(chunk.batch_indices[edge])
                    left = int(chunk.actor_i[edge])
                    right = int(chunk.actor_j[edge])
                    for band in range(BAND_COUNT):
                        if not bool(chunk.support_mask[edge, band]):
                            continue
                        pair_count = int(pair_counts[group][band])
                        pair_coefficient = (
                            grad_pair64[group, band] / pair_count
                        ).to(dtype=torch.float32)
                        objective = objective + torch.sum(
                            pair_tokens[local_edge, band] * pair_coefficient
                        )

                        routed_left, routed_right = _routed_half_edge_actors(
                            incidence_plan,
                            incidence_cursors,
                            group=group,
                            band=band,
                            left=left,
                            right=right,
                        )
                        for actor, half_edge in (
                            (routed_left, half_ij[local_edge, band]),
                            (routed_right, half_ji[local_edge, band]),
                        ):
                            degree = int(node_degrees[group][actor][band])
                            coefficient = (
                                grad_delta[group, actor, band] / degree
                                + global_coefficients[group, band]
                            )
                            raw_moment = state.node_m2[group, actor, band] / degree
                            clamp_active = raw_moment >= 0.0
                            moment_term = (
                                2.0
                                * grad_moment[group, actor, band]
                                * (
                                    half_edge.detach().to(dtype=torch.float64)
                                    - state.node_mean[group, actor, band]
                                )
                                / degree
                            )
                            coefficient = coefficient + torch.where(
                                clamp_active,
                                moment_term,
                                torch.zeros_like(moment_term),
                            )
                            objective = objective + torch.sum(
                                half_edge * coefficient.to(dtype=torch.float32)
                            )

                local_gradients = torch.autograd.grad(
                    objective,
                    parameters,
                    allow_unused=True,
                )
            for index, gradient in enumerate(local_gradients):
                if gradient is not None:
                    accumulated[index].add_(gradient)
            del (
                half_ij,
                half_ji,
                local_gradients,
                objective,
                pair_tokens,
                relation_ij,
                relation_ji,
            )
    _assert_incidence_census_consumed(incidence_plan, incidence_cursors)
    return tuple(accumulated)


class _StreamingEdgeSummaryFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        model: _StreamingEdgeModel,
        batch: PreparedActivityBatch,
        edge_chunk_size: int,
        pair_chunk_stream: _PairChunkStream | None,
        *parameters: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        ctx.set_materialize_grads(False)
        ctx.model = model
        ctx.batch = batch
        ctx.edge_chunk_size = edge_chunk_size
        ctx.pair_chunk_stream = pair_chunk_stream
        device_type = next(model.parameters()).device.type
        ctx.autocast_device_type = device_type
        ctx.autocast_enabled = torch.is_autocast_enabled(device_type)
        ctx.autocast_dtype = torch.get_autocast_dtype(device_type)
        ctx.autocast_cache_enabled = torch.is_autocast_cache_enabled()
        ctx.save_for_backward(*parameters)
        summaries, _ = _compute_edge_summaries(
            model,
            batch,
            edge_chunk_size=edge_chunk_size,
            pair_chunk_stream=pair_chunk_stream,
        )
        ctx.mark_non_differentiable(
            summaries.valid_pair_count,
            summaries.topology_node_mask,
        )
        return (
            summaries.pair_component,
            summaries.node_statistics,
            summaries.valid_pair_count,
            summaries.topology_node_mask,
        )

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any,
        grad_pair_component: Tensor | None,
        grad_node_statistics: Tensor | None,
        grad_valid_pair_count: Tensor | None,
        grad_topology_node_mask: Tensor | None,
    ) -> tuple[None, None, None, None, *tuple[Tensor, ...]]:
        del grad_valid_pair_count, grad_topology_node_mask
        parameters = cast(tuple[Tensor, ...], ctx.saved_tensors)
        with torch.autocast(
            device_type=ctx.autocast_device_type,
            enabled=ctx.autocast_enabled,
            dtype=ctx.autocast_dtype,
            cache_enabled=ctx.autocast_cache_enabled,
        ):
            with torch.no_grad():
                _, state = _compute_edge_summaries(
                    ctx.model,
                    ctx.batch,
                    edge_chunk_size=ctx.edge_chunk_size,
                    pair_chunk_stream=ctx.pair_chunk_stream,
                )
        # Exit and re-enter autocast so a backend cannot reuse a no-grad cached
        # low-precision parameter view during the differentiable replay.
        with torch.autocast(
            device_type=ctx.autocast_device_type,
            enabled=ctx.autocast_enabled,
            dtype=ctx.autocast_dtype,
            cache_enabled=ctx.autocast_cache_enabled,
        ):
            gradients = _stream_parameter_vjp(
                ctx.model,
                ctx.batch,
                edge_chunk_size=ctx.edge_chunk_size,
                pair_chunk_stream=ctx.pair_chunk_stream,
                state=state,
                grad_pair_component=grad_pair_component,
                grad_node_statistics=grad_node_statistics,
                parameters=parameters,
            )
        return (None, None, None, None, *gradients)


def streaming_edge_summaries(
    model: _StreamingEdgeModel,
    batch: PreparedActivityBatch,
    *,
    edge_chunk_size: int,
    pair_chunk_stream: _PairChunkStream | None = None,
) -> StreamingEdgeSummaries:
    """Return exact summaries with chunk-recomputed edge-parameter gradients."""

    checked = validate_prepared_activity_batch(batch)
    all_parameters = _edge_parameters(model)
    parameters = tuple(
        parameter for parameter in all_parameters if parameter.requires_grad
    )
    if not torch.is_grad_enabled() or not parameters:
        summaries, _ = _compute_edge_summaries(
            model,
            checked,
            edge_chunk_size=edge_chunk_size,
            pair_chunk_stream=pair_chunk_stream,
        )
        return summaries
    outputs = _StreamingEdgeSummaryFunction.apply(
        model,
        checked,
        edge_chunk_size,
        pair_chunk_stream,
        *parameters,
    )
    return StreamingEdgeSummaries(*outputs)


__all__ = [
    "StreamingEdgeSummaries",
    "streaming_edge_summaries",
]
