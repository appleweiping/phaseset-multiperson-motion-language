"""Temporal actor-edge incidence, ordered capture readout and language loss.

This is the V2 research module, not a licensed-data runner or a trained model.
Edges are processed in fixed 64-edge blocks. Checkpointed blocks recompute
physical relations and neural activations in backward rather than retaining
all edge-token activations. All-pair time remains quadratic in actor count.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .directional_phase import (
    ACTOR_FEATURE_DIM,
    PHASE_FEATURE_DIM,
    DirectionalPhaseField,
    local_pair_chunk,
)
from .objectives import variable_positive_symmetric_infonce


class MaskedTemporalEncoder(nn.Module):
    """Two shared temporal layers; missing positions do not update either state.

    ``encode_chunk`` carries both layers across ordered chunks of one track.
    Reset its state for a different capture or a declared track discontinuity.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.cells = nn.ModuleList([nn.GRUCell(width, width), nn.GRUCell(width, width)])

    def forward(self, sequence: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        outputs, states = self.encode_chunk(sequence, mask)
        return outputs, states[-1]

    def encode_chunk(
        self, sequence: Tensor, mask: Tensor, states: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        if sequence.ndim != 3 or mask.shape != sequence.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("temporal inputs require [N,P,D] and bool[N,P]")
        expected = (len(self.cells), sequence.shape[0], sequence.shape[2])
        if states is None:
            states = sequence.new_zeros(expected)
        elif states.shape != expected or states.dtype != sequence.dtype or states.device != sequence.device:
            raise ValueError("chunk states must match [2,N,D], dtype and device")
        layer_states = list(states.unbind(0))
        outputs = []
        for position in range(sequence.shape[1]):
            valid = mask[:, position, None]
            value = sequence[:, position]
            for layer, cell in enumerate(self.cells):
                proposal = cell(value, layer_states[layer])
                layer_states[layer] = torch.where(valid, proposal, layer_states[layer])
                value = layer_states[layer]
            outputs.append(torch.where(valid, value, torch.zeros_like(value)))
        if not outputs:
            raise ValueError("temporal sequence is empty")
        return torch.stack(outputs, dim=1), torch.stack(layer_states)


@dataclass(frozen=True)
class TemporalCoordinationOutput:
    patch_tokens: Tensor
    patch_mask: Tensor
    embedding: Tensor
    topology_nodes: Tensor
    topology_mask: Tensor
    valid_pair_count: Tensor


class TemporalIncidenceEncoder(nn.Module):
    """Shared endpoint-bound relations followed along their time trajectories."""

    def __init__(
        self,
        width: int = 512,
        *,
        use_topology: bool = True,
        strip_phase: bool = False,
        checkpoint_blocks: bool = True,
    ) -> None:
        super().__init__()
        if type(width) is not int or width < 2:
            raise ValueError("width must be an integer >=2")
        self.width = width
        self.use_topology = use_topology
        self.strip_phase = strip_phase
        self.checkpoint_blocks = checkpoint_blocks
        self.actor_projection = nn.Sequential(nn.Linear(ACTOR_FEATURE_DIM, width), nn.GELU())
        self.half_edge = nn.Sequential(
            nn.Linear(2 * width + 6 * PHASE_FEATURE_DIM + 3, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.symmetric_pair = nn.Sequential(nn.Linear(3 * width, width), nn.GELU())
        self.node_mlp = nn.Sequential(
            nn.Linear(3 * width + 2, width), nn.GELU(), nn.Linear(width, width)
        )
        self.node_temporal = MaskedTemporalEncoder(width)
        self.edge_delta = nn.Sequential(
            nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, width)
        )
        self.edge_temporal = MaskedTemporalEncoder(width)
        self.group_temporal = MaskedTemporalEncoder(width)
        self.topology_scale = nn.Parameter(torch.tensor(0.1))

    def _half_edges(
        self, field: DirectionalPhaseField, hidden: Tensor, start: int, stop: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        chunk = local_pair_chunk(field, start, stop)
        device = hidden.device
        endpoints = torch.tensor(chunk.endpoints.copy(), dtype=torch.int64, device=device)
        forward = torch.tensor(chunk.features.copy(), dtype=hidden.dtype, device=device)
        reverse = torch.tensor(chunk.reverse_features(), dtype=hidden.dtype, device=device)
        if self.strip_phase:
            forward[..., :2] = 0.0
            reverse[..., :2] = 0.0
        left, right = endpoints[:, 0], endpoints[:, 1]
        roots = torch.tensor(field.root_positions.copy(), dtype=hidden.dtype, device=device)
        root_mask = torch.tensor(field.root_patch_mask.copy(), device=device)
        relative_root = roots[right] - roots[left]
        relative_root = torch.where(
            (root_mask[left] & root_mask[right])[..., None],
            relative_root,
            torch.zeros_like(relative_root),
        )
        left_hidden, right_hidden = hidden[left], hidden[right]
        a = self.half_edge(
            torch.cat(
                (
                    left_hidden,
                    right_hidden,
                    forward.flatten(start_dim=2),
                    relative_root,
                ),
                dim=-1,
            )
        )
        b = self.half_edge(
            torch.cat(
                (
                    right_hidden,
                    left_hidden,
                    reverse.flatten(start_dim=2),
                    -relative_root,
                ),
                dim=-1,
            )
        )
        mask = torch.tensor(chunk.support_mask.any(axis=-1).copy(), device=device)
        actor_mask = torch.tensor(field.actor_patch_mask.copy(), device=device)
        mask = mask & actor_mask[left] & actor_mask[right]
        return (
            torch.where(mask[..., None], a, torch.zeros_like(a)),
            torch.where(mask[..., None], b, torch.zeros_like(b)),
            endpoints,
            mask,
        )

    def _node_block(
        self, field: DirectionalPhaseField, hidden: Tensor, start: int, stop: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        a, b, endpoints, mask = self._half_edges(field, hidden, start, stop)
        # K x 64 incidence block, not an all-pair K x K adjacency or score map.
        left = F.one_hot(endpoints[:, 0], field.actor_count).T.to(torch.float64)
        right = F.one_hot(endpoints[:, 1], field.actor_count).T.to(torch.float64)
        shape = (field.actor_count, field.patch_count, self.width)
        a64, b64 = a.to(torch.float64), b.to(torch.float64)
        total = (left @ a64.flatten(1) + right @ b64.flatten(1)).reshape(shape)
        second = (left @ a64.square().flatten(1) + right @ b64.square().flatten(1)).reshape(shape)
        degree = (left + right) @ mask.to(torch.float64)
        return total, second, degree

    def _edge_block(
        self,
        field: DirectionalPhaseField,
        hidden: Tensor,
        nodes: Tensor,
        topology_mask: Tensor,
        start: int,
        stop: int,
    ) -> tuple[Tensor, Tensor]:
        a, b, endpoints, mask = self._half_edges(field, hidden, start, stop)
        pair = self.symmetric_pair(torch.cat((a + b, (a - b).abs(), a * b), dim=-1))
        left_nodes, right_nodes = nodes[endpoints[:, 0]], nodes[endpoints[:, 1]]
        delta = self.edge_delta(
            torch.cat(
                (pair, left_nodes + right_nodes, (left_nodes - right_nodes).abs()),
                dim=-1,
            )
        )
        eligible = topology_mask[endpoints[:, 0]] | topology_mask[endpoints[:, 1]]
        delta = torch.where(eligible[..., None], delta, torch.zeros_like(delta))
        encoded, _ = self.edge_temporal(pair + self.topology_scale.tanh() * delta, mask)
        return encoded.to(torch.float64).sum(dim=0), mask.to(torch.int64).sum(dim=0)

    def _run_block(self, function, *inputs):
        if self.checkpoint_blocks and torch.is_grad_enabled():
            return checkpoint(function, *inputs, use_reentrant=False)
        return function(*inputs)

    def forward(self, field: DirectionalPhaseField) -> TemporalCoordinationOutput:
        device = next(self.parameters()).device
        actor_values = torch.tensor(field.actor_features.copy(), dtype=torch.float32, device=device)
        actor_mask = torch.tensor(field.actor_patch_mask.copy(), device=device)
        hidden = self.actor_projection(actor_values)
        hidden = torch.where(actor_mask[..., None], hidden, torch.zeros_like(hidden))
        shape = (field.actor_count, field.patch_count, self.width)
        total = torch.zeros(shape, dtype=torch.float64, device=device)
        second = torch.zeros_like(total)
        degree = torch.zeros(shape[:2], dtype=torch.float64, device=device)
        for start in range(0, field.pair_count, 64):
            stop = min(field.pair_count, start + 64)

            def node_block(value, block_start=start, block_stop=stop):
                return self._node_block(field, value, block_start, block_stop)

            local_total, local_second, local_degree = self._run_block(node_block, hidden)
            total, second, degree = (
                total + local_total,
                second + local_second,
                degree + local_degree,
            )
        denominator = degree.clamp_min(1)[..., None]
        mean = total / denominator
        m2 = (second / denominator - mean.square()).clamp_min(0)
        global_mean = total.sum(dim=0) / degree.sum(dim=0).clamp_min(1)[:, None]
        coverage = degree / (field.actor_count - 1)
        stats = torch.cat(
            (
                hidden,
                (mean - global_mean).to(hidden.dtype),
                m2.to(hidden.dtype),
                coverage[..., None].to(hidden.dtype),
                (1 - coverage)[..., None].to(hidden.dtype),
            ),
            dim=-1,
        )
        topology_mask = actor_mask & (degree >= 2) & self.use_topology
        nodes = self.node_mlp(stats)
        nodes = torch.where(topology_mask[..., None], nodes, torch.zeros_like(nodes))
        nodes, _ = self.node_temporal(nodes, topology_mask)
        nodes = torch.where(topology_mask[..., None], nodes, torch.zeros_like(nodes))
        patch_sum = torch.zeros((field.patch_count, self.width), dtype=torch.float64, device=device)
        pair_count = torch.zeros(field.patch_count, dtype=torch.int64, device=device)
        for start in range(0, field.pair_count, 64):
            stop = min(field.pair_count, start + 64)

            def edge_block(value, node_value, block_start=start, block_stop=stop):
                return self._edge_block(
                    field, value, node_value, topology_mask, block_start, block_stop
                )

            local_sum, local_count = self._run_block(edge_block, hidden, nodes)
            patch_sum, pair_count = patch_sum + local_sum, pair_count + local_count
        patch_mask = pair_count > 0
        patches = (patch_sum / pair_count.clamp_min(1)[:, None]).to(hidden.dtype)
        patches, embedding = self.group_temporal(patches[None], patch_mask[None])
        return TemporalCoordinationOutput(
            F.normalize(patches[0], dim=-1),
            patch_mask,
            F.normalize(embedding[0], dim=-1),
            nodes,
            topology_mask,
            pair_count,
        )


class OrderedCaptureReadout(nn.Module):
    """Sort by physical window start, then encode the complete ordered capture."""

    def __init__(self, width: int = 512) -> None:
        super().__init__()
        self.temporal = MaskedTemporalEncoder(width)

    def forward(self, windows: Tensor, start_seconds: Tensor, mask: Tensor) -> Tensor:
        if (
            windows.ndim != 2
            or start_seconds.shape != windows.shape[:1]
            or mask.shape != windows.shape[:1]
        ):
            raise ValueError("capture needs [W,D] windows and [W] starts/mask")
        if mask.dtype != torch.bool or not bool(torch.isfinite(start_seconds).all()):
            raise ValueError("capture starts must be finite and mask bool")
        ordered = torch.argsort(start_seconds, stable=True)
        observed_starts = start_seconds[ordered][mask[ordered]]
        if observed_starts.numel() > 1 and bool(
            (observed_starts[1:] <= observed_starts[:-1]).any()
        ):
            raise ValueError("valid capture windows must have distinct start times")
        _, embedding = self.temporal(windows[ordered][None], mask[ordered][None])
        return F.normalize(embedding[0], dim=-1)


class CalibratedCoordinationScore(nn.Module):
    """Blend both branches as cosines on one learned similarity scale."""

    def __init__(self, initial_temperature: float = 0.07, initial_mixture: float = 0.1) -> None:
        super().__init__()
        if not math.isfinite(initial_temperature) or initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive")
        if not math.isfinite(initial_mixture) or not 0 < initial_mixture < 1:
            raise ValueError("initial_mixture must be strictly between zero and one")
        self.mixture_logit = nn.Parameter(torch.tensor(math.log(initial_mixture / (1 - initial_mixture))))
        self.log_scale = nn.Parameter(torch.tensor(math.log(1 / initial_temperature)))

    def forward(
        self,
        global_cosine: Tensor,
        coordination_cosine: Tensor,
        *,
        coordination_support: Tensor | None = None,
    ) -> Tensor:
        if global_cosine.shape != coordination_cosine.shape:
            raise ValueError("both cosine score matrices must have the same shape")
        for value in (global_cosine, coordination_cosine):
            if not bool(torch.isfinite(value).all()) or bool((value.abs() > 1.00001).any()):
                raise ValueError("calibration consumes cosines, not temperature-scaled logits")
        alpha = self.mixture_logit.sigmoid()
        if coordination_support is not None:
            if coordination_support.dtype != torch.bool:
                raise ValueError("coordination_support must be boolean")
            if coordination_support.shape == global_cosine.shape[:1] and global_cosine.ndim == 2:
                coordination_support = coordination_support[:, None]
            elif coordination_support.shape != global_cosine.shape:
                raise ValueError("coordination_support must match scores or the motion row axis")
            alpha = torch.where(coordination_support, alpha, torch.zeros_like(alpha))
        scale = self.log_scale.clamp(math.log(1e-3), math.log(100)).exp()
        return scale * ((1 - alpha) * global_cosine + alpha * coordination_cosine)


def complete_relation_text_scores(
    relation_tokens: Tensor, relation_mask: Tensor, text_embeddings: Tensor
) -> Tensor:
    """Match whole endpoint-bound tokens to text; never recombine separate maxima."""
    if relation_tokens.ndim != 3 or relation_mask.shape != relation_tokens.shape[:2]:
        raise ValueError("relations need [B,R,D] tokens and [B,R] mask")
    if relation_mask.dtype != torch.bool or text_embeddings.ndim != 2:
        raise ValueError("relation mask must be bool and text embeddings [M,D]")
    if relation_tokens.shape[-1] != text_embeddings.shape[-1]:
        raise ValueError("relation and text projections must share a width")
    similarities = torch.einsum(
        "brd,md->brm",
        F.normalize(relation_tokens, dim=-1),
        F.normalize(text_embeddings, dim=-1),
    )
    selected = similarities.masked_fill(~relation_mask[..., None], -torch.inf).amax(dim=1)
    return torch.where(relation_mask.any(dim=1)[:, None], selected, torch.zeros_like(selected))


def verified_counterfactual_loss(
    positive_scores: Tensor,
    negative_scores: Tensor,
    verified_negative_mask: Tensor,
    *,
    margin: float = 0.2,
) -> Tensor:
    """A caller must supply a verified-false mask; generic descriptions get no loss."""
    if (
        positive_scores.shape != negative_scores.shape
        or verified_negative_mask.shape != positive_scores.shape
        or verified_negative_mask.dtype != torch.bool
    ):
        raise ValueError("counterfactual scores and bool verified mask must have matching shapes")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and nonnegative")
    penalties = F.softplus(margin + negative_scores - positive_scores)
    if not bool(verified_negative_mask.any()):
        return (positive_scores.sum() + negative_scores.sum()) * 0.0
    return penalties[verified_negative_mask].mean()


def coordination_objective(
    scores: Tensor,
    positive_mask: Tensor,
    *,
    cf_positive_scores: Tensor,
    cf_negative_scores: Tensor,
    verified_negative_mask: Tensor,
    cf_weight: float = 0.2,
    margin: float = 0.2,
) -> Tensor:
    if not math.isfinite(cf_weight) or cf_weight < 0:
        raise ValueError("cf_weight must be finite and nonnegative")
    _, _, contrastive = variable_positive_symmetric_infonce(scores, positive_mask)
    counterfactual = verified_counterfactual_loss(
        cf_positive_scores, cf_negative_scores, verified_negative_mask, margin=margin
    )
    return contrastive + cf_weight * counterfactual
