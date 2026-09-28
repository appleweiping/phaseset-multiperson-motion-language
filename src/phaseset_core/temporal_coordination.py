"""Temporal actor-edge incidence, ordered capture readout and language loss.

This is the V2 research module, not a licensed-data runner or a trained model.
Edges are processed in fixed 64-edge blocks. Checkpointed blocks recompute
physical relations and neural activations in backward rather than retaining
all edge-token activations. All-pair time remains quadratic in actor count.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
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
from .dct_relations import DctRelationField, local_dct_pair_chunk
from .objectives import variable_positive_symmetric_infonce


RelationField = DirectionalPhaseField | DctRelationField


def _relation_pair_chunk(field: RelationField, start: int, stop: int):
    if type(field) is DctRelationField:
        return local_dct_pair_chunk(field, start, stop)
    if type(field) is DirectionalPhaseField:
        return local_pair_chunk(field, start, stop)
    raise TypeError("relation field must be a physical phase or A6 DCT field")


def _shuffle_incidence_values(
    a: Tensor,
    b: Tensor,
    mask: Tensor,
    *,
    seed: int,
    block_start: int,
) -> tuple[Tensor, Tensor]:
    """A4: shuffle supported half-edge values, never their endpoint slots.

    The operation is per capture, patch and canonical edge microblock. It
    preserves every patch's half-edge multiset, edge support, degree and
    coverage; only the values entering the actor-incidence node statistics are
    reassigned. Pair tokens and their temporal/text path remain untouched.
    """
    if (
        a.shape != b.shape
        or a.ndim != 3
        or mask.shape != a.shape[:2]
        or mask.dtype != torch.bool
        or type(seed) is not int
        or not 0 <= seed < 2**63
        or type(block_start) is not int
        or block_start < 0
    ):
        raise ValueError("A4 requires valid support and a fixed seed")
    edges, patches, width = a.shape
    values = torch.stack((a, b), dim=1).permute(2, 0, 1, 3).reshape(patches, 2 * edges, width)
    valid = mask.detach().cpu().T.repeat_interleave(2, dim=1).tolist()
    rows: list[list[int]] = []
    for patch, observed in enumerate(valid):
        mapping = list(range(2 * edges))
        slots = [index for index, present in enumerate(observed) if present]
        length = len(slots)
        if length > 1:
            digest = hashlib.sha256(
                b"phaseset-v2-A4-incidence-v1\0"
                + seed.to_bytes(8, "big")
                + block_start.to_bytes(8, "big")
                + patch.to_bytes(8, "big")
                + length.to_bytes(8, "big")
            ).digest()
            offset = int.from_bytes(digest[:8], "big") % length
            step = int.from_bytes(digest[8:16], "big") % length or 1
            while math.gcd(step, length) != 1:
                step = (step + 1) % length or 1
            if step == 1 and offset == 0:
                offset = 1
            for position, slot in enumerate(slots):
                mapping[slot] = slots[(position * step + offset) % length]
        rows.append(mapping)
    indices = torch.tensor(rows, dtype=torch.int64, device=a.device)
    shuffled = values.gather(1, indices[..., None].expand(-1, -1, width))
    paired = shuffled.reshape(patches, edges, 2, width).permute(1, 2, 0, 3)
    return paired[:, 0].contiguous(), paired[:, 1].contiguous()


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
        elif (
            states.shape != expected
            or states.dtype != sequence.dtype
            or states.device != sequence.device
        ):
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


class OrderFreeTemporalEncoder(nn.Module):
    """A2: equal-capacity independent patch MLP followed by masked mean.

    No state, position embedding, convolution or ordered reduction is used.
    The 6x expansion matches two width-preserving GRUCells within 0.1% in
    trainable parameters, without inert capacity-padding tensors.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        if type(width) is not int or width < 2:
            raise ValueError("width must be an integer >=2")
        self.width = width
        self.pointwise = nn.Sequential(
            nn.Linear(width, 6 * width),
            nn.GELU(),
            nn.Linear(6 * width, width),
        )

    def forward(self, sequence: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        if (
            sequence.ndim != 3
            or sequence.shape[2] != self.width
            or mask.shape != sequence.shape[:2]
            or mask.dtype != torch.bool
            or sequence.shape[1] == 0
        ):
            raise ValueError("A2 requires nonempty [N,P,width] and bool[N,P]")
        outputs = self.pointwise(sequence)
        outputs = torch.where(mask[..., None], outputs, torch.zeros_like(outputs))
        count = mask.sum(dim=1).clamp_min(1)
        embedding = (
            outputs.to(torch.float64).sum(dim=1) / count[:, None].to(torch.float64)
        ).to(outputs.dtype)
        embedding = torch.where(mask.any(dim=1)[:, None], embedding, torch.zeros_like(embedding))
        return outputs, embedding


@dataclass(frozen=True)
class TemporalCoordinationOutput:
    patch_tokens: Tensor
    patch_mask: Tensor
    embedding: Tensor
    topology_nodes: Tensor
    topology_mask: Tensor
    valid_pair_count: Tensor


@dataclass(frozen=True)
class TemporalRelationScores:
    coordination: TemporalCoordinationOutput
    cosine: Tensor
    periodic_support: Tensor


class TemporalIncidenceEncoder(nn.Module):
    """Shared endpoint-bound relations followed along their time trajectories."""

    def __init__(
        self,
        width: int = 512,
        *,
        use_topology: bool = True,
        strip_phase: bool = False,
        incidence_shuffle_seed: int | None = None,
        order_free: bool = False,
        checkpoint_blocks: bool = True,
    ) -> None:
        super().__init__()
        if type(width) is not int or width < 2:
            raise ValueError("width must be an integer >=2")
        if incidence_shuffle_seed is not None and (
            type(incidence_shuffle_seed) is not int
            or not 0 <= incidence_shuffle_seed < 2**63
            or not use_topology
        ):
            raise ValueError("A4 needs a fixed seed and active incidence topology")
        if type(order_free) is not bool or (
            order_free and (not use_topology or strip_phase or incidence_shuffle_seed is not None)
        ):
            raise ValueError("A2 must remove order from the otherwise full phase system")
        self.width = width
        self.use_topology = use_topology
        self.strip_phase = strip_phase
        self.incidence_shuffle_seed = incidence_shuffle_seed
        self.order_free = order_free
        self.checkpoint_blocks = checkpoint_blocks
        temporal_encoder = OrderFreeTemporalEncoder if order_free else MaskedTemporalEncoder
        self.actor_projection = nn.Sequential(nn.Linear(ACTOR_FEATURE_DIM, width), nn.GELU())
        self.actor_temporal = temporal_encoder(width)
        self.half_edge = nn.Sequential(
            nn.Linear(2 * width + 6 * PHASE_FEATURE_DIM + 3, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.symmetric_pair = nn.Sequential(nn.Linear(3 * width, width), nn.GELU())
        self.node_mlp = nn.Sequential(
            nn.Linear(3 * width + 2, width), nn.GELU(), nn.Linear(width, width)
        )
        self.node_temporal = temporal_encoder(width)
        self.edge_delta = nn.Sequential(
            nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, width)
        )
        self.edge_temporal = temporal_encoder(width)
        self.group_temporal = temporal_encoder(width)
        self.topology_scale = nn.Parameter(torch.tensor(0.1))

    def _half_edges(
        self, field: RelationField, hidden: Tensor, start: int, stop: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        chunk = _relation_pair_chunk(field, start, stop)
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
        self, field: RelationField, hidden: Tensor, start: int, stop: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        a, b, endpoints, mask = self._half_edges(field, hidden, start, stop)
        if self.incidence_shuffle_seed is not None:
            if type(field) is not DirectionalPhaseField:
                raise TypeError("A4 requires a physical phase field")
            a, b = _shuffle_incidence_values(
                a,
                b,
                mask,
                seed=self.incidence_shuffle_seed,
                block_start=start,
            )
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
        field: RelationField,
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

    def _encode_context(self, field: RelationField) -> tuple[Tensor, Tensor, Tensor]:
        device = next(self.parameters()).device
        actor_values = torch.tensor(field.actor_features.copy(), dtype=torch.float32, device=device)
        actor_mask = torch.tensor(field.actor_patch_mask.copy(), device=device)
        hidden = self.actor_projection(actor_values)
        hidden = torch.where(actor_mask[..., None], hidden, torch.zeros_like(hidden))
        # Preserve each actor's own history before forming endpoint-bound
        # relations. A field spans one capture; no set pooling or window reset
        # is inserted between local patches along a consistent actor track.
        hidden, _ = self.actor_temporal(hidden, actor_mask)
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
        return hidden, nodes, topology_mask

    def _readout(
        self, field: RelationField, hidden: Tensor, nodes: Tensor, topology_mask: Tensor
    ) -> TemporalCoordinationOutput:
        device = hidden.device
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

    def forward(self, field: RelationField) -> TemporalCoordinationOutput:
        return self._readout(field, *self._encode_context(field))

    def _directed_text_block(
        self,
        field: RelationField,
        hidden: Tensor,
        nodes: Tensor,
        topology_mask: Tensor,
        text: Tensor,
        start: int,
        stop: int,
    ) -> tuple[Tensor, Tensor]:
        a, b, endpoints, track_support = self._half_edges(field, hidden, start, stop)
        left, right = endpoints[:, 0], endpoints[:, 1]
        eligible = topology_mask[left] | topology_mask[right]
        # Unlike the unordered pair readout, language packets retain ordered
        # endpoints: a(i->j), n_i, n_j and b(j->i), n_j, n_i. Both directions
        # share the same update and temporal weights. No actor IDs enter them.
        delta_a = self.edge_delta(torch.cat((a, nodes[left], nodes[right]), dim=-1))
        delta_b = self.edge_delta(torch.cat((b, nodes[right], nodes[left]), dim=-1))
        delta_a = torch.where(eligible[..., None], delta_a, torch.zeros_like(delta_a))
        delta_b = torch.where(eligible[..., None], delta_b, torch.zeros_like(delta_b))
        packets = torch.cat(
            (a + self.topology_scale.tanh() * delta_a, b + self.topology_scale.tanh() * delta_b)
        )
        histories, _ = self.edge_temporal(packets, torch.cat((track_support, track_support)))
        # Zero-energy observations can carry track context but are not reliable
        # physical evidence. Phase uses Morlet floors; A6 uses its independently
        # fitted DCT floors. Neither treats mere track coverage as evidence.
        physical = _relation_pair_chunk(field, start, stop)
        observable = torch.tensor(physical.phase_mask.any(axis=-1).copy(), device=hidden.device)
        observable = observable & track_support
        language_mask = torch.cat((observable, observable))
        similarities = F.normalize(histories, dim=-1) @ text.T
        best = similarities.masked_fill(~language_mask[..., None], -torch.inf).amax(dim=(0, 1))
        return best, language_mask.any()

    def score_text(
        self, field: RelationField, text_embeddings: Tensor
    ) -> TemporalRelationScores:
        """Stream whole directed packets against whole captions on the full timeline.

        The fixed pre-pilot rule is half ordered-capture cosine, half maximum
        directed-packet cosine. Both consume the same adapted sentence vector;
        subject/action/object maxima are never recombined. The maximum covers
        both endpoint orientations and all observable patches in 64-edge
        blocks, with backward recomputation. Text/gallery batching is explicit;
        excessive block score storage fails rather than dropping edges/time.
        """
        device = next(self.parameters()).device
        if (
            text_embeddings.ndim != 2
            or text_embeddings.shape[0] == 0
            or text_embeddings.shape[1] != self.width
            or text_embeddings.device != device
            or text_embeddings.dtype != torch.float32
            or not bool(torch.isfinite(text_embeddings).all())
        ):
            raise ValueError("text requires nonempty finite float32[Q,width] on the model device")
        projected_bytes = (
            2 * min(64, field.pair_count) * field.patch_count * len(text_embeddings) * 4
        )
        if projected_bytes > 512 * 1024**2:
            raise MemoryError(
                "RESOURCE_LIMIT: directed relation score block; batch the text gallery"
            )
        hidden, nodes, topology_mask = self._encode_context(field)
        output = self._readout(field, hidden, nodes, topology_mask)
        text = F.normalize(text_embeddings, dim=-1)
        best = text.new_full((len(text),), -torch.inf)
        support = torch.zeros((), dtype=torch.bool, device=device)
        for start in range(0, field.pair_count, 64):
            stop = min(field.pair_count, start + 64)

            def text_block(value, node_value, text_value, block_start=start, block_stop=stop):
                return self._directed_text_block(
                    field, value, node_value, topology_mask, text_value, block_start, block_stop
                )

            local_best, local_support = self._run_block(text_block, hidden, nodes, text)
            best = torch.maximum(best, local_best)
            support = support | local_support
        whole_capture = output.embedding @ text.T
        cosine = torch.where(support, 0.5 * (whole_capture + best), torch.zeros_like(best))
        return TemporalRelationScores(output, cosine, support)


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
        self.mixture_logit = nn.Parameter(
            torch.tensor(math.log(initial_mixture / (1 - initial_mixture)))
        )
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


def weak_counterfactual_loss(
    positive_scores: Tensor,
    negative_scores: Tensor,
    included_weak_mask: Tensor,
    *,
    margin: float = 0.2,
) -> Tensor:
    """Disclosed caption-derived training targets, never human-verified truth.

    This uses the same registered margin formula but a distinct weak-label
    entry point. An included bit admits a weak training target only; it is
    not a false-event judgement, independent evaluation label or provenance.
    """
    if (
        positive_scores.ndim != 1
        or positive_scores.shape != negative_scores.shape
        or included_weak_mask.shape != positive_scores.shape
        or included_weak_mask.dtype != torch.bool
        or positive_scores.device != negative_scores.device
        or positive_scores.device != included_weak_mask.device
    ):
        raise ValueError("weak scores and bool inclusion mask need matching vectors/devices")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and nonnegative")
    penalties = F.softplus(margin + negative_scores - positive_scores)
    if not bool(included_weak_mask.any()):
        return (positive_scores.sum() + negative_scores.sum()) * 0.0
    return penalties[included_weak_mask].mean()


def weak_coordination_objective(
    scores: Tensor,
    positive_mask: Tensor,
    *,
    cf_positive_scores: Tensor,
    cf_negative_scores: Tensor,
    included_weak_mask: Tensor,
    cf_weight: float = 0.2,
    margin: float = 0.2,
) -> Tensor:
    """Human-only retrieval plus explicitly weak, training-only CF supervision."""
    if not math.isfinite(cf_weight) or cf_weight < 0:
        raise ValueError("cf_weight must be finite and nonnegative")
    _, _, contrastive = variable_positive_symmetric_infonce(scores, positive_mask)
    counterfactual = weak_counterfactual_loss(
        cf_positive_scores, cf_negative_scores, included_weak_mask, margin=margin
    )
    return contrastive + cf_weight * counterfactual
