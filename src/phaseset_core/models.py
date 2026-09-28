"""Permutation-invariant, streamed PhaseSet group-token encoder."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Final

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .contracts import (
    ACTIVITY_DIM,
    COORDINATE_DIM,
    JOINT_COUNT,
    PreparedActivityBatch,
    PreparedGroupBatch,
    skeleton_to_activity,
    validate_prepared_activity_batch,
    validate_prepared_group_batch,
)
from .periodic import (
    BAND_COUNT,
    DEFAULT_EDGE_CHUNK_SIZE,
    PairChunk,
    ResourceLimitError,
    TOKEN_WIDTH,
    iter_unordered_pair_chunks,
    validate_edge_budget,
    validate_energy_floors,
)
from .streaming_autograd import StreamingEdgeSummaries, streaming_edge_summaries
from .torch_periodic import differentiable_edge_summaries

if TYPE_CHECKING:
    from .periodic_descriptor_cache_v2 import CachedPairChunkStream


STATUS: Final = "DATA_FREE_PHASESET_MODEL_NONPRODUCTION_AUTHORITY0"
DEFAULT_EDGE_BUDGET: Final = 32_768


class PhaseSetModelError(ValueError):
    """Raised when model inputs violate the multi-person contract."""


@dataclass(frozen=True, slots=True)
class GroupTokenOutput:
    """Band-indexed group tokens and auditable decomposition."""

    tokens: Tensor
    band_mask: Tensor
    pair_component: Tensor
    topology_delta: Tensor
    valid_pair_count: Tensor
    topology_node_count: Tensor


# Compatibility alias retained for the alpha namespace.  The public contract
# name is GroupTokenOutput.
PhaseSetOutput = GroupTokenOutput


def _fixed_binary_tree_sum(values: list[Tensor]) -> Tensor:
    """Reduce one canonical microblock with a fixed adjacent binary tree."""

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


class SharedActorEncoder(nn.Module):
    """One temporal encoder shared by every actor and every group size."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(ACTIVITY_DIM, embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, activities: Tensor, frame_mask: Tensor, actor_mask: Tensor) -> Tensor:
        if activities.ndim != 4 or activities.shape[-1] != ACTIVITY_DIM:
            raise PhaseSetModelError("activities must have shape [B,K,T,5]")
        hidden = self.output_projection(F.gelu(self.input_projection(activities)))
        valid = frame_mask[:, None, :, None] & actor_mask[:, :, None, None]
        hidden = torch.where(valid, hidden, torch.zeros_like(hidden))
        denominator = frame_mask.sum(dim=1, dtype=hidden.dtype).clamp_min(1.0)
        pooled = hidden.sum(dim=2) / denominator[:, None, None]
        return torch.where(actor_mask[..., None], pooled, torch.zeros_like(pooled))


@dataclass(frozen=True, slots=True)
class GroupBaseOutput:
    """Permutation-invariant base embedding plus auditable actor embeddings."""

    group_embedding: Tensor
    actor_embeddings: Tensor
    actor_mask: Tensor


def _validate_base_widths(
    hidden_dim: int,
    heads: int,
    ffn_dim: int,
    dropout: float,
) -> None:
    if type(hidden_dim) is not int or hidden_dim < 2:
        raise TypeError("hidden_dim must be an exact int >=2")
    if type(heads) is not int or heads < 1 or hidden_dim % heads:
        raise TypeError("heads must be an exact positive divisor of hidden_dim")
    if type(ffn_dim) is not int or ffn_dim < hidden_dim:
        raise TypeError("ffn_dim must be an exact int >=hidden_dim")
    if type(dropout) not in (float, int) or not 0.0 <= float(dropout) < 1.0:
        raise TypeError("dropout must be a finite built-in number in [0,1)")


def _sinusoidal_time_encoding(
    time_steps: int,
    width: int,
    *,
    device: torch.device,
) -> Tensor:
    positions = torch.arange(time_steps, dtype=torch.float32, device=device)[:, None]
    even_count = (width + 1) // 2
    exponents = torch.arange(even_count, dtype=torch.float32, device=device)
    scales = torch.exp(-math.log(10_000.0) * (2.0 * exponents / width))
    encoding = torch.zeros((time_steps, width), dtype=torch.float32, device=device)
    encoding[:, 0::2] = torch.sin(positions * scales)
    if width > 1:
        encoding[:, 1::2] = torch.cos(positions * scales[: width // 2])
    return encoding


def _public_skeleton_tensors(
    batch: PreparedGroupBatch,
    device: torch.device,
) -> tuple[PreparedGroupBatch, Tensor, Tensor, Tensor, Tensor]:
    checked = validate_prepared_group_batch(batch)
    skeletons = torch.tensor(
        checked.skeletons,
        dtype=torch.float32,
        device=device,
    ).contiguous()
    actor_mask = torch.tensor(
        checked.actor_mask,
        dtype=torch.bool,
        device=device,
    ).contiguous()
    frame_mask = torch.tensor(
        checked.frame_mask,
        dtype=torch.bool,
        device=device,
    ).contiguous()
    track_mask = torch.tensor(
        checked.track_mask,
        dtype=torch.bool,
        device=device,
    ).contiguous()
    return checked, skeletons, actor_mask, frame_mask, track_mask


def _skeleton_features(skeletons: Tensor, track_mask: Tensor) -> Tensor:
    batch_size, actors, time_steps, _, _ = skeletons.shape
    coordinates = skeletons.reshape(
        batch_size,
        actors,
        time_steps,
        JOINT_COUNT * COORDINATE_DIM,
    )
    return torch.cat((coordinates, track_mask.to(dtype=torch.float32)), dim=-1)


class SharedSkeletonTemporalEncoder(nn.Module):
    """One identity-free temporal Transformer shared by every valid actor."""

    def __init__(
        self,
        *,
        hidden_dim: int = 512,
        heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        _validate_base_widths(hidden_dim, heads, ffn_dim, dropout)
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(
            JOINT_COUNT * COORDINATE_DIM + JOINT_COUNT,
            hidden_dim,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(
            layer,
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        skeletons: Tensor,
        actor_mask: Tensor,
        frame_mask: Tensor,
        track_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        features = _skeleton_features(skeletons, track_mask)
        batch_size, actors, time_steps, _ = features.shape
        time_mask = (
            actor_mask[:, :, None]
            & frame_mask[:, None, :]
            & track_mask.any(dim=-1)
        )
        actor_embeddings = torch.zeros(
            (batch_size, actors, self.hidden_dim),
            dtype=features.dtype,
            device=features.device,
        )
        actor_sequences = torch.zeros(
            (batch_size, actors, time_steps, self.hidden_dim),
            dtype=features.dtype,
            device=features.device,
        )
        # Each actor is evaluated as a fixed-shape singleton.  This makes the
        # frozen runtime independent of K_pad and of unrelated batch members.
        for group in range(batch_size):
            valid_length = int(frame_mask[group].sum().item())
            for actor in range(actors):
                if not bool(actor_mask[group, actor].item()):
                    continue
                selected_features = features[
                    group : group + 1,
                    actor,
                    :valid_length,
                ]
                selected_mask = time_mask[
                    group : group + 1,
                    actor,
                    :valid_length,
                ]
                hidden = self.input_projection(selected_features)
                hidden = hidden + _sinusoidal_time_encoding(
                    valid_length,
                    self.hidden_dim,
                    device=hidden.device,
                )[None, :, :]
                hidden = self.temporal(hidden, src_key_padding_mask=~selected_mask)
                hidden = self.output_norm(hidden)
                hidden = torch.where(
                    selected_mask[..., None],
                    hidden,
                    torch.zeros_like(hidden),
                )
                denominator = selected_mask.sum(
                    dim=1, dtype=hidden.dtype
                ).clamp_min(1.0)
                actor_embeddings[group, actor] = hidden.sum(dim=1)[0] / denominator[0]
                actor_sequences[group, actor, :valid_length] = hidden[0]
        return actor_embeddings, actor_sequences, time_mask


class PMAPool(nn.Module):
    """One learned seed pools an unordered set with no actor-position feature."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seed = nn.Parameter(torch.empty((1, 1, hidden_dim), dtype=torch.float32))
        nn.init.normal_(self.seed, mean=0.0, std=hidden_dim**-0.5)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, actors: Tensor, actor_mask: Tensor) -> Tensor:
        outputs: list[Tensor] = []
        for group in range(actors.shape[0]):
            actor_count = int(actor_mask[group].sum().item())
            values = actors[group : group + 1, :actor_count]
            attended, _ = self.attention(
                self.seed,
                values,
                values,
                need_weights=False,
            )
            hidden = self.norm1(self.seed + attended)
            outputs.append(self.norm2(hidden + self.ffn(hidden))[0, 0])
        return torch.stack(outputs)


class ActorMeanBase(nn.Module):
    """B0: shared temporal actor encoder followed by a masked actor mean."""

    system_id: Final = "B0"

    def __init__(
        self,
        *,
        hidden_dim: int = 512,
        heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.actor_encoder = SharedSkeletonTemporalEncoder(
            hidden_dim=hidden_dim,
            heads=heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

    def forward(self, batch: PreparedGroupBatch) -> GroupBaseOutput:
        device = next(self.parameters()).device
        _, skeletons, actor_mask, frame_mask, track_mask = _public_skeleton_tensors(
            batch, device
        )
        actor_embeddings, _, _ = self.actor_encoder(
            skeletons,
            actor_mask,
            frame_mask,
            track_mask,
        )
        group = torch.stack(
            [
                actor_embeddings[row, : int(actor_mask[row].sum().item())].mean(dim=0)
                for row in range(actor_embeddings.shape[0])
            ]
        )
        return GroupBaseOutput(
            group.contiguous(),
            actor_embeddings.contiguous(),
            actor_mask,
        )


class SetPMABase(nn.Module):
    """B1: shared temporal actor encoder followed by set PMA pooling."""

    system_id: Final = "B1"

    def __init__(
        self,
        *,
        hidden_dim: int = 512,
        heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.actor_encoder = SharedSkeletonTemporalEncoder(
            hidden_dim=hidden_dim,
            heads=heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.pool = PMAPool(hidden_dim, heads, ffn_dim, dropout)

    def forward(self, batch: PreparedGroupBatch) -> GroupBaseOutput:
        device = next(self.parameters()).device
        _, skeletons, actor_mask, frame_mask, track_mask = _public_skeleton_tensors(
            batch, device
        )
        actor_embeddings, _, _ = self.actor_encoder(
            skeletons,
            actor_mask,
            frame_mask,
            track_mask,
        )
        group = self.pool(actor_embeddings, actor_mask)
        return GroupBaseOutput(
            group.contiguous(),
            actor_embeddings.contiguous(),
            actor_mask,
        )


SOCIAL_QUERY_CHUNK_SIZE: Final = 64
SOCIAL_FRAME_ROW_CHUNK_SIZE: Final = 4


def _streaming_social_self_attention(
    attention: nn.MultiheadAttention,
    values: Tensor,
    actor_mask: Tensor,
) -> Tensor:
    """Exact self-attention algebra without materializing a full K-by-K score map."""

    if (
        values.ndim != 3
        or actor_mask.shape != values.shape[:2]
        or not attention._qkv_same_embed_dim
        or attention.in_proj_weight is None
    ):
        raise PhaseSetModelError("streaming social attention received an invalid shape")
    row_count, actor_count, width = values.shape
    heads = attention.num_heads
    head_dim = width // heads
    projected = F.linear(values, attention.in_proj_weight, attention.in_proj_bias)
    query, key, value = projected.chunk(3, dim=-1)
    query = query.reshape(row_count, actor_count, heads, head_dim).transpose(1, 2)
    key = key.reshape(row_count, actor_count, heads, head_dim).transpose(1, 2)
    value = value.reshape(row_count, actor_count, heads, head_dim).transpose(1, 2)
    result = torch.empty_like(query)
    query_chunk = min(SOCIAL_QUERY_CHUNK_SIZE, max(1, actor_count - 1))
    scale = head_dim**-0.5
    for row_start in range(0, row_count, SOCIAL_FRAME_ROW_CHUNK_SIZE):
        row_stop = min(row_count, row_start + SOCIAL_FRAME_ROW_CHUNK_SIZE)
        local_key = key[row_start:row_stop]
        local_value = value[row_start:row_stop]
        local_mask = actor_mask[row_start:row_stop, None, None, :]
        transposed_key = local_key.transpose(-2, -1)
        for query_start in range(0, actor_count, query_chunk):
            query_stop = min(actor_count, query_start + query_chunk)
            local_query = query[
                row_start:row_stop,
                :,
                query_start:query_stop,
            ]
            scores = torch.matmul(local_query * scale, transposed_key)
            scores = scores.masked_fill(~local_mask, float("-inf"))
            probabilities = torch.softmax(scores, dim=-1)
            probabilities = F.dropout(
                probabilities,
                p=float(attention.dropout),
                training=attention.training,
            )
            result[
                row_start:row_stop,
                :,
                query_start:query_stop,
            ] = torch.matmul(probabilities, local_value)
    combined = result.transpose(1, 2).reshape(row_count, actor_count, width)
    return F.linear(
        combined,
        attention.out_proj.weight,
        attention.out_proj.bias,
    ).contiguous()


class SocialTemporalLayer(nn.Module):
    """One temporal-attention then per-frame actor-set-attention layer."""

    def __init__(self, hidden_dim: int, heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim,
            heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.social_attention = nn.MultiheadAttention(
            hidden_dim,
            heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.social_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, value: Tensor, time_mask: Tensor, actor_mask: Tensor) -> Tensor:
        batch_size, actors, time_steps, width = value.shape
        flat = value.reshape(batch_size * actors, time_steps, width)
        flat_mask = time_mask.reshape(batch_size * actors, time_steps)
        valid_actor_rows = actor_mask.reshape(batch_size * actors)
        selected = flat[valid_actor_rows]
        selected_mask = flat_mask[valid_actor_rows]
        attended, _ = self.temporal_attention(
            selected,
            selected,
            selected,
            key_padding_mask=~selected_mask,
            need_weights=False,
        )
        selected = self.temporal_norm(selected + attended)
        selected = torch.where(
            selected_mask[..., None],
            selected,
            torch.zeros_like(selected),
        )
        temporal = torch.zeros_like(flat)
        temporal[valid_actor_rows] = selected
        temporal = temporal.reshape(batch_size, actors, time_steps, width)

        by_frame = temporal.permute(0, 2, 1, 3).reshape(
            batch_size * time_steps,
            actors,
            width,
        )
        frame_actor_mask = time_mask.permute(0, 2, 1).reshape(
            batch_size * time_steps,
            actors,
        )
        valid_frame_rows = frame_actor_mask.any(dim=1)
        selected_frames = by_frame[valid_frame_rows]
        selected_frame_mask = frame_actor_mask[valid_frame_rows]
        social = _streaming_social_self_attention(
            self.social_attention,
            selected_frames,
            selected_frame_mask,
        )
        selected_frames = self.social_norm(selected_frames + social)
        selected_frames = torch.where(
            selected_frame_mask[..., None],
            selected_frames,
            torch.zeros_like(selected_frames),
        )
        social_rows = torch.zeros_like(by_frame)
        social_rows[valid_frame_rows] = selected_frames
        output = social_rows.reshape(batch_size, time_steps, actors, width).permute(
            0, 2, 1, 3
        )
        output = self.output_norm(output + self.ffn(output))
        return torch.where(time_mask[..., None], output, torch.zeros_like(output))


class SocialTemporalBase(nn.Module):
    """B2: four alternating temporal/social layers and final PMA pooling."""

    system_id: Final = "B2"

    def __init__(
        self,
        *,
        hidden_dim: int = 512,
        heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        _validate_base_widths(hidden_dim, heads, ffn_dim, dropout)
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(
            JOINT_COUNT * COORDINATE_DIM + JOINT_COUNT,
            hidden_dim,
        )
        self.layers = nn.ModuleList(
            SocialTemporalLayer(hidden_dim, heads, ffn_dim, dropout)
            for _ in range(4)
        )
        self.pool = PMAPool(hidden_dim, heads, ffn_dim, dropout)

    def forward(self, batch: PreparedGroupBatch) -> GroupBaseOutput:
        device = next(self.parameters()).device
        _, skeletons, actor_mask, frame_mask, track_mask = _public_skeleton_tensors(
            batch, device
        )
        all_features = _skeleton_features(skeletons, track_mask)
        actor_embeddings = torch.zeros(
            (skeletons.shape[0], skeletons.shape[1], self.hidden_dim),
            dtype=torch.float32,
            device=device,
        )
        group_rows: list[Tensor] = []
        for group_index in range(skeletons.shape[0]):
            actor_count = int(actor_mask[group_index].sum().item())
            valid_length = int(frame_mask[group_index].sum().item())
            local_actor_mask = actor_mask[
                group_index : group_index + 1,
                :actor_count,
            ]
            local_track = track_mask[
                group_index : group_index + 1,
                :actor_count,
                :valid_length,
            ]
            local_time_mask = (
                local_actor_mask[:, :, None]
                & frame_mask[
                    group_index : group_index + 1,
                    None,
                    :valid_length,
                ]
                & local_track.any(dim=-1)
            )
            local_features = all_features[
                group_index : group_index + 1,
                :actor_count,
                :valid_length,
            ]
            hidden = self.input_projection(local_features)
            hidden = hidden + _sinusoidal_time_encoding(
                valid_length,
                self.hidden_dim,
                device=device,
            )[None, None, :, :]
            hidden = torch.where(
                local_time_mask[..., None], hidden, torch.zeros_like(hidden)
            )
            for layer in self.layers:
                hidden = layer(hidden, local_time_mask, local_actor_mask)
            denominator = local_time_mask.sum(
                dim=2, dtype=hidden.dtype
            ).clamp_min(1.0)
            local_actors = hidden.sum(dim=2) / denominator[..., None]
            actor_embeddings[group_index, :actor_count] = local_actors[0]
            group_rows.append(self.pool(local_actors, local_actor_mask)[0])
        group = torch.stack(group_rows)
        return GroupBaseOutput(
            group.contiguous(),
            actor_embeddings.contiguous(),
            actor_mask,
        )


def build_group_base(
    system_id: str,
    *,
    hidden_dim: int = 512,
    heads: int = 8,
    ffn_dim: int = 2048,
    dropout: float = 0.1,
) -> nn.Module:
    """Construct exactly one registered base-qualification candidate."""

    candidates: dict[str, type[nn.Module]] = {
        "B0": ActorMeanBase,
        "B1": SetPMABase,
        "B2": SocialTemporalBase,
    }
    if system_id not in candidates:
        raise PhaseSetModelError("base system_id must be one of B0, B1, B2")
    return candidates[system_id](
        hidden_dim=hidden_dim,
        heads=heads,
        ffn_dim=ffn_dim,
        dropout=dropout,
    )


class PhaseSetEncoder(nn.Module):
    """Six-band group encoder with O(K^2) time and O(K) resident state.

    The fixed PhasePair descriptor is streamed edge-by-edge.  No dense KxK
    tensor is constructed.  Actor commitments canonicalize the stream before
    this module sees it, while commitments themselves are never model features.
    """

    def __init__(
        self,
        *,
        embedding_dim: int = 512,
        hidden_dim: int = 256,
        energy_floors: np.ndarray | None = None,
        topology_scale: float = 1.0,
        edge_budget: int = DEFAULT_EDGE_BUDGET,
    ) -> None:
        super().__init__()
        if type(embedding_dim) is not int or embedding_dim < 2:
            raise TypeError("embedding_dim must be an exact int >=2")
        if type(hidden_dim) is not int or hidden_dim < 2:
            raise TypeError("hidden_dim must be an exact int >=2")
        if type(topology_scale) not in (float, int) or not np.isfinite(topology_scale):
            raise TypeError("topology_scale must be a finite built-in number")
        if type(edge_budget) is not int or edge_budget < 1:
            raise TypeError("edge_budget must be an exact positive int")
        floors = (
            np.zeros((BAND_COUNT,), dtype=np.float64)
            if energy_floors is None
            else energy_floors
        )
        self._energy_floors = validate_energy_floors(floors)
        self.embedding_dim = embedding_dim
        self.edge_budget = edge_budget
        # ``topology_scale`` is the initialization of the registered per-band
        # lambda.  The forward path always applies tanh, so the residual scale
        # remains learned and bounded.
        self.topology_lambda = nn.Parameter(
            torch.full((BAND_COUNT,), float(topology_scale), dtype=torch.float32)
        )
        self.half_edge_encoder = nn.Sequential(
            nn.Linear(TOKEN_WIDTH, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.pair_encoder = nn.Sequential(
            nn.Linear(3 * embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        # Raw degree is deliberately absent.  The last two scalars are support
        # coverage d/(K-1) and its complementary missingness.
        self.topology_encoder = nn.Sequential(
            nn.Linear(2 * embedding_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.common_postprocess = nn.LayerNorm(embedding_dim)

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def _edge_forward(
        self,
        relation_ij: Tensor,
        relation_ji: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply the registered directed-half-edge and symmetric-pair MLPs."""

        half_ij = self.half_edge_encoder(relation_ij)
        half_ji = self.half_edge_encoder(relation_ji)
        pair_tokens = self.pair_encoder(
            torch.cat(
                (
                    half_ij + half_ji,
                    torch.abs(half_ij - half_ji),
                    half_ij * half_ji,
                ),
                dim=-1,
            )
        )
        return half_ij, half_ji, pair_tokens

    def _iter_pair_chunks(
        self,
        batch: PreparedActivityBatch,
        *,
        edge_chunk_size: int,
    ) -> Iterator[PairChunk]:
        """Yield the registered edge stream.

        Control systems may override this protected seam with a frozen,
        non-phase feature stream.  The default remains the PhaseSet Morlet
        stream and therefore preserves the public full-model behaviour.
        """

        return iter_unordered_pair_chunks(
            batch,
            energy_floors=self._energy_floors,
            edge_chunk_size=edge_chunk_size,
            edge_budget=self.edge_budget,
        )

    def _expected_descriptor_stream_kind(self) -> str | None:
        """Return the only persisted descriptor family this encoder accepts."""

        return "FULL_RELATION"

    def _validate_cached_descriptor_stream(
        self,
        batch: PreparedGroupBatch,
        descriptor_stream: CachedPairChunkStream,
    ) -> CachedPairChunkStream:
        """Bind a verified descriptor stream to this exact batch and model."""

        from .periodic_descriptor_cache_v2 import (
            CachedPairChunkStream,
            energy_floors_sha256,
            prepared_group_batch_sha256,
        )

        if type(descriptor_stream) is not CachedPairChunkStream:
            raise PhaseSetModelError(
                "cached execution requires an exact verified descriptor stream"
            )
        expected_kind = self._expected_descriptor_stream_kind()
        if expected_kind is None:
            raise PhaseSetModelError(
                "this registered system does not admit cached descriptor execution"
            )
        if descriptor_stream.stream_kind != expected_kind:
            raise PhaseSetModelError(
                "cached descriptor stream kind does not match the registered system"
            )
        budget = validate_edge_budget(self.edge_budget)
        expected_edges = sum(count * (count - 1) // 2 for count in batch.actor_counts)
        if expected_edges > budget:
            raise ResourceLimitError(
                required_edges=expected_edges,
                edge_budget=budget,
            )
        cached_batch = descriptor_stream.batch
        if cached_batch.batch_input_sha256 != prepared_group_batch_sha256(batch):
            raise PhaseSetModelError(
                "cached descriptor stream does not bind the actual model batch"
            )
        if cached_batch.energy_floors_sha256 != energy_floors_sha256(
            self._energy_floors
        ):
            raise PhaseSetModelError(
                "cached descriptor stream does not bind the model energy floors"
            )
        if cached_batch.edge_count != expected_edges:
            raise PhaseSetModelError(
                "cached descriptor stream does not contain the complete batch edge set"
            )
        return descriptor_stream

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
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Identity hook used by the incidence-shuffled registered control."""

        del actor_i, actor_j, batch_indices, support_mask
        return half_ij, half_ji, pair_tokens

    def forward(
        self,
        batch: PreparedGroupBatch,
        *,
        edge_chunk_size: int = DEFAULT_EDGE_CHUNK_SIZE,
        include_topology: bool = True,
    ) -> GroupTokenOutput:
        """Encode the public 22-joint skeleton contract."""

        checked = validate_prepared_group_batch(batch)
        return self.forward_activity(
            skeleton_to_activity(checked),
            edge_chunk_size=edge_chunk_size,
            include_topology=include_topology,
        )

    def forward_cached(
        self,
        batch: PreparedGroupBatch,
        *,
        descriptor_stream: CachedPairChunkStream,
        edge_chunk_size: int = DEFAULT_EDGE_CHUNK_SIZE,
        include_topology: bool = True,
    ) -> GroupTokenOutput:
        """Encode with an explicitly verified, batch-bound descriptor stream."""

        if type(include_topology) is not bool:
            raise TypeError("include_topology must be an exact built-in bool")
        checked = validate_prepared_group_batch(batch)
        stream = self._validate_cached_descriptor_stream(checked, descriptor_stream)
        return self._forward_activity(
            skeleton_to_activity(checked),
            edge_chunk_size=edge_chunk_size,
            include_topology=include_topology,
            descriptor_stream=stream,
        )

    def forward_activity(
        self,
        batch: PreparedActivityBatch,
        *,
        edge_chunk_size: int = DEFAULT_EDGE_CHUNK_SIZE,
        include_topology: bool = True,
    ) -> GroupTokenOutput:
        """Internal/legacy test seam for already derived five-channel activity."""

        if type(include_topology) is not bool:
            raise TypeError("include_topology must be an exact built-in bool")
        checked = validate_prepared_activity_batch(batch)
        return self._forward_activity(
            checked,
            edge_chunk_size=edge_chunk_size,
            include_topology=include_topology,
            descriptor_stream=None,
        )

    def _forward_activity(
        self,
        checked: PreparedActivityBatch,
        *,
        edge_chunk_size: int,
        include_topology: bool,
        descriptor_stream: CachedPairChunkStream | None,
    ) -> GroupTokenOutput:
        """Shared reduction path for uncached and explicitly cached execution."""

        summaries = streaming_edge_summaries(
            self,
            checked,
            edge_chunk_size=edge_chunk_size,
            pair_chunk_stream=descriptor_stream,
        )
        return self._finish_summaries(
            summaries,
            actor_positions=tuple(
                tuple(range(count)) for count in checked.actor_counts
            ),
            include_topology=include_topology,
        )

    def forward_skeleton_autograd_oracle(
        self,
        skeletons: Tensor,
        actor_mask: Tensor,
        frame_mask: Tensor,
        track_mask: Tensor,
        *,
        include_topology: bool = True,
    ) -> GroupTokenOutput:
        """Run the explicit differentiable CPU skeleton/Morlet oracle.

        This qualification path accepts arbitrary valid-actor positions and
        retains gradients to ``skeletons``.  It shares the registered edge,
        topology, and postprocess parameters with the production model.  The
        production NumPy descriptor streamer remains intentionally distinct
        and is not represented as differentiable by this method.
        """

        if type(include_topology) is not bool:
            raise TypeError("include_topology must be an exact built-in bool")
        summaries = differentiable_edge_summaries(
            self,
            skeletons,
            actor_mask,
            frame_mask,
            track_mask,
            energy_floors=self._energy_floors,
            edge_budget=self.edge_budget,
        )
        actor_positions = tuple(
            tuple(
                int(position)
                for position in torch.nonzero(row, as_tuple=False)[:, 0].tolist()
            )
            for row in actor_mask
        )
        return self._finish_summaries(
            summaries,
            actor_positions=actor_positions,
            include_topology=include_topology,
        )

    def _finish_summaries(
        self,
        summaries: StreamingEdgeSummaries,
        *,
        actor_positions: tuple[tuple[int, ...], ...],
        include_topology: bool,
    ) -> GroupTokenOutput:
        """Apply the shared topology and postprocess heads to edge summaries."""

        device = self._device()
        pair_component = summaries.pair_component
        pair_count_tensor = summaries.valid_pair_count
        band_mask = pair_count_tensor > 0
        statistics = summaries.node_statistics
        topology_mask = summaries.topology_node_mask
        topology_masks = topology_mask.tolist()
        raw_node_delta = self.topology_encoder(statistics)
        # The false branch is materialized +0.  The topology encoder remains in
        # the graph, so an all-K=2 batch receives allocated, exactly zero grads.
        node_delta = torch.where(
            topology_mask[..., None],
            raw_node_delta,
            torch.zeros_like(raw_node_delta),
        )

        topology_rows: list[Tensor] = []
        topology_count_rows: list[list[int]] = []
        for group, positions in enumerate(actor_positions):
            bands = []
            counts = []
            for band in range(BAND_COUNT):
                values = [
                    node_delta[group, actor, band].to(dtype=torch.float64)
                    for actor in positions
                    if topology_masks[group][actor][band]
                ]
                count = len(values)
                if count:
                    value = (_fixed_binary_tree_sum(values) / count).to(
                        dtype=torch.float32
                    )
                else:
                    # Preserve an autograd path through every hard-gated node so
                    # K=2 produces allocated, bitwise-positive zero gradients.
                    linked_zeros = [
                        node_delta[group, actor, band].to(dtype=torch.float64)
                        for actor in positions
                    ]
                    value = _fixed_binary_tree_sum(linked_zeros).to(dtype=torch.float32)
                bands.append(value)
                counts.append(count)
            topology_rows.append(torch.stack(bands))
            topology_count_rows.append(counts)
        topology_delta = torch.stack(topology_rows)
        topology_count_tensor = torch.tensor(
            topology_count_rows,
            dtype=torch.int64,
            device=device,
        )

        effective_topology = (
            topology_delta if include_topology else torch.zeros_like(topology_delta)
        )
        bounded_scale = torch.tanh(self.topology_lambda)[None, :, None]
        combined = pair_component + bounded_scale * effective_topology
        processed = self.common_postprocess(combined)
        tokens = torch.where(band_mask[..., None], processed, torch.zeros_like(processed))
        if not bool(torch.isfinite(tokens).all().item()):
            raise PhaseSetModelError("group tokens became nonfinite")
        return GroupTokenOutput(
            tokens=tokens.contiguous(),
            band_mask=band_mask.contiguous(),
            pair_component=pair_component.contiguous(),
            topology_delta=effective_topology.contiguous(),
            valid_pair_count=pair_count_tensor.contiguous(),
            topology_node_count=topology_count_tensor.contiguous(),
        )


__all__ = [
    "ActorMeanBase",
    "DEFAULT_EDGE_BUDGET",
    "GroupBaseOutput",
    "GroupTokenOutput",
    "PMAPool",
    "PhaseSetEncoder",
    "PhaseSetModelError",
    "PhaseSetOutput",
    "ResourceLimitError",
    "SetPMABase",
    "SharedSkeletonTemporalEncoder",
    "SocialTemporalBase",
    "SocialTemporalLayer",
    "STATUS",
    "SharedActorEncoder",
    "build_group_base",
]
