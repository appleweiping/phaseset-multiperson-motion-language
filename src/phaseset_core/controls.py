"""Executable, capacity-matched PhaseSet residual systems 00--08.

The module is deliberately data-free: it constructs feature policies and
group-token encoders, but it does not train a model, open a dataset, or claim an
experimental result.  Systems 01--08 share the exact same trainable module
topology.  Their only differences are frozen input gates, the registered
actor-local marginal, DCT, and observation-availability edge streams, and the
deterministic incidence reassignment used by system 06.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
from typing import TYPE_CHECKING, Final

import numpy as np
import torch
from torch import Tensor, nn

from phasepair_core.signal import TOKEN_EPSILON

from .contracts import (
    PreparedActivityBatch,
    PreparedGroupBatch,
    skeleton_to_activity,
    validate_prepared_activity_batch,
    validate_prepared_group_batch,
)
from .models import DEFAULT_EDGE_BUDGET, GroupTokenOutput, PhaseSetEncoder
from .morlet import MORLET_FREQUENCIES_HZ, SAMPLE_RATE_HZ
from .periodic import (
    BAND_COUNT,
    DEFAULT_EDGE_CHUNK_SIZE,
    TOKEN_WIDTH,
    PairChunk,
    ResourceLimitError,
    iter_marginal_power_pair_chunks,
    iter_observation_availability_pair_chunks,
    validate_edge_budget,
    validate_edge_chunk_size,
    validate_energy_floors,
)

if TYPE_CHECKING:
    from .periodic_descriptor_cache_v2 import CachedPairChunkStream


STATUS: Final = "DATA_FREE_EXECUTABLE_PHASESET_CONTROLS_NONPRODUCTION_AUTHORITY0"
SYSTEM_IDS: Final = tuple(f"{index:02d}" for index in range(9))
RESIDUAL_SYSTEM_IDS: Final = tuple(f"{index:02d}" for index in range(1, 9))


class PhaseSetControlError(ValueError):
    """Raised when a registered control-system contract is violated."""


@dataclass(frozen=True, slots=True)
class ResidualSystemSpec:
    system_id: str
    name: str
    feature_policy: str
    include_topology: bool
    residual_enabled: bool = True


SYSTEM_SPECS: Final = (
    ResidualSystemSpec("00", "qualified group base", "NO_RESIDUAL", False, False),
    ResidualSystemSpec("01", "generic six tokens", "BAND_ID_ONLY", False),
    ResidualSystemSpec("02", "marginal Morlet power", "MARGINAL_POWER", False),
    ResidualSystemSpec("03", "mean/difference DCT", "MEAN_DIFFERENCE_DCT", False),
    ResidualSystemSpec("04", "PhasePair pair-bag", "FULL_RELATION", False),
    ResidualSystemSpec("05", "coverage/missing-only", "COVERAGE_MISSING_ONLY", True),
    ResidualSystemSpec("06", "incidence-shuffled", "SHUFFLED_INCIDENCE", True),
    ResidualSystemSpec("07", "phase-stripped full", "PHASE_STRIPPED", True),
    ResidualSystemSpec("08", "PhaseSet full", "FULL_RELATION", True),
)
_SPEC_BY_ID: Final = {spec.system_id: spec for spec in SYSTEM_SPECS}


def system_spec(system_id: object) -> ResidualSystemSpec:
    if type(system_id) is not str or system_id not in _SPEC_BY_ID:
        raise PhaseSetControlError("system_id must be one of 00--08")
    return _SPEC_BY_ID[system_id]


def gate_relation_descriptors(tokens: Tensor, feature_policy: str) -> Tensor:
    """Apply one frozen information gate to directed 13D descriptors."""

    if (
        type(tokens) is not Tensor
        or tokens.ndim != 3
        or tokens.shape[-2:]
        != (
            BAND_COUNT,
            TOKEN_WIDTH,
        )
    ):
        raise PhaseSetControlError("tokens must be a torch Tensor [E,6,13]")
    zeros = torch.zeros_like(tokens[..., :7])
    band_ids = tokens[..., 7:]
    if feature_policy == "BAND_ID_ONLY":
        return torch.cat((zeros, band_ids), dim=-1).contiguous()
    if feature_policy == "MARGINAL_POWER":
        return torch.cat(
            (tokens[..., :2], torch.zeros_like(tokens[..., 2:7]), band_ids),
            dim=-1,
        ).contiguous()
    if feature_policy == "PHASE_STRIPPED":
        # 0:2 are endpoint log powers; 2 is coherence.  Cosine, sine,
        # normalized lag, and signed-phase-validity (3:7) are all removed.
        return torch.cat(
            (tokens[..., :3], torch.zeros_like(tokens[..., 3:7]), band_ids),
            dim=-1,
        ).contiguous()
    if feature_policy in {"FULL_RELATION", "SHUFFLED_INCIDENCE"}:
        return tokens
    if feature_policy == "COVERAGE_MISSING_ONLY":
        return torch.cat((zeros, band_ids), dim=-1).contiguous()
    if feature_policy == "MEAN_DIFFERENCE_DCT":
        return tokens
    raise PhaseSetControlError("unknown feature policy")


def _affine_permutation(
    length: int,
    *,
    seed: int,
    band: int,
    edge_identity_digest: bytes,
    device: torch.device,
) -> Tensor:
    if length <= 1:
        return torch.arange(length, dtype=torch.int64, device=device)
    digest = hashlib.sha256(
        f"phaseset-incidence-v2/{seed}/{band}/{length}/".encode()
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
    positions = torch.arange(length, dtype=torch.int64, device=device)
    return torch.remainder(positions * step + offset, length)


def _edge_identity_digest(actor_i: Tensor, actor_j: Tensor) -> bytes:
    payload = bytearray(b"phaseset-incidence-edge-block-v1\x00")
    identities = tuple(
        (int(left), int(right))
        for left, right in zip(
            actor_i.detach().cpu().tolist(),
            actor_j.detach().cpu().tolist(),
            strict=True,
        )
    )
    if identities != tuple(sorted(set(identities))) or any(
        left < 0 or left >= right for left, right in identities
    ):
        raise PhaseSetControlError("edge identities must be unique canonical pairs")
    for left, right in identities:
        payload.extend(left.to_bytes(8, "big", signed=False))
        payload.extend(right.to_bytes(8, "big", signed=False))
    return hashlib.sha256(payload).digest()


def shuffle_half_edge_incidence(
    half_ij: Tensor,
    half_ji: Tensor,
    support_mask: Tensor,
    *,
    batch_indices: Tensor,
    actor_i: Tensor,
    actor_j: Tensor,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Reassign valid half-edge values while preserving support at every node.

    Canonical commitment ordering is applied by ``PreparedActivityBatch`` before
    this function is reached.  The explicit seed then selects a stateless
    permutation inside every group-local canonical 64-edge microblock.  Only
    values move: endpoint slots and their support mask stay fixed, so every
    node's degree and coverage remain unchanged and each group's valid
    half-edge multiset is exact.  Batch-row numbers never seed the permutation.
    """

    if type(seed) is not int or not 0 <= seed < 2**63:
        raise PhaseSetControlError("incidence seed must be an int in [0,2^63)")
    if (
        type(half_ij) is not Tensor
        or type(half_ji) is not Tensor
        or type(support_mask) is not Tensor
        or half_ij.shape != half_ji.shape
        or half_ij.ndim != 3
        or half_ij.shape[1] != BAND_COUNT
        or support_mask.dtype != torch.bool
        or tuple(support_mask.shape) != tuple(half_ij.shape[:2])
        or support_mask.device != half_ij.device
        or any(type(value) is not Tensor for value in (batch_indices, actor_i, actor_j))
        or any(value.dtype != torch.int64 for value in (batch_indices, actor_i, actor_j))
        or any(value.ndim != 1 for value in (batch_indices, actor_i, actor_j))
        or any(
            int(value.shape[0]) != int(half_ij.shape[0])
            for value in (batch_indices, actor_i, actor_j)
        )
        or any(
            value.device != half_ij.device
            for value in (batch_indices, actor_i, actor_j)
        )
    ):
        raise PhaseSetControlError("invalid half-edge shuffle tensors")
    if bool((batch_indices < 0).any().item()):
        raise PhaseSetControlError("batch indices must be nonnegative")
    stacked = torch.stack((half_ij, half_ji), dim=1)
    shuffled_bands: list[Tensor] = []
    groups = torch.unique(batch_indices, sorted=True)
    group_blocks: list[tuple[Tensor, Tensor, bytes]] = []
    slot_offsets = torch.arange(2, dtype=torch.int64, device=half_ij.device)[None, :]
    for group in groups:
        edge_indices = torch.nonzero(
            batch_indices == group,
            as_tuple=False,
        ).flatten()
        identity_digest = _edge_identity_digest(
            actor_i.index_select(0, edge_indices),
            actor_j.index_select(0, edge_indices),
        )
        slot_indices = (edge_indices[:, None] * 2 + slot_offsets).reshape(-1)
        group_blocks.append((edge_indices, slot_indices, identity_digest))
    for band in range(BAND_COUNT):
        values = stacked[:, :, band, :].reshape(-1, stacked.shape[-1])
        shuffled_values = values
        for edge_indices, slot_indices, identity_digest in group_blocks:
            valid_slots = slot_indices[
                support_mask.index_select(0, edge_indices)[:, band]
                .repeat_interleave(2)
            ]
            permutation = _affine_permutation(
                int(valid_slots.numel()),
                seed=seed,
                band=band,
                edge_identity_digest=identity_digest,
                device=values.device,
            )
            selected = values.index_select(
                0,
                valid_slots.index_select(0, permutation),
            )
            shuffled_values = shuffled_values.index_copy(0, valid_slots, selected)
        shuffled_bands.append(shuffled_values)
    shuffled = torch.stack(shuffled_bands, dim=1).reshape_as(stacked)
    return shuffled[:, 0].contiguous(), shuffled[:, 1].contiguous()


def _dct_pair_descriptor(
    left: np.ndarray,
    right: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    floors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a symmetric marginal-spectrum control with no cross endpoint term.

    For each endpoint independently, the cosine and sine quadratures at the
    registered DCT-II frequency are squared and added.  This is the power
    obtained by a cosine transform of that endpoint's autocorrelation, while
    avoiding any ``left * right`` cross term.  The pair features are then the
    symmetric mean and absolute difference of the two endpoint powers.
    """

    time_steps = int(left.shape[0])
    samples = np.arange(time_steps, dtype=np.float64) + 0.5
    tokens = np.zeros((BAND_COUNT, TOKEN_WIDTH), dtype=np.float32)
    support = np.zeros((BAND_COUNT,), dtype=np.bool_)
    left64 = left.astype(np.float64)
    right64 = right.astype(np.float64)
    for band, frequency in enumerate(MORLET_FREQUENCIES_HZ):
        tokens[band, 7 + band] = np.float32(1.0)
        if time_steps < 2:
            continue
        dct_bin = max(1, int(round(2.0 * time_steps * frequency / SAMPLE_RATE_HZ)))
        dct_bin = min(dct_bin, time_steps - 1)
        angles = math.pi * dct_bin * samples / time_steps
        cosine_basis = np.cos(angles)
        sine_basis = np.sin(angles)
        left_energy = 0.0
        right_energy = 0.0
        left_channels = 0
        right_channels = 0
        for channel in range(left.shape[1]):
            for values, mask, side in (
                (left64, left_mask, "left"),
                (right64, right_mask, "right"),
            ):
                channel_mask = mask[:, channel]
                observed = int(np.count_nonzero(channel_mask))
                if observed < 2:
                    continue
                scale = math.sqrt(2.0 / observed)
                selected = np.where(channel_mask, values[:, channel], 0.0)
                cosine = scale * float(np.dot(selected, cosine_basis))
                sine = scale * float(np.dot(selected, sine_basis))
                energy = cosine * cosine + sine * sine
                if side == "left":
                    left_energy += energy
                    left_channels += 1
                else:
                    right_energy += energy
                    right_channels += 1
        if left_channels:
            left_energy /= left_channels
        if right_channels:
            right_energy /= right_channels
        mean_energy = 0.5 * (left_energy + right_energy)
        difference_energy = abs(left_energy - right_energy)
        total = left_energy + right_energy
        tokens[band, :3] = (
            np.float32(math.log(mean_energy + TOKEN_EPSILON)),
            np.float32(math.log(difference_energy + TOKEN_EPSILON)),
            np.float32(difference_energy / (total + TOKEN_EPSILON)),
        )
        # The supplied per-band DCT floor is common to both endpoints and is
        # applied strictly and independently.  Unilateral motion never creates
        # a valid relation edge.
        support[band] = (
            left_energy > float(floors[band])
            and right_energy > float(floors[band])
        )
    return np.ascontiguousarray(tokens), np.ascontiguousarray(support)


def iter_generic_token_chunks(
    batch: PreparedActivityBatch,
    *,
    edge_chunk_size: int = DEFAULT_EDGE_CHUNK_SIZE,
    edge_budget: int = DEFAULT_EDGE_BUDGET,
) -> Iterator[PairChunk]:
    """Yield motion-independent band-ID edges for the generic-token control."""

    checked = validate_prepared_activity_batch(batch)
    chunk_size = validate_edge_chunk_size(edge_chunk_size)
    budget = validate_edge_budget(edge_budget)
    required_edges = sum(count * (count - 1) // 2 for count in checked.actor_counts)
    if required_edges > budget:
        raise ResourceLimitError(required_edges=required_edges, edge_budget=budget)

    generic = np.zeros((BAND_COUNT, TOKEN_WIDTH), dtype=np.float32)
    generic[:, 7:] = np.eye(BAND_COUNT, dtype=np.float32)
    all_bands = np.ones((BAND_COUNT,), dtype=np.bool_)
    groups: list[int] = []
    actor_i: list[int] = []
    actor_j: list[int] = []

    def flush() -> PairChunk:
        edge_count = len(groups)
        tokens = np.broadcast_to(generic, (edge_count, BAND_COUNT, TOKEN_WIDTH)).copy()
        support = np.broadcast_to(all_bands, (edge_count, BAND_COUNT)).copy()
        return PairChunk(
            np.ascontiguousarray(groups, dtype=np.int64),
            np.ascontiguousarray(actor_i, dtype=np.int64),
            np.ascontiguousarray(actor_j, dtype=np.int64),
            np.ascontiguousarray(tokens, dtype=np.float32),
            np.ascontiguousarray(tokens.copy(), dtype=np.float32),
            np.ascontiguousarray(support, dtype=np.bool_),
        )

    for group, actor_count in enumerate(checked.actor_counts):
        for left_actor in range(actor_count):
            for right_actor in range(left_actor + 1, actor_count):
                groups.append(group)
                actor_i.append(left_actor)
                actor_j.append(right_actor)
                if len(groups) == chunk_size:
                    yield flush()
                    groups.clear()
                    actor_i.clear()
                    actor_j.clear()
    if groups:
        yield flush()


def iter_mean_difference_dct_chunks(
    batch: PreparedActivityBatch,
    *,
    energy_floors: np.ndarray,
    edge_chunk_size: int = DEFAULT_EDGE_CHUNK_SIZE,
    edge_budget: int = DEFAULT_EDGE_BUDGET,
) -> Iterator[PairChunk]:
    """Yield a phase-free DCT edge stream in canonical actor/edge order."""

    checked = validate_prepared_activity_batch(batch)
    chunk_size = validate_edge_chunk_size(edge_chunk_size)
    budget = validate_edge_budget(edge_budget)
    floors = validate_energy_floors(energy_floors)
    required_edges = sum(count * (count - 1) // 2 for count in checked.actor_counts)
    if required_edges > budget:
        raise ResourceLimitError(required_edges=required_edges, edge_budget=budget)

    groups: list[int] = []
    actor_i: list[int] = []
    actor_j: list[int] = []
    forward_tokens: list[np.ndarray] = []
    reverse_tokens: list[np.ndarray] = []
    supports: list[np.ndarray] = []

    def flush() -> PairChunk:
        return PairChunk(
            np.ascontiguousarray(groups, dtype=np.int64),
            np.ascontiguousarray(actor_i, dtype=np.int64),
            np.ascontiguousarray(actor_j, dtype=np.int64),
            np.ascontiguousarray(np.stack(forward_tokens), dtype=np.float32),
            np.ascontiguousarray(np.stack(reverse_tokens), dtype=np.float32),
            np.ascontiguousarray(np.stack(supports), dtype=np.bool_),
        )

    for group, actor_count in enumerate(checked.actor_counts):
        valid_length = checked.valid_lengths[group]
        for left_actor in range(actor_count):
            for right_actor in range(left_actor + 1, actor_count):
                tokens, support = _dct_pair_descriptor(
                    checked.activities[group, left_actor, :valid_length],
                    checked.activities[group, right_actor, :valid_length],
                    checked.activity_mask[group, left_actor, :valid_length],
                    checked.activity_mask[group, right_actor, :valid_length],
                    floors,
                )
                groups.append(group)
                actor_i.append(left_actor)
                actor_j.append(right_actor)
                forward_tokens.append(tokens)
                reverse_tokens.append(tokens.copy())
                supports.append(support)
                if len(groups) == chunk_size:
                    yield flush()
                    groups.clear()
                    actor_i.clear()
                    actor_j.clear()
                    forward_tokens.clear()
                    reverse_tokens.clear()
                    supports.clear()
    if groups:
        yield flush()


class _ControlledPhaseSetEncoder(PhaseSetEncoder):
    def __init__(self, *, system_id: str, incidence_seed: int, **kwargs: object) -> None:
        self.control_spec = system_spec(system_id)
        if type(incidence_seed) is not int or not 0 <= incidence_seed < 2**63:
            raise PhaseSetControlError("incidence_seed must be an int in [0,2^63)")
        self.incidence_seed = incidence_seed
        super().__init__(**kwargs)

    def _edge_forward(
        self,
        relation_ij: Tensor,
        relation_ji: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        policy = self.control_spec.feature_policy
        gated_ij = gate_relation_descriptors(relation_ij, policy)
        gated_ji = gate_relation_descriptors(relation_ji, policy)
        half_ij, half_ji, pair_tokens = super()._edge_forward(gated_ij, gated_ji)
        if policy == "COVERAGE_MISSING_ONLY":
            # Retain zero-gradient graph links so every system has the same
            # trainable parameter topology while only coverage/missingness can
            # reach the topology MLP.
            half_ij = half_ij - half_ij
            half_ji = half_ji - half_ji
            pair_tokens = pair_tokens - pair_tokens
        return half_ij, half_ji, pair_tokens

    def _iter_pair_chunks(
        self,
        batch: PreparedActivityBatch,
        *,
        edge_chunk_size: int,
    ) -> Iterator[PairChunk]:
        if self.control_spec.feature_policy == "BAND_ID_ONLY":
            return iter_generic_token_chunks(
                batch,
                edge_chunk_size=edge_chunk_size,
                edge_budget=self.edge_budget,
            )
        if self.control_spec.feature_policy == "MARGINAL_POWER":
            return iter_marginal_power_pair_chunks(
                batch,
                edge_chunk_size=edge_chunk_size,
                edge_budget=self.edge_budget,
            )
        if self.control_spec.feature_policy == "MEAN_DIFFERENCE_DCT":
            return iter_mean_difference_dct_chunks(
                batch,
                energy_floors=self._energy_floors,
                edge_chunk_size=edge_chunk_size,
                edge_budget=self.edge_budget,
            )
        if self.control_spec.feature_policy == "COVERAGE_MISSING_ONLY":
            return iter_observation_availability_pair_chunks(
                batch,
                edge_chunk_size=edge_chunk_size,
                edge_budget=self.edge_budget,
            )
        return super()._iter_pair_chunks(batch, edge_chunk_size=edge_chunk_size)

    def _expected_descriptor_stream_kind(self) -> str | None:
        policy = self.control_spec.feature_policy
        if policy == "MARGINAL_POWER":
            return "MARGINAL_POWER"
        if policy == "MEAN_DIFFERENCE_DCT":
            return "MEAN_DIFFERENCE_DCT"
        if policy in ("FULL_RELATION", "SHUFFLED_INCIDENCE", "PHASE_STRIPPED"):
            return "FULL_RELATION"
        return None

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
        if self.control_spec.feature_policy != "SHUFFLED_INCIDENCE":
            return half_ij, half_ji, pair_tokens
        # System 06 is routed group-globally by the streaming reducer after a
        # support census.  Returning the differentiable values unchanged here
        # avoids any microblock-local permutation or O(K^2 D) activation cache.
        del support_mask, batch_indices, actor_i, actor_j
        return half_ij, half_ji, pair_tokens


class PhaseSetSystem(nn.Module):
    """Uniform token interface for frozen systems 00--08."""

    def __init__(
        self,
        system_id: str,
        *,
        embedding_dim: int = 512,
        hidden_dim: int = 256,
        energy_floors: np.ndarray | None = None,
        topology_scale: float = 1.0,
        edge_budget: int = DEFAULT_EDGE_BUDGET,
        incidence_seed: int = 1729,
    ) -> None:
        super().__init__()
        self.spec = system_spec(system_id)
        self.embedding_dim = embedding_dim
        self.register_buffer("_device_anchor", torch.zeros((), dtype=torch.float32))
        self.encoder: _ControlledPhaseSetEncoder | None
        if self.spec.residual_enabled:
            self.encoder = _ControlledPhaseSetEncoder(
                system_id=system_id,
                incidence_seed=incidence_seed,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                energy_floors=energy_floors,
                topology_scale=topology_scale,
                edge_budget=edge_budget,
            )
        else:
            self.encoder = None

    @property
    def system_id(self) -> str:
        return self.spec.system_id

    def _zero_output(self, batch: PreparedActivityBatch) -> GroupTokenOutput:
        checked = validate_prepared_activity_batch(batch)
        batch_size = checked.batch_size
        device = self._device_anchor.device
        tokens = torch.zeros(
            (batch_size, BAND_COUNT, self.embedding_dim),
            dtype=torch.float32,
            device=device,
        )
        mask = torch.zeros((batch_size, BAND_COUNT), dtype=torch.bool, device=device)
        counts = torch.zeros((batch_size, BAND_COUNT), dtype=torch.int64, device=device)
        return GroupTokenOutput(
            tokens, mask, tokens.clone(), tokens.clone(), counts, counts.clone()
        )

    def forward(
        self,
        batch: PreparedGroupBatch,
        *,
        edge_chunk_size: int = DEFAULT_EDGE_CHUNK_SIZE,
    ) -> GroupTokenOutput:
        checked = validate_prepared_group_batch(batch)
        return self.forward_activity(
            skeleton_to_activity(checked),
            edge_chunk_size=edge_chunk_size,
        )

    def forward_cached(
        self,
        batch: PreparedGroupBatch,
        *,
        descriptor_stream: CachedPairChunkStream,
        edge_chunk_size: int = DEFAULT_EDGE_CHUNK_SIZE,
    ) -> GroupTokenOutput:
        """Run an enabled system from its exact registered cached input family."""

        checked = validate_prepared_group_batch(batch)
        if self.encoder is None:
            raise PhaseSetControlError(
                "system 00 does not admit cached descriptor execution"
            )
        return self.encoder.forward_cached(
            checked,
            descriptor_stream=descriptor_stream,
            edge_chunk_size=edge_chunk_size,
            include_topology=self.spec.include_topology,
        )

    def forward_activity(
        self,
        batch: PreparedActivityBatch,
        *,
        edge_chunk_size: int = DEFAULT_EDGE_CHUNK_SIZE,
    ) -> GroupTokenOutput:
        checked = validate_prepared_activity_batch(batch)
        if self.encoder is None:
            return self._zero_output(checked)
        return self.encoder.forward_activity(
            checked,
            edge_chunk_size=edge_chunk_size,
            include_topology=self.spec.include_topology,
        )


def build_phaseset_system(
    system_id: str,
    **kwargs: object,
) -> PhaseSetSystem:
    """Construct exactly one frozen final-system row without training it."""

    return PhaseSetSystem(system_id, **kwargs)


def trainable_parameter_count(module: nn.Module) -> int:
    if not isinstance(module, nn.Module):
        raise TypeError("module must be a torch.nn.Module")
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def audit_control_parameter_counts(
    systems: Mapping[str, PhaseSetSystem],
) -> tuple[tuple[str, int], ...]:
    """Audit real module parameters, requiring systems 01--07 within 1% of 08."""

    if type(systems) is not dict or set(systems) != set(RESIDUAL_SYSTEM_IDS):
        raise PhaseSetControlError("systems must be an exact dict containing 01--08")
    counts: dict[str, int] = {}
    for system_id in RESIDUAL_SYSTEM_IDS:
        system = systems[system_id]
        if type(system) is not PhaseSetSystem or system.system_id != system_id:
            raise PhaseSetControlError("system mapping identity mismatch")
        counts[system_id] = trainable_parameter_count(system)
    full = counts["08"]
    if full <= 0:
        raise PhaseSetControlError("full system has no trainable parameters")
    for system_id in RESIDUAL_SYSTEM_IDS[:-1]:
        if Fraction(abs(counts[system_id] - full), full) > Fraction(1, 100):
            raise PhaseSetControlError(f"system {system_id} exceeds the +/-1% bound")
    return tuple((system_id, counts[system_id]) for system_id in RESIDUAL_SYSTEM_IDS)


__all__ = [
    "PhaseSetControlError",
    "PhaseSetSystem",
    "RESIDUAL_SYSTEM_IDS",
    "ResidualSystemSpec",
    "STATUS",
    "SYSTEM_IDS",
    "SYSTEM_SPECS",
    "audit_control_parameter_counts",
    "build_phaseset_system",
    "gate_relation_descriptors",
    "iter_generic_token_chunks",
    "iter_mean_difference_dct_chunks",
    "shuffle_half_edge_incidence",
    "system_spec",
    "trainable_parameter_count",
]
