"""Explicit full-parent TMR-Set adaptation with track-consistent set reconstruction.

TMR's actual VAE encoders/decoder and all auxiliary objectives are retained.
Shared per-actor/window encoding, set PMA, full-window temporal hierarchy, and
dynamic output slots are disclosed adaptations, not the original single-person
architecture. Slot-to-track assignment is solved once over the entire parent,
never independently at each window. No actor ordinal enters the encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .continuous_capture import PreparedContinuousCapture
from .models import PMAPool
from .tmr_components import (
    ACTORStyleDecoder,
    ACTORStyleEncoder,
    Gaussian,
    TMRLossWeights,
    diagonal_kl,
    filtered_contrastive_loss,
    sample_diagonal_gaussian,
)


@dataclass(frozen=True)
class TMRGroupWindow:
    features: Tensor  # [K,<=200,66], shared capture origin, not per-actor centering
    observed: Tensor  # same shape, feature-level observation mask
    actor_commitments: tuple[bytes, ...]  # lineage/reduction ordering only
    start_seconds: float

    def __post_init__(self):
        x = self.features
        if x.dtype != torch.float32 or x.ndim != 3 or x.shape[0] < 2 or x.shape[2] != 66:
            raise ValueError("TMR-Set windows need float32 [K>=2,T,66] body22 coordinates")
        if not 1 <= x.shape[1] <= 200 or not bool(torch.isfinite(x).all()):
            raise ValueError("window must contain 1..200 finite frames; never truncate")
        if (
            self.observed.dtype != torch.bool
            or self.observed.shape != x.shape
            or self.observed.device != x.device
        ):
            raise ValueError("observed must match every body coordinate")
        if not bool(self.observed.flatten(1).any(1).all()):
            raise ValueError("every actor needs observed coordinates")
        unobserved = x[~self.observed]
        if bool((unobserved != 0).any()) or bool(torch.signbit(unobserved).any()):
            raise ValueError("unobserved coordinates must be exact +0")
        if len(self.actor_commitments) != len(x) or len(set(self.actor_commitments)) != len(x):
            raise ValueError("actor lineage must uniquely cover every actor")
        if any(type(key) is not bytes or len(key) != 32 for key in self.actor_commitments):
            raise ValueError("actor commitments must be bytes32, not model features")
        if not math.isfinite(self.start_seconds) or self.start_seconds < 0:
            raise ValueError("absolute window start must be finite and nonnegative")

    def canonical(self, device: torch.device) -> tuple[Tensor, Tensor]:
        order = sorted(range(len(self.actor_commitments)), key=self.actor_commitments.__getitem__)
        index = torch.tensor(order, dtype=torch.int64, device=self.features.device)
        return (
            self.features.index_select(0, index).to(device),
            self.observed.index_select(0, index).to(device),
        )


@dataclass(frozen=True)
class TMRGroupCapture:
    windows: tuple[TMRGroupWindow, ...]

    def __post_init__(self):
        if not self.windows:
            raise ValueError("a complete parent needs all its accepted windows")
        actors = set(self.windows[0].actor_commitments)
        starts = tuple(window.start_seconds for window in self.windows)
        if starts != tuple(sorted(set(starts))):
            raise ValueError("windows must preserve strict absolute chronology including gaps")
        if any(set(window.actor_commitments) != actors for window in self.windows):
            raise ValueError("track membership cannot change between windows")

    @classmethod
    def from_prepared(cls, capture: PreparedContinuousCapture) -> TMRGroupCapture:
        if type(capture) is not PreparedContinuousCapture:
            raise ValueError("need an admitted complete prepared parent")
        return cls(
            tuple(
                TMRGroupWindow(
                    torch.from_numpy(window.skeletons.copy()).flatten(2),
                    torch.from_numpy(window.track_mask.copy())
                    .unsqueeze(-1)
                    .expand(-1, -1, -1, 3)
                    .flatten(2),
                    window.actor_commitments,
                    window.source_start_frame / 30.0,
                )
                for window in capture.windows()
            )
        )


@dataclass(frozen=True)
class TMRTextBatch:
    tokens: Tensor  # frozen DistilBERT [Q,L,768]
    token_mask: Tensor
    sentences: Tensor  # separate frozen normalized MPNet [Q,768]

    def __post_init__(self):
        if self.tokens.ndim != 3 or self.tokens.shape[2] != 768 or len(self.tokens) < 1:
            raise ValueError("TMR requires token-level [Q,L,768] language features")
        if self.token_mask.dtype != torch.bool or self.token_mask.shape != self.tokens.shape[:2]:
            raise ValueError("text token mask must cover the token sequence")
        if not bool(self.token_mask.any(1).all()):
            raise ValueError("every text row needs a valid token")
        if self.sentences.shape != (len(self.tokens), 768):
            raise ValueError("one separate 768D MPNet sentence embedding is needed per row")
        for values in (self.tokens, self.sentences):
            if (
                values.dtype != torch.float32
                or values.requires_grad
                or not bool(torch.isfinite(values).all())
            ):
                raise ValueError("pretrained language features must be frozen finite float32")


def absolute_sincos(positions: Tensor, width: int) -> Tensor:
    if width % 2:
        raise ValueError("sin/cos width must be even")
    scales = torch.exp(
        torch.arange(0, width, 2, device=positions.device, dtype=torch.float32)
        * (-math.log(10000.0) / width)
    )
    angles = positions.to(torch.float32)[:, None] * scales
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)


def minimum_assignment(cost: Tensor) -> tuple[int, ...]:
    """Exact square Hungarian assignment (O(K^3)), lowest-index tie breaking.

    Only a K-by-K scalar cost table is materialized; no K-by-K-by-time tensor.
    The discrete choice uses detached costs. Gradients flow through selected
    costs, as in a standard set reconstruction objective, not the argmin.
    """
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1] or len(cost) < 1:
        raise ValueError("matching needs one nonempty square cost table")
    if not bool(torch.isfinite(cost).all()):
        raise ValueError("nonfinite reconstruction costs cannot be assigned")
    values = cost.detach().to(device="cpu", dtype=torch.float64).tolist()
    count = len(values)
    u, v = [0.0] * (count + 1), [0.0] * (count + 1)
    p, way = [0] * (count + 1), [0] * (count + 1)
    for row in range(1, count + 1):
        p[0] = row
        current = 0
        minimum = [math.inf] * (count + 1)
        used = [False] * (count + 1)
        while True:
            used[current] = True
            active_row = p[current]
            delta, following = math.inf, 0
            for column in range(1, count + 1):
                if not used[column]:
                    reduced = values[active_row - 1][column - 1] - u[active_row] - v[column]
                    if reduced < minimum[column]:
                        minimum[column], way[column] = reduced, current
                    if minimum[column] < delta:
                        delta, following = minimum[column], column
            for column in range(count + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            current = following
            if p[current] == 0:
                break
        while current:
            previous = way[current]
            p[current] = p[previous]
            current = previous
    assignment = [0] * count
    for column in range(1, count + 1):
        assignment[p[column] - 1] = column - 1
    return tuple(assignment)


def track_costs(prediction: Tensor, target: Tensor, observed: Tensor) -> Tensor:
    """Per-slot/track unnormalized loss sums, without dense pair/frame features."""
    if prediction.shape != target.shape or observed.shape != target.shape:
        raise ValueError("slots/targets/masks must share [K,T,66] shape")
    columns = []
    for actor in range(len(target)):
        element = F.smooth_l1_loss(
            prediction, target[actor].expand_as(prediction), reduction="none"
        )
        columns.append(element.masked_fill(~observed[actor][None], 0).to(torch.float64).sum((1, 2)))
    return torch.stack(columns, dim=1)


class TMRSet(nn.Module):
    """Trainable full-parent set VAE, with meaningful text-to-group reconstruction.

    Six-layer original ACTOR blocks remain 256D/1024FFN/4heads/.1. The extra
    capture ACTOR hierarchy sees every accepted window and its absolute time.
    Reconstructed slot count comes from observed K only for the auxiliary
    decoder; no ordinal, commitment, or count feature is appended to retrieval.
    Caller still owns split, asset/provenance, seed, and training-budget admission.
    """

    system_id = "TMR-Set"

    def __init__(
        self,
        *,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        checkpoint_windows: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.checkpoint_windows = checkpoint_windows
        kwargs = dict(
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.actor_encoder = ACTORStyleEncoder(66, **kwargs)
        self.actor_pool = PMAPool(latent_dim, num_heads, ff_size, dropout)
        self.capture_encoder = ACTORStyleEncoder(latent_dim, **kwargs)
        self.text_encoder = ACTORStyleEncoder(768, **kwargs)
        self.motion_decoder = ACTORStyleDecoder(66, **kwargs)
        self.slot_projection = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.GELU())
        self.time_projection = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.GELU())
        self.loss_weights = TMRLossWeights()

    def encode_capture(self, capture: TMRGroupCapture) -> Gaussian:
        device = self.actor_encoder.tokens.device
        summaries = []
        for window in capture.windows:

            def encode(witness, window=window):
                x, observed = window.canonical(witness.device)
                actors = self.actor_encoder(x, observed.any(-1))[:, 0]
                return self.actor_pool(
                    actors[None], torch.ones(1, len(actors), device=device, dtype=torch.bool)
                )[0]

            summaries.append(
                checkpoint(
                    encode, self.actor_encoder.tokens, use_reentrant=False, preserve_rng_state=True
                )
                if self.checkpoint_windows and torch.is_grad_enabled()
                else encode(self.actor_encoder.tokens)
            )
        times = torch.tensor([window.start_seconds for window in capture.windows], device=device)
        sequence = torch.stack(summaries) + absolute_sincos(times, self.latent_dim)
        encoded = self.capture_encoder(
            sequence[None], torch.ones(1, len(sequence), device=device, dtype=torch.bool)
        )
        return encoded[0, 0], encoded[0, 1]

    def reconstruct_capture(self, z: Tensor, capture: TMRGroupCapture) -> Tensor:
        actor_count = len(capture.windows[0].actor_commitments)
        slot_positions = torch.arange(actor_count, dtype=torch.float32, device=z.device)
        slot_context = self.slot_projection(absolute_sincos(slot_positions, self.latent_dim))
        costs = torch.zeros(actor_count, actor_count, dtype=torch.float64, device=z.device)
        observed_count = 0
        for window in capture.windows:
            # Bind the window and count now; checkpoint replay must not use the
            # last loop iteration. Explicit z discovers CUDA's RNG device.
            def decode(latent, slots, window=window):
                target, observed = window.canonical(latent.device)
                time = torch.tensor([window.start_seconds], device=latent.device)
                memory = (
                    latent[None]
                    + slots
                    + self.time_projection(absolute_sincos(time, self.latent_dim))
                )
                # Slot masks cannot borrow a particular target track's missing
                # pattern before matching. Decode all positions, then mask cost.
                mask = torch.ones(
                    actor_count, target.shape[1], device=latent.device, dtype=torch.bool
                )
                return track_costs(self.motion_decoder(memory, mask), target, observed)

            costs = costs + (
                checkpoint(decode, z, slot_context, use_reentrant=False, preserve_rng_state=True)
                if self.checkpoint_windows and torch.is_grad_enabled()
                else decode(z, slot_context)
            )
            observed_count += int(window.observed.sum().item())
        assignment = torch.tensor(minimum_assignment(costs), device=z.device, dtype=torch.int64)
        selected = costs[torch.arange(actor_count, device=z.device), assignment].sum()
        return (selected / observed_count).to(torch.float32)

    def text_distributions(self, text: TMRTextBatch) -> Gaussian:
        device = self.text_encoder.tokens.device
        encoded = self.text_encoder(text.tokens.to(device), text.token_mask.to(device))
        return encoded[:, 0], encoded[:, 1]

    def score(self, captures: tuple[TMRGroupCapture, ...], text: TMRTextBatch) -> Tensor:
        motion = torch.stack([self.encode_capture(capture)[0] for capture in captures])
        text_mu, _ = self.text_distributions(text)
        return F.normalize(motion, dim=1) @ F.normalize(text_mu, dim=1).T / 0.1

    def compute_loss(
        self, captures: tuple[TMRGroupCapture, ...], text: TMRTextBatch, positive: Tensor
    ) -> dict[str, Tensor]:
        device = self.text_encoder.tokens.device
        positive = positive.to(device)
        if positive.dtype != torch.bool or positive.shape != (len(captures), len(text.tokens)):
            raise ValueError("known positives must match complete parents and all human text rows")
        if not bool(positive.any(0).all()) or not bool(positive.any(1).all()):
            raise ValueError("each parent/text must have a known positive")
        text_dist = self.text_distributions(text)
        text_z = sample_diagonal_gaussian(text_dist)
        motion_rows = [self.encode_capture(capture) for capture in captures]
        motion_dist = tuple(torch.stack([row[index] for row in motion_rows]) for index in (0, 1))
        motion_z = sample_diagonal_gaussian(motion_dist)
        recons, kls, latents = [], [], []
        for row, capture in enumerate(captures):
            indices = positive[row].nonzero().flatten()
            # Each parent's captions receive equal weight within that parent;
            # caption-rich parents do not multiply their auxiliary weight.
            text_recons = torch.stack(
                [self.reconstruct_capture(text_z[index], capture) for index in indices]
            ).mean()
            recons.append(text_recons + self.reconstruct_capture(motion_z[row], capture))
            td = tuple(part[indices] for part in text_dist)
            md = tuple(part[row].expand_as(td[index]) for index, part in enumerate(motion_dist))
            unit = (torch.zeros_like(md[0]), torch.zeros_like(md[1]))
            kls.append(
                diagonal_kl(td, md)
                + diagonal_kl(md, td)
                + diagonal_kl(md, unit)
                + diagonal_kl(td, unit)
            )
            latents.append(
                F.smooth_l1_loss(text_z[indices], motion_z[row].expand_as(text_z[indices]))
            )
        losses = {
            "recons": torch.stack(recons).mean(),
            "kl": torch.stack(kls).mean(),
            "latent": torch.stack(latents).mean(),
            "contrastive": filtered_contrastive_loss(
                motion_z, text_z, positive, sentence_embeddings=text.sentences.to(device)
            ),
        }
        losses["loss"] = self.loss_weights.total(losses)
        return losses
