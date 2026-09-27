"""TMR's ACTOR VAE and complete objective, not yet a TMR-Set baseline.

Adapted from Mathux/TMR at 6d74688730d15d43b0a755ce2b0e1f2d76138fc1
(MIT, copyright 2023 Mathis Petrovich; see LICENSE_TMR.md).
Defaults follow the training YAML, not the upstream class defaults. Frozen
DistilBERT token features and separate normalized MPNet sentence features must
be supplied by the caller. No language weights, split access, or optimizer live
here. The rectangular multi-positive/filter extension is explicitly documented
in docs/TMR_COMPONENTS.md; it does not create new ground-truth positives.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

Gaussian = tuple[Tensor, Tensor]


class PositionalEncoding(nn.Module):
    """Original analytic sin/cos encoding with an explicit length bound."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        if d_model < 2 or d_model % 2 or max_len < 1:
            raise ValueError("an even positive dimension and positive max_len are required")
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] > self.pe.shape[0]:
            raise ValueError("sequence exceeds explicit positional bound; never truncate it")
        return self.dropout(x + self.pe.permute(1, 0, 2)[:, : x.shape[1]])


def _sequence(x: Tensor, mask: Tensor, nfeats: int) -> None:
    if x.ndim != 3 or x.shape[-1] != nfeats or mask.shape != x.shape[:2]:
        raise ValueError("expected [batch,time,features] and matching [batch,time] mask")
    if mask.dtype != torch.bool or mask.device != x.device or not bool(mask.any(1).all()):
        raise ValueError("every sequence needs at least one valid position on its device")


class ACTORStyleEncoder(nn.Module):
    """Two distribution tokens, six-layer temporal Transformer, shared weights."""

    def __init__(
        self,
        nfeats: int,
        *,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()
        self.nfeats = nfeats
        self.projection = nn.Linear(nfeats, latent_dim)
        self.tokens = nn.Parameter(torch.randn(2, latent_dim))
        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout, max_len)
        layer = nn.TransformerEncoderLayer(
            latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation="gelu",
            batch_first=True,
        )
        self.seqTransEncoder = nn.TransformerEncoder(layer, num_layers)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        _sequence(x, mask, self.nfeats)
        x = self.projection(x)
        tokens = self.tokens.unsqueeze(0).expand(len(x), -1, -1)
        xseq = torch.cat((tokens, x), 1)
        token_mask = torch.ones((len(x), 2), dtype=torch.bool, device=x.device)
        aug_mask = torch.cat((token_mask, mask), 1)
        encoded = self.seqTransEncoder(
            self.sequence_pos_encoding(xseq),
            src_key_padding_mask=~aug_mask,
        )
        return encoded[:, :2]


class ACTORStyleDecoder(nn.Module):
    """Single latent memory and zero time queries plus analytic positions."""

    def __init__(
        self,
        nfeats: int,
        *,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()
        self.nfeats = nfeats
        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout, max_len)
        layer = nn.TransformerDecoderLayer(
            latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation="gelu",
            batch_first=True,
        )
        self.seqTransDecoder = nn.TransformerDecoder(layer, num_layers)
        self.final_layer = nn.Linear(latent_dim, nfeats)

    def forward(self, z: Tensor, mask: Tensor) -> Tensor:
        if z.ndim != 2 or mask.ndim != 2 or len(z) != len(mask):
            raise ValueError("expected [batch,latent] and [batch,time] mask")
        if mask.dtype != torch.bool or mask.device != z.device or not bool(mask.any(1).all()):
            raise ValueError("every decoder sequence needs at least one valid position")
        queries = torch.zeros(len(z), mask.shape[1], z.shape[1], device=z.device, dtype=z.dtype)
        output = self.seqTransDecoder(
            tgt=self.sequence_pos_encoding(queries),
            memory=z[:, None],
            tgt_key_padding_mask=~mask,
        )
        return self.final_layer(output).masked_fill(~mask[..., None], 0.0)


def sample_diagonal_gaussian(distribution: Gaussian, *, sample_mean: bool = False) -> Tensor:
    mu, logvar = distribution
    if sample_mean:
        return mu
    # Keep upstream TEMOS's operation order, not the algebraically equivalent
    # exp(logvar / 2): this also makes its RNG/gradient oracle meaningful.
    std = logvar.exp().pow(0.5)
    return mu + torch.empty_like(std).normal_() * std


def diagonal_kl(q: Gaussian, p: Gaussian) -> Tensor:
    """Upstream KLLoss's element mean, not a sum over latent coordinates."""
    mu_q, logvar_q = q
    mu_p, logvar_p = p
    log_ratio = logvar_q - logvar_p
    return (0.5 * (log_ratio.exp() + (mu_p - mu_q).pow(2) / logvar_p.exp() - 1 - log_ratio)).mean()


def reconstruction_loss(
    predicted: Tensor, target: Tensor, observed: Tensor | None = None
) -> Tensor:
    """Original padded mean by default; explicit observed-only extension for gaps."""
    if predicted.shape != target.shape:
        raise ValueError("reconstruction must have exactly the target shape")
    if observed is None:
        return F.smooth_l1_loss(predicted, target)
    if (
        observed.dtype != torch.bool
        or observed.shape != target.shape
        or observed.device != target.device
    ):
        raise ValueError("observed must be a feature-level boolean mask matching the target")
    if not bool(observed.any()):
        raise ValueError("cannot reconstruct a sequence without observed features")
    return F.smooth_l1_loss(predicted[observed], target[observed])


def tmr_auxiliary_losses(
    *,
    text_reconstruction: Tensor,
    motion_reconstruction: Tensor,
    target: Tensor,
    text_latent: Tensor,
    motion_latent: Tensor,
    text_distribution: Gaussian,
    motion_distribution: Gaussian,
    observed: Tensor | None = None,
) -> dict[str, Tensor]:
    """Both reconstructions, all four KLs, and the sampled latent alignment."""
    if text_latent.shape != motion_latent.shape:
        raise ValueError("auxiliary pairs must be aligned before computing TMR losses")
    reference = (torch.zeros_like(motion_distribution[0]), torch.zeros_like(motion_distribution[1]))
    return {
        "recons": reconstruction_loss(text_reconstruction, target, observed)
        + reconstruction_loss(motion_reconstruction, target, observed),
        "kl": diagonal_kl(text_distribution, motion_distribution)
        + diagonal_kl(motion_distribution, text_distribution)
        + diagonal_kl(motion_distribution, reference)
        + diagonal_kl(text_distribution, reference),
        "latent": F.smooth_l1_loss(text_latent, motion_latent),
    }


def semantic_negative_mask(
    positive: Tensor, sentence_embeddings: Tensor, threshold: float
) -> Tensor:
    """Remove similar negatives only; never turn similarity into a positive label.

    For a motion with multiple known captions, use maximum similarity against
    those captions. Diagonal single-positive batches reduce to TMR's filter.
    These frozen sentence features must come only from the active learning pool.
    """
    if not 0 < threshold <= 1:
        raise ValueError("the enabled TMR threshold must be in (0,1]")
    if sentence_embeddings.ndim != 2 or len(sentence_embeddings) != positive.shape[1]:
        raise ValueError("need one separate sentence-similarity embedding per text row")
    sentence_embeddings = sentence_embeddings.detach().to(device=positive.device)
    if not bool(torch.isfinite(sentence_embeddings).all()):
        raise ValueError("sentence embeddings must be finite")
    norms = sentence_embeddings.norm(dim=1)
    if not bool(torch.allclose(norms, torch.ones_like(norms), rtol=1e-4, atol=1e-5)):
        raise ValueError("TMR sentence-similarity embeddings must already be normalized")
    similar = sentence_embeddings @ sentence_embeddings.T > (2 * threshold - 1)
    filtered = (positive.to(torch.float32) @ similar.to(torch.float32)) > 0
    return filtered & ~positive


def filtered_contrastive_loss(
    motion_latent: Tensor,
    text_latent: Tensor,
    positive: Tensor,
    *,
    sentence_embeddings: Tensor | None = None,
    temperature: float = 0.1,
    threshold_selfsim: float = 0.8,
) -> Tensor:
    """Variable-positive symmetric InfoNCE with TMR negative filtering."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if (
        motion_latent.ndim != 2
        or text_latent.ndim != 2
        or motion_latent.shape[1] != text_latent.shape[1]
    ):
        raise ValueError("motion and text need compatible latent matrices")
    if positive.dtype != torch.bool or positive.shape != (len(motion_latent), len(text_latent)):
        raise ValueError("positive must be a [motion,text] boolean matrix")
    if positive.device != motion_latent.device or text_latent.device != motion_latent.device:
        raise ValueError("latents and positive matrix must share a device")
    if not bool(positive.any(0).all()) or not bool(positive.any(1).all()):
        raise ValueError("each motion and text needs at least one known positive")
    logits = F.normalize(motion_latent, dim=-1) @ F.normalize(text_latent, dim=-1).T / temperature
    if sentence_embeddings is not None and threshold_selfsim:
        logits = logits.masked_fill(
            semantic_negative_mask(positive, sentence_embeddings, threshold_selfsim),
            -torch.inf,
        )
    positives = logits.masked_fill(~positive, -torch.inf)
    return 0.5 * (
        (torch.logsumexp(logits, 1) - torch.logsumexp(positives, 1)).mean()
        + (torch.logsumexp(logits, 0) - torch.logsumexp(positives, 0)).mean()
    )


@dataclass(frozen=True)
class TMRLossWeights:
    recons: float = 1.0
    latent: float = 1e-5
    kl: float = 1e-5
    contrastive: float = 0.1

    def total(self, losses: dict[str, Tensor]) -> Tensor:
        return sum(
            getattr(self, key) * losses[key] for key in ("recons", "kl", "latent", "contrastive")
        )


class TMRSinglePerson(nn.Module):
    """Full original-task VAE forward/objective, not the multi-person adapter.

    Input features remain caller-defined (the original task's motion feature
    preprocessing is not replaced with skeleton flattening here). Text input is
    token-level [B,L,768] by default, not a CLIP sentence embedding. This module
    is useful for implementation equivalence checks, not a completed replication.
    """

    def __init__(self, motion_nfeats: int, text_nfeats: int = 768, **actor_kwargs):
        super().__init__()
        self.motion_encoder = ACTORStyleEncoder(motion_nfeats, **actor_kwargs)
        self.text_encoder = ACTORStyleEncoder(text_nfeats, **actor_kwargs)
        self.motion_decoder = ACTORStyleDecoder(motion_nfeats, **actor_kwargs)
        self.loss_weights = TMRLossWeights()

    def encode(self, x: Tensor, mask: Tensor, *, modality: str, sample_mean: bool = True):
        if modality not in ("text", "motion"):
            raise ValueError("modality must be text or motion")
        encoder = self.text_encoder if modality == "text" else self.motion_encoder
        encoded = encoder(x, mask)
        distribution = (encoded[:, 0], encoded[:, 1])
        return sample_diagonal_gaussian(distribution, sample_mean=sample_mean), distribution

    def compute_loss(
        self,
        motion: Tensor,
        motion_mask: Tensor,
        text_tokens: Tensor,
        text_mask: Tensor,
        sentence_embeddings: Tensor,
        *,
        observed: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if len(motion) != len(text_tokens):
            raise ValueError("original-task auxiliary pairs require equal batch sizes")
        # Preserve upstream stochastic execution order: text first, motion next.
        text_z, text_dist = self.encode(text_tokens, text_mask, modality="text", sample_mean=False)
        text_motion = self.motion_decoder(text_z, motion_mask)
        motion_z, motion_dist = self.encode(
            motion, motion_mask, modality="motion", sample_mean=False
        )
        motion_motion = self.motion_decoder(motion_z, motion_mask)
        losses = tmr_auxiliary_losses(
            text_reconstruction=text_motion,
            motion_reconstruction=motion_motion,
            target=motion,
            text_latent=text_z,
            motion_latent=motion_z,
            text_distribution=text_dist,
            motion_distribution=motion_dist,
            observed=observed,
        )
        positive = torch.eye(len(motion), dtype=torch.bool, device=motion.device)
        losses["contrastive"] = filtered_contrastive_loss(
            motion_z,
            text_z,
            positive,
            sentence_embeddings=sentence_embeddings,
        )
        losses["loss"] = self.loss_weights.total(losses)
        return losses
