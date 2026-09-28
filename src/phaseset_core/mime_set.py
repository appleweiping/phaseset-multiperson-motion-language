"""MIME's dyadic equations and an explicit shared complete-parent extension.

Independent paper-defined implementation, not released author code. Original
135D SMPL inputs/role-specific streams and shared body22 multi-person inputs
are distinct APIs. See docs/MIME_SET_ADAPTATION.md. Language is trainable CLIP
in mime_language.py; a frozen sentence-vector replacement is not MIME here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .tmr_set import TMRGroupCapture, absolute_sincos
from .wamo_set import positive_contrastive


@dataclass(frozen=True)
class MIMEConfig:
    width: int = 512
    heads: int = 4
    layers: int = 4
    ff_size: int = 2048
    dropout: float = 0.1
    epsilon: float = 1e-8

    def __post_init__(self):
        if any(
            type(x) is not int or x < 1 for x in (self.width, self.heads, self.layers, self.ff_size)
        ):
            raise ValueError("positive integer model dimensions required")
        if self.width % 2 or self.width % self.heads:
            raise ValueError("even width must be divisible by heads")
        if not (0 <= self.dropout < 1 and math.isfinite(self.epsilon) and self.epsilon > 0):
            raise ValueError("invalid dropout or root-energy epsilon")


def pair_features(
    a: Tensor,
    b: Tensor,
    roots_a: Tensor,
    roots_b: Tensor,
    root_delta_a: Tensor,
    root_delta_b: Tensor,
    joint_valid: Tensor,
    epsilon: float = 1e-8,
) -> tuple[Tensor, Tensor, Tensor]:
    """Paper Eq.3..6; either original135 or disclosed body22-coordinate inputs."""
    difference = roots_a - roots_b
    distance = difference.norm(dim=-1, keepdim=True)
    aa = torch.cat((a, distance, difference), -1)
    bb = torch.cat((b, distance, -difference), -1)
    energy_a = root_delta_a.square().sum(-1, keepdim=True)
    energy_b = root_delta_b.square().sum(-1, keepdim=True)
    rho = energy_a / (energy_a + energy_b + epsilon)
    relation = torch.cat((aa, bb, bb - aa, rho), -1)
    # Missing pelvis does not become an invented zero-distance interaction.
    relation = relation.masked_fill(~joint_valid[..., None], 0)
    root_augmentation = joint_valid[..., None]
    aa = torch.cat((a, aa[..., a.shape[-1] :].masked_fill(~root_augmentation, 0)), -1)
    bb = torch.cat((b, bb[..., b.shape[-1] :].masked_fill(~root_augmentation, 0)), -1)
    return aa, bb, relation


class QueryPool(nn.Module):
    """Original dot-product query pooling, not mean or first-frame pooling."""

    def __init__(self, width: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(width) / math.sqrt(width))

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        if not bool(valid.any(-1).all()):
            raise ValueError("query pooling needs a jointly observed frame")
        scores = (x * self.query).sum(-1).masked_fill(~valid, -torch.inf)
        return (scores.softmax(-1)[..., None] * x).sum(-2)


class CoAttentionLayer(nn.Module):
    """Pre-LN self attention, simultaneous full-time cross attention, GELU FFN.

    Shared=True ties weights across endpoints, not across layers. Shared=False
    retains original two-role distinct self/cross/FFN blocks. Neither path uses
    updated A when computing B's cross attention. Padding keys are masked.
    """

    def __init__(self, config: MIMEConfig, *, shared: bool):
        super().__init__()
        self.shared = shared
        count = 1 if shared else 2
        self.self_norms = nn.ModuleList([nn.LayerNorm(config.width) for _ in range(count)])
        self.cross_norms = nn.ModuleList([nn.LayerNorm(config.width) for _ in range(count)])
        self.ff_norms = nn.ModuleList([nn.LayerNorm(config.width) for _ in range(count)])
        self.self_attention = nn.ModuleList(
            [
                nn.MultiheadAttention(config.width, config.heads, config.dropout, batch_first=True)
                for _ in range(count)
            ]
        )
        self.cross_attention = nn.ModuleList(
            [
                nn.MultiheadAttention(config.width, config.heads, config.dropout, batch_first=True)
                for _ in range(count)
            ]
        )
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.width, config.ff_size),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.ff_size, config.width),
                )
                for _ in range(count)
            ]
        )
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(
        self, a: Tensor, b: Tensor, mask_a: Tensor, mask_b: Tensor
    ) -> tuple[Tensor, Tensor]:
        masks = (mask_a, mask_b)
        inputs = (a, b)
        self_updated = []
        for endpoint, (x, mask) in enumerate(zip(inputs, masks, strict=True)):
            index = 0 if self.shared else endpoint
            normalized = self.self_norms[index](x)
            attended = self.self_attention[index](
                normalized, normalized, normalized, key_padding_mask=~mask, need_weights=False
            )[0]
            self_updated.append(
                (x + self.residual_dropout(attended)).masked_fill(~mask[..., None], 0)
            )
        cross_sources = tuple(
            self.cross_norms[0 if self.shared else endpoint](x)
            for endpoint, x in enumerate(self_updated)
        )
        outputs = []
        for endpoint, (x, mask) in enumerate(zip(self_updated, masks, strict=True)):
            index = 0 if self.shared else endpoint
            other = 1 - endpoint
            attended = self.cross_attention[index](
                cross_sources[endpoint],
                cross_sources[other],
                cross_sources[other],
                key_padding_mask=~masks[other],
                need_weights=False,
            )[0]
            crossed = x + self.residual_dropout(attended)
            output = crossed + self.residual_dropout(
                self.ffns[index](self.ff_norms[index](crossed))
            )
            outputs.append(output.masked_fill(~mask[..., None], 0))
        return outputs[0], outputs[1]


class OriginalMIMEMotion(nn.Module):
    """Original role-aware dyadic motion architecture; not the group adapter.

    Inputs [B,2,T,135] must truly be root displacement+22 local6Drotations;
    roots [B,2,T,3] supply absolute shared-frame pelvis positions. No conversion
    of body22 XYZ to fictitious rotations happens here. No internal truncation.
    Original Inter-X host separately enforces its disclosed <=300/30fps task.
    """

    def __init__(self, config: MIMEConfig = MIMEConfig()):
        super().__init__()
        self.config = config
        self.projections = nn.ModuleList([nn.Linear(139, config.width) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(config.width) for _ in range(2)])
        self.roles = nn.Parameter(torch.randn(2, config.width) / math.sqrt(config.width))
        self.relation = nn.Linear(418, config.width)
        self.relation_norm = nn.LayerNorm(config.width)
        self.relation_type = nn.Parameter(torch.randn(config.width) / math.sqrt(config.width))
        self.blocks = nn.ModuleList(
            [CoAttentionLayer(config, shared=False) for _ in range(config.layers)]
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * config.width, config.width),
            nn.GELU(),
            nn.Linear(config.width, config.width),
        )
        self.pool = QueryPool(config.width)

    def forward(self, motion: Tensor, roots: Tensor, mask: Tensor) -> Tensor:
        if motion.ndim != 4 or motion.shape[1] != 2 or motion.shape[-1] != 135:
            raise ValueError("original MIME needs true [B,2,T,135] dyadic SMPL inputs")
        if (
            roots.shape != (*motion.shape[:3], 3)
            or mask.shape != motion.shape[:3]
            or mask.dtype != torch.bool
        ):
            raise ValueError("root positions and masks must match both motion streams")
        if not bool(mask.any(-1).all()) or not bool((mask[:, 0] & mask[:, 1]).any(-1).all()):
            raise ValueError("original dyad needs valid actor and jointly valid frames")
        if not bool(torch.isfinite(motion).all()) or not bool(torch.isfinite(roots).all()):
            raise ValueError("original dyad must be finite")
        joint = mask[:, 0] & mask[:, 1]
        aa, bb, relation = pair_features(
            motion[:, 0],
            motion[:, 1],
            roots[:, 0],
            roots[:, 1],
            motion[:, 0, :, :3],
            motion[:, 1, :, :3],
            joint,
            self.config.epsilon,
        )
        rel = (self.relation_norm(self.relation(relation)) + self.relation_type).masked_fill(
            ~joint[..., None], 0
        )
        positions = absolute_sincos(
            torch.arange(motion.shape[2], device=motion.device), self.config.width
        )
        streams = [
            (self.norms[i](self.projections[i](x)) + self.roles[i] + positions + rel).masked_fill(
                ~mask[:, i, :, None], 0
            )
            for i, x in enumerate((aa, bb))
        ]
        a, b = streams
        for block in self.blocks:
            a, b = block(a, b, mask[:, 0], mask[:, 1])
        return self.pool(self.fusion(torch.cat((a, b), -1)), joint)


class MIMESetMotion(nn.Module):
    """Shared body22 endpoint/co-attention model over every unordered edge.

    Directed relational encoding is injected into its own endpoint. Shared
    stream weights plus symmetric sum/absolute-difference/product fusion give
    an unordered pair embedding. Canonical lineage order fixes RNG/reductions;
    lineage is not a neural input. Every window's all edges enter its mean,
    followed by an absolute-gap-aware full-parent temporal readout.
    """

    system_id = "MIME-Set"

    def __init__(self, config: MIMEConfig = MIMEConfig(), *, checkpoint_edges: bool = True):
        super().__init__()
        self.config = config
        self.checkpoint_edges = checkpoint_edges
        self.actor_projection = nn.Linear(70, config.width)
        self.actor_norm = nn.LayerNorm(config.width)
        self.relation_projection = nn.Linear(211, config.width)
        self.relation_norm = nn.LayerNorm(config.width)
        self.relation_type = nn.Parameter(torch.randn(config.width) / math.sqrt(config.width))
        self.blocks = nn.ModuleList(
            [CoAttentionLayer(config, shared=True) for _ in range(config.layers)]
        )
        self.fusion = nn.Sequential(
            nn.Linear(3 * config.width, config.width),
            nn.GELU(),
            nn.Linear(config.width, config.width),
        )
        self.pool = QueryPool(config.width)
        layer = nn.TransformerEncoderLayer(
            config.width,
            config.heads,
            config.ff_size,
            config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.capture_transformer = nn.TransformerEncoder(
            layer, config.layers, enable_nested_tensor=False
        )
        self.capture_pool = QueryPool(config.width)

    @staticmethod
    def root_delta(x: Tensor, observed: Tensor) -> Tensor:
        roots = x[..., :3]
        result = torch.cat((torch.zeros_like(roots[:, :1]), roots[:, 1:] - roots[:, :-1]), 1)
        valid = observed[..., :3].all(-1)
        adjacent = torch.cat((torch.zeros_like(valid[:, :1]), valid[:, 1:] & valid[:, :-1]), 1)
        return result.masked_fill(~adjacent[..., None], 0)

    def pair(
        self,
        a: Tensor,
        b: Tensor,
        observed_a: Tensor,
        observed_b: Tensor,
        delta_a: Tensor,
        delta_b: Tensor,
    ) -> Tensor:
        mask_a, mask_b = observed_a.any(-1), observed_b.any(-1)
        joint = observed_a[..., :3].all(-1) & observed_b[..., :3].all(-1)
        aa, bb, ab = pair_features(
            a, b, a[..., :3], b[..., :3], delta_a, delta_b, joint, self.config.epsilon
        )
        _, _, ba = pair_features(
            b, a, b[..., :3], a[..., :3], delta_b, delta_a, joint, self.config.epsilon
        )
        times = absolute_sincos(torch.arange(a.shape[1], device=a.device), self.config.width)
        ra = (self.relation_norm(self.relation_projection(ab)) + self.relation_type).masked_fill(
            ~joint[..., None], 0
        )
        rb = (self.relation_norm(self.relation_projection(ba)) + self.relation_type).masked_fill(
            ~joint[..., None], 0
        )
        a = (self.actor_norm(self.actor_projection(aa)) + times + ra).masked_fill(
            ~mask_a[..., None], 0
        )
        b = (self.actor_norm(self.actor_projection(bb)) + times + rb).masked_fill(
            ~mask_b[..., None], 0
        )
        for block in self.blocks:
            a, b = block(a, b, mask_a, mask_b)
        symmetric = torch.cat((a + b, (a - b).abs(), a * b), -1)
        return self.pool(self.fusion(symmetric), joint)

    def forward(self, capture: TMRGroupCapture) -> Tensor:
        device = self.actor_projection.weight.device
        windows = []
        for window in capture.windows:
            x, observed = window.canonical(device)
            deltas = self.root_delta(x, observed)
            total = torch.zeros(self.config.width, device=device, dtype=torch.float64)
            count = 0
            for i in range(len(x)):
                for j in range(i + 1, len(x)):

                    def encode(witness, i=i, j=j, x=x, observed=observed, deltas=deltas):
                        return self.pair(
                            x[i : i + 1],
                            x[j : j + 1],
                            observed[i : i + 1],
                            observed[j : j + 1],
                            deltas[i : i + 1],
                            deltas[j : j + 1],
                        )[0]

                    value = (
                        checkpoint(
                            encode,
                            self.actor_projection.weight,
                            use_reentrant=False,
                            preserve_rng_state=True,
                        )
                        if self.checkpoint_edges and torch.is_grad_enabled()
                        else encode(self.actor_projection.weight)
                    )
                    total = total + value.to(torch.float64)
                    count += 1
            windows.append((total / count).float())
        times = torch.tensor([window.start_seconds for window in capture.windows], device=device)
        sequence = torch.stack(windows)[None] + absolute_sincos(times, self.config.width)[None]
        mask = torch.ones(1, len(windows), dtype=torch.bool, device=device)
        encoded = self.capture_transformer(sequence, src_key_padding_mask=~mask)
        return self.capture_pool(encoded, mask)[0]


class MIMERetrieval(nn.Module):
    """Motion + *trainable* CLIP language + learned positive logit scale.

    Language module owns verified local tokenizer/tower and accepts text tokens,
    not frozen sentence embeddings. Host owns curriculum/splits/optimizer.
    """

    def __init__(self, motion: nn.Module, language: nn.Module, *, initial_scale: float = 1 / 0.07):
        super().__init__()
        self.motion = motion
        self.language = language
        if not math.isfinite(initial_scale) or initial_scale <= 0:
            raise ValueError("positive logit-scale initialization needed")
        self.logit_scale = nn.Parameter(torch.tensor(math.log(initial_scale)))

    def score(self, captures: tuple[TMRGroupCapture, ...], tokens) -> Tensor:
        motion = torch.stack([self.motion(capture) for capture in captures])
        text = self.language(tokens)
        return self.logit_scale.exp() * (F.normalize(motion, dim=1) @ F.normalize(text, dim=1).T)

    def compute_loss(
        self, captures: tuple[TMRGroupCapture, ...], tokens, positive: Tensor
    ) -> Tensor:
        scores = self.score(captures, tokens)
        # MIME original averages the two directions (unlike WaMo's sum).
        return positive_contrastive(scores, positive.to(scores.device)) / 2

    def score_dyads(self, motion: Tensor, roots: Tensor, mask: Tensor, tokens) -> Tensor:
        if not isinstance(self.motion, OriginalMIMEMotion):
            raise ValueError(
                "original dyadic scoring requires original role-specific motion encoder"
            )
        if any(count != 1 for count in tokens.segment_counts):
            raise ValueError(
                "TEXT_CONTEXT_LIMIT: original MIME does not use group text segmentation"
            )
        encoded = self.motion(motion, roots, mask)
        text = self.language(tokens)
        return self.logit_scale.exp() * (F.normalize(encoded, dim=1) @ F.normalize(text, dim=1).T)

    def compute_loss_dyads(
        self, motion: Tensor, roots: Tensor, mask: Tensor, tokens, positive: Tensor
    ) -> Tensor:
        scores = self.score_dyads(motion, roots, mask, tokens)
        return positive_contrastive(scores, positive.to(scores.device)) / 2
