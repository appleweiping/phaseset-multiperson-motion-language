"""Paper-defined WaMo, explicitly adapted to complete multi-person parents.

No official implementation was available at the audited upstream revision.
See docs/WAMO_SET_ADAPTATION.md for disclosed choices versus paper defaults.
Learnable SWT/ISWT, intra/inter frequency features, both reconstructions and
original/partially shuffled frame classification are all trainable here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .models import PMAPool
from .tmr_set import TMRGroupCapture, absolute_sincos


@dataclass(frozen=True)
class WaMoConfig:
    # Paper-specified: width, levels, temperature, temporal groups, fraction.
    width: int = 256
    levels: int = 3
    temperature: float = 0.07
    temporal_groups: int = 16
    shuffle_fraction: float = 0.25
    # Disclosed implementation choices: not purported author-code defaults.
    heads: int = 4
    layers: int = 2
    ff_size: int = 1024
    dropout: float = 0.1
    low_kernel: int = 9
    high_kernel: int = 3
    reconstruction_weight: float = 1.0
    ordering_weight: float = 1.0

    def __post_init__(self):
        integers = (
            self.width,
            self.levels,
            self.temporal_groups,
            self.heads,
            self.layers,
            self.ff_size,
            self.low_kernel,
            self.high_kernel,
        )
        if any(type(value) is not int or value < 1 for value in integers):
            raise ValueError("dimensions, layers and kernel sizes must be positive integers")
        if self.width % 2 or self.width % self.heads:
            raise ValueError("even width must be divisible by attention heads")
        if not (
            self.low_kernel > self.high_kernel and self.low_kernel % 2 and self.high_kernel % 2
        ):
            raise ValueError("odd low-frequency kernel must exceed high-frequency kernel")
        if not (0 <= self.dropout < 1 and 0 <= self.shuffle_fraction <= 1):
            raise ValueError("invalid dropout or shuffled frame fraction")
        for value in (self.temperature, self.reconstruction_weight, self.ordering_weight):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("temperature and objective weights must be finite positive")


class LearnableSWT(nn.Module):
    """Dilated undecimated per-coordinate analysis and independent synthesis.

    Circular boundaries; Haar initialization. Synthesis has the redundant SWT
    factor 1/2, so each initial analysis/synthesis stage reconstructs exactly.
    Neither filter orthogonality nor perfect reconstruction is forced after
    learning. Output order is d1,...,dS,aS. Never decimate or remove a level.
    """

    def __init__(self, levels: int = 3):
        super().__init__()
        if type(levels) is not int or levels < 1:
            raise ValueError("SWT needs a positive decomposition level")
        self.levels = levels
        scale = 1 / math.sqrt(2)
        self.analysis_low = nn.Parameter(torch.tensor([scale, scale]))
        self.analysis_high = nn.Parameter(torch.tensor([-scale, scale]))
        self.synthesis_low = nn.Parameter(torch.tensor([scale / 2, scale / 2]))
        self.synthesis_high = nn.Parameter(torch.tensor([-scale / 2, scale / 2]))

    @staticmethod
    def _filter(x: Tensor, taps: Tensor, dilation: int, direction: int) -> Tensor:
        return sum(
            tap * torch.roll(x, direction * dilation * index, dims=1)
            for index, tap in enumerate(taps)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        if x.ndim != 3 or x.shape[1] < 1:
            raise ValueError("SWT expects [actors,time,coordinates]")
        low, details = x, []
        for level in range(self.levels):
            details.append(self._filter(low, self.analysis_high, 2**level, -1))
            low = self._filter(low, self.analysis_low, 2**level, -1)
        return (*details, low)

    def inverse(self, bands: tuple[Tensor, ...]) -> Tensor:
        if len(bands) != self.levels + 1 or any(x.shape != bands[0].shape for x in bands):
            raise ValueError("ISWT requires all same-length details plus final approximation")
        low = bands[-1]
        for level in reversed(range(self.levels)):
            low = self._filter(low, self.synthesis_low, 2**level, 1) + self._filter(
                bands[level], self.synthesis_high, 2**level, 1
            )
        return low


def temporal_labels(length: int, groups: int, device: torch.device) -> Tensor:
    """Consecutive approximately equal groups; no motion-dependent targets."""
    return torch.arange(length, device=device, dtype=torch.int64) * groups // length


def partial_frame_permutation(length: int, fraction: float, device: torch.device) -> Tensor:
    """Shuffle a uniformly selected frame subset, not temporal blocks.

    One permutation is shared by all actors in a window. Origin labels and
    observation masks must follow it; labels are never encoder inputs.
    """
    if length < 1 or not 0 <= fraction <= 1:
        raise ValueError("invalid frame subset")
    order = torch.arange(length, device=device)
    count = math.floor(length * fraction)
    if count >= 2:
        selected = torch.randperm(length, device=device)[:count]
        order[selected] = selected[torch.randperm(count, device=device)]
    return order


class AdditivePool(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.hidden = nn.Linear(width, width)
        # A common score bias cancels in softmax, so do not add a dead parameter.
        self.score = nn.Linear(width, 1, bias=False)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        logits = self.score(torch.tanh(self.hidden(x))).squeeze(-1)
        weights = logits.masked_fill(~mask, -torch.inf).softmax(dim=1)
        return (x * weights[..., None]).sum(dim=1)


class TemporalFeatures(nn.Module):
    def __init__(self, config: WaMoConfig):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            config.width,
            config.heads,
            config.ff_size,
            config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.width = config.width

    def forward(self, x: Tensor, mask: Tensor, positions: Tensor) -> Tensor:
        encoded = self.transformer(
            x + absolute_sincos(positions, self.width)[None],
            src_key_padding_mask=~mask,
        )
        return encoded.masked_fill(~mask[..., None], 0)


def _mlp(source: int, target: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(source, hidden), nn.GELU(), nn.Linear(hidden, target))


def positive_contrastive(scores: Tensor, positive: Tensor) -> Tensor:
    """WaMo bidirectional SUM (not half), rectangular known-positive extension."""
    if positive.dtype != torch.bool or positive.shape != scores.shape:
        raise ValueError("positive mask must match all motion/text scores")
    if not bool(positive.any(0).all()) or not bool(positive.any(1).all()):
        raise ValueError("every motion/text must have a known positive")
    selected = scores.masked_fill(~positive, -torch.inf)
    return (torch.logsumexp(scores, 1) - torch.logsumexp(selected, 1)).mean() + (
        torch.logsumexp(scores, 0) - torch.logsumexp(selected, 0)
    ).mean()


class WaMoSet(nn.Module):
    """Shared actor/window WaMo with unordered actor PMA and whole-parent time.

    Reconstruction is an actor-aligned autoencoder (not a text-to-set decoder).
    Every accepted window contributes motion features and both auxiliary tasks.
    Temporal group labels are window-local under this disclosed hierarchy.
    Actual absolute starts, including gaps, reach only the capture hierarchy.
    """

    system_id = "WaMo-Set"

    def __init__(self, config: WaMoConfig = WaMoConfig(), *, checkpoint_windows: bool = True):
        super().__init__()
        self.config = config
        self.checkpoint_windows = checkpoint_windows
        self.wavelet = LearnableSWT(config.levels)
        kernels = [config.high_kernel] * config.levels + [config.low_kernel]
        self.convolutions = nn.ModuleList(
            [nn.Conv1d(66, 66, kernel, groups=66) for kernel in kernels]
        )
        self.band_projections = nn.ModuleList(
            [_mlp(66, config.width, config.width) for _ in kernels]
        )
        self.intra_transformers = nn.ModuleList([TemporalFeatures(config) for _ in kernels])
        self.fusion = _mlp((config.levels + 1) * config.width, config.width, config.width)
        self.inter_transformer = TemporalFeatures(config)
        self.frame_pool = AdditivePool(config.width)
        self.actor_pool = PMAPool(config.width, config.heads, config.ff_size, config.dropout)
        self.capture_transformer = TemporalFeatures(config)
        self.capture_pool = AdditivePool(config.width)
        self.intra_decoders = nn.ModuleList([_mlp(config.width, 66, config.width) for _ in kernels])
        self.inter_decoder = _mlp(config.width, 66, config.width)
        self.temporal_classifier = nn.Linear(config.width, config.temporal_groups)
        self.text_projection = nn.Linear(768, config.width)

    def frame_features(self, x: Tensor, observed: Tensor) -> tuple[tuple[Tensor, ...], Tensor]:
        mask = observed.any(-1)
        times = torch.arange(x.shape[1], device=x.device, dtype=torch.float32)
        bands = self.wavelet(x)

        def convolve(conv, band):
            # Modular extension works even for T < kernel/2; circular pad()
            # cannot wrap more than once and would reject admitted short inputs.
            half = conv.kernel_size[0] // 2
            index = torch.arange(-half, x.shape[1] + half, device=x.device) % x.shape[1]
            return conv(band[:, index].transpose(1, 2)).transpose(1, 2)

        intra = tuple(
            transformer(projection(convolve(conv, band)), mask, times)
            for conv, projection, transformer, band in zip(
                self.convolutions,
                self.band_projections,
                self.intra_transformers,
                bands,
                strict=True,
            )
        )
        inter = self.inter_transformer(self.fusion(torch.cat(intra, -1)), mask, times)
        return intra, inter

    def _window(
        self, x: Tensor, observed: Tensor, order: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor]:
        intra, inter = self.frame_features(x, observed)
        frame_mask = observed.any(-1)
        actors = self.frame_pool(inter, frame_mask)
        summary = self.actor_pool(
            actors[None], torch.ones(1, len(actors), device=x.device, dtype=torch.bool)
        )[0]
        if order is None:
            zero = x.new_zeros((), dtype=torch.float64)
            return summary, zero, zero
        reconstructed = self.wavelet.inverse(
            tuple(
                decoder(features)
                for decoder, features in zip(self.intra_decoders, intra, strict=True)
            )
        )
        inter_reconstructed = self.inter_decoder(inter)
        rec_sum = sum(
            F.smooth_l1_loss(prediction, x, reduction="none")
            .masked_fill(~observed, 0)
            .to(torch.float64)
            .sum()
            for prediction in (reconstructed, inter_reconstructed)
        )
        labels = temporal_labels(x.shape[1], self.config.temporal_groups, x.device)
        _, shuffled = self.frame_features(x[:, order], observed[:, order])
        ce_sum = sum(
            F.cross_entropy(
                self.temporal_classifier(features).transpose(1, 2),
                target[None].expand(len(x), -1),
                reduction="none",
            )
            .masked_fill(~mask, 0)
            .to(torch.float64)
            .sum()
            for features, target, mask in (
                (inter, labels, frame_mask),
                (shuffled, labels[order], frame_mask[:, order]),
            )
        )
        return summary, rec_sum, ce_sum

    def encode_capture(
        self, capture: TMRGroupCapture, *, auxiliary: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]:
        device = self.wavelet.analysis_low.device
        summaries, reconstruction, ordering = [], [], []
        observations, frames = 0, 0
        for window in capture.windows:
            x, observed = window.canonical(device)
            order = (
                partial_frame_permutation(x.shape[1], self.config.shuffle_fraction, device)
                if auxiliary
                else None
            )

            def encode(witness, x=x, observed=observed, order=order):
                # Explicit witness tells checkpoint which device's RNG to replay.
                return self._window(x, observed, order)

            summary, rec, dmsp = (
                checkpoint(
                    encode, self.wavelet.analysis_low, use_reentrant=False, preserve_rng_state=True
                )
                if self.checkpoint_windows and torch.is_grad_enabled()
                else encode(self.wavelet.analysis_low)
            )
            summaries.append(summary)
            reconstruction.append(rec)
            ordering.append(dmsp)
            observations += int(observed.sum().item())
            frames += int(observed.any(-1).sum().item())
        sequence = torch.stack(summaries)[None]
        times = torch.tensor([window.start_seconds for window in capture.windows], device=device)
        mask = torch.ones(1, len(summaries), device=device, dtype=torch.bool)
        full = self.capture_pool(self.capture_transformer(sequence, mask, times), mask)[0]
        return (
            full,
            (torch.stack(reconstruction).sum() / observations).to(torch.float32),
            (torch.stack(ordering).sum() / frames).to(torch.float32),
        )

    def encode_text(self, cls: Tensor) -> Tensor:
        if (
            cls.ndim != 2
            or cls.shape[1] != 768
            or len(cls) < 1
            or cls.dtype != torch.float32
            or cls.requires_grad
            or not bool(torch.isfinite(cls).all())
        ):
            raise ValueError("need frozen finite float32 DistilBERT CLS [texts,768]")
        return self.text_projection(cls.to(self.text_projection.weight.device))

    def score(self, captures: tuple[TMRGroupCapture, ...], cls: Tensor) -> Tensor:
        motion = torch.stack([self.encode_capture(capture)[0] for capture in captures])
        return (
            F.normalize(motion, dim=1)
            @ F.normalize(self.encode_text(cls), dim=1).T
            / self.config.temperature
        )

    def compute_loss(
        self, captures: tuple[TMRGroupCapture, ...], cls: Tensor, positive: Tensor
    ) -> dict[str, Tensor]:
        rows = [self.encode_capture(capture, auxiliary=True) for capture in captures]
        motion = torch.stack([row[0] for row in rows])
        text = self.encode_text(cls)
        scores = F.normalize(motion, dim=1) @ F.normalize(text, dim=1).T / self.config.temperature
        losses = dict(
            contrastive=positive_contrastive(scores, positive.to(scores.device)),
            reconstruction=torch.stack([row[1] for row in rows]).mean(),
            ordering=torch.stack([row[2] for row in rows]).mean(),
        )
        losses["loss"] = (
            losses["contrastive"]
            + self.config.reconstruction_weight * losses["reconstruction"]
            + self.config.ordering_weight * losses["ordering"]
        )
        return losses
