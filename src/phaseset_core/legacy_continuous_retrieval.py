"""Whole-parent scorer for the unchanged old PhaseSet representation.

This is a score seam, not a qualified training host or a floor estimator. The
caller must bind an independently fitted old scalar floor to its private input
manifest before using this with native captures. No final-test data are opened.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .capture_validation import _validate_text_batch
from .continuous_capture import PreparedContinuousCapture
from .controls import PhaseSetSystem
from .frozen_clip_text import FrozenClipTextBatch
from .legacy_capture import encode_legacy_capture_tokens
from .objectives import PhaseSetRetrievalHead
from .periodic import DEFAULT_EDGE_CHUNK_SIZE, validate_edge_chunk_size
from .pipeline import collate_group_samples, edge_budget_batches
from .temporal_coordination import LegacyPhaseSetCalibratedHead
from .training import BaseRetrievalSystem


@dataclass(frozen=True)
class LegacyWholeCaptureScoreOutput:
    scores: Tensor
    global_cosine: Tensor
    legacy_relation_cosine: Tensor
    periodic_support: Tensor
    text_receipt_sha256: str
    capture_keys: tuple[str, ...]


class LegacyWholeCaptureRetrievalSystem(nn.Module):
    """Frozen B2 plus old six-token head, with either old or A9 scoring.

    Both modes use the same old 13D descriptor, six group tokens, old text-band
    projector and complete-capture masked mean. A9 alone replaces the old
    logit-plus-residual score with the common-scale cosine calibration.
    """

    def __init__(
        self,
        frozen_b2: BaseRetrievalSystem,
        legacy_encoder: PhaseSetSystem,
        *,
        score_mode: str,
        edge_chunk_size: int = DEFAULT_EDGE_CHUNK_SIZE,
        checkpoint_windows: bool = True,
        base_window_batch_size: int = 1,
        base_edge_budget: int = 32768,
    ) -> None:
        super().__init__()
        if (
            not isinstance(frozen_b2, BaseRetrievalSystem)
            or frozen_b2.system_id != "B2"
            or frozen_b2.embedding_dim != 512
        ):
            raise ValueError("old whole-capture controls require the frozen 512D B2")
        if (
            type(legacy_encoder) is not PhaseSetSystem
            or legacy_encoder.system_id != "08"
            or legacy_encoder.embedding_dim != 512
            or legacy_encoder.encoder is None
        ):
            raise ValueError("old whole-capture controls require exact system 08 at 512D")
        if score_mode not in ("old", "A9"):
            raise ValueError("score_mode must be old or A9")
        if type(checkpoint_windows) is not bool:
            raise TypeError("checkpoint_windows must be an exact bool")
        if type(base_window_batch_size) is not int or base_window_batch_size < 1:
            raise ValueError("base_window_batch_size must be positive")
        if type(base_edge_budget) is not int or base_edge_budget < 1:
            raise ValueError("base_edge_budget must be positive")
        self.edge_chunk_size = validate_edge_chunk_size(edge_chunk_size)
        self.checkpoint_windows = checkpoint_windows
        self.base_window_batch_size = base_window_batch_size
        self.base_edge_budget = base_edge_budget
        self.score_mode = score_mode
        self.frozen_b2 = frozen_b2.requires_grad_(False).eval()
        self.legacy_encoder = legacy_encoder
        self.head = (
            PhaseSetRetrievalHead(512)
            if score_mode == "old"
            else LegacyPhaseSetCalibratedHead(512)
        )
        self._floor_sha256 = self._current_floor_sha256()

    def _current_floor_sha256(self) -> str:
        floors = self.legacy_encoder.encoder._energy_floors
        if type(floors) is not np.ndarray or floors.dtype != np.float64 or floors.shape != (6,):
            raise ValueError("old scalar floor vector changed shape or type")
        return hashlib.sha256(floors.tobytes(order="C")).hexdigest()

    def get_extra_state(self) -> Tensor:
        payload = {
            "schema": "phaseset-v2-legacy-whole-capture-v1",
            "score_mode": self.score_mode,
            "old_scalar_floor_sha256": self._floor_sha256,
            "edge_chunk_size": self.edge_chunk_size,
            "checkpoint_windows": self.checkpoint_windows,
            "base_window_batch_size": self.base_window_batch_size,
            "base_edge_budget": self.base_edge_budget,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
        return torch.tensor(list(encoded), dtype=torch.uint8)

    def set_extra_state(self, state: object) -> None:
        if (
            type(state) is not Tensor
            or state.dtype != torch.uint8
            or state.ndim != 1
            or not torch.equal(state.detach().cpu(), self.get_extra_state())
            or self._current_floor_sha256() != self._floor_sha256
        ):
            raise ValueError("old mode, scalar floor or capture readout changed on resume")

    def train(self, mode: bool = True):
        super().train(mode)
        self.frozen_b2.eval()
        return self

    @torch.no_grad()
    def encode_global(self, capture: PreparedContinuousCapture) -> Tensor:
        self.frozen_b2.eval()
        windows = capture.windows()
        batches = edge_budget_batches(
            windows,
            max_total_edges=self.base_edge_budget,
            max_batch_size=self.base_window_batch_size,
        )
        rows = torch.cat(
            [self.frozen_b2.encode_trainable(collate_group_samples(batch)) for batch in batches]
        )
        if rows.shape != (len(windows), 512):
            raise RuntimeError("frozen B2 omitted a complete-capture window")
        return F.normalize(rows.to(torch.float64).mean(dim=0).to(torch.float32), dim=0)

    def score(
        self,
        captures: tuple[PreparedContinuousCapture, ...],
        text_batch: FrozenClipTextBatch,
    ) -> LegacyWholeCaptureScoreOutput:
        if (
            type(captures) is not tuple
            or not captures
            or any(type(capture) is not PreparedContinuousCapture for capture in captures)
        ):
            raise ValueError("old scorer requires complete prepared capture rows")
        if self._current_floor_sha256() != self._floor_sha256:
            raise ValueError("old scalar floor vector changed after construction")
        text, _, receipt_sha = _validate_text_batch(text_batch, allow_row_selection=True)
        device = next(self.legacy_encoder.parameters()).device
        if next(self.frozen_b2.parameters()).device != device:
            raise ValueError("frozen B2 and old encoder must share a device")
        text = text.to(device).contiguous()
        global_rows, tokens, masks = [], [], []
        for capture in captures:
            global_rows.append(self.encode_global(capture))
            old = encode_legacy_capture_tokens(
                self.legacy_encoder,
                capture,
                edge_chunk_size=self.edge_chunk_size,
                checkpoint_windows=self.checkpoint_windows,
            )
            tokens.append(old.tokens[0])
            masks.append(old.band_mask[0])
        global_embeddings = torch.stack(global_rows).contiguous()
        group_tokens = torch.stack(tokens).contiguous()
        band_mask = torch.stack(masks).contiguous()
        cosine = F.normalize(global_embeddings, dim=-1) @ F.normalize(text, dim=-1).T
        if self.score_mode == "old":
            base_logits = self.frozen_b2.scores(global_embeddings, text).detach().contiguous()
            output = self.head(group_tokens, band_mask, text, base_logits)
            relation_cosine = output.periodic_scores
        else:
            output = self.head(group_tokens, band_mask, text, cosine.detach().contiguous())
            relation_cosine = output.legacy_relation_cosine
        return LegacyWholeCaptureScoreOutput(
            output.scores.contiguous(),
            cosine.detach().contiguous(),
            relation_cosine,
            band_mask.any(dim=1).contiguous(),
            receipt_sha,
            tuple(capture.source_sha256 for capture in captures),
        )
