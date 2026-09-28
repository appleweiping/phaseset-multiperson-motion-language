"""Train registered group bases on whole parents, not independent windows.

The input is the admitted prepared skeleton timeline, without an unused
Morlet cache or physical floor. This is the trainable version of V2's fixed
global-branch readout: every accepted window, fixed float64 mean, float32 L2.
It does not grant data rights, a formal start, or a qualified base checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .capture_validation import _validate_text_batch
from .continuous_capture import PreparedContinuousCapture
from .frozen_clip_text import FrozenClipTextBatch
from .pipeline import collate_group_samples, edge_budget_batches
from .training import BaseRetrievalSystem


@dataclass(frozen=True)
class ContinuousBaseScoreOutput:
    scores: Tensor
    motion_embeddings: Tensor
    text_receipt_sha256: str
    capture_keys: tuple[str, ...]


class ContinuousBaseRetrievalSystem(nn.Module):
    """Trainable B0/B1/B2, complete-parent pooling and base logit temperature.

    Shared-actor architectures are reused without shrinking their registered
    512D configuration. Window checkpointing recomputes neural activations in
    backward; it does not pretend the complete physical input occupies O(1)
    memory. The outer parent-score replay keeps only one parent graph alive.
    No text adapter, CF term, coordination branch or trainable window pool is
    added to a base. Window batching is an explicit, resume-bound setting.
    """

    def __init__(
        self,
        base: BaseRetrievalSystem,
        *,
        checkpoint_windows: bool = True,
        base_window_batch_size: int = 1,
        base_edge_budget: int = 32768,
    ):
        super().__init__()
        if (
            not isinstance(base, BaseRetrievalSystem)
            or base.system_id not in ("B0", "B1", "B2")
            or base.embedding_dim != 512
        ):
            raise ValueError("whole-parent bases require registered B0/B1/B2 with 512D output")
        if any(not parameter.requires_grad for parameter in base.parameters()):
            raise ValueError("a base-learning system cannot contain frozen base parameters")
        if type(checkpoint_windows) is not bool:
            raise ValueError("checkpoint_windows must be an explicit bool")
        if type(base_window_batch_size) is not int or base_window_batch_size < 1:
            raise ValueError("base_window_batch_size must be a positive integer")
        if type(base_edge_budget) is not int or base_edge_budget < 1:
            raise ValueError("base_edge_budget must be a positive integer")
        self.base = base
        self.checkpoint_windows = checkpoint_windows
        self.base_window_batch_size = base_window_batch_size
        self.base_edge_budget = base_edge_budget

    @property
    def system_id(self) -> str:
        return self.base.system_id

    def encode_capture(self, capture: PreparedContinuousCapture) -> Tensor:
        if type(capture) is not PreparedContinuousCapture:
            raise ValueError("base input must be an admitted complete prepared capture")
        windows = capture.windows()
        if not windows:
            raise ValueError("base capture needs at least one accepted window")
        batches = edge_budget_batches(
            windows,
            max_total_edges=self.base_edge_budget,
            max_batch_size=self.base_window_batch_size,
        )
        embeddings = []
        for window_batch in batches:
            # Bind each batch now: late-binding a loop variable would replay
            # the last window for every checkpoint during backward.
            def encode(_device_witness, window_batch=window_batch):
                return self.base.encode_trainable(collate_group_samples(window_batch))

            embeddings.append(
                # Checkpoint discovers CUDA RNG devices from explicit tensor
                # inputs, not parameters captured only inside a closure. The
                # scalar is a device witness, not a new encoder input/feature.
                checkpoint(
                    encode, self.base.logit_scale, use_reentrant=False, preserve_rng_state=True
                )
                if self.checkpoint_windows and torch.is_grad_enabled()
                else encode(self.base.logit_scale)
            )
        rows = torch.cat(embeddings)
        if rows.shape != (len(windows), 512):
            raise RuntimeError("whole-parent base encoding omitted or malformed a window")
        return F.normalize(rows.to(torch.float64).mean(dim=0).to(torch.float32), dim=0)

    def score(
        self,
        captures: tuple[PreparedContinuousCapture, ...],
        text_batch: FrozenClipTextBatch,
    ) -> ContinuousBaseScoreOutput:
        if (
            type(captures) is not tuple
            or not captures
            or any(type(capture) is not PreparedContinuousCapture for capture in captures)
        ):
            raise ValueError("base scoring requires complete prepared capture rows")
        text, _, receipt_sha = _validate_text_batch(text_batch, allow_row_selection=True)
        device = next(self.base.parameters()).device
        motion = torch.stack([self.encode_capture(capture) for capture in captures])
        scores = self.base.scores(motion.contiguous(), text.to(device).contiguous())
        return ContinuousBaseScoreOutput(
            scores, motion, receipt_sha, tuple(capture.source_sha256 for capture in captures)
        )
