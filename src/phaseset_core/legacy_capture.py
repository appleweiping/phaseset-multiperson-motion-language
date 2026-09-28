"""Use every accepted complete-capture window with the old six-token model.

This is the motion-representation seam shared by the registered old-head and
A9 score-calibrated controls. It is not a training host or data admission.
The legacy PhaseSet encoder still sees each original 200-frame window; only
the capture-level pooling is new. No rejected interval is silently scored.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .continuous_capture import PreparedContinuousCapture
from .controls import PhaseSetSystem
from .models import _fixed_binary_tree_sum
from .periodic import BAND_COUNT, DEFAULT_EDGE_CHUNK_SIZE
from .pipeline import collate_group_samples


@dataclass(frozen=True)
class LegacyCaptureTokenOutput:
    tokens: Tensor  # float32 [1,6,D]
    band_mask: Tensor  # bool [1,6]
    supporting_window_count: Tensor  # int64 [6]
    accepted_window_count: int
    timeline_window_count: int


def encode_legacy_capture_tokens(
    encoder: PhaseSetSystem,
    capture: PreparedContinuousCapture,
    *,
    edge_chunk_size: int = DEFAULT_EDGE_CHUNK_SIZE,
    checkpoint_windows: bool = True,
) -> LegacyCaptureTokenOutput:
    """Masked-mean the old token of every accepted 10-second window.

    The encoder is the exact old full system 08, not a V2 temporal packet
    encoder. It computes all unordered actor pairs for each accepted window.
    A band with no observed support has an exact positive-zero output, and
    the returned support count remains separate from the neural features.
    """
    if type(encoder) is not PhaseSetSystem or encoder.system_id != "08":
        raise TypeError("complete legacy capture requires exact PhaseSet system 08")
    if type(capture) is not PreparedContinuousCapture:
        raise TypeError("complete legacy capture requires prepared physical input")
    if type(edge_chunk_size) is not int or edge_chunk_size < 1:
        raise ValueError("edge_chunk_size must be a positive exact integer")
    if type(checkpoint_windows) is not bool:
        raise TypeError("checkpoint_windows must be an exact bool")
    windows = capture.windows()
    values: list[Tensor] = []
    counts = torch.zeros((BAND_COUNT,), dtype=torch.int64, device=encoder._device_anchor.device)
    anchor = torch.zeros((), dtype=torch.float32, device=encoder._device_anchor.device)
    for window in windows:
        batch = collate_group_samples((window,))
        if checkpoint_windows and torch.is_grad_enabled():
            # Bind this exact window in the closure: backward may recompute it
            # after the Python loop has advanced to a later capture interval.
            def encode(_anchor: Tensor, fixed_batch=batch) -> tuple[Tensor, Tensor]:
                result = encoder(fixed_batch, edge_chunk_size=edge_chunk_size)
                return result.tokens, result.band_mask

            tokens, band_mask = checkpoint(encode, anchor, use_reentrant=False)
        else:
            output = encoder(batch, edge_chunk_size=edge_chunk_size)
            tokens, band_mask = output.tokens, output.band_mask
        mask = band_mask[0]
        values.append(torch.where(
            mask[:, None], tokens[0].to(torch.float64),
            torch.zeros_like(tokens[0], dtype=torch.float64),
        ))
        counts = counts + mask.to(torch.int64)
    if not values:
        raise ValueError("capture has no accepted window")
    sums = _fixed_binary_tree_sum(values)
    support = counts > 0
    pooled = torch.where(
        support[:, None],
        sums / counts.clamp_min(1)[:, None],
        torch.zeros_like(sums),
    ).to(torch.float32)
    pooled = torch.where(support[:, None], pooled, torch.zeros_like(pooled))
    return LegacyCaptureTokenOutput(
        pooled[None].contiguous(),
        support[None].contiguous(),
        counts.contiguous(),
        len(windows),
        len(capture.decisions),
    )
