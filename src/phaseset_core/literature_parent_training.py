"""Full-negative, one-complete-parent-at-a-time WaMo/MIME backward.

No optimizer, task seal, data discovery or TMR approximation. TMR's VAE and
set reconstruction need a separate faithful seam; they are not routed here.
Private hosts still admit native sources, language assets and learning folds.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .continuous_capture import PreparedContinuousCapture
from .mime_language import TrainableMIMECLIP
from .mime_set import MIMERetrieval, MIMESetMotion
from .parent_retrieval_task import ParentRetrievalBatch
from .tmr_set import TMRGroupCapture
from .training import _capture_rng, _restore_rng
from .wamo_set import WaMoSet, positive_contrastive


@dataclass(frozen=True)
class LiteratureParentBackward:
    loss: float
    loss_components: dict[str, float]
    scores: Tensor
    parent_count: int
    text_count: int
    replayed_parents: int


def _validate_learning(labels: ParentRetrievalBatch, components: tuple[str, ...]) -> None:
    if (
        type(components) is not tuple
        or not components
        or len(set(components)) != len(components)
        or any(type(component) is not str or not component for component in components)
    ):
        raise ValueError("explicit unique learning components are required")
    if len(labels.parents) < 2 or any(
        parent.component not in components or parent.split == "test" for parent in labels.parents
    ):
        raise ValueError("need two or more admitted learning parents, never final test")
    if len(set(labels.motion_source_keys)) != len(labels.parents) or len(
        set(labels.motion_positive_keys)
    ) != len(labels.parents):
        raise ValueError("released siblings/source aliases cannot be independent parents")


def _group(load_capture, source: str) -> TMRGroupCapture:
    capture = load_capture(source)
    if type(capture) is not PreparedContinuousCapture or capture.source_sha256 != source:
        raise ValueError("loader must return the admitted complete prepared source")
    return TMRGroupCapture.from_prepared(capture)


def _backward(
    system: nn.Module,
    labels: ParentRetrievalBatch,
    load_capture: Callable[[str], PreparedContinuousCapture],
    encode: Callable[[TMRGroupCapture], tuple[Tensor, ...]],
    objective: Callable[[tuple[Tensor, ...]], tuple[dict[str, Tensor], Tensor]],
    progress: Callable[[str, int], None] | None,
) -> LiteratureParentBackward:
    if not torch.is_grad_enabled():
        raise ValueError("training backward cannot run under no_grad")
    if any(p.dtype != torch.float32 for p in system.parameters()):
        raise ValueError("this numerical seam is FP32 only; BF16 needs qualification")
    was_training, after = system.training, None
    system.train()
    system.zero_grad(set_to_none=True)
    states, cached = [], []
    try:
        for index, source in enumerate(labels.motion_source_keys):
            if progress is not None:
                progress("cache", index)
            states.append(_capture_rng())
            capture = _group(load_capture, source)
            with torch.no_grad():
                rows = tuple(value.detach() for value in encode(capture))
            if any(not bool(torch.isfinite(value).all()) for value in rows):
                raise ValueError("complete-parent representation/auxiliary cache is nonfinite")
            cached.append(rows)
            del capture
        leaves = tuple(
            torch.stack([rows[index] for rows in cached]).detach().requires_grad_(True)
            for index in range(len(cached[0]))
        )
        losses, scores = objective(leaves)
        if scores.shape != (len(labels.parents), len(labels.captions)) or any(
            not bool(torch.isfinite(value).all()) for value in (*losses.values(), scores)
        ):
            raise ValueError("full human gallery/objective is malformed or nonfinite")
        # Includes any stochastic trainable text encoding, just as the dense
        # objective. Backward checkpoints preserve the post-forward RNG state.
        after = _capture_rng()
        losses["loss"].backward()
        if any(value.grad is None for value in leaves):
            raise RuntimeError("full objective did not differentiate every cached branch")
        for index, (source, state) in enumerate(
            zip(labels.motion_source_keys, states, strict=True)
        ):
            if progress is not None:
                progress("replay", index)
            _restore_rng(state)
            capture = _group(load_capture, source)
            outputs = encode(capture)
            if len(outputs) != len(cached[index]) or any(
                not torch.equal(value.detach(), expected)
                for value, expected in zip(outputs, cached[index], strict=True)
            ):
                raise RuntimeError("complete-parent replay changed an encoded/auxiliary value")
            torch.autograd.backward(outputs, tuple(value.grad[index] for value in leaves))
            del outputs, capture
        for name, parameter in system.named_parameters():
            if parameter.requires_grad and (
                parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
            ):
                raise ValueError(f"missing or nonfinite trainable gradient: {name}")
        return LiteratureParentBackward(
            float(losses["loss"].detach()),
            {name: float(value.detach()) for name, value in losses.items()},
            scores.detach().cpu(),
            len(labels.parents),
            len(labels.captions),
            len(cached),
        )
    finally:
        if after is not None:
            _restore_rng(after)
        system.train(was_training)


def backward_loaded_wamo_parent_batch(
    system: WaMoSet,
    labels: ParentRetrievalBatch,
    *,
    learning_components: tuple[str, ...],
    load_capture: Callable[[str], PreparedContinuousCapture],
    load_cls: Callable[[tuple[str, ...]], Tensor],
    progress: Callable[[str, int], None] | None = None,
) -> LiteratureParentBackward:
    """Exact WaMo SUM InfoNCE + both reconstructions + both ordering branches.

    ``load_cls`` must use the admitted frozen DistilBERT assets and every
    provided exact human row. Asset identity remains the private host's input
    binding, not an authority invented by this callable interface.
    """
    if not isinstance(system, WaMoSet):
        raise TypeError("need the actual complete-parent WaMo model")
    _validate_learning(labels, learning_components)
    cls = load_cls(labels.captions)
    device = system.wavelet.analysis_low.device
    positive = labels.positive_mask(device=device)

    def objective(rows):
        motion, reconstruction, ordering = rows
        text = system.encode_text(cls)
        scores = F.normalize(motion, dim=1) @ F.normalize(text, dim=1).T / system.config.temperature
        losses = dict(
            contrastive=positive_contrastive(scores, positive),
            reconstruction=reconstruction.mean(),
            ordering=ordering.mean(),
        )
        losses["loss"] = (
            losses["contrastive"]
            + system.config.reconstruction_weight * losses["reconstruction"]
            + system.config.ordering_weight * losses["ordering"]
        )
        return losses, scores

    return _backward(
        system,
        labels,
        load_capture,
        lambda capture: system.encode_capture(capture, auxiliary=True),
        objective,
        progress,
    )


def backward_loaded_mime_parent_batch(
    system: MIMERetrieval,
    labels: ParentRetrievalBatch,
    *,
    learning_components: tuple[str, ...],
    load_capture: Callable[[str], PreparedContinuousCapture],
    progress: Callable[[str, int], None] | None = None,
) -> LiteratureParentBackward:
    """Exact MIME averaged InfoNCE, full batch negatives and trainable CLIP.

    The original role-aware dyadic Inter-X protocol is intentionally not
    admitted here. Curriculum supplies labels upstream, never replaces loss.
    All exact official rows are tokenized once; text gradients are not frozen.
    """
    if (
        not isinstance(system, MIMERetrieval)
        or not isinstance(system.motion, MIMESetMotion)
        or not isinstance(system.language, TrainableMIMECLIP)
    ):
        raise TypeError("need shared full-parent MIME plus genuinely trainable CLIP module")
    _validate_learning(labels, learning_components)
    tokens = system.language.tokenize(labels.captions)
    positive = labels.positive_mask(device=system.logit_scale.device)

    def objective(rows):
        text = system.language(tokens)
        scores = system.logit_scale.exp() * (
            F.normalize(rows[0], dim=1) @ F.normalize(text, dim=1).T
        )
        return {"loss": positive_contrastive(scores, positive) / 2}, scores

    return _backward(
        system, labels, load_capture, lambda capture: (system.motion(capture),), objective, progress
    )
