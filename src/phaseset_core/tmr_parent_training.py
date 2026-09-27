"""Full-negative whole-parent TMR VAE backward with faithful stochastic order.

Text sampling, every motion distribution, batch motion sampling, and then
every parent's text/motion set reconstruction retain TMRSet.compute_loss's
order. All KL, latent, reconstruction, and filtered contrastive terms remain.
This is a numerical seam, not an optimizer, asset admission or experiment seal.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch.nn import functional as F

from .continuous_capture import PreparedContinuousCapture
from .literature_parent_training import LiteratureParentBackward, _group, _validate_learning
from .parent_retrieval_task import ParentRetrievalBatch
from .tmr_components import diagonal_kl, filtered_contrastive_loss, sample_diagonal_gaussian
from .tmr_set import TMRSet, TMRTextBatch
from .training import _capture_rng, _restore_rng


def backward_loaded_tmr_parent_batch(
    system: TMRSet,
    labels: ParentRetrievalBatch,
    *,
    learning_components: tuple[str, ...],
    load_capture: Callable[[str], PreparedContinuousCapture],
    load_text: Callable[[tuple[str, ...]], TMRTextBatch],
    progress: Callable[[str, int], None] | None = None,
) -> LiteratureParentBackward:
    """Retain the complete TMR-Set loss and every full-batch negative.

    The private host supplies exact frozen DistilBERT tokens and MPNet filter
    features, not learned/generated positives. Only distributions, sampled
    latents and scalar reconstruction values are cached. Decoder replay solves
    the whole-parent slot/track assignment exactly as the dense model does.
    """
    if not isinstance(system, TMRSet):
        raise TypeError("need the actual complete-parent TMR set VAE")
    _validate_learning(labels, learning_components)
    if not torch.is_grad_enabled() or any(p.dtype != torch.float32 for p in system.parameters()):
        raise ValueError("TMR replay requires grad-enabled FP32; BF16 is not qualified here")
    text = load_text(labels.captions)
    if type(text) is not TMRTextBatch or len(text.tokens) != len(labels.captions):
        raise ValueError("all admitted exact human rows need frozen TMR language features")
    device = system.text_encoder.tokens.device
    positive = labels.positive_mask(device=device)
    was_training, after = system.training, None
    system.train()
    system.zero_grad(set_to_none=True)
    encoder_states, decoder_states, cached, reconstructions = [], [], [], []

    def announce(phase, index):
        if progress is not None:
            progress(phase, index)

    def reconstruct(capture, text_latents, motion_latent):
        # One full-track Hungarian assignment for each reconstruction over all
        # windows; never independently match slots in individual windows.
        return torch.stack(
            [system.reconstruct_capture(latent, capture) for latent in text_latents]
        ).mean() + system.reconstruct_capture(motion_latent, capture)

    try:
        text_dist = system.text_distributions(text)
        text_z = sample_diagonal_gaussian(text_dist)
        for index, source in enumerate(labels.motion_source_keys):
            announce("encoder-cache", index)
            encoder_states.append(_capture_rng())
            capture = _group(load_capture, source)
            with torch.no_grad():
                distribution = tuple(value.detach() for value in system.encode_capture(capture))
            if any(not bool(torch.isfinite(value).all()) for value in distribution):
                raise ValueError("complete-parent TMR distribution is nonfinite")
            cached.append(distribution)
            del capture
        motion_dist = tuple(
            torch.stack([row[index] for row in cached]).detach().requires_grad_(True)
            for index in (0, 1)
        )
        # A single B-by-D normal draw, after all encoders. Drawing separately
        # inside a per-parent loop changes the dense model's stochastic order.
        motion_z = sample_diagonal_gaussian(motion_dist)
        kls, latents, caption_indices = [], [], []
        for row, source in enumerate(labels.motion_source_keys):
            announce("decoder-cache", row)
            decoder_states.append(_capture_rng())
            capture = _group(load_capture, source)
            indices = positive[row].nonzero().flatten()
            caption_indices.append(indices)
            with torch.no_grad():
                reconstructed = reconstruct(capture, text_z[indices], motion_z[row])
            if not bool(torch.isfinite(reconstructed)):
                raise ValueError("complete-parent set reconstruction is nonfinite")
            reconstructions.append(reconstructed.detach())
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
            del capture
        losses = dict(
            recons=torch.stack(reconstructions).mean(),
            kl=torch.stack(kls).mean(),
            latent=torch.stack(latents).mean(),
            contrastive=filtered_contrastive_loss(
                motion_z, text_z, positive, sentence_embeddings=text.sentences.to(device)
            ),
        )
        losses["loss"] = system.loss_weights.total(losses)
        # Return raw full scores; semantic filtering changes loss denominators
        # only, and must never invent positive labels or truncate the gallery.
        scores = F.normalize(motion_z, dim=1) @ F.normalize(text_z, dim=1).T / 0.1
        if any(not bool(torch.isfinite(value).all()) for value in (*losses.values(), scores)):
            raise ValueError("full TMR gallery/objective is nonfinite")
        after = _capture_rng()

        text_reconstruction_vjp = torch.zeros_like(text_z)
        motion_reconstruction_vjp = torch.zeros_like(motion_z)
        for row, (source, state, indices) in enumerate(
            zip(labels.motion_source_keys, decoder_states, caption_indices, strict=True)
        ):
            announce("decoder-replay", row)
            _restore_rng(state)
            capture = _group(load_capture, source)
            tz = text_z[indices].detach().requires_grad_(True)
            mz = motion_z[row].detach().requires_grad_(True)
            reconstructed = reconstruct(capture, tz, mz)
            if not torch.equal(reconstructed.detach(), reconstructions[row]):
                raise RuntimeError("complete-parent TMR reconstruction replay changed a value")
            # Identical derivative of the original equal-parent mean and its
            # registered reconstruction coefficient; no auxiliary term omitted.
            (reconstructed * (system.loss_weights.recons / len(labels.parents))).backward()
            if tz.grad is None or mz.grad is None:
                raise RuntimeError("set reconstruction did not differentiate both sampled latents")
            text_reconstruction_vjp[indices] += tz.grad
            motion_reconstruction_vjp[row] = mz.grad
            del reconstructed, capture, tz, mz

        _restore_rng(after)
        # The detached reconstruction scalar reports the true dense loss.
        # Its missing paths are supplied by both explicit sampled-latent VJPs;
        # decoder parameter gradients were accumulated in the replay above.
        torch.autograd.backward(
            (losses["loss"], text_z, motion_z),
            (None, text_reconstruction_vjp, motion_reconstruction_vjp),
        )
        if any(value.grad is None for value in motion_dist):
            raise RuntimeError("full TMR objective did not differentiate both motion distributions")
        for row, (source, state) in enumerate(
            zip(labels.motion_source_keys, encoder_states, strict=True)
        ):
            announce("encoder-replay", row)
            _restore_rng(state)
            capture = _group(load_capture, source)
            outputs = system.encode_capture(capture)
            if any(
                not torch.equal(actual.detach(), expected)
                for actual, expected in zip(outputs, cached[row], strict=True)
            ):
                raise RuntimeError("complete-parent TMR encoder replay changed a value")
            torch.autograd.backward(outputs, tuple(value.grad[row] for value in motion_dist))
            del outputs, capture
        for name, parameter in system.named_parameters():
            if parameter.requires_grad and (
                parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
            ):
                raise ValueError(f"missing or nonfinite TMR trainable gradient: {name}")
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
