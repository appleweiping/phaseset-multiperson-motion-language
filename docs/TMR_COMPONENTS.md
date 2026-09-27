# TMR implementation components — not a trained TMR-Set result

The [component implementation](../src/phaseset_core/tmr_components.py) follows
the [official pinned TMR source](https://github.com/Mathux/TMR/tree/6d74688730d15d43b0a755ce2b0e1f2d76138fc1),
under its [MIT license](../LICENSE_TMR.md). Defaults are the actual
training YAML: latent 256, FFN 1024, six layers, four heads, dropout 0.1, GELU.
The complete objective keeps both SmoothL1 motion reconstructions, all four
element-mean KL divergences, sampled latent SmoothL1 alignment and filtered
contrastive loss. Weights are 1, 1e-5, 1e-5, 0.1; temperature is 0.1.

The encoder prepends two learned distribution tokens. The decoder attends to
one sampled latent using zero time queries plus sinusoidal positions. Retrieval
uses distribution means. The caller supplies frozen DistilBERT token features
(default dimension 768), and separate normalized MPNet sentence features for
negative filtering; global CLIP vectors are not substituted for either.

The original diagonal, one-caption objective is available through
`TMRSinglePerson`. This is an architecture/loss equivalence tool, not evidence
that the original dataset preprocessing or converged original-task replication
has been completed. It does not implement the group decoder, language asset
loader, complete-parent training host, sampler, or pilot admission.

Two explicit extensions are provided for later adaptation:

- `filtered_contrastive_loss` accepts a rectangular known-positive matrix. A
  motion–text negative is filtered when that text exceeds cosine **0.6** against
  any known caption of that motion (official threshold 0.8 maps to 2t−1).
  Known positives are always preserved. Similarity never creates a positive.
  With a diagonal positive matrix this reduces to the original filter and
  symmetric cross-entropy, up to frozen floating-point tolerance.
- Reconstruction defaults to the original mean over the padded target tensor.
  An explicit feature-level `observed` mask instead computes the mean only over
  observed values, preserving genuine capture gaps. This changes normalization
  and must be disclosed/frozen for the group adapter; gaps are not zero-valued
  reconstruction targets. No set reconstruction assignment is implied here.

Position capacity defaults to the upstream 5000. Longer sequences require an
explicit larger `max_len` using the same analytic encoding; there is no silent
truncation. Encoder capacity includes its two prefix tokens. Dense attention
over whole native parents has not been profiled or approved for training.

Training-only pool selection remains the host's responsibility. Semantic
filtering must not inspect held-out/test captions and must not be applied to the
study's common human-only full-gallery retrieval metrics.
