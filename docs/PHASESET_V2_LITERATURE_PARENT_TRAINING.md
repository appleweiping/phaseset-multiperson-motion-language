# Full-negative WaMo and MIME parent backward

`literature_parent_training.py` preserves each model's complete batch objective
while loading and replaying one entire prepared capture graph at a time.
This is a numerical seam, not the production optimizer/supervisor, a study
freeze, a converged baseline, or an authorization to open final-test data.
TMR is deliberately absent from this callable: its VAE samples, global filtering
and track-consistent set reconstruction cannot be replaced by either of these
objectives. Its separate faithful path is described in
[TMR parent backward](PHASESET_V2_TMR_PARENT_TRAINING.md).

For WaMo, every capture's complete motion representation, both reconstruction
branches and both original/shuffled ordering branches are cached without an
activation graph. All exact human rows receive admitted frozen DistilBERT CLS
features once. The original summed bidirectional variable-positive InfoNCE and
equal-parent auxiliary means are differentiated over the entire B-by-Q gallery.
Its VJPs replay every complete capture, including original auxiliary branches.

For MIME, all complete motion rows are cached, then all exact human captions
are tokenized once and encoded by the genuinely trainable CLIP tower. The
original averaged bidirectional known-positive objective differentiates text,
learned logit scale and motion leaves jointly. Motion VJPs are replayed one
parent at a time. No text gradients are frozen and no microbatch negatives
are dropped. Role-aware original dyadic Inter-X inputs are a separate protocol.

Only explicit learning components are admitted; historical development C00 may
be a learning component in its authorized outer fold, but final-test rows may
never enter. Loaders must return the same admitted complete physical source
on cache and replay. Exact representation/auxiliary equality is checked before
backward. The private host still binds official data/asset identity, augmentation,
component census, runtime, budget and real human provenance; callbacks do not
invent those authorities. It must not retain the entire physical population.

RNG snapshots include each physical load and stochastic encoding; replay restores
the next update's post-text-forward state. Shared-parameter gradient accumulation
order can differ from dense autograd, so independent FP32 gradient oracles use
frozen tolerances while loss and cached representation equality remain exact.
FP32 only; hardware/BF16/resource qualification and full optimizer/checkpoint/
resume/native learning remain separate. Software fixtures are explicitly analytic,
including rejected timeline gaps, and cannot populate any paper result table.
