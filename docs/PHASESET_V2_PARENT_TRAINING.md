# Complete-parent numerical training seam

This module is the numerical seam used by the private V2 host, not a scientific
task seal, production attempt scheduler, dataset download or human-CF verifier.
It does not launch an optimizer, choose checkpoints, open final-test data or
declare that the 87-stage plan is launchable.

`parent_epoch_batches` visits every parent in the explicit learning components
once, uses one of seeds 1729/2718/31415, and includes every human caption row.
Its batch-size unit is **complete parents**, not released segments or windows.
The common batch/epoch/max-step settings are still to be frozen by the bounded
pilot. A singleton tail is rebalanced without dropping/repeating a parent;
an impossible ceiling fails explicitly. Historical main validation C00 may be
a learning component in the appropriate independent outer fold; final-test
parents may not. The host takes the precise component sets from the dated study
matrix, not from a sampler-selected score or fallback split.

`backward_parent_batch` binds complete physical source order, annotation-family
positives, all official human caption commitments and exact text bytes to the
frozen CLIP pool. CF sentences may follow the human rows without acquiring fake
retrieval positives. Explicit source/column/verified masks select their losses;
only real blind-human provenance admitted by the private host can establish a
true `verified_false` bit. Synthetic test bits are fixtures, not human evidence.

The full B-by-Q objective includes cross-microbatch negatives and variable
positives. A no-grad score pass plus a leaf-score VJP replays one complete
capture graph at a time, including the shared text adapter and calibration.
When extra CF columns follow the human prefix, that prefix is made contiguous
at the legacy loss boundary; the copy remains differentiable into all scores.
Within each capture, the existing ordered temporal/edge checkpointing remains.
This bounds live *multi-capture neural activation graphs*; it does not make
all-pair compute linear, eliminate physical-cache storage or bound all host RSS.
RNG snapshots preserve replay and the next update's stochastic state. B2 is
frozen, and gradients must be finite before the caller clips/steps/checkpoints.
This seam is FP32-qualified; hardware BF16 needs separate qualification and
integration. The complete private run host, registered budget accounting,
faithful TMR/WaMo/MIME, validation selection and science freeze remain required.
