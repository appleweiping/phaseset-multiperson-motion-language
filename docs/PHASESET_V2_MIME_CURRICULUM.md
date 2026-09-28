# MIME-Set curriculum component

This is one paper-defined sampler component, **not a qualified MIME baseline**.
It does not implement original role-aware/co-attention motion encoding, original
fine-tuned CLIP, original Inter-X evaluation, training or scientific task seal.

The [MIME v1 paper](https://arxiv.org/html/2607.22702v1) specifies uniform warmup,
training-anchor cosine neighbors, no-replacement batch coverage and a cosine
hardness ramp. This component uses three uniform epochs and a ten-epoch ramp
to 0.25. The zero-based convention is explicit: epochs 0..2 are uniform, epoch
3 starts at zero, epoch 12 reaches maximum. For descending-cosine eligible
neighbors, the window center is `floor((1-hardness)*(N_i-1))`.

Adaptations to the complete-parent group task are explicit: one anchor is the
normalized mean of **all** admitted frozen CLIP human sentence embeddings of
that parent, not a chosen caption or validation/held-out neighbor. This is not
original MIME's fine-tuned language tower. The private bounded pilot freezes
the disclosed neighbor-window width, which the paper does not specify, and
the same effective-parent batch policy as the other systems. A singleton tail
is rebalanced without dropping/repeating parents; all human rows remain
retrieval positives. Historical C00 can enter an outer fold's learning set,
but final-test parents cannot. The host supplies exact learning components.

The whole-parent anchor function now accepts the authenticated row selection
returned by `ParentHumanClipRows`, instead of requiring the complete original
development preparation batch to be re-encoded for every training subset.
It still checks exactly all learning-human commitments and original caption
digests: validation, held-out or appended weak columns cannot become anchors.
This changes the group sampling seam only, not original dyadic MIME text
admission or the legacy capture/storage receipt interfaces.

`MIMEParentTrainingHost` connects this sampler to real method-specific updates,
validation and exact resume. Anchor identity and explicit neighbor width are
resume-bound. For a pilot population smaller than the batch ceiling, the
curriculum cannot change the all-parent negative set; it can change order only.
Report that limitation rather than claiming a curriculum benefit from pilot
scores. The main population can form multiple no-replacement batches.

The scoped registered-server CPU qualification executed ten cases with no
skips: three actual host/row-selection cases plus seven existing sampler
cases. It checks independent all-human anchor means, held-out/changed-text
rejection, post-construction caller tensor mutation isolation, actual epoch
hooks and optimizer updates, and bitwise model/optimizer/scheduler/RNG resume.
Neighbor-width and anchor drift are refused before training data/text loads.
Production budget history is unchanged. This is analytic software evidence,
not native pilot learning, curriculum benefit or retrieval performance.

Tests use analytic text/motion-free fixtures; no resulting number is baseline
performance or human verification. Original dyadic MIME and the shared group
adapter, original auxiliary/input fidelity checks, actual profile, convergence,
fair pilot selection and the registered formal matrix remain required.
