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

Tests use analytic text/motion-free fixtures; no resulting number is baseline
performance or human verification. Original dyadic MIME and the shared group
adapter, original auxiliary/input fidelity checks, actual profile, convergence,
fair pilot selection and the registered formal matrix remain required.
