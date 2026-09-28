# Full-negative whole-parent TMR VAE backward

`tmr_parent_training.py` is a numerical seam, not a production experiment or
completed original-task replication. It uses the actual `TMRSet`, frozen
DistilBERT token features, separate MPNet negative-filter features, family-derived
positives, and every accepted window with its absolute chronology and gaps.

Stochastic order matches `TMRSet.compute_loss`: encode/sample all texts, encode
all parents, draw all motion latents together, then reconstruct each parent's
positive captions and motion. Reconstruction uses the full-parent Hungarian
assignment, never independent per-window identities. Both reconstructions, four
KL terms, sampled latent alignment, and filtered full-batch contrastive loss
retain their coefficients and equal-parent weighting.

Only Gaussian parameters, sampled latents and scalar reconstruction values are
cached. Decoder replay differentiates both sampled latent paths and decoder
parameters; the global objective then differentiates text and cached motion
distributions, followed by one-parent-at-a-time encoder replay. This bounds the
number of resident motion/decoder graphs, not the work within one complete
parent or the full text batch. It does not promise arbitrary-size memory use.

Tests compare the original dense VAE loss, every parameter gradient, RNG, full
gallery, and physical drift rejection. Loss/RNG are exact in the frozen runtime;
shared FP32 gradient accumulation order may differ (rtol 3e-5, atol 3e-6).
These analytic oracles are not native optimizer, CUDA/BF16, convergence, pilot,
profile, production-host, or scientific-result evidence. Those gates remain
separate and require actual server receipts before formal training.
