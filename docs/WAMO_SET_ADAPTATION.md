# WaMo-Set: paper-defined complete-parent adaptation

This is an independently written implementation of the methods in
[WaMo v2](https://arxiv.org/html/2508.03343v2), not released author code or a
claim of original-benchmark reproduction. The audited
[official repository](https://github.com/3DAgentWorld/WaMo) at
`cef15e9c75e736e3351f052e39e0aee12b13ad0e` contains no implementation;
the latest read-only tree check also found only README and attributes.

The retained components are learnable, dilated stationary wavelet analysis
without temporal decimation; separate intra-frequency temporal encoders;
concatenation and inter-frequency temporal encoding; additive frame attention;
independent learnable inverse SWT plus intra-band reconstruction MLPs;
inter-feature reconstruction; and original/partially shuffled temporal-group
classification. Both reconstructions use SmoothL1; both ordering branches use
cross-entropy. Text is projected frozen DistilBERT CLS, not CLIP or an MPNet
sentence replacement. The caller supplies verified local frozen language.

Paper-specified settings: decomposition level 3, latent width 256, temperature
0.07, 16 temporal groups, shuffled fraction 0.25, Haar/db1 initialization
(Table 9), Adam learning rate 1e-4 and cosine schedule. The model implements the
paper's *sum* of directional contrastive losses, not a halved objective.
For variable numbers of human captions, known positives use logsumexp in the
numerator and all gallery candidates in the denominator. No semantic filtering
or pseudo-positive mining is introduced.

Unpublished details are disclosed project choices: circular SWT boundaries;
two-tap Haar analysis `[1,1]/sqrt(2)` and `[-1,1]/sqrt(2)`; independent synthesis
initialized to half those taps to account for undecimated redundancy; depthwise
coordinate convolutions of width 9 (low) and 3 (high); two Transformer layers,
four heads, FFN 1024, GELU and dropout 0.1; two-layer projection/decoder MLPs;
unit reconstruction and ordering weights. These are configurable choices,
not purported author defaults. Learnable filters are not constrained to remain
orthogonal or perfectly reconstructing. Initial reconstruction is tested.
The existing three WaMo short-pilot slots, not an additional search, govern
any permitted configuration selection before the common contract freezes.

Multi-person/whole-parent extensions:

- Reuse the admitted complete body22 capture contract; shared per-actor
  processing of *every* accepted 200-frame window, unordered actor PMA,
  followed by a capture temporal encoder and additive pooling.
- The capture hierarchy receives actual absolute window starts; omitted
  invalid windows leave their temporal gaps rather than being collapsed.
  This bounded-window wavelet hierarchy differs from a single-person transform
  across an entire continuous trajectory and must be reported as such.
- Actor commitments choose a deterministic reduction order only; no ordinal,
  actor-specific weights or commitment features enter the model. Dynamic K
  has no fixed slot bank. Reconstruction is actor-aligned autoencoding, not
  text-to-motion set reconstruction, so no Hungarian matching is appropriate.
- DMSP groups are defined within each accepted window, approximately evenly
  by `floor(frame_index * 16 / window_length)`. A uniform subset of
  `floor(0.25*T)` frames is permuted; positions outside the subset are untouched.
  The same permutation moves all actors, masks and original labels together.
  This concrete frame-subset realization and window-local objective are
  disclosed adaptations. Labels never enter the encoders; positional codes
  refer to current sequence positions, not shuffled origin positions.
- Unobserved coordinates remain zero and do not contribute reconstruction;
  entirely unobserved actor frames do not contribute either ordering loss.
  Window losses accumulate coordinate/frame sums, normalize over the complete
  parent's observed coordinates/actor frames, then average parents, so a longer
  parent or more captions do not multiply its auxiliary weight. Masks are
  still visible through the input and must be covered by missingness controls.
- The model is not a production trainer: provenance, split selection,
  optimizer/schedule, full effective negative gallery, budgets and immutable
  attempt admission remain the host's responsibility. Qualification alone
  does not establish learning, original-task reproduction or paper results.
