# Old PhaseSet and A9 complete-parent score seam

`LegacyWholeCaptureRetrievalSystem` scores every accepted ten-second window of
an already prepared capture, with no actor or edge sampling. Its legacy motion
branch remains the old system-08 Morlet/13D/six-band encoder; each band token is
the masked mean over all supporting windows. The frozen B2 branch uses the same
complete-parent mean and normalization as the V2 base readout.

Two score modes share this representation:

- `old`: frozen B2 temperature-scaled logits plus the original bounded
  `PhaseSetRetrievalHead` relation residual.
- `A9`: frozen B2 cosine and the same old relation cosine under the common
  learned cosine calibration. This changes score scale, not relation features
  or the old text-band projection architecture.

The mode, window batching, edge chunk, scalar-floor values and typed floor
receipt are bound to the scorer state. The receipt's method is the unchanged
old 20 Hz five-speed-channel Morlet actor-local power, with a linear fifth
percentile per band over every accepted actor-window in the declared training
parent census. Observed exact-zero powers count; absent Morlet support does
not. Each band's observed and missing counts must total the actor-window
census. The receipt binds the private source-manifest digest, population,
physical kernel identity, floor bytes and diagnostics. The data-free test uses
a synthetic receipt and does not imply that native floors were fitted.

The generic complete-parent training host deliberately does **not** accept
this scorer yet. The registered real training-only scalar-floor receipt,
old/A9 run identities, predecessor checkpoint binding and score-plus-VJP host
qualification are separate prerequisites before formal training or sealed
evaluation.

The data-free CPU qualification checks both modes' score decomposition, frozen
B2 gradients, floor mutation rejection, host rejection, and the existing old
capture/A9 regression suites. Passing those checks is software qualification,
not a retrieval result or evidence for a paper claim.
