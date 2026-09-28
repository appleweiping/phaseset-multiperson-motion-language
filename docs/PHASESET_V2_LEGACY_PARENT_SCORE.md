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

The complete-parent training host admits this scorer only when the source,
scorer and host bindings name the same typed scalar-floor receipt, its training
population matches the host, and the run ID/seed/system/B2 predecessor is one
of the six old/A9 rows in the V2 matrix. The current host candidate also pins
main training/validation components, 20 epochs and its optimizer/CF defaults;
these defaults are not a completed pilot freeze. A short synthetic run cannot
be labeled as a formal row. The base and V2 host paths reject a
legacy receipt. The host still does not establish data rights,
provenance or run budget by itself; the private operator must verify those
inputs and the real server qualification before formal training. No final-test
interface is added.

`build_v2_legacy_control` constructs only registered V2-019/020/021 (old)
and V2-049/050/051 (A9) rows from the exact frozen matrix. It reuses the
same validation-selected B2 checkpoint/terminal verifier as the V2 residual
factory, requires the same-seed B2 predecessor and main population, and
checks that the typed old scalar floor names the admitted training source.
`make_v2_legacy_parent_host` then binds that build, floor, predecessor and
source to the complete-parent host. The build explicitly carries
`launch_authority=False`: construction and synthetic tests cannot admit a
native optimizer step, decide the pilot schedule, or turn an unverified floor
receipt into real-data evidence.

Data-free tests check both modes' score decomposition, frozen B2 gradients,
floor mutation rejection, unbound-host rejection, guarded score/VJP replay and
short-run mislabeling rejection, and six registered factory rows on synthetic
inputs. Passing those checks is
software qualification, not a retrieval result or evidence for a paper claim.
