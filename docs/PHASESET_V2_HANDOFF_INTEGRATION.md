# 2026-09-26 handoff integration

The user-supplied Markdown handoff resolves the missing component definitions.
It is preserved privately; no ZIP, licensed data, body model or download grant
is asserted to have been supplied with it.

The 87-stage JSON now encodes the exact C01/C02 pilot learning, C03 selection,
ten out-of-fold components in three folds, and sealed C09/C11/C15 exclusion.
Handoff fold indices 0/1/2 map explicitly to existing run split labels 1/2/3.
The total remains 87 planned stages, not 87 completed experiments. Strong
comparator selection and exact step limits still require the real pilot.

The bounded implementation changes are:

- Mean/difference DCT uses `(a+b)/2` and `a-b`, not half the difference.
- Two masked recurrent layers carry both track states across temporal chunks.
- Shared-cosine fusion starts at alpha 0.1 and accepts an explicit support mask
  to fall back to the global branch for unsupported motion rows.
- Pilot allocation includes B2, each pilot is capped at 20% of its formal
  optimizer steps, and mechanism head capacity tolerance is 5% without dummy
  parameters. Any permitted 60/40 epoch expansion precedes formal freeze and
  remains subject to the 300 GPU-hour cap.

## Observed software verification

Fresh GPT-5.6-Sol review found no blocking issue; semantic review remains
provisional. Server-only execution of the V2 coordination, Morlet and legacy
PhasePair signal suites: **65 passed, 1 CUDA-only skip in 9.89 s**. The separate
512D, three-actor CUDA forward/backward witness: **1 passed in 5.17 s**, RTX
A6000, peak allocated **180,931,072 bytes**, reserved **205,520,896 bytes**.
These are analytic mechanism checks, not motion-language retrieval results.

No real native-data training, human relational challenge, sealed-test scores,
paper aggregate or scientific acceptance is implied. Whole-capture actor/edge
state orchestration, long-capture halo convolution, relation-packet integration
with the host runner, and legitimate literature adaptations remain to finish.
The prior exact-commit CI and anonymous-codeload receipts retain their original
scope and are not relabeled as verification of this new revision.

## Asset discovery correction

The current [official Embody page](https://www.meta.com/emerging-tech/codec-avatars/embody-3d/)
exposes an organization-email field and XRCIA acceptance/download form.
Do not insist on an approval email as the only possible access evidence.
The [official repository](https://github.com/facebookresearch/embody-3d#download-data)
documents the resulting download links. Neither this discovery nor the
repository's code license establishes that restricted captures may be used or
deployed anywhere; the actual data license and grant must be respected.
The [SMPL-X model instructions](https://github.com/vchoutas/smplx#downloading-the-model)
require registration and the model license. A Python package or pytest fixture
with a model-like filename is not the licensed body-model asset.
