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
state host orchestration, disk-backed response streaming, relation-packet integration
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

## Capture-core follow-up, 2026-09-26

Shared two-layer actor temporal states now precede half-edge construction;
actor histories are not pooled or reset between local patches in one field.
Morlet convolution uses complete kernel halos around each central response
block (default 1024 frames), computing each response once. The resulting
complete response cache still has an explicit 1 GiB limit: this is bounded
temporary convolution, **not** a completed disk-streaming host.

The updated modules passed server CPU checks: **70 passed, 1 CUDA-only skip**
in 13.64 seconds, including chunk lengths 1/7/31/64 versus unchunked responses,
gap-mask equality, actor-state continuity, K=2 exact-zero regression, and the
legacy signal/Morlet tests. Before/after source snapshots are identical.
This is software qualification; formal V2 training remains 0/87.

## Continuous native capture follow-up, 2026-09-26

`prepare_continuous_capture` now preserves one group origin/yaw and the complete
absolute timeline across accepted ten-second intervals. The original 300-frame
missing-data decisions are unchanged; rejected windows remain zero/masked on
the timeline, rather than concatenating their neighbors. Anti-alias FIR chunks
include the full halo, and targets touching rejected FIR support are masked.
Original short-gap observation masks remain false. Complete windows are used;
any trailing source frames are explicitly recorded.

`continuous_directional_phase_field` consumes the unbatched capture seam,
preserving signed velocity and six-band support across window boundaries. The
legacy short-window `PreparedGroupBatch` contract has not been relaxed.
Actor, node, edge and group temporal encoders therefore see one whole field,
not separately initialized ten-second fields. Base-window views share the
same capture-wide coordinates and retain their absolute source starts.

Server-only checks passed **106 tests, 1 CUDA-only skip in 18.24 s**, including
FIR chunk lengths 1/7/31/300/1024, actor permutation bit equality, rejected
intervals, retained short gaps, shared yaw, absolute window offsets, cross-seam
Morlet support, the original preprocessing/pipeline and legacy signal suites.
Source snapshots before and after execution are identical.

A real native four-actor, 120-second training capture passed the complete
official-model conversion and **untrained** 512D temporal forward in **63.55 s**:
3600 source frames, `[4,2400,22,3]` at 20 Hz, 12 accepted windows, 119 local
patches and six complete-graph edges. There was no caption semantic reading,
trained retrieval score, optimization or test evaluation. Zero energy floors
were solely a numerical fixture, not production calibration.

This closes the continuous numeric seam and a native full-capture forward,
not the full-corpus training host, disk-backed cache/backward scheduling,
training-only floor fit, literature baselines or scientific experiments.
The response cache still enforces its 1 GiB RAM limit without sampling actors
or edges. Formal training remains **0/87**.

## Updated CUDA mechanism witness, 2026-09-26

The same continuous-capture source passed a fresh server-only 512D, three-actor
CUDA forward/backward check: **1 passed in 11.30 s**, NVIDIA RTX A6000, peak
allocated **206,455,808 bytes**, reserved **249,561,088 bytes**. Admission
required observed device use below 500 MiB; the allocator was bounded to 2 GiB.
Source snapshots match and the final attempt exit is zero. This supersedes no
earlier receipt: the earlier peaks above belong to their earlier source.

The witness uses an analytic input and no optimizer or retrieval scores. It
does not qualify full-corpus training, establish accuracy, or count as one of
the 87 formal stages. Actual development body-model conversion is a separate
private server job; final-test captures are excluded from that conversion.
