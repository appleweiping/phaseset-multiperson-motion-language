# Signed-vector calibration and continuous physical cache

This extends the V2 continuous numeric seam. It is not the completed training
runner, literature-baseline reproduction, or a retrieval-results release.

## Calibration population and arithmetic

`actor_patch_power_observations` collects each actor/local-patch/band once,
without enumerating edges, reading captions, or consulting energy thresholds.
It uses the same response centers, scalar masks and complex squared-energy
units as `local_pair_chunk`. Marginal actor support is not replaced by the
smaller intersection with any particular partner.

The estimator remains the previously specified linear fifth percentile:
`h=(n-1)/20`, with exact integer rank and fractional interpolation. Observed
exact-zero power is included; missing kernel support is excluded and counted.
This policy is not tuned on scores. The V2 population is **signed-vector
actor/local-patch power**, not the legacy five-speed-channel window population;
legacy fitted values are not reusable here.

`fit_directional_energy_floors` requires the full expected capture census and
that invocation's training components. Repeated, unexpected or incomplete
captures and non-training components fail without a returned fit. Runtime
limits fail explicitly rather than sample observations. Final C09/C11/C15
are never calibration components. C00 can be training in appropriate external
folds; it is excluded from main calibration. C03 is excluded from pilot/fold
calibration but is part of the main training split. Every fit must use its own
declared training census, never reuse another fold's thresholds.

## Response storage and reuse

`write_continuous_phase_cache` writes a fresh **private** directory containing
complete complex128 actor/band responses and bool masks as NPY memory maps.
Convolution keeps the original full kernel halo and response arithmetic.
Centers, signed actor features, root features, masks and absolute intervals
are preserved. There is no actor, edge, time or missing-interval sampling.
Disk bytes/free space are admitted before the directory is created, existing
attempts are not overwritten, and `metadata.json` is written last.

The physical response cache does not fit or bake in thresholds. Its zero-floor
field is collection-only. `load_continuous_phase_cache` requires explicit
fitted floors and expected capture source lineage; thresholds are applied when
constructing local relations, without reconvolving motion. Main/pilot/fold
fits may reuse fixed physical responses but cannot share learned checkpoints
or calibration populations. The existing execution manifest remains
responsible for preprocessing/source admission and file-integrity checks.

The original RAM API and its 1 GiB response gate remain unchanged. The disk
backend avoids retaining the complete response arrays as heap allocations;
linear velocity/features, OS page cache and neural activations still consume
memory. It does **not** establish constant resident memory or finish long-
capture neural backward scheduling, augmentation, production CLI integration,
or full-corpus training. All real-data artifacts remain private.

Software and native qualification results are recorded only after server
execution. Synthetic correctness and a one-capture native fit do not amount
to full-training calibration, formal optimization or scientific acceptance.

## Observed server qualification, 2026-09-27

The calibration/cache tests together with continuous preprocessing, V2
coordination, pipeline and legacy signal/Morlet regression passed **133 tests,
1 CUDA-only skip in 22.17 seconds** on server CPU. Source before/after matched.
Checks include exact RAM/mapped-response parity, pair features, immutable
thresholds, unequal actor masks, calibration isolation and cached gradients.

The same previously qualified **real four-actor, 120-second training capture**
then passed disk-cache creation, one-capture calibration, read-only reload and
**untrained 512D forward/backward** in **12.42 seconds**, using one CPU thread.
The cache contains 64,038,532 bytes; measured cache write was 1.51 seconds,
calibration 0.15 seconds, forward 2.82 seconds and backward 5.67 seconds.
There are 2,400 frames, 119 patches, six edges and 49 finite gradient tensors,
with nonzero half-edge/topology gradients. These are single qualification
measurements, not p50/p95 benchmarks or evidence of retrieval improvement.

The one-capture thresholds are **not** the complete training/fold fit. No
captions, final-test input, optimizer steps or retrieval scores were used.
Formal training remains **0/87**; production host integration, complete fits,
faithful literature baselines, pilot and scientific evaluation are unfinished.

## Full-development storage and split routing

`continuous_capture_io.load_prepared_continuous_capture` restores a complete
body22 timeline from the preparation manifest's completed record and its
three private arrays. The record, array hashes, headers, canonical actor
lineage and complete window decisions are checked; mmap arrays stay read-only.
This does not replace dataset rights/split admission by the execution manifest.

`calibration_populations.calibration_populations` reads the V2 matrix's exact
user-supplied component assignments. It returns five distinct training routes:

- Main: the twelve registered training components; C00 remains validation.
- Pilot: C01/C02 train, C03 selects.
- Fold 0: development excluding C03 and held-out C00/C06/C10/C14.
- Fold 1: development excluding C03 and held-out C04/C07/C12.
- Fold 2: development excluding C03 and held-out C05/C08/C13.

Thus C00 is training data in folds 1/2, not in main, pilot or fold 0. C03 is
main training data but never pilot/fold training data. C09/C11/C15 cannot be
admitted. Distinct folds refit floors and models; no main-checkpoint reuse.
This helper does not change `PLANNED_NOT_LAUNCHABLE` or launch training.

The private development consumer admits only completed body records, checks
the frozen eligible census, creates whole-capture physical caches, and waits
for the producer's successful source-bound receipt before fitting any floors.
If the producer finishes while a cache snapshot is being processed, it must
re-read and drain all newly completed records before verifying the terminal
census. No partial sample can masquerade as complete main/fold calibration.
The full body/caption/model inputs, physical arrays and fitted population
files remain private; the public release contains APIs and data-free tests.

The extended server CPU qualification passed **146 tests, one CUDA-only skip
in 21.06 seconds**, including three private orchestration regressions. The
producer-completion race is tested with mocks explicitly marked as sequencing
tests, not as research data or results. The public storage/population tests
are data-free and included in the package's source release.

Three already prepared **real native development captures** separately passed
admitted-record restoration, complete physical-cache creation, read-only
reload and artifact recording in **2.94 seconds** (94,609,294 cache bytes).
This is a bounded input-preparation qualification, not a 512D model profile,
training-population fit or retrieval result. No captions, final-test input,
optimizer steps or fitted floors were used. The complete 494-capture consumer
has been launched; it is not yet declared complete, and formal training is 0/87.

The disk cache is an input-preparation stage, not a completed V2 training
host. Shared-yaw actor/root recomputation, long-trajectory backward scheduling,
CLIP/counterfactual integration, faithful baselines, actual profile/pilot and
the 87-stage formal matrix remain separate required work.
