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
host. CLIP/counterfactual integration, faithful baselines, actual profile/pilot
and the 87-stage formal matrix remain separate required work.

## Whole-capture training input and shared yaw

`continuous_training_input.ContinuousTrainingInput.load` binds the body22 and
physical records through the consumer's `prepared_record`, verifies their
fixed artifact files, and restores one complete, immutable capture. Load once
per admitted capture rather than rehashing every response on every step.
Rights, split admission and the fitted calibration population still belong to
the frozen execution manifest. A positive key is `source_sha256`, never the
possibly reused actor-set/group commitment.

`view(yaw_delta=0, allow_shared_yaw=False)` reuses the read-only physical maps.
A nonzero view rotates **all actors and all frames together** about the same
existing group origin; it never recenters on a person or resets at a window.
Original masks, rejected intervals, absolute times and capture identity stay.
The transform explicitly operates on already prepared float32 body22; it does
not claim bitwise equivalence to rotating raw parameters before resampling.

Nonzero yaw recomputes complete signed velocity, Morlet responses, actor
signed-mean/per-channel-RMS and root statistics using the same frontend and
the unchanged fitted floors. It does **not** rotate the RMS vector, reuse stale
zero-yaw actor/root values, or assume floating-point phase invariance. The
RAM/edge resource gates apply to the whole capture; an oversized augmentation
fails explicitly, with no zero-yaw fallback, dropped actors or shortened time.
This correctness-first route reconvolves motion and its cost must be measured
in the actual host profile; augmentation is not advertised as free or mmap-only.

The caller must explicitly decide text yaw eligibility. Captions describing
world directions cannot silently receive incompatible motion augmentation;
this data-loading module neither reads nor rewrites any caption. A rejected
yaw request fails rather than silently proceeding without augmentation.
The frozen patch/hop/epsilon configuration cannot change between views.

Data-free tests cover exact masks and whole timeline, rotation about the
shared origin, correct RMS rather than the rotated-RMS shortcut, complete-field
parity with direct recomputation, full-trajectory gradients, cache drift and
resource/text-eligibility failures. These are implementation checks, not
retrieval improvements or formal optimizer runs.

The extended server CPU suite passed **154 tests, one CUDA-only skip in
26.21 seconds**, with unchanged source and invocation files. The same fixed
real four-actor, 120-second training capture then passed the admitted
zero-yaw cache view, full nonzero-yaw recomputation and two untrained **512D
forward/backward** evaluations in **20.49 seconds** total. Complete fields,
embeddings and all parameter gradients match direct augmented-field execution
bitwise in this frozen runtime; 49 gradient tensors are finite and the
half-edge/topology branches have nonzero gradients. Peak RSS was 1,180,908 KiB.

Measured admission was 0.49 seconds and yaw reconvolution 1.69 seconds. These
single-capture qualification timings do not establish end-to-end throughput,
p50/p95 latency or a GPU cost bound. Floors came from the prior fixed
one-capture QA fit, **not** the full main/pilot/fold population. No captions,
final-test input, optimizer or retrieval scores were used. The production
optimizer/CLIP/counterfactual/baseline host and 87 formal stages remain undone.

## Complete-capture sentence score and loss seam

`continuous_retrieval.ContinuousRetrievalSystem` reuses the fixed 512D B2
anchor and admitted `FrozenClipTextBatch` receipts. B2 stays frozen and in
evaluation mode even when the residual system trains. Every accepted window
is encoded in the same capture frame; its fixed mean supplies only the global
branch. No legacy base temperature is added to the bounded coordination cosine.

The new shared text adapter is 512 -> GELU -> 512. The coordination encoder
keeps its original complete ordered actor/node/pair/group histories. For
language, each **directed** packet retains its ordered half-edge and both
ordered endpoint node contexts; both orientations share update/temporal
weights. Complete trajectories are streamed in 64-edge blocks and recomputed
in backward. Tokens are matched to whole sentence vectors, not independently
maximized subject/action/object fields. No actor ordinal or lineage ID is a
neural feature.

The pre-pilot rule is fixed in the experiment matrix:
`r = 0.5 * ordered_capture_cosine + 0.5 * max_directed_packet_cosine`.
This bounded weighted local cosine is a scoring implementation, not a fourth
novelty claim. Group temporal parameters are effective, not dummy capacity.
Language maxima include only energy-observable patches; zero-energy track
support can retain history but cannot enable the periodic score. Low coherence
still does not remove an edge. A capture with no observable periodic evidence
has exact-zero coordination cosine and falls back to the global branch.

No all-pair packet tensor is retained. A block score matrix over text rows has
an explicit 512 MiB storage gate; callers must batch a larger gallery, never
drop actors, edges or time to fit it. Activation, physical-cache and complete
capture host memory are not constant-memory claims. Actual timing and cost
must be established in the later hardware profile.

The official holistic schema's **all `scene_explained` human sentence rows**
are used, without concatenating mood/atmosphere questionnaire fields or
machine-fusing annotations. Row count is variable. Source capture SHA identifies
the actual motion input, not its positive-target family. Released siblings can
share the same parent holistic file. `forward` requires separately admitted
motion/text positive-family keys and labels all siblings as positives; repeated
actor sets still do not create positives. Schema selection is not proof of motion observability;
official file provenance and human relation truth remain separate requirements.

`score` accepts arbitrary gallery/CF sentence batches without assigning any
positive labels; `forward` adds admitted target-family positives only for a valid
contrastive batch. A gallery chunk need not contain all matching motion rows,
and false-CF text is never forced to carry a fabricated positive capture key.

The loss interface requires explicit counterfactual scores and verified-false
masks. It cannot certify those human labels, and no false negative is invented
from generic captions. Empty explicit tensors cover implementation-only
contrastive checks or the no-CF control, not a completed full-CF training run.
The existing low-level atomic checkpoint and RNG helpers can be reused without
relaxing the old qualified-winner/execution guards. This new seam does not yet
claim a formal optimizer scheduler, full corpus calibration, faithful TMR/WaMo/
MIME adaptations, a trained B2 checkpoint, GPU profile/pilot, verified-CF
dataset, sealed evaluation or any of the 87 formal training stages.

The expanded server CPU regression passed **164 tests, one CUDA-only skip in
46.50 seconds**, with unchanged source and invocation files. This includes the
small synthetic optimizer/save/resume sequence and all prior continuous,
Morlet and legacy pair checks. The first immutable attempt had 163 passes and
one overly strict stationary-gradient sign-bit assertion: zero derivatives
included IEEE negative zero. Only that test was corrected to finite numerical
zero; the exact-positive-zero score and separate K=2 topology value/gradient
bit-pattern checks remain strict. The failed attempt is retained, not rewritten
or counted as a formal training restart.

A real development-only audit found **494 released segments but 253 parent
capture/holistic annotation families**. Exactly 138 families repeat across
segments; each file family maps to one parent stem, with no cross-component or
train/validation family duplication. The 494 files contain 2,613 sentence rows
including these copies (2,144 train, 469 validation); their per-segment row-count
histogram is 1:2, 4:20, 5:437, 9:4, 10:31. These are not 2,613 independent human
annotations. Final-test captions were not read.

The first native qualification's two segments shared a holistic file; its
finite backward was an execution check, not valid evidence of independent
retrieval targets. This prompted the explicit positive-family interface before
formal training. Parent-level motion/task aggregation, deduplicated gallery,
positive mapping, independent sampling/statistics and effective-test sealing
still require a unified contract freeze. Existing per-segment caches are input
artifacts, not proof that that task is closed. Do not treat sibling segments as
false negatives, bootstrap them as independent parent captures, invent a split
leak, change the participant-disjoint split, or concatenate missing time away.

After separating source and positive-family identities, the fresh server suite
again passed **164 tests, one CUDA-only skip in 46.98 seconds** with unchanged
source/invocation. Two **different** real C01 annotation families then passed
actual frozen-CLIP, complete **120/260-second** motion, all **12/26** accepted
base windows, **119/259** ordered patches, 10 human sentences and untrained
512D forward/backward in **151.26 seconds** total. All 55 trainable gradient
tensors were finite, required branches had nonzero gradients, frozen B2 stayed
unchanged, and the original group readout remained bitwise identical to the
previous qualified source in this runtime. Forward/backward were 51.69/71.94
seconds; peak RSS was 1,856,512 KiB. This is CPU execution qualification, not a
GPU throughput profile or retrieval result.

The same real-file audit counted **1,334 human answer-row slots when each
annotation family is counted once**, not 2,613 independent annotations and not
1,334 independently verified relation examples. No text truncation occurred in
this two-family check. It used one-capture QA floors and random untrained B2:
zero optimizer, pilot, formal, verified-CF and final-test runs. Complete parent
task admission, true human relation evidence and the 87-stage study remain
unfinished; their pending state must not be replaced by this qualification.

## Raw parent assembly, not concatenated preprocessed children

`parent_capture.ParentCaptureEpisode` accepts caller-licensed raw world body-22
tracks with native source provenance and their actual half-open 30 Hz frame
intervals. `assemble_parent_capture` aligns the same real actors across releases
and preserves the complete parent timeline. Unreleased intervals retain false
masks and exact positive-zero coordinates; they are not stationary motion,
interpolated poses or deleted time. Overlap or changed actor membership needs
an explicit data-contract resolution. Lineage is not a neural input.

Only after assembly is the existing capture-wide origin, shared yaw, FIR and
30-to-20 Hz grid applied once. Individually recentered/resampled child caches
cannot reproduce this raw-input operation. The assembly budget is explicit
and linear in actors and frames, not a claim about combined neural peak memory.
The existing complete-10-second window policy is unchanged: a shorter final
source tail is declared separately, not hidden or padded with invented poses.

The expanded registered-server CPU regression passed **170 tests, one CUDA-only
skip in 44.21 seconds**. It covers actor/episode permutation, direct full-world
preprocessing, original absolute intervals, 15/150-second holes, short native
tracking loss, tail accounting and input/resource rejection. Exact K=2 topology,
legacy pair checks and same-runtime bitwise invariance remain strict. An
independent directed-packet oracle uses frozen `rtol=2e-6, atol=2e-7`: separately
executed and batched GRUs produced at most 2.98e-8 observed score rounding
differences on Windows. This is not tolerance for actor permutation or old
readout equality, and no model/runtime guard was changed.

A preselected real development parent then passed raw licensed geometry,
assembly, physical-cache loading, native frozen CLIP and untrained 512D
forward/backward in **248.26 seconds**. Its three native releases contain
**7,650 observed frames** within an **8,550-frame (285-second)** parent span,
including **900 missing frames (30 seconds)**. The existing policy retains
**5,600 target frames**, declares the remaining 150 source frames, accepts
24 ten-second windows and rejects four. All 28 gap-only patches have no
actor/edge physical evidence. Raw actor/episode permutation is bitwise equal;
all 55 gradient tensors are finite and frozen random B2 remains unchanged.
Five real human sentence rows have no CLIP truncation. Forward/backward took
35.31/48.90 seconds; peak RSS was 1,969,516 KiB.

This is a CPU execution qualification using one-parent QA floors and scalar
score differentiation, **not** an all-positive one-parent InfoNCE experiment,
trained checkpoint, GPU profile, pilot, verified-counterfactual dataset or
formal run. Complete development-parent artifacts, five full train-only
calibrations, the unified parent gallery/sampling/statistics contract, faithful
literature controls, human relation verification and all 87 stages are still
required before final research delivery. The original failed invocation is
preserved; its existing SMPL-X library-path omission was corrected without
installing packages or changing the qualified model source.
