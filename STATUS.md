# PhaseSet project status

**Evidence snapshot:** 2026-09-08 (UTC)
**Research state:** `ACTIVE / AUTHORITY0 / NO_REAL_DATA_RESULT / NO_CLAIM`
**Published release:** `v0.2.1 / PUBLIC_DATA_FREE_STABLE_CORE_PORTABLE_TEST_HOTFIX`
**Final target:** `v1.0.0 / REAL_EXPERIMENTS_AND_PAPER / NOT_COMPLETE`

PhaseSet is the active multi-person successor to PhasePair. The in-place public
migration and `v0.2.0-alpha.1` pre-release are complete; the immutable
annotated PhasePair `v0.1.0` tag remains the dyadic compatibility baseline.
Implemented code, passing synthetic tests, and frozen plans are software
evidence, not training results or scientific claims.

## Server continuation

- Complete two-store descriptor-cache construction and cached residual
  checkpoint/resume now pass one combined 270-test Linux server cohort in
  429.82 seconds, with source unchanged. It also includes the qualification
  timer/runtime corrections and Windows readonly-shard fault-fixture fix.
  The actual official CLIP outputs retain their two exact embedding goldens
  and unchanged RNG under the new training-source binding. The same source
  subsequently passed 1360 full Linux tests, 2 existing skips and 39 subtests
  in 743.26 seconds, with source unchanged. Cross-platform CI acceptance is
  tracked separately; no formal run is
  implied. See [cache construction](docs/PERIODIC_CACHE_EXECUTION.md) and
  [cached training/resume](docs/CACHED_RESIDUAL_TRAINING.md).

- The cached complete-capture evaluator and private host qualification
  controller now pass one combined Linux server regression: 1334 tests,
  2 existing skips and 39 subtests in 643.90 seconds, with source unchanged.
  This closes software integration only. Native data access, all formal
  training, actual cohort qualification and sealed scores remain absent.
  The current successor now implements early POSIX/main-thread timer
  admission, exception-safe timer restoration, and qualify-base-only runtime
  classification for post-admission source/output/lease failures. Static review
  passed, and its focused Linux server run passed 145 tests in 64.24 seconds
  with source unchanged. That successor is not part of the observed 1334-test
  source; it is included in the newer 1360-test cohort above. Windows CI
  acceptance remains separate. See
  [host runtime boundaries](docs/HOST_BASE_QUALIFICATION.md).

- The combined capture-lineage, prepared-window cache-plan and CUDA-workspace
  correction passed 157 focused and 1285 full Linux server tests, two existing
  skips and 39 subtests, with source unchanged. A separate actual K=32 GPU
  retry completed B0/B1/B2 at full registered width and T=200: valid outputs,
  unchanged model state, and exact zero allocated/reserved memory after each
  release. Peak allocations were 81,548,800 / 94,160,384 / 315,323,392 bytes,
  below the unchanged 2 GiB cap. The probe used untrained models and one warmup
  plus one observed forward per model; it is not the formal 81-visit latency
  session, a trained cohort, retrieval evidence or a performance benchmark.
  Earlier failed attempts remain retained. See
  [CUDA lifecycle correction](docs/BASE_COHORT_QUALIFICATION.md).

- Capture-storage v2 now derives every descriptor context from the consumed
  manifest/NPZ and canonical global window ordinal while keeping v1 and full
  holistic-gallery semantics intact. It passed 92 focused and 1265 full
  server tests, two existing skips and 39 subtests, source unchanged.
  A separate prepared-window periodic plan passed 76 combined tests, admitting
  the exact twenty-epoch source/cache census for each registered seed and
  reopening each selected shard without fallback. A separate complete-capture
  plan now binds the holistic capture source and its cache without relabeling
  prepared-window auxiliary validation. At that predecessor milestone the plans
  were not yet integrated into training/checkpoint/resume; the current explicit
  cached lifecycle is described above.
  See [capture storage](docs/CAPTURE_PREPARED_STORAGE.md) and
  [window cache plan](docs/PERIODIC_TRAINING_CACHE_PLAN.md).

- The complete-capture plan enumerates the actual twenty-epoch prepared source
  and every holistic validation window for one registered seed, requiring its
  closed cache census before returning explicit sealed streams. The residual
  runtime now has a separate cached encoding entry, and a cached holistic
  evaluator shares the original complete pooling/gallery oracle while changing
  only periodic descriptor acquisition. The original uncached APIs remain.
  A focused server suite covering the cached residual model, frozen-text source
  binding and complete-capture plan passed 121 tests in 271.25 seconds with
  source unchanged. The complete combined regression then passed 1305 tests,
  2 existing skips and 39 subtests in 613.86 seconds, again with source
  unchanged. Three actual official CLIP forwards retained the established
  outputs and caller RNG. The later cached holistic evaluator passed 108
  focused server tests in 151.03 seconds with source unchanged; its subsequent
  combined 1334-test regression is reported above. These were
  software-source enumeration and text observations, not twenty epochs of
  training. At that predecessor milestone, training-loop, checkpoint/resume
  and host cache-plan consumption were unfinished. The newer core cached
  lifecycle is reported above; host end-to-end injection remains separate. See
  [complete-capture cache plan](docs/PERIODIC_CAPTURE_TRAINING_CACHE.md) and
  [capture validation](docs/CAPTURE_VALIDATION.md).

- Prepared-v2 batches now retain their exact descriptor source/window/epoch/yaw
  contexts. Repeated rotation of an already contextualized batch is rejected.
  The live CLIP training-source pin was updated and actual official text
  embedding bytes remained unchanged. A separate actual diagnostic localized
  a Torch/NVIDIA UUID-prefix mismatch; strict full-UUID matching now supports
  the observed representation without bypassing device or resource checks.
  The combined Linux server suite passed 136 focused tests and 1257 full
  tests, 2 existing skips and 39 subtests in 413.42 s, source unchanged.
  Complete training cache-plan/checkpoint/resume integration was still pending
  at that predecessor milestone;
  these observations are not real-data training or a K32 performance result.
  See [source lineage](docs/PREPARED_DATA_V2.md) and
  [CUDA identity](docs/BASE_COHORT_QUALIFICATION.md).

- Explicit reader-backed cached model execution binds the actual complete
  batch, energy floors and stream family, and enforces the live encoder edge
  budget before consuming descriptors. The same stream is
  replayed through forward incidence, backward moments and parameter VJP,
  without an uncached fallback. The model/cache/control suite passed 98 Linux
  server tests; full regression passed 1228 tests, 2 existing skips and
  39 subtests in 401.07 s, source unchanged. These are software fixtures,
  not real data or measured acceleration. At that predecessor milestone plan
  admission covered training rows and the complete epoch census, while the
  training-loop/checkpoint/resume consumption reported above was unfinished.
  See [descriptor model integration](docs/PERIODIC_DESCRIPTOR_CACHE_V2.md).
  The host now retains the canonical admitted plan/matrix/config instead of
  reading fields absent from the actual command intent. Commands and handler
  identity are checked before retaining the admission snapshot. Real
  CLI-to-backend fixture regression passed 94 focused tests; full regression
  passed 1232 tests, 2 existing skips and 39 subtests in 401.58 s, source
  unchanged. The new CLI fixtures intentionally stop at the model-runtime
  seam and retain FAILED terminals; they are not completed training runs.
  See [host integration](docs/HOST_RUNTIME_INTEGRATION.md).

- The complete frozen base-latency runner, exact qualification assembler,
  typed progress observer and durable POSIX journal are implemented. Separately,
  a weight-independent three-stream periodic descriptor cache now preserves
  exact canonical batches and bounded immutable shards. The merged Linux
  server source passed 1211 tests, 2 existing skips and 39 subtests in 362.75 s
  with source unchanged. Journal integration passed 73 combined tests;
  descriptor storage/streaming then passed 19 focused tests after a test-only
  hardlink/oversize case split. These are explicit
  software fixtures, not a real nine-run cohort, formal CUDA latency session,
  winner or licensed-data cache. The later explicit cache-model API is reported
  above. Current host end-to-end cached training/resume and host qualification
  command execution with real cohort inputs remain unfinished. See
  [base qualification](docs/BASE_COHORT_QUALIFICATION.md),
  [progress journal](docs/LATENCY_PROGRESS_JOURNAL.md) and
  [descriptor cache](docs/PERIODIC_DESCRIPTOR_CACHE_V2.md).

- The window-caption provenance component now checks the complete chain from
  same-window fusion inputs/outputs through ordered frozen text rows to the
  materialized training census. An actual official-CLIP probe exposed and fixed
  a prepared-data bug: inference chunk size had been mistaken for total caption
  count. Exact chunk coverage and all existing row/tensor/digest checks are now
  retained together. The unchanged probe subsequently passed three actual
  encoding chunks, prepared-tree reload and exact repeated provenance checks,
  without re-encoding or observed CPU Torch RNG change. The original failure
  remains retained. The source passed 53 focused / 1155 complete Linux server
  tests, two existing skips and 39 subtests; complete regression took 316.45 s
  with source unchanged. Fusion receipts, captions and motion in this probe
  were explicit software fixtures. No fusion backend, licensed data, motion
  training or formal supervision selection is claimed. See
  [caption-role and text-chunk binding](docs/CAPTION_ROLE_PROVENANCE.md).

- The completed-base resolver now reconstructs the exact nine registered
  successful chains, preserves failed predecessors, authenticates terminal-bound
  latest and distinct validation-best checkpoint bytes, and revalidates the
  complete registered state. The complete-capture scorer strictly loads those
  selected states, derives live parameter censuses and recomputes full-gallery
  score identities and exact capture-macro fractions. Their integration passed
  119 focused / 1119 complete server tests with 2 existing skips and 39 subtests.
  A separate unmocked full-width B0/B1/B2 selected-state-to-gallery witness
  passed all three lanes in 19.36 s on analytic K2/K3, T200 captures. It uses
  explicit checkpoint-cursor and text fixtures, not completed formal epochs,
  real captions or an actual nine-run cohort. The merged source at that stage
  passed 1124 complete Linux server tests, 2 existing skips and 39 subtests in
  315.67 s with the source census unchanged. Latency and qualification assembly
  were unfinished at that snapshot; their newer software verification is
  reported above. A private host qualification controller is now implemented,
  and its corrected focused server suite passed 131 tests in 37.07 seconds
  with source unchanged. The first attempt's seven fixture API failures and
  123 passes remain retained; only the test file changed for the successful
  retry. These software fixtures do not constitute actual qualification of a
  trained cohort. See
  [completed-base capture scoring](docs/BASE_COHORT_VALIDATION.md).

- Strict known-host SSH login to the registered host now succeeds. The initial
  read-only inventory is complete. Existing workloads were not interrupted.
- A separate data-volume Python 3.12.12 / NumPy 2.4.6 / Torch 2.12.0+cu126
  environment is installed with a clean dependency check. The complete Linux
  CPU regression including capture host composition, storage portability and
  frozen-text restoration passed: 1077 tests, 2 skipped, and 39 subtests, in
  296.55 s. Its focused integration passed 103 tests. The preceding capture
  host integration passed 96 focused / 1070 complete tests; the earlier storage
  integration passed 110 focused / 1043 complete tests. The preceding capture-training integration passed 1012
  complete / 84 focused tests and 6 focused subtests. Source bytes were unchanged before and after
  both suites. The prior prepared-data v2 host integration passed 94 focused /
  964 complete tests. The prior residual-resume integration
  passed 948 tests and 39 subtests.
  The earlier frozen-CLIP integration passed 932 tests and 39 subtests.
  The earlier host-only suite passed 917 tests and its 25 focused checks.
  This includes synthetic model/lifecycle checks, not main-data training.
- A separately bounded shared-GPU functionality probe passed FP64 forward and
  FP32 forward/backward using 8x8 tensors. It did not stop existing workloads
  and does not qualify BF16, model execution, performance, or the full runtime.
- A subsequent actual-width PhaseSet periodic-core CUDA observation passed
  with 512-dimensional tokens and a 256-dimensional hidden layer: K2/K3
  forward/backward, actor permutations, K2 topology output/gradients as exact
  positive zero, and K13 (78 edges) bitwise equality at chunks 64/128/256.
  Maximum observed CPU/CUDA token difference was 1.431e-6 within the frozen
  FP32 tolerance. The engineering fixture peaked at 80,722,944 allocated bytes
  under a 2 GiB allocator cap; this is not a performance benchmark. Its first
  attempt stopped before model construction because physical FB total and
  CUDA context-usable total were incorrectly equated; that failed receipt is
  retained. The corrected memory instrumentation did not relax a model gate.
  This does not qualify BF16, the learned group base, or the full training runtime.
- Separate actual-width B0/B1/B2 FP32 CUDA forward/backward lanes passed the
  registered parameter census, physical permutation/padding equality, finite
  gradients, frozen-text preservation, and CPU/CUDA tolerance checks. Their
  small analytic fixtures are not training or performance results. A subsequent
  residual retrieval lane failed its frozen-base CPU/CUDA embedding comparison
  after zero-residual scoring checks. That failed receipt is preserved. A
  same-weight diagnostic isolated an observed native-MHA-fastpath dependency.
  The frozen runtime now disables that fastpath on CPU and CUDA and restores
  caller flags. With unchanged tolerance, the new CUDA retrieval-head seam
  passed: frozen-base maximum difference 7.153e-7, score difference 4.471e-8,
  and finite nonzero exercised head gradients. This did not run the periodic
  core or an optimizer. A subsequent B0 repeat stopped before model creation
  for insufficient free memory. After memory recovered, a distinct retry chain
  passed retrieval/B0/B1/B2 with the same reviewed source and tolerance; the
  original resource failure remains preserved. Formal-training and BF16
  qualification remain pending. See [runtime policy](docs/FROZEN_MHA_RUNTIME.md).
- Full-width FP32 CUDA lifecycle checks subsequently passed for B0/B1/B2 and
  system 08: width 512, 8 heads, FFN 2048, full 200-frame body22 windows and analytic
  K2/K3 groups. Three uninterrupted optimizer updates exactly matched a newly
  constructed runtime restoring update 2 and completing update 3, including
  model, optimizer, scheduler, RNG, cursor and manifests. All four lanes stayed
  within a 2 GiB allocator cap; B2 retained all four social-temporal layers.
  These bounded fixtures used effective batch 2, completed no epoch, selected
  no validation checkpoint, and are not formal runs or performance benchmarks.
  See [observed lifecycle scope](docs/CUDA_LIFECYCLE_OBSERVATION.md).
- The native Embody loader and concrete licensed SMPL-X body-22 evaluator are
  implemented. Seven adapter contract tests passed on Linux/Python 3.12.12/
  NumPy 2.4.6; these use explicit fixtures, not licensed assets or main data.
- The immutable prepared-data v2 writer/reader now connects canonical body-22
  windows and frozen variable-count text features to the private-host factory.
  It verifies the exact consumed index/manifest/payload bytes, rejects split
  overlap, and applies reproducible shared-group yaw from original bytes each
  training epoch. Validation is unrotated. Window-positive families support the
  auxiliary task; a separate combined index now connects the private-host
  holistic capture source without relabeling auxiliary validation data.
  Actor-set commitments must not be used as unique capture/window row IDs.
  See [prepared-data contract](docs/PREPARED_DATA_V2.md).
- The complete-window capture evaluator now pools unnormalized embeddings
  before one full-gallery scoring call and preserves exact capture-macro R@1
  fractions. The training API uses this concrete val-only source for checkpoint
  selection and binds its full census on resume. The standalone caller passed
  42 focused / 997 complete server tests; the training integration passed 54
  focused tests including actual optimizer/checkpoint/resume equality, followed
  by complete 1002- and 1012-test regressions and actual CLIP requalifications. Two
  legacy fixture failures from the first full attempt are retained and explained
  in [test stability](docs/LEGACY_TEST_STABILITY.md). These are analytic lifecycle
  checks, not formal runs. Actual scored-census consistency and scoped live
  capture-function drift checks were added after provisional composition review.
  The private host now composes authenticated window training and capture
  validation from disk, checking actual actor/caption/window overlap before
  backend construction and forwarding the exact capture source through both
  base/residual and resume paths. Licensed source/caption provenance and
  participant-disjoint input preparation remain incomplete. See
  [capture validation](docs/CAPTURE_VALIDATION.md).
- A bounded validation-only capture storage writer/reader now preserves the
  complete window plan, original padding, variable caption counts and original
  frozen text receipts. Receipt-bound restoration performs no CLIP inference.
  An actual official-model cache probe caught a snapshot-file ordering bug that
  the initial structural fixtures had missed. The failed attempt is preserved;
  the corrected source passed the 110 / 1043 tests above plus three official
  forwards and two exact disk-restoration roundtrips, with no extra forward.
  The original output goldens and observed CPU Torch RNG were unchanged.
  Storage is not a rights, annotation, split or anonymization proof. See
  [capture storage](docs/CAPTURE_PREPARED_STORAGE.md).
- The preceding capture-storage commit failed 14 tests on each Windows CI
  runtime while its Ubuntu audit succeeded. A read-only filesystem diagnostic
  found different path-stat and descriptor-stat `ctime` values for the same
  unchanged file. The correction preserves complete before/after checks within
  each interface, compares stable identity fields across interfaces, and keeps
  all content-digest and bounded-read checks. Seven added cases cover this
  distinction and actual drift. The 103 / 1077 server tests above and the
  unchanged real CLIP cache observation passed. Windows acceptance of this
  correction requires its own GitHub CI result; no failed check was skipped.
- The next Windows run passed those storage-stat cases but exposed a separate
  test helper's default CRLF translation. The test-only LF correction and two
  authenticated CRLF negatives passed 71 focused server tests. Subsequent
  GitHub Actions for commit `db71a83` passed both Windows Python versions and
  Ubuntu's public-tree/wheel audit. Both preceding failed runs remain retained;
  production canonical-byte checks were not relaxed. See
  [fixture stability](docs/LEGACY_TEST_STABILITY.md).
- Neither Embody approval nor a licensed neutral model has been observed.
  Open-license Multi-TPC acquisition completed with its official byte count and
  MD5 verified. Its archive inventory has 322 files. The bounded audit confirms
  19 Euler rotation triplets plus three translation anchors, not body22 joint
  positions, and unresolved modality-alignment/group-identity details. It is
  supplementary conversation data, never replacement
  confirmatory data. The pinned CLIP snapshot's eight files also match the
  previously acquired official SHA-256 values. Actual frozen CLIP text CPU
  loading and two public-caption forwards passed, with bitwise repeat outputs
  and unchanged RNG. The production-shaped frozen text adapter subsequently
  passed 15 server tests and three real-model forwards: the unchanged repeat
  golden plus a separate 226-to-77-token long-caption observation. It verifies
  the exact retained prefix and EOS, restores caller RNG, and emits text-free
  receipts. See [frozen CLIP text adapter](docs/FROZEN_CLIP_TEXT_ADAPTER.md).
  This does not qualify GPU text execution or real-data training.
  The MHA-policy successor separately passed 33 focused/949 complete server
  tests and three official CLIP forwards, preserving the original output
  golden and RNG under correctly changed code/runtime/receipt identities.
  The capture-training successor also passed three official CLIP forwards with
  both prior output digests and RNG unchanged under its new source identity.
- The earlier text-adapter commit was downloaded from anonymous codeload,
  manifest/content/link checked, built offline into a wheel, and installed into
  a separate server target. All 56 wheel Python modules match the public Git
  index. Three real CLIP forwards from that installed wheel produced a complete
  observation byte-identical to the previously verified source copy.
- The subsequent public capture-storage commit was also anonymously downloaded,
  checked against its 223-file manifest, and built offline into a separate
  wheel target. All 60 Python modules matched the source. Three actual CLIP
  forwards and two disk restorations from the installed wheel produced the
  same complete observation as the server source. This Linux packaging check
  does not close the Windows filesystem-portability failure described above.
- Native Linux last-bit Morlet formula drift is repaired using strictly
  verified canonical coefficient bytes, while preserving both original hashes.
  Windows still requires its original exact formula digest; Linux requires its
  measured exact native formula fingerprint plus a bounded tap comparison.
  See [Morlet portability](docs/MORLET_PORTABILITY.md). POSIX checkpoint cleanup
  also now pins its inode so a concurrently replaced file is not deleted.
- The current no-result manuscript now builds using the hash-pinned official
  ICASSP 2027 template: five pages, technical content through page four,
  references-only page five, all fonts embedded, no unresolved citations or
  overfull boxes. All result cells and pending author metadata remain explicit.
- The concrete private-host bridge now connects base/residual training to
  live checkpoint receipts, a periodic heartbeat, and a whole-attempt OS lease.
  Base and residual resume verify the predecessor checkpoint chain, including
  process-loss recovery and authenticated latest/older-best materialization.
  Fifty-one focused server tests passed, including actual tiny residual
  optimization/checkpoint/resume equality and zero execution on invalid prior
  checkpoint bytes. See [resume details](docs/RESIDUAL_RESUME.md).
  Preparation and sealed evaluation host commands remain unwired. The separate
  base-qualification controller passed 131 focused server tests after a
  test-only correction; the first attempt's seven fixture API failures and
  123 passes remain retained. No actual nine-run completion is claimed; see
  [host base qualification](docs/HOST_BASE_QUALIFICATION.md).
  Four actual tiny server fits verified observer behavior without changing
  checkpoint/model/loss/validation outputs. These are engineering fixtures,
  not formal training attempts. GitHub CI passed both Windows runtimes and the
  Ubuntu public-tree/wheel audit for the integration commit.

This continuation is not a completed scientific release. Private receipts,
data, model assets, connection details, and download material are not public.

## Completed in this migration

- Full encrypted snapshots of the public repository and private research
  workspace, plus a verified all-refs bundle and SHA-256 receipts.
- Clean migration branch from exact public commit
  `123bf9d2017f09f11c812c6450f854595d2e29aa`.
- New `phaseset-core` distribution identity and `phaseset_core` namespace;
  legacy `phasepair_core` remains present.
- Frozen public and private ARIS research/experiment contracts for dynamic-K,
  permutation-invariant, streamed multi-person computation.
- Public/private boundary, migration policy, contribution policy, ownership,
  and security rules.
- Executable capacity-matched systems 00--08. System 01 is strictly
  motion/Morlet-independent fixed band-ID input; system 03 uses only symmetric
  endpoint self-power statistics and requires bilateral support, so it has no
  cross-endpoint phase or lag. Marginal-power, pair-only, coverage-only,
  incidence-shuffled, phase-stripped, and full paths are also implemented.
- Licensed-format-independent numeric capture preparation with body-22
  extraction, anti-aliased 30-to-20 Hz conversion, shared group transforms,
  exact masks, dynamic padding, salted lineage, and all-edge resource limits.
- Deterministic base/residual optimization, variable-positive symmetric
  InfoNCE, frozen-base residuals, edge-budget gradient caching, validation-only
  selection, immutable checkpoints, exact resume tests, and strict loading of
  each winning seed checkpoint with its frozen base logit-scale.
- A complete nine-row base-qualification receipt schema and closed validator:
  exact score fractions,
  parameter counts, integer per-run latency, nine terminal digests, nine
  validation-selected checkpoint digests, validation-manifest/query-census/
  evaluator digests, nine unique score-artifact digests, and three seed-ordered
  winner checkpoint digests. The production bridge recomputes the canonical
  artifact and requires a separately authenticated nine-row authorization.
  The resource tie-break uses the sorted middle of each
  base's three integer latency measurements. Every formal loader requires an
  externally trusted expected qualification digest; residual construction
  additionally requires the externally trusted complete-cohort capacity-audit
  digest before constructing any system. A checkpoint proves best-at-write;
  final-run best status remains bound to the immutable terminal/qualification
  evidence.
- Default 256-edge runtime chunks with fixed 64-edge canonical microblocks;
  system 06 uses a group-global, per-band half-edge bijection backed by only
  `O(E)` integer routing metadata, while B2 never materializes a complete
  `K x K` attention-score map.
- A frozen numerical runtime with deterministic-algorithm errors, highest
  float32 matmul precision, TF32/reduced-precision reductions disabled,
  deterministic cuDNN, a mandatory deterministic CUDA workspace setting, and
  disabled native MHA fastpath on CPU/CUDA,
  an environment-v2 hardware/runtime/build/thread inventory. Live drift is
  rejected and ambient host flags are restored on exit.
- A differentiable small-batch Torch CPU skeleton/Morlet oracle for input-
  gradient permutation qualification. Production periodic preparation retains
  an explicit non-differentiable NumPy descriptor-stream boundary.
- Digest-only auxiliary caption fusion with a frozen backend manifest, a
  trusted-host verify-and-consume gate for test admission, and an explicitly
  provisional same-family semantic-review status.
- Digest-only host-injected production adapter. The public CLI remains
  authority zero unless an authenticated private host injects that adapter.
  Qualification, evaluation, bootstrap, and render-paper completions are
  semantically recomputed. Rendering re-verifies the sealed-test ledger, exact
  27-row score census, checkpoint/terminal/environment-bound resource census,
  deterministic claim ladder, and exact publication artifact set.

## Registered scientific census

- Native Embody 3D census: 572 eligible K>=3 captures, 69 participants,
  20.673 scene hours, 27 K=3 and 545 K=4.
- Participant-disjoint split: 400 train / 96 validation / 76 test captures.
  Private membership and participant identifiers are not public artifacts.
- All K=3 captures are sealed test cases. Training does not observe K=3.
- Base qualification: 3 architectures x 3 seeds = 9 attempts.
- Periodic residuals: 8 systems x 3 seeds = 24 attempts.
- Formal training census: 33 attempts; final score matrix: 9 systems x 3
  seeds.
- Test runs once after validation and aggregation code are frozen. Inference
  independently evaluates all 27 system-by-seed score tables. It never averages
  logits across seeds. Each capture's three paired seed effects are averaged in
  fixed seed order; 100,000 bootstrap draws resample captures only, never seeds.
  H1--H8 share one index stream and Holm correction applies only to H2--H8.

No real-data attempt, score row, confidence interval, corrected decision, or
model-quality claim currently exists.

## External gates

Embody 3D requires a real applicant to submit the official release form with
true identity, institution, and email. No automation may invent those facts.
Private download URLs and licensed assets stay outside Git.

The earlier connection-refused state was superseded by successful strict
known-host login on 2026-09-08 UTC. Only the registered port and host remain
in scope. Resource availability, numerical qualification, and actual data
access are separate observed requirements; SSH success alone is not a passed
training environment.

The GitHub repository was renamed in place to
`appleweiping/phaseset-multiperson-motion-language`. Its numeric repository ID,
main ref, immutable v0.1.0 tag object, release, and latest successful Actions
run were unchanged. The old web URL returns a permanent redirect, and old/new
anonymous git and codeload endpoints resolve to the same refs and post-rename
archive bytes. The separate `periodic-motion-language` repository contains one
deprecation notice linking PhaseSet and is archived without history rewrite.

## Release provenance boundary

This source tree deliberately does not self-attest whether its declared
version has been published. A `v0.2.1` release exists only when the external
annotated tag, GitHub release, Actions results, repository identity, complete
asset inventory, and anonymous codeload receipts agree on the same commit and
tree while the immutable `v0.1.0` object remains unchanged. A branch name,
package version, changelog date, or statement inside the candidate tree is not
such evidence.

## Remaining empirical completion gates

1. Obtain legitimate Embody access and close private split/caption provenance.
2. Qualify the Linux/CUDA runtime, precision, canonical reductions, synthetic
   lifecycle, and disposable overfit.
3. Execute all 33 attempts with immutable terminals, then perform exactly one
   sealed-test evaluation.
4. Generate statistics, figures, tables, paper, and slides from one frozen
   aggregate; retain negative or inconclusive outcomes.
5. Publish the safe `v1.0.0` empirical release and verify a clean anonymous
   codeload can install, test, build the paper, and validate release hashes.

Until these gates have real evidence, PhaseSet is a data-free implementation
and execution-contract milestone, not a completed empirical paper.
