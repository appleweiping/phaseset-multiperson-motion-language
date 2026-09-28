# Base cohort latency and qualification

The latency runner and deterministic assembler extend the
[completed-base resolver and capture scorer](BASE_COHORT_VALIDATION.md).
They implement a software path from nine completed training chains to the
existing qualification selector. No real nine-run cohort, winner or latency
benchmark has yet been observed.

## One complete latency session

`run_base_cohort_latency_session` reuses the admitted, validation-selected
checkpoint bytes for all three bases and all three seeds. It checks the
installed source, selected state, runtime, validation census and live parameter
census against the resolver and scorer. It does not accept supplied timings
or parameter counts.

The frozen workload is an analytic K=32, T=200, body-22 group at the registered
512-dimensional model width. It is an engineering timing workload, not a
replacement dataset. Nine cyclic rounds visit all nine rows: 81 visits in
total. Each visit has five warmups and eleven synchronized timed forwards,
giving 99 samples per row. Output copying and finite-value checking are inside
the defined forward interval; progress-journal I/O is outside it. The statistic
is the exact integer median of all 99 positive nanosecond samples.

The runner records the actual CUDA UUID, resolved device, runtime and precision.
Its shared-device policy requires at least 8 GiB free and a 2 GiB allocator cap.
An admission, resource, identity or execution failure holds the entire session.
It does not discard slow rows, lower K, remove layers, switch checkpoints or
select a quiet subset of visits. The controller must predeclare and freeze an
appropriate outer wall timeout before the actual session. A short functionality
probe's timeout is not the formal session timeout. Existing unrelated device
workloads are not interrupted.

The injected runtime seam is explicitly nonformal and exists for contract
tests. It cannot produce an accepted actual-runtime qualification merely by
finishing the same number of loops.

An actual registered-server diagnostic found that NVIDIA returned the complete
`GPU-`-prefixed UUID while Torch's CUDA UUID object rendered the same payload
without that prefix. The runtime now accepts only those two complete canonical
lowercase representations from Torch and compares the resulting full UUID
exactly against both registration and the NVIDIA observation. Short forms,
whitespace, uppercase payloads, repeated prefixes, MIG identifiers and changed
payloads are rejected; this conversion cannot select or change the device.
Tests prove invalid identities fail before allocator setup or tensor creation,
and accepted representations preserve the original allocator cap. The original
failed K32 observations remain retained; this identity correction alone is not
a completed K32 model run or a formal latency session.

The following actual K32 retry passed device initialization but failed the
strict zero-allocation check when releasing B0. No completed model row was
returned. Its failure and unchanged-source receipts remain retained; the
following diagnosis and correction did not change the workload or limit.

A separate actual diagnostic then found that B0's warmup and observed forward
returned without an earlier exception. All model parameters and buffers were
on CPU, but one 32 MiB process-local cuBLAS workspace remained. Garbage
collection did not remove it; the installed cuBLAS workspace-clear API reduced
both allocated and reserved memory to zero. That diagnostic still rethrew the
original failure and did not produce a completed latency row.

The release path now explicitly synchronizes and clears this process's cuBLAS
workspaces before emptying the allocator cache and applying the original exact
zero guard. Missing, noncallable or failing cleanup APIs raise a typed error;
real remaining allocations still reject release. Device selection, the 2 GiB
cap, the 8 GiB threshold, warmups, timing boundaries and visit order are
unchanged. This cleanup does not stop or alter any other process.

The corrected release path, capture lineage and window cache plan passed 157
focused and 1285 full server tests, two existing skips and 39 subtests in
512.46 seconds, with source unchanged. A subsequent actual K32 GPU retry
completed B0, B1 and B2 using untrained registered-width models, T=200, one
warmup and one observed FP32 forward each. All outputs were finite CUDA
`[1,512]` tensors, model state was unchanged and each production release passed
the unchanged exact-zero allocated/reserved guard.

| Model | Peak allocated bytes | Peak reserved bytes |
| --- | ---: | ---: |
| B0 ActorMean | 81,548,800 | 102,760,448 |
| B1 SetPMA | 94,160,384 | 119,537,664 |
| B2 SocialTemporal | 315,323,392 | 375,390,208 |

These are bounded diagnostic resource observations under the original 2 GiB
cap and 8 GiB free-memory admission. They do not constitute the formal
81-visit/99-samples-per-row session, latency qualification rows, trained
checkpoints or retrieval-quality results. No diagnostic duration is used as a
qualification tie-break. Every earlier failed attempt remains retained.

The integrated UUID/source-lineage/text-boundary CPU regression passed 136
focused tests and 1257 full tests, two existing skips and 39 subtests in
413.42 seconds with source unchanged. Its CUDA API stand-ins are explicitly
software tests, not a substitute for the separate actual device observation.

## Qualification assembly

`assemble_base_cohort_qualification` accepts the resolved cohort, admission,
complete capture-scoring result and complete actual latency session. It accepts
no submitted score rows, winner, timing samples or evaluator digest. It checks
the complete row order and binds each terminal, selected checkpoint, selected
state, query census, precision, runtime and installed-source identity across
the three observations.

Each `BaseScore` obtains its exact `Fraction` metric from the capture scorer,
its parameter count from matching scorer/latency live censuses, and its latency
from recomputing the retained 99-sample median. Its combined artifact digest
binds both the score and latency observations. The evaluator identity is the
installed production qualification artifact, not an unrelated module digest.

The assembler calls the existing `experiments.qualify_base` and production
canonical verifier. Selection remains: highest exact three-seed mean primary
score; then fewer parameters, lower three-seed median latency, and smaller base
ID. It does not implement a second competing selection rule. Selected bytes
and scoped source dependencies are checked for stability around assembly.

The result remains a software artifact with `authority=0`, `production=False`
and `result_claimed=False`. The trusted private controller must use actual
independently retained execution evidence through the existing backend flow.
This does not introduce an extra user-signature requirement, turn hashes into
data rights, or authorize access to sealed test data.

## Private host composition

The [host qualification controller](HOST_BASE_QUALIFICATION.md) implements the
private orchestration seam from the existing `qualify-base` CLI request to this
resolver, scorer, latency runner, journal and assembler. The CLI still supplies
only the attempt root; it cannot submit a winner, score, timing, parameter
count, checkpoint choice or authorization object.

Before the scorer places a model on CUDA, the controller applies the same
registered-device identity, minimum free-memory and allocator-cap policy used
by latency. After scoring it releases scorer references, synchronizes, clears
this process's cuBLAS workspaces, empties the allocator cache and retains the
exact-zero guard before starting the independent latency session. A scoring
failure remains primary even if cleanup also fails, and resource causes remain
HOLD outcomes rather than being converted into model-quality failures.

After a complete journal-agreeing latency result, the controller calls the
existing assembler, retains its exact bytes and lets the existing production
adapter recompute them before the backend callback. The controller creates no
new scientific selector or human-authorization mechanism.

## Progress, verification and remaining work

The optional typed observer connects to the
[durable progress journal](LATENCY_PROGRESS_JOURNAL.md). A journal terminal is
not a substitute for the actual latency call outcome and its receipt.

The latency contract passed 45 focused server tests; qualification assembly
passed a 53-test combined suite. The final observer/journal integration passed
73 tests in 22.63 s. The merged source, also containing the descriptor cache,
passed 1211 tests, two existing skips and 39 subtests in 362.75 s on the Linux
experiment server, with its complete source census unchanged. These counts
overlap and must not be added as independent experiments.

Tests use explicit software fixtures and injected observations. No actual
81-visit CUDA timing session, real completed training cohort, qualified winner
or data-dependent host `qualify-base` completion is claimed. The host controller
passed 131 focused server tests in 37.07 seconds with source unchanged,
including the real CLI-to-backend path and a structurally supplied qualification
callback. The first attempt's 123 passes and seven fixture API failures remain
retained. Its correction changed only the test file, not production guards.
These software tests did not produce real nine-run inputs or a qualified winner.
