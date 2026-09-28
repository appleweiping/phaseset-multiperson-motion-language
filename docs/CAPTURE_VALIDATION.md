# Holistic capture validation and training selection

`capture_pooling.py` and `capture_validation.py` implement the main task's
numeric model-to-gallery path. `PhaseSetTrainingRuntime.fit(train_source,
capture_validation_source)` uses it for validation checkpoint selection. The
private host can load the on-disk capture source, but upstream licensed-data
preparation and provenance remain separate work; this is not a declaration of
data readiness.

## Complete capture inputs

`CaptureWindowPlan` binds a separate capture identity, distinct accepted-window
identities, and chronological source starts on the 300-frame/30-fps grid.
Upstream preparation must freeze the complete accepted nonoverlapping window
census before model outputs exist. Rejected windows and tails belong in its
audit; a plan cannot be assembled from whichever outputs happened to finish.

`CaptureValidationWindow` supplies one complete body-22 window as B=1,
200-frame `PreparedGroupBatch`. All windows of a capture must preserve its
valid actor set, group commitment, and K. Padding and input actor positions may
differ. Several distinct captures may legitimately share the same actor set;
actor-set group commitments are never used as unique capture or window row IDs.

`CaptureValidationCapture` combines the complete plan/windows with an exact
`FrozenClipTextBatch` and its participant-component label. Its text receipt
binds the actual feature bytes, but cannot establish that unseen text came
from an official human holistic annotation. That provenance and data rights
must be supplied by the private source audit. Variable positive text counts are
supported; no fixed three-caption assumption is introduced.

`CaptureValidationSource` is val-only and requires at least two captures.
It canonicalizes capture and window order and checks duplicate/missing/foreign
identities. Test cannot be relabeled through a split argument. Source plans,
commitments, embeddings, captions, and component membership remain private.

Optional [v2 storage lineage](CAPTURE_PREPARED_STORAGE.md) binds every window
to its actual consumed manifest/NPZ and canonical global ordinal. A source
must contain either no such lineage or a complete, single-manifest sequence.
It does not change capture pooling, text order or gallery semantics, and it
does not by itself connect descriptor caching to this evaluator. The separate
prepared-window cache plan cannot replace holistic capture validation. The
[complete-capture descriptor plan](PERIODIC_CAPTURE_TRAINING_CACHE.md) provides
that distinct typed connection.

## Encoding and pooling

The evaluator encodes every admitted window at B=1 with a registered retrieval
system interface and width 512. It uses the existing frozen numerical context,
eval/no-grad, and the configured qualified precision. CPU/CUDA device aliases
are resolved to actual tensor devices. Caller training modes and numerical
flags are restored even when an encoder fails.

Base embeddings are summed chronologically with the fixed adjacent float64
binary tree and divided by the complete accepted-window count. Periodic tokens
use the same tree but divide by each band's supported-window count. Results
are cast once to float32; unsupported bands remain exact positive zero.

Only after pooling does the existing retrieval scorer normalize features and
compute the full capture/text gallery in FP32. The implementation does not
average normalized window embeddings, window scores, or window recall values.

An explicit byte budget covers supplied encoded arrays, and the configured
edge budget admits every pair in every window. CUDA also checks a static
model/gallery tensor estimate; this is not a total allocator/RSS guarantee.
Resource failures reject the complete operation, never sample windows,
captures, people, or edges. CUDA OOM remains an explicit resource failure.

## Explicit cached descriptor path

`run_capture_validation_cached` uses the same complete source, fixed-tree
pooling, full gallery, objective and capture-macro metric implementation as the
uncached runner. It accepts an admitted
`PeriodicDescriptorCaptureTrainingPlan`, binds it to the residual system and
complete training configuration, and asks the plan to rebuild and validate the
whole capture source before any model forward or score.

For each actual window in that rebuilt source, the plan reopens one verified
`CachedPairChunkStream`; `ResidualRetrievalSystem.encode_trainable_cached`
passes it explicitly to the registered periodic encoder. There is no silent
cache-miss fallback and no cache of frozen-base or learned residual outputs.
The attached frozen CLIP feature batch is validated and consumed exactly as in
the uncached path; cached validation does not re-encode text. Prepared-window
auxiliary validation is not accepted as a substitute for the holistic source.

The original `run_capture_validation` API remains unchanged. Both entry points
share the same core, and the cached wrapper/core are included in the scoped
live-function identity checks. These checks detect in-process drift within the
declared runtime; they are not arbitrary-Python isolation.

## Metrics and checkpoints

The gallery contains one motion row per capture and every admitted holistic
text positive. The existing evaluator produces integer R@1 hits. Exact
`Fraction` arithmetic computes each capture's T2M and M2T contributions and
their equally weighted capture mean. The primary metric is their half-sum;
captions are not independent capture clusters. Variable-positive symmetric
InfoNCE is computed on this same full gallery.

Training admission snapshots the complete validation census, including the
upstream manifest and window/text feature identities. The existing checkpoint
`val_manifest_sha256` stores that combined census digest. A changed gallery
is rejected on resume, and a capture source cannot become a training iterator.
At each epoch the training loop uses this evaluator's capture primary metric
for the normal best-checkpoint selection and records all encoded window edges.
It first creates an owned snapshot and computes edge counts from that snapshot,
then verifies the evaluator's returned scored census before consuming its metric.
Formal runtime checks also bind the capture class/runner and the key imported
pooling, evaluation, and objective functions. This is scoped live-function drift
detection, not a security guarantee against arbitrary Python execution.

The checkpoint schema retains its historical float metric field; it is not
presented as an exact qualification artifact. Qualification must recompute
exact fractions from the selected checkpoint and full capture gallery. The
window-oriented validation source remains available for the separately
declared auxiliary/legacy task and is not main-task evidence.

## Observed scope and remaining work

The standalone caller's focused server suite passed 42 tests, including actual
512-wide B0 and PhaseSet encoders; its full regression passed 997 tests,
2 skipped, and 39 subtests. The subsequent training integration passed 54
focused tests, including actual tiny optimizer/checkpoint/new-runtime resume
equality and rejection of a changed capture census. Full regression also
exposed two legacy fixture-timing/lifetime failures
in a complete run (1000 passed, two failed), preserved as a failed attempt.
The [test-only stability repair](LEGACY_TEST_STABILITY.md) subsequently passed
74 focused tests and 6 subtests, then 1002 complete tests, 2 skipped, and
39 subtests in 297.31 seconds, with unchanged source bytes. Three actual
official CLIP forwards requalified the changed training source while retaining
the pre-existing output golden, long-caption observation, and caller RNG.
An available same-author provisional composition review then identified the
scored-census and live-function consistency gaps described above. Their narrow
repair passed 84 focused tests and 6 subtests, then 1012 complete tests,
2 skipped, and 39 subtests in 286.46 seconds. Three more actual CLIP forwards
retained the original output values and caller RNG with correctly updated
source identities. Earlier attempts are not overwritten. Initial review used
the ARIS local-only fallback; the subsequent same-author review remains
provisional, not independent scientific acceptance.

These tests use labeled analytic fixtures. They do not substitute for native
Embody captures, official holistic descriptions, a rights grant, complete host
command composition, the nine-run base qualification, or sealed evaluation.
See [host integration](HOST_RUNTIME_INTEGRATION.md) and
[prepared window storage](PREPARED_DATA_V2.md).

The complete capture descriptor plan and cached residual model entry later
passed a 121-test focused server suite in 271.25 seconds with source unchanged;
the complete combined regression passed 1305 tests, 2 existing skips and 39
subtests in 613.86 seconds, also with source unchanged. Three official CLIP
forwards retained the established outputs and caller RNG. Those suites
enumerated the twenty-epoch software source census but did not train for twenty
epochs or produce a checkpoint. The cached capture runner then passed 108
focused server tests in 151.03 seconds with source unchanged. Its complete
combined regression, including host qualification integration, passed 1334
tests, 2 existing skips and 39 subtests in 643.90 seconds, source unchanged. Training,
checkpoint/resume and host lifecycle integration for the plan remain pending.
