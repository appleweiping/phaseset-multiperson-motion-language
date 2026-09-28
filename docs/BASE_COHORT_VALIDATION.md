# Completed-base cohort and capture validation

The completed-base resolver and capture scorer connect the registered training
ledger to actual validation-selected model states. They do not select a winner
without the separately required complete latency session and qualification
assembly, and they do not establish data rights.

## Completed attempts, latest checkpoints and selected checkpoints

`resolve_completed_base_cohort` consumes the trusted host's attempt registry
and admitted plan, matrix, installed-source, runtime and data bindings. It
requires exactly one successful chain for each registered B0/B1/B2 by seed
row. Failed and held predecessors remain in the validated resume chain rather
than becoming additional score rows.

The terminal-bound latest checkpoint is not automatically the best checkpoint.
The resolver authenticates and decodes that payload, follows its exact best
filename and digest, and independently validates the selected payload. It
retains immutable owned bytes rather than returning a mutable checkpoint path.
Latest and selected checkpoints must agree on best metric and schedule; the
selected validation cursor and cumulative counters cannot exceed the latest
state. Selected-best-at-write evidence alone is not a completed-run receipt.

The checks reuse the actual registered constructors and checkpoint validators:
closed headers, full model state, optimizer, scheduler, RNG, cursor, seed-bound
initialization, code, environment and behavior. Bounded no-follow regular-file
reads precede restricted CPU checkpoint decoding. The recursive attempt chain
is rejected before a 65th entry, and checkpoint directory iteration has a fixed
100,000-entry bound. A single payload is bounded at 2 GiB and all retained
selected payloads at 4 GiB. Resource rejection never silently drops a base row.

## Actual complete-capture scoring

`score_resolved_base_cohort` accepts those resolved bytes and one concrete
validation-only `CaptureValidationSource`. Its full census must equal the
checkpoint's admitted validation binding. This census includes the upstream
manifest, complete window plan, numeric inputs, text receipts and identities;
the upstream manifest digest remains separately identified.

For every registered row, the scorer revalidates the selected bytes, constructs
the real seed-bound base, strictly loads its complete state, verifies the live
state and behavior, and derives the full trainable parameter census including
`logit_scale`. There is no random-checkpoint fallback, caller-supplied factory,
parameter count or score input. Each architecture's parameter rows must agree
across all three seeds.

The existing audited [capture evaluator](CAPTURE_VALIDATION.md) encodes every
accepted window, pools unnormalized embeddings per capture and evaluates one
complete motion/text gallery. The scorer independently checks gallery IDs,
positive relations, group sizes, component labels, score bytes and exact
capture-macro R@1 fractions. It checks source census and selected-byte stability
before and after scoring. One failed row prevents a complete cohort result.

The complete float64 score matrix has a 256 MiB preflight bound. No capture,
caption, window, actor or edge is sampled to fit this bound. Returned score
bytes are immutable little-endian float64; canonical observation bytes bind
their digest to the actual selected checkpoint, terminal, runtime and query
census. These private query identities and scores are not automatically safe
to publish.

## Acceptance boundary

These APIs emit software observations, not a completed qualification or new
execution authority. The trusted controller must still combine actual nine-run
terminals, scores and parameter censuses with one complete frozen latency
session, then call the existing selector and production evidence assembly.
No caller may authenticate a qualification merely by submitting matching
hashes. The exact formal training-caption policy must also be clarified before
real training; the scorer does not choose that policy or access sealed test.

Server contract tests cover restricted checkpoint state validation, exact row
ordering, latest/best linkage, immutable inputs, actual score-derived metrics,
live parameter counts and malformed dependencies. Explicit analytic fixtures
are not real training attempts or evidence of retrieval quality. In the
observed integrated candidate, 119 focused and 1119 complete Linux server
tests passed, with two existing skips and 39 subtests; source comparisons
passed. A separate three-lane server witness subsequently passed for actual
registered width-512 B0/B1/B2. Without replacing constructors, checkpoint
validators, encoders, pooling or scoring, it loaded complete selected model
states and scored two analytic T200 captures with K2/K3. The checkpoint cursor
and single initialized optimizer-state row were explicit software fixtures,
not formal epochs or a completed attempt ledger. Its text embeddings were
analytic unit vectors, not captions encoded by CLIP. All three tests passed in
19.36 seconds with source unchanged. This tests numeric integration only;
later cross-platform/public-package checks retain their independent scope.
The merged source, including that actual-width witness and the previously
verified LF-only host fixtures, subsequently passed 1124 complete Linux server
tests, two existing skips and 39 subtests in 315.67 seconds. The full source
census was unchanged across the run. This merged regression does not turn the
analytic checkpoints into completed formal training attempts.
