# Window-caption provenance and frozen-text chunk coverage

`phaseset_core.caption_role_provenance` joins the content records that already
exist during private preparation: same-window human actor descriptions,
machine-fused description/paraphrase outputs, ordered frozen CLIP receipt rows,
and the materialized prepared-data v2 training census. It does not run a fusion
model, select a scientific supervision policy, authorize data, or train motion
models.

## Content binding

`validate_candidate_machine_fused_window_training(...)` checks the proposed
window-supervision mapping while original text is available. Each admitted
window has its own positive family, with the fused description followed by
its paraphrase. Human input text/source digests, actor membership, group/source
lineage, window start, fusion provenance, exact UTF-8 output digests, CLIP row
order and embedding bytes must agree. Canonical actor ordering after storage
is accepted without changing membership. The complete reloaded train census
must contain each bound motion and caption exactly once.

The returned private digest-only receipt retains machine-fused/non-human roles.
It is not a claim that a language model preserved semantics, nor a data-rights
certificate. Commitments can remain linkable and must stay private. A trusted
preparation process supplies already verified source summaries; caller-supplied
digests alone are not authentication. Runtime source bytes must be independently
verified before import, not inferred from a static review record.

The candidate is bounded to 16,384 windows, 32,768 caption rows, and 64 MiB of
inspected float32 text embeddings per representation. This is accounting after
objects or batches are materialized, not an aggregate allocator cap. The
validator never samples the census to fit a bound.

## Actual CLIP integration correction

The frozen adapter's `receipt.batch_size` is its inference chunk size, not its
total caption count. Prepared-data v2 now checks an exact integer chunk size
within the existing adapter bound and reconstructs the exact contiguous
`chunk_ranges` over every caption row. A chunk size larger than the census is
valid and yields one shorter chunk. Missing, reordered, overlapping or oversized
ranges and Boolean/malformed sizes are rejected. Existing row, shape, stride,
byte, full/individual embedding digest and positive-family checks remain.

A registered-server probe using the actual official CLIP weights first exposed
the old false equality after encoding four texts in two chunks. Its failure
was retained. After the correction, the unchanged probe encoded four training
fixture texts and two separate auxiliary-validation fixture texts in three
actual chunks, wrote and reloaded the prepared tree, and reproduced the exact
provenance receipt without another encode call. Observed CPU Torch RNG was
unchanged. All 53 focused server tests passed with source bytes unchanged.
A subsequent complete Linux server regression passed 1,155 tests, two existing
skips and 39 subtests in 316.45 seconds; its source census was also unchanged.

The texts, fusion receipts and analytic motion in this probe were explicit
software fixtures. No fusion backend, licensed dataset, motion-model training,
real cohort, retrieval-quality metric or formal supervision choice was involved.

## Scientific boundary

This validator implements a candidate hybrid: machine-fused same-window
description/paraphrase supervision for window InfoNCE, with separate official
human holistic capture validation/test. The candidate has not amended the
frozen experiment contract. It must not silently turn auxiliary captions into
human primary labels or copy whole-capture descriptions onto individual windows.
All systems/seeds must share whichever policy is explicitly preregistered
before real training. Primary capture pooling and checkpoint selection remain
the existing [capture-validation contract](CAPTURE_VALIDATION.md).

This is a preparation component, not the complete `prepare-data` command.
Licensed native inputs, participant-disjoint source preparation, actual fusion
execution and full private command composition remain required. See
[prepared-data v2](PREPARED_DATA_V2.md) and
[the frozen text adapter](FROZEN_CLIP_TEXT_ADAPTER.md).
