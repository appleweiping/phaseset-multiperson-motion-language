# Immutable periodic descriptor cache

`periodic_descriptor_cache_v2` stores weight-independent physical pair
descriptors, not learned tokens or activations. The existing small periodic
cache stores energy-floor metadata; this separate format stores all three
descriptor streams: full relations, marginal power, and mean/difference DCT.
No licensed motion payload or private cache artifact is included here.

## Exact cache identity

A cache key binds the complete canonical numeric group batch, masks, padding,
actor/group lineage commitments, fixed Morlet oracle and energy floors. It
also binds explicit source/environment, manifest, batch, split, window ordinal,
seed, epoch and exact yaw identities. Commitments determine canonical reduction
order but do not become neural-network inputs.

These caller-supplied digests provide integrity and equality binding only;
cache construction does not authenticate their origin or grant dataset access,
consent, split, training or execution authority.

Reordering actors preserves canonical identity; changing dynamic batch packing
is a distinct key, not an inferred cache hit. Shared yaw includes floating-point
rotation and rounding, so another epoch's apparently rotation-invariant motion
is not assumed bitwise interchangeable. There is no approximate cross-epoch
reuse or silent uncached fallback for a configured wrong cache.

The bounded format preserves directed descriptor arrays, validity and endpoint
indices in canonical order. Re-iteration can yield requested runtime chunks
without changing the logical stream, including chunks spanning batch groups.
The same physical full-relation stream can underlie the controls that transform
it; marginal and DCT require their separately identified streams.

## Storage and resource boundaries

The writer checks expected pair count and decoded byte requirements from actor
counts before materializing a stream. The fixed decoded census is 1914 bytes
per edge for all stored arrays. Per-shard, decoded, total-byte and edge bounds
are explicit; an additional exact serialized-size check follows encoding.
Resource rejection does not drop actors, sample edges or truncate an artifact.

Artifacts are exclusively created and the index is published last. The reader
performs bounded owned reads, raw digest checks, strict schema checks and exact
dtype/shape/offset/payload digest validation. It checks static symlinks,
single-link payloads and byte bounds; loaded arrays are non-writeable. On
platforms providing `O_NOFOLLOW`, the owned read also requests no-follow
opening. On Windows the module is import-safe, but cache use requires a trusted,
stable, non-concurrently-mutated tree and does not claim POSIX-equivalent
no-follow or junction protection. Static symlink checks, single-link checks,
hashes and read-only flags are integrity hardening, not filesystem isolation
or data authorization.

## Descriptor storage verification

Nineteen server tests passed in 36.21 s with source unchanged. They cover all
three streams, a 186-edge K12/K16 mixed batch, chunks 64/128/256 including a
cross-group boundary, malformed authenticated artifacts, bounds, and unchanged
CPU Torch RNG. A small test-only encoder seam also checks exact cached versus
uncached parameter gradients through the existing custom backward replay.
The hardlink and oversize checks are independent: only an explicit hardlink
creation capability error can skip that case; unknown I/O errors fail, and the
oversize check still runs. All nineteen ran successfully in this Linux
observation. Its private receipt digest is
`8edbb675a64773fe104a1eae7b4bf398262b61827199152833da1e0765237f41`;
the receipt itself and its private paths are not published.

That earlier test seam deliberately supplied fixture chunks and was not a
production cache provider. The unchanged-source 1211-test regression used the
preceding production module and eighteen-test file; the test-only case split
was separately verified by the nineteen-test observation above.

## Explicit model entry and backward replay

`PhaseSetEncoder.forward_cached` and `PhaseSetSystem.forward_cached` now
accept an exact reader-backed `CachedPairChunkStream`. They revalidate the
actual complete `PreparedGroupBatch`, its canonical numeric digest, the
reader-bound six-band energy-floor digest, the expected stream family and the
complete unordered-edge census. Configured cached execution never silently
falls back to a physical recomputation.

| Consumer | Required cached stream |
| --- | --- |
| Plain encoder; systems 04, 06, 07, 08 | FULL_RELATION |
| System 02 | MARGINAL_POWER |
| System 03 | MEAN_DIFFERENCE_DCT |
| Systems 00, 01, 05 | Cached execution rejected |

Systems 06 and 07 still apply the registered incidence and phase-stripping
transformations. System 04 retains the shared postprocess with topology
disabled. No learned token, intermediate activation or gradient is stored in
the descriptor cache.

The same re-iterable stream is explicitly passed through forward reduction,
system-06 incidence census, no-grad backward moment reconstruction, and
differentiable microblock VJP replay. It is retained on that autograd call's
context, not installed globally or as mutable model state. Legacy `forward`
and `forward_activity` remain uncached.

Cached execution preserves the encoder's exact edge-budget contract. It
rejects E greater than the budget with the existing `ResourceLimitError` and
`RESOURCE_LIMIT` fields before consuming descriptors; equality is accepted.
This checks a real early integration defect: bypassing the uncached iterator
must not also bypass its resource admission.

## Observed model integration and remaining work

The combined descriptor/model/control/custom-autograd suite passed 98 tests
in 90.90 s on the Linux experiment server, with source unchanged. It includes
the 186-edge two-group stream at chunks 64/128/256, exact output and parameter
gradient comparisons, all six registered cached consumers, actor-storage
permutation, wrong batch/floors/family, caller RNG preservation, and cached
forward/backward with an instance-local forbidden uncached iterator, including
system 06. The 185/186 edge-budget boundary is independently exercised.

The complete integrated source then passed 1228 tests, two existing skips and
39 subtests in 401.07 s with source unchanged. These are CPU software fixtures,
not licensed-data training, CUDA performance measurements or retrieval-quality
evidence.

The [prepared-v2 source](PREPARED_DATA_V2.md) now derives immutable descriptor
contexts from the actual consumed NPZ, split manifest, global window ordinal,
seed, epoch and applied yaw. Legacy batches retain `None`, and a contextualized
batch cannot be rotated again while keeping stale yaw metadata.

The [prepared-window cache plan](PERIODIC_TRAINING_CACHE_PLAN.md) now admits
an exact complete 20-epoch training census and one val/epoch-zero census for
each official seed. Its selector reopens the authenticated reader on every
request; the reader exposes only a read-only complete key-census tuple.
This remains an engineering window-validation seam, not holistic capture
validation. A separate [complete-capture plan](PERIODIC_CAPTURE_TRAINING_CACHE.md)
now binds the same complete training census to every actual holistic
validation window. The residual system has an explicit reader-backed cached
encoding method, and [capture validation](CAPTURE_VALIDATION.md) can consume
that plan through its shared complete-gallery evaluator. Training-loop,
initialization and checkpoint/resume consumption still require integration.
The model checks actual batch bytes but does not invent source window ordinals
or silently select a different epoch/yaw context. Host preparation must supply
the correct already verified reader handle. No real-data cache, acceleration,
measured throughput or model-quality result is claimed.
