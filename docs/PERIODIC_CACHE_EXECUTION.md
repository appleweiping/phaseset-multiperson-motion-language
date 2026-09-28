# Complete periodic descriptor cache construction

`periodic_cache_execution` builds and consumes one seed-specific private cache
tree. It composes the actual v2 descriptor writers/readers and complete plan
admission; it does not estimate energy floors, choose a base, run a model or
authorize data.

The entry points are `load_qualified_energy_floor_input`,
`build_periodic_descriptor_cache_for_seed` and
`consume_periodic_descriptor_cache_build`. They require one official seed, its
formal twenty-epoch residual configuration, canonical selected-base
qualification, the exact winning checkpoint, supplied qualified floor bytes,
an admitted prepared-v2 train source, a complete holistic capture-val source,
source/environment identities and explicit resource bounds.

The writer enumerates every train batch in epochs 0..19 and every accepted
capture-validation window once, then admits the complete plans for
02/03/04/06/07/08. Systems 01/05 do not receive fabricated descriptor plans.
No person, edge or window is silently dropped to fit a resource limit.

The train writer is constrained by both its per-cache and combined limits;
the capture writer receives only the positive remaining shard/byte allowance.
An exceeding shard is rejected before it is written. File reads compare
portable path/descriptor identity and independently verify full before/after
metadata on each interface, including the Windows ctime distinction.

## Immutable output and consumption

The output root must be new. It contains `train/`, `capture-validation/`, six
`plans/<system>.json` files, and a final `build.json`. The last file binds the
seed/configuration, selected base/terminal/checkpoint, consumed floor
record/payload/vector, both source and cache identities/censuses, six admitted
plans, code/environment and resource bounds. It is published only after the
complete build and admission succeed. Partial failed roots are preserved and
are not resumed or overwritten by this version.

Consumption requires a separately retained expected build-manifest digest.
It reopens both readers, repeats all six complete admissions, compares stored
plan bytes, checks the live checkpoint and training-code identities, and
reconstructs the exact canonical manifest. A self-consistent replacement JSON
file or an asserted success boolean is insufficient.

Cache contents remain private: commitments can still be linkable, and
descriptor shards must not be treated as automatically publishable data.

## Verification boundary

The combined 270-test registered-server cohort includes actual complete
twenty-epoch and full-capture analytic construction/consumption, no-overwrite,
wrong split/checkpoint, mutated plan, extra member, strict floor payload,
aggregate disk limits and canonical type-drift tests. It passed with unchanged
source. Full regression on the same source passed 1360 tests, 2 existing skips
and 39 subtests in 743.26 seconds. These are software tests, not licensed-data
construction or training.

The existing floor-only record proves linkage to supplied seed/base bytes;
it does not prove how the six values were scientifically fitted. A separately
specified actor-once training-only floor collector and provenance are still
required for formal use. Diagnostic constants are not a substitute. Host CLI
dispatch and typed production evidence also remain separate integration work.
See [cached residual lifecycle](CACHED_RESIDUAL_TRAINING.md).
