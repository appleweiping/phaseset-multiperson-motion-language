# Prepared-window periodic descriptor plans

`admit_periodic_descriptor_training_plan` derives a sealed plan from exact
formal-residual configuration, prepared-v2 train/val sources and authenticated
`PeriodicDescriptorCacheV2` readers. It accepts no caller-authored plan digest,
key subset or cache-miss fallback.

This API remains an engineering prepared-window seam. It does not implement the
main task's holistic capture validation, start training, restore a checkpoint
or establish data rights. A separate
[complete-capture plan](PERIODIC_CAPTURE_TRAINING_CACHE.md) now covers the
holistic validation source without relabeling this auxiliary validation input.

## Complete requests, one seed at a time

Admission enumerates the actual training source for epochs 0 through 19 and
the actual prepared validation source once at epoch zero, using one of the
three registered seeds. Every request binds its consumed manifest/NPZ,
window identities and ordinals, exact yaw bits, full numeric batch including
padding, edge count, energy floors and descriptor family.

The keys actually required by those requests must exactly equal each reader's
complete `cache_key_census`. Missing, additional, duplicate or corrupt rows
reject admission. The edge budget is checked before shard opening, with the
existing `ResourceLimitError`; no person or edge is silently removed.

System 02 uses marginal-power descriptors, 03 uses mean/difference DCT, and
04/06/07/08 use full relations. The registered model still applies incidence
shuffling or phase stripping where required. Base systems and residual
systems 01/05 reject this cache-plan API.

## Stateless selection

`plan.open_training_batch(batch)` and `plan.open_validation_batch(batch)`
recompute the actual request identity, match one admitted row, revalidate the
reader and complete key census, enforce the edge budget, and reopen the real
shard before returning its sealed stream. Repeated selection is repeated
verification, not an execution cursor or a cached learned output.

The plan binds actual source contexts, not unobservable iterator call
arguments. Prepared validation contexts are always epoch zero; a future
training integration must explicitly request validation epoch zero as well.

Admission preserves Python, NumPy, CPU Torch and already-initialized CUDA RNG
states. It does not initialize CUDA to inspect its RNG. If an input source
unexpectedly consumes randomness, the saved states are restored and admission
fails. These checks are ordinary API integrity, not arbitrary-Python isolation.

## Observed scope

The combined plan/source/cache/model suite passed 76 tests in 184.61 seconds
on the Linux experiment server, with source unchanged. Tests build actual
descriptor artifacts from explicitly analytic prepared-data fixtures and
exercise all three seeds, twenty epochs, family mapping, closed census,
resource order, request mutants and repeated reader opens.

No optimization, trained checkpoint, real-data cache, speedup, formal result or
holistic validation is claimed by that observation. The complete-capture plan
and explicit cached model entry are implemented as separate typed APIs. Their
training-loop, initialization/report, checkpoint and resume integration remains
unfinished; this prepared-window plan is not upgraded into main-task evidence
by those later components.
