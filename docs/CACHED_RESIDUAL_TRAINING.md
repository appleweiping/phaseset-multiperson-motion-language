# Cached residual training and resume

The residual lifecycle accepts an explicitly admitted
`PeriodicDescriptorCaptureTrainingPlan` for systems 02/03/04/06/07/08. It does
not autodetect caches, substitute a window-level validation task for the full
capture gallery, or fall back to uncached descriptors on a miss.

Pass the same exact plan to both
`construct_registered_residual_seed_bound_system(..., descriptor_plan=plan)`
and `PhaseSetTrainingRuntime(..., descriptor_plan=plan)`. Admission checks its
system, official seed, formal twenty-epoch configuration, edge budget, energy
floors, train manifest and complete holistic capture census. Systems 01/05 and
base systems do not accept descriptor plans.

Every gradient-cache first pass and differentiable replay reopens the current
batch through `plan.open_training_batch(batch)` and calls
`encode_trainable_cached`. Readers, streams and learned activations are not
stored in checkpoints. A replay error restores the post-objective RNG state
and propagates before an optimizer or scheduler step; no failed cache read is
replaced by recomputation. Holistic validation calls the shared
`run_capture_validation_cached` implementation for every accepted window.

## Serialized compatibility

Uncached initialization, checkpoint and report schemas and bytes remain v1;
they do not gain nullable cache fields. Cached execution uses additive schemas:

- `phaseset-training-initialization-binding-v2-cached-descriptor-plan`
- `phaseset-training-checkpoint-v2-cached-descriptor-plan`
- `phaseset-training-report-v2-cached-descriptor-plan`

The cached checkpoint binds the complete plan digest and a dataloader digest
over the train manifest, holistic capture census and that plan. Resume must
reconstruct the same live plan and revalidate those identities before restoring
state. Cached and uncached checkpoints are not implicitly migrated into one
another. Paths, descriptor payloads, reader objects and stream cursors are
excluded from serialization.

## Evidence and remaining integration

The combined registered-server software cohort passed 270 focused tests in
429.82 seconds with source unchanged. It includes real analytic cache writers,
readers and plan admission, small-model lifecycle tests, failure injection and
the platform corrections. The official frozen CLIP adapter also retained its
two exact embedding goldens and unchanged RNG after the training-source pin
was updated. The same source then passed 1360 full Linux tests, 2 existing
skips and 39 subtests in 743.26 seconds, with source unchanged. These
observations are not formal model-quality results.

Full-width cached numerical equivalence and host-level plan/configuration and
checkpoint-ledger injection are separate work. The current CLI host does not
yet dispatch this cached lifecycle end to end. Native data, fitted scientific
floors, the nine qualified bases and the formal residual runs remain required.
See [cache construction](PERIODIC_CACHE_EXECUTION.md) and
[complete plan admission](PERIODIC_CAPTURE_TRAINING_CACHE.md).
