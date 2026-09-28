# Complete-capture periodic descriptor plan

`PeriodicDescriptorCaptureTrainingPlan` is the authority-zero planning seam
for one residual system and one registered seed. It combines the existing
twenty-epoch prepared-training request census with the main task's complete
holistic capture-validation source. It does not run training, select a
checkpoint, grant data access or claim a result.

## Complete admission

`admit_periodic_descriptor_capture_training_plan` accepts an exact residual
`TrainingConfig`, a prepared-v2 training source, a holistic
`CaptureValidationSource`, and separate authenticated descriptor-cache readers
for training and capture validation. It accepts no caller-authored plan digest,
partial key list or cache-miss fallback.

Admission enumerates every actual training row for epochs 0 through 19 using
the selected seed. It separately enumerates every window in every holistic
validation capture at validation epoch zero and positive-zero yaw. Capture
requests derive their source NPZ, storage manifest and canonical global window
ordinal from the loaded capture source; a loop index or source start frame is
not substituted for lineage.

The required training and capture key sets must each equal the complete census
of their corresponding cache reader. Source manifests, full source censuses,
capture plans, window and caption counts, numeric batches including padding,
energy floors, edge limits and descriptor families remain bound. Admission
preserves Python, NumPy and CPU Torch RNG, and preserves CUDA RNG when CUDA was
already initialized. These checks establish consistency of supplied artifacts,
not data rights or protection from arbitrary code in the process.

## Explicit selection

`plan.open_training_batch(batch)` revalidates one exact contextualized training
row and returns its sealed descriptor stream. Before cached holistic validation,
`plan.validate_capture_validation_source(source)` rebuilds and verifies the
entire capture source and cache census. The evaluator then calls
`plan.open_capture_validation_window(capture, window_position)` only for windows
from that rebuilt source.

Systems 02 and 03 select marginal-power and mean/difference-DCT descriptors;
systems 04, 06, 07 and 08 select full-relation descriptors. Systems 01 and 05,
base models, missing shards, wrong batches, foreign families and changed reader
censuses are rejected rather than routed to an uncached fallback.

`ResidualRetrievalSystem.encode_trainable_cached` is the corresponding model
entry. It supplies the verified stream to the registered PhaseSet encoder while
leaving the frozen base, residual head and learned outputs uncached. The
uncached model method remains available for existing callers.

`run_capture_validation_cached` shares the original holistic evaluator's
complete pooling, gallery, objective and capture-macro metric core. Its only
alternate branch is periodic descriptor acquisition. It validates the whole
capture source before any model forward or gallery score and consumes the
already attached frozen text features without invoking CLIP again. The original
`run_capture_validation` signature and uncached behavior remain available.

## Verification and remaining integration

A registered-server focused software suite covering the cached residual model,
frozen-text source binding and complete capture plan passed 121 tests in
271.25 seconds with source unchanged. The complete combined regression then
passed 1305 tests, 2 existing skips and 39 subtests in 613.86 seconds, again
with source unchanged. Three actual official CLIP forwards retained the
established output values and caller RNG. Those observations used software
fixtures and did not perform optimization, finish an epoch, load licensed data
or produce a checkpoint.

The later cached holistic runner passed 108 focused server tests in 151.03
seconds with source unchanged. Its complete combined regression with host
qualification integration passed 1334 tests, 2 existing skips and 39 subtests
in 643.90 seconds, source unchanged. The complete training lifecycle still needs a typed
plan handoff for initialization, all epochs, checkpoint/report binding and
resume. Host storage loading and standalone plan admission are not that lifecycle
integration. A real-data cache, acceleration claim, formal score and
qualification remain absent.
