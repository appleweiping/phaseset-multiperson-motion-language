# Base and residual host resume

The private host supports both registered base-qualification and residual
training attempts through `phaseset resume`. The public CLI still requires the
existing authenticated host adapter; this feature does not manufacture data
rights, a qualified base, or an experiment result.

## Command contract

Base resume accepts an existing predecessor checkpoint and a new attempt
directory. Residual resume additionally requires `--base-checkpoint` and
`--periodic-cache`, the same artifact inputs as `run-residual`. Base resume
rejects these residual-only arguments. Resume has no stop-after-step option.

The verified predecessor run ID determines the seed, system, and legal role.
An explicit system argument must agree. Both attempts must share their private
attempt root. Symlinked paths, absent artifacts, incompatible roles, or a
checkpoint other than the terminal-selected checkpoint are rejected.

The residual host reconstructs, in order, the base qualification, selected
frozen base, validated periodic-cache energy floors, seed capacity audit, and
registered residual constructor. The unchanged training runtime then verifies
its configuration, data, code/environment, initialization, optimizer, RNG, and
resume-chain bindings before continuing.

## Immutable attempts and checkpoint copies

Resume creates a new attempt rather than rewriting its predecessor. A process
lease excludes a simultaneous writer. Process-loss recovery acquires the same
lease and uses already persisted checkpoint evidence; it does not guess from
PID age or invent a missing checkpoint.

The host copies the authenticated latest checkpoint into the new attempt with
exclusive creation, a regular-file descriptor check, streaming SHA-256, and
fsync. It reads and rehashes the bytes from the same open destination descriptor
before decoding an owned byte buffer with `weights_only=True`. Unverified
checkpoint bytes are never passed to the decoder.

If that payload refers to an older best-validation checkpoint, only the named,
digest-bound best file is copied too. This allows the unchanged runtime to find
its best checkpoint under the new attempt directory. The copies do not replace
the predecessor ledger or alter checkpoint payloads.

A failed copy is retained inside the new failed attempt for diagnosis. The host
does not use a check-then-unlink sequence that could delete a replacement file.
Failure artifacts and a truthful FAILED terminal are written; retained partial
bytes are not accepted as a resumable checkpoint. The predecessor is unchanged.

## Cache provenance boundary

Existing checkpoints bind the consumed six float64 energy floors through the
registered residual initialization, together with the selected frozen base,
qualification, capacity audit, and factory/code identity. They do not bind the
historical periodic-cache filename, complete file SHA, or descriptors that the
runtime did not consume. Resume validates the supplied cache and reconstructs
the same consumed state; it does not retroactively claim file provenance.

## Observed verification

On the registered Linux server, 51 focused tests passed in 50.29 seconds with
unchanged source bytes. Coverage includes role/artifact rejection, real attempt
ledgers, latest and older-best materialization, changed checkpoint bytes rejected
before decoding, zero runtime calls on invalid prior-best input, failure
terminalization, and a tiny genuine optimization/checkpoint/interruption/resume
comparison against uninterrupted execution.

Two earlier test-fixture failures were retained and corrected without changing
the host implementation: an incomplete fake report and an outdated deletion
expectation. The tiny actual runtime test already passed in that first attempt.
These are engineering tests, not formal PhaseSet training or scientific results.

The complete server regression with this integration then passed 948 tests,
two skipped, and 39 subtests in 376.00 seconds, again with unchanged source bytes.
