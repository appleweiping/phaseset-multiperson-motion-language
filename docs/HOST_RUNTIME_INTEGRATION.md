# PhaseSet private-host runtime integration

The composition root is `src/phaseset_core/host.py`. It contains no server endpoint,
credential, dataset root, or receipt payload.

## Implemented production chain

`host.RuntimeAdapterFactory(private_host_config)` is passed directly to
`phaseset_core.cli.main(..., runtime_adapter_factory=factory)`.  The factory:

1. loads an existing private receipt record;
2. hashes every receipt artifact named by that record, including the rights,
   runtime, execution, source/prepared-data, split, caption, scoring-census,
   and hard-gallery records required by `PrivateReceiptAssertions`;
3. reads every prepared NPZ once into a bounded buffer, authenticates that exact
   buffer, and parses the same bytes without loading pickle data;
4. constructs the repository's `ProductionRuntimeAdapter`.

It never creates authority, rights, runtime, or execution receipts.  If an
artifact is absent or changed, the factory raises the existing
`ExecutionHold` codes and the public CLI remains authority zero.

Admission and command execution have distinct typed contracts.
`RuntimeAdmissionRequest` carries the frozen plan, matrix, training-config,
command census and handler digest; it has no per-command `command` field.
`CommandIntent` carries the selected command and run, not matrix/config fields.
The backend checks all five canonical admission bindings before retaining its
one-shot snapshot. New base/residual attempt records obtain their three
experiment digests from that retained admission, while dispatch checks the
actual intent against the parsed request. A rejected admission cannot poison
the snapshot and prevent a later canonical admission. Resume continues to
inherit its verified predecessor binding. No extra operator hash approval is
introduced.

The factory dispatches v1 and v2 prepared indexes using the actual index bytes
bound to the private prepared-data receipt. The v2 loader also verifies each
consumed split manifest before decoding its numeric batches. Malformed batch
semantics are translated into the existing execution hold before a training
backend is constructed. The legacy v1 loader now checks its consumed manifest
against the index digest rather than relying on a separate earlier path read.

`PrivatePreparedDataSource` maps a v1 immutable split manifest to the public
`TrainingDataSource` protocol.  Each NPZ contains exactly:

- `skeletons`: float32 `[B,K,200,22,3]`
- `actor_mask`: bool `[B,K]`
- `frame_mask`: bool `[B,200]`
- `track_mask`: bool `[B,K,200,22]`
- `actor_commitments`: uint8 `[B,K,32]` (ignored where actor_mask is false)
- `group_commitments`: uint8 `[B,32]`
- `text_embeddings`: float32 `[Q,D]`
- `motion_positive_ids`: uint8 `[B,32]`
- `text_positive_ids`: uint8 `[Q,32]`
- `text_commitments`: uint8 `[Q,32]`

The v2 contract fixes the text width at 512, accepts variable caption counts,
and supplies epoch-specific deterministic whole-group yaw from immutable
unrotated training bytes. Its validation source is unrotated. See
[prepared-data v2](PREPARED_DATA_V2.md) for the writer, bounded numeric format,
split checks, and provenance limits. Neither loader establishes data rights.

## Main holistic capture validation

The additional closed `phaseset-host-capture-validation-index-v1` composes an
existing prepared training index with validation-only
[capture storage](CAPTURE_PREPARED_STORAGE.md). The combined index bytes are
bound to the same private prepared-data receipt. Both the outer index and the
referenced training index must equal canonical ASCII JSON bytes, not merely
parse to equivalent objects. Duplicate keys and noncanonical encodings are
rejected even when a supplied digest matches them.

Its `train` row has exactly `kind/path/sha256`. The kind is
`prepared-training-index-v1` or `prepared-training-index-v2` and must match the
actual referenced schema. That complete old index is validated, then only its
training source is used. Its auxiliary window-validation source is never
relabeled as holistic capture data.

Its `val` row has exactly `kind/path/sha256/source_census_sha256/`
`upstream_manifest_sha256`, with kind `capture-validation-storage-v1`. The host
loads the actual storage bytes with the retained expected digest and checks
the reconstructed source type, `val` split, complete census and upstream
manifest. The runtime receives this `CaptureValidationSource` unchanged.
Base and residual runs, including both resume paths, use the same composition.
Checkpoint and scoring-time census binding remain in the existing training API.

Before constructing the backend, the host streams every actual training batch
and rejects actor/track or caption commitments shared with capture validation.
Prepared v2 additionally compares its documented window-positive IDs with
capture window commitments; v1 arbitrary positive families are not falsely
treated as window IDs. Source manifests/censuses are checked before and after
the scan. The fixed scan order uses local deterministic source RNG and preserves
Python, NumPy and Torch global RNG. Actor commitments are not participant IDs:
this check complements, but never replaces, the authenticated participant split
audit.

Preflight materializes the bounded capture source and streams the training
tree once. The single-batch byte limit still applies; train scan I/O is linear
in total artifact bytes, and there is no new aggregate train-byte limit.
Capture identity sets are linear in the materialized validation census. This
operation neither loads CLIP nor opens the sealed test split. Direct legacy
v1/v2 window-task dispatch retains its prior behavior.

`run-base` uses the formal seed-bound B0--B2 constructor and
`PhaseSetTrainingRuntime`.  Its checkpoint directory must be
`<attempt-root>/<attempt-id>/model-checkpoints`, allowing the harness to build
the public write-once attempt/checkpoint/terminal ledger around the runtime
checkpoint.  A controlled interruption is terminalized as
`CONTROLLED_INTERRUPTION`, retaining a resumable checkpoint.

The optional `checkpoint_observer` adds a post-atomic-write
observer to the public runtime without changing model, optimizer, sampler, or
numerical behavior. The host uses it to write live checkpoint receipts and a
30-second heartbeat chain while `fit()` is active. The final update checkpoint
of an epoch is deliberately deferred because the runtime immediately writes a
validation checkpoint at the same global step, while `AttemptStore` requires
strictly increasing receipt steps. Python failures and controlled interrupts
terminalize against the latest live receipt. A report checkpoint restored from
the predecessor is explicitly receipted into the new attempt even when `fit()`
makes no progress and emits no observer event.

Each active attempt holds an OS-released exclusive lock at
`<attempt>/.lifetime.lock` from attempt creation through successful or failed
terminal persistence. Heartbeats are immutable files under
`<attempt>/heartbeats/`; checkpoint receipts are immutable files under
`<attempt>/checkpoints/`; runtime payloads remain under
`<attempt>/model-checkpoints/`. If a process is lost before it writes a
terminal, `resume` may close the orphan only after nonblocking acquisition of
the lifetime lock and a second complete attempt-chain verification while that
lock remains held. It never infers death from PID identity or heartbeat age.

An ordinary runtime or checkpoint-observer exception writes `failure.json` and
a `HOST_EXECUTION_FAILED` terminal, then returns a `FAILED BackendExecution`
whose artifacts are those actual files. `BaseException` paths attempt the same
terminalization but are re-raised. If the failure or terminal cannot be
persisted (for example, disk failure), the host propagates the error and does
not manufacture a backend artifact.

`resume` accepts base and residual attempts. The predecessor checkpoint
must be under the same attempt root; the harness verifies its terminal and
checkpoint receipt, creates a distinct resume-linked attempt with
`AttemptStore.create_resumed`, and passes that exact `ResumeRecord` to the
training runtime. Residual resume also requires `--base-checkpoint` and
`--periodic-cache`; base resume rejects those arguments. The host verifies and
copies the latest and any older best checkpoint into the new attempt before
decoding owned authenticated bytes. Failed partial copies are retained, not
deleted by a path-based cleanup race. See [resume details](RESIDUAL_RESUME.md).

## Executable invocation

Install the package first. The
private-host module accepts its private host configuration before `--` and
passes everything after `--` through the public parser:

```bash
python -m phaseset_core.host --host-config private-run/host.json -- \
  preflight --config configs/phaseset/training.json

python -m phaseset_core.host --host-config private-run/host.json -- \
  run-base \
  --run-id phaseset-run-v1/BASE_QUALIFICATION/1729/B0 \
  --checkpoint-dir private-run/attempts/base-1729-b0-a/model-checkpoints \
  --config configs/phaseset/training.json

python -m phaseset_core.host --host-config private-run/host.json -- \
  run-residual \
  --run-id phaseset-run-v1/RESIDUAL_TRAIN/1729/08 \
  --base-checkpoint private-run/base/1729/selected-validation.pt \
  --periodic-cache private-run/cache/1729 \
  --checkpoint-dir private-run/attempts/residual-1729-08-a/model-checkpoints \
  --config configs/phaseset/training.json
```

A trusted Python composition root can equivalently call:

```python
from phaseset_core.host import main

raise SystemExit(main(PUBLIC_CLI_ARGUMENTS, config_path=PRIVATE_HOST_CONFIG))
```

The original private host config schema is `phaseset-private-host-v1`. It points to a
prepared index, an existing receipt record plus the exact artifact file for
each receipt digest, the installed source-tree digest, and runtime facts
(device, BF16 qualification decision, edge budget, checkpoint cadence, and
resume metadata).  Paths may be absolute or relative to the private config.
They are never serialized by the public CLI.

The backward-compatible `phaseset-private-host-v2` requires the
closed `base_qualification` configuration. Its `qualify-base` implementation
resolves the actual nine terminal histories, rescoring their selected
checkpoints on complete validation captures before an independent CUDA
latency observation and qualification assembly. See
[host base qualification](HOST_BASE_QUALIFICATION.md) for the exact command,
latency in-process wall-time bound, runtime admission, evidence and failure semantics. Existing
v1 configurations are not silently upgraded or assigned a qualification.

For residual runs, the optional `residual` object points to the one canonical
base-qualification artifact and one canonical capacity-audit artifact per
registered seed. The host process derives and verifies their SHA-256 values;
the operator does not transcribe individual hashes. `--periodic-cache` names a
seed directory containing canonical `record.json` (`PeriodicCacheRecord`) and
the digest-bound `cache-content.npz`. The payload must expose float64
`energy_floors[6]`. The host reads the payload once into a bounded buffer and
hashes and parses those same bytes. It verifies seed, winning-base run, base
terminal, qualification, and cache-content bindings before the public residual
factory is called.

The execution controller is responsible for creating the single private
receipt record from the user's already granted execution scope and the actual
rights/runtime/data measurements. This harness consumes and verifies that
record; it does not ask the user to sign each digest and does not create a
rights assertion when Embody approval is unknown.

## Deliberate gaps (no placeholder completion)

- Embody rights are not known. Until a real rights artifact participates in
  the receipt record, preflight/run-base/run-residual/resume remain held.
- The Embody-format conversion is outside this harness.  It must produce the
  exact prepared index/split/NPZ seam above and bind it to the authenticated
  prepared-data receipt.
- The main capture disk composition above is implemented. Producing its inputs
  from licensed captures and verified official holistic captions remains the
  upstream preparation responsibility. Auxiliary window positive families and
  repeating actor-set commitments never substitute for actual capture row IDs.
- `run-residual` is wired through the strict qualified-base loader, canonical
  periodic-cache record, energy floors, canonical all-system capacity audit,
  closed residual constructor, and the same attempt ledger. Residual resume
  reconstructs those same dependencies and passes the verified predecessor
  record to the unchanged runtime. Historical cache-file SHA provenance is not
  retroactively claimed when only energy floors were consumed.
- `prepare-data`, `audit-split`, `build-periodic-cache`,
  `evaluate`, `bootstrap`, and `render-paper` are not
  implemented here.  Calling one raises before a `BackendExecution` exists.
- The public `resume` parser has no stop-after-step option. It accepts B0/B1/B2
  and registered final system IDs; the host implements base and residual resume
  and recovers identity from the verified attempt ledger.
- The residual encoder and complete-capture evaluator now expose explicit
  reader-backed descriptor-cache APIs. The training loop now consumes explicit
  admitted plans, including cached checkpoint/resume. This host's residual
  construction/checkpoint/resume path does not yet inject those plans.
  The existing host cache argument authenticates the complete cache payload
  and consumes its energy floors, while its model path still recomputes
  descriptors. That historical behavior must not be represented as descriptor
  cache consumption in reports. See the
  [complete-capture cache plan](PERIODIC_CAPTURE_TRAINING_CACHE.md).
- A hard power loss before the first atomic runtime checkpoint has no resumable
  payload. The host refuses such a resume instead of manufacturing a checkpoint
  receipt. SIGKILL/power loss cannot write a terminal synchronously; terminal
  recovery occurs on the authorized resume operation.
- Runtime checkpoint fields do not expose separate sampler, dataloader, and
  dropout blobs.  The host receipt records the exact cursor/manifests and RNG
  payload under those ledger identities; a future public checkpoint-to-ledger
  helper would remove this local encoding convention.

`score_execution_census_sha256` is treated as the preregistered census of score
executions required by admission. It is never interpreted as a post-training
score table and therefore does not make base training depend on future scores.

## Observed tests and limits

On the Linux server, 12 host contract tests passed, including a genuine child
process lifetime-lock test and injected persistence failures. Four separate
tiny CPU fits passed checkpoint-observer integration: durable update/validation
callbacks, propagation of observer failure with a resumable checkpoint,
unchanged model/checkpoint/loss/validation values with an attached observer,
and a real checkpoint payload receipted by the attempt ledger. These use
`synthetic_contract=True` and do not assert production admission or results.

Report/failure JSON is file-fsynced and, on POSIX, parent-directory-fsynced
before a terminal is persisted. A failed fsync cannot create a success terminal.
Actual Embody rights, full model/runtime qualification, all registered runs,
and sealed test remain separate requirements.

The subsequent v2 prepared-source integration passed 94 focused server tests
and a separate complete suite of 964 tests, 2 skipped, and 39 subtests. These
include actual malformed numeric payloads and manifest/index replacement
cases; they do not assert a completed preparation command or a real-data run.

The complete-capture host composition subsequently passed 96 focused tests
and 1070 complete Linux server tests, with 2 skips and 39 subtests. A later
storage-portability correction passed 103 focused / 1077 complete tests,
again with 2 skips and 39 subtests, plus the unchanged official CLIP forward
and disk-restoration observation. Source comparisons passed throughout.
The new tests include canonical index bytes, real train/validation commitment
overlap, unchanged ambient RNG, and exact dispatch through base/residual/resume.
They use explicit analytic fixtures and do not establish data access or a
completed real training run.

The portability correction follows observed failures on both Windows CI
versions at the preceding capture-storage commit. On the inspected Windows
runtime, path stat and descriptor stat returned different `ctime` semantics
for one unchanged file. Cross-interface identity checks now compare device,
inode, size and modification time; each interface separately retains its full
before/after identity check including `ctime`. Content digests, bounded reads,
regular-file and symlink checks remain mandatory. Tests simulate the stable
cross-interface difference and reject drift within either interface. Windows
CI for the correction is a separate external acceptance check.

The later admission/intent correction passed 94 focused Linux server tests
and 1232 full tests, two existing skips and 39 subtests in 401.58 seconds.
Source censuses remained unchanged. New tests enter the real public CLI,
request-aware factory, production admission and backend with real typed
requests. They create actual attempt ledgers, then deliberately raise at the
typed base/residual model-runtime seam. Correct experiment bindings and
durable FAILED terminals must exist; no runtime checkpoint is manufactured.
Negative admissions with changed matrix, command census or handler digest
leave the snapshot unbound and still allow the subsequent canonical request.
These tests close the concrete missing-field/poisoning defects, not a complete
licensed-data CLI training run or the unimplemented commands listed above.

The host qualification integration passed 131 focused Linux server tests in
37.07 seconds with source unchanged. Tests enter the real CLI/factory/adapter/
backend path, validate typed holds and independently recheck completion
artifacts. A preceding run with seven fixture/API failures is retained; only
the tests changed for this rerun, not production code. These are software
fixtures, not nine trained bases, formal CUDA latency rows or a selected base.

The subsequent controller boundary correction passed 145 focused tests and was
included in a 270-test combined cached-training/cache-construction cohort.
Both observed runs kept the source unchanged. Early main-thread/POSIX timer
checks, cleanup preservation and qualify-base-only runtime HOLD classification
are implemented; Windows CI acceptance is a separate observation.

The [descriptor cache builder](PERIODIC_CACHE_EXECUTION.md) and
[cached residual lifecycle](CACHED_RESIDUAL_TRAINING.md) are callable core
interfaces, not yet an end-to-end cached CLI host. Typed build evidence,
private configuration, direct plan injection and ledger/resume linkage remain
explicit follow-up integration. The existing uncached host does not silently
select a cache based on files found on disk.
