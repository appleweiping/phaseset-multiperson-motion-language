# Private host base qualification

The private host controller composes the already defined base-cohort resolver,
holistic capture scorer, CUDA latency session, durable progress journal,
qualification assembler and production adapter. It does not change the public
selection rule and does not let a CLI caller submit scientific results.

## Request and input boundary

The public command remains `qualify-base --terminal-root`. The authenticated
host configuration supplies the closed attempt registry, output and journal
locations, registered device identity, frozen protocol/launcher identities and
latency in-process wall-time bound. Concrete host names, ports, device identifiers, private paths,
receipts and data stay outside the public tree.

The controller requires the exact registered nine-run history: each run keeps
failed predecessors and ends in its only successful terminal. The resolver
then verifies terminal-bound latest checkpoint bytes and the distinct selected
validation-best checkpoint where applicable. Missing or inconsistent rows stop
before scoring. A registry digest records consumed bytes; it is not proof of
permission or data provenance.

## Closed execution order

For a formal call, the controller performs this sequence:

1. Load the canonical registry and resolve all nine completed chains and their
   selected checkpoint payloads.
2. Before capture scoring, observe the registered CUDA device, require the
   registered minimum free memory, establish the process allocator cap and
   require an initially clean allocator.
3. Recompute the complete holistic validation gallery for every selected base
   checkpoint. Submitted scores and parameter counts are not accepted.
4. Release scorer objects, clear this process's cuBLAS workspaces, empty the
   allocator cache and require zero allocated and reserved bytes. The device
   identity and free-memory observation are checked again.
5. Run the complete cyclic latency session with its typed write-once observer.
   The returned session and durable journal terminal must agree.
6. Assemble the qualification with the existing selector, retain the exact
   qualification and dependency bytes, and pass them to the existing
   production adapter for independent recomputation and backend callback.

Scoring and latency share the same physical-device and precision binding, but
latency still performs its own admission. The scoring preflight does not weaken
the latency gate. Cleanup affects only allocations and workspaces owned by the
current process and does not stop other jobs.

## Failure and timeout behavior

CUDA identity, free-memory, allocator, workspace and OOM failures are retained
as HOLD outcomes. If scoring and cleanup both fail, the original scoring error
and its resource cause remain primary while the cleanup type is recorded
separately. A successful score cannot enter latency unless cleanup reaches the
exact-zero boundary.

The in-process interval timer is supported only where the required POSIX timer
API exists. The current controller checks that API, the Python main thread
and the absence of an inherited deadline before registry resolution or scoring,
then rechecks immediately before latency timer installation. Its cleanup always
attempts both disarm and prior-handler restoration, and a cleanup error cannot
replace the primary scoring or latency error. Timer capability or restoration
drift is a runtime HOLD, not a scientific failure.
Deployment must also provide its separately frozen hard timeout because a
process signal cannot guarantee that a blocked native call returns. A hard kill
may leave only a valid prefix of the progress journal and the start record;
that prefix is not a qualification.

An injected runtime exists only for software contract tests. Even a complete
synthetic journal from that seam returns a nonformal HOLD and cannot reach
qualification assembly or authentication.

## Verification status

The controller passed 131 focused server tests in 37.07 seconds with source
unchanged. Tests cover resource admission/release, failure preservation, timer
portability, the actual CLI-to-backend path and a structurally supplied
host-to-production-adapter callback. The first attempt's 123 passes and seven
fixture API failures in 35.21 seconds remain retained. A test-only successor
fixed journal bindings, CLI stream arguments, fake-CUDA test isolation and a
class-name assertion; production controller and host bytes were unchanged.
These software fixtures do not establish a real nine-run cohort, formal
81-visit CUDA latency session, qualified winner or data-dependent completion.

The combined source, including cached complete-capture validation, subsequently
passed 1334 tests, 2 existing skips and 39 subtests in 643.90 seconds on the
Linux server, with source unchanged. No formal cohort was produced by this
regression.

Follow-up code now implements the two reviewed runtime-boundary
corrections: the early and exception-safe timer behavior described above, and
qualify-base-only classification of post-admission source/output/lease runtime
failures as HOLD outcomes rather than CLI input errors. Other commands and the
v1 private-host schema retain their prior behavior. Fault-injection tests use a
typed portable timer facade so Windows does not need to emulate POSIX signals;
a separate test exercises the real POSIX handler and timer restoration.

The successor passed static checks, fresh static review, and 145 focused Linux
server tests in 64.24 seconds with source unchanged. Those code and test changes
are not included in the observed 1334-test source, and cross-platform CI
acceptance remains pending, so this paragraph does not claim the portability
defect is empirically closed. Formal qualification remains unexecuted; this
feature-branch milestone is not an empirical release.

The earlier K=32 B0/B1/B2 diagnostic verified the corrected per-process CUDA
release mechanics at registered width, but used untrained models, one warmup
and one observed forward per model. It is not the formal latency session or a
host qualification result. Earlier failed observations remain part of the
history.
