# Complete native parent CPU profile through the production journal

One real, bounded CPU invocation completed on 2026-09-27, 16:54:29–17:03:37 UTC.
It joined the already qualified [complete-parent backward interface](PHASESET_V2_PARENT_TRAINING.md),
development disk source, original human CLIP rows and
[StudyProcess](PHASESET_V2_STUDY_PROCESS.md) to the existing production budget.
This is **native CPU seam qualification**, not a trained result or GPU profile.

## Actual input and scope

Two native training parents were selected in immutable byte-key order before
model scores, without timeline cropping or window sampling. They contain four
actors each, 5,800 and 6,000 frames, all 26 and 29 accepted windows, and all five
official human holistic rows each. The other development parents remained in
the admitted task metadata; final-test inputs were absent. Main-training-only
physical floors and the already closed body/physical/language caches were used.

Full-width 512-D B2 and V2 performed a complete rectangular 2-motion × 10-text
negative-gallery cache followed by RNG replay and vector–Jacobian backward.
Each system loaded each entire parent once for caching and again for replay.
The untrained seeded B2 state was reused; it was frozen for the V2 head.
There was no optimizer, weight update, CF pair or training-yaw admission.

| System | Complete cache + VJP seconds | Trainable gradient tensors |
|---|---:|---:|
| B2 | 264.5908735031262 | 88 |
| V2 | 266.6414572298527 | 55 |

Every trainable gradient was present and finite; at least one gradient element
per system was nonzero. This does **not** assert every individual parameter
tensor had a nonzero element. All model state bytes remained unchanged and the
V2 frozen base received no gradients. The recorded peak process RSS was
4,113,476 KiB. These single complete-batch CPU timings are not pure forward
latencies, p50/p95, GPU throughput, a 128-parent batch cost, inference ratio or
a prediction that all stages fit the 300-GPU-hour envelope.

## Production lifecycle and costs

FP32, seed 1729, CPU/interop one thread, no visible CUDA, offline warm environment,
nice 10 and a 16-GiB address-space bound were fixed. The child had a 1,800-second
watchdog and the production reservation included its full cleanup envelope.
It exited normally: all three wrapper/numerical/source-check exits were zero,
both stderr files empty, and source/invocation bytes unchanged. The controller
verified its own process group had exited; no timeout or forced kill occurred.

The existing production journal appended one real `RESERVE` and `SETTLE`,
without rebootstrap or alteration of its original historical import. Actual
controller wall time was 543.8673342310358 seconds, GPU cost zero, optimizer
cursor zero, and no pending/unknown-cursor attempt remained. Historical cost
stayed 34 conservatively accounted GPU seconds. Pilot and formal counts stayed
**0/12 and 0/87**; dated formal step limits remain unfrozen.

The private metadata-only archive was independently downloaded and verified:
28 hashed files plus the receipt itself, 29 flat members total. Raw inputs,
caption text, arrays, model assets and checkpoints were not in that archive or
the public repository.

- Receipt SHA-256: `d243b5a8547f97636d04d0ade3b521aa3b16573ee0ba824381d5bed766271a0b`.
- Archive SHA-256: `43df64c735d5dca3fbdd3bac8a8201ea13c17937ba410cde5f03174ab7808e5a`.

Fresh GPT-5.6-Sol/xhigh review of the three new private callers found no blocking
or non-blocking issues; same-family semantic review remains provisional. The
actual server evidence closes only the scope above. Real GPU profile, admitted
training transforms, bounded pilot, 87-stage learning, sealed evaluation,
statistics and paper delivery still require their own actual execution.
The later [label-source amendment](PHASESET_V2_RELATION_SUPERVISION_AMENDMENT.md)
cancels waiting for new annotators, but does not fabricate human CF evidence.
