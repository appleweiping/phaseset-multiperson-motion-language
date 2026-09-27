# Complete-parent base and residual training host

`continuous_parent_host.py` connects the qualified full-gallery gradient replay
to real FP32 AdamW updates, checkpoint selection and restart. It is not the old
window-unit training recipe, a formal-study authorization, a trained B2 anchor,
or a literature-baseline implementation. A software qualification is not a
native learning result. The formal study still requires the common bounded
pilot, explicit parent-unit amendment, floors, hardware profile and 87-stage
freeze. The legacy human relationship audit cannot be supplied by this code;
the [user amendment](PHASESET_V2_RELATION_SUPERVISION_AMENDMENT.md) now permits
a separate weak-training branch, not replacement of independent motion truth.

## Inputs and updates

The private `ParentTrainingSource` loads one admitted **whole** capture at a
time from disk. Every accepted window remains in the frozen global branch;
every physical timeline interval remains in the ordered coordination branch.
The source must reproduce the same shared-yaw transform during replay, and
respect the pre-admitted language/yaw rule. No shorter timeline or physical
batch materialization is introduced by the host.

For a **base** stage, the source returns `PreparedContinuousCapture` directly,
not a fabricated residual view. `ContinuousBaseRetrievalSystem` reuses B0/B1/B2
without shrinking their registered architectures. Every accepted window is
encoded in absolute order, then pooled with the same fixed float64 mean and
float32 L2 normalization as the residual's global anchor. The base encoder and
its bounded logit temperature are trainable; no text adapter, phase cache,
physical floor, coordination evidence or learned window pool is added.
Non-reentrant window checkpointing preserves dropout RNG and binds each actual
window to its own replay closure. The checkpoint invocation includes an
explicit tensor on the model's device so
CUDA RNG is discoverable; closure-only parameters would not provide that.
CPU invocation checks do not qualify CUDA numerical gradients. It bounds
stored neural activations, not the
size of the complete physical skeleton input. Native memory/cost qualification
is still required. The outer full-gallery parent-score replay is shared by
base and residual stages; the architectures and input types are not conflated.

The effective batch is selected by explicit learning components and includes
all official human holistic rows of every parent. Only the B-by-Q score matrix
and RNG snapshots are cached. The full variable-positive symmetric objective
is differentiated once; each complete capture graph is replayed separately.
The physical loader is called separately for cache and replay. It must not
silently keep the entire learning population in memory. Loading and neural
activation storage are separate from the total disk-cache footprint.

Optional CF sentences follow the human retrieval prefix. Their columns have no
fabricated retrieval positive. A verified-false bit is not human provenance:
the private manifest must bind the actual blind two-human records. Unverified
CFs contribute no CF term. No label generation or data discovery occurs here.
`ParentWeakCounterfactualRows` instead uses `included_weak` and the separate
weak loss/count fields. Its source is disclosed machine-caption contradiction,
never human `verified_false`. The original retrieval prefix and all validation
rows stay human-only. Base stages reject CF rows, extra text columns and any CF weight. They optimize
only the existing variable-positive symmetric retrieval InfoNCE.

Optimizer: AdamW, weight decay 0.01, clipping 1.0, 5% warmup followed by cosine
decay, default residual learning rate 3e-4 and 20 complete epochs. Seeds remain
1729, 2718 and 31415. Batch size is explicitly **parents**, not 128 windows.
The short qualification schedules are not the formal/pilot freeze. There is
no automatic BF16 fallback; FP32 is the only supported path in this host.
`ParentHostConfig.for_base` explicitly sets the distinct base defaults:
30 epochs, learning rate 2e-4, zero CF loss and `stage="base"`. The stage must
match the actual scorer. A base has no frozen-anchor checkpoint binding;
residual stages still require that exact frozen B2 binding and enforce its
unchanged state. No checkpoint produced by a fixture is a qualified base.

## Validation and selection

Learning and checkpoint-selection components are explicit, disjoint and
non-test. Historical main-validation components can be learning components in
an authorized outer fold; held-out components are never opened by the host.
Validation loads every parent and every exact human row of its complete
gallery, with no CF extras or augmentation. Scores are computed against the
same full text pool, one whole motion at a time. Evaluation consumes no training
RNG and restores the previous model mode.

The reused endpoint is capture-macro bidirectional R@1: motion-to-text counts
any positive; text-to-motion is averaged first within each capture family,
then equally across captures. Ties are broken by fixed lineage commitments,
not input order. This is explicitly not caption-count-weighted R@1. Only a
strictly better validation endpoint replaces the selected checkpoint; exact
ties preserve the first. There is no final-test method or test-based selector.

## Checkpoints and attempts

Each `fit` owns a fresh attempt directory with an immutable run manifest,
numbered progress events, atomic write-once checkpoints and terminal evidence.
The private run manifest contains the full **active learning/selection**
parent/text census and must remain private. Unused held-out/test rows of a
larger supplied task are not serialized or made part of this host's resume
identity. It binds configuration, input/physical-floor/CLIP/CF
manifest, code including the source adapter, qualified runtime and frozen
base identity. This caller-provided binding is not a substitute for real
qualification or permission.

Checkpoints contain model, AdamW, scheduler, Python/NumPy/Torch RNG, complete
epoch/batch cursor, parents seen, validation history and selected artifact.
Resume verifies the exact predecessor digest, payload, same execution inputs,
cursor/census, frozen anchor and optimizer/scheduler progress. A resumed host
must be built from the same seed-bound **initial** model. Resume uses a new
attempt; it never overwrites or mutates the predecessor. Interruption at an
epoch's last update still completes that epoch's validation before returning.
Zero-update and already-completed resume are explicit cases.

These event/terminal records intentionally carry no formal-study authority.
The private production adapter remains responsible for resource admission,
locks, budget accounting, attempt-ledger linkage, signal/error classification,
long-operation heartbeat and publication-safe result export. Failed disk or
process writes cannot guarantee an on-disk terminal; the remote supervisor
must close that failure from its own evidence. Never publish private host
manifests, checkpoints, caption text or licensed physical cache files.

## Qualification scope

Tests exercise actual registered 512D B0/B1/B2 forward and all-parameter
checkpointed backward with active dropout, plus an independent per-window
pooling oracle. Separate, clearly marked analytic fixtures exercise the real
base/residual host optimizer loop,
fixtures, full gallery/caption census, first-best selection, bitwise
uninterrupted-versus-resumed model/optimizer/scheduler/RNG, zero/mid/end/completed
resume, concrete input drift, truncated checkpoints and wrong progress cursor.
They also retain the existing independent dense-versus-replay gradient tests.
None is evidence of real native learning, a paper metric, GPU cost, human CF
truth, or successful completion of any of the 87 formal stages.

A separate native CPU qualification has now replayed two complete four-person
captures through the actual 512D V2 score/loss backward with ten original human
rows and six disclosed machine weak negatives. All 55 trainable gradient
tensors were finite and weights stayed unchanged; no optimizer was created.
See the [weak-pool qualification scope](PHASESET_V2_WEAK_CLIP_POOL.md).
This closes that input/loss seam, not native learning or a hardware cost claim.
