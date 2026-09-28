# Complete-parent literature optimizer host

`phaseset_core.literature_parent_host.LiteratureParentTrainingHost` uses the
same explicit Adam(W), warmup/cosine, gradient clipping, validation checkpoint selector,
immutable attempt and checkpoint/resume loop as the complete-parent base and
residual host. The original constructor still admits only its actual base/V2
scorers; literature models enter through their own strict constructor.
`ParentHostConfig.for_literature` supplies the disclosed method-specific
optimizer/decay defaults; those settings are exact resume inputs.

Actual MIME group training uses `MIMEParentTrainingHost`, not the generic
uniform-batch literature host. Its `_batches` hook invokes the qualified
training-only MIME curriculum at every epoch, including resume/census checks.
The caller prepares anchors once from all learning-parent human CLIP rows;
neighbor width and an owned snapshot of anchor values join the checkpoint manifest. The genuinely
trainable MIME CLIP tower remains separate from these frozen sampling anchors.

## Objectives and inputs

- TMR-Set: complete stochastic VAE objective, both full-track reconstructions,
  four KL terms, latent alignment and MPNet-filtered full-batch contrastive loss.
- WaMo-Set: both reconstructions, original/shuffled temporal classification,
  and the original **sum** of contrastive directions.
- MIME-Set: genuine trainable CLIP language tower and learned score scale;
  the original **average** of contrastive directions. Original role-aware
  Inter-X MIME is a distinct interface and is not admitted by this group host.

The host dispatches the already-qualified per-parent cache/VJP/replay seams,
not a common replacement loss. Every effective batch retains its entire
rectangular motion-by-human-caption negative gallery, variable positives,
and every accepted window at its absolute time, including windows after gaps.
Source reload augmentation must depend on source/seed/epoch, independently of
model dropout and VAE sampling RNG. It must reproduce the same arrays each time.

`LiteratureParentSource` supplies entire `PreparedContinuousCapture` objects,
frozen TMR token/sentence features, and frozen WaMo DistilBERT CLS features.
MIME uses its own actual trainable tokenizer/tower. Language asset identities,
source adapter, preprocessing and complete census belong in the operator's
private input/code bindings; no private download URLs or captions are supplied
by this public package.

## Validation and restart

All validation human rows are encoded once; motion parents are streamed one
at a time without auxiliary loss or augmentation. Scores follow the methods'
public `score` equations, with TMR distribution means at evaluation. The
selector remains capture-macro bidirectional R@1 against registered family
positives, deterministic lineage tie breaks, and first-checkpoint retention
on exact ties. Model mode and dropout/sampling RNG are restored afterwards.

The manifest additionally records the method and its loss configuration;
resume rejects changed objective coefficients as well as execution bindings,
initial state, task, scalar/module configuration and optimizer settings.
The shared checkpoint loop retains optimizer/scheduler/RNG state, source census,
cursor checks, predecessor evidence and terminal failures.

## Qualification is not research completion

The analytic tests exercise real optimizer updates, method-specific losses,
full validation endpoint, dropout/VAE bitwise interrupted resume, objective
drift refusal, and exclusion of held-out/test rows. They are expressly not
native optimization, a pilot, original-task convergence or a hardware profile.
The existing base/residual host and original model/seam tests must pass too;
the real Transformers CLIP component must execute, not skip.

Native default-model replay qualification separately uses complete real
development captures, all human rows, and each original dense objective as
the gradient oracle. It proves objective preservation, not baseline accuracy.

No constructor, software test or code review authorizes formal training.
The dated whole-parent training-unit amendment, hardware profile, bounded
pilot, study/budget freeze and rights remain private admission steps. No
final-test interface is provided here. Same-family semantic review remains
provisional; research metrics, statistics and paper results are still pending.

See the [WaMo/MIME replay contract](PHASESET_V2_LITERATURE_PARENT_TRAINING.md),
[TMR replay contract](PHASESET_V2_TMR_PARENT_TRAINING.md), and
[complete-parent host](PHASESET_V2_PARENT_HOST.md).
