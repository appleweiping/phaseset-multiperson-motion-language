# Complete-parent retrieval labels

`phaseset_core.parent_retrieval_task` is a provenance/label index, not a data
license, physical admission gate, final task seal, or trained model. It reads
no paths and opens no test data on its own.

## Unit of retrieval

Released episodes may share one native parent and one human holistic annotation
file. They must first be placed on the actual parent world-coordinate timeline,
including unreleased gaps, and preprocessed once. They are not independent
motion rows or negative examples of each other. The host separately admits
the complete body/cache, declared tail, source lineage, and population-specific
training-only physical floors.

The development annotation audit found 494 eligible released episodes belonging
to 253 native parent/holistic-file families: 202 training and 51 validation
parents. Reading each official `scene_explained` list once yields 1,334 human
answer rows. An exact UTF-8 text audit found no repeated rows within or across
these families. This is not semantic-equivalence or relation-truth verification;
test annotations were not read by that diagnostic.

## Interface

The private host supplies `ParentCaptionRecord.from_official_bytes(...)` with
the complete-parent source identity, admitted annotation-family digest,
component, split, original annotation bytes and expected native human row
count. Every `scene_explained` row is retained verbatim. Other questionnaire
fields are not fused into motion captions. A digest cannot by itself prove that
an annotation is official, licensed or skeleton-observable.

`ParentRetrievalTask(records, expected_family_keys=...)` verifies that the
explicit complete population is present before selecting any subset. Repeated
source/family records, missing or extra families, and a participant component
crossing splits are errors. It does not silently deduplicate released siblings
or infer which parent should be retained.

`gallery(split=...)` returns one motion row per admitted parent and every human
caption row for that split. `batch(family_keys)` preserves the supplied parent
order and rejects repeated or unadmitted parents. Positive keys use the parent
annotation family; source keys remain separate. The resulting rectangular
variable-positive mask has no fixed three-caption assumption. Text commitments
follow the existing source/family/row-ordinal CLIP receipt convention and are
lineage only, never tokenizer or neural features.

Sampling schedules and full-population physical admission remain the training
host's responsibility. All bases, literature controls and V2 variants must use
the same admitted parent task. Capture-cluster statistics use independent
parents, not released episodes, windows or individual captions as replicates.

## Evidence boundary

Data-free metadata tests are not human truth. A two-real-parent frozen-random-B2
forward/loss/backward witness, if passed, establishes execution and label
plumbing only; it is not an optimizer run, pilot, formal stage or retrieval
result. Two real blinded human annotators are still required for verified
relation counterfactuals. Complete-population floors, task freeze, faithful
literature baselines, the fixed 87-stage matrix and sealed evaluation remain
separate gates. No final-test discovery or unsealing is performed by this API.
