# Frozen human CLIP rows for complete-parent training

`ParentHumanClipRows` takes one original sealed CLIP preparation batch and the
complete caller-admitted development `ParentRetrievalTask`. All official human
occurrences must be present, in canonical task order, with their exact UTF-8
text digest and source/family/ordinal commitment. Equal sentence strings are
not deduplicated. No motion-derived information or fitted learning statistics
enter these frozen text features. Final-test parents are rejected.

Prepare with the existing pinned offline CPU-only CLIP adapter, save its
original receipt and raw projected FP32 embeddings, and authenticate them with
the existing original-batch rehydrator. Select rows in the actual parent batch
order using `select_frozen_clip_text_rows`. This does not construct a CLIP tower
in the training process or relax the preparation runtime contract.

Each selection has the distinct `phaseset-frozen-clip-row-selection-v1` schema
and `ROW_SELECTION_AUTHORITY0` status. It records original row indices, the
original receipt identity and unchanged token/embedding/commitment provenance.
It is explicitly a cached-row operation, not a new encoder invocation. Original
runtime/model/source identities describe preparation, not the selection host.
No caption or commitment can be supplied anew to the selection factory.
Nested selection and duplicate indices are rejected. Persist the original
batch; the original rehydrator deliberately rejects derived selection receipts.

Only the continuous base/V2 scoring and parent-training validators opt into
this derived schema. Legacy prepared-capture validation/storage admission is
unchanged. Returned embeddings remain snapshot-owned contiguous frozen CPU
FP32; the scoring module moves these numbers to its model device as usual.
This promises exact reuse of the stored preparation rows, not universally
bitwise-identical Transformer outputs under different inference batch shapes
or hardware.

The existing CLIP-native 77-token truncation policy is retained, with each
original token count and truncation flag preserved. Actual private preparation
reports how many captions exceed it. This differs openly from the untruncated
[TMR/WaMo language rows](PHASESET_V2_FROZEN_LANGUAGE_ROWS.md). MIME still trains
its own genuine CLIP tower; this cache is not its text training path.

`DevelopmentParentDiskSource(..., human_clip_rows=...)` supplies its human-only
`text(labels, training=...)` interface with empty counterfactual indices. Empty
CF rows mean no evidence: neither a negative truth label nor permission to run
the formal V2 CF objective without the two real blind human verifications.
The full human-verified CF pool, semantic yaw policy, run-specific floors,
production locks, heartbeat, budget, hardware profile and study freeze remain
separate inputs and obligations. No pilot or formal stage is authorized here.

Analytic tests cover ordered exact reuse, ownership, RNG preservation, altered
ancestry/text/task rejection, original-schema exclusion and base/V2 optimizer,
full validation and bitwise interruption/resume. Actual native preparation must
separately verify every admitted human occurrence and all three seed orderings.
Neither software tests nor feature preparation are native learning results.

See the [development disk source](PHASESET_V2_PARENT_DISK_SOURCE.md) and
[complete-parent training host](PHASESET_V2_PARENT_HOST.md).
