# Fixed TMR/WaMo human-language features

`phaseset_core.tmr_feature_cache` stores actual frozen TMR language outputs
for private complete-parent development data. It is not an encoder, a new
annotation, a label filter, or evidence that arbitrary input files are official.
The private preparation caller uses the admitted pinned DistilBERT and MPNet
snapshots through `FrozenTMRLanguage`, without training or content truncation.
WaMo reuses the actual DistilBERT CLS token, not a TMR projection or MPNet/CLIP
substitute. MIME's own genuinely trainable CLIP is **not** cached here.

Only contiguous masked right padding is removed. Every content token and
special token is retained. Reload dynamically restores exact positive-zero
padding and masks; valid tokens, CLS and sentence features match the stored
preparation outputs exactly. The preparation batch shape and frozen runtime
are part of that feature snapshot; this is not a universal bitwise claim
across different Transformer batching layouts or hardware.

Exact duplicate UTF-8 text can reuse a fixed feature file. Annotation
occurrences, caption commitments, complete-parent sources, families and all
variable positive rows remain distinct in `ParentRetrievalTask`. The cache
never samples captions or changes their whitespace, wording or order.

Files are written once into a fresh private directory. Loading verifies
closed manifest entries, complete array headers, finite FP32 values, token
lengths and file identity. Unknown rows or changed artifacts fail explicitly;
they do not trigger a substitute model, online download or regeneration.
The private caller additionally verifies the admitted development population,
official human-file lineage, exact feature reload, unchanged model weights
and RNG, and source/asset identity before and after preparation. The final
test population is excluded, and no learned training statistic is fitted.

Analytic tests cover ordering, repeated occurrences, masks, padding, CLS,
file drift, unknown rows, and TMR text distribution/gradient preservation.
They are not proof of pretrained assets. That requires the separate actual
private generation receipt. Neither a cache, these tests, nor a same-family
review is optimization, a pilot, convergence, a hardware profile, a retrieval
result or completion of the research project. No data or model-derived
feature files belong in this public repository.

See the [literature optimizer host](PHASESET_V2_LITERATURE_PARENT_HOST.md)
and [TMR adaptation](TMR_SET_ADAPTATION.md).
