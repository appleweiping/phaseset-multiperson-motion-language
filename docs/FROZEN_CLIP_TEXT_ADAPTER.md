# PhaseSet frozen CLIP text adapter

The frozen CPU feature path passed its registered Linux server observation on
2026-09-08. Fifteen adapter tests passed; the actual pretrained model completed
the two-forward golden repeat and one separate overlength forward. This is a
feature-path engineering result, not a dataset, training, or quality result.

Implementation: `src/phaseset_core/frozen_clip_text.py`; tests:
`tests/test_phaseset_frozen_clip_text.py`. The server probe imported the canonical
`phaseset_core.frozen_clip_text` module from a separately verified source copy.
Model files and private execution receipts are not distributed in this tree.

## Selection made for PhaseSet

PhaseSet consumes the official frozen pretrained projection from
`CLIPTextModelWithProjection`:

```text
caption
  -> fixed CLIP tokenizer (77 tokens)
  -> frozen CLIP text_model
  -> final-LayerNorm pooled EOS representation
  -> frozen official text_projection
  -> raw finite contiguous CPU float32 [Q,512]
```

The adapter does not L2-normalize this output. The existing retrieval systems
normalize embeddings at scoring time. It also does not add the legacy
PhasePair bridge-owned trainable `text.project` layer. That old layer belonged
to a different training design and is not needed to place the official CLIP
projection in PhaseSet's registered 512-dimensional space.

This choice does not eliminate PhaseSet's downstream `TextBandMLP` in residual
retrieval. `TextBandMLP` remains part of the trainable PhaseSet retrieval head,
where it converts an already-frozen global text embedding into band queries.
It is not part of the CLIP tower and is not run by this adapter.

The public `RetrievalTrainingBatch` boundary already enforces finite contiguous
CPU float32 `[Q,D]`, detaches/clones it, requires one commitment per row, and
uses commitments only for positive-family/lineage logic. This adapter tightens
the producer width to exactly 512 and calls the installed training embedding
validator before returning.

## API

```python
adapter = load_frozen_clip_text_adapter(
    verified_snapshot_path,
    max_batch_size=64,
)
batch = adapter.encode(
    ordered_caption_tuple,
    ordered_caption_commitment_tuple,
    batch_size=32,
)
text_embeddings = batch.embeddings
receipt_bytes = batch.receipt.canonical_json_bytes()
```

`Q` is variable and preserves input order. The caller must choose an explicit
batch size no larger than the load-time maximum; both are capped at 256.
Caption count, individual UTF-8 bytes, and total UTF-8 bytes are bounded.
Commitments must be unique exact `bytes[32]` rows. They are never passed to the
tokenizer/model, never influence embeddings, and appear only as hex lineage in
the path-free, text-free receipt.

Long text uses the single frozen `clip_native_truncate_77` policy already used
by the legacy loader. Each batch is first tokenized with special tokens and no
truncation solely to measure the complete token count. The formal tokenizer
call then uses `max_length=77`, `padding="max_length"`, and `truncation=True`;
an overlength row retains CLIP's native prefix and tokenizer EOS token 49407.
The adapter separately preserves the model configuration's pinned legacy
`eos_token_id=2` compatibility branch; those two identities are intentionally
not conflated. The receipt exposes the original and encoded token counts plus
`truncated=true`,
and binds the full caption UTF-8 digest and complete padded input-ID/attention-
mask tensor digests. It never prints raw text or token IDs, so truncation cannot
be silent even though the text itself remains private.

The formal token tensor is also compared element-for-element with a local
expectation constructed from the complete untruncated token row: short rows are
padded without changing their tokens, while long rows retain the first 76
native tokens and place tokenizer EOS 49407 at position 77. The attention mask
must exactly match that expected row. A count/EOS-only match is insufficient.

The authorized prepare-data host retains the complete caption. The 512-vector
represents only CLIP's native truncated-prefix semantics when `truncated=true`;
it is not claimed to preserve the full caption's meaning. Segmenting, LLM
summarization, or multi-window aggregation would be a different scientific
contract and is not performed here.

Batch size and ordered chunk ranges are provenance. This implementation does
not promise bitwise equality across different CPU batch sizes, so a different
batch size produces a different frozen-cache key even when caption order is
unchanged. The cache key binds method, source, runtime, snapshot, batch/chunk
order, full-text digests, and tokenization digests, but excludes lineage
commitments and every model/system identifier downstream of CLIP. B0/B1/B2 and
all residual systems must reuse the same receipt-bound frozen embedding cache;
they must not run separate text conversions.

`FrozenClipTextBatch.embeddings` returns a fresh owned clone on every access so
a caller cannot mutate the receipt-bound stored value through the public
property. `RetrievalTrainingBatch` takes another snapshot at its boundary.

## Snapshot and runtime trust

The only accepted snapshot is `openai/clip-vit-base-patch32` revision
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. The module contains literal name,
size, and SHA-256 pins for its eight official files. It requires an absolute,
resolved, non-reparse directory containing exactly those eight regular,
single-link files—no subdirectories, metadata, cache files, aliases, or extras.

All eight files are hashed before and after model loading. They are hashed
again before and after every complete `encode` call. This is a bounded
best-effort stability check, not an operating-system atomic snapshot.

The loader is exact and text-only:

- `CLIPTokenizerFast.from_pretrained(local_path, local_files_only=True,
  trust_remote_code=False)`;
- `CLIPTextModelWithProjection.from_pretrained` with the same offline flags,
  `low_cpu_mem_usage=False`, and loading information returned;
- a temporary empty cache plus offline environment and blocked common network
  connection entry points;
- exact loading-info census: no missing/mismatched/error rows and only the
  expected vision/logit keys discarded from the full checkpoint;
- exact model config, 197 live text/projection parameters, one buffer, no vision
  module, all parameters finite CPU float32 and permanently `requires_grad=False`;
- `eval()` plus `torch.inference_mode()` for every forward;
- live parameter/buffer content manifest checked before and after each encode;
- caller Torch RNG preserved across load and every success or failure of
  encode; any RNG advance during encode is a closed hold after the caller state
  is restored.

The implementation is deliberately pinned to the newly observed CPU identity:
CPython 3.12.12, Linux x86-64 little-endian, Torch 2.12.0+cu126 with CUDA hidden,
Transformers 4.57.3, tokenizers 0.22.2, huggingface-hub 0.36.2, and NumPy 2.4.6.
It requires one intra-op/inter-op thread, deterministic algorithms, and highest
float32 matmul precision, and native MHA fastpath disabled. The caller must
install that false flag; the adapter checks and records it without changing it.
A different runtime is a new qualification lane, not
a permissive fallback.

The exact Transformers model/tokenizer sources and installed
`phaseset_core.training` source are pinned. The adapter's own source is
hashed before/after load and encode and recorded in every receipt. The receipt
cross-binds runtime, source, snapshot, ordered caption text digests, lineage
commitments, per-row embedding digests, and the full output digest without
including captions or filesystem paths.

## Prepare-data integration boundary

The eventual prepare-data host, not this module, owns:

1. authorized caption access and deterministic caption selection;
2. split and positive-family assignment;
3. caption commitments and manifest provenance;
4. secure storage of raw text and frozen arrays;
5. checking that the receipt and array digest still agree before constructing
   `RetrievalTrainingBatch`.

The adapter accepts captions already selected and authorized by the host. It
cannot discover data, read caption files, infer splits, issue commitments,
authorize training, or convert a receipt into authority. Its receipt always
contains `authority=0`, `production=false`, and `training_authorized=false`.
Successful feature preparation is not a scientific result or a data-rights
witness.

## Tested adapter coverage

`test_phaseset_frozen_clip_text.py` uses a deterministic fake tokenizer/model
with real Torch tensors; it never imports Transformers or loads CLIP. It covers:

- variable ordered `Q`, explicit chunking, exact `[Q,512]` output;
- native truncation above 77 tokens with original/encoded counts, a visible
  truncation flag, exact retained-prefix tokens/mask, terminal EOS, and no raw
  caption in the receipt;
- commitments affecting lineage/receipt but never embedding values;
- input-order preservation;
- text/path-free authority-zero receipts;
- literal offline loader arguments and frozen eval state;
- extra snapshot files and post-load snapshot mutation;
- non-finite forward output and live parameter mutation;
- exact bounded integer batch size;
- cache identity changing with batch/chunk order but not lineage commitments;
- returned tensor ownership.
- formal-token body mutation before forward and model RNG advance with caller-
  state restoration.

Additional coverage can include mutation tests for missing files, aliases/hardlinks,
source/runtime drift, tokenizer shape/key drift, loading-info/config drift,
caption UTF-8/size/count limits, duplicate commitments, output dtype/device/
layout drift, network attempts, warning emission, and receipt canonicalization.

## Observed server qualification

The fifteen adapter tests passed in 11.45 seconds. The separate real-model
observation completed with exit zero and empty stderr; source hashes were equal
before and after execution. The two original public fixture rows produced
bitwise-equal embeddings and receipts across repeated forwards, with the same
previously observed combined output SHA-256:
`4a2ab93dab393a38bd0e78ee737fa31813c8e35ef3fcb5f4f3108705a502f58e`.

A third forward used a separate public overlength fixture. The original 226
tokens became exactly 77, with the complete expected prefix, attention mask,
and terminal tokenizer EOS 49407 checked before forwarding. Its observed output
SHA-256 was
`3f89662e5f6c06d4c7f9c1e2e77a78ad73dbe4da0a7be4a56a35df27060ebcba`.
That value is an observation, not a self-created acceptance golden. Caller RNG
state was preserved across all three calls. Independent same-family code review
remains provisional; these actual tests close only the measured feature-path
checks. Prepare-data integration, data rights, and formal experiments are separate.

The full Linux regression with this adapter then passed 932 tests, one skipped,
and 39 subtests in 287.60 seconds, with unchanged source bytes. The model weights
are not required for that unit/regression suite; actual loading is verified by
the separate private snapshot-bound observation above.

After the [frozen MHA policy](FROZEN_MHA_RUNTIME.md) changed the pinned training
source and runtime identity, a separate server requalification passed 33
focused tests and 949 full-suite tests, 2 skipped, and 39 subtests. Three actual
official CLIP forwards preserved both output values above and caller RNG.
The new receipt contains the new source/runtime identity and false MHA flag;
it is intentionally not byte-identical to the historical receipt. These checks
do not qualify CUDA text, full training, or real-data experiments.

The later capture-training integration changed the exact training source to
157160 bytes, SHA-256
`9f6c21e354a17facda89a542d53e39013477c5e1b1a259781e2446cd394f6d06`.
After its 1012-test complete server regression passed, another three official
CLIP forwards passed. Both output digests above, repeated embedding equality,
and caller RNG were unchanged. The adapter receipt correctly records the new
training and adapter source identities; no golden was regenerated. This is a
source-bound CPU feature-path requalification, not a real-data training result.

## Receipt-bound disk restoration

`rehydrate_frozen_clip_text_batch` restores an already encoded batch from its
owned CPU float32 features and complete original canonical JSON receipt. The
caller supplies the independently retained expected receipt digest. Closed
field/type validation checks all official file pins, historical source/runtime
manifests, caption lineage, chunk ranges, cache identity, per-row feature hashes
and the complete feature hash. It does not load CLIP or run inference. A
historical receipt is not rewritten to claim today's encoding source.

The first separate real-model restoration probe found a snapshot-order bug:
the live encoder records files in filename order, whereas restoration compared
against pin declaration order. Unit fixtures had repeated that wrong ordering.
The failed attempt is retained. Restoration and fixtures were corrected to the
encoder's actual ordering; a negative case now rejects declaration-order rows.
No official pin, live encoding behavior or output golden was changed.

The corrected integration passed110 focused tests and1043 complete tests,
2 existing skips and39 subtests. Three subsequent actual official forwards
preserved both original output digests above. Two separately persisted cache
batches then restored exact features, lineage, complete receipts and cache
keys, with no extra forward and unchanged observed CPU Torch RNG state.
Source comparisons passed, and all44 receipt members were independently
rehashed after transfer. The actual cache-probe private receipt commitment is
`b8dfe62f3773f81ff117d31da703cf60130eaf0b6890407bc203cdff35bbd2ac`.
Private cache payloads and raw operational receipts are not public assets.

See [prepared capture storage](CAPTURE_PREPARED_STORAGE.md) for the bounded
on-disk consumer. Neither restoration nor a matching receipt proves caption
rights, official annotation identity, participant-disjointness or model quality.
