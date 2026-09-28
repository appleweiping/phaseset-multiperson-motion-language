# Prepared capture-validation storage

`phaseset_core.capture_prepared_storage` persists an already admitted,
validation-only `CaptureValidationSource` without discarding windows or
re-encoding text. It preserves full chronological 200-frame body22 windows,
original actor padding, variable caption counts, and each original frozen-CLIP
feature batch and canonical encoding receipt.

```python
from phaseset_core.capture_prepared_storage import (
    load_capture_validation_source,
    write_capture_validation_source,
)

written = write_capture_validation_source(new_absolute_private_root, source)
restored = load_capture_validation_source(
    written.manifest_path,
    expected_manifest_sha256=written.manifest_sha256,
)
assert restored.census_sha256 == source.census_sha256
```

The caller must retain the expected manifest digest through its independently
authenticated preparation record. Reading a digest from the same untrusted
artifact is not authentication. The writer requires a new root; existing
artifacts are not overwritten.

## Descriptor source lineage in v2

`write_capture_validation_source_v2` writes the additive v2 schema. The same
loader accepts both versions. V1 remains the default writer and restores no
descriptor lineage; it rejects attempts to discard existing v2 lineage.

For each v2 window, the loader derives `CaptureDescriptorWindowSource` from
the exact consumed storage-manifest digest, the verified window NPZ digest,
and its canonical global ordinal. Global ordinals cover all windows in sorted
capture-commitment and capture-plan order. They are neither capture-local
positions nor source-frame starts.

`window.descriptor_context(seed=...)` produces that window's val/epoch-zero/
positive-zero-yaw context. The source rejects mixed v1/v2 lineage, multiple
storage manifests and any noncanonical ordinal census. The upstream scientific
manifest and complete source census retain their existing meanings; the
storage-manifest binding is separate to avoid a circular digest.

The server integration passed 92 focused and 1265 full tests, two existing
skips and 39 subtests, with source unchanged. These are storage/software
fixtures, not access approval, human annotation verification or a real cache.

## Existing storage and provenance guarantees

The manifest binds exact capture/window/caption counts, complete window plans,
the upstream manifest and source census, every numeric artifact, original text
receipts, caption lineage, output feature hashes and embedding cache keys.
Window arrays are deterministic uncompressed NPZ/NPY1.0, with no pickle or
object arrays. Text features and receipt JSON are stored separately.

The reader hashes and decodes the same bounded byte buffers. It checks closed
canonical JSON, file and directory census, regular non-symlink file identity,
stable opened-descriptor metadata, ZIP metadata and CRC, NPY headers and payload
lengths, exact dtypes/shapes/masks, and final reconstructed source census.
Header-based logical materialized-byte limits are enforced before NumPy loads
the arrays. Resource failures reject the complete source; they never sample
windows, people or captions. These limits are not a total-process-RSS bound.
Root ancestors must be trusted and stable; the loader does not claim
descriptor-relative protection against hostile writable ancestors.

Frozen text restoration calls only
`rehydrate_frozen_clip_text_batch(embeddings, receipt_json_bytes=...,
expected_receipt_sha256=...)` in the owning text module. That API verifies the
complete original canonical receipt, full/per-row feature bytes, fixed model
and tokenizer identity, historical encoding source/runtime manifests,
caption lineage, chunk ranges and cache key before reconstructing the owned
batch. It does not load CLIP or perform inference. Historical training/adapter
source identities remain historical; live adapter loading retains its own
strict current source pins.

This is private data storage, not an anonymization or data-release format.
Capture, actor, group, caption and component commitments can remain linkable.
The upstream controller must ensure component labels contain no identifying
text. Stored bodies, features, captions, operational paths and receipts do not
belong in the public repository.

Storage integrity does not establish data rights, participant-disjointness,
official holistic annotation identity, model quality, or permission to open
the sealed test split. This API accepts validation only. Cross-source training
and validation admission belongs to the host and split-audit layer; a pair of
individually valid sources must not be assumed disjoint.
