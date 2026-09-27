from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

import numpy as np
import pytest

from phaseset_core.continuous_capture import prepare_continuous_capture
from phaseset_core.continuous_capture_io import load_prepared_continuous_capture
from phaseset_core.pipeline import PrivateCaptureArrays


def _artifacts(root):
    joints = np.zeros((600, 3, 22, 3), dtype=np.float32)
    joints[..., 0] = np.arange(600, dtype=np.float32)[:, None, None] / 30
    prepared = prepare_continuous_capture(
        PrivateCaptureArrays(
            joints,
            np.ones(joints.shape[:-1], dtype=np.bool_),
            tuple(bytes([i + 1]) * 32 for i in range(3)),
            "a" * 64,
        )
    )
    arrays = {
        "skeletons.npy": prepared.skeletons,
        "track_mask.npy": prepared.track_mask,
        "actor_commitments.npy": np.array(
            [list(value) for value in prepared.actor_commitments], dtype=np.uint8
        ),
    }
    digests = {}
    for name, array in arrays.items():
        np.save(root / name, array, allow_pickle=False)
        digests[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    metadata = {
        "shape": list(prepared.skeletons.shape),
        "source_sha256": prepared.source_sha256,
        "group_commitment": prepared.group_commitment.hex(),
        "artifact_sha256s": digests,
        "window_decisions": [asdict(value) for value in prepared.decisions],
        "trailing_source_frames": prepared.trailing_source_frames,
        "reference_frame_20fps": prepared.reference_frame,
        "group_center": prepared.group_center.tolist(),
        "augmentation_yaw": prepared.augmentation_yaw,
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return prepared, metadata


def test_completed_body_artifacts_restore_exact_full_timeline(tmp_path):
    reference, metadata = _artifacts(tmp_path)
    restored = load_prepared_continuous_capture(tmp_path, expected_metadata=metadata)
    np.testing.assert_array_equal(
        reference.skeletons.view(np.uint32), restored.skeletons.view(np.uint32)
    )
    np.testing.assert_array_equal(reference.track_mask, restored.track_mask)
    assert reference.actor_commitments == restored.actor_commitments
    assert reference.decisions == restored.decisions
    assert restored.frame_count == 400 and len(restored.windows()) == 2
    assert not restored.skeletons.flags.writeable and not restored.track_mask.flags.writeable
    assert reference.source_sha256 == restored.source_sha256


def test_prepared_artifact_drift_is_rejected(tmp_path):
    _, metadata = _artifacts(tmp_path)
    with (tmp_path / "track_mask.npy").open("ab") as stream:
        stream.write(b"drift")
    with pytest.raises(ValueError, match="artifact differs"):
        load_prepared_continuous_capture(tmp_path, expected_metadata=metadata)


def test_prepared_metadata_must_match_admitted_record(tmp_path):
    _, metadata = _artifacts(tmp_path)
    altered = {**metadata, "augmentation_yaw": 1.0}
    with pytest.raises(ValueError, match="metadata differs"):
        load_prepared_continuous_capture(tmp_path, expected_metadata=altered)


def test_unexpected_array_name_is_not_followed(tmp_path):
    _, metadata = _artifacts(tmp_path)
    metadata["artifact_sha256s"]["unexpected.npy"] = "a" * 64
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="three body22 arrays"):
        load_prepared_continuous_capture(tmp_path, expected_metadata=metadata)
