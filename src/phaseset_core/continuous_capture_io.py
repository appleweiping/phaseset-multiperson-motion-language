"""Read the private body22 artifacts produced by continuous preparation.

This is the V2 full-timeline storage seam, not the legacy window store.
The frozen execution manifest remains responsible for dataset/split admission.
No caption, model asset or participant name is needed to load these arrays.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .continuous_capture import PreparedContinuousCapture
from .preprocessing import WindowDecision


PREPARED_ARRAY_NAMES = ("skeletons.npy", "track_mask.npy", "actor_commitments.npy")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prepared_continuous_capture(
    directory: str | Path,
    *,
    expected_metadata: dict[str, Any],
) -> PreparedContinuousCapture:
    """Verify the admitted preparation record and load read-only body maps.

    ``expected_metadata`` is the completed record in the producer's private
    manifest. Only the three fixed array names are read. Full array checks
    and timeline/canonical lineage validation are intentionally retained.
    """
    root = Path(directory)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata != expected_metadata:
        raise ValueError("continuous prepared metadata differs from admitted manifest")
    digests = metadata["artifact_sha256s"]
    if set(digests) != set(PREPARED_ARRAY_NAMES):
        raise ValueError("continuous prepared record requires the three body22 arrays")
    for name in PREPARED_ARRAY_NAMES:
        if _file_sha256(root / name) != digests[name]:
            raise ValueError(f"continuous prepared artifact differs: {name}")
    skeletons = np.asarray(np.load(root / "skeletons.npy", allow_pickle=False, mmap_mode="r"))
    tracking = np.asarray(np.load(root / "track_mask.npy", allow_pickle=False, mmap_mode="r"))
    commitments = np.load(root / "actor_commitments.npy", allow_pickle=False)
    if (
        commitments.dtype != np.dtype(np.uint8)
        or commitments.shape != (skeletons.shape[0], 32)
        or list(skeletons.shape) != metadata["shape"]
    ):
        raise ValueError("continuous prepared array headers differ from manifest")
    return PreparedContinuousCapture(
        skeletons=skeletons,
        track_mask=tracking,
        actor_commitments=tuple(value.tobytes() for value in commitments),
        source_sha256=metadata["source_sha256"],
        group_commitment=bytes.fromhex(metadata["group_commitment"]),
        decisions=tuple(WindowDecision(**value) for value in metadata["window_decisions"]),
        trailing_source_frames=metadata["trailing_source_frames"],
        reference_frame=metadata["reference_frame_20fps"],
        group_center=np.array(metadata["group_center"], dtype=np.float64),
        augmentation_yaw=metadata["augmentation_yaw"],
    )
