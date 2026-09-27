"""Complete development-parent disk views with model-RNG-independent yaw.

Private callers supply the admitted human records, exact preparation/cache
records and explicit caption yaw eligibility. This module discovers no data,
opens no final test, infers no semantic permission and holds no model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from .continuous_capture_io import load_prepared_continuous_capture
from .continuous_training_input import ContinuousTrainingInput, shared_yaw_capture
from .parent_retrieval_task import ParentCaptionRecord, ParentRetrievalTask
from .tmr_feature_cache import FrozenTMRRowCache, caption_key
from .training import OFFICIAL_SEEDS


@dataclass(frozen=True)
class ParentDiskRecord:
    parent: ParentCaptionRecord
    body_directory: Path
    prepared_record: dict
    physical_directory: Path | None = None
    physical_record: dict | None = None


def deterministic_parent_yaw(source_sha256: str, *, seed: int, epoch: int) -> float:
    """One group rotation per entire source/seed/epoch, no global RNG draws."""
    if type(seed) is not int or seed not in OFFICIAL_SEEDS:
        raise ValueError("augmentation requires an official fixed seed")
    if type(epoch) is not int or not 0 <= epoch < 2**64:
        raise ValueError("augmentation epoch must be a nonnegative integer")
    value = hashlib.sha256(
        b"phaseset-parent-shared-yaw-v1\0"
        + bytes.fromhex(source_sha256)
        + seed.to_bytes(8, "big")
        + epoch.to_bytes(8, "big")
    ).digest()
    # Explicit fixed 53-bit mapping avoids platform RNG/version dependence.
    fraction = (int.from_bytes(value[:8], "big") >> 11) / 2**53
    return 2 * math.pi * fraction - math.pi


class DevelopmentParentDiskSource:
    """Stream exact full timelines; never keep a dataset-sized RAM cache.

    TMR/WaMo and base ``capture`` views load body arrays only. Residual ``view``
    additionally loads the complete physical cache and the operator-admitted
    training-only floors, recomputing all physical features for nonzero yaw.
    No stale zero-yaw approximation, timeline truncation or sampled window
    fallback exists. Validation always uses the original unaugmented source.

    ``yaw_eligibility`` must cover every admitted family explicitly. Its
    human-text rule/provenance belongs in the private input manifest; a bool
    here cannot establish that world-direction captions permit a rotation.
    This is a source adapter, not the production lock/budget/pilot authority.
    """

    def __init__(
        self,
        task: ParentRetrievalTask,
        records: tuple[ParentDiskRecord, ...],
        *,
        yaw_eligibility: Mapping[str, bool],
        language_rows: FrozenTMRRowCache,
        energy_floors: np.ndarray | None = None,
    ):
        if type(task) is not ParentRetrievalTask or any(
            type(record) is not ParentDiskRecord for record in records
        ):
            raise TypeError("need the admitted complete task and exact disk records")
        self.records = {record.parent.annotation_family_sha256: record for record in records}
        families = {parent.annotation_family_sha256 for parent in task.parents}
        if len(self.records) != len(records) or set(self.records) != families:
            raise ValueError("disk source must cover the complete admitted task exactly once")
        self.yaw_eligibility = dict(yaw_eligibility)
        if set(self.yaw_eligibility) != families or any(
            type(value) is not bool for value in self.yaw_eligibility.values()
        ):
            raise ValueError("every human-caption family needs explicit boolean yaw eligibility")
        if type(language_rows) is not FrozenTMRRowCache:
            raise TypeError("need the closed frozen TMR/WaMo human feature rows")
        for parent in task.parents:
            record = self.records[parent.annotation_family_sha256]
            if parent.split == "test":
                raise ValueError("final test is not admitted by a development source")
            if record.parent != parent or (
                record.prepared_record["source_sha256"] != parent.source_sha256
                or record.prepared_record["augmentation_yaw"] != 0.0
            ):
                raise ValueError("disk preparation differs from the unaugmented parent task")
            if any(
                caption_key(caption) not in language_rows.records for caption in parent.captions
            ):
                raise ValueError("all official human rows must exist in the closed language cache")
            if energy_floors is not None and (
                record.physical_directory is None
                or record.physical_record is None
                or record.physical_record["prepared_record"] != record.prepared_record
            ):
                raise ValueError("phase views require complete matching physical records")
        self.energy_floors = None
        if energy_floors is not None:
            floors = np.array(energy_floors, dtype=np.float64, copy=True)
            if floors.shape != (6,) or not np.isfinite(floors).all() or (floors < 0).any():
                raise ValueError("physical floors need six finite nonnegative values")
            floors.setflags(write=False)
            self.energy_floors = floors
        self.language_rows = language_rows

    def _record_and_yaw(self, parent, *, seed, epoch, training):
        if type(parent) is not ParentCaptionRecord or (
            parent.annotation_family_sha256 not in self.records
        ):
            raise ValueError("parent is outside the admitted development task")
        record = self.records[parent.annotation_family_sha256]
        if record.parent != parent:
            raise ValueError("parent human/source lineage differs from the admitted task")
        if type(training) is not bool:
            raise ValueError("training view selection must be explicit")
        angle = deterministic_parent_yaw(parent.source_sha256, seed=seed, epoch=epoch)
        if not training or not self.yaw_eligibility[parent.annotation_family_sha256]:
            angle = 0.0
        return record, angle

    def capture(self, parent, *, seed, epoch, training):
        record, angle = self._record_and_yaw(parent, seed=seed, epoch=epoch, training=training)
        capture = load_prepared_continuous_capture(
            record.body_directory, expected_metadata=record.prepared_record
        )
        return shared_yaw_capture(capture, yaw_delta=angle)

    def view(self, parent, *, seed, epoch, training):
        if self.energy_floors is None:
            return self.capture(parent, seed=seed, epoch=epoch, training=training)
        record, angle = self._record_and_yaw(parent, seed=seed, epoch=epoch, training=training)
        complete = ContinuousTrainingInput.load(
            record.body_directory,
            record.physical_directory,
            cache_record=record.physical_record,
            energy_floors=self.energy_floors,
        )
        return complete.view(
            yaw_delta=angle,
            allow_shared_yaw=self.yaw_eligibility[parent.annotation_family_sha256],
        )

    def tmr_text(self, captions):
        return self.language_rows.tmr_text(captions)

    def wamo_cls(self, captions):
        return self.language_rows.wamo_cls(captions)
