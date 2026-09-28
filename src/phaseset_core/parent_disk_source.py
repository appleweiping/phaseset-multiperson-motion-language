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
from .continuous_training_input import (
    ContinuousTrainingInput,
    ContinuousTrainingView,
    shared_yaw_capture,
)
from .dct_calibration import DctFloorReceipt
from .dct_relations import DctViewContext, require_dct_working_budget
from .legacy_scalar_calibration import LegacyScalarFloorReceipt
from .directional_phase import LocalPhaseConfig, VELOCITY_MODES
from .speed_calibration import SpeedFloorReceipt
from .parent_retrieval_task import ParentCaptionRecord, ParentRetrievalTask
from .parent_clip_rows import ParentHumanClipRows
from .parent_weak_clip_rows import ParentWeakClipRows
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

    TMR/WaMo, base and typed legacy ``capture`` views load body arrays only. V2 ``view``
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
        dct_floor_receipt: DctFloorReceipt | None = None,
        human_clip_rows: ParentHumanClipRows | None = None,
        weak_clip_rows: ParentWeakClipRows | None = None,
        velocity_mode: str = "signed_vector",
        speed_floor_receipt: SpeedFloorReceipt | None = None,
        legacy_floor_receipt: LegacyScalarFloorReceipt | None = None,
        legacy_training_source_manifest_sha256: str | None = None,
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
        if energy_floors is not None and dct_floor_receipt is not None:
            raise ValueError("phase and A6 floors must not be admitted together")
        if legacy_floor_receipt is not None:
            if (
                type(legacy_floor_receipt) is not LegacyScalarFloorReceipt
                or energy_floors is not None
                or dct_floor_receipt is not None
                or speed_floor_receipt is not None
                or legacy_training_source_manifest_sha256
                != legacy_floor_receipt.training_source_manifest_sha256
            ):
                raise ValueError("legacy capture-only source requires its typed scalar receipt alone")
        elif legacy_training_source_manifest_sha256 is not None:
            raise ValueError("legacy training manifest cannot enter a nonlegacy source")
        self.legacy_floor_receipt = legacy_floor_receipt
        self.legacy_training_source_manifest_sha256 = legacy_training_source_manifest_sha256
        if velocity_mode not in VELOCITY_MODES or (
            velocity_mode == "speed_only" and energy_floors is None
        ):
            raise ValueError("speed-only source requires an independent phase cache and floors")
        if velocity_mode == "speed_only":
            if type(speed_floor_receipt) is not SpeedFloorReceipt:
                raise ValueError("A1 source requires an independent speed floor receipt")
            speed_floor_receipt.require_config(LocalPhaseConfig())
            if not np.array_equal(energy_floors, speed_floor_receipt.floors):
                raise ValueError("A1 source floors differ from its training receipt")
        elif speed_floor_receipt is not None:
            raise ValueError("speed floor receipt cannot enter a signed-vector source")
        self.velocity_mode = velocity_mode
        self.speed_floor_receipt = speed_floor_receipt
        if dct_floor_receipt is not None:
            if type(dct_floor_receipt) is not DctFloorReceipt:
                raise TypeError("A6 disk source requires a typed DCT floor receipt")
            dct_floor_receipt.require_config(LocalPhaseConfig())
        self.dct_floor_receipt = dct_floor_receipt
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
        if human_clip_rows is not None and (
            type(human_clip_rows) is not ParentHumanClipRows
            or human_clip_rows.parents
            != {parent.annotation_family_sha256: parent for parent in task.parents}
        ):
            raise ValueError("human CLIP rows must cover the same complete development task")
        self.human_clip_rows = human_clip_rows
        if weak_clip_rows is not None and (
            type(weak_clip_rows) is not ParentWeakClipRows
            or weak_clip_rows.human is not human_clip_rows
        ):
            raise ValueError("weak rows must use the same admitted human row provider")
        self.weak_clip_rows = weak_clip_rows

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
        if self.dct_floor_receipt is not None:
            record, _ = self._record_and_yaw(parent, seed=seed, epoch=epoch, training=training)
            shape = record.prepared_record.get("shape")
            if (
                type(shape) is not list
                or len(shape) != 4
                or any(type(value) is not int for value in shape)
                or shape[2:] != [22, 3]
            ):
                raise ValueError("A6 prepared shape is invalid")
            require_dct_working_budget(shape[0], shape[1], LocalPhaseConfig())
            capture = self.capture(parent, seed=seed, epoch=epoch, training=training)
            return ContinuousTrainingView(
                capture,
                DctViewContext.from_capture(capture, LocalPhaseConfig(), self.dct_floor_receipt),
                False,
            )
        if self.energy_floors is None:
            return self.capture(parent, seed=seed, epoch=epoch, training=training)
        record, angle = self._record_and_yaw(parent, seed=seed, epoch=epoch, training=training)
        complete = ContinuousTrainingInput.load(
            record.body_directory,
            record.physical_directory,
            cache_record=record.physical_record,
            energy_floors=self.energy_floors,
            expected_velocity_mode=self.velocity_mode,
        )
        if self.speed_floor_receipt is not None:
            self.speed_floor_receipt.require_config(complete.cached_field.config)
        return complete.view(
            yaw_delta=angle,
            allow_shared_yaw=self.yaw_eligibility[parent.annotation_family_sha256],
        )

    def tmr_text(self, captions):
        return self.language_rows.tmr_text(captions)

    def wamo_cls(self, captions):
        return self.language_rows.wamo_cls(captions)

    def text(self, labels, *, training):
        """Original human evaluation; separately admitted weak training pool."""
        if self.human_clip_rows is None:
            raise ValueError(
                "base/V2 human text requires the closed original CLIP preparation batch"
            )
        provider = self.weak_clip_rows if self.weak_clip_rows is not None else self.human_clip_rows
        return provider.text(labels, training=training)
