"""Admitted whole-capture input views, with consistent shared-yaw augmentation.

Zero-yaw views reuse immutable disk responses. Nonzero yaw rotates the entire
prepared timeline and recomputes *all* physical/actor/root fields, including
float32 coordinate rounding. Rotating cached per-channel RMS is incorrect.
This correctness-first RAM route is bounded, not a claim of free augmentation
or a complete optimizer/CLIP/baseline host. Profile its cost before training.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .continuous_capture import PreparedContinuousCapture
from .continuous_capture_io import _file_sha256, load_prepared_continuous_capture
from .continuous_phase_cache import load_continuous_phase_cache
from .directional_phase import (
    DirectionalPhaseField,
    LocalPhaseConfig,
    continuous_directional_phase_field,
)


def shared_yaw_capture(
    capture: PreparedContinuousCapture, *, yaw_delta: float
) -> PreparedContinuousCapture:
    """Rotate every actor/time about the existing common origin; never recenter.

    The input is already antialiased body22. This is an explicit post-prepare
    float32 training transform, not bitwise equivalence to rotating raw SMPL-X
    parameters before resampling. Masks, rejected intervals and lineage stay.
    """
    if type(capture) is not PreparedContinuousCapture:
        raise TypeError("capture must be exactly PreparedContinuousCapture")
    if (
        isinstance(yaw_delta, bool)
        or not isinstance(yaw_delta, (int, float))
        or not math.isfinite(yaw_delta)
        or not math.isfinite(capture.augmentation_yaw + yaw_delta)
    ):
        raise ValueError("yaw_delta and the total yaw must be finite real angles")
    if yaw_delta == 0.0:
        return capture
    values = capture.skeletons.astype(np.float64)
    cosine, sine = math.cos(yaw_delta), math.sin(yaw_delta)
    rotated = np.empty_like(values)
    rotated[..., 0] = cosine * values[..., 0] + sine * values[..., 2]
    rotated[..., 1] = values[..., 1]
    rotated[..., 2] = -sine * values[..., 0] + cosine * values[..., 2]
    skeletons = np.ascontiguousarray(
        np.where(capture.track_mask[..., None], rotated, 0.0), dtype=np.float32
    )
    skeletons[skeletons == 0.0] = 0.0
    return replace(
        capture,
        skeletons=skeletons,
        augmentation_yaw=capture.augmentation_yaw + yaw_delta,
    )


@dataclass(frozen=True)
class ContinuousTrainingView:
    capture: PreparedContinuousCapture
    phase_field: DirectionalPhaseField
    reused_physical_cache: bool

    @property
    def positive_capture_key(self) -> str:
        """Repeated actor sets are not positives; identity is the source capture."""
        return self.capture.source_sha256


@dataclass(frozen=True)
class ContinuousTrainingInput:
    """Load once per admitted capture; request complete views during training.

    Dataset/split/rights/calibration-population admission remains the frozen
    execution manifest's job. No text is inspected here. The caller must decide
    whether a caption permits yaw (e.g. world-direction language may not).
    """

    capture: PreparedContinuousCapture
    cached_field: DirectionalPhaseField

    @classmethod
    def load(
        cls,
        body_directory: str | Path,
        cache_directory: str | Path,
        *,
        cache_record: dict[str, Any],
        energy_floors: np.ndarray,
    ) -> ContinuousTrainingInput:
        prepared_record = cache_record["prepared_record"]
        if prepared_record["augmentation_yaw"] != 0.0:
            raise ValueError("this development cache route requires unaugmented body22")
        capture = load_prepared_continuous_capture(
            body_directory, expected_metadata=prepared_record
        )
        if cache_record["source_sha256"] != capture.source_sha256:
            raise ValueError("physical and prepared records refer to different captures")
        root = Path(cache_directory)
        names = {
            "metadata.json",
            "actor-features.npy",
            "actor-patch-mask.npy",
            "root-positions.npy",
            "root-patch-mask.npy",
        } | {
            f"{prefix}-{band}.npy"
            for prefix in ("response", "response-mask", "centers")
            for band in range(1, 7)
        }
        digests = cache_record["artifact_sha256s"]
        if set(digests) != names:
            raise ValueError("physical manifest must contain the complete fixed cache arrays")
        for name in sorted(names):
            if _file_sha256(root / name) != digests[name]:
                raise ValueError(f"physical cache artifact differs: {name}")
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if metadata["frame_count"] != capture.frame_count:
            raise ValueError("physical cache timeline differs from prepared body22")
        field = load_continuous_phase_cache(
            root, expected_source_sha256=capture.source_sha256, energy_floors=energy_floors
        )
        if field.actor_count != capture.actor_count or field.patch_count != cache_record["patches"]:
            raise ValueError("physical cache actor/patch census differs")
        return cls(capture, field)

    def view(
        self,
        *,
        yaw_delta: float = 0.0,
        allow_shared_yaw: bool,
        config: LocalPhaseConfig | None = None,
    ) -> ContinuousTrainingView:
        if type(allow_shared_yaw) is not bool:
            raise ValueError("caption yaw eligibility must be an explicit boolean")
        rotated = shared_yaw_capture(self.capture, yaw_delta=yaw_delta)
        if rotated is self.capture:
            if config is not None and config != self.cached_field.config:
                raise ValueError("a reused cache cannot change the physical configuration")
            return ContinuousTrainingView(self.capture, self.cached_field, True)
        if not allow_shared_yaw:
            raise ValueError("nonzero yaw is not admitted for this capture's text")
        selected_config = self.cached_field.config if config is None else config
        if any(
            getattr(selected_config, name) != getattr(self.cached_field.config, name)
            for name in ("patch_frames", "hop_frames", "epsilon")
        ):
            raise ValueError("augmentation cannot change the frozen physical configuration")
        # No zero-yaw features or approximate RMS/complex rotations are reused.
        # The existing RAM/edge gates explicitly reject oversized full captures.
        field = continuous_directional_phase_field(
            rotated, energy_floors=self.cached_field.energy_floors, config=selected_config
        )
        return ContinuousTrainingView(rotated, field, False)
