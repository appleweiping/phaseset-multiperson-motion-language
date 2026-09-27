"""Place native raw body22 episodes on their actual parent 30 Hz timeline.

The caller admits official parent/actor lineage before supplying episodes.
These are licensed-format raw ``PrivateCaptureArrays`` (world coordinates),
never independently centered/resampled child ``PreparedContinuousCapture``s.
Unreleased intervals are explicitly unobserved, not stationary or interpolated
motion. One subsequent ``prepare_continuous_capture`` applies a common origin,
yaw, FIR, target sampling grid and the unchanged ten-second acceptance rules.
This is a numeric input seam, not parent task/split/gallery/statistical admission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np

from .pipeline import PrivateCaptureArrays, PreparationResourceLimit, _capture_lineage_sha256
from .preprocessing import PreprocessingError


@dataclass(frozen=True)
class ParentCaptureEpisode:
    """A raw released episode with verified absolute, half-open 30 Hz bounds.

    Official inclusive filename ends must be converted to ``stop=end+1`` by
    the private data adapter and checked against the actual array length.
    One actor commitment follows the same real track across all episodes.
    """

    capture: PrivateCaptureArrays = field(repr=False)
    source_start_frame: int
    source_stop_frame: int

    def __post_init__(self) -> None:
        if type(self.capture) is not PrivateCaptureArrays:
            raise TypeError("parent episodes require raw PrivateCaptureArrays")
        if self.capture.source_sha256 is None:
            raise PreprocessingError("parent episodes require explicit native source provenance")
        if (
            type(self.source_start_frame) is not int
            or type(self.source_stop_frame) is not int
            or not 0 <= self.source_start_frame < self.source_stop_frame
            or self.source_stop_frame - self.source_start_frame != self.capture.frame_count
        ):
            raise PreprocessingError("episode absolute interval must match its real raw frames")


@dataclass(frozen=True)
class ParentCaptureAssembly:
    capture: PrivateCaptureArrays = field(repr=False)
    source_start_frame: int
    source_stop_frame: int
    episode_intervals: tuple[tuple[int, int], ...]
    missing_intervals: tuple[tuple[int, int], ...]
    episode_source_sha256s: tuple[str, ...]


def assemble_parent_capture(
    episodes: tuple[ParentCaptureEpisode, ...], *, max_assembly_bytes: int = 1_073_741_824
) -> ParentCaptureAssembly:
    """Align all native actors and episodes without compacting time or motion.

    Array time zero is the first released episode's original start; the exact
    absolute bounds remain on the assembly. Gaps and native missing tracks
    keep false masks and exact +0 coordinates. No one is removed, no pose is
    extrapolated, and no child is recentered. Overlapping releases or changed
    actor sets require explicit data-contract resolution rather than guessing.
    Memory grows linearly with the true parent span and actor count; an explicit
    resource limit precedes assembly allocation. This does not assert the
    combined assembly/preprocessing/phase/neural peak fits the same budget.
    """
    if (
        type(episodes) is not tuple
        or not episodes
        or any(type(episode) is not ParentCaptureEpisode for episode in episodes)
    ):
        raise TypeError("episodes must be a nonempty exact ParentCaptureEpisode tuple")
    if type(max_assembly_bytes) is not int or max_assembly_bytes < 1:
        raise ValueError("max_assembly_bytes must be a positive exact int")
    ordered = tuple(sorted(episodes, key=lambda episode: episode.source_start_frame))
    actors = tuple(sorted(ordered[0].capture.actor_commitments))
    for index, episode in enumerate(ordered):
        if tuple(sorted(episode.capture.actor_commitments)) != actors:
            raise PreprocessingError("parent episode actor membership changed")
        if index and episode.source_start_frame < ordered[index - 1].source_stop_frame:
            raise PreprocessingError("overlapping parent episodes require data-contract resolution")
        track = episode.capture.track_mask[:, :, :22]
        if not np.array_equal(track, np.broadcast_to(track[..., :1], track.shape)):
            raise PreprocessingError("parent body22 tracking must be uniform per actor-frame")

    first, stop = ordered[0].source_start_frame, ordered[-1].source_stop_frame
    frames = stop - first
    dtype = np.result_type(*(episode.capture.joints.dtype for episode in ordered))
    # Assembly and PrivateCaptureArrays each own a numeric/mask copy; include
    # conservative boolean validation/assignment scratch without K*K storage.
    projected_bytes = frames * len(actors) * 22 * (2 * 3 * dtype.itemsize + 16)
    if projected_bytes > max_assembly_bytes:
        raise PreparationResourceLimit(
            "RESOURCE_LIMIT: parent raw timeline assembly exceeds budget"
        )
    joints = np.zeros((frames, len(actors), 22, 3), dtype=dtype)
    tracking = np.zeros(joints.shape[:-1], dtype=np.bool_)
    missing, source_digests = [], []
    lineage = hashlib.sha256(b"phaseset-native-parent-timeline-v1\x00")
    for actor in actors:
        lineage.update(actor)
    previous_stop = first
    for episode in ordered:
        raw = episode.capture
        order = [raw.actor_commitments.index(actor) for actor in actors]
        start = episode.source_start_frame - first
        end = episode.source_stop_frame - first
        mask = raw.track_mask[:, order, :22]
        joints[start:end] = np.where(mask[..., None], raw.joints[:, order, :22], 0.0)
        tracking[start:end] = mask
        if episode.source_start_frame > previous_stop:
            missing.append((previous_stop, episode.source_start_frame))
        previous_stop = episode.source_stop_frame
        digest = _capture_lineage_sha256(raw)
        source_digests.append(digest)
        lineage.update(episode.source_start_frame.to_bytes(8, "big"))
        lineage.update(episode.source_stop_frame.to_bytes(8, "big"))
        lineage.update(digest.encode("ascii"))
    joints[joints == 0.0] = 0.0
    return ParentCaptureAssembly(
        capture=PrivateCaptureArrays(joints, tracking, actors, lineage.hexdigest()),
        source_start_frame=first,
        source_stop_frame=stop,
        episode_intervals=tuple((e.source_start_frame, e.source_stop_frame) for e in ordered),
        missing_intervals=tuple(missing),
        episode_source_sha256s=tuple(source_digests),
    )
