"""Faithful TMR/WaMo/MIME optimizer hosts using the shared parent loop.

No final-test interface or generic contrastive substitute. Native data/assets,
whole-parent amendment, hardware profile and bounded study/pilot admission
remain private operator gates; this constructor cannot grant them.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Protocol

import torch
from torch import Tensor
from torch.nn import functional as F

from .continuous_capture import PreparedContinuousCapture
from .continuous_parent_host import (
    ContinuousParentTrainingHost,
    ParentHostBindings,
    ParentHostConfig,
)
from .continuous_parent_training import ParentBackwardResult
from .literature_parent_training import (
    _group,
    backward_loaded_mime_parent_batch,
    backward_loaded_wamo_parent_batch,
)
from .mime_language import TrainableMIMECLIP
from .mime_set import MIMERetrieval, MIMESetMotion
from .mime_parent_curriculum import MimeParentAnchors, mime_parent_epoch_batches
from .parent_retrieval_task import ParentCaptionRecord, ParentRetrievalTask
from .tmr_parent_training import backward_loaded_tmr_parent_batch
from .tmr_set import TMRSet, TMRTextBatch
from .training import _capture_macro_bidirectional_r1, _capture_rng, _restore_rng, _stable_hash
from .wamo_set import WaMoSet


class LiteratureParentSource(Protocol):
    """Entire disk-backed source, deterministic seed/epoch augmentation.

    Every reload must return identical physical arrays for that source,
    seed/epoch/training tuple, independently of model dropout/sampling RNG.
    Validation returns unaugmented data. Language methods receive all exact
    human rows in order and use the operator-admitted frozen asset snapshots.
    MIME uses its own genuinely trainable tokenizer/tower, not these features.
    """

    def capture(
        self, parent: ParentCaptionRecord, *, seed: int, epoch: int, training: bool
    ) -> PreparedContinuousCapture: ...

    def tmr_text(self, captions: tuple[str, ...]) -> TMRTextBatch: ...

    def wamo_cls(self, captions: tuple[str, ...]) -> Tensor: ...


class LiteratureParentTrainingHost(ContinuousParentTrainingHost):
    """Same schedule, clipping, validation selector and resume as B0/B1/B2.

    Each method keeps its own qualified full-negative objective/replay. Text
    is encoded once per full validation gallery; complete motion parents are
    streamed one at a time. Inter-X's original role-aware MIME is separate.
    Adam versus AdamW and weight decay are explicit resume-bound configuration,
    not silently forced to the internal base/residual AdamW .01 default.
    """

    def __init__(
        self,
        system: TMRSet | WaMoSet | MIMERetrieval,
        task: ParentRetrievalTask,
        source: LiteratureParentSource,
        config: ParentHostConfig,
        bindings: ParentHostBindings,
    ):
        if isinstance(system, TMRSet):
            method, objective = "TMR-Set", asdict(system.loss_weights)
        elif isinstance(system, WaMoSet):
            method, objective = "WaMo-Set", asdict(system.config)
        elif (
            isinstance(system, MIMERetrieval)
            and isinstance(system.motion, MIMESetMotion)
            and isinstance(system.language, TrainableMIMECLIP)
        ):
            method, objective = "MIME-Set", asdict(system.motion.config)
        else:
            raise TypeError("need actual TMR/WaMo/shared MIME models, not original dyadic MIME")
        if config.stage != "base" or config.cf_weight != 0 or config.cf_margin != 0:
            raise ValueError(
                "literature baselines use their original losses, not the V2 CF residual"
            )
        if bindings.frozen_base_checkpoint_sha256 is not None:
            raise ValueError("literature models train independently, not on a frozen B2 checkpoint")
        self._is_base = True  # independent trainable model, shared loop has no anchor
        self._method = method
        self._run_identity = None  # literature rows have no V2 mechanism predecessor
        self._initialize(system, task, source, config, bindings, None)
        self._manifest.update(
            schema="phaseset-literature-parent-host-v1",
            literature_method=method,
            objective_configuration=objective,
        )

    def _capture(self, parent, *, training):
        capture = self.source.capture(
            parent, seed=self.config.seed, epoch=self._epoch, training=training
        )
        if (
            type(capture) is not PreparedContinuousCapture
            or capture.source_sha256 != parent.source_sha256
        ):
            raise ValueError("loader returned a different complete physical source")
        if not training and capture.augmentation_yaw != 0.0:
            raise ValueError("validation must use the admitted unaugmented complete source")
        return capture

    def _backward_batch(self, labels):
        by_source = {row.source_sha256: row for row in labels.parents}
        args = dict(
            learning_components=self.config.train_components,
            load_capture=lambda key: self._capture(by_source[key], training=True),
            progress=lambda phase, index: self._event(phase, capture_index=index),
        )
        if self._method == "TMR-Set":
            result = backward_loaded_tmr_parent_batch(
                self.system, labels, load_text=self.source.tmr_text, **args
            )
        elif self._method == "WaMo-Set":
            result = backward_loaded_wamo_parent_batch(
                self.system, labels, load_cls=self.source.wamo_cls, **args
            )
        else:
            result = backward_loaded_mime_parent_batch(self.system, labels, **args)
        self._event(
            "literature_objective", method=self._method, loss_components=result.loss_components
        )
        return ParentBackwardResult(
            result.loss,
            result.scores,
            result.parent_count,
            result.text_count,
            0,
            result.replayed_parents,
        )

    @torch.no_grad()
    def _validate(self):
        labels = self._gallery
        previous_rng, was_training = _capture_rng(), self.system.training
        self.system.eval()
        try:
            if self._method == "TMR-Set":
                text = self.source.tmr_text(labels.captions)
                if type(text) is not TMRTextBatch or len(text.tokens) != len(labels.captions):
                    raise ValueError("validation needs all exact frozen TMR human rows")
                text_embedding = self.system.text_distributions(text)[0]
                scale = 1 / 0.1
            elif self._method == "WaMo-Set":
                cls = self.source.wamo_cls(labels.captions)
                if len(cls) != len(labels.captions):
                    raise ValueError("validation needs all exact frozen DistilBERT CLS rows")
                text_embedding = self.system.encode_text(cls)
                scale = 1 / self.system.config.temperature
            else:
                text_embedding = self.system.language(
                    self.system.language.tokenize(labels.captions)
                )
                scale = self.system.logit_scale.exp()
            text_embedding = F.normalize(text_embedding, dim=1)
            scores = []
            for index, parent in enumerate(labels.parents):
                self._event("validation_capture", capture_index=index)
                capture = _group(
                    lambda key: self._capture(parent, training=False), parent.source_sha256
                )
                if self._method == "TMR-Set":
                    motion = self.system.encode_capture(capture)[0]
                elif self._method == "WaMo-Set":
                    motion = self.system.encode_capture(capture)[0]
                else:
                    motion = self.system.motion(capture)
                scores.append((scale * (F.normalize(motion[None], dim=1) @ text_embedding.T)).cpu())
                del capture, motion
            logits = torch.cat(scores).contiguous()
            if logits.shape != (len(labels.parents), len(labels.captions)) or not bool(
                torch.isfinite(logits).all()
            ):
                raise ValueError("literature full human gallery is malformed or nonfinite")
            return _capture_macro_bidirectional_r1(
                logits,
                labels.positive_mask(device=torch.device("cpu")),
                tuple(bytes.fromhex(key) for key in labels.motion_positive_keys),
                tuple(bytes.fromhex(key) for key in labels.text_positive_keys),
                tuple(bytes.fromhex(key) for key in labels.motion_source_keys),
                labels.caption_commitments,
            )
        finally:
            self.system.train(was_training)
            _restore_rng(previous_rng)


class MIMEParentTrainingHost(LiteratureParentTrainingHost):
    """Actual shared MIME optimizer with its training-only curriculum hook.

    Anchors are prepared once from every official learning-parent human row,
    independently of the trainable MIME language tower. Neighbor width and
    anchor identity join the existing exact checkpoint/resume manifest.
    Uniform warmup and cosine hardness follow the qualified public sampler;
    this grants neither pilot slots nor original dyadic reproduction.
    """

    def __init__(
        self,
        system: MIMERetrieval,
        task: ParentRetrievalTask,
        source: LiteratureParentSource,
        config: ParentHostConfig,
        bindings: ParentHostBindings,
        *,
        anchors: MimeParentAnchors,
        neighbor_window: int,
    ):
        if not isinstance(system, MIMERetrieval) or not isinstance(system.motion, MIMESetMotion):
            raise TypeError("curriculum host requires the actual shared MIME group scorer")
        # Own the sampling values: a frozen dataclass alone cannot stop the
        # caller mutating its tensor after this host binds the manifest.
        self._curriculum_anchors = MimeParentAnchors(
            anchors.family_keys, anchors.embeddings.clone()
        )
        self._neighbor_window = neighbor_window
        super().__init__(system, task, source, config, bindings)
        self._manifest["mime_curriculum"] = {
            "neighbor_window": neighbor_window,
            "anchor_family_keys": self._curriculum_anchors.family_keys,
            "anchor_embeddings_sha256": _stable_hash(self._curriculum_anchors.embeddings),
            "sampler": "mime_parent_epoch_batches",
            "uniform_epochs": 3,
            "ramp_maximum_epoch_zero_based": 12,
            "maximum_hardness": 0.25,
            "source": "all_learning_parent_human_frozen_clip_mean",
        }

    def _batches(self, epoch: int):
        return mime_parent_epoch_batches(
            self.task,
            self._curriculum_anchors,
            train_components=self.config.train_components,
            batch_size=self.config.parent_batch_size,
            neighbor_window=self._neighbor_window,
            seed=self.config.seed,
            epoch=epoch,
        )
