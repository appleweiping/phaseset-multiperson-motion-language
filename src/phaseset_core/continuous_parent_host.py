"""Complete-parent FP32 base/residual optimizer, validation and restart host.

This runs real updates, not the superseded window-unit training recipe. The
private operator still admits rights, physical floors, human CF provenance,
hardware/budget and the dated 87-stage matrix. Constructing this host supplies
none of those authorities. Literature baselines remain separate implementations.
No final-test interface exists here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
from typing import Protocol

import numpy as np
import torch

from .continuous_base_retrieval import ContinuousBaseRetrievalSystem
from .continuous_capture import PreparedContinuousCapture
from .continuous_parent_training import (
    ParentCounterfactualRows,
    backward_loaded_parent_batch,
    parent_epoch_batches,
    parent_input_capture,
    validate_parent_text_batch,
)
from .continuous_retrieval import ContinuousRetrievalSystem
from .continuous_training_input import ContinuousTrainingView
from .frozen_clip_text import FrozenClipTextBatch
from .parent_retrieval_task import ParentCaptionRecord, ParentRetrievalBatch, ParentRetrievalTask
from .parent_run_monitor import ParentRunMonitor, parent_failure_code
from .training import (
    OFFICIAL_SEEDS,
    CheckpointArtifact,
    TrainingCheckpointError,
    _atomic_torch_save,
    _capture_macro_bidirectional_r1,
    _capture_rng,
    _frozen_numerical_runtime,
    _learning_rate_multiplier,
    _load_torch_checkpoint,
    _restore_rng,
    _stable_hash,
)


class ParentTrainingSource(Protocol):
    """Admitted disk-backed inputs; no private data discovery in public code.

    ``view`` must return the entire source and reproduce its augmentation on
    replay. Evaluation views must be unaugmented. ``text`` starts with all
    exact human rows; training-only, genuinely verified CFs may follow. The
    two-human truth records are bound by the private input manifest, not by
    the returned bool bits. No test or held-out rows may be loaded incidentally.
    """

    def view(
        self, parent: ParentCaptionRecord, *, seed: int, epoch: int, training: bool
    ) -> ContinuousTrainingView | PreparedContinuousCapture: ...

    def text(
        self, labels: ParentRetrievalBatch, *, training: bool
    ) -> tuple[FrozenClipTextBatch, ParentCounterfactualRows]: ...


@dataclass(frozen=True)
class ParentHostConfig:
    seed: int
    train_components: tuple[str, ...]
    validation_components: tuple[str, ...]
    epochs: int = 20
    parent_batch_size: int = 128
    learning_rate: float = 3e-4
    cf_weight: float = 0.2
    cf_margin: float = 0.2
    stage: str = "residual"

    @classmethod
    def for_base(
        cls,
        seed: int,
        train_components: tuple[str, ...],
        validation_components: tuple[str, ...],
        **settings,
    ):
        """Explicit base defaults; short qualifications are not a study freeze."""
        return cls(
            seed,
            train_components,
            validation_components,
            **{
                "epochs": 30,
                "learning_rate": 2e-4,
                "cf_weight": 0.0,
                "cf_margin": 0.0,
                "stage": "base",
                **settings,
            },
        )

    def __post_init__(self):
        if type(self.seed) is not int or self.seed not in OFFICIAL_SEEDS:
            raise ValueError("host requires one of the three fixed seeds")
        for components in (self.train_components, self.validation_components):
            if (
                type(components) is not tuple
                or not components
                or len(set(components)) != len(components)
                or any(type(value) is not str or not value for value in components)
            ):
                raise ValueError("learning/validation components must be explicit and unique")
        if set(self.train_components) & set(self.validation_components):
            raise ValueError("learning and checkpoint-selection components cannot overlap")
        if type(self.epochs) is not int or self.epochs < 1:
            raise ValueError("epoch count must be a positive integer")
        if type(self.parent_batch_size) is not int or self.parent_batch_size < 2:
            raise ValueError("parent batch size must be at least two")
        for value in (self.learning_rate, self.cf_weight, self.cf_margin):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("loss/optimizer scalars must be finite nonnegative numbers")
        if self.learning_rate == 0:
            raise ValueError("learning rate must be positive")
        if self.stage not in ("base", "residual"):
            raise ValueError("host stage must explicitly be base or residual")
        if self.stage == "base" and (self.cf_weight != 0 or self.cf_margin != 0):
            raise ValueError("a base stage has no CF loss")


@dataclass(frozen=True)
class ParentHostBindings:
    """Digests of the already admitted private execution inputs, not authority.

    The input manifest includes physical source/cache/floor/CLIP identities,
    complete parent census, yaw rules and human verification provenance. The
    code manifest covers this host, the full scorer and private source adapter.
    Runtime binding comes from the real frozen environment qualification.
    """

    input_manifest_sha256: str
    code_manifest_sha256: str
    runtime_sha256: str
    frozen_base_checkpoint_sha256: str | None

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "frozen_base_checkpoint_sha256" and value is None:
                continue
            if (
                type(value) is not str
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError("execution bindings must be lowercase SHA-256 keys")


@dataclass(frozen=True)
class ParentHostReport:
    outcome: str
    global_step: int
    completed_epochs: int
    parents_seen: int
    latest_checkpoint: CheckpointArtifact
    best_checkpoint: CheckpointArtifact | None
    best_validation_r1: float | None
    validation_history: tuple[tuple[int, float], ...]


def _write_json_once(path: Path, payload: dict) -> None:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(raw)
        stream.flush()


class ContinuousParentTrainingHost:
    """AdamW/5%-warmup/cosine, clip1, full-gallery validation, immutable resume.

    FP32 only; no automatic BF16, shorter timeline, dropped tail, sampled
    negatives or toy-model fallback. One effective update is one complete
    parent batch; the same full-gallery score VJP is replayed per parent.
    Validation R@1 is capture-macro in both directions, with any-positive
    M2T and deterministic lineage tie breaks, using the existing endpoint.
    Strict improvement selects checkpoints; exact ties retain the first.
    """

    def __init__(
        self,
        system: ContinuousRetrievalSystem | ContinuousBaseRetrievalSystem,
        task: ParentRetrievalTask,
        source: ParentTrainingSource,
        config: ParentHostConfig,
        bindings: ParentHostBindings,
    ):
        if not isinstance(system, (ContinuousRetrievalSystem, ContinuousBaseRetrievalSystem)):
            raise TypeError("this host requires an actual complete-parent base or V2 scorer")
        self._is_base = isinstance(system, ContinuousBaseRetrievalSystem)
        if config.stage != ("base" if self._is_base else "residual"):
            raise ValueError("host stage does not match the actual scorer")
        if (bindings.frozen_base_checkpoint_sha256 is None) != self._is_base:
            raise ValueError("only a residual stage binds a frozen B2 checkpoint")
        self._initialize(system, task, source, config, bindings)

    def _initialize(self, system, task, source, config, bindings):
        """Shared optimizer/checkpoint loop; public constructors admit scorers."""
        if any(parameter.dtype != torch.float32 for parameter in system.parameters()):
            raise ValueError("this host has only been qualified for FP32")
        self.system, self.task, self.source = system, task, source
        self.config, self.bindings = config, bindings
        # Establish the complete component census before any physical/text load.
        self._steps_per_epoch = len(self._batches(0))
        validation = tuple(
            row for row in task.parents if row.component in config.validation_components
        )
        if {row.component for row in validation} != set(config.validation_components) or any(
            row.split == "test" for row in validation
        ):
            raise ValueError("validation components must be fully admitted development parents")
        self._gallery = task.batch(tuple(row.annotation_family_sha256 for row in validation))
        self._total_steps = self._steps_per_epoch * config.epochs
        self._parameters = [
            parameter for parameter in system.parameters() if parameter.requires_grad
        ]
        if not self._parameters or (
            not self._is_base and any(p.requires_grad for p in system.frozen_b2.parameters())
        ):
            raise ValueError("the residual must be trainable and its B2 anchor frozen")
        if self._is_base and any(not p.requires_grad for p in system.parameters()):
            raise ValueError("base-learning parameters must all remain trainable")
        self._base_state = None if self._is_base else _stable_hash(system.frozen_b2.state_dict())
        self._manifest = {
            "schema": "phaseset-complete-parent-host-v1",
            "bindings": asdict(bindings),
            "config": asdict(config),
            "task": [
                asdict(parent)
                for parent in task.parents
                if parent.component in set(config.train_components + config.validation_components)
            ],
            "initial_model_sha256": _stable_hash(system.state_dict()),
            # State tensors omit actual scorer controls and dropout settings.
            # Store the scalar configuration/module tree in the same manifest.
            "scorer_configuration": [
                {
                    "name": name,
                    "class": f"{type(module).__module__}.{type(module).__qualname__}",
                    "scalars": {
                        key: value
                        for key, value in vars(module).items()
                        if not key.startswith("_")
                        and key not in ("training", "dump_patches", "call_super_init")
                        and (value is None or type(value) in (bool, int, float, str))
                    },
                }
                for name, module in system.named_modules()
            ],
            "frozen_base_state_sha256": self._base_state,
            "steps_per_epoch": self._steps_per_epoch,
            "total_steps": self._total_steps,
        }
        self._optimizer = torch.optim.AdamW(
            self._parameters, lr=config.learning_rate, weight_decay=0.01
        )
        self._scheduler = torch.optim.lr_scheduler.LambdaLR(
            self._optimizer, lambda step: _learning_rate_multiplier(step, self._total_steps)
        )
        self._epoch = self._offset = self._step = self._parents_seen = self._sequence = 0
        self._history: list[tuple[int, float]] = []
        self._best: CheckpointArtifact | None = None
        self._best_value: float | None = None
        self._latest: CheckpointArtifact | None = None
        self._root: Path | None = None
        self._used = False
        self._event_sequence = 0
        self._monitor: ParentRunMonitor | None = None

    def _batches(self, epoch: int):
        return parent_epoch_batches(
            self.task,
            train_components=self.config.train_components,
            batch_size=self.config.parent_batch_size,
            seed=self.config.seed,
            epoch=epoch,
        )

    def _backward_batch(self, labels: ParentRetrievalBatch):
        by_source = {row.source_sha256: row for row in labels.parents}
        text, cf = self.source.text(labels, training=True)
        return backward_loaded_parent_batch(
            self.system,
            labels,
            text,
            load_view=lambda key: self.source.view(
                by_source[key], seed=self.config.seed, epoch=self._epoch, training=True
            ),
            counterfactuals=cf,
            cf_weight=self.config.cf_weight,
            margin=self.config.cf_margin,
            progress=lambda phase, index: self._event(phase, capture_index=index),
        )

    def _event(self, phase: str, **fields):
        assert self._root is not None
        event = {
            "phase": phase,
            "observed_at_utc": datetime.now(timezone.utc).isoformat(),
            "global_step": self._step,
            "epoch": self._epoch,
            "parent_batch_offset": self._offset,
            **fields,
        }
        _write_json_once(
            self._root / f"event-{self._event_sequence:08d}.json",
            event,
        )
        self._event_sequence += 1
        if self._monitor is not None and phase != "terminal":
            self._monitor.observe(event)

    @torch.no_grad()
    def _validate(self) -> float:
        labels = self._gallery
        previous_rng, was_training = _capture_rng(), self.system.training
        scores = []
        self.system.eval()
        try:
            text, cf = self.source.text(labels, training=False)
            if validate_parent_text_batch(labels, text) != len(labels.captions) or cf.source_keys:
                raise ValueError(
                    "validation pool must contain only all official human retrieval rows"
                )
            for index, parent in enumerate(labels.parents):
                self._event("validation_capture", capture_index=index)
                view = self.source.view(
                    parent, seed=self.config.seed, epoch=self._epoch, training=False
                )
                capture = parent_input_capture(self.system, view)
                if capture.source_sha256 != parent.source_sha256 or capture.augmentation_yaw != 0.0:
                    raise ValueError("validation requires the admitted unaugmented whole source")
                scores.append(self.system.score((view,), text).scores.detach().cpu())
                del view, capture
            logits = torch.cat(scores).contiguous()
            if logits.shape != (len(labels.parents), len(labels.captions)) or not bool(
                torch.isfinite(logits).all()
            ):
                raise ValueError("validation full-gallery score census is malformed/nonfinite")
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

    def _checkpoint(self, *, select: bool = False) -> CheckpointArtifact:
        assert self._root is not None
        self._event("checkpoint_write")
        self._sequence += 1
        path = self._root / f"checkpoint-{self._step:012d}-{self._sequence:06d}.pt"
        best = (
            (str(path.resolve()), None)
            if select
            else (None if self._best is None else (str(self._best.path), self._best.sha256))
        )
        payload = {
            "manifest": self._manifest,
            "global_step": self._step,
            "epoch": self._epoch,
            "parent_batch_offset": self._offset,
            "parents_seen": self._parents_seen,
            "checkpoint_sequence": self._sequence,
            "validation_history": self._history,
            "best": best,
            "best_validation_r1": self._best_value,
            "model": self.system.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "scheduler": self._scheduler.state_dict(),
            "rng": _capture_rng(),
        }
        payload["state_sha256"] = _stable_hash(payload)
        self._latest = _atomic_torch_save(path, payload)
        if select:
            self._best = self._latest
        if self._monitor is not None:
            self._monitor.checkpoint(self._latest, best=self._best)
        return self._latest

    def _resume(self, path: Path, digest: str):
        payload, artifact = _load_torch_checkpoint(path, expected_sha256=digest)
        saved_digest = payload.pop("state_sha256", None)
        if _stable_hash(payload) != saved_digest or payload["manifest"] != self._manifest:
            raise TrainingCheckpointError("parent checkpoint state or execution inputs drifted")
        epoch, offset, step = (
            payload[key] for key in ("epoch", "parent_batch_offset", "global_step")
        )
        if any(type(value) is not int for value in (epoch, offset, step)) or not (
            0 <= epoch <= self.config.epochs
            and 0 <= offset <= self._steps_per_epoch
            and step == epoch * self._steps_per_epoch + offset
            and step <= self._total_steps
            and (epoch < self.config.epochs or offset == 0)
        ):
            raise TrainingCheckpointError("parent checkpoint cursor is inconsistent")
        batches = self._batches(0)
        expected_parents = epoch * sum(len(batch.parents) for batch in batches) + sum(
            len(batch.parents) for batch in self._batches(epoch)[:offset]
        )
        history = payload["validation_history"]
        if payload["parents_seen"] != expected_parents or (
            type(history) is not list
            or len(history) != epoch
            or any(
                type(row) is not tuple
                or len(row) != 2
                or row[0] != index
                or not (type(row[1]) is float and math.isfinite(row[1]) and 0 <= row[1] <= 1)
                for index, row in enumerate(history)
            )
        ):
            raise TrainingCheckpointError("parent checkpoint census/validation history drifted")
        model = payload["model"]
        base = {
            key.removeprefix("frozen_b2."): value
            for key, value in model.items()
            if key.startswith("frozen_b2.")
        }
        if (not self._is_base and _stable_hash(base) != self._base_state) or any(
            not bool(torch.isfinite(value).all()) for value in model.values()
        ):
            raise TrainingCheckpointError("parent checkpoint changed the anchor or is nonfinite")
        self.system.load_state_dict(model, strict=True)
        self._optimizer.load_state_dict(payload["optimizer"])
        self._scheduler.load_state_dict(payload["scheduler"])
        expected_lr = self.config.learning_rate * _learning_rate_multiplier(step, self._total_steps)
        if (
            self._scheduler.last_epoch != step
            or any(
                group["lr"] != expected_lr or group["weight_decay"] != 0.01
                for group in self._optimizer.param_groups
            )
            or any(
                int(state["step"]) != step
                or not bool(torch.isfinite(state["exp_avg"]).all())
                or not bool(torch.isfinite(state["exp_avg_sq"]).all())
                for state in self._optimizer.state.values()
            )
        ):
            raise TrainingCheckpointError("parent checkpoint optimizer/scheduler drifted")
        self._epoch, self._offset, self._step = epoch, offset, step
        self._parents_seen = payload["parents_seen"]
        self._sequence = payload["checkpoint_sequence"]
        if type(self._sequence) is not int or self._sequence < 1:
            raise TrainingCheckpointError("checkpoint sequence must be a positive integer")
        self._history = history
        self._best_value = payload["best_validation_r1"]
        best = payload["best"]
        if best is not None:
            best_path, best_digest = best
            if best_digest is None:
                if Path(best_path) != artifact.path:
                    raise TrainingCheckpointError("selected checkpoint self-reference differs")
                self._best = artifact
            else:
                _, self._best = _load_torch_checkpoint(Path(best_path), expected_sha256=best_digest)
        if (self._best is None) != (not history) or self._best_value != (
            max(row[1] for row in history) if history else None
        ):
            raise TrainingCheckpointError(
                "parent checkpoint best selector disagrees with validation"
            )
        self._latest = artifact
        _restore_rng(payload["rng"])

    def fit(
        self,
        run_directory: str | Path,
        *,
        resume_checkpoint: str | Path | None = None,
        resume_sha256: str | None = None,
        stop_after_steps: int | None = None,
        checkpoint_keep_recent: int | None = None,
    ) -> ParentHostReport:
        """Run or resume into a distinct immutable attempt directory.

        A deliberate interruption ends at an update boundary. If that boundary
        is the end of an epoch, its validation is still performed before stop;
        the selected checkpoint cannot be bypassed by resume. Terminal records
        never claim research completion or grant a formal start/pilot budget.
        ``checkpoint_keep_recent>=2`` explicitly retires this attempt's older
        unreferenced payloads only; None preserves the legacy keep-all policy.
        """
        if self._used:
            raise RuntimeError("a host instance may own only one attempt")
        if (resume_checkpoint is None) != (resume_sha256 is None):
            raise ValueError("resume requires the exact predecessor artifact digest")
        if stop_after_steps is not None and (
            type(stop_after_steps) is not int or not 0 <= stop_after_steps <= self._total_steps
        ):
            raise ValueError("stop point must be an update boundary within this schedule")
        if checkpoint_keep_recent is not None and (
            type(checkpoint_keep_recent) is not int or checkpoint_keep_recent < 2
        ):
            raise ValueError("checkpoint retention must keep at least two recent payloads")
        self._used = True
        self._root = Path(run_directory).resolve()
        self._root.mkdir(parents=True, exist_ok=False, mode=0o700)
        self._monitor = ParentRunMonitor(self._root, keep_recent=checkpoint_keep_recent)
        outcome, failure, primary_error = "FAILED", None, None
        try:
            self._monitor.start()
            _write_json_once(
                self._root / "run.json",
                {
                    "manifest": self._manifest,
                    "predecessor_checkpoint": None
                    if resume_checkpoint is None
                    else str(Path(resume_checkpoint).resolve()),
                    "predecessor_sha256": resume_sha256,
                    "checkpoint_keep_recent": checkpoint_keep_recent,
                    "formal_authority": False,
                },
            )
            with _frozen_numerical_runtime(next(self.system.parameters()).device):
                if resume_checkpoint is None:
                    random.seed(self.config.seed)
                    np.random.seed(self.config.seed)
                    torch.manual_seed(self.config.seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(self.config.seed)
                else:
                    self._resume(Path(resume_checkpoint), resume_sha256)
                if stop_after_steps is not None and stop_after_steps < self._step:
                    raise ValueError("stop point precedes resumed progress")
                self._event("start")
                while self._epoch < self.config.epochs:
                    batches = self._batches(self._epoch)
                    if self._offset == len(batches):
                        value = self._validate()
                        selected = self._best_value is None or value > self._best_value
                        if selected:
                            self._best_value = value
                        self._history.append((self._epoch, value))
                        self._epoch += 1
                        self._offset = 0
                        self._checkpoint(select=selected)
                        self._event(
                            "validation", capture_macro_bidirectional_r1=value, selected=selected
                        )
                        continue
                    if stop_after_steps is not None and self._step == stop_after_steps:
                        break
                    labels = batches[self._offset]
                    self._event("training_batch")
                    result = self._backward_batch(labels)
                    self._monitor.check()
                    norm = torch.nn.utils.clip_grad_norm_(
                        self._parameters, 1.0, error_if_nonfinite=True
                    )
                    lr = self._optimizer.param_groups[0]["lr"]
                    self._optimizer.step()
                    self._scheduler.step()
                    if (
                        not self._is_base
                        and _stable_hash(self.system.frozen_b2.state_dict()) != self._base_state
                    ):
                        raise RuntimeError("training mutated the shared frozen B2 anchor")
                    self._step += 1
                    self._offset += 1
                    self._parents_seen += len(labels.parents)
                    self._checkpoint()
                    self._event(
                        "update",
                        loss=result.loss,
                        parents=len(labels.parents),
                        human_rows=result.retrieval_caption_count,
                        verified_cf=result.verified_counterfactual_count,
                        gradient_norm=float(norm),
                        learning_rate=lr,
                    )
                outcome = "COMPLETED" if self._epoch == self.config.epochs else "INTERRUPTED"
                if self._latest is None:
                    self._checkpoint()
                assert self._latest is not None
                return ParentHostReport(
                    outcome,
                    self._step,
                    self._epoch,
                    self._parents_seen,
                    self._latest,
                    self._best,
                    self._best_value,
                    tuple(self._history),
                )
        except BaseException as error:
            failure = type(error).__name__
            primary_error = error
            raise
        finally:
            cleanup_error = None
            try:
                self._monitor.finish(outcome=outcome, failure=primary_error)
            except BaseException as error:
                cleanup_error = error
                if primary_error is None:
                    outcome, failure, primary_error = "FAILED", type(error).__name__, error
                else:
                    primary_error.add_note(f"monitor close also failed: {error!r}")
            try:
                self._event("terminal", outcome=outcome, failure_class=failure)
                _write_json_once(
                    self._root / "terminal.json",
                    {
                        "outcome": outcome,
                        "failure_class": failure,
                        "failure_code": parent_failure_code(primary_error),
                        "global_step": self._step,
                        "completed_epochs": self._epoch,
                        "parents_seen": self._parents_seen,
                        "latest_checkpoint": None
                        if self._latest is None
                        else str(self._latest.path),
                        "latest_checkpoint_sha256": None
                        if self._latest is None
                        else self._latest.sha256,
                        "best_validation_r1": self._best_value,
                        "monitor": self._monitor.summary(),
                        "formal_authority": False,
                    },
                )
            except BaseException as error:
                if primary_error is None:
                    primary_error = error
                    raise
                primary_error.add_note(f"terminal evidence write also failed: {error!r}")
            finally:
                try:
                    self._monitor.release()
                except BaseException as error:
                    if primary_error is None:
                        raise
                    primary_error.add_note(f"lifetime lease release also failed: {error!r}")
            if cleanup_error is not None and primary_error is cleanup_error:
                raise cleanup_error
