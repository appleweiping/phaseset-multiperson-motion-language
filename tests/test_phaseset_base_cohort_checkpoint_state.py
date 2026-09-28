from __future__ import annotations

import copy
import hashlib
import random
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from phaseset_core import base_cohort_resolver as resolver
from phaseset_core import training


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


class _TinyBase(nn.Module):
    """Small engineering fixture; never a registered or qualified PhaseSet base."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.25, -0.5], dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor([0.125], dtype=torch.float32))


class _ValueDriftTinyBase(_TinyBase):
    def load_state_dict(self, *args: object, **kwargs: object) -> object:
        result = super().load_state_dict(*args, **kwargs)
        with torch.no_grad():
            self.weight.add_(1.0)
        return result


def _config() -> training.TrainingConfig:
    return training.TrainingConfig(
        stage="base",
        seed=1729,
        device="cpu",
        request_bf16=False,
        bf16_runtime_qualified=False,
        edge_budget=32_768,
        checkpoint_every_updates=1,
        synthetic_contract=False,
    )


def _binding() -> SimpleNamespace:
    return SimpleNamespace(
        behavior_sha256=_sha("tiny-behavior"),
        code_artifact_sha256=_sha("tiny-code"),
        environment_sha256=_sha("tiny-environment"),
        factory_sha256=_sha("tiny-factory"),
        initial_optimizable_state_sha256=_sha("tiny-initial-state"),
        sha256=_sha("tiny-initialization-binding"),
    )


def _payload(
    config: training.TrainingConfig,
    *,
    global_step: int,
    total_steps: int,
) -> dict[str, object]:
    source = _TinyBase()
    optimizer = torch.optim.AdamW(
        source.parameters(),
        lr=config.learning_rate,
        weight_decay=training.WEIGHT_DECAY,
    )

    def multiplier(step: int) -> float:
        return training._learning_rate_multiplier(step, total_steps)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    for parameter in source.parameters():
        parameter.grad = torch.full_like(parameter, 0.5)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    binding = _binding()
    return {
        "behavior_sha256": binding.behavior_sha256,
        "code_artifact_sha256": binding.code_artifact_sha256,
        "environment_sha256": binding.environment_sha256,
        "factory_sha256": binding.factory_sha256,
        "initial_optimizable_state_sha256": binding.initial_optimizable_state_sha256,
        "initialization_binding_sha256": binding.sha256,
        "model": copy.deepcopy(source.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "rng": copy.deepcopy(training._capture_rng()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
    }


@pytest.fixture
def installed_tiny_builder(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    binding = _binding()

    def construct(
        system_id: str,
        _config_value: training.TrainingConfig,
    ) -> tuple[nn.Module, SimpleNamespace]:
        assert system_id == "B0"
        return _TinyBase(), binding

    def state_sha256(system: nn.Module, **_kwargs: object) -> str:
        return training._stable_hash(system.state_dict())

    monkeypatch.setattr(
        resolver.training_module,
        "construct_registered_base_seed_bound_system",
        construct,
    )
    monkeypatch.setattr(
        resolver.training_module,
        "training_system_state_sha256",
        state_sha256,
    )
    monkeypatch.setattr(
        resolver.training_module,
        "training_system_behavior_sha256",
        lambda _system: binding.behavior_sha256,
    )
    return binding


def test_real_model_optimizer_scheduler_and_rng_state_validate_without_rng_drift(
    installed_tiny_builder: SimpleNamespace,
) -> None:
    del installed_tiny_builder
    config = _config()
    payload = _payload(config, global_step=1, total_steps=8)
    before = training._stable_hash(training._capture_rng())
    resolver._validate_model_and_optimizer(
        payload,
        config=config,
        system_id="B0",
        global_step=1,
        total_steps=8,
    )
    assert training._stable_hash(training._capture_rng()) == before


@pytest.mark.parametrize("mutation", ["missing_key", "wrong_shape"])
def test_real_strict_model_restore_rejects_key_and_shape_corruption(
    installed_tiny_builder: SimpleNamespace,
    mutation: str,
) -> None:
    del installed_tiny_builder
    config = _config()
    payload = _payload(config, global_step=1, total_steps=8)
    model = payload["model"]
    assert isinstance(model, dict)
    if mutation == "missing_key":
        model.pop("bias")
    else:
        model["weight"] = torch.zeros((1, 2), dtype=torch.float32)
    with pytest.raises(resolver.BaseCohortResolutionError, match="model/optimizer state"):
        resolver._validate_model_and_optimizer(
            payload,
            config=config,
            system_id="B0",
            global_step=1,
            total_steps=8,
        )


def test_loaded_model_value_digest_mismatch_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    installed_tiny_builder: SimpleNamespace,
) -> None:
    config = _config()
    payload = _payload(config, global_step=1, total_steps=8)
    monkeypatch.setattr(
        resolver.training_module,
        "construct_registered_base_seed_bound_system",
        lambda _system_id, _config_value: (_ValueDriftTinyBase(), installed_tiny_builder),
    )
    with pytest.raises(resolver.BaseCohortResolutionError, match="differs from checkpoint bytes"):
        resolver._validate_model_and_optimizer(
            payload,
            config=config,
            system_id="B0",
            global_step=1,
            total_steps=8,
        )


@pytest.mark.parametrize("mutation", ["optimizer_step", "scheduler_cursor"])
def test_real_optimizer_scheduler_cursor_corruption_is_rejected(
    installed_tiny_builder: SimpleNamespace,
    mutation: str,
) -> None:
    del installed_tiny_builder
    config = _config()
    payload = _payload(config, global_step=1, total_steps=8)
    if mutation == "optimizer_step":
        optimizer = payload["optimizer"]
        assert isinstance(optimizer, dict)
        states = optimizer["state"]
        assert isinstance(states, dict)
        first = next(iter(states.values()))
        assert isinstance(first, dict)
        first["step"] = torch.tensor(2.0)
    else:
        scheduler = payload["scheduler"]
        assert isinstance(scheduler, dict)
        scheduler["last_epoch"] = 2
    with pytest.raises(resolver.BaseCohortResolutionError, match="model/optimizer state"):
        resolver._validate_model_and_optimizer(
            payload,
            config=config,
            system_id="B0",
            global_step=1,
            total_steps=8,
        )


def test_rng_partial_restore_failure_restores_ambient_state(
    installed_tiny_builder: SimpleNamespace,
) -> None:
    del installed_tiny_builder
    config = _config()
    payload = _payload(config, global_step=1, total_steps=8)
    rng = payload["rng"]
    assert isinstance(rng, dict)
    rng["python"] = random.Random(991).getstate()
    rng.pop("numpy_keys")
    before = training._stable_hash(training._capture_rng())
    with pytest.raises(resolver.BaseCohortResolutionError, match="RNG state"):
        resolver._validate_model_and_optimizer(
            payload,
            config=config,
            system_id="B0",
            global_step=1,
            total_steps=8,
        )
    assert training._stable_hash(training._capture_rng()) == before


def test_registered_builder_identity_mismatch_is_rejected_before_state_restore(
    installed_tiny_builder: SimpleNamespace,
) -> None:
    del installed_tiny_builder
    config = _config()
    payload = _payload(config, global_step=1, total_steps=8)
    payload["factory_sha256"] = _sha("substituted-factory")
    before = training._stable_hash(training._capture_rng())
    with pytest.raises(resolver.BaseCohortResolutionError, match="factory_sha256"):
        resolver._validate_model_and_optimizer(
            payload,
            config=config,
            system_id="B0",
            global_step=1,
            total_steps=8,
        )
    assert training._stable_hash(training._capture_rng()) == before
