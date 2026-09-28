"""Fail-closed command surface for the PhaseSet execution protocol."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Protocol, Sequence, TextIO

from phaseset_core.execution import (
    AUTHORITY,
    COMMANDS,
    DataFreeRunner,
    ExecutionContractError,
    ExecutionHold,
    RuntimeAdapter,
    RuntimeAdmission,
    canonical_command_result_bytes,
    canonical_preflight_bytes,
    canonical_public_training_config_bytes,
    canonical_runtime_admission_bytes,
    load_training_config,
)
from phaseset_core.experiments import BASE_SYSTEMS, FINAL_SYSTEMS, SEEDS, parse_run_id


EXIT_OK = 0
EXIT_USAGE_OR_CONTRACT = 2
EXIT_HOLD = 3
EXIT_EXECUTION_FAILED = 4
_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "phaseset" / "training.json"
_BASE_SYSTEM_IDS = tuple(system_id for system_id, _ in BASE_SYSTEMS)
_FINAL_SYSTEM_IDS = tuple(system_id for system_id, _ in FINAL_SYSTEMS)
_RESIDUAL_SYSTEM_IDS = tuple(system_id for system_id in _FINAL_SYSTEM_IDS if system_id != "00")
_RESUME_SYSTEM_IDS = _BASE_SYSTEM_IDS + _FINAL_SYSTEM_IDS


@dataclass(frozen=True, slots=True)
class CLICommandRequest:
    """Private, in-process command parameters passed to a trusted adapter factory.

    This object is deliberately not serializable by the public CLI. Paths may
    identify private data or execution state, so they are delivered only to a
    host-injected factory and can never appear in stdout, stderr, admission
    JSON, or command-result JSON.
    """

    command: str
    run_id: str | None = None
    seed: int | None = None
    split: str | None = None
    system_id: str | None = None
    source_manifest: Path | None = None
    prepared_root: Path | None = None
    split_manifest: Path | None = None
    terminal_root: Path | None = None
    base_checkpoint: Path | None = None
    periodic_cache: Path | None = None
    cache_root: Path | None = None
    checkpoint_directory: Path | None = None
    resume_checkpoint: Path | None = None
    stop_after_global_step: int | None = None
    evaluation_manifest: Path | None = None
    statistical_report: Path | None = None
    attempt_directory: Path | None = None

    def __post_init__(self) -> None:
        if type(self.command) is not str or self.command not in COMMANDS:
            raise ExecutionContractError("CLI command request is outside the closed census")
        for name in (
            "source_manifest",
            "prepared_root",
            "split_manifest",
            "terminal_root",
            "base_checkpoint",
            "periodic_cache",
            "cache_root",
            "checkpoint_directory",
            "resume_checkpoint",
            "evaluation_manifest",
            "statistical_report",
            "attempt_directory",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Path):
                raise TypeError(f"{name} must be a pathlib.Path")
        if self.stop_after_global_step is not None and (
            type(self.stop_after_global_step) is not int or self.stop_after_global_step < 0
        ):
            raise ExecutionContractError("stop_after_global_step must be nonnegative")

    @property
    def carries_factory_only_parameters(self) -> bool:
        """Whether a legacy already-built adapter would silently lose input."""

        path_values = (
            self.source_manifest,
            self.prepared_root,
            self.split_manifest,
            self.terminal_root,
            self.base_checkpoint,
            self.periodic_cache,
            self.cache_root,
            self.checkpoint_directory,
            self.resume_checkpoint,
            self.evaluation_manifest,
            self.statistical_report,
            self.attempt_directory,
        )
        return (
            any(value is not None for value in path_values)
            or self.stop_after_global_step is not None
            or (self.command in {"evaluate", "resume"} and self.system_id is not None)
        )


class RuntimeAdapterFactory(Protocol):
    """Trusted host seam binding parsed CLI parameters to one runtime adapter.

    A production host normally constructs a request-specific private backend,
    authenticates its receipt assertions, and returns a
    ``ProductionRuntimeAdapter``. The public executable never imports a backend
    from a path, endpoint, environment variable, or command-line string.
    """

    def __call__(self, request: CLICommandRequest) -> RuntimeAdapter:
        """Return the adapter bound to exactly this in-process request."""


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="public fail-closed training configuration",
    )


def _add_training_checkpoint_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint-dir",
        dest="checkpoint_directory",
        type=Path,
        help="private immutable checkpoint output directory (factory-only)",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="private checkpoint to resume after backend verification (factory-only)",
    )
    parser.add_argument(
        "--stop-after-global-step",
        type=int,
        help="controlled interruption point used by the checkpoint/resume runtime",
    )


def _load_cli_config(path: Path) -> bytes:
    try:
        return load_training_config(path)
    except FileNotFoundError:
        if path == _DEFAULT_CONFIG:
            return canonical_public_training_config_bytes()
        raise


def _build_command_request(namespace: argparse.Namespace) -> CLICommandRequest:
    command = namespace.command
    run_id = getattr(namespace, "run_id", None)
    seed = getattr(namespace, "seed", None)
    split = getattr(namespace, "split", None)
    system_id = getattr(namespace, "system_id", None)
    if command in {"run-base", "run-residual"}:
        role, _, derived_system_id = parse_run_id(run_id)
        expected_role = "BASE_QUALIFICATION" if command == "run-base" else "RESIDUAL_TRAIN"
        if role != expected_role:
            raise ExecutionContractError("run_id role does not match the CLI command")
        if system_id is not None and system_id != derived_system_id:
            raise ExecutionContractError("explicit system_id differs from run_id")
        system_id = derived_system_id
    elif command == "build-periodic-cache" and seed not in SEEDS:
        raise ExecutionContractError("periodic cache seed is outside the frozen census")
    return CLICommandRequest(
        command=command,
        run_id=run_id,
        seed=seed,
        split=split,
        system_id=system_id,
        source_manifest=getattr(namespace, "source_manifest", None),
        prepared_root=getattr(namespace, "prepared_root", None),
        split_manifest=getattr(namespace, "split_manifest", None),
        terminal_root=getattr(namespace, "terminal_root", None),
        base_checkpoint=getattr(namespace, "base_checkpoint", None),
        periodic_cache=getattr(namespace, "periodic_cache", None),
        cache_root=getattr(namespace, "cache_root", None),
        checkpoint_directory=getattr(namespace, "checkpoint_directory", None),
        resume_checkpoint=getattr(namespace, "resume_checkpoint", None),
        stop_after_global_step=getattr(namespace, "stop_after_global_step", None),
        evaluation_manifest=getattr(namespace, "evaluation_manifest", None),
        statistical_report=getattr(namespace, "statistical_report", None),
        attempt_directory=getattr(namespace, "attempt_dir", None),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phaseset",
        description="PhaseSet authority-zero execution protocol",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="enumerate external prerequisite holds")
    _add_config(preflight)

    prepare_data = subparsers.add_parser(
        "prepare-data", help="request registered private data preparation"
    )
    _add_config(prepare_data)
    prepare_data.add_argument("--source-manifest", type=Path)
    prepare_data.add_argument("--prepared-root", type=Path)

    audit_split = subparsers.add_parser("audit-split", help="audit a private split manifest")
    _add_config(audit_split)
    audit_split.add_argument("--split-manifest", type=Path)

    run_base = subparsers.add_parser("run-base", help="request one registered base run")
    _add_config(run_base)
    run_base.add_argument("--run-id", required=True)
    run_base.add_argument("--system-id", choices=_BASE_SYSTEM_IDS)
    _add_training_checkpoint_options(run_base)

    qualify = subparsers.add_parser("qualify-base", help="qualify all nine base terminals")
    _add_config(qualify)
    qualify.add_argument("--terminal-root", type=Path)

    cache = subparsers.add_parser(
        "build-periodic-cache", help="request a qualified seed-specific periodic cache"
    )
    _add_config(cache)
    cache.add_argument("--seed", required=True, type=int)
    cache.add_argument("--base-checkpoint", type=Path)
    cache.add_argument("--cache-root", type=Path)

    residual = subparsers.add_parser("run-residual", help="request one registered residual run")
    _add_config(residual)
    residual.add_argument("--run-id", required=True)
    residual.add_argument("--system-id", choices=_RESIDUAL_SYSTEM_IDS)
    residual.add_argument("--base-checkpoint", type=Path)
    residual.add_argument("--periodic-cache", type=Path)
    _add_training_checkpoint_options(residual)

    evaluate = subparsers.add_parser(
        "evaluate", help="request validation or sealed-test evaluation"
    )
    _add_config(evaluate)
    evaluate.add_argument("--split", required=True, choices=("validation", "test"))
    evaluate.add_argument("--system-id", choices=_FINAL_SYSTEM_IDS)
    evaluate.add_argument("--terminal-root", type=Path)

    bootstrap = subparsers.add_parser(
        "bootstrap", help="request the registered 100k paired-capture bootstrap"
    )
    _add_config(bootstrap)
    bootstrap.add_argument("--evaluation-manifest", type=Path)

    render = subparsers.add_parser(
        "render-paper", help="render only from a frozen statistical aggregate"
    )
    _add_config(render)
    render.add_argument("--statistical-report", type=Path)

    resume = subparsers.add_parser("resume", help="request a new attempt from a checkpoint")
    _add_config(resume)
    resume.add_argument("--attempt-dir", type=Path)
    resume.add_argument("--checkpoint", dest="resume_checkpoint", type=Path)
    resume.add_argument("--system-id", choices=_RESUME_SYSTEM_IDS)
    resume.add_argument("--base-checkpoint", type=Path)
    resume.add_argument("--periodic-cache", type=Path)

    return parser


def _emit(raw: bytes, stream: TextIO) -> None:
    stream.write(raw.decode("ascii"))
    stream.flush()


def _hold_bytes(command: str, hold_codes: tuple[str, ...]) -> bytes:
    return _canonical_bytes(
        {
            "authority": AUTHORITY,
            "command": command,
            "execution_authorized": False,
            "external_receipt_verified": False,
            "hold_codes": list(hold_codes),
            "production": False,
            "result_claimed": False,
            "status": "HELD",
        }
    )


def _contract_error_bytes(command: str, code: str) -> bytes:
    return _canonical_bytes(
        {
            "authority": AUTHORITY,
            "command": command,
            "error_code": code,
            "execution_authorized": False,
            "production": False,
            "result_claimed": False,
            "status": "REJECTED",
        }
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    runtime_adapter: RuntimeAdapter | None = None,
    runtime_adapter_factory: RuntimeAdapterFactory | None = None,
) -> int:
    """Parse one command and dispatch only through an injected, admitted runtime.

    ``runtime_adapter`` remains the compatibility seam for path-free adapters.
    Hosts that need private paths, a system selection, or checkpoint/resume
    controls must inject ``runtime_adapter_factory`` so the parsed request is
    actually bound before admission. The normal installed CLI supplies neither
    seam and therefore remains authority zero and held.
    """

    output = sys.stdout if stdout is None else stdout
    errors = sys.stderr if stderr is None else stderr
    parser = build_parser()
    try:
        namespace = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)

    command = namespace.command
    if command not in COMMANDS:
        _emit(_contract_error_bytes(str(command), "COMMAND_OUTSIDE_CENSUS"), errors)
        return EXIT_USAGE_OR_CONTRACT
    try:
        request = _build_command_request(namespace)
        if runtime_adapter is not None and runtime_adapter_factory is not None:
            raise ExecutionContractError("only one runtime injection seam may be used")
        config_raw = _load_cli_config(namespace.config)
        selected_adapter = runtime_adapter
        if runtime_adapter_factory is not None:
            if not callable(runtime_adapter_factory):
                raise TypeError("runtime_adapter_factory must be callable")
            selected_adapter = runtime_adapter_factory(request)
        elif runtime_adapter is not None and request.carries_factory_only_parameters:
            raise ExecutionContractError(
                "private CLI parameters require a request-aware runtime adapter factory"
            )
        runner = DataFreeRunner(config_raw, runtime_adapter=selected_adapter)
        if command == "preflight":
            report = runner.preflight()
            if type(report) is RuntimeAdmission:
                _emit(canonical_runtime_admission_bytes(report), output)
                return EXIT_OK
            _emit(canonical_preflight_bytes(report), output)
            return EXIT_OK if report.ready else EXIT_HOLD

        result = runner.execute_command(
            command,
            run_id=getattr(namespace, "run_id", None),
            seed=getattr(namespace, "seed", None),
            split=getattr(namespace, "split", None),
        )
        _emit(canonical_command_result_bytes(result), output)
        if result.outcome == "COMPLETED":
            return EXIT_OK
        if result.outcome == "HELD":
            return EXIT_HOLD
        return EXIT_EXECUTION_FAILED
    except FileNotFoundError:
        _emit(_hold_bytes(command, ("HOLD_TRAINING_CONFIG_ABSENT",)), output)
        return EXIT_HOLD
    except ExecutionHold as exc:
        _emit(_hold_bytes(command, exc.hold_codes), output)
        return EXIT_HOLD
    except (ExecutionContractError, TypeError, ValueError):
        _emit(_contract_error_bytes(command, "EXECUTION_CONTRACT_REJECTED"), errors)
        return EXIT_USAGE_OR_CONTRACT
    except Exception:
        _emit(_contract_error_bytes(command, "RUNTIME_ADAPTER_FAILED_CLOSED"), errors)
        return EXIT_EXECUTION_FAILED

    _emit(_contract_error_bytes(command, "PUBLIC_RUNNER_CANNOT_EXECUTE"), errors)
    return EXIT_USAGE_OR_CONTRACT


if __name__ == "__main__":  # pragma: no cover - exercised through main in tests
    raise SystemExit(main())


__all__ = [
    "CLICommandRequest",
    "EXIT_HOLD",
    "EXIT_OK",
    "EXIT_USAGE_OR_CONTRACT",
    "RuntimeAdapterFactory",
    "build_parser",
    "main",
]
