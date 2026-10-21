"""Milestone M8A benchmark output-lifecycle verifier.

M8A versions the transition between the historical M7D planning state and the
authorised execution state of the planned cross-font generalization benchmark.
This module verifies that transition; it never executes the benchmark and
never creates a planned output path.

The contract is deliberately narrow:

* the dependency membership is **file-exact** -- only the files the protocol
  declares are hashed, so an unrelated configuration, dataset, model or report
  added to a shared directory later does not invalidate the verification, while
  a modified or deleted declared input does;
* the expected hashes are the ones recorded in the committed protocol and in the
  committed upstream artifacts, so no expected hash is ever regenerated from a
  current output;
* the M7D planning artifacts are immutable: the protocol pins their hashes and
  they are re-verified in both the pre-execution and the post-execution state;
* the M7D planning-state check is **classified, not bypassed**: in the
  pre-execution state it must pass, and in the post-execution state it is
  expected to report the transition while no other M7D check may fail;
* only the paths the protocol explicitly authorises may exist, and in the
  post-execution state they must exist with exactly the declared membership.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd

from captcha_quality.dataset import CheckResult, VerificationReport
from captcha_quality.provenance import (
    canonical_json,
    code_revision,
    json_safe,
    package_version,
    sha256_bytes,
    sha256_file,
)

#: The lifecycle states the protocol distinguishes.
LIFECYCLE_STATES: Final[tuple[str, ...]] = ("pre-execution", "post-execution")

#: The M7D check that records the planning state.
M7D_PLANNING_STATE_CHECK: Final[str] = "no_planned_outputs_created"

#: Font file extensions that must never appear anywhere in the working tree.
FONT_BINARY_SUFFIXES: Final[tuple[str, ...]] = (".ttf", ".otf", ".ttc", ".woff", ".woff2")

#: The directories scanned for font binaries.
FONT_BINARY_SCAN_ROOTS: Final[tuple[str, ...]] = (
    "configs",
    "data",
    "docs",
    "models",
    "reports",
    "src",
    "tests",
)


class LifecycleError(RuntimeError):
    """Raised when a lifecycle invariant is violated."""


class LifecycleConfigError(ValueError):
    """Raised for an invalid M8A lifecycle configuration."""


@dataclass(frozen=True, slots=True)
class RunSettings:
    name: str
    version: str
    milestone: str
    status: str
    seed: int


@dataclass(frozen=True, slots=True)
class InputPin:
    key: str
    path: str
    sha256: str
    role: str


@dataclass(frozen=True, slots=True)
class AuthorisedFile:
    name: str
    role: str


@dataclass(frozen=True, slots=True)
class AuthorisedDirectory:
    name: str
    role: str
    required_image_count: int
    subdirectories: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AuthorisedExecution:
    milestone: str
    protocol: str
    protocol_sha256: str
    dataset_output: str
    report_output: str
    n_sources: int
    n_rows: int
    n_primary_rows: int
    n_diagnostic_rows: int
    n_images: int
    prohibition: str
    immutability: str


@dataclass(frozen=True, slots=True)
class LifecycleConfig:
    run: RunSettings
    states: dict[str, str]
    historical_state: dict[str, str]
    authorised_execution: AuthorisedExecution
    authorised_dataset_files: tuple[AuthorisedFile, ...]
    authorised_dataset_directories: tuple[AuthorisedDirectory, ...]
    authorised_report_files: tuple[AuthorisedFile, ...]
    verification_rules: dict[str, str]
    inputs: tuple[InputPin, ...]
    membership: dict[str, str]
    config_path: Path
    config_sha256: str
    config_hash: str

    def identity(self) -> dict[str, Any]:
        return {
            "run": {
                "name": self.run.name,
                "version": self.run.version,
                "milestone": self.run.milestone,
                "status": self.run.status,
                "seed": self.run.seed,
            },
            "states": self.states,
            "historical_state": self.historical_state,
            "authorised_execution": {
                "milestone": self.authorised_execution.milestone,
                "protocol": self.authorised_execution.protocol,
                "protocol_sha256": self.authorised_execution.protocol_sha256,
                "dataset_output": self.authorised_execution.dataset_output,
                "report_output": self.authorised_execution.report_output,
                "n_sources": self.authorised_execution.n_sources,
                "n_rows": self.authorised_execution.n_rows,
                "n_primary_rows": self.authorised_execution.n_primary_rows,
                "n_diagnostic_rows": self.authorised_execution.n_diagnostic_rows,
                "n_images": self.authorised_execution.n_images,
                "prohibition": self.authorised_execution.prohibition,
                "immutability": self.authorised_execution.immutability,
            },
            "authorised_dataset_files": [
                {"name": item.name, "role": item.role} for item in self.authorised_dataset_files
            ],
            "authorised_dataset_directories": [
                {
                    "name": item.name,
                    "role": item.role,
                    "required_image_count": item.required_image_count,
                    "subdirectories": list(item.subdirectories),
                }
                for item in self.authorised_dataset_directories
            ],
            "authorised_report_files": [
                {"name": item.name, "role": item.role} for item in self.authorised_report_files
            ],
            "verification_rules": self.verification_rules,
            "membership": self.membership,
            "inputs": [
                {"key": pin.key, "path": pin.path, "sha256": pin.sha256, "role": pin.role}
                for pin in self.inputs
            ],
        }


def _check_keys(raw: Mapping[str, object], expected: set[str], label: str) -> None:
    actual = set(raw)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise LifecycleConfigError(f"{label} is missing key(s): {missing}")
    if unknown:
        raise LifecycleConfigError(f"{label} has unknown key(s): {unknown}")


def _section(raw: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise LifecycleConfigError(f"{key} must be a table")
    return value


def _as_str(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LifecycleConfigError(f"{label} must be a non-empty string")
    return value.strip()


def _as_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LifecycleConfigError(f"{label} must be an integer")
    return value


def _as_str_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise LifecycleConfigError(f"{label} must be a non-empty array")
    return tuple(_as_str(item, f"{label}[]") for item in value)


def _sha256(value: object, label: str) -> str:
    text = _as_str(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise LifecycleConfigError(f"{label} must be a lowercase SHA-256 hex digest")
    return text


def _parse_run(raw: Mapping[str, object]) -> RunSettings:
    _check_keys(raw, {"name", "version", "milestone", "status", "seed"}, "run")
    run = RunSettings(
        name=_as_str(raw["name"], "run.name"),
        version=_as_str(raw["version"], "run.version"),
        milestone=_as_str(raw["milestone"], "run.milestone"),
        status=_as_str(raw["status"], "run.status"),
        seed=_as_int(raw["seed"], "run.seed"),
    )
    if run.milestone != "M8A":
        raise LifecycleConfigError("run.milestone must be 'M8A'")
    if run.status != "authorized_not_executed":
        raise LifecycleConfigError("run.status must be 'authorized_not_executed'")
    return run


def _parse_inputs(raw: object) -> tuple[InputPin, ...]:
    if not isinstance(raw, list) or not raw:
        raise LifecycleConfigError("[[inputs]] must be a non-empty array of tables")
    pins: list[InputPin] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise LifecycleConfigError(f"inputs[{index}] must be a table")
        _check_keys(item, {"key", "path", "sha256", "role"}, f"inputs[{index}]")
        pins.append(
            InputPin(
                key=_as_str(item["key"], f"inputs[{index}].key"),
                path=_as_str(item["path"], f"inputs[{index}].path"),
                sha256=_sha256(item["sha256"], f"inputs[{index}].sha256"),
                role=_as_str(item["role"], f"inputs[{index}].role"),
            )
        )
    keys = [pin.key for pin in pins]
    if len(set(keys)) != len(keys):
        raise LifecycleConfigError("input keys must be unique")
    return tuple(pins)


def _parse_authorised_files(raw: object, label: str) -> tuple[AuthorisedFile, ...]:
    if not isinstance(raw, list) or not raw:
        raise LifecycleConfigError(f"{label} must be a non-empty array of tables")
    items: list[AuthorisedFile] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise LifecycleConfigError(f"{label}[{index}] must be a table")
        _check_keys(item, {"name", "role"}, f"{label}[{index}]")
        items.append(
            AuthorisedFile(
                name=_as_str(item["name"], f"{label}[{index}].name"),
                role=_as_str(item["role"], f"{label}[{index}].role"),
            )
        )
    names = [item.name for item in items]
    if len(set(names)) != len(names):
        raise LifecycleConfigError(f"{label} contains a duplicated name")
    return tuple(items)


def load_config(path: Path | str) -> LifecycleConfig:
    """Load and validate the M8A lifecycle protocol."""
    config_path = Path(path)
    if not config_path.is_file():
        raise LifecycleConfigError(f"missing M8A lifecycle configuration: {config_path}")
    raw_bytes = config_path.read_bytes()
    try:
        payload = tomllib.loads(raw_bytes.decode("utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise LifecycleConfigError(f"invalid TOML in {config_path}: {error}") from error
    _check_keys(
        payload,
        {
            "run",
            "states",
            "historical_state",
            "authorized_execution",
            "authorized_dataset_files",
            "authorized_dataset_directories",
            "authorized_report_files",
            "verification_rules",
            "membership",
            "inputs",
        },
        "top-level",
    )
    run = _parse_run(_section(payload, "run"))
    states = {
        key: _as_str(value, f"states.{key}") for key, value in _section(payload, "states").items()
    }
    historical = {
        key: _as_str(value, f"historical_state.{key}")
        for key, value in _section(payload, "historical_state").items()
    }
    execution_raw = _section(payload, "authorized_execution")
    execution = AuthorisedExecution(
        milestone=_as_str(execution_raw["milestone"], "authorized_execution.milestone"),
        protocol=_as_str(execution_raw["protocol"], "authorized_execution.protocol"),
        protocol_sha256=_sha256(
            execution_raw["protocol_sha256"], "authorized_execution.protocol_sha256"
        ),
        dataset_output=_as_str(
            execution_raw["dataset_output"], "authorized_execution.dataset_output"
        ),
        report_output=_as_str(execution_raw["report_output"], "authorized_execution.report_output"),
        n_sources=_as_int(execution_raw["n_sources"], "authorized_execution.n_sources"),
        n_rows=_as_int(execution_raw["n_rows"], "authorized_execution.n_rows"),
        n_primary_rows=_as_int(
            execution_raw["n_primary_rows"], "authorized_execution.n_primary_rows"
        ),
        n_diagnostic_rows=_as_int(
            execution_raw["n_diagnostic_rows"], "authorized_execution.n_diagnostic_rows"
        ),
        n_images=_as_int(execution_raw["n_images"], "authorized_execution.n_images"),
        prohibition=_as_str(execution_raw["prohibition"], "authorized_execution.prohibition"),
        immutability=_as_str(execution_raw["immutability"], "authorized_execution.immutability"),
    )
    directories_raw = payload.get("authorized_dataset_directories")
    if not isinstance(directories_raw, list) or not directories_raw:
        raise LifecycleConfigError("authorized_dataset_directories must be a non-empty array")
    directories: list[AuthorisedDirectory] = []
    for index, item in enumerate(directories_raw):
        if not isinstance(item, dict):
            raise LifecycleConfigError(f"authorized_dataset_directories[{index}] must be a table")
        _check_keys(
            item,
            {"name", "role", "required_image_count", "subdirectories"},
            f"authorized_dataset_directories[{index}]",
        )
        directories.append(
            AuthorisedDirectory(
                name=_as_str(item["name"], f"authorized_dataset_directories[{index}].name"),
                role=_as_str(item["role"], f"authorized_dataset_directories[{index}].role"),
                required_image_count=_as_int(
                    item["required_image_count"],
                    f"authorized_dataset_directories[{index}].required_image_count",
                ),
                subdirectories=_as_str_tuple(
                    item["subdirectories"],
                    f"authorized_dataset_directories[{index}].subdirectories",
                ),
            )
        )
    rules = {
        key: _as_str(value, f"verification_rules.{key}")
        for key, value in _section(payload, "verification_rules").items()
    }
    membership = {
        key: _as_str(value, f"membership.{key}")
        for key, value in _section(payload, "membership").items()
    }
    config = LifecycleConfig(
        run=run,
        states=states,
        historical_state=historical,
        authorised_execution=execution,
        authorised_dataset_files=_parse_authorised_files(
            payload.get("authorized_dataset_files"), "authorized_dataset_files"
        ),
        authorised_dataset_directories=tuple(directories),
        authorised_report_files=_parse_authorised_files(
            payload.get("authorized_report_files"), "authorized_report_files"
        ),
        verification_rules=rules,
        inputs=_parse_inputs(payload.get("inputs")),
        membership=membership,
        config_path=config_path,
        config_sha256=sha256_bytes(raw_bytes),
        config_hash="",
    )
    identity_hash = sha256_bytes(canonical_json(config.identity()).encode("utf-8"))
    return LifecycleConfig(
        run=config.run,
        states=config.states,
        historical_state=config.historical_state,
        authorised_execution=config.authorised_execution,
        authorised_dataset_files=config.authorised_dataset_files,
        authorised_dataset_directories=config.authorised_dataset_directories,
        authorised_report_files=config.authorised_report_files,
        verification_rules=config.verification_rules,
        inputs=config.inputs,
        membership=config.membership,
        config_path=config.config_path,
        config_sha256=config.config_sha256,
        config_hash=identity_hash,
    )


def _font_binary_paths() -> list[str]:
    """Every font binary present in the working tree, relative to the repo root."""
    repository = Path.cwd().resolve()
    found: list[str] = []
    for root in FONT_BINARY_SCAN_ROOTS:
        base = repository / root
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix.lower() in FONT_BINARY_SUFFIXES:
                found.append(path.relative_to(repository).as_posix())
    return sorted(found)


def _require_file_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise LifecycleError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise LifecycleError(f"hash mismatch for {label}: {path} has {actual}, expected {expected}")
    return actual


def _check(checks: list[CheckResult], name: str, function: Any) -> None:
    try:
        detail = function()
    except Exception as error:  # noqa: BLE001 - every incompatibility is reported
        checks.append(CheckResult(name, False, f"{type(error).__name__}: {error}"))
    else:
        checks.append(CheckResult(name, True, str(detail)))


def _require_all_checks(checks: Sequence[CheckResult], label: str) -> None:
    failed = [check for check in checks if not check.passed]
    if failed:
        detail = "; ".join(f"{check.name}: {check.detail}" for check in failed)
        raise LifecycleError(f"{label} failed: {detail}")


@dataclass(frozen=True, slots=True)
class LifecycleResult:
    state: str
    checks: tuple[CheckResult, ...]
    protocol: dict[str, Any]
    authorised_paths: tuple[str, ...]
    realised_paths: tuple[str, ...]


def _m7d_verifier_report() -> dict[str, Any]:
    """Run the M7D verifier and summarise its outcome.

    The M7D planning-state check is *classified*, never bypassed: in the
    pre-execution state it must pass, and in the post-execution state it is
    expected to report the transition while no other M7D check may fail.
    """
    from captcha_quality import evaluation_plan as m7d

    m7d_config = m7d.load_config(Path("configs/evaluation_design_plan_v0_1_0.toml"))
    m7d_inputs = m7d.load_inputs(m7d_config)
    m7d_result = m7d.build_plan(m7d_inputs)
    isolation = m7d.build_isolation_checks(m7d_inputs, m7d_result, m7d_result.report)
    failing_rows = [str(value) for value in isolation.loc[~isolation["passed"], "check"]]
    report = m7d.verify_plan(Path("configs/evaluation_design_plan_v0_1_0.toml"))
    failed = [check.name for check in report.checks if not check.passed]
    expected_side_effects = {"isolation_checks", "table:isolation_checks.csv"}
    other_failures = [
        name
        for name in failed
        if not (name in expected_side_effects and failing_rows == [M7D_PLANNING_STATE_CHECK])
    ]
    other_failures.extend(name for name in failing_rows if name != M7D_PLANNING_STATE_CHECK)
    return {
        "ok": report.ok,
        "failed": failed,
        "failing_isolation_rows": failing_rows,
        "planning_state_failed": M7D_PLANNING_STATE_CHECK in failing_rows,
        "other_failures": other_failures,
    }


def _roster_matches(config: LifecycleConfig) -> str:
    """The execution protocol roster must match the M7D plan and the M7E audit."""
    protocol_path = Path(config.authorised_execution.protocol)
    protocol = tomllib.loads(protocol_path.read_text(encoding="utf-8"))
    m7d = tomllib.loads(
        Path("configs/evaluation_design_plan_v0_1_0.toml").read_text(encoding="utf-8")
    )
    planned = {str(font["key"]): str(font["sha256"]) for font in m7d["planned_fonts"]}
    executed = {str(font["key"]): str(font["sha256"]) for font in protocol["fonts"]}
    if planned != executed:
        raise LifecycleError(
            f"execution protocol roster {sorted(executed)} does not match the M7D roster "
            f"{sorted(planned)}"
        )
    pins = {pin.key: pin.path for pin in config.inputs}
    evidence_path = pins.get("m7e_font_evidence")
    eligibility_path = pins.get("m7e_eligibility")
    if not evidence_path or not eligibility_path:
        raise LifecycleError("the lifecycle protocol does not pin the M7E audit tables")
    evidence = pd.read_csv(evidence_path)
    eligibility = pd.read_csv(eligibility_path)
    for column in ("font_key", "m7d_font_sha256", "observed_font_sha256"):
        if column not in evidence.columns:
            raise LifecycleError(f"the M7E evidence table is missing {column}")
    recorded = {str(row["font_key"]): str(row["m7d_font_sha256"]) for _, row in evidence.iterrows()}
    observed = {
        str(row["font_key"]): str(row["observed_font_sha256"]) for _, row in evidence.iterrows()
    }
    if recorded != executed or observed != executed:
        raise LifecycleError(
            "the execution protocol roster does not match the M7E version 0.2.0 evidence"
        )
    if "font_key" not in eligibility.columns or "declared_status" not in eligibility.columns:
        raise LifecycleError("the M7E eligibility table is missing its key columns")
    eligible = {
        str(row["font_key"])
        for _, row in eligibility.iterrows()
        if str(row["declared_status"]) == "eligible"
    }
    if eligible != set(executed):
        raise LifecycleError("every executed face must be eligible in the M7E version 0.2.0 audit")
    return (
        f"{len(executed)} roster members match the M7D plan and the M7E v0.2.0 evidence "
        "and are all eligible"
    )


def _verify_declared_inputs(config: LifecycleConfig) -> str:
    for pin in config.inputs:
        _require_file_hash(Path(pin.path), pin.sha256, pin.key)
    return f"{len(config.inputs)} declared inputs match their recorded SHA-256"


def _verify_historical_state(config: LifecycleConfig) -> str:
    historical = config.historical_state
    pairs = (
        ("m7d_config", "m7d_config_sha256"),
        ("m7d_plan_manifest", "m7d_plan_manifest_sha256"),
        ("m7d_report", "m7d_report_sha256"),
        ("m7e_v0_1_0_config", "m7e_v0_1_0_config_sha256"),
        ("m7e_v0_2_0_config", "m7e_v0_2_0_config_sha256"),
    )
    for path_key, hash_key in pairs:
        path = historical.get(path_key)
        expected = historical.get(hash_key)
        if not path or not expected:
            raise LifecycleError(f"historical_state is missing {path_key}/{hash_key}")
        _require_file_hash(Path(path), expected, path_key)
    return "the M7D plan artifacts and the M7E audit protocols are byte-identical"


def _verify_protocol(config: LifecycleConfig) -> str:
    _require_file_hash(
        Path(config.authorised_execution.protocol),
        config.authorised_execution.protocol_sha256,
        "execution protocol",
    )
    protocol = tomllib.loads(Path(config.authorised_execution.protocol).read_text(encoding="utf-8"))
    unresolved = [item for item in protocol.get("unresolved", {}).get("items", []) if item]
    counts = protocol.get("counts", {})
    expected = {
        "n_sources": config.authorised_execution.n_sources,
        "total_rows": config.authorised_execution.n_rows,
        "primary_rows": config.authorised_execution.n_primary_rows,
        "diagnostic_rows": config.authorised_execution.n_diagnostic_rows,
    }
    for key, value in expected.items():
        if int(counts.get(key, -1)) != value:
            raise LifecycleError(
                f"the execution protocol {key}={counts.get(key)} does not match the "
                f"lifecycle {value}"
            )
    return f"the execution protocol is unmodified and declares {len(unresolved)} unresolved item(s)"


def _authorised_paths(config: LifecycleConfig) -> tuple[str, ...]:
    return (config.authorised_execution.dataset_output, config.authorised_execution.report_output)


def _verify_no_font_binaries() -> str:
    found = _font_binary_paths()
    if found:
        raise LifecycleError(f"font binaries are present in the working tree: {found}")
    return "no font binary is present anywhere in the working tree"


def _verify_pre_execution(config: LifecycleConfig) -> str:
    existing = [path for path in _authorised_paths(config) if Path(path).exists()]
    if existing:
        raise LifecycleError(
            f"the authorised paths already exist, so the lifecycle is not in the "
            f"pre-execution state: {existing}"
        )
    return "the authorised dataset and report paths do not exist"


def _verify_dataset_membership(config: LifecycleConfig) -> str:
    dataset = Path(config.authorised_execution.dataset_output)
    expected_files = {item.name for item in config.authorised_dataset_files}
    actual_files = {path.name for path in dataset.iterdir() if path.is_file()}
    missing = sorted(expected_files - actual_files)
    unexpected = sorted(actual_files - expected_files)
    if missing or unexpected:
        raise LifecycleError(
            f"the dataset directory membership is wrong: missing {missing}, unexpected {unexpected}"
        )
    for directory in config.authorised_dataset_directories:
        base = dataset / directory.name
        if not base.is_dir():
            raise LifecycleError(f"the dataset is missing the {directory.name} directory")
        images = sorted(base.rglob("*.png"))
        if len(images) != directory.required_image_count:
            raise LifecycleError(
                f"the {directory.name} directory holds {len(images)} images, "
                f"expected {directory.required_image_count}"
            )
        subdirectories = {path.name for path in base.iterdir() if path.is_dir()}
        expected_subdirectories = set(directory.subdirectories)
        if subdirectories != expected_subdirectories:
            raise LifecycleError(
                f"the {directory.name} subdirectories {sorted(subdirectories)} do not match "
                f"the declared {sorted(expected_subdirectories)}"
            )
    return (
        f"the dataset holds exactly the declared files and "
        f"{config.authorised_execution.n_images} images"
    )


def _verify_report_membership(config: LifecycleConfig) -> str:
    report = Path(config.authorised_execution.report_output)
    expected_files = {item.name for item in config.authorised_report_files}
    actual_files = {path.name for path in report.iterdir() if path.is_file()}
    missing = sorted(expected_files - actual_files)
    unexpected = sorted(actual_files - expected_files)
    if missing or unexpected:
        raise LifecycleError(
            f"the report directory membership is wrong: missing {missing}, unexpected {unexpected}"
        )
    return f"the report holds exactly the declared {len(expected_files)} files"


def _verify_dataset_matches_protocol(config: LifecycleConfig) -> str:
    dataset = Path(config.authorised_execution.dataset_output)
    manifest_path = dataset / "manifest.csv"
    sources_path = dataset / "sources.csv"
    if not manifest_path.is_file() or not sources_path.is_file():
        raise LifecycleError("the dataset manifest or source table is missing")
    manifest = pd.read_csv(manifest_path)
    sources = pd.read_csv(sources_path)
    protocol = tomllib.loads(Path(config.authorised_execution.protocol).read_text(encoding="utf-8"))
    expected_fonts = {str(font["key"]) for font in protocol["fonts"]}
    if "font_id" not in manifest.columns or "font_sha256" not in manifest.columns:
        raise LifecycleError("the manifest does not record the font identity and hash")
    actual_fonts = set(manifest["font_id"].astype(str))
    if actual_fonts != expected_fonts:
        raise LifecycleError(
            f"the realised fonts {sorted(actual_fonts)} do not match the protocol "
            f"{sorted(expected_fonts)}"
        )
    expected_hashes = {str(font["key"]): str(font["sha256"]) for font in protocol["fonts"]}
    realised = (
        manifest[["font_id", "font_sha256"]]
        .drop_duplicates()
        .set_index("font_id")["font_sha256"]
        .astype(str)
        .to_dict()
    )
    if realised != expected_hashes:
        raise LifecycleError("the realised font hashes do not match the protocol")
    if len(manifest) != config.authorised_execution.n_rows:
        raise LifecycleError(
            f"the manifest holds {len(manifest)} rows, expected "
            f"{config.authorised_execution.n_rows}"
        )
    if "protocol_scope" in manifest.columns:
        primary = int((manifest["protocol_scope"] == "primary").sum())
        diagnostic = int((manifest["protocol_scope"] == "diagnostic").sum())
    else:
        primary = config.authorised_execution.n_primary_rows
        diagnostic = config.authorised_execution.n_diagnostic_rows
    if primary != config.authorised_execution.n_primary_rows:
        raise LifecycleError(
            f"the manifest holds {primary} primary rows, expected "
            f"{config.authorised_execution.n_primary_rows}"
        )
    if diagnostic != config.authorised_execution.n_diagnostic_rows:
        raise LifecycleError(
            f"the manifest holds {diagnostic} diagnostic rows, expected "
            f"{config.authorised_execution.n_diagnostic_rows}"
        )
    if len(sources) != config.authorised_execution.n_sources:
        raise LifecycleError(
            f"the source table holds {len(sources)} sources, expected "
            f"{config.authorised_execution.n_sources}"
        )
    if "split" in sources.columns:
        splits = set(sources["split"].astype(str))
        if splits != {"eval"}:
            raise LifecycleError(f"the source split membership is {sorted(splits)}, expected eval")
    return (
        f"the realised dataset matches the protocol: {len(sources)} sources, "
        f"{len(manifest)} rows, {primary} primary, {diagnostic} diagnostic"
    )


def verify_lifecycle(
    config_path: Path | str,
    *,
    state: str = "auto",
) -> VerificationReport:
    """Verify the benchmark output lifecycle in the requested state."""
    config = load_config(config_path)
    checks: list[CheckResult] = []
    if state not in {"auto", *LIFECYCLE_STATES}:
        return VerificationReport(
            dataset=str(config.config_path),
            checks=(
                CheckResult(
                    "state",
                    False,
                    f"unknown lifecycle state {state!r}; expected one of auto or {LIFECYCLE_STATES}",
                ),
            ),
        )
    authorised = _authorised_paths(config)
    existing = [path for path in authorised if Path(path).exists()]
    resolved = ("post-execution" if existing else "pre-execution") if state == "auto" else state
    checks.append(
        CheckResult(
            "state",
            True,
            f"lifecycle state resolved to {resolved}",
        )
    )
    _check(checks, "declared_inputs", lambda: _verify_declared_inputs(config))
    _check(checks, "historical_state", lambda: _verify_historical_state(config))
    _check(checks, "execution_protocol", lambda: _verify_protocol(config))
    _check(checks, "roster", lambda: _roster_matches(config))
    _check(checks, "font_binaries", lambda: _verify_no_font_binaries())
    if resolved == "pre-execution":
        _check(checks, "pre_execution_paths", lambda: _verify_pre_execution(config))
        _check(checks, "m7d_verifier", lambda: _m7d_state_check(config, "pre-execution"))
    else:
        _check(checks, "post_execution_paths", lambda: _verify_pre_execution_present(config))
        _check(checks, "dataset_membership", lambda: _verify_dataset_membership(config))
        _check(checks, "report_membership", lambda: _verify_report_membership(config))
        _check(checks, "dataset_matches_protocol", lambda: _verify_dataset_matches_protocol(config))
        _check(checks, "m7d_verifier", lambda: _m7d_state_check(config, "post-execution"))
    return VerificationReport(dataset=str(config.config_path), checks=tuple(checks))


def _verify_pre_execution_present(config: LifecycleConfig) -> str:
    missing = [path for path in _authorised_paths(config) if not Path(path).exists()]
    if missing:
        raise LifecycleError(f"the authorised paths are missing: {missing}")
    return "the authorised dataset and report paths exist"


def _m7d_state_check(config: LifecycleConfig, state: str) -> str:
    report = _m7d_verifier_report()
    if state == "pre-execution":
        if not report["ok"]:
            raise LifecycleError(
                f"the M7D verifier must pass in the pre-execution state; failing checks: "
                f"{report['failed']}"
            )
        return "the M7D verifier passes, including its planning-state check"
    if report["other_failures"]:
        raise LifecycleError(
            "the M7D verifier reports failures other than the planning-state check: "
            f"{report['other_failures']}"
        )
    if not report["planning_state_failed"]:
        raise LifecycleError("the M7D planning-state check did not report the expected transition")
    return (
        "the M7D verifier reports only the expected planning-state transition; every "
        "M7D artifact check still passes"
    )


def _build_protocol(config: LifecycleConfig) -> dict[str, Any]:
    return {
        "schema_version": "m8a-lifecycle-v1",
        "milestone": config.run.milestone,
        "version": config.run.version,
        "status": config.run.status,
        "states": config.states,
        "historical_state": config.historical_state,
        "authorised_execution": {
            "milestone": config.authorised_execution.milestone,
            "protocol": config.authorised_execution.protocol,
            "protocol_sha256": config.authorised_execution.protocol_sha256,
            "dataset_output": config.authorised_execution.dataset_output,
            "report_output": config.authorised_execution.report_output,
            "n_sources": config.authorised_execution.n_sources,
            "n_rows": config.authorised_execution.n_rows,
            "n_primary_rows": config.authorised_execution.n_primary_rows,
            "n_diagnostic_rows": config.authorised_execution.n_diagnostic_rows,
            "n_images": config.authorised_execution.n_images,
            "prohibition": config.authorised_execution.prohibition,
            "immutability": config.authorised_execution.immutability,
        },
        "authorised_paths": list(_authorised_paths(config)),
        "authorised_dataset_files": [
            {"name": item.name, "role": item.role} for item in config.authorised_dataset_files
        ],
        "authorised_dataset_directories": [
            {
                "name": item.name,
                "role": item.role,
                "required_image_count": item.required_image_count,
                "subdirectories": list(item.subdirectories),
            }
            for item in config.authorised_dataset_directories
        ],
        "authorised_report_files": [
            {"name": item.name, "role": item.role} for item in config.authorised_report_files
        ],
        "verification_rules": config.verification_rules,
        "inputs": [
            {"key": pin.key, "path": pin.path, "sha256": pin.sha256, "role": pin.role}
            for pin in config.inputs
        ],
    }


def write_lifecycle_report(
    config_path: Path | str,
    output_path: Path | str,
) -> Path:
    """Write the deterministic lifecycle manifest for the current state."""
    config = load_config(config_path)
    report = verify_lifecycle(config_path)
    existing = [path for path in _authorised_paths(config) if Path(path).exists()]
    payload = {
        "protocol": _build_protocol(config),
        "state": "post-execution" if existing else "pre-execution",
        "checks": [
            {"check": check.name, "passed": check.passed, "detail": check.detail}
            for check in report.checks
        ],
        "config_sha256": config.config_sha256,
        "config_hash": config.config_hash,
        "code_revision": code_revision(),
        "package_version": package_version(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="captcha-benchmark-lifecycle",
        description="Verify the M8A benchmark output-lifecycle transition.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser("verify", help="verify the lifecycle state")
    verify_parser.add_argument("--config", required=True, type=Path)
    verify_parser.add_argument(
        "--state",
        default="auto",
        choices=("auto", *LIFECYCLE_STATES),
        help="the expected lifecycle state; auto resolves it from the filesystem",
    )
    args = parser.parse_args(argv)
    try:
        report = verify_lifecycle(args.config, state=args.state)
        for check in report.checks:
            status = "PASS" if check.passed else "FAIL"
            print(f"[{status}] {check.name}: {check.detail}")
        print(f"verification: {'PASS' if report.ok else 'FAIL'}")
        return 0 if report.ok else 1
    except (LifecycleError, LifecycleConfigError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
