"""Milestone M8E benchmark provenance and ground-truth integrity verification.

This module is a maintenance contract around the completed M8C benchmark. It
does not execute the benchmark, generate an image, train or select a model, or
write a benchmark artifact. It supplies two deliberately separate modes:

``historical-evidence``
    Re-verify the recorded M8C ground truth, input manifest, run metadata,
    execution dependency closure and external environment. The historical run
    is *not* retroactively described as clean: M8C recorded ``75c8ef4-dirty``.

``future-execution``
    Require a clean, committed working tree and absent output paths before any
    caller action runs. Future input manifests must have unique keys; the one
    historical ``text_case_manifest`` duplicate is allowed only because this
    contract declares and validates that exact multiplicity.

The expected ground-truth byte hash is a committed configuration value. It is
never computed from the current ground-truth file during verification. Error
messages report paths, counts and digests, never ground-truth text strings.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import platform
import stat
import subprocess
import sys
import tomllib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Final, TypeVar

import pandas as pd

from captcha_quality import matched_font_benchmark as mfb
from captcha_quality.benchmark_preflight import load_case_manifest, load_execution_protocol
from captcha_quality.dataset import CheckResult, VerificationReport
from captcha_quality.provenance import sha256_bytes, sha256_file

#: The explicit contract version. New verifier behaviour must use a new value.
INTEGRITY_CONTRACT: Final[str] = "m8e-integrity-v1"

#: The schema of the committed integrity configuration.
INTEGRITY_CONFIG_SCHEMA: Final[str] = "m8e-integrity-config-v1"

#: The default committed configuration path, relative to the repository root.
DEFAULT_CONFIG: Final[Path] = Path("configs/benchmark_integrity_v0_1_0.toml")

#: The exact input-manifest schema used by the M8C report.
INPUT_MANIFEST_COLUMNS: Final[tuple[str, ...]] = (
    "key",
    "path",
    "sha256",
    "expected_sha256",
    "matches",
    "role",
    "kind",
)

T = TypeVar("T")


class IntegrityError(RuntimeError):
    """Raised when a provenance or integrity invariant is violated."""


class IntegrityConfigError(ValueError):
    """Raised for an invalid M8E integrity configuration."""


@dataclass(frozen=True, slots=True)
class BenchmarkIdentity:
    """The historical benchmark whose evidence is being verified."""

    name: str
    dataset: str
    report: str
    protocol: str
    protocol_sha256: str
    lifecycle: str
    lifecycle_sha256: str
    run_metadata: str
    run_metadata_sha256: str
    root_module: str
    historical_execution_revision: str


@dataclass(frozen=True, slots=True)
class GroundTruthPin:
    """The committed byte-level ground-truth contract."""

    path: str
    sha256: str
    rows: int
    columns: tuple[str, ...]
    unique_columns: tuple[str, ...]
    regular_file: bool


@dataclass(frozen=True, slots=True)
class LegacyDuplicate:
    """An explicitly allowed historical manifest multiplicity."""

    key: str
    count: int
    identical_columns: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class InputManifestPin:
    """The historical input-manifest identity and key rules."""

    path: str
    sha256: str
    rows: int
    unique_keys: int
    key_column: str
    columns: tuple[str, ...]
    legacy_duplicates: tuple[LegacyDuplicate, ...]


@dataclass(frozen=True, slots=True)
class DependencyPin:
    """One exact in-repository file in the execution closure."""

    key: str
    path: str
    sha256: str
    role: str


@dataclass(frozen=True, slots=True)
class PackagePin:
    """One external distribution version and the modules it provides."""

    name: str
    version: str
    modules: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EnvironmentPin:
    """The runtime constraints that affect benchmark reproducibility."""

    python_version: str
    python_implementation: str
    system: str
    machine: str
    packages: tuple[PackagePin, ...]


@dataclass(frozen=True, slots=True)
class RepositoryPolicy:
    """The clean-tree requirements for future executions."""

    require_clean_worktree: bool
    require_committed_revision: bool
    require_absent_outputs: bool


@dataclass(frozen=True, slots=True)
class IntegrityConfig:
    """A validated M8E integrity configuration."""

    schema_version: str
    contract: str
    benchmark: BenchmarkIdentity
    ground_truth: GroundTruthPin
    input_manifest: InputManifestPin
    dependencies: tuple[DependencyPin, ...]
    environment: EnvironmentPin
    repository: RepositoryPolicy
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class RepositoryState:
    """The strict Git state used by the execution guard."""

    root: Path
    revision: str
    branch: str
    status: str

    @property
    def clean(self) -> bool:
        return not self.status.strip()


@dataclass(frozen=True, slots=True)
class ExecutionTicket:
    """Proof that a future execution passed the provenance preflight."""

    revision: str
    branch: str
    config_sha256: str
    output_paths: tuple[str, ...]


# ---------------------------------------------------------------------------
# configuration parsing
# ---------------------------------------------------------------------------


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise IntegrityConfigError(f"{label} must be a table")
    return value


def _check_keys(
    raw: Mapping[str, object],
    expected: set[str],
    label: str,
    *,
    optional: set[str] | None = None,
) -> None:
    allowed = expected | (optional or set())
    actual = set(raw)
    missing = sorted(expected - actual)
    unknown = sorted(actual - allowed)
    if missing:
        raise IntegrityConfigError(f"{label} is missing key(s): {missing}")
    if unknown:
        raise IntegrityConfigError(f"{label} has unknown key(s): {unknown}")


def _as_str(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IntegrityConfigError(f"{label} must be a non-empty string")
    return value.strip()


def _as_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise IntegrityConfigError(f"{label} must be a non-negative integer")
    return value


def _as_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise IntegrityConfigError(f"{label} must be a boolean")
    return value


def _as_str_list(value: object, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise IntegrityConfigError(f"{label} must be an array")
    if not value and not allow_empty:
        raise IntegrityConfigError(f"{label} must be a non-empty array")
    return tuple(_as_str(item, f"{label}[]") for item in value)


def _as_sha256(value: object, label: str) -> str:
    text = _as_str(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise IntegrityConfigError(f"{label} must be a lowercase SHA-256 hex digest")
    return text


def _as_commit(value: object, label: str) -> str:
    text = _as_str(value, label)
    if len(text) != 40 or any(character not in "0123456789abcdef" for character in text):
        raise IntegrityConfigError(f"{label} must be a lowercase full Git commit id")
    return text


def _safe_relative_path(value: object, label: str) -> str:
    text = _as_str(value, label)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts or path.name in {"", ".", ".."}:
        raise IntegrityConfigError(f"{label} must be a relative path inside the repository")
    return path.as_posix()


def _parse_benchmark(raw: Mapping[str, object]) -> BenchmarkIdentity:
    _check_keys(
        raw,
        {
            "name",
            "dataset",
            "report",
            "protocol",
            "protocol_sha256",
            "lifecycle",
            "lifecycle_sha256",
            "run_metadata",
            "run_metadata_sha256",
            "root_module",
            "historical_execution_revision",
        },
        "benchmark",
    )
    return BenchmarkIdentity(
        name=_as_str(raw["name"], "benchmark.name"),
        dataset=_safe_relative_path(raw["dataset"], "benchmark.dataset"),
        report=_safe_relative_path(raw["report"], "benchmark.report"),
        protocol=_safe_relative_path(raw["protocol"], "benchmark.protocol"),
        protocol_sha256=_as_sha256(raw["protocol_sha256"], "benchmark.protocol_sha256"),
        lifecycle=_safe_relative_path(raw["lifecycle"], "benchmark.lifecycle"),
        lifecycle_sha256=_as_sha256(raw["lifecycle_sha256"], "benchmark.lifecycle_sha256"),
        run_metadata=_safe_relative_path(raw["run_metadata"], "benchmark.run_metadata"),
        run_metadata_sha256=_as_sha256(raw["run_metadata_sha256"], "benchmark.run_metadata_sha256"),
        root_module=_safe_relative_path(raw["root_module"], "benchmark.root_module"),
        historical_execution_revision=_as_str(
            raw["historical_execution_revision"],
            "benchmark.historical_execution_revision",
        ),
    )


def _parse_ground_truth(raw: Mapping[str, object]) -> GroundTruthPin:
    _check_keys(
        raw,
        {
            "path",
            "sha256",
            "rows",
            "columns",
            "unique_columns",
            "regular_file",
        },
        "ground_truth",
    )
    columns = _as_str_list(raw["columns"], "ground_truth.columns")
    unique_columns = _as_str_list(raw["unique_columns"], "ground_truth.unique_columns")
    unknown = sorted(set(unique_columns) - set(columns))
    if unknown:
        raise IntegrityConfigError(
            f"ground_truth.unique_columns contains unknown column(s): {unknown}"
        )
    if not _as_bool(raw["regular_file"], "ground_truth.regular_file"):
        raise IntegrityConfigError("ground_truth.regular_file must be true")
    return GroundTruthPin(
        path=_safe_relative_path(raw["path"], "ground_truth.path"),
        sha256=_as_sha256(raw["sha256"], "ground_truth.sha256"),
        rows=_as_int(raw["rows"], "ground_truth.rows"),
        columns=columns,
        unique_columns=unique_columns,
        regular_file=True,
    )


def _parse_legacy_duplicates(raw: object) -> tuple[LegacyDuplicate, ...]:
    if not isinstance(raw, list):
        raise IntegrityConfigError("input_manifest.legacy_duplicates must be an array")
    items: list[LegacyDuplicate] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise IntegrityConfigError(f"input_manifest.legacy_duplicates[{index}] must be a table")
        _check_keys(
            item,
            {"key", "count", "identical_columns", "reason"},
            f"input_manifest.legacy_duplicates[{index}]",
        )
        count = _as_int(item["count"], f"input_manifest.legacy_duplicates[{index}].count")
        if count < 2:
            raise IntegrityConfigError(
                f"input_manifest.legacy_duplicates[{index}].count must be at least 2"
            )
        items.append(
            LegacyDuplicate(
                key=_as_str(item["key"], f"input_manifest.legacy_duplicates[{index}].key"),
                count=count,
                identical_columns=_as_str_list(
                    item["identical_columns"],
                    f"input_manifest.legacy_duplicates[{index}].identical_columns",
                ),
                reason=_as_str(item["reason"], f"input_manifest.legacy_duplicates[{index}].reason"),
            )
        )
    keys = [item.key for item in items]
    if len(set(keys)) != len(keys):
        raise IntegrityConfigError("input_manifest.legacy_duplicates contains a duplicate key")
    return tuple(items)


def _parse_input_manifest(raw: Mapping[str, object]) -> InputManifestPin:
    _check_keys(
        raw,
        {
            "path",
            "sha256",
            "rows",
            "unique_keys",
            "key_column",
            "columns",
            "legacy_duplicates",
        },
        "input_manifest",
    )
    columns = _as_str_list(raw["columns"], "input_manifest.columns")
    key_column = _as_str(raw["key_column"], "input_manifest.key_column")
    if key_column not in columns:
        raise IntegrityConfigError("input_manifest.key_column must be one of its columns")
    rows = _as_int(raw["rows"], "input_manifest.rows")
    unique_keys = _as_int(raw["unique_keys"], "input_manifest.unique_keys")
    if unique_keys > rows:
        raise IntegrityConfigError("input_manifest.unique_keys cannot exceed its rows")
    legacy = _parse_legacy_duplicates(raw["legacy_duplicates"])
    for duplicate in legacy:
        unknown = sorted(set(duplicate.identical_columns) - set(columns))
        if unknown:
            raise IntegrityConfigError(
                f"legacy duplicate {duplicate.key!r} names unknown column(s): {unknown}"
            )
        if key_column in duplicate.identical_columns:
            raise IntegrityConfigError(
                f"legacy duplicate {duplicate.key!r} must not compare the key column"
            )
    return InputManifestPin(
        path=_safe_relative_path(raw["path"], "input_manifest.path"),
        sha256=_as_sha256(raw["sha256"], "input_manifest.sha256"),
        rows=rows,
        unique_keys=unique_keys,
        key_column=key_column,
        columns=columns,
        legacy_duplicates=legacy,
    )


def _parse_dependencies(raw: object) -> tuple[DependencyPin, ...]:
    if not isinstance(raw, list) or not raw:
        raise IntegrityConfigError("[[dependencies]] must be a non-empty array of tables")
    dependencies: list[DependencyPin] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise IntegrityConfigError(f"dependencies[{index}] must be a table")
        _check_keys(item, {"key", "path", "sha256", "role"}, f"dependencies[{index}]")
        path = _safe_relative_path(item["path"], f"dependencies[{index}].path")
        if not path.startswith("src/captcha_quality/") or not path.endswith(".py"):
            raise IntegrityConfigError(
                f"dependencies[{index}].path must be a Python module under src/captcha_quality"
            )
        dependencies.append(
            DependencyPin(
                key=_as_str(item["key"], f"dependencies[{index}].key"),
                path=path,
                sha256=_as_sha256(item["sha256"], f"dependencies[{index}].sha256"),
                role=_as_str(item["role"], f"dependencies[{index}].role"),
            )
        )
    keys = [item.key for item in dependencies]
    paths = [item.path for item in dependencies]
    if len(set(keys)) != len(keys):
        raise IntegrityConfigError("dependency keys must be unique")
    if len(set(paths)) != len(paths):
        raise IntegrityConfigError("dependency paths must be unique")
    return tuple(dependencies)


def _parse_environment(raw: Mapping[str, object]) -> EnvironmentPin:
    _check_keys(
        raw,
        {
            "python_version",
            "python_implementation",
            "system",
            "machine",
            "packages",
        },
        "environment",
    )
    packages_raw = raw["packages"]
    if not isinstance(packages_raw, list) or not packages_raw:
        raise IntegrityConfigError("environment.packages must be a non-empty array")
    packages: list[PackagePin] = []
    for index, item in enumerate(packages_raw):
        if not isinstance(item, dict):
            raise IntegrityConfigError(f"environment.packages[{index}] must be a table")
        _check_keys(item, {"name", "version", "modules"}, f"environment.packages[{index}]")
        packages.append(
            PackagePin(
                name=_as_str(item["name"], f"environment.packages[{index}].name"),
                version=_as_str(item["version"], f"environment.packages[{index}].version"),
                modules=_as_str_list(
                    item["modules"],
                    f"environment.packages[{index}].modules",
                    allow_empty=True,
                ),
            )
        )
    names = [item.name for item in packages]
    if len(set(names)) != len(names):
        raise IntegrityConfigError("environment package names must be unique")
    modules = [module for item in packages for module in item.modules]
    if len(set(modules)) != len(modules):
        raise IntegrityConfigError("environment package modules must be uniquely declared")
    return EnvironmentPin(
        python_version=_as_str(raw["python_version"], "environment.python_version"),
        python_implementation=_as_str(
            raw["python_implementation"], "environment.python_implementation"
        ),
        system=_as_str(raw["system"], "environment.system"),
        machine=_as_str(raw["machine"], "environment.machine"),
        packages=tuple(packages),
    )


def _parse_repository(raw: Mapping[str, object]) -> RepositoryPolicy:
    _check_keys(
        raw,
        {"require_clean_worktree", "require_committed_revision", "require_absent_outputs"},
        "repository",
    )
    policy = RepositoryPolicy(
        require_clean_worktree=_as_bool(
            raw["require_clean_worktree"], "repository.require_clean_worktree"
        ),
        require_committed_revision=_as_bool(
            raw["require_committed_revision"], "repository.require_committed_revision"
        ),
        require_absent_outputs=_as_bool(
            raw["require_absent_outputs"], "repository.require_absent_outputs"
        ),
    )
    if not policy.require_clean_worktree:
        raise IntegrityConfigError("repository.require_clean_worktree must be true")
    if not policy.require_committed_revision:
        raise IntegrityConfigError("repository.require_committed_revision must be true")
    if not policy.require_absent_outputs:
        raise IntegrityConfigError("repository.require_absent_outputs must be true")
    return policy


def load_config(path: Path | str) -> IntegrityConfig:
    """Load and strictly validate an M8E integrity configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise IntegrityConfigError(f"missing M8E integrity configuration: {config_path}")
    raw_bytes = config_path.read_bytes()
    try:
        payload = tomllib.loads(raw_bytes.decode("utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise IntegrityConfigError(f"invalid TOML in {config_path}: {error}") from error
    _check_keys(
        payload,
        {
            "schema",
            "contract",
            "benchmark",
            "ground_truth",
            "input_manifest",
            "repository",
            "environment",
            "dependencies",
        },
        "top-level",
    )
    schema_version = _as_str(payload["schema"], "schema")
    contract = _as_str(payload["contract"], "contract")
    if schema_version != INTEGRITY_CONFIG_SCHEMA:
        raise IntegrityConfigError(
            f"unsupported integrity schema {schema_version!r}; expected {INTEGRITY_CONFIG_SCHEMA!r}"
        )
    if contract != INTEGRITY_CONTRACT:
        raise IntegrityConfigError(
            f"unsupported integrity contract {contract!r}; expected {INTEGRITY_CONTRACT!r}"
        )
    benchmark = _parse_benchmark(_require_mapping(payload["benchmark"], "benchmark"))
    dependencies = _parse_dependencies(payload["dependencies"])
    if benchmark.root_module not in {item.path for item in dependencies}:
        raise IntegrityConfigError("benchmark.root_module must be a declared dependency")
    return IntegrityConfig(
        schema_version=schema_version,
        contract=contract,
        benchmark=benchmark,
        ground_truth=_parse_ground_truth(_require_mapping(payload["ground_truth"], "ground_truth")),
        input_manifest=_parse_input_manifest(
            _require_mapping(payload["input_manifest"], "input_manifest")
        ),
        dependencies=dependencies,
        environment=_parse_environment(_require_mapping(payload["environment"], "environment")),
        repository=_parse_repository(_require_mapping(payload["repository"], "repository")),
        path=config_path,
        sha256=sha256_bytes(raw_bytes),
    )


# ---------------------------------------------------------------------------
# small verification helpers
# ---------------------------------------------------------------------------


def _repo_root(root: Path | str | None = None) -> Path:
    if root is None:
        return Path(__file__).resolve().parents[2]
    return Path(root).absolute()


def _absolute(root: Path, relative: str | Path) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else root / path


def _check(checks: list[CheckResult], name: str, function: Callable[[], Any]) -> None:
    try:
        detail = function()
    except Exception as error:  # noqa: BLE001 - every incompatibility is reported
        checks.append(CheckResult(name, False, f"{type(error).__name__}: {error}"))
    else:
        checks.append(CheckResult(name, True, str(detail)))


def _require_regular_file(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise IntegrityError(f"{label} is missing: {path}") from error
    except OSError as error:
        raise IntegrityError(f"{label} cannot be inspected: {path}: {error}") from error
    if not stat.S_ISREG(mode):
        raise IntegrityError(f"{label} is not a regular file: {path}")


def _require_file_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise IntegrityError(f"{label} is missing: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise IntegrityError(f"{label} SHA-256 mismatch: observed {observed}, expected {expected}")
    return f"{label} SHA-256 matches {expected}"


def _first_lines(text: str, limit: int = 5) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) <= limit:
        return "; ".join(lines)
    return "; ".join([*lines[:limit], f"... {len(lines) - limit} more"])


# ---------------------------------------------------------------------------
# ground truth
# ---------------------------------------------------------------------------


def verify_ground_truth(
    config: IntegrityConfig,
    *,
    repo_root: Path | str | None = None,
    dataset_root: Path | str | None = None,
    verify_semantics: bool = True,
) -> VerificationReport:
    """Verify ground truth by exact bytes, schema, rows, uniqueness and joins.

    The expected digest comes only from the committed integrity configuration;
    it is never calculated from the current file. A byte change, including a
    pure row reordering that preserves parsed semantics, fails.
    """
    root = _repo_root(repo_root)
    pin = config.ground_truth
    if dataset_root is None:
        dataset_dir = _absolute(root, config.benchmark.dataset)
        path = _absolute(root, pin.path)
    else:
        dataset_dir = Path(dataset_root).absolute()
        path = dataset_dir / Path(pin.path).name
    checks: list[CheckResult] = []

    def _file() -> str:
        _require_regular_file(path, "ground_truth.csv")
        raw = path.read_bytes()
        observed = sha256_bytes(raw)
        if observed != pin.sha256:
            raise IntegrityError(
                f"ground_truth.csv SHA-256 mismatch: observed {observed}, expected {pin.sha256}"
            )
        return f"exact SHA-256 {pin.sha256} over {len(raw)} bytes"

    def _table() -> str:
        frame = pd.read_csv(io.BytesIO(path.read_bytes()), keep_default_na=False)
        actual = tuple(frame.columns)
        if actual != pin.columns:
            raise IntegrityError(
                f"ground_truth.csv columns must be exactly {list(pin.columns)}, got {list(actual)}"
            )
        if len(frame) != pin.rows:
            raise IntegrityError(f"ground_truth.csv has {len(frame)} rows, expected {pin.rows}")
        duplicates = [column for column in pin.unique_columns if frame[column].duplicated().any()]
        if duplicates:
            raise IntegrityError(
                f"ground_truth.csv has duplicate value(s) in: {sorted(duplicates)}"
            )
        return (
            f"exactly {len(frame)} rows and columns {list(pin.columns)}; "
            f"unique {list(pin.unique_columns)}"
        )

    _check(checks, "ground_truth:file", _file)
    _check(checks, "ground_truth:table", _table)

    if verify_semantics:

        def _semantics() -> str:
            protocol_path = _absolute(root, config.benchmark.protocol)
            _require_file_hash(
                protocol_path, config.benchmark.protocol_sha256, "execution protocol"
            )
            protocol = load_execution_protocol(protocol_path)
            text_path = _absolute(root, str(protocol.text_case_manifest["path"]))
            case_manifest = load_case_manifest(text_path)
            bundle = mfb.load_dataset(dataset_dir)
            mfb._verify_source_tables(bundle, protocol, case_manifest)
            return "ground truth joins one-to-one to the frozen source plan and lineage"

        _check(checks, "ground_truth:semantic_joins", _semantics)

    return VerificationReport(dataset=str(dataset_dir), checks=tuple(checks))


# ---------------------------------------------------------------------------
# dependency closure
# ---------------------------------------------------------------------------


def _module_candidates(name: str) -> Iterable[str]:
    parts = [part for part in name.split(".") if part]
    for index in range(len(parts), 0, -1):
        yield ".".join(parts[:index])


def _internal_import_names(tree: ast.AST, current_module: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "captcha_quality":
                    continue
                if alias.name.startswith("captcha_quality."):
                    names.add(alias.name[len("captcha_quality.") :])
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                base = current_module.split(".")[: -node.level]
                module = ".".join([*base, module] if module else base)
            if module == "captcha_quality":
                names.update(alias.name for alias in node.names)
            elif module.startswith("captcha_quality."):
                rest = module[len("captcha_quality.") :]
                names.add(rest)
                names.update(f"{rest}.{alias.name}" for alias in node.names)
    return names


def _resolve_module(package_root: Path, name: str) -> Path | None:
    for candidate_name in _module_candidates(name):
        relative = Path(*candidate_name.split("."))
        module = package_root / f"{relative}.py"
        if module.is_file():
            return module.absolute()
        package = package_root / relative / "__init__.py"
        if package.is_file():
            return package.absolute()
    return None


def discover_dependency_closure(root_module: Path, package_root: Path) -> tuple[Path, ...]:
    """Discover the exact in-repository import closure of ``root_module``.

    Only imports reachable from the root module are followed. An unrelated file
    added to the package or to any shared directory is deliberately ignored.
    """
    root_module = Path(root_module).absolute()
    package_root = Path(package_root).absolute()
    if not root_module.is_file():
        raise IntegrityError(f"execution root module is missing: {root_module}")
    queue = [root_module]
    discovered: dict[Path, None] = {}
    while queue:
        path = queue.pop()
        if path in discovered:
            continue
        if not path.is_file():
            raise IntegrityError(f"imported module is missing: {path}")
        discovered[path] = None
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as error:
            raise IntegrityError(f"cannot parse imported module {path}: {error}") from error
        try:
            current = path.relative_to(package_root).with_suffix("").as_posix().replace("/", ".")
        except ValueError:
            current = path.stem
        for name in _internal_import_names(tree, current):
            resolved = _resolve_module(package_root, name)
            if resolved is not None:
                queue.append(resolved)
    return tuple(
        sorted(
            discovered,
            key=lambda path: path.relative_to(package_root).as_posix(),
        )
    )


def _external_import_names(paths: Sequence[Path]) -> set[str]:
    names: set[str] = set()
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            candidates: list[str] = []
            if isinstance(node, ast.Import):
                candidates = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                candidates = [node.module.split(".")[0]]
            for name in candidates:
                if name == "captcha_quality" or name == "__future__":
                    continue
                if name in sys.stdlib_module_names:
                    continue
                names.add(name)
    return names


def _verify_environment(pin: EnvironmentPin) -> str:
    failures: list[str] = []
    if platform.python_version() != pin.python_version:
        failures.append(f"Python {platform.python_version()} != pinned {pin.python_version}")
    if platform.python_implementation() != pin.python_implementation:
        failures.append(
            f"implementation {platform.python_implementation()} != "
            f"pinned {pin.python_implementation}"
        )
    if platform.system() != pin.system:
        failures.append(f"system {platform.system()} != pinned {pin.system}")
    if platform.machine() != pin.machine:
        failures.append(f"machine {platform.machine()} != pinned {pin.machine}")
    for package in pin.packages:
        try:
            observed = metadata.version(package.name)
        except metadata.PackageNotFoundError:
            failures.append(f"{package.name} is not installed")
            continue
        if observed != package.version:
            failures.append(f"{package.name} {observed} != pinned {package.version}")
    if failures:
        raise IntegrityError("; ".join(failures))
    return (
        f"Python {pin.python_version} {pin.python_implementation} on "
        f"{pin.system}/{pin.machine}; {len(pin.packages)} package versions match"
    )


def verify_dependency_closure(
    config: IntegrityConfig,
    *,
    repo_root: Path | str | None = None,
    verify_environment: bool = True,
) -> VerificationReport:
    """Verify the complete execution closure against exact committed hashes."""
    root = _repo_root(repo_root)
    checks: list[CheckResult] = []
    package_root = root / "src" / "captcha_quality"
    root_module = _absolute(root, config.benchmark.root_module)
    declared = {item.path: item for item in config.dependencies}
    closure: tuple[Path, ...] = ()

    def _membership() -> str:
        nonlocal closure
        closure = discover_dependency_closure(root_module, package_root)
        observed = {path.relative_to(root).as_posix() for path in closure}
        missing_pins = sorted(observed - set(declared))
        unexpected = sorted(set(declared) - observed)
        if missing_pins or unexpected:
            raise IntegrityError(
                "dependency closure mismatch; unpinned imported module(s)="
                f"{missing_pins}, declared but not imported={unexpected}"
            )
        return (
            f"all {len(observed)} discovered in-repository modules are explicitly "
            "pinned; unrelated package files are out of scope"
        )

    _check(checks, "dependencies:membership", _membership)

    def _hashes() -> str:
        failures: list[str] = []
        for dependency in config.dependencies:
            path = _absolute(root, dependency.path)
            if not path.is_file():
                failures.append(f"{dependency.key}: missing {dependency.path}")
                continue
            observed = sha256_file(path)
            if observed != dependency.sha256:
                failures.append(
                    f"{dependency.key}: observed {observed} != expected {dependency.sha256}"
                )
        if failures:
            raise IntegrityError("; ".join(failures))
        return f"{len(config.dependencies)} / {len(config.dependencies)} pinned files match"

    _check(checks, "dependencies:hashes", _hashes)

    def _external() -> str:
        if not closure:
            raise IntegrityError("dependency closure was not discovered")
        observed = _external_import_names(closure)
        declared_modules = {
            module for package in config.environment.packages for module in package.modules
        }
        unexpected = sorted(observed - declared_modules)
        if unexpected:
            raise IntegrityError(
                f"execution closure imports undeclared external module(s): {unexpected}"
            )
        return f"all {len(observed)} external import roots are declared"

    _check(checks, "dependencies:external_imports", _external)
    if verify_environment:
        _check(checks, "environment:runtime", lambda: _verify_environment(config.environment))
    return VerificationReport(dataset=str(root), checks=tuple(checks))


# ---------------------------------------------------------------------------
# input manifest
# ---------------------------------------------------------------------------


def validate_input_manifest_frame(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    key_column: str,
    legacy_duplicates: Sequence[LegacyDuplicate] = (),
) -> str:
    """Validate exact columns and key uniqueness for one input manifest.

    Duplicate keys are rejected. A caller may pass ``legacy_duplicates`` only
    for a documented historical evidence mode; each declared multiplicity is
    then checked exactly, including the declared agreement columns.
    """
    actual = tuple(frame.columns)
    expected = tuple(columns)
    if actual != expected:
        raise IntegrityError(
            f"input manifest columns must be exactly {list(expected)}, got {list(actual)}"
        )
    if key_column not in frame.columns:
        raise IntegrityError(f"input manifest is missing its key column {key_column!r}")
    if frame[key_column].astype(str).str.len().eq(0).any():
        raise IntegrityError("input manifest contains an empty key")
    counts = frame[key_column].astype(str).value_counts()
    duplicated = {str(key): int(count) for key, count in counts.items() if int(count) > 1}
    allowed = {item.key: item for item in legacy_duplicates}
    unexpected = sorted(set(duplicated) - set(allowed))
    if unexpected:
        raise IntegrityError(f"duplicate input-manifest keys are not allowed: {unexpected}")
    for key, duplicate in allowed.items():
        observed = duplicated.get(key, 0)
        if observed == 0:
            raise IntegrityError(
                f"declared historical duplicate key {key!r} is absent from the manifest"
            )
        if observed != duplicate.count:
            raise IntegrityError(
                f"historical duplicate key {key!r} has multiplicity {observed}, "
                f"expected {duplicate.count}"
            )
        group = frame[frame[key_column].astype(str) == key]
        compared = group[list(duplicate.identical_columns)].astype(str)
        if len(compared.drop_duplicates()) != 1:
            raise IntegrityError(
                f"historical duplicate key {key!r} does not agree on "
                f"{list(duplicate.identical_columns)}"
            )
    if allowed:
        detail = (
            f"{len(counts)} unique keys across {len(frame)} rows; "
            f"{len(allowed)} historical duplicate key(s) validated"
        )
    else:
        detail = f"{len(counts)} unique keys across {len(frame)} rows"
    return detail


def _manifest_matches(frame: pd.DataFrame) -> str:
    if "matches" not in frame.columns:
        raise IntegrityError("input manifest is missing the 'matches' column")
    values = frame["matches"].astype(str).str.lower()
    missing = int((~values.isin({"true", "1"})).sum())
    if missing:
        raise IntegrityError(f"input manifest has {missing} row(s) whose pin does not match")
    if "sha256" in frame.columns and "expected_sha256" in frame.columns:
        expected = frame["expected_sha256"].astype(str)
        observed = frame["sha256"].astype(str)
        mismatched = int(((expected != "") & (observed != expected)).sum())
        if mismatched:
            raise IntegrityError(
                f"input manifest has {mismatched} row(s) whose observed and expected hashes differ"
            )
    return f"all {len(frame)} input rows are pinned and matched"


def verify_historical_input_manifest(
    config: IntegrityConfig,
    *,
    repo_root: Path | str | None = None,
) -> VerificationReport:
    """Verify the preserved historical M8C input manifest byte-for-byte."""
    root = _repo_root(repo_root)
    pin = config.input_manifest
    path = _absolute(root, pin.path)
    checks: list[CheckResult] = []

    def _hash() -> str:
        _require_regular_file(path, "input_manifest.csv")
        observed = sha256_file(path)
        if observed != pin.sha256:
            raise IntegrityError(
                f"input_manifest.csv SHA-256 mismatch: observed {observed}, expected {pin.sha256}"
            )
        return f"exact historical SHA-256 {pin.sha256}"

    def _frame() -> pd.DataFrame:
        return pd.read_csv(path, keep_default_na=False)

    def _table() -> str:
        frame = _frame()
        actual = tuple(frame.columns)
        if actual != pin.columns:
            raise IntegrityError(
                f"input_manifest.csv columns must be exactly {list(pin.columns)}, "
                f"got {list(actual)}"
            )
        if len(frame) != pin.rows:
            raise IntegrityError(f"input_manifest.csv has {len(frame)} rows, expected {pin.rows}")
        unique = int(frame[pin.key_column].nunique())
        if unique != pin.unique_keys:
            raise IntegrityError(
                f"input_manifest.csv has {unique} unique keys, expected {pin.unique_keys}"
            )
        return f"{len(frame)} rows and {unique} unique keys"

    def _keys() -> str:
        return validate_input_manifest_frame(
            _frame(),
            columns=pin.columns,
            key_column=pin.key_column,
            legacy_duplicates=pin.legacy_duplicates,
        )

    _check(checks, "input_manifest:file", _hash)
    _check(checks, "input_manifest:table", _table)
    _check(checks, "input_manifest:keys", _keys)
    _check(checks, "input_manifest:matches", lambda: _manifest_matches(_frame()))
    return VerificationReport(dataset=str(path), checks=tuple(checks))


def verify_future_input_manifest(
    path: Path | str,
    *,
    expected_sha256: str | None = None,
    expected_rows: int | None = None,
    columns: Sequence[str] = INPUT_MANIFEST_COLUMNS,
    key_column: str = "key",
) -> VerificationReport:
    """Validate a newly generated input manifest for a future execution.

    Future manifests must have unique keys. There is deliberately no legacy
    duplicate allowance in this entry point.
    """
    manifest_path = Path(path)
    expected_columns = tuple(columns)
    expected_hash = _as_sha256(expected_sha256, "expected_sha256") if expected_sha256 else ""
    checks: list[CheckResult] = []

    def _hash() -> str:
        if not expected_hash:
            return "no expected byte hash supplied for the future manifest"
        _require_regular_file(manifest_path, "future input_manifest.csv")
        observed = sha256_file(manifest_path)
        if observed != expected_hash:
            raise IntegrityError(
                f"future input_manifest.csv SHA-256 mismatch: observed {observed}, "
                f"expected {expected_hash}"
            )
        return f"future input manifest SHA-256 matches {expected_hash}"

    def _frame() -> pd.DataFrame:
        return pd.read_csv(manifest_path, keep_default_na=False)

    def _table() -> str:
        frame = _frame()
        if tuple(frame.columns) != expected_columns:
            raise IntegrityError(
                f"future input manifest columns must be exactly {list(expected_columns)}, "
                f"got {list(frame.columns)}"
            )
        if expected_rows is not None and len(frame) != expected_rows:
            raise IntegrityError(
                f"future input manifest has {len(frame)} rows, expected {expected_rows}"
            )
        return f"{len(frame)} rows and columns {list(expected_columns)}"

    _check(checks, "input_manifest:file", _hash)
    _check(checks, "input_manifest:table", _table)
    _check(
        checks,
        "input_manifest:keys",
        lambda: validate_input_manifest_frame(
            _frame(),
            columns=expected_columns,
            key_column=key_column,
            legacy_duplicates=(),
        ),
    )
    _check(checks, "input_manifest:matches", lambda: _manifest_matches(_frame()))
    return VerificationReport(dataset=str(manifest_path), checks=tuple(checks))


# ---------------------------------------------------------------------------
# historical evidence
# ---------------------------------------------------------------------------


def verify_historical_evidence(
    config_path: Path | str = DEFAULT_CONFIG,
    *,
    repo_root: Path | str | None = None,
) -> VerificationReport:
    """Verify the recorded M8C evidence without asserting a clean historical run."""
    root = _repo_root(repo_root)
    config = load_config(_absolute(root, config_path))
    checks: list[CheckResult] = [
        CheckResult(
            "contract",
            True,
            f"{config.contract} ({config.schema_version})",
        ),
        CheckResult(
            "mode",
            True,
            "historical-evidence mode; the original dirty-tree execution is not "
            "retroactively described as clean",
        ),
    ]
    for name, path, expected in (
        ("benchmark:protocol", config.benchmark.protocol, config.benchmark.protocol_sha256),
        ("benchmark:lifecycle", config.benchmark.lifecycle, config.benchmark.lifecycle_sha256),
        (
            "benchmark:run_metadata",
            config.benchmark.run_metadata,
            config.benchmark.run_metadata_sha256,
        ),
    ):
        _check(
            checks,
            name,
            lambda path=path, expected=expected, name=name: _require_file_hash(
                _absolute(root, path), expected, name
            ),
        )

    def _historical_revision() -> str:
        payload = json.loads(
            _absolute(root, config.benchmark.run_metadata).read_text(encoding="utf-8")
        )
        code = payload.get("code", {})
        recorded = str(code.get("revision", ""))
        if recorded != config.benchmark.historical_execution_revision:
            raise IntegrityError(
                f"recorded historical revision {recorded!r} != "
                f"{config.benchmark.historical_execution_revision!r}"
            )
        module_pin = next(
            item for item in config.dependencies if item.path == config.benchmark.root_module
        )
        if str(code.get("module_sha256", "")) != module_pin.sha256:
            raise IntegrityError(
                "run_metadata.json module hash does not match the pinned execution module"
            )
        return (
            f"historical revision is honestly recorded as {recorded!r}; "
            "this verifier does not claim it was clean"
        )

    _check(checks, "benchmark:historical_revision", _historical_revision)
    checks.extend(verify_ground_truth(config, repo_root=root).checks)
    checks.extend(verify_dependency_closure(config, repo_root=root).checks)
    checks.extend(verify_historical_input_manifest(config, repo_root=root).checks)
    return VerificationReport(dataset=str(root / config.benchmark.dataset), checks=tuple(checks))


# ---------------------------------------------------------------------------
# future-execution provenance guard
# ---------------------------------------------------------------------------


def repository_state(repo_root: Path | str) -> RepositoryState:
    """Read the strict Git state of ``repo_root``; fail closed outside Git."""
    root = Path(repo_root).absolute()
    if not root.is_dir():
        raise IntegrityError(f"repository root is not a directory: {root}")

    def _git(*args: str) -> str:
        try:
            completed = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                check=False,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise IntegrityError(f"cannot run git in {root}: {error}") from error
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise IntegrityError(f"git {' '.join(args)} failed in {root}: {detail}")
        return completed.stdout.strip()

    revision = _git("rev-parse", "HEAD")
    if len(revision) != 40:
        raise IntegrityError(f"repository has no full committed revision: {root}")
    try:
        branch = _git("symbolic-ref", "--short", "-q", "HEAD")
    except IntegrityError:
        branch = ""
    status = _git("status", "--porcelain", "--untracked-files=normal")
    return RepositoryState(root=root, revision=revision, branch=branch, status=status)


def require_clean_worktree(
    repo_root: Path | str,
    *,
    expected_commit: str | None = None,
) -> RepositoryState:
    """Require a clean tree and, when supplied, the exact expected commit."""
    state = repository_state(repo_root)
    if expected_commit is not None:
        expected = _as_commit(expected_commit, "expected_commit")
        if state.revision != expected:
            raise IntegrityError(
                f"HEAD is {state.revision}, expected committed revision {expected}"
            )
    if not state.clean:
        raise IntegrityError(
            f"working tree is not clean at {state.revision}: {_first_lines(state.status)}"
        )
    return state


def _require_absent_outputs(root: Path, output_paths: Sequence[Path | str]) -> str:
    existing: list[str] = []
    for output in output_paths:
        path = _absolute(root, output)
        if path.exists():
            existing.append(str(path))
    if existing:
        raise IntegrityError(f"future output path(s) already exist: {existing}")
    return f"{len(output_paths)} future output path(s) are absent"


def verify_future_preflight(
    config_path: Path | str = DEFAULT_CONFIG,
    *,
    repo_root: Path | str | None = None,
    expected_commit: str,
    output_paths: Sequence[Path | str] = (),
) -> VerificationReport:
    """Verify the conditions required before any future execution creates data."""
    root = _repo_root(repo_root)
    config = load_config(_absolute(root, config_path))
    checks: list[CheckResult] = [
        CheckResult("contract", True, f"{config.contract} ({config.schema_version})"),
        CheckResult(
            "mode",
            True,
            "future-execution mode; a clean committed tree is required before "
            "any output path is created",
        ),
    ]

    def _clean() -> str:
        state = require_clean_worktree(root, expected_commit=expected_commit)
        return f"clean committed tree at {state.revision} (branch {state.branch or 'detached'})"

    _check(checks, "repository:clean", _clean)
    _check(
        checks,
        "repository:outputs_absent",
        lambda: _require_absent_outputs(root, output_paths),
    )
    checks.extend(verify_dependency_closure(config, repo_root=root).checks)
    return VerificationReport(dataset=str(root), checks=tuple(checks))


def run_guarded_execution(
    *,
    repo_root: Path | str,
    config_path: Path | str = DEFAULT_CONFIG,
    expected_commit: str,
    output_paths: Sequence[Path | str],
    action: Callable[[ExecutionTicket], T],
) -> T:
    """Run ``action`` only after clean-tree, absent-output and dependency checks.

    The check order is intentional: the dirty-tree rejection happens before any
    output path is created, and the action is never called on failure.
    """
    root = _repo_root(repo_root)
    config = load_config(_absolute(root, config_path))
    state = require_clean_worktree(root, expected_commit=expected_commit)
    _require_absent_outputs(root, output_paths)
    report = verify_dependency_closure(config, repo_root=root)
    if not report.ok:
        failures = "; ".join(f"{check.name}: {check.detail}" for check in report.failures())
        raise IntegrityError(f"future execution dependency preflight failed: {failures}")
    ticket = ExecutionTicket(
        revision=state.revision,
        branch=state.branch,
        config_sha256=config.sha256,
        output_paths=tuple(str(_absolute(root, path)) for path in output_paths),
    )
    return action(ticket)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_report(report: VerificationReport) -> None:
    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")
    print(f"verification: {'PASS' if report.ok else 'FAIL'}")


def _verify_command(args: argparse.Namespace) -> int:
    report = verify_historical_evidence(args.config, repo_root=args.repo_root)
    _print_report(report)
    return 0 if report.ok else 1


def _preflight_command(args: argparse.Namespace) -> int:
    report = verify_future_preflight(
        args.config,
        repo_root=args.repo_root,
        expected_commit=args.expected_commit,
        output_paths=args.output,
    )
    _print_report(report)
    return 0 if report.ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="captcha-benchmark-integrity",
        description="Verify M8C provenance, ground-truth bytes and future execution guards.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser(
        "verify", help="verify the historical M8C evidence and dependency closure"
    )
    verify_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    verify_parser.add_argument("--repo-root", type=Path, default=None)
    verify_parser.set_defaults(handler=_verify_command)

    preflight_parser = subparsers.add_parser(
        "preflight", help="verify the clean-tree conditions for a future execution"
    )
    preflight_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    preflight_parser.add_argument("--repo-root", type=Path, default=None)
    preflight_parser.add_argument("--expected-commit", required=True)
    preflight_parser.add_argument(
        "--output",
        type=Path,
        action="append",
        default=[],
        help="a future output path that must not exist yet (repeatable)",
    )
    preflight_parser.set_defaults(handler=_preflight_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (IntegrityError, IntegrityConfigError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except Exception as error:  # noqa: BLE001 - fail closed with a non-zero exit
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


__all__ = [
    "DEFAULT_CONFIG",
    "INTEGRITY_CONFIG_SCHEMA",
    "INTEGRITY_CONTRACT",
    "ExecutionTicket",
    "IntegrityConfig",
    "IntegrityConfigError",
    "IntegrityError",
    "RepositoryState",
    "build_parser",
    "discover_dependency_closure",
    "load_config",
    "main",
    "repository_state",
    "require_clean_worktree",
    "run_guarded_execution",
    "validate_input_manifest_frame",
    "verify_dependency_closure",
    "verify_future_input_manifest",
    "verify_future_preflight",
    "verify_ground_truth",
    "verify_historical_evidence",
    "verify_historical_input_manifest",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
