"""M9A read-only interference error analysis.

M9A analyses the frozen M8C cross-font benchmark predictions. It never
generates a benchmark, loads a model, fits, refits, calibrates, selects or
tunes anything. The only writes are the explicitly scoped files under
``reports/interference_error_analysis_v0_1_0``.

The target is ``ssim_target``. PSNR is diagnostic only and is not analysed.
The independent unit is the source group (one font rendered with one committed
text case). Bootstrap replicates resample whole source groups.

A positive residual is ``ssim_target - prediction`` (underprediction); a
negative residual is overprediction.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy import stats

from captcha_quality.provenance import (
    code_revision,
    json_safe,
    package_version,
    sha256_file,
)

# ---------------------------------------------------------------------------
# frozen analysis contract
# ---------------------------------------------------------------------------

ANALYSIS_NAME: Final[str] = "interference-error-analysis"
ANALYSIS_VERSION: Final[str] = "v0.1.0"
CONFIG_SCHEMA: Final[str] = "m9a-interference-error-analysis-config-v1"
PROTOCOL_SCHEMA: Final[str] = "m9a-interference-error-analysis-protocol-v1"
DEFAULT_CONFIG: Final[Path] = Path("configs/interference_error_analysis_v0_1_0.toml")
REPORT_BASENAME: Final[str] = "interference_error_analysis_v0_1_0"
REPORT_PREFIX: Final[str] = "interference_error_analysis_v"

NEAR_DEGENERATE_SD: Final[float] = 0.005
LARGE_ERROR_THRESHOLD: Final[float] = 0.20

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "intensity_mean",
    "intensity_std",
    "contrast_rms",
    "ink_fraction",
    "laplacian_variance",
    "edge_density",
    "histogram_entropy",
    "noise_sigma_immerkaer",
)

PREDICTION_COLUMNS: Final[tuple[str, ...]] = (
    "row_id",
    "sample_id",
    "source_id",
    "case_id",
    "case_index",
    "font_id",
    "font_group",
    "variant_family",
    "severity_level",
    "amplitude_px",
    "protocol_scope",
    "ssim_target",
    "psnr_target",
    "psnr_infinite",
    "prediction",
    "residual",
    "abs_residual",
)

MANIFEST_COLUMNS: Final[tuple[str, ...]] = (
    "row_id",
    "sample_id",
    "source_id",
    "split_group_id",
    "split",
    "case_id",
    "case_index",
    "font_id",
    "font_sha256",
    "text_length",
    "text_sha256",
    "render_seed",
    "variant_id",
    "variant_family",
    "severity_level",
    "amplitude_px",
    "variant_seed",
    "image_path",
    "reference_path",
    "ssim_target",
    "psnr_target",
    "psnr_infinite",
    "height",
    "width",
    "image_sha256",
    "reference_sha256",
    "dataset_version",
    "generation_config_hash",
    "protocol_scope",
    "primary",
    "corruption_parameters_json",
)

SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "source_id",
    "case_id",
    "case_index",
    "split_group_id",
    "split",
    "font_id",
    "font_path",
    "font_sha256",
    "text_length",
    "text_sha256",
    "render_seed",
    "render_config_hash",
    "background_level",
    "height",
    "width",
    "clean_image_path",
    "clean_image_sha256",
    "generation_config_hash",
    "dataset_version",
)

SOURCE_CASE_COLUMNS: Final[tuple[str, ...]] = (
    "source_id",
    "case_id",
    "case_index",
    "font_id",
    "text_length",
    "text_sha256",
    "render_seed",
    "split",
    "split_group_id",
)

GROUND_TRUTH_COLUMNS: Final[tuple[str, ...]] = (
    "source_id",
    "case_id",
    "font_id",
    "text",
)

FEATURE_COLUMNS: Final[tuple[str, ...]] = ("sample_id", *FEATURE_NAMES)

REQUIRED_INPUT_KEYS: Final[tuple[str, ...]] = (
    "predictions",
    "manifest",
    "features",
    "sources",
    "source_cases",
    "ground_truth",
    "run_metadata",
    "model_artifact",
    "model_card",
    "m8e_config",
)

RUN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "name",
        "version",
        "seed",
        "n_bootstrap",
        "confidence_level",
        "output_reports",
    }
)
ANALYSIS_KEYS: Final[frozenset[str]] = frozenset(
    {
        "near_degenerate_sd",
        "large_error_threshold",
        "group_column",
        "independent_unit",
        "bootstrap_method",
        "feature_association_method",
        "multiple_comparison_method",
        "large_error_rule",
        "feature_association_rule",
        "font_contrast_rule",
        "source_characteristic_rule",
    }
)
INPUT_KEYS: Final[frozenset[str]] = frozenset({"key", "path", "sha256", "rows", "role"})

FAMILY_ORDER: Final[tuple[str, ...]] = (
    "gaussian_blur",
    "gaussian_noise",
    "thin_lines",
    "arc_strokes",
    "elastic",
)
SEVERITY_FAMILIES: Final[tuple[str, ...]] = (
    "gaussian_blur",
    "gaussian_noise",
    "thin_lines",
    "arc_strokes",
    "elastic",
)
PRIMARY_SEVERITIES: Final[dict[str, tuple[int, ...]]] = {
    "gaussian_blur": (1, 2, 3, 4),
    "gaussian_noise": (1, 2, 3, 4),
    "thin_lines": (1, 2, 3, 4),
    "arc_strokes": (1, 2, 3, 4),
    "elastic": (2, 3, 4),
}
INTERFERENCE_FAMILIES: Final[tuple[str, ...]] = ("thin_lines", "arc_strokes")

SCOPE_DESCRIPTIONS: Final[dict[str, str]] = {
    "all_primary": "all 5,700 primary rows",
    "original_blur_noise": "primary Gaussian blur and Gaussian noise rows",
    "gaussian_blur": "primary Gaussian blur rows",
    "gaussian_noise": "primary Gaussian noise rows",
    "interference": "primary thin-line and arc-stroke rows",
    "thin_lines": "primary thin-line rows",
    "arc_strokes": "primary arc-stroke rows",
    "geometric_primary": "primary elastic geometric rows",
    "clean_reference": "diagnostic clean-reference rows (target is constant)",
}
SCOPE_ORDER: Final[tuple[str, ...]] = tuple(SCOPE_DESCRIPTIONS)
SUMMARY_SCOPES: Final[tuple[str, ...]] = (
    "all_primary",
    "original_blur_noise",
    "gaussian_blur",
    "gaussian_noise",
    "interference",
    "thin_lines",
    "arc_strokes",
    "geometric_primary",
    "clean_reference",
)
FONT_SCOPES: Final[tuple[str, ...]] = (
    "all_primary",
    "original_blur_noise",
    "gaussian_blur",
    "gaussian_noise",
    "interference",
    "thin_lines",
    "arc_strokes",
    "geometric_primary",
)
SOURCE_SCOPES: Final[tuple[str, ...]] = FONT_SCOPES
FEATURE_SCOPES: Final[tuple[str, ...]] = (
    "all_primary",
    "interference",
    "thin_lines",
    "arc_strokes",
    "gaussian_blur",
    "gaussian_noise",
    "geometric_primary",
    "thin_lines_s1",
    "thin_lines_s2",
    "thin_lines_s3",
    "thin_lines_s4",
    "arc_strokes_s1",
    "arc_strokes_s2",
    "arc_strokes_s3",
    "arc_strokes_s4",
)

PAIRED_CONDITION_CONTRASTS: Final[tuple[tuple[str, str, str], ...]] = (
    ("interference_minus_gaussian_blur", "interference", "gaussian_blur"),
    ("interference_minus_gaussian_noise", "interference", "gaussian_noise"),
    ("interference_minus_geometric_primary", "interference", "geometric_primary"),
    ("thin_lines_minus_arc_strokes", "thin_lines", "arc_strokes"),
)

REPORT_FILES: Final[tuple[str, ...]] = (
    "README.md",
    "analysis_protocol.json",
    "summary_metrics.csv",
    "condition_calibration.csv",
    "font_condition_metrics.csv",
    "severity_metrics.csv",
    "source_group_metrics.csv",
    "case_metrics.csv",
    "source_characteristic_associations.csv",
    "feature_error_associations.csv",
    "paired_condition_contrasts.csv",
    "error_concentration.csv",
    "hypotheses_and_followups.md",
    "verification_checks.csv",
)
FIGURE_FILES: Final[tuple[str, ...]] = (
    "fig1_condition_error_profile.svg",
    "fig2_interference_severity_calibration.svg",
    "fig3_font_interference_mae.svg",
    "fig4_feature_error_associations.svg",
    "fig5_source_error_concentration.svg",
)
OUTPUT_FILES: Final[tuple[str, ...]] = (
    *REPORT_FILES,
    *(f"figures/{name}" for name in FIGURE_FILES),
)

FORBIDDEN_IMPORTS: Final[frozenset[str]] = frozenset(
    {
        "sklearn",
        "joblib",
        "PIL",
        "cv2",
        "skimage",
        "captcha_quality.matched_font_benchmark",
        "captcha_quality.features",
        "captcha_quality.corruptions",
        "captcha_quality.render",
        "captcha_quality.benchmark",
    }
)
FORBIDDEN_CALL_NAMES: Final[frozenset[str]] = frozenset(
    {
        "fit",
        "fit_transform",
        "predict",
        "predict_proba",
        "joblib.load",
        "render_clean",
        "generate_dataset",
        "apply_corruption",
        "apply_strokes",
        "run_benchmark",
    }
)

EXTERNAL_VERIFIERS: Final[tuple[dict[str, Any], ...]] = (
    {
        "key": "m8e_integrity",
        "command": [
            "-m",
            "captcha_quality.benchmark_integrity",
            "verify",
        ],
        "expected": "m8e-integrity-v1 historical-evidence mode PASS",
    },
    {
        "key": "m8c_lifecycle_post_execution",
        "command": [
            "-m",
            "captcha_quality.benchmark_lifecycle",
            "verify",
            "--config",
            "configs/benchmark_lifecycle_v0_3_0.toml",
            "--state",
            "post-execution",
        ],
        "expected": "M8C lifecycle post-execution PASS",
    },
    {
        "key": "m8c_full_replay",
        "command": [
            "-m",
            "captcha_quality.matched_font_benchmark",
            "verify",
            "--lifecycle",
            "configs/benchmark_lifecycle_v0_2_0.toml",
            "--lifecycle-post",
            "configs/benchmark_lifecycle_v0_3_0.toml",
        ],
        "expected": "M8C table replay and exact report membership PASS",
    },
)


class AnalysisError(RuntimeError):
    """Raised when an M9A analysis input or invariant fails closed."""


class AnalysisConfigError(AnalysisError):
    """Raised when the M9A configuration is malformed or unsafe."""


# ---------------------------------------------------------------------------
# configuration and input validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InputPin:
    key: str
    path: str
    sha256: str
    rows: int
    role: str


@dataclass(frozen=True, slots=True)
class RunSettings:
    name: str
    version: str
    seed: int
    n_bootstrap: int
    confidence_level: float
    output_reports: Path


@dataclass(frozen=True, slots=True)
class AnalysisSettings:
    near_degenerate_sd: float
    large_error_threshold: float
    group_column: str
    independent_unit: str
    bootstrap_method: str
    feature_association_method: str
    multiple_comparison_method: str
    large_error_rule: str
    feature_association_rule: str
    font_contrast_rule: str
    source_characteristic_rule: str


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    run: RunSettings
    analysis: AnalysisSettings
    inputs: tuple[InputPin, ...]
    config_path: Path
    config_sha256: str
    repo_root: Path

    def identity(self) -> dict[str, Any]:
        return {
            "run": {
                "name": self.run.name,
                "version": self.run.version,
                "seed": self.run.seed,
                "n_bootstrap": self.run.n_bootstrap,
                "confidence_level": self.run.confidence_level,
                "output_reports": str(self.run.output_reports),
            },
            "analysis": asdict(self.analysis),
            "inputs": [asdict(item) for item in self.inputs],
        }


@dataclass(frozen=True, slots=True)
class Check:
    check_id: str
    category: str
    expected: str
    actual: str
    passed: bool
    details: str = ""


@dataclass(frozen=True, slots=True)
class AnalysisInputs:
    predictions: pd.DataFrame
    manifest: pd.DataFrame
    features: pd.DataFrame
    sources: pd.DataFrame
    source_cases: pd.DataFrame
    ground_truth: pd.DataFrame
    run_metadata: Mapping[str, Any]
    records: tuple[dict[str, Any], ...]
    checks: tuple[Check, ...]
    external_verification: tuple[dict[str, Any], ...]
    joined: pd.DataFrame


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    summary_metrics: pd.DataFrame
    condition_calibration: pd.DataFrame
    font_condition_metrics: pd.DataFrame
    severity_metrics: pd.DataFrame
    source_group_metrics: pd.DataFrame
    case_metrics: pd.DataFrame
    source_characteristic_associations: pd.DataFrame
    feature_error_associations: pd.DataFrame
    paired_condition_contrasts: pd.DataFrame
    error_concentration: pd.DataFrame
    checks: tuple[Check, ...]
    external_verification: tuple[dict[str, Any], ...]


def _toml_section(payload: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise AnalysisConfigError(f"[{name}] must be a table")
    return dict(value)


def _check_keys(section: Mapping[str, Any], allowed: frozenset[str], name: str) -> None:
    unknown = sorted(set(section) - allowed)
    missing = sorted(allowed - set(section))
    if unknown:
        raise AnalysisConfigError(f"unknown key(s) in [{name}]: {unknown}")
    if missing:
        raise AnalysisConfigError(f"missing key(s) in [{name}]: {missing}")


def _as_nonempty_str(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnalysisConfigError(f"{label} must be a non-empty string")
    return value


def _as_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AnalysisConfigError(f"{label} must be an integer")
    return value


def _as_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise AnalysisConfigError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise AnalysisConfigError(f"{label} must be finite")
    return result


def _as_sha256(value: object, label: str) -> str:
    text = _as_nonempty_str(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise AnalysisConfigError(f"{label} must be a lower-case SHA-256 hex digest")
    return text


def load_config(
    path: Path | str = DEFAULT_CONFIG, *, repo_root: Path | None = None
) -> AnalysisConfig:
    """Load and strictly validate the M9A analysis configuration."""
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise AnalysisConfigError(f"configuration does not exist: {config_path}")
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise AnalysisConfigError("configuration root must be a table")

    allowed_root = {"schema", "run", "analysis", "inputs"}
    unknown_root = sorted(set(raw) - allowed_root)
    if unknown_root:
        raise AnalysisConfigError(f"unknown top-level key(s): {unknown_root}")
    if raw.get("schema") != CONFIG_SCHEMA:
        raise AnalysisConfigError(f"schema must be {CONFIG_SCHEMA!r}")

    run_raw = _toml_section(raw, "run")
    _check_keys(run_raw, RUN_KEYS, "run")
    run = RunSettings(
        name=_as_nonempty_str(run_raw["name"], "run.name"),
        version=_as_nonempty_str(run_raw["version"], "run.version"),
        seed=_as_int(run_raw["seed"], "run.seed"),
        n_bootstrap=_as_int(run_raw["n_bootstrap"], "run.n_bootstrap"),
        confidence_level=_as_float(run_raw["confidence_level"], "run.confidence_level"),
        output_reports=Path(_as_nonempty_str(run_raw["output_reports"], "run.output_reports")),
    )
    if run.name != ANALYSIS_NAME:
        raise AnalysisConfigError(f"run.name must be {ANALYSIS_NAME!r}")
    if run.version != ANALYSIS_VERSION:
        raise AnalysisConfigError(f"run.version must be {ANALYSIS_VERSION!r}")
    if run.seed < 0:
        raise AnalysisConfigError("run.seed must be >= 0")
    if run.n_bootstrap < 100:
        raise AnalysisConfigError("run.n_bootstrap must be >= 100")
    if not 0.0 < run.confidence_level < 1.0:
        raise AnalysisConfigError("run.confidence_level must be strictly between 0 and 1")

    analysis_raw = _toml_section(raw, "analysis")
    _check_keys(analysis_raw, ANALYSIS_KEYS, "analysis")
    analysis = AnalysisSettings(
        near_degenerate_sd=_as_float(
            analysis_raw["near_degenerate_sd"], "analysis.near_degenerate_sd"
        ),
        large_error_threshold=_as_float(
            analysis_raw["large_error_threshold"], "analysis.large_error_threshold"
        ),
        group_column=_as_nonempty_str(analysis_raw["group_column"], "analysis.group_column"),
        independent_unit=_as_nonempty_str(
            analysis_raw["independent_unit"], "analysis.independent_unit"
        ),
        bootstrap_method=_as_nonempty_str(
            analysis_raw["bootstrap_method"], "analysis.bootstrap_method"
        ),
        feature_association_method=_as_nonempty_str(
            analysis_raw["feature_association_method"],
            "analysis.feature_association_method",
        ),
        multiple_comparison_method=_as_nonempty_str(
            analysis_raw["multiple_comparison_method"],
            "analysis.multiple_comparison_method",
        ),
        large_error_rule=_as_nonempty_str(
            analysis_raw["large_error_rule"], "analysis.large_error_rule"
        ),
        feature_association_rule=_as_nonempty_str(
            analysis_raw["feature_association_rule"],
            "analysis.feature_association_rule",
        ),
        font_contrast_rule=_as_nonempty_str(
            analysis_raw["font_contrast_rule"], "analysis.font_contrast_rule"
        ),
        source_characteristic_rule=_as_nonempty_str(
            analysis_raw["source_characteristic_rule"],
            "analysis.source_characteristic_rule",
        ),
    )
    if analysis.near_degenerate_sd != NEAR_DEGENERATE_SD:
        raise AnalysisConfigError(
            f"analysis.near_degenerate_sd must be the frozen M8C value {NEAR_DEGENERATE_SD}"
        )
    if analysis.large_error_threshold != LARGE_ERROR_THRESHOLD:
        raise AnalysisConfigError(
            f"analysis.large_error_threshold must be the predeclared value {LARGE_ERROR_THRESHOLD}"
        )
    if analysis.group_column != "source_id":
        raise AnalysisConfigError("analysis.group_column must be 'source_id'")

    inputs_raw = raw.get("inputs")
    if not isinstance(inputs_raw, list) or not inputs_raw:
        raise AnalysisConfigError("[[inputs]] must be a non-empty list")
    inputs: list[InputPin] = []
    seen: set[str] = set()
    for index, item in enumerate(inputs_raw):
        if not isinstance(item, dict):
            raise AnalysisConfigError(f"inputs[{index}] must be a table")
        _check_keys(item, INPUT_KEYS, f"inputs[{index}]")
        key = _as_nonempty_str(item["key"], f"inputs[{index}].key")
        if key in seen:
            raise AnalysisConfigError(f"duplicate input key: {key}")
        seen.add(key)
        rows = _as_int(item["rows"], f"inputs[{index}].rows")
        if rows < 0:
            raise AnalysisConfigError(f"inputs[{index}].rows must be >= 0")
        inputs.append(
            InputPin(
                key=key,
                path=_as_nonempty_str(item["path"], f"inputs[{index}].path"),
                sha256=_as_sha256(item["sha256"], f"inputs[{index}].sha256"),
                rows=rows,
                role=_as_nonempty_str(item["role"], f"inputs[{index}].role"),
            )
        )
    if set(seen) != set(REQUIRED_INPUT_KEYS):
        raise AnalysisConfigError(
            "inputs must declare exactly these keys: " + ", ".join(REQUIRED_INPUT_KEYS)
        )

    output = run.output_reports
    if not output.is_absolute():
        output = root / output
    config = AnalysisConfig(
        run=RunSettings(
            name=run.name,
            version=run.version,
            seed=run.seed,
            n_bootstrap=run.n_bootstrap,
            confidence_level=run.confidence_level,
            output_reports=output.resolve(),
        ),
        analysis=analysis,
        inputs=tuple(inputs),
        config_path=config_path,
        config_sha256=sha256_file(config_path),
        repo_root=root,
    )
    validate_output_directory(config.repo_root, config.run.output_reports)
    return config


def validate_output_directory(repo_root: Path, output_directory: Path) -> Path:
    """Fail closed unless *output_directory* is a new M9A report path.

    The guard deliberately does not authorise a configurable override into a
    historical dataset, model or report directory.
    """
    root = repo_root.resolve()
    output = output_directory.resolve()
    reports = (root / "reports").resolve()
    if output == reports or not output.is_relative_to(reports):
        raise AnalysisConfigError(f"output directory must be below {reports}")
    if not output.name.startswith(REPORT_PREFIX):
        raise AnalysisConfigError(
            f"output directory name must start with {REPORT_PREFIX!r}: {output.name}"
        )
    forbidden = (
        root / "data",
        root / "models",
        root / "configs",
        root / "src",
        root / "tests",
        root / ".git",
        root / "reports" / "matched_font_benchmark_v0_1_0",
        root / "reports" / "matched_font_benchmark_provenance_audit_v0_1_0",
    )
    for path in forbidden:
        resolved = path.resolve()
        if output == resolved or output.is_relative_to(resolved):
            raise AnalysisConfigError(f"output directory is a forbidden historical path: {output}")
    return output


def _check(
    checks: list[Check],
    check_id: str,
    category: str,
    expected: str,
    actual: str,
    passed: bool,
    details: str = "",
) -> Check:
    result = Check(check_id, category, expected, actual, bool(passed), details)
    checks.append(result)
    return result


def _require(
    checks: list[Check],
    check_id: str,
    category: str,
    expected: str,
    actual: str,
    passed: bool,
    details: str = "",
) -> None:
    result = _check(checks, check_id, category, expected, actual, passed, details)
    if not result.passed:
        raise AnalysisError(f"{check_id}: expected {expected!r}; observed {actual!r}. {details}")


def _read_csv_pinned(
    path: Path,
    *,
    expected_sha256: str,
    expected_rows: int,
    expected_columns: Sequence[str],
    usecols: Sequence[str] | None = None,
) -> pd.DataFrame:
    if not path.is_file():
        raise AnalysisError(f"required input is missing: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise AnalysisError(
            f"input hash mismatch for {path}: expected {expected_sha256}, observed {actual_sha256}"
        )
    header = pd.read_csv(path, nrows=0)
    if list(header.columns) != list(expected_columns):
        raise AnalysisError(
            f"input schema mismatch for {path}: expected {list(expected_columns)}, "
            f"observed {list(header.columns)}"
        )
    if usecols is None:
        frame = pd.read_csv(path, keep_default_na=False)
    else:
        frame = pd.read_csv(path, usecols=list(usecols), keep_default_na=False)
    if len(frame) != expected_rows:
        raise AnalysisError(
            f"input row-count mismatch for {path}: expected {expected_rows}, observed {len(frame)}"
        )
    return frame


def _require_unique(frame: pd.DataFrame, column: str, label: str) -> None:
    if column not in frame.columns:
        raise AnalysisError(f"{label} is missing key column {column!r}")
    duplicates = int(frame[column].duplicated().sum())
    if duplicates:
        raise AnalysisError(f"{label} has {duplicates} duplicate {column!r} value(s)")


def _require_no_missing(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing_columns = sorted(set(columns) - set(frame.columns))
    if missing_columns:
        raise AnalysisError(f"{label} is missing column(s): {missing_columns}")
    for column in columns:
        values = frame[column]
        if values.isna().any() or (values.astype(str).str.len() == 0).any():
            raise AnalysisError(f"{label} has missing value(s) in {column!r}")


def _same_values(left: pd.Series, right: pd.Series, label: str) -> None:
    if len(left) != len(right):
        raise AnalysisError(f"{label}: length mismatch {len(left)} vs {len(right)}")
    if not left.reset_index(drop=True).equals(right.reset_index(drop=True)):
        raise AnalysisError(f"{label}: values differ")


def _same_float_values(left: pd.Series, right: pd.Series, label: str, atol: float = 1e-12) -> None:
    if len(left) != len(right):
        raise AnalysisError(f"{label}: length mismatch {len(left)} vs {len(right)}")
    if not np.allclose(
        left.to_numpy(dtype=np.float64),
        right.to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=atol,
    ):
        raise AnalysisError(f"{label}: values differ beyond atol={atol}")


def _git_output(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout


def _full_git_revision(repo_root: Path) -> str:
    return _git_output(repo_root, "rev-parse", "HEAD").strip() or "unknown"


def _working_tree_dirty(repo_root: Path) -> bool:
    return bool(_git_output(repo_root, "status", "--porcelain").strip())


def _run_external_verifiers(
    repo_root: Path,
    checks: list[Check],
) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for item in EXTERNAL_VERIFIERS:
        command = [sys.executable, *item["command"]]
        try:
            completed = subprocess.run(
                command,
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            _require(
                checks,
                f"external_{item['key']}",
                "historical_verifier",
                item["expected"],
                f"failed to execute: {exc}",
                False,
            )
            continue
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        last_line = next((line for line in reversed(stdout.splitlines()) if line.strip()), "")
        passed = completed.returncode == 0 and "PASS" in stdout
        _require(
            checks,
            f"external_{item['key']}",
            "historical_verifier",
            item["expected"],
            f"exit={completed.returncode}; last_line={last_line[:200]}",
            passed,
            details=(
                f"stdout_sha256={sha256_text(stdout)}; "
                f"stderr_sha256={sha256_text(stderr)}; "
                f"command={' '.join(item['command'])}"
            ),
        )
        records.append(
            {
                "key": item["key"],
                "command": " ".join(item["command"]),
                "returncode": completed.returncode,
                "passed": passed,
                "last_line": last_line,
                "stdout_sha256": sha256_text(stdout),
                "stderr_sha256": sha256_text(stderr),
            }
        )
    return tuple(records)


def sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("numpy", "pandas", "scipy", "captcha-quality-analyzer"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    versions["captcha_quality_package"] = package_version()
    return versions


def load_inputs(
    config: AnalysisConfig,
    *,
    run_verifiers: bool = True,
    require_clean: bool = True,
) -> AnalysisInputs:
    """Load and verify the frozen M8C inputs without fitting or generating.

    ``require_clean=False`` exists only for focused tests and iterative local
    development before the analysis code is committed. The report-producing CLI
    always uses the default and records the clean/dirty state.
    """
    checks: list[Check] = []
    root = config.repo_root
    records: list[dict[str, Any]] = []
    pins = {pin.key: pin for pin in config.inputs}

    dirty = _working_tree_dirty(root)
    _require(
        checks,
        "working_tree_clean_at_start",
        "repository",
        "clean working tree before the analysis writes its new report",
        "clean" if not dirty else "dirty",
        (not dirty) or (not require_clean),
        details="The report records the exact revision and this state.",
    )
    revision = _full_git_revision(root)
    _check(
        checks,
        "analysis_revision_recorded",
        "repository",
        "a full Git revision",
        revision,
        revision != "unknown",
    )

    for key, pin in pins.items():
        path = root / pin.path
        if not path.is_file():
            _require(
                checks,
                f"input_{key}_exists",
                "input",
                "regular file",
                "missing",
                False,
                details=str(path),
            )
        actual = sha256_file(path)
        _require(
            checks,
            f"input_{key}_sha256",
            "input",
            pin.sha256,
            actual,
            actual == pin.sha256,
            details=pin.path,
        )
        records.append(
            {
                "key": key,
                "path": pin.path,
                "role": pin.role,
                "expected_sha256": pin.sha256,
                "observed_sha256": actual,
                "expected_rows": pin.rows,
                "matches": actual == pin.sha256,
            }
        )

    predictions = _read_csv_pinned(
        root / pins["predictions"].path,
        expected_sha256=pins["predictions"].sha256,
        expected_rows=pins["predictions"].rows,
        expected_columns=PREDICTION_COLUMNS,
    )
    manifest = _read_csv_pinned(
        root / pins["manifest"].path,
        expected_sha256=pins["manifest"].sha256,
        expected_rows=pins["manifest"].rows,
        expected_columns=MANIFEST_COLUMNS,
    )
    features = _read_csv_pinned(
        root / pins["features"].path,
        expected_sha256=pins["features"].sha256,
        expected_rows=pins["features"].rows,
        expected_columns=FEATURE_COLUMNS,
    )
    sources = _read_csv_pinned(
        root / pins["sources"].path,
        expected_sha256=pins["sources"].sha256,
        expected_rows=pins["sources"].rows,
        expected_columns=SOURCE_COLUMNS,
    )
    source_cases = _read_csv_pinned(
        root / pins["source_cases"].path,
        expected_sha256=pins["source_cases"].sha256,
        expected_rows=pins["source_cases"].rows,
        expected_columns=SOURCE_CASE_COLUMNS,
    )
    ground_truth = _read_csv_pinned(
        root / pins["ground_truth"].path,
        expected_sha256=pins["ground_truth"].sha256,
        expected_rows=pins["ground_truth"].rows,
        expected_columns=GROUND_TRUTH_COLUMNS,
        usecols=["source_id", "case_id", "font_id"],
    )
    ground_truth = ground_truth.loc[:, ["source_id", "case_id", "font_id"]]
    run_metadata = json.loads((root / pins["run_metadata"].path).read_text(encoding="utf-8"))
    if not isinstance(run_metadata, dict):
        raise AnalysisError("run_metadata.json root must be an object")

    _validate_table_keys(
        predictions, manifest, features, sources, source_cases, ground_truth, checks
    )
    _validate_row_coverage(predictions, manifest, checks)
    _validate_reference_coverage(predictions, features, sources, source_cases, ground_truth, checks)
    _validate_model_provenance(root, pins, run_metadata, checks)
    _validate_ground_truth_policy(root, pins["ground_truth"].path, checks)

    joined = _build_joined_frame(
        predictions,
        manifest,
        features,
        sources,
        source_cases,
        ground_truth,
    )
    _validate_joined_structure(joined, config, checks)

    external: tuple[dict[str, Any], ...] = ()
    if run_verifiers:
        external = _run_external_verifiers(root, checks)
        _check(
            checks,
            "external_verifier_count",
            "historical_verifier",
            f"{len(EXTERNAL_VERIFIERS)} external verifier records",
            f"{len(external)} records",
            len(external) == len(EXTERNAL_VERIFIERS),
        )
    else:
        _check(
            checks,
            "external_verifiers_skipped",
            "historical_verifier",
            "external historical verifiers executed",
            "skipped by caller",
            True,
            details="Only used by focused tests; the report run executes the verifiers.",
        )

    _rehash_inputs(root, pins, checks, label="after_analysis")
    return AnalysisInputs(
        predictions=predictions,
        manifest=manifest,
        features=features,
        sources=sources,
        source_cases=source_cases,
        ground_truth=ground_truth,
        run_metadata=run_metadata,
        records=tuple(records),
        checks=tuple(checks),
        external_verification=external,
        joined=joined,
    )


def _validate_table_keys(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    sources: pd.DataFrame,
    source_cases: pd.DataFrame,
    ground_truth: pd.DataFrame,
    checks: list[Check],
) -> None:
    tables = {
        "predictions": (predictions, "row_id"),
        "manifest": (manifest, "row_id"),
        "features": (features, "sample_id"),
        "sources": (sources, "source_id"),
        "source_cases": (source_cases, "source_id"),
        "ground_truth": (ground_truth, "source_id"),
    }
    for label, (frame, key) in tables.items():
        _require_unique(frame, key, label)
        _require(
            checks,
            f"unique_{label}_{key}",
            "join_key",
            f"unique {key}",
            f"{frame[key].nunique()} unique / {len(frame)} rows",
            frame[key].nunique() == len(frame),
        )
    for label, (frame, key) in tables.items():
        missing = int(frame[key].astype(str).str.len().eq(0).sum())
        _require(
            checks,
            f"nonmissing_{label}_{key}",
            "join_key",
            "no empty key",
            f"{missing} empty",
            missing == 0,
        )


def _validate_row_coverage(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    checks: list[Check],
) -> None:
    for key in ("row_id", "sample_id"):
        pred_values = set(predictions[key])
        manifest_values = set(manifest[key])
        missing = len(manifest_values - pred_values)
        extra = len(pred_values - manifest_values)
        _require(
            checks,
            f"coverage_predictions_{key}",
            "join",
            "exact one-to-one key coverage",
            f"missing={missing}; extra={extra}",
            missing == 0 and extra == 0,
        )
    _require(
        checks,
        "coverage_predictions_rows",
        "join",
        f"{len(manifest)} prediction rows",
        f"{len(predictions)} prediction rows",
        len(predictions) == len(manifest),
    )


def _validate_reference_coverage(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    sources: pd.DataFrame,
    source_cases: pd.DataFrame,
    ground_truth: pd.DataFrame,
    checks: list[Check],
) -> None:
    """Require exact key coverage for every reference table, both directions."""
    references = (
        ("features", predictions, features, "sample_id"),
        ("sources", predictions, sources, "source_id"),
        ("source_cases", sources, source_cases, "source_id"),
        ("ground_truth", sources, ground_truth, "source_id"),
    )
    for label, anchor, reference, key in references:
        anchor_values = set(anchor[key])
        reference_values = set(reference[key])
        missing = len(anchor_values - reference_values)
        extra = len(reference_values - anchor_values)
        _require(
            checks,
            f"coverage_{label}_{key}",
            "join",
            f"exact {key} coverage between anchor and {label}",
            f"missing={missing}; extra={extra}",
            missing == 0 and extra == 0,
        )
        expected_rows = int(anchor[key].nunique())
        _require(
            checks,
            f"rows_{label}",
            "join",
            f"{expected_rows} {label} rows",
            f"{len(reference)} {label} rows",
            len(reference) == expected_rows,
        )


def _validate_model_provenance(
    root: Path,
    pins: Mapping[str, InputPin],
    run_metadata: Mapping[str, Any],
    checks: list[Check],
) -> None:
    inference = run_metadata.get("inference")
    _require(
        checks,
        "run_metadata_inference_present",
        "model_provenance",
        "an inference section",
        "present" if isinstance(inference, dict) else "missing",
        isinstance(inference, dict),
    )
    assert isinstance(inference, dict)
    _require(
        checks,
        "model_fit_not_performed",
        "model_provenance",
        "inference.model_fit_performed is false",
        str(inference.get("model_fit_performed")),
        inference.get("model_fit_performed") is False,
    )
    _require(
        checks,
        "calibration_not_performed",
        "model_provenance",
        "inference.calibration_performed is false",
        str(inference.get("calibration_performed")),
        inference.get("calibration_performed") is False,
    )
    _require(
        checks,
        "feature_change_not_performed",
        "model_provenance",
        "inference.feature_change_performed is false",
        str(inference.get("feature_change_performed")),
        inference.get("feature_change_performed") is False,
    )
    _require(
        checks,
        "frozen_feature_names",
        "model_provenance",
        str(FEATURE_NAMES),
        str(tuple(inference.get("feature_names", ()))),
        tuple(inference.get("feature_names", ())) == FEATURE_NAMES,
    )
    _require(
        checks,
        "prediction_count",
        "model_provenance",
        "6300 predictions",
        str(inference.get("prediction_count")),
        inference.get("prediction_count") == 6300,
    )
    _require(
        checks,
        "model_family",
        "model_provenance",
        "expanded_random_forest",
        str(inference.get("model_family")),
        inference.get("model_family") == "expanded_random_forest",
    )
    _require(
        checks,
        "model_artifact_hash",
        "model_provenance",
        pins["model_artifact"].sha256,
        str(inference.get("artifact_sha256")),
        inference.get("artifact_sha256") == pins["model_artifact"].sha256,
    )
    _require(
        checks,
        "model_card_hash",
        "model_provenance",
        pins["model_card"].sha256,
        str(inference.get("card_sha256")),
        inference.get("card_sha256") == pins["model_card"].sha256,
    )
    model_card = json.loads((root / pins["model_card"].path).read_text(encoding="utf-8"))
    _require(
        checks,
        "model_card_artifact_hash",
        "model_provenance",
        pins["model_artifact"].sha256,
        str(model_card.get("artifact_sha256")),
        model_card.get("artifact_sha256") == pins["model_artifact"].sha256,
    )
    _require(
        checks,
        "model_card_feature_names",
        "model_provenance",
        str(FEATURE_NAMES),
        str(tuple(model_card.get("feature_names", ()))),
        tuple(model_card.get("feature_names", ())) == FEATURE_NAMES,
    )
    _require(
        checks,
        "model_card_target",
        "model_provenance",
        "ssim_target",
        str(model_card.get("target")),
        model_card.get("target") == "ssim_target",
    )


def _validate_ground_truth_policy(
    root: Path,
    ground_truth_path: str,
    checks: list[Check],
) -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", ground_truth_path],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    ignored = subprocess.run(
        ["git", "check-ignore", ground_truth_path],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    _require(
        checks,
        "ground_truth_untracked",
        "ground_truth_policy",
        "ground_truth.csv is not tracked",
        "tracked" if tracked.returncode == 0 else "untracked",
        tracked.returncode != 0,
    )
    _require(
        checks,
        "ground_truth_ignored",
        "ground_truth_policy",
        "ground_truth.csv is ignored",
        "ignored" if ignored.returncode == 0 else "not ignored",
        ignored.returncode == 0,
    )
    _require(
        checks,
        "ground_truth_text_not_loaded",
        "ground_truth_policy",
        "only source_id, case_id and font_id are loaded",
        "text column excluded at read time",
        True,
    )


def _build_joined_frame(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    sources: pd.DataFrame,
    source_cases: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    manifest_subset = manifest.loc[
        :,
        [
            "row_id",
            "sample_id",
            "source_id",
            "case_id",
            "case_index",
            "font_id",
            "variant_family",
            "severity_level",
            "protocol_scope",
            "ssim_target",
        ],
    ].rename(
        columns={
            column: f"{column}_manifest"
            for column in (
                "sample_id",
                "source_id",
                "case_id",
                "case_index",
                "font_id",
                "variant_family",
                "severity_level",
                "protocol_scope",
                "ssim_target",
            )
        }
    )
    joined = predictions.merge(manifest_subset, on="row_id", how="left", validate="one_to_one")
    for column in (
        "sample_id",
        "source_id",
        "case_id",
        "case_index",
        "font_id",
        "variant_family",
        "severity_level",
        "protocol_scope",
    ):
        _same_values(joined[column], joined[f"{column}_manifest"], f"predictions/manifest {column}")
    _same_float_values(
        joined["ssim_target"],
        joined["ssim_target_manifest"],
        "predictions/manifest ssim_target",
    )
    joined = joined.drop(
        columns=[
            f"{column}_manifest"
            for column in (
                "sample_id",
                "source_id",
                "case_id",
                "case_index",
                "font_id",
                "variant_family",
                "severity_level",
                "protocol_scope",
                "ssim_target",
            )
        ]
    )

    joined = joined.merge(features, on="sample_id", how="left", validate="one_to_one")
    source_subset = sources.loc[
        :,
        [
            "source_id",
            "case_id",
            "case_index",
            "split_group_id",
            "split",
            "text_length",
            "background_level",
        ],
    ].rename(
        columns={
            column: f"{column}_source"
            for column in (
                "case_id",
                "case_index",
                "split_group_id",
                "split",
                "text_length",
                "background_level",
            )
        }
    )
    joined = joined.merge(source_subset, on="source_id", how="left", validate="many_to_one")
    for column in ("case_id", "case_index"):
        _same_values(joined[column], joined[f"{column}_source"], f"predictions/sources {column}")
    joined = joined.drop(columns=[f"{column}_source" for column in ("case_id", "case_index")])
    joined = joined.rename(
        columns={
            "split_group_id_source": "split_group_id",
            "split_source": "split",
            "text_length_source": "text_length",
            "background_level_source": "background_level",
        }
    )

    case_subset = source_cases.loc[
        :,
        ["source_id", "case_id", "case_index", "font_id", "text_length", "split", "split_group_id"],
    ].rename(
        columns={
            column: f"{column}_case"
            for column in (
                "case_id",
                "case_index",
                "font_id",
                "text_length",
                "split",
                "split_group_id",
            )
        }
    )
    joined = joined.merge(case_subset, on="source_id", how="left", validate="many_to_one")
    for column in ("case_id", "case_index", "font_id"):
        _same_values(joined[column], joined[f"{column}_case"], f"predictions/source_cases {column}")
    _same_values(
        joined["text_length"], joined["text_length_case"], "sources/source_cases text_length"
    )
    _same_values(joined["split"], joined["split_case"], "sources/source_cases split")
    _same_values(
        joined["split_group_id"],
        joined["split_group_id_case"],
        "sources/source_cases split_group_id",
    )
    joined = joined.drop(
        columns=[
            f"{column}_case"
            for column in (
                "case_id",
                "case_index",
                "font_id",
                "text_length",
                "split",
                "split_group_id",
            )
        ]
    )

    gt_subset = ground_truth.rename(
        columns={
            "case_id": "case_id_gt",
            "font_id": "font_id_gt",
        }
    )
    joined = joined.merge(gt_subset, on="source_id", how="left", validate="many_to_one")
    _same_values(joined["case_id"], joined["case_id_gt"], "predictions/ground_truth case_id")
    _same_values(joined["font_id"], joined["font_id_gt"], "predictions/ground_truth font_id")
    joined = joined.drop(columns=["case_id_gt", "font_id_gt"])

    joined = joined.copy()
    joined["residual_recomputed"] = joined["ssim_target"] - joined["prediction"]
    joined["abs_residual_recomputed"] = joined["residual_recomputed"].abs()
    return joined


def _validate_joined_structure(
    joined: pd.DataFrame,
    config: AnalysisConfig,
    checks: list[Check],
) -> None:
    _require_no_missing(
        joined,
        [
            "row_id",
            "sample_id",
            "source_id",
            "case_id",
            "font_id",
            "font_group",
            "variant_family",
            "severity_level",
            "protocol_scope",
            "ssim_target",
            "prediction",
            *FEATURE_NAMES,
            "background_level",
            "text_length",
            "split",
        ],
        "joined analysis frame",
    )
    _require_unique(joined, "row_id", "joined analysis frame")
    _require_unique(joined, "sample_id", "joined analysis frame")
    _require(
        checks,
        "joined_row_count",
        "join",
        "6300 joined rows",
        str(len(joined)),
        len(joined) == 6300,
    )
    _require(
        checks,
        "joined_source_count",
        "join",
        "300 source groups",
        str(joined["source_id"].nunique()),
        joined["source_id"].nunique() == 300,
    )
    _require(
        checks,
        "joined_font_count",
        "join",
        "10 fonts",
        str(joined["font_id"].nunique()),
        joined["font_id"].nunique() == 10,
    )
    _require(
        checks,
        "joined_case_count",
        "join",
        "30 text cases",
        str(joined["case_id"].nunique()),
        joined["case_id"].nunique() == 30,
    )
    target = joined["ssim_target"].to_numpy(dtype=np.float64)
    prediction = joined["prediction"].to_numpy(dtype=np.float64)
    _require(
        checks,
        "target_prediction_finite",
        "values",
        "all target and prediction values finite",
        f"target_finite={np.isfinite(target).all()}; prediction_finite={np.isfinite(prediction).all()}",
        bool(np.isfinite(target).all() and np.isfinite(prediction).all()),
    )
    _require(
        checks,
        "target_prediction_range",
        "values",
        "target and prediction in [0, 1]",
        f"target=[{target.min():.6f}, {target.max():.6f}]; prediction=[{prediction.min():.6f}, {prediction.max():.6f}]",
        bool(
            target.min() >= 0.0
            and target.max() <= 1.0
            and prediction.min() >= 0.0
            and prediction.max() <= 1.0
        ),
    )
    residual = target - prediction
    residual_delta = float(np.max(np.abs(residual - joined["residual"].to_numpy(dtype=np.float64))))
    abs_delta = float(
        np.max(np.abs(np.abs(residual) - joined["abs_residual"].to_numpy(dtype=np.float64)))
    )
    _require(
        checks,
        "residual_definition",
        "values",
        "recorded residual equals ssim_target - prediction",
        f"max_abs_delta={residual_delta:.3e}",
        residual_delta <= 1e-12,
    )
    _require(
        checks,
        "abs_residual_definition",
        "values",
        "recorded abs_residual equals |ssim_target - prediction|",
        f"max_abs_delta={abs_delta:.3e}",
        abs_delta <= 1e-12,
    )

    source_sizes = joined.groupby("source_id", sort=True).size()
    _require(
        checks,
        "source_row_count",
        "source_structure",
        "every source has 21 rows",
        f"unique_sizes={sorted(source_sizes.unique().tolist())}",
        bool((source_sizes == 21).all()),
    )
    expected_counts = (
        joined.groupby(["variant_family", "protocol_scope"], sort=True).size().to_dict()
    )
    expected = {
        ("clean", "diagnostic"): 300,
        ("elastic", "diagnostic"): 300,
        ("elastic", "primary"): 900,
        ("gaussian_blur", "primary"): 1200,
        ("gaussian_noise", "primary"): 1200,
        ("thin_lines", "primary"): 1200,
        ("arc_strokes", "primary"): 1200,
    }
    _require(
        checks,
        "condition_row_counts",
        "design",
        str(expected),
        str(expected_counts),
        expected_counts == expected,
    )
    primary = joined["protocol_scope"].eq("primary")
    _require(
        checks,
        "primary_row_count",
        "design",
        "5700 primary rows",
        str(int(primary.sum())),
        int(primary.sum()) == 5700,
    )
    _require(
        checks,
        "diagnostic_row_count",
        "design",
        "600 diagnostic rows",
        str(int((~primary).sum())),
        int((~primary).sum()) == 600,
    )
    font_case = joined.loc[primary, ["font_id", "case_id", "source_id"]].drop_duplicates()
    _require(
        checks,
        "font_case_source_design",
        "design",
        "300 unique font-case sources, one per font and case",
        f"rows={len(font_case)}; unique_sources={font_case['source_id'].nunique()}",
        len(font_case) == 300 and font_case["source_id"].nunique() == 300,
    )
    per_font_case = font_case.groupby("font_id", sort=True).size()
    _require(
        checks,
        "font_case_balance",
        "design",
        "every font observes all 30 cases",
        f"unique_counts={sorted(per_font_case.unique().tolist())}",
        bool((per_font_case == 30).all()),
    )
    _require(
        checks,
        "interference_rows",
        "design",
        "2400 interference rows (1200 thin + 1200 arc)",
        str(int(joined["variant_family"].isin(INTERFERENCE_FAMILIES).sum())),
        int(joined["variant_family"].isin(INTERFERENCE_FAMILIES).sum()) == 2400,
    )
    _require(
        checks,
        "r2_suppression_rule",
        "metrics",
        "target SD < 0.005 suppresses R²",
        f"near_degenerate_sd={config.analysis.near_degenerate_sd}",
        config.analysis.near_degenerate_sd == NEAR_DEGENERATE_SD,
    )


def _rehash_inputs(
    root: Path,
    pins: Mapping[str, InputPin],
    checks: list[Check],
    *,
    label: str,
) -> None:
    mismatches: list[str] = []
    for key, pin in pins.items():
        path = root / pin.path
        if not path.is_file() or sha256_file(path) != pin.sha256:
            mismatches.append(key)
    _require(
        checks,
        f"inputs_unchanged_{label}",
        "immutability",
        "all pinned inputs unchanged",
        "unchanged" if not mismatches else f"changed={mismatches}",
        not mismatches,
    )


# ---------------------------------------------------------------------------
# descriptive metrics and source-grouped uncertainty
# ---------------------------------------------------------------------------


def scope_frame(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Return the rows for one predeclared M9A scope."""
    if scope not in SCOPE_DESCRIPTIONS and "_s" not in scope:
        raise AnalysisError(f"unknown scope: {scope}")
    if scope == "all_primary":
        selected = frame["protocol_scope"].eq("primary")
    elif scope == "original_blur_noise":
        selected = frame["protocol_scope"].eq("primary") & frame["variant_family"].isin(
            ["gaussian_blur", "gaussian_noise"]
        )
    elif scope == "gaussian_blur":
        selected = frame["protocol_scope"].eq("primary") & frame["variant_family"].eq(
            "gaussian_blur"
        )
    elif scope == "gaussian_noise":
        selected = frame["protocol_scope"].eq("primary") & frame["variant_family"].eq(
            "gaussian_noise"
        )
    elif scope == "interference":
        selected = frame["protocol_scope"].eq("primary") & frame["variant_family"].isin(
            INTERFERENCE_FAMILIES
        )
    elif scope == "thin_lines":
        selected = frame["protocol_scope"].eq("primary") & frame["variant_family"].eq("thin_lines")
    elif scope == "arc_strokes":
        selected = frame["protocol_scope"].eq("primary") & frame["variant_family"].eq("arc_strokes")
    elif scope == "geometric_primary":
        selected = frame["protocol_scope"].eq("primary") & frame["variant_family"].eq("elastic")
    elif scope == "clean_reference":
        selected = frame["protocol_scope"].eq("diagnostic") & frame["variant_family"].eq("clean")
    elif (
        scope.endswith("_s1")
        or scope.endswith("_s2")
        or scope.endswith("_s3")
        or scope.endswith("_s4")
    ):
        family, severity_text = scope.rsplit("_s", 1)
        if family not in SEVERITY_FAMILIES:
            raise AnalysisError(f"unknown severity family in scope: {scope}")
        selected = (
            frame["protocol_scope"].eq("primary")
            & frame["variant_family"].eq(family)
            & frame["severity_level"].eq(int(severity_text))
        )
    else:
        raise AnalysisError(f"unknown scope: {scope}")
    result = frame.loc[selected].copy()
    if result.empty:
        raise AnalysisError(f"scope {scope!r} is empty")
    return result


def _scope_role(scope: str) -> str:
    return "diagnostic" if scope == "clean_reference" else "primary"


def _point_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    near_degenerate_sd: float,
    large_error_threshold: float,
) -> dict[str, Any]:
    if target.size == 0 or prediction.size == 0:
        raise AnalysisError("metrics require at least one row")
    if target.shape != prediction.shape:
        raise AnalysisError(
            f"target/prediction shapes differ: {target.shape} vs {prediction.shape}"
        )
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise AnalysisError("metrics require finite target and prediction values")
    residual = target - prediction
    abs_residual = np.abs(residual)
    ss_res = float(np.sum(residual**2))
    target_mean = float(np.mean(target))
    target_sd = float(np.std(target, ddof=0))
    ss_tot = float(np.sum((target - target_mean) ** 2))
    r2_defined = bool(target_sd >= near_degenerate_sd and ss_tot > 0.0)
    r2 = float(1.0 - ss_res / ss_tot) if r2_defined else math.nan
    if target_sd < near_degenerate_sd:
        r2_status = f"suppressed_target_sd_lt_{near_degenerate_sd:g}"
    elif ss_tot <= 0.0:
        r2_status = "suppressed_constant_target"
    else:
        r2_status = "reported"
    prediction_mean = float(np.mean(prediction))
    prediction_sd = float(np.std(prediction, ddof=0))
    prediction_target_sd_ratio = float(prediction_sd / target_sd) if target_sd > 0.0 else math.nan
    if target_sd > 0.0 and prediction_sd > 0.0:
        prediction_target_corr = float(np.corrcoef(target, prediction)[0, 1])
    else:
        prediction_target_corr = math.nan
    if target_sd > 0.0:
        slope, intercept = np.polyfit(target, prediction, 1)
        calibration_slope = float(slope)
        calibration_intercept = float(intercept)
    else:
        calibration_slope = math.nan
        calibration_intercept = math.nan
    if target.size > 2 and target_sd > 0.0 and np.std(residual, ddof=0) > 0.0:
        residual_target_spearman = float(stats.spearmanr(target, residual).statistic)
    else:
        residual_target_spearman = math.nan
    return {
        "n_rows": int(target.size),
        "mae": float(np.mean(abs_residual)),
        "rmse": float(math.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(residual)),
        "sd_residual": float(np.std(residual, ddof=0)),
        "r2": r2,
        "r2_defined": r2_defined,
        "r2_suppressed": not r2_defined,
        "r2_status": r2_status,
        "target_mean": target_mean,
        "target_sd": target_sd,
        "target_variance": float(np.var(target, ddof=0)),
        "target_min": float(np.min(target)),
        "target_max": float(np.max(target)),
        "target_q05": float(np.quantile(target, 0.05)),
        "target_q25": float(np.quantile(target, 0.25)),
        "target_q50": float(np.quantile(target, 0.50)),
        "target_q75": float(np.quantile(target, 0.75)),
        "target_q95": float(np.quantile(target, 0.95)),
        "prediction_mean": prediction_mean,
        "prediction_sd": prediction_sd,
        "prediction_min": float(np.min(prediction)),
        "prediction_max": float(np.max(prediction)),
        "prediction_target_sd_ratio": prediction_target_sd_ratio,
        "prediction_target_corr": prediction_target_corr,
        "calibration_slope": calibration_slope,
        "calibration_intercept": calibration_intercept,
        "residual_target_spearman": residual_target_spearman,
        "residual_q05": float(np.quantile(residual, 0.05)),
        "residual_q25": float(np.quantile(residual, 0.25)),
        "residual_q50": float(np.quantile(residual, 0.50)),
        "residual_q75": float(np.quantile(residual, 0.75)),
        "residual_q95": float(np.quantile(residual, 0.95)),
        "abs_error_q50": float(np.quantile(abs_residual, 0.50)),
        "abs_error_q90": float(np.quantile(abs_residual, 0.90)),
        "abs_error_q95": float(np.quantile(abs_residual, 0.95)),
        "abs_error_q99": float(np.quantile(abs_residual, 0.99)),
        "large_error_count": int(np.sum(abs_residual >= large_error_threshold)),
        "large_error_proportion": float(np.mean(abs_residual >= large_error_threshold)),
        "large_error_threshold": float(large_error_threshold),
    }


def compute_metrics(
    frame: pd.DataFrame,
    *,
    near_degenerate_sd: float = NEAR_DEGENERATE_SD,
    large_error_threshold: float = LARGE_ERROR_THRESHOLD,
) -> dict[str, Any]:
    """Compute descriptive metrics for an aligned subset."""
    target = frame["ssim_target"].to_numpy(dtype=np.float64)
    prediction = frame["prediction"].to_numpy(dtype=np.float64)
    result = _point_metrics(
        target,
        prediction,
        near_degenerate_sd=near_degenerate_sd,
        large_error_threshold=large_error_threshold,
    )
    result["n_sources"] = int(frame["source_id"].nunique())
    result["n_fonts"] = int(frame["font_id"].nunique())
    result["n_cases"] = int(frame["case_id"].nunique())
    return result


def _within_source_response(frame: pd.DataFrame) -> dict[str, float]:
    ratios: list[float] = []
    slopes: list[float] = []
    correlations: list[float] = []
    for _, group in frame.groupby("source_id", sort=True):
        target = group["ssim_target"].to_numpy(dtype=np.float64)
        prediction = group["prediction"].to_numpy(dtype=np.float64)
        target_sd = float(np.std(target, ddof=0))
        prediction_sd = float(np.std(prediction, ddof=0))
        if target_sd > 0.0:
            ratios.append(prediction_sd / target_sd)
            slope, _ = np.polyfit(target, prediction, 1)
            slopes.append(float(slope))
        if target_sd > 0.0 and prediction_sd > 0.0:
            correlations.append(float(np.corrcoef(target, prediction)[0, 1]))
    return {
        "within_source_prediction_target_sd_ratio_median": float(np.median(ratios))
        if ratios
        else math.nan,
        "within_source_calibration_slope_median": float(np.median(slopes)) if slopes else math.nan,
        "within_source_prediction_target_corr_median": float(np.median(correlations))
        if correlations
        else math.nan,
    }


def _source_sse_and_mae(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    sse = frame.groupby("source_id", sort=True).apply(
        lambda group: float(
            np.sum(
                (
                    group["ssim_target"].to_numpy(dtype=np.float64)
                    - group["prediction"].to_numpy(dtype=np.float64)
                )
                ** 2
            )
        ),
        include_groups=False,
    )
    mae = frame.groupby("source_id", sort=True)["abs_residual"].mean()
    return sse, mae


def _concentration_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    sse, mae = _source_sse_and_mae(frame)
    total = float(sse.sum())
    sorted_sse = np.sort(sse.to_numpy(dtype=np.float64))[::-1]
    n_sources = len(sorted_sse)

    def top_share(fraction: float) -> float:
        count = max(1, int(math.ceil(fraction * n_sources)))
        return float(np.sum(sorted_sse[:count]) / total) if total > 0.0 else math.nan

    return {
        "n_sources": n_sources,
        "total_sse": total,
        "top_5pct_source_sse_share": top_share(0.05),
        "top_10pct_source_sse_share": top_share(0.10),
        "top_20pct_source_sse_share": top_share(0.20),
        "max_source_sse_share": float(sorted_sse[0] / total) if total > 0.0 else math.nan,
        "source_sse_cv": float(np.std(sorted_sse, ddof=0) / np.mean(sorted_sse))
        if np.mean(sorted_sse) > 0.0
        else math.nan,
        "source_mae_q50": float(np.quantile(mae.to_numpy(dtype=np.float64), 0.50)),
        "source_mae_q90": float(np.quantile(mae.to_numpy(dtype=np.float64), 0.90)),
        "source_mae_q95": float(np.quantile(mae.to_numpy(dtype=np.float64), 0.95)),
    }


def source_bootstrap_indices(
    source_ids: Sequence[str],
    *,
    n_bootstrap: int,
    seed: int,
) -> list[np.ndarray]:
    """Return whole-source-group bootstrap draws.

    Each element is an array of source identifiers of length ``n_groups``. Every
    row of a selected source remains in the same replicate; individual images
    are never resampled.
    """
    groups = np.array(sorted(set(source_ids)), dtype=object)
    if groups.size == 0:
        raise AnalysisError("cannot bootstrap an empty source set")
    rng = np.random.default_rng(seed)
    return [rng.choice(groups, size=groups.size, replace=True) for _ in range(n_bootstrap)]


def _fast_point_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    near_degenerate_sd: float,
) -> dict[str, float]:
    residual = target - prediction
    target_mean = float(np.mean(target))
    target_sd = float(np.std(target, ddof=0))
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((target - target_mean) ** 2))
    r2 = (
        float(1.0 - ss_res / ss_tot)
        if target_sd >= near_degenerate_sd and ss_tot > 0.0
        else math.nan
    )
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(math.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(residual)),
        "r2": r2,
    }


def _bootstrap_metric_ci(
    frame: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
    confidence_level: float,
    near_degenerate_sd: float,
) -> dict[str, Any]:
    target = frame["ssim_target"].to_numpy(dtype=np.float64)
    prediction = frame["prediction"].to_numpy(dtype=np.float64)
    indices = {
        group: np.asarray(index, dtype=np.int64)
        for group, index in frame.groupby("source_id", sort=True).indices.items()
    }
    groups = np.array(sorted(indices), dtype=object)
    rng = np.random.default_rng(seed)
    mae_values: list[float] = []
    rmse_values: list[float] = []
    bias_values: list[float] = []
    r2_values: list[float] = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(groups, size=groups.size, replace=True)
        index = np.concatenate([indices[group] for group in sampled])
        metrics = _fast_point_metrics(
            target[index],
            prediction[index],
            near_degenerate_sd=near_degenerate_sd,
        )
        mae_values.append(metrics["mae"])
        rmse_values.append(metrics["rmse"])
        bias_values.append(metrics["bias"])
        if math.isfinite(metrics["r2"]):
            r2_values.append(metrics["r2"])
    low = 100.0 * (1.0 - confidence_level) / 2.0
    high = 100.0 - low

    def interval(values: list[float]) -> tuple[float, float]:
        if not values:
            return math.nan, math.nan
        array = np.asarray(values, dtype=np.float64)
        return float(np.percentile(array, low)), float(np.percentile(array, high))

    mae_low, mae_high = interval(mae_values)
    rmse_low, rmse_high = interval(rmse_values)
    bias_low, bias_high = interval(bias_values)
    r2_low, r2_high = interval(r2_values)
    return {
        "mae_ci_low": mae_low,
        "mae_ci_high": mae_high,
        "rmse_ci_low": rmse_low,
        "rmse_ci_high": rmse_high,
        "bias_ci_low": bias_low,
        "bias_ci_high": bias_high,
        "r2_ci_low": r2_low,
        "r2_ci_high": r2_high,
        "r2_ci_defined": bool(r2_values),
        "n_bootstrap": int(n_bootstrap),
        "bootstrap_seed": int(seed),
        "bootstrap_low_pct": float(low),
        "bootstrap_high_pct": float(high),
        "n_bootstrap_sources": int(groups.size),
    }


def paired_bootstrap_ci(
    values: Sequence[float],
    *,
    n_bootstrap: int,
    seed: int,
    confidence_level: float,
) -> tuple[float, float]:
    """Percentile interval for the mean of paired source/case values."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = [
        float(np.mean(rng.choice(array, size=array.size, replace=True))) for _ in range(n_bootstrap)
    ]
    low = 100.0 * (1.0 - confidence_level) / 2.0
    high = 100.0 - low
    return float(np.percentile(means, low)), float(np.percentile(means, high))


def _bh_qvalues(pvalues: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg q-values, preserving input order and NaN values."""
    array = np.asarray(pvalues, dtype=np.float64)
    result = np.full(array.shape, math.nan)
    finite = np.isfinite(array)
    if not finite.any():
        return result.tolist()
    values = array[finite]
    order = np.argsort(values)[::-1]
    ranked = values[order]
    count = len(ranked)
    adjusted = np.empty(count, dtype=np.float64)
    running = 1.0
    for position, pvalue in enumerate(ranked):
        rank = count - position
        running = min(running, float(pvalue) * count / rank)
        adjusted[position] = running
    adjusted = np.clip(adjusted, 0.0, 1.0)
    finite_positions = np.flatnonzero(finite)
    result[finite_positions[order]] = adjusted
    return result.tolist()


def _spearman(frame: pd.DataFrame, feature: str, outcome: str) -> tuple[float, float, int]:
    subset = frame.loc[:, [feature, outcome]].dropna()
    if len(subset) < 3 or subset[feature].nunique() < 2 or subset[outcome].nunique() < 2:
        return math.nan, math.nan, int(len(subset))
    result = stats.spearmanr(subset[feature], subset[outcome])
    return float(result.statistic), float(result.pvalue), int(len(subset))


# ---------------------------------------------------------------------------
# predeclared analysis tables
# ---------------------------------------------------------------------------


def _summary_row(frame: pd.DataFrame, scope: str, config: AnalysisConfig) -> dict[str, Any]:
    metrics = compute_metrics(
        frame,
        near_degenerate_sd=config.analysis.near_degenerate_sd,
        large_error_threshold=config.analysis.large_error_threshold,
    )
    bootstrap = _bootstrap_metric_ci(
        frame,
        n_bootstrap=config.run.n_bootstrap,
        seed=config.run.seed,
        confidence_level=config.run.confidence_level,
        near_degenerate_sd=config.analysis.near_degenerate_sd,
    )
    row: dict[str, Any] = {
        "scope": scope,
        "description": SCOPE_DESCRIPTIONS[scope],
        "protocol_role": _scope_role(scope),
        **metrics,
        **bootstrap,
        **_within_source_response(frame),
        **_concentration_metrics(frame),
    }
    return row


def build_summary_metrics(joined: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    rows = [_summary_row(scope_frame(joined, scope), scope, config) for scope in SUMMARY_SCOPES]
    return pd.DataFrame(rows)


def build_condition_calibration(joined: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope in SUMMARY_SCOPES:
        frame = scope_frame(joined, scope)
        metrics = compute_metrics(
            frame,
            near_degenerate_sd=config.analysis.near_degenerate_sd,
            large_error_threshold=config.analysis.large_error_threshold,
        )
        rows.append(
            {
                "scope": scope,
                "description": SCOPE_DESCRIPTIONS[scope],
                "protocol_role": _scope_role(scope),
                "n_rows": metrics["n_rows"],
                "n_sources": metrics["n_sources"],
                "target_mean": metrics["target_mean"],
                "target_sd": metrics["target_sd"],
                "prediction_mean": metrics["prediction_mean"],
                "prediction_sd": metrics["prediction_sd"],
                "prediction_target_sd_ratio": metrics["prediction_target_sd_ratio"],
                "prediction_target_corr": metrics["prediction_target_corr"],
                "calibration_slope": metrics["calibration_slope"],
                "calibration_intercept": metrics["calibration_intercept"],
                "residual_target_spearman": metrics["residual_target_spearman"],
                "mean_residual": metrics["bias"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "r2_status": metrics["r2_status"],
                **_within_source_response(frame),
            }
        )
    return pd.DataFrame(rows)


def _paired_font_outcomes(
    frame: pd.DataFrame,
    scope: str,
) -> tuple[pd.Series, pd.Series]:
    scoped = scope_frame(frame, scope)
    per_source = scoped.groupby(["font_id", "case_id"], sort=True)["abs_residual"].mean()
    wide = per_source.unstack("font_id")
    return wide, per_source


def build_font_condition_metrics(joined: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    font_groups = (
        joined.loc[:, ["font_id", "font_group"]]
        .drop_duplicates()
        .set_index("font_id")["font_group"]
        .to_dict()
    )
    interference = scope_frame(joined, "interference")
    interference_targets = interference.groupby(["font_id", "severity_level"], sort=True)[
        "ssim_target"
    ].mean()
    for scope in FONT_SCOPES:
        wide, _ = _paired_font_outcomes(joined, scope)
        for font_id in sorted(wide.columns):
            frame = scope_frame(joined, scope)
            font_frame = frame.loc[frame["font_id"].eq(font_id)]
            metrics = compute_metrics(
                font_frame,
                near_degenerate_sd=config.analysis.near_degenerate_sd,
                large_error_threshold=config.analysis.large_error_threshold,
            )
            bootstrap = _bootstrap_metric_ci(
                font_frame,
                n_bootstrap=config.run.n_bootstrap,
                seed=config.run.seed,
                confidence_level=config.run.confidence_level,
                near_degenerate_sd=config.analysis.near_degenerate_sd,
            )
            other_mean = wide.drop(columns=[font_id]).mean(axis=1)
            differences = (wide[font_id] - other_mean).dropna()
            paired_low, paired_high = paired_bootstrap_ci(
                differences.to_numpy(dtype=np.float64),
                n_bootstrap=config.run.n_bootstrap,
                seed=config.run.seed,
                confidence_level=config.run.confidence_level,
            )
            if scope == "interference":
                target_s1 = float(interference_targets.loc[(font_id, 1)])
                target_s4 = float(interference_targets.loc[(font_id, 4)])
                target_drop = target_s1 - target_s4
            else:
                target_s1 = math.nan
                target_s4 = math.nan
                target_drop = math.nan
            rows.append(
                {
                    "font_id": font_id,
                    "font_group": font_groups[font_id],
                    "scope": scope,
                    "description": SCOPE_DESCRIPTIONS[scope],
                    "n_rows": metrics["n_rows"],
                    "n_sources": metrics["n_sources"],
                    "n_cases": metrics["n_cases"],
                    "mae": metrics["mae"],
                    "mae_ci_low": bootstrap["mae_ci_low"],
                    "mae_ci_high": bootstrap["mae_ci_high"],
                    "rmse": metrics["rmse"],
                    "rmse_ci_low": bootstrap["rmse_ci_low"],
                    "rmse_ci_high": bootstrap["rmse_ci_high"],
                    "bias": metrics["bias"],
                    "bias_ci_low": bootstrap["bias_ci_low"],
                    "bias_ci_high": bootstrap["bias_ci_high"],
                    "r2": metrics["r2"],
                    "r2_status": metrics["r2_status"],
                    "target_mean": metrics["target_mean"],
                    "target_sd": metrics["target_sd"],
                    "prediction_mean": metrics["prediction_mean"],
                    "prediction_sd": metrics["prediction_sd"],
                    "prediction_target_sd_ratio": metrics["prediction_target_sd_ratio"],
                    "calibration_slope": metrics["calibration_slope"],
                    "target_s1_mean": target_s1,
                    "target_s4_mean": target_s4,
                    "target_drop_s1_to_s4": target_drop,
                    "large_error_count": metrics["large_error_count"],
                    "large_error_proportion": metrics["large_error_proportion"],
                    "paired_difference_mean": float(differences.mean()),
                    "paired_difference_ci_low": paired_low,
                    "paired_difference_ci_high": paired_high,
                    "paired_cases_worse": int((differences > 0).sum()),
                    "n_paired_cases": int(len(differences)),
                    "paired_difference_rule": config.analysis.font_contrast_rule,
                    "n_bootstrap": config.run.n_bootstrap,
                }
            )
    return pd.DataFrame(rows)


def build_severity_metrics(joined: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    expected = {
        (family, severity)
        for family, severities in PRIMARY_SEVERITIES.items()
        for severity in severities
    }
    observed = {
        (family, int(severity))
        for family, severity in joined.loc[
            joined["protocol_scope"].eq("primary"), ["variant_family", "severity_level"]
        ]
        .drop_duplicates()
        .to_numpy()
    }
    missing = sorted(expected - observed)
    if missing:
        raise AnalysisError(f"missing primary severity group(s): {missing}")
    for family in SEVERITY_FAMILIES:
        for severity in PRIMARY_SEVERITIES[family]:
            frame = scope_frame(joined, f"{family}_s{severity}")
            metrics = compute_metrics(
                frame,
                near_degenerate_sd=config.analysis.near_degenerate_sd,
                large_error_threshold=config.analysis.large_error_threshold,
            )
            bootstrap = _bootstrap_metric_ci(
                frame,
                n_bootstrap=config.run.n_bootstrap,
                seed=config.run.seed,
                confidence_level=config.run.confidence_level,
                near_degenerate_sd=config.analysis.near_degenerate_sd,
            )
            rows.append(
                {
                    "family": family,
                    "severity_level": severity,
                    "scope": f"{family}_s{severity}",
                    "n_rows": metrics["n_rows"],
                    "n_sources": metrics["n_sources"],
                    "target_mean": metrics["target_mean"],
                    "target_sd": metrics["target_sd"],
                    "prediction_mean": metrics["prediction_mean"],
                    "prediction_sd": metrics["prediction_sd"],
                    "prediction_target_sd_ratio": metrics["prediction_target_sd_ratio"],
                    "calibration_slope": metrics["calibration_slope"],
                    "mae": metrics["mae"],
                    "mae_ci_low": bootstrap["mae_ci_low"],
                    "mae_ci_high": bootstrap["mae_ci_high"],
                    "rmse": metrics["rmse"],
                    "rmse_ci_low": bootstrap["rmse_ci_low"],
                    "rmse_ci_high": bootstrap["rmse_ci_high"],
                    "bias": metrics["bias"],
                    "bias_ci_low": bootstrap["bias_ci_low"],
                    "bias_ci_high": bootstrap["bias_ci_high"],
                    "r2": metrics["r2"],
                    "r2_status": metrics["r2_status"],
                    "large_error_count": metrics["large_error_count"],
                    "large_error_proportion": metrics["large_error_proportion"],
                    "large_error_threshold": metrics["large_error_threshold"],
                    "n_bootstrap": config.run.n_bootstrap,
                }
            )
    return pd.DataFrame(rows)


def build_source_group_metrics(joined: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope in SOURCE_SCOPES:
        frame = scope_frame(joined, scope)
        sse, _ = _source_sse_and_mae(frame)
        total_sse = float(sse.sum())
        for source_id, group in frame.groupby("source_id", sort=True):
            metrics = compute_metrics(
                group,
                near_degenerate_sd=config.analysis.near_degenerate_sd,
                large_error_threshold=config.analysis.large_error_threshold,
            )
            first = group.iloc[0]
            rows.append(
                {
                    "scope": scope,
                    "source_id": source_id,
                    "case_id": first["case_id"],
                    "case_index": int(first["case_index"]),
                    "font_id": first["font_id"],
                    "font_group": first["font_group"],
                    "split": first["split"],
                    "split_group_id": first["split_group_id"],
                    "background_level": int(first["background_level"]),
                    "text_length": int(first["text_length"]),
                    "n_rows": metrics["n_rows"],
                    "sse": float(sse.loc[source_id]),
                    "sse_share": float(sse.loc[source_id] / total_sse)
                    if total_sse > 0.0
                    else math.nan,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "bias": metrics["bias"],
                    "r2": metrics["r2"],
                    "r2_status": metrics["r2_status"],
                    "target_mean": metrics["target_mean"],
                    "target_sd": metrics["target_sd"],
                    "prediction_mean": metrics["prediction_mean"],
                    "prediction_sd": metrics["prediction_sd"],
                    "prediction_target_sd_ratio": metrics["prediction_target_sd_ratio"],
                    "calibration_slope": metrics["calibration_slope"],
                    "large_error_count": metrics["large_error_count"],
                    "large_error_proportion": metrics["large_error_proportion"],
                    "large_error_threshold": metrics["large_error_threshold"],
                }
            )
    result = pd.DataFrame(rows)
    result["source_mae_rank"] = (
        result.groupby("scope", sort=False)["mae"].rank(ascending=False, method="min").astype(int)
    )
    return result


def build_case_metrics(joined: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    frame = scope_frame(joined, "interference")
    rows: list[dict[str, Any]] = []
    for case_id, group in frame.groupby("case_id", sort=True):
        metrics = compute_metrics(
            group,
            near_degenerate_sd=config.analysis.near_degenerate_sd,
            large_error_threshold=config.analysis.large_error_threshold,
        )
        per_source = group.groupby("source_id", sort=True)["abs_residual"].mean()
        first = group.iloc[0]
        rows.append(
            {
                "case_id": case_id,
                "case_index": int(first["case_index"]),
                "text_length": int(first["text_length"]),
                "n_rows": metrics["n_rows"],
                "n_fonts": metrics["n_fonts"],
                "n_sources": metrics["n_sources"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "bias": metrics["bias"],
                "r2": metrics["r2"],
                "r2_status": metrics["r2_status"],
                "target_mean": metrics["target_mean"],
                "target_sd": metrics["target_sd"],
                "prediction_mean": metrics["prediction_mean"],
                "prediction_sd": metrics["prediction_sd"],
                "large_error_count": metrics["large_error_count"],
                "large_error_proportion": metrics["large_error_proportion"],
                "source_mae_min": float(per_source.min()),
                "source_mae_max": float(per_source.max()),
                "source_mae_sd": float(per_source.std(ddof=0)),
            }
        )
    result = pd.DataFrame(rows)
    result["case_mae_rank"] = result["mae"].rank(ascending=False, method="min").astype(int)
    return result


def build_source_characteristic_associations(
    joined: pd.DataFrame,
    config: AnalysisConfig,
    *,
    source_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if source_metrics is None:
        source_metrics = build_source_group_metrics(joined, config)
    interference = source_metrics.loc[source_metrics["scope"].eq("interference")]
    rows: list[dict[str, Any]] = []
    for font_id in ["all", *sorted(interference["font_id"].unique())]:
        subset = (
            interference
            if font_id == "all"
            else interference.loc[interference["font_id"].eq(font_id)]
        )
        for characteristic in ("background_level", "text_length"):
            rho, pvalue, n_rows = _spearman(subset, characteristic, "mae")
            rows.append(
                {
                    "scope": "interference_source_mae",
                    "font_id": font_id,
                    "characteristic": characteristic,
                    "n_sources": int(subset["source_id"].nunique()),
                    "n_rows": n_rows,
                    "spearman_rho": rho,
                    "p_value": pvalue,
                    "exploratory": True,
                    "rule": config.analysis.source_characteristic_rule,
                }
            )
    result = pd.DataFrame(rows)
    result["bh_q_value"] = _bh_qvalues(result["p_value"].tolist())
    result["direction"] = np.where(
        result["spearman_rho"] > 0.0,
        "positive",
        np.where(result["spearman_rho"] < 0.0, "negative", "zero"),
    )
    return result


def build_feature_error_associations(joined: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target_metric in ("abs_residual", "residual"):
        for scope in FEATURE_SCOPES:
            frame = scope_frame(joined, scope)
            for feature in FEATURE_NAMES:
                rho, pvalue, n_rows = _spearman(frame, feature, target_metric)
                rows.append(
                    {
                        "target_metric": target_metric,
                        "scope": scope,
                        "feature": feature,
                        "n_rows": n_rows,
                        "n_sources": int(frame["source_id"].nunique()),
                        "spearman_rho": rho,
                        "p_value": pvalue,
                        "exploratory": True,
                        "rule": config.analysis.feature_association_rule,
                    }
                )
    result = pd.DataFrame(rows)
    for metric in ("abs_residual", "residual"):
        mask = result["target_metric"].eq(metric)
        result.loc[mask, "bh_q_value"] = _bh_qvalues(result.loc[mask, "p_value"].tolist())
    result["direction"] = np.where(
        result["spearman_rho"] > 0.0,
        "positive",
        np.where(result["spearman_rho"] < 0.0, "negative", "zero"),
    )
    return result


def build_paired_condition_contrasts(joined: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    source_mae = {
        scope: scope_frame(joined, scope).groupby("source_id", sort=True)["abs_residual"].mean()
        for scope in {item for triple in PAIRED_CONDITION_CONTRASTS for item in triple[1:]}
    }
    rows: list[dict[str, Any]] = []
    for name, scope_a, scope_b in PAIRED_CONDITION_CONTRASTS:
        left = source_mae[scope_a]
        right = source_mae[scope_b]
        common = left.index.intersection(right.index)
        differences = (left.loc[common] - right.loc[common]).dropna()
        ci_low, ci_high = paired_bootstrap_ci(
            differences.to_numpy(dtype=np.float64),
            n_bootstrap=config.run.n_bootstrap,
            seed=config.run.seed,
            confidence_level=config.run.confidence_level,
        )
        rows.append(
            {
                "contrast": name,
                "scope_a": scope_a,
                "scope_b": scope_b,
                "n_sources": int(len(differences)),
                "aggregate_mae_a": float(left.loc[common].mean()),
                "aggregate_mae_b": float(right.loc[common].mean()),
                "aggregate_mae_difference": float(
                    left.loc[common].mean() - right.loc[common].mean()
                ),
                "mean_source_difference": float(differences.mean()),
                "median_source_difference": float(differences.median()),
                "difference_ci_low": ci_low,
                "difference_ci_high": ci_high,
                "proportion_a_worse": float((differences > 0.0).mean()),
                "n_bootstrap": config.run.n_bootstrap,
                "bootstrap_seed": config.run.seed,
                "bootstrap_method": config.analysis.bootstrap_method,
            }
        )
    return pd.DataFrame(rows)


def build_error_concentration(joined: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    rows = []
    for scope in SUMMARY_SCOPES:
        metrics = _concentration_metrics(scope_frame(joined, scope))
        rows.append({"scope": scope, "description": SCOPE_DESCRIPTIONS[scope], **metrics})
    return pd.DataFrame(rows)


def build_analysis_result(inputs: AnalysisInputs, config: AnalysisConfig) -> AnalysisResult:
    """Build every predeclared table from the frozen inputs."""
    joined = inputs.joined
    checks = list(inputs.checks)
    summary = build_summary_metrics(joined, config)
    calibration = build_condition_calibration(joined, config)
    font = build_font_condition_metrics(joined, config)
    severity = build_severity_metrics(joined, config)
    source = build_source_group_metrics(joined, config)
    case = build_case_metrics(joined, config)
    characteristics = build_source_characteristic_associations(
        joined, config, source_metrics=source
    )
    features = build_feature_error_associations(joined, config)
    paired = build_paired_condition_contrasts(joined, config)
    concentration = build_error_concentration(joined, config)

    _check(
        checks,
        "summary_scope_membership",
        "analysis",
        f"{len(SUMMARY_SCOPES)} predeclared scopes",
        f"{len(summary)} rows",
        len(summary) == len(SUMMARY_SCOPES),
    )
    _check(
        checks,
        "font_condition_membership",
        "analysis",
        f"{len(FONT_SCOPES)} scopes x 10 fonts",
        f"{len(font)} rows",
        len(font) == len(FONT_SCOPES) * 10,
    )
    _check(
        checks,
        "severity_membership",
        "analysis",
        "19 primary family/severity groups (elastic 0.5 px is diagnostic)",
        f"{len(severity)} rows",
        len(severity) == 19,
    )
    _check(
        checks,
        "source_group_membership",
        "analysis",
        f"{len(SOURCE_SCOPES)} scopes x 300 sources",
        f"{len(source)} rows",
        len(source) == len(SOURCE_SCOPES) * 300,
    )
    _check(
        checks,
        "case_membership",
        "analysis",
        "30 text cases",
        f"{len(case)} rows",
        len(case) == 30,
    )
    _check(
        checks,
        "condition_calibration_membership",
        "analysis",
        f"{len(SUMMARY_SCOPES)} calibration scopes",
        f"{len(calibration)} rows",
        len(calibration) == len(SUMMARY_SCOPES),
    )
    _check(
        checks,
        "source_characteristic_membership",
        "analysis",
        "2 characteristics x (overall + 10 fonts)",
        f"{len(characteristics)} rows",
        len(characteristics) == 2 * 11,
    )
    _check(
        checks,
        "error_concentration_membership",
        "analysis",
        f"{len(SUMMARY_SCOPES)} concentration scopes",
        f"{len(concentration)} rows",
        len(concentration) == len(SUMMARY_SCOPES),
    )
    _check(
        checks,
        "feature_association_membership",
        "analysis",
        f"{len(FEATURE_SCOPES)} scopes x 8 features x 2 outcomes",
        f"{len(features)} rows",
        len(features) == len(FEATURE_SCOPES) * 8 * 2,
    )
    _check(
        checks,
        "paired_contrast_membership",
        "analysis",
        f"{len(PAIRED_CONDITION_CONTRASTS)} paired contrasts",
        f"{len(paired)} rows",
        len(paired) == len(PAIRED_CONDITION_CONTRASTS),
    )
    _check(
        checks,
        "clean_r2_suppressed",
        "metrics",
        "constant clean-reference target suppresses R²",
        str(summary.loc[summary["scope"].eq("clean_reference"), "r2_status"].iloc[0]),
        summary.loc[summary["scope"].eq("clean_reference"), "r2_status"].iloc[0] != "reported",
    )
    return AnalysisResult(
        summary_metrics=summary,
        condition_calibration=calibration,
        font_condition_metrics=font,
        severity_metrics=severity,
        source_group_metrics=source,
        case_metrics=case,
        source_characteristic_associations=characteristics,
        feature_error_associations=features,
        paired_condition_contrasts=paired,
        error_concentration=concentration,
        checks=tuple(checks),
        external_verification=inputs.external_verification,
    )


# ---------------------------------------------------------------------------
# dependency-free deterministic SVG figures
# ---------------------------------------------------------------------------

SVG_FONT: Final[str] = "Helvetica, Arial, sans-serif"
COLOR_TEXT: Final[str] = "#1f2937"
COLOR_MUTED: Final[str] = "#6b7280"
COLOR_GRID: Final[str] = "#d1d5db"
COLOR_INTERFERENCE: Final[str] = "#b91c1c"
COLOR_THIN: Final[str] = "#dc2626"
COLOR_ARC: Final[str] = "#1d4ed8"
COLOR_BLUR: Final[str] = "#2563eb"
COLOR_NOISE: Final[str] = "#0f766e"
COLOR_GEOMETRIC: Final[str] = "#15803d"
COLOR_ALL: Final[str] = "#4b5563"


def _svg_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: float = 12,
    anchor: str = "start",
    fill: str = COLOR_TEXT,
    weight: str = "normal",
    rotate: float | None = None,
) -> str:
    transform = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="{SVG_FONT}" font-size="{size:.2f}" '
        f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}"{transform}>'
        f"{_svg_escape(text)}</text>"
    )


def _svg_rect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str,
    stroke: str = "none",
    stroke_width: float = 1.0,
    rx: float = 0.0,
) -> str:
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
        f'rx="{rx:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.2f}"/>'
    )


def _svg_line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str,
    width: float = 1.0,
    dash: str | None = None,
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{stroke}" stroke-width="{width:.2f}"{dash_attr}/>'
    )


def _svg_circle(x: float, y: float, radius: float, *, fill: str, stroke: str = "none") -> str:
    return (
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1"/>'
    )


def _svg_polyline(
    points: Sequence[tuple[float, float]],
    *,
    stroke: str,
    width: float = 2.0,
    dash: str | None = None,
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (
        f'<polyline points="{coordinates}" fill="none" stroke="{stroke}" '
        f'stroke-width="{width:.2f}"{dash_attr}/>'
    )


def _svg_document(width: int, height: int, title: str, description: str, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">\n'
        f'<title id="title">{_svg_escape(title)}</title>\n'
        f'<desc id="desc">{_svg_escape(description)}</desc>\n'
        f"{_svg_rect(0, 0, width, height, fill='#ffffff')}\n"
        f"{body}\n</svg>\n"
    )


def _fmt(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _axis_x(
    left: float,
    right: float,
    y: float,
    minimum: float,
    maximum: float,
    ticks: Sequence[float],
    *,
    label: str,
) -> str:
    parts = [
        _svg_line(left, y, right, y, stroke=COLOR_TEXT, width=1.2),
    ]
    for tick in ticks:
        x = left + (tick - minimum) / (maximum - minimum) * (right - left)
        parts.append(_svg_line(x, y, x, y + 5, stroke=COLOR_TEXT, width=1.0))
        parts.append(_svg_text(x, y + 20, f"{tick:g}", size=11, anchor="middle", fill=COLOR_MUTED))
    parts.append(_svg_text((left + right) / 2, y + 43, label, size=12, anchor="middle"))
    return "".join(parts)


def _figure_condition_profile(result: AnalysisResult) -> str:
    scopes = [
        "gaussian_blur",
        "gaussian_noise",
        "interference",
        "thin_lines",
        "arc_strokes",
        "geometric_primary",
        "all_primary",
    ]
    labels = {
        "gaussian_blur": "Gaussian blur",
        "gaussian_noise": "Gaussian noise",
        "interference": "Interference",
        "thin_lines": "Thin lines",
        "arc_strokes": "Arc strokes",
        "geometric_primary": "Geometric (primary)",
        "all_primary": "All primary",
    }
    colors = {
        "gaussian_blur": COLOR_BLUR,
        "gaussian_noise": COLOR_NOISE,
        "interference": COLOR_INTERFERENCE,
        "thin_lines": COLOR_THIN,
        "arc_strokes": COLOR_ARC,
        "geometric_primary": COLOR_GEOMETRIC,
        "all_primary": COLOR_ALL,
    }
    summary = result.summary_metrics.set_index("scope")
    width, height = 1040, 610
    left, right, top = 175, 790, 78
    row_height = 54
    maximum = float(summary.loc[scopes, "mae_ci_high"].max()) * 1.08
    parts: list[str] = [
        _svg_text(34, 34, "Prediction error by corruption family", size=20, weight="bold"),
        _svg_text(
            34,
            56,
            "MAE with a source-grouped 95% percentile interval; bias = target - prediction.",
            size=12,
            fill=COLOR_MUTED,
        ),
    ]
    for index, scope in enumerate(scopes):
        y = top + index * row_height
        row = summary.loc[scope]
        parts.append(_svg_text(left - 14, y + 18, labels[scope], size=13, anchor="end"))
        bar_length = (float(row["mae"]) / maximum) * (right - left)
        parts.append(_svg_rect(left, y, max(1.0, bar_length), 26, fill=colors[scope], rx=3))
        low_x = left + (float(row["mae_ci_low"]) / maximum) * (right - left)
        high_x = left + (float(row["mae_ci_high"]) / maximum) * (right - left)
        parts.append(_svg_line(low_x, y + 13, high_x, y + 13, stroke="#111827", width=1.3))
        parts.append(_svg_line(low_x, y + 7, low_x, y + 19, stroke="#111827", width=1.3))
        parts.append(_svg_line(high_x, y + 7, high_x, y + 19, stroke="#111827", width=1.3))
        parts.append(
            _svg_text(
                right + 14,
                y + 18,
                f"MAE {_fmt(float(row['mae']))}   bias {_fmt(float(row['bias']), 3)}",
                size=12,
            )
        )
    parts.append(
        _axis_x(
            left,
            right,
            top + len(scopes) * row_height + 8,
            0.0,
            maximum,
            [0.0, 0.02, 0.04, 0.06, 0.08, 0.10],
            label="MAE (SSIM units)",
        )
    )
    parts.append(
        _svg_text(
            34,
            height - 24,
            "Positive bias means underprediction; negative bias means overprediction. "
            "Intervals resample whole source groups, not images.",
            size=11,
            fill=COLOR_MUTED,
        )
    )
    return _svg_document(
        width,
        height,
        "Prediction error by corruption family",
        "Horizontal bars show MAE for blur, noise, interference and geometric conditions. "
        "Interference has the largest MAE and a negative bias. Intervals are source-grouped "
        "percentile bootstrap intervals.",
        "".join(parts),
    )


def _figure_severity_calibration(result: AnalysisResult) -> str:
    severity = result.severity_metrics
    width, height = 980, 570
    left, right, top, bottom = 90, 930, 80, 470
    y_min, y_max = 0.64, 1.0
    parts: list[str] = [
        _svg_text(34, 34, "Interference severity: target vs prediction", size=20, weight="bold"),
        _svg_text(
            34,
            56,
            "Mean SSIM at each predeclared severity rung; one row per source group per rung.",
            size=12,
            fill=COLOR_MUTED,
        ),
    ]

    def x_position(rung: int) -> float:
        return left + (rung - 1) / 3 * (right - left)

    def y_position(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    for tick in (0.7, 0.8, 0.9, 1.0):
        y = y_position(tick)
        parts.append(_svg_line(left, y, right, y, stroke=COLOR_GRID, width=0.8, dash="4 4"))
        parts.append(
            _svg_text(left - 12, y + 4, f"{tick:.1f}", size=11, anchor="end", fill=COLOR_MUTED)
        )
    for rung in (1, 2, 3, 4):
        x = x_position(rung)
        parts.append(_svg_line(x, top, x, bottom, stroke=COLOR_GRID, width=0.8, dash="4 4"))
        parts.append(_svg_text(x, bottom + 24, f"s{rung}", size=12, anchor="middle"))
    parts.append(
        _svg_text((left + right) / 2, height - 26, "Severity level", size=13, anchor="middle")
    )
    parts.append(
        _svg_text(26, (top + bottom) / 2, "Mean SSIM", size=13, anchor="middle", rotate=-90)
    )

    for family, color, label in (
        ("thin_lines", COLOR_THIN, "Thin lines"),
        ("arc_strokes", COLOR_ARC, "Arc strokes"),
    ):
        subset = severity.loc[severity["family"].eq(family)].sort_values("severity_level")
        target_points = [
            (x_position(int(row.severity_level)), y_position(float(row.target_mean)))
            for row in subset.itertuples()
        ]
        prediction_points = [
            (x_position(int(row.severity_level)), y_position(float(row.prediction_mean)))
            for row in subset.itertuples()
        ]
        parts.append(_svg_polyline(target_points, stroke=color, width=2.6))
        parts.append(_svg_polyline(prediction_points, stroke=color, width=2.6, dash="7 5"))
        for points in (target_points, prediction_points):
            for x, y in points:
                parts.append(_svg_circle(x, y, 4.2, fill=color, stroke="#ffffff"))
        parts.append(
            _svg_text(
                right - 130,
                top + 10 if family == "thin_lines" else top + 30,
                label,
                size=13,
                fill=color,
                weight="bold",
            )
        )
    parts.append(_svg_text(right - 130, top + 54, "solid = target", size=11, fill=COLOR_MUTED))
    parts.append(_svg_text(right - 130, top + 70, "dashed = prediction", size=11, fill=COLOR_MUTED))
    parts.append(
        _svg_text(
            34,
            height - 58,
            "At s4 the mean prediction is near 0.90 while the mean thin-line target is 0.70 and the "
            "mean arc target is 0.74. The gap is a descriptive calibration failure, not a causal test.",
            size=11,
            fill=COLOR_MUTED,
        )
    )
    return _svg_document(
        width,
        height,
        "Interference severity target and prediction",
        "Line chart of mean target and mean prediction across thin-line and arc-stroke severities. "
        "Predictions remain near 0.90 while targets fall at severe rungs.",
        "".join(parts),
    )


def _figure_font_interference(result: AnalysisResult) -> str:
    table = result.font_condition_metrics.loc[
        result.font_condition_metrics["scope"].eq("interference")
    ].sort_values("mae")
    width, height = 1000, 590
    left, right, top = 190, 760, 82
    row_height = 42
    maximum = float(table["mae_ci_high"].max()) * 1.10
    parts: list[str] = [
        _svg_text(34, 34, "Interference MAE by font", size=20, weight="bold"),
        _svg_text(
            34,
            56,
            "One point per font with a within-font source-grouped 95% interval (30 sources).",
            size=12,
            fill=COLOR_MUTED,
        ),
    ]
    for index, row in enumerate(table.itertuples()):
        y = top + index * row_height
        color = (
            COLOR_INTERFERENCE
            if row.font_id == "courier_new"
            else (COLOR_GEOMETRIC if row.font_id == "impact" else COLOR_ALL)
        )
        parts.append(_svg_text(left - 16, y + 5, row.font_id, size=13, anchor="end"))
        x = left + (float(row.mae) / maximum) * (right - left)
        low = left + (float(row.mae_ci_low) / maximum) * (right - left)
        high = left + (float(row.mae_ci_high) / maximum) * (right - left)
        parts.append(_svg_line(low, y, high, y, stroke=color, width=2.2))
        parts.append(_svg_line(low, y - 5, low, y + 5, stroke=color, width=1.2))
        parts.append(_svg_line(high, y - 5, high, y + 5, stroke=color, width=1.2))
        parts.append(_svg_circle(x, y, 4.6, fill=color, stroke="#ffffff"))
        parts.append(
            _svg_text(
                right + 12,
                y + 4,
                f"{_fmt(float(row.mae))}  (large {100.0 * float(row.large_error_proportion):.1f}%)",
                size=11,
            )
        )
    parts.append(
        _axis_x(
            left,
            right,
            top + len(table) * row_height + 2,
            0.0,
            maximum,
            [0.0, 0.03, 0.06, 0.09, 0.12, 0.15],
            label="Interference MAE (SSIM units)",
        )
    )
    parts.append(
        _svg_text(
            34,
            height - 24,
            "Font identity is entangled with glyph shape and renderer. The contrast is descriptive, "
            "not a causal font effect.",
            size=11,
            fill=COLOR_MUTED,
        )
    )
    return _svg_document(
        width,
        height,
        "Interference MAE by font",
        "Dot plot of interference MAE for ten fonts. Courier New is highest and Impact is lowest. "
        "Intervals resample sources within each font.",
        "".join(parts),
    )


def _diverging_color(value: float, limit: float = 1.0) -> tuple[str, str]:
    clipped = max(-limit, min(limit, value))
    if clipped >= 0.0:
        fraction = clipped / limit
        start = np.array([255, 255, 255], dtype=float)
        end = np.array([153, 27, 27], dtype=float)
    else:
        fraction = -clipped / limit
        start = np.array([255, 255, 255], dtype=float)
        end = np.array([29, 78, 216], dtype=float)
    color = start + (end - start) * fraction
    red, green, blue = (int(round(channel)) for channel in color)
    return f"#{red:02x}{green:02x}{blue:02x}", ("#ffffff" if abs(clipped) > 0.55 else COLOR_TEXT)


def _figure_feature_heatmap(result: AnalysisResult) -> str:
    scopes = [
        "thin_lines",
        "arc_strokes",
        "thin_lines_s1",
        "thin_lines_s2",
        "thin_lines_s3",
        "thin_lines_s4",
        "arc_strokes_s1",
        "arc_strokes_s2",
        "arc_strokes_s3",
        "arc_strokes_s4",
    ]
    table = result.feature_error_associations.loc[
        result.feature_error_associations["target_metric"].eq("abs_residual")
    ]
    pivot = table.pivot(index="feature", columns="scope", values="spearman_rho").reindex(
        index=FEATURE_NAMES, columns=scopes
    )
    width, height = 1270, 560
    left, top = 235, 105
    cell_width, cell_height = 92, 42
    parts: list[str] = [
        _svg_text(34, 34, "Feature rank association with absolute error", size=20, weight="bold"),
        _svg_text(
            34,
            56,
            "Spearman rho within condition/severity strata; exploratory and not source-clustered inferential tests.",
            size=12,
            fill=COLOR_MUTED,
        ),
    ]
    for column_index, scope in enumerate(scopes):
        x = left + column_index * cell_width + cell_width / 2
        label = (
            scope.replace("thin_lines", "thin").replace("arc_strokes", "arc").replace("_s", " s")
        )
        parts.append(_svg_text(x, top - 12, label, size=10, anchor="middle", fill=COLOR_MUTED))
    for row_index, feature in enumerate(FEATURE_NAMES):
        y = top + row_index * cell_height
        parts.append(_svg_text(left - 14, y + 26, feature, size=11, anchor="end"))
        for column_index, scope in enumerate(scopes):
            value = float(pivot.loc[feature, scope])
            fill, text_color = _diverging_color(value)
            x = left + column_index * cell_width
            parts.append(_svg_rect(x + 1, y + 1, cell_width - 2, cell_height - 2, fill=fill))
            parts.append(
                _svg_text(
                    x + cell_width / 2,
                    y + 26,
                    f"{value:+.2f}",
                    size=10,
                    anchor="middle",
                    fill=text_color,
                )
            )
    legend_x, legend_y = left, top + len(FEATURE_NAMES) * cell_height + 30
    for index, value in enumerate((-1.0, -0.5, 0.0, 0.5, 1.0)):
        fill, _ = _diverging_color(value)
        parts.append(
            _svg_rect(legend_x + index * 46, legend_y, 44, 16, fill=fill, stroke=COLOR_GRID)
        )
        parts.append(
            _svg_text(
                legend_x + index * 46 + 22,
                legend_y + 32,
                f"{value:+.1f}",
                size=10,
                anchor="middle",
                fill=COLOR_MUTED,
            )
        )
    parts.append(_svg_text(legend_x, legend_y - 10, "Spearman rho", size=11, fill=COLOR_MUTED))
    parts.append(
        _svg_text(
            34,
            height - 24,
            "Larger positive values mean higher feature values accompany larger absolute error. "
            "Severity and source composition remain confounded; no feature is selected.",
            size=11,
            fill=COLOR_MUTED,
        )
    )
    return _svg_document(
        width,
        height,
        "Feature rank association with absolute error",
        "Heatmap of Spearman correlations between the eight existing features and absolute error "
        "for thin-line and arc-stroke scopes, including severity strata.",
        "".join(parts),
    )


def _figure_concentration(result: AnalysisResult) -> str:
    scopes = [
        ("interference", COLOR_INTERFERENCE, "Interference"),
        ("gaussian_blur", COLOR_BLUR, "Gaussian blur"),
        ("gaussian_noise", COLOR_NOISE, "Gaussian noise"),
        ("geometric_primary", COLOR_GEOMETRIC, "Geometric"),
    ]
    width, height = 920, 580
    left, right, top, bottom = 90, 820, 80, 470
    parts: list[str] = [
        _svg_text(34, 34, "Source-group concentration of squared error", size=20, weight="bold"),
        _svg_text(
            34,
            56,
            "Sources are ranked from largest to smallest squared error; whole source groups.",
            size=12,
            fill=COLOR_MUTED,
        ),
    ]
    for tick in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        x = left + tick * (right - left)
        y = bottom - tick * (bottom - top)
        parts.append(_svg_line(left, y, right, y, stroke=COLOR_GRID, width=0.8, dash="4 4"))
        parts.append(_svg_line(x, top, x, bottom, stroke=COLOR_GRID, width=0.8, dash="4 4"))
        parts.append(
            _svg_text(x, bottom + 22, f"{tick:.1f}", size=11, anchor="middle", fill=COLOR_MUTED)
        )
        parts.append(
            _svg_text(left - 14, y + 4, f"{tick:.1f}", size=11, anchor="end", fill=COLOR_MUTED)
        )
    parts.append(
        _svg_text(
            (left + right) / 2,
            height - 30,
            "Cumulative share of source groups",
            size=13,
            anchor="middle",
        )
    )
    parts.append(
        _svg_text(
            26,
            (top + bottom) / 2,
            "Cumulative share of squared error",
            size=13,
            anchor="middle",
            rotate=-90,
        )
    )
    for scope, color, label in scopes:
        subset = result.source_group_metrics.loc[result.source_group_metrics["scope"].eq(scope)]
        sse = np.sort(subset["sse"].to_numpy(dtype=np.float64))[::-1]
        cumulative = np.cumsum(sse) / np.sum(sse)
        x_values = np.arange(1, len(sse) + 1) / len(sse)
        points = [
            (left + xv * (right - left), bottom - yv * (bottom - top))
            for xv, yv in zip(x_values, cumulative, strict=True)
        ]
        parts.append(_svg_polyline(points, stroke=color, width=2.2))
        index = min(len(cumulative) - 1, max(0, int(0.2 * len(cumulative)) - 1))
        label_y = bottom - float(cumulative[index]) * (bottom - top)
        parts.append(
            _svg_text(left + 0.21 * (right - left), label_y - 8, label, size=11, fill=color)
        )
    parts.append(
        _svg_text(
            34,
            height - 58,
            "Blur and noise errors are concentrated in a few sources; interference error is broad. "
            "This is descriptive, not a hypothesis test.",
            size=11,
            fill=COLOR_MUTED,
        )
    )
    return _svg_document(
        width,
        height,
        "Source-group concentration of squared error",
        "Lorenz-style curves for interference, blur, noise and geometric conditions. The interference "
        "curve stays close to the diagonal, showing broad rather than source-outlier error.",
        "".join(parts),
    )


def build_figures(result: AnalysisResult, output_directory: Path) -> dict[str, str]:
    figures_directory = output_directory / "figures"
    figures_directory.mkdir(parents=True, exist_ok=True)
    content = {
        "fig1_condition_error_profile.svg": _figure_condition_profile(result),
        "fig2_interference_severity_calibration.svg": _figure_severity_calibration(result),
        "fig3_font_interference_mae.svg": _figure_font_interference(result),
        "fig4_feature_error_associations.svg": _figure_feature_heatmap(result),
        "fig5_source_error_concentration.svg": _figure_concentration(result),
    }
    for name, text in content.items():
        (figures_directory / name).write_text(text, encoding="utf-8")
    return {f"figures/{name}": sha256_text(text) for name, text in content.items()}


# ---------------------------------------------------------------------------
# report text, protocol and fail-closed package writing
# ---------------------------------------------------------------------------


def _row(frame: pd.DataFrame, **filters: object) -> pd.Series:
    subset = frame
    for column, value in filters.items():
        subset = subset.loc[subset[column].eq(value)]
    if len(subset) != 1:
        raise AnalysisError(f"expected exactly one row for {filters}; observed {len(subset)}")
    return subset.iloc[0]


def _ci(row: pd.Series, point: str, low: str, high: str, digits: int = 4) -> str:
    return (
        f"{_fmt(float(row[point]), digits)} "
        f"[{_fmt(float(row[low]), digits)}, {_fmt(float(row[high]), digits)}]"
    )


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _top_feature_rows(result: AnalysisResult, count: int = 10) -> pd.DataFrame:
    table = result.feature_error_associations.loc[
        result.feature_error_associations["target_metric"].eq("abs_residual")
    ].copy()
    table["abs_rho"] = table["spearman_rho"].abs()
    return table.sort_values(["abs_rho", "scope", "feature"], ascending=[False, True, True]).head(
        count
    )


def build_readme(inputs: AnalysisInputs, result: AnalysisResult, config: AnalysisConfig) -> str:
    summary = result.summary_metrics.set_index("scope")
    interference = summary.loc["interference"]
    blur = summary.loc["gaussian_blur"]
    noise = summary.loc["gaussian_noise"]
    geometric = summary.loc["geometric_primary"]
    thin = summary.loc["thin_lines"]
    arc = summary.loc["arc_strokes"]
    thin_s4 = _row(result.severity_metrics, family="thin_lines", severity_level=4)
    arc_s4 = _row(result.severity_metrics, family="arc_strokes", severity_level=4)
    fonts = result.font_condition_metrics.loc[
        result.font_condition_metrics["scope"].eq("interference")
    ].sort_values("mae")
    courier = _row(fonts, font_id="courier_new")
    impact = _row(fonts, font_id="impact")
    concentration = result.error_concentration.set_index("scope")
    check_passed = sum(1 for check in result.checks if check.passed)
    check_total = len(result.checks)
    top_features = _top_feature_rows(result, 8)

    summary_rows = []
    for scope in SUMMARY_SCOPES:
        row = summary.loc[scope]
        r2 = "n/a (suppressed)" if not bool(row["r2_defined"]) else _fmt(float(row["r2"]), 4)
        summary_rows.append(
            [
                scope,
                f"{int(row['n_rows'])} / {int(row['n_sources'])}",
                _ci(row, "mae", "mae_ci_low", "mae_ci_high"),
                _fmt(float(row["rmse"])),
                _fmt(float(row["bias"])),
                r2,
                f"{100.0 * float(row['large_error_proportion']):.2f}%",
            ]
        )
    font_rows = []
    for row in fonts.itertuples():
        font_rows.append(
            [
                row.font_id,
                _ci(pd.Series(row._asdict()), "mae", "mae_ci_low", "mae_ci_high"),
                _fmt(float(row.bias)),
                _fmt(float(row.r2)),
                f"{100.0 * float(row.large_error_proportion):.1f}%",
                _ci(
                    pd.Series(row._asdict()),
                    "paired_difference_mean",
                    "paired_difference_ci_low",
                    "paired_difference_ci_high",
                ),
                f"{int(row.paired_cases_worse)}/30",
            ]
        )
    severity_rows = []
    for family in ("thin_lines", "arc_strokes"):
        for severity in (1, 2, 3, 4):
            row = _row(result.severity_metrics, family=family, severity_level=severity)
            severity_rows.append(
                [
                    family,
                    severity,
                    _fmt(float(row["target_mean"])),
                    _fmt(float(row["prediction_mean"])),
                    _fmt(float(row["mae"])),
                    _fmt(float(row["bias"])),
                    f"{100.0 * float(row['large_error_proportion']):.1f}%",
                ]
            )
    concentration_rows = []
    for scope in (
        "gaussian_blur",
        "gaussian_noise",
        "interference",
        "thin_lines",
        "arc_strokes",
        "geometric_primary",
    ):
        row = concentration.loc[scope]
        concentration_rows.append(
            [
                scope,
                f"{100.0 * float(row['top_5pct_source_sse_share']):.1f}%",
                f"{100.0 * float(row['top_10pct_source_sse_share']):.1f}%",
                _fmt(float(row["source_mae_q50"])),
                _fmt(float(row["source_mae_q90"])),
            ]
        )
    feature_rows = [
        [
            row.scope,
            row.feature,
            f"{float(row.spearman_rho):+.3f}",
            f"{float(row.bh_q_value):.2e}" if math.isfinite(float(row.bh_q_value)) else "n/a",
            int(row.n_rows),
        ]
        for row in top_features.itertuples()
    ]
    input_rows = [
        [
            record["key"],
            f"`{record['path']}`",
            record["observed_sha256"][:16] + "…",
            record["expected_rows"] if record["expected_rows"] else "—",
        ]
        for record in inputs.records
    ]

    return f"""# Interference error analysis v0.1.0

## Purpose and scope

This is a read-only analysis of the frozen M8C cross-font benchmark. It asks
where prediction error is concentrated under thin-line and arc-stroke
interference, whether particular fonts or source groups drive the failure,
which existing features are associated with the error, and whether the pattern
is specific to interference.

The target is `ssim_target`. PSNR is diagnostic only and is not analysed. The
independent unit is the **source group**: one pinned font rendered with one
committed text case. Whole source groups are resampled; individual corrupted
images are never treated as independent.

This analysis did **not** train, refit, tune, calibrate, select or replace a
model; did not add or select a feature; did not generate a dataset; and did not
write into any M8C, M8D or M8E artifact. It does not measure human readability,
CAPTCHA solvability, security or production robustness.

## Frozen inputs and provenance

{_md_table(["Input", "Path", "SHA-256 prefix", "Rows"], input_rows)}

The M5C champion artifact is hash-checked but never loaded. Its recorded
`inference.model_fit_performed`, `calibration_performed` and
`feature_change_performed` flags are all false, and its eight feature names are
exactly the frozen M8C feature set. The full input hashes, commit, working-tree
state, software versions, bootstrap seed and verifier outcomes are recorded in
`analysis_protocol.json`.

The local `ground_truth.csv` is integrity-pinned but only its `source_id`,
`case_id` and `font_id` columns are read. The text column is never loaded or
written to the report.

## Main findings

### 1. Interference is worse than the other primary conditions, with opposite bias

{_md_table(["Scope", "Rows / sources", "MAE (95% CI)", "RMSE", "Bias", "R²", "Large error ≥ 0.20"], summary_rows)}

Interference MAE is {_fmt(float(interference["mae"]))} versus
{_fmt(float(blur["mae"]))} for blur, {_fmt(float(noise["mae"]))} for noise and
{_fmt(float(geometric["mae"]))} for primary geometric distortion. The
interference bias is {_fmt(float(interference["bias"]))}: predictions are too
high (overprediction) on average. Blur and noise have positive bias, meaning
underprediction. The negative interference R²
({_fmt(float(interference["r2"]))}) means the model is worse than a constant
condition-mean predictor in squared-error terms on this slice.

### 2. Severe interference drives the large errors

At the predeclared exploratory threshold `|residual| >= 0.20`, thin lines have
{100.0 * float(thin["large_error_proportion"]):.2f}% large errors and arcs
{100.0 * float(arc["large_error_proportion"]):.2f}%. The failure is concentrated
at the highest rung: thin s4 has a large-error proportion of
{100.0 * float(thin_s4["large_error_proportion"]):.1f}% and arc s4
{100.0 * float(arc_s4["large_error_proportion"]):.1f}%. At thin s4 the mean
target is {_fmt(float(thin_s4["target_mean"]))} while the mean prediction is
{_fmt(float(thin_s4["prediction_mean"]))}; at arc s4 the corresponding values are
{_fmt(float(arc_s4["target_mean"]))} and {_fmt(float(arc_s4["prediction_mean"]))}.

{_md_table(["Family", "Severity", "Mean target", "Mean prediction", "MAE", "Bias", "Large error"], severity_rows)}

### 3. The model responds monotonically but with compressed amplitude

Within each source group, thin-line predictions have a median prediction/target
SD ratio of {_fmt(float(thin["within_source_prediction_target_sd_ratio_median"]))}
and a median slope of
{_fmt(float(thin["within_source_calibration_slope_median"]))}; arc values are
{_fmt(float(arc["within_source_prediction_target_sd_ratio_median"]))} and
{_fmt(float(arc["within_source_calibration_slope_median"]))}. Blur and noise are
near 1.0. The within-source rank correlations are nevertheless high
({_fmt(float(thin["within_source_prediction_target_corr_median"]))} for thin and
{_fmt(float(arc["within_source_prediction_target_corr_median"]))} for arc), and
residual-target Spearman correlations are
{_fmt(float(thin["residual_target_spearman"]))} and
{_fmt(float(arc["residual_target_spearman"]))}. The pattern is attenuation or
range compression, not a completely condition-insensitive constant predictor.
The attenuation is descriptive; its cause is not identified here.

### 4. Font differences are large but entangled with glyph and target difficulty

{_md_table(["Font", "MAE (95% CI)", "Bias", "R²", "Large error", "Paired diff vs other fonts (95% CI)", "Cases worse"], font_rows)}

`courier_new` has the largest interference MAE
({_fmt(float(courier["mae"]))}); `impact` has the smallest
({_fmt(float(impact["mae"]))}). The paired-by-case contrast is
{_fmt(float(courier["paired_difference_mean"]))} for Courier New
(95% CI [{_fmt(float(courier["paired_difference_ci_low"]))},
{_fmt(float(courier["paired_difference_ci_high"]))}]; worse on
{int(courier["paired_cases_worse"])}/30 cases) and
{_fmt(float(impact["paired_difference_mean"]))} for Impact
(95% CI [{_fmt(float(impact["paired_difference_ci_low"]))},
{_fmt(float(impact["paired_difference_ci_high"]))}]; worse on
{int(impact["paired_cases_worse"])}/30). However, the target itself is less
damaged for Impact: its interference target mean is
{_fmt(float(impact["target_mean"]))} versus
{_fmt(float(courier["target_mean"]))} for Courier New, and its s1-to-s4 target
drop is {_fmt(float(impact["target_drop_s1_to_s4"]))} versus
{_fmt(float(courier["target_drop_s1_to_s4"]))}. Font identity, glyph shape,
renderer and target degradation are entangled; this is not a causal font effect.

### 5. Interference error is broad across source groups, unlike blur/noise

{_md_table(["Scope", "Top 5% source SSE share", "Top 10% source SSE share", "Source MAE median", "Source MAE q90"], concentration_rows)}

The top 5% of interference sources account for
{100.0 * float(concentration.loc["interference", "top_5pct_source_sse_share"]):.1f}%
of interference squared error, compared with
{100.0 * float(concentration.loc["gaussian_blur", "top_5pct_source_sse_share"]):.1f}%
for blur and
{100.0 * float(concentration.loc["gaussian_noise", "top_5pct_source_sse_share"]):.1f}%
for noise. Interference is worse than blur for
{100.0 * float(_row(result.paired_condition_contrasts, contrast="interference_minus_gaussian_blur")["proportion_a_worse"]):.1f}%
of sources and worse than geometric distortion for
{100.0 * float(_row(result.paired_condition_contrasts, contrast="interference_minus_geometric_primary")["proportion_a_worse"]):.1f}%.
This is a broad, systematic error, not a small number of outlying sources.

### 6. Existing feature associations are strong but exploratory

{_md_table(["Scope", "Feature", "Spearman rho vs abs residual", "BH q", "Rows"], feature_rows)}

Within interference and within severity strata, `laplacian_variance`,
`edge_density` and `noise_sigma_immerkaer` tend to be positively associated with
absolute error, while `histogram_entropy`, contrast and ink fraction tend to be
negatively associated. These are descriptive rank associations from the
existing eight-feature table. They are not source-clustered inferential tests,
they do not establish that the model uses a feature in a particular way, and
they are not a feature-selection result.

## Figures

- `figures/fig1_condition_error_profile.svg` — MAE and bias by primary condition.
- `figures/fig2_interference_severity_calibration.svg` — mean target versus mean prediction across interference severities.
- `figures/fig3_font_interference_mae.svg` — per-font interference MAE with within-font source-grouped intervals.
- `figures/fig4_feature_error_associations.svg` — exploratory Spearman heatmap for the eight existing features.
- `figures/fig5_source_error_concentration.svg` — cumulative squared-error concentration by source group.

## Files

| File | Purpose |
| --- | --- |
| `summary_metrics.csv` | scope-level metrics, intervals, calibration and concentration |
| `condition_calibration.csv` | target/prediction calibration by condition |
| `font_condition_metrics.csv` | font x condition metrics and paired-by-case contrasts |
| `severity_metrics.csv` | primary family/severity metrics; elastic 0.5 px is diagnostic and excluded |
| `source_group_metrics.csv` | one row per source group and scope |
| `case_metrics.csv` | text-case interference summaries |
| `source_characteristic_associations.csv` | exploratory background/text-length associations |
| `feature_error_associations.csv` | exploratory feature rank associations |
| `paired_condition_contrasts.csv` | source-paired condition comparisons |
| `error_concentration.csv` | source-level squared-error concentration |
| `hypotheses_and_followups.md` | at most three testable hypotheses and future acceptance criteria |
| `verification_checks.csv` | fail-closed checks and actual outcomes |
| `analysis_protocol.json` | inputs, hashes, decisions, seed, software and output hashes |

## Reproduction

From the repository root, on the pinned environment:

```bash
.venv/bin/python -m captcha_quality.interference_error_analysis run \\
  --config configs/interference_error_analysis_v0_1_0.toml
```

The run refuses to overwrite an existing report unless `--overwrite` is given.
It refuses to write anywhere outside a new
`reports/interference_error_analysis_v*` directory. It runs the M8E integrity
verifier, the M8C post-execution lifecycle verifier and the full M8C replay
before writing the report.

## Verification

{check_passed}/{check_total} pre-package analysis checks passed.
`verification_checks.csv` records the complete package check set as well,
including output membership, immutability and the external historical
verifiers. The package records the exact Git revision, the clean/dirty state at
the start, the input hashes, the model artifact and card hashes, and the
external verifier outcomes. The external verifiers are also run again by the
final project-level checks.

## Limitations

- Synthetic image-similarity analysis only; SSIM is not human readability,
  solvability, security or production robustness.
- Ten purposively selected fonts and thirty committed text cases are not a
  random sample of fonts, users or deployments. Font identity is entangled with
  glyph shape, renderer and target degradation.
- Source-grouped intervals describe this source distribution; they do not
  estimate a font or user population.
- The 0.20 large-error threshold was fixed in the protocol before these subgroup
  tables were generated, but earlier reconnaissance had inspected aggregate and
  severity summaries. Every threshold-based subgroup proportion is therefore
  labelled exploratory.
- Feature associations are descriptive and exploratory. They are not corrected
  for source clustering, do not identify mechanism and do not justify selecting
  or adding a feature.
- The elastic 0.5 px rung is diagnostic and is excluded from the primary
  severity table. Clean-reference R² is suppressed because its target is
  constant.
- M3/M5C metrics are not treated as like-for-like comparisons to M8C. M8C is a
  new cross-font benchmark with a different source, font and condition
  composition.
"""


def build_hypotheses(result: AnalysisResult) -> str:
    summary = result.summary_metrics.set_index("scope")
    thin = summary.loc["thin_lines"]
    arc = summary.loc["arc_strokes"]
    blur = summary.loc["gaussian_blur"]
    noise = summary.loc["gaussian_noise"]
    interference = summary.loc["interference"]
    thin_s4 = _row(result.severity_metrics, family="thin_lines", severity_level=4)
    arc_s4 = _row(result.severity_metrics, family="arc_strokes", severity_level=4)
    concentration = result.error_concentration.set_index("scope")
    fonts = result.font_condition_metrics.loc[
        result.font_condition_metrics["scope"].eq("interference")
    ]
    courier = _row(fonts, font_id="courier_new")
    impact = _row(fonts, font_id="impact")
    times = _row(fonts, font_id="times_new_roman")
    features = result.feature_error_associations.loc[
        result.feature_error_associations["target_metric"].eq("abs_residual")
    ]
    laplacian_thin = _row(features, scope="thin_lines", feature="laplacian_variance")
    edge_thin = _row(features, scope="thin_lines", feature="edge_density")
    laplacian_arc = _row(features, scope="arc_strokes", feature="laplacian_variance")
    edge_arc = _row(features, scope="arc_strokes", feature="edge_density")
    entropy_thin = _row(features, scope="thin_lines", feature="histogram_entropy")
    entropy_arc = _row(features, scope="arc_strokes", feature="histogram_entropy")

    return f"""# Hypotheses and follow-ups

These are three testable explanations motivated by the descriptive M9A results.
They are not confirmed mechanisms. None was tested by fitting, calibrating or
selecting a model in M9A. Confidence labels reflect the strength and consistency
of the observed evidence, not intuition about the model.

## H1. Attenuated response to severe interference

**Claim.** The frozen model tracks the ordering of interference severity but
compresses the magnitude of its response, so severe thin-line and arc-stroke
damage is predicted as near-clean quality. This attenuation is the main source
of the interference MAE, negative bias and negative R².

**Evidence.**
- Interference MAE is {_fmt(float(interference["mae"]))} versus
  {_fmt(float(blur["mae"]))} for blur and {_fmt(float(noise["mae"]))} for noise;
  interference bias is {_fmt(float(interference["bias"]))}.
- At thin s4 the mean target is {_fmt(float(thin_s4["target_mean"]))} but the mean
  prediction is {_fmt(float(thin_s4["prediction_mean"]))}; at arc s4 the values
  are {_fmt(float(arc_s4["target_mean"]))} and
  {_fmt(float(arc_s4["prediction_mean"]))}.
- The within-source prediction/target SD ratio is
  {_fmt(float(thin["within_source_prediction_target_sd_ratio_median"]))} for thin
  lines and {_fmt(float(arc["within_source_prediction_target_sd_ratio_median"]))}
  for arcs, compared with
  {_fmt(float(blur["within_source_prediction_target_sd_ratio_median"]))} for blur
  and {_fmt(float(noise["within_source_prediction_target_sd_ratio_median"]))} for
  noise. The corresponding within-source slopes are
  {_fmt(float(thin["within_source_calibration_slope_median"]))} and
  {_fmt(float(arc["within_source_calibration_slope_median"]))}.
- Residual-target Spearman correlations are
  {_fmt(float(thin["residual_target_spearman"]))} and
  {_fmt(float(arc["residual_target_spearman"]))}: lower targets are
  overpredicted and higher targets are underpredicted.
- The top 5% of sources account for only
  {100.0 * float(concentration.loc["interference", "top_5pct_source_sse_share"]):.1f}%
  of interference squared error, so this is not carried by a few extreme
  sources.

**Evidence against / alternatives.** Within-source rank correlations are still
{_fmt(float(thin["within_source_prediction_target_corr_median"]))} for thin lines
and {_fmt(float(arc["within_source_prediction_target_corr_median"]))} for arcs, so
the model is not condition-blind. Mild s1-s3 rungs have few large errors; most
of the failure is at s4. The frozen M5C training description covers original
blur/noise and geometric groups, while prior M7B documentation describes thin
and arc interference as new out-of-distribution slices; training-distribution
shift is therefore an alternative explanation for the attenuation, not a proven
cause. Impact has a smaller target drop and a much smaller error, so a
target-difficulty explanation also contributes.

**Confidence.** High for the descriptive attenuation; moderate for the
out-of-distribution training explanation.

**Falsifying follow-up.** In a future, predeclared experiment, fit a monotone
calibration or a retrained model on source-disjoint *training* sources that
include the thin/arc ladder, then evaluate once on held-out sources. If
interference error is purely an attenuation problem, the intervention should
reduce interference MAE/RMSE and the s4 gap without damaging retention slices.
If it does not, attenuation alone is insufficient.

**Independent unit and split.** Source group. Split train/validation/test by
source group with no case or font overlap between training and test; hold out
at least two fonts and a disjoint set of text cases. Never resample images.

**Metric and predeclared acceptance criterion.** Primary metric: interference
RMSE on held-out source groups; secondary: s4 thin/arc MAE and large-error
proportion. Accept only if interference RMSE falls by at least 20% relative to
the frozen champion with a source-grouped 95% interval lower bound above zero,
while blur/noise RMSE increases by no more than 5% relative and clean-reference
MAE remains at or below 0.005.

## H2. Font/glyph-specific difficulty modifies the failure

**Claim.** Interference failure is not uniform across fonts. Thin-stroke or
otherwise difficult glyph sets (notably Courier New, Times New Roman and
Verdana) have larger interference error and larger target degradation, while
Impact is comparatively protected. Font identity is a modifier of the failure,
not an isolated cause.

**Evidence.**
- Courier New interference MAE is {_fmt(float(courier["mae"]))}, Times New Roman
  is {_fmt(float(times["mae"]))} and Impact is {_fmt(float(impact["mae"]))}.
- The paired-by-case contrast versus the other nine fonts is
  {_fmt(float(courier["paired_difference_mean"]))} for Courier New
  (95% CI [{_fmt(float(courier["paired_difference_ci_low"]))},
  {_fmt(float(courier["paired_difference_ci_high"]))}]; worse on
  {int(courier["paired_cases_worse"])}/30 cases) and
  {_fmt(float(impact["paired_difference_mean"]))} for Impact
  (95% CI [{_fmt(float(impact["paired_difference_ci_low"]))},
  {_fmt(float(impact["paired_difference_ci_high"]))}]; worse on
  {int(impact["paired_cases_worse"])}/30).
- The interference target is less damaged for Impact
  (mean {_fmt(float(impact["target_mean"]))};
  s1-to-s4 drop {_fmt(float(impact["target_drop_s1_to_s4"]))}) than for Courier
  New (mean {_fmt(float(courier["target_mean"]))};
  drop {_fmt(float(courier["target_drop_s1_to_s4"]))}).

**Evidence against / alternatives.** Font identity is entangled with glyph
shape, stroke width, renderer and the SSIM target itself. The per-font error
ordering partly follows target difficulty, so the evidence does not identify a
causal font property. All fonts still show negative interference bias (Impact
included), and the font table is descriptive. A font effect that disappears
after matching on target severity would not support a font-specific mechanism.

**Confidence.** Moderate.

**Falsifying follow-up.** Predeclare a held-out-font experiment with at least
two new thin-stroke and two new thick-stroke fonts, render the same cases and
corruptions, and compare interference error after matching on target severity
and text length. If the Courier-vs-Impact contrast disappears after matching,
the font-specific hypothesis is falsified.

**Independent unit and split.** Source group, with fonts held out entirely from
model fitting or calibration. Cases and fonts must be disjoint across
train/validation/test; do not reuse a source group across splits.

**Metric and predeclared acceptance criterion.** Metric: paired-by-case
interference MAE difference between thin-stroke and thick-stroke font groups,
matched on target severity. Accept only if the source-grouped 95% interval
excludes zero and the direction replicates in at least two independently held-
out fonts. Do not claim a font-population effect from ten purposive fonts.

## H3. Existing features conflate interference with benign edge/sharpness variation

**Claim.** The existing eight image features describe edge and sharpness
structure, but the two features most associated with interference error are
`laplacian_variance` and `edge_density`. The model may therefore represent
interference as a sharpness/edge phenomenon that overlaps with benign texture,
rather than as a distinct severe degradation.

**Evidence.**
- Within thin lines, `laplacian_variance` has Spearman rho
  {float(laplacian_thin["spearman_rho"]):+.3f} and `edge_density`
  {float(edge_thin["spearman_rho"]):+.3f} with absolute error; for arcs the
  corresponding values are {float(laplacian_arc["spearman_rho"]):+.3f} and
  {float(edge_arc["spearman_rho"]):+.3f} for the same edge family.
- `histogram_entropy` is negatively associated with absolute error
  ({float(entropy_thin["spearman_rho"]):+.3f} thin, {float(entropy_arc["spearman_rho"]):+.3f} arc),
  and the associations persist inside several severity strata.
- The strongest associations are edge/sharpness features
  (`laplacian_variance`, `edge_density`), which are plausible benign-texture
  cues. This is consistent with, but does not demonstrate, conflation.

**Evidence against / alternatives.** These are descriptive rank associations,
not a demonstration that the random forest uses a feature as a benign cue. The
features are also correlated with source content and severity, and the analysis
does not inspect tree splits, importances or counterfactual feature changes.
Within-source prediction-target rank correlations remain high, which is evidence
against a complete failure to represent interference. Feature associations may
be consequences of the target/source composition rather than causes of error.
No feature is recommended, added or selected by M9A.

**Confidence.** Low.

**Falsifying follow-up.** On held-out fonts and source groups, compare
interference and benign-condition rows matched on the eight existing features and
on target SSIM. If matched residuals show no systematic interference excess, the
conflation hypothesis is falsified. A second, independent test can ask whether
the feature-error associations survive within font x severity strata; if they
vanish, source composition explains them.

**Independent unit and split.** Source group; held-out fonts and disjoint cases
for any model-fitting or calibration step. Matching must be done without using
test outcomes to choose features or thresholds.

**Metric and predeclared acceptance criterion.** Metric: source-grouped mean
residual difference between matched interference and benign rows, and the
within-font/severity partial rank association of `laplacian_variance` and
`edge_density` with absolute error. Accept the hypothesis only if the matched
residual difference is at least 0.02 SSIM with a 95% interval excluding zero and
the feature association replicates in at least two held-out fonts. Otherwise
treat it as unsupported.

## What M9A does not establish

- It does not identify a causal mechanism for the interference error.
- It does not establish that any font is intrinsically worse for CAPTCHA
  readability, solvability or security.
- It does not justify adding a feature, changing the target, or replacing the
  frozen champion.
- It does not treat the M3 or M5C test metrics as like-for-like comparisons to
  M8C.
"""


def build_protocol(
    inputs: AnalysisInputs,
    result: AnalysisResult,
    config: AnalysisConfig,
    output_hashes: Mapping[str, str],
    *,
    checks: Sequence[Check] | None = None,
) -> dict[str, Any]:
    checked = tuple(checks) if checks is not None else result.checks
    clean_check = next(
        (check for check in checked if check.check_id == "working_tree_clean_at_start"), None
    )
    return {
        "schema": PROTOCOL_SCHEMA,
        "analysis": {
            "name": ANALYSIS_NAME,
            "version": ANALYSIS_VERSION,
            "package_version": package_version(),
            "command": (
                "python -m captcha_quality.interference_error_analysis run "
                "--config configs/interference_error_analysis_v0_1_0.toml"
            ),
            "config_path": str(config.config_path.relative_to(config.repo_root)),
            "config_sha256": config.config_sha256,
            "config_identity": config.identity(),
            "git_revision": _full_git_revision(config.repo_root),
            "git_revision_short": code_revision(),
            "working_tree_clean_at_start": bool(clean_check.passed) if clean_check else None,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "packages": _package_versions(),
        },
        "inputs": [dict(record) for record in inputs.records],
        "source_structure": {
            "joined_rows": int(len(inputs.joined)),
            "sources": int(inputs.joined["source_id"].nunique()),
            "fonts": int(inputs.joined["font_id"].nunique()),
            "cases": int(inputs.joined["case_id"].nunique()),
            "primary_rows": int(inputs.joined["protocol_scope"].eq("primary").sum()),
            "interference_rows": int(
                inputs.joined["variant_family"].isin(INTERFERENCE_FAMILIES).sum()
            ),
            "rows_per_source": sorted(
                inputs.joined.groupby("source_id", sort=True).size().unique().tolist()
            ),
        },
        "metric_definitions": {
            "target": "ssim_target",
            "psnr": "not analysed; diagnostic only",
            "residual": "ssim_target - prediction",
            "residual_sign": "positive = underprediction; negative = overprediction",
            "mae": "mean absolute residual",
            "rmse": "population root mean squared residual",
            "bias": "mean signed residual",
            "r2": "1 - sum(residual^2) / sum((target - target_mean)^2)",
            "r2_suppression": "R² is suppressed when target SD < 0.005 (frozen M8C rule)",
            "large_error": "|residual| >= 0.20 SSIM units; predeclared exploratory threshold",
            "calibration_slope": "ordinary least-squares slope of prediction on target",
            "within_source": "per-source prediction/target SD ratio, slope and rank correlation; medians reported",
        },
        "analysis_decisions": asdict(config.analysis),
        "bootstrap": {
            "method": config.analysis.bootstrap_method,
            "seed": config.run.seed,
            "n_bootstrap": config.run.n_bootstrap,
            "confidence_level": config.run.confidence_level,
            "percentiles": [2.5, 97.5],
            "independent_unit": config.analysis.independent_unit,
            "source_group_column": config.analysis.group_column,
            "paired_font_contrast": config.analysis.font_contrast_rule,
        },
        "thresholds": {
            "large_error_threshold": config.analysis.large_error_threshold,
            "large_error_exploratory": True,
            "near_degenerate_sd": config.analysis.near_degenerate_sd,
        },
        "scope_definitions": SCOPE_DESCRIPTIONS,
        "ground_truth_handling": {
            "path": "data/processed/matched_font_benchmark_v0.1.0/ground_truth.csv",
            "sha256": next(
                record["expected_sha256"]
                for record in inputs.records
                if record["key"] == "ground_truth"
            ),
            "rows": 300,
            "columns_read": ["source_id", "case_id", "font_id"],
            "text_loaded": False,
            "tracked": False,
            "ignored": True,
        },
        "model_context": {
            "model_family": "expanded_random_forest",
            "model_fit_performed": False,
            "calibration_performed": False,
            "feature_change_performed": False,
            "feature_names": list(FEATURE_NAMES),
            "artifact_path": "models/retention_aware_selection_v0_1_0/selected_policy.joblib",
            "artifact_sha256": next(
                record["expected_sha256"]
                for record in inputs.records
                if record["key"] == "model_artifact"
            ),
        },
        "external_verification": list(inputs.external_verification),
        "verification_check_count": len(checked),
        "verification_checks_passed": sum(1 for check in checked if check.passed),
        "output_files": dict(output_hashes),
        "limitations": [
            "Synthetic image-similarity analysis only; SSIM is not human readability, solvability, security or production robustness.",
            "Ten purposive fonts and thirty committed text cases are not a random population sample.",
            "Font identity is entangled with glyph shape, renderer, case composition and the SSIM target.",
            "The 0.20 large-error threshold was predeclared for this analysis but earlier reconnaissance had inspected aggregate and severity summaries; threshold-based subgroup proportions are exploratory.",
            "Feature associations are descriptive rank correlations, not source-clustered inferential tests, and do not identify mechanism.",
            "The elastic 0.5 px rung is diagnostic and excluded from the primary severity table.",
            "M3 and M5C metrics are not like-for-like comparisons to M8C.",
        ],
    }


def _verification_frame(checks: Sequence[Check]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "check_id": check.check_id,
                "category": check.category,
                "expected": check.expected,
                "actual": check.actual,
                "passed": check.passed,
                "details": check.details,
            }
            for check in checks
        ]
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.12g", na_rep="")


def _write_json(payload: object, path: Path) -> None:
    text = json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")


def _relative_files(directory: Path) -> list[str]:
    return sorted(
        str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file()
    )


def _static_analysis_scope_check(checks: list[Check]) -> None:
    path = Path(__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden_imports: list[str] = []
    forbidden_calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in FORBIDDEN_IMPORTS:
                    forbidden_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module in FORBIDDEN_IMPORTS:
            forbidden_imports.append(str(node.module))
        elif isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name):
                name = function.id
            elif isinstance(function, ast.Attribute):
                name = function.attr
            else:
                continue
            if name in FORBIDDEN_CALL_NAMES:
                forbidden_calls.append(name)
    _check(
        checks,
        "analysis_code_scope",
        "scope",
        "no benchmark-generation, model-loading, fitting or prediction calls",
        f"forbidden_imports={sorted(set(forbidden_imports))}; forbidden_calls={sorted(set(forbidden_calls))}",
        not forbidden_imports and not forbidden_calls,
    )


def _append_package_checks(
    checks: list[Check],
    inputs: AnalysisInputs,
    config: AnalysisConfig,
    output_directory: Path,
) -> None:
    _static_analysis_scope_check(checks)
    _rehash_inputs(
        config.repo_root,
        {pin.key: pin for pin in config.inputs},
        checks,
        label="after_report_write",
    )
    all_files = _relative_files(output_directory)
    _check(
        checks,
        "output_file_count",
        "package",
        f"{len(OUTPUT_FILES)} declared output files",
        f"{len(all_files)} files",
        len(all_files) == len(OUTPUT_FILES),
    )
    empty = [
        relative for relative in all_files if (output_directory / relative).stat().st_size == 0
    ]
    _check(
        checks,
        "output_files_nonempty",
        "package",
        "every output file is non-empty",
        f"empty={empty}",
        not empty,
    )
    figure_files = [name for name in all_files if name.startswith("figures/")]
    _check(
        checks,
        "figure_count",
        "package",
        f"{len(FIGURE_FILES)} SVG figures",
        f"{len(figure_files)} figures",
        len(figure_files) == len(FIGURE_FILES),
    )
    malformed = []
    for name in FIGURE_FILES:
        try:
            ET.parse(output_directory / "figures" / name)
        except ET.ParseError:
            malformed.append(name)
    _check(
        checks,
        "figure_svg_well_formed",
        "package",
        "every declared figure is well-formed SVG",
        f"malformed={malformed}",
        not malformed,
    )
    text_headers = []
    for name in (
        "summary_metrics.csv",
        "condition_calibration.csv",
        "font_condition_metrics.csv",
        "severity_metrics.csv",
        "source_group_metrics.csv",
        "case_metrics.csv",
        "source_characteristic_associations.csv",
        "feature_error_associations.csv",
        "paired_condition_contrasts.csv",
        "error_concentration.csv",
    ):
        header = (output_directory / name).read_text(encoding="utf-8").splitlines()[0]
        text_headers.extend(header.split(","))
    _check(
        checks,
        "ground_truth_text_not_written",
        "ground_truth_policy",
        "no 'text' column in any output CSV",
        "absent" if "text" not in text_headers else "present",
        "text" not in text_headers,
    )


def write_report_package(
    inputs: AnalysisInputs,
    result: AnalysisResult,
    config: AnalysisConfig,
    *,
    overwrite: bool = False,
) -> Path:
    """Write the new M9A report package atomically.

    The only authorised destination is a new
    ``reports/interference_error_analysis_v*`` directory.
    """
    output_directory = validate_output_directory(config.repo_root, config.run.output_reports)
    if output_directory.exists() and not overwrite:
        raise AnalysisError(
            f"report directory already exists; pass --overwrite to replace it: {output_directory}"
        )
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.staging-",
            dir=output_directory.parent,
        )
    )
    try:
        _write_csv(result.summary_metrics, staging / "summary_metrics.csv")
        _write_csv(result.condition_calibration, staging / "condition_calibration.csv")
        _write_csv(result.font_condition_metrics, staging / "font_condition_metrics.csv")
        _write_csv(result.severity_metrics, staging / "severity_metrics.csv")
        _write_csv(result.source_group_metrics, staging / "source_group_metrics.csv")
        _write_csv(result.case_metrics, staging / "case_metrics.csv")
        _write_csv(
            result.source_characteristic_associations,
            staging / "source_characteristic_associations.csv",
        )
        _write_csv(result.feature_error_associations, staging / "feature_error_associations.csv")
        _write_csv(result.paired_condition_contrasts, staging / "paired_condition_contrasts.csv")
        _write_csv(result.error_concentration, staging / "error_concentration.csv")
        build_figures(result, staging)
        (staging / "README.md").write_text(build_readme(inputs, result, config), encoding="utf-8")
        (staging / "hypotheses_and_followups.md").write_text(
            build_hypotheses(result), encoding="utf-8"
        )

        checks = list(result.checks)
        _write_csv(_verification_frame(checks), staging / "verification_checks.csv")
        preliminary_hashes = {
            relative: sha256_file(staging / relative)
            for relative in _relative_files(staging)
            if relative != "analysis_protocol.json"
        }
        protocol = build_protocol(inputs, result, config, preliminary_hashes, checks=checks)
        _write_json(protocol, staging / "analysis_protocol.json")

        _append_package_checks(checks, inputs, config, staging)
        actual = _relative_files(staging)
        expected = sorted(OUTPUT_FILES)
        _check(
            checks,
            "output_membership_exact",
            "package",
            "exact declared output file set",
            f"missing={sorted(set(expected) - set(actual))}; extra={sorted(set(actual) - set(expected))}",
            set(actual) == set(expected),
        )
        _write_csv(_verification_frame(checks), staging / "verification_checks.csv")
        final_hashes = {
            relative: sha256_file(staging / relative)
            for relative in _relative_files(staging)
            if relative != "analysis_protocol.json"
        }
        protocol = build_protocol(inputs, result, config, final_hashes, checks=checks)
        _write_json(protocol, staging / "analysis_protocol.json")
        final_actual = _relative_files(staging)
        if set(final_actual) != set(expected):
            raise AnalysisError(
                f"final output membership mismatch: missing={sorted(set(expected) - set(final_actual))}; "
                f"extra={sorted(set(final_actual) - set(expected))}"
            )
        if any(not check.passed for check in checks):
            failed = [check.check_id for check in checks if not check.passed]
            raise AnalysisError(f"failed package checks: {failed}")

        if output_directory.exists():
            if not overwrite:
                raise AnalysisError(f"report directory appeared during writing: {output_directory}")
            shutil.rmtree(output_directory)
        staging.rename(output_directory)
        return output_directory
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def run_analysis(
    config_path: Path | str = DEFAULT_CONFIG,
    *,
    overwrite: bool = False,
    run_verifiers: bool = True,
    require_clean: bool = True,
) -> Path:
    started = time.perf_counter()
    config = load_config(config_path)
    inputs = load_inputs(
        config,
        run_verifiers=run_verifiers,
        require_clean=require_clean,
    )
    result = build_analysis_result(inputs, config)
    output = write_report_package(inputs, result, config, overwrite=overwrite)
    elapsed = time.perf_counter() - started
    recorded = pd.read_csv(output / "verification_checks.csv")
    print(f"[PASS] report: {output.relative_to(config.repo_root)}")
    print(
        f"[PASS] checks: {int(recorded['passed'].sum())}/{len(recorded)} "
        f"package checks; {elapsed:.1f}s"
    )
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="captcha-interference-error-analysis",
        description=(
            "Read-only M9A analysis of frozen M8C thin-line and arc-stroke "
            "prediction errors. No model is loaded, fitted or selected."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="write the new M9A report package")
    run_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="path to the frozen analysis configuration",
    )
    run_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing M9A report directory after a complete staging run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        try:
            run_analysis(args.config, overwrite=args.overwrite)
        except (AnalysisError, OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
