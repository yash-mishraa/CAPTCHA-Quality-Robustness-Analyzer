"""Milestone M7C cross-font evaluation analysis.

M7C is a read-only analysis of the frozen M7B held-out-font predictions and the
frozen M5C familiar-font predictions. It does not generate data, train, refit,
recalibrate, select or tune a model, change a target or split, or write into any
prior dataset/model/report directory.

The independent unit is a source group. Bootstrap replicates resample whole
source groups and keep every row of a sampled source together. The three
held-out fonts are a purposive sample, not a sample of font identities. Font
differences describe the observed source distribution; they do not identify a
font-population effect. The target is SSIM against an exact synthetic clean
reference, not human readability, solve rate, security or production
robustness.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import shutil
import sys
import time
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from captcha_quality.corruptions import derive_noise_seed
from captcha_quality.dataset import CheckResult, VerificationReport
from captcha_quality.modelling import TARGET, regression_metrics
from captcha_quality.provenance import (
    canonical_json,
    code_revision,
    json_safe,
    package_version,
    sha256_bytes,
    sha256_file,
)

ANALYSIS_VERSION: Final[str] = "v0.1.0"
CLASSIFICATIONS: Final[tuple[str, ...]] = ("supported", "suggestive", "unsupported")
FONT_ORDER: Final[tuple[str, ...]] = ("georgia", "monaco", "impact")
FAMILY_ORDER: Final[tuple[str, ...]] = (
    "clean",
    "gaussian_blur",
    "gaussian_noise",
    "thin_lines",
    "arc_strokes",
    "elastic",
)
SCOPE_ORDER: Final[tuple[str, ...]] = (
    "all_primary",
    "original_blur_noise",
    "gaussian_blur",
    "gaussian_noise",
    "interference",
    "thin_lines",
    "arc_strokes",
    "geometric_primary",
    "geometric_primary_1px",
    "geometric_primary_2px",
    "geometric_primary_3px",
    "clean_reference",
    "diagnostic_0_5px",
)
PRIMARY_SCOPES: Final[tuple[str, ...]] = SCOPE_ORDER[:11]
DIAGNOSTIC_SCOPES: Final[tuple[str, ...]] = SCOPE_ORDER[11:]
PAIRED_CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "gaussian_noise_minus_gaussian_blur": ("gaussian_noise", "gaussian_blur"),
    "thin_lines_minus_arc_strokes": ("thin_lines", "arc_strokes"),
    "interference_minus_original_blur_noise": (
        "interference",
        "original_blur_noise",
    ),
    "geometric_primary_minus_original_blur_noise": (
        "geometric_primary",
        "original_blur_noise",
    ),
    "geometric_2px_minus_1px": ("geometric_primary_2px", "geometric_primary_1px"),
    "geometric_3px_minus_2px": ("geometric_primary_3px", "geometric_primary_2px"),
    "geometric_3px_minus_1px": ("geometric_primary_3px", "geometric_primary_1px"),
}
FAMILIAR_SCOPE_MAP: Final[dict[str, str]] = {
    "original_blur_noise": "original_blur_noise",
    "gaussian_blur": "original_blur_noise_gaussian_blur",
    "gaussian_noise": "original_blur_noise_gaussian_noise",
    "geometric_primary": "geometric_primary",
    "geometric_primary_1px": "geometric_primary_severity_2",
    "geometric_primary_2px": "geometric_primary_severity_3",
    "geometric_primary_3px": "geometric_primary_severity_4",
    "clean_reference": "clean_reference",
    "diagnostic_0_5px": "diagnostic_0_5",
}
FAMILIAR_SEVERITY_TO_AMPLITUDE: Final[dict[int, float]] = {2: 1.0, 3: 2.0, 4: 3.0}
CLAIM_IDS: Final[tuple[str, ...]] = tuple(f"C{index:02d}" for index in range(1, 14))
REQUIRED_INPUT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "m7b_config",
        "m7b_manifest",
        "m7b_sources",
        "m7b_features",
        "m7b_summary",
        "m7b_predictions",
        "m7b_overall_metrics",
        "m7b_font_metrics",
        "m7b_family_metrics",
        "m7b_source_metrics",
        "m7b_uncertainty",
        "m7b_comparison",
        "m7b_target_distributions",
        "m7b_dataset_composition",
        "m7b_isolation_checks",
        "m7b_run_metadata",
        "m7b_report",
        "m5c_config",
        "m5c_test_predictions",
        "m5c_test_slice_metrics",
        "m5c_uncertainty",
        "m5c_paired_uncertainty",
        "m5c_run_metadata",
        "m5c_report",
        "m5c_model_card",
        "m5c_model_artifact",
        "m5a_config",
        "m5a_manifest",
        "m7a_config",
        "m7a_uncertainty",
        "m7a_error_concentration",
    }
)
OUTPUT_TABLE_NAMES: Final[tuple[str, ...]] = (
    "input_manifest.csv",
    "font_metrics.csv",
    "family_metrics.csv",
    "severity_metrics.csv",
    "source_metrics.csv",
    "error_concentration.csv",
    "uncertainty.csv",
    "paired_differences.csv",
    "font_differences.csv",
    "benchmark_comparison.csv",
    "claim_evidence.csv",
    "isolation_checks.csv",
)
REPORT_SECTIONS: Final[tuple[str, ...]] = (
    "## Executive summary",
    "## 1. Read-only protocol and pinned inputs",
    "## 2. Data and source accounting",
    "## 3. Per-font performance",
    "## 4. Corruption-family analysis",
    "## 5. Geometric severity",
    "## 6. Source-level error concentration",
    "## 7. Uncertainty and paired comparisons",
    "## 8. Familiar versus held-out benchmark",
    "## 9. Claim-to-evidence audit",
    "## 10. Limitations",
    "## 11. Reproduce and verify",
    "## 12. Research-integrity boundary",
)
M7B_PREDICTION_COLUMNS: Final[tuple[str, ...]] = (
    "sample_id",
    "source_id",
    "font_id",
    "text_length",
    "variant_family",
    "severity_level",
    "amplitude_px",
    "protocol_scope",
    "image_sha256",
    "ssim_target",
    "prediction",
    "residual",
    "abs_residual",
)
M5C_PREDICTION_COLUMNS: Final[tuple[str, ...]] = (
    "model",
    "scope",
    "split",
    "sample_id",
    "source_id",
    "font_id",
    "text_length",
    "family",
    "severity_level",
    "ssim_target",
    "prediction",
    "residual",
    "abs_residual",
)
REPORTS_ROOT: Final[Path] = Path("reports")
PROTECTED_REPORT_ROOTS: Final[tuple[str, ...]] = (
    "reports/benchmark_v0_1_0",
    "reports/captcha_quality_v0.1.0",
    "reports/captcha_quality_v0.1.0-pilot",
    "reports/dataset_adequacy_audit_clean_run_1",
    "reports/dataset_adequacy_audit_clean_run_2",
    "reports/dataset_adequacy_audit_v0_1_0",
    "reports/geometric_distortion_pilot",
    "reports/geometric_robustness_benchmark_v0_1_0",
    "reports/heldout_font_benchmark_v0_1_0",
    "reports/local_structure_study_v0_1_0",
    "reports/pilot",
    "reports/retention_aware_selection_v0_1_0",
    "reports/retention_aware_training_v0_1_0",
    "reports/robustness_benchmark_v0_1_0",
    "reports/robustness_pilot",
    "reports/spatial_feature_study_v0_1_0",
    "reports/spatial_resolution_study_v0_1_0",
    "reports/structure_tensor_study_v0_1_0",
)
ALLOWED_FONT_DIFFERENCE_RULES: Final[frozenset[str]] = frozenset(
    {"any_pairwise_primary_rmse_ci_excludes_zero"}
)
ALLOWED_BLUR_NOISE_RULES: Final[frozenset[str]] = frozenset({"all_font_rmse_above_familiar_point"})
ALLOWED_GEOMETRIC_RULES: Final[frozenset[str]] = frozenset({"severity_rmse_range_ratio_ge_1_5"})
ALLOWED_CONCENTRATION_RULES: Final[frozenset[str]] = frozenset(
    {"top3_share_ge_0_25_and_effective_fraction_le_0_5"}
)


class CrossFontError(RuntimeError):
    """Raised when a frozen input or analysis invariant is violated."""


class CrossFontConfigError(ValueError):
    """Raised for an invalid M7C configuration."""


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
    group_column: str
    independent_unit: str
    bootstrap_method: str
    near_degenerate_sd: float
    font_ids: tuple[str, ...]
    family_order: tuple[str, ...]
    primary_scopes: tuple[str, ...]
    diagnostic_scopes: tuple[str, ...]
    geometric_amplitudes_primary_px: tuple[float, ...]
    geometric_amplitude_diagnostic_px: float
    paired_contrasts: tuple[str, ...]
    font_difference_scopes: tuple[str, ...]
    familiar_comparison_scopes: tuple[str, ...]
    top_k_sources: int
    font_difference_rule: str
    blur_noise_consistency_rule: str
    geometric_concealment_rule: str
    concentration_rule: str


@dataclass(frozen=True, slots=True)
class InputPin:
    key: str
    path: str
    sha256: str
    role: str


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    run: RunSettings
    analysis: AnalysisSettings
    inputs: tuple[InputPin, ...]
    config_path: Path
    config_sha256: str
    config_hash: str

    def identity(self) -> dict[str, Any]:
        """Return the canonical config identity used for replay checks."""
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


def _check_keys(section: Mapping[str, object], allowed: set[str], name: str) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise CrossFontConfigError(f"unknown key(s) in [{name}]: {unknown}")
    missing = sorted(allowed - set(section))
    if missing:
        raise CrossFontConfigError(f"missing key(s) in [{name}]: {missing}")


def _section(payload: Mapping[str, object], name: str) -> dict[str, object]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise CrossFontConfigError(f"[{name}] must be a table")
    return dict(value)


def _as_str(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CrossFontConfigError(f"{label} must be a non-empty string")
    return value


def _as_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CrossFontConfigError(f"{label} must be an integer")
    return value


def _as_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CrossFontConfigError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise CrossFontConfigError(f"{label} must be finite")
    return result


def _as_str_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise CrossFontConfigError(f"{label} must be a list of strings")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise CrossFontConfigError(f"{label} must not contain duplicates")
    return result


def _as_float_tuple(value: object, label: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise CrossFontConfigError(f"{label} must be a list of numbers")
    result = tuple(_as_float(item, label) for item in value)
    if len(set(result)) != len(result):
        raise CrossFontConfigError(f"{label} must not contain duplicates")
    return result


def _sha256(value: object, label: str) -> str:
    text = _as_str(value, label)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise CrossFontConfigError(f"{label} must be a lowercase 64-character SHA-256")
    return text


def _parse_run(raw: Mapping[str, object]) -> RunSettings:
    _check_keys(
        raw,
        {
            "name",
            "version",
            "seed",
            "n_bootstrap",
            "confidence_level",
            "output_reports",
        },
        "run",
    )
    run = RunSettings(
        name=_as_str(raw["name"], "run.name"),
        version=_as_str(raw["version"], "run.version"),
        seed=_as_int(raw["seed"], "run.seed"),
        n_bootstrap=_as_int(raw["n_bootstrap"], "run.n_bootstrap"),
        confidence_level=_as_float(raw["confidence_level"], "run.confidence_level"),
        output_reports=Path(_as_str(raw["output_reports"], "run.output_reports")),
    )
    if run.seed < 0:
        raise CrossFontConfigError("run.seed must be >= 0")
    if run.n_bootstrap < 1:
        raise CrossFontConfigError("run.n_bootstrap must be >= 1")
    if not 0.0 < run.confidence_level < 1.0:
        raise CrossFontConfigError("run.confidence_level must be strictly between 0 and 1")
    return run


def _parse_analysis(raw: Mapping[str, object]) -> AnalysisSettings:
    _check_keys(
        raw,
        {
            "group_column",
            "independent_unit",
            "bootstrap_method",
            "near_degenerate_sd",
            "font_ids",
            "family_order",
            "primary_scopes",
            "diagnostic_scopes",
            "geometric_amplitudes_primary_px",
            "geometric_amplitude_diagnostic_px",
            "paired_contrasts",
            "font_difference_scopes",
            "familiar_comparison_scopes",
            "top_k_sources",
            "font_difference_rule",
            "blur_noise_consistency_rule",
            "geometric_concealment_rule",
            "concentration_rule",
        },
        "analysis",
    )
    settings = AnalysisSettings(
        group_column=_as_str(raw["group_column"], "analysis.group_column"),
        independent_unit=_as_str(raw["independent_unit"], "analysis.independent_unit"),
        bootstrap_method=_as_str(raw["bootstrap_method"], "analysis.bootstrap_method"),
        near_degenerate_sd=_as_float(raw["near_degenerate_sd"], "analysis.near_degenerate_sd"),
        font_ids=_as_str_tuple(raw["font_ids"], "analysis.font_ids"),
        family_order=_as_str_tuple(raw["family_order"], "analysis.family_order"),
        primary_scopes=_as_str_tuple(raw["primary_scopes"], "analysis.primary_scopes"),
        diagnostic_scopes=_as_str_tuple(raw["diagnostic_scopes"], "analysis.diagnostic_scopes"),
        geometric_amplitudes_primary_px=_as_float_tuple(
            raw["geometric_amplitudes_primary_px"],
            "analysis.geometric_amplitudes_primary_px",
        ),
        geometric_amplitude_diagnostic_px=_as_float(
            raw["geometric_amplitude_diagnostic_px"],
            "analysis.geometric_amplitude_diagnostic_px",
        ),
        paired_contrasts=_as_str_tuple(raw["paired_contrasts"], "analysis.paired_contrasts"),
        font_difference_scopes=_as_str_tuple(
            raw["font_difference_scopes"], "analysis.font_difference_scopes"
        ),
        familiar_comparison_scopes=_as_str_tuple(
            raw["familiar_comparison_scopes"], "analysis.familiar_comparison_scopes"
        ),
        top_k_sources=_as_int(raw["top_k_sources"], "analysis.top_k_sources"),
        font_difference_rule=_as_str(raw["font_difference_rule"], "analysis.font_difference_rule"),
        blur_noise_consistency_rule=_as_str(
            raw["blur_noise_consistency_rule"],
            "analysis.blur_noise_consistency_rule",
        ),
        geometric_concealment_rule=_as_str(
            raw["geometric_concealment_rule"],
            "analysis.geometric_concealment_rule",
        ),
        concentration_rule=_as_str(raw["concentration_rule"], "analysis.concentration_rule"),
    )
    if settings.group_column != "source_id" or settings.independent_unit != "source_id":
        raise CrossFontConfigError("M7C requires source_id as the independent unit")
    if settings.bootstrap_method != "source_grouped_percentile":
        raise CrossFontConfigError("analysis.bootstrap_method must be 'source_grouped_percentile'")
    if settings.near_degenerate_sd <= 0.0:
        raise CrossFontConfigError("analysis.near_degenerate_sd must be positive")
    if settings.font_ids != FONT_ORDER:
        raise CrossFontConfigError(f"analysis.font_ids must be {FONT_ORDER}")
    if settings.family_order != FAMILY_ORDER:
        raise CrossFontConfigError(f"analysis.family_order must be {FAMILY_ORDER}")
    if set(settings.primary_scopes) | set(settings.diagnostic_scopes) != set(SCOPE_ORDER):
        raise CrossFontConfigError("primary and diagnostic scopes must partition the scope set")
    if set(settings.primary_scopes) & set(settings.diagnostic_scopes):
        raise CrossFontConfigError("primary and diagnostic scopes must be disjoint")
    if settings.primary_scopes != PRIMARY_SCOPES or settings.diagnostic_scopes != DIAGNOSTIC_SCOPES:
        raise CrossFontConfigError("M7C scope order must match the frozen analysis protocol")
    amplitudes = settings.geometric_amplitudes_primary_px
    if amplitudes != (1.0, 2.0, 3.0):
        raise CrossFontConfigError("primary geometric amplitudes must be [1.0, 2.0, 3.0]")
    if settings.geometric_amplitude_diagnostic_px != 0.5:
        raise CrossFontConfigError("diagnostic geometric amplitude must be 0.5 px")
    unknown_contrasts = sorted(set(settings.paired_contrasts) - set(PAIRED_CONTRASTS))
    if unknown_contrasts:
        raise CrossFontConfigError(f"unknown paired contrast(s): {unknown_contrasts}")
    unknown_font_scopes = sorted(set(settings.font_difference_scopes) - set(SCOPE_ORDER))
    if unknown_font_scopes:
        raise CrossFontConfigError(f"unknown font-difference scope(s): {unknown_font_scopes}")
    unknown_familiar = sorted(set(settings.familiar_comparison_scopes) - set(FAMILIAR_SCOPE_MAP))
    if unknown_familiar:
        raise CrossFontConfigError(f"unknown familiar comparison scope(s): {unknown_familiar}")
    if settings.top_k_sources < 1:
        raise CrossFontConfigError("analysis.top_k_sources must be >= 1")
    if settings.font_difference_rule not in ALLOWED_FONT_DIFFERENCE_RULES:
        raise CrossFontConfigError("unknown analysis.font_difference_rule")
    if settings.blur_noise_consistency_rule not in ALLOWED_BLUR_NOISE_RULES:
        raise CrossFontConfigError("unknown analysis.blur_noise_consistency_rule")
    if settings.geometric_concealment_rule not in ALLOWED_GEOMETRIC_RULES:
        raise CrossFontConfigError("unknown analysis.geometric_concealment_rule")
    if settings.concentration_rule not in ALLOWED_CONCENTRATION_RULES:
        raise CrossFontConfigError("unknown analysis.concentration_rule")
    return settings


def _parse_inputs(raw: object) -> tuple[InputPin, ...]:
    if not isinstance(raw, list) or not raw:
        raise CrossFontConfigError("[[inputs]] must be a non-empty array of tables")
    pins: list[InputPin] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise CrossFontConfigError(f"inputs[{index}] must be a table")
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
        raise CrossFontConfigError("input keys must be unique")
    missing_keys = sorted(REQUIRED_INPUT_KEYS - set(keys))
    if missing_keys:
        raise CrossFontConfigError(f"missing required input pin(s): {missing_keys}")
    unknown_keys = sorted(set(keys) - REQUIRED_INPUT_KEYS)
    if unknown_keys:
        raise CrossFontConfigError(f"unknown input pin(s): {unknown_keys}")
    return tuple(pins)


def load_config(path: Path | str) -> AnalysisConfig:
    """Load and validate the M7C configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise CrossFontConfigError(f"missing M7C configuration: {config_path}")
    raw_bytes = config_path.read_bytes()
    try:
        payload = tomllib.loads(raw_bytes.decode("utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise CrossFontConfigError(f"invalid TOML in {config_path}: {error}") from error
    _check_keys(payload, {"run", "analysis", "inputs"}, "top-level")
    run = _parse_run(_section(payload, "run"))
    analysis = _parse_analysis(_section(payload, "analysis"))
    inputs = _parse_inputs(payload.get("inputs"))
    config = AnalysisConfig(
        run=run,
        analysis=analysis,
        inputs=inputs,
        config_path=config_path,
        config_sha256=sha256_bytes(raw_bytes),
        config_hash="",
    )
    identity_hash = sha256_bytes(canonical_json(config.identity()).encode("utf-8"))
    return AnalysisConfig(
        run=config.run,
        analysis=config.analysis,
        inputs=config.inputs,
        config_path=config.config_path,
        config_sha256=config.config_sha256,
        config_hash=identity_hash,
    )


@dataclass(frozen=True, slots=True)
class AnalysisInputs:
    """Verified frozen inputs and the derived analysis frames."""

    config: AnalysisConfig
    m7b_predictions: pd.DataFrame
    m7b_manifest: pd.DataFrame
    m7b_sources: pd.DataFrame
    m7b_summary: dict[str, Any]
    m5c_predictions: pd.DataFrame
    m5c_metrics: pd.DataFrame
    m5c_uncertainty: pd.DataFrame
    m7a_uncertainty: pd.DataFrame
    m5a_manifest: pd.DataFrame
    model_card: dict[str, Any]
    m7b_metadata: dict[str, Any]
    m5c_metadata: dict[str, Any]
    input_hashes: dict[str, str]
    input_manifest: pd.DataFrame
    checks: tuple[CheckResult, ...]


def _pin_by_key(config: AnalysisConfig, key: str) -> InputPin:
    for pin in config.inputs:
        if pin.key == key:
            return pin
    raise CrossFontError(f"configuration is missing required input pin: {key}")


def _require_file_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise CrossFontError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise CrossFontError(
            f"input hash mismatch for {label}: {path} has {actual}, expected {expected}"
        )
    return actual


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise CrossFontError(f"missing {label}: {path}")
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise CrossFontError(f"cannot read {label}: {path}: {error}") from error


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise CrossFontError(f"missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CrossFontError(f"cannot read {label}: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise CrossFontError(f"{label} must contain a JSON object: {path}")
    return payload


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise CrossFontError(f"{label} is missing column(s): {missing}")


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
        raise CrossFontError(f"{label} failed: {detail}")


def _scope_mask(frame: pd.DataFrame, scope: str) -> pd.Series:
    if scope == "all_primary":
        return frame["protocol_scope"].eq("primary")
    if scope == "original_blur_noise":
        return frame["variant_family"].isin(["gaussian_blur", "gaussian_noise"])
    if scope == "gaussian_blur":
        return frame["variant_family"].eq("gaussian_blur")
    if scope == "gaussian_noise":
        return frame["variant_family"].eq("gaussian_noise")
    if scope == "interference":
        return frame["variant_family"].isin(["thin_lines", "arc_strokes"])
    if scope == "thin_lines":
        return frame["variant_family"].eq("thin_lines")
    if scope == "arc_strokes":
        return frame["variant_family"].eq("arc_strokes")
    if scope == "geometric_primary":
        return frame["variant_family"].eq("elastic") & frame["amplitude_px"].isin([1.0, 2.0, 3.0])
    if scope == "geometric_primary_1px":
        return frame["variant_family"].eq("elastic") & frame["amplitude_px"].eq(1.0)
    if scope == "geometric_primary_2px":
        return frame["variant_family"].eq("elastic") & frame["amplitude_px"].eq(2.0)
    if scope == "geometric_primary_3px":
        return frame["variant_family"].eq("elastic") & frame["amplitude_px"].eq(3.0)
    if scope == "clean_reference":
        return frame["variant_family"].eq("clean")
    if scope == "diagnostic_0_5px":
        return frame["variant_family"].eq("elastic") & frame["amplitude_px"].eq(0.5)
    raise CrossFontError(f"unknown analysis scope: {scope}")


def _scope_frame(frame: pd.DataFrame, scope: str, font_id: str = "all") -> pd.DataFrame:
    subset = frame.loc[_scope_mask(frame, scope)].copy()
    if font_id != "all":
        subset = subset.loc[subset["font_id"] == font_id].copy()
    return subset


def _target_stats(values: pd.Series) -> dict[str, float | int | bool]:
    array = values.to_numpy(dtype=np.float64)
    if array.size == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "sd": math.nan,
            "variance": math.nan,
            "min": math.nan,
            "max": math.nan,
            "q05": math.nan,
            "q25": math.nan,
            "q50": math.nan,
            "q75": math.nan,
            "q95": math.nan,
        }
    quantiles = np.quantile(array, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=0)),
        "variance": float(np.var(array, ddof=0)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "q05": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "q50": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q95": float(quantiles[4]),
    }


def _metric_values(
    frame: pd.DataFrame,
    *,
    near_degenerate_sd: float,
    image_column: str | None = "image_sha256",
) -> dict[str, Any]:
    if frame.empty:
        return {
            "n_rows": 0,
            "n_sources": 0,
            "n_distinct_images": 0,
            "target_mean": math.nan,
            "target_sd": math.nan,
            "target_variance": math.nan,
            "target_min": math.nan,
            "target_max": math.nan,
            "target_q05": math.nan,
            "target_q25": math.nan,
            "target_q50": math.nan,
            "target_q75": math.nan,
            "target_q95": math.nan,
            "mae": math.nan,
            "rmse": math.nan,
            "r2_raw": math.nan,
            "r2": math.nan,
            "r2_defined": False,
            "r2_suppressed": False,
            "near_degenerate": True,
            "mean_residual": math.nan,
            "sd_residual": math.nan,
        }
    target = frame[TARGET].to_numpy(dtype=np.float64)
    predicted = frame["prediction"].to_numpy(dtype=np.float64)
    metrics = regression_metrics(target, predicted)
    stats = _target_stats(frame[TARGET])
    near_degenerate = bool(float(stats["sd"]) < near_degenerate_sd)
    r2_defined = bool(metrics["r2_defined"]) and not near_degenerate
    residual = target - predicted
    return {
        "n_rows": int(len(frame)),
        "n_sources": int(frame["source_id"].nunique()),
        "n_distinct_images": (
            int(frame[image_column].nunique()) if image_column is not None else math.nan
        ),
        "target_mean": float(stats["mean"]),
        "target_sd": float(stats["sd"]),
        "target_variance": float(stats["variance"]),
        "target_min": float(stats["min"]),
        "target_max": float(stats["max"]),
        "target_q05": float(stats["q05"]),
        "target_q25": float(stats["q25"]),
        "target_q50": float(stats["q50"]),
        "target_q75": float(stats["q75"]),
        "target_q95": float(stats["q95"]),
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "r2_raw": float(metrics["r2"]),
        "r2": float(metrics["r2"]) if r2_defined else math.nan,
        "r2_defined": r2_defined,
        "r2_suppressed": bool(metrics["r2_defined"] and near_degenerate),
        "near_degenerate": near_degenerate,
        "mean_residual": float(np.mean(residual)),
        "sd_residual": float(np.std(residual, ddof=0)),
    }


def _indices_by_group(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    groups = frame["source_id"].to_numpy()
    unique = sorted(set(groups))
    return {group: np.flatnonzero(groups == group) for group in unique}


def _bootstrap_group_metrics(
    frame: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Source-grouped percentile bootstrap for MAE and RMSE."""
    if frame.empty:
        return {
            "mae_ci_low": math.nan,
            "mae_ci_high": math.nan,
            "rmse_ci_low": math.nan,
            "rmse_ci_high": math.nan,
            "n_groups": 0,
            "n_samples": 0,
            "n_bootstrap": n_bootstrap,
        }
    target = frame[TARGET].to_numpy(dtype=np.float64)
    predicted = frame["prediction"].to_numpy(dtype=np.float64)
    indices = _indices_by_group(frame)
    groups = np.array(sorted(indices))
    rng = np.random.default_rng(seed)
    mae_values: list[float] = []
    rmse_values: list[float] = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        index = np.concatenate([indices[group] for group in sampled])
        metrics = regression_metrics(target[index], predicted[index])
        mae_values.append(float(metrics["mae"]))
        rmse_values.append(float(metrics["rmse"]))
    low = 100.0 * (1.0 - confidence_level) / 2.0
    high = 100.0 - low
    return {
        "mae_ci_low": float(np.percentile(mae_values, low)),
        "mae_ci_high": float(np.percentile(mae_values, high)),
        "rmse_ci_low": float(np.percentile(rmse_values, low)),
        "rmse_ci_high": float(np.percentile(rmse_values, high)),
        "n_groups": int(len(groups)),
        "n_samples": int(len(frame)),
        "n_bootstrap": int(n_bootstrap),
    }


def _paired_metric_difference(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    *,
    metric: str,
    n_bootstrap: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Paired source-grouped bootstrap for one metric difference A - B."""
    if metric not in {"mae", "rmse"}:
        raise CrossFontError(f"unsupported paired metric: {metric}")
    sources_a = set(frame_a["source_id"])
    sources_b = set(frame_b["source_id"])
    common = sorted(sources_a & sources_b)
    if not common or sources_a != sources_b:
        return {
            "eligible": False,
            "reason": "source sets differ or are empty",
            "estimate_a": math.nan,
            "estimate_b": math.nan,
            "difference": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "n_groups": 0,
            "n_samples_a": int(len(frame_a)),
            "n_samples_b": int(len(frame_b)),
        }
    indices_a = _indices_by_group(frame_a)
    indices_b = _indices_by_group(frame_b)
    if any(len(indices_a[source]) == 0 or len(indices_b[source]) == 0 for source in common):
        return {
            "eligible": False,
            "reason": "one or more paired sources have no rows on one side",
            "estimate_a": math.nan,
            "estimate_b": math.nan,
            "difference": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "n_groups": 0,
            "n_samples_a": int(len(frame_a)),
            "n_samples_b": int(len(frame_b)),
        }
    groups = np.array(common)
    target_a = frame_a[TARGET].to_numpy(dtype=np.float64)
    pred_a = frame_a["prediction"].to_numpy(dtype=np.float64)
    target_b = frame_b[TARGET].to_numpy(dtype=np.float64)
    pred_b = frame_b["prediction"].to_numpy(dtype=np.float64)
    estimate_a = float(regression_metrics(target_a, pred_a)[metric])
    estimate_b = float(regression_metrics(target_b, pred_b)[metric])
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        index_a = np.concatenate([indices_a[source] for source in sampled])
        index_b = np.concatenate([indices_b[source] for source in sampled])
        metric_a = float(regression_metrics(target_a[index_a], pred_a[index_a])[metric])
        metric_b = float(regression_metrics(target_b[index_b], pred_b[index_b])[metric])
        differences.append(metric_a - metric_b)
    low = 100.0 * (1.0 - confidence_level) / 2.0
    high = 100.0 - low
    return {
        "eligible": True,
        "reason": "",
        "estimate_a": estimate_a,
        "estimate_b": estimate_b,
        "difference": estimate_a - estimate_b,
        "ci_low": float(np.percentile(differences, low)),
        "ci_high": float(np.percentile(differences, high)),
        "n_groups": int(len(groups)),
        "n_samples_a": int(len(frame_a)),
        "n_samples_b": int(len(frame_b)),
    }


def _unpaired_metric_difference(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    *,
    metric: str,
    n_bootstrap: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Independent source-grouped bootstrap for one metric difference A - B."""
    if metric not in {"mae", "rmse"}:
        raise CrossFontError(f"unsupported unpaired metric: {metric}")
    if frame_a.empty or frame_b.empty:
        return {
            "eligible": False,
            "reason": "one side is empty",
            "estimate_a": math.nan,
            "estimate_b": math.nan,
            "difference": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "n_groups_a": 0,
            "n_groups_b": 0,
            "n_samples_a": int(len(frame_a)),
            "n_samples_b": int(len(frame_b)),
        }
    groups_a = np.array(sorted(set(frame_a["source_id"])))
    groups_b = np.array(sorted(set(frame_b["source_id"])))
    indices_a = _indices_by_group(frame_a)
    indices_b = _indices_by_group(frame_b)
    target_a = frame_a[TARGET].to_numpy(dtype=np.float64)
    pred_a = frame_a["prediction"].to_numpy(dtype=np.float64)
    target_b = frame_b[TARGET].to_numpy(dtype=np.float64)
    pred_b = frame_b["prediction"].to_numpy(dtype=np.float64)
    estimate_a = float(regression_metrics(target_a, pred_a)[metric])
    estimate_b = float(regression_metrics(target_b, pred_b)[metric])
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(n_bootstrap):
        sampled_a = rng.choice(groups_a, size=len(groups_a), replace=True)
        sampled_b = rng.choice(groups_b, size=len(groups_b), replace=True)
        index_a = np.concatenate([indices_a[source] for source in sampled_a])
        index_b = np.concatenate([indices_b[source] for source in sampled_b])
        metric_a = float(regression_metrics(target_a[index_a], pred_a[index_a])[metric])
        metric_b = float(regression_metrics(target_b[index_b], pred_b[index_b])[metric])
        differences.append(metric_a - metric_b)
    low = 100.0 * (1.0 - confidence_level) / 2.0
    high = 100.0 - low
    return {
        "eligible": True,
        "reason": "",
        "estimate_a": estimate_a,
        "estimate_b": estimate_b,
        "difference": estimate_a - estimate_b,
        "ci_low": float(np.percentile(differences, low)),
        "ci_high": float(np.percentile(differences, high)),
        "n_groups_a": int(len(groups_a)),
        "n_groups_b": int(len(groups_b)),
        "n_samples_a": int(len(frame_a)),
        "n_samples_b": int(len(frame_b)),
    }


def _validate_m7b_inputs(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    sources: pd.DataFrame,
    summary: Mapping[str, Any],
    config: AnalysisConfig,
) -> None:
    if tuple(predictions.columns) != M7B_PREDICTION_COLUMNS:
        raise CrossFontError(f"M7B predictions schema changed: {tuple(predictions.columns)}")
    if len(predictions) != 1260:
        raise CrossFontError(f"M7B predictions must have 1260 rows, found {len(predictions)}")
    if predictions["sample_id"].duplicated().any():
        raise CrossFontError("M7B predictions contain duplicate sample_id values")
    if predictions["source_id"].nunique() != 60:
        raise CrossFontError("M7B predictions must contain 60 source groups")
    font_counts = predictions.groupby("font_id")["source_id"].nunique().to_dict()
    expected_font_counts = {font: 20 for font in FONT_ORDER}
    if font_counts != expected_font_counts:
        raise CrossFontError(f"M7B source counts by font are {font_counts}")
    if predictions["image_sha256"].nunique() != 1260:
        raise CrossFontError("M7B predictions do not contain 1260 distinct image hashes")
    if not np.isfinite(predictions["ssim_target"]).all():
        raise CrossFontError("M7B predictions contain non-finite targets")
    if not np.isfinite(predictions["prediction"]).all():
        raise CrossFontError("M7B predictions contain non-finite predictions")
    if not predictions["ssim_target"].between(0.0, 1.0).all():
        raise CrossFontError("M7B targets fall outside [0, 1]")
    residual = predictions["ssim_target"] - predictions["prediction"]
    if not np.allclose(residual, predictions["residual"], atol=1e-12, rtol=0.0):
        raise CrossFontError("M7B residual column does not equal target - prediction")
    if not np.allclose(residual.abs(), predictions["abs_residual"], atol=1e-12, rtol=0.0):
        raise CrossFontError("M7B abs_residual column does not equal |target - prediction|")
    scope_counts = predictions["protocol_scope"].value_counts().to_dict()
    if scope_counts != {"primary": 1140, "diagnostic": 120}:
        raise CrossFontError(f"M7B protocol-scope counts are {scope_counts}")
    if set(predictions["font_id"]) != set(FONT_ORDER):
        raise CrossFontError("M7B predictions contain an unexpected font id")
    if set(predictions["variant_family"]) != set(FAMILY_ORDER):
        raise CrossFontError("M7B predictions contain an unexpected family")
    _require_columns(sources, ["source_id", "font_id"], "M7B sources")
    if set(sources["source_id"]) != set(predictions["source_id"]):
        raise CrossFontError("M7B sources and predictions disagree on source ids")
    if set(sources["font_id"]) != set(FONT_ORDER):
        raise CrossFontError("M7B sources contain an unexpected font id")
    summary_counts = summary.get("counts", {})
    if not isinstance(summary_counts, Mapping):
        raise CrossFontError("M7B dataset summary has no counts object")
    if int(summary_counts.get("rows", -1)) != 1260:
        raise CrossFontError("M7B dataset summary row count is not 1260")
    if int(summary_counts.get("sources", -1)) != 60:
        raise CrossFontError("M7B dataset summary source count is not 60")
    if summary_counts.get("splits") != {"eval": 60}:
        raise CrossFontError("M7B dataset summary split policy is not one eval split")
    if summary.get("config_sha256") != _pin_by_key(config, "m7b_config").sha256:
        raise CrossFontError("M7B dataset summary does not match the pinned M7B config")
    _require_columns(
        manifest,
        [
            "sample_id",
            "source_id",
            "font_id",
            "split",
            "variant_family",
            "severity_level",
            "amplitude_px",
            "protocol_scope",
            "image_sha256",
            "ssim_target",
        ],
        "M7B manifest",
    )
    if len(manifest) != len(predictions):
        raise CrossFontError("M7B manifest and predictions row counts differ")
    joined = predictions.merge(
        manifest[
            [
                "sample_id",
                "source_id",
                "font_id",
                "variant_family",
                "severity_level",
                "amplitude_px",
                "protocol_scope",
                "image_sha256",
                "ssim_target",
            ]
        ],
        on="sample_id",
        suffixes=("_prediction", "_manifest"),
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(predictions):
        raise CrossFontError("M7B predictions and manifest do not share every sample_id")
    for column in ("source_id", "font_id", "variant_family", "protocol_scope", "image_sha256"):
        if not (joined[f"{column}_prediction"] == joined[f"{column}_manifest"]).all():
            raise CrossFontError(f"M7B predictions and manifest disagree on {column}")
    if not np.allclose(
        joined["ssim_target_prediction"],
        joined["ssim_target_manifest"],
        atol=1e-12,
        rtol=0.0,
    ):
        raise CrossFontError("M7B predictions and manifest disagree on ssim_target")
    amplitude_prediction = joined["amplitude_px_prediction"].to_numpy(dtype=np.float64)
    amplitude_manifest = joined["amplitude_px_manifest"].to_numpy(dtype=np.float64)
    if not np.allclose(amplitude_prediction, amplitude_manifest, equal_nan=True, atol=1e-12):
        raise CrossFontError("M7B predictions and manifest disagree on amplitude_px")
    severity_prediction = joined["severity_level_prediction"].to_numpy(dtype=np.float64)
    severity_manifest = joined["severity_level_manifest"].to_numpy(dtype=np.float64)
    if not np.allclose(severity_prediction, severity_manifest, equal_nan=True, atol=1e-12):
        raise CrossFontError("M7B predictions and manifest disagree on severity_level")
    if set(predictions["source_id"]) & set(manifest.loc[manifest["split"] != "eval", "source_id"]):
        raise CrossFontError("M7B predictions include a non-eval source")


def _validate_m5c_inputs(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    uncertainty: pd.DataFrame,
) -> None:
    if tuple(predictions.columns) != M5C_PREDICTION_COLUMNS:
        raise CrossFontError(f"M5C predictions schema changed: {tuple(predictions.columns)}")
    selected = predictions.loc[predictions["model"] == "expanded_random_forest"].copy()
    if selected.empty:
        raise CrossFontError("M5C predictions do not contain expanded_random_forest")
    if len(selected) != 396:
        raise CrossFontError(f"M5C selected model must have 396 rows, found {len(selected)}")
    if selected["source_id"].nunique() != 12:
        raise CrossFontError("M5C selected model must contain 12 familiar source groups")
    if not np.isfinite(selected["ssim_target"]).all():
        raise CrossFontError("M5C predictions contain non-finite targets")
    if not np.isfinite(selected["prediction"]).all():
        raise CrossFontError("M5C predictions contain non-finite predictions")
    residual = selected["ssim_target"] - selected["prediction"]
    if not np.allclose(residual, selected["residual"], atol=1e-12, rtol=0.0):
        raise CrossFontError("M5C residual column does not equal target - prediction")
    if not np.allclose(residual.abs(), selected["abs_residual"], atol=1e-12, rtol=0.0):
        raise CrossFontError("M5C abs_residual column does not equal |target - prediction|")
    selected_metrics = metrics.loc[metrics["model"] == "expanded_random_forest"]
    if len(selected_metrics) != 10:
        raise CrossFontError("M5C selected model must have 10 frozen metric rows")
    for _, row in selected_metrics.iterrows():
        scope = str(row["scope"])
        frame = selected.loc[selected["scope"] == scope]
        if frame.empty:
            raise CrossFontError(f"M5C predictions are missing scope {scope}")
        computed = regression_metrics(frame[TARGET], frame["prediction"])
        if int(computed["n_rows"]) != int(row["n_samples"]):
            raise CrossFontError(f"M5C row count mismatch for scope {scope}")
        if int(frame["source_id"].nunique()) != int(row["n_sources"]):
            raise CrossFontError(f"M5C source count mismatch for scope {scope}")
        if not math.isclose(float(computed["mae"]), float(row["mae"]), rel_tol=1e-9, abs_tol=1e-12):
            raise CrossFontError(f"M5C MAE replay mismatch for scope {scope}")
        if not math.isclose(
            float(computed["rmse"]), float(row["rmse"]), rel_tol=1e-9, abs_tol=1e-12
        ):
            raise CrossFontError(f"M5C RMSE replay mismatch for scope {scope}")
        residual_scope = frame[TARGET] - frame["prediction"]
        if not math.isclose(
            float(residual_scope.mean()),
            float(row["mean_residual"]),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise CrossFontError(f"M5C mean-residual replay mismatch for scope {scope}")
        if not math.isclose(
            float(residual_scope.std(ddof=1)),
            float(row["sd_residual"]),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise CrossFontError(f"M5C residual-SD replay mismatch for scope {scope}")
    required = {"model", "scope", "metric", "n_groups", "n_bootstrap"}
    _require_columns(uncertainty, required, "M5C uncertainty")
    selected_uncertainty = uncertainty.loc[uncertainty["model"] == "expanded_random_forest"]
    if selected_uncertainty.empty:
        raise CrossFontError("M5C uncertainty has no expanded_random_forest rows")
    if set(selected_uncertainty["n_groups"]) != {12}:
        raise CrossFontError("M5C uncertainty does not use 12 independent source groups")
    if set(selected_uncertainty["n_bootstrap"]) != {1000}:
        raise CrossFontError("M5C uncertainty does not use 1000 bootstrap replicates")
    if set(selected_uncertainty["metric"]) != {"mae", "rmse", "r2"}:
        raise CrossFontError("M5C uncertainty metrics changed")


def _validate_m5a_mapping(manifest: pd.DataFrame) -> None:
    _require_columns(
        manifest,
        ["variant_family", "severity_level", "amplitude_px"],
        "M5A manifest",
    )
    elastic = manifest.loc[manifest["variant_family"] == "elastic"]
    observed = (
        elastic.groupby("severity_level")["amplitude_px"]
        .agg(lambda values: tuple(sorted(set(float(value) for value in values))))
        .to_dict()
    )
    expected = {
        1: (0.5,),
        **{
            severity: (amplitude,) for severity, amplitude in FAMILIAR_SEVERITY_TO_AMPLITUDE.items()
        },
    }
    if observed != expected:
        raise CrossFontError(
            f"M5A severity-to-amplitude mapping changed: observed {observed}, expected {expected}"
        )


def _validate_m7a_convention(uncertainty: pd.DataFrame) -> None:
    required = {"method", "confidence_level", "independent_groups", "n_bootstrap"}
    _require_columns(uncertainty, required, "M7A uncertainty")
    if not (uncertainty["method"] == "source_grouped_percentile").all():
        raise CrossFontError("M7A uncertainty does not use source-grouped percentile intervals")
    if not np.allclose(uncertainty["confidence_level"], 0.95):
        raise CrossFontError("M7A uncertainty confidence level is not 0.95")
    if set(uncertainty["independent_groups"]) != {12}:
        raise CrossFontError("M7A uncertainty does not use 12 independent source groups")
    if int(uncertainty["n_bootstrap"].min()) != 2000:
        raise CrossFontError("M7A uncertainty does not use 2000 bootstrap replicates")


def _validate_model_identity(card: Mapping[str, Any], config: AnalysisConfig) -> None:
    if card.get("model_family") != "expanded_random_forest":
        raise CrossFontError("M5C champion model family is not expanded_random_forest")
    if card.get("calibration") != "none":
        raise CrossFontError("M5C champion calibration is not none")
    if card.get("artifact_sha256") != _pin_by_key(config, "m5c_model_artifact").sha256:
        raise CrossFontError("M5C champion card and artifact pin disagree")
    feature_names = card.get("feature_names")
    if not isinstance(feature_names, list) or len(feature_names) != 8:
        raise CrossFontError("M5C champion card does not record eight frozen features")


def _validate_m7b_metadata(metadata: Mapping[str, Any], config: AnalysisConfig) -> None:
    model = metadata.get("model")
    if not isinstance(model, Mapping):
        raise CrossFontError("M7B metadata has no model identity")
    if model.get("artifact_sha256") != _pin_by_key(config, "m5c_model_artifact").sha256:
        raise CrossFontError("M7B metadata model artifact differs from the pinned champion")
    if model.get("calibration") != "none":
        raise CrossFontError("M7B metadata model calibration is not none")


def _validate_m5c_metadata(metadata: Mapping[str, Any]) -> None:
    selection = metadata.get("selection")
    if not isinstance(selection, Mapping):
        raise CrossFontError("M5C metadata has no selection record")
    if selection.get("selected_model") != "expanded_random_forest":
        raise CrossFontError("M5C metadata selected model is not expanded_random_forest")
    if selection.get("selected_calibration") != "none":
        raise CrossFontError("M5C metadata selected calibration is not none")


def load_inputs(config: AnalysisConfig) -> AnalysisInputs:
    """Verify every pinned input and load the frozen analysis frames."""
    input_hashes: dict[str, str] = {}
    manifest_rows: list[dict[str, Any]] = []
    paths: dict[str, Path] = {}
    for pin in config.inputs:
        path = Path(pin.path)
        actual = _require_file_hash(path, pin.sha256, pin.key)
        input_hashes[pin.key] = actual
        paths[pin.key] = path
        manifest_rows.append(
            {
                "key": pin.key,
                "role": pin.role,
                "path": pin.path,
                "sha256": actual,
                "bytes": int(path.stat().st_size),
                "verified": True,
            }
        )
    input_manifest = pd.DataFrame(manifest_rows)
    m7b_predictions = _read_csv(paths["m7b_predictions"], "M7B predictions")
    m7b_manifest = _read_csv(paths["m7b_manifest"], "M7B manifest")
    m7b_sources = _read_csv(paths["m7b_sources"], "M7B sources")
    m7b_summary = _read_json(paths["m7b_summary"], "M7B dataset summary")
    m5c_predictions = _read_csv(paths["m5c_test_predictions"], "M5C test predictions")
    m5c_metrics = _read_csv(paths["m5c_test_slice_metrics"], "M5C test slice metrics")
    m5c_uncertainty = _read_csv(paths["m5c_uncertainty"], "M5C uncertainty")
    m7a_uncertainty = _read_csv(paths["m7a_uncertainty"], "M7A uncertainty")
    m5a_manifest = _read_csv(paths["m5a_manifest"], "M5A manifest")
    model_card = _read_json(paths["m5c_model_card"], "M5C champion card")
    m7b_metadata = _read_json(paths["m7b_run_metadata"], "M7B run metadata")
    m5c_metadata = _read_json(paths["m5c_run_metadata"], "M5C run metadata")
    checks: list[CheckResult] = []
    _check(
        checks,
        "m7b_predictions",
        lambda: (
            _validate_m7b_inputs(m7b_predictions, m7b_manifest, m7b_sources, m7b_summary, config)
            or "schema, counts, targets and manifest identity verified"
        ),
    )
    _check(
        checks,
        "m5c_predictions",
        lambda: (
            _validate_m5c_inputs(m5c_predictions, m5c_metrics, m5c_uncertainty)
            or "schema, counts, metrics and residual replay verified"
        ),
    )
    _check(
        checks,
        "m5a_severity_mapping",
        lambda: _validate_m5a_mapping(m5a_manifest) or "severity 1/2/3/4 maps to 0.5/1/2/3 px",
    )
    _check(
        checks,
        "m7a_bootstrap_convention",
        lambda: (
            _validate_m7a_convention(m7a_uncertainty)
            or "source-grouped percentile, 12 groups, 2000 replicates"
        ),
    )
    _check(
        checks,
        "model_identity",
        lambda: (
            _validate_model_identity(model_card, config)
            or "expanded_random_forest / no calibration / eight features"
        ),
    )
    _check(
        checks,
        "m7b_model_identity",
        lambda: (
            _validate_m7b_metadata(m7b_metadata, config)
            or "M7B metadata model matches the pinned champion"
        ),
    )
    _check(
        checks,
        "m5c_model_identity",
        lambda: (
            _validate_m5c_metadata(m5c_metadata)
            or "M5C metadata selection matches the pinned champion"
        ),
    )
    _require_all_checks(checks, "M7C input checks")
    return AnalysisInputs(
        config=config,
        m7b_predictions=m7b_predictions,
        m7b_manifest=m7b_manifest,
        m7b_sources=m7b_sources,
        m7b_summary=m7b_summary,
        m5c_predictions=m5c_predictions.loc[
            m5c_predictions["model"] == "expanded_random_forest"
        ].copy(),
        m5c_metrics=m5c_metrics,
        m5c_uncertainty=m5c_uncertainty,
        m7a_uncertainty=m7a_uncertainty,
        m5a_manifest=m5a_manifest,
        model_card=model_card,
        m7b_metadata=m7b_metadata,
        m5c_metadata=m5c_metadata,
        input_hashes=input_hashes,
        input_manifest=input_manifest,
        checks=tuple(checks),
    )


def _protocol_role(scope: str) -> str:
    return "primary" if scope in PRIMARY_SCOPES else "diagnostic"


def _metric_row(
    metrics: pd.DataFrame,
    *,
    scope: str,
    font_id: str = "all",
) -> pd.Series:
    subset = metrics.loc[(metrics["scope"] == scope) & (metrics["font_id"] == font_id)]
    if len(subset) != 1:
        raise CrossFontError(f"expected one metric row for scope={scope}, font={font_id}")
    return subset.iloc[0]


def build_family_metrics(inputs: AnalysisInputs) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope in SCOPE_ORDER:
        for font_id in ("all", *FONT_ORDER):
            frame = _scope_frame(inputs.m7b_predictions, scope, font_id)
            values = _metric_values(
                frame, near_degenerate_sd=inputs.config.analysis.near_degenerate_sd
            )
            rows.append(
                {
                    "scope": scope,
                    "protocol_role": _protocol_role(scope),
                    "font_id": font_id,
                    **values,
                }
            )
    return pd.DataFrame(rows)


def build_font_metrics(family_metrics: pd.DataFrame) -> pd.DataFrame:
    summary_scopes = (
        "all_primary",
        "original_blur_noise",
        "interference",
        "geometric_primary",
        "clean_reference",
        "diagnostic_0_5px",
    )
    subset = family_metrics.loc[
        (family_metrics["font_id"] != "all") & (family_metrics["scope"].isin(summary_scopes))
    ].copy()
    return subset.reset_index(drop=True)


def build_severity_metrics(inputs: AnalysisInputs) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for font_id in ("all", *FONT_ORDER):
        primary = inputs.m7b_predictions.loc[
            (inputs.m7b_predictions["variant_family"] == "elastic")
            & (inputs.m7b_predictions["amplitude_px"].isin([1.0, 2.0, 3.0]))
        ]
        if font_id != "all":
            primary = primary.loc[primary["font_id"] == font_id]
        pivot = primary.groupby(["source_id", "amplitude_px"])[TARGET].mean().unstack()
        if {1.0, 2.0, 3.0}.issubset(pivot.columns):
            monotonic = (pivot[1.0] > pivot[2.0]) & (pivot[2.0] > pivot[3.0])
            n_sources_monotonic = int(monotonic.sum())
            n_sources_primary = int(len(pivot))
        else:
            n_sources_monotonic = 0
            n_sources_primary = 0
        aggregate_means = {
            amplitude: float(primary.loc[primary["amplitude_px"] == amplitude, TARGET].mean())
            for amplitude in (1.0, 2.0, 3.0)
        }
        aggregate_order_consistent = bool(
            aggregate_means[1.0] > aggregate_means[2.0] > aggregate_means[3.0]
        )
        for amplitude in (0.5, 1.0, 2.0, 3.0):
            frame = inputs.m7b_predictions.loc[
                (inputs.m7b_predictions["variant_family"] == "elastic")
                & (inputs.m7b_predictions["amplitude_px"] == amplitude)
            ]
            if font_id != "all":
                frame = frame.loc[frame["font_id"] == font_id]
            values = _metric_values(
                frame, near_degenerate_sd=inputs.config.analysis.near_degenerate_sd
            )
            severity_level = int(frame["severity_level"].iloc[0]) if not frame.empty else -1
            rows.append(
                {
                    "amplitude_px": amplitude,
                    "protocol_role": "diagnostic" if amplitude == 0.5 else "primary",
                    "font_id": font_id,
                    "severity_level": severity_level,
                    "aggregate_target_order_consistent": aggregate_order_consistent,
                    "n_sources_monotonic": n_sources_monotonic,
                    "n_sources_primary": n_sources_primary,
                    "source_monotonic_fraction": (
                        n_sources_monotonic / n_sources_primary if n_sources_primary else math.nan
                    ),
                    **values,
                }
            )
    return pd.DataFrame(rows)


def build_source_metrics(inputs: AnalysisInputs) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope in SCOPE_ORDER:
        frame = _scope_frame(inputs.m7b_predictions, scope)
        for (source_id, font_id), group in frame.groupby(["source_id", "font_id"], sort=True):
            values = _metric_values(
                group, near_degenerate_sd=inputs.config.analysis.near_degenerate_sd
            )
            residual = group[TARGET].to_numpy(dtype=np.float64) - group["prediction"].to_numpy(
                dtype=np.float64
            )
            rows.append(
                {
                    "scope": scope,
                    "protocol_role": _protocol_role(scope),
                    "source_id": source_id,
                    "font_id": font_id,
                    "n_rows": int(len(group)),
                    "n_distinct_images": int(group["image_sha256"].nunique()),
                    "target_mean": values["target_mean"],
                    "target_sd": values["target_sd"],
                    "mae": values["mae"],
                    "rmse": values["rmse"],
                    "r2_raw": values["r2_raw"],
                    "r2": values["r2"],
                    "r2_defined": values["r2_defined"],
                    "r2_suppressed": values["r2_suppressed"],
                    "near_degenerate": values["near_degenerate"],
                    "mean_residual": values["mean_residual"],
                    "sd_residual": values["sd_residual"],
                    "sse": float(np.sum(residual**2)),
                }
            )
    source_metrics = pd.DataFrame(rows)
    totals = source_metrics.groupby("scope")["sse"].transform("sum")
    source_metrics["sse_share"] = np.where(
        totals.to_numpy(dtype=np.float64) > 0.0,
        source_metrics["sse"].to_numpy(dtype=np.float64) / totals.to_numpy(dtype=np.float64),
        0.0,
    )
    source_metrics["sse_rank"] = (
        source_metrics.groupby("scope")["sse"].rank(method="min", ascending=False).astype(int)
    )
    return source_metrics


def build_error_concentration(
    source_metrics: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    top_k = config.analysis.top_k_sources
    for scope in SCOPE_ORDER:
        for font_id in ("all", *FONT_ORDER):
            if font_id == "all":
                subset = source_metrics.loc[source_metrics["scope"] == scope]
            else:
                subset = source_metrics.loc[
                    (source_metrics["scope"] == scope) & (source_metrics["font_id"] == font_id)
                ]
            if subset.empty:
                continue
            sse = subset["sse"].to_numpy(dtype=np.float64)
            total_sse = float(np.sum(sse))
            n_sources = int(len(subset))
            if total_sse > 0.0:
                sorted_sse = np.sort(sse)[::-1]
                shares = sorted_sse / total_sse
                top1_share = float(shares[0])
                top_k_share = float(np.sum(shares[: min(top_k, n_sources)]))
                hhi = float(np.sum((sse / total_sse) ** 2))
                effective_sources = float(1.0 / hhi) if hhi > 0.0 else math.nan
                effective_fraction = (
                    effective_sources / n_sources if math.isfinite(effective_sources) else math.nan
                )
                cumulative = np.cumsum(shares)
                sources_for_50 = int(np.searchsorted(cumulative, 0.5) + 1)
                sources_for_80 = int(np.searchsorted(cumulative, 0.8) + 1)
                cv_sse = float(np.std(sse, ddof=0) / np.mean(sse))
                max_index = subset["sse"].idxmax()
                max_source = str(subset.loc[max_index, "source_id"])
                max_sse = float(subset.loc[max_index, "sse"])
            else:
                top1_share = math.nan
                top_k_share = math.nan
                hhi = math.nan
                effective_sources = math.nan
                effective_fraction = math.nan
                sources_for_50 = 0
                sources_for_80 = 0
                cv_sse = math.nan
                max_source = ""
                max_sse = math.nan
            rows.append(
                {
                    "scope": scope,
                    "protocol_role": _protocol_role(scope),
                    "font_id": font_id,
                    "n_sources": n_sources,
                    "n_rows": int(subset["n_rows"].sum()),
                    "total_sse": total_sse,
                    "top_k": top_k,
                    "top1_source": max_source,
                    "top1_share": top1_share,
                    "top_k_share": top_k_share,
                    "hhi": hhi,
                    "effective_sources": effective_sources,
                    "effective_fraction": effective_fraction,
                    "sources_for_50pct": sources_for_50,
                    "sources_for_80pct": sources_for_80,
                    "cv_sse": cv_sse,
                    "max_source_sse": max_sse,
                    "caveat": (
                        "Descriptive influence summary; no source is excluded and "
                        "no model is refit."
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_uncertainty(inputs: AnalysisInputs) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    config = inputs.config
    for scope in SCOPE_ORDER:
        for font_id in ("all", *FONT_ORDER):
            frame = _scope_frame(inputs.m7b_predictions, scope, font_id)
            values = _metric_values(frame, near_degenerate_sd=config.analysis.near_degenerate_sd)
            bootstrap = _bootstrap_group_metrics(
                frame,
                n_bootstrap=config.run.n_bootstrap,
                seed=derive_noise_seed(config.run.seed, "m7c_bootstrap", scope, font_id),
                confidence_level=config.run.confidence_level,
            )
            for metric in ("mae", "rmse"):
                rows.append(
                    {
                        "scope": scope,
                        "protocol_role": _protocol_role(scope),
                        "font_id": font_id,
                        "metric": metric,
                        "estimate": values[metric],
                        "ci_low": bootstrap[f"{metric}_ci_low"],
                        "ci_high": bootstrap[f"{metric}_ci_high"],
                        "n_groups": bootstrap["n_groups"],
                        "n_samples": bootstrap["n_samples"],
                        "n_bootstrap": bootstrap["n_bootstrap"],
                        "confidence_level": config.run.confidence_level,
                        "method": config.analysis.bootstrap_method,
                        "target_sd": values["target_sd"],
                        "target_near_degenerate": values["near_degenerate"],
                        "caveat": (
                            "Source-grouped percentile interval; descriptive uncertainty "
                            "summary, not a significance test or a font-population estimate."
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_paired_differences(inputs: AnalysisInputs) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    config = inputs.config
    for contrast in config.analysis.paired_contrasts:
        scope_a, scope_b = PAIRED_CONTRASTS[contrast]
        for font_id in ("all", *FONT_ORDER):
            frame_a = _scope_frame(inputs.m7b_predictions, scope_a, font_id)
            frame_b = _scope_frame(inputs.m7b_predictions, scope_b, font_id)
            for metric in ("mae", "rmse"):
                result = _paired_metric_difference(
                    frame_a,
                    frame_b,
                    metric=metric,
                    n_bootstrap=config.run.n_bootstrap,
                    seed=derive_noise_seed(
                        config.run.seed, "m7c_paired", contrast, font_id, metric
                    ),
                    confidence_level=config.run.confidence_level,
                )
                rows.append(
                    {
                        "contrast": contrast,
                        "scope_a": scope_a,
                        "scope_b": scope_b,
                        "font_id": font_id,
                        "metric": metric,
                        "estimate_a": result["estimate_a"],
                        "estimate_b": result["estimate_b"],
                        "difference": result["difference"],
                        "ci_low": result["ci_low"],
                        "ci_high": result["ci_high"],
                        "n_groups": result["n_groups"],
                        "n_samples_a": result["n_samples_a"],
                        "n_samples_b": result["n_samples_b"],
                        "n_bootstrap": config.run.n_bootstrap,
                        "confidence_level": config.run.confidence_level,
                        "method": config.analysis.bootstrap_method,
                        "paired": True,
                        "eligible": result["eligible"],
                        "reason": result["reason"],
                        "caveat": (
                            "Paired by source; positive difference means higher error on "
                            "side A. Descriptive percentile interval, not a significance test."
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_font_differences(inputs: AnalysisInputs) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    config = inputs.config
    font_pairs = (
        (FONT_ORDER[0], FONT_ORDER[1]),
        (FONT_ORDER[0], FONT_ORDER[2]),
        (FONT_ORDER[1], FONT_ORDER[2]),
    )
    for scope in config.analysis.font_difference_scopes:
        for font_a, font_b in font_pairs:
            frame_a = _scope_frame(inputs.m7b_predictions, scope, font_a)
            frame_b = _scope_frame(inputs.m7b_predictions, scope, font_b)
            for metric in ("mae", "rmse"):
                result = _unpaired_metric_difference(
                    frame_a,
                    frame_b,
                    metric=metric,
                    n_bootstrap=config.run.n_bootstrap,
                    seed=derive_noise_seed(
                        config.run.seed, "m7c_font_difference", scope, font_a, font_b, metric
                    ),
                    confidence_level=config.run.confidence_level,
                )
                rows.append(
                    {
                        "scope": scope,
                        "protocol_role": _protocol_role(scope),
                        "font_a": font_a,
                        "font_b": font_b,
                        "metric": metric,
                        "estimate_a": result["estimate_a"],
                        "estimate_b": result["estimate_b"],
                        "difference": result["difference"],
                        "ci_low": result["ci_low"],
                        "ci_high": result["ci_high"],
                        "n_groups_a": result["n_groups_a"],
                        "n_groups_b": result["n_groups_b"],
                        "n_samples_a": result["n_samples_a"],
                        "n_samples_b": result["n_samples_b"],
                        "n_bootstrap": config.run.n_bootstrap,
                        "confidence_level": config.run.confidence_level,
                        "method": config.analysis.bootstrap_method,
                        "paired": False,
                        "eligible": result["eligible"],
                        "reason": result["reason"],
                        "caveat": (
                            "Unpaired font groups with distinct sources. Difference describes "
                            "the observed source distribution; three purposive fonts are not a "
                            "font population, and this is not a significance test."
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_benchmark_comparison(inputs: AnalysisInputs) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    config = inputs.config
    familiar = inputs.m5c_predictions
    for scope in config.analysis.familiar_comparison_scopes:
        familiar_scope = FAMILIAR_SCOPE_MAP[scope]
        heldout = _scope_frame(inputs.m7b_predictions, scope)
        familiar_frame = familiar.loc[familiar["scope"] == familiar_scope].copy()
        if familiar_frame.empty:
            raise CrossFontError(f"M5C predictions are missing scope {familiar_scope}")
        heldout_values = _metric_values(
            heldout, near_degenerate_sd=config.analysis.near_degenerate_sd
        )
        familiar_values = _metric_values(
            familiar_frame,
            near_degenerate_sd=config.analysis.near_degenerate_sd,
            image_column=None,
        )
        heldout_bootstrap = _bootstrap_group_metrics(
            heldout,
            n_bootstrap=config.run.n_bootstrap,
            seed=derive_noise_seed(config.run.seed, "m7c_benchmark_heldout", scope),
            confidence_level=config.run.confidence_level,
        )
        familiar_bootstrap = _bootstrap_group_metrics(
            familiar_frame,
            n_bootstrap=config.run.n_bootstrap,
            seed=derive_noise_seed(config.run.seed, "m7c_benchmark_familiar", scope),
            confidence_level=config.run.confidence_level,
        )
        for metric in ("mae", "rmse"):
            result = _unpaired_metric_difference(
                heldout,
                familiar_frame,
                metric=metric,
                n_bootstrap=config.run.n_bootstrap,
                seed=derive_noise_seed(config.run.seed, "m7c_benchmark_difference", scope, metric),
                confidence_level=config.run.confidence_level,
            )
            heldout_estimate = float(heldout_values[metric])
            familiar_estimate = float(familiar_values[metric])
            familiar_ci_low = float(familiar_bootstrap[f"{metric}_ci_low"])
            familiar_ci_high = float(familiar_bootstrap[f"{metric}_ci_high"])
            rows.append(
                {
                    "m7b_scope": scope,
                    "familiar_scope": familiar_scope,
                    "metric": metric,
                    "m7b_estimate": heldout_estimate,
                    "familiar_estimate": familiar_estimate,
                    "difference": result["difference"],
                    "ci_low": result["ci_low"],
                    "ci_high": result["ci_high"],
                    "m7b_ci_low": heldout_bootstrap[f"{metric}_ci_low"],
                    "m7b_ci_high": heldout_bootstrap[f"{metric}_ci_high"],
                    "familiar_ci_low": familiar_ci_low,
                    "familiar_ci_high": familiar_ci_high,
                    "heldout_point_outside_familiar_ci": bool(
                        heldout_estimate < familiar_ci_low or heldout_estimate > familiar_ci_high
                    ),
                    "m7b_n_rows": heldout_values["n_rows"],
                    "m7b_n_sources": heldout_values["n_sources"],
                    "m7b_n_distinct_images": heldout_values["n_distinct_images"],
                    "familiar_n_rows": familiar_values["n_rows"],
                    "familiar_n_sources": familiar_values["n_sources"],
                    "m7b_target_mean": heldout_values["target_mean"],
                    "m7b_target_sd": heldout_values["target_sd"],
                    "familiar_target_mean": familiar_values["target_mean"],
                    "familiar_target_sd": familiar_values["target_sd"],
                    "m7b_near_degenerate": heldout_values["near_degenerate"],
                    "familiar_near_degenerate": familiar_values["near_degenerate"],
                    "m7b_r2": heldout_values["r2"],
                    "familiar_r2": familiar_values["r2"],
                    "paired": False,
                    "eligible": result["eligible"],
                    "comparison_type": "unpaired_descriptive",
                    "comparable": "partial",
                    "caveat": (
                        "Unpaired benchmarks; fonts, source composition/counts and corruption "
                        "mix differ. Descriptive only; not causal, not equivalence, and not a "
                        "significance test."
                    ),
                }
            )
    return pd.DataFrame(rows)


def _ci_excludes_zero(row: pd.Series) -> bool:
    if not bool(row["eligible"]):
        return False
    low = float(row["ci_low"])
    high = float(row["ci_high"])
    return bool(math.isfinite(low) and math.isfinite(high) and (low > 0.0 or high < 0.0))


def _claim_row(
    *,
    claim_id: str,
    claim: str,
    classification: str,
    evidence: str,
    limitations: str,
    primary_metric: str,
    primary_value: float,
    supporting_count: int,
    scope: str,
    rule: str,
    rule_met: bool,
) -> dict[str, Any]:
    if classification not in CLASSIFICATIONS:
        raise CrossFontError(f"invalid claim classification: {classification}")
    if not evidence.strip() or not limitations.strip():
        raise CrossFontError(f"claim {claim_id} has empty evidence or limitations")
    return {
        "claim_id": claim_id,
        "claim": claim,
        "classification": classification,
        "evidence": evidence,
        "limitations": limitations,
        "primary_metric": primary_metric,
        "primary_value": primary_value,
        "supporting_count": int(supporting_count),
        "scope": scope,
        "rule": rule,
        "rule_met": bool(rule_met),
    }


def build_claim_evidence(
    inputs: AnalysisInputs,
    family_metrics: pd.DataFrame,
    severity_metrics: pd.DataFrame,
    error_concentration: pd.DataFrame,
    font_differences: pd.DataFrame,
    benchmark_comparison: pd.DataFrame,
) -> pd.DataFrame:
    config = inputs.config
    primary_font_rows = family_metrics.loc[
        (family_metrics["scope"] == "all_primary") & (family_metrics["font_id"] != "all")
    ]
    font_rmse = {str(row["font_id"]): float(row["rmse"]) for _, row in primary_font_rows.iterrows()}
    primary_pairs = font_differences.loc[
        (font_differences["scope"] == "all_primary") & (font_differences["metric"] == "rmse")
    ]
    excluding = primary_pairs.loc[primary_pairs.apply(_ci_excludes_zero, axis=1)]
    pair_summary = "; ".join(
        f"{row['font_a']}-{row['font_b']} {float(row['difference']):+.4f} "
        f"[{float(row['ci_low']):+.4f}, {float(row['ci_high']):+.4f}]"
        for _, row in primary_pairs.iterrows()
    )
    max_abs_pair = (
        float(primary_pairs["difference"].abs().max()) if len(primary_pairs) else math.nan
    )
    if len(excluding) > 0:
        font_classification = "supported"
    elif max_abs_pair > 0.0:
        font_classification = "suggestive"
    else:
        font_classification = "unsupported"
    font_evidence = (
        "Observed all-primary RMSE by font: "
        + ", ".join(f"{font}={font_rmse[font]:.4f}" for font in FONT_ORDER)
        + ". Unpaired source-grouped difference intervals: "
        + pair_summary
        + "."
    )
    font_limitations = (
        "Font groups contain distinct sources and are unpaired. The three fonts are a "
        "purposive sample, not a random sample of font identities. A supported label "
        "applies only to the observed source distribution and does not establish a "
        "general font effect."
    )
    claims: list[dict[str, Any]] = [
        _claim_row(
            claim_id="C01",
            claim=(
                "The frozen champion's all-primary error differs across the three tested "
                "held-out fonts on the observed source distribution."
            ),
            classification=font_classification,
            evidence=font_evidence,
            limitations=font_limitations,
            primary_metric="max_abs_all_primary_rmse_pair_difference",
            primary_value=max_abs_pair,
            supporting_count=len(excluding),
            scope="all_primary",
            rule=config.analysis.font_difference_rule,
            rule_met=len(excluding) > 0,
        )
    ]

    familiar_blur_row = benchmark_comparison.loc[
        (benchmark_comparison["m7b_scope"] == "original_blur_noise")
        & (benchmark_comparison["metric"] == "rmse")
    ].iloc[0]
    familiar_blur_rmse = float(familiar_blur_row["familiar_estimate"])
    blur_font_rows = family_metrics.loc[
        (family_metrics["scope"] == "original_blur_noise") & (family_metrics["font_id"] != "all")
    ]
    blur_font_rmse = {
        str(row["font_id"]): float(row["rmse"]) for _, row in blur_font_rows.iterrows()
    }
    above = [font for font in FONT_ORDER if blur_font_rmse[font] > familiar_blur_rmse]
    all_above = len(above) == len(FONT_ORDER)
    if all_above:
        blur_classification = "supported"
    elif above:
        blur_classification = "suggestive"
    else:
        blur_classification = "unsupported"
    blur_evidence = (
        f"Familiar original blur/noise RMSE is {familiar_blur_rmse:.4f}. Held-out "
        "original blur/noise RMSE by font: "
        + ", ".join(f"{font}={blur_font_rmse[font]:.4f}" for font in FONT_ORDER)
        + f". {len(above)} of 3 held-out fonts exceed the familiar point estimate."
    )
    blur_limitations = (
        "The familiar and held-out benchmarks are unmatched: fonts, source counts and "
        "source composition differ. Exceeding a point estimate is a descriptive "
        "comparison, not a causal font effect or a matched significance test."
    )
    claims.append(
        _claim_row(
            claim_id="C02",
            claim=(
                "Blur/noise RMSE is higher than the familiar benchmark point estimate in "
                "all three held-out fonts."
            ),
            classification=blur_classification,
            evidence=blur_evidence,
            limitations=blur_limitations,
            primary_metric="heldout_blur_noise_rmse_above_familiar_count",
            primary_value=float(len(above)),
            supporting_count=len(above),
            scope="original_blur_noise",
            rule=config.analysis.blur_noise_consistency_rule,
            rule_met=all_above,
        )
    )
    blur_ratio = max(blur_font_rmse.values()) / min(blur_font_rmse.values())
    homogeneity_classification = "unsupported" if blur_ratio >= 2.0 else "suggestive"
    claims.append(
        _claim_row(
            claim_id="C03",
            claim=(
                "The magnitude of blur/noise degradation is similar across the three "
                "held-out fonts."
            ),
            classification=homogeneity_classification,
            evidence=(
                "Held-out blur/noise RMSE ranges from "
                f"{min(blur_font_rmse.values()):.4f} to {max(blur_font_rmse.values()):.4f} "
                f"across the three fonts (max/min ratio {blur_ratio:.3f})."
            ),
            limitations=(
                "Three purposive fonts and unmatched sources cannot establish homogeneity "
                "or a font-population distribution. A ratio below 2 is not evidence of "
                "equivalence."
            ),
            primary_metric="blur_noise_rmse_max_min_ratio",
            primary_value=blur_ratio,
            supporting_count=0,
            scope="original_blur_noise",
            rule="descriptive_homogeneity_ratio_lt_2",
            rule_met=blur_ratio < 2.0,
        )
    )

    severity_primary = severity_metrics.loc[
        (severity_metrics["font_id"] == "all") & (severity_metrics["protocol_role"] == "primary")
    ].sort_values("amplitude_px")
    severity_rmse = {
        float(row["amplitude_px"]): float(row["rmse"]) for _, row in severity_primary.iterrows()
    }
    severity_ratio = max(severity_rmse.values()) / min(severity_rmse.values())
    severity_classification = "supported" if severity_ratio >= 1.5 else "unsupported"
    severity_evidence = (
        "All-font primary geometric RMSE by amplitude: "
        + ", ".join(
            f"{amplitude:g}px={severity_rmse[amplitude]:.4f}" for amplitude in sorted(severity_rmse)
        )
        + f". The max/min severity RMSE ratio is {severity_ratio:.3f}; the aggregate "
        "geometric RMSE is a single summary of these levels."
    )
    claims.append(
        _claim_row(
            claim_id="C04",
            claim=("Aggregate geometric performance conceals severity-specific differences."),
            classification=severity_classification,
            evidence=severity_evidence,
            limitations=(
                "Severity slices are nested and correlated within source. The comparison is "
                "descriptive and does not identify a monotonic causal dose response."
            ),
            primary_metric="geometric_severity_rmse_max_min_ratio",
            primary_value=severity_ratio,
            supporting_count=len(severity_rmse),
            scope="geometric_primary",
            rule=config.analysis.geometric_concealment_rule,
            rule_met=severity_ratio >= 1.5,
        )
    )

    concentration = error_concentration.loc[
        (error_concentration["scope"] == "all_primary") & (error_concentration["font_id"] == "all")
    ].iloc[0]
    top_k_share = float(concentration["top_k_share"])
    effective_fraction = float(concentration["effective_fraction"])
    n_sources = int(concentration["n_sources"])
    top_k = int(concentration["top_k"])
    concentration_rule_met = bool(top_k_share >= 0.25 and effective_fraction <= 0.50)
    uniform_top_k_share = top_k / n_sources if n_sources else math.nan
    if concentration_rule_met:
        concentration_classification = "supported"
    elif top_k_share > 2.0 * uniform_top_k_share:
        concentration_classification = "suggestive"
    else:
        concentration_classification = "unsupported"
    claims.append(
        _claim_row(
            claim_id="C05",
            claim="A small number of source groups dominate the all-primary error.",
            classification=concentration_classification,
            evidence=(
                f"All-primary top-{top_k} source share is {top_k_share:.3f}; "
                f"effective source count is {float(concentration['effective_sources']):.2f} "
                f"of {n_sources} ({effective_fraction:.3f}); "
                f"{int(concentration['sources_for_50pct'])} sources account for 50% of SSE. "
                "The uniform top-k share is "
                f"{uniform_top_k_share:.3f}."
            ),
            limitations=(
                "Concentration is a descriptive influence summary. No source is excluded, "
                "no model is refit, and source-level shares are not a significance test."
            ),
            primary_metric="all_primary_top_k_sse_share",
            primary_value=top_k_share,
            supporting_count=(
                int(concentration["sources_for_50pct"])
                if concentration_classification == "supported"
                else 0
            ),
            scope="all_primary",
            rule=config.analysis.concentration_rule,
            rule_met=concentration_rule_met,
        )
    )

    unsupported_claims = (
        (
            "C06",
            "The frozen champion generalizes to arbitrary unseen fonts.",
            "The experiment contains three held-out fonts (Georgia, Monaco, Impact) with "
            "20 source groups each. These are a purposive sample, not a sample of font "
            "identities, and the source-grouped intervals do not estimate font-population "
            "variation.",
            "A font-population claim would require a designed sample of font identities and "
            "renderers, not three purposively selected typefaces.",
            "font_population",
        ),
        (
            "C07",
            "The frozen champion generalizes to unseen renderers, resolutions or formats.",
            "M7B holds the renderer, canvas, grayscale/PNG pipeline and corruption "
            "definitions fixed. Only the typeface changes; no renderer, resolution or "
            "format contrast is present in the frozen predictions.",
            "No variation in renderer, resolution or format was generated or evaluated.",
            "rendering_conditions",
        ),
        (
            "C08",
            "The results predict human readability or CAPTCHA solve rates.",
            "The target is synthetic SSIM against an exact source-specific clean reference. "
            "No human readability rating, response time or solve outcome is present in the "
            "frozen artifacts.",
            "SSIM is not a validated proxy for human readability or solve difficulty.",
            "human_outcomes",
        ),
        (
            "C09",
            "The results establish CAPTCHA security or production robustness.",
            "The experiment contains no attack, solver, deployment or security outcome. It "
            "measures image similarity to a clean reference under synthetic corruptions.",
            "Security and production robustness require adversarial and deployment evidence "
            "that this benchmark does not contain.",
            "security",
        ),
        (
            "C10",
            "The familiar-versus-held-out difference is caused by changing the font.",
            "The familiar M5C and held-out M7B benchmarks differ in fonts, source counts and "
            "composition, and partially in corruption mix and target distributions. The "
            "benchmark comparison is explicitly unpaired and descriptive.",
            "No source is matched across benchmarks, so the difference cannot be assigned "
            "to the font condition.",
            "familiar_vs_heldout",
        ),
        (
            "C11",
            "Close aggregate geometric RMSE point estimates establish geometric equivalence.",
            "The aggregate geometric RMSE point estimates are close, but severity-specific "
            "RMSEs differ and the benchmarks are unmatched. No equivalence margin or matched "
            "design was prespecified.",
            "Similar point estimates are not an equivalence test; aggregate similarity can "
            "conceal severity-specific and source-composition differences.",
            "geometric_equivalence",
        ),
        (
            "C12",
            "The 0.5 px elastic rung supports stable R² or model-ranking claims.",
            "The 0.5 px diagnostic has near-degenerate target variance and its R² is "
            "suppressed by the frozen near-degenerate rule. It is excluded from the primary "
            "geometric scopes.",
            "The diagnostic rung is retained for continuity with M5A, not for model "
            "selection, stable R² interpretation or ranking.",
            "diagnostic_0_5px",
        ),
        (
            "C13",
            "Per-font differences generalize to the population of all fonts.",
            "Each font has 20 independent source groups, but there are only three font "
            "identities. Source-level resampling estimates uncertainty for the observed "
            "sources, not variability across font identities.",
            "Three fonts cannot characterize a font population; no random font sample or "
            "hierarchical font model is available.",
            "font_population",
        ),
    )
    for claim_id, claim, evidence, limitations, scope in unsupported_claims:
        claims.append(
            _claim_row(
                claim_id=claim_id,
                claim=claim,
                classification="unsupported",
                evidence=evidence,
                limitations=limitations,
                primary_metric="design_limitation",
                primary_value=math.nan,
                supporting_count=0,
                scope=scope,
                rule="not_supported_by_frozen_design",
                rule_met=False,
            )
        )
    claim_frame = pd.DataFrame(claims)
    if tuple(claim_frame["claim_id"]) != CLAIM_IDS:
        raise CrossFontError("claim table does not contain the predeclared claim ids")
    return claim_frame


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    font_metrics: pd.DataFrame
    family_metrics: pd.DataFrame
    severity_metrics: pd.DataFrame
    source_metrics: pd.DataFrame
    error_concentration: pd.DataFrame
    uncertainty: pd.DataFrame
    paired_differences: pd.DataFrame
    font_differences: pd.DataFrame
    benchmark_comparison: pd.DataFrame
    claim_evidence: pd.DataFrame
    protocol: dict[str, Any]


def build_protocol(config: AnalysisConfig) -> dict[str, Any]:
    """Return the predeclared, machine-readable M7C protocol."""
    return {
        "analysis": {
            "name": config.run.name,
            "version": config.run.version,
            "config_sha256": config.config_sha256,
            "config_hash": config.config_hash,
        },
        "independent_unit": config.analysis.independent_unit,
        "group_column": config.analysis.group_column,
        "bootstrap": {
            "method": config.analysis.bootstrap_method,
            "seed": config.run.seed,
            "n_bootstrap": config.run.n_bootstrap,
            "confidence_level": config.run.confidence_level,
            "interval": "percentile",
            "paired_unit": "source_id",
        },
        "near_degenerate_sd": config.analysis.near_degenerate_sd,
        "top_k_sources": config.analysis.top_k_sources,
        "primary_scopes": list(config.analysis.primary_scopes),
        "diagnostic_scopes": list(config.analysis.diagnostic_scopes),
        "font_ids": list(config.analysis.font_ids),
        "family_order": list(config.analysis.family_order),
        "geometric_amplitudes_primary_px": list(config.analysis.geometric_amplitudes_primary_px),
        "geometric_amplitude_diagnostic_px": config.analysis.geometric_amplitude_diagnostic_px,
        "paired_contrasts": list(config.analysis.paired_contrasts),
        "font_difference_scopes": list(config.analysis.font_difference_scopes),
        "familiar_comparison_scopes": list(config.analysis.familiar_comparison_scopes),
        "claim_ids": list(CLAIM_IDS),
        "claim_rules": {
            "font_difference": config.analysis.font_difference_rule,
            "blur_noise_consistency": config.analysis.blur_noise_consistency_rule,
            "geometric_concealment": config.analysis.geometric_concealment_rule,
            "concentration": config.analysis.concentration_rule,
        },
        "interpretation": (
            "Descriptive uncertainty summaries for the observed synthetic source "
            "distribution. No significance testing, no causal font effect, no font "
            "population estimate, and no claim about human readability, solve rates, "
            "security or production robustness."
        ),
    }


def analyze(inputs: AnalysisInputs) -> AnalysisResult:
    """Build every deterministic M7C table from the frozen inputs."""
    family_metrics = build_family_metrics(inputs)
    font_metrics = build_font_metrics(family_metrics)
    severity_metrics = build_severity_metrics(inputs)
    source_metrics = build_source_metrics(inputs)
    error_concentration = build_error_concentration(source_metrics, inputs.config)
    uncertainty = build_uncertainty(inputs)
    paired_differences = build_paired_differences(inputs)
    font_differences = build_font_differences(inputs)
    benchmark_comparison = build_benchmark_comparison(inputs)
    claim_evidence = build_claim_evidence(
        inputs,
        family_metrics,
        severity_metrics,
        error_concentration,
        font_differences,
        benchmark_comparison,
    )
    return AnalysisResult(
        font_metrics=font_metrics,
        family_metrics=family_metrics,
        severity_metrics=severity_metrics,
        source_metrics=source_metrics,
        error_concentration=error_concentration,
        uncertainty=uncertainty,
        paired_differences=paired_differences,
        font_differences=font_differences,
        benchmark_comparison=benchmark_comparison,
        claim_evidence=claim_evidence,
        protocol=build_protocol(inputs.config),
    )


def _format_cell(value: object, digits: int = 6) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        number = float(value)
        if not math.isfinite(number):
            return "NA"
        return f"{number:.{digits}f}"
    text = str(value)
    if not text:
        return "NA"
    return text.replace("|", "\\|")


def _md_table(frame: pd.DataFrame, columns: Sequence[str], *, digits: int = 6) -> str:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise CrossFontError(f"report table is missing columns: {missing}")
    if frame.empty:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, separator]
    for _, row in frame.iterrows():
        lines.append(
            "| " + " | ".join(_format_cell(row[column], digits=digits) for column in columns) + " |"
        )
    return "\n".join(lines)


def _font_summary_table(family_metrics: pd.DataFrame) -> pd.DataFrame:
    scopes = (
        "all_primary",
        "original_blur_noise",
        "interference",
        "geometric_primary",
        "clean_reference",
        "diagnostic_0_5px",
    )
    subset = family_metrics.loc[
        (family_metrics["font_id"] != "all") & (family_metrics["scope"].isin(scopes))
    ]
    rmse = subset.pivot(index="font_id", columns="scope", values="rmse")
    mae = subset.pivot(index="font_id", columns="scope", values="mae")
    display = pd.DataFrame({"font_id": rmse.index})
    for scope in scopes:
        display[f"{scope}_mae"] = mae.loc[rmse.index, scope].to_numpy()
        display[f"{scope}_rmse"] = rmse[scope].to_numpy()
    return display


def build_report(inputs: AnalysisInputs, result: AnalysisResult) -> str:
    config = inputs.config
    claim_counts = result.claim_evidence["classification"].value_counts().to_dict()
    all_primary = _metric_row(result.family_metrics, scope="all_primary", font_id="all")
    font_rows = result.family_metrics.loc[
        (result.family_metrics["scope"] == "all_primary")
        & (result.family_metrics["font_id"] != "all")
    ]
    font_rmse = {str(row["font_id"]): float(row["rmse"]) for _, row in font_rows.iterrows()}
    blur_rows = result.family_metrics.loc[
        (result.family_metrics["scope"] == "original_blur_noise")
        & (result.family_metrics["font_id"] != "all")
    ]
    blur_rmse = {str(row["font_id"]): float(row["rmse"]) for _, row in blur_rows.iterrows()}
    familiar_blur = result.benchmark_comparison.loc[
        (result.benchmark_comparison["m7b_scope"] == "original_blur_noise")
        & (result.benchmark_comparison["metric"] == "rmse")
    ].iloc[0]
    geometric_rows = result.family_metrics.loc[
        (
            result.family_metrics["scope"].isin(
                ["geometric_primary_1px", "geometric_primary_2px", "geometric_primary_3px"]
            )
        )
        & (result.family_metrics["font_id"] == "all")
    ].sort_values("scope")
    geometric_rmse = {str(row["scope"]): float(row["rmse"]) for _, row in geometric_rows.iterrows()}
    concentration = result.error_concentration.loc[
        (result.error_concentration["scope"] == "all_primary")
        & (result.error_concentration["font_id"] == "all")
    ].iloc[0]
    diagnostic = _metric_row(result.family_metrics, scope="diagnostic_0_5px", font_id="all")

    input_display = inputs.input_manifest.copy()
    input_display["sha256"] = input_display["sha256"].str.slice(0, 16) + "…"
    data_display = result.family_metrics.loc[result.family_metrics["font_id"] == "all"][
        [
            "scope",
            "protocol_role",
            "n_rows",
            "n_sources",
            "n_distinct_images",
            "target_mean",
            "target_sd",
            "mae",
            "rmse",
            "r2",
            "near_degenerate",
        ]
    ]
    family_display = result.family_metrics.loc[
        (result.family_metrics["font_id"] != "all")
        & (
            result.family_metrics["scope"].isin(
                [
                    "original_blur_noise",
                    "gaussian_blur",
                    "gaussian_noise",
                    "interference",
                    "thin_lines",
                    "arc_strokes",
                    "geometric_primary",
                ]
            )
        )
    ][
        [
            "font_id",
            "scope",
            "n_rows",
            "n_sources",
            "target_mean",
            "target_sd",
            "mae",
            "rmse",
            "mean_residual",
            "r2",
            "r2_suppressed",
        ]
    ]
    severity_display = result.severity_metrics[
        [
            "amplitude_px",
            "protocol_role",
            "font_id",
            "severity_level",
            "n_rows",
            "n_sources",
            "target_mean",
            "target_sd",
            "mae",
            "rmse",
            "mean_residual",
            "r2",
            "r2_suppressed",
            "aggregate_target_order_consistent",
            "n_sources_monotonic",
            "n_sources_primary",
        ]
    ]
    concentration_display = result.error_concentration.loc[
        (result.error_concentration["scope"] == "all_primary")
        | (
            (result.error_concentration["font_id"] == "all")
            & (
                result.error_concentration["scope"].isin(
                    ["original_blur_noise", "geometric_primary"]
                )
            )
        )
    ][
        [
            "scope",
            "font_id",
            "n_sources",
            "n_rows",
            "top_k_share",
            "top1_share",
            "effective_sources",
            "effective_fraction",
            "sources_for_50pct",
            "sources_for_80pct",
            "top1_source",
        ]
    ]
    uncertainty_display = result.uncertainty.loc[
        (result.uncertainty["font_id"] != "all")
        & (
            result.uncertainty["scope"].isin(
                ["all_primary", "original_blur_noise", "geometric_primary"]
            )
        )
    ][
        [
            "scope",
            "font_id",
            "metric",
            "estimate",
            "ci_low",
            "ci_high",
            "n_groups",
            "n_samples",
        ]
    ]
    paired_display = result.paired_differences.loc[result.paired_differences["font_id"] == "all"][
        [
            "contrast",
            "metric",
            "estimate_a",
            "estimate_b",
            "difference",
            "ci_low",
            "ci_high",
            "n_groups",
            "eligible",
        ]
    ]
    font_difference_display = result.font_differences.loc[
        (result.font_differences["scope"] == "all_primary")
        & (result.font_differences["metric"] == "rmse")
    ][
        [
            "font_a",
            "font_b",
            "estimate_a",
            "estimate_b",
            "difference",
            "ci_low",
            "ci_high",
            "n_groups_a",
            "n_groups_b",
            "eligible",
        ]
    ]
    benchmark_display = result.benchmark_comparison[
        [
            "m7b_scope",
            "familiar_scope",
            "metric",
            "m7b_estimate",
            "familiar_estimate",
            "difference",
            "ci_low",
            "ci_high",
            "m7b_n_rows",
            "m7b_n_sources",
            "familiar_n_rows",
            "familiar_n_sources",
            "heldout_point_outside_familiar_ci",
        ]
    ]
    claim_display = result.claim_evidence[
        ["claim_id", "claim", "classification", "primary_metric", "primary_value"]
    ]

    lines: list[str] = []
    lines.append("# M7C — Cross-Font Evaluation Analysis")
    lines.append("")
    lines.append(
        "**Experiment:** EXP-0016  \n"
        "**Status:** Complete — read-only analysis of frozen M7B and M5C predictions.  \n"
        "**Model:** frozen M5C `expanded_random_forest`, calibration `none`.  \n"
        "**Scope:** synthetic SSIM against exact clean references. This report does not "
        "measure human readability, CAPTCHA solve rates, security or production robustness."
    )
    lines.append("")
    lines.append("## Executive summary")
    lines.append("")
    lines.append(
        f"The frozen champion has all-primary MAE {float(all_primary['mae']):.6f}, "
        f"RMSE {float(all_primary['rmse']):.6f} and R² {float(all_primary['r2']):.6f} "
        f"over {int(all_primary['n_rows'])} rows, "
        f"{int(all_primary['n_distinct_images'])} distinct images and "
        f"{int(all_primary['n_sources'])} independent source groups."
    )
    lines.append("")
    lines.append(
        "Per-font all-primary RMSE is "
        + ", ".join(f"{font} {font_rmse[font]:.6f}" for font in FONT_ORDER)
        + ". The font groups are unpaired and contain distinct sources, so these are "
        "differences on the observed source distribution, not a font-population effect."
    )
    lines.append("")
    lines.append(
        "Original blur/noise RMSE is "
        + ", ".join(f"{font} {blur_rmse[font]:.6f}" for font in FONT_ORDER)
        + f", compared with the familiar M5C point estimate "
        f"{float(familiar_blur['familiar_estimate']):.6f}. The comparison is unmatched "
        "and descriptive."
    )
    lines.append("")
    lines.append(
        "The aggregate primary geometric RMSE is "
        f"{_metric_row(result.family_metrics, scope='geometric_primary', font_id='all')['rmse']:.6f}, "
        "but the all-font severity RMSEs are "
        + ", ".join(
            f"{scope.replace('geometric_primary_', '')} {value:.6f}"
            for scope, value in geometric_rmse.items()
        )
        + ". The aggregate therefore does not describe a single severity-level error."
    )
    lines.append("")
    lines.append(
        f"All-primary top-{int(concentration['top_k'])} source share is "
        f"{float(concentration['top_k_share']):.3f}; the effective source count is "
        f"{float(concentration['effective_sources']):.2f} of "
        f"{int(concentration['n_sources'])}. Error concentration is descriptive only: "
        "no source is excluded and no model is refit."
    )
    lines.append("")
    lines.append(
        "Claim audit: "
        + ", ".join(
            f"{classification} {claim_counts.get(classification, 0)}"
            for classification in CLASSIFICATIONS
        )
        + ". Supported labels apply only within the observed synthetic benchmark; "
        "they do not establish arbitrary-font, human-outcome or security claims."
    )
    lines.append("")
    lines.append("## 1. Read-only protocol and pinned inputs")
    lines.append("")
    lines.append(
        f"Configuration: `{config.config_path}`  \n"
        f"Config SHA-256: `{config.config_sha256}`  \n"
        f"Config identity hash: `{config.config_hash}`  \n"
        f"Seed: `{config.run.seed}`; bootstrap replicates: `{config.run.n_bootstrap}`; "
        f"confidence level: `{config.run.confidence_level}`; independent unit: "
        f"`{config.analysis.independent_unit}`; interval: percentile.  \n"
        f"Inputs pinned and verified: `{len(config.inputs)}`."
    )
    lines.append("")
    lines.append(
        "M7C reads the frozen M7B predictions/targets and the frozen M5C familiar-font "
        "predictions. It does not generate a dataset, fit a model, recalibrate, select, "
        "tune or write into a prior dataset/model/report directory."
    )
    lines.append("")
    lines.append(
        "**Primary scopes:** "
        + ", ".join(f"`{scope}`" for scope in config.analysis.primary_scopes)
        + ". **Diagnostic scopes:** "
        + ", ".join(f"`{scope}`" for scope in config.analysis.diagnostic_scopes)
        + "."
    )
    lines.append("")
    lines.append(
        "**Paired contrasts** preserve source pairing where both sides contain the same "
        "source: "
        + ", ".join(f"`{contrast}`" for contrast in config.analysis.paired_contrasts)
        + ". **Font differences are unpaired** because each font has distinct source "
        "groups. **Familiar-versus-held-out differences are unpaired** because the "
        "benchmarks have different fonts, sources and corruption mixes."
    )
    lines.append("")
    lines.append(_md_table(input_display, ["key", "role", "sha256", "bytes"], digits=0))
    lines.append("")
    lines.append("## 2. Data and source accounting")
    lines.append("")
    lines.append(
        "M7B contains 60 held-out sources, 20 per font, and 1,260 rows (1,140 primary "
        "and 120 diagnostic). Each row has a distinct image hash. The primary scope "
        "combines original blur/noise, thin-line/arc interference and 1/2/3 px elastic "
        "geometric distortion. The clean reference and the 0.5 px elastic rung are "
        "diagnostic-only."
    )
    lines.append("")
    lines.append(
        _md_table(
            data_display,
            [
                "scope",
                "protocol_role",
                "n_rows",
                "n_sources",
                "n_distinct_images",
                "target_mean",
                "target_sd",
                "mae",
                "rmse",
                "r2",
                "near_degenerate",
            ],
        )
    )
    lines.append("")
    lines.append("## 3. Per-font performance")
    lines.append("")
    lines.append(
        "Residual convention: `residual = ssim_target - prediction`. A positive mean "
        "residual means the target SSIM is higher than the prediction (the model "
        "under-predicted image quality); a negative mean residual means it over-predicted "
        "quality. The target is SSIM to an exact clean reference, not a human rating."
    )
    lines.append("")
    lines.append(
        _md_table(
            _font_summary_table(result.family_metrics),
            [
                "font_id",
                "all_primary_mae",
                "all_primary_rmse",
                "original_blur_noise_rmse",
                "interference_rmse",
                "geometric_primary_rmse",
                "clean_reference_rmse",
                "diagnostic_0_5px_rmse",
            ],
        )
    )
    lines.append("")
    lines.append(
        "The table reports all-primary MAE/RMSE plus family-specific RMSE. Fonts are not "
        "ranked as universally easy or difficult: each font is a different set of 20 "
        "sources, and the target distribution differs by font and corruption family."
    )
    lines.append("")
    lines.append("## 4. Corruption-family analysis")
    lines.append("")
    lines.append(
        "The blur/noise family contains Gaussian blur and Gaussian noise variants; "
        "interference contains thin lines and arc strokes; geometric primary contains "
        "1, 2 and 3 px elastic warps. The clean reference is a diagnostic control."
    )
    lines.append("")
    lines.append(
        _md_table(
            family_display,
            [
                "font_id",
                "scope",
                "n_rows",
                "n_sources",
                "target_mean",
                "target_sd",
                "mae",
                "rmse",
                "mean_residual",
                "r2",
                "r2_suppressed",
            ],
        )
    )
    lines.append("")
    lines.append(
        "The familiar M5C original blur/noise RMSE is "
        f"{float(familiar_blur['familiar_estimate']):.6f}. All three held-out fonts have "
        "a higher point estimate, but the comparison is unmatched: familiar sources are "
        "Arial/Courier New/Times New Roman, held-out sources are Georgia/Monaco/Impact, "
        "and source counts and composition differ. This is not a causal font effect."
    )
    lines.append("")
    lines.append("## 5. Geometric severity")
    lines.append("")
    lines.append(
        "M7B stores elastic severity levels 2, 3 and 4 for 1, 2 and 3 px amplitudes; "
        "severity 1 is the 0.5 px diagnostic rung. The mapping is verified against the "
        "frozen M5A manifest. Target ordering is checked both in aggregate and per source."
    )
    lines.append("")
    lines.append(
        _md_table(
            severity_display,
            [
                "amplitude_px",
                "protocol_role",
                "font_id",
                "severity_level",
                "n_rows",
                "n_sources",
                "target_mean",
                "target_sd",
                "mae",
                "rmse",
                "mean_residual",
                "r2",
                "r2_suppressed",
                "aggregate_target_order_consistent",
                "n_sources_monotonic",
                "n_sources_primary",
            ],
        )
    )
    lines.append("")
    lines.append(
        f"The 0.5 px diagnostic has target SD {float(diagnostic['target_sd']):.6f}, "
        f"below the near-degenerate margin {config.analysis.near_degenerate_sd:.6f}. Its "
        "R² is suppressed and it is not used for model selection, stable R² claims or "
        "ranking."
    )
    lines.append("")
    lines.append("## 6. Source-level error concentration")
    lines.append("")
    lines.append(
        "Per-source MAE, RMSE, residual summaries and SSE shares are in "
        "`source_metrics.csv`; the table below shows the concentration summary. The "
        "number of independent sources is explicit in every row."
    )
    lines.append("")
    lines.append(
        _md_table(
            concentration_display,
            [
                "scope",
                "font_id",
                "n_sources",
                "n_rows",
                "top_k_share",
                "top1_share",
                "effective_sources",
                "effective_fraction",
                "sources_for_50pct",
                "sources_for_80pct",
                "top1_source",
            ],
        )
    )
    lines.append("")
    lines.append("## 7. Uncertainty and paired comparisons")
    lines.append("")
    lines.append(
        "Intervals resample whole source groups with replacement and retain all rows of "
        "a sampled source. They are descriptive percentile intervals, not significance "
        "tests. Paired contrasts use the same sampled sources on both sides; font "
        "differences use independent font groups and are unpaired."
    )
    lines.append("")
    lines.append(
        _md_table(
            uncertainty_display,
            [
                "scope",
                "font_id",
                "metric",
                "estimate",
                "ci_low",
                "ci_high",
                "n_groups",
                "n_samples",
            ],
        )
    )
    lines.append("")
    lines.append("**Paired within-source contrasts (all fonts pooled):**")
    lines.append("")
    lines.append(
        _md_table(
            paired_display,
            [
                "contrast",
                "metric",
                "estimate_a",
                "estimate_b",
                "difference",
                "ci_low",
                "ci_high",
                "n_groups",
                "eligible",
            ],
        )
    )
    lines.append("")
    lines.append("**Unpaired font differences for all-primary RMSE:**")
    lines.append("")
    lines.append(
        _md_table(
            font_difference_display,
            [
                "font_a",
                "font_b",
                "estimate_a",
                "estimate_b",
                "difference",
                "ci_low",
                "ci_high",
                "n_groups_a",
                "n_groups_b",
                "eligible",
            ],
        )
    )
    lines.append("")
    lines.append(
        "A difference interval that excludes zero is still a descriptive statement about "
        "the observed source distribution. It is not evidence that a font identity causes "
        "the difference or that the result generalizes to other fonts."
    )
    lines.append("")
    lines.append("## 8. Familiar versus held-out benchmark")
    lines.append("")
    lines.append(
        "The familiar M5C and held-out M7B metrics are reproduced from the frozen "
        "artifacts. They are not matched: fonts, source composition and counts, and "
        "partially corruption mix and target distributions differ. The comparison is "
        "descriptive, not causal and not an equivalence test."
    )
    lines.append("")
    lines.append(
        _md_table(
            benchmark_display,
            [
                "m7b_scope",
                "familiar_scope",
                "metric",
                "m7b_estimate",
                "familiar_estimate",
                "difference",
                "ci_low",
                "ci_high",
                "m7b_n_rows",
                "m7b_n_sources",
                "familiar_n_rows",
                "familiar_n_sources",
                "heldout_point_outside_familiar_ci",
            ],
        )
    )
    lines.append("")
    lines.append("## 9. Claim-to-evidence audit")
    lines.append("")
    lines.append(
        "Classifications are `supported`, `suggestive` and `unsupported`. A supported "
        "label is restricted to the observed synthetic benchmark and the predeclared "
        "evidence rule; it is not upgraded because a point estimate looks large."
    )
    lines.append("")
    lines.append(
        _md_table(
            claim_display,
            ["claim_id", "claim", "classification", "primary_metric", "primary_value"],
        )
    )
    lines.append("")
    lines.append("**Claim evidence and limitations:**")
    lines.append("")
    for _, row in result.claim_evidence.iterrows():
        lines.append(
            f"- **{row['claim_id']} — {row['classification']}:** {row['claim']} "
            f"Evidence: {row['evidence']} Limitations: {row['limitations']}"
        )
    lines.append("")
    lines.append("## 10. Limitations")
    lines.append("")
    limitations = (
        "Only three held-out font identities are evaluated; they are a purposive sample, "
        "not a random sample of fonts.",
        "Font groups contain distinct source groups and are unpaired. Source-level "
        "resampling does not estimate variation across font identities.",
        "The familiar and held-out benchmarks are unmatched in fonts, source composition "
        "and counts, and partially in corruption mix and target distributions.",
        "Intervals are descriptive uncertainty summaries. No significance test, causal "
        "identification or equivalence test is performed.",
        "The source groups are 60 synthetic sources; intervals with few groups can be "
        "wide and unstable.",
        "The target is SSIM to an exact synthetic clean reference. It is not human "
        "readability, solve rate, security or production robustness.",
        "Renderer, canvas, grayscale/PNG pipeline and corruption definitions are fixed; "
        "no renderer, resolution or format generalization is tested.",
        "Corruption variants are nested within source. Source aggregation avoids treating "
        "variants as independent, but it does not create new independent sources.",
        "The 0.5 px elastic rung has near-degenerate target variance and is diagnostic "
        "only; its R² is suppressed.",
    )
    for item in limitations:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 11. Reproduce and verify")
    lines.append("")
    lines.append("```bash")
    lines.append(
        "uv run --extra image --extra ml captcha-cross-font-analysis run "
        "--config configs/cross_font_analysis_v0_1_0.toml --overwrite"
    )
    lines.append(
        "uv run --extra image --extra ml captcha-cross-font-analysis verify "
        "--config configs/cross_font_analysis_v0_1_0.toml"
    )
    lines.append("```")
    lines.append("")
    lines.append(
        "All outputs are written only under the configured Git-ignored report directory. "
        "The verifier rechecks input hashes, protected assets, deterministic tables, "
        "report sections and the claim table."
    )
    lines.append("")
    lines.append("## 12. Research-integrity boundary")
    lines.append("")
    lines.append(
        "M7C analyzes synthetic image similarity. It does not measure human CAPTCHA "
        "readability, solve rates, security or production robustness. The results must "
        "not be presented as evidence for those outcomes, and the three tested fonts must "
        "not be presented as a representative sample of all fonts."
    )
    lines.append("")
    return "\n".join(lines)


def build_isolation_checks(
    inputs: AnalysisInputs,
    result: AnalysisResult,
    report: str,
) -> pd.DataFrame:
    checks: list[CheckResult] = list(inputs.checks)
    config = inputs.config

    _check(
        checks,
        "input_manifest_complete",
        lambda: (
            f"{len(inputs.input_manifest)} pinned inputs verified"
            if len(inputs.input_manifest) == len(config.inputs)
            and bool(inputs.input_manifest["verified"].all())
            else (_ for _ in ()).throw(CrossFontError("input manifest is incomplete"))
        ),
    )
    _check(
        checks,
        "residual_convention",
        lambda: (
            "residual = ssim_target - prediction"
            if np.allclose(
                inputs.m7b_predictions[TARGET] - inputs.m7b_predictions["prediction"],
                inputs.m7b_predictions["residual"],
                atol=1e-12,
                rtol=0.0,
            )
            else (_ for _ in ()).throw(CrossFontError("residual convention changed"))
        ),
    )
    expected_scope_rows = {
        "all_primary": 1140,
        "original_blur_noise": 480,
        "gaussian_blur": 240,
        "gaussian_noise": 240,
        "interference": 480,
        "thin_lines": 240,
        "arc_strokes": 240,
        "geometric_primary": 180,
        "geometric_primary_1px": 60,
        "geometric_primary_2px": 60,
        "geometric_primary_3px": 60,
        "clean_reference": 60,
        "diagnostic_0_5px": 60,
    }
    _check(
        checks,
        "scope_row_counts",
        lambda: (
            "all frozen scope row counts match"
            if all(
                len(_scope_frame(inputs.m7b_predictions, scope)) == expected
                for scope, expected in expected_scope_rows.items()
            )
            else (_ for _ in ()).throw(CrossFontError("scope row counts changed"))
        ),
    )
    _check(
        checks,
        "font_source_counts",
        lambda: (
            "20 sources per held-out font"
            if inputs.m7b_predictions.groupby("font_id")["source_id"].nunique().to_dict()
            == {font: 20 for font in FONT_ORDER}
            else (_ for _ in ()).throw(CrossFontError("font source counts changed"))
        ),
    )
    _check(
        checks,
        "source_sse_accounting",
        lambda: (
            "source SSE sums equal aggregate SSE for every scope"
            if all(
                math.isclose(
                    float(
                        result.source_metrics.loc[
                            result.source_metrics["scope"] == scope, "sse"
                        ].sum()
                    ),
                    float(
                        np.sum(
                            (
                                _scope_frame(inputs.m7b_predictions, scope)[TARGET]
                                - _scope_frame(inputs.m7b_predictions, scope)["prediction"]
                            )
                            ** 2
                        )
                    ),
                    rel_tol=1e-10,
                    abs_tol=1e-12,
                )
                for scope in SCOPE_ORDER
            )
            else (_ for _ in ()).throw(CrossFontError("source SSE accounting failed"))
        ),
    )
    _check(
        checks,
        "uncertainty_group_counts",
        lambda: (
            "bootstrap group counts equal source counts"
            if all(
                int(row["n_groups"])
                == int(
                    _metric_row(
                        result.family_metrics,
                        scope=str(row["scope"]),
                        font_id=str(row["font_id"]),
                    )["n_sources"]
                )
                for _, row in result.uncertainty.iterrows()
            )
            else (_ for _ in ()).throw(CrossFontError("bootstrap group counts differ"))
        ),
    )
    _check(
        checks,
        "paired_contrast_eligibility",
        lambda: (
            "every configured paired contrast has eligible source pairs"
            if all(
                bool(group["eligible"].any())
                for _, group in result.paired_differences.groupby("contrast")
            )
            else (_ for _ in ()).throw(CrossFontError("a paired contrast is ineligible"))
        ),
    )
    _check(
        checks,
        "severity_accounting",
        lambda: (
            "0.5/1/2/3 px severity slices are present with 20 or 60 sources"
            if set(result.severity_metrics["amplitude_px"]) == {0.5, 1.0, 2.0, 3.0}
            and all(
                int(row["n_sources"]) == (60 if row["font_id"] == "all" else 20)
                for _, row in result.severity_metrics.iterrows()
            )
            else (_ for _ in ()).throw(CrossFontError("severity accounting failed"))
        ),
    )
    _check(
        checks,
        "claim_table_completeness",
        lambda: (
            "13 claims with valid classifications and evidence"
            if tuple(result.claim_evidence["claim_id"]) == CLAIM_IDS
            and set(result.claim_evidence["classification"]).issubset(CLASSIFICATIONS)
            and bool(result.claim_evidence["evidence"].str.len().gt(0).all())
            and bool(result.claim_evidence["limitations"].str.len().gt(0).all())
            else (_ for _ in ()).throw(CrossFontError("claim table is incomplete"))
        ),
    )
    _check(
        checks,
        "benchmark_comparison_completeness",
        lambda: (
            "every configured familiar comparison scope has two metrics"
            if set(result.benchmark_comparison["m7b_scope"])
            == set(config.analysis.familiar_comparison_scopes)
            and set(result.benchmark_comparison["metric"]) == {"mae", "rmse"}
            and bool(result.benchmark_comparison["eligible"].all())
            else (_ for _ in ()).throw(CrossFontError("benchmark comparison is incomplete"))
        ),
    )
    _check(
        checks,
        "report_sections",
        lambda: (
            "all report sections present"
            if all(section in report for section in REPORT_SECTIONS)
            else (_ for _ in ()).throw(CrossFontError("report is missing a section"))
        ),
    )
    _check(
        checks,
        "report_claim_ids",
        lambda: (
            "all claim ids appear in the report"
            if all(claim_id in report for claim_id in CLAIM_IDS)
            else (_ for _ in ()).throw(CrossFontError("report is missing a claim id"))
        ),
    )
    return pd.DataFrame(
        [{"check": check.name, "passed": check.passed, "detail": check.detail} for check in checks]
    )


def _safe_output_path(path: Path | str) -> Path:
    resolved = Path(path).resolve()
    reports_root = REPORTS_ROOT.resolve()
    if resolved == reports_root or not resolved.is_relative_to(reports_root):
        raise CrossFontError(
            f"refusing to write outside the Git-ignored reports directory: {resolved}"
        )
    for protected_text in PROTECTED_REPORT_ROOTS:
        protected = Path(protected_text).resolve()
        if resolved == protected or resolved.is_relative_to(protected):
            raise CrossFontError(
                f"refusing to write into a protected frozen report path: {resolved}"
            )
    return resolved


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(json_safe(dict(payload)), indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")


def _fingerprint_tree(root: Path, *, skip_images: bool = True) -> dict[str, str]:
    root = root.resolve()
    if not root.exists():
        return {}
    repository = Path.cwd().resolve()
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(repository).as_posix()
        if skip_images and "/images/" in relative:
            continue
        result[relative] = sha256_file(path)
    return result


def _protected_fingerprint() -> dict[str, str]:
    fingerprint: dict[str, str] = {}
    roots = [Path("configs"), Path("data/processed"), Path("models")]
    roots.extend(Path(path) for path in PROTECTED_REPORT_ROOTS)
    for root in roots:
        fingerprint.update(_fingerprint_tree(root, skip_images=True))
    return fingerprint


def _fingerprint_hash(fingerprint: Mapping[str, str]) -> str:
    return sha256_bytes(canonical_json(fingerprint).encode("utf-8"))


def _output_hashes(output_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "run_metadata.json":
            hashes[path.name] = sha256_file(path)
    return hashes


def _config_with_output(config: AnalysisConfig, output: Path | None) -> AnalysisConfig:
    if output is None:
        return config
    run = config.run.__class__(
        name=config.run.name,
        version=config.run.version,
        seed=config.run.seed,
        n_bootstrap=config.run.n_bootstrap,
        confidence_level=config.run.confidence_level,
        output_reports=Path(output),
    )
    candidate = AnalysisConfig(
        run=run,
        analysis=config.analysis,
        inputs=config.inputs,
        config_path=config.config_path,
        config_sha256=config.config_sha256,
        config_hash="",
    )
    return AnalysisConfig(
        run=candidate.run,
        analysis=candidate.analysis,
        inputs=candidate.inputs,
        config_path=candidate.config_path,
        config_sha256=candidate.config_sha256,
        config_hash=sha256_bytes(canonical_json(candidate.identity()).encode("utf-8")),
    )


def _build_metadata(
    inputs: AnalysisInputs,
    result: AnalysisResult,
    *,
    output_hashes: Mapping[str, str],
    frozen_assets: Mapping[str, str],
    frozen_assets_hash: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    config = inputs.config
    all_primary = _metric_row(result.family_metrics, scope="all_primary", font_id="all")
    claim_counts = result.claim_evidence["classification"].value_counts().to_dict()
    font_source_counts = inputs.m7b_predictions.groupby("font_id")["source_id"].nunique().to_dict()
    return {
        "analysis": {
            "name": config.run.name,
            "version": config.run.version,
            "config_path": str(config.config_path),
            "config_sha256": config.config_sha256,
            "config_hash": config.config_hash,
            "code_revision": code_revision(),
            "package_version": package_version(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "model": {
            "model_family": inputs.model_card.get("model_family"),
            "calibration": inputs.model_card.get("calibration"),
            "artifact_sha256": inputs.model_card.get("artifact_sha256"),
            "feature_names": inputs.model_card.get("feature_names"),
        },
        "inputs": inputs.input_hashes,
        "input_manifest_sha256": output_hashes.get("input_manifest.csv"),
        "protocol_sha256": output_hashes.get("protocol.json"),
        "report_sha256": output_hashes.get("report.md"),
        "summary": {
            "rows": int(len(inputs.m7b_predictions)),
            "sources": int(inputs.m7b_predictions["source_id"].nunique()),
            "primary_rows": int((inputs.m7b_predictions["protocol_scope"] == "primary").sum()),
            "diagnostic_rows": int(
                (inputs.m7b_predictions["protocol_scope"] == "diagnostic").sum()
            ),
            "font_source_counts": font_source_counts,
            "all_primary": {
                "mae": float(all_primary["mae"]),
                "rmse": float(all_primary["rmse"]),
                "r2": float(all_primary["r2"]),
                "n_rows": int(all_primary["n_rows"]),
                "n_sources": int(all_primary["n_sources"]),
                "n_distinct_images": int(all_primary["n_distinct_images"]),
            },
            "claims": {
                classification: int(claim_counts.get(classification, 0))
                for classification in CLASSIFICATIONS
            },
        },
        "frozen_assets": dict(frozen_assets),
        "frozen_assets_hash": frozen_assets_hash,
        "output_hashes": dict(output_hashes),
        "elapsed_seconds": round(float(elapsed_seconds), 3),
    }


def run_analysis(
    config_path: Path | str,
    *,
    output_reports: Path | str | None = None,
    overwrite: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Run M7C and write deterministic outputs to a Git-ignored report directory."""
    started = time.perf_counter()
    config = _config_with_output(load_config(config_path), output_reports)
    output_dir = _safe_output_path(config.run.output_reports)
    if output_dir.exists() and not overwrite:
        raise CrossFontError(
            f"output directory already exists: {output_dir}; pass --overwrite to replace it"
        )
    frozen_before = _protected_fingerprint()
    inputs = load_inputs(config)
    result = analyze(inputs)
    report = build_report(inputs, result)
    checks = build_isolation_checks(inputs, result, report)
    _require_all_checks(
        [
            CheckResult(str(row["check"]), bool(row["passed"]), str(row["detail"]))
            for _, row in checks.iterrows()
        ],
        "M7C analysis checks",
    )
    if output_dir.exists():
        if not overwrite:
            raise CrossFontError(f"output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(inputs.input_manifest, output_dir / "input_manifest.csv")
    _write_csv(result.font_metrics, output_dir / "font_metrics.csv")
    _write_csv(result.family_metrics, output_dir / "family_metrics.csv")
    _write_csv(result.severity_metrics, output_dir / "severity_metrics.csv")
    _write_csv(result.source_metrics, output_dir / "source_metrics.csv")
    _write_csv(result.error_concentration, output_dir / "error_concentration.csv")
    _write_csv(result.uncertainty, output_dir / "uncertainty.csv")
    _write_csv(result.paired_differences, output_dir / "paired_differences.csv")
    _write_csv(result.font_differences, output_dir / "font_differences.csv")
    _write_csv(result.benchmark_comparison, output_dir / "benchmark_comparison.csv")
    _write_csv(result.claim_evidence, output_dir / "claim_evidence.csv")
    _write_csv(checks, output_dir / "isolation_checks.csv")
    _write_json(result.protocol, output_dir / "protocol.json")
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    output_hashes = _output_hashes(output_dir)
    frozen_after = _protected_fingerprint()
    if frozen_before != frozen_after:
        raise CrossFontError("protected frozen assets changed during the M7C run")
    metadata = _build_metadata(
        inputs,
        result,
        output_hashes=output_hashes,
        frozen_assets=frozen_after,
        frozen_assets_hash=_fingerprint_hash(frozen_after),
        elapsed_seconds=time.perf_counter() - started,
    )
    _write_json(metadata, output_dir / "run_metadata.json")
    return output_dir, metadata


def _compare_table(
    checks: list[CheckResult],
    name: str,
    expected: pd.DataFrame,
    path: Path,
) -> None:
    try:
        actual = pd.read_csv(path)
        expected = expected.copy()
        for column in expected.columns:
            if pd.api.types.is_object_dtype(expected[column]) or isinstance(
                expected[column].dtype, pd.StringDtype
            ):
                expected[column] = expected[column].fillna("")
                if column in actual.columns:
                    actual[column] = actual[column].fillna("")
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=1e-9,
            atol=1e-9,
        )
    except Exception as error:  # noqa: BLE001 - report every replay mismatch
        checks.append(CheckResult(f"table:{name}", False, f"{type(error).__name__}: {error}"))
    else:
        checks.append(CheckResult(f"table:{name}", True, "replayed table matches"))


def verify_analysis(
    config_path: Path | str,
    *,
    output_reports: Path | str | None = None,
) -> VerificationReport:
    """Replay M7C from frozen inputs and verify a written report directory."""
    config = _config_with_output(load_config(config_path), output_reports)
    output_dir = _safe_output_path(config.run.output_reports)
    checks: list[CheckResult] = []
    if not output_dir.is_dir():
        return VerificationReport(
            dataset=str(output_dir),
            checks=(CheckResult("output_directory", False, f"missing: {output_dir}"),),
        )
    inputs = load_inputs(config)
    result = analyze(inputs)
    report = build_report(inputs, result)
    isolation = build_isolation_checks(inputs, result, report)
    expected_tables = {
        "input_manifest.csv": inputs.input_manifest,
        "font_metrics.csv": result.font_metrics,
        "family_metrics.csv": result.family_metrics,
        "severity_metrics.csv": result.severity_metrics,
        "source_metrics.csv": result.source_metrics,
        "error_concentration.csv": result.error_concentration,
        "uncertainty.csv": result.uncertainty,
        "paired_differences.csv": result.paired_differences,
        "font_differences.csv": result.font_differences,
        "benchmark_comparison.csv": result.benchmark_comparison,
        "claim_evidence.csv": result.claim_evidence,
        "isolation_checks.csv": isolation,
    }
    for name, expected in expected_tables.items():
        _compare_table(checks, name, expected, output_dir / name)
    checks.append(
        CheckResult(
            "isolation_checks",
            bool(isolation["passed"].all()),
            "all recomputed isolation checks passed",
        )
    )
    try:
        actual_protocol = json.loads((output_dir / "protocol.json").read_text(encoding="utf-8"))
        if actual_protocol != json_safe(result.protocol):
            raise CrossFontError("protocol.json does not match the recomputed protocol")
    except Exception as error:  # noqa: BLE001
        checks.append(CheckResult("protocol", False, f"{type(error).__name__}: {error}"))
    else:
        checks.append(CheckResult("protocol", True, "recomputed protocol matches"))
    try:
        actual_report = (output_dir / "report.md").read_text(encoding="utf-8")
        if actual_report != report:
            raise CrossFontError("report.md does not match the recomputed report")
    except Exception as error:  # noqa: BLE001
        checks.append(CheckResult("report", False, f"{type(error).__name__}: {error}"))
    else:
        checks.append(CheckResult("report", True, "recomputed report matches"))
    metadata_path = output_dir / "run_metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001
        checks.append(CheckResult("metadata", False, f"{type(error).__name__}: {error}"))
        return VerificationReport(dataset=str(output_dir), checks=tuple(checks))
    checks.append(CheckResult("metadata", True, "run_metadata.json is readable"))
    checks.append(
        CheckResult(
            "config_sha256",
            metadata.get("analysis", {}).get("config_sha256") == config.config_sha256,
            "recorded config SHA-256 matches",
        )
    )
    checks.append(
        CheckResult(
            "config_hash",
            metadata.get("analysis", {}).get("config_hash") == config.config_hash,
            "recorded config identity hash matches",
        )
    )
    checks.append(
        CheckResult(
            "input_hashes",
            metadata.get("inputs") == inputs.input_hashes,
            "recorded input hashes match",
        )
    )
    current_output_hashes = _output_hashes(output_dir)
    checks.append(
        CheckResult(
            "output_hashes",
            metadata.get("output_hashes") == current_output_hashes,
            "recorded output hashes match",
        )
    )
    current_frozen = _protected_fingerprint()
    checks.append(
        CheckResult(
            "frozen_assets",
            metadata.get("frozen_assets") == current_frozen
            and metadata.get("frozen_assets_hash") == _fingerprint_hash(current_frozen),
            "protected frozen assets are unchanged",
        )
    )
    return VerificationReport(dataset=str(output_dir), checks=tuple(checks))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="captcha-cross-font-analysis",
        description="Read-only M7C cross-font evaluation analysis.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run the frozen cross-font analysis")
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--output-reports", type=Path)
    run_parser.add_argument("--overwrite", action="store_true")
    verify_parser = subparsers.add_parser(
        "verify", help="replay and verify a written cross-font analysis"
    )
    verify_parser.add_argument("--config", required=True, type=Path)
    verify_parser.add_argument("--output-reports", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            output_dir, metadata = run_analysis(
                args.config,
                output_reports=args.output_reports,
                overwrite=args.overwrite,
            )
            print(f"wrote M7C analysis to {output_dir}")
            print(json.dumps(json_safe(metadata["summary"]), indent=2, sort_keys=True))
            return 0
        report = verify_analysis(
            args.config,
            output_reports=args.output_reports,
        )
        for check in report.checks:
            status = "PASS" if check.passed else "FAIL"
            print(f"[{status}] {check.name}: {check.detail}")
        print(f"verification: {'PASS' if report.ok else 'FAIL'}")
        return 0 if report.ok else 1
    except (CrossFontError, CrossFontConfigError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
