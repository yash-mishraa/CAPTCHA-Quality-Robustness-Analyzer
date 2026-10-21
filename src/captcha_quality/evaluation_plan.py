"""Milestone M7D evaluation-design and dataset-expansion plan.

M7D is a read-only research-planning milestone. It selects one next benchmark,
pre-registers its design and analysis, and stops before any dataset generation,
image rendering, model fitting, recalibration, prediction change or evaluation.
It reads only frozen M3/M5C/M7A/M7B/M7C artifacts, verifies their hashes and
selected numerical claims, and writes a deterministic plan to a Git-ignored
report directory.

The selected primary direction is A, a broader unseen-font benchmark, with the
matched text-case control from direction C incorporated into the design. The
ten typefaces are a purposive roster, not a random sample of a font population.
The independent units are distinguished explicitly: font identity, matched text
case, source, clean reference, corrupted variant, renderer, benchmark row and
source group. The target remains synthetic SSIM against an exact clean
reference; it is not human readability, solve rate, security or production
robustness.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import sys
import time
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from PIL import ImageFont

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
from captcha_quality.render import DEFAULT_CHARSET

#: Directions compared by this milestone.
DIRECTION_ORDER: Final[tuple[str, ...]] = ("A", "B", "C")
DIRECTION_TITLES: Final[dict[str, str]] = {
    "A": "Broader unseen-font generalization with matched text cases",
    "B": "Unseen-renderer generalization",
    "C": "Matched font-transfer benchmark",
}

#: The frozen M7B corruption protocol, reproduced exactly.
EXPECTED_CORRUPTIONS: Final[dict[str, tuple[tuple[str, tuple[float, ...]], ...]]] = {
    "clean": (("diagnostic", (0.0,)),),
    "gaussian_blur": (("primary", (1.0, 2.0, 3.5, 5.0)),),
    "gaussian_noise": (("primary", (6.0, 12.0, 20.0, 32.0)),),
    "thin_lines": (("primary", (1.0, 2.0, 4.0, 7.0)),),
    "arc_strokes": (("primary", (1.0, 2.0, 4.0, 7.0)),),
    "elastic": (("primary", (1.0, 2.0, 3.0)), ("diagnostic", (0.5,))),
}

FONT_ROLES: Final[tuple[str, ...]] = ("familiar", "held-out")
FONT_STRATA: Final[tuple[str, ...]] = (
    "serif",
    "monospace",
    "grotesque_sans",
    "humanist_sans",
    "display_sans",
    "display_script",
)

REQUIRED_INPUT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "m3_config",
        "m3_summary",
        "m3_sources",
        "m3_manifest",
        "m5c_config",
        "m5c_model_card",
        "m5c_model_artifact",
        "m5c_test_predictions",
        "m5c_test_slice_metrics",
        "m5c_run_metadata",
        "m5c_report",
        "m7a_config",
        "m7a_next_direction_comparison",
        "m7a_claim_evidence",
        "m7a_run_metadata",
        "m7b_config",
        "m7b_summary",
        "m7b_sources",
        "m7b_manifest",
        "m7b_predictions",
        "m7b_overall_metrics",
        "m7b_font_metrics",
        "m7b_family_metrics",
        "m7b_uncertainty",
        "m7b_comparison",
        "m7b_run_metadata",
        "m7b_report",
        "m7c_config",
        "m7c_font_metrics",
        "m7c_family_metrics",
        "m7c_severity_metrics",
        "m7c_source_metrics",
        "m7c_error_concentration",
        "m7c_uncertainty",
        "m7c_benchmark_comparison",
        "m7c_claim_evidence",
        "m7c_run_metadata",
        "m7c_report",
    }
)
OUTPUT_TABLE_NAMES: Final[tuple[str, ...]] = (
    "input_manifest.csv",
    "font_inventory.csv",
    "direction_assessment.csv",
    "research_questions.csv",
    "independent_units.csv",
    "split_hierarchy.csv",
    "endpoints.csv",
    "sample_size_scenarios.csv",
    "leakage_controls.csv",
    "success_criteria.csv",
    "stopping_rules.csv",
    "claims_and_limits.csv",
    "evidence_snapshot.csv",
    "isolation_checks.csv",
)
REPORT_SECTIONS: Final[tuple[str, ...]] = (
    "## Executive summary",
    "## 1. Status and research-integrity boundary",
    "## 2. Preflight and pinned evidence",
    "## 3. Evidence from M7A-M7C",
    "## 4. Direction comparison and decision",
    "## 5. Research questions and independent units",
    "## 6. Split hierarchy and leakage controls",
    "## 7. Proposed dataset expansion",
    "## 8. Font selection and verification",
    "## 9. Pre-registered endpoints and uncertainty",
    "## 10. Sample-size and precision rationale",
    "## 11. Frozen model and baseline policy",
    "## 12. Success criteria, stopping rules and stopping interpretation",
    "## 13. Valid interpretation of null, mixed and negative findings",
    "## 14. What the plan cannot establish",
    "## 15. Implementation handoff and verification",
    "## 16. Non-execution statement",
)
REPORTS_ROOT: Final[Path] = Path("reports")
PROTECTED_REPORT_ROOTS: Final[tuple[str, ...]] = (
    "reports/benchmark_v0_1_0",
    "reports/captcha_quality_v0.1.0",
    "reports/captcha_quality_v0.1.0-pilot",
    "reports/cross_font_analysis_v0_1_0",
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

# ---------------------------------------------------------------------------
# Scoped protected-dependency membership
# ---------------------------------------------------------------------------
# The original M7D contract fingerprinted three *shared, growing* trees
# (``configs/``, ``data/processed/`` and ``models/``) plus the historical
# milestone report roots. That made M7D verification depend on the complete
# future contents of directories M7D never declared as inputs: adding an
# unrelated milestone configuration (for example the configuration of a later
# milestone) changed the tree fingerprint and failed M7D verification even
# though every M7D input, table, manifest and report artifact was
# byte-identical. The same rule also covered transient, Git-ignored scratch
# copies that are generated outputs rather than frozen dependencies.
#
# M7D's actual dependencies are a *file* set, not a tree: every upstream
# dependency is individually hash-pinned in the M7D planning configuration,
# which is itself committed to Git and whose SHA-256 is recorded. The contract
# below therefore declares members at file level, which is the role-appropriate
# granularity for this milestone:
#
#   * ``config_files``  - the M7D planning configuration itself. Its expected
#     hash is the SHA-256 recorded when M7D ran, cross-checked against the
#     recorded asset snapshot and against the recomputed file.
#   * ``input_files``   - the exact upstream evidence files M7D pins (the M3,
#     M5C, M7A, M7B and M7C protocols, datasets, predictions, metrics, tables,
#     metadata and reports). Each must be present and must match the
#     configuration-pinned hash.
#   * ``output_files``  - the artifacts M7D itself writes. The file set must be
#     exact, the sixteen non-manifest artifacts must match their recorded
#     hashes, and every table, the plan manifest and the report are recomputed
#     from the pinned inputs and compared.
#   * ``ignored_paths`` - transient scratch directories and the two
#     *planned* future benchmark output paths. They are not M7D dependencies,
#     must not be required to exist, and are excluded so the exclusion is
#     auditable rather than implicit.
#
# Expected hashes are never regenerated from the current tree: they are the
# hashes committed in the M7D configuration and recorded in the M7D run
# snapshot, and the two anchors must agree.
M7D_PROTECTED_DEPENDENCY_MEMBERSHIP_VERSION: Final[str] = "m7d-scoped-dependencies-v1"
M7D_PLANNING_CONFIG_FILE: Final[str] = "configs/evaluation_design_plan_v0_1_0.toml"
M7D_IGNORED_PATHS: Final[tuple[str, ...]] = (
    "data/processed/_m7b_clean_run_1_snapshot",
    "data/processed/_m7b_clean_run_2",
    "data/processed/_m7b_probe",
    "data/processed/_m7b_probe2",
    "data/processed/_m7b_probe3",
    "data/processed/_m7b_probe4",
    "data/processed/repro_a",
    "data/processed/repro_b",
    "data/processed/matched_font_benchmark_v0.1.0",
    "reports/matched_font_benchmark_v0_1_0",
)
M7D_OUTPUT_FILES: Final[tuple[str, ...]] = (
    "claims_and_limits.csv",
    "direction_assessment.csv",
    "endpoints.csv",
    "evidence_snapshot.csv",
    "font_inventory.csv",
    "independent_units.csv",
    "input_manifest.csv",
    "isolation_checks.csv",
    "leakage_controls.csv",
    "plan_manifest.json",
    "report.md",
    "research_questions.csv",
    "run_metadata.json",
    "sample_size_scenarios.csv",
    "split_hierarchy.csv",
    "stopping_rules.csv",
    "success_criteria.csv",
)
M7C_CHAMPION_ARTIFACT_SHA256: Final[str] = (
    "c9b1625de3bb503f22fa14f9b6468b028e609b6c7febe7c307dafd0e3e1156e0"
)
M7C_CHAMPION_CARD_SHA256: Final[str] = (
    "9a75725ba1ccf82ab42f009aa744a102cce86a6a9bb1d1aa86431362213507f8"
)
M7C_CHAMPION_NAME: Final[str] = "expanded_random_forest"
M7C_CHAMPION_CALIBRATION: Final[str] = "none"
M3_EXPECTED_FONTS: Final[tuple[str, ...]] = ("arial", "courier_new", "times_new_roman")
M7B_EXPECTED_FONTS: Final[tuple[str, ...]] = ("georgia", "monaco", "impact")
PLANNED_HELDOUT_FONTS: Final[tuple[str, ...]] = (
    "georgia",
    "monaco",
    "impact",
    "verdana",
    "trebuchet_ms",
    "chalkduster",
    "luminari",
)
PLANNED_FAMILIAR_FONTS: Final[tuple[str, ...]] = M3_EXPECTED_FONTS
#: Held-out faces that M7B already used as an evaluation set. They were never
#: used for fitting, calibration or selection, so reusing them keeps three of
#: the seven directly comparable with M7B, but it does mean that only four of
#: the seven held-out faces are genuinely new.
M7B_REUSED_HELDOUT_FONTS: Final[tuple[str, ...]] = M7B_EXPECTED_FONTS
PLANNED_NEW_HELDOUT_FONTS: Final[tuple[str, ...]] = tuple(
    font for font in PLANNED_HELDOUT_FONTS if font not in M7B_REUSED_HELDOUT_FONTS
)


class PlanError(RuntimeError):
    """Raised when a frozen input or plan invariant is violated."""


class PlanConfigError(ValueError):
    """Raised for an invalid M7D configuration."""


@dataclass(frozen=True, slots=True)
class RunSettings:
    name: str
    version: str
    seed: int
    output_reports: Path


@dataclass(frozen=True, slots=True)
class PlanSettings:
    milestone: str
    chosen_direction: str
    status: str
    primary_question: str
    n_text_cases: int
    rows_per_source: int
    primary_rows_per_source: int
    diagnostic_rows_per_source: int
    dataset_version: str
    dataset_output: str
    benchmark_report_output: str
    renderer: str
    canvas_width: int
    canvas_height: int
    image_mode: str
    image_format: str
    text_lengths: tuple[int, ...]
    text_length_weights: tuple[float, ...]
    primary_scope: str
    primary_metric: str
    source_group: str
    bootstrap_method: str
    n_bootstrap: int
    confidence_level: float
    near_degenerate_sd: float


@dataclass(frozen=True, slots=True)
class PrecisionSettings:
    text_case_counts: tuple[int, ...]
    paired_sd_assumptions: tuple[float, ...]
    font_counts: tuple[int, ...]
    font_sd_assumptions: tuple[float, ...]
    interval_multiplier: float


@dataclass(frozen=True, slots=True)
class CorruptionPlan:
    family: str
    unit: str
    role: str
    levels: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FontPlan:
    key: str
    role: str
    family: str
    stratum: str
    path: str
    sha256: str
    license_status: str
    rationale: str


@dataclass(frozen=True, slots=True)
class InputPin:
    key: str
    path: str
    sha256: str
    role: str


@dataclass(frozen=True, slots=True)
class PlanConfig:
    run: RunSettings
    plan: PlanSettings
    precision: PrecisionSettings
    corruptions: tuple[CorruptionPlan, ...]
    fonts: tuple[FontPlan, ...]
    inputs: tuple[InputPin, ...]
    config_path: Path
    config_sha256: str
    config_hash: str

    def identity(self) -> dict[str, Any]:
        return {
            "run": {
                "name": self.run.name,
                "version": self.run.version,
                "seed": self.run.seed,
                "output_reports": str(self.run.output_reports),
            },
            "plan": {
                "milestone": self.plan.milestone,
                "chosen_direction": self.plan.chosen_direction,
                "status": self.plan.status,
                "primary_question": self.plan.primary_question,
                "n_text_cases": self.plan.n_text_cases,
                "rows_per_source": self.plan.rows_per_source,
                "primary_rows_per_source": self.plan.primary_rows_per_source,
                "diagnostic_rows_per_source": self.plan.diagnostic_rows_per_source,
                "dataset_version": self.plan.dataset_version,
                "dataset_output": self.plan.dataset_output,
                "benchmark_report_output": self.plan.benchmark_report_output,
                "renderer": self.plan.renderer,
                "canvas_width": self.plan.canvas_width,
                "canvas_height": self.plan.canvas_height,
                "image_mode": self.plan.image_mode,
                "image_format": self.plan.image_format,
                "text_lengths": self.plan.text_lengths,
                "text_length_weights": self.plan.text_length_weights,
                "primary_scope": self.plan.primary_scope,
                "primary_metric": self.plan.primary_metric,
                "source_group": self.plan.source_group,
                "bootstrap_method": self.plan.bootstrap_method,
                "n_bootstrap": self.plan.n_bootstrap,
                "confidence_level": self.plan.confidence_level,
                "near_degenerate_sd": self.plan.near_degenerate_sd,
            },
            "precision": {
                "text_case_counts": self.precision.text_case_counts,
                "paired_sd_assumptions": self.precision.paired_sd_assumptions,
                "font_counts": self.precision.font_counts,
                "font_sd_assumptions": self.precision.font_sd_assumptions,
                "interval_multiplier": self.precision.interval_multiplier,
            },
            "corruptions": [
                {
                    "family": corruption.family,
                    "unit": corruption.unit,
                    "role": corruption.role,
                    "levels": corruption.levels,
                }
                for corruption in self.corruptions
            ],
            "fonts": [
                {
                    "key": font.key,
                    "role": font.role,
                    "family": font.family,
                    "stratum": font.stratum,
                    "path": font.path,
                    "sha256": font.sha256,
                    "license_status": font.license_status,
                    "rationale": font.rationale,
                }
                for font in self.fonts
            ],
            "inputs": [
                {
                    "key": pin.key,
                    "path": pin.path,
                    "sha256": pin.sha256,
                    "role": pin.role,
                }
                for pin in self.inputs
            ],
        }


def _check_keys(raw: Mapping[str, object], expected: set[str], label: str) -> None:
    actual = set(raw)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise PlanConfigError(f"{label} is missing key(s): {missing}")
    if unknown:
        raise PlanConfigError(f"{label} has unknown key(s): {unknown}")


def _section(raw: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise PlanConfigError(f"{key} must be a table")
    return value


def _as_str(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlanConfigError(f"{label} must be a non-empty string")
    return value.strip()


def _as_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanConfigError(f"{label} must be an integer")
    return value


def _as_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PlanConfigError(f"{label} must be a number")
    return float(value)


def _as_int_tuple(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise PlanConfigError(f"{label} must be a non-empty array")
    return tuple(_as_int(item, f"{label}[]") for item in value)


def _as_float_tuple(value: object, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise PlanConfigError(f"{label} must be a non-empty array")
    return tuple(_as_float(item, f"{label}[]") for item in value)


def _sha256(value: object, label: str) -> str:
    text = _as_str(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise PlanConfigError(f"{label} must be a lowercase SHA-256 hex digest")
    return text


def _parse_run(raw: Mapping[str, object]) -> RunSettings:
    _check_keys(raw, {"name", "version", "seed", "output_reports"}, "run")
    run = RunSettings(
        name=_as_str(raw["name"], "run.name"),
        version=_as_str(raw["version"], "run.version"),
        seed=_as_int(raw["seed"], "run.seed"),
        output_reports=Path(_as_str(raw["output_reports"], "run.output_reports")),
    )
    if run.seed < 0:
        raise PlanConfigError("run.seed must be >= 0")
    return run


def _parse_plan(raw: Mapping[str, object]) -> PlanSettings:
    _check_keys(
        raw,
        {
            "milestone",
            "chosen_direction",
            "status",
            "primary_question",
            "n_text_cases",
            "rows_per_source",
            "primary_rows_per_source",
            "diagnostic_rows_per_source",
            "dataset_version",
            "dataset_output",
            "benchmark_report_output",
            "renderer",
            "canvas_width",
            "canvas_height",
            "image_mode",
            "image_format",
            "text_lengths",
            "text_length_weights",
            "primary_scope",
            "primary_metric",
            "source_group",
            "bootstrap_method",
            "n_bootstrap",
            "confidence_level",
            "near_degenerate_sd",
        },
        "plan",
    )
    plan = PlanSettings(
        milestone=_as_str(raw["milestone"], "plan.milestone"),
        chosen_direction=_as_str(raw["chosen_direction"], "plan.chosen_direction"),
        status=_as_str(raw["status"], "plan.status"),
        primary_question=_as_str(raw["primary_question"], "plan.primary_question"),
        n_text_cases=_as_int(raw["n_text_cases"], "plan.n_text_cases"),
        rows_per_source=_as_int(raw["rows_per_source"], "plan.rows_per_source"),
        primary_rows_per_source=_as_int(
            raw["primary_rows_per_source"], "plan.primary_rows_per_source"
        ),
        diagnostic_rows_per_source=_as_int(
            raw["diagnostic_rows_per_source"], "plan.diagnostic_rows_per_source"
        ),
        dataset_version=_as_str(raw["dataset_version"], "plan.dataset_version"),
        dataset_output=_as_str(raw["dataset_output"], "plan.dataset_output"),
        benchmark_report_output=_as_str(
            raw["benchmark_report_output"], "plan.benchmark_report_output"
        ),
        renderer=_as_str(raw["renderer"], "plan.renderer"),
        canvas_width=_as_int(raw["canvas_width"], "plan.canvas_width"),
        canvas_height=_as_int(raw["canvas_height"], "plan.canvas_height"),
        image_mode=_as_str(raw["image_mode"], "plan.image_mode"),
        image_format=_as_str(raw["image_format"], "plan.image_format"),
        text_lengths=_as_int_tuple(raw["text_lengths"], "plan.text_lengths"),
        text_length_weights=_as_float_tuple(raw["text_length_weights"], "plan.text_length_weights"),
        primary_scope=_as_str(raw["primary_scope"], "plan.primary_scope"),
        primary_metric=_as_str(raw["primary_metric"], "plan.primary_metric"),
        source_group=_as_str(raw["source_group"], "plan.source_group"),
        bootstrap_method=_as_str(raw["bootstrap_method"], "plan.bootstrap_method"),
        n_bootstrap=_as_int(raw["n_bootstrap"], "plan.n_bootstrap"),
        confidence_level=_as_float(raw["confidence_level"], "plan.confidence_level"),
        near_degenerate_sd=_as_float(raw["near_degenerate_sd"], "plan.near_degenerate_sd"),
    )
    if plan.milestone != "M7D":
        raise PlanConfigError("plan.milestone must be 'M7D'")
    if plan.chosen_direction not in DIRECTION_ORDER:
        raise PlanConfigError("plan.chosen_direction must be one of A, B or C")
    if plan.status != "planned_not_executed":
        raise PlanConfigError("plan.status must be 'planned_not_executed'")
    if plan.n_text_cases < 20:
        raise PlanConfigError("plan.n_text_cases must be at least 20")
    if plan.canvas_width <= 0 or plan.canvas_height <= 0:
        raise PlanConfigError("canvas dimensions must be positive")
    if plan.image_mode != "L" or plan.image_format != "PNG":
        raise PlanConfigError("M7D requires 8-bit grayscale PNG as the planned format")
    if plan.text_lengths != (4, 5, 6):
        raise PlanConfigError("plan.text_lengths must be [4, 5, 6]")
    if len(plan.text_length_weights) != len(plan.text_lengths):
        raise PlanConfigError("text length weights must align with text lengths")
    if not math.isclose(sum(plan.text_length_weights), 1.0, abs_tol=1e-9):
        raise PlanConfigError("plan.text_length_weights must sum to 1")
    if any(weight <= 0.0 for weight in plan.text_length_weights):
        raise PlanConfigError("text length weights must be positive")
    if plan.primary_scope != "all_primary":
        raise PlanConfigError("plan.primary_scope must be 'all_primary'")
    if plan.primary_metric != "rmse":
        raise PlanConfigError("plan.primary_metric must be 'rmse'")
    if plan.source_group != "matched_text_case":
        raise PlanConfigError("plan.source_group must be 'matched_text_case'")
    if plan.bootstrap_method != "matched_case_grouped_percentile":
        raise PlanConfigError("plan.bootstrap_method must be 'matched_case_grouped_percentile'")
    if plan.n_bootstrap < 1:
        raise PlanConfigError("plan.n_bootstrap must be >= 1")
    if not 0.0 < plan.confidence_level < 1.0:
        raise PlanConfigError("plan.confidence_level must be between 0 and 1")
    if plan.near_degenerate_sd <= 0.0:
        raise PlanConfigError("plan.near_degenerate_sd must be positive")
    if not plan.dataset_output.startswith("data/processed/"):
        raise PlanConfigError("planned dataset output must be under data/processed/")
    if not plan.benchmark_report_output.startswith("reports/"):
        raise PlanConfigError("planned benchmark report must be under reports/")
    if plan.rows_per_source != plan.primary_rows_per_source + plan.diagnostic_rows_per_source:
        raise PlanConfigError("primary plus diagnostic rows must equal rows_per_source")
    return plan


def _parse_precision(raw: Mapping[str, object]) -> PrecisionSettings:
    _check_keys(
        raw,
        {
            "text_case_counts",
            "paired_sd_assumptions",
            "font_counts",
            "font_sd_assumptions",
            "interval_multiplier",
        },
        "precision",
    )
    settings = PrecisionSettings(
        text_case_counts=_as_int_tuple(raw["text_case_counts"], "precision.text_case_counts"),
        paired_sd_assumptions=_as_float_tuple(
            raw["paired_sd_assumptions"], "precision.paired_sd_assumptions"
        ),
        font_counts=_as_int_tuple(raw["font_counts"], "precision.font_counts"),
        font_sd_assumptions=_as_float_tuple(
            raw["font_sd_assumptions"], "precision.font_sd_assumptions"
        ),
        interval_multiplier=_as_float(raw["interval_multiplier"], "precision.interval_multiplier"),
    )
    if any(count < 2 for count in settings.text_case_counts):
        raise PlanConfigError("precision.text_case_counts must all be >= 2")
    if any(count < 2 for count in settings.font_counts):
        raise PlanConfigError("precision.font_counts must all be >= 2")
    if any(value <= 0.0 for value in settings.paired_sd_assumptions):
        raise PlanConfigError("paired SD assumptions must be positive")
    if any(value <= 0.0 for value in settings.font_sd_assumptions):
        raise PlanConfigError("font SD assumptions must be positive")
    if settings.interval_multiplier <= 0.0:
        raise PlanConfigError("precision.interval_multiplier must be positive")
    return settings


def _parse_corruptions(raw: object, plan: PlanSettings) -> tuple[CorruptionPlan, ...]:
    if not isinstance(raw, list) or not raw:
        raise PlanConfigError("[[corruptions]] must be a non-empty array")
    parsed: list[CorruptionPlan] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise PlanConfigError(f"corruptions[{index}] must be a table")
        _check_keys(item, {"family", "unit", "role", "levels"}, f"corruptions[{index}]")
        corruption = CorruptionPlan(
            family=_as_str(item["family"], f"corruptions[{index}].family"),
            unit=_as_str(item["unit"], f"corruptions[{index}].unit"),
            role=_as_str(item["role"], f"corruptions[{index}].role"),
            levels=_as_float_tuple(item["levels"], f"corruptions[{index}].levels"),
        )
        if corruption.role not in {"primary", "diagnostic"}:
            raise PlanConfigError(f"corruptions[{index}].role must be primary or diagnostic")
        key = (corruption.family, corruption.role)
        if key in seen:
            raise PlanConfigError(f"duplicate corruption family/role: {key}")
        seen.add(key)
        parsed.append(corruption)
    expected = {
        (family, role): levels
        for family, roles in EXPECTED_CORRUPTIONS.items()
        for role, levels in roles
    }
    actual = {(corruption.family, corruption.role): corruption.levels for corruption in parsed}
    if actual != expected:
        raise PlanConfigError(
            "planned corruptions must reproduce the frozen M7B ladders exactly: "
            f"expected {expected}, got {actual}"
        )
    rows_per_source = sum(len(corruption.levels) for corruption in parsed)
    primary_rows = sum(
        len(corruption.levels) for corruption in parsed if corruption.role == "primary"
    )
    diagnostic_rows = rows_per_source - primary_rows
    if rows_per_source != plan.rows_per_source:
        raise PlanConfigError(
            f"corruption ladder implies {rows_per_source} rows/source, "
            f"config says {plan.rows_per_source}"
        )
    if primary_rows != plan.primary_rows_per_source:
        raise PlanConfigError(
            f"corruption ladder implies {primary_rows} primary rows/source, "
            f"config says {plan.primary_rows_per_source}"
        )
    if diagnostic_rows != plan.diagnostic_rows_per_source:
        raise PlanConfigError(
            f"corruption ladder implies {diagnostic_rows} diagnostic rows/source, "
            f"config says {plan.diagnostic_rows_per_source}"
        )
    return tuple(parsed)


def _parse_fonts(raw: object, plan: PlanSettings) -> tuple[FontPlan, ...]:
    if not isinstance(raw, list) or not raw:
        raise PlanConfigError("[[planned_fonts]] must be a non-empty array")
    fonts: list[FontPlan] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise PlanConfigError(f"planned_fonts[{index}] must be a table")
        _check_keys(
            item,
            {
                "key",
                "role",
                "family",
                "stratum",
                "path",
                "sha256",
                "license_status",
                "rationale",
            },
            f"planned_fonts[{index}]",
        )
        font = FontPlan(
            key=_as_str(item["key"], f"planned_fonts[{index}].key"),
            role=_as_str(item["role"], f"planned_fonts[{index}].role"),
            family=_as_str(item["family"], f"planned_fonts[{index}].family"),
            stratum=_as_str(item["stratum"], f"planned_fonts[{index}].stratum"),
            path=_as_str(item["path"], f"planned_fonts[{index}].path"),
            sha256=_sha256(item["sha256"], f"planned_fonts[{index}].sha256"),
            license_status=_as_str(
                item["license_status"], f"planned_fonts[{index}].license_status"
            ),
            rationale=_as_str(item["rationale"], f"planned_fonts[{index}].rationale"),
        )
        if font.role not in FONT_ROLES:
            raise PlanConfigError(f"planned_fonts[{index}].role must be familiar or held-out")
        if font.stratum not in FONT_STRATA:
            raise PlanConfigError(f"planned_fonts[{index}].stratum is unknown")
        if font.license_status != "review_required_before_generation":
            raise PlanConfigError(
                "planned_fonts license_status must record that local-use review is still required"
            )
        fonts.append(font)
    keys = [font.key for font in fonts]
    if len(set(keys)) != len(keys):
        raise PlanConfigError("planned font keys must be unique")
    paths = [font.path for font in fonts]
    if len(set(paths)) != len(paths):
        raise PlanConfigError("planned font paths must be unique")
    hashes = [font.sha256 for font in fonts]
    if len(set(hashes)) != len(hashes):
        raise PlanConfigError("planned font hashes must be unique")
    familiar = tuple(font.key for font in fonts if font.role == "familiar")
    heldout = tuple(font.key for font in fonts if font.role == "held-out")
    if familiar != PLANNED_FAMILIAR_FONTS:
        raise PlanConfigError(f"familiar fonts must be {PLANNED_FAMILIAR_FONTS}")
    if heldout != PLANNED_HELDOUT_FONTS:
        raise PlanConfigError(f"held-out fonts must be {PLANNED_HELDOUT_FONTS}")
    if len(fonts) != 10:
        raise PlanConfigError("M7D's planned roster must contain exactly ten fonts")
    sources = plan.n_text_cases * len(fonts)
    if sources != 300:
        raise PlanConfigError(
            f"planned roster and text cases imply {sources} sources; expected 300"
        )
    return tuple(fonts)


def _parse_inputs(raw: object) -> tuple[InputPin, ...]:
    if not isinstance(raw, list) or not raw:
        raise PlanConfigError("[[inputs]] must be a non-empty array of tables")
    pins: list[InputPin] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise PlanConfigError(f"inputs[{index}] must be a table")
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
        raise PlanConfigError("input keys must be unique")
    missing = sorted(REQUIRED_INPUT_KEYS - set(keys))
    unknown = sorted(set(keys) - REQUIRED_INPUT_KEYS)
    if missing:
        raise PlanConfigError(f"missing required input pin(s): {missing}")
    if unknown:
        raise PlanConfigError(f"unknown input pin(s): {unknown}")
    return tuple(pins)


def load_config(path: Path | str) -> PlanConfig:
    """Load and validate the M7D plan configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise PlanConfigError(f"missing M7D configuration: {config_path}")
    raw_bytes = config_path.read_bytes()
    try:
        payload = tomllib.loads(raw_bytes.decode("utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise PlanConfigError(f"invalid TOML in {config_path}: {error}") from error
    _check_keys(
        payload,
        {"run", "plan", "corruptions", "precision", "planned_fonts", "inputs"},
        "top-level",
    )
    run = _parse_run(_section(payload, "run"))
    plan = _parse_plan(_section(payload, "plan"))
    corruptions = _parse_corruptions(payload.get("corruptions"), plan)
    precision = _parse_precision(_section(payload, "precision"))
    fonts = _parse_fonts(payload.get("planned_fonts"), plan)
    inputs = _parse_inputs(payload.get("inputs"))
    config = PlanConfig(
        run=run,
        plan=plan,
        precision=precision,
        corruptions=corruptions,
        fonts=fonts,
        inputs=inputs,
        config_path=config_path,
        config_sha256=sha256_bytes(raw_bytes),
        config_hash="",
    )
    identity_hash = sha256_bytes(canonical_json(config.identity()).encode("utf-8"))
    return PlanConfig(
        run=config.run,
        plan=config.plan,
        precision=config.precision,
        corruptions=config.corruptions,
        fonts=config.fonts,
        inputs=config.inputs,
        config_path=config.config_path,
        config_sha256=config.config_sha256,
        config_hash=identity_hash,
    )


@dataclass(frozen=True, slots=True)
class PlanInputs:
    config: PlanConfig
    m3_summary: dict[str, Any]
    m3_sources: pd.DataFrame
    m3_manifest: pd.DataFrame
    m5c_model_card: dict[str, Any]
    m5c_test_predictions: pd.DataFrame
    m5c_test_slice_metrics: pd.DataFrame
    m5c_run_metadata: dict[str, Any]
    m7a_directions: pd.DataFrame
    m7a_claims: pd.DataFrame
    m7a_run_metadata: dict[str, Any]
    m7b_summary: dict[str, Any]
    m7b_sources: pd.DataFrame
    m7b_manifest: pd.DataFrame
    m7b_predictions: pd.DataFrame
    m7b_overall_metrics: pd.DataFrame
    m7b_font_metrics: pd.DataFrame
    m7b_family_metrics: pd.DataFrame
    m7b_comparison: pd.DataFrame
    m7b_run_metadata: dict[str, Any]
    m7c_font_metrics: pd.DataFrame
    m7c_family_metrics: pd.DataFrame
    m7c_severity_metrics: pd.DataFrame
    m7c_source_metrics: pd.DataFrame
    m7c_error_concentration: pd.DataFrame
    m7c_uncertainty: pd.DataFrame
    m7c_benchmark_comparison: pd.DataFrame
    m7c_claims: pd.DataFrame
    m7c_run_metadata: dict[str, Any]
    input_hashes: dict[str, str]
    input_manifest: pd.DataFrame
    font_inventory: pd.DataFrame
    checks: tuple[CheckResult, ...]


def _pin_by_key(config: PlanConfig, key: str) -> InputPin:
    for pin in config.inputs:
        if pin.key == key:
            return pin
    raise PlanError(f"configuration is missing required input pin: {key}")


def _require_file_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise PlanError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise PlanError(
            f"input hash mismatch for {label}: {path} has {actual}, expected {expected}"
        )
    return actual


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise PlanError(f"missing {label}: {path}")
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise PlanError(f"cannot read {label}: {path}: {error}") from error


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise PlanError(f"missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PlanError(f"cannot read {label}: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise PlanError(f"{label} must contain a JSON object: {path}")
    return payload


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise PlanError(f"{label} is missing column(s): {missing}")


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
        raise PlanError(f"{label} failed: {detail}")


def _glyph_missing(font: FontPlan) -> tuple[bool, bool, str, str]:
    """Return available, hash_verified, missing_glyphs and a status detail."""
    path = Path(font.path)
    if not path.is_file():
        return False, False, "unavailable", f"missing font file: {path}"
    actual_hash = sha256_file(path)
    hash_verified = actual_hash == font.sha256
    try:
        face = ImageFont.truetype(str(path), 32)
    except Exception as error:  # noqa: BLE001 - font parse errors are data errors
        return True, hash_verified, "unreadable", f"{type(error).__name__}: {error}"
    missing: list[str] = []
    for character in DEFAULT_CHARSET:
        try:
            bbox = face.getbbox(character)
        except Exception:  # noqa: BLE001 - missing glyph confirms unusable coverage
            missing.append(character)
            continue
        if bbox is None or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            missing.append(character)
    if missing:
        return True, hash_verified, "".join(missing), "missing glyph coverage"
    return True, hash_verified, "", "font file and charset coverage verified"


def _font_m7b_status(font: FontPlan) -> str:
    """Record how a planned font relates to the frozen M7B evaluation set."""
    if font.role == "familiar":
        return "familiar_m3_development_font"
    if font.key in M7B_REUSED_HELDOUT_FONTS:
        return "held_out_in_m7b_evaluation_only"
    return "new_held_out_face"


def _font_inventory(config: PlanConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for font in config.fonts:
        available, hash_verified, missing, detail = _glyph_missing(font)
        path = Path(font.path)
        rows.append(
            {
                "font_id": font.key,
                "role": font.role,
                "family": font.family,
                "stratum": font.stratum,
                "path": font.path,
                "sha256": font.sha256,
                "bytes": path.stat().st_size if path.is_file() else math.nan,
                "available": available,
                "hash_verified": hash_verified,
                "glyph_coverage_ok": available and not missing and missing != "unreadable",
                "missing_glyphs": missing,
                "license_status": font.license_status,
                "license_claim": "not_asserted",
                "planned_sources": config.plan.n_text_cases,
                "planned_rows": config.plan.n_text_cases * config.plan.rows_per_source,
                "planned_primary_rows": (
                    config.plan.n_text_cases * config.plan.primary_rows_per_source
                ),
                "planned_diagnostic_rows": (
                    config.plan.n_text_cases * config.plan.diagnostic_rows_per_source
                ),
                "rationale": font.rationale,
                "m7b_status": _font_m7b_status(font),
                "verification_detail": detail,
            }
        )
    return pd.DataFrame(rows)


def _summary_count(summary: Mapping[str, Any], key: str, label: str) -> Any:
    counts = summary.get("counts")
    if not isinstance(counts, dict) or key not in counts:
        raise PlanError(f"{label} is missing counts.{key}")
    return counts[key]


def _split_counts(sources: pd.DataFrame) -> dict[str, int]:
    if "split" not in sources.columns:
        raise PlanError("source table is missing split")
    return {str(key): int(value) for key, value in sources["split"].value_counts().items()}


def load_inputs(config: PlanConfig) -> PlanInputs:
    """Verify every pinned input and load the frozen evidence frames."""
    input_hashes: dict[str, str] = {}
    manifest_rows: list[dict[str, Any]] = []
    for pin in config.inputs:
        path = Path(pin.path)
        digest = _require_file_hash(path, pin.sha256, pin.key)
        input_hashes[pin.key] = digest
        manifest_rows.append(
            {
                "key": pin.key,
                "path": pin.path,
                "sha256": pin.sha256,
                "role": pin.role,
                "bytes": path.stat().st_size,
                "verified": digest == pin.sha256,
            }
        )
    input_manifest = pd.DataFrame(manifest_rows)
    font_inventory = _font_inventory(config)

    m3_summary = _read_json(Path(_pin_by_key(config, "m3_summary").path), "M3 summary")
    m3_sources = _read_csv(Path(_pin_by_key(config, "m3_sources").path), "M3 sources")
    m3_manifest = _read_csv(Path(_pin_by_key(config, "m3_manifest").path), "M3 manifest")
    m5c_model_card = _read_json(
        Path(_pin_by_key(config, "m5c_model_card").path), "M5C champion model card"
    )
    m5c_predictions = _read_csv(
        Path(_pin_by_key(config, "m5c_test_predictions").path), "M5C test predictions"
    )
    m5c_metrics = _read_csv(
        Path(_pin_by_key(config, "m5c_test_slice_metrics").path), "M5C test-slice metrics"
    )
    m5c_metadata = _read_json(
        Path(_pin_by_key(config, "m5c_run_metadata").path), "M5C run metadata"
    )
    m7a_directions = _read_csv(
        Path(_pin_by_key(config, "m7a_next_direction_comparison").path),
        "M7A next-direction comparison",
    )
    m7a_claims = _read_csv(
        Path(_pin_by_key(config, "m7a_claim_evidence").path), "M7A claim evidence"
    )
    m7a_metadata = _read_json(
        Path(_pin_by_key(config, "m7a_run_metadata").path), "M7A run metadata"
    )
    m7b_summary = _read_json(Path(_pin_by_key(config, "m7b_summary").path), "M7B summary")
    m7b_sources = _read_csv(Path(_pin_by_key(config, "m7b_sources").path), "M7B sources")
    m7b_manifest = _read_csv(Path(_pin_by_key(config, "m7b_manifest").path), "M7B manifest")
    m7b_predictions = _read_csv(
        Path(_pin_by_key(config, "m7b_predictions").path), "M7B predictions"
    )
    m7b_overall = _read_csv(
        Path(_pin_by_key(config, "m7b_overall_metrics").path), "M7B overall metrics"
    )
    m7b_font = _read_csv(Path(_pin_by_key(config, "m7b_font_metrics").path), "M7B per-font metrics")
    m7b_family = _read_csv(
        Path(_pin_by_key(config, "m7b_family_metrics").path), "M7B family metrics"
    )
    m7b_comparison = _read_csv(
        Path(_pin_by_key(config, "m7b_comparison").path), "M7B familiar comparison"
    )
    m7b_metadata = _read_json(
        Path(_pin_by_key(config, "m7b_run_metadata").path), "M7B run metadata"
    )
    m7c_font = _read_csv(Path(_pin_by_key(config, "m7c_font_metrics").path), "M7C per-font metrics")
    m7c_family = _read_csv(
        Path(_pin_by_key(config, "m7c_family_metrics").path), "M7C family metrics"
    )
    m7c_severity = _read_csv(
        Path(_pin_by_key(config, "m7c_severity_metrics").path),
        "M7C geometric severity metrics",
    )
    m7c_source = _read_csv(
        Path(_pin_by_key(config, "m7c_source_metrics").path), "M7C source metrics"
    )
    m7c_concentration = _read_csv(
        Path(_pin_by_key(config, "m7c_error_concentration").path),
        "M7C error concentration",
    )
    m7c_uncertainty = _read_csv(
        Path(_pin_by_key(config, "m7c_uncertainty").path), "M7C grouped uncertainty"
    )
    m7c_comparison = _read_csv(
        Path(_pin_by_key(config, "m7c_benchmark_comparison").path),
        "M7C familiar-versus-held-out comparison",
    )
    m7c_claims = _read_csv(
        Path(_pin_by_key(config, "m7c_claim_evidence").path), "M7C claim evidence"
    )
    m7c_metadata = _read_json(
        Path(_pin_by_key(config, "m7c_run_metadata").path), "M7C run metadata"
    )

    _require_columns(
        m3_manifest,
        ("sample_id", "source_id", "font_id", "split", "height", "width", "image_sha256"),
        "M3 manifest",
    )
    _require_columns(m3_sources, ("source_id", "split", "font_id"), "M3 sources")
    _require_columns(
        m5c_predictions,
        ("model", "scope", "split", "source_id", "ssim_target", "prediction"),
        "M5C test predictions",
    )
    _require_columns(
        m5c_metrics,
        ("model", "scope", "split", "mae", "rmse", "n_samples", "n_sources"),
        "M5C test-slice metrics",
    )
    _require_columns(
        m7b_predictions,
        (
            "source_id",
            "font_id",
            "variant_family",
            "severity_level",
            "amplitude_px",
            "protocol_scope",
            "ssim_target",
            "prediction",
        ),
        "M7B predictions",
    )
    _require_columns(
        m7b_overall,
        ("scope", "font_id", "protocol_role", "mae", "rmse", "r2", "n_rows", "n_sources"),
        "M7B overall metrics",
    )
    _require_columns(
        m7b_font,
        ("scope", "font_id", "protocol_role", "mae", "rmse", "r2", "n_rows", "n_sources"),
        "M7B per-font metrics",
    )
    _require_columns(
        m7b_family,
        (
            "family",
            "severity",
            "font_id",
            "protocol_role",
            "mae",
            "rmse",
            "r2",
            "n_rows",
            "n_sources",
        ),
        "M7B family metrics",
    )
    _require_columns(
        m7c_font,
        ("scope", "font_id", "protocol_role", "mae", "rmse", "r2", "n_rows", "n_sources"),
        "M7C per-font metrics",
    )
    _require_columns(
        m7c_family,
        ("scope", "font_id", "protocol_role", "mae", "rmse", "r2", "n_rows", "n_sources"),
        "M7C family metrics",
    )
    _require_columns(
        m7c_severity,
        ("amplitude_px", "font_id", "protocol_role", "mae", "rmse", "r2"),
        "M7C severity metrics",
    )
    _require_columns(
        m7c_concentration,
        ("scope", "font_id", "top1_share", "top_k_share", "effective_sources"),
        "M7C error concentration",
    )
    _require_columns(
        m7c_comparison,
        ("m7b_scope", "familiar_scope", "metric", "m7b_estimate", "familiar_estimate"),
        "M7C benchmark comparison",
    )
    _require_columns(m7a_directions, ("direction", "title", "recommended"), "M7A directions")
    _require_columns(m7a_claims, ("claim_id", "classification"), "M7A claims")
    _require_columns(m7c_claims, ("claim_id", "classification"), "M7C claims")

    checks: list[CheckResult] = []
    _check(checks, "input_pins", lambda: f"{len(config.inputs)} pinned inputs verified")
    _check(
        checks,
        "font_inventory",
        lambda: (
            "all planned font files, hashes and glyph coverage verified"
            if bool(font_inventory["available"].all())
            and bool(font_inventory["hash_verified"].all())
            and bool(font_inventory["glyph_coverage_ok"].all())
            else (_ for _ in ()).throw(PlanError("one or more planned fonts are unusable"))
        ),
    )
    _check(
        checks,
        "license_gate",
        lambda: (
            "all planned fonts record local-use review as pending"
            if set(font_inventory["license_status"]) == {"review_required_before_generation"}
            else (_ for _ in ()).throw(PlanError("license status is not recorded"))
        ),
    )
    _check(
        checks,
        "m3_counts",
        lambda: _m3_count_detail(m3_summary, m3_sources, m3_manifest),
    )
    _check(
        checks,
        "m7b_counts",
        lambda: _m7b_count_detail(m7b_summary, m7b_sources, m7b_manifest, m7b_predictions),
    )
    _check(
        checks,
        "model_identity",
        lambda: _model_identity_detail(m5c_model_card, input_hashes),
    )
    _check(
        checks,
        "m7c_counts",
        lambda: _m7c_count_detail(m7b_predictions, m7c_font, m7c_family),
    )
    if any(not check.passed for check in checks):
        _require_all_checks(checks, "M7D input checks")

    return PlanInputs(
        config=config,
        m3_summary=m3_summary,
        m3_sources=m3_sources,
        m3_manifest=m3_manifest,
        m5c_model_card=m5c_model_card,
        m5c_test_predictions=m5c_predictions,
        m5c_test_slice_metrics=m5c_metrics,
        m5c_run_metadata=m5c_metadata,
        m7a_directions=m7a_directions,
        m7a_claims=m7a_claims,
        m7a_run_metadata=m7a_metadata,
        m7b_summary=m7b_summary,
        m7b_sources=m7b_sources,
        m7b_manifest=m7b_manifest,
        m7b_predictions=m7b_predictions,
        m7b_overall_metrics=m7b_overall,
        m7b_font_metrics=m7b_font,
        m7b_family_metrics=m7b_family,
        m7b_comparison=m7b_comparison,
        m7b_run_metadata=m7b_metadata,
        m7c_font_metrics=m7c_font,
        m7c_family_metrics=m7c_family,
        m7c_severity_metrics=m7c_severity,
        m7c_source_metrics=m7c_source,
        m7c_error_concentration=m7c_concentration,
        m7c_uncertainty=m7c_uncertainty,
        m7c_benchmark_comparison=m7c_comparison,
        m7c_claims=m7c_claims,
        m7c_run_metadata=m7c_metadata,
        input_hashes=input_hashes,
        input_manifest=input_manifest,
        font_inventory=font_inventory,
        checks=tuple(checks),
    )


def _m3_count_detail(
    summary: Mapping[str, Any], sources: pd.DataFrame, manifest: pd.DataFrame
) -> str:
    if _summary_count(summary, "sources", "M3 summary") != 60:
        raise PlanError("M3 summary source count is not 60")
    if _summary_count(summary, "samples", "M3 summary") != 540:
        raise PlanError("M3 summary sample count is not 540")
    if len(sources) != 60 or len(manifest) != 540:
        raise PlanError("M3 tables do not contain 60 sources / 540 rows")
    if tuple(sorted(sources["font_id"].unique())) != tuple(sorted(M3_EXPECTED_FONTS)):
        raise PlanError("M3 font set is not the frozen familiar-font set")
    splits = _split_counts(sources)
    if splits != {"train": 36, "val": 12, "test": 12}:
        raise PlanError(f"unexpected M3 split counts: {splits}")
    if set(manifest["height"]) != {64} or set(manifest["width"]) != {192}:
        raise PlanError("M3 canvas is not 192x64")
    return "60 sources, 540 rows, 36/12/12 split, 192x64 canvas"


def _m7b_count_detail(
    summary: Mapping[str, Any],
    sources: pd.DataFrame,
    manifest: pd.DataFrame,
    predictions: pd.DataFrame,
) -> str:
    if _summary_count(summary, "sources", "M7B summary") != 60:
        raise PlanError("M7B summary source count is not 60")
    if _summary_count(summary, "rows", "M7B summary") != 1260:
        raise PlanError("M7B summary row count is not 1260")
    if _summary_count(summary, "images", "M7B summary") != 1260:
        raise PlanError("M7B summary image count is not 1260")
    if len(sources) != 60 or len(manifest) != 1260 or len(predictions) != 1260:
        raise PlanError("M7B tables do not contain 60 sources / 1260 rows")
    per_font = sources.groupby("font_id")["source_id"].nunique().to_dict()
    if per_font != {"georgia": 20, "impact": 20, "monaco": 20}:
        raise PlanError(f"unexpected M7B per-font source counts: {per_font}")
    if int((predictions["protocol_scope"] == "primary").sum()) != 1140:
        raise PlanError("M7B primary row count is not 1140")
    if int((predictions["protocol_scope"] == "diagnostic").sum()) != 120:
        raise PlanError("M7B diagnostic row count is not 120")
    return "60 sources, 1260 rows, 20 sources per held-out font, 1140 primary rows"


def _model_identity_detail(model_card: Mapping[str, Any], input_hashes: Mapping[str, str]) -> str:
    if model_card.get("model_family") != M7C_CHAMPION_NAME:
        raise PlanError("M5C champion model_family does not match the frozen baseline")
    if model_card.get("calibration") != M7C_CHAMPION_CALIBRATION:
        raise PlanError("M5C champion calibration is not 'none'")
    if model_card.get("artifact_sha256") != M7C_CHAMPION_ARTIFACT_SHA256:
        raise PlanError("M5C champion artifact hash does not match the recorded baseline")
    if input_hashes.get("m5c_model_artifact") != M7C_CHAMPION_ARTIFACT_SHA256:
        raise PlanError("pinned M5C artifact hash does not match the model card")
    if input_hashes.get("m5c_model_card") != M7C_CHAMPION_CARD_SHA256:
        raise PlanError("pinned M5C model card hash does not match the baseline")
    return "expanded_random_forest, calibration none, artifact and card hashes match"


def _m7c_count_detail(
    predictions: pd.DataFrame, m7c_font: pd.DataFrame, m7c_family: pd.DataFrame
) -> str:
    if len(predictions) != 1260:
        raise PlanError("M7B predictions changed row count")
    primary = predictions.loc[predictions["protocol_scope"] == "primary"]
    if len(primary) != 1140 or primary["source_id"].nunique() != 60:
        raise PlanError("M7C primary accounting does not match 1140 rows / 60 sources")
    all_primary = m7c_family.loc[
        (m7c_family["scope"] == "all_primary") & (m7c_family["font_id"] == "all")
    ]
    if len(all_primary) != 1:
        raise PlanError("M7C all-primary row is missing or duplicated")
    if int(all_primary.iloc[0]["n_rows"]) != 1140:
        raise PlanError("M7C all-primary row count is not 1140")
    if len(m7c_font) == 0:
        raise PlanError("M7C per-font metrics are empty")
    return "1140 primary rows, 60 source groups, all-primary accounting matches"


def _scope_mask_predictions(frame: pd.DataFrame, scope: str) -> pd.Series:
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
    raise PlanError(f"unknown prediction scope: {scope}")


def _scope_frame(frame: pd.DataFrame, scope: str, font_id: str | None = None) -> pd.DataFrame:
    subset = frame.loc[_scope_mask_predictions(frame, scope)].copy()
    if font_id is not None:
        subset = subset.loc[subset["font_id"] == font_id].copy()
    return subset


def _metric_values(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "n_rows": 0,
            "n_sources": 0,
            "n_distinct_images": 0,
            "target_mean": math.nan,
            "target_sd": math.nan,
            "mae": math.nan,
            "rmse": math.nan,
            "r2": math.nan,
            "r2_defined": False,
        }
    target = frame[TARGET].to_numpy(dtype=np.float64)
    predicted = frame["prediction"].to_numpy(dtype=np.float64)
    metrics = regression_metrics(target, predicted)
    return {
        "n_rows": int(len(frame)),
        "n_sources": int(frame["source_id"].nunique()),
        "n_distinct_images": (
            int(frame["image_sha256"].nunique()) if "image_sha256" in frame else math.nan
        ),
        "target_mean": float(np.mean(target)),
        "target_sd": float(np.std(target, ddof=0)),
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "r2": float(metrics["r2"]),
        "r2_defined": bool(metrics["r2_defined"]),
    }


def _recompute_m7b(
    predictions: pd.DataFrame, scope: str, font_id: str | None = None
) -> dict[str, Any]:
    return _metric_values(_scope_frame(predictions, scope, font_id))


def _recompute_m5c(
    predictions: pd.DataFrame, scope: str, model: str = M7C_CHAMPION_NAME
) -> dict[str, Any]:
    subset = predictions.loc[
        (predictions["model"] == model)
        & (predictions["split"] == "test")
        & (predictions["scope"] == scope)
    ].copy()
    if subset.empty:
        raise PlanError(f"no M5C test rows for scope {scope}")
    target = subset[TARGET].to_numpy(dtype=np.float64)
    predicted = subset["prediction"].to_numpy(dtype=np.float64)
    metrics = regression_metrics(target, predicted)
    return {
        "n_rows": int(len(subset)),
        "n_sources": int(subset["source_id"].nunique()),
        "target_mean": float(np.mean(target)),
        "target_sd": float(np.std(target, ddof=0)),
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "r2": float(metrics["r2"]),
        "r2_defined": bool(metrics["r2_defined"]),
    }


def _table_row(
    frame: pd.DataFrame,
    *,
    scope: str,
    font_id: str,
    role: str | None = None,
    amplitude: float | None = None,
) -> pd.Series:
    subset = frame.loc[(frame["scope"] == scope) & (frame["font_id"] == font_id)].copy()
    if role is not None:
        subset = subset.loc[subset["protocol_role"] == role]
    if amplitude is not None:
        subset = subset.loc[subset["amplitude_px"] == amplitude]
    if len(subset) != 1:
        raise PlanError(
            f"expected exactly one row for scope={scope}, font={font_id}, "
            f"role={role}, amplitude={amplitude}; found {len(subset)}"
        )
    return subset.iloc[0]


def _claim_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = Counter(str(value) for value in frame["classification"])
    return {
        "supported": int(counts.get("supported", 0)),
        "suggestive": int(counts.get("suggestive", 0)),
        "unsupported": int(counts.get("unsupported", 0)),
    }


def _claim_count_text(counts: Mapping[str, int]) -> str:
    return f"{counts['supported']}/{counts['suggestive']}/{counts['unsupported']}"


def _stable_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "undefined"
        return f"{value:.15g}"
    return str(value)


def _values_match(published: Any, reproduced: Any) -> bool:
    if isinstance(published, int | float) and not isinstance(published, bool):
        if not isinstance(reproduced, int | float) or isinstance(reproduced, bool):
            return False
        return math.isclose(float(published), float(reproduced), rel_tol=1e-9, abs_tol=1e-12)
    return str(published) == str(reproduced)


def _evidence_snapshot(inputs: PlanInputs) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(
        evidence_id: str,
        family: str,
        source_key: str,
        scope: str,
        metric: str,
        published: Any,
        reproduced: Any,
        unit: str,
        note: str,
    ) -> None:
        rows.append(
            {
                "evidence_id": evidence_id,
                "family": family,
                "source_key": source_key,
                "source_sha256": inputs.input_hashes[source_key],
                "scope": scope,
                "metric": metric,
                "published_value": _stable_value(published),
                "reproduced_value": _stable_value(reproduced),
                "unit": unit,
                "reproduced": _values_match(published, reproduced),
                "note": note,
            }
        )

    m3_splits = _split_counts(inputs.m3_sources)
    add(
        "E01",
        "M3",
        "m3_summary",
        "dataset",
        "sources",
        _summary_count(inputs.m3_summary, "sources", "M3 summary"),
        len(inputs.m3_sources),
        "count",
        "Frozen development dataset source count.",
    )
    add(
        "E02",
        "M3",
        "m3_summary",
        "dataset",
        "rows",
        _summary_count(inputs.m3_summary, "samples", "M3 summary"),
        len(inputs.m3_manifest),
        "count",
        "Frozen development dataset row count.",
    )
    for split, expected in (("train", 36), ("val", 12), ("test", 12)):
        add(
            f"E03-{split}",
            "M3",
            "m3_sources",
            f"split:{split}",
            "sources",
            expected,
            m3_splits.get(split, 0),
            "count",
            "Source-grouped split count; no source crosses splits.",
        )
    add(
        "E04",
        "M3",
        "m3_summary",
        "fonts",
        "font_count",
        len(M3_EXPECTED_FONTS),
        inputs.m3_sources["font_id"].nunique(),
        "count",
        "Three familiar fonts are shared by every M3 split.",
    )

    m7b_per_font = inputs.m7b_sources.groupby("font_id")["source_id"].nunique().to_dict()
    add(
        "E05",
        "M7B",
        "m7b_summary",
        "dataset",
        "sources",
        _summary_count(inputs.m7b_summary, "sources", "M7B summary"),
        len(inputs.m7b_sources),
        "count",
        "Held-out-font benchmark source count.",
    )
    add(
        "E06",
        "M7B",
        "m7b_summary",
        "dataset",
        "rows",
        _summary_count(inputs.m7b_summary, "rows", "M7B summary"),
        len(inputs.m7b_manifest),
        "count",
        "Held-out-font benchmark row count.",
    )
    add(
        "E07",
        "M7B",
        "m7b_predictions",
        "all_primary",
        "primary_rows",
        1140,
        int((inputs.m7b_predictions["protocol_scope"] == "primary").sum()),
        "count",
        "Primary rows exclude clean and 0.5 px elastic diagnostics.",
    )
    add(
        "E08",
        "M7B",
        "m7b_predictions",
        "diagnostic",
        "diagnostic_rows",
        120,
        int((inputs.m7b_predictions["protocol_scope"] == "diagnostic").sum()),
        "count",
        "Diagnostic rows are clean and 0.5 px elastic only.",
    )
    for font in M7B_EXPECTED_FONTS:
        add(
            f"E09-{font}",
            "M7B",
            "m7b_sources",
            f"font:{font}",
            "sources",
            20,
            m7b_per_font.get(font, 0),
            "count",
            "Each held-out font contributes 20 independent sources.",
        )

    m7b_all_primary = _table_row(
        inputs.m7b_overall_metrics, scope="all_primary", font_id="all", role="primary"
    )
    m7b_all_primary_recomputed = _recompute_m7b(inputs.m7b_predictions, "all_primary")
    for index, metric in enumerate(("mae", "rmse", "r2"), start=10):
        add(
            f"E{index:02d}",
            "M7B",
            "m7b_overall_metrics",
            "all_primary",
            metric,
            float(m7b_all_primary[metric]),
            float(m7b_all_primary_recomputed[metric]),
            "metric",
            "All-primary metric reproduced from the frozen prediction rows.",
        )
    for index, font in enumerate(M7B_EXPECTED_FONTS, start=13):
        published = float(
            _table_row(
                inputs.m7b_font_metrics,
                scope="all_primary",
                font_id=font,
                role="primary",
            )["rmse"]
        )
        reproduced = float(_recompute_m7b(inputs.m7b_predictions, "all_primary", font)["rmse"])
        add(
            f"E{index:02d}",
            "M7B",
            "m7b_font_metrics",
            f"all_primary:{font}",
            "rmse",
            published,
            reproduced,
            "metric",
            "Per-font all-primary RMSE; font groups remain unpaired.",
        )
    for scope, eid in (
        ("original_blur_noise", "E16"),
        ("geometric_primary", "E17"),
        ("interference", "E18"),
    ):
        published = float(
            _table_row(
                inputs.m7b_overall_metrics,
                scope=scope,
                font_id="all",
                role="primary",
            )["rmse"]
        )
        reproduced = float(_recompute_m7b(inputs.m7b_predictions, scope)["rmse"])
        add(
            eid,
            "M7B",
            "m7b_overall_metrics",
            scope,
            "rmse",
            published,
            reproduced,
            "metric",
            "All-font family RMSE reproduced from the frozen rows.",
        )

    m7c_all_primary = _table_row(
        inputs.m7c_family_metrics,
        scope="all_primary",
        font_id="all",
        role="primary",
    )
    for index, metric in enumerate(("mae", "rmse", "r2"), start=19):
        add(
            f"E{index:02d}",
            "M7C",
            "m7c_family_metrics",
            "all_primary",
            metric,
            float(m7c_all_primary[metric]),
            float(m7b_all_primary_recomputed[metric]),
            "metric",
            "M7C all-primary metric independently reproduced from M7B predictions.",
        )
    for index, font in enumerate(M7B_EXPECTED_FONTS, start=22):
        published = float(
            _table_row(
                inputs.m7c_font_metrics,
                scope="all_primary",
                font_id=font,
                role="primary",
            )["rmse"]
        )
        reproduced = float(_recompute_m7b(inputs.m7b_predictions, "all_primary", font)["rmse"])
        add(
            f"E{index:02d}",
            "M7C",
            "m7c_font_metrics",
            f"all_primary:{font}",
            "rmse",
            published,
            reproduced,
            "metric",
            "M7C per-font RMSE reproduced from M7B predictions.",
        )
    for index, font in enumerate(M7B_EXPECTED_FONTS, start=25):
        published = float(
            _table_row(
                inputs.m7c_font_metrics,
                scope="original_blur_noise",
                font_id=font,
                role="primary",
            )["rmse"]
        )
        reproduced = float(
            _recompute_m7b(inputs.m7b_predictions, "original_blur_noise", font)["rmse"]
        )
        add(
            f"E{index:02d}",
            "M7C",
            "m7c_font_metrics",
            f"original_blur_noise:{font}",
            "rmse",
            published,
            reproduced,
            "metric",
            "Blur/noise error is strongly uneven across the three held-out fonts.",
        )
    for index, amplitude in enumerate((1.0, 2.0, 3.0), start=28):
        scope = f"geometric_primary_{int(amplitude)}px"
        published = float(
            inputs.m7c_severity_metrics.loc[
                (inputs.m7c_severity_metrics["font_id"] == "all")
                & (inputs.m7c_severity_metrics["amplitude_px"] == amplitude)
                & (inputs.m7c_severity_metrics["protocol_role"] == "primary")
            ].iloc[0]["rmse"]
        )
        reproduced = float(_recompute_m7b(inputs.m7b_predictions, scope)["rmse"])
        add(
            f"E{index:02d}",
            "M7C",
            "m7c_severity_metrics",
            scope,
            "rmse",
            published,
            reproduced,
            "metric",
            "Aggregate geometric error hides a non-monotonic severity pattern.",
        )

    primary = inputs.m7b_predictions.loc[
        inputs.m7b_predictions["protocol_scope"] == "primary"
    ].copy()
    primary["squared_error"] = (primary[TARGET] - primary["prediction"]) ** 2
    source_sse = primary.groupby("source_id", sort=True)["squared_error"].sum()
    shares = source_sse / source_sse.sum()
    top3_share = float(shares.nlargest(3).sum())
    hhi = float(np.sum(shares.to_numpy(dtype=np.float64) ** 2))
    effective_sources = 1.0 / hhi
    concentration = inputs.m7c_error_concentration.loc[
        (inputs.m7c_error_concentration["scope"] == "all_primary")
        & (inputs.m7c_error_concentration["font_id"] == "all")
    ].iloc[0]
    add(
        "E31",
        "M7C",
        "m7c_error_concentration",
        "all_primary",
        "top3_share",
        float(concentration["top_k_share"]),
        top3_share,
        "share",
        "Source-level squared-error concentration is descriptive.",
    )
    add(
        "E32",
        "M7C",
        "m7c_error_concentration",
        "all_primary",
        "effective_sources",
        float(concentration["effective_sources"]),
        effective_sources,
        "count",
        "Effective source count is 1/HHI, not a significance test.",
    )

    for eid, scope in (("E33", "original_blur_noise"), ("E34", "geometric_primary")):
        comparison = inputs.m7c_benchmark_comparison.loc[
            (inputs.m7c_benchmark_comparison["m7b_scope"] == scope)
            & (inputs.m7c_benchmark_comparison["metric"] == "rmse")
        ].iloc[0]
        m7b_value = float(_recompute_m7b(inputs.m7b_predictions, scope)["rmse"])
        familiar_value = float(_recompute_m5c(inputs.m5c_test_predictions, scope)["rmse"])
        add(
            eid,
            "M7C",
            "m7c_benchmark_comparison",
            scope,
            "difference_rmse",
            float(comparison["difference"]),
            m7b_value - familiar_value,
            "metric",
            "Unmatched familiar-versus-held-out difference; descriptive only.",
        )

    for scope, eid in (
        ("m3_test_overall", "E35"),
        ("original_blur_noise", "E36"),
        ("geometric_primary", "E37"),
        ("clean_reference", "E38"),
    ):
        published = inputs.m5c_test_slice_metrics.loc[
            (inputs.m5c_test_slice_metrics["model"] == M7C_CHAMPION_NAME)
            & (inputs.m5c_test_slice_metrics["split"] == "test")
            & (inputs.m5c_test_slice_metrics["scope"] == scope)
        ].iloc[0]
        reproduced = _recompute_m5c(inputs.m5c_test_predictions, scope)
        add(
            eid,
            "M5C",
            "m5c_test_slice_metrics",
            scope,
            "rmse",
            float(published["rmse"]),
            float(reproduced["rmse"]),
            "metric",
            "Familiar-font champion metric reproduced from stored test predictions.",
        )

    add(
        "E39",
        "M5C",
        "m5c_model_artifact",
        "model",
        "artifact_sha256",
        M7C_CHAMPION_ARTIFACT_SHA256,
        inputs.input_hashes["m5c_model_artifact"],
        "sha256",
        "Frozen champion identity; no model is loaded, fitted or recalibrated by M7D.",
    )
    add(
        "E40",
        "M5C",
        "m5c_model_card",
        "model",
        "card_sha256",
        M7C_CHAMPION_CARD_SHA256,
        inputs.input_hashes["m5c_model_card"],
        "sha256",
        "Frozen champion model-card identity.",
    )
    m7a_counts = _claim_counts(inputs.m7a_claims)
    m7c_counts = _claim_counts(inputs.m7c_claims)
    add(
        "E41",
        "M7A",
        "m7a_claim_evidence",
        "claims",
        "supported/suggestive/unsupported",
        _claim_count_text(
            {
                "supported": int(
                    inputs.m7a_run_metadata.get("claims", {}).get("counts", {}).get("supported", -1)
                ),
                "suggestive": int(
                    inputs.m7a_run_metadata.get("claims", {})
                    .get("counts", {})
                    .get("suggestive", -1)
                ),
                "unsupported": int(
                    inputs.m7a_run_metadata.get("claims", {})
                    .get("counts", {})
                    .get("unsupported", -1)
                ),
            }
        ),
        _claim_count_text(m7a_counts),
        "count",
        "M7A audit claim classifications, recomputed from the claim table.",
    )
    add(
        "E42",
        "M7C",
        "m7c_claim_evidence",
        "claims",
        "supported/suggestive/unsupported",
        _claim_count_text(
            {
                "supported": int(
                    inputs.m7c_run_metadata.get("summary", {})
                    .get("claims", {})
                    .get("supported", -1)
                ),
                "suggestive": int(
                    inputs.m7c_run_metadata.get("summary", {})
                    .get("claims", {})
                    .get("suggestive", -1)
                ),
                "unsupported": int(
                    inputs.m7c_run_metadata.get("summary", {})
                    .get("claims", {})
                    .get("unsupported", -1)
                ),
            }
        ),
        _claim_count_text(m7c_counts),
        "count",
        "M7C audit claim classifications, recomputed from the claim table.",
    )
    return pd.DataFrame(rows)


def _direction_assessment(inputs: PlanInputs) -> pd.DataFrame:
    comparison = inputs.m7c_benchmark_comparison
    original = comparison.loc[
        (comparison["m7b_scope"] == "original_blur_noise") & (comparison["metric"] == "rmse")
    ].iloc[0]
    geometric = comparison.loc[
        (comparison["m7b_scope"] == "geometric_primary") & (comparison["metric"] == "rmse")
    ].iloc[0]
    rows = [
        {
            "direction": "A",
            "title": DIRECTION_TITLES["A"],
            "scientific_value": (
                "Highest: widens the held-out font set and can test whether the M7C "
                "three-font pattern persists across a pre-specified, typographically "
                "diverse roster. It directly targets the unsupported arbitrary-unseen-font "
                "and font-population claims. Three of the seven held-out faces were "
                "already used in M7B as an evaluation set, so four identities are "
                "genuinely new; the roster widens the held-out set rather than "
                "replacing it."
            ),
            "uncertainty_resolved": (
                "Partially resolves M7C C06/C13 and, through matching, C10. It does not "
                "resolve renderer, resolution, format, human-readability or security claims."
            ),
            "required_independent_units": (
                "Ten purposive font identities; 30 matched text cases; 300 font-specific "
                "sources; 5,700 primary variants. Font identity, text case and source are "
                "kept distinct."
            ),
            "implementation_cost": (
                "Moderate-high: four new held-out font files beyond M7B, a new matched-case "
                "generator, 6,300 rows and a zero-shot evaluation. No new model is trained."
            ),
            "leakage_risk": (
                "Manageable with font-level holdout, matched cases kept whole, and no use of "
                "held-out labels for fitting, calibration or selection. The ten fonts remain "
                "purposive, so population inference is still unavailable."
            ),
            "reproducibility_risk": (
                "Moderate: font files are machine-dependent, so exact path/hash/glyph checks "
                "and fail-closed generation are mandatory."
            ),
            "primary_metrics": (
                "Matched held-out-minus-familiar all-primary RMSE contrast; per-font RMSE; "
                "family and geometric-severity slices."
            ),
            "supportable_conclusions": (
                "Conditional statements about the frozen champion across the ten tested "
                "typefaces and matched corruption realisations."
            ),
            "cannot_establish": (
                "Arbitrary-font or font-population generalization, causal font effects, "
                "unseen-renderer generalization, human readability, solve rates or "
                "security; nor independent confirmation of the M7B three-font pattern, "
                "because three of the seven held-out faces were already used in M7B."
            ),
            "decision": "selected",
            "selected": True,
            "rationale": (
                "A is the only direction that expands the font-identity axis while retaining "
                "the internal-validity control that M7C showed was missing. The M7C unmatched "
                "blur/noise difference is +0.0726 RMSE and the matched geometric difference "
                "is -0.0002; these cannot be interpreted as font effects without matching. "
                "Adopting C's matched cases inside A avoids repeating the same three fonts "
                "without fixing the confound."
            ),
        },
        {
            "direction": "B",
            "title": DIRECTION_TITLES["B"],
            "scientific_value": (
                "High in principle: changes the renderer rather than the typeface and probes "
                "a distinct condition axis. It would address the unsupported unseen-renderer "
                "claim that M7A/M7C identify."
            ),
            "uncertainty_resolved": (
                "Would address M7C C07, but only for the specific renderer pair. It would not "
                "resolve the unmatched font comparison or the three-font limitation."
            ),
            "required_independent_units": (
                "At least two renderer implementations, with renderer-level replication "
                "ideally including multiple engines, canvas sizes and formats."
            ),
            "implementation_cost": (
                "High: a second renderer, new calibration of glyph placement and corruption "
                "behaviour, plus a separate reference-generation and verification path."
            ),
            "leakage_risk": (
                "High confounding: changing renderer also changes antialiasing, glyph "
                "placement, font fallback and effective spatial frequency. With two "
                "renderers there is no meaningful renderer-population inference."
            ),
            "reproducibility_risk": (
                "High: renderer versions and platform libraries are harder to pin than a "
                "single TTF file."
            ),
            "primary_metrics": (
                "Renderer-paired all-primary and family RMSE contrasts on matched sources."
            ),
            "supportable_conclusions": (
                "Descriptive statements about the tested renderer pair, conditional on the "
                "same fonts, corruption ladders and target."
            ),
            "cannot_establish": (
                "General renderer robustness, font-population effects, human readability or "
                "security."
            ),
            "decision": "deferred",
            "selected": False,
            "rationale": (
                "B is scientifically important but a larger and more confounded step. The "
                "current evidence cannot separate renderer from antialiasing, placement and "
                "format, and two renderers are not a renderer sample. It should follow the "
                "font-identity work as a separately versioned protocol."
            ),
        },
        {
            "direction": "C",
            "title": DIRECTION_TITLES["C"],
            "scientific_value": (
                "High internal validity: matches text strings and corruption seeds across "
                "familiar and held-out fonts, directly addressing M7C's unmatched comparison."
            ),
            "uncertainty_resolved": (
                "Would resolve M7C C10 for the three existing held-out fonts. It would not "
                "expand the font-identity set and would leave C06/C13 unresolved."
            ),
            "required_independent_units": (
                "Six fonts (three familiar, three held-out), 30 matched text cases and 180 sources."
            ),
            "implementation_cost": (
                "Moderate: a matched-case generator for six already pinned fonts and 3,780 "
                "rows. No new model is trained."
            ),
            "leakage_risk": (
                "Low if cases are kept whole and held-out labels are never used for fitting. "
                "The three held-out fonts remain a purposive sample."
            ),
            "reproducibility_risk": ("Low: all six font files already have recorded hashes."),
            "primary_metrics": (
                "Matched held-out-minus-familiar all-primary RMSE on three held-out fonts."
            ),
            "supportable_conclusions": (
                "A matched statement about the three existing held-out fonts and their "
                "corruption-specific contrasts."
            ),
            "cannot_establish": (
                "Broader unseen-font behavior, font-population generalization or renderer "
                "generalization."
            ),
            "decision": "matching_control_incorporated_into_A",
            "selected": False,
            "rationale": (
                "C is the best pure control design but too incremental as the primary "
                "direction: it reuses the same three held-out fonts that M7C already shows "
                "cannot support a font-population reading. Its matched-case and matched-"
                "corruption logic is therefore adopted as a design control inside A, not "
                "discarded."
            ),
        },
    ]
    frame = pd.DataFrame(rows)
    frame.attrs["original_difference"] = float(original["difference"])
    frame.attrs["geometric_difference"] = float(geometric["difference"])
    return frame


def _research_questions() -> pd.DataFrame:
    rows = [
        {
            "question_id": "RQ1",
            "role": "primary",
            "question": (
                "Does the frozen champion's all-primary RMSE differ between held-out and "
                "familiar fonts when text case and corruption realisations are matched?"
            ),
            "estimand": (
                "Average of the 21 matched held-out/familiar font-pair contrasts, each "
                "computed on the same 30 text cases; signed direction is held-out minus "
                "familiar."
            ),
            "independent_unit": "matched_text_case for uncertainty; font_pair for descriptive spread",
            "scope": "all_primary",
            "status": "planned",
            "interpretation": (
                "A conditional contrast for the ten-font roster, not a causal font effect or "
                "a font-population interval."
            ),
        },
        {
            "question_id": "RQ2",
            "role": "primary_supporting",
            "question": "How does all-primary RMSE vary across the pre-specified ten fonts?",
            "estimand": "Per-font RMSE with matched-case grouped intervals and leave-one-font-out range.",
            "independent_unit": "font_identity",
            "scope": "all_primary",
            "status": "planned",
            "interpretation": "Descriptive across the roster; no population interval.",
        },
        {
            "question_id": "RQ3",
            "role": "secondary",
            "question": ("Do font contrasts differ by corruption family and geometric severity?"),
            "estimand": "Per-font and paired held-out/familiar RMSE by family and 1/2/3 px elastic rung.",
            "independent_unit": "matched_text_case",
            "scope": "blur, noise, thin lines, arc strokes, geometric primary",
            "status": "planned",
            "interpretation": (
                "Predeclared family/severity slices; no post-hoc search over all possible "
                "subgroups and no universal ranking."
            ),
        },
        {
            "question_id": "RQ4",
            "role": "secondary",
            "question": "Is source-level error concentrated in a few sources?",
            "estimand": "Top-1/top-3 squared-error share, HHI and effective source count per font.",
            "independent_unit": "source",
            "scope": "all_primary",
            "status": "planned",
            "interpretation": "Descriptive influence summary; no source is trimmed or refit.",
        },
        {
            "question_id": "RQ5",
            "role": "diagnostic",
            "question": "Are target distributions non-degenerate for every primary slice?",
            "estimand": "Target n, mean, SD, quantiles and near-degeneracy flag.",
            "independent_unit": "source for the target distribution",
            "scope": "all scopes",
            "status": "planned",
            "interpretation": (
                "A primary slice with target SD below 0.005 is flagged; R² is suppressed and "
                "the slice cannot support primary ranking claims."
            ),
        },
        {
            "question_id": "RQ6",
            "role": "out_of_scope",
            "question": "Does the champion generalize to unseen renderers or formats?",
            "estimand": "Not estimated by this design; renderer is fixed.",
            "independent_unit": "renderer_identity (not sampled)",
            "scope": "none",
            "status": "not_answered",
            "interpretation": "Requires a separately authorized renderer benchmark.",
        },
        {
            "question_id": "RQ7",
            "role": "future",
            "question": "Would a future model improve on the frozen champion?",
            "estimand": "A predeclared margin on an untouched final test set, with retention constraints.",
            "independent_unit": "untouched test source groups",
            "scope": "future authorized experiment",
            "status": "not_authorized",
            "interpretation": (
                "M7D does not train or compare models. Any future model comparison needs a "
                "separate protocol and a test set never used for fitting, calibration or selection."
            ),
        },
    ]
    return pd.DataFrame(rows)


def _independent_units() -> pd.DataFrame:
    rows = [
        {
            "unit": "font_identity",
            "definition": "One typeface file/face selected in the pre-specified ten-font roster.",
            "independence": (
                "One purposive identity; not exchangeable with unobserved fonts and not a "
                "random draw from a font universe."
            ),
            "used_for": "Font-level descriptive spread, leave-one-font-out sensitivity, roster coverage.",
            "not_valid_for": "Font-population confidence intervals or causal font effects.",
            "grouping_key": "font_id",
        },
        {
            "unit": "matched_text_case",
            "definition": (
                "One text string sampled once and rendered in every font with the same render "
                "and corruption seed schedule."
            ),
            "independence": "The pairing unit for familiar-versus-held-out contrasts.",
            "used_for": "Primary matched contrast, paired family/severity contrasts, uncertainty.",
            "not_valid_for": "An independent sample of fonts.",
            "grouping_key": "case_id",
        },
        {
            "unit": "source",
            "definition": "One clean render of one text case in one font.",
            "independence": (
                "All corrupted variants of a source are correlated; the source is the "
                "resampling unit for per-font metrics."
            ),
            "used_for": "Per-font source metrics, source-grouped uncertainty, target distributions.",
            "not_valid_for": "Treating variants as independent observations.",
            "grouping_key": "source_id",
        },
        {
            "unit": "clean_reference",
            "definition": "The exact PNG clean render used as the reference for one source.",
            "independence": "One per source; not a separate experimental condition.",
            "used_for": "SSIM target calculation and immutability checks.",
            "not_valid_for": "An additional sample or an independent model input.",
            "grouping_key": "source_id",
        },
        {
            "unit": "corrupted_variant",
            "definition": "One corruption family/severity applied to one source.",
            "independence": "Nested within source; correlated by construction.",
            "used_for": "Row-level targets, family and severity summaries.",
            "not_valid_for": "Confidence intervals, effective sample size or significance tests.",
            "grouping_key": "source_id + variant_id",
        },
        {
            "unit": "renderer",
            "definition": "Pillow ImageDraw + FreeType at 192x64, 8-bit grayscale PNG.",
            "independence": "One fixed condition, not a sampled factor.",
            "used_for": "Scope statement and reproducibility metadata.",
            "not_valid_for": "Renderer-generalization claims.",
            "grouping_key": "renderer_id",
        },
        {
            "unit": "benchmark_row",
            "definition": "One manifest row with one image, reference, target and prediction.",
            "independence": "Nested in variant, source, case and font.",
            "used_for": "Aggregation only.",
            "not_valid_for": "Effective sample size or independent replication.",
            "grouping_key": "sample_id",
        },
        {
            "unit": "font_pair",
            "definition": "One held-out font paired with one familiar font on the same cases.",
            "independence": "Pairs share fonts and cases; they are a descriptive construction.",
            "used_for": "Primary contrast construction and spread across 21 pairs.",
            "not_valid_for": "Independent replication or population inference.",
            "grouping_key": "heldout_font_id + familiar_font_id",
        },
        {
            "unit": "source_group",
            "definition": "All rows belonging to one font-specific source.",
            "independence": "The bootstrap resampling unit; all variants stay together.",
            "used_for": "Matched-case grouped percentile bootstrap.",
            "not_valid_for": "Font-level population inference.",
            "grouping_key": "source_id",
        },
    ]
    return pd.DataFrame(rows)


def _split_hierarchy() -> pd.DataFrame:
    rows = [
        {
            "level": "benchmark_eval",
            "scope": "M7D proposed benchmark",
            "rule": "All 300 planned sources are in one untouched eval split; no training, calibration or selection.",
            "grouping_key": "source_id",
            "train_allowed": False,
            "eval_allowed": True,
            "leakage_control": "Zero-shot evaluation of the frozen M5C champion only.",
            "notes": "The benchmark is not a training dataset.",
        },
        {
            "level": "font_identity",
            "scope": "future model development",
            "rule": "Held-out font identities never appear in training, validation, feature selection or calibration.",
            "grouping_key": "font_id",
            "train_allowed": False,
            "eval_allowed": True,
            "leakage_control": "Font-level holdout; exact font SHA-256 checked against development data.",
            "notes": "Ten fonts are still a purposive roster.",
        },
        {
            "level": "matched_text_case",
            "scope": "future source-level split",
            "rule": "If a source-level split is required, all font renders of a case stay in the same partition.",
            "grouping_key": "case_id",
            "train_allowed": True,
            "eval_allowed": True,
            "leakage_control": "Case-grouped partitioning; no near-duplicate text across partitions.",
            "notes": "Matched cases exist to be paired, not to be split across training and test.",
        },
        {
            "level": "source",
            "scope": "all analyses",
            "rule": "All variants of a source stay together in every split, metric and bootstrap replicate.",
            "grouping_key": "source_id",
            "train_allowed": True,
            "eval_allowed": True,
            "leakage_control": "Source-grouped splits and source-grouped uncertainty.",
            "notes": "Variants are never treated as independent sources.",
        },
        {
            "level": "variant",
            "scope": "all analyses",
            "rule": "A variant is nested within its source; never split or resampled independently.",
            "grouping_key": "source_id + variant_id",
            "train_allowed": True,
            "eval_allowed": True,
            "leakage_control": "Source grouping; no row-level bootstrap.",
            "notes": "Corrupted variants share the same clean reference.",
        },
        {
            "level": "future_final_test",
            "scope": "future authorized training",
            "rule": "Reserve at least two held-out font identities and disjoint text cases for a final test set never used for selection.",
            "grouping_key": "font_id + case_id",
            "train_allowed": False,
            "eval_allowed": True,
            "leakage_control": "Two-axis holdout: font identity and text case; test labels never used for fitting or calibration.",
            "notes": "This is a future protocol requirement, not part of M7D.",
        },
        {
            "level": "renderer",
            "scope": "M7D proposed benchmark",
            "rule": "Keep the frozen renderer, canvas, image mode and format fixed.",
            "grouping_key": "renderer_id",
            "train_allowed": False,
            "eval_allowed": True,
            "leakage_control": "No renderer variation; a renderer study is a separate version.",
            "notes": "Direction B is deferred.",
        },
    ]
    return pd.DataFrame(rows)


def _endpoints() -> pd.DataFrame:
    rows = [
        {
            "endpoint_id": "P1",
            "role": "primary",
            "metric": "rmse",
            "scope": "all_primary",
            "unit": "matched_text_case; font_pair described separately",
            "contrast": "held-out minus familiar, averaged over 21 font pairs",
            "uncertainty": "matched-case grouped percentile bootstrap, 2,000 replicates, 95%",
            "degenerate_policy": "no row-level CI; target SD >= 0.005; no optional stopping",
            "success_interpretation": (
                "Direction and signed interval conditional on the ten-font roster; zero "
                "overlap does not establish equivalence or a causal font effect."
            ),
            "status": "planned",
        },
        {
            "endpoint_id": "P1d",
            "role": "primary_supporting",
            "metric": "mae, rmse, r2",
            "scope": "all_primary",
            "unit": "font_identity for descriptive spread; source for per-font uncertainty",
            "contrast": "per font",
            "uncertainty": "source-grouped percentile bootstrap; leave-one-font-out range",
            "degenerate_policy": "R² suppressed when target SD < 0.005 or undefined",
            "success_interpretation": "Descriptive per-font profile; not a population interval.",
            "status": "planned",
        },
        {
            "endpoint_id": "S1",
            "role": "secondary",
            "metric": "rmse",
            "scope": "original_blur_noise",
            "unit": "matched_text_case",
            "contrast": "held-out minus familiar, per font pair",
            "uncertainty": "matched-case grouped percentile bootstrap",
            "degenerate_policy": "no row-level CI; report target distribution",
            "success_interpretation": "Blur/noise-specific contrast; no pooling with geometric family.",
            "status": "planned",
        },
        {
            "endpoint_id": "S2",
            "role": "secondary",
            "metric": "rmse",
            "scope": "gaussian_blur, gaussian_noise, thin_lines, arc_strokes, interference",
            "unit": "matched_text_case",
            "contrast": "per font and held-out minus familiar",
            "uncertainty": "matched-case grouped percentile bootstrap",
            "degenerate_policy": "predeclared families only; no search over arbitrary subgroups",
            "success_interpretation": "Family-specific profile; no universal ranking.",
            "status": "planned",
        },
        {
            "endpoint_id": "S3",
            "role": "secondary",
            "metric": "rmse",
            "scope": "geometric_primary and 1/2/3 px rungs",
            "unit": "matched_text_case",
            "contrast": "per font and held-out minus familiar",
            "uncertainty": "matched-case grouped percentile bootstrap",
            "degenerate_policy": "0.5 px excluded from primary; 1 px R² interpreted with target SD",
            "success_interpretation": "Severity-specific profile; aggregate cannot substitute for rungs.",
            "status": "planned",
        },
        {
            "endpoint_id": "S4",
            "role": "diagnostic",
            "metric": "mae, rmse",
            "scope": "clean_reference",
            "unit": "source",
            "contrast": "per font",
            "uncertainty": "source-grouped descriptive interval",
            "degenerate_policy": "clean target is constant; R² undefined; diagnostic only",
            "success_interpretation": "Clean-reference retention check; not a degradation endpoint.",
            "status": "planned",
        },
        {
            "endpoint_id": "S5",
            "role": "diagnostic",
            "metric": "target mean, sd, quantiles",
            "scope": "all scopes",
            "unit": "source and row",
            "contrast": "per font/family/severity",
            "uncertainty": "none; descriptive distribution",
            "degenerate_policy": "target SD < 0.005 flags near-degenerate; R² suppressed",
            "success_interpretation": "Confirms whether a slice can support its stated metric.",
            "status": "planned",
        },
        {
            "endpoint_id": "S6",
            "role": "secondary",
            "metric": "top1/top3 share, HHI, effective sources",
            "scope": "all_primary",
            "unit": "source",
            "contrast": "per font",
            "uncertainty": "descriptive; no resampling of row-level errors",
            "degenerate_policy": "no source trimming; report top source identity",
            "success_interpretation": "Influence summary only; not a significance test.",
            "status": "planned",
        },
        {
            "endpoint_id": "S7",
            "role": "secondary",
            "metric": "font-level range and SD of per-font RMSE",
            "scope": "all_primary",
            "unit": "font_identity",
            "contrast": "across ten fonts",
            "uncertainty": "leave-one-font-out sensitivity; no population interval",
            "degenerate_policy": "n=10 is explicitly insufficient for precise font-population inference",
            "success_interpretation": "Roster description only; cannot support arbitrary-font generalization.",
            "status": "planned",
        },
        {
            "endpoint_id": "S8",
            "role": "secondary",
            "metric": "paired within-source differences",
            "scope": "noise minus blur; thin lines minus arc strokes; geometric severity adjacent",
            "unit": "source",
            "contrast": "within-source",
            "uncertainty": "source-grouped percentile bootstrap",
            "degenerate_policy": "pairs require identical source sets; otherwise ineligible",
            "success_interpretation": "Corruption-family/severity profile; no cross-font causal claim.",
            "status": "planned",
        },
        {
            "endpoint_id": "F1",
            "role": "future",
            "metric": "predeclared improvement margin",
            "scope": "future untouched test set",
            "unit": "untouched test source groups",
            "contrast": "future model minus frozen champion",
            "uncertainty": "grouped interval and paired comparison",
            "degenerate_policy": "no fitting, calibration or selection on test labels",
            "success_interpretation": (
                "Only a separately authorized experiment can define a practically meaningful "
                "margin; M7D does not invent a universal SSIM threshold."
            ),
            "status": "not_authorized",
        },
    ]
    return pd.DataFrame(rows)


def _sample_size_scenarios(inputs: PlanInputs, config: PlanConfig) -> pd.DataFrame:
    source_metrics = inputs.m7c_source_metrics.loc[
        inputs.m7c_source_metrics["scope"] == "all_primary"
    ]
    if source_metrics.empty:
        raise PlanError("M7C source metrics do not contain all_primary rows")
    source_sd = float(source_metrics["rmse"].std(ddof=1))
    if not math.isfinite(source_sd) or source_sd <= 0.0:
        raise PlanError("cannot derive a positive source-level planning SD")
    multiplier = config.precision.interval_multiplier
    rows: list[dict[str, Any]] = []
    for count in config.precision.text_case_counts:
        standard_error = source_sd / math.sqrt(count)
        rows.append(
            {
                "scenario_id": f"source-mean-n{count}",
                "level": "matched_text_case_mean_rmse",
                "n_units": count,
                "assumed_sd": source_sd,
                "assumed_sd_basis": (
                    "Observed M7C all-primary per-source RMSE SD; descriptive planning input, "
                    "not a population parameter."
                ),
                "expected_se": standard_error,
                "ci95_halfwidth": standard_error * multiplier,
                "interval_multiplier": multiplier,
                "selected": count == config.plan.n_text_cases,
                "notes": (
                    "Normal-approximation precision for a mean of source-level RMSE; the "
                    "benchmark will use a grouped percentile bootstrap."
                ),
            }
        )
    for count in config.precision.text_case_counts:
        for assumed_sd in config.precision.paired_sd_assumptions:
            standard_error = assumed_sd / math.sqrt(count)
            rows.append(
                {
                    "scenario_id": f"paired-n{count}-sd{assumed_sd:.3f}",
                    "level": "matched_font_pair_difference",
                    "n_units": count,
                    "assumed_sd": assumed_sd,
                    "assumed_sd_basis": (
                        "Illustrative within-case paired-difference SD; no matched cross-font "
                        "data exist yet, so this is a sensitivity scenario, not an estimate."
                    ),
                    "expected_se": standard_error,
                    "ci95_halfwidth": standard_error * multiplier,
                    "interval_multiplier": multiplier,
                    "selected": count == config.plan.n_text_cases,
                    "notes": (
                        "The primary contrast pairs fonts within a text case; smaller paired "
                        "SD than the unpaired source SD is the expected but unverified gain."
                    ),
                }
            )
    for count in config.precision.font_counts:
        for assumed_sd in config.precision.font_sd_assumptions:
            standard_error = assumed_sd / math.sqrt(count)
            rows.append(
                {
                    "scenario_id": f"font-mean-n{count}-sd{assumed_sd:.3f}",
                    "level": "font_mean_rmse",
                    "n_units": count,
                    "assumed_sd": assumed_sd,
                    "assumed_sd_basis": (
                        "Illustrative between-font SD; the three observed M7B fonts are not a "
                        "font sample and cannot supply a population variance."
                    ),
                    "expected_se": standard_error,
                    "ci95_halfwidth": standard_error * multiplier,
                    "interval_multiplier": multiplier,
                    "selected": count == 10,
                    "notes": (
                        "Descriptive font-roster precision only. The normal multiplier "
                        "understates small-df t uncertainty; no font-population claim is made."
                    ),
                }
            )
    return pd.DataFrame(rows)


def _leakage_controls() -> pd.DataFrame:
    rows = [
        (
            "L01",
            "Font-level holdout",
            "Held-out fonts absent from M3 training/validation/test and from model selection.",
            "Check exact font keys and SHA-256 against M3 sources and config.",
            "Stop dataset generation.",
        ),
        (
            "L02",
            "Frozen model identity",
            "Use the exact M5C artifact and card hashes.",
            "Pin artifact/card; load no model in M7D; future evaluation checks hashes and schema.",
            "Stop evaluation.",
        ),
        (
            "L03",
            "No test-label fitting",
            "No fitting, recalibration, selection or threshold tuning on held-out labels.",
            "Protocol audit and code review; predictions are generated once from the frozen artifact.",
            "Stop evaluation and report protocol violation.",
        ),
        (
            "L04",
            "Matched-case grouping",
            "All font renders of a case share render and corruption realisations.",
            "Case IDs and derived seeds recorded in the manifest; no optional re-draw.",
            "Stop generation.",
        ),
        (
            "L05",
            "Source grouping",
            "All variants of a source stay together in splits and bootstrap replicates.",
            "Source-grouped split assignment and grouped resampling.",
            "Stop analysis.",
        ),
        (
            "L06",
            "No row-level independence",
            "Rows, variants and pixels are never treated as independent samples.",
            "Uncertainty only at case/source/font levels; no row bootstrap.",
            "Reject the analysis.",
        ),
        (
            "L07",
            "Future training separation",
            "Any future model development gets a separate dataset with untouched font/case test groups.",
            "Separate config, split hash and authorization; final test labels sealed.",
            "Stop future experiment.",
        ),
        (
            "L08",
            "Manifest and split hashes",
            "Record config, manifest, split, font and target hashes.",
            "Fail-closed verification before evaluation.",
            "Stop evaluation.",
        ),
        (
            "L09",
            "No font substitution",
            "A missing or changed font stops the run; no fallback font is chosen after seeing outcomes.",
            "Pre-registered roster and hash gate.",
            "Stop dataset generation.",
        ),
        (
            "L10",
            "Target recomputation",
            "SSIM is recomputed from exact image/reference bytes.",
            "Dataset verifier recomputes every target and clean-reference hash.",
            "Stop evaluation.",
        ),
        (
            "L11",
            "Output protection",
            "Write only to the versioned Git-ignored report directory.",
            "Path allowlist and protected-root fingerprint before/after.",
            "Abort run.",
        ),
        (
            "L12",
            "No post-hoc endpoint search",
            "Only predeclared families, severities and contrasts are primary.",
            "Endpoint table fixed before generation; exploratory slices labelled.",
            "Report as exploratory or reject claim.",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=("control_id", "control", "requirement", "enforcement", "failure_action"),
    )


def _success_criteria() -> pd.DataFrame:
    rows = [
        (
            "G01",
            "Integrity",
            "All 40 pinned inputs match recorded SHA-256 and schemas.",
            "Before plan and before benchmark evaluation.",
            "Stop.",
            "Frozen evidence cannot be trusted.",
        ),
        (
            "G02",
            "Integrity",
            "Ten font files exist, match hashes and cover the full charset.",
            "Before dataset generation.",
            "Stop and revise plan.",
            "No silent fallback to another font.",
        ),
        (
            "G03",
            "Licensing",
            "Local-use/licensing review is recorded for every font before generation.",
            "Before dataset generation.",
            "Stop.",
            "M7D does not assert a license.",
        ),
        (
            "G04",
            "Coverage",
            "300 sources, 5,700 primary rows and 600 diagnostic rows exactly.",
            "After generation, before evaluation.",
            "Stop.",
            "Matched design is incomplete.",
        ),
        (
            "G05",
            "Coverage",
            "30 text cases per font and 10 fonts; every case rendered in every font.",
            "After generation.",
            "Stop.",
            "Matched pairing is incomplete.",
        ),
        (
            "G06",
            "Leakage",
            "No held-out font appears in M3 development data or model selection.",
            "Before evaluation.",
            "Stop.",
            "Unseen-font claim is invalid.",
        ),
        (
            "G07",
            "Leakage",
            "No case/source crosses a future training/test boundary and no variant is split from its source.",
            "Before any future training.",
            "Stop.",
            "Grouped evaluation is invalid.",
        ),
        (
            "G08",
            "Rendering",
            "Two independent generations are byte-identical except runtime metadata; target replay matches every row.",
            "After generation.",
            "Stop.",
            "Provenance is not reproducible.",
        ),
        (
            "G09",
            "Targets",
            "Primary target SD >= 0.005 in every required slice; clean and 0.5 px are diagnostic only.",
            "Before evaluation.",
            "Stop or mark slice diagnostic.",
            "R²/ranking would be unstable.",
        ),
        (
            "G10",
            "Model",
            "Artifact and card hashes match the frozen champion; no fit/calibration call occurs.",
            "Before and during evaluation.",
            "Stop.",
            "Baseline identity is broken.",
        ),
        (
            "G11",
            "Analysis",
            "All primary/secondary endpoints, uncertainty and missing-data policies are reported as pre-registered.",
            "After evaluation.",
            "Report limitation; no optional extension.",
            "Selective reporting is invalid.",
        ),
        (
            "G12",
            "Reproducibility",
            "Plan and benchmark reports/tables replay byte-identically from pinned inputs.",
            "After every run.",
            "Stop.",
            "Claim is not reproducible.",
        ),
        (
            "G13",
            "Interpretation",
            "Null/mixed/negative findings are reported without changing endpoints or thresholds.",
            "Report review.",
            "Reject claim.",
            "Results must not drive the success rule.",
        ),
        (
            "G14",
            "Future margin",
            "Any future improvement claim uses a predeclared practical margin on an untouched test set.",
            "Before future training.",
            "Do not claim improvement.",
            "M7D does not invent a universal SSIM threshold.",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=("criterion_id", "category", "criterion", "gate", "failure_action", "rationale"),
    )


def _stopping_rules() -> pd.DataFrame:
    rows = [
        (
            "R01",
            "preflight",
            "Stop if any pinned input is missing, hash-mismatched or schema-incompatible.",
            "Fix or restore the frozen input; do not rewrite history.",
            "Evidence must be reproducible.",
        ),
        (
            "R02",
            "font verification",
            "Stop if any planned font is missing, changes hash, loses glyph coverage or lacks a recorded license review.",
            "Revise the plan version before generation; never substitute after outcomes.",
            "Font identity is the treatment condition.",
        ),
        (
            "R03",
            "generation",
            "Generate exactly 30 cases per font and 10 fonts; no optional extension after seeing model outcomes.",
            "Publish the fixed sample as planned; report precision as a limitation.",
            "Prevents outcome-dependent sampling.",
        ),
        (
            "R04",
            "pre-evaluation",
            "Stop if source/case counts, manifest hash, target replay or primary target variance fail.",
            "Do not evaluate the model until the dataset gate passes.",
            "Prevents invalid target use.",
        ),
        (
            "R05",
            "model evaluation",
            "Evaluate the frozen champion once; no iterative peeking, recalibration or endpoint changes.",
            "Report all predeclared endpoints or stop the analysis.",
            "Protects zero-shot interpretation.",
        ),
        (
            "R06",
            "precision",
            "If intervals are wider than the planning scenarios, report that limitation; do not collect more data after seeing results.",
            "A new versioned benchmark is required for more precision.",
            "Prevents precision-driven optional stopping.",
        ),
        (
            "R07",
            "interpretation",
            "Stop any claim that exceeds the ten-font roster, fixed renderer, synthetic SSIM target or observed cases.",
            "Downgrade to a descriptive/conditional statement.",
            "Research-integrity boundary.",
        ),
        (
            "R08",
            "future improvement",
            "Stop a future model comparison without a predeclared practical margin, retention constraints and untouched test set.",
            "No improvement claim.",
            "Prevents post-hoc thresholds.",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=("rule_id", "stage", "rule", "action", "rationale"),
    )


def _claims_and_limits() -> pd.DataFrame:
    rows = [
        (
            "M7D-C01",
            "M7D is a planning milestone, not an executed dataset or benchmark.",
            "established",
            "This report, config and output manifest; no dataset or prediction artifact is created by M7D.",
            "Status field and non-execution checks.",
            "Any claim that the proposed benchmark has already run.",
        ),
        (
            "M7D-C02",
            "M7C found descriptive per-font differences on the observed three held-out fonts.",
            "established",
            "M7C all-primary RMSE Georgia 0.0852, Monaco 0.0761, Impact 0.1134.",
            "E22-E24 evidence rows; unpaired font groups.",
            "A font-population effect or causal attribution.",
        ),
        (
            "M7D-C03",
            "M7C's familiar-versus-held-out comparison is unmatched.",
            "established",
            "M7C comparison type is unpaired_descriptive; fonts, source composition and corruption mix differ.",
            "M7C benchmark-comparison table and limitations.",
            "A causal font or renderer effect.",
        ),
        (
            "M7D-C04",
            "A matched-case design can estimate familiar-versus-held-out contrasts conditional on the roster.",
            "planned",
            "Proposed 30 matched cases, 10 fonts, 300 sources, 5,700 primary rows.",
            "Design and endpoint tables; not yet executed.",
            "A completed estimate or a font-population interval.",
        ),
        (
            "M7D-C05",
            "The proposed benchmark can describe variation across the ten tested typefaces.",
            "planned",
            "Font inventory and per-font endpoints.",
            "Roster is purposive and fixed before outcomes.",
            "Arbitrary unseen-font generalization.",
        ),
        (
            "M7D-C06",
            "Ten purposive fonts estimate a font population.",
            "not_supported",
            "No random sampling frame or font-level probability sample exists.",
            "Font-level uncertainty is descriptive and leave-one-font-out only.",
            "A population interval, ranking or universal claim.",
        ),
        (
            "M7D-C07",
            "The proposed benchmark establishes unseen-renderer generalization.",
            "not_supported",
            "Renderer, canvas, image mode and format are fixed.",
            "Direction B is deferred to a separate protocol.",
            "Any renderer-generalization claim.",
        ),
        (
            "M7D-C08",
            "The proposed benchmark measures human readability, solve rate or security.",
            "not_supported",
            "Target is SSIM against a synthetic clean reference; no human or solver outcomes.",
            "Research-integrity boundary repeated in report.",
            "Construct or security claims.",
        ),
        (
            "M7D-C09",
            "The frozen champion can be evaluated zero-shot without recalibration.",
            "planned",
            "M5C artifact/card are pinned; no fit or calibrator is called by M7D.",
            "Frozen-model policy and success criteria.",
            "A model-improvement claim.",
        ),
        (
            "M7D-C10",
            "A null or mixed result would be evidence of equivalence or universal robustness.",
            "not_supported",
            "No equivalence margin is predeclared; the roster and renderer are fixed.",
            "Interpretation rules S1-S8 and report section 13.",
            "Equivalence, non-inferiority or universal robustness.",
        ),
        (
            "M7D-C11",
            "A negative result would establish CAPTCHA insecurity.",
            "not_supported",
            "SSIM is not a solve-rate or security measure.",
            "Target-definition boundary.",
            "Security or deployment claims.",
        ),
        (
            "M7D-C12",
            "Future model improvement requires a separate authorized experiment.",
            "established",
            "M7D trains nothing; future work needs a new protocol and untouched test set.",
            "Stopping rule R08 and endpoint F1.",
            "An implicit permission to refit or tune on the new benchmark.",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=("claim_id", "claim", "status", "evidence", "design_support", "cannot_establish"),
    )


@dataclass(frozen=True, slots=True)
class PlanResult:
    evidence_snapshot: pd.DataFrame
    direction_assessment: pd.DataFrame
    research_questions: pd.DataFrame
    independent_units: pd.DataFrame
    split_hierarchy: pd.DataFrame
    font_inventory: pd.DataFrame
    endpoints: pd.DataFrame
    sample_size_scenarios: pd.DataFrame
    leakage_controls: pd.DataFrame
    success_criteria: pd.DataFrame
    stopping_rules: pd.DataFrame
    claims_and_limits: pd.DataFrame
    protocol: dict[str, Any]
    report: str


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [json_safe(record) for record in frame.to_dict(orient="records")]


def _planned_counts(config: PlanConfig) -> dict[str, int]:
    fonts = len(config.fonts)
    sources = config.plan.n_text_cases * fonts
    return {
        "fonts": fonts,
        "familiar_fonts": sum(1 for font in config.fonts if font.role == "familiar"),
        "held_out_fonts": sum(1 for font in config.fonts if font.role == "held-out"),
        "text_cases": config.plan.n_text_cases,
        "sources": sources,
        "rows": sources * config.plan.rows_per_source,
        "primary_rows": sources * config.plan.primary_rows_per_source,
        "diagnostic_rows": sources * config.plan.diagnostic_rows_per_source,
    }


def _protocol(
    inputs: PlanInputs,
    *,
    evidence: pd.DataFrame,
    directions: pd.DataFrame,
    questions: pd.DataFrame,
    units: pd.DataFrame,
    splits: pd.DataFrame,
    fonts: pd.DataFrame,
    endpoints: pd.DataFrame,
    sample_size: pd.DataFrame,
    leakage: pd.DataFrame,
    success: pd.DataFrame,
    stopping: pd.DataFrame,
    claims: pd.DataFrame,
) -> dict[str, Any]:
    config = inputs.config
    counts = _planned_counts(config)
    source_metrics = inputs.m7c_source_metrics.loc[
        inputs.m7c_source_metrics["scope"] == "all_primary"
    ]
    source_sd = float(source_metrics["rmse"].std(ddof=1))
    font_rmse = (
        inputs.m7c_font_metrics.loc[
            (inputs.m7c_font_metrics["scope"] == "all_primary")
            & (inputs.m7c_font_metrics["protocol_role"] == "primary")
            & (inputs.m7c_font_metrics["font_id"].isin(M7B_EXPECTED_FONTS))
        ]
        .set_index("font_id")["rmse"]
        .to_dict()
    )
    return {
        "schema_version": "m7d-plan-v1",
        "milestone": config.plan.milestone,
        "status": config.plan.status,
        "chosen_direction": config.plan.chosen_direction,
        "chosen_title": DIRECTION_TITLES[config.plan.chosen_direction],
        "primary_question": config.plan.primary_question,
        "directions": _records(directions),
        "evidence": _records(evidence),
        "planned_counts": counts,
        "dataset_design": {
            "dataset_version": config.plan.dataset_version,
            "dataset_output": config.plan.dataset_output,
            "benchmark_report_output": config.plan.benchmark_report_output,
            "n_text_cases": config.plan.n_text_cases,
            "n_fonts": counts["fonts"],
            "n_familiar_fonts": counts["familiar_fonts"],
            "n_held_out_fonts": counts["held_out_fonts"],
            "n_sources": counts["sources"],
            "n_rows": counts["rows"],
            "n_primary_rows": counts["primary_rows"],
            "n_diagnostic_rows": counts["diagnostic_rows"],
            "renderer": config.plan.renderer,
            "canvas": [config.plan.canvas_width, config.plan.canvas_height],
            "image_mode": config.plan.image_mode,
            "image_format": config.plan.image_format,
            "target": TARGET,
            "target_definition": "SSIM between each corrupted image and its exact source-specific clean reference",
            "psnr_role": "diagnostic only",
            "text_lengths": config.plan.text_lengths,
            "text_length_weights": config.plan.text_length_weights,
            "charset": DEFAULT_CHARSET,
            "held_out_fonts_reused_from_m7b": list(M7B_REUSED_HELDOUT_FONTS),
            "new_held_out_fonts": list(PLANNED_NEW_HELDOUT_FONTS),
            "roster_increment_note": (
                "Georgia, Monaco and Impact were used once in M7B as an evaluation "
                "set only; the champion was never fit, calibrated or selected on "
                "them. Reusing them keeps three of the seven held-out faces "
                "directly comparable with M7B, but only four of the seven are "
                "genuinely new typefaces, so the roster widens the held-out set "
                "rather than replacing it, and the M7B three-font pattern is not "
                "independent of this roster."
            ),
        },
        "corruption_protocol": [
            {
                "family": corruption.family,
                "unit": corruption.unit,
                "role": corruption.role,
                "levels": corruption.levels,
            }
            for corruption in config.corruptions
        ],
        "font_roster": _records(fonts),
        "research_questions": _records(questions),
        "independent_units": _records(units),
        "split_hierarchy": _records(splits),
        "endpoints": _records(endpoints),
        "uncertainty": {
            "method": config.plan.bootstrap_method,
            "n_bootstrap": config.plan.n_bootstrap,
            "confidence_level": config.plan.confidence_level,
            "source_group": config.plan.source_group,
            "primary_unit": "matched_text_case",
            "font_level_policy": "descriptive only; leave-one-font-out sensitivity; no population interval",
            "row_level_policy": "never independent; no row-level bootstrap",
            "near_degenerate_sd": config.plan.near_degenerate_sd,
            "missing_policy": (
                "Any missing primary row fails the completeness gate. Failed renders are not "
                "imputed. Diagnostic-only missingness is reported explicitly."
            ),
        },
        "sample_size": {
            "observed_m7c_source_sd": source_sd,
            "observed_m7c_font_rmse": font_rmse,
            "text_case_counts": config.precision.text_case_counts,
            "paired_sd_assumptions": config.precision.paired_sd_assumptions,
            "font_counts": config.precision.font_counts,
            "font_sd_assumptions": config.precision.font_sd_assumptions,
            "selected_text_cases": config.plan.n_text_cases,
            "selected_fonts": counts["fonts"],
            "scenarios": _records(sample_size),
            "caveat": (
                "Precision scenarios are illustrative, not power calculations. The paired "
                "and font-level SD assumptions are not estimated from matched data; the "
                "three-font pilot cannot supply a font-population variance."
            ),
        },
        "frozen_model": {
            "name": M7C_CHAMPION_NAME,
            "calibration": M7C_CHAMPION_CALIBRATION,
            "artifact_sha256": M7C_CHAMPION_ARTIFACT_SHA256,
            "card_sha256": M7C_CHAMPION_CARD_SHA256,
            "feature_schema": inputs.m5c_model_card.get("feature_names"),
            "policy": (
                "Zero-shot evaluation only. No fit, partial_fit, calibrator, selector, target "
                "change or test-label-driven decision."
            ),
            "future_training": (
                "Separate authorized experiment with untouched font/case test groups and a "
                "predeclared practical margin."
            ),
        },
        "leakage_controls": _records(leakage),
        "success_criteria": _records(success),
        "stopping_rules": _records(stopping),
        "claims_and_limits": _records(claims),
        "limitations": [
            "Ten typefaces are a purposive roster, not a random sample of a font population.",
            "Three of the seven held-out faces (Georgia, Monaco, Impact) were already used in M7B as an evaluation set, so only four held-out identities are genuinely new and the M7B three-font pattern is not independent of this roster.",
            "The roster includes three display/script fonts as deliberate high-variation stress conditions; it is not representative of production CAPTCHA fonts.",
            "Thirty matched text cases provide limited source-level precision; the paired SD is unknown before generation.",
            "One renderer, one 192x64 canvas and one 8-bit grayscale PNG format are fixed; renderer generalization is untested.",
            "SSIM is full-reference synthetic image similarity, not human readability, solve rate, security or production robustness.",
            "Matched text cases reduce but do not eliminate composition differences because glyph metrics and ink distributions differ by font.",
            "Font paths, availability and licensing are machine-dependent; generation fails closed rather than substituting a font.",
            "No model is trained, refitted, recalibrated, selected or tuned in M7D.",
            "The clean target is constant and the 0.5 px elastic rung is near-degenerate; both are diagnostic only.",
            "No equivalence or non-inferiority margin is defined in M7D; a null contrast cannot establish equivalence.",
        ],
        "non_execution": [
            "No dataset was generated.",
            "No image was rendered.",
            "No model was trained, refitted, recalibrated, selected or tuned.",
            "No prediction was changed.",
            "No frozen benchmark artifact was modified.",
            "All proposed work is planned, not completed.",
        ],
        "output_contract": {
            "tables": list(OUTPUT_TABLE_NAMES),
            "manifest": "plan_manifest.json",
            "report": "report.md",
            "metadata": "run_metadata.json",
        },
    }


def _fmt(value: Any, digits: int = 6) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "undefined"
        return f"{value:.{digits}f}"
    return str(value)


def _md_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, separator]
    for _, row in frame.iterrows():
        cells = [str(row[column]).replace("|", "\\|") for column in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _build_report(inputs: PlanInputs, tables: dict[str, pd.DataFrame]) -> str:
    config = inputs.config
    counts = _planned_counts(config)
    evidence = tables["evidence_snapshot"]
    directions = tables["direction_assessment"]
    units = tables["independent_units"]
    splits = tables["split_hierarchy"]
    fonts = tables["font_inventory"]
    endpoints = tables["endpoints"]
    sample = tables["sample_size_scenarios"]
    success = tables["success_criteria"]
    stopping = tables["stopping_rules"]
    claims = tables["claims_and_limits"]
    concentration = inputs.m7c_error_concentration.loc[
        (inputs.m7c_error_concentration["scope"] == "all_primary")
        & (inputs.m7c_error_concentration["font_id"] == "all")
    ].iloc[0]
    m7c_all = _table_row(
        inputs.m7c_family_metrics, scope="all_primary", font_id="all", role="primary"
    )
    m7c_fonts = inputs.m7c_font_metrics.loc[
        (inputs.m7c_font_metrics["scope"] == "all_primary")
        & (inputs.m7c_font_metrics["protocol_role"] == "primary")
        & (inputs.m7c_font_metrics["font_id"].isin(M7B_EXPECTED_FONTS))
    ].set_index("font_id")
    m7c_blur = inputs.m7c_font_metrics.loc[
        (inputs.m7c_font_metrics["scope"] == "original_blur_noise")
        & (inputs.m7c_font_metrics["protocol_role"] == "primary")
        & (inputs.m7c_font_metrics["font_id"].isin(M7B_EXPECTED_FONTS))
    ].set_index("font_id")
    severity = inputs.m7c_severity_metrics.loc[
        (inputs.m7c_severity_metrics["font_id"] == "all")
        & (inputs.m7c_severity_metrics["protocol_role"] == "primary")
    ].set_index("amplitude_px")
    comparison = inputs.m7c_benchmark_comparison
    original_cmp = comparison.loc[
        (comparison["m7b_scope"] == "original_blur_noise") & (comparison["metric"] == "rmse")
    ].iloc[0]
    geometric_cmp = comparison.loc[
        (comparison["m7b_scope"] == "geometric_primary") & (comparison["metric"] == "rmse")
    ].iloc[0]
    selected_source = sample.loc[
        sample["selected"] & (sample["level"] == "matched_text_case_mean_rmse")
    ].iloc[0]
    selected_paired = sample.loc[
        sample["selected"] & (sample["level"] == "matched_font_pair_difference")
    ]
    selected_font = sample.loc[sample["selected"] & (sample["level"] == "font_mean_rmse")]
    m7c_counts = _claim_counts(inputs.m7c_claims)
    font_table = _md_table(
        fonts,
        (
            "font_id",
            "role",
            "stratum",
            "m7b_status",
            "sha256",
            "glyph_coverage_ok",
            "license_status",
        ),
    )
    direction_table = _md_table(
        directions,
        ("direction", "scientific_value", "leakage_risk", "decision", "selected"),
    )
    unit_table = _md_table(
        units,
        ("unit", "independence", "used_for", "not_valid_for"),
    )
    split_table = _md_table(
        splits,
        ("level", "rule", "grouping_key", "train_allowed", "eval_allowed"),
    )
    endpoint_table = _md_table(
        endpoints,
        ("endpoint_id", "role", "metric", "scope", "contrast", "uncertainty"),
    )
    success_table = _md_table(
        success,
        ("criterion_id", "category", "criterion", "gate", "failure_action"),
    )
    stopping_table = _md_table(
        stopping,
        ("rule_id", "stage", "rule", "action"),
    )
    claims_table = _md_table(
        claims,
        ("claim_id", "claim", "status", "cannot_establish"),
    )
    evidence_table = _md_table(
        evidence.loc[
            evidence["evidence_id"].isin(
                [
                    "E10",
                    "E11",
                    "E13",
                    "E14",
                    "E15",
                    "E16",
                    "E17",
                    "E22",
                    "E23",
                    "E24",
                    "E25",
                    "E26",
                    "E27",
                    "E28",
                    "E29",
                    "E30",
                    "E31",
                    "E32",
                    "E33",
                    "E34",
                ]
            )
        ],
        (
            "evidence_id",
            "family",
            "scope",
            "metric",
            "published_value",
            "reproduced_value",
            "reproduced",
        ),
    )
    source_sd = float(selected_source["assumed_sd"])
    text = f"""# M7D - Evaluation design and dataset-expansion plan

- **Experiment:** EXP-0017
- **Status:** Planned, not executed.
- **Chosen direction:** A - broader unseen-font generalization with matched text cases.
- **Frozen model:** M5C `{M7C_CHAMPION_NAME}`, calibration `{M7C_CHAMPION_CALIBRATION}`, artifact SHA-256 `{M7C_CHAMPION_ARTIFACT_SHA256}`.
- **Scope:** synthetic SSIM against an exact clean reference. This plan does not measure human readability, CAPTCHA solve rates, security or production robustness.

## Executive summary

M7D selects **Direction A**: a broader unseen-font benchmark with the matched
text-case control from Direction C incorporated into the design. The plan
pre-registers ten typefaces, 30 text cases, 300 font-specific sources, 6,300
rows, 5,700 primary rows and 600 diagnostic rows. The frozen M5C champion would
be evaluated zero-shot; no model would be trained, recalibrated, selected or
tuned, and no held-out label would be used for fitting or selection. Three of
the seven held-out faces (Georgia, Monaco, Impact) were already used in M7B as
an *evaluation* set, so four held-out identities are genuinely new and the
M7B three-font pattern is not independently re-tested by this design.

The decision is driven by the frozen M7C evidence, not by a new model result.
M7C's all-primary MAE is {_fmt(float(m7c_all["mae"]))} and RMSE is {_fmt(float(m7c_all["rmse"]))} (R^2 {_fmt(float(m7c_all["r2"]))}) over 1,140 rows from 60 sources. Per-font all-primary RMSE is Georgia {_fmt(float(m7c_fonts.loc["georgia", "rmse"]))}, Monaco {_fmt(float(m7c_fonts.loc["monaco", "rmse"]))} and Impact {_fmt(float(m7c_fonts.loc["impact", "rmse"]))}. Original blur/noise RMSE is {_fmt(float(m7c_blur.loc["georgia", "rmse"]))}, {_fmt(float(m7c_blur.loc["monaco", "rmse"]))} and **{_fmt(float(m7c_blur.loc["impact", "rmse"]))}** versus {_fmt(float(original_cmp["familiar_estimate"]))} on the familiar benchmark; primary geometric RMSE is {_fmt(float(geometric_cmp["m7b_estimate"]))} versus {_fmt(float(geometric_cmp["familiar_estimate"]))}. The familiar-versus-held-out blur/noise difference is +{_fmt(float(original_cmp["difference"]))} and the geometric difference is {_fmt(float(geometric_cmp["difference"]))}.
Those comparisons are unmatched in fonts, source composition and corruption
mix, so they cannot be read as font effects. The proposed design pairs every
held-out font with every familiar font on the same 30 text cases and matched
corruption realisations; the matching changes the estimand from an unpaired
roster comparison to a conditional matched contrast.

The plan does **not** claim that ten purposive typefaces estimate a font
population. It does **not** test unseen renderers, human readability, solve
rates or security. It does **not** authorize a model change. Every dataset,
benchmark, endpoint and success criterion below is planned, not completed.

## 1. Status and research-integrity boundary

M7D is a read-only planning milestone. It reads the frozen M3/M5C/M7A/M7B/M7C
artifacts, verifies the pinned hashes and selected numerical claims, and writes
only to `reports/evaluation_design_plan_v0_1_0/` (Git-ignored). It does not
generate a dataset, render an image, train or refit a model, recalibrate a
prediction, change a target or split, or modify any frozen benchmark artifact.

The target boundary is unchanged: the only outcome named here is SSIM between a
corrupted synthetic image and its exact clean reference. A lower SSIM error is
not evidence of human readability, CAPTCHA solve rate, security or real-world
robustness. The clean target is constant and the 0.5 px elastic rung is
diagnostic only.

## 2. Preflight and pinned evidence

The plan pins {len(config.inputs)} upstream files by SHA-256: the M3 dataset
config/summary/sources/manifest, the M5C config/model card/artifact/test
predictions/metrics/metadata/report, the M7A config/direction table/claims/
metadata, the M7B config/dataset summary/sources/manifest/predictions/metrics/
comparison/metadata/report, and the M7C config/font/family/severity/source/
concentration/uncertainty/comparison/claims/metadata/report. All pins were
verified before this plan was written. The M3 dataset is 60 sources and 540
rows on a 36/12/12 split at 192x64; M7B is 60 sources and 1,260 rows (20 sources
per held-out font) with 1,140 primary and 120 diagnostic rows. The M5C champion
artifact and card hashes match the recorded baseline.

The M7D module also verifies the ten planned font files by path, SHA-256 and
charset coverage. All ten files were present and covered the charset on the
current development machine. Presence is not a guarantee of future availability
or licensing: `license_status` is `review_required_before_generation`, and the
future dataset generator must stop rather than substitute a font.

## 3. Evidence from M7A-M7C

All {len(evidence)} selected evidence values were recomputed from the pinned
frames (or, for hashes, from the pinned files), not copied from prior prose.
The full audit is in `evidence_snapshot.csv`; all rows have
`reproduced = true`.

{evidence_table}

The salient M7A-M7C evidence is:

- M7A has 3 supported / 2 suggestive / 7 unsupported claims over 60 sources and
  12 independent test groups; its recommendation was explicitly held-out fonts
  or rendering conditions, not a new model.
- M7B/M7C find a large three-font blur/noise difference and a near-identical
  aggregate geometric point estimate, but the geometric aggregate conceals a
  non-monotonic severity profile: all-font RMSE is {_fmt(float(severity.loc[1.0, "rmse"]))}
  at 1 px, {_fmt(float(severity.loc[2.0, "rmse"]))} at 2 px and
  {_fmt(float(severity.loc[3.0, "rmse"]))} at 3 px (max/min 3.20).
- All-primary error is broadly distributed rather than dominated by a handful
  of sources: top-1 share {_fmt(float(concentration["top1_share"]))}, top-3 share
  {_fmt(float(concentration["top_k_share"]))}, HHI {_fmt(float(concentration["hhi"]))}
  and effective source count {_fmt(float(concentration["effective_sources"]))} of 60.
- The M7C claim audit classifies {_claim_count_text(m7c_counts)} claims as
  supported/suggestive/unsupported. It explicitly leaves arbitrary unseen fonts,
  unseen renderers/resolutions/formats, human readability or solve rates,
  security, a causal font effect, geometric equivalence and font-population
  generalization unsupported.

## 4. Direction comparison and decision

{direction_table}

**Decision: A.** Direction A is the only option that expands the font-identity
axis while retaining the matched-case control. Direction C alone would repair
the unmatched comparison but reuse the same three held-out fonts; Direction B
would add a renderer axis but confound renderer with antialiasing, placement,
format and effective spatial frequency, and two renderers are not a renderer
sample. Direction B is deferred, not rejected permanently. Direction C's
matching logic is incorporated into A as a design feature rather than treated as
a separate primary benchmark.

The practical consequence is a broader but still conditional claim: after the
proposed benchmark, the project could say how the frozen champion behaves
across **these ten pre-specified typefaces** and **this fixed renderer**, with
matched text cases and corruption realisations. It could not say that the
champion generalizes to arbitrary fonts, font populations, renderers, human
readers or real CAPTCHAs.

## 5. Research questions and independent units

The plan separates the font identity, matched text case, font-specific source,
clean reference, corrupted variant, renderer, benchmark row, font pair and source
group. These are not interchangeable.

{unit_table}

**Primary question.** {config.plan.primary_question}

The primary estimand is the average of the 21 held-out/familiar font-pair
contrasts, each computed on the same 30 text cases; the signed direction is
held-out minus familiar. The matched text case is the uncertainty unit. The
font pair is a descriptive construction. The font identity is the unit for
font-level spread, not a random draw from a font population. Source grouping
alone cannot establish font-population generalization because it resamples
cases within the observed fonts, not new fonts.

## 6. Split hierarchy and leakage controls

{split_table}

The proposed benchmark is eval-only: all 300 sources are in one untouched
`eval` split and no model is fit. If a future authorized training experiment is
added, at least two held-out font identities and disjoint text cases must be
reserved for a final test set that is never used for fitting, calibration or
selection. A source-level split must keep all font renders of a case together;
a variant is never split from its source. The complete control table is in
`leakage_controls.csv` and is summarised in section 12.

## 7. Proposed dataset expansion

| Item | Planned value |
| --- | --- |
| Dataset version | `{config.plan.dataset_version}` |
| Dataset output | `{config.plan.dataset_output}` (Git-ignored) |
| Benchmark report output | `{config.plan.benchmark_report_output}` (Git-ignored) |
| Fonts | {counts["fonts"]} ({counts["familiar_fonts"]} familiar, {counts["held_out_fonts"]} held-out) |
| Matched text cases | {counts["text_cases"]} |
| Sources | {counts["sources"]} |
| Rows | {counts["rows"]} |
| Primary rows | {counts["primary_rows"]} |
| Diagnostic rows | {counts["diagnostic_rows"]} |
| Renderer | {config.plan.renderer} |
| Canvas / mode / format | {config.plan.canvas_width}x{config.plan.canvas_height}, {config.plan.image_mode}, {config.plan.image_format} |
| Target | `{TARGET}` (SSIM to exact clean reference) |
| Bootstrap | {config.plan.n_bootstrap} matched-case grouped percentile replicates, {config.plan.confidence_level:.0%} |

Each text case is sampled once from the frozen charset and length distribution
(4/5/6 characters, weights 0.30/0.40/0.30), then rendered in all ten fonts. The
same case ID, render seed schedule and corruption seed schedule are used across
fonts, so a font contrast is paired by construction. Font-specific sources and
clean references still differ because glyph metrics differ; matching reduces
but does not eliminate source-composition differences.

The corruption protocol is the frozen M7B ladder: Gaussian blur at 1/2/3.5/5
px, Gaussian noise at 6/12/20/32 sigma, thin lines at 1/2/4/7, arc strokes at
1/2/4/7, and elastic warp at 1/2/3 px. The clean image and the 0.5 px elastic
rung are diagnostic only. The future manifest must record `case_id`,
`source_id`, `font_id`, font SHA-256, render seed, corruption seed, variant seed,
text SHA-256, image/reference SHA-256, target, dataset version and generation
hash. Any missing primary row fails the completeness gate; there is no
imputation and no silent re-render.

## 8. Font selection and verification

The roster was fixed before any model outcome on the new benchmark. It is a
purposive maximum-variation sample, not a random sample:

{font_table}

Selection criteria were: (1) the three M3 fonts remain the familiar reference;
(2) held-out fonts must be absent from M3 development data; (3) the roster spans
serif, monospace, grotesque/humanist sans and display/script typography; (4)
each file must be a single-face TTF with full charset coverage on the current
machine; (5) each file must be hashable and locally available. The display/script
fonts are deliberate stress conditions, not representative production CAPTCHA
fonts. Licensing is **not** asserted by M7D. Before generation, each font needs a
recorded local-use/license review; if any file is missing, changes hash, loses a
glyph or lacks approval, generation stops and a new plan version is required.
No substitute font may be chosen after seeing outcomes.

**How much of the roster is actually new.** Georgia, Monaco and Impact were
already used in M7B as an *evaluation* set; the champion was never fit,
calibrated or selected on them, so they remain valid held-out identities and
keep three of the seven directly comparable with M7B. Only four of the seven
held-out faces — Verdana, Trebuchet MS, Chalkduster and Luminari — are
genuinely new typefaces. The roster therefore **widens** the held-out set
rather than replacing it, and the M7B three-font pattern is not independent of
this roster: a per-font result on Georgia, Monaco or Impact reuses a face whose
M7B error profile is already known, while the four new faces are the only
fully novel font identities. The `m7b_status` column of
`font_inventory.csv` records this distinction per font, and the claim that the
champion behaves differently "across the ten tested typefaces" must be read as
a claim about seven identities, four of which are new.

## 9. Pre-registered endpoints and uncertainty

{endpoint_table}

The primary endpoint is the matched held-out-minus-familiar all-primary RMSE
contrast. Per-font all-primary metrics support it descriptively. Secondary
endpoints cover original blur/noise, each declared corruption family, geometric
severity, clean-reference retention, target distributions, source concentration
and paired within-source contrasts. There is no search over arbitrary
subgroups. R^2 is suppressed when the target SD is below 0.005 or the target is
constant. The clean and 0.5 px slices are diagnostic and cannot support primary
ranking claims.

Uncertainty uses {config.plan.n_bootstrap} matched-case grouped percentile
bootstrap replicates. A replicate resamples whole text cases and keeps every
font, source and variant of a case together. Per-font intervals resample
font-specific sources, not rows. Font-level summaries use leave-one-font-out
sensitivity and descriptive ranges; they do not get a population interval.
There is no row-level bootstrap. Missing primary rows fail the gate; missing
diagnostic rows are reported explicitly and never imputed.

## 10. Sample-size and precision rationale

The source-level planning SD is {source_sd:.6f}, computed from the 60 frozen M7C
all-primary per-source RMSE values. With 30 matched cases, the normal-
approximation SE for a mean source RMSE is {float(selected_source["expected_se"]):.6f}
and the 95% half-width is {float(selected_source["ci95_halfwidth"]):.6f}. This
is a precision scenario, not a power calculation.

For the matched contrast, the paired-difference SD is unknown before the new
data exist. The plan therefore reports sensitivity across assumed SDs rather
than fabricating an effect size. At 30 cases:

| Assumed paired SD | Expected SE | 95% half-width |
| ---: | ---: | ---: |
"""
    for _, row in selected_paired.sort_values("assumed_sd").iterrows():
        text += (
            f"| {float(row['assumed_sd']):.3f} | {float(row['expected_se']):.6f} | "
            f"{float(row['ci95_halfwidth']):.6f} |\n"
        )
    text += """
For font-level precision, ten fonts are still insufficient for a precise
population interval. The illustrative sensitivity is:

| Assumed between-font SD | Expected SE | 95% half-width |
| ---: | ---: | ---: |
"""
    for _, row in selected_font.sort_values("assumed_sd").iterrows():
        text += (
            f"| {float(row['assumed_sd']):.3f} | {float(row['expected_se']):.6f} | "
            f"{float(row['ci95_halfwidth']):.6f} |\n"
        )
    text += f"""
These font-level numbers use a normal multiplier and understate small-df t
uncertainty; they are descriptive roster precision, not a font-population
estimate. The three observed M7B font RMSEs (0.0852, 0.0761, 0.1134) are not a
font sample and were not used as a population variance. If a future protocol
makes font-population inference primary, 12-16 or more independent font
identities, or a hierarchical model with a defined sampling frame, would be
required. The plan selects 30 cases and 10 fonts as a fixed design; it does not
extend after seeing results.

## 11. Frozen model and baseline policy

The baseline is exactly the M5C `{M7C_CHAMPION_NAME}` champion with calibration
`{M7C_CHAMPION_CALIBRATION}`, artifact SHA-256
`{M7C_CHAMPION_ARTIFACT_SHA256}` and card SHA-256
`{M7C_CHAMPION_CARD_SHA256}`. The eight-feature schema, target, renderer and
corruption definitions are frozen. The future evaluator may load the serialized
estimator and predict; it must not call `fit`, `partial_fit`, a calibrator or a
selector, and must not use held-out labels to choose features, thresholds,
endpoints or a model.

M7D itself does not load the model or create predictions. Any future model
training or comparison requires a separate authorized experiment, a new
versioned protocol, untouched test groups and a predeclared practical improvement
margin. M7D does not invent a universal SSIM threshold; the margin must be
justified in that future protocol by measurement precision, retention constraints
and a practical decision rule.

## 12. Success criteria, stopping rules and stopping interpretation

{success_table}

{stopping_table}

The design stops on data or protocol failure, not on whether the results look
favourable. The sample is fixed at 30 cases and 10 fonts; if intervals are wider
than planned, that is a reported limitation, not a reason to collect more after
seeing model outcomes. A future model-improvement claim has no standing unless a
practical margin, retention constraints and an untouched test set were fixed
before evaluation.

## 13. Valid interpretation of null, mixed and negative findings

- **Null contrast.** A CI that includes zero means no detected matched
  familiar-versus-held-out difference on this roster and these corruptions. It
  does **not** establish equivalence, universal robustness or font-population
  homogeneity. No equivalence margin was predeclared.
- **Mixed result.** A font-by-corruption interaction is a valid conditional
  finding. The plan reports family and severity slices separately and does not
  average them into a single universal ranking. A reversal (for example, a font
  that is worse on blur/noise but better on interference) is not evidence that
  the font is generally better.
- **Negative result.** Held-out fonts worse than familiar fonts is a conditional
  descriptive result. It does not identify a causal font effect, prove that a
  production CAPTCHA is insecure or predict human solve rates. It may motivate a
  future model experiment, but it does not authorize one.
- **Positive result.** A narrower or smaller contrast likewise does not prove
  arbitrary-font generalization. It is conditional on the ten-font roster, the
  fixed renderer, the synthetic target and the observed cases.

## 14. What the plan cannot establish

- Arbitrary unseen-font generalization or a font-population effect.
- Independent confirmation of the M7B three-font pattern: three of the seven
  held-out faces are the same typefaces M7B already used as an evaluation set,
  so only four held-out identities are genuinely new.
- Unseen-renderer, resolution or image-format generalization.
- Human readability, CAPTCHA solve rate, security or production robustness.
- A causal effect of typeface independent of glyph metrics, ink coverage and
  source composition.
- Equivalence or non-inferiority without a predeclared margin.
- A stable ranking at the 0.5 px elastic rung or on the constant clean target.
- A model improvement or a new champion.

The full claim table is in `claims_and_limits.csv`:

{claims_table}

## 15. Implementation handoff and verification

The plan is versioned by `configs/evaluation_design_plan_v0_1_0.toml` and
implemented by `src/captcha_quality/evaluation_plan.py` with the
`captcha-evaluation-plan` CLI. Future dataset generation is **not** implemented
by this milestone. When authorized, it should add a separate versioned config
and generator that consumes this plan's roster, counts, corruption ladder,
manifest fields and fail-closed gates.

```bash
uv sync --extra dev --extra image --extra ml

uv run --extra image --extra ml captcha-evaluation-plan run \
    --config configs/evaluation_design_plan_v0_1_0.toml --overwrite

uv run --extra image --extra ml captcha-evaluation-plan verify \
    --config configs/evaluation_design_plan_v0_1_0.toml
```

`run` writes only to `reports/evaluation_design_plan_v0_1_0/`. `verify` reloads
all pinned inputs, replays the plan core, compares every deterministic table,
`plan_manifest.json` and `report.md`, and rechecks input, output and protected-
asset hashes. Two clean runs are byte-identical except for runtime metadata.

## 16. Non-execution statement

M7D generated no dataset, rendered no image, trained no model, changed no
prediction and modified no frozen artifact. The proposed dataset, benchmark,
endpoints, sample size, success criteria and stopping rules are planned only.
The next action is review of this plan, not execution.

- No dataset was generated.
- No image was rendered.
- No model was trained, refitted, recalibrated, selected or tuned.
- No prediction was changed.
- No frozen benchmark artifact was modified.
- All proposed work is planned, not completed.
"""
    return text


def build_plan(inputs: PlanInputs) -> PlanResult:
    """Build every deterministic M7D table, manifest and report."""
    evidence = _evidence_snapshot(inputs)
    if not bool(evidence["reproduced"].all()):
        failed = evidence.loc[~evidence["reproduced"], "evidence_id"].tolist()
        raise PlanError(f"evidence reproduction failed for: {failed}")
    directions = _direction_assessment(inputs)
    questions = _research_questions()
    units = _independent_units()
    splits = _split_hierarchy()
    fonts = inputs.font_inventory.copy()
    endpoints = _endpoints()
    sample_size = _sample_size_scenarios(inputs, inputs.config)
    leakage = _leakage_controls()
    success = _success_criteria()
    stopping = _stopping_rules()
    claims = _claims_and_limits()
    protocol = _protocol(
        inputs,
        evidence=evidence,
        directions=directions,
        questions=questions,
        units=units,
        splits=splits,
        fonts=fonts,
        endpoints=endpoints,
        sample_size=sample_size,
        leakage=leakage,
        success=success,
        stopping=stopping,
        claims=claims,
    )
    tables = {
        "evidence_snapshot": evidence,
        "direction_assessment": directions,
        "research_questions": questions,
        "independent_units": units,
        "split_hierarchy": splits,
        "font_inventory": fonts,
        "endpoints": endpoints,
        "sample_size_scenarios": sample_size,
        "leakage_controls": leakage,
        "success_criteria": success,
        "stopping_rules": stopping,
        "claims_and_limits": claims,
    }
    report = _build_report(inputs, tables)
    return PlanResult(
        evidence_snapshot=evidence,
        direction_assessment=directions,
        research_questions=questions,
        independent_units=units,
        split_hierarchy=splits,
        font_inventory=fonts,
        endpoints=endpoints,
        sample_size_scenarios=sample_size,
        leakage_controls=leakage,
        success_criteria=success,
        stopping_rules=stopping,
        claims_and_limits=claims,
        protocol=protocol,
        report=report,
    )


def build_isolation_checks(inputs: PlanInputs, result: PlanResult, report: str) -> pd.DataFrame:
    """Independently check the plan's completeness, purity and non-execution."""
    checks: list[CheckResult] = []
    checks.extend(inputs.checks)
    _check(
        checks,
        "chosen_direction",
        lambda: (
            "Direction A selected with matched text-case control"
            if inputs.config.plan.chosen_direction == "A"
            and bool(
                result.direction_assessment.loc[
                    result.direction_assessment["direction"] == "A", "selected"
                ].iloc[0]
            )
            else (_ for _ in ()).throw(PlanError("direction decision is inconsistent"))
        ),
    )
    _check(
        checks,
        "planned_counts",
        lambda: (
            "300 sources, 6300 rows, 5700 primary, 600 diagnostic"
            if _planned_counts(inputs.config)
            == {
                "fonts": 10,
                "familiar_fonts": 3,
                "held_out_fonts": 7,
                "text_cases": 30,
                "sources": 300,
                "rows": 6300,
                "primary_rows": 5700,
                "diagnostic_rows": 600,
            }
            else (_ for _ in ()).throw(PlanError("planned counts are inconsistent"))
        ),
    )
    _check(
        checks,
        "font_roster",
        lambda: (
            "3 familiar and 7 held-out fonts, all hash and glyph verified"
            if len(result.font_inventory) == 10
            and int((result.font_inventory["role"] == "familiar").sum()) == 3
            and int((result.font_inventory["role"] == "held-out").sum()) == 7
            and bool(result.font_inventory["hash_verified"].all())
            and bool(result.font_inventory["glyph_coverage_ok"].all())
            else (_ for _ in ()).throw(PlanError("font roster is incomplete"))
        ),
    )
    _check(
        checks,
        "evidence_reproduced",
        lambda: (
            f"all {len(result.evidence_snapshot)} evidence values reproduced"
            if bool(result.evidence_snapshot["reproduced"].all())
            else (_ for _ in ()).throw(PlanError("one or more evidence values did not reproduce"))
        ),
    )
    required_units = {
        "font_identity",
        "matched_text_case",
        "source",
        "clean_reference",
        "corrupted_variant",
        "renderer",
        "benchmark_row",
        "font_pair",
        "source_group",
    }
    _check(
        checks,
        "independent_unit_coverage",
        lambda: (
            "all required independent-unit distinctions documented"
            if required_units.issubset(set(result.independent_units["unit"]))
            else (_ for _ in ()).throw(PlanError("independent-unit table is incomplete"))
        ),
    )
    _check(
        checks,
        "endpoint_coverage",
        lambda: (
            "primary, supporting, secondary and diagnostic endpoints present"
            if {"P1", "P1d", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "F1"}.issubset(
                set(result.endpoints["endpoint_id"])
            )
            else (_ for _ in ()).throw(PlanError("endpoint table is incomplete"))
        ),
    )
    _check(
        checks,
        "sample_size_scenarios",
        lambda: (
            "source, paired and font-level sensitivity scenarios present"
            if set(result.sample_size_scenarios["level"])
            == {
                "matched_text_case_mean_rmse",
                "matched_font_pair_difference",
                "font_mean_rmse",
            }
            else (_ for _ in ()).throw(PlanError("sample-size scenarios are incomplete"))
        ),
    )
    _check(
        checks,
        "report_sections",
        lambda: (
            "all report sections present"
            if all(section in report for section in REPORT_SECTIONS)
            else (_ for _ in ()).throw(PlanError("report is missing a section"))
        ),
    )
    _check(
        checks,
        "non_execution_statement",
        lambda: (
            "report states that no dataset, image, model or prediction was produced"
            if "No dataset was generated." in report
            and "No image was rendered." in report
            and "No model was trained" in report
            and "planned, not executed" in report.lower()
            else (_ for _ in ()).throw(PlanError("non-execution statement is missing"))
        ),
    )
    _check(
        checks,
        "claim_limits",
        lambda: (
            "font-population, renderer, readability and security limits are explicit"
            if {"M7D-C06", "M7D-C07", "M7D-C08", "M7D-C10", "M7D-C11"}.issubset(
                set(result.claims_and_limits["claim_id"])
            )
            else (_ for _ in ()).throw(PlanError("claim-limit table is incomplete"))
        ),
    )
    _check(
        checks,
        "stopping_rules",
        lambda: (
            "fixed sample and no outcome-dependent extension are documented"
            if "R03" in set(result.stopping_rules["rule_id"])
            and "R06" in set(result.stopping_rules["rule_id"])
            else (_ for _ in ()).throw(PlanError("stopping rules are incomplete"))
        ),
    )
    _check(
        checks,
        "success_criteria",
        lambda: (
            "integrity, leakage, target, model, reproducibility and future-margin gates present"
            if {"G01", "G06", "G09", "G10", "G12", "G14"}.issubset(
                set(result.success_criteria["criterion_id"])
            )
            else (_ for _ in ()).throw(PlanError("success criteria are incomplete"))
        ),
    )
    _check(
        checks,
        "no_planned_outputs_created",
        lambda: (
            "planned dataset and benchmark report directories are absent at M7D run time"
            if not Path(inputs.config.plan.dataset_output).exists()
            and not Path(inputs.config.plan.benchmark_report_output).exists()
            else (_ for _ in ()).throw(
                PlanError("a planned output already exists; M7D must not execute it")
            )
        ),
    )
    return pd.DataFrame(
        [{"check": check.name, "passed": check.passed, "detail": check.detail} for check in checks]
    )


def _safe_output_path(path: Path | str) -> Path:
    resolved = Path(path).resolve()
    reports_root = REPORTS_ROOT.resolve()
    if resolved == reports_root or not resolved.is_relative_to(reports_root):
        raise PlanError(f"refusing to write outside the Git-ignored reports directory: {resolved}")
    for protected_text in PROTECTED_REPORT_ROOTS:
        protected = Path(protected_text).resolve()
        if resolved == protected or resolved.is_relative_to(protected):
            raise PlanError(f"refusing to write into a protected frozen report path: {resolved}")
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


@dataclass(frozen=True, slots=True)
class ProtectedDependencyMembership:
    """Explicit membership of the frozen dependencies M7D verification protects.

    M7D declares its dependencies at file level because every upstream
    dependency is individually hash-pinned in the committed M7D planning
    configuration. Files that merely share a directory with a declared
    dependency are not M7D dependencies, so adding them does not invalidate
    M7D verification.
    """

    version: str
    config_files: tuple[str, ...]
    input_files: tuple[str, ...]
    output_files: tuple[str, ...]
    ignored_paths: tuple[str, ...]

    @property
    def members(self) -> tuple[str, ...]:
        """Every declared file the plan reads."""
        return self.config_files + self.input_files


def protected_dependency_membership(config: PlanConfig) -> ProtectedDependencyMembership:
    """Return the declared M7D protected-dependency membership."""
    return ProtectedDependencyMembership(
        version=M7D_PROTECTED_DEPENDENCY_MEMBERSHIP_VERSION,
        config_files=(M7D_PLANNING_CONFIG_FILE,),
        input_files=tuple(pin.path for pin in config.inputs),
        output_files=M7D_OUTPUT_FILES,
        ignored_paths=M7D_IGNORED_PATHS,
    )


def _is_ignored_path(path: str, membership: ProtectedDependencyMembership) -> bool:
    """True when a path is a declared ignored scratch or planned output path."""
    return any(
        path == ignored or path.startswith(f"{ignored}/") for ignored in membership.ignored_paths
    )


def _scoped_dependency_fingerprint(
    membership: ProtectedDependencyMembership,
) -> dict[str, str]:
    """Hash exactly the declared M7D dependencies that exist on disk."""
    fingerprint: dict[str, str] = {}
    for relative in membership.members:
        if _is_ignored_path(relative, membership):
            continue
        path = Path(relative)
        if not path.is_file():
            continue
        fingerprint[relative] = sha256_file(path)
    return fingerprint


def _scoped_dependency_state(
    membership: ProtectedDependencyMembership,
    expected: Mapping[str, str],
) -> dict[str, list[str]]:
    """Classify the declared dependencies against their expected hashes.

    ``missing`` covers a declared dependency that no longer exists, ``not_a_file``
    covers one that was replaced by something that is not a regular file, and
    ``modified`` covers one whose content hash differs from the expected hash.
    """
    missing: list[str] = []
    not_a_file: list[str] = []
    modified: list[str] = []
    for relative in sorted(expected):
        path = Path(relative)
        if not path.exists():
            missing.append(relative)
        elif not path.is_file():
            not_a_file.append(relative)
        elif sha256_file(path) != expected[relative]:
            modified.append(relative)
    return {"missing": missing, "not_a_file": not_a_file, "modified": modified}


def _output_hashes(output_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "run_metadata.json":
            hashes[path.name] = sha256_file(path)
    return hashes


def _config_with_output(config: PlanConfig, output: Path | None) -> PlanConfig:
    if output is None:
        return config
    run = RunSettings(
        name=config.run.name,
        version=config.run.version,
        seed=config.run.seed,
        output_reports=Path(output),
    )
    candidate = PlanConfig(
        run=run,
        plan=config.plan,
        precision=config.precision,
        corruptions=config.corruptions,
        fonts=config.fonts,
        inputs=config.inputs,
        config_path=config.config_path,
        config_sha256=config.config_sha256,
        config_hash="",
    )
    return PlanConfig(
        run=candidate.run,
        plan=candidate.plan,
        precision=candidate.precision,
        corruptions=candidate.corruptions,
        fonts=candidate.fonts,
        inputs=candidate.inputs,
        config_path=candidate.config_path,
        config_sha256=candidate.config_sha256,
        config_hash=sha256_bytes(canonical_json(candidate.identity()).encode("utf-8")),
    )


def _build_metadata(
    inputs: PlanInputs,
    result: PlanResult,
    *,
    output_hashes: Mapping[str, str],
    frozen_assets: Mapping[str, str],
    frozen_assets_hash: str,
    frozen_dependencies: Mapping[str, str] | None = None,
    frozen_dependencies_hash: str | None = None,
    elapsed_seconds: float,
) -> dict[str, Any]:
    config = inputs.config
    counts = _planned_counts(config)
    return {
        "plan": {
            "name": config.run.name,
            "version": config.run.version,
            "config_path": str(config.config_path),
            "config_sha256": config.config_sha256,
            "config_hash": config.config_hash,
            "code_revision": code_revision(),
            "package_version": package_version(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "milestone": config.plan.milestone,
            "status": config.plan.status,
            "chosen_direction": config.plan.chosen_direction,
            "planned_counts": counts,
        },
        "inputs": inputs.input_hashes,
        "input_manifest_sha256": output_hashes.get("input_manifest.csv"),
        "plan_manifest_sha256": output_hashes.get("plan_manifest.json"),
        "report_sha256": output_hashes.get("report.md"),
        "summary": {
            "evidence_rows": int(len(result.evidence_snapshot)),
            "all_evidence_reproduced": bool(result.evidence_snapshot["reproduced"].all()),
            "directions": int(len(result.direction_assessment)),
            "fonts": int(len(result.font_inventory)),
            "familiar_fonts": int((result.font_inventory["role"] == "familiar").sum()),
            "held_out_fonts": int((result.font_inventory["role"] == "held-out").sum()),
            "text_cases": config.plan.n_text_cases,
            "sources": counts["sources"],
            "rows": counts["rows"],
            "primary_rows": counts["primary_rows"],
            "diagnostic_rows": counts["diagnostic_rows"],
            "chosen_direction": config.plan.chosen_direction,
            "status": config.plan.status,
        },
        "planned_outputs_present_at_run": {
            "dataset": Path(config.plan.dataset_output).exists(),
            "benchmark_report": Path(config.plan.benchmark_report_output).exists(),
        },
        "frozen_assets": dict(frozen_assets),
        "frozen_assets_hash": frozen_assets_hash,
        "frozen_dependencies": dict(frozen_dependencies or {}),
        "frozen_dependencies_hash": frozen_dependencies_hash,
        "frozen_dependency_membership": (protected_dependency_membership(config).version),
        "output_hashes": dict(output_hashes),
        "runtime_seconds": round(float(elapsed_seconds), 3),
    }


def run_plan(
    config_path: Path | str,
    *,
    output_reports: Path | str | None = None,
    overwrite: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Run the read-only M7D plan and write deterministic Git-ignored outputs."""
    started = time.perf_counter()
    config = _config_with_output(load_config(config_path), output_reports)
    output_dir = _safe_output_path(config.run.output_reports)
    if output_dir.exists() and not overwrite:
        raise PlanError(
            f"output directory already exists: {output_dir}; pass --overwrite to replace it"
        )
    membership = protected_dependency_membership(config)
    dependencies_before = _scoped_dependency_fingerprint(membership)
    inputs = load_inputs(config)
    result = build_plan(inputs)
    checks = build_isolation_checks(inputs, result, result.report)
    _require_all_checks(
        [
            CheckResult(str(row["check"]), bool(row["passed"]), str(row["detail"]))
            for _, row in checks.iterrows()
        ],
        "M7D plan checks",
    )
    if output_dir.exists():
        if not overwrite:
            raise PlanError(f"output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(inputs.input_manifest, output_dir / "input_manifest.csv")
    _write_csv(result.font_inventory, output_dir / "font_inventory.csv")
    _write_csv(result.direction_assessment, output_dir / "direction_assessment.csv")
    _write_csv(result.research_questions, output_dir / "research_questions.csv")
    _write_csv(result.independent_units, output_dir / "independent_units.csv")
    _write_csv(result.split_hierarchy, output_dir / "split_hierarchy.csv")
    _write_csv(result.endpoints, output_dir / "endpoints.csv")
    _write_csv(result.sample_size_scenarios, output_dir / "sample_size_scenarios.csv")
    _write_csv(result.leakage_controls, output_dir / "leakage_controls.csv")
    _write_csv(result.success_criteria, output_dir / "success_criteria.csv")
    _write_csv(result.stopping_rules, output_dir / "stopping_rules.csv")
    _write_csv(result.claims_and_limits, output_dir / "claims_and_limits.csv")
    _write_csv(result.evidence_snapshot, output_dir / "evidence_snapshot.csv")
    _write_csv(checks, output_dir / "isolation_checks.csv")
    _write_json(result.protocol, output_dir / "plan_manifest.json")
    (output_dir / "report.md").write_text(result.report, encoding="utf-8")
    output_hashes = _output_hashes(output_dir)
    dependencies_after = _scoped_dependency_fingerprint(membership)
    if dependencies_before != dependencies_after:
        raise PlanError("declared protected dependencies changed during the M7D run")
    frozen_assets = _protected_fingerprint()
    metadata = _build_metadata(
        inputs,
        result,
        output_hashes=output_hashes,
        frozen_assets=frozen_assets,
        frozen_assets_hash=_fingerprint_hash(frozen_assets),
        frozen_dependencies=dependencies_after,
        frozen_dependencies_hash=_fingerprint_hash(dependencies_after),
        elapsed_seconds=time.perf_counter() - started,
    )
    _write_json(metadata, output_dir / "run_metadata.json")
    return output_dir, metadata


def _compare_table(
    checks: list[CheckResult], name: str, expected: pd.DataFrame, path: Path
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


def _expected_dependency_hashes(
    config: PlanConfig,
    metadata: Mapping[str, Any],
    membership: ProtectedDependencyMembership,
) -> dict[str, str]:
    """Return the expected hash of every declared M7D dependency.

    The primary anchor is the committed M7D planning configuration: the
    upstream input hashes are the configuration-pinned values, and the
    planning configuration's own expected hash is the value recorded when M7D
    ran. The hashes are never regenerated from the current tree.
    """
    expected = {pin.path: pin.sha256 for pin in config.inputs}
    recorded_config_hash = str(metadata.get("plan", {}).get("config_sha256", ""))
    for relative in membership.config_files:
        expected[relative] = recorded_config_hash
    return expected


def _verify_protected_dependencies(
    config: PlanConfig,
    metadata: Mapping[str, Any],
    output_dir: Path,
) -> list[CheckResult]:
    """Verify M7D's declared protected dependencies.

    The contract is scoped to the dependencies M7D actually declares, so an
    unrelated configuration or dataset added to a shared directory later does
    not invalidate M7D, while a modified, deleted or substituted declared
    dependency is still detected. The expected hashes are anchored to the
    committed M7D configuration and cross-checked against the recorded M7D run
    snapshot, so a modified manifest fails closed.
    """
    checks: list[CheckResult] = []
    membership = protected_dependency_membership(config)
    recorded = metadata.get("frozen_assets")
    if not isinstance(recorded, dict) or not recorded:
        checks.append(
            CheckResult(
                "frozen_dependency_manifest",
                False,
                "recorded frozen-asset snapshot is missing or malformed",
            )
        )
        return checks
    manifest_intact = metadata.get("frozen_assets_hash") == _fingerprint_hash(recorded)
    checks.append(
        CheckResult(
            "frozen_dependency_manifest",
            manifest_intact,
            (
                "recorded frozen-asset snapshot is internally consistent"
                if manifest_intact
                else "recorded frozen-asset snapshot does not match its recorded hash"
            ),
        )
    )
    if not manifest_intact:
        return checks
    expected = _expected_dependency_hashes(config, metadata, membership)
    recorded_subset = {
        relative: digest for relative, digest in recorded.items() if relative in expected
    }
    anchors_agree = recorded_subset == expected
    checks.append(
        CheckResult(
            "frozen_dependency_anchors",
            anchors_agree,
            (
                "the recorded M7D snapshot and the committed configuration agree on "
                f"{len(expected)} declared dependencies"
                if anchors_agree
                else "the recorded M7D snapshot disagrees with the committed configuration"
            ),
        )
    )
    if not anchors_agree:
        return checks
    state = _scoped_dependency_state(membership, expected)
    failed = {key: value for key, value in state.items() if value}
    if failed:
        detail = "; ".join(f"{key}: {value[:5]}" for key, value in sorted(failed.items()))
        checks.append(
            CheckResult(
                "frozen_dependencies",
                False,
                f"{len(expected)} declared dependencies checked; {detail}",
            )
        )
    else:
        checks.append(
            CheckResult(
                "frozen_dependencies",
                True,
                f"{len(expected)} declared protected dependencies are unchanged",
            )
        )
    actual_output_files = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    expected_output_files = sorted(membership.output_files)
    checks.append(
        CheckResult(
            "output_file_set",
            actual_output_files == expected_output_files,
            (
                "the M7D plan output file set is exactly as declared"
                if actual_output_files == expected_output_files
                else (
                    f"unexpected output file set: missing {sorted(set(expected_output_files) - set(actual_output_files))}, "
                    f"unexpected {sorted(set(actual_output_files) - set(expected_output_files))}"
                )
            ),
        )
    )
    return checks


def verify_plan(
    config_path: Path | str,
    *,
    output_reports: Path | str | None = None,
) -> VerificationReport:
    """Replay M7D and verify a written plan directory."""
    config = _config_with_output(load_config(config_path), output_reports)
    output_dir = _safe_output_path(config.run.output_reports)
    checks: list[CheckResult] = []
    if not output_dir.is_dir():
        return VerificationReport(
            dataset=str(output_dir),
            checks=(CheckResult("output_directory", False, f"missing: {output_dir}"),),
        )
    inputs = load_inputs(config)
    result = build_plan(inputs)
    isolation = build_isolation_checks(inputs, result, result.report)
    expected_tables = {
        "input_manifest.csv": inputs.input_manifest,
        "font_inventory.csv": result.font_inventory,
        "direction_assessment.csv": result.direction_assessment,
        "research_questions.csv": result.research_questions,
        "independent_units.csv": result.independent_units,
        "split_hierarchy.csv": result.split_hierarchy,
        "endpoints.csv": result.endpoints,
        "sample_size_scenarios.csv": result.sample_size_scenarios,
        "leakage_controls.csv": result.leakage_controls,
        "success_criteria.csv": result.success_criteria,
        "stopping_rules.csv": result.stopping_rules,
        "claims_and_limits.csv": result.claims_and_limits,
        "evidence_snapshot.csv": result.evidence_snapshot,
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
        actual_protocol = json.loads(
            (output_dir / "plan_manifest.json").read_text(encoding="utf-8")
        )
        if actual_protocol != json_safe(result.protocol):
            raise PlanError("plan_manifest.json does not match the recomputed protocol")
    except Exception as error:  # noqa: BLE001
        checks.append(CheckResult("plan_manifest", False, f"{type(error).__name__}: {error}"))
    else:
        checks.append(CheckResult("plan_manifest", True, "recomputed plan manifest matches"))
    try:
        actual_report = (output_dir / "report.md").read_text(encoding="utf-8")
        if actual_report != result.report:
            raise PlanError("report.md does not match the recomputed report")
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
            metadata.get("plan", {}).get("config_sha256") == config.config_sha256,
            "recorded config SHA-256 matches",
        )
    )
    checks.append(
        CheckResult(
            "config_hash",
            metadata.get("plan", {}).get("config_hash") == config.config_hash,
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
    checks.extend(_verify_protected_dependencies(config, metadata, output_dir))
    return VerificationReport(dataset=str(output_dir), checks=tuple(checks))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="captcha-evaluation-plan",
        description="Read-only M7D evaluation-design and dataset-expansion plan.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="build the M7D research plan")
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--output-reports", type=Path)
    run_parser.add_argument("--overwrite", action="store_true")
    verify_parser = subparsers.add_parser("verify", help="replay and verify a written M7D plan")
    verify_parser.add_argument("--config", required=True, type=Path)
    verify_parser.add_argument("--output-reports", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            output_dir, metadata = run_plan(
                args.config,
                output_reports=args.output_reports,
                overwrite=args.overwrite,
            )
            print(f"wrote M7D plan to {output_dir}")
            print(json.dumps(json_safe(metadata["summary"]), indent=2, sort_keys=True))
            return 0
        report = verify_plan(
            args.config,
            output_reports=args.output_reports,
        )
        for check in report.checks:
            status = "PASS" if check.passed else "FAIL"
            print(f"[{status}] {check.name}: {check.detail}")
        print(f"verification: {'PASS' if report.ok else 'FAIL'}")
        return 0 if report.ok else 1
    except (PlanError, PlanConfigError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
