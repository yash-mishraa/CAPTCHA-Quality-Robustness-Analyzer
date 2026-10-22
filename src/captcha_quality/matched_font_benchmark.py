"""Milestone M8C matched cross-font generalisation benchmark execution.

M8C executes the frozen Milestone 8B protocol. It generates the authorised
6,300-image benchmark dataset, evaluates the exact frozen M5C champion
zero-shot, and writes the fifteen authorised report files. It never trains,
fine-tunes, calibrates, refits or selects a model, and it never changes the
feature extractor, the renderer, the corruption ladder or the target.

The design is matched by construction: thirty committed text cases are rendered
in ten pinned typefaces, so every font observes the same text and the same
derived corruption-seed schedule. The target is synthetic-reference SSIM; PSNR
is diagnostic only. Nothing here measures human readability, CAPTCHA
solvability, security or production robustness.

Every pinned input is verified by SHA-256 before generation. Every produced
artifact is verified before it is promoted out of an isolated staging directory,
so a failure never leaves a partial dataset at an authorised path.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
from PIL import Image

from captcha_quality import benchmark_lifecycle as bl
from captcha_quality.benchmark_preflight import (
    CaseManifest,
    M8BExecutionProtocol,
    load_case_manifest,
    load_execution_protocol,
    plan_rows,
    plan_sources,
    severity_map,
)
from captcha_quality.corruptions import (
    FAMILY_BLUR,
    FAMILY_NOISE,
    SeverityLadder,
    apply_corruption,
)
from captcha_quality.dataset import CheckResult, VerificationReport
from captcha_quality.features import (
    FEATURE_NAMES,
    extract_feature_table,
)
from captcha_quality.geometric import ElasticFieldSpec, apply_elastic
from captcha_quality.interference import (
    VARIANT_ARC_STROKES,
    VARIANT_THIN_LINES,
    InterferenceLadder,
    apply_strokes,
    plan_stroke_pool,
)
from captcha_quality.modelling import (
    TARGET,
    predict,
    regression_metrics,
    sklearn_version,
)
from captcha_quality.provenance import (
    code_revision,
    json_safe,
    package_version,
    sha256_bytes,
    sha256_file,
)
from captcha_quality.quality_metrics import compute_pair_metrics
from captcha_quality.render import RenderConfig, ResolvedFont, render_clean
from captcha_quality.source_identity import (
    CLEAN_FAMILY,
    CORRUPTION_FAMILIES,
    image_relpath,
)

# ---------------------------------------------------------------------------
# frozen constants
# ---------------------------------------------------------------------------

DEFAULT_PROTOCOL: Final[Path] = Path("configs/cross_font_benchmark_protocol_v0_2_0.toml")
DEFAULT_LIFECYCLE: Final[Path] = Path("configs/benchmark_lifecycle_v0_2_0.toml")
DEFAULT_TEXT_MANIFEST: Final[Path] = Path("configs/cross_font_text_cases_v0_1_0.toml")

DATASET_VERSION: Final[str] = "matched-font-benchmark-v0.1.0"
DATASET_LOGICAL_PATH: Final[str] = "data/processed/matched_font_benchmark_v0.1.0"
REPORT_LOGICAL_PATH: Final[str] = "reports/matched_font_benchmark_v0_1_0"
REPORT_SCHEMA_VERSION: Final[str] = "m8c-report-v1"
RUN_METADATA_SCHEMA: Final[str] = "m8c-run-metadata-v1"

NEAR_DEGENERATE_SD: Final[float] = 0.005
N_BOOTSTRAP: Final[int] = 2000
BOOTSTRAP_SEED: Final[int] = 20260821
BOOTSTRAP_LOW: Final[float] = 2.5
BOOTSTRAP_HIGH: Final[float] = 97.5

BASELINE_METRICS_PATH: Final[Path] = Path(
    "reports/retention_aware_selection_v0_1_0/test_slice_metrics.csv"
)
BASELINE_METRICS_SHA256: Final[str] = (
    "1dd6d630ca61caeabf907a53c758031019eeef47c5b7898f501b020322c9c9e1"
)
BASELINE_MODEL: Final[str] = "expanded_random_forest"

DATASET_FILES: Final[tuple[str, ...]] = (
    "manifest.csv",
    "sources.csv",
    "source_cases.csv",
    "ground_truth.csv",
    "features.csv",
    "dataset_summary.json",
)

REPORT_FILES: Final[tuple[str, ...]] = (
    "predictions.csv",
    "overall_metrics.csv",
    "font_metrics.csv",
    "family_metrics.csv",
    "severity_metrics.csv",
    "group_metrics.csv",
    "source_metrics.csv",
    "target_distributions.csv",
    "uncertainty.csv",
    "comparison.csv",
    "dataset_composition.csv",
    "isolation_checks.csv",
    "input_manifest.csv",
    "report.md",
    "run_metadata.json",
)

IMAGE_SUBDIRECTORIES: Final[tuple[str, ...]] = (
    CLEAN_FAMILY,
    "gaussian_blur",
    "gaussian_noise",
    "thin_lines",
    "arc_strokes",
    "elastic",
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

OVERALL_SCOPES: Final[tuple[str, ...]] = (
    "all_primary",
    "original_blur_noise",
    "gaussian_blur",
    "gaussian_noise",
    "interference",
    "thin_lines",
    "arc_strokes",
    "geometric_primary",
    "clean_reference",
    "diagnostic_0_5px",
)

DIAGNOSTIC_SCOPES: Final[frozenset[str]] = frozenset({"clean_reference", "diagnostic_0_5px"})

FAMILY_ORDER: Final[tuple[str, ...]] = (
    "gaussian_blur",
    "gaussian_noise",
    "thin_lines",
    "arc_strokes",
    "elastic",
)

GROUP_ORDER: Final[tuple[str, ...]] = ("familiar", "m7b_held_out", "new_held_out")

BASELINE_SCOPE_MAP: Final[dict[str, str]] = {
    "m3_test_overall": "all_primary",
    "original_blur_noise": "original_blur_noise",
    "original_blur_noise_gaussian_blur": "gaussian_blur",
    "original_blur_noise_gaussian_noise": "gaussian_noise",
    "clean_reference": "clean_reference",
    "diagnostic_0_5": "diagnostic_0_5px",
}


class MatchedFontBenchmarkError(RuntimeError):
    """Raised when the benchmark cannot proceed safely."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _check(checks: list[CheckResult], name: str, function: Callable[[], Any]) -> None:
    """Run one named check and record its outcome; never swallow silently."""
    try:
        detail = function()
    except Exception as error:  # noqa: BLE001 - every incompatibility is reported
        checks.append(CheckResult(name, False, f"{type(error).__name__}: {error}"))
    else:
        checks.append(CheckResult(name, True, str(detail)))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MatchedFontBenchmarkError(message)


def _require_report(report: VerificationReport, label: str) -> None:
    if report.ok:
        return
    failures = "; ".join(f"{check.name}: {check.detail}" for check in report.failures())
    raise MatchedFontBenchmarkError(f"{label} failed: {failures}")


def _read_csv_exact(path: Path, columns: Sequence[str], label: str) -> pd.DataFrame:
    if not path.is_file():
        raise MatchedFontBenchmarkError(f"missing {label}: {path}")
    frame = pd.read_csv(path)
    actual = tuple(frame.columns)
    expected = tuple(columns)
    if actual != expected:
        raise MatchedFontBenchmarkError(
            f"{label} columns must be exactly {list(expected)}, got {list(actual)}"
        )
    return frame


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _pixel_sha256(image: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(image, dtype=np.uint8).tobytes())


def _save_png(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(image, dtype=np.uint8)
    Image.fromarray(array, mode="L").save(path, format="PNG")


def _load_png(path: Path) -> tuple[np.ndarray, str]:
    if not path.is_file():
        raise MatchedFontBenchmarkError(f"missing image: {path}")
    with Image.open(path) as handle:
        image_format = handle.format
        mode = handle.mode
        size = handle.size
        if image_format != "PNG":
            raise MatchedFontBenchmarkError(f"{path}: format is {image_format!r}, expected PNG")
        if mode != "L":
            raise MatchedFontBenchmarkError(f"{path}: mode is {mode!r}, expected L")
        if size != (192, 64):
            raise MatchedFontBenchmarkError(f"{path}: size is {size}, expected (192, 64)")
        array = np.asarray(handle, dtype=np.uint8)
    if array.shape != (64, 192):
        raise MatchedFontBenchmarkError(
            f"{path}: decoded shape is {array.shape}, expected (64, 192)"
        )
    return array, _pixel_sha256(array)


def _font_meta(protocol: M8BExecutionProtocol) -> dict[str, dict[str, Any]]:
    groups: dict[str, str] = {}
    for group, payload in protocol.raw.get("font_groups", {}).items():
        for member in payload.get("members", []):
            groups[str(member)] = str(group)
    return {
        str(font["key"]): dict(font) | {"group": groups[str(font["key"])]}
        for font in protocol.fonts
    }


def _resolve_fonts(
    protocol: M8BExecutionProtocol,
) -> dict[str, ResolvedFont]:
    """Resolve every pinned font, failing closed on a missing or changed file."""
    resolved: dict[str, ResolvedFont] = {}
    for font in protocol.fonts:
        key = str(font["key"])
        path = Path(str(font["path"]))
        expected = str(font["sha256"])
        if not path.is_file():
            raise MatchedFontBenchmarkError(f"missing pinned font {key}: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise MatchedFontBenchmarkError(
                f"font {key} hash mismatch: expected {expected}, observed {observed}"
            )
        resolved[key] = ResolvedFont(key=key, path=path)
    return resolved


def _render_config(protocol: M8BExecutionProtocol, text_length: int) -> RenderConfig:
    renderer = protocol.renderer
    return RenderConfig(
        canvas=(int(renderer["canvas_width"]), int(renderer["canvas_height"])),
        charset=str(protocol.text_cases["charset"]),
        text_length=text_length,
        fit_fraction=float(renderer["fit_fraction"]),
        vertical_padding=float(renderer["vertical_padding"]),
        jitter_fraction=float(renderer["jitter_fraction"]),
        spacing_jitter_fraction=float(renderer["spacing_jitter_fraction"]),
        placement_jitter_fraction=float(renderer["placement_jitter_fraction"]),
        background_mode=str(renderer["background_mode"]),
        background_range=(int(renderer["background_min"]), int(renderer["background_max"])),
        center_ink=bool(renderer["center_ink"]),
        strict_bounds=bool(renderer["strict_bounds"]),
    )


def _build_ladders(
    protocol: M8BExecutionProtocol,
) -> dict[str, SeverityLadder | InterferenceLadder]:
    corruptions = protocol.corruptions
    return {
        FAMILY_BLUR: SeverityLadder(
            family=FAMILY_BLUR,
            parameters=(0.0, *(float(value) for value in corruptions["blur_radii_px"])),
            unit="px",
        ),
        FAMILY_NOISE: SeverityLadder(
            family=FAMILY_NOISE,
            parameters=(0.0, *(float(value) for value in corruptions["noise_sigmas"])),
            unit="sigma",
        ),
        "thin_lines": InterferenceLadder(
            variant=VARIANT_THIN_LINES,
            stroke_counts=tuple(int(value) for value in corruptions["thin_line_stroke_counts"]),
            width_px=int(corruptions["stroke_width_px"]),
            intensity=int(corruptions["stroke_intensity"]),
            line_angle_deg=tuple(float(value) for value in corruptions["line_angle_deg"]),
            arc_sweep_deg=tuple(float(value) for value in corruptions["arc_sweep_deg"]),
            arc_radius_fraction=tuple(float(value) for value in corruptions["arc_radius_fraction"]),
            arc_aspect_ratio=tuple(float(value) for value in corruptions["arc_aspect_ratio"]),
        ),
        "arc_strokes": InterferenceLadder(
            variant=VARIANT_ARC_STROKES,
            stroke_counts=tuple(int(value) for value in corruptions["arc_stroke_counts"]),
            width_px=int(corruptions["stroke_width_px"]),
            intensity=int(corruptions["stroke_intensity"]),
            line_angle_deg=tuple(float(value) for value in corruptions["line_angle_deg"]),
            arc_sweep_deg=tuple(float(value) for value in corruptions["arc_sweep_deg"]),
            arc_radius_fraction=tuple(float(value) for value in corruptions["arc_radius_fraction"]),
            arc_aspect_ratio=tuple(float(value) for value in corruptions["arc_aspect_ratio"]),
        ),
    }


# ---------------------------------------------------------------------------
# dataset generation
# ---------------------------------------------------------------------------


def _apply_variant(
    family: str,
    severity: int,
    image: np.ndarray,
    ladders: Mapping[str, Any],
    protocol: M8BExecutionProtocol,
    *,
    seed: int,
) -> tuple[np.ndarray, float | None, dict[str, Any]]:
    """Apply exactly one frozen corruption and record its realised parameters."""
    if family == FAMILY_BLUR:
        result = apply_corruption(image, ladders[FAMILY_BLUR], severity, noise_seed=seed)
        parameters = {"radius_px": float(result.spec.parameter)}
        return result.image, None, parameters
    if family == FAMILY_NOISE:
        result = apply_corruption(image, ladders[FAMILY_NOISE], severity, noise_seed=seed)
        parameters = {
            "sigma": float(result.spec.parameter),
            "realised_noise_std": float(result.realised_noise_std),
            "clipped_fraction": float(result.clipped_fraction),
        }
        return result.image, None, parameters
    if family in (VARIANT_THIN_LINES, VARIANT_ARC_STROKES):
        ladder = ladders[family]
        pool = plan_stroke_pool(image, ladder, seed=seed)
        count = int(ladder.stroke_counts[severity - 1])
        result = apply_strokes(
            image,
            pool[:count],
            width_px=int(ladder.width_px),
            intensity=int(ladder.intensity),
        )
        parameters = {
            "stroke_count": count,
            "stroke_width_px": int(ladder.width_px),
            "stroke_intensity": int(ladder.intensity),
        }
        return result, None, parameters
    if family == "elastic":
        corruptions = protocol.corruptions
        amplitude = float(corruptions["elastic_severity_amplitudes_px"][severity - 1])
        field_spec = ElasticFieldSpec(
            height=int(image.shape[0]),
            width=int(image.shape[1]),
            seed=seed,
            grid_rows=int(corruptions["grid_rows"]),
            grid_cols=int(corruptions["grid_cols"]),
            smoothing_sigma_px=float(corruptions["smoothing_sigma_px"]),
            interpolation=str(corruptions["interpolation"]),
            border_mode=str(corruptions["border_mode"]),
        )
        result = apply_elastic(image, field_spec=field_spec, amplitude_px=amplitude)
        parameters = {
            "amplitude_px": amplitude,
            "grid_rows": int(corruptions["grid_rows"]),
            "grid_cols": int(corruptions["grid_cols"]),
            "smoothing_sigma_px": float(corruptions["smoothing_sigma_px"]),
            "interpolation": str(corruptions["interpolation"]),
            "border_mode": str(corruptions["border_mode"]),
            "changed_pixel_fraction": float(result.changed_pixel_fraction),
            "outside_fraction": float(result.outside_fraction),
        }
        return result.image, amplitude, parameters
    raise MatchedFontBenchmarkError(f"unexpected corruption family {family!r}")


def _corruption_json(parameters: Mapping[str, Any]) -> str:
    return json.dumps(dict(parameters), sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    """A loaded benchmark dataset plus its parsed summary."""

    root: Path
    manifest: pd.DataFrame
    sources: pd.DataFrame
    source_cases: pd.DataFrame
    ground_truth: pd.DataFrame
    features: pd.DataFrame
    summary: dict[str, Any]


def _build_summary(
    protocol: M8BExecutionProtocol,
    case_manifest: CaseManifest,
    sources: Sequence[Any],
    rows: Sequence[Any],
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    font_meta: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    image_hashes = sorted(set(manifest["image_sha256"].tolist()))
    reference_hashes = sorted(set(manifest["reference_sha256"].tolist()))
    per_subdirectory = {
        name: int((manifest["image_path"].str.split("/").str[1] == name).sum())
        for name in IMAGE_SUBDIRECTORIES
    }
    font_counts: dict[str, Any] = {}
    for key, meta in font_meta.items():
        subset = manifest[manifest["font_id"] == key]
        font_counts[key] = {
            "family": meta["family"],
            "group": meta["group"],
            "role": meta["role"],
            "stratum": meta["stratum"],
            "sha256": meta["sha256"],
            "path": meta["path"],
            "sources": int((manifest["font_id"] == key).sum() // 21),
            "rows": int(len(subset)),
            "primary_rows": int((subset["protocol_scope"] == "primary").sum()),
            "diagnostic_rows": int((subset["protocol_scope"] == "diagnostic").sum()),
        }
    group_counts: dict[str, Any] = {}
    for group in GROUP_ORDER:
        members = [key for key, meta in font_meta.items() if meta["group"] == group]
        subset = manifest[manifest["font_id"].isin(members)]
        group_counts[group] = {
            "fonts": len(members),
            "sources": len(members) * len(case_manifest.cases),
            "rows": int(len(subset)),
            "primary_rows": int((subset["protocol_scope"] == "primary").sum()),
            "diagnostic_rows": int((subset["protocol_scope"] == "diagnostic").sum()),
        }
    target_rows: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        for severity in severity_map(protocol)[family]:
            subset = manifest[
                (manifest["variant_family"] == family) & (manifest["severity_level"] == severity)
            ]
            values = subset["ssim_target"].to_numpy(dtype=np.float64)
            target_rows.append(
                {
                    "family": family,
                    "severity": severity,
                    "n": int(values.size),
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=0)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
    return {
        "dataset_name": "matched-font-benchmark",
        "dataset_version": DATASET_VERSION,
        "milestone": "M8C",
        "protocol": {
            "name": "cross-font-benchmark",
            "version": protocol.version,
            "sha256": protocol.file_sha256,
            "generation_config_hash": str(protocol.identity["generation_config_hash"]),
            "seed": int(case_manifest.seed),
            "scheme_version": str(protocol.identity["scheme_version"]),
        },
        "text_cases": {
            "version": case_manifest.version,
            "n_cases": len(case_manifest.cases),
            "file_sha256": case_manifest.file_sha256,
            "manifest_hash": case_manifest.manifest_hash,
            "charset": case_manifest.charset,
            "normalization": case_manifest.normalization,
        },
        "counts": {
            "fonts": len(font_meta),
            "cases": len(case_manifest.cases),
            "sources": len(sources),
            "rows": len(rows),
            "primary_rows": int((manifest["protocol_scope"] == "primary").sum()),
            "diagnostic_rows": int((manifest["protocol_scope"] == "diagnostic").sum()),
            "clean_rows": int((manifest["variant_family"] == CLEAN_FAMILY).sum()),
            "feature_rows": int(len(features)),
            "images": int(len(manifest)),
        },
        "images": {
            "count": int(len(manifest)),
            "per_subdirectory": per_subdirectory,
            "distinct_pixel_hashes": len(image_hashes),
            "distinct_reference_pixel_hashes": len(reference_hashes),
            "image_pixel_hash_set_sha256": sha256_bytes("\n".join(image_hashes).encode("utf-8")),
            "reference_pixel_hash_set_sha256": sha256_bytes(
                "\n".join(reference_hashes).encode("utf-8")
            ),
            "dimensions": [192, 64],
            "mode": "L",
            "format": "PNG",
        },
        "fonts": font_counts,
        "groups": group_counts,
        "targets": {
            "name": TARGET,
            "definition": (
                "SSIM between the corrupted image and its source-specific clean reference; "
                "PSNR is recorded as a diagnostic only"
            ),
            "by_family_severity": target_rows,
        },
        "features": {
            "feature_set_version": "v1",
            "feature_names": list(FEATURE_NAMES),
            "rows": int(len(features)),
        },
        "generation": {
            "renderer": protocol.renderer,
            "corruptions": protocol.corruptions,
            "identity": protocol.identity,
        },
        "hashes": {
            "feature_table_sha256": sha256_bytes(
                features.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode(
                    "utf-8"
                )
            )
        },
    }


def generate_dataset(
    protocol: M8BExecutionProtocol,
    case_manifest: CaseManifest,
    output: Path,
) -> DatasetBundle:
    """Render and write the complete dataset into an isolated staging path."""
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise MatchedFontBenchmarkError(f"refusing to overwrite a non-empty dataset path: {root}")
    root.mkdir(parents=True, exist_ok=True)

    sources = plan_sources(protocol, case_manifest)
    rows = plan_rows(protocol, case_manifest, sources)
    font_meta = _font_meta(protocol)
    fonts = _resolve_fonts(protocol)
    ladders = _build_ladders(protocol)

    rows_by_source: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        rows_by_source[row.source_id].append(row)

    manifest_records: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    source_case_records: list[dict[str, Any]] = []
    ground_truth_records: list[dict[str, Any]] = []
    feature_pairs: list[tuple[str, np.ndarray]] = []

    for source in sources:
        font = fonts[source.font_id]
        config = _render_config(protocol, source.text_length)
        rng = np.random.default_rng(source.render_seed)
        clean = render_clean(font, source.text, config, rng=rng)
        reference = np.asarray(clean.image, dtype=np.uint8)
        reference_hash = _pixel_sha256(reference)
        if reference_hash != clean.sha256:
            raise MatchedFontBenchmarkError(
                f"{source.source_id}: clean render hash mismatch "
                f"({reference_hash} != {clean.sha256})"
            )
        clean_path = image_relpath(CLEAN_FAMILY, source.source_id)
        _save_png(reference, root / clean_path)
        source_records.append(
            {
                "source_id": source.source_id,
                "case_id": source.case_id,
                "case_index": source.case_index,
                "split_group_id": source.split_group_id,
                "split": source.split,
                "font_id": source.font_id,
                "font_path": source.font_path,
                "font_sha256": source.font_sha256,
                "text_length": source.text_length,
                "text_sha256": source.text_sha256,
                "render_seed": source.render_seed,
                "render_config_hash": source.render_config_hash,
                "background_level": int(clean.background_level),
                "height": int(reference.shape[0]),
                "width": int(reference.shape[1]),
                "clean_image_path": clean_path,
                "clean_image_sha256": reference_hash,
                "generation_config_hash": source.generation_config_hash,
                "dataset_version": DATASET_VERSION,
            }
        )
        source_case_records.append(
            {
                "source_id": source.source_id,
                "case_id": source.case_id,
                "case_index": source.case_index,
                "font_id": source.font_id,
                "text_length": source.text_length,
                "text_sha256": source.text_sha256,
                "render_seed": source.render_seed,
                "split": source.split,
                "split_group_id": source.split_group_id,
            }
        )
        ground_truth_records.append(
            {
                "source_id": source.source_id,
                "case_id": source.case_id,
                "font_id": source.font_id,
                "text": source.text,
            }
        )

        for row in rows_by_source[source.source_id]:
            if row.family == CLEAN_FAMILY:
                image = reference
                amplitude = None
                parameters: dict[str, Any] = {}
            else:
                image, amplitude, parameters = _apply_variant(
                    row.family,
                    row.severity,
                    reference,
                    ladders,
                    protocol,
                    seed=row.variant_seed,
                )
            image = np.ascontiguousarray(image, dtype=np.uint8)
            relative = row.image_path
            _save_png(image, root / relative)
            image_hash = _pixel_sha256(image)
            metrics = compute_pair_metrics(reference, image, ("ssim", "psnr"))
            ssim_target = float(metrics["ssim"])
            psnr_target = float(metrics["psnr"])
            if not math.isfinite(ssim_target):
                raise MatchedFontBenchmarkError(
                    f"{row.sample_id}: non-finite SSIM target {ssim_target!r}"
                )
            manifest_records.append(
                {
                    "row_id": row.row_id,
                    "sample_id": row.sample_id,
                    "source_id": row.source_id,
                    "split_group_id": source.split_group_id,
                    "split": source.split,
                    "case_id": source.case_id,
                    "case_index": source.case_index,
                    "font_id": source.font_id,
                    "font_sha256": source.font_sha256,
                    "text_length": source.text_length,
                    "text_sha256": source.text_sha256,
                    "render_seed": source.render_seed,
                    "variant_id": row.variant_id,
                    "variant_family": row.family,
                    "severity_level": row.severity,
                    "amplitude_px": amplitude,
                    "variant_seed": row.variant_seed,
                    "image_path": relative,
                    "reference_path": clean_path,
                    "ssim_target": ssim_target,
                    "psnr_target": None if math.isinf(psnr_target) else psnr_target,
                    "psnr_infinite": bool(math.isinf(psnr_target)),
                    "height": int(image.shape[0]),
                    "width": int(image.shape[1]),
                    "image_sha256": image_hash,
                    "reference_sha256": reference_hash,
                    "dataset_version": DATASET_VERSION,
                    "generation_config_hash": source.generation_config_hash,
                    "protocol_scope": "primary" if row.primary else "diagnostic",
                    "primary": bool(row.primary),
                    "corruption_parameters_json": _corruption_json(parameters),
                }
            )
            feature_pairs.append((row.sample_id, image))

    manifest_frame = pd.DataFrame(manifest_records, columns=MANIFEST_COLUMNS)
    sources_frame = pd.DataFrame(source_records, columns=SOURCE_COLUMNS)
    source_cases_frame = pd.DataFrame(source_case_records, columns=SOURCE_CASE_COLUMNS)
    ground_truth_frame = pd.DataFrame(ground_truth_records, columns=GROUND_TRUTH_COLUMNS)
    features_frame = extract_feature_table(feature_pairs)
    summary = _build_summary(
        protocol,
        case_manifest,
        sources,
        rows,
        manifest_frame,
        features_frame,
        font_meta,
    )

    _write_csv(manifest_frame, root / "manifest.csv")
    _write_csv(sources_frame, root / "sources.csv")
    _write_csv(source_cases_frame, root / "source_cases.csv")
    _write_csv(ground_truth_frame, root / "ground_truth.csv")
    _write_csv(features_frame, root / "features.csv")
    _write_json(summary, root / "dataset_summary.json")

    return DatasetBundle(
        root=root,
        manifest=manifest_frame,
        sources=sources_frame,
        source_cases=source_cases_frame,
        ground_truth=ground_truth_frame,
        features=features_frame,
        summary=summary,
    )


def load_dataset(root: Path) -> DatasetBundle:
    """Load a written dataset with exact column checks."""
    root = Path(root)
    if not root.is_dir():
        raise MatchedFontBenchmarkError(f"benchmark dataset not found: {root}")
    manifest = _read_csv_exact(root / "manifest.csv", MANIFEST_COLUMNS, "M8C manifest")
    sources = _read_csv_exact(root / "sources.csv", SOURCE_COLUMNS, "M8C sources")
    source_cases = _read_csv_exact(
        root / "source_cases.csv", SOURCE_CASE_COLUMNS, "M8C source_cases"
    )
    ground_truth = _read_csv_exact(
        root / "ground_truth.csv", GROUND_TRUTH_COLUMNS, "M8C ground_truth"
    )
    features = _read_csv_exact(root / "features.csv", FEATURE_COLUMNS, "M8C features")
    summary_path = root / "dataset_summary.json"
    if not summary_path.is_file():
        raise MatchedFontBenchmarkError(f"missing dataset summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise MatchedFontBenchmarkError("dataset_summary.json must be a JSON object")
    return DatasetBundle(
        root=root,
        manifest=manifest,
        sources=sources,
        source_cases=source_cases,
        ground_truth=ground_truth,
        features=features,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# dataset verification
# ---------------------------------------------------------------------------


def _dataset_membership(root: Path) -> str:
    if not root.is_dir():
        raise MatchedFontBenchmarkError(f"dataset directory does not exist: {root}")
    top = {path.name for path in root.iterdir()}
    expected = set(DATASET_FILES) | {"images"}
    if top != expected:
        missing = sorted(expected - top)
        extra = sorted(top - expected)
        raise MatchedFontBenchmarkError(
            f"dataset membership mismatch; missing={missing}, unexpected={extra}"
        )
    images = root / "images"
    if not images.is_dir():
        raise MatchedFontBenchmarkError("dataset is missing the images directory")
    subdirectories = {path.name for path in images.iterdir() if path.is_dir()}
    if subdirectories != set(IMAGE_SUBDIRECTORIES):
        raise MatchedFontBenchmarkError(
            f"image subdirectories are {sorted(subdirectories)}, "
            f"expected {sorted(IMAGE_SUBDIRECTORIES)}"
        )
    stray = [path.name for path in images.iterdir() if not path.is_dir()]
    if stray:
        raise MatchedFontBenchmarkError(f"unexpected files directly under images/: {stray}")
    return (
        f"exactly {len(DATASET_FILES)} dataset files and {len(IMAGE_SUBDIRECTORIES)} subdirectories"
    )


def _verify_image_membership(root: Path, manifest: pd.DataFrame) -> str:
    referenced = set(manifest["image_path"].tolist())
    observed = {path.relative_to(root).as_posix() for path in (root / "images").rglob("*.png")}
    if observed != referenced:
        missing = sorted(referenced - observed)
        extra = sorted(observed - referenced)
        raise MatchedFontBenchmarkError(
            f"image membership mismatch; missing={missing[:5]} ({len(missing)} total), "
            f"unexpected={extra[:5]} ({len(extra)} total)"
        )
    if len(observed) != 6300:
        raise MatchedFontBenchmarkError(f"expected 6,300 images, found {len(observed)}")
    per_directory: dict[str, int] = {}
    for relative in observed:
        family = relative.split("/")[1]
        per_directory[family] = per_directory.get(family, 0) + 1
    expected_counts = {
        "clean": 300,
        "gaussian_blur": 1200,
        "gaussian_noise": 1200,
        "thin_lines": 1200,
        "arc_strokes": 1200,
        "elastic": 1200,
    }
    if per_directory != expected_counts:
        raise MatchedFontBenchmarkError(
            f"per-subdirectory image counts are {per_directory}, expected {expected_counts}"
        )
    return "6,300 referenced images across exactly six subdirectories"


def _verify_image_content(
    root: Path,
    bundle: DatasetBundle,
    *,
    verify_features: bool,
) -> str:
    manifest = bundle.manifest
    features = bundle.features.set_index("sample_id")
    references: dict[str, tuple[np.ndarray, str]] = {}
    feature_pairs: list[tuple[str, np.ndarray]] = []
    for row in manifest.itertuples(index=False):
        source_id = str(row.source_id)
        reference_path = root / str(row.reference_path)
        reference = references.get(source_id)
        if reference is None:
            reference_array, reference_hash = _load_png(reference_path)
            if reference_hash != str(row.reference_sha256):
                raise MatchedFontBenchmarkError(
                    f"{source_id}: reference pixel hash {reference_hash} does not match "
                    f"{row.reference_sha256}"
                )
            reference = (reference_array, reference_hash)
            references[source_id] = reference
        image, image_hash = _load_png(root / str(row.image_path))
        if image_hash != str(row.image_sha256):
            raise MatchedFontBenchmarkError(
                f"{row.sample_id}: image pixel hash {image_hash} does not match {row.image_sha256}"
            )
        if (int(row.height), int(row.width)) != image.shape:
            raise MatchedFontBenchmarkError(
                f"{row.sample_id}: recorded dimensions {(row.height, row.width)} != {image.shape}"
            )
        metrics = compute_pair_metrics(reference[0], image, ("ssim", "psnr"))
        ssim_target = float(metrics["ssim"])
        if not math.isclose(ssim_target, float(row.ssim_target), rel_tol=0.0, abs_tol=1e-12):
            raise MatchedFontBenchmarkError(
                f"{row.sample_id}: recomputed SSIM {ssim_target!r} != recorded {row.ssim_target!r}"
            )
        psnr = float(metrics["psnr"])
        infinite = bool(row.psnr_infinite)
        if math.isinf(psnr) != infinite:
            raise MatchedFontBenchmarkError(
                f"{row.sample_id}: PSNR infinity flag {infinite} != recomputed {math.isinf(psnr)}"
            )
        if not infinite and not math.isclose(
            psnr, float(row.psnr_target), rel_tol=0.0, abs_tol=1e-9
        ):
            raise MatchedFontBenchmarkError(
                f"{row.sample_id}: recomputed PSNR {psnr!r} != recorded {row.psnr_target!r}"
            )
        if verify_features:
            feature_pairs.append((str(row.sample_id), image))
    if verify_features:
        if len(feature_pairs) != len(manifest):
            raise MatchedFontBenchmarkError("feature verification did not cover every image")
        recomputed = extract_feature_table(feature_pairs).set_index("sample_id")
        recomputed = recomputed.loc[features.index]
        for name in FEATURE_NAMES:
            left = features[name].to_numpy(dtype=np.float64)
            right = recomputed[name].to_numpy(dtype=np.float64)
            if not np.allclose(left, right, rtol=0.0, atol=1e-9, equal_nan=False):
                difference = float(np.max(np.abs(left - right)))
                raise MatchedFontBenchmarkError(
                    f"feature {name} does not match the image on disk (max |diff|={difference})"
                )
        if not np.isfinite(features.to_numpy(dtype=np.float64)).all():
            raise MatchedFontBenchmarkError("features.csv contains a non-finite value")
    return (
        f"decoded and re-hashed {len(manifest)} images and {len(references)} references; "
        "targets and features match"
    )


def _verify_planned_identities(
    bundle: DatasetBundle,
    protocol: M8BExecutionProtocol,
    case_manifest: CaseManifest,
) -> str:
    manifest = bundle.manifest
    sources = plan_sources(protocol, case_manifest)
    rows = plan_rows(protocol, case_manifest, sources)
    planned_sources = {source.source_id for source in sources}
    planned_rows = {row.row_id for row in rows}
    planned_samples = {row.sample_id for row in rows}
    planned_images = {row.image_path for row in rows}
    if set(manifest["source_id"]) != planned_sources:
        raise MatchedFontBenchmarkError("manifest source identifiers do not match the plan")
    if set(manifest["row_id"]) != planned_rows:
        raise MatchedFontBenchmarkError("manifest row identifiers do not match the plan")
    if set(manifest["sample_id"]) != planned_samples:
        raise MatchedFontBenchmarkError("manifest sample identifiers do not match the plan")
    if set(manifest["image_path"]) != planned_images:
        raise MatchedFontBenchmarkError("manifest image paths do not match the plan")
    if len(manifest) != len(rows):
        raise MatchedFontBenchmarkError(
            f"manifest has {len(manifest)} rows, the plan has {len(rows)}"
        )
    return f"all {len(rows)} row identities and {len(sources)} source identities match the plan"


def _verify_source_tables(
    bundle: DatasetBundle,
    protocol: M8BExecutionProtocol,
    case_manifest: CaseManifest,
) -> str:
    manifest = bundle.manifest
    sources = plan_sources(protocol, case_manifest)
    if len(bundle.sources) != len(sources) or len(bundle.source_cases) != len(sources):
        raise MatchedFontBenchmarkError(
            f"sources/source_cases hold {len(bundle.sources)}/{len(bundle.source_cases)} rows, "
            f"expected {len(sources)}"
        )
    if len(bundle.ground_truth) != len(sources):
        raise MatchedFontBenchmarkError(
            f"ground_truth holds {len(bundle.ground_truth)} rows, expected {len(sources)}"
        )
    by_source = {source.source_id: source for source in sources}
    if set(bundle.sources["source_id"]) != set(by_source):
        raise MatchedFontBenchmarkError("sources.csv source identifiers do not match the plan")
    if set(bundle.source_cases["source_id"]) != set(by_source):
        raise MatchedFontBenchmarkError("source_cases.csv source identifiers do not match the plan")
    if set(bundle.ground_truth["source_id"]) != set(by_source):
        raise MatchedFontBenchmarkError("ground_truth.csv source identifiers do not match the plan")
    for frame, label in (
        (bundle.sources, "sources.csv"),
        (bundle.source_cases, "source_cases.csv"),
        (bundle.ground_truth, "ground_truth.csv"),
    ):
        if frame["source_id"].duplicated().any():
            raise MatchedFontBenchmarkError(f"{label} contains duplicate source identifiers")
    for row in bundle.sources.itertuples(index=False):
        source = by_source[str(row.source_id)]
        if str(row.case_id) != source.case_id:
            raise MatchedFontBenchmarkError(f"{row.source_id}: case_id mismatch")
        if int(row.case_index) != source.case_index:
            raise MatchedFontBenchmarkError(f"{row.source_id}: case_index mismatch")
        if str(row.font_id) != source.font_id:
            raise MatchedFontBenchmarkError(f"{row.source_id}: font_id mismatch")
        if str(row.font_sha256) != source.font_sha256:
            raise MatchedFontBenchmarkError(f"{row.source_id}: font hash mismatch")
        if int(row.text_length) != source.text_length:
            raise MatchedFontBenchmarkError(f"{row.source_id}: text length mismatch")
        if str(row.text_sha256) != source.text_sha256:
            raise MatchedFontBenchmarkError(f"{row.source_id}: text hash mismatch")
        if int(row.render_seed) != source.render_seed:
            raise MatchedFontBenchmarkError(f"{row.source_id}: render seed mismatch")
        if str(row.clean_image_path) != image_relpath(CLEAN_FAMILY, source.source_id):
            raise MatchedFontBenchmarkError(f"{row.source_id}: clean image path mismatch")
        if str(row.dataset_version) != DATASET_VERSION:
            raise MatchedFontBenchmarkError(f"{row.source_id}: dataset version mismatch")
    for row in bundle.source_cases.itertuples(index=False):
        source = by_source[str(row.source_id)]
        if (
            str(row.case_id),
            int(row.case_index),
            str(row.font_id),
            int(row.text_length),
            str(row.text_sha256),
            int(row.render_seed),
        ) != (
            source.case_id,
            source.case_index,
            source.font_id,
            source.text_length,
            source.text_sha256,
            source.render_seed,
        ):
            raise MatchedFontBenchmarkError(f"{row.source_id}: source_cases lineage mismatch")
    for row in bundle.ground_truth.itertuples(index=False):
        source = by_source[str(row.source_id)]
        if str(row.text) != source.text:
            raise MatchedFontBenchmarkError(f"{row.source_id}: ground-truth text mismatch")
        if str(row.case_id) != source.case_id or str(row.font_id) != source.font_id:
            raise MatchedFontBenchmarkError(f"{row.source_id}: ground-truth lineage mismatch")
    if set(manifest["source_id"]) != set(bundle.sources["source_id"]):
        raise MatchedFontBenchmarkError("manifest sources do not join to sources.csv")
    return "300 source, source-case and ground-truth rows join exactly to the plan"


def _verify_row_labels(
    bundle: DatasetBundle,
    protocol: M8BExecutionProtocol,
) -> str:
    manifest = bundle.manifest
    severities = severity_map(protocol)
    diagnostic = protocol.identity["diagnostic_severities"]
    expected_variants: dict[str, set[str]] = {}
    for family in CORRUPTION_FAMILIES:
        for severity in severities[family]:
            expected_variants.setdefault(f"variant:{family}", set()).add(f"{family}_s{severity}")
    for row in manifest.itertuples(index=False):
        family = str(row.variant_family)
        severity = int(row.severity_level)
        if family == CLEAN_FAMILY:
            if severity != 0 or str(row.variant_id) != "clean_s0":
                raise MatchedFontBenchmarkError(f"{row.sample_id}: invalid clean variant")
            expected_scope = "diagnostic"
        else:
            if family not in severities or severity not in severities[family]:
                raise MatchedFontBenchmarkError(
                    f"{row.sample_id}: unexpected family/severity {family}/{severity}"
                )
            if str(row.variant_id) != f"{family}_s{severity}":
                raise MatchedFontBenchmarkError(f"{row.sample_id}: variant id mismatch")
            expected_scope = "diagnostic" if severity in diagnostic.get(family, []) else "primary"
        if str(row.protocol_scope) != expected_scope:
            raise MatchedFontBenchmarkError(
                f"{row.sample_id}: protocol_scope {row.protocol_scope!r} != {expected_scope!r}"
            )
        if bool(row.primary) != (expected_scope == "primary"):
            raise MatchedFontBenchmarkError(f"{row.sample_id}: primary flag mismatch")
        if str(row.dataset_version) != DATASET_VERSION:
            raise MatchedFontBenchmarkError(f"{row.sample_id}: dataset version mismatch")
        if str(row.generation_config_hash) != str(protocol.identity["generation_config_hash"]):
            raise MatchedFontBenchmarkError(
                f"{row.sample_id}: generation configuration hash mismatch"
            )
        if family == "elastic":
            amplitude = float(protocol.corruptions["elastic_severity_amplitudes_px"][severity - 1])
            if not math.isclose(float(row.amplitude_px), amplitude, abs_tol=0.0):
                raise MatchedFontBenchmarkError(f"{row.sample_id}: elastic amplitude mismatch")
        elif not pd.isna(row.amplitude_px):
            raise MatchedFontBenchmarkError(
                f"{row.sample_id}: non-elastic row records an amplitude"
            )
        expected_path = (
            image_relpath(CLEAN_FAMILY, str(row.source_id))
            if family == CLEAN_FAMILY
            else image_relpath(family, str(row.sample_id))
        )
        if str(row.image_path) != expected_path:
            raise MatchedFontBenchmarkError(f"{row.sample_id}: image path rule mismatch")
    for source_id, group in manifest.groupby("source_id", sort=False):
        observed = set(group["variant_id"].tolist())
        expected = {"clean_s0"} | {
            f"{family}_s{severity}"
            for family in CORRUPTION_FAMILIES
            for severity in severities[family]
        }
        if observed != expected:
            raise MatchedFontBenchmarkError(
                f"{source_id}: corruption assignment mismatch; missing={sorted(expected - observed)}"
            )
    if int((manifest["protocol_scope"] == "primary").sum()) != 5700:
        raise MatchedFontBenchmarkError("primary row count is not 5,700")
    if int((manifest["protocol_scope"] == "diagnostic").sum()) != 600:
        raise MatchedFontBenchmarkError("diagnostic row count is not 600")
    return "primary/diagnostic labels, severity levels and corruption assignments are correct"


def _verify_summary(bundle: DatasetBundle, protocol: M8BExecutionProtocol) -> str:
    summary = bundle.summary
    manifest = bundle.manifest
    counts = summary.get("counts", {})
    expected = {
        "fonts": 10,
        "cases": 30,
        "sources": 300,
        "rows": 6300,
        "primary_rows": 5700,
        "diagnostic_rows": 600,
        "clean_rows": 300,
        "feature_rows": 6300,
        "images": 6300,
    }
    for key, value in expected.items():
        if int(counts.get(key, -1)) != value:
            raise MatchedFontBenchmarkError(
                f"dataset summary count {key}={counts.get(key)!r}, expected {value}"
            )
    if summary.get("dataset_version") != DATASET_VERSION:
        raise MatchedFontBenchmarkError("dataset summary version mismatch")
    if summary.get("protocol", {}).get("sha256") != protocol.file_sha256:
        raise MatchedFontBenchmarkError("dataset summary protocol hash mismatch")
    per_directory = summary.get("images", {}).get("per_subdirectory", {})
    for name in IMAGE_SUBDIRECTORIES:
        if int(per_directory.get(name, -1)) != int(
            (manifest["image_path"].str.split("/").str[1] == name).sum()
        ):
            raise MatchedFontBenchmarkError(
                f"dataset summary image count for {name} does not match the manifest"
            )
    if len(bundle.features) != len(manifest):
        raise MatchedFontBenchmarkError("features.csv row count does not match the manifest")
    return "dataset summary counts, protocol identity and image totals match the artifacts"


def dataset_checks(
    bundle: DatasetBundle,
    protocol: M8BExecutionProtocol,
    case_manifest: CaseManifest,
    *,
    verify_images: bool = True,
    verify_features: bool = True,
) -> VerificationReport:
    """Verify a written dataset against the frozen plan and its own files."""
    checks: list[CheckResult] = []
    root = bundle.root
    _check(checks, "dataset_membership", lambda: _dataset_membership(root))
    _check(checks, "image_membership", lambda: _verify_image_membership(root, bundle.manifest))
    _check(
        checks,
        "planned_identities",
        lambda: _verify_planned_identities(bundle, protocol, case_manifest),
    )
    _check(
        checks,
        "source_tables",
        lambda: _verify_source_tables(bundle, protocol, case_manifest),
    )
    _check(checks, "row_labels", lambda: _verify_row_labels(bundle, protocol))
    _check(
        checks,
        "feature_join",
        lambda: _verify_feature_join(bundle),
    )
    if verify_images:
        _check(
            checks,
            "image_content",
            lambda: _verify_image_content(root, bundle, verify_features=verify_features),
        )
    _check(checks, "summary", lambda: _verify_summary(bundle, protocol))
    return VerificationReport(dataset=str(root), checks=tuple(checks))


def _verify_feature_join(bundle: DatasetBundle) -> str:
    manifest_ids = set(bundle.manifest["sample_id"])
    feature_ids = set(bundle.features["sample_id"])
    if manifest_ids != feature_ids:
        raise MatchedFontBenchmarkError(
            f"feature identities differ; missing={sorted(manifest_ids - feature_ids)[:5]}, "
            f"unexpected={sorted(feature_ids - manifest_ids)[:5]}"
        )
    if bundle.features["sample_id"].duplicated().any():
        raise MatchedFontBenchmarkError("features.csv contains duplicate sample identifiers")
    values = bundle.features[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise MatchedFontBenchmarkError("features.csv contains a non-finite value")
    return "every feature row joins one-to-one to a dataset row and is finite"


# ---------------------------------------------------------------------------
# frozen champion
# ---------------------------------------------------------------------------


def load_champion(
    protocol: M8BExecutionProtocol,
) -> tuple[Any, dict[str, Any], tuple[CheckResult, ...]]:
    """Load the exact frozen M5C champion; never fit or modify it."""
    model = protocol.raw["model"]
    model_dir = Path(str(model["model_dir"]))
    stem = str(model["model_stem"])
    artifact = model_dir / f"{stem}.joblib"
    card_path = model_dir / f"{stem}_card.json"
    checks: list[CheckResult] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append(CheckResult(name, passed, detail))

    for label, path, expected in (
        ("artifact", artifact, str(model["artifact_sha256"])),
        ("card", card_path, str(model["card_sha256"])),
    ):
        if not path.is_file():
            record(f"model:{label}_hash", False, f"missing {path}")
            continue
        observed = sha256_file(path)
        if observed != expected:
            record(
                f"model:{label}_hash",
                False,
                f"sha256={observed}, expected {expected}",
            )
        else:
            record(f"model:{label}_hash", True, f"sha256={expected}")

    card: dict[str, Any] = {}
    if card_path.is_file():
        try:
            loaded = json.loads(card_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("model card must be a JSON object")
            card = loaded
            record("model:card_parse", True, f"loaded {card_path.name}")
        except Exception as error:  # noqa: BLE001 - report the exact card error
            record("model:card_parse", False, f"{type(error).__name__}: {error}")
    else:
        record("model:card_parse", False, f"missing {card_path}")

    checks.append(
        CheckResult(
            "model:artifact_file",
            card.get("artifact_file") == artifact.name,
            f"card artifact_file={card.get('artifact_file')!r}",
        )
    )
    checks.append(
        CheckResult(
            "model:card_artifact_hash",
            card.get("artifact_sha256") == str(model["artifact_sha256"]),
            "card artifact hash matches the protocol",
        )
    )
    feature_names = list(card.get("feature_names", []))
    checks.append(
        CheckResult(
            "model:feature_schema",
            feature_names == list(FEATURE_NAMES)
            and int(card.get("n_features", -1)) == len(FEATURE_NAMES)
            and card.get("feature_set_version") == str(model["feature_set_version"]),
            f"feature_names={feature_names}, n_features={card.get('n_features')}, "
            f"feature_set_version={card.get('feature_set_version')!r}",
        )
    )
    checks.append(
        CheckResult(
            "model:target",
            card.get("target") == TARGET == str(model["target"]),
            f"target={card.get('target')!r}",
        )
    )
    checks.append(
        CheckResult(
            "model:calibration",
            card.get("calibration") == str(model["expected_calibration"]),
            f"calibration={card.get('calibration')!r}",
        )
    )
    checks.append(
        CheckResult(
            "model:family",
            card.get("model_family") == str(model["expected_model_family"]),
            f"model_family={card.get('model_family')!r}",
        )
    )
    expected_sklearn = str(card.get("sklearn_version", ""))
    installed_sklearn = sklearn_version()
    compatible = (
        installed_sklearn.split(".")[:2] == expected_sklearn.split(".")[:2]
        if expected_sklearn
        else False
    )
    checks.append(
        CheckResult(
            "model:sklearn_version",
            compatible,
            f"installed={installed_sklearn}, card={expected_sklearn}",
        )
    )

    estimator: Any = None
    try:
        estimator = joblib.load(artifact)
        n_features = int(getattr(estimator, "n_features_in_", -1))
        if n_features != len(FEATURE_NAMES):
            raise MatchedFontBenchmarkError(
                f"estimator accepts {n_features} features, expected {len(FEATURE_NAMES)}"
            )
        record("model:load", True, f"loaded {artifact.name} with {n_features} inputs")
    except Exception as error:  # noqa: BLE001 - fail closed on any load error
        record("model:load", False, f"{type(error).__name__}: {error}")

    if not all(check.passed for check in checks):
        failures = "; ".join(
            f"{check.name}: {check.detail}" for check in checks if not check.passed
        )
        raise MatchedFontBenchmarkError(f"the frozen champion cannot be used: {failures}")
    return estimator, card, tuple(checks)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def _target_stats(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
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


def _metric_values(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        stats = _target_stats([])
        return {
            "n_rows": 0,
            "n_sources": 0,
            "n_fonts": 0,
            "mae": math.nan,
            "rmse": math.nan,
            "r2": math.nan,
            "r2_defined": False,
            "r2_suppressed": False,
            "r2_status": "undefined_empty",
            "mean_residual": math.nan,
            "sd_residual": math.nan,
            **{f"target_{key}": value for key, value in stats.items() if key != "n"},
        }
    actual = frame["ssim_target"].to_numpy(dtype=np.float64)
    predicted = frame["prediction"].to_numpy(dtype=np.float64)
    metrics = regression_metrics(actual, predicted)
    stats = _target_stats(actual)
    near_degenerate = stats["sd"] < NEAR_DEGENERATE_SD
    r2 = float(metrics["r2"]) if metrics["r2_defined"] and not near_degenerate else math.nan
    if near_degenerate:
        status = "suppressed_near_degenerate_target"
    elif not metrics["r2_defined"]:
        status = "undefined_constant_target"
    else:
        status = "reported"
    residual = actual - predicted
    return {
        "n_rows": int(len(frame)),
        "n_sources": int(frame["source_id"].nunique()),
        "n_fonts": int(frame["font_id"].nunique()),
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "r2": r2,
        "r2_defined": bool(metrics["r2_defined"]),
        "r2_suppressed": bool(near_degenerate),
        "r2_status": status,
        "mean_residual": float(np.mean(residual)),
        "sd_residual": float(np.std(residual, ddof=0)),
        **{f"target_{key}": value for key, value in stats.items() if key != "n"},
    }


def _scope_subset(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    family = frame["variant_family"]
    severity = frame["severity_level"]
    primary = frame["protocol_scope"] == "primary"
    if scope == "all_primary":
        return frame[primary]
    if scope == "original_blur_noise":
        return frame[primary & family.isin([FAMILY_BLUR, FAMILY_NOISE])]
    if scope in (FAMILY_BLUR, FAMILY_NOISE, "thin_lines", "arc_strokes"):
        return frame[primary & (family == scope)]
    if scope == "interference":
        return frame[primary & family.isin(["thin_lines", "arc_strokes"])]
    if scope == "elastic":
        return frame[primary & (family == "elastic")]
    if scope == "geometric_primary":
        return frame[primary & (family == "elastic")]
    if scope == "clean_reference":
        return frame[family == CLEAN_FAMILY]
    if scope == "diagnostic_0_5px":
        return frame[(family == "elastic") & (severity == 1)]
    raise MatchedFontBenchmarkError(f"unknown scope {scope!r}")


def _overall_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope in OVERALL_SCOPES:
        row = {
            "scope": scope,
            "protocol_role": "diagnostic" if scope in DIAGNOSTIC_SCOPES else "primary",
        }
        row.update(_metric_values(_scope_subset(frame, scope)))
        rows.append(row)
    return pd.DataFrame(rows)


def _font_metrics(frame: pd.DataFrame, font_meta: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for font_id, meta in font_meta.items():
        font_frame = frame[frame["font_id"] == font_id]
        for role, scope in (
            ("primary", "all_primary"),
            ("diagnostic", "clean_reference"),
            ("diagnostic", "diagnostic_0_5px"),
        ):
            row = {
                "font_id": font_id,
                "font_family": meta["family"],
                "font_group": meta["group"],
                "font_role": meta["role"],
                "stratum": meta["stratum"],
                "scope": scope,
                "protocol_role": role,
            }
            row.update(_metric_values(_scope_subset(font_frame, scope)))
            rows.append(row)
    return pd.DataFrame(rows)


def _family_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        subset = frame[frame["variant_family"] == family]
        row = {
            "variant_family": family,
            "scope": family,
            "protocol_role": "primary",
        }
        row.update(_metric_values(subset[subset["protocol_scope"] == "primary"]))
        rows.append(row)
    row = {"variant_family": "elastic", "scope": "diagnostic_0_5px", "protocol_role": "diagnostic"}
    row.update(_metric_values(_scope_subset(frame, "diagnostic_0_5px")))
    rows.append(row)
    row = {
        "variant_family": CLEAN_FAMILY,
        "scope": "clean_reference",
        "protocol_role": "diagnostic",
    }
    row.update(_metric_values(_scope_subset(frame, "clean_reference")))
    rows.append(row)
    return pd.DataFrame(rows)


def _severity_parameter(protocol: M8BExecutionProtocol, family: str, severity: int) -> float:
    corruptions = protocol.corruptions
    if family == CLEAN_FAMILY:
        return 0.0
    if family == FAMILY_BLUR:
        return float(corruptions["blur_radii_px"][severity - 1])
    if family == FAMILY_NOISE:
        return float(corruptions["noise_sigmas"][severity - 1])
    if family == "thin_lines":
        return float(corruptions["thin_line_stroke_counts"][severity - 1])
    if family == "arc_strokes":
        return float(corruptions["arc_stroke_counts"][severity - 1])
    if family == "elastic":
        return float(corruptions["elastic_severity_amplitudes_px"][severity - 1])
    raise MatchedFontBenchmarkError(f"unknown corruption family {family!r}")


def _severity_metrics(frame: pd.DataFrame, protocol: M8BExecutionProtocol) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    clean = _scope_subset(frame, "clean_reference")
    row = {
        "variant_family": CLEAN_FAMILY,
        "severity_level": 0,
        "severity_parameter": 0.0,
        "scope": "clean_reference",
        "protocol_role": "diagnostic",
    }
    row.update(_metric_values(clean))
    rows.append(row)
    for family in FAMILY_ORDER:
        for severity in severity_map(protocol)[family]:
            subset = frame[
                (frame["variant_family"] == family) & (frame["severity_level"] == severity)
            ]
            role = "diagnostic" if (subset["protocol_scope"] == "diagnostic").all() else "primary"
            row = {
                "variant_family": family,
                "severity_level": severity,
                "severity_parameter": _severity_parameter(protocol, family, severity),
                "scope": "diagnostic_0_5px" if role == "diagnostic" else family,
                "protocol_role": role,
            }
            row.update(_metric_values(subset))
            rows.append(row)
    return pd.DataFrame(rows)


def _group_metrics(frame: pd.DataFrame, font_meta: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group in GROUP_ORDER:
        members = [key for key, meta in font_meta.items() if meta["group"] == group]
        group_frame = frame[frame["font_id"].isin(members)]
        for role, scope in (
            ("primary", "all_primary"),
            ("diagnostic", "clean_reference"),
            ("diagnostic", "diagnostic_0_5px"),
        ):
            row = {
                "font_group": group,
                "n_fonts": len(members),
                "scope": scope,
                "protocol_role": role,
            }
            row.update(_metric_values(_scope_subset(group_frame, scope)))
            rows.append(row)
    return pd.DataFrame(rows)


def _source_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    primary = frame[frame["protocol_scope"] == "primary"]
    rows: list[dict[str, Any]] = []
    for source_id, group in primary.groupby("source_id", sort=False):
        metrics = regression_metrics(
            group["ssim_target"].to_numpy(dtype=np.float64),
            group["prediction"].to_numpy(dtype=np.float64),
        )
        first = group.iloc[0]
        rows.append(
            {
                "source_id": source_id,
                "case_id": first["case_id"],
                "case_index": int(first["case_index"]),
                "font_id": first["font_id"],
                "n_rows": int(len(group)),
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
                "r2": float(metrics["r2"]) if metrics["r2_defined"] else math.nan,
                "r2_defined": bool(metrics["r2_defined"]),
                "mean_residual": float(np.mean(group["residual"])),
                "sd_residual": float(np.std(group["residual"], ddof=0)),
                "target_mean": float(group["ssim_target"].mean()),
                "prediction_mean": float(group["prediction"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _target_distributions(
    frame: pd.DataFrame,
    protocol: M8BExecutionProtocol,
    font_meta: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    entries: list[dict[str, Any]] = []

    def add(
        scope: str,
        scope_type: str,
        subset: pd.DataFrame,
        *,
        family: str | None = None,
        severity: int | None = None,
        font_id: str | None = None,
        group: str | None = None,
        role: str = "primary",
    ) -> None:
        stats = _target_stats(subset["ssim_target"])
        finite_psnr = subset["psnr_target"].dropna().to_numpy(dtype=np.float64)
        near_degenerate = bool(stats["n"] > 0 and stats["sd"] < NEAR_DEGENERATE_SD)
        entries.append(
            {
                "scope": scope,
                "scope_type": scope_type,
                "variant_family": family,
                "severity_level": severity,
                "font_id": font_id,
                "font_group": group,
                "protocol_role": role,
                "n_rows": int(len(subset)),
                "n_sources": int(subset["source_id"].nunique()) if len(subset) else 0,
                "n_fonts": int(subset["font_id"].nunique()) if len(subset) else 0,
                "target_mean": stats["mean"],
                "target_sd": stats["sd"],
                "target_variance": stats["variance"],
                "target_min": stats["min"],
                "target_max": stats["max"],
                "target_q05": stats["q05"],
                "target_q25": stats["q25"],
                "target_q50": stats["q50"],
                "target_q75": stats["q75"],
                "target_q95": stats["q95"],
                "psnr_finite_mean": float(np.mean(finite_psnr)) if finite_psnr.size else math.nan,
                "psnr_infinite_rows": int(subset["psnr_infinite"].sum()),
                "near_degenerate": near_degenerate,
                "r2_status": (
                    "suppressed_near_degenerate_target" if near_degenerate else "reported"
                ),
            }
        )

    add("all_primary", "overall", _scope_subset(frame, "all_primary"))
    for family in FAMILY_ORDER:
        add(family, "family", _scope_subset(frame, family), family=family)
        for severity in severity_map(protocol)[family]:
            subset = frame[
                (frame["variant_family"] == family) & (frame["severity_level"] == severity)
            ]
            role = "diagnostic" if (subset["protocol_scope"] == "diagnostic").all() else "primary"
            label = "diagnostic_elastic_s1" if role == "diagnostic" else f"{family}_s{severity}"
            add(
                label,
                "severity",
                subset,
                family=family,
                severity=severity,
                role=role,
            )
    for font_id, meta in font_meta.items():
        add(
            f"font:{font_id}",
            "font",
            _scope_subset(frame[frame["font_id"] == font_id], "all_primary"),
            font_id=font_id,
            group=meta["group"],
        )
    for group in GROUP_ORDER:
        members = [key for key, meta in font_meta.items() if meta["group"] == group]
        add(
            f"group:{group}",
            "group",
            _scope_subset(frame[frame["font_id"].isin(members)], "all_primary"),
            group=group,
        )
    add("clean_reference", "diagnostic", _scope_subset(frame, "clean_reference"), role="diagnostic")
    add(
        "diagnostic_0_5px",
        "diagnostic",
        _scope_subset(frame, "diagnostic_0_5px"),
        family="elastic",
        severity=1,
        role="diagnostic",
    )
    return pd.DataFrame(entries)


def _percentile_interval(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan, math.nan
    return float(np.percentile(finite, BOOTSTRAP_LOW)), float(np.percentile(finite, BOOTSTRAP_HIGH))


def _bootstrap_scope(
    subset: pd.DataFrame,
    *,
    scheme: str,
    scope: str,
    group_column: str,
) -> list[dict[str, Any]]:
    if subset.empty:
        return [
            {
                "scheme": scheme,
                "scope": scope,
                "metric": metric,
                "estimate": math.nan,
                "ci_low": math.nan,
                "ci_high": math.nan,
                "n_groups": 0,
                "n_samples": 0,
                "n_bootstrap": N_BOOTSTRAP,
                "r2_defined_rate": math.nan,
                "note": "empty scope",
            }
            for metric in ("mae", "rmse", "r2")
        ]
    actual = subset["ssim_target"].to_numpy(dtype=np.float64)
    predicted = subset["prediction"].to_numpy(dtype=np.float64)
    groups = subset[group_column].to_numpy()
    unique = sorted(set(groups.tolist()))
    index_map = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    mae_values: list[float] = []
    rmse_values: list[float] = []
    r2_values: list[float] = []
    r2_defined: list[bool] = []
    for _ in range(N_BOOTSTRAP):
        chosen = rng.choice(len(unique), size=len(unique), replace=True)
        indices = np.concatenate([index_map[unique[index]] for index in chosen])
        sample_actual = actual[indices]
        sample_predicted = predicted[indices]
        metrics = regression_metrics(sample_actual, sample_predicted)
        sd = float(np.std(sample_actual, ddof=0))
        mae_values.append(float(metrics["mae"]))
        rmse_values.append(float(metrics["rmse"]))
        defined = bool(metrics["r2_defined"]) and sd >= NEAR_DEGENERATE_SD
        r2_defined.append(defined)
        r2_values.append(float(metrics["r2"]) if defined else math.nan)
    estimate = regression_metrics(actual, predicted)
    estimate_sd = float(np.std(actual, ddof=0))
    estimate_r2 = (
        float(estimate["r2"])
        if estimate["r2_defined"] and estimate_sd >= NEAR_DEGENERATE_SD
        else math.nan
    )
    rows: list[dict[str, Any]] = []
    for metric, estimates, estimate_value in (
        ("mae", np.asarray(mae_values), float(estimate["mae"])),
        ("rmse", np.asarray(rmse_values), float(estimate["rmse"])),
        ("r2", np.asarray(r2_values), estimate_r2),
    ):
        low, high = _percentile_interval(estimates)
        rows.append(
            {
                "scheme": scheme,
                "scope": scope,
                "metric": metric,
                "estimate": estimate_value,
                "ci_low": low,
                "ci_high": high,
                "n_groups": len(unique),
                "n_samples": int(len(subset)),
                "n_bootstrap": N_BOOTSTRAP,
                "r2_defined_rate": (float(np.mean(r2_defined)) if metric == "r2" else math.nan),
                "note": (
                    "descriptive percentile interval over observed groups; "
                    "not a significance test or a font-population estimate"
                ),
            }
        )
    return rows


def _uncertainty(frame: pd.DataFrame, font_meta: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source_specs: list[tuple[str, pd.DataFrame]] = [
        ("all_primary", _scope_subset(frame, "all_primary"))
    ]
    for family in FAMILY_ORDER:
        source_specs.append((family, _scope_subset(frame, family)))
    for font_id in font_meta:
        source_specs.append(
            (f"font:{font_id}", _scope_subset(frame[frame["font_id"] == font_id], "all_primary"))
        )
    for group in GROUP_ORDER:
        members = [key for key, meta in font_meta.items() if meta["group"] == group]
        source_specs.append(
            (
                f"group:{group}",
                _scope_subset(frame[frame["font_id"].isin(members)], "all_primary"),
            )
        )
    source_specs.append(("clean_reference", _scope_subset(frame, "clean_reference")))
    source_specs.append(("diagnostic_0_5px", _scope_subset(frame, "diagnostic_0_5px")))
    for scope, subset in source_specs:
        rows.extend(
            _bootstrap_scope(subset, scheme="source_grouped", scope=scope, group_column="source_id")
        )

    matched_specs: list[tuple[str, pd.DataFrame]] = [
        ("all_primary", _scope_subset(frame, "all_primary"))
    ]
    for group in GROUP_ORDER:
        members = [key for key, meta in font_meta.items() if meta["group"] == group]
        matched_specs.append(
            (
                f"group:{group}",
                _scope_subset(frame[frame["font_id"].isin(members)], "all_primary"),
            )
        )
    for font_id in font_meta:
        matched_specs.append(
            (f"font:{font_id}", _scope_subset(frame[frame["font_id"] == font_id], "all_primary"))
        )
    for scope, subset in matched_specs:
        rows.extend(
            _bootstrap_scope(subset, scheme="matched_case", scope=scope, group_column="case_id")
        )
    return pd.DataFrame(rows)


def _comparison(
    frame: pd.DataFrame,
    overall: pd.DataFrame,
    groups: pd.DataFrame,
) -> pd.DataFrame:
    baseline_path = BASELINE_METRICS_PATH
    if not baseline_path.is_file():
        raise MatchedFontBenchmarkError(f"missing recorded M5C baseline: {baseline_path}")
    observed_hash = sha256_file(baseline_path)
    if observed_hash != BASELINE_METRICS_SHA256:
        raise MatchedFontBenchmarkError(
            f"recorded M5C baseline hash {observed_hash} != {BASELINE_METRICS_SHA256}"
        )
    baseline = pd.read_csv(baseline_path)
    required = {
        "model",
        "split",
        "scope",
        "mae",
        "rmse",
        "r2",
        "r2_defined",
        "n_samples",
        "n_sources",
    }
    if not required.issubset(baseline.columns):
        raise MatchedFontBenchmarkError(
            f"recorded baseline is missing columns {sorted(required - set(baseline.columns))}"
        )
    rows: list[dict[str, Any]] = []
    for row in baseline.itertuples(index=False):
        if str(row.model) != BASELINE_MODEL or str(row.split) != "test":
            continue
        scope = str(row.scope)
        if scope not in BASELINE_SCOPE_MAP:
            continue
        rows.append(
            {
                "source": "recorded_m5c_familiar_baseline",
                "subset": scope,
                "matched_scope": BASELINE_SCOPE_MAP[scope],
                "protocol_role": (
                    "diagnostic" if scope in {"clean_reference", "diagnostic_0_5"} else "primary"
                ),
                "mae": float(row.mae),
                "rmse": float(row.rmse),
                "r2": float(row.r2) if bool(row.r2_defined) else math.nan,
                "r2_defined": bool(row.r2_defined),
                "n_rows": int(row.n_samples),
                "n_sources": int(row.n_sources),
                "note": (
                    "recorded familiar-font M5C test slice; read, never recomputed; "
                    "familiar fonts only"
                ),
            }
        )
    overall_index = overall.set_index("scope")
    for scope in (
        "all_primary",
        "original_blur_noise",
        "gaussian_blur",
        "gaussian_noise",
        "clean_reference",
        "diagnostic_0_5px",
    ):
        if scope not in overall_index.index:
            continue
        row = overall_index.loc[scope]
        rows.append(
            {
                "source": "new_matched_font_benchmark",
                "subset": scope,
                "matched_scope": scope,
                "protocol_role": str(row["protocol_role"]),
                "mae": float(row["mae"]),
                "rmse": float(row["rmse"]),
                "r2": float(row["r2"]) if bool(row["r2_defined"]) else math.nan,
                "r2_defined": bool(row["r2_defined"]),
                "n_rows": int(row["n_rows"]),
                "n_sources": int(row["n_sources"]),
                "note": (
                    "new ten-font matched benchmark; the primary rows aggregate familiar, "
                    "M7B held-out and new held-out fonts and are not a like-for-like "
                    "familiar-font comparison"
                ),
            }
        )
    group_index = groups.set_index(["font_group", "scope"])
    for group in GROUP_ORDER:
        key = (group, "all_primary")
        if key not in group_index.index:
            continue
        row = group_index.loc[key]
        rows.append(
            {
                "source": "new_matched_font_benchmark",
                "subset": f"group:{group}",
                "matched_scope": "all_primary",
                "protocol_role": "primary",
                "mae": float(row["mae"]),
                "rmse": float(row["rmse"]),
                "r2": float(row["r2"]) if bool(row["r2_defined"]) else math.nan,
                "r2_defined": bool(row["r2_defined"]),
                "n_rows": int(row["n_rows"]),
                "n_sources": int(row["n_sources"]),
                "note": (
                    "familiar-font group is the closest new analogue of the recorded baseline; "
                    "it is still a new matched-case dataset and not the M3 test slice"
                    if group == "familiar"
                    else "held-out font group; descriptive only"
                ),
            }
        )
    return pd.DataFrame(rows)


def _composition(
    bundle: DatasetBundle,
    protocol: M8BExecutionProtocol,
    case_manifest: CaseManifest,
    font_meta: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    manifest = bundle.manifest
    counts = protocol.counts
    rows: list[dict[str, Any]] = []

    def add(
        component: str,
        expected: Any,
        realised: Any,
        detail: str = "",
    ) -> None:
        if isinstance(expected, bool) or isinstance(realised, bool):
            matches = bool(expected) == bool(realised)
        elif isinstance(expected, (int, float)) and isinstance(realised, (int, float)):
            matches = math.isclose(float(expected), float(realised), abs_tol=0.0)
        else:
            matches = expected == realised
        rows.append(
            {
                "component": component,
                "expected": expected,
                "realised": realised,
                "matches": matches,
                "detail": detail,
            }
        )

    add("fonts", 10, int(manifest["font_id"].nunique()), "pinned roster")
    add("cases", 30, int(manifest["case_id"].nunique()), "committed matched cases")
    add("sources", int(counts["n_sources"]), len(bundle.sources), "one source per font-case pair")
    add("rows", int(counts["total_rows"]), len(manifest), "21 rows per source")
    add(
        "primary_rows",
        int(counts["primary_rows"]),
        int((manifest["protocol_scope"] == "primary").sum()),
        "diagnostic rows excluded",
    )
    add(
        "diagnostic_rows",
        int(counts["diagnostic_rows"]),
        int((manifest["protocol_scope"] == "diagnostic").sum()),
        "clean and 0.5 px elastic",
    )
    add(
        "clean_rows",
        int(counts["clean_rows"]),
        int((manifest["variant_family"] == CLEAN_FAMILY).sum()),
        "one per source",
    )
    for family, expected_key in (
        ("gaussian_blur", "blur_rows"),
        ("gaussian_noise", "noise_rows"),
        ("thin_lines", "thin_line_rows"),
        ("arc_strokes", "arc_stroke_rows"),
    ):
        add(
            f"rows:{family}",
            int(counts[expected_key]),
            int((manifest["variant_family"] == family).sum()),
            "four severities per source",
        )
    add(
        "rows:elastic_primary",
        int(counts["elastic_primary_rows"]),
        int(
            (
                (manifest["variant_family"] == "elastic")
                & (manifest["protocol_scope"] == "primary")
            ).sum()
        ),
        "severities 2-4",
    )
    add(
        "rows:elastic_diagnostic",
        int(counts["elastic_diagnostic_rows"]),
        int(
            (
                (manifest["variant_family"] == "elastic")
                & (manifest["protocol_scope"] == "diagnostic")
            ).sum()
        ),
        "severity 1 (0.5 px) only",
    )
    add("images", 6300, int(len(manifest)), "one image per row")
    add("feature_rows", 6300, len(bundle.features), "eight frozen features per row")
    add("source_cases_rows", 300, len(bundle.source_cases), "matched-case manifest")
    add("ground_truth_rows", 300, len(bundle.ground_truth), "literal text; Git-ignored")
    add(
        "duplicate_row_ids",
        0,
        int(manifest["row_id"].duplicated().sum()),
        "row identity must be unique",
    )
    add(
        "duplicate_sample_ids",
        0,
        int(manifest["sample_id"].duplicated().sum()),
        "sample identity must be unique",
    )
    add(
        "distinct_source_ids",
        300,
        int(manifest["source_id"].nunique()),
        "each source is repeated by its 21 variants",
    )
    source_sizes = manifest.groupby("source_id", sort=False).size()
    add(
        "min_rows_per_source",
        21,
        int(source_sizes.min()),
        "every source has exactly 21 rows",
    )
    add(
        "max_rows_per_source",
        21,
        int(source_sizes.max()),
        "every source has exactly 21 rows",
    )
    add(
        "missing_row_ids",
        0,
        int(manifest["row_id"].isna().sum()),
        "row identity is mandatory",
    )
    add(
        "missing_sample_ids",
        0,
        int(manifest["sample_id"].isna().sum()),
        "sample identity is mandatory",
    )
    add(
        "missing_image_paths",
        0,
        int(manifest["image_path"].isna().sum()),
        "image path is mandatory",
    )
    add(
        "missing_feature_values",
        0,
        int(bundle.features[list(FEATURE_NAMES)].isna().sum().sum()),
        "the frozen feature extractor must return eight finite values",
    )
    add(
        "nonfinite_ssim_targets",
        0,
        int((~np.isfinite(manifest["ssim_target"].to_numpy(dtype=np.float64))).sum()),
        "SSIM target must be finite",
    )
    add(
        "clean_psnr_infinite_rows",
        300,
        int(manifest["psnr_infinite"].sum()),
        "the clean reference has infinite PSNR; PSNR is diagnostic only",
    )
    for font_id in font_meta:
        subset = manifest[manifest["font_id"] == font_id]
        add(f"sources:{font_id}", 30, int(len(subset) // 21), font_id)
        add(f"rows:{font_id}", 630, int(len(subset)), font_id)
    for group in GROUP_ORDER:
        members = [key for key, meta in font_meta.items() if meta["group"] == group]
        subset = manifest[manifest["font_id"].isin(members)]
        add(f"fonts:{group}", len(members), int(len(members)), group)
        add(f"rows:{group}", len(members) * 630, int(len(subset)), group)
    for length in (4, 5, 6):
        add(
            f"text_length:{length}",
            True,
            bool((manifest["text_length"] == length).any()),
            "length is sampled by the pinned rule, not fixed per length",
        )
    add(
        "split",
        "eval",
        ", ".join(sorted(manifest["split"].unique())),
        "evaluation-only benchmark; no training split",
    )
    add(
        "distinct_image_pixel_hashes",
        int(len(manifest)),
        int(manifest["image_sha256"].nunique()),
        "image identity must be unique",
    )
    add(
        "distinct_reference_pixel_hashes",
        300,
        int(manifest["reference_sha256"].nunique()),
        "one clean reference per source",
    )
    return pd.DataFrame(rows)


def _input_manifest(
    protocol: M8BExecutionProtocol,
    lifecycle_path: Path,
    text_manifest_path: Path,
) -> pd.DataFrame:
    lifecycle = bl.load_config(lifecycle_path)
    rows: list[dict[str, Any]] = []

    def add(
        key: str,
        path: str,
        expected: str,
        role: str,
        kind: str,
    ) -> None:
        candidate = Path(path)
        exists = candidate.is_file()
        observed = sha256_file(candidate) if exists else ""
        rows.append(
            {
                "key": key,
                "path": path,
                "sha256": observed,
                "expected_sha256": expected,
                "matches": bool(exists and (not expected or observed == expected)),
                "role": role,
                "kind": kind,
            }
        )

    for pin in lifecycle.inputs:
        kind = "other"
        if pin.path.startswith("models/"):
            kind = "model"
        elif pin.path.startswith("configs/"):
            kind = "protocol"
        elif pin.path.startswith("src/"):
            kind = "code"
        elif pin.path.startswith("reports/"):
            kind = "evidence"
        elif pin.path.startswith("data/"):
            kind = "data"
        add(pin.key, pin.path, pin.sha256, pin.role, kind)
    for font in protocol.fonts:
        add(
            f"font:{font['key']}",
            str(font["path"]),
            str(font["sha256"]),
            f"pinned {font['family']} font file; evaluated only, never redistributed",
            "font",
        )
    add(
        "text_case_manifest",
        str(text_manifest_path),
        str(protocol.text_case_manifest["file_sha256"]),
        "committed matched text-case manifest",
        "protocol",
    )
    add(
        "lifecycle_protocol",
        str(lifecycle_path),
        "",
        "lifecycle protocol that authorises the output paths",
        "protocol",
    )
    add(
        "execution_module",
        "src/captcha_quality/matched_font_benchmark.py",
        "",
        "M8C execution implementation",
        "code",
    )
    add(
        "recorded_m5c_baseline",
        str(BASELINE_METRICS_PATH),
        BASELINE_METRICS_SHA256,
        "recorded familiar-font M5C test-slice metrics",
        "evidence",
    )
    return pd.DataFrame(rows)


def _isolation_checks(
    checks: Iterable[CheckResult],
    *,
    extra: Iterable[CheckResult] = (),
) -> pd.DataFrame:
    rows = [
        {"check": check.name, "passed": bool(check.passed), "detail": check.detail}
        for check in (*checks, *extra)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# report construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReportBundle:
    """The fourteen deterministic report artifacts plus their provenance."""

    predictions: pd.DataFrame
    overall: pd.DataFrame
    font: pd.DataFrame
    family: pd.DataFrame
    severity: pd.DataFrame
    group: pd.DataFrame
    source: pd.DataFrame
    targets: pd.DataFrame
    uncertainty: pd.DataFrame
    comparison: pd.DataFrame
    composition: pd.DataFrame
    isolation: pd.DataFrame
    inputs: pd.DataFrame
    report_text: str
    key_metrics: dict[str, Any]


def _merge_features(bundle: DatasetBundle) -> pd.DataFrame:
    merged = bundle.manifest.merge(
        bundle.features,
        on="sample_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(merged) != len(bundle.manifest):
        raise MatchedFontBenchmarkError("feature join changed the number of dataset rows")
    if merged[list(FEATURE_NAMES)].isna().any().any():
        raise MatchedFontBenchmarkError("feature join produced a missing feature value")
    return merged


def _model_info(protocol: M8BExecutionProtocol, card: Mapping[str, Any]) -> dict[str, Any]:
    model = protocol.raw["model"]
    return {
        "artifact_file": f"{model['model_dir']}/{model['model_stem']}.joblib",
        "artifact_sha256": str(model["artifact_sha256"]),
        "card_file": f"{model['model_dir']}/{model['model_stem']}_card.json",
        "card_sha256": str(model["card_sha256"]),
        "model_family": str(card.get("model_family")),
        "model_name": str(card.get("model_name")),
        "calibration": str(card.get("calibration")),
        "target": str(card.get("target")),
        "feature_set_version": str(card.get("feature_set_version")),
        "feature_names": list(card.get("feature_names", [])),
        "n_features": int(card.get("n_features", -1)),
        "sklearn_version": sklearn_version(),
        "model_card_sklearn_version": str(card.get("sklearn_version")),
        "package_version": package_version(),
        "selection": card.get("selection"),
    }


def _build_prediction_frame(
    bundle: DatasetBundle,
    estimator: Any,
    font_meta: Mapping[str, Mapping[str, Any]],
) -> tuple[pd.DataFrame, list[CheckResult]]:
    merged = _merge_features(bundle)
    raw = predict(estimator, merged)
    raw = np.asarray(raw, dtype=np.float64).reshape(-1)
    checks: list[CheckResult] = []
    checks.append(
        CheckResult(
            "inference:prediction_count",
            raw.size == len(merged) == 6300,
            f"{raw.size} predictions for {len(merged)} rows",
        )
    )
    checks.append(
        CheckResult(
            "inference:prediction_finite",
            bool(np.isfinite(raw).all()),
            f"{int((~np.isfinite(raw)).sum())} non-finite prediction(s)",
        )
    )
    checks.append(
        CheckResult(
            "inference:prediction_range",
            bool(((raw >= 0.0) & (raw <= 1.0)).all()),
            f"min={float(raw.min()):.6f}, max={float(raw.max()):.6f}",
        )
    )
    checks.append(
        CheckResult(
            "inference:identity_join",
            bool(
                merged["row_id"].is_unique
                and merged["sample_id"].is_unique
                and len(merged) == len(bundle.manifest)
            ),
            "predictions map one-to-one to unique dataset rows",
        )
    )
    checks.append(
        CheckResult(
            "inference:no_fit",
            True,
            "the execution path calls predict() only; no fit, partial_fit, calibration, "
            "selection, thresholding or refitting is performed",
        )
    )
    prediction = pd.DataFrame(
        {
            "row_id": merged["row_id"].astype(str),
            "sample_id": merged["sample_id"].astype(str),
            "source_id": merged["source_id"].astype(str),
            "case_id": merged["case_id"].astype(str),
            "case_index": merged["case_index"].astype(int),
            "font_id": merged["font_id"].astype(str),
            "font_group": merged["font_id"].map(
                {key: meta["group"] for key, meta in font_meta.items()}
            ),
            "variant_family": merged["variant_family"].astype(str),
            "severity_level": merged["severity_level"].astype(int),
            "amplitude_px": merged["amplitude_px"],
            "protocol_scope": merged["protocol_scope"].astype(str),
            "ssim_target": merged["ssim_target"].astype(float),
            "psnr_target": merged["psnr_target"],
            "psnr_infinite": merged["psnr_infinite"].astype(bool),
            "prediction": raw,
        }
    )
    prediction["residual"] = prediction["ssim_target"] - prediction["prediction"]
    prediction["abs_residual"] = prediction["residual"].abs()
    return prediction, checks


def _key_metrics(overall: pd.DataFrame, group: pd.DataFrame) -> dict[str, Any]:
    index = overall.set_index("scope")

    def extract(row: pd.Series) -> dict[str, Any]:
        return {
            "n_rows": int(row["n_rows"]),
            "n_sources": int(row["n_sources"]),
            "mae": float(row["mae"]),
            "rmse": float(row["rmse"]),
            "r2": (
                float(row["r2"])
                if bool(row["r2_defined"]) and not bool(row["r2_suppressed"])
                else None
            ),
            "r2_defined": bool(row["r2_defined"]),
            "r2_suppressed": bool(row["r2_suppressed"]),
            "r2_status": str(row["r2_status"]),
        }

    result: dict[str, Any] = {
        "overall_primary": extract(index.loc["all_primary"]),
        "clean_reference": extract(index.loc["clean_reference"]),
        "diagnostic_0_5px": extract(index.loc["diagnostic_0_5px"]),
    }
    group_index = group.set_index(["font_group", "scope"])
    for name in GROUP_ORDER:
        result[f"{name}_primary"] = extract(group_index.loc[(name, "all_primary")])
    return result


def _build_report_bundle(
    bundle: DatasetBundle,
    protocol: M8BExecutionProtocol,
    case_manifest: CaseManifest,
    estimator: Any,
    card: Mapping[str, Any],
    model_checks: Iterable[CheckResult],
    dataset_checks: Iterable[CheckResult],
    inputs: pd.DataFrame,
) -> ReportBundle:
    font_meta = _font_meta(protocol)
    predictions, inference_checks = _build_prediction_frame(bundle, estimator, font_meta)
    artifact = Path(str(protocol.raw["model"]["model_dir"])) / (
        f"{protocol.raw['model']['model_stem']}.joblib"
    )
    inference_checks.append(
        CheckResult(
            "inference:artifact_unchanged",
            sha256_file(artifact) == str(protocol.raw["model"]["artifact_sha256"]),
            f"artifact sha256={sha256_file(artifact)} after prediction",
        )
    )
    overall = _overall_metrics(predictions)
    font = _font_metrics(predictions, font_meta)
    family = _family_metrics(predictions)
    severity = _severity_metrics(predictions, protocol)
    group = _group_metrics(predictions, font_meta)
    source = _source_metrics(predictions)
    targets = _target_distributions(predictions, protocol, font_meta)
    uncertainty = _uncertainty(predictions, font_meta)
    comparison = _comparison(predictions, overall, group)
    composition = _composition(bundle, protocol, case_manifest, font_meta)
    isolation = _isolation_checks(
        dataset_checks,
        extra=(*model_checks, *inference_checks),
    )
    key_metrics = _key_metrics(overall, group)
    report_text = _report_text(
        protocol,
        card,
        bundle,
        predictions=predictions,
        overall=overall,
        font=font,
        family=family,
        severity=severity,
        group=group,
        targets=targets,
        uncertainty=uncertainty,
        comparison=comparison,
        composition=composition,
        inputs=inputs,
    )
    return ReportBundle(
        predictions=predictions,
        overall=overall,
        font=font,
        family=family,
        severity=severity,
        group=group,
        source=source,
        targets=targets,
        uncertainty=uncertainty,
        comparison=comparison,
        composition=composition,
        isolation=isolation,
        inputs=inputs,
        report_text=report_text,
        key_metrics=key_metrics,
    )


# ---------------------------------------------------------------------------
# readable report
# ---------------------------------------------------------------------------


def _format_value(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (np.bool_,)):
        return "yes" if bool(value) else "no"
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return "n/a"
        if math.isinf(number):
            return "inf"
        return f"{number:.{digits}f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _md_table(frame: pd.DataFrame, columns: Sequence[str], digits: int = 6) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame[list(columns)].itertuples(index=False):
        lines.append("| " + " | ".join(_format_value(value, digits) for value in row) + " |")
    return "\n".join(lines)


def _report_text(
    protocol: M8BExecutionProtocol,
    card: Mapping[str, Any],
    bundle: DatasetBundle,
    *,
    predictions: pd.DataFrame,
    overall: pd.DataFrame,
    font: pd.DataFrame,
    family: pd.DataFrame,
    severity: pd.DataFrame,
    group: pd.DataFrame,
    targets: pd.DataFrame,
    uncertainty: pd.DataFrame,
    comparison: pd.DataFrame,
    composition: pd.DataFrame,
    inputs: pd.DataFrame,
) -> str:
    overall_index = overall.set_index("scope")
    primary = overall_index.loc["all_primary"]
    font_primary = font[font["protocol_role"] == "primary"].copy()
    font_primary = font_primary.sort_values("mae")
    group_primary = group[group["protocol_role"] == "primary"].copy()
    family_primary = family[family["protocol_role"] == "primary"].copy()
    severity_primary = severity[severity["protocol_role"] == "primary"].copy()
    uncertainty_overall = uncertainty[
        (uncertainty["scope"] == "all_primary")
        & (uncertainty["metric"].isin(["mae", "rmse", "r2"]))
    ]
    baseline_rows = comparison[comparison["source"] == "recorded_m5c_familiar_baseline"]
    baseline_overall = baseline_rows[baseline_rows["subset"] == "m3_test_overall"]

    new_held_out = font_primary[font_primary["font_group"] == "new_held_out"]
    m7b_held_out = font_primary[font_primary["font_group"] == "m7b_held_out"]

    findings: list[str] = []
    if not baseline_overall.empty:
        baseline_mae = float(baseline_overall.iloc[0]["mae"])
        direction = "higher" if float(primary["mae"]) > baseline_mae else "lower"
        findings.append(
            f"The ten-font primary MAE is {_format_value(primary['mae'])} against a "
            f"recorded familiar-font M5C test-slice MAE of {_format_value(baseline_mae)}; "
            f"the new value is {direction}. The two are not a like-for-like comparison: "
            "the recorded slice contains familiar fonts only."
        )
    if bool(primary["r2_defined"]) and float(primary["r2"]) < 0.0:
        findings.append(
            f"The primary R-squared is {_format_value(primary['r2'])}, i.e. negative: the "
            "frozen champion is worse than predicting the slice mean on this benchmark. "
            "This is reported as a negative result and is not tuned away."
        )
    if not font_primary.empty:
        best = font_primary.iloc[0]
        worst = font_primary.iloc[-1]
        findings.append(
            f"Font-level MAE ranges from {_format_value(best['mae'])} "
            f"({best['font_id']}) to {_format_value(worst['mae'])} "
            f"({worst['font_id']}). These are descriptive differences over the observed "
            "matched cases, not a font-population effect."
        )
    negative_families = family_primary[family_primary["r2"].notna() & (family_primary["r2"] < 0.0)]
    if not negative_families.empty:
        names = ", ".join(
            f"{row.variant_family} ({_format_value(row.r2)})"
            for row in negative_families.itertuples(index=False)
        )
        findings.append(
            f"R-squared is negative for {names}: on those families the frozen champion is "
            "worse than predicting the slice mean. This mixed result is reported as observed "
            "and is not tuned away."
        )
    if not m7b_held_out.empty and not new_held_out.empty:
        findings.append(
            "Georgia, Monaco and Impact were already evaluated in M7B; these results are "
            "not an independent replication of the M7B three-font finding. Verdana, "
            "Trebuchet MS, Chalkduster and Luminari are genuinely new to project evaluation."
        )
    findings_text = "\n\n".join(f"- {item}" for item in findings)

    manifest_sha = sha256_file(bundle.root / "manifest.csv")
    text = f"""# Matched cross-font generalisation benchmark (M8C)

## Scope and question

This report evaluates the exact frozen M5C champion zero-shot on a matched
cross-font benchmark. The target is **SSIM between each corrupted synthetic
image and its source-specific clean reference**. PSNR is recorded as a
diagnostic only. Nothing in this report measures human readability, CAPTCHA
solvability, CAPTCHA security or production robustness.

The central generalisation limitation is the number of **font identities**, not
the number of text cases: seven held-out identities are observed over thirty
matched cases. Three hundred sources are not three hundred independent fonts.

## Frozen identity and protocol chain

| Item | Value |
| --- | --- |
| Milestone | M8C (execution of the M8B frozen protocol) |
| Execution protocol | `{protocol.path}` |
| Protocol SHA-256 | `{protocol.file_sha256}` |
| Generation configuration hash | `{protocol.identity["generation_config_hash"]}` |
| Text-case manifest | `{protocol.text_case_manifest["path"]}` |
| Text-case file SHA-256 | `{protocol.text_case_manifest["file_sha256"]}` |
| Canonical text-case manifest hash | `{protocol.text_case_manifest["manifest_hash"]}` |
| Champion | `{card.get("model_family")}` / calibration `{card.get("calibration")}` |
| Champion artifact SHA-256 | `{protocol.raw["model"]["artifact_sha256"]}` |
| Champion card SHA-256 | `{protocol.raw["model"]["card_sha256"]}` |
| Feature set | `{card.get("feature_set_version")}`: {", ".join(card.get("feature_names", []))} |
| Target | `{card.get("target")}` |
| Dataset version | `{DATASET_VERSION}` |
| Dataset manifest SHA-256 | `{manifest_sha}` |

The champion was loaded with `joblib` and used unchanged. No fit, partial fit,
calibration, selection, thresholding or refitting was performed. No feature was
added, removed, reordered or recomputed with a different definition.

## Dataset composition

{_md_table(composition, ["component", "expected", "realised", "matches"])}

The dataset contains exactly {len(predictions)} rows, {int((predictions["protocol_scope"] == "primary").sum())} primary rows and {int((predictions["protocol_scope"] == "diagnostic").sum())} diagnostic rows. Clean-reference rows and the 0.5 px elastic rows are diagnostic only and are excluded from every primary metric and from the primary bootstrap.

## Metric definitions and aggregation

- MAE is the mean absolute error, `mean(|y - y_hat|)`.
- RMSE is the population root mean squared error, `sqrt(mean((y - y_hat)^2))`.
- R-squared is `1 - SS_res / SS_tot`, where `SS_tot` is taken around the target
  mean. It is undefined for a constant target and is suppressed when the target
  standard deviation is below {NEAR_DEGENERATE_SD}. Suppressed and undefined
  slices remain in the tables with an explicit status.
- Primary scopes are the M7D/M8B primary scopes: all primary rows, original
  blur and noise, each blur and noise family, interference, thin lines, arc
  strokes and geometric primary. Clean-reference and 0.5 px elastic rows are
  diagnostic and are excluded from every primary metric, primary endpoint and
  primary bootstrap.
- Overall metrics aggregate all ten fonts and all 300 sources for the named
  scope. Font metrics restrict the same primary rows to one font; group,
  family and severity metrics apply the corresponding subset. Per-source
  metrics use the 19 primary rows of one source.
- Uncertainty uses 2,000 percentile-bootstrap replicates. The source-grouped
  scheme resamples whole source groups; the matched-case scheme resamples whole
  cases and keeps all ten font renders of a case together. Neither interval is
  a significance test or a font-population estimate.

## Primary results

{_md_table(overall[overall["protocol_role"] == "primary"], ["scope", "n_rows", "n_sources", "mae", "rmse", "r2", "r2_status"])}

By font:

{_md_table(font_primary, ["font_id", "font_family", "font_group", "n_rows", "mae", "rmse", "r2", "r2_status"])}

By font group:

{_md_table(group_primary, ["font_group", "n_fonts", "n_rows", "n_sources", "mae", "rmse", "r2", "r2_status"])}

By corruption family:

{_md_table(family_primary, ["variant_family", "n_rows", "n_sources", "mae", "rmse", "r2", "r2_status"])}

By corruption severity (primary slices only):

{_md_table(severity_primary, ["variant_family", "severity_level", "severity_parameter", "n_rows", "mae", "rmse", "r2", "r2_status"])}

## Clean-image and diagnostic results

These rows are reported separately and are never merged into a primary metric.

{_md_table(overall[overall["protocol_role"] == "diagnostic"], ["scope", "n_rows", "n_sources", "mae", "rmse", "r2", "r2_status"])}

The clean reference has a constant target of 1.0, so its target standard
deviation is zero and R-squared is undefined; MAE and RMSE are still reported.
The 0.5 px elastic row is diagnostic because the M7B/M7D agreement excludes
that near-identity condition from primary scoring.

## Uncertainty

Source-grouped and matched-case percentile bootstrap intervals, 2,000
replicates, resampling whole source groups or whole matched cases. The intervals
are descriptive uncertainty summaries over the observed sources and font
identities. They are not significance tests and do not estimate a font
population.

{_md_table(uncertainty_overall, ["scheme", "scope", "metric", "estimate", "ci_low", "ci_high", "n_groups", "n_samples", "r2_defined_rate"])}

## Recorded familiar-font baseline

The recorded M5C familiar-font test-slice metrics are shown beside the new
results exactly as read from `{BASELINE_METRICS_PATH}`. They were not recomputed,
refitted or selected. The new primary aggregate is over all ten fonts, so it is
not a like-for-like familiar-font comparison. The new `group:familiar` row is the
closest new analogue, but it is still a different matched-case dataset.

{_md_table(comparison, ["source", "subset", "matched_scope", "protocol_role", "mae", "rmse", "r2", "n_rows", "n_sources"])}

## Findings

{findings_text}

## Interpretation limits

- The target is synthetic-reference SSIM. PSNR is diagnostic only.
- No claim is made about human readability, CAPTCHA solvability, security or
  production robustness.
- Seven held-out font identities, not 300 cases, are the central
  generalisation limitation. Font-level differences are descriptive for the
  observed matched cases.
- Georgia, Monaco and Impact were already evaluated in M7B and are not an
  independent new-font replication. Verdana, Trebuchet MS, Chalkduster and
  Luminari are genuinely new held-out identities for this project.
- The ten faces are pinned Apple-bundled macOS builds. The result does not
  establish cross-platform generalisation.
- No post-hoc threshold, endpoint change, model selection or new champion was
  introduced after seeing these results.
- R-squared is suppressed where the target standard deviation is below
  {NEAR_DEGENERATE_SD}; those slices are shown with an explicit status rather
  than silently dropped.

## Provenance and environment

The report was generated by the M8C execution module. The full machine-readable
environment, input hashes, prediction counts and reproducibility comparison are
recorded in `run_metadata.json`.

| Item | Value |
| --- | --- |
| Python | `{platform.python_version()}` |
| Platform | `{platform.platform()}` |
| scikit-learn | `{sklearn_version()}` |
| numpy | `{_package_version("numpy")}` |
| pandas | `{_package_version("pandas")}` |
| Pillow | `{_package_version("pillow")}` |
| Code revision at execution | recorded in `run_metadata.json` |
| Inputs verified | {int(inputs["matches"].sum())} / {len(inputs)} |

## Output membership

The dataset has exactly six top-level files plus `images/` with 6,300 PNGs in
six subdirectories. The report directory has exactly fifteen files. No font
binary is present in the repository, dataset or report.

## Reproducibility

The prescribed check was performed by repeating the complete generation,
inference and report pipeline in an isolated temporary directory and comparing
identities, image bytes, tables and report text. The outcome, including any
metadata-only difference, is recorded in `run_metadata.json`.
"""
    return text


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


# ---------------------------------------------------------------------------
# execution metadata, writing and verification
# ---------------------------------------------------------------------------


def _environment() -> dict[str, Any]:
    packages = {}
    for name in (
        "numpy",
        "pandas",
        "pillow",
        "scipy",
        "scikit-image",
        "opencv-python-headless",
        "opencv-python",
        "scikit-learn",
        "joblib",
        "fonttools",
    ):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
    }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _dataset_file_hashes(root: Path) -> dict[str, str]:
    return {name: sha256_file(root / name) for name in DATASET_FILES}


def _report_file_hashes(root: Path) -> dict[str, str]:
    return {name: sha256_file(root / name) for name in REPORT_FILES if name != "run_metadata.json"}


def build_run_metadata(
    protocol: M8BExecutionProtocol,
    lifecycle_path: Path,
    bundle: DatasetBundle,
    report: ReportBundle,
    report_dir: Path,
    model_info: Mapping[str, Any],
    inputs: pd.DataFrame,
    *,
    runtime: Mapping[str, Any],
    reproducibility: Mapping[str, Any] | None,
) -> dict[str, Any]:
    dataset_hashes = _dataset_file_hashes(bundle.root)
    report_hashes = _report_file_hashes(report_dir)
    prediction_count = int(len(report.predictions))
    manifest = bundle.manifest
    return {
        "schema_version": RUN_METADATA_SCHEMA,
        "milestone": "M8C",
        "run": {
            "name": "matched-font-benchmark",
            "protocol_version": protocol.version,
            "protocol_path": str(protocol.path),
            "protocol_sha256": protocol.file_sha256,
            "dataset_version": DATASET_VERSION,
            "seed": int(load_case_manifest(Path(str(protocol.text_case_manifest["path"]))).seed),
        },
        "lifecycle": {
            "pre_execution_protocol": str(lifecycle_path),
            "pre_execution_sha256": sha256_file(lifecycle_path),
            "status_before": "authorized_not_executed",
            "status_after": "executed",
            "post_execution_protocol": "configs/benchmark_lifecycle_v0_3_0.toml",
            "post_execution_sha256": (
                sha256_file(Path("configs/benchmark_lifecycle_v0_3_0.toml"))
                if Path("configs/benchmark_lifecycle_v0_3_0.toml").is_file()
                else None
            ),
        },
        "code": {
            "revision": code_revision(),
            "package_version": package_version(),
            "module": "src/captcha_quality/matched_font_benchmark.py",
            "module_sha256": sha256_file(Path("src/captcha_quality/matched_font_benchmark.py")),
        },
        "environment": _environment(),
        "inputs": inputs.to_dict(orient="records"),
        "dataset": {
            "path": DATASET_LOGICAL_PATH,
            "version": DATASET_VERSION,
            "n_fonts": int(manifest["font_id"].nunique()),
            "n_cases": int(manifest["case_id"].nunique()),
            "n_sources": int(manifest["source_id"].nunique()),
            "n_rows": int(len(manifest)),
            "n_primary_rows": int((manifest["protocol_scope"] == "primary").sum()),
            "n_diagnostic_rows": int((manifest["protocol_scope"] == "diagnostic").sum()),
            "n_images": int(len(manifest)),
            "manifest_sha256": dataset_hashes["manifest.csv"],
            "sources_sha256": dataset_hashes["sources.csv"],
            "source_cases_sha256": dataset_hashes["source_cases.csv"],
            "ground_truth_sha256": dataset_hashes["ground_truth.csv"],
            "features_sha256": dataset_hashes["features.csv"],
            "dataset_summary_sha256": dataset_hashes["dataset_summary.json"],
            "image_pixel_hash_set_sha256": bundle.summary["images"]["image_pixel_hash_set_sha256"],
            "reference_pixel_hash_set_sha256": bundle.summary["images"][
                "reference_pixel_hash_set_sha256"
            ],
        },
        "inference": {
            **dict(model_info),
            "prediction_count": prediction_count,
            "dataset_row_count": int(len(manifest)),
            "join_integrity": bool(
                prediction_count == len(manifest)
                and report.predictions["row_id"].is_unique
                and report.predictions["sample_id"].is_unique
            ),
            "invalid_predictions": 0,
            "model_fit_performed": False,
            "calibration_performed": False,
            "selection_performed": False,
            "thresholding_performed": False,
            "feature_change_performed": False,
            "target_change_performed": False,
            "note": (
                "the exact frozen M5C champion was loaded and predict() was called once "
                "per dataset row; no fitting, calibration, selection or thresholding occurred"
            ),
        },
        "metrics": report.key_metrics,
        "outputs": {
            "membership": list(REPORT_FILES),
            "hashes": report_hashes,
        },
        "reproducibility": reproducibility
        if reproducibility is not None
        else {"performed": False, "reason": "not requested"},
        "runtime": dict(runtime),
    }


def write_reports(report: ReportBundle, report_dir: Path) -> None:
    """Write the fourteen deterministic report files (run_metadata is last)."""
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(report.predictions, report_dir / "predictions.csv")
    _write_csv(report.overall, report_dir / "overall_metrics.csv")
    _write_csv(report.font, report_dir / "font_metrics.csv")
    _write_csv(report.family, report_dir / "family_metrics.csv")
    _write_csv(report.severity, report_dir / "severity_metrics.csv")
    _write_csv(report.group, report_dir / "group_metrics.csv")
    _write_csv(report.source, report_dir / "source_metrics.csv")
    _write_csv(report.targets, report_dir / "target_distributions.csv")
    _write_csv(report.uncertainty, report_dir / "uncertainty.csv")
    _write_csv(report.comparison, report_dir / "comparison.csv")
    _write_csv(report.composition, report_dir / "dataset_composition.csv")
    _write_csv(report.isolation, report_dir / "isolation_checks.csv")
    _write_csv(report.inputs, report_dir / "input_manifest.csv")
    _write_text(report.report_text, report_dir / "report.md")


def _report_membership(report_dir: Path, *, require_metadata: bool) -> str:
    if not report_dir.is_dir():
        raise MatchedFontBenchmarkError(f"report directory does not exist: {report_dir}")
    observed = {path.name for path in report_dir.iterdir() if path.is_file()}
    expected = set(REPORT_FILES) if require_metadata else set(REPORT_FILES) - {"run_metadata.json"}
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise MatchedFontBenchmarkError(
            f"report membership mismatch; missing={missing}, unexpected={extra}"
        )
    stray = [path.name for path in report_dir.iterdir() if not path.is_file()]
    if stray:
        raise MatchedFontBenchmarkError(f"unexpected directories in report output: {stray}")
    return f"exactly {len(expected)} declared report files, no more and no fewer"


def _frames_equal(left: pd.DataFrame, right: pd.DataFrame, tolerance: float = 1e-12) -> bool:
    if tuple(left.columns) != tuple(right.columns) or len(left) != len(right):
        return False
    for name in left.columns:
        a = left[name]
        b = right[name]
        if pd.api.types.is_bool_dtype(a) or pd.api.types.is_bool_dtype(b):
            if not np.array_equal(a.to_numpy(dtype=bool), b.to_numpy(dtype=bool)):
                return False
        elif pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            if not np.allclose(
                a.to_numpy(dtype=np.float64),
                b.to_numpy(dtype=np.float64),
                rtol=0.0,
                atol=tolerance,
                equal_nan=True,
            ):
                return False
        else:
            left_text = a.where(a.notna(), "").astype(str).to_numpy()
            right_text = b.where(b.notna(), "").astype(str).to_numpy()
            if not np.array_equal(left_text, right_text):
                return False
    return True


def verify_outputs(
    protocol: M8BExecutionProtocol,
    case_manifest: CaseManifest,
    dataset_dir: Path,
    report_dir: Path,
    *,
    lifecycle_path: Path = DEFAULT_LIFECYCLE,
    require_run_metadata: bool = True,
) -> VerificationReport:
    """Independently verify a written dataset and its report directory."""
    checks: list[CheckResult] = []
    dataset_dir = Path(dataset_dir)
    report_dir = Path(report_dir)
    bundle = load_dataset(Path(dataset_dir))
    dataset_report = dataset_checks(
        bundle, protocol, case_manifest, verify_images=True, verify_features=True
    )
    checks.extend(dataset_report.checks)

    estimator: Any = None
    card: dict[str, Any] = {}
    model_checks: tuple[CheckResult, ...] = ()

    def _load_champion() -> str:
        nonlocal estimator, card, model_checks
        estimator, card, model_checks = load_champion(protocol)
        return "the frozen champion loads with the pinned artifact and card hashes"

    _check(checks, "champion", _load_champion)
    if estimator is None:
        return VerificationReport(dataset=str(dataset_dir), checks=tuple(checks))

    inputs = _input_manifest(
        protocol, lifecycle_path, Path(str(protocol.text_case_manifest["path"]))
    )
    checks.append(
        CheckResult(
            "inputs:pinned",
            bool(inputs["matches"].all()),
            f"{int(inputs['matches'].sum())} / {len(inputs)} pinned inputs match",
        )
    )

    def _rebuild() -> tuple[ReportBundle, dict[str, pd.DataFrame]]:
        report = _build_report_bundle(
            bundle,
            protocol,
            case_manifest,
            estimator,
            card,
            model_checks,
            dataset_report.checks,
            inputs,
        )
        frames = {
            "predictions": report.predictions,
            "overall_metrics": report.overall,
            "font_metrics": report.font,
            "family_metrics": report.family,
            "severity_metrics": report.severity,
            "group_metrics": report.group,
            "source_metrics": report.source,
            "target_distributions": report.targets,
            "uncertainty": report.uncertainty,
            "comparison": report.comparison,
            "dataset_composition": report.composition,
            "isolation_checks": report.isolation,
            "input_manifest": report.inputs,
        }
        return report, frames

    rebuilt: dict[str, ReportBundle] = {}

    def _verify_tables() -> str:
        report, frames = _rebuild()
        rebuilt["report"] = report
        for name, frame in frames.items():
            written = _read_csv_exact(
                report_dir / f"{name}.csv", tuple(frame.columns), f"written {name}"
            )
            if not _frames_equal(frame, written):
                raise MatchedFontBenchmarkError(f"{name}.csv does not match the replay")
        if (report_dir / "report.md").read_text(encoding="utf-8") != report.report_text:
            raise MatchedFontBenchmarkError("report.md does not match the replay")
        return "all thirteen tables and report.md replay exactly"

    _check(checks, "tables_replay", _verify_tables)

    def _verify_metadata() -> str:
        path = report_dir / "run_metadata.json"
        if not path.is_file():
            if require_run_metadata:
                raise MatchedFontBenchmarkError("run_metadata.json is missing")
            return "run_metadata.json is not required at this stage"
        payload = json.loads(path.read_text(encoding="utf-8"))
        membership = list(payload.get("outputs", {}).get("membership", []))
        if membership != list(REPORT_FILES):
            raise MatchedFontBenchmarkError(
                "run_metadata.json does not declare the exact fifteen report files"
            )
        hashes = payload.get("outputs", {}).get("hashes", {})
        for name in REPORT_FILES:
            if name == "run_metadata.json":
                continue
            expected = hashes.get(name)
            if expected is None:
                raise MatchedFontBenchmarkError(f"run_metadata.json is missing the hash of {name}")
            observed = sha256_file(report_dir / name)
            if observed != expected:
                raise MatchedFontBenchmarkError(
                    f"{name}: observed sha256 {observed} != recorded {expected}"
                )
        dataset = payload.get("dataset", {})
        observed_manifest = sha256_file(dataset_dir / "manifest.csv")
        if dataset.get("manifest_sha256") != observed_manifest:
            raise MatchedFontBenchmarkError("run_metadata.json dataset manifest hash mismatch")
        return "run metadata hashes and membership are consistent"

    _check(checks, "run_metadata", _verify_metadata)
    _check(
        checks,
        "report_membership",
        lambda: _report_membership(report_dir, require_metadata=require_run_metadata),
    )

    predictions = _read_csv_exact(
        report_dir / "predictions.csv", PREDICTION_COLUMNS, "written predictions"
    )
    checks.append(
        CheckResult(
            "predictions:count",
            len(predictions) == len(bundle.manifest) == 6300,
            f"{len(predictions)} predictions",
        )
    )
    checks.append(
        CheckResult(
            "predictions:identities",
            set(predictions["row_id"]) == set(bundle.manifest["row_id"])
            and set(predictions["sample_id"]) == set(bundle.manifest["sample_id"]),
            "prediction identities join one-to-one to the dataset",
        )
    )
    checks.append(
        CheckResult(
            "predictions:finite",
            bool(np.isfinite(predictions["prediction"]).all()),
            "all predictions are finite",
        )
    )
    return VerificationReport(dataset=str(dataset_dir), checks=tuple(checks))


# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------


def _relative_image_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted((root / "images").rglob("*.png"))
    }


def compare_runs(
    left_dataset: Path,
    left_report: Path,
    right_dataset: Path,
    right_report: Path,
) -> dict[str, Any]:
    """Compare two complete runs, distinguishing metadata-only differences."""
    left_files = _dataset_file_hashes(left_dataset)
    right_files = _dataset_file_hashes(right_dataset)
    dataset_differences = [
        name for name in DATASET_FILES if left_files.get(name) != right_files.get(name)
    ]
    left_images = _relative_image_hashes(left_dataset)
    right_images = _relative_image_hashes(right_dataset)
    image_differences = [
        name
        for name in sorted(set(left_images) | set(right_images))
        if left_images.get(name) != right_images.get(name)
    ]
    left_reports = _report_file_hashes(left_report)
    right_reports = _report_file_hashes(right_report)
    report_differences = [
        name
        for name in REPORT_FILES
        if name != "run_metadata.json" and left_reports.get(name) != right_reports.get(name)
    ]

    def core(path: Path) -> Any:
        payload = json.loads((path / "run_metadata.json").read_text(encoding="utf-8"))
        payload.pop("runtime", None)
        payload.pop("reproducibility", None)
        return json_safe(payload)

    metadata_core_identical = core(left_report) == core(right_report)
    differences = (
        [f"dataset:{name}" for name in dataset_differences]
        + [f"image:{name}" for name in image_differences]
        + [f"report:{name}" for name in report_differences]
    )
    if not metadata_core_identical:
        differences.append("run_metadata:core")
    return {
        "performed": True,
        "second_run": "isolated temporary directory",
        "second_run_path_recorded": False,
        "dataset_files_identical": not dataset_differences,
        "dataset_differences": dataset_differences,
        "images_identical": not image_differences,
        "image_count_compared": len(left_images),
        "image_differences": image_differences[:20],
        "report_files_identical": not report_differences,
        "report_differences": report_differences,
        "metadata_core_identical": metadata_core_identical,
        "metadata_only_differences": [
            "run_metadata.runtime",
            "run_metadata.reproducibility",
        ],
        "identical": not differences,
        "differences": differences[:50],
        "note": (
            "the second run repeated generation, inference and report writing from the "
            "same pinned inputs and the same working tree; image bytes, dataset tables, "
            "report tables and report text were compared directly"
        ),
    }


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def _verify_metadata_hashes(dataset_dir: Path, report_dir: Path) -> str:
    payload = json.loads((report_dir / "run_metadata.json").read_text(encoding="utf-8"))
    hashes = payload.get("outputs", {}).get("hashes", {})
    for name in REPORT_FILES:
        if name == "run_metadata.json":
            continue
        observed = sha256_file(report_dir / name)
        if observed != hashes.get(name):
            raise MatchedFontBenchmarkError(
                f"run metadata hash mismatch for {name}: {observed} != {hashes.get(name)}"
            )
    if payload.get("dataset", {}).get("manifest_sha256") != sha256_file(
        dataset_dir / "manifest.csv"
    ):
        raise MatchedFontBenchmarkError("run metadata dataset manifest hash mismatch")
    return "run metadata hashes are consistent after the final rewrite"


def execute_once(
    protocol: M8BExecutionProtocol,
    case_manifest: CaseManifest,
    dataset_dir: Path,
    report_dir: Path,
    estimator: Any,
    card: Mapping[str, Any],
    model_checks: Iterable[CheckResult],
    inputs: pd.DataFrame,
    *,
    lifecycle_path: Path,
    isolated: bool,
) -> tuple[DatasetBundle, ReportBundle, dict[str, Any]]:
    """Generate, predict and write one complete run into isolated staging paths."""
    started_at = _utc_now()
    started = time.perf_counter()
    bundle = generate_dataset(protocol, case_manifest, dataset_dir)
    dataset_report = dataset_checks(
        bundle, protocol, case_manifest, verify_images=True, verify_features=True
    )
    _require_report(dataset_report, "generated dataset verification")
    report = _build_report_bundle(
        bundle,
        protocol,
        case_manifest,
        estimator,
        card,
        model_checks,
        dataset_report.checks,
        inputs,
    )
    write_reports(report, report_dir)
    runtime = {
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "isolated_staging": bool(isolated),
    }
    model_info = _model_info(protocol, card)
    metadata = build_run_metadata(
        protocol,
        lifecycle_path,
        bundle,
        report,
        report_dir,
        model_info,
        inputs,
        runtime=runtime,
        reproducibility=None,
    )
    _write_json(metadata, report_dir / "run_metadata.json")
    verification = verify_outputs(
        protocol,
        case_manifest,
        dataset_dir,
        report_dir,
        lifecycle_path=lifecycle_path,
        require_run_metadata=True,
    )
    _require_report(verification, "staged output verification")
    return bundle, report, metadata


def run_pipeline(
    protocol: M8BExecutionProtocol,
    case_manifest: CaseManifest,
    dataset_output: Path,
    report_output: Path,
    *,
    lifecycle_path: Path = DEFAULT_LIFECYCLE,
    reproduce: bool = True,
    isolated: bool = False,
) -> dict[str, Any]:
    """Run the complete benchmark in staging, verify it, then promote it."""
    dataset_output = Path(dataset_output)
    report_output = Path(report_output)
    if dataset_output.exists():
        raise MatchedFontBenchmarkError(
            f"refusing to overwrite the authorised dataset path: {dataset_output}"
        )
    if report_output.exists():
        raise MatchedFontBenchmarkError(
            f"refusing to overwrite the authorised report path: {report_output}"
        )

    estimator, card, model_checks = load_champion(protocol)
    inputs = _input_manifest(
        protocol, lifecycle_path, Path(str(protocol.text_case_manifest["path"]))
    )
    if not bool(inputs["matches"].all()):
        failures = inputs[~inputs["matches"]]["key"].tolist()
        raise MatchedFontBenchmarkError(f"pinned inputs do not match: {failures}")

    with tempfile.TemporaryDirectory(prefix="m8c-stage-") as stage:
        stage_root = Path(stage)
        auth_dataset = stage_root / "dataset"
        auth_report = stage_root / "reports"
        bundle, report, metadata = execute_once(
            protocol,
            case_manifest,
            auth_dataset,
            auth_report,
            estimator,
            card,
            model_checks,
            inputs,
            lifecycle_path=lifecycle_path,
            isolated=isolated,
        )
        reproducibility: dict[str, Any] = {
            "performed": False,
            "reason": "reproduction disabled by the caller",
        }
        if reproduce:
            with tempfile.TemporaryDirectory(prefix="m8c-repro-") as repro_stage:
                repro_root = Path(repro_stage)
                repro_dataset = repro_root / "dataset"
                repro_report = repro_root / "reports"
                execute_once(
                    protocol,
                    case_manifest,
                    repro_dataset,
                    repro_report,
                    estimator,
                    card,
                    model_checks,
                    inputs,
                    lifecycle_path=lifecycle_path,
                    isolated=True,
                )
                reproducibility = compare_runs(
                    auth_dataset, auth_report, repro_dataset, repro_report
                )
        metadata = build_run_metadata(
            protocol,
            lifecycle_path,
            bundle,
            report,
            auth_report,
            _model_info(protocol, card),
            inputs,
            runtime=metadata["runtime"],
            reproducibility=reproducibility,
        )
        _write_json(metadata, auth_report / "run_metadata.json")
        _verify_metadata_hashes(auth_dataset, auth_report)
        _report_membership(auth_report, require_metadata=True)

        dataset_output.parent.mkdir(parents=True, exist_ok=True)
        report_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(auth_dataset), str(dataset_output))
        shutil.move(str(auth_report), str(report_output))

    return {
        "dataset": str(dataset_output),
        "report": str(report_output),
        "counts": {
            "fonts": 10,
            "cases": 30,
            "sources": 300,
            "rows": int(len(report.predictions)),
            "primary_rows": int((report.predictions["protocol_scope"] == "primary").sum()),
            "diagnostic_rows": int((report.predictions["protocol_scope"] == "diagnostic").sum()),
            "images": 6300,
            "reports": len(REPORT_FILES),
        },
        "metrics": report.key_metrics,
        "reproducibility": reproducibility,
        "verification_checks": len(
            verify_outputs(
                protocol,
                case_manifest,
                dataset_output,
                report_output,
                lifecycle_path=lifecycle_path,
                require_run_metadata=True,
            ).checks
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _print_verification(report: VerificationReport) -> None:
    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")
    print(f"verification: {'PASS' if report.ok else 'FAIL'}")


def _run_command(args: argparse.Namespace) -> int:
    protocol = load_execution_protocol(args.execution_protocol)
    case_manifest = load_case_manifest(
        args.manifest or Path(str(protocol.text_case_manifest["path"]))
    )
    lifecycle = bl.load_config(args.lifecycle)
    authorised = lifecycle.authorised_execution
    dataset_output = Path(args.dataset_output or authorised.dataset_output)
    report_output = Path(args.report_output or authorised.report_output)
    repo_root = _repo_root().resolve()

    if args.isolated:
        for path in (dataset_output, report_output):
            resolved = path.resolve()
            if resolved == repo_root or repo_root in resolved.parents:
                raise MatchedFontBenchmarkError(
                    f"isolated outputs must live outside the repository: {resolved}"
                )
    else:
        from captcha_quality.benchmark_preflight import verify_preflight

        preflight = verify_preflight(
            args.execution_protocol,
            lifecycle=args.lifecycle,
            manifest=args.manifest,
        )
        _print_verification(preflight)
        if not preflight.ok:
            return 1
        if lifecycle.run.status != "authorized_not_executed":
            raise MatchedFontBenchmarkError(
                f"the lifecycle status is {lifecycle.run.status!r}, expected "
                "'authorized_not_executed' before the first authoritative run"
            )
        if str(authorised.protocol_sha256) != str(protocol.file_sha256):
            raise MatchedFontBenchmarkError(
                "the execution protocol does not match the lifecycle authorisation"
            )

    result = run_pipeline(
        protocol,
        case_manifest,
        dataset_output,
        report_output,
        lifecycle_path=args.lifecycle,
        reproduce=not args.no_reproduce,
        isolated=args.isolated,
    )
    print(f"dataset: {result['dataset']}")
    print(f"report: {result['report']}")
    print("counts: " + ", ".join(f"{key}={value}" for key, value in result["counts"].items()))
    overall = result["metrics"]["overall_primary"]
    print(
        "primary: "
        f"mae={overall['mae']:.6f} rmse={overall['rmse']:.6f} "
        f"r2={_format_value(overall['r2'])} status={overall['r2_status']}"
    )
    reproducibility = result["reproducibility"]
    print(
        "reproducibility: "
        f"performed={reproducibility.get('performed')} "
        f"identical={reproducibility.get('identical')}"
    )
    print(f"final verification checks: {result['verification_checks']}")
    return 0


def _verify_command(args: argparse.Namespace) -> int:
    protocol = load_execution_protocol(args.execution_protocol)
    case_manifest = load_case_manifest(
        args.manifest or Path(str(protocol.text_case_manifest["path"]))
    )
    lifecycle = bl.load_config(args.lifecycle)
    dataset_dir = Path(args.dataset or lifecycle.authorised_execution.dataset_output)
    report_dir = Path(args.report or lifecycle.authorised_execution.report_output)
    report = verify_outputs(
        protocol,
        case_manifest,
        dataset_dir,
        report_dir,
        lifecycle_path=args.lifecycle,
        require_run_metadata=True,
    )
    checks = list(report.checks)
    if args.lifecycle_post is not None:
        post = bl.verify_lifecycle(args.lifecycle_post, state="post-execution")
        checks.extend(post.checks)
    combined = VerificationReport(dataset=str(dataset_dir), checks=tuple(checks))
    _print_verification(combined)
    return 0 if combined.ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="captcha-matched-font-benchmark",
        description=(
            "Execute and verify the frozen M8C matched cross-font generalisation benchmark."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="generate, predict and write the benchmark")
    run_parser.add_argument("--execution-protocol", type=Path, default=DEFAULT_PROTOCOL)
    run_parser.add_argument("--lifecycle", type=Path, default=DEFAULT_LIFECYCLE)
    run_parser.add_argument("--manifest", type=Path, default=None)
    run_parser.add_argument("--dataset-output", type=Path, default=None)
    run_parser.add_argument("--report-output", type=Path, default=None)
    run_parser.add_argument(
        "--isolated",
        action="store_true",
        help="stage a reproduction run outside the repository; skips the lifecycle precheck",
    )
    run_parser.add_argument(
        "--no-reproduce",
        action="store_true",
        help="skip the second isolated reproducibility run (debugging only)",
    )
    run_parser.set_defaults(handler=_run_command)

    verify_parser = subparsers.add_parser(
        "verify", help="independently replay and verify a written benchmark"
    )
    verify_parser.add_argument("--execution-protocol", type=Path, default=DEFAULT_PROTOCOL)
    verify_parser.add_argument("--lifecycle", type=Path, default=DEFAULT_LIFECYCLE)
    verify_parser.add_argument("--manifest", type=Path, default=None)
    verify_parser.add_argument("--dataset", type=Path, default=None)
    verify_parser.add_argument("--report", type=Path, default=None)
    verify_parser.add_argument(
        "--lifecycle-post",
        type=Path,
        default=None,
        help="optional post-execution lifecycle protocol to verify as well",
    )
    verify_parser.set_defaults(handler=_verify_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except MatchedFontBenchmarkError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except Exception as error:  # noqa: BLE001 - fail closed with a non-zero exit
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


__all__ = [
    "DATASET_FILES",
    "DATASET_VERSION",
    "REPORT_FILES",
    "DatasetBundle",
    "MatchedFontBenchmarkError",
    "ReportBundle",
    "build_parser",
    "build_run_metadata",
    "compare_runs",
    "dataset_checks",
    "execute_once",
    "generate_dataset",
    "load_champion",
    "load_dataset",
    "main",
    "run_pipeline",
    "verify_outputs",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
