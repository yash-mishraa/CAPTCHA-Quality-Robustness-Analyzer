"""Frozen-artifact contracts for the local research dashboard.

The dashboard is a read-only presentation layer.  Every path, column and
release hash in this module is pinned to the ``v1.0.0`` research artifacts.
Nothing here writes to the repository, invokes the frozen model, or reads the
untracked ground-truth table.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = REPO_ROOT / "data" / "processed" / "matched_font_benchmark_v0.1.0"
IMAGES_ROOT = DATA_ROOT / "images"
M8_REPORT_DIR = REPO_ROOT / "reports" / "matched_font_benchmark_v0_1_0"
M9_REPORT_DIR = REPO_ROOT / "reports" / "interference_error_analysis_v0_1_0"

# Release hashes pinned in docs/LIVE_DEMO_PLAN.md section 4.7.
CORE_HASHES = {
    "manifest.csv": "35ae405fd084070bf4d6c235638cbda6671573389e880dcca2a63d8b66049743",
    "features.csv": "fa2d79c85d682b26545dc2c4e3ec5ece2a41d85aea7e1ea1070f7256ff8519bb",
    "predictions.csv": "e6c1f1ba83bbb05c18fed8a9b0dd1c87014a6f6c3667475971b2834dfedd4dd4",
}

REPORT_HASHES = {
    ("m8", "overall_metrics.csv"): (
        "abbc28d5c73e15091edbf79cc086449d171c91af4233964529811634ed0a79fa"
    ),
    ("m9", "summary_metrics.csv"): (
        "6d68e465887e3053e20a7c582f79e8318900e8cd14fb48005f15a54c224f1303"
    ),
}

# UI allowlists.  Internal provenance columns are deliberately excluded so
# they cannot leak into a page by accident.
MANIFEST_COLUMNS = (
    "sample_id",
    "source_id",
    "case_id",
    "case_index",
    "font_id",
    "text_length",
    "variant_family",
    "severity_level",
    "amplitude_px",
    "image_path",
    "reference_path",
    "ssim_target",
    "height",
    "width",
    "image_sha256",
    "reference_sha256",
    "protocol_scope",
)

FEATURE_COLUMNS = (
    "intensity_mean",
    "intensity_std",
    "contrast_rms",
    "ink_fraction",
    "laplacian_variance",
    "edge_density",
    "histogram_entropy",
    "noise_sigma_immerkaer",
)

PREDICTION_COLUMNS = (
    "sample_id",
    "font_id",
    "font_group",
    "ssim_target",
    "prediction",
    "residual",
    "abs_residual",
)

CORE_STRING_COLUMNS = (
    "sample_id",
    "source_id",
    "case_id",
    "font_id",
    "font_group",
    "variant_family",
    "protocol_scope",
    "image_path",
    "reference_path",
    "image_sha256",
    "reference_sha256",
)

EXPECTED_ROWS = 6_300
EXPECTED_PRIMARY_ROWS = 5_700
EXPECTED_DIAGNOSTIC_ROWS = 600
EXPECTED_FONTS = 10
EXPECTED_CASES = 30
EXPECTED_SOURCES = 300
EXPECTED_ROWS_PER_SOURCE = 21
EXPECTED_HEIGHT = 64
EXPECTED_WIDTH = 192

PROTOCOL_SCOPES = ("primary", "diagnostic")
VARIANT_FAMILIES = (
    "clean",
    "gaussian_blur",
    "gaussian_noise",
    "thin_lines",
    "arc_strokes",
    "elastic",
)
FONT_IDS = (
    "arial",
    "courier_new",
    "times_new_roman",
    "georgia",
    "monaco",
    "impact",
    "verdana",
    "trebuchet_ms",
    "chalkduster",
    "luminari",
)

# Report tables the app may read.  This is an allowlist, not a blocklist.
# Each entry lists the columns the page logic depends on.
M8_TABLES = {
    "overall_metrics.csv": (
        "scope",
        "protocol_role",
        "n_rows",
        "n_sources",
        "n_fonts",
        "mae",
        "rmse",
        "r2",
        "r2_status",
    ),
    "family_metrics.csv": (
        "variant_family",
        "scope",
        "protocol_role",
        "n_rows",
        "n_sources",
        "mae",
        "rmse",
        "r2",
        "r2_status",
    ),
    "uncertainty.csv": (
        "scheme",
        "scope",
        "metric",
        "estimate",
        "ci_low",
        "ci_high",
        "n_groups",
        "n_samples",
        "note",
    ),
    "dataset_composition.csv": (
        "component",
        "expected",
        "realised",
        "matches",
        "detail",
    ),
}

M9_TABLES = {
    "summary_metrics.csv": (
        "scope",
        "protocol_role",
        "n_rows",
        "n_sources",
        "mae",
        "rmse",
        "r2",
        "large_error_proportion",
        "top_5pct_source_sse_share",
    ),
    "severity_metrics.csv": (
        "family",
        "severity_level",
        "scope",
        "n_rows",
        "n_sources",
        "target_mean",
        "prediction_mean",
        "mae",
        "large_error_proportion",
    ),
    "condition_calibration.csv": (
        "scope",
        "protocol_role",
        "n_rows",
        "n_sources",
        "prediction_target_sd_ratio",
        "calibration_slope",
        "within_source_prediction_target_sd_ratio_median",
        "within_source_calibration_slope_median",
    ),
    "error_concentration.csv": (
        "scope",
        "n_sources",
        "top_5pct_source_sse_share",
        "top_10pct_source_sse_share",
        "top_20pct_source_sse_share",
    ),
    "font_condition_metrics.csv": (
        "font_id",
        "font_group",
        "scope",
        "n_rows",
        "n_sources",
        "mae",
        "mae_ci_low",
        "mae_ci_high",
    ),
}

REPORT_TABLES = {"m8": M8_TABLES, "m9": M9_TABLES}
REPORT_DIRS = {"m8": M8_REPORT_DIR, "m9": M9_REPORT_DIR}

# These are never opened by the app.  The loader allowlist enforces the rule;
# the names are kept here as an explicit contract for readers and tests.
FORBIDDEN_FILES = (
    "ground_truth.csv",
    "sources.csv",
    "dataset_summary.json",
    "source_cases.csv",
    "run_metadata.json",
    "input_manifest.csv",
    "provenance.json",
)

DATA_ERROR_MESSAGE = (
    "The frozen benchmark files did not pass the read-only integrity check. "
    "No substitute data has been generated."
)

IMAGE_UNAVAILABLE_MESSAGE = "This record's image is unavailable."
IMAGE_HASH_MESSAGE = "This record failed its image integrity check."
NO_MATCH_MESSAGE = "No public row matches this selection."
DIAGNOSTIC_MESSAGE = "Diagnostic record: excluded from primary metrics."


class IntegrityError(RuntimeError):
    """A frozen artifact failed its read-only contract.

    ``str(error)`` is always the generic message shown to a visitor. The
    specific reason is available as ``error.detail`` for tests and local logs.
    """

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(DATA_ERROR_MESSAGE)


class ImageIntegrityError(IntegrityError):
    """A selected image failed its path, hash, or format check."""

    def __init__(self, detail: str, message: str = IMAGE_HASH_MESSAGE) -> None:
        self.detail = detail
        RuntimeError.__init__(self, message)
