"""Focused tests for the M8C matched cross-font benchmark execution.

The full benchmark renders 6,300 images and is exercised by the separately
commanded execution/verification run. These tests cover the deterministic
contracts that can fail cheaply: the frozen output membership, the primary and
diagnostic scope masks, the R-squared suppression rule, the report directory
contract, the pinned input manifest and the executed lifecycle status.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from captcha_quality import benchmark_lifecycle as bl
from captcha_quality import matched_font_benchmark as mfb
from captcha_quality.benchmark_preflight import load_case_manifest, load_execution_protocol

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPO_ROOT / "configs" / "cross_font_benchmark_protocol_v0_2_0.toml"
LIFECYCLE = REPO_ROOT / "configs" / "benchmark_lifecycle_v0_2_0.toml"
MANIFEST = REPO_ROOT / "configs" / "cross_font_text_cases_v0_1_0.toml"

pytestmark = pytest.mark.skipif(
    not PROTOCOL.is_file(), reason="the M8B execution protocol is not present"
)


@pytest.fixture(scope="module")
def protocol():
    return load_execution_protocol(PROTOCOL)


@pytest.fixture(scope="module")
def manifest():
    return load_case_manifest(MANIFEST)


def _frame() -> pd.DataFrame:
    """One primary row per family/severity plus the clean and 0.5 px rows."""
    rows = [
        {
            "source_id": "src-a",
            "font_id": "arial",
            "variant_family": "clean",
            "severity_level": 0,
            "protocol_scope": "diagnostic",
            "ssim_target": 1.0,
            "prediction": 0.99,
        }
    ]
    for family in ("gaussian_blur", "gaussian_noise", "thin_lines", "arc_strokes"):
        for severity in (1, 2, 3, 4):
            rows.append(
                {
                    "source_id": "src-a",
                    "font_id": "arial",
                    "variant_family": family,
                    "severity_level": severity,
                    "protocol_scope": "primary",
                    "ssim_target": 0.8 - severity * 0.05,
                    "prediction": 0.75 - severity * 0.05,
                }
            )
    for severity in (1, 2, 3, 4):
        rows.append(
            {
                "source_id": "src-a",
                "font_id": "arial",
                "variant_family": "elastic",
                "severity_level": severity,
                "protocol_scope": "primary" if severity > 1 else "diagnostic",
                "ssim_target": 0.9 - severity * 0.1,
                "prediction": 0.88 - severity * 0.1,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# frozen membership
# ---------------------------------------------------------------------------


def test_dataset_and_report_membership_are_frozen():
    assert mfb.DATASET_FILES == (
        "manifest.csv",
        "sources.csv",
        "source_cases.csv",
        "ground_truth.csv",
        "features.csv",
        "dataset_summary.json",
    )
    assert len(mfb.REPORT_FILES) == 15
    assert len(set(mfb.REPORT_FILES)) == 15
    assert "run_metadata.json" in mfb.REPORT_FILES
    assert "report.md" in mfb.REPORT_FILES


def test_report_membership_matches_the_authorising_lifecycle():
    lifecycle = bl.load_config(LIFECYCLE)
    declared = {item.name for item in lifecycle.authorised_report_files}
    assert declared == set(mfb.REPORT_FILES)


def test_dataset_membership_matches_the_authorising_lifecycle():
    lifecycle = bl.load_config(LIFECYCLE)
    declared = {item.name for item in lifecycle.authorised_dataset_files}
    assert declared == set(mfb.DATASET_FILES)


def test_image_subdirectories_cover_every_family():
    assert set(mfb.IMAGE_SUBDIRECTORIES) == {
        "clean",
        "gaussian_blur",
        "gaussian_noise",
        "thin_lines",
        "arc_strokes",
        "elastic",
    }


# ---------------------------------------------------------------------------
# scope masks and aggregation rules
# ---------------------------------------------------------------------------


def test_primary_masks_exclude_clean_and_diagnostic_elastic():
    frame = _frame()
    primary = mfb._scope_subset(frame, "all_primary")
    assert set(primary["protocol_scope"]) == {"primary"}
    assert len(primary) == 19
    assert "clean" not in set(primary["variant_family"])
    assert not ((primary["variant_family"] == "elastic") & (primary["severity_level"] == 1)).any()


def test_family_and_interference_masks_are_disjoint_and_complete():
    frame = _frame()
    blur = mfb._scope_subset(frame, "gaussian_blur")
    noise = mfb._scope_subset(frame, "gaussian_noise")
    interference = mfb._scope_subset(frame, "interference")
    geometric = mfb._scope_subset(frame, "geometric_primary")
    assert len(blur) == len(noise) == 4
    assert len(interference) == 8
    assert len(geometric) == 3
    combined = pd.concat([blur, noise, interference, geometric])
    assert len(combined) == 19


def test_original_blur_noise_scope_is_blur_plus_noise():
    frame = _frame()
    combined = mfb._scope_subset(frame, "original_blur_noise")
    assert set(combined["variant_family"]) == {"gaussian_blur", "gaussian_noise"}
    assert len(combined) == 8


def test_clean_and_diagnostic_scopes_are_separate():
    frame = _frame()
    clean = mfb._scope_subset(frame, "clean_reference")
    diagnostic = mfb._scope_subset(frame, "diagnostic_0_5px")
    assert len(clean) == 1 and bool((clean["variant_family"] == "clean").all())
    assert len(diagnostic) == 1
    assert int(diagnostic["severity_level"].iloc[0]) == 1


# ---------------------------------------------------------------------------
# metrics and R-squared suppression
# ---------------------------------------------------------------------------


def test_metric_values_report_r2_when_the_target_varies():
    frame = _frame()
    primary = mfb._scope_subset(frame, "all_primary")
    values = mfb._metric_values(primary)
    assert values["n_rows"] == 19
    assert values["n_sources"] == 1
    assert values["r2_status"] == "reported"
    assert np.isfinite(values["r2"])
    assert values["mae"] >= 0.0


def test_metric_values_suppress_r2_for_a_constant_target():
    frame = _frame()
    clean = mfb._scope_subset(frame, "clean_reference")
    values = mfb._metric_values(clean)
    assert values["r2_status"] == "suppressed_near_degenerate_target"
    assert values["r2_suppressed"] is True
    assert np.isnan(values["r2"])
    assert np.isfinite(values["mae"])


def test_metric_values_suppress_r2_below_the_frozen_threshold():
    frame = _frame().head(2).copy()
    frame["ssim_target"] = [0.9000, 0.9001]
    frame["prediction"] = [0.8999, 0.9002]
    values = mfb._metric_values(frame)
    assert values["r2_suppressed"] is True
    assert np.isnan(values["r2"])


def test_target_stats_are_population_statistics():
    stats = mfb._target_stats([0.0, 1.0])
    assert stats["n"] == 2
    assert stats["mean"] == pytest.approx(0.5)
    assert stats["sd"] == pytest.approx(0.5)
    assert stats["variance"] == pytest.approx(0.25)


def test_severity_parameters_match_the_frozen_ladders(protocol):
    assert mfb._severity_parameter(protocol, "gaussian_blur", 1) == pytest.approx(
        protocol.corruptions["blur_radii_px"][0]
    )
    assert mfb._severity_parameter(protocol, "gaussian_noise", 4) == pytest.approx(
        protocol.corruptions["noise_sigmas"][3]
    )
    assert mfb._severity_parameter(protocol, "elastic", 1) == pytest.approx(
        protocol.corruptions["elastic_severity_amplitudes_px"][0]
    )


# ---------------------------------------------------------------------------
# report directory contract
# ---------------------------------------------------------------------------


def test_report_membership_accepts_exactly_the_declared_files(tmp_path):
    for name in mfb.REPORT_FILES:
        (tmp_path / name).write_text("x\n", encoding="utf-8")
    assert "15" in mfb._report_membership(tmp_path, require_metadata=True)
    (tmp_path / "run_metadata.json").unlink()
    assert "14" in mfb._report_membership(tmp_path, require_metadata=False)


def test_report_membership_rejects_a_missing_file(tmp_path):
    for name in mfb.REPORT_FILES:
        (tmp_path / name).write_text("x\n", encoding="utf-8")
    (tmp_path / "predictions.csv").unlink()
    with pytest.raises(mfb.MatchedFontBenchmarkError, match="missing"):
        mfb._report_membership(tmp_path, require_metadata=True)


def test_report_membership_rejects_an_unexpected_file(tmp_path):
    for name in mfb.REPORT_FILES:
        (tmp_path / name).write_text("x\n", encoding="utf-8")
    (tmp_path / "extra.csv").write_text("x\n", encoding="utf-8")
    with pytest.raises(mfb.MatchedFontBenchmarkError, match="unexpected"):
        mfb._report_membership(tmp_path, require_metadata=True)


def test_frames_equal_normalises_missing_object_values():
    left = pd.DataFrame({"a": [1, 2], "b": ["x", None]})
    right = pd.DataFrame({"a": [1, 2], "b": ["x", np.nan]})
    assert mfb._frames_equal(left, right)
    right.loc[1, "b"] = "y"
    assert not mfb._frames_equal(left, right)


# ---------------------------------------------------------------------------
# pinned inputs and lifecycle
# ---------------------------------------------------------------------------


def test_font_roster_and_groups_are_the_frozen_ten(protocol):
    meta = mfb._font_meta(protocol)
    assert len(meta) == 10
    assert {key: value["group"] for key, value in meta.items()} == {
        "arial": "familiar",
        "courier_new": "familiar",
        "times_new_roman": "familiar",
        "georgia": "m7b_held_out",
        "monaco": "m7b_held_out",
        "impact": "m7b_held_out",
        "verdana": "new_held_out",
        "trebuchet_ms": "new_held_out",
        "chalkduster": "new_held_out",
        "luminari": "new_held_out",
    }


def test_input_manifest_matches_every_pinned_input(protocol):
    inputs = mfb._input_manifest(protocol, LIFECYCLE, MANIFEST)
    assert bool(inputs["matches"].all()), inputs[~inputs["matches"]].to_dict("records")
    keys = set(inputs["key"])
    for expected in (
        "font:arial",
        "font:luminari",
        "text_case_manifest",
        "lifecycle_protocol",
        "execution_module",
        "recorded_m5c_baseline",
    ):
        assert expected in keys
    assert len(inputs) == len(bl.load_config(LIFECYCLE).inputs) + 10 + 4


def test_executed_lifecycle_status_is_accepted(tmp_path):
    text = LIFECYCLE.read_text(encoding="utf-8")
    text = text.replace('version = "v0.2.0"', 'version = "v0.3.0"', 1)
    text = text.replace('milestone = "M8A"', 'milestone = "M8C"', 1)
    text = text.replace('status = "authorized_not_executed"', 'status = "executed"', 1)
    text = text.replace('supersedes = "v0.1.0"', 'supersedes = "v0.2.0"', 1)
    path = tmp_path / "lifecycle_v0_3_0.toml"
    path.write_text(text, encoding="utf-8")
    config = bl.load_config(path)
    assert config.run.status == "executed"
    assert config.run.milestone == "M8C"
    assert config.run.version == "v0.3.0"
    assert config.run.supersedes == "v0.2.0"
    # The declared input contract is unchanged by the transition.
    base = bl.load_config(LIFECYCLE)
    assert config.inputs == base.inputs
    assert config.authorised_execution == base.authorised_execution


def test_executed_lifecycle_verifies_post_execution():
    path = REPO_ROOT / "configs" / "benchmark_lifecycle_v0_3_0.toml"
    report = bl.verify_lifecycle(path, state="post-execution")
    assert report.ok, [f"{c.name}: {c.detail}" for c in report.failures()]
    mapped = {check.name: check for check in report.checks}
    assert mapped["dataset_membership"].passed
    assert mapped["report_membership"].passed
    assert mapped["dataset_matches_protocol"].passed
    assert "transition" in mapped["m7d_verifier"].detail


def test_protocol_and_text_manifest_are_readable_toml():
    for path in (PROTOCOL, MANIFEST):
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
