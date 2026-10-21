"""Tests for the M7D evaluation-design and dataset-expansion plan."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("sklearn")

from captcha_quality import evaluation_plan as ep
from captcha_quality.provenance import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "evaluation_design_plan_v0_1_0.toml"
M7B_DATA_ROOT = REPO_ROOT / "data" / "processed" / "heldout_font_benchmark_v0.1.0"

if not M7B_DATA_ROOT.exists():  # pragma: no cover - depends on generated artifacts
    pytest.skip(
        "frozen M7B inputs are not present; generate the upstream artifacts first",
        allow_module_level=True,
    )

_CONFIG = ep.load_config(CONFIG_PATH)
if not all(Path(font.path).is_file() for font in _CONFIG.fonts):  # pragma: no cover
    pytest.skip(
        "planned font files are not available on this machine",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def bundle():
    """Load the frozen inputs and build the deterministic plan once."""
    config = ep.load_config(CONFIG_PATH)
    inputs = ep.load_inputs(config)
    result = ep.build_plan(inputs)
    checks = ep.build_isolation_checks(inputs, result, result.report)
    return config, inputs, result, checks


@pytest.fixture()
def output_dir():
    """Use a Git-ignored scratch directory under reports/."""
    path = REPO_ROOT / "reports" / "_m7d_test_output"
    if path.exists():
        shutil.rmtree(path)
    try:
        yield path
    finally:
        if path.exists():
            shutil.rmtree(path)


def _evidence(result: ep.PlanResult, evidence_id: str) -> pd.Series:
    return result.evidence_snapshot.loc[
        result.evidence_snapshot["evidence_id"] == evidence_id
    ].iloc[0]


def test_config_counts_and_corruption_ladder(bundle):
    config, _, _, _ = bundle
    counts = ep._planned_counts(config)
    assert counts == {
        "fonts": 10,
        "familiar_fonts": 3,
        "held_out_fonts": 7,
        "text_cases": 30,
        "sources": 300,
        "rows": 6300,
        "primary_rows": 5700,
        "diagnostic_rows": 600,
    }
    assert config.plan.chosen_direction == "A"
    assert config.plan.status == "planned_not_executed"
    assert config.plan.n_bootstrap == 2000
    assert config.plan.near_degenerate_sd == 0.005
    ladder = {(item.family, item.role): item.levels for item in config.corruptions}
    assert ladder[("gaussian_blur", "primary")] == (1.0, 2.0, 3.5, 5.0)
    assert ladder[("gaussian_noise", "primary")] == (6.0, 12.0, 20.0, 32.0)
    assert ladder[("thin_lines", "primary")] == (1.0, 2.0, 4.0, 7.0)
    assert ladder[("arc_strokes", "primary")] == (1.0, 2.0, 4.0, 7.0)
    assert ladder[("elastic", "primary")] == (1.0, 2.0, 3.0)
    assert ladder[("elastic", "diagnostic")] == (0.5,)
    assert ladder[("clean", "diagnostic")] == (0.0,)


def test_font_inventory_and_glyph_coverage(bundle):
    _, inputs, result, _ = bundle
    fonts = result.font_inventory
    assert len(fonts) == 10
    assert set(fonts["font_id"]) == set(ep.PLANNED_FAMILIAR_FONTS) | set(ep.PLANNED_HELDOUT_FONTS)
    assert int((fonts["role"] == "familiar").sum()) == 3
    assert int((fonts["role"] == "held-out").sum()) == 7
    assert bool(fonts["available"].all())
    assert bool(fonts["hash_verified"].all())
    assert bool(fonts["glyph_coverage_ok"].all())
    assert set(fonts["license_status"]) == {"review_required_before_generation"}
    assert set(fonts["license_claim"]) == {"not_asserted"}
    assert set(fonts["planned_sources"]) == {30}
    assert set(fonts["planned_primary_rows"]) == {570}
    assert set(fonts["planned_diagnostic_rows"]) == {60}
    for _, row in fonts.iterrows():
        assert sha256_file(Path(row["path"])) == row["sha256"]
        assert row["missing_glyphs"] == ""
    assert len(inputs.font_inventory) == 10


def test_font_roster_reuse_from_m7b_is_explicit(bundle):
    _, _, result, _ = bundle
    fonts = result.font_inventory.set_index("font_id")
    for key in ("georgia", "monaco", "impact"):
        assert fonts.loc[key, "m7b_status"] == "held_out_in_m7b_evaluation_only"
    for key in ("verdana", "trebuchet_ms", "chalkduster", "luminari"):
        assert fonts.loc[key, "m7b_status"] == "new_held_out_face"
    for key in ("arial", "courier_new", "times_new_roman"):
        assert fonts.loc[key, "m7b_status"] == "familiar_m3_development_font"
    assert ep.PLANNED_NEW_HELDOUT_FONTS == (
        "verdana",
        "trebuchet_ms",
        "chalkduster",
        "luminari",
    )
    assert ep.M7B_REUSED_HELDOUT_FONTS == ("georgia", "monaco", "impact")
    design = result.protocol["dataset_design"]
    assert design["new_held_out_fonts"] == list(ep.PLANNED_NEW_HELDOUT_FONTS)
    assert design["held_out_fonts_reused_from_m7b"] == list(ep.M7B_REUSED_HELDOUT_FONTS)
    assert "not independent" in design["roster_increment_note"]
    assert any("already used in M7B" in limitation for limitation in result.protocol["limitations"])
    report = result.report
    assert "How much of the roster is actually new" in report
    assert "Verdana, Trebuchet MS, Chalkduster and Luminari" in report


def test_evidence_reproduction(bundle):
    _, _, result, _ = bundle
    evidence = result.evidence_snapshot
    assert len(evidence) == 46
    assert bool(evidence["reproduced"].all())
    assert evidence["source_sha256"].str.len().eq(64).all()
    assert set(evidence["family"]) == {"M3", "M5C", "M7A", "M7B", "M7C"}
    assert math.isclose(
        float(_evidence(result, "E11")["reproduced_value"]),
        0.0929520648953031,
        rel_tol=1e-12,
    )
    assert math.isclose(
        float(_evidence(result, "E13")["reproduced_value"]),
        0.0851977776011776,
        rel_tol=1e-12,
    )
    assert math.isclose(
        float(_evidence(result, "E33")["reproduced_value"]),
        0.072595467158,
        rel_tol=1e-10,
    )
    assert math.isclose(
        float(_evidence(result, "E34")["reproduced_value"]),
        -0.000167403803406,
        rel_tol=1e-10,
    )
    assert math.isclose(
        float(_evidence(result, "E31")["reproduced_value"]),
        0.0991411034039,
        rel_tol=1e-12,
    )
    assert math.isclose(
        float(_evidence(result, "E31")["published_value"]),
        float(_evidence(result, "E31")["reproduced_value"]),
        rel_tol=1e-12,
    )
    assert _evidence(result, "E41")["reproduced_value"] == "3/2/7"
    assert _evidence(result, "E42")["reproduced_value"] == "3/0/10"


def test_direction_decision(bundle):
    _, _, result, _ = bundle
    directions = result.direction_assessment.set_index("direction")
    assert set(directions.index) == {"A", "B", "C"}
    assert bool(directions.loc["A", "selected"]) is True
    assert bool(directions.loc["B", "selected"]) is False
    assert bool(directions.loc["C", "selected"]) is False
    assert directions.loc["A", "decision"] == "selected"
    assert directions.loc["B", "decision"] == "deferred"
    assert directions.loc["C", "decision"] == "matching_control_incorporated_into_A"
    assert "matched" in directions.loc["A", "rationale"].lower()
    assert "three held-out fonts" in directions.loc["C", "rationale"]
    assert "0.0726" in directions.loc["A", "rationale"]
    assert "-0.0002" in directions.loc["A", "rationale"]


def test_research_questions_and_independent_units(bundle):
    _, _, result, _ = bundle
    questions = result.research_questions.set_index("question_id")
    assert questions.loc["RQ1", "role"] == "primary"
    assert "held-out" in questions.loc["RQ1", "question"]
    assert "matched" in questions.loc["RQ1", "estimand"].lower()
    assert questions.loc["RQ6", "status"] == "not_answered"
    assert questions.loc["RQ7", "status"] == "not_authorized"
    units = set(result.independent_units["unit"])
    assert {
        "font_identity",
        "matched_text_case",
        "source",
        "clean_reference",
        "corrupted_variant",
        "renderer",
        "benchmark_row",
        "font_pair",
        "source_group",
    }.issubset(units)
    font_unit = result.independent_units.loc[
        result.independent_units["unit"] == "font_identity"
    ].iloc[0]
    assert "not a random draw" in font_unit["independence"]
    assert "population" in font_unit["not_valid_for"]
    variant = result.independent_units.loc[
        result.independent_units["unit"] == "corrupted_variant"
    ].iloc[0]
    assert "nested" in variant["independence"].lower()
    assert "confidence intervals" in variant["not_valid_for"].lower()


def test_split_hierarchy_and_leakage_controls(bundle):
    _, _, result, _ = bundle
    splits = result.split_hierarchy.set_index("level")
    assert splits.loc["benchmark_eval", "train_allowed"] is False or not bool(
        splits.loc["benchmark_eval", "train_allowed"]
    )
    assert "untouched eval" in splits.loc["benchmark_eval", "rule"]
    assert "never appear in training" in splits.loc["font_identity", "rule"]
    assert "all font renders of a case" in splits.loc["matched_text_case", "rule"]
    assert "variants" in splits.loc["source", "rule"].lower()
    assert "at least two held-out font identities" in splits.loc["future_final_test", "rule"]
    assert "renderer" in splits.loc["renderer", "rule"].lower()
    controls = result.leakage_controls
    assert len(controls) >= 12
    assert set(controls["control_id"]) == {f"L{i:02d}" for i in range(1, 13)}
    assert controls["enforcement"].str.len().gt(0).all()
    assert controls["failure_action"].str.len().gt(0).all()


def test_endpoint_coverage_and_policies(bundle):
    _, _, result, _ = bundle
    endpoints = result.endpoints.set_index("endpoint_id")
    assert endpoints.loc["P1", "role"] == "primary"
    assert endpoints.loc["P1", "metric"] == "rmse"
    assert endpoints.loc["P1", "scope"] == "all_primary"
    assert "matched_text_case" in endpoints.loc["P1", "unit"]
    assert "21 font pairs" in endpoints.loc["P1", "contrast"]
    assert "2,000" in endpoints.loc["P1", "uncertainty"]
    assert endpoints.loc["S4", "role"] == "diagnostic"
    assert "R² undefined" in endpoints.loc["S4", "degenerate_policy"]
    assert endpoints.loc["F1", "status"] == "not_authorized"
    assert "universal SSIM threshold" in endpoints.loc["F1", "success_interpretation"]
    assert not result.endpoints["endpoint_id"].duplicated().any()


def test_sample_size_scenarios(bundle):
    _, _, result, _ = bundle
    sample = result.sample_size_scenarios
    assert set(sample["level"]) == {
        "matched_text_case_mean_rmse",
        "matched_font_pair_difference",
        "font_mean_rmse",
    }
    source = sample.loc[sample["level"] == "matched_text_case_mean_rmse"]
    assert list(source["n_units"]) == [20, 30, 40, 60]
    assert int(source.loc[source["selected"], "n_units"].iloc[0]) == 30
    assert math.isclose(
        float(source.loc[source["selected"], "assumed_sd"].iloc[0]),
        0.022634,
        rel_tol=1e-4,
    )
    paired = sample.loc[sample["level"] == "matched_font_pair_difference"]
    assert set(paired.loc[paired["selected"], "n_units"]) == {30}
    assert set(paired.loc[paired["selected"], "assumed_sd"]) == {
        0.005,
        0.010,
        0.020,
    }
    font = sample.loc[sample["level"] == "font_mean_rmse"]
    assert set(font.loc[font["selected"], "n_units"]) == {10}
    assert set(font.loc[font["selected"], "assumed_sd"]) == {0.010, 0.020, 0.030}
    assert sample["assumed_sd_basis"].str.contains("not").any()
    assert not sample["scenario_id"].duplicated().any()


def test_success_and_stopping_rules(bundle):
    _, _, result, _ = bundle
    criteria = result.success_criteria.set_index("criterion_id")
    assert {"G01", "G06", "G09", "G10", "G12", "G14"}.issubset(criteria.index)
    assert "licensing" in criteria.loc["G03", "criterion"].lower()
    assert "byte-identical" in criteria.loc["G08", "criterion"]
    assert "universal SSIM threshold" in criteria.loc["G14", "rationale"]
    rules = result.stopping_rules.set_index("rule_id")
    assert "exactly 30 cases" in rules.loc["R03", "rule"]
    assert "do not collect more data" in rules.loc["R06", "rule"]
    assert "Stop any claim" in rules.loc["R07", "rule"]
    assert "predeclared practical margin" in rules.loc["R08", "rule"]


def test_claims_and_limits(bundle):
    _, _, result, _ = bundle
    claims = result.claims_and_limits.set_index("claim_id")
    assert len(claims) == 12
    assert claims.loc["M7D-C01", "status"] == "established"
    assert claims.loc["M7D-C06", "status"] == "not_supported"
    assert "font population" in claims.loc["M7D-C06", "claim"].lower()
    assert claims.loc["M7D-C07", "status"] == "not_supported"
    assert "unseen-renderer" in claims.loc["M7D-C07", "claim"].lower()
    assert claims.loc["M7D-C08", "status"] == "not_supported"
    assert "human readability" in claims.loc["M7D-C08", "claim"].lower()
    assert claims.loc["M7D-C10", "status"] == "not_supported"
    assert "equivalence" in claims.loc["M7D-C10", "claim"].lower()
    assert claims.loc["M7D-C12", "status"] == "established"


def test_report_sections_and_content(bundle):
    _, _, result, _ = bundle
    report = result.report
    assert all(section in report for section in ep.REPORT_SECTIONS)
    assert "EXP-0017" in report
    assert "Planned, not executed" in report
    assert "Direction A" in report
    assert "300 font-specific sources" in report
    assert "6,300" in report
    assert "5,700" in report
    assert "600" in report
    assert "0.092952" in report
    assert "0.085198" in report
    assert "0.076126" in report
    assert "0.113431" in report
    assert "0.072595" in report
    assert "-0.000167" in report
    assert "font population" in report.lower()
    assert "human readability" in report
    assert "No dataset was generated." in report
    assert "No image was rendered." in report
    assert "No model was trained" in report
    assert "planned, not completed" in report.lower()
    assert "universal SSIM threshold" in report


def test_isolation_checks(bundle):
    _, _, _, checks = bundle
    assert checks["check"].is_unique
    assert checks["passed"].all()
    assert set(checks["check"]).issuperset(
        {
            "input_pins",
            "font_inventory",
            "license_gate",
            "m3_counts",
            "m7b_counts",
            "model_identity",
            "m7c_counts",
            "chosen_direction",
            "planned_counts",
            "evidence_reproduced",
            "independent_unit_coverage",
            "endpoint_coverage",
            "sample_size_scenarios",
            "report_sections",
            "non_execution_statement",
            "claim_limits",
            "stopping_rules",
            "success_criteria",
            "no_planned_outputs_created",
        }
    )


def test_output_protection():
    with pytest.raises(ep.PlanError):
        ep._safe_output_path(REPO_ROOT / "configs" / "not_allowed")
    with pytest.raises(ep.PlanError):
        ep._safe_output_path(REPO_ROOT / "data" / "processed" / "not_allowed")
    with pytest.raises(ep.PlanError):
        ep._safe_output_path(REPO_ROOT / "models" / "not_allowed")
    with pytest.raises(ep.PlanError):
        ep._safe_output_path(REPO_ROOT / "reports" / "cross_font_analysis_v0_1_0" / "not_allowed")
    with pytest.raises(ep.PlanError):
        ep._safe_output_path(REPO_ROOT / "reports")
    allowed = ep._safe_output_path(REPO_ROOT / "reports" / "_m7d_test_output")
    assert allowed.name == "_m7d_test_output"


def test_run_and_verify_determinism(bundle, output_dir):
    config, _, _, _ = bundle
    first_dir, first_metadata = ep.run_plan(CONFIG_PATH, output_reports=output_dir, overwrite=True)
    assert first_dir == output_dir.resolve()
    first_hashes = ep._output_hashes(first_dir)
    assert "report.md" in first_hashes
    assert "plan_manifest.json" in first_hashes
    assert "run_metadata.json" not in first_hashes
    assert first_metadata["summary"]["all_evidence_reproduced"] is True
    assert first_metadata["summary"]["sources"] == 300
    assert first_metadata["summary"]["rows"] == 6300
    assert first_metadata["summary"]["primary_rows"] == 5700
    assert first_metadata["summary"]["diagnostic_rows"] == 600
    assert first_metadata["plan"]["chosen_direction"] == "A"
    assert first_metadata["plan"]["status"] == "planned_not_executed"
    assert first_metadata["planned_outputs_present_at_run"] == {
        "dataset": False,
        "benchmark_report": False,
    }
    verification = ep.verify_plan(CONFIG_PATH, output_reports=output_dir)
    assert verification.ok, verification.failures()
    second_dir, second_metadata = ep.run_plan(
        CONFIG_PATH, output_reports=output_dir, overwrite=True
    )
    assert second_dir == first_dir
    assert ep._output_hashes(second_dir) == first_hashes
    assert second_metadata["plan"]["config_hash"] == first_metadata["plan"]["config_hash"]
    assert ep.verify_plan(CONFIG_PATH, output_reports=output_dir).ok


def test_verify_detects_tampering(bundle, output_dir):
    ep.run_plan(CONFIG_PATH, output_reports=output_dir, overwrite=True)
    report_path = output_dir / "report.md"
    report_path.write_text(report_path.read_text(encoding="utf-8") + "\ntamper\n", encoding="utf-8")
    verification = ep.verify_plan(CONFIG_PATH, output_reports=output_dir)
    assert not verification.ok
    assert any(not check.passed for check in verification.checks)


def test_metadata_is_json_safe(bundle, output_dir):
    _, inputs, result, _ = bundle
    metadata = ep._build_metadata(
        inputs,
        result,
        output_hashes={"report.md": "a" * 64},
        frozen_assets={"configs/example.toml": "b" * 64},
        frozen_assets_hash="c" * 64,
        elapsed_seconds=1.0,
    )
    encoded = json.dumps(metadata, sort_keys=True, allow_nan=False)
    assert "NaN" not in encoded
    assert metadata["summary"]["evidence_rows"] == 46
    assert metadata["summary"]["all_evidence_reproduced"] is True
    assert metadata["summary"]["chosen_direction"] == "A"
    assert metadata["plan"]["config_sha256"] == inputs.config.config_sha256


def test_cli_run_and_verify(bundle, output_dir):
    _, _, _, _ = bundle
    assert (
        ep.main(
            [
                "run",
                "--config",
                str(CONFIG_PATH),
                "--output-reports",
                str(output_dir),
                "--overwrite",
            ]
        )
        == 0
    )
    assert (
        ep.main(
            [
                "verify",
                "--config",
                str(CONFIG_PATH),
                "--output-reports",
                str(output_dir),
            ]
        )
        == 0
    )
