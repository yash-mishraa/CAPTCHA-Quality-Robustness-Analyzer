"""Focused tests for the M9C research-prioritization and closure memo."""

from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "reports" / "research_prioritization_v0_1_0"

REQUIRED_FILES = {
    "README.md",
    "research_prioritization.md",
    "hypothesis_comparison.csv",
    "project_completion_checklist.md",
    "verification_checks.csv",
}

HYPOTHESIS_COLUMNS = [
    "hypothesis_id",
    "short_name",
    "source_confidence",
    "scientific_importance",
    "falsifiability",
    "explanatory_discrimination",
    "data_compute_requirements",
    "new_cases_or_fonts",
    "leakage_confounding_risk",
    "expected_information_gain",
    "project_relevance",
    "negative_result_value",
    "key_unresolved_assumptions",
    "implementation_complexity",
    "recommendation",
    "rationale",
]

VERIFICATION_COLUMNS = ["check_id", "category", "expected", "actual", "passed", "details"]

TRACKED_HISTORICAL_HASHES = {
    "reports/matched_font_benchmark_v0_1_0/predictions.csv": (
        "e6c1f1ba83bbb05c18fed8a9b0dd1c87014a6f6c3667475971b2834dfedd4dd4"
    ),
    "reports/matched_font_benchmark_v0_1_0/report.md": (
        "ef266df7d0e3022b7eb0b6ccf08bb97782a8d2741e4f980f51facf94be7ad85f"
    ),
    "reports/interference_error_analysis_v0_1_0/README.md": (
        "22de39a32f23f501789f1a771d2c0c4f70eaa0f527988810ff1313f91de5a7c8"
    ),
    "reports/interference_error_analysis_v0_1_0/hypotheses_and_followups.md": (
        "36bef0136f72f4e072f2da0d218883723750d7315969c5791037ee5771cbab24"
    ),
    "reports/interference_experiment_protocol_v0_1_0/experiment_protocol.md": (
        "5b53e3c70171530fda38787906600dbb441c72a9e62713dc2aaa6e9492dbbcc4"
    ),
    "reports/interference_experiment_protocol_v0_1_0/statistical_analysis_plan.json": (
        "048fcee530746055c9f6b00c62f6e325996975e3e7e8ce47d39c8dbc589df481"
    ),
}

LOCAL_MODEL_HASHES = {
    "models/retention_aware_selection_v0_1_0/selected_policy.joblib": (
        "c9b1625de3bb503f22fa14f9b6468b028e609b6c7febe7c307dafd0e3e1156e0"
    ),
    "models/retention_aware_selection_v0_1_0/selected_policy_card.json": (
        "9a75725ba1ccf82ab42f009aa744a102cce86a6a9bb1d1aa86431362213507f8"
    ),
}


def _package_text(name: str) -> str:
    return (PACKAGE / name).read_text(encoding="utf-8")


def _package_csv(name: str) -> list[dict[str, str]]:
    with (PACKAGE / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_package_membership_is_exact() -> None:
    assert PACKAGE.is_dir()
    actual = {path.name for path in PACKAGE.iterdir() if path.is_file()}
    assert actual == REQUIRED_FILES
    assert not any(path.is_dir() for path in PACKAGE.iterdir())


def test_hypothesis_comparison_schema_and_dispositions() -> None:
    rows = _package_csv("hypothesis_comparison.csv")
    assert rows
    assert list(rows[0]) == HYPOTHESIS_COLUMNS
    assert [row["hypothesis_id"] for row in rows] == ["H1", "H2", "H3"]
    assert [row["short_name"] for row in rows] == [
        "attenuated_response",
        "edge_sharpness_conflation",
        "font_glyph_difficulty",
    ]
    for row in rows:
        assert all(row[column].strip() for column in HYPOTHESIS_COLUMNS)
    assert "not authorized" in rows[0]["recommendation"].lower()
    assert "deprioritize" in rows[1]["recommendation"].lower()
    assert "deprioritize" in rows[2]["recommendation"].lower()
    assert "new fonts required" in rows[0]["new_cases_or_fonts"]
    assert "new thin-stroke" in rows[2]["new_cases_or_fonts"]


def test_decision_is_path_a_without_execution_authority() -> None:
    readme = _package_text("README.md")
    memo = _package_text("research_prioritization.md")
    combined = f"{readme}\n{memo}".lower()

    assert "path a" in combined
    assert "stop experimentation" in combined
    assert "does **not** authorize" in readme
    assert "does not authorize" in memo
    assert "no part of this recommendation authorizes execution" in combined
    assert "m9b" in combined and "not executed" in combined
    assert "not_tested" in combined
    assert "path b" in combined and "path c" in combined
    assert "h1" in combined and "h2" in combined and "h3" in combined
    assert "best candidate" in combined


def test_completion_checklist_has_required_sections_and_ids() -> None:
    checklist = _package_text("project_completion_checklist.md")
    lower = checklist.lower()

    assert "definition of done" in lower
    assert "mandatory closure requirements" in lower
    assert "optional future research" in lower
    assert "reopening rules" in lower
    assert "what would trigger reopening" in lower
    assert "what would not trigger reopening" in lower
    assert "better model score" in lower
    assert "not authorized" in lower
    for identifier in [f"M-{number:02d}" for number in range(1, 11)]:
        assert identifier in checklist
    for identifier in [f"O-{number:02d}" for number in range(1, 8)]:
        assert identifier in checklist


def test_verification_checks_schema_and_outcomes() -> None:
    rows = _package_csv("verification_checks.csv")
    assert rows
    assert list(rows[0]) == VERIFICATION_COLUMNS
    assert len({row["check_id"] for row in rows}) == len(rows)
    required = {
        "git_state_at_review",
        "m8e_integrity",
        "m8c_lifecycle_post",
        "m8c_replay",
        "m9b_protocol_validator",
        "ground_truth_policy",
        "memo_decision_path_a",
        "historical_artifacts_unchanged",
    }
    assert required <= {row["check_id"] for row in rows}
    for row in rows:
        assert all(row[column].strip() for column in VERIFICATION_COLUMNS)
        assert row["passed"] == "True"


def test_no_execution_or_data_artifacts_in_memo_package() -> None:
    assert {path.suffix for path in PACKAGE.iterdir()} == {".md", ".csv"}
    assert not any("ground_truth" in path.name for path in PACKAGE.iterdir())
    assert not any(
        path.suffix in {".json", ".toml", ".py", ".joblib"} for path in PACKAGE.iterdir()
    )


def test_tracked_historical_evidence_hashes_are_unchanged() -> None:
    missing = [
        relative for relative in TRACKED_HISTORICAL_HASHES if not (REPO_ROOT / relative).is_file()
    ]
    if missing:
        pytest.skip(f"historical artifacts absent: {missing}")
    for relative, expected in TRACKED_HISTORICAL_HASHES.items():
        assert _sha256(REPO_ROOT / relative) == expected, relative


def test_local_frozen_model_hashes_are_unchanged() -> None:
    missing = [relative for relative in LOCAL_MODEL_HASHES if not (REPO_ROOT / relative).is_file()]
    if missing:
        pytest.skip(f"frozen model artifacts absent: {missing}")
    for relative, expected in LOCAL_MODEL_HASHES.items():
        assert _sha256(REPO_ROOT / relative) == expected, relative


def test_m9b_protocol_validator_still_passes() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "captcha_quality.interference_experiment_protocol", "verify"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "26/26" in completed.stdout
