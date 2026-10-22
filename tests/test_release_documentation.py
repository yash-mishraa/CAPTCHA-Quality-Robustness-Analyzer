"""Focused release-documentation and claims checks for the v1.0.0 package."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import pytest

import captcha_quality

REPO_ROOT = Path(__file__).resolve().parents[1]

DOCUMENTATION = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "PROJECT_OVERVIEW.md",
    REPO_ROOT / "docs" / "ARCHITECTURE.md",
    REPO_ROOT / "docs" / "METHODOLOGY.md",
    REPO_ROOT / "docs" / "EXPERIMENT_LOG.md",
    REPO_ROOT / "docs" / "RESEARCH_SUMMARY.md",
    REPO_ROOT / "docs" / "REPRODUCIBILITY.md",
    REPO_ROOT / "docs" / "RELEASE_NOTES_v1.0.0.md",
    REPO_ROOT / "docs" / "RELEASE_CHECKLIST_v1.0.0.md",
]

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


def _text(path: Path) -> str:
    assert path.is_file(), path
    return path.read_text(encoding="utf-8")


def _normalise(value: str) -> str:
    return value.replace("−", "-").replace("\u00a0", " ")


def _links(text: str) -> list[str]:
    return re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_release_documentation_membership() -> None:
    for path in DOCUMENTATION:
        assert path.is_file(), path

    assert (REPO_ROOT / "LICENSE").is_file()
    assert (REPO_ROOT / "docs" / "BENCHMARK_INTEGRITY.md").is_file()


def test_package_version_is_release_version() -> None:
    assert captcha_quality.__version__ == "1.0.0"


def test_readme_has_required_release_sections_and_status() -> None:
    readme = _text(REPO_ROOT / "README.md")
    for heading in [
        "## Research question",
        "## Scope and non-claims",
        "## Contributions",
        "## Architecture and workflow",
        "## Repository map",
        "## Setup",
        "## Verification and tests",
        "## Results",
        "## Limitations",
        "## Research status",
    ]:
        assert heading in readme, heading

    assert "M0–M9C" in readme
    assert "M8C" in readme and "M9C" in readme
    assert "Path A" in readme
    assert "stop experimentation" in readme.lower()
    assert "M9B" in readme and "not" in readme.lower()
    assert "H1" in readme and "H2" in readme and "H3" in readme
    assert "not_tested" in readme.lower() or "not tested" in readme.lower()
    assert "does **not** measure human" in readme


def test_headline_metrics_are_consistent_with_authoritative_reports() -> None:
    combined = _normalise(
        "\n".join(_text(path) for path in [REPO_ROOT / "README.md", *DOCUMENTATION[5:8]])
    )

    for value in [
        "0.0157",
        "0.0226",
        "0.9865",
        "0.0156",
        "0.0220",
        "0.9872",
        "0.0493",
        "0.0590",
        "0.1855",
        "0.063427",
        "0.092952",
        "0.743537",
        "0.059607",
        "0.088610",
        "0.732898",
        "0.090825",
        "0.118525",
        "-0.430242",
        "0.102073",
        "0.132114",
        "-0.575158",
        "0.079578",
        "0.103162",
        "-0.298911",
        "10.6%",
        "0.241",
        "0.337",
    ]:
        assert value in combined, value

    assert "20261023" in combined
    assert "2,000" in combined
    assert "98.333%" in combined
    assert "c9b1625de3bb503f22fa14f9b6468b028e609b6c7febe7c307dafd0e3e1156e0" in combined
    assert "9a75725ba1ccf82ab42f009aa744a102cce86a6a9bb1d1aa86431362213507f8" in combined


def test_m9b_is_unexecuted_and_hypotheses_remain_not_tested() -> None:
    combined = _normalise(
        "\n".join(
            _text(path)
            for path in [
                REPO_ROOT / "README.md",
                REPO_ROOT / "docs" / "RESEARCH_SUMMARY.md",
                REPO_ROOT / "docs" / "EXPERIMENT_LOG.md",
                REPO_ROOT / "docs" / "RELEASE_NOTES_v1.0.0.md",
            ]
        )
    ).lower()

    assert "m9b" in combined
    assert "not executed" in combined or "unexecuted" in combined
    assert "not_tested" in combined or "not tested" in combined
    assert "path a" in combined
    assert "h1" in combined and "h2" in combined and "h3" in combined


def test_release_notes_and_checklist_do_not_claim_unsupported_scope() -> None:
    combined = _normalise(
        _text(REPO_ROOT / "docs" / "RELEASE_NOTES_v1.0.0.md")
        + "\n"
        + _text(REPO_ROOT / "docs" / "RELEASE_CHECKLIST_v1.0.0.md")
    ).lower()

    assert "project complete" in combined
    assert "v1.0.0 ready" in combined
    assert "human readability" in combined
    assert "does not" in combined or "cannot" in combined
    assert "no github release" in combined or "not_applicable" in combined


def test_closure_checklist_has_no_blocked_items_and_required_rows() -> None:
    checklist = _text(REPO_ROOT / "docs" / "RELEASE_CHECKLIST_v1.0.0.md")
    assert not re.search(r"\|\s*`?blocked`?\s*\|", checklist, flags=re.IGNORECASE)

    for identifier in [f"C-{number:02d}" for number in range(1, 26)]:
        assert identifier in checklist, identifier

    assert "not_applicable" in checklist
    assert "Ground-truth text" in checklist
    assert "No dataset, image, prediction" in checklist
    assert "M9C Path A" in checklist


def test_markdown_links_resolve() -> None:
    for path in DOCUMENTATION:
        text = _text(path)
        for target in _links(text):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            relative = target.split("#", 1)[0]
            if not relative:
                continue
            resolved = (path.parent / relative).resolve()
            assert resolved.exists(), f"{path.relative_to(REPO_ROOT)} -> {target}"


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


def test_ground_truth_is_not_tracked() -> None:
    completed = subprocess.run(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            "data/processed/matched_font_benchmark_v0.1.0/ground_truth.csv",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0


def test_architecture_and_repository_map_paths_exist() -> None:
    architecture = _text(REPO_ROOT / "docs" / "ARCHITECTURE.md")
    for module in [
        "render.py",
        "corruptions.py",
        "interference.py",
        "geometric.py",
        "quality_metrics.py",
        "features.py",
        "dataset.py",
        "splits.py",
        "benchmark.py",
        "modelling.py",
        "matched_font_benchmark.py",
        "benchmark_integrity.py",
        "interference_error_analysis.py",
        "interference_experiment_protocol.py",
    ]:
        assert module in architecture
        assert (REPO_ROOT / "src" / "captcha_quality" / module).is_file(), module
