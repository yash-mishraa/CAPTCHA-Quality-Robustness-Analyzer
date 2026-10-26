from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = REPO_ROOT / "docs" / "PUBLIC_SAFE_DEMO_PLAN.md"


def test_public_safe_plan_exists_and_is_non_empty() -> None:
    assert PLAN_PATH.exists(), f"Plan file not found at {PLAN_PATH}"
    content = PLAN_PATH.read_text(encoding="utf-8")
    assert len(content) > 1000


def test_public_safe_plan_evaluates_all_three_options() -> None:
    content = PLAN_PATH.read_text(encoding="utf-8")
    assert "Option A" in content
    assert "Option B" in content
    assert "Option C" in content
    assert "Results-Only" in content


def test_public_safe_plan_has_all_required_sections() -> None:
    content = PLAN_PATH.read_text(encoding="utf-8")

    required_sections = [
        "1. Evaluation of Public-Data Strategies",
        "2. Source-by-Source Public Asset Inventory",
        "3. Recommended Architecture: Results-Only Public Dashboard",
        "4. Exact Files Cleared for Public Display",
        "5. Required Application Changes for Public-Safe Mode",
        "6. Validation and Acceptance Criteria",
        "7. Explicit Licensing Assumptions and Unresolved Blockers",
        "8. Final Recommendation",
    ]

    for section in required_sections:
        assert section in content, f"Missing required section: {section}"


def test_public_safe_plan_recommends_option_a_and_declares_feasible() -> None:
    content = PLAN_PATH.read_text(encoding="utf-8")
    assert "public_safe_demo_feasible" in content

    final_section = content.split("## 8. Final Recommendation")[1]
    assert "public_safe_demo_feasible" in final_section
    assert "Option A" in final_section


def test_public_safe_plan_strictly_excludes_proprietary_images_and_ground_truth() -> None:
    content = PLAN_PATH.read_text(encoding="utf-8")
    assert "ground_truth.csv" in content
    assert "Strictly Excluded Assets" in content
    assert "images/**" in content
    assert "public_launch_blocked" in content
