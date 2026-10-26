from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = REPO_ROOT / "docs" / "LIVE_DEMO_READINESS_AUDIT.md"


def test_audit_document_exists_and_is_non_empty() -> None:
    assert AUDIT_PATH.exists(), f"Audit file not found at {AUDIT_PATH}"
    content = AUDIT_PATH.read_text(encoding="utf-8")
    assert len(content) > 1000


def test_audit_document_has_all_required_sections() -> None:
    content = AUDIT_PATH.read_text(encoding="utf-8")

    required_sections = [
        "1. Audit Scope and Exact Repository Revision",
        "2. Asset and Licensing Inventory",
        "3. Public Artifact Exposure and Privacy Review",
        "4. Deployment Feasibility Review",
        "5. Data-Integrity and Startup-Read-Only Review",
        "6. Documentation Discrepancy",
        "7. Findings Classification",
        "8. Exact Evidence for Every Classification",
        "9. Recommended Remediation",
        "10. Explicit Final Gate",
    ]

    for section in required_sections:
        assert section in content, f"Missing required section: {section}"


def test_audit_document_explicit_gate_is_public_launch_blocked() -> None:
    content = AUDIT_PATH.read_text(encoding="utf-8")
    assert "public_launch_blocked" in content
    # Final gate must explicitly record public_launch_blocked
    gate_section = content.split("## 10. Explicit Final Gate")[1]
    assert "public_launch_blocked" in gate_section
    assert "GATE: public_launch_ready" not in gate_section


def test_audit_documents_m7e_and_b11_blockers() -> None:
    content = AUDIT_PATH.read_text(encoding="utf-8")
    assert "M7E" in content
    assert "B11" in content
    assert "a product, a model, or a deployment" in content
    assert "ground_truth.csv" in content
