"""Tests for the M7E v0.2.0 font-roster licence resolution.

The v0.2.0 protocol supersedes v0.1.0 for the Luminari face only. These
tests pin the resolution: the decision is derived from recorded evidence, an
unsupported ``eligible`` decision is rejected, the roster membership and
provenance are preserved, the superseded v0.1.0 protocol keeps its own
conclusions and stays verifiable, and no historical artifact is modified.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from captcha_quality import evaluation_plan as ep
from captcha_quality import font_roster_audit as fra
from captcha_quality.provenance import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
V1_CONFIG = REPO_ROOT / "configs" / "font_roster_audit_v0_1_0.toml"
V2_CONFIG = REPO_ROOT / "configs" / "font_roster_audit_v0_2_0.toml"
M7D_REPORT_ROOT = REPO_ROOT / "reports" / "evaluation_design_plan_v0_1_0"

if not (M7D_REPORT_ROOT / "plan_manifest.json").is_file():  # pragma: no cover
    pytest.skip(
        "the frozen M7D plan artifacts are not present; run M7D first",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def v1():
    """The superseded protocol version, loaded but never modified."""
    config = fra.load_config(V1_CONFIG)
    inputs = fra.load_inputs(config)
    result = fra.build_audit(inputs)
    return config, inputs, result


@pytest.fixture(scope="module")
def v2():
    """The current protocol version."""
    config = fra.load_config(V2_CONFIG)
    inputs = fra.load_inputs(config)
    result = fra.build_audit(inputs)
    checks = fra.build_isolation_checks(inputs, result, result.report)
    return config, inputs, result, checks


def _verify_in(shadow: Path, config_name: str) -> fra.VerificationReport:
    previous = Path.cwd()
    os.chdir(shadow)
    try:
        return fra.verify_audit(Path("configs") / config_name)
    except (fra.AuditError, fra.AuditConfigError) as error:
        return fra.VerificationReport(
            dataset=str(shadow),
            checks=(
                fra.CheckResult("declared_dependency", False, f"{type(error).__name__}: {error}"),
            ),
        )
    finally:
        os.chdir(previous)


def _shadow_repo(tmp_path: Path, config_name: str) -> Path:
    """Build a shadow repository with the declared inputs and audit outputs."""
    root = tmp_path / "m7e_shadow"
    config = fra.load_config(REPO_ROOT / "configs" / config_name)
    for pin in config.inputs:
        target = root / pin.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(REPO_ROOT / pin.path)
    own = root / "configs" / config_name
    own.parent.mkdir(parents=True, exist_ok=True)
    own.symlink_to(REPO_ROOT / "configs" / config_name)
    output_dir = root / config.run.output_reports
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in fra.OUTPUT_TABLE_NAMES:
        (output_dir / name).symlink_to(REPO_ROOT / config.run.output_reports / name)
    for name in ("audit_manifest.json", "report.md", "run_metadata.json"):
        (output_dir / name).symlink_to(REPO_ROOT / config.run.output_reports / name)
    return root


# ---------------------------------------------------------------------------
# 1. the resolved decision and its basis
# ---------------------------------------------------------------------------


def test_luminari_is_resolved_to_eligible(v2):
    _, _, result, _ = v2
    eligibility = result.eligibility_table.set_index("font_key")
    assert eligibility.loc["luminari", "declared_status"] == "eligible"
    assert eligibility.loc["luminari", "derived_status"] == "eligible"
    assert bool(eligibility.loc["luminari", "decision_supported"])


def test_the_decision_records_its_evidence_and_scope(v2):
    _, _, result, _ = v2
    evidence = result.evidence_table.set_index("font_key")
    eligibility = result.eligibility_table.set_index("font_key")
    assert evidence.loc["luminari", "permits_rendering"] == "yes"
    assert evidence.loc["luminari", "permits_output_redistribution"] == "yes"
    assert evidence.loc["luminari", "permits_font_file_redistribution"] == "no"
    assert evidence.loc["luminari", "attribution_required"] == "no"
    assert evidence.loc["luminari", "reproducible_pin"] == "yes"
    assert evidence.loc["luminari", "license_source_url"] == "https://canadatype.com/eula/"
    excerpt = evidence.loc["luminari", "license_evidence_excerpt"]
    for provision in ("section 2E", "section 3", "section 11", "section 12", "section 14"):
        assert provision in excerpt, provision
    reason = eligibility.loc["luminari", "declared_reason"]
    assert "macOS 27 Software License Agreement" in reason
    assert "output device" in reason
    conditions = eligibility.loc["luminari", "declared_conditions"]
    assert "evaluation only" in conditions
    assert "font file" in conditions
    assert eligibility.loc["luminari", "unresolved_questions"].strip()


def test_all_ten_faces_are_eligible_and_the_gate_is_open(v2):
    _, _, result, checks = v2
    gate = fra._gate_summary(result.eligibility_table)
    assert gate["eligible"] == 10
    assert gate["review_required"] == 0
    assert gate["ineligible"] == 0
    assert gate["unavailable"] == 0
    assert gate["audit_gate"] == "passed"
    assert gate["benchmark_generation_gate"] == "open"
    assert checks["passed"].all()


# ---------------------------------------------------------------------------
# 2. missing or ambiguous evidence cannot produce eligible
# ---------------------------------------------------------------------------


def test_ambiguous_evidence_cannot_produce_eligible(v2):
    config, _, _, _ = v2
    evidence = {
        "availability_status": "installed_and_hash_verified",
        "permits_rendering": "yes",
        "permits_output_redistribution": "unverified",
        "permits_font_file_redistribution": "no",
        "attribution_required": "unverified",
        "reproducible_pin": "yes",
    }
    status, reason = fra._derive_status(evidence, config)
    assert status == "review_required"
    assert "unverified" in reason
    supported, detail = fra._declared_status_is_supported("eligible", status, evidence)
    assert supported is False
    assert "not supported" in detail


def test_missing_licence_source_is_rejected_by_the_parser():
    with pytest.raises(fra.AuditConfigError):
        fra._parse_evidence([{"font_key": "luminari"}])


def test_an_eligible_decision_without_evidence_is_rejected(v2):
    config, _, _, _ = v2
    evidence = {
        "availability_status": "not_installed",
        "permits_rendering": "yes",
        "permits_output_redistribution": "yes",
        "permits_font_file_redistribution": "no",
        "attribution_required": "no",
        "reproducible_pin": "yes",
    }
    status, _ = fra._derive_status(evidence, config)
    assert status == "unavailable"
    supported, _ = fra._declared_status_is_supported("eligible", status, evidence)
    assert supported is False


# ---------------------------------------------------------------------------
# 3. roster membership and provenance are preserved
# ---------------------------------------------------------------------------


def test_roster_membership_is_unchanged_between_protocol_versions(v1, v2):
    config_v1, _, result_v1 = v1
    config_v2, _, result_v2, _ = v2
    assert config_v1.roster == config_v2.roster
    assert config_v1.statuses == config_v2.statuses
    assert config_v1.derivation == config_v2.derivation
    assert config_v1.inputs == config_v2.inputs
    coverage_v1 = result_v1.roster_coverage.set_index("font_key")
    coverage_v2 = result_v2.roster_coverage.set_index("font_key")
    assert list(coverage_v1.index) == list(coverage_v2.index)
    assert coverage_v1["expected_role"].to_dict() == coverage_v2["expected_role"].to_dict()
    assert coverage_v1["distinct_suite"].to_dict() == coverage_v2["distinct_suite"].to_dict()


def test_only_the_luminari_record_changed_between_protocol_versions(v1, v2):
    config_v1, _, _ = v1
    config_v2, _, _, _ = v2
    evidence_v1 = {record.font_key: record.fields for record in config_v1.evidence}
    evidence_v2 = {record.font_key: record.fields for record in config_v2.evidence}
    changed = [key for key in evidence_v1 if evidence_v1[key] != evidence_v2[key]]
    assert changed == ["luminari"]
    decisions_v1 = {record.font_key: record for record in config_v1.decisions}
    decisions_v2 = {record.font_key: record for record in config_v2.decisions}
    changed = [key for key in decisions_v1 if decisions_v1[key] != decisions_v2[key]]
    assert changed == ["luminari"]


def test_the_m7b_font_reuse_distinction_is_preserved(v2):
    _, _, result, _ = v2
    coverage = result.roster_coverage.set_index("font_key")
    assert set(coverage.index[coverage["distinct_suite"] == "M7B held-out evaluation fonts"]) == {
        "georgia",
        "monaco",
        "impact",
    }
    assert set(coverage.index[coverage["distinct_suite"] == "M3 development fonts"]) == {
        "arial",
        "courier_new",
        "times_new_roman",
    }
    assert set(coverage.index[coverage["distinct_suite"] == "new to the project evaluation"]) == {
        "verdana",
        "trebuchet_ms",
        "chalkduster",
        "luminari",
    }


def test_the_superseded_protocol_keeps_its_own_conclusion(v1):
    config, _, result = v1
    eligibility = result.eligibility_table.set_index("font_key")
    assert eligibility.loc["luminari", "declared_status"] == "review_required"
    gate = fra._gate_summary(result.eligibility_table)
    assert gate["audit_gate"] == "partially_passed"
    assert gate["benchmark_generation_gate"] == "closed"
    assert "superseded" in result.report.lower()
    assert config.audit.supersedes == ""


def test_the_current_protocol_declares_its_supersession(v2):
    config, _, result, _ = v2
    assert "font_roster_audit_v0_1_0.toml" in config.audit.supersedes
    assert config.audit.experiment == "EXP-0019"
    assert config.audit.residual_questions.strip()
    assert "supersedes" not in result.report.lower()


# ---------------------------------------------------------------------------
# 4. historical artifacts remain unchanged
# ---------------------------------------------------------------------------


def test_the_superseded_protocol_is_still_verifiable(tmp_path):
    shadow = _shadow_repo(tmp_path, "font_roster_audit_v0_1_0.toml")
    report = _verify_in(shadow, "font_roster_audit_v0_1_0.toml")
    assert report.ok, [f"{c.name}: {c.detail}" for c in report.failures()]


def test_the_current_protocol_verifies(tmp_path):
    shadow = _shadow_repo(tmp_path, "font_roster_audit_v0_2_0.toml")
    report = _verify_in(shadow, "font_roster_audit_v0_2_0.toml")
    assert report.ok, [f"{c.name}: {c.detail}" for c in report.failures()]


def test_modifying_the_current_protocol_fails_verification(tmp_path):
    shadow = _shadow_repo(tmp_path, "font_roster_audit_v0_2_0.toml")
    target = shadow / "configs" / "font_roster_audit_v0_2_0.toml"
    content = target.read_bytes()
    target.unlink()
    target.write_bytes(content + b"\n# tampered\n")
    report = _verify_in(shadow, "font_roster_audit_v0_2_0.toml")
    assert not report.ok


def test_tampered_evidence_manifest_fails_closed(tmp_path):
    shadow = _shadow_repo(tmp_path, "font_roster_audit_v0_2_0.toml")
    manifest = shadow / "reports" / "font_roster_audit_v0_2_0" / "run_metadata.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["frozen_dependencies"]["configs/evaluation_design_plan_v0_1_0.toml"] = "0" * 64
    manifest.unlink()
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    report = _verify_in(shadow, "font_roster_audit_v0_2_0.toml")
    assert not report.ok


def test_historical_milestone_artifacts_are_untouched(v2):
    config, _, _, _ = v2
    for pin in config.inputs:
        assert sha256_file(REPO_ROOT / pin.path) == pin.sha256
    m7d_manifest = json.loads((M7D_REPORT_ROOT / "run_metadata.json").read_text(encoding="utf-8"))
    assert all(
        sha256_file(M7D_REPORT_ROOT / name) == digest
        for name, digest in m7d_manifest["output_hashes"].items()
    )
    assert sha256_file(V1_CONFIG) == (
        "5c8140e4bebb4ede473fdbf6499b476afc73157a595a4a16a16be61805038755"
    )


def test_the_planned_benchmark_paths_are_still_absent(v2):
    _, _, _, checks = v2
    assert checks.loc[checks["check"] == "planned_benchmark_paths_absent", "passed"].all()
    assert not (REPO_ROOT / "data" / "processed" / "matched_font_benchmark_v0.1.0").exists()
    assert not (REPO_ROOT / "reports" / "matched_font_benchmark_v0_1_0").exists()
    assert fra._font_binary_paths() == []


# ---------------------------------------------------------------------------
# 5. M7C and M7D verifiers still pass and still detect tampering
# ---------------------------------------------------------------------------


def test_m7c_and_m7d_verifiers_still_pass_and_detect_tampering(v2):
    from captcha_quality import cross_font_analysis as cfa

    m7c = cfa.verify_analysis(Path("configs/cross_font_analysis_v0_1_0.toml"))
    assert m7c.ok, [f"{c.name}: {c.detail}" for c in m7c.failures()]
    m7d = ep.verify_plan(Path("configs/evaluation_design_plan_v0_1_0.toml"))
    assert m7d.ok, [f"{c.name}: {c.detail}" for c in m7d.failures()]

    membership = cfa.ProtectedDependencyMembership(
        version="test-v1",
        config_files=("configs/a.toml",),
        dataset_roots=("data/processed/ds",),
        model_roots=(),
        report_roots=(),
        ignored_roots=(),
        output_files=(),
    )
    work = Path(os.environ.get("M7E_TMP", "/tmp")) / "m7c_probe_v2"
    if work.exists():
        shutil.rmtree(work)
    (work / "configs").mkdir(parents=True)
    (work / "configs" / "a.toml").write_text("a = 1\n", encoding="utf-8")
    dataset = work / "data" / "processed" / "ds"
    dataset.mkdir(parents=True)
    (dataset / "manifest.csv").write_text("x\n", encoding="utf-8")
    previous = Path.cwd()
    os.chdir(work)
    try:
        recorded = cfa._scoped_fingerprint(membership)
        (work / "configs" / "a.toml").write_text("a = 2\n", encoding="utf-8")
        current = cfa._scoped_fingerprint(membership)
    finally:
        os.chdir(previous)
        shutil.rmtree(work, ignore_errors=True)
    assert cfa._scoped_fingerprint_mismatches(recorded, current)["modified"] == ["configs/a.toml"]


# ---------------------------------------------------------------------------
# 6. determinism
# ---------------------------------------------------------------------------


def test_repeated_verification_of_the_current_protocol_is_deterministic(v2):
    first = fra.verify_audit(Path("configs/font_roster_audit_v0_2_0.toml"))
    second = fra.verify_audit(Path("configs/font_roster_audit_v0_2_0.toml"))
    assert first.ok and second.ok
    assert [(c.name, c.passed) for c in first.checks] == [(c.name, c.passed) for c in second.checks]
