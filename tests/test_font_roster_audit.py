"""Tests for the M7E font-roster licensing and availability audit."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from captcha_quality import benchmark_lifecycle as bl
from captcha_quality import font_roster_audit as fra
from captcha_quality.provenance import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "font_roster_audit_v0_1_0.toml"
M7D_REPORT_ROOT = REPO_ROOT / "reports" / "evaluation_design_plan_v0_1_0"

if not (M7D_REPORT_ROOT / "plan_manifest.json").is_file():  # pragma: no cover
    pytest.skip(
        "the frozen M7D plan artifacts are not present; run M7D first",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    """Load the frozen inputs in a tree where the planned benchmark is absent.

    The M7E protocol deliberately refuses to load its inputs once the planned
    benchmark exists. Building the audit in a isolated shadow tree preserves
    that pre-execution check while allowing the historical artifacts to be
    verified after the authorized M8C execution.
    """
    config = fra.load_config(CONFIG_PATH)
    shadow = _shadow_repo(tmp_path_factory.mktemp("m7e_bundle"))
    previous = Path.cwd()
    os.chdir(shadow)
    try:
        inputs = fra.load_inputs(config)
    finally:
        os.chdir(previous)
    result = fra.build_audit(inputs)
    checks = fra.build_isolation_checks(inputs, result, result.report)
    return config, inputs, result, checks


def _verify_in(shadow: Path) -> fra.VerificationReport:
    previous = Path.cwd()
    os.chdir(shadow)
    try:
        return fra.verify_audit(Path("configs/font_roster_audit_v0_1_0.toml"))
    except (fra.AuditError, fra.AuditConfigError) as error:
        return fra.VerificationReport(
            dataset=str(shadow),
            checks=(
                fra.CheckResult("declared_dependency", False, f"{type(error).__name__}: {error}"),
            ),
        )
    finally:
        os.chdir(previous)


def _check_map(report: fra.VerificationReport) -> dict[str, fra.CheckResult]:
    return {check.name: check for check in report.checks}


def _shadow_repo(tmp_path: Path) -> Path:
    """Build a shadow repository with the declared inputs and audit outputs."""
    root = tmp_path / "m7e_shadow"
    config = fra.load_config(CONFIG_PATH)
    for pin in (*config.inputs,):
        target = root / pin.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(REPO_ROOT / pin.path)
    own = root / "configs" / "font_roster_audit_v0_1_0.toml"
    own.parent.mkdir(parents=True, exist_ok=True)
    own.symlink_to(REPO_ROOT / "configs" / "font_roster_audit_v0_1_0.toml")
    output_dir = root / "reports" / "font_roster_audit_v0_1_0"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in fra.OUTPUT_TABLE_NAMES:
        target = output_dir / name
        target.symlink_to(REPO_ROOT / "reports" / "font_roster_audit_v0_1_0" / name)
    for name in ("audit_manifest.json", "report.md", "run_metadata.json"):
        target = output_dir / name
        target.symlink_to(REPO_ROOT / "reports" / "font_roster_audit_v0_1_0" / name)
    return root


# ---------------------------------------------------------------------------
# 1. a valid evidence record and eligibility decision
# ---------------------------------------------------------------------------


def test_evidence_records_and_decisions_are_complete(bundle):
    config, _, result, checks = bundle
    evidence = result.evidence_table
    assert len(evidence) == 10
    assert len(result.eligibility_table) == 10
    assert set(evidence["font_key"]) == {
        "arial",
        "chalkduster",
        "courier_new",
        "georgia",
        "impact",
        "luminari",
        "monaco",
        "times_new_roman",
        "trebuchet_ms",
        "verdana",
    }
    for field in fra.EVIDENCE_FIELDS:
        if field == "fs_type":
            continue
        assert evidence[field].astype(str).str.strip().ne("").all(), field
    decision_columns = {
        "status": "declared_status",
        "reason": "declared_reason",
        "conditions": "declared_conditions",
        "unresolved_questions": "unresolved_questions",
    }
    for field in fra.DECISION_FIELDS:
        column = decision_columns[field]
        assert result.eligibility_table[column].astype(str).str.strip().ne("").all(), field
    assert bool(result.eligibility_table["decision_supported"].all())
    assert checks["passed"].all()


def test_audit_passes_with_declared_dependencies_intact(bundle, tmp_path):
    config, inputs, result, checks = bundle
    shadow = _shadow_repo(tmp_path)
    report = _verify_in(shadow)
    assert report.ok, [f"{c.name}: {c.detail}" for c in report.failures()]
    mapped = _check_map(report)
    assert mapped["frozen_dependencies"].passed
    assert mapped["font_hashes"].passed
    assert mapped["output_file_set"].passed


# ---------------------------------------------------------------------------
# 2. missing license evidence
# ---------------------------------------------------------------------------


def test_missing_license_evidence_forces_review_required():
    config = fra.load_config(CONFIG_PATH)
    evidence = {
        "availability_status": "installed_and_hash_verified",
        "permits_rendering": "yes",
        "permits_output_redistribution": "unverified",
        "permits_font_file_redistribution": "no",
        "attribution_required": "no",
        "reproducible_pin": "yes",
    }
    status, reason = fra._derive_status(evidence, config)
    assert status == "review_required"
    assert "permits_output_redistribution" in reason


def test_missing_license_source_is_rejected_by_the_parser(tmp_path):
    payload = CONFIG_PATH.read_text(encoding="utf-8")
    assert "license_source_url" in payload
    with pytest.raises(fra.AuditConfigError):
        fra._parse_evidence([{"font_key": "arial"}])


# ---------------------------------------------------------------------------
# 3. unsupported or malformed license status
# ---------------------------------------------------------------------------


def test_malformed_license_status_is_rejected():
    with pytest.raises(fra.AuditConfigError):
        fra._parse_decisions(
            [
                {
                    "font_key": "arial",
                    "status": "probably fine",
                    "reason": "x",
                    "conditions": "y",
                    "unresolved_questions": "z",
                }
            ]
        )


def test_unknown_evidence_grade_is_rejected():
    with pytest.raises(fra.AuditConfigError):
        fra._parse_evidence(
            [
                {
                    "font_key": "arial",
                    **{field: "x" for field in fra.EVIDENCE_FIELDS if field != "fs_type"},
                    "fs_type": 0,
                }
            ]
        )


# ---------------------------------------------------------------------------
# 4. an eligible decision without adequate evidence
# ---------------------------------------------------------------------------


def test_eligible_decision_without_evidence_is_unsupported():
    evidence = {
        "availability_status": "installed_and_hash_verified",
        "permits_rendering": "yes",
        "permits_output_redistribution": "unverified",
        "permits_font_file_redistribution": "no",
        "attribution_required": "no",
        "reproducible_pin": "yes",
    }
    supported, detail = fra._declared_status_is_supported("eligible", "review_required", evidence)
    assert supported is False
    assert "not supported" in detail


def test_ineligible_decision_requires_a_positive_prohibition():
    evidence = {
        "availability_status": "installed_and_hash_verified",
        "permits_rendering": "yes",
        "permits_output_redistribution": "unverified",
        "permits_font_file_redistribution": "no",
        "attribution_required": "no",
        "reproducible_pin": "yes",
    }
    supported, _ = fra._declared_status_is_supported("ineligible", "review_required", evidence)
    assert supported is False
    prohibited = dict(evidence, permits_output_redistribution="no")
    supported, _ = fra._declared_status_is_supported("ineligible", "ineligible", prohibited)
    assert supported is True


def test_unavailable_decision_requires_an_unobtainable_face():
    evidence = {
        "availability_status": "installed_and_hash_verified",
        "permits_rendering": "yes",
        "permits_output_redistribution": "yes",
        "permits_font_file_redistribution": "no",
        "attribution_required": "no",
        "reproducible_pin": "yes",
    }
    supported, _ = fra._declared_status_is_supported("unavailable", "eligible", evidence)
    assert supported is False
    missing = dict(evidence, availability_status="not_installed")
    supported, _ = fra._declared_status_is_supported("unavailable", "unavailable", missing)
    assert supported is True


def test_review_required_is_always_accepted():
    evidence = {
        "availability_status": "installed_and_hash_verified",
        "permits_rendering": "yes",
        "permits_output_redistribution": "unverified",
        "permits_font_file_redistribution": "no",
        "attribution_required": "no",
        "reproducible_pin": "yes",
    }
    supported, _ = fra._declared_status_is_supported("review_required", "eligible", evidence)
    assert supported is True


# ---------------------------------------------------------------------------
# 5. a missing or duplicated roster member
# ---------------------------------------------------------------------------


def test_missing_roster_member_is_detected(bundle):
    config, inputs, _, _ = bundle
    with pytest.raises(fra.AuditError):
        fra._roster_matches_plan(config, inputs.m7d_planned_fonts[:-1])


def test_duplicated_roster_member_is_detected(bundle):
    config, inputs, _, _ = bundle
    duplicated = [*inputs.m7d_planned_fonts, inputs.m7d_planned_fonts[0]]
    with pytest.raises(fra.AuditError):
        fra._roster_matches_plan(config, duplicated)


def test_roster_coverage_matches_the_m7d_plan(bundle):
    _, _, result, _ = bundle
    coverage = result.roster_coverage.set_index("font_key")
    assert len(coverage) == 10
    assert bool(coverage["in_m7d_plan"].all())
    assert bool(coverage["has_evidence_record"].all())
    assert bool(coverage["has_decision_record"].all())
    assert set(coverage.index[coverage["expected_role"] == "familiar"]) == {
        "arial",
        "courier_new",
        "times_new_roman",
    }
    assert set(coverage.index[coverage["distinct_suite"] == "M7B held-out evaluation fonts"]) == {
        "georgia",
        "monaco",
        "impact",
    }


# ---------------------------------------------------------------------------
# 6. modified or deleted declared evidence
# ---------------------------------------------------------------------------


def test_modified_declared_input_fails_verification(tmp_path):
    shadow = _shadow_repo(tmp_path)
    target = shadow / "configs" / "evaluation_design_plan_v0_1_0.toml"
    content = target.read_bytes()
    target.unlink()
    target.write_bytes(content + b"\n# tampered\n")
    report = _verify_in(shadow)
    assert not report.ok


def test_deleted_declared_input_fails_verification(tmp_path):
    shadow = _shadow_repo(tmp_path)
    (shadow / "configs" / "evaluation_design_plan_v0_1_0.toml").unlink()
    report = _verify_in(shadow)
    assert not report.ok


def test_modified_audit_output_fails_verification(tmp_path):
    shadow = _shadow_repo(tmp_path)
    target = shadow / "reports" / "font_roster_audit_v0_1_0" / "eligibility.csv"
    target.unlink()
    target.write_text("font_key,declared_status\narial,eligible\n", encoding="utf-8")
    report = _verify_in(shadow)
    assert not report.ok


# ---------------------------------------------------------------------------
# 7. an unexpected protected output
# ---------------------------------------------------------------------------


def test_unexpected_protected_output_fails_verification(tmp_path):
    shadow = _shadow_repo(tmp_path)
    extra = shadow / "reports" / "font_roster_audit_v0_1_0" / "unexpected.csv"
    extra.write_text("x\n", encoding="utf-8")
    report = _verify_in(shadow)
    assert not report.ok
    assert not _check_map(report)["output_file_set"].passed


# ---------------------------------------------------------------------------
# 8. a missing, malformed or tampered dependency manifest
# ---------------------------------------------------------------------------


def test_missing_dependency_manifest_fails_closed(tmp_path):
    shadow = _shadow_repo(tmp_path)
    (shadow / "reports" / "font_roster_audit_v0_1_0" / "run_metadata.json").unlink()
    report = _verify_in(shadow)
    assert not report.ok
    assert not _check_map(report)["metadata"].passed


def test_malformed_dependency_manifest_fails_closed(tmp_path):
    shadow = _shadow_repo(tmp_path)
    manifest = shadow / "reports" / "font_roster_audit_v0_1_0" / "run_metadata.json"
    manifest.unlink()
    manifest.write_text("{not json", encoding="utf-8")
    report = _verify_in(shadow)
    assert not report.ok
    assert not _check_map(report)["metadata"].passed


def test_tampered_dependency_manifest_fails_closed(tmp_path):
    shadow = _shadow_repo(tmp_path)
    manifest = shadow / "reports" / "font_roster_audit_v0_1_0" / "run_metadata.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["frozen_dependencies"]["configs/evaluation_design_plan_v0_1_0.toml"] = "0" * 64
    manifest.unlink()
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    report = _verify_in(shadow)
    assert not report.ok
    assert not _check_map(report)["frozen_dependencies"].passed


def test_tampered_recorded_input_hashes_fail_closed(tmp_path):
    shadow = _shadow_repo(tmp_path)
    manifest = shadow / "reports" / "font_roster_audit_v0_1_0" / "run_metadata.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["inputs"]["m7d_config"] = "0" * 64
    manifest.unlink()
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    report = _verify_in(shadow)
    assert not report.ok
    assert not _check_map(report)["input_hashes"].passed


# ---------------------------------------------------------------------------
# 9. unrelated additions to shared directories
# ---------------------------------------------------------------------------


def test_unrelated_config_data_model_and_report_additions_are_accepted(tmp_path):
    shadow = _shadow_repo(tmp_path)
    (shadow / "configs" / "some_future_milestone_v0_1_0.toml").write_text(
        "# unrelated\n", encoding="utf-8"
    )
    dataset = shadow / "data" / "processed" / "some_future_dataset_v0.1.0"
    dataset.mkdir(parents=True)
    (dataset / "manifest.csv").write_text("x\n", encoding="utf-8")
    model = shadow / "models" / "some_future_model"
    model.mkdir(parents=True)
    (model / "candidate.joblib").write_text("x\n", encoding="utf-8")
    report_dir = shadow / "reports" / "some_future_report_v0_1_0"
    report_dir.mkdir(parents=True)
    (report_dir / "report.md").write_text("x\n", encoding="utf-8")
    report = _verify_in(shadow)
    assert report.ok, [f"{c.name}: {c.detail}" for c in report.failures()]


# ---------------------------------------------------------------------------
# 10. deterministic output and verification
# ---------------------------------------------------------------------------


def test_repeated_verification_is_deterministic(bundle, tmp_path):
    shadow = _shadow_repo(tmp_path)
    first = _verify_in(shadow)
    second = _verify_in(shadow)
    assert first.ok and second.ok
    assert [(c.name, c.passed) for c in first.checks] == [(c.name, c.passed) for c in second.checks]


def test_two_runs_produce_identical_outputs(tmp_path):
    shadow = _shadow_repo(tmp_path)
    output_dir = shadow / "reports" / "_m7e_test_output"
    previous = Path.cwd()
    os.chdir(shadow)
    try:
        first_dir, first_metadata = fra.run_audit(
            CONFIG_PATH, output_reports=output_dir, overwrite=True
        )
        first_hashes = fra._output_hashes(first_dir)
        second_dir, second_metadata = fra.run_audit(
            CONFIG_PATH, output_reports=output_dir, overwrite=True
        )
        assert second_dir == first_dir
        assert fra._output_hashes(second_dir) == first_hashes
        assert first_metadata["summary"] == second_metadata["summary"]
        assert fra.verify_audit(CONFIG_PATH, output_reports=output_dir).ok
    finally:
        os.chdir(previous)


# ---------------------------------------------------------------------------
# 11. M7C and M7D verifiers remain intact
# ---------------------------------------------------------------------------


def test_m7c_and_m7d_verifiers_still_pass_and_still_detect_tampering(bundle):
    from captcha_quality import cross_font_analysis as cfa

    m7c = cfa.verify_analysis(Path("configs/cross_font_analysis_v0_1_0.toml"))
    assert m7c.ok, [f"{c.name}: {c.detail}" for c in m7c.failures()]
    transition = bl._m7d_verifier_report()
    assert transition["planning_state_failed"] is True
    assert transition["other_failures"] == []

    # M7C still detects a modified declared dependency.
    membership = cfa.ProtectedDependencyMembership(
        version="test-v1",
        config_files=("configs/a.toml",),
        dataset_roots=("data/processed/ds",),
        model_roots=(),
        report_roots=(),
        ignored_roots=(),
        output_files=(),
    )
    (tmp := Path(os.environ.get("M7E_TMP", "/tmp"))).mkdir(parents=True, exist_ok=True)
    work = tmp / "m7c_probe"
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
# Additional milestone-specific invariants
# ---------------------------------------------------------------------------


def test_no_font_binary_is_present_in_the_working_tree(bundle):
    _, _, _, checks = bundle
    assert checks.loc[checks["check"] == "no_font_binaries_committed", "passed"].all()
    assert fra._font_binary_paths() == []


def test_m7d_lifecycle_check_is_preserved(bundle):
    _, _, _, checks = bundle
    assert checks.loc[checks["check"] == "planned_benchmark_paths_absent", "passed"].all()
    # The pre-execution M7E check is not weakened: it still refuses the real
    # repository now that the authorized M8C outputs exist.
    config = fra.load_config(CONFIG_PATH)
    with pytest.raises(fra.AuditError, match="planned_benchmark_paths_absent"):
        fra.load_inputs(config)
    assert (REPO_ROOT / "data" / "processed" / "matched_font_benchmark_v0.1.0").is_dir()
    assert (REPO_ROOT / "reports" / "matched_font_benchmark_v0_1_0").is_dir()


def test_gate_is_reported_and_the_unresolved_face_is_retained(bundle):
    _, _, result, _ = bundle
    gate = fra._gate_summary(result.eligibility_table)
    assert gate["total_faces"] == 10
    assert gate["eligible"] == 9
    assert gate["review_required"] == 1
    assert gate["ineligible"] == 0
    assert gate["unavailable"] == 0
    assert gate["audit_gate"] == "partially_passed"
    assert gate["benchmark_generation_gate"] == "closed"
    statuses = result.eligibility_table.set_index("font_key")["declared_status"]
    assert statuses["luminari"] == "review_required"
    assert "luminari" in set(result.eligibility_table["font_key"])


def test_report_records_the_scope_and_the_boundary(bundle):
    _, _, result, _ = bundle
    report = result.report
    flattened = " ".join(report.split())
    for section in fra.REPORT_SECTIONS:
        assert section in report
    assert "EXP-0018" in report
    assert "human readability" in report
    assert "font binaries" in report
    assert "no_font_binaries_committed" not in report
    assert "artificial-intelligence training restriction" in flattened
    assert "Georgia, Monaco and Impact are not new to the project" in flattened


def test_evidence_table_records_the_observed_font_metadata(bundle):
    _, _, result, _ = bundle
    evidence = result.evidence_table.set_index("font_key")
    assert bool((evidence["observed_font_sha256"] == evidence["m7d_font_sha256"]).all())
    assert bool((evidence["availability_status"] == "installed_and_hash_verified").all())
    assert evidence.loc["georgia", "observed_family"] == "Georgia"
    assert int(evidence.loc["georgia", "observed_fs_type"]) == 8
    assert evidence.loc["monaco", "observed_license_text"] == ""


def test_historical_artifacts_are_untouched(bundle):
    """The audit must not modify the M7D plan or any pinned input."""
    config, inputs, _, _ = bundle
    for pin in config.inputs:
        assert sha256_file(REPO_ROOT / pin.path) == pin.sha256
    m7d_manifest = json.loads((M7D_REPORT_ROOT / "run_metadata.json").read_text(encoding="utf-8"))
    assert all(
        sha256_file(M7D_REPORT_ROOT / name) == digest
        for name, digest in m7d_manifest["output_hashes"].items()
    )


def test_protocol_declares_no_font_file_redistribution(bundle):
    config, _, _, _ = bundle
    assert config.audit.font_file_redistribution_required is False
    assert "never copies a font binary" in config.audit.font_file_redistribution_rationale
    assert set(config.statuses) == {
        fra.StatusMeaning(status=name, meaning=item.meaning)
        for name, item in zip(
            fra.ELIGIBILITY_STATUSES,
            [s for s in config.statuses],
            strict=True,
        )
    }
    assert {item.status for item in config.statuses} == set(fra.ELIGIBILITY_STATUSES)


def test_roster_config_matches_the_audited_membership(bundle):
    config, _, _, _ = bundle
    assert config.roster.expected_count == 10
    assert len(config.roster.expected_familiar) == 3
    assert len(config.roster.expected_held_out) == 7
    assert config.roster.source == "configs/evaluation_design_plan_v0_1_0.toml"
    assert set(config.roster.project_statuses) == set(fra.PROJECT_STATUSES)


def test_evidence_table_has_one_row_per_roster_member(bundle):
    _, _, result, _ = bundle
    assert not result.evidence_table["font_key"].duplicated().any()
    assert not result.eligibility_table["font_key"].duplicated().any()
    assert set(result.evidence_table["font_key"]) == set(result.eligibility_table["font_key"])
    assert isinstance(result.evidence_table, pd.DataFrame)
