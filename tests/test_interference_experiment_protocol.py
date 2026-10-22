"""Focused tests for the M9B preregistered interference experiment protocol."""

from __future__ import annotations

import copy
import csv
from pathlib import Path

import pytest

from captcha_quality import interference_experiment_protocol as protocol

REPO_ROOT = Path(__file__).resolve().parents[1]


def _package() -> protocol.ProtocolPackage:
    return protocol.load_package(REPO_ROOT)


def _check_by_id(checks: tuple[protocol.Check, ...], check_id: str) -> protocol.Check:
    matches = [check for check in checks if check.check_id == check_id]
    assert len(matches) == 1
    return matches[0]


def test_package_membership_and_required_files() -> None:
    package = _package()
    checks = protocol.validate_package_membership(package)
    assert _check_by_id(checks, "package_membership").passed is True
    assert sorted(protocol.REQUIRED_FILES) == sorted(
        str(path.relative_to(package.root)) for path in package.root.rglob("*") if path.is_file()
    )


def test_plan_schema_and_required_fields() -> None:
    plan = _package().plan
    checks = protocol.validate_plan_structure(plan)
    assert _check_by_id(checks, "plan_schema").passed is True
    assert _check_by_id(checks, "plan_milestone").passed is True
    assert _check_by_id(checks, "plan_status").passed is True
    assert _check_by_id(checks, "experiment_required_fields").passed is True
    assert _check_by_id(checks, "hypothesis_ids_unique").passed is True
    assert len(plan["experiments"]) == 3
    assert {experiment["hypothesis_id"] for experiment in plan["experiments"]} == {"H1", "H2", "H3"}
    for experiment in plan["experiments"]:
        assert experiment["primary_estimand_units"]


def test_hypothesis_matrix_and_plan_are_consistent() -> None:
    package = _package()
    checks = protocol.validate_matrix(package.matrix, package.plan)
    assert _check_by_id(checks, "matrix_columns").passed is True
    assert _check_by_id(checks, "matrix_row_count").passed is True
    assert _check_by_id(checks, "matrix_ids_unique").passed is True
    assert _check_by_id(checks, "matrix_plan_consistency").passed is True
    assert len(package.matrix) == 3
    assert {row["hypothesis_id"] for row in package.matrix} == {"H1", "H2", "H3"}


def test_allowed_decision_states_and_no_preemption() -> None:
    package = _package()
    plan_checks = protocol.validate_plan_structure(package.plan)
    matrix_checks = protocol.validate_matrix(package.matrix, package.plan)
    assert _check_by_id(plan_checks, "allowed_decision_states").passed is True
    assert _check_by_id(plan_checks, "decision_state_registry").passed is True
    assert _check_by_id(matrix_checks, "matrix_decision_states").passed is True
    assert _check_by_id(matrix_checks, "matrix_not_preempted").passed is True
    assert set(protocol.ALLOWED_DECISION_STATES) == {
        "not_tested",
        "support",
        "falsify",
        "inconclusive",
        "not_testable",
    }


def test_m8c_derived_testing_is_labelled_exploratory() -> None:
    package = _package()
    checks = protocol.validate_plan_structure(package.plan)
    assert _check_by_id(checks, "m8c_derived_testing_is_exploratory").passed is True
    assert _check_by_id(checks, "m8c_is_not_confirmatory").passed is True
    for experiment in package.plan["experiments"]:
        assert experiment["m8c_role"] == "exploratory_only"
        assert experiment["data_regime"].startswith("confirmatory_new_independent_source_set")
    assert package.plan["data_regimes"]["m8c_observed"]["confirmatory"] is False
    assert package.plan["data_regimes"]["confirmatory"]["confirmatory"] is True


def test_missing_field_fails_closed() -> None:
    plan = copy.deepcopy(_package().plan)
    del plan["experiments"][0]["support_rule"]
    checks = protocol.validate_plan_structure(plan)
    failed = _check_by_id(checks, "experiment_required_fields")
    assert failed.passed is False
    assert "support_rule" in failed.details


def test_duplicate_hypothesis_id_fails_closed() -> None:
    plan = copy.deepcopy(_package().plan)
    plan["experiments"][1]["hypothesis_id"] = plan["experiments"][0]["hypothesis_id"]
    checks = protocol.validate_plan_structure(plan)
    assert _check_by_id(checks, "hypothesis_ids_unique").passed is False


def test_invalid_decision_state_fails_closed() -> None:
    plan = copy.deepcopy(_package().plan)
    plan["experiments"][0]["current_decision_state"] = "supported"
    checks = protocol.validate_plan_structure(plan)
    assert _check_by_id(checks, "allowed_decision_states").passed is False


def test_matrix_plan_mismatch_fails_closed() -> None:
    package = _package()
    matrix = [dict(row) for row in package.matrix]
    matrix[0]["support_rule"] = "a different support rule"
    checks = protocol.validate_matrix(matrix, package.plan)
    assert _check_by_id(checks, "matrix_plan_consistency").passed is False


def test_verification_table_schema_and_all_passed() -> None:
    package = _package()
    checks = protocol.validate_verification_checks(package.verification_checks)
    assert _check_by_id(checks, "verification_columns").passed is True
    assert _check_by_id(checks, "verification_ids_unique").passed is True
    assert _check_by_id(checks, "verification_all_passed").passed is True
    assert all(row["passed"] == "True" for row in package.verification_checks)


def test_failed_verification_row_fails_closed() -> None:
    rows = [dict(row) for row in _package().verification_checks]
    rows[0]["passed"] = "False"
    checks = protocol.validate_verification_checks(rows)
    assert _check_by_id(checks, "verification_all_passed").passed is False


def test_provenance_hashes_and_m9a_revision_chain() -> None:
    plan = _package().plan
    checks = protocol.validate_provenance(REPO_ROOT, plan)
    assert _check_by_id(checks, "provenance_hashes").passed is True
    assert _check_by_id(checks, "m9a_revision_chain").passed is True
    assert plan["predecessor"]["m9a_analysis_revision"] == protocol.M9A_ANALYSIS_REVISION
    assert plan["predecessor"]["m9a_report_revision"] == protocol.M9A_REPORT_REVISION


def test_provenance_hash_mismatch_fails_closed() -> None:
    plan = copy.deepcopy(_package().plan)
    plan["provenance"]["inputs"][0]["sha256"] = "0" * 64
    checks = protocol.validate_provenance(REPO_ROOT, plan)
    assert _check_by_id(checks, "provenance_hashes").passed is False


def test_no_experiment_model_fit_or_generation_entrypoint() -> None:
    checks = protocol.validate_static_scope()
    assert _check_by_id(checks, "static_scope_no_execution").passed is True
    cli_checks = protocol.validate_cli_scope()
    assert _check_by_id(cli_checks, "cli_only_verify").passed is True

    parser = protocol.build_parser()
    subparsers = [
        action for action in parser._actions if action.__class__.__name__ == "_SubParsersAction"
    ]
    assert len(subparsers) == 1
    assert sorted(subparsers[0].choices) == ["verify"]


def test_validator_cli_accepts_verify_and_rejects_run(capsys: pytest.CaptureFixture[str]) -> None:
    assert protocol.main(["verify", "--repo-root", str(REPO_ROOT)]) == 0
    output = capsys.readouterr().out
    assert "verification: PASS" in output

    with pytest.raises(SystemExit) as excinfo:
        protocol.main(["run"])
    assert excinfo.value.code == 2


def test_sample_size_and_precision_are_not_invented() -> None:
    sample_size = _package().plan["common_design"]["sample_size"]
    assert sample_size["current_data_can_support_confirmatory_claim"] is False
    assert sample_size["pilot_required"] is True
    assert sample_size["power_claim"] is False
    assert sample_size["optional_stopping"] is False
    assert sample_size["precision_targets"] == {
        "H1_simultaneous_CI_half_width_max": 0.05,
        "H2_simultaneous_CI_half_width_max": 0.01,
        "H3_simultaneous_CI_half_width_max": 0.01,
    }


def test_ground_truth_policy_and_non_claims() -> None:
    plan = _package().plan
    ground_truth = next(
        item for item in plan["provenance"]["inputs"] if item["key"] == "m8c_ground_truth"
    )
    assert "hash-check only" in ground_truth["role"]
    assert "text never read" in ground_truth["role"]
    assert plan["scope"]["m8c_can_be_confirmatory"] is False
    assert plan["scope"]["m9a_associations_are_confirmatory"] is False
    assert plan["next_milestone"]["m9c_is_automatic"] is False


def test_hypothesis_matrix_has_support_falsify_and_followup_columns() -> None:
    path = REPO_ROOT / protocol.PROTOCOL_DIRECTORY / protocol.MATRIX_FILENAME
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames is not None
    for column in (
        "support_rule",
        "falsify_rule",
        "inconclusive_rule",
        "followup_if_supported",
        "followup_if_falsified",
        "followup_if_inconclusive",
    ):
        assert column in reader.fieldnames
    for row in rows:
        assert row["support_rule"]
        assert row["falsify_rule"]
        assert row["inconclusive_rule"]
        assert row["followup_if_supported"]
        assert row["followup_if_falsified"]
        assert row["followup_if_inconclusive"]
