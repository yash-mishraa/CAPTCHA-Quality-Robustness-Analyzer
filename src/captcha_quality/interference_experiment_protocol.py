"""M9B preregistered interference experiment protocol validator.

This module validates a static protocol package. It does not run an experiment,
generate a benchmark, load a model, fit, calibrate, select, predict or add a
feature. Its only command is ``verify``.

The protocol itself lives in
``reports/interference_experiment_protocol_v0_1_0``. The validator checks:

* required package membership;
* the machine-readable schema and required fields;
* unique hypothesis identifiers;
* allowed decision states;
* consistency between the hypothesis matrix and the statistical plan;
* the exploratory labelling of every M8C-derived analysis;
* provenance hashes for the frozen historical inputs;
* the absence of model-fitting or benchmark-generation entry points.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

PROTOCOL_DIRECTORY: Final[str] = "reports/interference_experiment_protocol_v0_1_0"
PROTOCOL_SCHEMA: Final[str] = "m9b-interference-experiment-protocol-v1"
PLAN_FILENAME: Final[str] = "statistical_analysis_plan.json"
MATRIX_FILENAME: Final[str] = "hypothesis_matrix.csv"
VERIFICATION_FILENAME: Final[str] = "verification_checks.csv"

REQUIRED_FILES: Final[tuple[str, ...]] = (
    "README.md",
    "experiment_protocol.md",
    PLAN_FILENAME,
    MATRIX_FILENAME,
    "data_requirements.md",
    "decision_gates.md",
    VERIFICATION_FILENAME,
)

HYPOTHESIS_IDS: Final[tuple[str, ...]] = ("H1", "H2", "H3")

ALLOWED_DECISION_STATES: Final[tuple[str, ...]] = (
    "not_tested",
    "support",
    "falsify",
    "inconclusive",
    "not_testable",
)

REQUIRED_EXPERIMENT_FIELDS: Final[tuple[str, ...]] = (
    "hypothesis_id",
    "short_name",
    "claim",
    "design_type",
    "data_regime",
    "m8c_role",
    "primary_outcome",
    "residual_convention",
    "independent_variable",
    "primary_estimand",
    "primary_estimand_definition",
    "primary_estimand_units",
    "primary_metric",
    "independent_unit",
    "analysis_unit",
    "reference_condition",
    "splits",
    "support_rule",
    "falsify_rule",
    "inconclusive_rule",
    "secondary_metrics",
    "diagnostic_metrics",
    "main_confounders",
    "what_cannot_be_separated",
    "new_data_required",
    "followup_if_supported",
    "followup_if_falsified",
    "followup_if_inconclusive",
    "current_decision_state",
)

REQUIRED_MATRIX_COLUMNS: Final[tuple[str, ...]] = REQUIRED_EXPERIMENT_FIELDS

MATRIX_PLAN_FIELDS: Final[tuple[str, ...]] = REQUIRED_EXPERIMENT_FIELDS

LIST_EXPERIMENT_FIELDS: Final[frozenset[str]] = frozenset(
    {"secondary_metrics", "diagnostic_metrics", "main_confounders"}
)

REQUIRED_CHECK_COLUMNS: Final[tuple[str, ...]] = (
    "check_id",
    "category",
    "expected",
    "actual",
    "passed",
    "details",
)

FORBIDDEN_IMPORTS: Final[frozenset[str]] = frozenset(
    {
        "joblib",
        "sklearn",
        "PIL",
        "cv2",
        "skimage",
        "subprocess",
        "captcha_quality.benchmark",
        "captcha_quality.corruptions",
        "captcha_quality.dataset",
        "captcha_quality.features",
        "captcha_quality.interference_error_analysis",
        "captcha_quality.matched_font_benchmark",
        "captcha_quality.modelling",
        "captcha_quality.render",
    }
)

FORBIDDEN_CALL_NAMES: Final[frozenset[str]] = frozenset(
    {
        "fit",
        "fit_transform",
        "calibrate",
        "predict",
        "predict_proba",
        "joblib.load",
        "render_clean",
        "generate_dataset",
        "apply_corruption",
        "apply_strokes",
        "run_benchmark",
        "train",
    }
)

EXPECTED_SEED: Final[int] = 20261023
EXPECTED_BOOTSTRAP_REPLICATES: Final[int] = 2000
EXPECTED_PRIMARY_CONFIDENCE: Final[float] = 0.9833333333333333

M9A_ANALYSIS_REVISION: Final[str] = "37bc86dba3cac8d7cfe2d9e84d83da300974866f"
M9A_REPORT_REVISION: Final[str] = "e3d8ed36b497347b87606536000114746d27d709"


class ProtocolError(RuntimeError):
    """Raised when the protocol package cannot be loaded or parsed."""


@dataclass(frozen=True)
class Check:
    """One fail-closed verification result."""

    check_id: str
    category: str
    expected: str
    actual: str
    passed: bool
    details: str = ""

    def as_row(self) -> dict[str, str | bool]:
        return {
            "check_id": self.check_id,
            "category": self.category,
            "expected": self.expected,
            "actual": self.actual,
            "passed": self.passed,
            "details": self.details,
        }


@dataclass(frozen=True)
class ProtocolPackage:
    """The parsed static protocol package."""

    root: Path
    plan: Mapping[str, Any]
    matrix: tuple[dict[str, str], ...]
    verification_checks: tuple[dict[str, str], ...]


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_no_duplicate_pairs,
        )
    except OSError as exc:
        raise ProtocolError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"{path} root must be a JSON object")
    return payload


def _load_csv(path: Path) -> tuple[dict[str, str], ...]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ProtocolError(f"{path} has no header")
            rows = tuple(dict(row) for row in reader)
    except OSError as exc:
        raise ProtocolError(f"cannot read {path}: {exc}") from exc
    return rows


def load_package(repo_root: Path | str) -> ProtocolPackage:
    """Load the static package and fail closed on missing or malformed files."""
    root = Path(repo_root).resolve() / PROTOCOL_DIRECTORY
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise ProtocolError(f"missing protocol file(s): {', '.join(missing)}")
    plan = _load_json(root / PLAN_FILENAME)
    matrix = _load_csv(root / MATRIX_FILENAME)
    checks = _load_csv(root / VERIFICATION_FILENAME)
    return ProtocolPackage(root=root, plan=plan, matrix=matrix, verification_checks=checks)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without interpreting its contents."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(
    check_id: str,
    category: str,
    expected: object,
    actual: object,
    passed: bool,
    details: str = "",
) -> Check:
    return Check(
        check_id=check_id,
        category=category,
        expected=str(expected),
        actual=str(actual),
        passed=bool(passed),
        details=details,
    )


def _nonempty(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_nonempty(item) for item in value)
    return value is not None


def validate_package_membership(package: ProtocolPackage) -> tuple[Check, ...]:
    """Check that the package has exactly the declared file membership."""
    observed = sorted(
        str(path.relative_to(package.root)) for path in package.root.rglob("*") if path.is_file()
    )
    expected = sorted(REQUIRED_FILES)
    return (
        _check(
            "package_membership",
            "package",
            f"{len(expected)} declared files",
            f"{len(observed)} files",
            observed == expected,
            f"missing={sorted(set(expected) - set(observed))}; extra={sorted(set(observed) - set(expected))}",
        ),
    )


def validate_plan_structure(plan: Mapping[str, Any]) -> tuple[Check, ...]:
    """Validate the machine-readable plan schema and required fields."""
    checks: list[Check] = []
    checks.append(
        _check(
            "plan_schema",
            "schema",
            PROTOCOL_SCHEMA,
            plan.get("schema"),
            plan.get("schema") == PROTOCOL_SCHEMA,
        )
    )
    checks.append(
        _check(
            "plan_milestone",
            "scope",
            "M9B",
            plan.get("milestone"),
            plan.get("milestone") == "M9B",
        )
    )
    checks.append(
        _check(
            "plan_status",
            "scope",
            "preregistered_not_executed",
            plan.get("status"),
            plan.get("status") == "preregistered_not_executed",
        )
    )

    experiments = plan.get("experiments")
    if not isinstance(experiments, list):
        checks.append(
            _check(
                "experiments_list",
                "schema",
                "list of experiments",
                type(experiments).__name__,
                False,
            )
        )
        return tuple(checks)

    ids = [
        experiment.get("hypothesis_id")
        for experiment in experiments
        if isinstance(experiment, Mapping)
    ]
    checks.append(
        _check(
            "hypothesis_ids_unique",
            "schema",
            list(HYPOTHESIS_IDS),
            ids,
            len(ids) == len(set(ids)) and set(ids) == set(HYPOTHESIS_IDS),
            f"count={len(ids)}",
        )
    )

    missing_fields: list[str] = []
    empty_fields: list[str] = []
    invalid_states: list[str] = []
    non_exploratory: list[str] = []
    for experiment in experiments:
        if not isinstance(experiment, Mapping):
            missing_fields.append("experiment is not an object")
            continue
        hypothesis_id = str(experiment.get("hypothesis_id", "unknown"))
        missing = [field for field in REQUIRED_EXPERIMENT_FIELDS if field not in experiment]
        if missing:
            missing_fields.append(f"{hypothesis_id}: {', '.join(missing)}")
            continue
        empty = [field for field in REQUIRED_EXPERIMENT_FIELDS if not _nonempty(experiment[field])]
        if empty:
            empty_fields.append(f"{hypothesis_id}: {', '.join(empty)}")
        state = experiment.get("current_decision_state")
        if state not in ALLOWED_DECISION_STATES:
            invalid_states.append(f"{hypothesis_id}={state}")
        if experiment.get("m8c_role") != "exploratory_only":
            non_exploratory.append(str(hypothesis_id))

    checks.append(
        _check(
            "experiment_required_fields",
            "schema",
            f"{len(REQUIRED_EXPERIMENT_FIELDS)} fields per experiment",
            f"missing={len(missing_fields)}; empty={len(empty_fields)}",
            not missing_fields and not empty_fields,
            "; ".join(missing_fields + empty_fields),
        )
    )
    checks.append(
        _check(
            "allowed_decision_states",
            "decision",
            list(ALLOWED_DECISION_STATES),
            sorted({str(experiment.get("current_decision_state")) for experiment in experiments}),
            not invalid_states,
            f"invalid={invalid_states}",
        )
    )
    checks.append(
        _check(
            "m8c_derived_testing_is_exploratory",
            "leakage",
            "every experiment has m8c_role=exploratory_only",
            f"non_exploratory={non_exploratory}",
            not non_exploratory,
        )
    )

    decision_states = plan.get("decision_states")
    allowed = decision_states.get("allowed") if isinstance(decision_states, Mapping) else None
    checks.append(
        _check(
            "decision_state_registry",
            "decision",
            list(ALLOWED_DECISION_STATES),
            allowed,
            isinstance(allowed, list) and set(allowed) == set(ALLOWED_DECISION_STATES),
        )
    )

    scope = plan.get("scope")
    checks.append(
        _check(
            "m8c_is_not_confirmatory",
            "leakage",
            "false",
            {
                "m8c_can_be_confirmatory": scope.get("m8c_can_be_confirmatory")
                if isinstance(scope, Mapping)
                else None,
                "m9a_associations_are_confirmatory": scope.get("m9a_associations_are_confirmatory")
                if isinstance(scope, Mapping)
                else None,
            },
            isinstance(scope, Mapping)
            and scope.get("m8c_can_be_confirmatory") is False
            and scope.get("m9a_associations_are_confirmatory") is False,
        )
    )

    common = plan.get("common_design")
    policy = common.get("statistical_policy") if isinstance(common, Mapping) else None
    if isinstance(policy, Mapping):
        policy_ok = (
            policy.get("resampling_seed") == EXPECTED_SEED
            and policy.get("resampling_replicates") == EXPECTED_BOOTSTRAP_REPLICATES
            and policy.get("primary_confidence_level") == EXPECTED_PRIMARY_CONFIDENCE
        )
        actual_policy = {
            "seed": policy.get("resampling_seed"),
            "replicates": policy.get("resampling_replicates"),
            "primary_confidence": policy.get("primary_confidence_level"),
        }
    else:
        policy_ok = False
        actual_policy = None
    checks.append(
        _check(
            "frozen_statistical_settings",
            "statistics",
            {
                "seed": EXPECTED_SEED,
                "replicates": EXPECTED_BOOTSTRAP_REPLICATES,
                "primary_confidence": EXPECTED_PRIMARY_CONFIDENCE,
            },
            actual_policy,
            policy_ok,
        )
    )

    cross_checks = plan.get("next_milestone")
    checks.append(
        _check(
            "m9c_is_not_automatic",
            "decision",
            "m9c_is_automatic=false",
            cross_checks.get("m9c_is_automatic") if isinstance(cross_checks, Mapping) else None,
            isinstance(cross_checks, Mapping) and cross_checks.get("m9c_is_automatic") is False,
        )
    )

    provenance = plan.get("provenance")
    inputs = provenance.get("inputs") if isinstance(provenance, Mapping) else None
    checks.append(
        _check(
            "provenance_inputs_present",
            "provenance",
            "non-empty list",
            len(inputs) if isinstance(inputs, list) else None,
            isinstance(inputs, list) and bool(inputs),
        )
    )
    return tuple(checks)


def _normalise_matrix_value(value: object) -> str:
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return str(value)


def validate_matrix(
    matrix: Sequence[Mapping[str, str]], plan: Mapping[str, Any]
) -> tuple[Check, ...]:
    """Validate matrix schema, IDs, decision states and plan consistency."""
    checks: list[Check] = []
    if not matrix:
        return (_check("matrix_rows", "schema", "3 hypothesis rows", "0 rows", False),)

    columns = tuple(matrix[0].keys())
    checks.append(
        _check(
            "matrix_columns",
            "schema",
            list(REQUIRED_MATRIX_COLUMNS),
            list(columns),
            set(columns) == set(REQUIRED_MATRIX_COLUMNS),
        )
    )
    checks.append(
        _check(
            "matrix_row_count",
            "schema",
            len(HYPOTHESIS_IDS),
            len(matrix),
            len(matrix) == len(HYPOTHESIS_IDS),
        )
    )

    ids = [row.get("hypothesis_id", "") for row in matrix]
    checks.append(
        _check(
            "matrix_ids_unique",
            "schema",
            list(HYPOTHESIS_IDS),
            ids,
            len(ids) == len(set(ids)) and set(ids) == set(HYPOTHESIS_IDS),
        )
    )

    states = [row.get("current_decision_state", "") for row in matrix]
    checks.append(
        _check(
            "matrix_decision_states",
            "decision",
            list(ALLOWED_DECISION_STATES),
            states,
            all(state in ALLOWED_DECISION_STATES for state in states),
        )
    )
    checks.append(
        _check(
            "matrix_not_preempted",
            "decision",
            "all current states are not_tested",
            states,
            all(state == "not_tested" for state in states),
        )
    )

    experiments = plan.get("experiments")
    plan_by_id = (
        {
            str(experiment["hypothesis_id"]): experiment
            for experiment in experiments
            if isinstance(experiment, Mapping)
        }
        if isinstance(experiments, list)
        else {}
    )
    mismatches: list[str] = []
    for row in matrix:
        hypothesis_id = row.get("hypothesis_id", "")
        experiment = plan_by_id.get(hypothesis_id)
        if experiment is None:
            mismatches.append(f"{hypothesis_id}: absent from plan")
            continue
        for field in MATRIX_PLAN_FIELDS:
            matrix_value = row.get(field, "")
            plan_value = _normalise_matrix_value(experiment.get(field, ""))
            if matrix_value != plan_value:
                mismatches.append(f"{hypothesis_id}.{field}: matrix != plan")
    checks.append(
        _check(
            "matrix_plan_consistency",
            "consistency",
            f"{len(MATRIX_PLAN_FIELDS)} fields for each hypothesis",
            f"{len(mismatches)} mismatch(es)",
            not mismatches,
            "; ".join(mismatches),
        )
    )
    return tuple(checks)


def validate_verification_checks(
    rows: Sequence[Mapping[str, str]],
) -> tuple[Check, ...]:
    """Validate the recorded verification table and require all checks to pass."""
    checks: list[Check] = []
    if not rows:
        return (_check("verification_rows", "verification", "non-empty", "0 rows", False),)
    columns = tuple(rows[0].keys())
    checks.append(
        _check(
            "verification_columns",
            "verification",
            list(REQUIRED_CHECK_COLUMNS),
            list(columns),
            set(columns) == set(REQUIRED_CHECK_COLUMNS),
        )
    )
    ids = [row.get("check_id", "") for row in rows]
    checks.append(
        _check(
            "verification_ids_unique",
            "verification",
            f"{len(rows)} unique check ids",
            f"{len(set(ids))} unique / {len(ids)} rows",
            len(ids) == len(set(ids)),
        )
    )
    failed = [row.get("check_id", "") for row in rows if row.get("passed") != "True"]
    checks.append(
        _check(
            "verification_all_passed",
            "verification",
            "every recorded check passed",
            f"{len(rows) - len(failed)} / {len(rows)} passed",
            not failed,
            f"failed={failed}",
        )
    )
    return tuple(checks)


def validate_provenance(
    repo_root: Path | str,
    plan: Mapping[str, Any],
    *,
    verify_hashes: bool = True,
) -> tuple[Check, ...]:
    """Verify the frozen input hashes and the M9A revision chain."""
    root = Path(repo_root).resolve()
    provenance = plan.get("provenance")
    inputs = provenance.get("inputs") if isinstance(provenance, Mapping) else None
    if not isinstance(inputs, list):
        return (_check("provenance_inputs", "provenance", "list", None, False),)

    if not verify_hashes:
        return (
            _check(
                "provenance_hashes",
                "provenance",
                f"{len(inputs)} SHA-256 values",
                "skipped",
                True,
                "hash verification explicitly disabled",
            ),
        )

    mismatches: list[str] = []
    checked = 0
    for item in inputs:
        if not isinstance(item, Mapping):
            mismatches.append("non-object provenance entry")
            continue
        path = root / str(item.get("path", ""))
        expected = item.get("sha256")
        if not path.is_file():
            mismatches.append(f"{item.get('key')}: missing {path}")
            continue
        observed = sha256_file(path)
        checked += 1
        if observed != expected:
            mismatches.append(f"{item.get('key')}: expected {expected}, observed {observed}")

    revision_ok = (
        isinstance(provenance, Mapping)
        and provenance.get("design_revision") == M9A_REPORT_REVISION
        and provenance.get("origin_main_at_design_start") == M9A_REPORT_REVISION
    )
    predecessor = plan.get("predecessor")
    predecessor_ok = (
        isinstance(predecessor, Mapping)
        and predecessor.get("m9a_analysis_revision") == M9A_ANALYSIS_REVISION
        and predecessor.get("m9a_report_revision") == M9A_REPORT_REVISION
    )
    return (
        _check(
            "provenance_hashes",
            "provenance",
            f"{len(inputs)} SHA-256 values",
            f"{checked}/{len(inputs)} checked",
            not mismatches,
            "; ".join(mismatches),
        ),
        _check(
            "m9a_revision_chain",
            "provenance",
            {
                "design_revision": M9A_REPORT_REVISION,
                "m9a_analysis_revision": M9A_ANALYSIS_REVISION,
                "m9a_report_revision": M9A_REPORT_REVISION,
            },
            {
                "design_revision": provenance.get("design_revision")
                if isinstance(provenance, Mapping)
                else None,
                "m9a_analysis_revision": predecessor.get("m9a_analysis_revision")
                if isinstance(predecessor, Mapping)
                else None,
                "m9a_report_revision": predecessor.get("m9a_report_revision")
                if isinstance(predecessor, Mapping)
                else None,
            },
            revision_ok and predecessor_ok,
        ),
    )


def validate_static_scope() -> tuple[Check, ...]:
    """Fail if this module contains a fitting, prediction or generation path."""
    path = Path(__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden_imports: list[str] = []
    forbidden_calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in FORBIDDEN_IMPORTS:
                    forbidden_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module in FORBIDDEN_IMPORTS:
            forbidden_imports.append(str(node.module))
        elif isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name):
                name = function.id
            elif isinstance(function, ast.Attribute):
                name = function.attr
            else:
                continue
            if name in FORBIDDEN_CALL_NAMES:
                forbidden_calls.append(name)
    return (
        _check(
            "static_scope_no_execution",
            "scope",
            "no fitting, prediction, calibration or generation imports/calls",
            f"forbidden_imports={sorted(set(forbidden_imports))}; forbidden_calls={sorted(set(forbidden_calls))}",
            not forbidden_imports and not forbidden_calls,
        ),
    )


def validate_cli_scope() -> tuple[Check, ...]:
    """Fail if the protocol validator exposes anything but ``verify``."""
    parser = build_parser()
    actions = [
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    ]
    choices = sorted(actions[0].choices) if actions else []
    return (
        _check(
            "cli_only_verify",
            "scope",
            ["verify"],
            choices,
            choices == ["verify"],
        ),
    )


def validate_protocol(repo_root: Path | str, *, verify_hashes: bool = True) -> tuple[Check, ...]:
    """Run all static protocol checks and return the complete check set."""
    package = load_package(repo_root)
    checks: list[Check] = []
    checks.extend(validate_package_membership(package))
    checks.extend(validate_plan_structure(package.plan))
    checks.extend(validate_matrix(package.matrix, package.plan))
    checks.extend(validate_verification_checks(package.verification_checks))
    checks.extend(validate_provenance(repo_root, package.plan, verify_hashes=verify_hashes))
    checks.extend(validate_static_scope())
    checks.extend(validate_cli_scope())
    return tuple(checks)


def format_checks_csv(checks: Sequence[Check]) -> str:
    """Format checks as a CSV table."""
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=REQUIRED_CHECK_COLUMNS,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(check.as_row() for check in checks)
    return buffer.getvalue()


def format_checks_text(checks: Sequence[Check]) -> str:
    """Format checks as a compact human-readable table."""
    lines = [
        "check_id\tcategory\tpassed\texpected\tactual",
    ]
    for check in checks:
        lines.append(
            f"{check.check_id}\t{check.category}\t{check.passed}\t{check.expected}\t{check.actual}"
        )
    passed = sum(1 for check in checks if check.passed)
    lines.append(f"summary\t\t{passed}/{len(checks)} checks passed")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the validator CLI. Only ``verify`` is available."""
    parser = argparse.ArgumentParser(
        prog="captcha-interference-experiment-protocol",
        description=(
            "Validate the static M9B interference experiment protocol. "
            "This command never runs an experiment or fits a model."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify", help="validate the protocol package")
    verify.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="repository root; defaults to the package root",
    )
    verify.add_argument(
        "--no-hash-check",
        action="store_true",
        help="skip historical provenance hash verification (not for final validation)",
    )
    verify.add_argument(
        "--csv",
        action="store_true",
        help="print the checks as CSV instead of the compact text table",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the static protocol validator."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "verify":
        parser.error(f"unknown command: {args.command}")
    repo_root = args.repo_root or Path(__file__).resolve().parents[2]
    try:
        checks = validate_protocol(repo_root, verify_hashes=not args.no_hash_check)
    except ProtocolError as exc:
        print(f"protocol error: {exc}", file=sys.stderr)
        print("verification: FAIL", file=sys.stderr)
        return 1
    if args.csv:
        print(format_checks_csv(checks), end="")
    else:
        print(format_checks_text(checks))
    passed = all(check.passed for check in checks)
    print(f"verification: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
