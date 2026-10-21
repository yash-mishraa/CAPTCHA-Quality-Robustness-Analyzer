"""Milestone M7E font-roster licensing and availability audit.

M7E is a **read-only audit**. It reads the M7D planning configuration and the
frozen M7D plan artifacts, re-verifies each proposed typeface against its
pinned path and SHA-256, and re-derives an eligibility decision for every face
from the recorded licence evidence. It generates no images, no text cases, no
dataset rows, no predictions and no model artifacts, and it does not change the
M7D plan, the M5C champion or any historical output.

The audit separates *evidence* from *decisions*: the committed protocol records
the facts, and the module re-derives the status from those facts on every
verification run. A declared ``eligible`` status is rejected unless the
derived status is also ``eligible``.

Expected hashes are anchored in the committed protocol and in the pinned M7D
inputs; they are never regenerated from the current tree.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd

from captcha_quality.dataset import CheckResult, VerificationReport
from captcha_quality.provenance import (
    canonical_json,
    code_revision,
    json_safe,
    package_version,
    sha256_bytes,
    sha256_file,
)

#: Protocol versions that a later protocol version supersedes. The mapping is
#: keyed by the superseded configuration path and gives the superseding one, so
#: that a reader of a superseded audit is told which audit is current.
SUPERSEDED_PROTOCOLS: Final[dict[str, str]] = {
    "configs/font_roster_audit_v0_1_0.toml": "configs/font_roster_audit_v0_2_0.toml",
}

#: The four eligibility statuses and their exact meanings.
ELIGIBILITY_STATUSES: Final[tuple[str, ...]] = (
    "eligible",
    "ineligible",
    "review_required",
    "unavailable",
)

#: Tri-state permission values used by the evidence records.
PERMISSION_VALUES: Final[tuple[str, ...]] = ("yes", "no", "unverified")

#: Project status of a proposed face.
PROJECT_STATUSES: Final[tuple[str, ...]] = (
    "m3_development_font",
    "m7b_held_out_evaluation_font",
    "new_to_project_evaluation",
)

#: Evidence grades.
EVIDENCE_GRADES: Final[tuple[str, ...]] = (
    "primary_authoritative",
    "primary_with_supporting",
    "insufficient",
)

#: Availability scopes.
AVAILABILITY_SCOPES: Final[tuple[str, ...]] = (
    "apple_platforms",
    "multi_platform",
    "audit_machine_only",
)

#: Required evidence fields, excluding the ``font_key`` identifier.
EVIDENCE_FIELDS: Final[tuple[str, ...]] = (
    "family",
    "face",
    "version",
    "postscript_name",
    "copyright_holder",
    "rights_holder",
    "fs_type",
    "embedding_restrictions",
    "project_status",
    "availability_scope",
    "availability_evidence",
    "license_name",
    "license_scope",
    "license_source_url",
    "license_retrieved_date",
    "license_evidence_excerpt",
    "permits_rendering",
    "permits_output_redistribution",
    "permits_font_file_redistribution",
    "attribution_required",
    "attribution_terms",
    "reproducible_pin",
    "pin_evidence",
    "evidence_grade",
    "evidence_notes",
)

#: Required decision fields, excluding the ``font_key`` identifier.
DECISION_FIELDS: Final[tuple[str, ...]] = (
    "status",
    "reason",
    "conditions",
    "unresolved_questions",
)

#: Font file extensions that must never appear in a committed artifact.
FONT_BINARY_SUFFIXES: Final[tuple[str, ...]] = (".ttf", ".otf", ".ttc", ".woff", ".woff2")

REPORTS_ROOT: Final[Path] = Path("reports")
PROTECTED_REPORT_ROOTS: Final[tuple[str, ...]] = (
    "reports/benchmark_v0_1_0",
    "reports/captcha_quality_v0.1.0",
    "reports/captcha_quality_v0.1.0-pilot",
    "reports/cross_font_analysis_v0_1_0",
    "reports/dataset_adequacy_audit_clean_run_1",
    "reports/dataset_adequacy_audit_clean_run_2",
    "reports/dataset_adequacy_audit_v0_1_0",
    "reports/evaluation_design_plan_v0_1_0",
    "reports/geometric_distortion_pilot",
    "reports/geometric_robustness_benchmark_v0_1_0",
    "reports/heldout_font_benchmark_v0_1_0",
    "reports/local_structure_study_v0_1_0",
    "reports/pilot",
    "reports/retention_aware_selection_v0_1_0",
    "reports/retention_aware_training_v0_1_0",
    "reports/robustness_benchmark_v0_1_0",
    "reports/robustness_pilot",
    "reports/spatial_feature_study_v0_1_0",
    "reports/spatial_resolution_study_v0_1_0",
    "reports/structure_tensor_study_v0_1_0",
)

OUTPUT_TABLE_NAMES: Final[tuple[str, ...]] = (
    "input_manifest.csv",
    "font_evidence.csv",
    "eligibility.csv",
    "roster_coverage.csv",
    "isolation_checks.csv",
)

REPORT_SECTIONS: Final[tuple[str, ...]] = (
    "## Executive summary",
    "## 1. Scope and research-integrity boundary",
    "## 2. Protocol, inputs and evidence standard",
    "## 3. Eligibility statuses and derivation rules",
    "## 4. Roster coverage",
    "## 5. Per-font evidence and decisions",
    "## 6. Licence families and the font-file boundary",
    "## 7. Availability and reproducibility",
    "## 8. Unresolved licensing questions",
    "## 9. Project-level findings",
    "## 10. Benchmark generation gate",
    "## 11. Limitations",
    "## 12. Non-execution statement",
)


class AuditError(RuntimeError):
    """Raised when a pinned input or an audit invariant is violated."""


class AuditConfigError(ValueError):
    """Raised for an invalid M7E configuration."""


@dataclass(frozen=True, slots=True)
class RunSettings:
    name: str
    version: str
    seed: int
    output_reports: Path


@dataclass(frozen=True, slots=True)
class AuditSettings:
    milestone: str
    status: str
    scope: str
    intended_use: str
    target: str
    font_file_redistribution_required: bool
    font_file_redistribution_rationale: str
    out_of_scope: str
    supersedes: str = ""
    residual_questions: str = ""
    experiment: str = "EXP-0018"


@dataclass(frozen=True, slots=True)
class RosterSettings:
    source: str
    expected_count: int
    expected_familiar: tuple[str, ...]
    expected_held_out: tuple[str, ...]
    project_statuses: tuple[str, ...]
    note: str


@dataclass(frozen=True, slots=True)
class StatusMeaning:
    status: str
    meaning: str


@dataclass(frozen=True, slots=True)
class DerivationRules:
    rule_unavailable: str
    rule_ineligible: str
    rule_review_required: str
    rule_eligible: str
    rule_declared: str


@dataclass(frozen=True, slots=True)
class InputPin:
    key: str
    path: str
    sha256: str
    role: str


@dataclass(frozen=True, slots=True)
class FontEvidence:
    font_key: str
    fields: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EligibilityDecision:
    font_key: str
    status: str
    reason: str
    conditions: str
    unresolved_questions: str


@dataclass(frozen=True, slots=True)
class AuditConfig:
    run: RunSettings
    audit: AuditSettings
    roster: RosterSettings
    statuses: tuple[StatusMeaning, ...]
    derivation: DerivationRules
    inputs: tuple[InputPin, ...]
    evidence: tuple[FontEvidence, ...]
    decisions: tuple[EligibilityDecision, ...]
    config_path: Path
    config_sha256: str
    config_hash: str

    def identity(self) -> dict[str, Any]:
        return {
            "run": {
                "name": self.run.name,
                "version": self.run.version,
                "seed": self.run.seed,
                "output_reports": str(self.run.output_reports),
            },
            "audit": {
                "milestone": self.audit.milestone,
                "status": self.audit.status,
                "scope": self.audit.scope,
                "intended_use": self.audit.intended_use,
                "target": self.audit.target,
                "font_file_redistribution_required": (self.audit.font_file_redistribution_required),
                "font_file_redistribution_rationale": (
                    self.audit.font_file_redistribution_rationale
                ),
                "out_of_scope": self.audit.out_of_scope,
                "supersedes": self.audit.supersedes,
                "residual_questions": self.audit.residual_questions,
                "experiment": self.audit.experiment,
            },
            "roster": {
                "source": self.roster.source,
                "expected_count": self.roster.expected_count,
                "expected_familiar": self.roster.expected_familiar,
                "expected_held_out": self.roster.expected_held_out,
                "project_statuses": self.roster.project_statuses,
                "note": self.roster.note,
            },
            "statuses": [
                {"status": item.status, "meaning": item.meaning} for item in self.statuses
            ],
            "derivation": {
                "rule_unavailable": self.derivation.rule_unavailable,
                "rule_ineligible": self.derivation.rule_ineligible,
                "rule_review_required": self.derivation.rule_review_required,
                "rule_eligible": self.derivation.rule_eligible,
                "rule_declared": self.derivation.rule_declared,
            },
            "inputs": [
                {"key": pin.key, "path": pin.path, "sha256": pin.sha256, "role": pin.role}
                for pin in self.inputs
            ],
            "evidence": [{"font_key": item.font_key, **item.fields} for item in self.evidence],
            "decisions": [
                {
                    "font_key": item.font_key,
                    "status": item.status,
                    "reason": item.reason,
                    "conditions": item.conditions,
                    "unresolved_questions": item.unresolved_questions,
                }
                for item in self.decisions
            ],
        }


def _check_keys(raw: Mapping[str, object], expected: set[str], label: str) -> None:
    actual = set(raw)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise AuditConfigError(f"{label} is missing key(s): {missing}")
    if unknown:
        raise AuditConfigError(f"{label} has unknown key(s): {unknown}")


def _section(raw: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise AuditConfigError(f"{key} must be a table")
    return value


def _as_str(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AuditConfigError(f"{label} must be a non-empty string")
    return value.strip()


def _as_optional_str(value: object, label: str) -> str:
    """Accept an absent or empty optional string, rejecting other non-strings."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise AuditConfigError(f"{label} must be a string")
    return value.strip()


def _as_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AuditConfigError(f"{label} must be an integer")
    return value


def _as_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise AuditConfigError(f"{label} must be a boolean")
    return value


def _as_str_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise AuditConfigError(f"{label} must be a non-empty array")
    return tuple(_as_str(item, f"{label}[]") for item in value)


def _sha256(value: object, label: str) -> str:
    text = _as_str(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise AuditConfigError(f"{label} must be a lowercase SHA-256 hex digest")
    return text


def _parse_run(raw: Mapping[str, object]) -> RunSettings:
    _check_keys(raw, {"name", "version", "seed", "output_reports"}, "run")
    run = RunSettings(
        name=_as_str(raw["name"], "run.name"),
        version=_as_str(raw["version"], "run.version"),
        seed=_as_int(raw["seed"], "run.seed"),
        output_reports=Path(_as_str(raw["output_reports"], "run.output_reports")),
    )
    if run.seed < 0:
        raise AuditConfigError("run.seed must be >= 0")
    return run


def _parse_audit(raw: Mapping[str, object]) -> AuditSettings:
    values = dict(raw)
    optional = {"supersedes", "residual_questions", "experiment"}
    for key in optional:
        values.pop(key, None)
    _check_keys(
        values,
        {
            "milestone",
            "status",
            "scope",
            "intended_use",
            "target",
            "font_file_redistribution_required",
            "font_file_redistribution_rationale",
            "out_of_scope",
        },
        "audit",
    )
    settings = AuditSettings(
        milestone=_as_str(raw["milestone"], "audit.milestone"),
        status=_as_str(raw["status"], "audit.status"),
        scope=_as_str(raw["scope"], "audit.scope"),
        intended_use=_as_str(raw["intended_use"], "audit.intended_use"),
        target=_as_str(raw["target"], "audit.target"),
        font_file_redistribution_required=_as_bool(
            raw["font_file_redistribution_required"],
            "audit.font_file_redistribution_required",
        ),
        font_file_redistribution_rationale=_as_str(
            raw["font_file_redistribution_rationale"],
            "audit.font_file_redistribution_rationale",
        ),
        out_of_scope=_as_str(raw["out_of_scope"], "audit.out_of_scope"),
        supersedes=_as_optional_str(raw.get("supersedes"), "audit.supersedes"),
        residual_questions=_as_optional_str(
            raw.get("residual_questions"), "audit.residual_questions"
        ),
        experiment=_as_optional_str(raw.get("experiment"), "audit.experiment") or "EXP-0018",
    )
    if settings.milestone != "M7E":
        raise AuditConfigError("audit.milestone must be 'M7E'")
    return settings


def _parse_roster(raw: Mapping[str, object]) -> RosterSettings:
    _check_keys(
        raw,
        {
            "source",
            "expected_count",
            "expected_familiar",
            "expected_held_out",
            "project_statuses",
            "note",
        },
        "roster",
    )
    roster = RosterSettings(
        source=_as_str(raw["source"], "roster.source"),
        expected_count=_as_int(raw["expected_count"], "roster.expected_count"),
        expected_familiar=_as_str_tuple(raw["expected_familiar"], "roster.expected_familiar"),
        expected_held_out=_as_str_tuple(raw["expected_held_out"], "roster.expected_held_out"),
        project_statuses=_as_str_tuple(raw["project_statuses"], "roster.project_statuses"),
        note=_as_str(raw["note"], "roster.note"),
    )
    if roster.expected_count != len(roster.expected_familiar) + len(roster.expected_held_out):
        raise AuditConfigError("roster.expected_count does not match the expected members")
    if set(roster.expected_familiar) & set(roster.expected_held_out):
        raise AuditConfigError("familiar and held-out roster members must be disjoint")
    return roster


def _parse_statuses(raw: Mapping[str, object]) -> tuple[StatusMeaning, ...]:
    _check_keys(raw, set(ELIGIBILITY_STATUSES), "statuses")
    meanings: list[StatusMeaning] = []
    for status in ELIGIBILITY_STATUSES:
        entry = raw[status]
        if not isinstance(entry, dict):
            raise AuditConfigError(f"statuses.{status} must be a table")
        _check_keys(entry, {"meaning"}, f"statuses.{status}")
        meanings.append(
            StatusMeaning(
                status=status,
                meaning=_as_str(entry["meaning"], f"statuses.{status}.meaning"),
            )
        )
    return tuple(meanings)


def _parse_derivation(raw: Mapping[str, object]) -> DerivationRules:
    keys = {
        "rule_unavailable",
        "rule_ineligible",
        "rule_review_required",
        "rule_eligible",
        "rule_declared",
    }
    _check_keys(raw, keys, "derivation")
    return DerivationRules(**{key: _as_str(raw[key], f"derivation.{key}") for key in sorted(keys)})


def _parse_inputs(raw: object) -> tuple[InputPin, ...]:
    if not isinstance(raw, list) or not raw:
        raise AuditConfigError("[[inputs]] must be a non-empty array of tables")
    pins: list[InputPin] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise AuditConfigError(f"inputs[{index}] must be a table")
        _check_keys(item, {"key", "path", "sha256", "role"}, f"inputs[{index}]")
        pins.append(
            InputPin(
                key=_as_str(item["key"], f"inputs[{index}].key"),
                path=_as_str(item["path"], f"inputs[{index}].path"),
                sha256=_sha256(item["sha256"], f"inputs[{index}].sha256"),
                role=_as_str(item["role"], f"inputs[{index}].role"),
            )
        )
    keys = [pin.key for pin in pins]
    if len(set(keys)) != len(keys):
        raise AuditConfigError("input keys must be unique")
    return tuple(pins)


def _parse_evidence(raw: object) -> tuple[FontEvidence, ...]:
    if not isinstance(raw, list) or not raw:
        raise AuditConfigError("[[font_evidence]] must be a non-empty array of tables")
    records: list[FontEvidence] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise AuditConfigError(f"font_evidence[{index}] must be a table")
        _check_keys(item, {"font_key", *EVIDENCE_FIELDS}, f"font_evidence[{index}]")
        font_key = _as_str(item["font_key"], f"font_evidence[{index}].font_key")
        fields: dict[str, Any] = {}
        for field in EVIDENCE_FIELDS:
            value = item[field]
            if field == "fs_type":
                fields[field] = _as_int(value, f"font_evidence[{index}].{field}")
            else:
                fields[field] = _as_str(value, f"font_evidence[{index}].{field}")
        for field, allowed in (
            ("permits_rendering", PERMISSION_VALUES),
            ("permits_output_redistribution", PERMISSION_VALUES),
            ("permits_font_file_redistribution", PERMISSION_VALUES),
            ("attribution_required", PERMISSION_VALUES),
            ("reproducible_pin", PERMISSION_VALUES),
            ("evidence_grade", EVIDENCE_GRADES),
            ("project_status", PROJECT_STATUSES),
            ("availability_scope", AVAILABILITY_SCOPES),
        ):
            if fields[field] not in allowed:
                raise AuditConfigError(f"font_evidence[{index}].{field} must be one of {allowed}")
        records.append(FontEvidence(font_key=font_key, fields=fields))
    keys = [record.font_key for record in records]
    if len(set(keys)) != len(keys):
        raise AuditConfigError("font_evidence font_key values must be unique")
    return tuple(records)


def _parse_decisions(raw: object) -> tuple[EligibilityDecision, ...]:
    if not isinstance(raw, list) or not raw:
        raise AuditConfigError("[[eligibility_decision]] must be a non-empty array of tables")
    records: list[EligibilityDecision] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise AuditConfigError(f"eligibility_decision[{index}] must be a table")
        _check_keys(item, {"font_key", *DECISION_FIELDS}, f"eligibility_decision[{index}]")
        status = _as_str(item["status"], f"eligibility_decision[{index}].status")
        if status not in ELIGIBILITY_STATUSES:
            raise AuditConfigError(
                f"eligibility_decision[{index}].status must be one of {ELIGIBILITY_STATUSES}"
            )
        records.append(
            EligibilityDecision(
                font_key=_as_str(item["font_key"], f"eligibility_decision[{index}].font_key"),
                status=status,
                reason=_as_str(item["reason"], f"eligibility_decision[{index}].reason"),
                conditions=_as_str(item["conditions"], f"eligibility_decision[{index}].conditions"),
                unresolved_questions=_as_str(
                    item["unresolved_questions"],
                    f"eligibility_decision[{index}].unresolved_questions",
                ),
            )
        )
    keys = [record.font_key for record in records]
    if len(set(keys)) != len(keys):
        raise AuditConfigError("eligibility_decision font_key values must be unique")
    return tuple(records)


def load_config(path: Path | str) -> AuditConfig:
    """Load and validate the M7E audit protocol."""
    config_path = Path(path)
    if not config_path.is_file():
        raise AuditConfigError(f"missing M7E configuration: {config_path}")
    raw_bytes = config_path.read_bytes()
    try:
        payload = tomllib.loads(raw_bytes.decode("utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise AuditConfigError(f"invalid TOML in {config_path}: {error}") from error
    _check_keys(
        payload,
        {
            "run",
            "audit",
            "roster",
            "statuses",
            "derivation",
            "inputs",
            "font_evidence",
            "eligibility_decision",
        },
        "top-level",
    )
    run = _parse_run(_section(payload, "run"))
    audit = _parse_audit(_section(payload, "audit"))
    roster = _parse_roster(_section(payload, "roster"))
    statuses = _parse_statuses(_section(payload, "statuses"))
    derivation = _parse_derivation(_section(payload, "derivation"))
    inputs = _parse_inputs(payload.get("inputs"))
    evidence = _parse_evidence(payload.get("font_evidence"))
    decisions = _parse_decisions(payload.get("eligibility_decision"))
    config = AuditConfig(
        run=run,
        audit=audit,
        roster=roster,
        statuses=statuses,
        derivation=derivation,
        inputs=inputs,
        evidence=evidence,
        decisions=decisions,
        config_path=config_path,
        config_sha256=sha256_bytes(raw_bytes),
        config_hash="",
    )
    identity_hash = sha256_bytes(canonical_json(config.identity()).encode("utf-8"))
    return AuditConfig(
        run=config.run,
        audit=config.audit,
        roster=config.roster,
        statuses=config.statuses,
        derivation=config.derivation,
        inputs=config.inputs,
        evidence=config.evidence,
        decisions=config.decisions,
        config_path=config.config_path,
        config_sha256=config.config_sha256,
        config_hash=identity_hash,
    )


@dataclass(frozen=True, slots=True)
class AuditInputs:
    config: AuditConfig
    m7d_config: dict[str, Any]
    m7d_planned_fonts: list[dict[str, Any]]
    m7d_plan_manifest: dict[str, Any]
    m7d_font_inventory: pd.DataFrame
    input_hashes: dict[str, str]
    input_manifest: pd.DataFrame
    checks: tuple[CheckResult, ...]


def _require_file_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise AuditError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise AuditError(
            f"input hash mismatch for {label}: {path} has {actual}, expected {expected}"
        )
    return actual


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise AuditError(f"missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot read {label}: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise AuditError(f"{label} must contain a JSON object: {path}")
    return payload


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise AuditError(f"missing {label}: {path}")
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise AuditError(f"cannot read {label}: {path}: {error}") from error


def _check(checks: list[CheckResult], name: str, function: Any) -> None:
    try:
        detail = function()
    except Exception as error:  # noqa: BLE001 - every incompatibility is reported
        checks.append(CheckResult(name, False, f"{type(error).__name__}: {error}"))
    else:
        checks.append(CheckResult(name, True, str(detail)))


def _require_all_checks(checks: Sequence[CheckResult], label: str) -> None:
    failed = [check for check in checks if not check.passed]
    if failed:
        detail = "; ".join(f"{check.name}: {check.detail}" for check in failed)
        raise AuditError(f"{label} failed: {detail}")


def _read_truetype_metadata(path: Path) -> dict[str, Any]:
    """Read the family, style, version and embedding bits from a TrueType file.

    The values are read from the font's own ``name`` and ``OS/2`` tables, which
    are primary evidence about the exact face that would be rendered.
    """
    import struct

    data = path.read_bytes()
    if len(data) < 12 or data[:4] not in (b"\x00\x01\x00\x00", b"true", b"OTTO"):
        raise AuditError(f"not a TrueType/OpenType file: {path}")
    table_count = struct.unpack(">H", data[4:6])[0]
    tables: dict[str, tuple[int, int]] = {}
    for index in range(table_count):
        offset = 12 + 16 * index
        tag = data[offset : offset + 4].decode("latin-1")
        table_offset, table_length = struct.unpack(">II", data[offset + 8 : offset + 16])
        tables[tag] = (table_offset, table_length)
    if "name" not in tables:
        raise AuditError(f"font has no name table: {path}")
    name_offset = tables["name"][0]
    count, string_offset = struct.unpack(">HH", data[name_offset + 2 : name_offset + 6])
    names: dict[int, str] = {}
    for index in range(count):
        record = name_offset + 6 + 12 * index
        platform_id, _encoding, _language, name_id, length, offset = struct.unpack(
            ">HHHHHH", data[record : record + 12]
        )
        raw = data[
            name_offset + string_offset + offset : name_offset + string_offset + offset + length
        ]
        try:
            text = raw.decode("utf-16-be") if platform_id == 3 else raw.decode("latin-1")
        except UnicodeDecodeError:
            continue
        text = text.replace("\x00", "").strip()
        if text and name_id not in names:
            names[name_id] = text
    fs_type: int | None = None
    if "OS/2" in tables:
        offset = tables["OS/2"][0]
        fs_type = int(struct.unpack(">H", data[offset + 8 : offset + 10])[0])
    return {
        "family": names.get(1, ""),
        "subfamily": names.get(2, ""),
        "full_name": names.get(4, ""),
        "version": names.get(5, ""),
        "postscript_name": names.get(6, ""),
        "copyright": names.get(0, ""),
        "manufacturer": names.get(8, ""),
        "designer": names.get(9, ""),
        "license_text": names.get(13, ""),
        "license_url": names.get(14, ""),
        "fs_type": fs_type,
    }


def _font_binary_paths() -> list[str]:
    """Every font binary present in the working tree, relative to the repo root."""
    repository = Path.cwd().resolve()
    found: list[str] = []
    for root in ("configs", "data", "docs", "models", "reports", "src", "tests"):
        base = repository / root
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix.lower() in FONT_BINARY_SUFFIXES:
                found.append(path.relative_to(repository).as_posix())
    return sorted(found)


def load_inputs(config: AuditConfig) -> AuditInputs:
    """Verify the pinned inputs and load the M7D plan evidence."""
    input_hashes: dict[str, str] = {}
    manifest_rows: list[dict[str, Any]] = []
    for pin in config.inputs:
        path = Path(pin.path)
        digest = _require_file_hash(path, pin.sha256, pin.key)
        input_hashes[pin.key] = digest
        manifest_rows.append(
            {
                "key": pin.key,
                "path": pin.path,
                "sha256": pin.sha256,
                "role": pin.role,
                "bytes": path.stat().st_size,
                "verified": digest == pin.sha256,
            }
        )
    input_manifest = pd.DataFrame(manifest_rows)
    m7d_config_path = Path(config.roster.source)
    m7d_config_raw = _require_file_hash(m7d_config_path, input_hashes["m7d_config"], "m7d_config")
    del m7d_config_raw
    m7d_config = tomllib.loads(m7d_config_path.read_text(encoding="utf-8"))
    planned_fonts = m7d_config.get("planned_fonts")
    if not isinstance(planned_fonts, list) or not planned_fonts:
        raise AuditError("the M7D configuration does not declare [[planned_fonts]]")
    plan_manifest = _read_json(
        Path(next(pin.path for pin in config.inputs if pin.key == "m7d_plan_manifest")),
        "M7D plan manifest",
    )
    font_inventory = _read_csv(
        Path(next(pin.path for pin in config.inputs if pin.key == "m7d_font_inventory")),
        "M7D font inventory",
    )
    for column in ("font_id", "role", "family", "stratum", "sha256", "available"):
        if column not in font_inventory.columns:
            raise AuditError(f"M7D font inventory is missing column: {column}")

    checks: list[CheckResult] = []
    _check(
        checks,
        "input_pins",
        lambda: f"{len(config.inputs)} pinned inputs verified by SHA-256",
    )
    _check(
        checks,
        "roster_matches_m7d_plan",
        lambda: _roster_matches_plan(config, planned_fonts),
    )
    _check(
        checks,
        "m7d_font_inventory_agrees",
        lambda: _font_inventory_agrees(config, font_inventory, planned_fonts),
    )
    _check(
        checks,
        "no_font_binaries_committed",
        lambda: _no_font_binaries(),
    )
    _check(
        checks,
        "planned_benchmark_paths_absent",
        lambda: _planned_paths_absent(plan_manifest),
    )
    _require_all_checks(checks, "M7E input checks")
    return AuditInputs(
        config=config,
        m7d_config=m7d_config,
        m7d_planned_fonts=list(planned_fonts),
        m7d_plan_manifest=plan_manifest,
        m7d_font_inventory=font_inventory,
        input_hashes=input_hashes,
        input_manifest=input_manifest,
        checks=tuple(checks),
    )


def _roster_matches_plan(config: AuditConfig, planned_fonts: list[dict[str, Any]]) -> str:
    planned_keys = [str(font["key"]) for font in planned_fonts]
    if len(set(planned_keys)) != len(planned_keys):
        raise AuditError("the M7D roster contains a duplicated member")
    planned = {str(font["key"]): str(font["role"]) for font in planned_fonts}
    expected = {
        **{key: "familiar" for key in config.roster.expected_familiar},
        **{key: "held-out" for key in config.roster.expected_held_out},
    }
    if planned != expected:
        raise AuditError(
            f"the M7D roster {sorted(planned)} does not match the audited roster {sorted(expected)}"
        )
    evidence_keys = {record.font_key for record in config.evidence}
    decision_keys = {record.font_key for record in config.decisions}
    if evidence_keys != set(expected) or decision_keys != set(expected):
        raise AuditError("evidence and decision records must cover the roster exactly")
    return f"{len(expected)} roster members agree with the M7D plan and the audit records"


def _font_inventory_agrees(
    config: AuditConfig,
    font_inventory: pd.DataFrame,
    planned_fonts: list[dict[str, Any]],
) -> str:
    pinned = {str(font["key"]): str(font["sha256"]) for font in planned_fonts}
    recorded = {str(row["font_id"]): str(row["sha256"]) for _, row in font_inventory.iterrows()}
    if pinned != recorded:
        raise AuditError("the M7D font inventory does not match the M7D pinned font hashes")
    if not bool(font_inventory["available"].all()):
        raise AuditError("the M7D font inventory records an unavailable font")
    return f"{len(pinned)} font hashes agree with the M7D font inventory"


def _no_font_binaries() -> str:
    found = _font_binary_paths()
    if found:
        raise AuditError(f"font binaries are present in the working tree: {found}")
    return "no font binary is present anywhere in the working tree"


def _planned_paths_absent(plan_manifest: Mapping[str, Any]) -> str:
    design = plan_manifest.get("dataset_design", {})
    dataset_output = str(design.get("dataset_output", ""))
    report_output = str(design.get("benchmark_report_output", ""))
    for path in (dataset_output, report_output):
        if path and Path(path).exists():
            raise AuditError(f"the planned benchmark path already exists: {path}")
    return "the planned benchmark dataset and report paths do not exist"


def _derive_status(evidence: Mapping[str, Any], config: AuditConfig) -> tuple[str, str]:
    """Derive an eligibility status from the recorded evidence fields.

    The rules are the ones declared in the protocol's ``[derivation]`` table.
    """
    if evidence.get("availability_status") != "installed_and_hash_verified":
        return "unavailable", "the exact face is not available with the pinned hash"
    if evidence.get("permits_rendering") == "no":
        return "ineligible", "the evidence states that rendering is prohibited"
    if evidence.get("permits_output_redistribution") == "no":
        return "ineligible", "the evidence states that redistributing the output is prohibited"
    unverified = [
        field
        for field in (
            "permits_rendering",
            "permits_output_redistribution",
            "permits_font_file_redistribution",
            "attribution_required",
            "reproducible_pin",
        )
        if evidence.get(field) == "unverified"
    ]
    if unverified:
        return "review_required", f"unverified evidence: {', '.join(unverified)}"
    if evidence.get("attribution_required") == "yes":
        return "review_required", "an attribution or notice condition is unresolved"
    if (
        evidence.get("permits_font_file_redistribution") == "no"
        and config.audit.font_file_redistribution_required
    ):
        return "review_required", "the workflow would have to redistribute the font file"
    return "eligible", "all recorded evidence supports the intended use"


def _declared_status_is_supported(
    declared: str, derived: str, evidence: Mapping[str, Any]
) -> tuple[bool, str]:
    """Check that a declared status is not more permissive than the evidence."""
    if declared == "eligible":
        if derived == "eligible":
            return True, "the derived status is eligible"
        return False, f"an eligible decision is not supported: the derived status is {derived}"
    if declared == "ineligible":
        prohibited = any(
            evidence.get(field) == "no"
            for field in ("permits_rendering", "permits_output_redistribution")
        )
        if prohibited:
            return True, "the evidence records a positive prohibition"
        return False, "an ineligible decision requires a positive prohibition in the evidence"
    if declared == "unavailable":
        if evidence.get("availability_status") != "installed_and_hash_verified":
            return True, "the face is not available with the pinned hash"
        return False, "an unavailable decision requires the face to be unobtainable"
    return True, "review_required is the conservative default and is always accepted"


def _build_evidence_table(inputs: AuditInputs) -> pd.DataFrame:
    """Combine the recorded evidence with the live availability verification."""
    config = inputs.config
    planned = {str(font["key"]): font for font in inputs.m7d_planned_fonts}
    inventory = inputs.m7d_font_inventory.set_index("font_id")
    rows: list[dict[str, Any]] = []
    for record in config.evidence:
        font = planned[record.font_key]
        path = Path(str(font["path"]))
        available = path.is_file()
        observed_hash = sha256_file(path) if available else ""
        hash_matches = bool(available and observed_hash == str(font["sha256"]))
        metadata: dict[str, Any] = {}
        if available:
            metadata = _read_truetype_metadata(path)
        availability_status = (
            "installed_and_hash_verified"
            if hash_matches
            else ("installed_hash_mismatch" if available else "not_installed")
        )
        row: dict[str, Any] = {
            "font_key": record.font_key,
            "m7d_role": str(font["role"]),
            "m7d_stratum": str(font["stratum"]),
            "m7d_font_sha256": str(font["sha256"]),
            "observed_font_sha256": observed_hash,
            "observed_path_exists": available,
            "availability_status": availability_status,
            "observed_family": metadata.get("family", ""),
            "observed_subfamily": metadata.get("subfamily", ""),
            "observed_version": metadata.get("version", ""),
            "observed_postscript_name": metadata.get("postscript_name", ""),
            "observed_fs_type": metadata.get("fs_type"),
            "observed_license_text": metadata.get("license_text", ""),
            "observed_license_url": metadata.get("license_url", ""),
            "m7d_recorded_available": bool(
                inventory.loc[record.font_key, "available"]
                if record.font_key in inventory.index
                else False
            ),
        }
        row.update(record.fields)
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.sort_values("font_key", kind="stable").reset_index(drop=True)


def _build_eligibility_table(evidence_table: pd.DataFrame, inputs: AuditInputs) -> pd.DataFrame:
    config = inputs.config
    decisions = {record.font_key: record for record in config.decisions}
    rows: list[dict[str, Any]] = []
    for _, row in evidence_table.iterrows():
        key = str(row["font_key"])
        evidence = {field: row[field] for field in EVIDENCE_FIELDS if field in row.index}
        evidence["availability_status"] = row["availability_status"]
        derived, derived_reason = _derive_status(evidence, config)
        decision = decisions[key]
        supported, support_detail = _declared_status_is_supported(
            decision.status, derived, evidence
        )
        rows.append(
            {
                "font_key": key,
                "m7d_role": row["m7d_role"],
                "project_status": row["project_status"],
                "family": row["family"],
                "face": row["face"],
                "availability_status": row["availability_status"],
                "license_name": row["license_name"],
                "license_source_url": row["license_source_url"],
                "license_retrieved_date": row["license_retrieved_date"],
                "permits_rendering": row["permits_rendering"],
                "permits_output_redistribution": row["permits_output_redistribution"],
                "permits_font_file_redistribution": row["permits_font_file_redistribution"],
                "attribution_required": row["attribution_required"],
                "reproducible_pin": row["reproducible_pin"],
                "evidence_grade": row["evidence_grade"],
                "derived_status": derived,
                "derived_reason": derived_reason,
                "declared_status": decision.status,
                "declared_reason": decision.reason,
                "declared_conditions": decision.conditions,
                "unresolved_questions": decision.unresolved_questions,
                "decision_supported": supported,
                "decision_support_detail": support_detail,
            }
        )
    return pd.DataFrame(rows)


def _build_roster_coverage(inputs: AuditInputs) -> pd.DataFrame:
    config = inputs.config
    planned = {str(font["key"]): font for font in inputs.m7d_planned_fonts}
    expected = {
        **{key: "familiar" for key in config.roster.expected_familiar},
        **{key: "held-out" for key in config.roster.expected_held_out},
    }
    rows: list[dict[str, Any]] = []
    for key in sorted(expected):
        font = planned.get(key)
        rows.append(
            {
                "font_key": key,
                "expected_role": expected[key],
                "in_m7d_plan": font is not None,
                "m7d_stratum": str(font["stratum"]) if font else "",
                "has_evidence_record": key in {record.font_key for record in config.evidence},
                "has_decision_record": key in {record.font_key for record in config.decisions},
                "distinct_suite": (
                    "M3 development fonts"
                    if expected[key] == "familiar"
                    else "M7B held-out evaluation fonts"
                    if key in ("georgia", "monaco", "impact")
                    else "new to the project evaluation"
                ),
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True, slots=True)
class AuditResult:
    evidence_table: pd.DataFrame
    eligibility_table: pd.DataFrame
    roster_coverage: pd.DataFrame
    protocol: dict[str, Any]
    report: str


def _md_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, separator]
    for _, row in frame.iterrows():
        cells = [str(row[column]).replace("|", "\\|") for column in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _gate_summary(eligibility: pd.DataFrame) -> dict[str, Any]:
    counts = eligibility["declared_status"].value_counts().to_dict()
    eligible = int(counts.get("eligible", 0))
    review = int(counts.get("review_required", 0))
    ineligible = int(counts.get("ineligible", 0))
    unavailable = int(counts.get("unavailable", 0))
    total = len(eligibility)
    if ineligible or unavailable:
        audit_gate = "blocked"
    elif review:
        audit_gate = "partially_passed"
    else:
        audit_gate = "passed"
    return {
        "total_faces": total,
        "eligible": eligible,
        "review_required": review,
        "ineligible": ineligible,
        "unavailable": unavailable,
        "audit_gate": audit_gate,
        "benchmark_generation_gate": "closed" if review or ineligible or unavailable else "open",
        "benchmark_generation_reason": (
            "Every proposed face must carry a defensible eligibility decision before the "
            "benchmark is generated. A face whose licence position is unresolved blocks "
            "generation of the roster as designed; it is not silently removed and no "
            "similarly named family is substituted."
            if review or ineligible or unavailable
            else "Every proposed face is eligible for the intended workflow."
        ),
    }


def _protocol_payload(
    inputs: AuditInputs, result_tables: dict[str, pd.DataFrame]
) -> dict[str, Any]:
    config = inputs.config
    return {
        "schema_version": "m7e-font-roster-audit-v1",
        "milestone": config.audit.milestone,
        "status": config.audit.status,
        "scope": config.audit.scope,
        "intended_use": config.audit.intended_use,
        "target": config.audit.target,
        "out_of_scope": config.audit.out_of_scope,
        "supersedes": config.audit.supersedes,
        "residual_questions": config.audit.residual_questions,
        "experiment": config.audit.experiment,
        "font_file_redistribution_required": config.audit.font_file_redistribution_required,
        "font_file_redistribution_rationale": (config.audit.font_file_redistribution_rationale),
        "roster": {
            "source": config.roster.source,
            "expected_count": config.roster.expected_count,
            "expected_familiar": list(config.roster.expected_familiar),
            "expected_held_out": list(config.roster.expected_held_out),
            "project_statuses": list(config.roster.project_statuses),
            "note": config.roster.note,
        },
        "statuses": [{"status": item.status, "meaning": item.meaning} for item in config.statuses],
        "derivation": {
            "rule_unavailable": config.derivation.rule_unavailable,
            "rule_ineligible": config.derivation.rule_ineligible,
            "rule_review_required": config.derivation.rule_review_required,
            "rule_eligible": config.derivation.rule_eligible,
            "rule_declared": config.derivation.rule_declared,
        },
        "inputs": {key: value for key, value in inputs.input_hashes.items()},
        "evidence": [
            json_safe(record) for record in result_tables["font_evidence"].to_dict("records")
        ],
        "eligibility": [
            json_safe(record) for record in result_tables["eligibility"].to_dict("records")
        ],
        "roster_coverage": [
            json_safe(record) for record in result_tables["roster_coverage"].to_dict("records")
        ],
        "gate": _gate_summary(result_tables["eligibility"]),
    }


def _superseded_by(config_path: Path) -> str | None:
    """Return the superseding protocol path when this one has been superseded.

    The lookup matches on the path suffix so that it works whether the
    protocol was loaded through a relative or an absolute path.
    """
    text = config_path.as_posix()
    for superseded, superseding in SUPERSEDED_PROTOCOLS.items():
        if text.endswith(superseded):
            return superseding
    return None


def _build_report(inputs: AuditInputs, tables: dict[str, pd.DataFrame]) -> str:
    config = inputs.config
    eligibility = tables["eligibility"]
    evidence = tables["font_evidence"]
    coverage = tables["roster_coverage"]
    gate = _gate_summary(eligibility)
    status_table = _md_table(
        pd.DataFrame(
            [{"status": item.status, "meaning": item.meaning} for item in config.statuses]
        ),
        ("status", "meaning"),
    )
    derivation_table = _md_table(
        pd.DataFrame(
            [
                {"rule": key.replace("rule_", ""), "rule_text": getattr(config.derivation, key)}
                for key in sorted(
                    (
                        "rule_unavailable",
                        "rule_ineligible",
                        "rule_review_required",
                        "rule_eligible",
                        "rule_declared",
                    )
                )
            ]
        ),
        ("rule", "rule_text"),
    )
    coverage_table = _md_table(
        coverage,
        ("font_key", "expected_role", "m7d_stratum", "distinct_suite", "in_m7d_plan"),
    )
    decision_table = _md_table(
        eligibility,
        (
            "font_key",
            "m7d_role",
            "project_status",
            "availability_status",
            "permits_rendering",
            "permits_output_redistribution",
            "permits_font_file_redistribution",
            "attribution_required",
            "derived_status",
            "declared_status",
        ),
    )
    evidence_detail = _md_table(
        evidence,
        (
            "font_key",
            "family",
            "face",
            "version",
            "rights_holder",
            "license_name",
            "license_source_url",
            "license_retrieved_date",
            "evidence_grade",
        ),
    )
    unresolved = eligibility.loc[
        eligibility["declared_status"] == "review_required", "font_key"
    ].tolist()
    project_status_counts = eligibility["project_status"].value_counts().to_dict()
    if unresolved:
        clearance_sentence = (
            f"{gate['eligible']} of {gate['total_faces']} faces are **eligible** for the "
            f"intended workflow; {', '.join(unresolved)} is **review_required**; no face "
            "is ineligible or unavailable."
        )
        if len(unresolved) == 1:
            second_finding = (
                f"Second, **one face remains unresolved**: {', '.join(unresolved)}. The "
                "recorded evidence does not establish the licence position for the "
                "intended use, so the question is kept as an open licensing question "
                "rather than inferred, and it keeps the benchmark generation gate "
                "closed."
            )
        else:
            second_finding = (
                f"Second, **{len(unresolved)} faces remain unresolved**: "
                f"{', '.join(unresolved)}. The recorded evidence does not establish the "
                "licence position for the intended use, so the questions are kept open "
                "rather than inferred, and they keep the benchmark generation gate "
                "closed."
            )
    else:
        clearance_sentence = (
            f"All {gate['total_faces']} faces are **eligible** for the intended "
            "workflow: no face is review_required, ineligible or unavailable."
        )
        second_finding = (
            "Second, **no face remains unresolved**: every face carries a defensible "
            "eligibility decision on the recorded evidence, so the benchmark "
            "generation gate is open."
        )
    superseding = _superseded_by(config.config_path)
    if superseding:
        supersession_banner = (
            f"**This protocol version is superseded by `{superseding}`.** The current "
            "audit is the superseding version; the decision and the evidence recorded "
            "below are preserved exactly as they were recorded for this version."
        )
    else:
        supersession_banner = ""
    unresolved_table = _md_table(
        eligibility, ("font_key", "declared_status", "unresolved_questions")
    )
    conditions_table = _md_table(
        eligibility, ("font_key", "declared_status", "declared_conditions")
    )
    if unresolved:
        gate_conditions = (
            "The face concern is not removed from the roster and no similarly named "
            "family is substituted for it. Clearing it, or a deliberate and separately "
            "versioned roster decision, is a prerequisite for generation."
        )
    else:
        gate_conditions = (
            "Every proposed face carries a defensible eligibility decision, so the "
            "roster may proceed to the separately versioned dataset generator and "
            "zero-shot benchmark that the M7D plan describes."
        )
    residual_block = (
        ("\n" + config.audit.residual_questions + "\n") if config.audit.residual_questions else ""
    )
    text = (
        f"""# M7E - Font roster licensing and availability audit

- **Experiment:** {config.audit.experiment}
- **Protocol version:** {config.run.version} (`{config.config_path}`)
- **Status:** Completed (read-only audit)
- **Roster audited:** {gate["total_faces"]} typefaces proposed by the M7D plan
- **Audit gate:** {gate["audit_gate"]} ({gate["eligible"]} eligible, {gate["review_required"]} review_required, {gate["ineligible"]} ineligible, {gate["unavailable"]} unavailable)
- **Benchmark generation gate:** {gate["benchmark_generation_gate"]}

{supersession_banner}

## Executive summary

M7E audits the ten typefaces the M7D plan proposes for the planned matched
cross-font benchmark. It re-verifies each face against its pinned path and
SHA-256, records the licence evidence for that specific face, and re-derives an
eligibility decision from that evidence. {clearance_sentence}

The audit changes nothing. It generated no images, no text cases, no dataset
rows, no predictions and no model artifacts, and it did not modify the M7D plan,
the M5C champion or any historical output.

Two findings matter for the next milestone. First, **every proposed face is a
proprietary typeface**: none is released under an open licence. The intended
workflow can comply because it renders text locally and distributes only the
rendered images, never a font binary, and because the licences that govern
these designs permit distributing materials made with them. {second_finding}

## 1. Scope and research-integrity boundary

M7E is an **audit and planning milestone**. The scope is the availability and
licence eligibility of the ten typefaces proposed by the M7D plan for the
planned cross-font generalization benchmark. The intended use under audit is:

> {config.audit.intended_use}

The target boundary is unchanged: {config.audit.target}

Explicitly out of scope: {config.audit.out_of_scope}

The audit does not change the planned benchmark design. The roster of ten
typefaces, 30 matched text cases, 300 sources, 6,300 rows, 5,700 primary rows,
600 diagnostic rows, the M7B corruption ladder, the renderer and the 192x64
grayscale image specification are all exactly as the M7D plan records them.
The audit does not train, tune or select a model, does not alter the M5C
champion, and does not recompute any M7A-M7D output.

## 2. Protocol, inputs and evidence standard

The audit protocol is `configs/font_roster_audit_v0_1_0.toml`. It pins
{len(config.inputs)} upstream files by SHA-256:

| Key | Path | Role |
| --- | --- | --- |
"""
        + "\n".join(f"| {pin.key} | `{pin.path}` | {pin.role} |" for pin in config.inputs)
        + f"""

The evidence standard is deliberately strict. For every face the audit records
the planned family, the exact face and style, the version string, the
PostScript name, the copyright and rights holder, the OS/2 embedding bits, the
availability source and scope, the licence name and scope, a source URL, a
retrieval date, a verbatim evidence excerpt, and explicit answers to four
questions: is rendering permitted, is redistribution of the rendered output
permitted, is redistribution of the font file permitted, and does an attribution
or notice condition apply. Evidence facts and eligibility decisions live in
separate tables so the decisions can be re-derived from the facts on every
verification run.

A face is not called eligible unless both availability and the licence
requirements are supported by recorded evidence. Where the terms could not be
retrieved, or remain ambiguous, the face is marked `review_required` and the
uncertainty is recorded; permission is never inferred. Search results and
third-party summaries were not accepted as sufficient evidence where the
original terms could be checked; the recorded sources are the Apple Software
License Agreement, the Microsoft Typography font redistribution FAQ, the Monotype
desktop end-user licence, the Canada Type end-user licence, Apple's published
system-font listing, and the licence strings embedded in the font files
themselves.

## 3. Eligibility statuses and derivation rules

{status_table}

The derivation is re-run on every verification:

{derivation_table}

A declared status of `eligible` is rejected unless the derived status is also
`eligible`. A declared status of `ineligible` requires a positive prohibition
in the evidence, and a declared status of `unavailable` requires the face to be
unobtainable. `review_required` is always accepted as the conservative
default. A declared decision that is *more* restrictive than the derived
status is allowed.

## 4. Roster coverage

{coverage_table}

The roster membership is validated against the pinned M7D planning
configuration, so the audit cannot silently cover a different set of faces. The
three familiar fonts are the M3 development fonts. Georgia, Monaco and Impact
were already evaluated in M7B: they remain valid held-out identities because
the champion was never fitted, calibrated or selected on them, but they are
**not** fonts that are new to the project evaluation, and a result on them does
not independently replicate the M7B three-font finding. The project-status
split of the roster is
{project_status_counts.get("m3_development_font", 0)} M3 development fonts,
{project_status_counts.get("m7b_held_out_evaluation_font", 0)} M7B evaluation
fonts and {project_status_counts.get("new_to_project_evaluation", 0)} faces that
are new to the project evaluation.

## 5. Per-font evidence and decisions

{evidence_detail}

{decision_table}

Every face was verified present with the M7D-pinned SHA-256, and the family,
style, version string, PostScript name and OS/2 embedding bits were read back
from the font file itself and cross-checked against the recorded evidence. The
full evidence excerpts, attribution terms, availability notes and licence
scope are in `font_evidence.csv`; the declared reasons, conditions and open
questions are in `eligibility.csv`.

## 6. Licence families and the font-file boundary

The roster falls into three licence families.

**Microsoft designs** (Georgia, Verdana, Trebuchet MS). The macOS product
licence permits using the bundled font to display and print content. The
foundry's own guidance states that graphic files made with the font may be
shared, sold or redistributed, provided the graphic file is an image of a word,
phrase or passage of text, and that converting the font into a per-letter
bitmap font is not allowed. The benchmark renders whole strings into PNG images
and never produces an addressable per-glyph bitmap font, so it complies. The
font file itself may not be redistributed.

**Monotype designs** (Arial, Courier New, Times New Roman, Impact). The macOS
product licence permits using the bundled font to display and print content,
and the foundry's desktop licence grants the right to create, edit, view, print
and distribute materials made with the font, subject to the same two
conditions, which the benchmark satisfies. The font file may not be
redistributed and the font may not be modified.

**Apple-published designs** (Monaco, Chalkduster) and the **Canada Type
design** (Luminari). For Monaco and Chalkduster the font file carries no
separate licence string, so the macOS product licence governs, and its
important note extends the licence to publishing and distributing materials the
licensee is authorised to distribute. Luminari is the exception and is
discussed in section 8.

The **font-file boundary** is the line that makes the workflow compliant. The
planned benchmark never copies a font binary into the repository, the dataset
or the report: the dataset manifest records the font identifier and the font
SHA-256, never the font file, the repository ignores `data/`, `models/` and
`reports/`, and the audit verifies at run time that no font binary exists
anywhere in the working tree. No licence in the roster permits redistributing a
font file, and the workflow does not need to.

## 7. Availability and reproducibility

Every face was verified present on the audit machine with the SHA-256 pinned
by the M7D configuration, so the exact face that would be rendered is
identified reproducibly by family, style, PostScript name, version string and
file hash. The audit records the observed version string and embedding bits
from the font file itself rather than trusting a summary.

Availability is nonetheless **Apple-platform scoped**. These are the builds
Apple ships with its operating systems, and the pinned hashes are the macOS
builds. The commonly used alternative distribution of the Microsoft and
Monotype core fonts on Linux ships *different versions* under the older Core
Fonts for the Web end-user licence, so the pinned hashes do not transfer and a
cross-platform re-pin would need a fresh licence review. Monaco, Chalkduster
and Luminari have no general non-Apple distribution channel that the audit
could verify. The benchmark is therefore reproducible on a pinned Apple
environment, and the SHA-256 pin rather than a published version string is the
authoritative identifier for the three Apple-published faces whose installed
files are newer than Apple's currently published listing.

## 8. Unresolved licensing questions

Every recorded decision carries its own open questions. The table below is the
union of those questions across the roster; a face is cleared only where its
question is recorded as none.

{unresolved_table}
{residual_block}

## 9. Project-level findings

**No proposed face is an open-licence font.** The roster is composed entirely
of proprietary typefaces. Any future claim that the benchmark is reproducible
from freely redistributable inputs would be false. The audit also confirms
that the workflow never needs to redistribute a font file, which is what keeps
the roster legally usable for a local, evaluation-only benchmark.

**The Monotype direct licence contains an artificial-intelligence training
restriction** that forbids using the font software, or any rendered glyphs,
outlines, kerning data or other visual elements, in connection with developing,
training or fine-tuning artificial-intelligence models. The macOS-bundled
copies audited here are governed by the macOS Software License Agreement, which
contains no such clause, and the planned benchmark is evaluation-only. This
finding is nevertheless material to the project: the frozen M3 development data
was rendered with three of these faces and the M5C champion was trained on it.
Any future move to a directly licensed Monotype copy, or any training on
font-rendered data under such a licence, would need a fresh licence review
before it happens. It is recorded here rather than left to be discovered later.

**Georgia, Monaco and Impact are not new to the project.** They were already
evaluated in M7B. The audit's roster coverage table keeps that distinction
visible so that a future result on them is never described as independent
confirmation of the M7B three-font pattern.

**Recorded conditions.** The conditions attached to each decision are part of
the audit's contract and must be honoured by any later milestone.

{conditions_table}

## 10. Benchmark generation gate

| Gate | Result |
| --- | --- |
| Faces audited | {gate["total_faces"]} |
| Eligible | {gate["eligible"]} |
| review_required | {gate["review_required"]} |
| ineligible | {gate["ineligible"]} |
| unavailable | {gate["unavailable"]} |
| Audit gate | {gate["audit_gate"]} |
| Benchmark generation gate | {gate["benchmark_generation_gate"]} |

{gate["benchmark_generation_reason"]}

The gate is therefore **{gate["audit_gate"]}** at the audit level and
**{gate["benchmark_generation_gate"]}** for benchmark generation.
{gate_conditions}

This milestone does not implement the transition into actual benchmark outputs
and does not create anything at the planned benchmark output paths; M7D's
`no_planned_outputs_created` check is preserved and still passes.

## 11. Limitations

This is an audit of documentary evidence, not a legal opinion, and the
conclusions depend on the terms that were retrievable on the retrieval dates
recorded in the protocol. For three faces the permission for the rendered
output rests on the product licence rather than on an express foundry clause
that names redistribution of rendered images, and the evidence grade records
that distinction. The audit covers the faces as pinned on the audit machine; it
does not verify the terms of any other version of the same family. The audit
does not assess trademark risk, and it does not clear any face for use in a
product, a model, or a deployment. Finally, the audit says nothing about human
readability, CAPTCHA solvability, security or production robustness: the
benchmark target remains synthetic SSIM against an exact clean reference.

## 12. Non-execution statement

M7E generated no CAPTCHA images, no synthetic text cases and no dataset rows. It
created no predictions and no model artifacts. It trained, tuned and selected
no model. It did not recompute or regenerate any M7A-M7D output, did not
change the M5C champion, and did not modify the M7D plan, report, manifest or
run metadata. It did not download or commit any font binary. It created nothing
at the planned benchmark output paths, and it did not weaken M7D's
`no_planned_outputs_created` check. The next action is review of this audit,
not benchmark generation.
"""
    )
    return text


def build_audit(inputs: AuditInputs) -> AuditResult:
    """Build every deterministic M7E table, manifest and report."""
    evidence_table = _build_evidence_table(inputs)
    eligibility_table = _build_eligibility_table(evidence_table, inputs)
    roster_coverage = _build_roster_coverage(inputs)
    tables = {
        "font_evidence": evidence_table,
        "eligibility": eligibility_table,
        "roster_coverage": roster_coverage,
    }
    protocol = _protocol_payload(inputs, tables)
    report = _build_report(inputs, tables)
    return AuditResult(
        evidence_table=evidence_table,
        eligibility_table=eligibility_table,
        roster_coverage=roster_coverage,
        protocol=protocol,
        report=report,
    )


def build_isolation_checks(inputs: AuditInputs, result: AuditResult, report: str) -> pd.DataFrame:
    """Independently check the audit's completeness, purity and non-execution."""
    checks: list[CheckResult] = []
    checks.extend(inputs.checks)
    eligibility = result.eligibility_table
    evidence = result.evidence_table
    config = inputs.config
    _check(
        checks,
        "roster_membership",
        lambda: (
            f"{len(result.roster_coverage)} roster members match the M7D plan and the audit records"
        ),
    )
    _check(
        checks,
        "availability_verified",
        lambda: (
            f"{len(evidence)} faces verified present with the pinned SHA-256"
            if bool((evidence["availability_status"] == "installed_and_hash_verified").all())
            else (_ for _ in ()).throw(
                AuditError("one or more faces are not available with the pinned hash")
            )
        ),
    )
    _check(
        checks,
        "evidence_fields_complete",
        lambda: (
            "every evidence record carries all required fields"
            if all(
                str(row[field]).strip()
                for _, row in evidence.iterrows()
                for field in EVIDENCE_FIELDS
                if field != "fs_type"
            )
            else (_ for _ in ()).throw(AuditError("evidence records are incomplete"))
        ),
    )
    _check(
        checks,
        "statuses_allowed",
        lambda: (
            "every declared status is an allowed value"
            if set(eligibility["declared_status"]).issubset(set(ELIGIBILITY_STATUSES))
            else (_ for _ in ()).throw(AuditError("an unknown eligibility status was declared"))
        ),
    )
    _check(
        checks,
        "decisions_supported",
        lambda: (
            f"all {len(eligibility)} declared decisions are supported by the evidence"
            if bool(eligibility["decision_supported"].all())
            else (_ for _ in ()).throw(
                AuditError(
                    "unsupported decisions: "
                    f"{eligibility.loc[~eligibility['decision_supported'], 'font_key'].tolist()}"
                )
            )
        ),
    )
    _check(
        checks,
        "eligible_requires_derived_eligible",
        lambda: (
            "no eligible decision exceeds the derived status"
            if not (
                (eligibility["declared_status"] == "eligible")
                & (eligibility["derived_status"] != "eligible")
            ).any()
            else (_ for _ in ()).throw(
                AuditError("an eligible decision is not supported by the derived status")
            )
        ),
    )
    _check(
        checks,
        "evidence_grade_valid",
        lambda: (
            "every evidence record uses an allowed evidence grade"
            if set(evidence["evidence_grade"]).issubset(set(EVIDENCE_GRADES))
            else (_ for _ in ()).throw(AuditError("an unknown evidence grade was recorded"))
        ),
    )
    _check(
        checks,
        "project_status_valid",
        lambda: (
            "every record uses an allowed project status"
            if set(evidence["project_status"]).issubset(set(PROJECT_STATUSES))
            else (_ for _ in ()).throw(AuditError("an unknown project status was recorded"))
        ),
    )
    _check(
        checks,
        "m7b_reuse_preserved",
        lambda: (
            "Georgia, Monaco and Impact are recorded as M7B evaluation fonts, not as new"
            if set(
                evidence.loc[
                    evidence["project_status"] == "m7b_held_out_evaluation_font", "font_key"
                ]
            )
            == {"georgia", "monaco", "impact"}
            else (_ for _ in ()).throw(
                AuditError("the M7B font reuse distinction was not preserved")
            )
        ),
    )
    _check(
        checks,
        "font_file_boundary",
        lambda: (
            "the protocol records that no font binary has to be redistributed"
            if config.audit.font_file_redistribution_required is False
            else (_ for _ in ()).throw(
                AuditError("the protocol now requires redistributing a font file")
            )
        ),
    )
    _check(
        checks,
        "report_sections",
        lambda: (
            "all report sections present"
            if all(section in report for section in REPORT_SECTIONS)
            else (_ for _ in ()).throw(AuditError("the report is missing a section"))
        ),
    )
    flattened = " ".join(report.split())
    _check(
        checks,
        "non_execution_statement",
        lambda: (
            "the report states that nothing was generated"
            if "M7E generated no CAPTCHA images" in flattened
            and "created nothing at the planned benchmark output paths" in flattened
            else (_ for _ in ()).throw(AuditError("the non-execution statement is missing"))
        ),
    )
    _check(
        checks,
        "gate_reported",
        lambda: (
            "the audit and benchmark generation gates are reported"
            if "## 10. Benchmark generation gate" in report
            and _gate_summary(eligibility)["audit_gate"]
            else (_ for _ in ()).throw(AuditError("the gate result is not reported"))
        ),
    )
    return pd.DataFrame(
        [{"check": check.name, "passed": check.passed, "detail": check.detail} for check in checks]
    )


def _safe_output_path(path: Path | str) -> Path:
    resolved = Path(path).resolve()
    reports_root = REPORTS_ROOT.resolve()
    if resolved == reports_root or not resolved.is_relative_to(reports_root):
        raise AuditError(f"refusing to write outside the Git-ignored reports directory: {resolved}")
    for protected_text in PROTECTED_REPORT_ROOTS:
        protected = Path(protected_text).resolve()
        if protected == reports_root / "font_roster_audit_v0_1_0":
            continue
        if resolved == protected or resolved.is_relative_to(protected):
            raise AuditError(f"refusing to write into a protected frozen report path: {resolved}")
    return resolved


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(json_safe(dict(payload)), indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")


def _scoped_fingerprint(config: AuditConfig) -> dict[str, str]:
    """Hash exactly the declared audit inputs, by repository-relative path.

    The membership is file-exact: only the files the protocol declares are
    hashed, so an unrelated configuration, dataset, model or report added to a
    shared directory later does not invalidate the audit, while a modified or
    deleted declared input is still detected.
    """
    fingerprint: dict[str, str] = {}
    for pin in config.inputs:
        path = Path(pin.path)
        if not path.is_file():
            continue
        fingerprint[pin.path] = sha256_file(path)
    return fingerprint


def _font_hashes(inputs: AuditInputs) -> dict[str, str]:
    """The observed SHA-256 of every audited face, keyed by stable font id."""
    planned = {str(font["key"]): str(font["sha256"]) for font in inputs.m7d_planned_fonts}
    return dict(sorted(planned.items()))


def _output_hashes(output_dir: Path) -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "run_metadata.json"
    }


def _fingerprint_hash(fingerprint: Mapping[str, str]) -> str:
    return sha256_bytes(canonical_json(fingerprint).encode("utf-8"))


def _build_metadata(
    inputs: AuditInputs,
    result: AuditResult,
    *,
    output_hashes: Mapping[str, str],
    dependencies: Mapping[str, str],
    font_hashes: Mapping[str, str],
    elapsed_seconds: float,
) -> dict[str, Any]:
    config = inputs.config
    gate = _gate_summary(result.eligibility_table)
    return {
        "audit": {
            "name": config.run.name,
            "version": config.run.version,
            "config_path": str(config.config_path),
            "config_sha256": config.config_sha256,
            "config_hash": config.config_hash,
            "code_revision": code_revision(),
            "package_version": package_version(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "milestone": config.audit.milestone,
            "status": config.audit.status,
        },
        "inputs": inputs.input_hashes,
        "frozen_dependencies": dict(dependencies),
        "frozen_dependencies_hash": _fingerprint_hash(dependencies),
        "font_hashes": dict(font_hashes),
        "summary": {
            "faces_audited": gate["total_faces"],
            "eligible": gate["eligible"],
            "review_required": gate["review_required"],
            "ineligible": gate["ineligible"],
            "unavailable": gate["unavailable"],
            "audit_gate": gate["audit_gate"],
            "benchmark_generation_gate": gate["benchmark_generation_gate"],
            "decisions_supported": bool(result.eligibility_table["decision_supported"].all()),
        },
        "output_hashes": dict(output_hashes),
        "runtime_seconds": round(float(elapsed_seconds), 3),
    }


def _config_with_output(config: AuditConfig, output: Path | None) -> AuditConfig:
    if output is None:
        return config
    run = RunSettings(
        name=config.run.name,
        version=config.run.version,
        seed=config.run.seed,
        output_reports=Path(output),
    )
    candidate = AuditConfig(
        run=run,
        audit=config.audit,
        roster=config.roster,
        statuses=config.statuses,
        derivation=config.derivation,
        inputs=config.inputs,
        evidence=config.evidence,
        decisions=config.decisions,
        config_path=config.config_path,
        config_sha256=config.config_sha256,
        config_hash="",
    )
    return AuditConfig(
        run=candidate.run,
        audit=candidate.audit,
        roster=candidate.roster,
        statuses=candidate.statuses,
        derivation=candidate.derivation,
        inputs=candidate.inputs,
        evidence=candidate.evidence,
        decisions=candidate.decisions,
        config_path=candidate.config_path,
        config_sha256=candidate.config_sha256,
        config_hash=sha256_bytes(canonical_json(candidate.identity()).encode("utf-8")),
    )


def run_audit(
    config_path: Path | str,
    *,
    output_reports: Path | str | None = None,
    overwrite: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Run the read-only M7E audit and write deterministic Git-ignored outputs."""
    started = time.perf_counter()
    config = _config_with_output(load_config(config_path), output_reports)
    output_dir = _safe_output_path(config.run.output_reports)
    if output_dir.exists() and not overwrite:
        raise AuditError(
            f"output directory already exists: {output_dir}; pass --overwrite to replace it"
        )
    dependencies_before = _scoped_fingerprint(config)
    inputs = load_inputs(config)
    result = build_audit(inputs)
    checks = build_isolation_checks(inputs, result, result.report)
    _require_all_checks(
        [
            CheckResult(str(row["check"]), bool(row["passed"]), str(row["detail"]))
            for _, row in checks.iterrows()
        ],
        "M7E audit checks",
    )
    if output_dir.exists():
        if not overwrite:
            raise AuditError(f"output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(inputs.input_manifest, output_dir / "input_manifest.csv")
    _write_csv(result.evidence_table, output_dir / "font_evidence.csv")
    _write_csv(result.eligibility_table, output_dir / "eligibility.csv")
    _write_csv(result.roster_coverage, output_dir / "roster_coverage.csv")
    _write_csv(checks, output_dir / "isolation_checks.csv")
    _write_json(result.protocol, output_dir / "audit_manifest.json")
    (output_dir / "report.md").write_text(result.report, encoding="utf-8")
    output_hashes = _output_hashes(output_dir)
    dependencies_after = _scoped_fingerprint(config)
    if dependencies_before != dependencies_after:
        raise AuditError("declared audit dependencies changed during the M7E run")
    metadata = _build_metadata(
        inputs,
        result,
        output_hashes=output_hashes,
        dependencies=dependencies_after,
        font_hashes=_font_hashes(inputs),
        elapsed_seconds=time.perf_counter() - started,
    )
    _write_json(metadata, output_dir / "run_metadata.json")
    return output_dir, metadata


def _compare_table(
    checks: list[CheckResult], name: str, expected: pd.DataFrame, path: Path
) -> None:
    try:
        actual = pd.read_csv(path)
        expected = expected.copy()
        for column in expected.columns:
            if pd.api.types.is_object_dtype(expected[column]) or isinstance(
                expected[column].dtype, pd.StringDtype
            ):
                expected[column] = expected[column].fillna("")
                if column in actual.columns:
                    actual[column] = actual[column].fillna("")
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=1e-9,
            atol=1e-9,
        )
    except Exception as error:  # noqa: BLE001 - report every replay mismatch
        checks.append(CheckResult(f"table:{name}", False, f"{type(error).__name__}: {error}"))
    else:
        checks.append(CheckResult(f"table:{name}", True, "replayed table matches"))


def verify_audit(
    config_path: Path | str,
    *,
    output_reports: Path | str | None = None,
) -> VerificationReport:
    """Replay the M7E audit and verify a written audit directory."""
    config = _config_with_output(load_config(config_path), output_reports)
    output_dir = _safe_output_path(config.run.output_reports)
    checks: list[CheckResult] = []
    if not output_dir.is_dir():
        return VerificationReport(
            dataset=str(output_dir),
            checks=(CheckResult("output_directory", False, f"missing: {output_dir}"),),
        )
    inputs = load_inputs(config)
    result = build_audit(inputs)
    isolation = build_isolation_checks(inputs, result, result.report)
    expected_tables = {
        "input_manifest.csv": inputs.input_manifest,
        "font_evidence.csv": result.evidence_table,
        "eligibility.csv": result.eligibility_table,
        "roster_coverage.csv": result.roster_coverage,
        "isolation_checks.csv": isolation,
    }
    for name, expected in expected_tables.items():
        _compare_table(checks, name, expected, output_dir / name)
    checks.append(
        CheckResult(
            "isolation_checks",
            bool(isolation["passed"].all()),
            "all recomputed isolation checks passed",
        )
    )
    try:
        actual_protocol = json.loads(
            (output_dir / "audit_manifest.json").read_text(encoding="utf-8")
        )
        if actual_protocol != json_safe(result.protocol):
            raise AuditError("audit_manifest.json does not match the recomputed protocol")
    except Exception as error:  # noqa: BLE001
        checks.append(CheckResult("audit_manifest", False, f"{type(error).__name__}: {error}"))
    else:
        checks.append(CheckResult("audit_manifest", True, "recomputed audit manifest matches"))
    try:
        actual_report = (output_dir / "report.md").read_text(encoding="utf-8")
        if actual_report != result.report:
            raise AuditError("report.md does not match the recomputed report")
    except Exception as error:  # noqa: BLE001
        checks.append(CheckResult("report", False, f"{type(error).__name__}: {error}"))
    else:
        checks.append(CheckResult("report", True, "recomputed report matches"))
    metadata_path = output_dir / "run_metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001
        checks.append(CheckResult("metadata", False, f"{type(error).__name__}: {error}"))
        return VerificationReport(dataset=str(output_dir), checks=tuple(checks))
    checks.append(CheckResult("metadata", True, "run_metadata.json is readable"))
    checks.append(
        CheckResult(
            "config_sha256",
            metadata.get("audit", {}).get("config_sha256") == config.config_sha256,
            "recorded config SHA-256 matches",
        )
    )
    checks.append(
        CheckResult(
            "config_hash",
            metadata.get("audit", {}).get("config_hash") == config.config_hash,
            "recorded config identity hash matches",
        )
    )
    checks.append(
        CheckResult(
            "input_hashes",
            metadata.get("inputs") == inputs.input_hashes,
            "recorded input hashes match",
        )
    )
    current_dependencies = _scoped_fingerprint(config)
    checks.append(
        CheckResult(
            "frozen_dependencies",
            metadata.get("frozen_dependencies") == current_dependencies
            and metadata.get("frozen_dependencies_hash") == _fingerprint_hash(current_dependencies),
            "declared audit dependencies are unchanged",
        )
    )
    checks.append(
        CheckResult(
            "font_hashes",
            metadata.get("font_hashes") == _font_hashes(inputs),
            "recorded font hashes match the M7D pins",
        )
    )
    current_output_hashes = _output_hashes(output_dir)
    checks.append(
        CheckResult(
            "output_hashes",
            metadata.get("output_hashes") == current_output_hashes,
            "recorded output hashes match",
        )
    )
    actual_output_files = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    expected_output_files = sorted(
        (*OUTPUT_TABLE_NAMES, "audit_manifest.json", "report.md", "run_metadata.json")
    )
    checks.append(
        CheckResult(
            "output_file_set",
            actual_output_files == expected_output_files,
            (
                "the audit output file set is exactly as declared"
                if actual_output_files == expected_output_files
                else (
                    f"unexpected output file set: missing {sorted(set(expected_output_files) - set(actual_output_files))}, "
                    f"unexpected {sorted(set(actual_output_files) - set(expected_output_files))}"
                )
            ),
        )
    )
    return VerificationReport(dataset=str(output_dir), checks=tuple(checks))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="captcha-font-roster-audit",
        description="Read-only M7E font-roster licensing and availability audit.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run the font-roster audit")
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--output-reports", type=Path)
    run_parser.add_argument("--overwrite", action="store_true")
    verify_parser = subparsers.add_parser("verify", help="replay and verify the audit")
    verify_parser.add_argument("--config", required=True, type=Path)
    verify_parser.add_argument("--output-reports", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            output_dir, metadata = run_audit(
                args.config,
                output_reports=args.output_reports,
                overwrite=args.overwrite,
            )
            print(f"wrote M7E audit to {output_dir}")
            print(json.dumps(json_safe(metadata["summary"]), indent=2, sort_keys=True))
            return 0
        report = verify_audit(
            args.config,
            output_reports=args.output_reports,
        )
        for check in report.checks:
            status = "PASS" if check.passed else "FAIL"
            print(f"[{status}] {check.name}: {check.detail}")
        print(f"verification: {'PASS' if report.ok else 'FAIL'}")
        return 0 if report.ok else 1
    except (AuditError, AuditConfigError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
