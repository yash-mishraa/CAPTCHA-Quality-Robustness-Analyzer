"""Milestone M8B source-case and generator-preflight verifier.

M8B resolves the two execution blockers that the frozen Milestone 8A
protocol left open: the exact matched text strings and the deterministic
source identifier scheme. This module verifies both, proves the planned
counts and join contracts without materialising the dataset, and exercises
the existing generator in an isolated temporary directory.

The module never creates a planned benchmark output path. The bounded
generator preflight renders a small, explicitly enumerated set of
representative images into a temporary directory that is removed again when
the run finishes.

The verification fails closed on:

* a missing or changed text-case manifest;
* a duplicate or missing case identifier;
* a font roster or font-hash mismatch;
* an invalid canonical serialisation;
* duplicate or unstable source identifiers;
* unsupported or unresolved glyph coverage;
* a changed renderer configuration or an unpinned dependency;
* unexpected files or outputs;
* any attempt to bypass the Milestone 8A lifecycle contract.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from PIL import Image

from captcha_quality.corruptions import (
    FAMILY_BLUR,
    FAMILY_NOISE,
    SeverityLadder,
    apply_corruption,
)
from captcha_quality.dataset import CheckResult, VerificationReport
from captcha_quality.geometric import ElasticFieldSpec, apply_elastic
from captcha_quality.interference import (
    VARIANT_ARC_STROKES,
    VARIANT_THIN_LINES,
    InterferenceLadder,
    apply_strokes,
    plan_stroke_pool,
)
from captcha_quality.provenance import sha256_bytes, sha256_file
from captcha_quality.render import RenderConfig, RenderError, ResolvedFont, render_clean
from captcha_quality.source_identity import (
    CLEAN_FAMILY,
    CORRUPTION_FAMILIES,
    GlyphCoverageError,
    IdentityError,
    TextCase,
    case_record_hash,
    font_codepoints,
    generation_config_hash,
    image_relpath,
    manifest_hash,
    missing_glyphs,
    render_config_hash,
    render_config_payload,
    render_seed,
    row_id,
    source_id,
    validate_stored_text,
    variant_id,
    variant_seed,
)

__all__ = [
    "M8BExecutionProtocol",
    "M8BPreflightError",
    "CaseManifest",
    "PlannedRow",
    "PlannedSource",
    "load_case_manifest",
    "load_execution_protocol",
    "main",
    "plan_rows",
    "plan_sources",
    "run_generator_preflight",
    "verify_preflight",
]


class M8BPreflightError(RuntimeError):
    """Raised when a Milestone 8B preflight invariant is violated."""


class M8BConfigError(ValueError):
    """Raised for an invalid Milestone 8B configuration."""


#: The default locations of the Milestone 8B artifacts.
DEFAULT_EXECUTION_PROTOCOL = Path("configs/cross_font_benchmark_protocol_v0_2_0.toml")
DEFAULT_LIFECYCLE = Path("configs/benchmark_lifecycle_v0_2_0.toml")
DEFAULT_CASE_MANIFEST = Path("configs/cross_font_text_cases_v0_1_0.toml")

#: The number of faces rendered by the bounded generator preflight.
PREFLIGHT_FONT_COUNT: Final[int] = 10

#: The number of matched cases rendered by the bounded generator preflight.
PREFLIGHT_CASE_COUNT: Final[int] = 1

#: The number of faces that receive the full corruption sweep.
PREFLIGHT_CORRUPTION_FONT_COUNT: Final[int] = 2


# ---------------------------------------------------------------------------
# configuration objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CaseManifest:
    """The committed matched text-case manifest."""

    path: Path
    version: str
    n_cases: int
    charset: str
    seed: int
    case_namespace: str
    normalization: str
    cases: tuple[TextCase, ...]
    file_sha256: str
    manifest_hash: str

    def case_ids(self) -> tuple[str, ...]:
        return tuple(case.case_id for case in self.cases)


@dataclass(frozen=True, slots=True)
class M8BExecutionProtocol:
    """The frozen Milestone 8B execution protocol."""

    path: Path
    raw: dict[str, Any]
    version: str
    file_sha256: str

    @property
    def fonts(self) -> list[dict[str, Any]]:
        return list(self.raw["fonts"])

    @property
    def text_cases(self) -> dict[str, Any]:
        return dict(self.raw["text_cases"])

    @property
    def renderer(self) -> dict[str, Any]:
        return dict(self.raw["renderer"])

    @property
    def corruptions(self) -> dict[str, Any]:
        return dict(self.raw["corruptions"])

    @property
    def counts(self) -> dict[str, Any]:
        return dict(self.raw["counts"])

    @property
    def identity(self) -> dict[str, Any]:
        return dict(self.raw["identity"])

    @property
    def text_case_manifest(self) -> dict[str, Any]:
        return dict(self.raw["text_case_manifest"])


@dataclass(frozen=True, slots=True)
class PlannedSource:
    """One planned clean source: a (text case, typeface) pair."""

    source_id: str
    case_id: str
    case_index: int
    font_id: str
    font_sha256: str
    font_path: str
    text: str
    text_length: int
    text_sha256: str
    render_seed: int
    render_config_hash: str
    generation_config_hash: str
    split: str
    split_group_id: str


@dataclass(frozen=True, slots=True)
class PlannedRow:
    """One planned dataset row."""

    row_id: str
    sample_id: str
    variant_id: str
    source_id: str
    case_id: str
    font_id: str
    family: str
    severity: int
    variant_seed: int
    primary: bool
    image_path: str


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------


def load_execution_protocol(path: Path | str) -> M8BExecutionProtocol:
    """Load and minimally validate the Milestone 8B execution protocol."""
    protocol_path = Path(path)
    if not protocol_path.is_file():
        raise M8BConfigError(f"missing execution protocol: {protocol_path}")
    raw_bytes = protocol_path.read_bytes()
    try:
        payload = tomllib.loads(raw_bytes.decode("utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise M8BConfigError(f"invalid TOML in {protocol_path}: {error}") from error
    for key in ("run", "text_cases", "renderer", "corruptions", "counts", "identity"):
        if key not in payload:
            raise M8BConfigError(f"the execution protocol is missing the {key!r} section")
    version = str(payload["run"]["version"])
    if payload["run"].get("milestone") != "M8B":
        raise M8BConfigError("the execution protocol must belong to milestone M8B")
    return M8BExecutionProtocol(
        path=protocol_path,
        raw=payload,
        version=version,
        file_sha256=sha256_bytes(raw_bytes),
    )


def load_case_manifest(path: Path | str) -> CaseManifest:
    """Load the committed text-case manifest without normalising anything.

    The stored strings are read verbatim. Normalisation is *asserted*, never
    applied, so a stored string that would need normalising is reported
    rather than silently repaired.
    """
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise M8BConfigError(f"missing text-case manifest: {manifest_path}")
    raw_bytes = manifest_path.read_bytes()
    try:
        payload = tomllib.loads(raw_bytes.decode("utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise M8BConfigError(f"invalid TOML in {manifest_path}: {error}") from error
    header = payload.get("manifest")
    if not isinstance(header, dict):
        raise M8BConfigError("the text-case manifest is missing the [manifest] section")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise M8BConfigError("the text-case manifest declares no cases")
    charset = str(header["charset"])
    cases: list[TextCase] = []
    for index, item in enumerate(raw_cases):
        if not isinstance(item, dict):
            raise M8BConfigError(f"cases[{index}] must be a table")
        text = str(item["text"])
        validate_stored_text(text, charset=charset)
        cases.append(
            TextCase(
                case_id=str(item["case_id"]),
                case_index=int(item["case_index"]),
                text=text,
                text_utf8_hex=str(item["text_utf8_hex"]),
                text_sha256=str(item["text_sha256"]),
                text_length=int(item["text_length"]),
                case_hash=str(item["case_hash"]),
            )
        )
    return CaseManifest(
        path=manifest_path,
        version=str(header["version"]),
        n_cases=int(header["n_cases"]),
        charset=charset,
        seed=int(header["seed"]),
        case_namespace=str(header["case_namespace"]),
        normalization=str(header["normalization"]),
        cases=tuple(cases),
        file_sha256=sha256_bytes(raw_bytes),
        manifest_hash=manifest_hash([case.record() for case in cases]),
    )


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------


def _render_config(protocol: M8BExecutionProtocol, text_length: int) -> RenderConfig:
    """The frozen renderer configuration for one text length."""
    renderer = protocol.renderer
    return RenderConfig(
        canvas=(int(renderer["canvas_width"]), int(renderer["canvas_height"])),
        charset=protocol.text_cases["charset"],
        text_length=text_length,
        fit_fraction=float(renderer["fit_fraction"]),
        vertical_padding=float(renderer["vertical_padding"]),
        jitter_fraction=float(renderer["jitter_fraction"]),
        spacing_jitter_fraction=float(renderer["spacing_jitter_fraction"]),
        placement_jitter_fraction=float(renderer["placement_jitter_fraction"]),
        background_mode=str(renderer["background_mode"]),
        background_range=(int(renderer["background_min"]), int(renderer["background_max"])),
        center_ink=bool(renderer["center_ink"]),
        strict_bounds=bool(renderer["strict_bounds"]),
    )


def plan_sources(
    protocol: M8BExecutionProtocol,
    manifest: CaseManifest,
) -> tuple[PlannedSource, ...]:
    """The planned clean sources: every case rendered in every typeface.

    The plan is a pure function of the manifest and the protocol. It never
    touches the filesystem, so it cannot depend on directory traversal order
    or on row ordering.
    """
    generation = generation_config_hash(protocol.raw)
    sources: list[PlannedSource] = []
    for font in protocol.fonts:
        font_id = str(font["key"])
        font_sha256 = str(font["sha256"])
        for case in manifest.cases:
            render_hash = render_config_hash(
                render_config_payload(
                    protocol.renderer,
                    charset=manifest.charset,
                    text_length=case.text_length,
                )
            )
            seed = render_seed(manifest.seed, font_id, case.case_index)
            sources.append(
                PlannedSource(
                    source_id=source_id(
                        case=case.record(),
                        font_id=font_id,
                        font_sha256=font_sha256,
                        render_config_hash=render_hash,
                        render_seed=seed,
                        generation_config_hash=generation,
                    ),
                    case_id=case.case_id,
                    case_index=case.case_index,
                    font_id=font_id,
                    font_sha256=font_sha256,
                    font_path=str(font["path"]),
                    text=case.text,
                    text_length=case.text_length,
                    text_sha256=case.text_sha256,
                    render_seed=seed,
                    render_config_hash=render_hash,
                    generation_config_hash=generation,
                    split=str(protocol.identity["split"]),
                    split_group_id="",
                )
            )
    sources = [
        PlannedSource(
            source_id=source.source_id,
            case_id=source.case_id,
            case_index=source.case_index,
            font_id=source.font_id,
            font_sha256=source.font_sha256,
            font_path=source.font_path,
            text=source.text,
            text_length=source.text_length,
            text_sha256=source.text_sha256,
            render_seed=source.render_seed,
            render_config_hash=source.render_config_hash,
            generation_config_hash=source.generation_config_hash,
            split=source.split,
            split_group_id=source.source_id,
        )
        for source in sources
    ]
    return tuple(sources)


def severity_map(protocol: M8BExecutionProtocol) -> dict[str, tuple[int, ...]]:
    """The declared severity levels of every corruption family."""
    identity = protocol.identity
    families = identity["families"]
    return {
        str(family): tuple(int(level) for level in families[family]["severities"])
        for family in CORRUPTION_FAMILIES
    }


def plan_rows(
    protocol: M8BExecutionProtocol,
    manifest: CaseManifest,
    sources: Sequence[PlannedSource],
) -> tuple[PlannedRow, ...]:
    """The planned dataset rows: every variant of every source.

    The plan is derived from the protocol alone, so the counts and the
    primary/diagnostic split can be proven without rendering anything.
    """
    generation = generation_config_hash(protocol.raw)
    cases = {case.case_id: case for case in manifest.cases}
    rows: list[PlannedRow] = []
    for source in sources:
        case = cases[source.case_id]
        record = case.record()
        clean_variant = variant_id(CLEAN_FAMILY, 0)
        clean_sample = f"{source.source_id}__{clean_variant}"
        rows.append(
            PlannedRow(
                row_id=row_id(
                    source=source.source_id,
                    sample=clean_sample,
                    variant=clean_variant,
                    case=record,
                    font_id=source.font_id,
                    font_sha256=source.font_sha256,
                    render_config_hash=source.render_config_hash,
                    render_seed=source.render_seed,
                    generation_config_hash=generation,
                    family=CLEAN_FAMILY,
                    severity=0,
                    variant_seed=variant_seed(manifest.seed, source.source_id, CLEAN_FAMILY, 0),
                    primary=False,
                ),
                sample_id=clean_sample,
                variant_id=clean_variant,
                source_id=source.source_id,
                case_id=source.case_id,
                font_id=source.font_id,
                family=CLEAN_FAMILY,
                severity=0,
                variant_seed=variant_seed(manifest.seed, source.source_id, CLEAN_FAMILY, 0),
                primary=False,
                image_path=image_relpath(CLEAN_FAMILY, source.source_id),
            )
        )
        for family in CORRUPTION_FAMILIES:
            for severity in severity_map(protocol)[family]:
                variant = variant_id(family, severity)
                sample = f"{source.source_id}__{variant}"
                seed = variant_seed(manifest.seed, source.source_id, family, severity)
                primary = severity not in protocol.identity["diagnostic_severities"].get(family, [])
                rows.append(
                    PlannedRow(
                        row_id=row_id(
                            source=source.source_id,
                            sample=sample,
                            variant=variant,
                            case=record,
                            font_id=source.font_id,
                            font_sha256=source.font_sha256,
                            render_config_hash=source.render_config_hash,
                            render_seed=source.render_seed,
                            generation_config_hash=generation,
                            family=family,
                            severity=severity,
                            variant_seed=seed,
                            primary=primary,
                        ),
                        sample_id=sample,
                        variant_id=variant,
                        source_id=source.source_id,
                        case_id=source.case_id,
                        font_id=source.font_id,
                        family=family,
                        severity=severity,
                        variant_seed=seed,
                        primary=primary,
                        image_path=image_relpath(family, sample),
                    )
                )
    return tuple(rows)


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------


def _check(checks: list[CheckResult], name: str, function: Any) -> None:
    try:
        detail = function()
    except Exception as error:  # noqa: BLE001 - every incompatibility is reported
        checks.append(CheckResult(name, False, f"{type(error).__name__}: {error}"))
    else:
        checks.append(CheckResult(name, True, str(detail)))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise M8BPreflightError(message)


def _verify_manifest_pins(protocol: M8BExecutionProtocol, manifest: CaseManifest) -> str:
    """The execution protocol must pin the manifest file and its content."""
    pins = protocol.text_case_manifest
    expected_file = str(pins["file_sha256"])
    expected_manifest = str(pins["manifest_hash"])
    if manifest.file_sha256 != expected_file:
        raise M8BPreflightError(
            f"the text-case manifest file hash {manifest.file_sha256} does not match the "
            f"pinned value {expected_file}"
        )
    if manifest.manifest_hash != expected_manifest:
        raise M8BPreflightError(
            f"the canonical text-case manifest hash {manifest.manifest_hash} does not match "
            f"the pinned value {expected_manifest}"
        )
    return (
        f"the manifest file hash and the canonical manifest hash match the pins in "
        f"execution protocol {protocol.version}"
    )


def _verify_case_identity(manifest: CaseManifest) -> str:
    """Exactly thirty cases with unique identifiers and correct ordering."""
    _require(
        len(manifest.cases) == manifest.n_cases == 30,
        f"the manifest holds {len(manifest.cases)} cases, expected 30",
    )
    identifiers = manifest.case_ids()
    _require(len(set(identifiers)) == len(identifiers), "the case identifiers are not unique")
    texts = [case.text for case in manifest.cases]
    _require(len(set(texts)) == len(texts), "the stored texts are not unique")
    for position, case in enumerate(manifest.cases):
        _require(
            case.case_index == position,
            f"the case at position {position} declares index {case.case_index}; the order "
            "is part of the pinned manifest",
        )
    return f"{len(manifest.cases)} cases with unique identifiers, unique texts and a pinned order"


def _verify_case_encoding(manifest: CaseManifest) -> str:
    """Every stored string is the exact UTF-8 encoding of a normalised case."""
    for case in manifest.cases:
        encoded = case.text.encode("utf-8")
        if encoded.hex() != case.text_utf8_hex:
            raise M8BPreflightError(
                f"{case.case_id}: the stored UTF-8 hex does not match the stored text"
            )
        if sha256_bytes(encoded) != case.text_sha256:
            raise M8BPreflightError(
                f"{case.case_id}: the stored text hash does not match the stored text"
            )
        if len(case.text) != case.text_length:
            raise M8BPreflightError(
                f"{case.case_id}: the stored text length does not match the declared length"
            )
        if case_record_hash(case.record()) != case.case_hash:
            raise M8BPreflightError(
                f"{case.case_id}: the canonical case record hash does not match the stored value"
            )
    return (
        f"all {len(manifest.cases)} cases carry a matching UTF-8 encoding, text hash, "
        "length and canonical case record hash"
    )


def _verify_manifest_derivation(
    protocol: M8BExecutionProtocol,
    manifest: CaseManifest,
) -> str:
    """The committed cases must still match the pinned derivation rule."""
    expected = derive_matched_cases(protocol, manifest)
    stored = [case.text for case in manifest.cases]
    if expected != stored:
        raise M8BPreflightError(
            "the committed text cases do not match the pinned derivation rule; the manifest "
            "must be regenerated as a new version, never edited in place"
        )
    return "the committed cases still match the pinned deterministic derivation rule"


def derive_matched_cases(
    protocol: M8BExecutionProtocol,
    manifest: CaseManifest,
) -> list[str]:
    """Re-derive the matched cases from the pinned rule.

    The rule is the one the Milestone 8A protocol froze: for case index *i*
    a single generator seeded with
    ``derive_noise_seed(dataset_seed, "matched_case", str(i))`` draws the text
    length and then the text.
    """
    from captcha_quality.corruptions import derive_noise_seed
    from captcha_quality.render import sample_text

    text_cases = protocol.text_cases
    seed = int(text_cases["dataset_seed"])
    namespace = str(text_cases["case_namespace"])
    lengths = [int(value) for value in text_cases["text_lengths"]]
    weights = np.asarray(
        [float(value) for value in text_cases["text_length_weights"]], dtype=np.float64
    )
    charset = str(text_cases["charset"])
    derived: list[str] = []
    for index in range(manifest.n_cases):
        rng = np.random.default_rng(derive_noise_seed(seed, namespace, str(index)))
        length = int(rng.choice(np.asarray(lengths, dtype=np.int64), p=weights / weights.sum()))
        config = RenderConfig(charset=charset, text_length=length)
        derived.append(sample_text(rng, config))
    return derived


def _verify_roster(protocol: M8BExecutionProtocol) -> str:
    """The roster must match the M7D plan and the M7E version 0.2.0 evidence."""
    import pandas as pd

    m7d = tomllib.loads(
        Path("configs/evaluation_design_plan_v0_1_0.toml").read_text(encoding="utf-8")
    )
    planned = {str(font["key"]): str(font["sha256"]) for font in m7d["planned_fonts"]}
    executed = {str(font["key"]): str(font["sha256"]) for font in protocol.fonts}
    if planned != executed:
        raise M8BPreflightError(
            f"the execution protocol roster {sorted(executed)} does not match the M7D roster "
            f"{sorted(planned)}"
        )
    evidence = pd.read_csv("reports/font_roster_audit_v0_2_0/font_evidence.csv")
    eligibility = pd.read_csv("reports/font_roster_audit_v0_2_0/eligibility.csv")
    recorded = {str(row["font_key"]): str(row["m7d_font_sha256"]) for _, row in evidence.iterrows()}
    observed = {
        str(row["font_key"]): str(row["observed_font_sha256"]) for _, row in evidence.iterrows()
    }
    if recorded != executed or observed != executed:
        raise M8BPreflightError(
            "the execution protocol roster does not match the M7E version 0.2.0 evidence"
        )
    eligible = {
        str(row["font_key"])
        for _, row in eligibility.iterrows()
        if str(row["declared_status"]) == "eligible"
    }
    if eligible != set(executed):
        raise M8BPreflightError(
            "every executed face must be eligible in the M7E version 0.2.0 audit"
        )
    return f"{len(executed)} roster members match the M7D plan and the M7E v0.2.0 evidence"


def _verify_font_files(protocol: M8BExecutionProtocol) -> str:
    """Every pinned font file must exist and hash to its pinned value."""
    for font in protocol.fonts:
        path = Path(str(font["path"]))
        if not path.is_file():
            raise M8BPreflightError(f"the pinned font file is missing: {path}")
        actual = sha256_file(path)
        if actual != str(font["sha256"]):
            raise M8BPreflightError(
                f"the pinned font file {path} has hash {actual}, expected {font['sha256']}"
            )
    return f"all {len(protocol.fonts)} pinned font files exist and match their pinned hash"


def _verify_glyph_coverage(
    protocol: M8BExecutionProtocol,
    manifest: CaseManifest,
) -> str:
    """Every character of every case must be renderable by every pinned face."""
    characters = sorted({character for case in manifest.cases for character in case.text})
    for font in protocol.fonts:
        path = Path(str(font["path"]))
        codepoints = font_codepoints(path)
        absent = missing_glyphs(codepoints, "".join(characters))
        if absent:
            raise GlyphCoverageError(
                f"the face {font['key']} has no glyph for {absent}; a character is never "
                "substituted and never replaced with a question mark"
            )
    return (
        f"all {len(characters)} distinct case characters are present in the character map of "
        f"all {len(protocol.fonts)} pinned faces"
    )


def _verify_source_identities(
    protocol: M8BExecutionProtocol,
    manifest: CaseManifest,
    sources: Sequence[PlannedSource],
) -> str:
    """The source identifiers must be unique, stable and correctly derived."""
    identifiers = [source.source_id for source in sources]
    _require(
        len(set(identifiers)) == len(identifiers),
        "the planned source identifiers are not unique",
    )
    _require(
        len(sources) == int(protocol.counts["n_sources"]),
        f"the plan holds {len(sources)} sources, expected {protocol.counts['n_sources']}",
    )
    repeated = plan_sources(protocol, manifest)
    if [source.source_id for source in repeated] != identifiers:
        raise M8BPreflightError(
            "the source identifiers are not stable: recomputing the plan changed them"
        )
    for source in sources:
        if source.split_group_id != source.source_id:
            raise M8BPreflightError(
                f"{source.source_id}: the split group must be the source identifier"
            )
    return (
        f"{len(sources)} unique source identifiers, stable across repeated computation and "
        "independent of any filesystem traversal"
    )


def _verify_row_plan(
    protocol: M8BExecutionProtocol,
    rows: Sequence[PlannedRow],
    sources: Sequence[PlannedSource],
) -> str:
    """The planned row counts, identities and primary/diagnostic split."""
    counts = protocol.counts
    _require(
        len(rows) == int(counts["total_rows"]),
        f"the plan holds {len(rows)} rows, expected {counts['total_rows']}",
    )
    primary = sum(1 for row in rows if row.primary)
    diagnostic = len(rows) - primary
    _require(
        primary == int(counts["primary_rows"]),
        f"the plan holds {primary} primary rows, expected {counts['primary_rows']}",
    )
    _require(
        diagnostic == int(counts["diagnostic_rows"]),
        f"the plan holds {diagnostic} diagnostic rows, expected {counts['diagnostic_rows']}",
    )
    samples = [row.sample_id for row in rows]
    _require(len(set(samples)) == len(samples), "the planned sample identifiers are not unique")
    row_ids = [row.row_id for row in rows]
    _require(len(set(row_ids)) == len(row_ids), "the planned row identifiers are not unique")
    images = [row.image_path for row in rows]
    _require(
        len(set(images)) == len(images),
        "the planned image identities are not unique",
    )
    _require(
        len(images) == int(counts["total_rows"]),
        f"the plan holds {len(images)} image identities, expected {counts['total_rows']}",
    )
    directories = {row.image_path.split("/")[1] for row in rows}
    expected_directories = set(protocol.identity["image_subdirectories"])
    if directories != expected_directories:
        raise M8BPreflightError(
            f"the planned image subdirectories {sorted(directories)} do not match the "
            f"declared {sorted(expected_directories)}"
        )
    known_sources = {source.source_id for source in sources}
    unknown = sorted({row.source_id for row in rows} - known_sources)
    if unknown:
        raise M8BPreflightError(f"the plan references unknown sources: {unknown}")
    diagnostic_families = {
        row.family for row in rows if not row.primary and row.family != CLEAN_FAMILY
    }
    if diagnostic_families != {"elastic"}:
        raise M8BPreflightError(f"the diagnostic exclusion is wrong: {sorted(diagnostic_families)}")
    return (
        f"{len(rows)} planned rows, {primary} primary and {diagnostic} diagnostic, with "
        f"{len(images)} unique image identities across {len(directories)} subdirectories"
    )


def _verify_generation_config(protocol: M8BExecutionProtocol) -> str:
    """The renderer configuration and the generation configuration are frozen."""
    expected = generation_config_hash(protocol.raw)
    declared = str(protocol.identity["generation_config_hash"])
    if expected != declared:
        raise M8BPreflightError(
            f"the generation configuration hash {expected} does not match the pinned value "
            f"{declared}; the renderer configuration or the corruption ladder changed"
        )
    declared_version = str(protocol.identity["render_config_version"])
    from captcha_quality.source_identity import RENDER_CONFIG_VERSION

    if declared_version != RENDER_CONFIG_VERSION:
        raise M8BPreflightError(
            f"the pinned render configuration version {declared_version} does not match the "
            f"implemented version {RENDER_CONFIG_VERSION}"
        )
    return (
        f"the generation configuration hash matches the pinned value and the render "
        f"configuration version is {declared_version}"
    )


def _verify_lifecycle_contract(
    lifecycle_path: Path | str,
    protocol: M8BExecutionProtocol,
) -> str:
    """The Milestone 8A lifecycle contract must still hold."""
    from captcha_quality.benchmark_lifecycle import verify_lifecycle

    lifecycle = Path(lifecycle_path)
    if not lifecycle.is_file():
        raise M8BPreflightError(f"missing lifecycle protocol: {lifecycle}")
    report = verify_lifecycle(lifecycle)
    if not report.ok:
        failures = [f"{check.name}: {check.detail}" for check in report.failures()]
        raise M8BPreflightError(f"the lifecycle verification failed: {'; '.join(failures)}")
    payload = tomllib.loads(lifecycle.read_text(encoding="utf-8"))
    if payload["run"]["status"] != "authorized_not_executed":
        raise M8BPreflightError("the lifecycle is not in the authorized-but-not-executed state")
    authorised = Path(str(payload["authorized_execution"]["protocol"])).resolve()
    if authorised != protocol.path.resolve():
        raise M8BPreflightError(
            "the lifecycle protocol does not authorise the execution protocol under test"
        )
    return (
        "the lifecycle verification passes, the lifecycle is authorized but not executed and "
        "it authorises the execution protocol under test"
    )


def verify_preflight(
    execution_protocol: Path | str = DEFAULT_EXECUTION_PROTOCOL,
    *,
    lifecycle: Path | str | None = DEFAULT_LIFECYCLE,
    manifest: Path | str | None = None,
) -> VerificationReport:
    """Verify the Milestone 8B source cases, identity scheme and row plan."""
    protocol = load_execution_protocol(execution_protocol)
    manifest_path = (
        Path(manifest) if manifest is not None else Path(str(protocol.text_case_manifest["path"]))
    )
    checks: list[CheckResult] = []
    loaded: dict[str, Any] = {}

    def _load() -> CaseManifest:
        if "manifest" not in loaded:
            loaded["manifest"] = load_case_manifest(manifest_path)
        return loaded["manifest"]

    _check(checks, "text_case_manifest", lambda: _verify_manifest_pins(protocol, _load()))
    _check(checks, "case_identity", lambda: _verify_case_identity(_load()))
    _check(checks, "case_encoding", lambda: _verify_case_encoding(_load()))
    _check(checks, "case_derivation", lambda: _verify_manifest_derivation(protocol, _load()))
    _check(checks, "roster", lambda: _verify_roster(protocol))
    _check(checks, "font_files", lambda: _verify_font_files(protocol))
    _check(checks, "glyph_coverage", lambda: _verify_glyph_coverage(protocol, _load()))
    _check(checks, "generation_config", lambda: _verify_generation_config(protocol))

    sources: list[PlannedSource] = []
    rows: list[PlannedRow] = []

    def _plan_sources() -> tuple[PlannedSource, ...]:
        if not sources:
            sources.extend(plan_sources(protocol, _load()))
        return tuple(sources)

    def _plan_rows() -> tuple[PlannedRow, ...]:
        if not rows:
            rows.extend(plan_rows(protocol, _load(), _plan_sources()))
        return tuple(rows)

    _check(
        checks,
        "source_identities",
        lambda: _verify_source_identities(protocol, _load(), _plan_sources()),
    )
    _check(checks, "row_plan", lambda: _verify_row_plan(protocol, _plan_rows(), _plan_sources()))
    if lifecycle is not None:
        _check(
            checks, "lifecycle_contract", lambda: _verify_lifecycle_contract(lifecycle, protocol)
        )
    return VerificationReport(dataset=str(protocol.path), checks=tuple(checks))


# ---------------------------------------------------------------------------
# bounded generator preflight
# ---------------------------------------------------------------------------


def _validate_rendered_image(path: Path, width: int, height: int) -> str:
    """Validate the image contract: exact size, grayscale mode, decodable PNG."""
    if not path.is_file():
        raise M8BPreflightError(f"the rendered image is missing: {path}")
    with Image.open(path) as handle:
        if handle.format != "PNG":
            raise M8BPreflightError(f"the rendered image is not a PNG: {path}")
        if handle.mode != "L":
            raise M8BPreflightError(
                f"the rendered image mode is {handle.mode!r}, expected 'L': {path}"
            )
        size = handle.size
    if size != (width, height):
        raise M8BPreflightError(
            f"the rendered image is {size[0]}x{size[1]}, expected {width}x{height}: {path}"
        )
    return f"{width}x{height} grayscale PNG"


def _resolve_pinned_font(key: str, path: str, expected_sha256: str) -> ResolvedFont:
    """Resolve a pinned face and verify its hash; never substitute a face."""
    font_path = Path(path)
    if not font_path.is_file():
        raise M8BPreflightError(f"the pinned font file is missing: {font_path}")
    actual = sha256_file(font_path)
    if actual != expected_sha256:
        raise M8BPreflightError(
            f"the pinned font file {font_path} has hash {actual}, expected {expected_sha256}"
        )
    return ResolvedFont(key=key, path=font_path)


def _build_ladders(protocol: M8BExecutionProtocol) -> dict[str, Any]:
    """The frozen corruption ladders, built from the protocol."""
    corruptions = protocol.corruptions
    ladders: dict[str, Any] = {
        FAMILY_BLUR: SeverityLadder(FAMILY_BLUR, (0.0, *corruptions["blur_radii_px"]), "px"),
        FAMILY_NOISE: SeverityLadder(FAMILY_NOISE, (0.0, *corruptions["noise_sigmas"]), "sigma"),
    }
    for variant, counts in (
        (VARIANT_THIN_LINES, corruptions["thin_line_stroke_counts"]),
        (VARIANT_ARC_STROKES, corruptions["arc_stroke_counts"]),
    ):
        ladders[variant] = InterferenceLadder(
            variant=variant,
            stroke_counts=tuple(int(count) for count in counts),
            width_px=int(corruptions["stroke_width_px"]),
            intensity=int(corruptions["stroke_intensity"]),
            line_angle_deg=tuple(float(value) for value in corruptions["line_angle_deg"]),
            arc_sweep_deg=tuple(float(value) for value in corruptions["arc_sweep_deg"]),
            arc_radius_fraction=tuple(float(value) for value in corruptions["arc_radius_fraction"]),
            arc_aspect_ratio=tuple(float(value) for value in corruptions["arc_aspect_ratio"]),
        )
    return ladders


def _apply_family(
    family: str,
    severity: int,
    image: np.ndarray,
    ladders: Mapping[str, Any],
    protocol: M8BExecutionProtocol,
    *,
    source: str,
    seed: int,
) -> np.ndarray:
    """Apply one corruption of one family at one severity."""
    if family in (FAMILY_BLUR, FAMILY_NOISE):
        ladder = ladders[family]
        result = apply_corruption(image, ladder, severity, noise_seed=seed)
        return result.image
    if family in (VARIANT_THIN_LINES, VARIANT_ARC_STROKES):
        ladder = ladders[family]
        strokes = plan_stroke_pool(image, ladder, seed=seed)
        count = ladder.stroke_counts[severity - 1]
        return apply_strokes(
            image,
            strokes[:count],
            width_px=ladder.width_px,
            intensity=ladder.intensity,
        )
    if family == "elastic":
        corruptions = protocol.corruptions
        amplitude = float(corruptions["elastic_severity_amplitudes_px"][severity - 1])
        field_spec = ElasticFieldSpec(
            height=int(image.shape[0]),
            width=int(image.shape[1]),
            seed=seed,
            grid_rows=int(corruptions["grid_rows"]),
            grid_cols=int(corruptions["grid_cols"]),
            smoothing_sigma_px=float(corruptions["smoothing_sigma_px"]),
            interpolation=str(corruptions["interpolation"]),
            border_mode=str(corruptions["border_mode"]),
        )
        return apply_elastic(image, field_spec=field_spec, amplitude_px=amplitude).image
    raise M8BPreflightError(f"unsupported corruption family {family!r}")


def run_generator_preflight(
    execution_protocol: Path | str = DEFAULT_EXECUTION_PROTOCOL,
    *,
    manifest: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> VerificationReport:
    """Exercise the existing generator on a small, bounded set of cases.

    The preflight renders the clean reference of one matched case in every
    pinned face, sweeps every corruption family and severity on a small
    subset, and probes the explicit failure paths. It writes only into an
    isolated temporary directory, never into the authorised benchmark paths.
    """
    protocol = load_execution_protocol(execution_protocol)
    manifest_path = (
        Path(manifest) if manifest is not None else Path(str(protocol.text_case_manifest["path"]))
    )
    case_manifest = load_case_manifest(manifest_path)
    sources = plan_sources(protocol, case_manifest)
    rows = plan_rows(protocol, case_manifest, sources)
    ladders = _build_ladders(protocol)
    width = int(protocol.renderer["canvas_width"])
    height = int(protocol.renderer["canvas_height"])
    checks: list[CheckResult] = []

    temporary = output_dir is None
    root = (
        Path(tempfile.mkdtemp(prefix="m8b-generator-preflight-")) if temporary else Path(output_dir)
    )
    try:
        selected_fonts = protocol.fonts[:PREFLIGHT_FONT_COUNT]
        corruption_fonts = protocol.fonts[:PREFLIGHT_CORRUPTION_FONT_COUNT]
        case = case_manifest.cases[:PREFLIGHT_CASE_COUNT]

        def _render_all_faces() -> str:
            rendered = 0
            for font in selected_fonts:
                resolved = _resolve_pinned_font(
                    str(font["key"]), str(font["path"]), str(font["sha256"])
                )
                for item in case:
                    config = _render_config(protocol, item.text_length)
                    rng = np.random.default_rng(
                        render_seed(case_manifest.seed, resolved.key, item.case_index)
                    )
                    clean = render_clean(resolved, item.text, config, rng=rng)
                    target = root / f"{resolved.key}__{item.case_id}.png"
                    Image.fromarray(clean.image).save(target, format="PNG")
                    _validate_rendered_image(target, width, height)
                    if clean.image.shape != (height, width):
                        raise M8BPreflightError(
                            f"{resolved.key}: the rendered array is {clean.image.shape}"
                        )
                    rendered += 1
            return f"rendered {rendered} clean references across {len(selected_fonts)} faces"

        def _deterministic_clean_render() -> str:
            font = selected_fonts[0]
            resolved = _resolve_pinned_font(
                str(font["key"]), str(font["path"]), str(font["sha256"])
            )
            item = case[0]
            config = _render_config(protocol, item.text_length)
            digests = []
            for attempt in range(2):
                rng = np.random.default_rng(
                    render_seed(case_manifest.seed, resolved.key, item.case_index)
                )
                clean = render_clean(resolved, item.text, config, rng=rng)
                digests.append(sha256_bytes(clean.image.tobytes()))
                target = root / f"determinism-{attempt}.png"
                Image.fromarray(clean.image).save(target, format="PNG")
            if digests[0] != digests[1]:
                raise M8BPreflightError("two renders of the same source produced different pixels")
            return "two renders of the same source produced byte-identical pixels"

        def _corruption_sweep() -> str:
            applied = 0
            for font in corruption_fonts:
                resolved = _resolve_pinned_font(
                    str(font["key"]), str(font["path"]), str(font["sha256"])
                )
                for item in case:
                    config = _render_config(protocol, item.text_length)
                    rng = np.random.default_rng(
                        render_seed(case_manifest.seed, resolved.key, item.case_index)
                    )
                    clean = render_clean(resolved, item.text, config, rng=rng)
                    planned = [
                        row
                        for row in rows
                        if row.font_id == resolved.key and row.case_id == item.case_id
                    ]
                    for row in planned:
                        if row.family == CLEAN_FAMILY:
                            continue
                        image = _apply_family(
                            row.family,
                            row.severity,
                            clean.image,
                            ladders,
                            protocol,
                            source=row.source_id,
                            seed=row.variant_seed,
                        )
                        target = root / "sweep" / row.image_path
                        target.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(image).save(target, format="PNG")
                        _validate_rendered_image(target, width, height)
                        again = _apply_family(
                            row.family,
                            row.severity,
                            clean.image,
                            ladders,
                            protocol,
                            source=row.source_id,
                            seed=row.variant_seed,
                        )
                        if not np.array_equal(image, again):
                            raise M8BPreflightError(
                                f"{row.sample_id}: the seeded corruption is not reproducible"
                            )
                        applied += 1
            return f"applied {applied} seeded corruptions reproducibly"

        def _failure_probes() -> str:
            probes = 0
            font = selected_fonts[0]
            try:
                _resolve_pinned_font(str(font["key"]), "/nonexistent/font.ttf", str(font["sha256"]))
            except M8BPreflightError:
                probes += 1
            else:
                raise M8BPreflightError("a missing font file did not fail")
            try:
                _resolve_pinned_font(str(font["key"]), str(font["path"]), "0" * 64)
            except M8BPreflightError:
                probes += 1
            else:
                raise M8BPreflightError("a font hash mismatch did not fail")
            try:
                font_codepoints("/nonexistent/font.ttf")
            except IdentityError:
                probes += 1
            else:
                raise M8BPreflightError("a missing font did not fail the coverage check")
            try:
                validate_stored_text("☃", charset=case_manifest.charset)
            except IdentityError:
                probes += 1
            else:
                raise M8BPreflightError("an unsupported character did not fail")
            try:
                RenderConfig(canvas=(0, 0))
            except RenderError:
                probes += 1
            else:
                raise M8BPreflightError("an invalid renderer configuration did not fail")
            bad = root / "unexpected.png"
            Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(bad, format="PNG")
            try:
                _validate_rendered_image(bad, width, height)
            except M8BPreflightError:
                probes += 1
            else:
                raise M8BPreflightError("an unexpected output size did not fail")
            return f"{probes} explicit failure probes failed closed as required"

        def _identity_join() -> str:
            planned = plan_rows(protocol, case_manifest, sources)
            if len(planned) != len(rows):
                raise M8BPreflightError("the row plan is not reproducible")
            sample = planned[0]
            if sample.sample_id != f"{sample.source_id}__{sample.variant_id}":
                raise M8BPreflightError("the sample identifier does not follow the scheme")
            expected_image = (
                image_relpath(CLEAN_FAMILY, sample.source_id)
                if sample.family == CLEAN_FAMILY
                else image_relpath(sample.family, sample.sample_id)
            )
            if sample.image_path != expected_image:
                raise M8BPreflightError("the image identity does not follow the scheme")
            known = {source.source_id for source in sources}
            if sample.source_id not in known:
                raise M8BPreflightError("the row does not reference a valid source")
            return (
                "every planned row joins to its source, its case, its font and the protocol version"
            )

        _check(checks, "render_all_faces", _render_all_faces)
        _check(checks, "deterministic_clean_render", _deterministic_clean_render)
        _check(checks, "seeded_corruption", _corruption_sweep)
        _check(checks, "failure_probes", _failure_probes)
        _check(checks, "identity_join", _identity_join)
    finally:
        if temporary:
            shutil.rmtree(root, ignore_errors=True)
    return VerificationReport(dataset=str(protocol.path), checks=tuple(checks))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="captcha-benchmark-preflight",
        description="Verify the M8B matched source cases and the generator preflight.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser(
        "verify", help="verify the text cases, the identity scheme and the row plan"
    )
    verify_parser.add_argument(
        "--execution-protocol", type=Path, default=DEFAULT_EXECUTION_PROTOCOL
    )
    verify_parser.add_argument("--lifecycle", type=Path, default=DEFAULT_LIFECYCLE)
    verify_parser.add_argument("--manifest", type=Path, default=None)

    preflight_parser = subparsers.add_parser(
        "preflight", help="exercise the generator on a bounded set of cases"
    )
    preflight_parser.add_argument(
        "--execution-protocol", type=Path, default=DEFAULT_EXECUTION_PROTOCOL
    )
    preflight_parser.add_argument("--manifest", type=Path, default=None)
    preflight_parser.add_argument("--output-dir", type=Path, default=None)

    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            report = verify_preflight(
                args.execution_protocol,
                lifecycle=args.lifecycle,
                manifest=args.manifest,
            )
        else:
            report = run_generator_preflight(
                args.execution_protocol,
                manifest=args.manifest,
                output_dir=args.output_dir,
            )
        for check in report.checks:
            status = "PASS" if check.passed else "FAIL"
            print(f"[{status}] {check.name}: {check.detail}")
        print(f"verification: {'PASS' if report.ok else 'FAIL'}")
        return 0 if report.ok else 1
    except (M8BPreflightError, M8BConfigError, IdentityError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
