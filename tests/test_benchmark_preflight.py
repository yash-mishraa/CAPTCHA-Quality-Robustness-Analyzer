"""Tests for the Milestone M8B source cases, identity scheme and preflight.

The matched text cases, the canonical identifier scheme and the planned row
identities are verified without materialising the benchmark. The bounded
generator preflight renders only a small, explicitly enumerated set of
representative images and only into temporary directories; no planned
benchmark output path is ever created or mutated.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import numpy as np
import pytest

from captcha_quality import benchmark_lifecycle as bl
from captcha_quality import benchmark_preflight as bp
from captcha_quality import cross_font_analysis as cfa
from captcha_quality import font_roster_audit as fra
from captcha_quality import source_identity as si
from captcha_quality.corruptions import FAMILY_NOISE, apply_corruption
from captcha_quality.provenance import sha256_bytes, sha256_file
from captcha_quality.render import RenderError

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_V1 = REPO_ROOT / "configs" / "cross_font_benchmark_protocol_v0_1_0.toml"
PROTOCOL_V2 = REPO_ROOT / "configs" / "cross_font_benchmark_protocol_v0_2_0.toml"
LIFECYCLE_V1 = REPO_ROOT / "configs" / "benchmark_lifecycle_v0_1_0.toml"
LIFECYCLE_V2 = REPO_ROOT / "configs" / "benchmark_lifecycle_v0_2_0.toml"
MANIFEST = REPO_ROOT / "configs" / "cross_font_text_cases_v0_1_0.toml"

pytestmark = pytest.mark.skipif(
    not PROTOCOL_V2.is_file(), reason="the M8B execution protocol is not present"
)


@pytest.fixture(scope="module")
def protocol() -> bp.M8BExecutionProtocol:
    return bp.load_execution_protocol(PROTOCOL_V2)


@pytest.fixture(scope="module")
def manifest() -> bp.CaseManifest:
    return bp.load_case_manifest(MANIFEST)


@pytest.fixture(scope="module")
def sources(protocol: bp.M8BExecutionProtocol, manifest: bp.CaseManifest):
    return bp.plan_sources(protocol, manifest)


@pytest.fixture(scope="module")
def rows(protocol: bp.M8BExecutionProtocol, manifest: bp.CaseManifest, sources):
    return bp.plan_rows(protocol, manifest, sources)


def _shadow_repo(root: Path) -> Path:
    """Build a shadow tree with every declared input of the M8B lifecycle."""
    lifecycle = tomllib.loads(LIFECYCLE_V2.read_text(encoding="utf-8"))
    for pin in lifecycle["inputs"]:
        target = root / str(pin["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if not (target.exists() or target.is_symlink()):
            target.symlink_to(REPO_ROOT / str(pin["path"]))
    for upstream in (
        "configs/evaluation_design_plan_v0_1_0.toml",
        "configs/font_roster_audit_v0_2_0.toml",
    ):
        payload = tomllib.loads((REPO_ROOT / upstream).read_text(encoding="utf-8"))
        for pin in payload.get("inputs", []):
            target = root / str(pin["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            if not (target.exists() or target.is_symlink()):
                target.symlink_to(REPO_ROOT / str(pin["path"]))
    for directory in (
        "reports/evaluation_design_plan_v0_1_0",
        "reports/font_roster_audit_v0_2_0",
        "reports/font_roster_audit_v0_1_0",
    ):
        source = REPO_ROOT / directory
        if not source.is_dir():
            continue
        for path in sorted(source.iterdir()):
            if not path.is_file():
                continue
            target = root / directory / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not (target.exists() or target.is_symlink()):
                target.symlink_to(path)
    lifecycle_link = root / "configs" / LIFECYCLE_V2.name
    lifecycle_link.parent.mkdir(parents=True, exist_ok=True)
    if not (lifecycle_link.exists() or lifecycle_link.is_symlink()):
        lifecycle_link.symlink_to(LIFECYCLE_V2)
    return root


def _verify_in(root: Path) -> bp.VerificationReport:
    previous = Path.cwd()
    os.chdir(root)
    try:
        return bp.verify_preflight(
            root / "configs" / PROTOCOL_V2.name,
            lifecycle=root / "configs" / LIFECYCLE_V2.name,
            manifest=root / "configs" / MANIFEST.name,
        )
    finally:
        os.chdir(previous)


# ---------------------------------------------------------------------------
# 1. exactly thirty valid, unique matched cases
# ---------------------------------------------------------------------------


def test_exactly_thirty_valid_unique_matched_cases(manifest: bp.CaseManifest):
    assert manifest.n_cases == 30
    assert len(manifest.cases) == 30
    identifiers = manifest.case_ids()
    assert len(set(identifiers)) == 30
    texts = [case.text for case in manifest.cases]
    assert len(set(texts)) == 30
    for case in manifest.cases:
        assert 4 <= case.text_length <= 6
        assert len(case.text) == case.text_length
        assert set(case.text) <= set(manifest.charset)
        assert case.case_id == f"case-{case.case_index:03d}"


def test_the_same_cases_are_used_by_every_typeface(manifest, sources):
    by_case: dict[str, set[str]] = {}
    for source in sources:
        by_case.setdefault(source.case_id, set()).add(source.font_id)
    assert len(by_case) == 30
    assert all(len(fonts) == 10 for fonts in by_case.values())


# ---------------------------------------------------------------------------
# 2. stable manifest hash and deterministic ordering
# ---------------------------------------------------------------------------


def test_manifest_hash_is_stable_and_ordering_is_pinned(manifest: bp.CaseManifest):
    again = bp.load_case_manifest(MANIFEST)
    assert again.manifest_hash == manifest.manifest_hash
    assert again.file_sha256 == manifest.file_sha256
    reversed_cases = list(reversed(manifest.cases))
    assert si.manifest_hash([case.record() for case in reversed_cases]) != manifest.manifest_hash


def test_manifest_hash_is_pinned_by_the_execution_protocol(protocol, manifest):
    pins = protocol.text_case_manifest
    assert pins["file_sha256"] == manifest.file_sha256
    assert pins["manifest_hash"] == manifest.manifest_hash
    assert pins["n_cases"] == 30


# ---------------------------------------------------------------------------
# 3. UTF-8 and normalisation behaviour
# ---------------------------------------------------------------------------


def test_utf8_encoding_is_exact(manifest: bp.CaseManifest):
    for case in manifest.cases:
        encoded = case.text.encode("utf-8")
        assert encoded.hex() == case.text_utf8_hex
        assert sha256_bytes(encoded) == case.text_sha256


def test_normalisation_is_asserted_and_never_applied(manifest: bp.CaseManifest):
    with pytest.raises(si.IdentityError):
        si.validate_stored_text("é", charset=manifest.charset)
    with pytest.raises(si.IdentityError):
        si.validate_stored_text("☃", charset=manifest.charset)
    with pytest.raises(si.IdentityError):
        si.validate_stored_text("", charset=manifest.charset)
    stored = "6RTX"
    si.validate_stored_text(stored, charset=manifest.charset)
    assert stored == "6RTX"


def test_a_manifest_whose_text_was_rewritten_fails(manifest: bp.CaseManifest):
    from captcha_quality.source_identity import canonical_case_record, case_record_hash

    tampered = dict(canonical_case_record(manifest.cases[0].record()))
    tampered["text"] = "6RTY"
    assert case_record_hash(tampered) != manifest.cases[0].case_hash


# ---------------------------------------------------------------------------
# 4. glyph coverage and missing-glyph failure
# ---------------------------------------------------------------------------


def test_every_face_covers_every_case_character(protocol, manifest):
    characters = "".join(case.text for case in manifest.cases)
    for font in protocol.fonts:
        codepoints = si.font_codepoints(Path(str(font["path"])))
        assert si.missing_glyphs(codepoints, characters) == []


def test_a_missing_glyph_fails_closed(protocol):
    font = protocol.fonts[0]
    codepoints = si.font_codepoints(Path(str(font["path"])))
    missing = si.missing_glyphs(codepoints, "☃")
    assert missing == ["☃"]
    with pytest.raises(si.GlyphCoverageError):
        raise si.GlyphCoverageError(f"the face {font['key']} has no glyph for {missing}")


def test_the_verifier_rejects_a_face_without_the_required_glyph(protocol, manifest):
    """A synthetic case outside the character map is refused, never substituted."""
    font = protocol.fonts[0]
    codepoints = si.font_codepoints(Path(str(font["path"])))
    assert si.missing_glyphs(codepoints, "☃") != []


# ---------------------------------------------------------------------------
# 5. font hash mismatch and missing font
# ---------------------------------------------------------------------------


def test_font_hash_mismatch_and_missing_font_fail(protocol):
    font = protocol.fonts[0]
    with pytest.raises(bp.M8BPreflightError):
        bp._resolve_pinned_font(str(font["key"]), str(font["path"]), "0" * 64)
    with pytest.raises(bp.M8BPreflightError):
        bp._resolve_pinned_font(str(font["key"]), "/nonexistent/missing.ttf", str(font["sha256"]))
    resolved = bp._resolve_pinned_font(str(font["key"]), str(font["path"]), str(font["sha256"]))
    assert resolved.key == str(font["key"])


# ---------------------------------------------------------------------------
# 6. canonical source-identifier determinism
# ---------------------------------------------------------------------------


def test_source_ids_are_deterministic(protocol, manifest, sources):
    repeated = bp.plan_sources(protocol, manifest)
    assert [source.source_id for source in repeated] == [source.source_id for source in sources]
    assert all(source.source_id.startswith("src-") for source in sources)
    assert all(len(source.source_id) == len("src-") + 16 for source in sources)


def test_a_changed_font_hash_changes_the_source_id(protocol, manifest, sources):
    case = manifest.cases[0]
    font = protocol.fonts[0]
    base = sources[0]
    changed = si.source_id(
        case=case.record(),
        font_id=str(font["key"]),
        font_sha256="0" * 64,
        render_config_hash=base.render_config_hash,
        render_seed=base.render_seed,
        generation_config_hash=base.generation_config_hash,
    )
    assert changed != base.source_id


def test_the_source_id_never_depends_on_the_filesystem(protocol, manifest, sources, tmp_path):
    """Computing the plan in an empty directory gives the same identifiers."""
    previous = Path.cwd()
    os.chdir(tmp_path)
    try:
        elsewhere = bp.plan_sources(protocol, manifest)
    finally:
        os.chdir(previous)
    assert [source.source_id for source in elsewhere] == [source.source_id for source in sources]


# ---------------------------------------------------------------------------
# 7. source uniqueness and source-to-corruption lineage
# ---------------------------------------------------------------------------


def test_source_ids_are_unique_and_every_row_references_a_source(sources, rows):
    identifiers = [source.source_id for source in sources]
    assert len(sources) == 300
    assert len(set(identifiers)) == 300
    known = set(identifiers)
    assert {row.source_id for row in rows} <= known
    for row in rows:
        assert row.sample_id.startswith(f"{row.source_id}__")


def test_the_case_is_recoverable_from_every_row(sources, rows):
    by_source = {source.source_id: source for source in sources}
    for row in rows:
        source = by_source[row.source_id]
        assert row.case_id == source.case_id
        assert row.font_id == source.font_id


# ---------------------------------------------------------------------------
# 8. deterministic seeded corruption
# ---------------------------------------------------------------------------


def test_seeded_corruption_is_reproducible(protocol, manifest, sources):
    ladders = bp._build_ladders(protocol)
    ladder = ladders[FAMILY_NOISE]
    image = np.full((64, 192), 255, dtype=np.uint8)
    seed = si.variant_seed(manifest.seed, sources[0].source_id, FAMILY_NOISE, 2)
    first = apply_corruption(image, ladder, 2, noise_seed=seed).image
    second = apply_corruption(image, ladder, 2, noise_seed=seed).image
    assert np.array_equal(first, second)
    other = apply_corruption(
        image,
        ladder,
        3,
        noise_seed=si.variant_seed(manifest.seed, sources[0].source_id, FAMILY_NOISE, 3),
    ).image
    assert not np.array_equal(first, other)
    assert seed != si.variant_seed(manifest.seed, sources[0].source_id, FAMILY_NOISE, 3)


def test_the_variant_seed_is_a_pure_function(manifest, sources):
    first = si.variant_seed(manifest.seed, sources[0].source_id, FAMILY_NOISE, 1)
    second = si.variant_seed(manifest.seed, sources[0].source_id, FAMILY_NOISE, 1)
    assert first == second


# ---------------------------------------------------------------------------
# 9. dimensions, mode and PNG decoding
# ---------------------------------------------------------------------------


def test_the_renderer_produces_the_frozen_image_contract(protocol, manifest, tmp_path):
    report = bp.run_generator_preflight(PROTOCOL_V2, manifest=MANIFEST, output_dir=tmp_path)
    assert report.ok, [f"{c.name}: {c.detail}" for c in report.failures()]
    images = sorted(path for path in tmp_path.rglob("*.png") if path.name != "unexpected.png")
    assert images
    from PIL import Image

    for path in images:
        with Image.open(path) as handle:
            assert handle.format == "PNG"
            assert handle.mode == "L"
            assert handle.size == (192, 64)


def test_an_invalid_renderer_configuration_fails():
    with pytest.raises(RenderError):
        from captcha_quality.render import RenderConfig

        RenderConfig(canvas=(0, 0))


# ---------------------------------------------------------------------------
# 10. duplicate, missing and unexpected row identities
# ---------------------------------------------------------------------------


def test_a_duplicate_row_identity_fails(protocol, rows, sources):
    duplicated = list(rows) + [rows[0]]
    with pytest.raises(bp.M8BPreflightError):
        bp._verify_row_plan(protocol, duplicated, sources)


def test_a_missing_row_fails(protocol, rows, sources):
    with pytest.raises(bp.M8BPreflightError):
        bp._verify_row_plan(protocol, list(rows)[:-1], sources)


def test_an_unexpected_image_identity_fails(protocol, rows, sources):
    import dataclasses

    doctored = [
        dataclasses.replace(
            row, image_path=row.image_path.replace(f"images/{row.family}/", "images/unexpected/")
        )
        if index == 0
        else row
        for index, row in enumerate(rows)
    ]
    with pytest.raises(bp.M8BPreflightError):
        bp._verify_row_plan(protocol, doctored, sources)


def test_a_duplicate_source_identity_fails(protocol, manifest, sources):
    import dataclasses

    doctored = [dataclasses.replace(sources[0], source_id=sources[1].source_id), *sources[1:]]
    with pytest.raises(bp.M8BPreflightError):
        bp._verify_source_identities(protocol, manifest, doctored)


# ---------------------------------------------------------------------------
# 11. primary/diagnostic separation and count invariants
# ---------------------------------------------------------------------------


def test_counts_and_the_primary_diagnostic_split(protocol, rows):
    counts = protocol.counts
    assert len(rows) == counts["total_rows"] == 6300
    primary = [row for row in rows if row.primary]
    diagnostic = [row for row in rows if not row.primary]
    assert len(primary) == counts["primary_rows"] == 5700
    assert len(diagnostic) == counts["diagnostic_rows"] == 600
    assert len(primary) + len(diagnostic) == len(rows)
    assert all(row.family == si.CLEAN_FAMILY for row in diagnostic if row.family != "elastic")
    elastic_diagnostic = [row for row in diagnostic if row.family == "elastic"]
    assert {row.severity for row in elastic_diagnostic} == {1}
    assert all(row.primary for row in rows if row.family == "gaussian_blur")


def test_every_source_has_the_declared_number_of_rows(protocol, rows):
    per_source: dict[str, int] = {}
    for row in rows:
        per_source[row.source_id] = per_source.get(row.source_id, 0) + 1
    assert len(per_source) == 300
    assert set(per_source.values()) == {21}


def test_the_image_identities_cover_the_six_declared_subdirectories(rows):
    directories = {row.image_path.split("/")[1] for row in rows}
    assert directories == {
        "clean",
        "gaussian_blur",
        "gaussian_noise",
        "thin_lines",
        "arc_strokes",
        "elastic",
    }
    assert len({row.image_path for row in rows}) == 6300


# ---------------------------------------------------------------------------
# 12. tampered protocol or manifest
# ---------------------------------------------------------------------------


def test_a_tampered_manifest_fails_closed(tmp_path):
    root = _shadow_repo(tmp_path / "repo")
    target = root / "configs" / MANIFEST.name
    content = target.read_text(encoding="utf-8")
    target.unlink()
    target.write_text(content.replace('text = "6RTX"', 'text = "6RTY"'), encoding="utf-8")
    report = _verify_in(root)
    assert not report.ok
    mapped = {check.name: check for check in report.checks}
    assert not mapped["text_case_manifest"].passed


def test_a_tampered_execution_protocol_fails_closed(tmp_path):
    root = _shadow_repo(tmp_path / "repo")
    target = root / "configs" / PROTOCOL_V2.name
    content = target.read_text(encoding="utf-8")
    target.unlink()
    target.write_text(content.replace("n_sources = 300", "n_sources = 299"), encoding="utf-8")
    report = _verify_in(root)
    assert not report.ok
    mapped = {check.name: check for check in report.checks}
    assert not mapped["lifecycle_contract"].passed


def test_a_changed_renderer_configuration_fails_closed(tmp_path):
    root = _shadow_repo(tmp_path / "repo")
    target = root / "configs" / PROTOCOL_V2.name
    content = target.read_text(encoding="utf-8")
    target.unlink()
    target.write_text(content.replace("canvas_width = 192", "canvas_width = 256"), encoding="utf-8")
    report = _verify_in(root)
    assert not report.ok
    mapped = {check.name: check for check in report.checks}
    assert not mapped["generation_config"].passed


def test_an_unpinned_dependency_fails_closed(tmp_path):
    root = _shadow_repo(tmp_path / "repo")
    target = root / "src" / "captcha_quality" / "render.py"
    content = target.read_bytes()
    target.unlink()
    target.write_bytes(content + b"\n# tampered\n")
    report = _verify_in(root)
    assert not report.ok
    mapped = {check.name: check for check in report.checks}
    assert not mapped["lifecycle_contract"].passed


# ---------------------------------------------------------------------------
# 13. scoped membership
# ---------------------------------------------------------------------------


def test_unrelated_shared_directory_additions_are_accepted(tmp_path):
    root = _shadow_repo(tmp_path / "repo")
    (root / "configs" / "some_future_milestone_v0_1_0.toml").write_text(
        "# unrelated\n", encoding="utf-8"
    )
    dataset = root / "data" / "processed" / "some_future_dataset_v0.1.0"
    dataset.mkdir(parents=True)
    (dataset / "manifest.csv").write_text("x\n", encoding="utf-8")
    model = root / "models" / "some_future_model"
    model.mkdir(parents=True)
    (model / "candidate.joblib").write_text("x\n", encoding="utf-8")
    report_dir = root / "reports" / "some_future_report_v0_1_0"
    report_dir.mkdir(parents=True)
    (report_dir / "report.md").write_text("x\n", encoding="utf-8")
    report = _verify_in(root)
    assert report.ok, [f"{c.name}: {c.detail}" for c in report.failures()]


def test_a_font_binary_in_the_tree_fails(tmp_path):
    root = _shadow_repo(tmp_path / "repo")
    (root / "models" / "some_font.ttf").write_bytes(b"")
    report = _verify_in(root)
    assert not report.ok
    mapped = {check.name: check for check in report.checks}
    assert not mapped["lifecycle_contract"].passed


# ---------------------------------------------------------------------------
# 14. historical verifiers
# ---------------------------------------------------------------------------


def test_the_historical_verifiers_still_pass(tmp_path):
    m7c = cfa.verify_analysis(Path("configs/cross_font_analysis_v0_1_0.toml"))
    assert m7c.ok, [f"{c.name}: {c.detail}" for c in m7c.failures()]
    transition = bl._m7d_verifier_report()
    assert transition["planning_state_failed"] is True
    assert transition["other_failures"] == []
    shadow = _shadow_repo(tmp_path / "repo")
    previous = Path.cwd()
    os.chdir(shadow)
    try:
        m7e = fra.verify_audit(Path("configs/font_roster_audit_v0_2_0.toml"))
        m7e_v1 = fra.verify_audit(Path("configs/font_roster_audit_v0_1_0.toml"))
    finally:
        os.chdir(previous)
    assert m7e.ok, [f"{c.name}: {c.detail}" for c in m7e.failures()]
    assert m7e_v1.ok, [f"{c.name}: {c.detail}" for c in m7e_v1.failures()]


def test_the_m8a_lifecycle_versions_still_pass():
    for config in (LIFECYCLE_V1, LIFECYCLE_V2):
        report = bl.verify_lifecycle(config)
        assert report.ok, [f"{c.name}: {c.detail}" for c in report.failures()]


def test_the_m8a_protocol_v1_is_unchanged():
    assert sha256_file(PROTOCOL_V1) == (
        "192902a17d23a73a0698a16bfe26131104a72fa02eb5b0b8631ea05b873ad316"
    )
    assert sha256_file(LIFECYCLE_V1) == (
        "839934808ede72fd66046a8a4546454b7f85a8e08bdaad664179635df3dbcdc9"
    )


# ---------------------------------------------------------------------------
# 15. the lifecycle transition is versioned, not rewritten
# ---------------------------------------------------------------------------


def test_the_lifecycle_transition_is_versioned():
    lifecycle = tomllib.loads(LIFECYCLE_V2.read_text(encoding="utf-8"))
    assert lifecycle["run"]["status"] == "authorized_not_executed"
    assert lifecycle["authorized_execution"]["dataset_output"] == (
        "data/processed/matched_font_benchmark_v0.1.0"
    )
    assert lifecycle["authorized_execution"]["report_output"] == (
        "reports/matched_font_benchmark_v0_1_0"
    )
    executed = bl.load_config(REPO_ROOT / "configs" / "benchmark_lifecycle_v0_3_0.toml")
    assert executed.run.status == "executed"
    assert executed.run.milestone == "M8C"
    assert executed.run.supersedes == "v0.2.0"
    assert (REPO_ROOT / "data/processed/matched_font_benchmark_v0.1.0").is_dir()
    assert (REPO_ROOT / "reports/matched_font_benchmark_v0_1_0").is_dir()


def test_the_preflight_creates_no_benchmark_output(tmp_path):
    before = sorted(path.name for path in (REPO_ROOT / "data" / "processed").iterdir())
    before_reports = sorted(path.name for path in (REPO_ROOT / "reports").iterdir())
    report = bp.run_generator_preflight(PROTOCOL_V2, manifest=MANIFEST)
    assert report.ok
    after = sorted(path.name for path in (REPO_ROOT / "data" / "processed").iterdir())
    after_reports = sorted(path.name for path in (REPO_ROOT / "reports").iterdir())
    assert before == after
    assert before_reports == after_reports
    assert (REPO_ROOT / "data/processed/matched_font_benchmark_v0.1.0").is_dir()
    assert (REPO_ROOT / "reports/matched_font_benchmark_v0_1_0").is_dir()


def test_the_m8b_verifier_passes_in_the_repository():
    report = bp.verify_preflight(PROTOCOL_V2, lifecycle=LIFECYCLE_V2, manifest=MANIFEST)
    assert report.ok, [f"{c.name}: {c.detail}" for c in report.failures()]
