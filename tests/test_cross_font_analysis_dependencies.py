"""Regression tests for the M7C scoped protected-dependency contract.

M7C originally fingerprinted the whole ``configs/``, ``data/processed/`` and
``models/`` trees. That made its verification depend on the complete future
contents of shared directories M7C never declared as inputs, so adding the
unrelated M7D planning configuration failed M7C verification even though every
M7C input, table, prediction, metric and report artifact was byte-identical.

These tests pin the replacement contract: M7C verification is scoped to the
dependencies M7C actually declares, and a modified, deleted or substituted
declared dependency is still detected. The tests never mutate the real frozen
repository artifacts; they operate on isolated temporary shadow trees.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("sklearn")

from captcha_quality import cross_font_analysis as cfa
from captcha_quality.provenance import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "cross_font_analysis_v0_1_0.toml"
M7B_DATA_ROOT = REPO_ROOT / "data" / "processed" / "heldout_font_benchmark_v0.1.0"
M7C_REPORT_ROOT = REPO_ROOT / "reports" / "cross_font_analysis_v0_1_0"

if not M7B_DATA_ROOT.exists() or not M7C_REPORT_ROOT.exists():  # pragma: no cover
    pytest.skip(
        "frozen M7C inputs are not present; generate the upstream artifacts first",
        allow_module_level=True,
    )


def _real_membership() -> cfa.ProtectedDependencyMembership:
    return cfa.protected_dependency_membership()


def _declared_members() -> list[str]:
    """Every file the real membership declares, without touching the disk."""
    membership = _real_membership()
    members = list(membership.config_files)
    for root in membership.roots:
        base = REPO_ROOT / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(REPO_ROOT).as_posix()
            if "/images/" in relative:
                continue
            if cfa._is_ignored_member(relative, membership):
                continue
            members.append(relative)
    return sorted(members)


@pytest.fixture()
def shadow_repo(tmp_path: Path) -> Path:
    """Build an isolated shadow repository that mirrors the frozen M7C tree.

    Files are symlinked so the fixture stays cheap and, crucially, so that no
    test can mutate the real frozen artifacts. The M7C output directory is a
    real directory whose members are symlinks, so a test can replace a single
    symlink with a modified copy when it needs to simulate tampering.
    """
    root = tmp_path / "m7c_shadow"
    for relative in _declared_members():
        source = REPO_ROOT / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
    output_dir = root / "reports" / "cross_font_analysis_v0_1_0"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in cfa.M7C_OUTPUT_FILES:
        source = M7C_REPORT_ROOT / name
        target = output_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
    return root


def _replace_with_copy(path: Path, content: bytes) -> None:
    """Replace a symlinked member with a real, modified copy."""
    path.unlink()
    path.write_bytes(content)


def _verify_in(shadow: Path) -> cfa.VerificationReport:
    """Verify M7C inside the shadow tree.

    The config path is passed relative to the shadow working directory, which
    is how the CLI invokes it, so the recomputed report embeds the same
    relative configuration path as the stored one.
    """
    previous = Path.cwd()
    os.chdir(shadow)
    try:
        return cfa.verify_analysis(Path("configs/cross_font_analysis_v0_1_0.toml"))
    except cfa.CrossFontError as error:
        return cfa.VerificationReport(
            dataset=str(shadow),
            checks=(
                cfa.CheckResult("declared_dependency", False, f"{type(error).__name__}: {error}"),
            ),
        )
    finally:
        os.chdir(previous)


def _check_map(report: cfa.VerificationReport) -> dict[str, cfa.CheckResult]:
    return {check.name: check for check in report.checks}


# ---------------------------------------------------------------------------
# Unit level: the scoped membership helpers, fully isolated
# ---------------------------------------------------------------------------


def _tiny_membership(tmp_path: Path) -> cfa.ProtectedDependencyMembership:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "a.toml").write_text("a = 1\n", encoding="utf-8")
    (tmp_path / "configs" / "b.toml").write_text("b = 2\n", encoding="utf-8")
    dataset = tmp_path / "data" / "processed" / "ds"
    dataset.mkdir(parents=True)
    (dataset / "manifest.csv").write_text("x\n", encoding="utf-8")
    (dataset / "sources.csv").write_text("y\n", encoding="utf-8")
    scratch = tmp_path / "data" / "processed" / "scratch"
    scratch.mkdir()
    (scratch / "notes.txt").write_text("transient\n", encoding="utf-8")
    return cfa.ProtectedDependencyMembership(
        version="test-v1",
        config_files=("configs/a.toml",),
        dataset_roots=("data/processed/ds",),
        model_roots=(),
        report_roots=(),
        ignored_roots=("data/processed/scratch",),
        output_files=(),
    )


def test_scoped_fingerprint_covers_only_declared_members(tmp_path, monkeypatch):
    membership = _tiny_membership(tmp_path)
    monkeypatch.chdir(tmp_path)
    fingerprint = cfa._scoped_fingerprint(membership)
    assert set(fingerprint) == {
        "configs/a.toml",
        "data/processed/ds/manifest.csv",
        "data/processed/ds/sources.csv",
    }


def test_unrelated_new_config_does_not_change_scoped_fingerprint(tmp_path, monkeypatch):
    membership = _tiny_membership(tmp_path)
    monkeypatch.chdir(tmp_path)
    before = cfa._scoped_fingerprint(membership)
    (tmp_path / "configs" / "unrelated.toml").write_text("c = 3\n", encoding="utf-8")
    after = cfa._scoped_fingerprint(membership)
    assert before == after


def test_unrelated_new_dataset_root_does_not_change_scoped_fingerprint(tmp_path, monkeypatch):
    membership = _tiny_membership(tmp_path)
    monkeypatch.chdir(tmp_path)
    before = cfa._scoped_fingerprint(membership)
    other = tmp_path / "data" / "processed" / "other_dataset"
    other.mkdir()
    (other / "manifest.csv").write_text("z\n", encoding="utf-8")
    assert cfa._scoped_fingerprint(membership) == before


def test_scoped_fingerprint_detects_modified_declared_member(tmp_path, monkeypatch):
    membership = _tiny_membership(tmp_path)
    monkeypatch.chdir(tmp_path)
    recorded = cfa._scoped_fingerprint(membership)
    (tmp_path / "configs" / "a.toml").write_text("a = 99\n", encoding="utf-8")
    current = cfa._scoped_fingerprint(membership)
    mismatches = cfa._scoped_fingerprint_mismatches(recorded, current)
    assert mismatches["modified"] == ["configs/a.toml"]
    assert not mismatches["missing"]
    assert not mismatches["unexpected"]


def test_scoped_fingerprint_detects_missing_declared_member(tmp_path, monkeypatch):
    membership = _tiny_membership(tmp_path)
    monkeypatch.chdir(tmp_path)
    recorded = cfa._scoped_fingerprint(membership)
    (tmp_path / "data" / "processed" / "ds" / "sources.csv").unlink()
    current = cfa._scoped_fingerprint(membership)
    mismatches = cfa._scoped_fingerprint_mismatches(recorded, current)
    assert mismatches["missing"] == ["data/processed/ds/sources.csv"]


def test_scoped_fingerprint_detects_unexpected_member_in_declared_root(tmp_path, monkeypatch):
    membership = _tiny_membership(tmp_path)
    monkeypatch.chdir(tmp_path)
    recorded = cfa._scoped_fingerprint(membership)
    (tmp_path / "data" / "processed" / "ds" / "extra.csv").write_text("e\n", encoding="utf-8")
    current = cfa._scoped_fingerprint(membership)
    mismatches = cfa._scoped_fingerprint_mismatches(recorded, current)
    assert mismatches["unexpected"] == ["data/processed/ds/extra.csv"]


def test_expected_scoped_fingerprint_uses_recorded_hashes_only(tmp_path, monkeypatch):
    membership = _tiny_membership(tmp_path)
    monkeypatch.chdir(tmp_path)
    recorded = {
        "configs/a.toml": "a" * 64,
        "configs/b.toml": "b" * 64,
        "data/processed/ds/manifest.csv": "c" * 64,
        "data/processed/ds/sources.csv": "d" * 64,
        "data/processed/scratch/notes.txt": "e" * 64,
    }
    expected = cfa._expected_scoped_fingerprint(recorded, membership)
    assert expected == {
        "configs/a.toml": "a" * 64,
        "data/processed/ds/manifest.csv": "c" * 64,
        "data/processed/ds/sources.csv": "d" * 64,
    }


def test_expected_scoped_fingerprint_is_not_derived_from_the_current_tree(tmp_path, monkeypatch):
    """A modified member must still mismatch against the recorded hash."""
    membership = _tiny_membership(tmp_path)
    monkeypatch.chdir(tmp_path)
    recorded = {"configs/a.toml": "0" * 64}
    expected = cfa._expected_scoped_fingerprint(recorded, membership)
    current = cfa._scoped_fingerprint(membership)
    mismatches = cfa._scoped_fingerprint_mismatches(expected, current)
    assert mismatches["modified"] == ["configs/a.toml"]


# ---------------------------------------------------------------------------
# Integration level: the real M7C verifier, against an isolated shadow tree
# ---------------------------------------------------------------------------


def test_m7c_verification_passes_with_declared_dependencies_intact(shadow_repo):
    report = _verify_in(shadow_repo)
    assert report.ok, [f"{c.name}: {c.detail}" for c in report.failures()]
    checks = _check_map(report)
    assert checks["frozen_dependency_manifest"].passed
    assert checks["frozen_dependencies"].passed
    assert checks["output_file_set"].passed


def test_m7c_verification_passes_in_the_real_repository():
    """The real repository, including the unrelated M7D configuration."""
    report = cfa.verify_analysis(Path("configs/cross_font_analysis_v0_1_0.toml"))
    assert report.ok, [f"{c.name}: {c.detail}" for c in report.failures()]
    checks = _check_map(report)
    assert checks["frozen_dependencies"].passed
    assert "unchanged" in checks["frozen_dependencies"].detail


def test_adding_an_unrelated_config_does_not_invalidate_m7c(shadow_repo):
    before = _verify_in(shadow_repo)
    assert before.ok
    unrelated = shadow_repo / "configs" / "some_future_milestone_v0_1_0.toml"
    unrelated.write_text("# unrelated future configuration\n", encoding="utf-8")
    after = _verify_in(shadow_repo)
    assert after.ok, [f"{c.name}: {c.detail}" for c in after.failures()]
    assert _check_map(after)["frozen_dependencies"].passed


def test_adding_an_unrelated_config_to_the_real_configs_directory_is_not_a_dependency():
    """The real M7D configuration is present and is not an M7C dependency."""
    membership = _real_membership()
    assert "configs/evaluation_design_plan_v0_1_0.toml" not in membership.config_files
    for root in membership.roots:
        assert not root.startswith("configs")
    assert (REPO_ROOT / "configs" / "evaluation_design_plan_v0_1_0.toml").is_file()
    expected = cfa._expected_scoped_fingerprint(
        json.loads((M7C_REPORT_ROOT / "run_metadata.json").read_text(encoding="utf-8"))[
            "frozen_assets"
        ],
        membership,
    )
    assert "configs/evaluation_design_plan_v0_1_0.toml" not in expected


def test_modifying_a_pinned_input_fails_verification(shadow_repo):
    pinned = shadow_repo / "configs" / "cross_font_analysis_v0_1_0.toml"
    _replace_with_copy(pinned, pinned.read_bytes() + b"\n# tampered\n")
    report = _verify_in(shadow_repo)
    assert not report.ok
    assert any(not check.passed for check in report.checks)


def test_modifying_a_pinned_report_input_fails_verification(shadow_repo):
    pinned = shadow_repo / "reports" / "heldout_font_benchmark_v0_1_0" / "predictions.csv"
    _replace_with_copy(pinned, pinned.read_bytes() + b"tampered\n")
    report = _verify_in(shadow_repo)
    assert not report.ok
    assert any(not check.passed for check in report.checks)


def test_deleting_a_required_pinned_input_fails_verification(shadow_repo):
    pinned = shadow_repo / "reports" / "heldout_font_benchmark_v0_1_0" / "predictions.csv"
    pinned.unlink()
    report = _verify_in(shadow_repo)
    assert not report.ok
    assert any(not check.passed for check in report.checks)


def test_deleting_a_declared_dependency_member_fails_verification(shadow_repo):
    """A declared but non-pinned dependency is still protected.

    The M3 ground-truth table is part of the declared membership but is not one
    of M7C's 31 hash-pinned inputs, so it exercises the scoped fingerprint
    rather than the pinned-input guard.
    """
    member = shadow_repo / "data" / "processed" / "captcha_quality_v0.1.0" / "ground_truth.csv"
    member.unlink()
    report = _verify_in(shadow_repo)
    assert not report.ok
    checks = _check_map(report)
    assert not checks["frozen_dependencies"].passed
    assert "missing" in checks["frozen_dependencies"].detail


def test_altering_a_protected_m7c_output_artifact_fails_verification(shadow_repo):
    target = shadow_repo / "reports" / "cross_font_analysis_v0_1_0" / "font_metrics.csv"
    _replace_with_copy(target, b"font_id,rmse\nghost,0.0\n")
    report = _verify_in(shadow_repo)
    assert not report.ok
    checks = _check_map(report)
    assert not checks["table:font_metrics.csv"].passed
    assert not checks["output_hashes"].passed


def test_adding_an_unexpected_output_artifact_fails_verification(shadow_repo):
    extra = shadow_repo / "reports" / "cross_font_analysis_v0_1_0" / "unexpected.csv"
    extra.write_text("x\n", encoding="utf-8")
    report = _verify_in(shadow_repo)
    assert not report.ok
    assert not _check_map(report)["output_file_set"].passed


def test_missing_dependency_manifest_fails_closed(shadow_repo):
    (shadow_repo / "reports" / "cross_font_analysis_v0_1_0" / "run_metadata.json").unlink()
    report = _verify_in(shadow_repo)
    assert not report.ok
    assert not _check_map(report)["metadata"].passed


def test_malformed_dependency_manifest_fails_closed(shadow_repo):
    manifest = shadow_repo / "reports" / "cross_font_analysis_v0_1_0" / "run_metadata.json"
    _replace_with_copy(manifest, b"{not json")
    report = _verify_in(shadow_repo)
    assert not report.ok
    assert not _check_map(report)["metadata"].passed


def test_tampered_dependency_manifest_snapshot_fails_closed(shadow_repo):
    """Editing the recorded snapshot must be detected, not silently accepted."""
    manifest = shadow_repo / "reports" / "cross_font_analysis_v0_1_0" / "run_metadata.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["frozen_assets"]["configs/cross_font_analysis_v0_1_0.toml"] = "0" * 64
    _replace_with_copy(manifest, json.dumps(payload, indent=2, sort_keys=True).encode())
    report = _verify_in(shadow_repo)
    assert not report.ok
    assert not _check_map(report)["frozen_dependency_manifest"].passed


def test_repeated_verification_is_deterministic(shadow_repo):
    first = _verify_in(shadow_repo)
    second = _verify_in(shadow_repo)
    assert first.ok and second.ok
    assert [(c.name, c.passed) for c in first.checks] == [(c.name, c.passed) for c in second.checks]


def test_m7d_configuration_and_verification_are_unaffected():
    """The M7D milestone keeps its own configuration, hashes and verifier."""
    from captcha_quality import evaluation_plan as ep

    m7d_config = REPO_ROOT / "configs" / "evaluation_design_plan_v0_1_0.toml"
    assert m7d_config.is_file()
    config = ep.load_config(m7d_config)
    assert config.config_sha256 == (
        "e9f236b383172217e85096c1e06e81adcbb3fd85dc5c18f6537d824e1c6300d3"
    )
    assert config.config_hash == (
        "6464989d0c165d4749d2ece04bd7f8435d3499edf38e4dcd5f2097abd44b4537"
    )
    report = ep.verify_plan(m7d_config)
    assert report.ok, [f"{c.name}: {c.detail}" for c in report.failures()]


def test_membership_membership_is_documented_and_declared():
    membership = _real_membership()
    assert membership.version == "m7c-scoped-dependencies-v1"
    assert "configs/cross_font_analysis_v0_1_0.toml" in membership.config_files
    assert "data/processed/heldout_font_benchmark_v0.1.0" in membership.dataset_roots
    assert "models/retention_aware_selection_v0_1_0" in membership.model_roots
    assert membership.report_roots == cfa.PROTECTED_REPORT_ROOTS
    assert "data/processed/repro_a" in membership.ignored_roots
    assert set(membership.output_files) == set(cfa.M7C_OUTPUT_FILES)


def test_declared_membership_is_covered_by_the_recorded_snapshot():
    """Every declared member must exist in the M7C run-time record."""
    recorded = json.loads((M7C_REPORT_ROOT / "run_metadata.json").read_text(encoding="utf-8"))[
        "frozen_assets"
    ]
    membership = _real_membership()
    expected = cfa._expected_scoped_fingerprint(recorded, membership)
    assert expected, "the recorded snapshot must cover the declared membership"
    for relative in membership.config_files:
        assert relative in expected
    for root in membership.roots:
        members = [path for path in expected if path.startswith(f"{root}/")]
        assert members, f"declared root has no recorded members: {root}"


def test_shadowed_members_are_symlinks_and_the_real_tree_is_untouched(shadow_repo):
    """The fixture never materialises the real frozen artifacts."""
    for relative in _declared_members():
        path = shadow_repo / relative
        if path.is_file():
            assert path.is_symlink() or relative.startswith("reports/cross_font")
    for name in cfa.M7C_OUTPUT_FILES:
        path = shadow_repo / "reports" / "cross_font_analysis_v0_1_0" / name
        assert path.is_symlink()
    # the real tree is unchanged by the fixture
    assert (
        sha256_file(REPO_ROOT / "configs" / "cross_font_analysis_v0_1_0.toml")
        == (
            json.loads((M7C_REPORT_ROOT / "run_metadata.json").read_text(encoding="utf-8"))[
                "analysis"
            ]["config_sha256"]
        )
    )


def test_scratch_fixture_cleanup_is_safe(shadow_repo):
    """The shadow tree can be removed without touching the repository."""
    unrelated = shadow_repo / "configs" / "some_future_milestone_v0_1_0.toml"
    if unrelated.exists():
        unrelated.unlink()
    assert (REPO_ROOT / "configs" / "cross_font_analysis_v0_1_0.toml").is_file()
    assert (REPO_ROOT / "configs" / "evaluation_design_plan_v0_1_0.toml").is_file()
