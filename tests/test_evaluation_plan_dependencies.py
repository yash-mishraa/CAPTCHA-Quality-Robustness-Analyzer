"""Regression tests for the M7D scoped protected-dependency contract.

M7D originally fingerprinted the whole ``configs/``, ``data/processed/`` and
``models/`` trees. That made its verification depend on the complete future
contents of shared directories M7D never declared as inputs, so adding an
unrelated milestone configuration would fail M7D verification even though every
M7D input, table, manifest and report artifact was byte-identical.

These tests pin the replacement contract: M7D verification is scoped to the
dependencies M7D actually declares, and a modified, deleted or substituted
declared dependency is still detected. The tests operate on isolated temporary
shadow trees and never mutate the real frozen repository artifacts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("sklearn")

from captcha_quality import cross_font_analysis as cfa
from captcha_quality import evaluation_plan as ep
from captcha_quality.provenance import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "evaluation_design_plan_v0_1_0.toml"
M7B_DATA_ROOT = REPO_ROOT / "data" / "processed" / "heldout_font_benchmark_v0.1.0"
M7D_REPORT_ROOT = REPO_ROOT / "reports" / "evaluation_design_plan_v0_1_0"
M7C_CONFIG_PATH = REPO_ROOT / "configs" / "cross_font_analysis_v0_1_0.toml"

if not M7B_DATA_ROOT.exists() or not M7D_REPORT_ROOT.exists():  # pragma: no cover
    pytest.skip(
        "frozen M7D inputs are not present; generate the upstream artifacts first",
        allow_module_level=True,
    )


def _declared_members() -> list[str]:
    """The M7D planning configuration plus every pinned upstream dependency."""
    config = ep.load_config(CONFIG_PATH)
    return [ep.M7D_PLANNING_CONFIG_FILE, *(pin.path for pin in config.inputs)]


@pytest.fixture()
def shadow_repo(tmp_path: Path) -> Path:
    """Build an isolated shadow repository mirroring the frozen M7D tree.

    Members are symlinked so the fixture stays cheap and, crucially, so that no
    test can mutate the real frozen artifacts. The M7D output directory is a
    real directory whose members are symlinks, so a test can replace a single
    symlink with a modified copy when it needs to simulate tampering.
    """
    root = tmp_path / "m7d_shadow"
    for relative in _declared_members():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(REPO_ROOT / relative)
    output_dir = root / "reports" / "evaluation_design_plan_v0_1_0"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ep.M7D_OUTPUT_FILES:
        target = output_dir / name
        target.symlink_to(M7D_REPORT_ROOT / name)
    return root


def _replace_with_copy(path: Path, content: bytes) -> None:
    """Replace a symlinked member with a real, modified copy."""
    path.unlink()
    path.write_bytes(content)


def _verify_in(shadow: Path) -> ep.VerificationReport:
    """Verify M7D inside the shadow tree.

    The config path is passed relative to the shadow working directory, which
    is how the CLI invokes it, so the recomputed report embeds the same
    relative configuration path as the stored one.
    """
    previous = Path.cwd()
    os.chdir(shadow)
    try:
        return ep.verify_plan(Path("configs/evaluation_design_plan_v0_1_0.toml"))
    except (ep.PlanError, ep.PlanConfigError) as error:
        return ep.VerificationReport(
            dataset=str(shadow),
            checks=(ep.CheckResult("declared_dependency", False, str(error)),),
        )
    finally:
        os.chdir(previous)


def _check_map(report: ep.VerificationReport) -> dict[str, ep.CheckResult]:
    return {check.name: check for check in report.checks}


# ---------------------------------------------------------------------------
# 1. verification passes with all declared dependencies intact
# ---------------------------------------------------------------------------


def test_m7d_verification_passes_with_declared_dependencies_intact(shadow_repo):
    report = _verify_in(shadow_repo)
    assert report.ok, [f"{c.name}: {c.detail}" for c in report.failures()]
    checks = _check_map(report)
    assert checks["frozen_dependency_manifest"].passed
    assert checks["frozen_dependency_anchors"].passed
    assert checks["frozen_dependencies"].passed
    assert checks["output_file_set"].passed
    assert checks["config_sha256"].passed
    assert checks["config_hash"].passed
    assert checks["input_hashes"].passed
    assert checks["output_hashes"].passed


def test_m7d_verification_passes_in_the_real_repository():
    """The real repository, including the unrelated M7C configuration."""
    report = ep.verify_plan(Path("configs/evaluation_design_plan_v0_1_0.toml"))
    assert report.ok, [f"{c.name}: {c.detail}" for c in report.failures()]
    checks = _check_map(report)
    assert checks["frozen_dependencies"].passed
    assert "unchanged" in checks["frozen_dependencies"].detail


# ---------------------------------------------------------------------------
# 2/3. unrelated additions to shared directories are accepted
# ---------------------------------------------------------------------------


def test_adding_an_unrelated_config_does_not_invalidate_m7d(shadow_repo):
    (shadow_repo / "configs" / "future_milestone_v0_1_0.toml").write_text(
        "# an unrelated future configuration\n", encoding="utf-8"
    )
    report = _verify_in(shadow_repo)
    assert report.ok, [f"{c.name}: {c.detail}" for c in report.failures()]
    assert _check_map(report)["frozen_dependencies"].passed


def test_adding_an_unrelated_file_to_another_shared_directory_is_accepted(shadow_repo):
    dataset = shadow_repo / "data" / "processed" / "some_future_dataset_v0.1.0"
    dataset.mkdir(parents=True)
    (dataset / "manifest.csv").write_text("x\n", encoding="utf-8")
    model = shadow_repo / "models" / "some_future_model"
    model.mkdir(parents=True)
    (model / "candidate.joblib").write_text("x\n", encoding="utf-8")
    report = _verify_in(shadow_repo)
    assert report.ok, [f"{c.name}: {c.detail}" for c in report.failures()]


def test_adding_an_unrelated_report_directory_is_accepted(shadow_repo):
    report_dir = shadow_repo / "reports" / "some_future_report_v0_1_0"
    report_dir.mkdir(parents=True)
    (report_dir / "report.md").write_text("x\n", encoding="utf-8")
    report = _verify_in(shadow_repo)
    assert report.ok, [f"{c.name}: {c.detail}" for c in report.failures()]


def test_an_unrelated_addition_is_not_a_declared_dependency():
    config = ep.load_config(CONFIG_PATH)
    membership = ep.protected_dependency_membership(config)
    assert "configs/future_milestone_v0_1_0.toml" not in membership.members
    assert membership.config_files == (ep.M7D_PLANNING_CONFIG_FILE,)
    assert len(membership.input_files) == 38
    assert membership.ignored_paths == ep.M7D_IGNORED_PATHS


# ---------------------------------------------------------------------------
# 4. modifying the M7D planning configuration fails
# ---------------------------------------------------------------------------


def test_modifying_the_m7d_planning_configuration_fails_verification(shadow_repo):
    config_path = shadow_repo / "configs" / "evaluation_design_plan_v0_1_0.toml"
    _replace_with_copy(config_path, config_path.read_bytes() + b"\n# tampered\n")
    report = _verify_in(shadow_repo)
    assert not report.ok
    checks = _check_map(report)
    assert not checks["config_sha256"].passed
    assert not checks["frozen_dependencies"].passed


def test_deleting_the_m7d_planning_configuration_fails_verification(shadow_repo):
    (shadow_repo / "configs" / "evaluation_design_plan_v0_1_0.toml").unlink()
    report = _verify_in(shadow_repo)
    assert not report.ok


# ---------------------------------------------------------------------------
# 5. modifying or deleting a declared upstream dependency fails
# ---------------------------------------------------------------------------


def test_modifying_a_declared_upstream_dependency_fails_verification(shadow_repo):
    dependency = shadow_repo / "reports" / "cross_font_analysis_v0_1_0" / "font_metrics.csv"
    _replace_with_copy(dependency, dependency.read_bytes() + b"tampered\n")
    report = _verify_in(shadow_repo)
    assert not report.ok


def test_deleting_a_declared_upstream_dependency_fails_verification(shadow_repo):
    dependency = shadow_repo / "reports" / "cross_font_analysis_v0_1_0" / "font_metrics.csv"
    dependency.unlink()
    report = _verify_in(shadow_repo)
    assert not report.ok


def test_modifying_a_declared_dataset_dependency_fails_verification(shadow_repo):
    dependency = (
        shadow_repo / "data" / "processed" / "heldout_font_benchmark_v0.1.0" / "manifest.csv"
    )
    _replace_with_copy(dependency, dependency.read_bytes() + b"tampered\n")
    report = _verify_in(shadow_repo)
    assert not report.ok


def test_deleting_a_declared_model_dependency_fails_verification(shadow_repo):
    dependency = (
        shadow_repo / "models" / "retention_aware_selection_v0_1_0" / "selected_policy.joblib"
    )
    dependency.unlink()
    report = _verify_in(shadow_repo)
    assert not report.ok


# ---------------------------------------------------------------------------
# 6/7. protected M7D outputs
# ---------------------------------------------------------------------------


def test_modifying_a_protected_m7d_output_fails_verification(shadow_repo):
    target = shadow_repo / "reports" / "evaluation_design_plan_v0_1_0" / "font_inventory.csv"
    _replace_with_copy(target, b"font_id,role\nghost,familiar\n")
    report = _verify_in(shadow_repo)
    assert not report.ok
    checks = _check_map(report)
    assert not checks["table:font_inventory.csv"].passed
    assert not checks["output_hashes"].passed


def test_deleting_a_protected_m7d_output_fails_verification(shadow_repo):
    (
        shadow_repo / "reports" / "evaluation_design_plan_v0_1_0" / "sample_size_scenarios.csv"
    ).unlink()
    report = _verify_in(shadow_repo)
    assert not report.ok
    checks = _check_map(report)
    assert not checks["output_file_set"].passed


def test_adding_an_unexpected_protected_output_fails_verification(shadow_repo):
    extra = shadow_repo / "reports" / "evaluation_design_plan_v0_1_0" / "unexpected.csv"
    extra.write_text("x\n", encoding="utf-8")
    report = _verify_in(shadow_repo)
    assert not report.ok
    assert not _check_map(report)["output_file_set"].passed


# ---------------------------------------------------------------------------
# 8. missing, malformed or tampered dependency manifest fails closed
# ---------------------------------------------------------------------------


def test_missing_dependency_manifest_fails_closed(shadow_repo):
    (shadow_repo / "reports" / "evaluation_design_plan_v0_1_0" / "run_metadata.json").unlink()
    report = _verify_in(shadow_repo)
    assert not report.ok
    assert not _check_map(report)["metadata"].passed


def test_malformed_dependency_manifest_fails_closed(shadow_repo):
    manifest = shadow_repo / "reports" / "evaluation_design_plan_v0_1_0" / "run_metadata.json"
    _replace_with_copy(manifest, b"{not json")
    report = _verify_in(shadow_repo)
    assert not report.ok
    assert not _check_map(report)["metadata"].passed


def test_tampered_dependency_manifest_snapshot_fails_closed(shadow_repo):
    """Editing the recorded snapshot must be detected, not silently accepted."""
    manifest = shadow_repo / "reports" / "evaluation_design_plan_v0_1_0" / "run_metadata.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["frozen_assets"]["reports/cross_font_analysis_v0_1_0/font_metrics.csv"] = "0" * 64
    _replace_with_copy(manifest, json.dumps(payload, indent=2, sort_keys=True).encode())
    report = _verify_in(shadow_repo)
    assert not report.ok
    assert not _check_map(report)["frozen_dependency_manifest"].passed


def test_consistently_retampered_dependency_manifest_fails_closed(shadow_repo):
    """Editing the snapshot *and* its recorded hash must still fail closed.

    The second anchor is the committed M7D planning configuration, so a
    manifest rewritten to agree with itself still disagrees with the committed
    configuration-pinned hashes.
    """
    manifest = shadow_repo / "reports" / "evaluation_design_plan_v0_1_0" / "run_metadata.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["frozen_assets"]["reports/cross_font_analysis_v0_1_0/font_metrics.csv"] = "0" * 64
    payload["frozen_assets_hash"] = ep._fingerprint_hash(payload["frozen_assets"])
    _replace_with_copy(manifest, json.dumps(payload, indent=2, sort_keys=True).encode())
    report = _verify_in(shadow_repo)
    assert not report.ok
    assert not _check_map(report)["frozen_dependency_anchors"].passed


def test_tampered_recorded_input_hashes_fail_closed(shadow_repo):
    manifest = shadow_repo / "reports" / "evaluation_design_plan_v0_1_0" / "run_metadata.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["inputs"]["m7c_font_metrics"] = "0" * 64
    _replace_with_copy(manifest, json.dumps(payload, indent=2, sort_keys=True).encode())
    report = _verify_in(shadow_repo)
    assert not report.ok
    assert not _check_map(report)["input_hashes"].passed


# ---------------------------------------------------------------------------
# 9. the M7C scoped verifier is unaffected and still detects tampering
# ---------------------------------------------------------------------------


def test_m7c_scoped_verifier_is_unaffected():
    report = cfa.verify_analysis(Path("configs/cross_font_analysis_v0_1_0.toml"))
    assert report.ok, [f"{c.name}: {c.detail}" for c in report.failures()]
    checks = {check.name: check for check in report.checks}
    assert checks["frozen_dependencies"].passed
    assert checks["frozen_dependency_manifest"].passed
    assert cfa.protected_dependency_membership().version == "m7c-scoped-dependencies-v1"


def test_m7c_scoped_verifier_still_detects_a_modified_dependency(tmp_path, monkeypatch):
    """The M7C contract must keep detecting genuine dependency modifications."""
    membership = cfa.ProtectedDependencyMembership(
        version="test-v1",
        config_files=("configs/a.toml",),
        dataset_roots=("data/processed/ds",),
        model_roots=(),
        report_roots=(),
        ignored_roots=(),
        output_files=(),
    )
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "a.toml").write_text("a = 1\n", encoding="utf-8")
    dataset = tmp_path / "data" / "processed" / "ds"
    dataset.mkdir(parents=True)
    (dataset / "manifest.csv").write_text("x\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    recorded = cfa._scoped_fingerprint(membership)
    (tmp_path / "configs" / "a.toml").write_text("a = 2\n", encoding="utf-8")
    current = cfa._scoped_fingerprint(membership)
    mismatches = cfa._scoped_fingerprint_mismatches(recorded, current)
    assert mismatches["modified"] == ["configs/a.toml"]


# ---------------------------------------------------------------------------
# 10. repeated verification is deterministic
# ---------------------------------------------------------------------------


def test_repeated_m7d_verification_is_deterministic(shadow_repo):
    first = _verify_in(shadow_repo)
    second = _verify_in(shadow_repo)
    assert first.ok and second.ok
    assert [(c.name, c.passed) for c in first.checks] == [(c.name, c.passed) for c in second.checks]


# ---------------------------------------------------------------------------
# Unit level: the scoped helpers
# ---------------------------------------------------------------------------


def test_scoped_fingerprint_covers_only_declared_members(tmp_path, monkeypatch):
    membership = ep.ProtectedDependencyMembership(
        version="test-v1",
        config_files=("configs/plan.toml",),
        input_files=("data/processed/ds/manifest.csv",),
        output_files=(),
        ignored_paths=("data/processed/scratch",),
    )
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "plan.toml").write_text("a = 1\n", encoding="utf-8")
    (tmp_path / "configs" / "other.toml").write_text("b = 2\n", encoding="utf-8")
    dataset = tmp_path / "data" / "processed" / "ds"
    dataset.mkdir(parents=True)
    (dataset / "manifest.csv").write_text("x\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    fingerprint = ep._scoped_dependency_fingerprint(membership)
    assert set(fingerprint) == {"configs/plan.toml", "data/processed/ds/manifest.csv"}


def test_scoped_state_classifies_missing_and_modified(tmp_path, monkeypatch):
    membership = ep.ProtectedDependencyMembership(
        version="test-v1",
        config_files=("configs/plan.toml",),
        input_files=("data/processed/ds/manifest.csv",),
        output_files=(),
        ignored_paths=(),
    )
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "plan.toml").write_text("a = 1\n", encoding="utf-8")
    dataset = tmp_path / "data" / "processed" / "ds"
    dataset.mkdir(parents=True)
    (dataset / "manifest.csv").write_text("x\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    expected = {
        "configs/plan.toml": sha256_file(tmp_path / "configs" / "plan.toml"),
        "data/processed/ds/manifest.csv": sha256_file(dataset / "manifest.csv"),
    }
    assert ep._scoped_dependency_state(membership, expected) == {
        "missing": [],
        "not_a_file": [],
        "modified": [],
    }
    (tmp_path / "configs" / "plan.toml").write_text("a = 2\n", encoding="utf-8")
    (dataset / "manifest.csv").unlink()
    state = ep._scoped_dependency_state(membership, expected)
    assert state["modified"] == ["configs/plan.toml"]
    assert state["missing"] == ["data/processed/ds/manifest.csv"]


def test_m7d_membership_is_explicit_and_documented():
    config = ep.load_config(CONFIG_PATH)
    membership = ep.protected_dependency_membership(config)
    assert membership.version == "m7d-scoped-dependencies-v1"
    assert membership.config_files == ("configs/evaluation_design_plan_v0_1_0.toml",)
    assert len(membership.input_files) == 38
    assert set(membership.input_files) == {pin.path for pin in config.inputs}
    assert set(membership.output_files) == set(ep.M7D_OUTPUT_FILES)
    assert len(membership.output_files) == 17
    assert "data/processed/matched_font_benchmark_v0.1.0" in membership.ignored_paths
    assert "reports/matched_font_benchmark_v0_1_0" in membership.ignored_paths


def test_historical_m7d_artifacts_are_untouched():
    """The regression tests must not have modified the recorded plan."""
    meta = json.loads((M7D_REPORT_ROOT / "run_metadata.json").read_text(encoding="utf-8"))
    assert all(
        sha256_file(M7D_REPORT_ROOT / name) == digest
        for name, digest in meta["output_hashes"].items()
    )
    assert all(
        sha256_file(REPO_ROOT / relative) == digest
        for relative, digest in meta["frozen_assets"].items()
    )
    assert sha256_file(CONFIG_PATH) == meta["plan"]["config_sha256"]
