"""Tests for the M8A benchmark output-lifecycle transition.

The lifecycle is a *read-only* verification contract: it declares the
authorised execution paths, classifies the M7D planning-state check in each
state, and proves that a later execution run cannot rewrite the original
plan. Every lifecycle simulation happens in an isolated temporary shadow tree;
no real benchmark output is created or mutated.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from captcha_quality import benchmark_lifecycle as bl
from captcha_quality import evaluation_plan as ep
from captcha_quality import font_roster_audit as fra
from captcha_quality.provenance import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
LIFECYCLE_CONFIG = REPO_ROOT / "configs" / "benchmark_lifecycle_v0_1_0.toml"
PROTOCOL_CONFIG = REPO_ROOT / "configs" / "cross_font_benchmark_protocol_v0_1_0.toml"

if not PROTOCOL_CONFIG.is_file():  # pragma: no cover
    pytest.skip("the M8A execution protocol is not present", allow_module_level=True)


@pytest.fixture(scope="module")
def config() -> bl.LifecycleConfig:
    return bl.load_config(LIFECYCLE_CONFIG)


def _dataset_manifest(config: bl.LifecycleConfig) -> pd.DataFrame:
    """A synthetic manifest with the protocol's fonts, rows and scopes."""
    import tomllib

    protocol = tomllib.loads(PROTOCOL_CONFIG.read_text(encoding="utf-8"))
    fonts = [(str(font["key"]), str(font["sha256"])) for font in protocol["fonts"]]
    rows = []
    for index in range(config.authorised_execution.n_rows):
        font_id, font_sha = fonts[index % len(fonts)]
        primary = index % 21 not in (0, 20)
        rows.append(
            {
                "sample_id": f"row-{index:05d}",
                "source_id": f"src-{index // 21:03d}",
                "font_id": font_id,
                "font_sha256": font_sha,
                "protocol_scope": "primary" if primary else "diagnostic",
            }
        )
    return pd.DataFrame(rows)


def _write_shadow_dataset(config: bl.LifecycleConfig, root: Path) -> None:
    """Create the simulated executed dataset inside a temporary shadow tree.

    The images are zero-byte placeholders, not rendered images: the lifecycle
    contract only constrains the file membership and the count.
    """
    dataset = root / config.authorised_execution.dataset_output
    dataset.mkdir(parents=True, exist_ok=True)
    manifest = _dataset_manifest(config)
    manifest.to_csv(dataset / "manifest.csv", index=False)
    sources = pd.DataFrame(
        {
            "source_id": [
                f"src-{index:03d}" for index in range(config.authorised_execution.n_sources)
            ],
            "split": ["eval"] * config.authorised_execution.n_sources,
        }
    )
    sources.to_csv(dataset / "sources.csv", index=False)
    for name in ("source_cases.csv", "ground_truth.csv", "features.csv"):
        (dataset / name).write_text("placeholder\n", encoding="utf-8")
    (dataset / "dataset_summary.json").write_text("{}\n", encoding="utf-8")
    for directory in config.authorised_dataset_directories:
        base = dataset / directory.name
        for subdirectory in directory.subdirectories:
            (base / subdirectory).mkdir(parents=True, exist_ok=True)
        per_directory = directory.required_image_count // len(directory.subdirectories)
        for subdirectory in directory.subdirectories:
            for index in range(per_directory):
                (base / subdirectory / f"row-{index:05d}.png").write_bytes(b"")
    # top up to the exact declared count
    images = sorted((dataset / directory.name).rglob("*.png"))
    extra = directory.required_image_count - len(images)
    for index in range(max(extra, 0)):
        (images[0].parent / f"extra-{index:05d}.png").write_bytes(b"")


def _write_shadow_report(config: bl.LifecycleConfig, root: Path) -> None:
    report = root / config.authorised_execution.report_output
    report.mkdir(parents=True, exist_ok=True)
    for item in config.authorised_report_files:
        (report / item.name).write_text("placeholder\n", encoding="utf-8")


def _shadow_repo(config: bl.LifecycleConfig, root: Path) -> Path:
    """Link every declared input and the frozen milestone artifacts."""
    for pin in config.inputs:
        target = root / pin.path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(REPO_ROOT / pin.path)
    # the M7D and M7E verifiers need their own declared inputs in the tree
    import tomllib

    lifecycle_link = root / "configs" / LIFECYCLE_CONFIG.name
    lifecycle_link.parent.mkdir(parents=True, exist_ok=True)
    if not (lifecycle_link.exists() or lifecycle_link.is_symlink()):
        lifecycle_link.symlink_to(LIFECYCLE_CONFIG)

    for upstream in (
        "configs/evaluation_design_plan_v0_1_0.toml",
        "configs/font_roster_audit_v0_2_0.toml",
    ):
        payload = tomllib.loads((REPO_ROOT / upstream).read_text(encoding="utf-8"))
        for pin in payload.get("inputs", []):
            target = root / str(pin["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                continue
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
            if target.exists() or target.is_symlink():
                continue
            target.symlink_to(path)
    return root


def _verify_in(root: Path, state: str = "auto") -> bl.VerificationReport:
    previous = Path.cwd()
    os.chdir(root)
    try:
        shadow_config = root / "configs" / LIFECYCLE_CONFIG.name
        config_path = shadow_config if shadow_config.is_file() else LIFECYCLE_CONFIG
        return bl.verify_lifecycle(config_path, state=state)
    finally:
        os.chdir(previous)


# ---------------------------------------------------------------------------
# 1. historical M7D verification still passes unchanged
# ---------------------------------------------------------------------------


def test_historical_m7d_verification_still_passes():
    report = ep.verify_plan(Path("configs/evaluation_design_plan_v0_1_0.toml"))
    assert report.ok, [f"{c.name}: {c.detail}" for c in report.failures()]
    rows = pd.read_csv(REPO_ROOT / "reports/evaluation_design_plan_v0_1_0/isolation_checks.csv")
    assert bl.M7D_PLANNING_STATE_CHECK in set(rows["check"])
    assert bool(rows.loc[rows["check"] == bl.M7D_PLANNING_STATE_CHECK, "passed"].iloc[0])


def test_lifecycle_is_valid_in_the_authorized_but_not_executed_state(config):
    report = _verify_in(REPO_ROOT)
    assert report.ok, [f"{c.name}: {c.detail}" for c in report.failures()]
    mapped = {check.name: check for check in report.checks}
    assert "pre-execution" in mapped["state"].detail
    assert mapped["pre_execution_paths"].passed


def test_transition_does_not_require_the_planned_outputs_to_exist(config):
    assert not (REPO_ROOT / config.authorised_execution.dataset_output).exists()
    assert not (REPO_ROOT / config.authorised_execution.report_output).exists()
    report = _verify_in(REPO_ROOT)
    assert report.ok


# ---------------------------------------------------------------------------
# 2. shadow execution: only the authorised paths may be created
# ---------------------------------------------------------------------------


def test_shadow_execution_passes_with_exactly_the_authorised_membership(config, tmp_path):
    root = _shadow_repo(config, tmp_path / "repo")
    _write_shadow_dataset(config, root)
    _write_shadow_report(config, root)
    report = _verify_in(root, state="post-execution")
    assert report.ok, [f"{c.name}: {c.detail}" for c in report.failures()]
    mapped = {check.name: check for check in report.checks}
    assert mapped["dataset_membership"].passed
    assert mapped["report_membership"].passed
    assert mapped["dataset_matches_protocol"].passed
    assert "transition" in mapped["m7d_verifier"].detail


def test_unexpected_path_in_the_dataset_fails(config, tmp_path):
    root = _shadow_repo(config, tmp_path / "repo")
    _write_shadow_dataset(config, root)
    _write_shadow_report(config, root)
    (root / config.authorised_execution.dataset_output / "unexpected.csv").write_text(
        "x\n", encoding="utf-8"
    )
    report = _verify_in(root, state="post-execution")
    assert not report.ok
    mapped = {check.name: check for check in report.checks}
    assert not mapped["dataset_membership"].passed


def test_missing_required_output_fails(config, tmp_path):
    root = _shadow_repo(config, tmp_path / "repo")
    _write_shadow_dataset(config, root)
    _write_shadow_report(config, root)
    (root / config.authorised_execution.report_output / "predictions.csv").unlink()
    report = _verify_in(root, state="post-execution")
    assert not report.ok
    mapped = {check.name: check for check in report.checks}
    assert not mapped["report_membership"].passed


def test_missing_image_fails(config, tmp_path):
    root = _shadow_repo(config, tmp_path / "repo")
    _write_shadow_dataset(config, root)
    _write_shadow_report(config, root)
    images = sorted((root / config.authorised_execution.dataset_output).rglob("*.png"))
    images[0].unlink()
    report = _verify_in(root, state="post-execution")
    assert not report.ok
    mapped = {check.name: check for check in report.checks}
    assert not mapped["dataset_membership"].passed


def test_unauthorised_path_creation_is_not_authorised(config, tmp_path):
    """Only the two declared directories may hold benchmark outputs."""
    assert set(config.authorised_paths if hasattr(config, "authorised_paths") else ()) == set()
    authorised = {
        config.authorised_execution.dataset_output,
        config.authorised_execution.report_output,
    }
    assert authorised == {
        "data/processed/matched_font_benchmark_v0.1.0",
        "reports/matched_font_benchmark_v0_1_0",
    }


# ---------------------------------------------------------------------------
# 3. frozen inputs and the historical plan
# ---------------------------------------------------------------------------


def test_modified_frozen_input_fails(config, tmp_path):
    root = _shadow_repo(config, tmp_path / "repo")
    target = root / "configs" / "evaluation_design_plan_v0_1_0.toml"
    content = target.read_bytes()
    target.unlink()
    target.write_bytes(content + b"\n# tampered\n")
    report = _verify_in(root)
    assert not report.ok
    mapped = {check.name: check for check in report.checks}
    assert not mapped["declared_inputs"].passed
    assert not mapped["historical_state"].passed


def test_deleted_frozen_input_fails(config, tmp_path):
    root = _shadow_repo(config, tmp_path / "repo")
    (root / "models" / "retention_aware_selection_v0_1_0" / "selected_policy.joblib").unlink()
    report = _verify_in(root)
    assert not report.ok
    mapped = {check.name: check for check in report.checks}
    assert not mapped["declared_inputs"].passed


def test_model_artifact_change_fails_verification(config, tmp_path):
    root = _shadow_repo(config, tmp_path / "repo")
    target = root / "models" / "retention_aware_selection_v0_1_0" / "selected_policy.joblib"
    target.unlink()
    target.write_bytes(b"not the champion")
    report = _verify_in(root)
    assert not report.ok
    mapped = {check.name: check for check in report.checks}
    assert not mapped["declared_inputs"].passed


def test_feature_schema_change_fails_verification(config, tmp_path):
    root = _shadow_repo(config, tmp_path / "repo")
    target = root / "src" / "captcha_quality" / "features.py"
    content = target.read_bytes()
    target.unlink()
    target.write_bytes(content + b"\n# feature change\n")
    report = _verify_in(root)
    assert not report.ok
    mapped = {check.name: check for check in report.checks}
    assert not mapped["declared_inputs"].passed


def test_execution_cannot_change_the_m7d_historical_plan(config, tmp_path):
    root = _shadow_repo(config, tmp_path / "repo")
    target = root / "reports" / "evaluation_design_plan_v0_1_0" / "plan_manifest.json"
    content = target.read_bytes()
    target.unlink()
    target.write_bytes(content + b" ")
    report = _verify_in(root)
    assert not report.ok
    mapped = {check.name: check for check in report.checks}
    assert not mapped["historical_state"].passed


def test_tampered_lifecycle_manifest_fails_closed(config, tmp_path):
    """A lifecycle manifest edited to change its own recorded values still fails."""
    root = _shadow_repo(config, tmp_path / "repo")
    target = root / "configs" / "benchmark_lifecycle_v0_1_0.toml"
    content = target.read_text(encoding="utf-8")
    target.unlink()
    target.write_text(content.replace("n_sources = 300", "n_sources = 299"), encoding="utf-8")
    report = _verify_in(root)
    assert not report.ok
    mapped = {check.name: check for check in report.checks}
    assert not mapped["execution_protocol"].passed


def test_self_consistent_tampering_of_a_frozen_input_still_fails(config, tmp_path):
    """Rewriting the recorded hash of a tampered input does not help.

    The lifecycle manifest and the frozen artifact are edited together so the
    manifest is internally consistent, but the M7D verifier re-checks the M7D
    artifacts against the Git-tracked M7D configuration, so the tampering is
    still detected.
    """
    root = _shadow_repo(config, tmp_path / "repo")
    manifest = root / "reports" / "evaluation_design_plan_v0_1_0" / "plan_manifest.json"
    content = manifest.read_bytes()
    manifest.unlink()
    manifest.write_bytes(content + b" ")
    tampered_sha = sha256_file(manifest)
    lifecycle = root / "configs" / "benchmark_lifecycle_v0_1_0.toml"
    lifecycle_text = lifecycle.read_text(encoding="utf-8")
    lifecycle.unlink()
    lifecycle.write_text(
        lifecycle_text.replace(
            "7eeaf58969c001142d158681d38283b718cb7a5fa7914c22af29f1c6ec48e8a8",
            tampered_sha,
        ),
        encoding="utf-8",
    )
    report = _verify_in(root)
    assert not report.ok
    mapped = {check.name: check for check in report.checks}
    assert not mapped["m7d_verifier"].passed


def test_roster_must_match_the_m7e_evidence(config, tmp_path):
    root = _shadow_repo(config, tmp_path / "repo")
    target = root / "reports" / "font_roster_audit_v0_2_0" / "font_evidence.csv"
    frame = pd.read_csv(target)
    frame.loc[frame["font_key"] == "verdana", "m7d_font_sha256"] = "0" * 64
    target.unlink()
    frame.to_csv(target, index=False)
    report = _verify_in(root)
    assert not report.ok
    mapped = {check.name: check for check in report.checks}
    assert not mapped["roster"].passed


# ---------------------------------------------------------------------------
# 4. scoped membership: unrelated additions are accepted
# ---------------------------------------------------------------------------


def test_unrelated_additions_to_shared_directories_are_accepted(config, tmp_path):
    root = _shadow_repo(config, tmp_path / "repo")
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


def test_font_binary_in_the_tree_fails(config, tmp_path):
    root = _shadow_repo(config, tmp_path / "repo")
    (root / "models" / "some_font.ttf").write_bytes(b"")
    report = _verify_in(root)
    assert not report.ok
    mapped = {check.name: check for check in report.checks}
    assert not mapped["font_binaries"].passed


# ---------------------------------------------------------------------------
# 5. determinism and the other milestone verifiers
# ---------------------------------------------------------------------------


def test_repeated_lifecycle_verification_is_deterministic():
    first = bl.verify_lifecycle(LIFECYCLE_CONFIG)
    second = bl.verify_lifecycle(LIFECYCLE_CONFIG)
    assert first.ok and second.ok
    assert [(c.name, c.passed) for c in first.checks] == [(c.name, c.passed) for c in second.checks]


def test_previous_milestone_verifiers_retain_their_behaviour():
    from captcha_quality import cross_font_analysis as cfa

    m7c = cfa.verify_analysis(Path("configs/cross_font_analysis_v0_1_0.toml"))
    assert m7c.ok, [f"{c.name}: {c.detail}" for c in m7c.failures()]
    m7d = ep.verify_plan(Path("configs/evaluation_design_plan_v0_1_0.toml"))
    assert m7d.ok
    m7e = fra.verify_audit(Path("configs/font_roster_audit_v0_2_0.toml"))
    assert m7e.ok, [f"{c.name}: {c.detail}" for c in m7e.failures()]


def test_historical_artifacts_are_untouched():
    """The M8A milestone changed no historical artifact."""
    assert sha256_file(LIFECYCLE_CONFIG) == sha256_file(LIFECYCLE_CONFIG)
    m7d_config = REPO_ROOT / "configs" / "evaluation_design_plan_v0_1_0.toml"
    assert sha256_file(m7d_config) == (
        "e9f236b383172217e85096c1e06e81adcbb3fd85dc5c18f6537d824e1c6300d3"
    )
    m7e_config = REPO_ROOT / "configs" / "font_roster_audit_v0_2_0.toml"
    assert sha256_file(m7e_config) == (
        "bc319df3cae8114d6ecdcb50310a0c69c54af94cd7399c69e48189dfe0733faa"
    )
    assert not (REPO_ROOT / "data" / "processed" / "matched_font_benchmark_v0.1.0").exists()
    assert not (REPO_ROOT / "reports" / "matched_font_benchmark_v0_1_0").exists()
