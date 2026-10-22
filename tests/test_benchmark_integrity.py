"""Focused M8E provenance and ground-truth integrity regression tests.

These tests never regenerate the benchmark, render an image, call a model or
mutate an authoritative artifact. Tamper cases use isolated temporary shadow
trees. The historical M8C evidence is read from the committed artifacts and
from the local, Git-ignored ground_truth.csv.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from captcha_quality import benchmark_integrity as bi
from captcha_quality import benchmark_lifecycle as bl
from captcha_quality import matched_font_benchmark as mfb
from captcha_quality.provenance import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "benchmark_integrity_v0_1_0.toml"
PROTOCOL = REPO_ROOT / "configs" / "cross_font_benchmark_protocol_v0_2_0.toml"
DATASET = REPO_ROOT / "data" / "processed" / "matched_font_benchmark_v0.1.0"
REPORT = REPO_ROOT / "reports" / "matched_font_benchmark_v0_1_0"
GROUND_TRUTH = DATASET / "ground_truth.csv"
INPUT_MANIFEST = REPORT / "input_manifest.csv"
RUN_METADATA = REPORT / "run_metadata.json"

HISTORICAL_HASHES = {
    "reports/matched_font_benchmark_v0_1_0/input_manifest.csv": (
        "70f8b0ca8dfc2a9c6742858f8f0bfa3b863ca78fb9a4aacf4bac0a17952d3d71"
    ),
    "reports/matched_font_benchmark_v0_1_0/run_metadata.json": (
        "930ce8d8767a3e52919e5736ff8a9dfe4b74b008c54e2cc2107fd221d0944ebd"
    ),
    "src/captcha_quality/matched_font_benchmark.py": (
        "3ec0e93f5d7aa2a773ade5260de4f344c41a68899772767a3b3b7e9ade387506"
    ),
}

ARTIFACTS_PRESENT = (
    GROUND_TRUTH.is_file()
    and INPUT_MANIFEST.is_file()
    and RUN_METADATA.is_file()
    and (DATASET / "manifest.csv").is_file()
    and (REPORT / "predictions.csv").is_file()
)
requires_artifacts = pytest.mark.skipif(
    not ARTIFACTS_PRESENT, reason="the M8C artifacts are not present in this checkout"
)


@pytest.fixture(scope="module")
def config() -> bi.IntegrityConfig:
    return bi.load_config(CONFIG)


def _shadow_dataset(tmp_path: Path) -> Path:
    """A temporary dataset that shares every non-ground-truth file by symlink."""
    shadow = tmp_path / "dataset"
    shadow.mkdir(parents=True, exist_ok=True)
    for name in (
        "manifest.csv",
        "sources.csv",
        "source_cases.csv",
        "features.csv",
        "dataset_summary.json",
    ):
        (shadow / name).symlink_to(DATASET / name)
    return shadow


def _write_ground_truth(shadow: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(shadow / "ground_truth.csv", index=False, lineterminator="\n")


def _failure_names(report: bi.VerificationReport) -> set[str]:
    return {check.name for check in report.failures()}


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _shadow_repository(tmp_path: Path) -> tuple[Path, Path, str]:
    """Create a committed shadow repository with the exact execution closure."""
    root = tmp_path / "shadow_repo"
    for dependency in bi.load_config(CONFIG).dependencies:
        target = root / dependency.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(REPO_ROOT / dependency.path)
    config_path = root / "configs" / CONFIG.name
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.symlink_to(CONFIG)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "m8e-tests@example.invalid")
    _git(root, "config", "user.name", "M8E tests")
    _git(root, "add", "-A")
    _git(root, "-c", "commit.gpgsign=false", "commit", "-qm", "shadow closure")
    return root, config_path, _git(root, "rev-parse", "HEAD")


@requires_artifacts
def test_valid_unchanged_ground_truth_passes(config: bi.IntegrityConfig) -> None:
    report = bi.verify_ground_truth(config, repo_root=REPO_ROOT)
    assert report.ok
    assert {check.name for check in report.checks} == {
        "ground_truth:file",
        "ground_truth:table",
        "ground_truth:semantic_joins",
    }
    assert sha256_file(GROUND_TRUTH) == config.ground_truth.sha256
    metadata = json.loads(RUN_METADATA.read_text(encoding="utf-8"))
    assert metadata["dataset"]["ground_truth_sha256"] == config.ground_truth.sha256


@requires_artifacts
def test_ground_truth_byte_modification_fails(config: bi.IntegrityConfig, tmp_path: Path) -> None:
    shadow = _shadow_dataset(tmp_path)
    frame = pd.read_csv(GROUND_TRUTH, keep_default_na=False)
    frame.loc[frame.index[0], "text"] = "ZZZZZ"
    _write_ground_truth(shadow, frame)
    report = bi.verify_ground_truth(config, repo_root=REPO_ROOT, dataset_root=shadow)
    assert not report.ok
    assert "ground_truth:file" in _failure_names(report)
    detail = " ".join(check.detail for check in report.failures())
    assert "ZZZZZ" not in detail
    assert str(frame.loc[frame.index[0], "case_id"]) not in detail


@requires_artifacts
def test_ground_truth_reordering_fails_even_when_parsed_rows_are_equivalent(
    config: bi.IntegrityConfig, tmp_path: Path
) -> None:
    shadow = _shadow_dataset(tmp_path)
    frame = pd.read_csv(GROUND_TRUTH, keep_default_na=False)
    _write_ground_truth(shadow, frame.iloc[::-1].reset_index(drop=True))
    report = bi.verify_ground_truth(config, repo_root=REPO_ROOT, dataset_root=shadow)
    assert not report.ok
    assert "ground_truth:file" in _failure_names(report)
    # The join is order-independent, so the semantic check alone would pass.
    semantic = [check for check in report.checks if check.name == "ground_truth:semantic_joins"]
    assert semantic and semantic[0].passed


@requires_artifacts
def test_missing_and_non_regular_ground_truth_fail(
    config: bi.IntegrityConfig, tmp_path: Path
) -> None:
    missing = _shadow_dataset(tmp_path / "missing")
    missing_report = bi.verify_ground_truth(config, repo_root=REPO_ROOT, dataset_root=missing)
    assert not missing_report.ok
    assert "ground_truth:file" in _failure_names(missing_report)

    linked = _shadow_dataset(tmp_path / "linked")
    (linked / "ground_truth.csv").symlink_to(GROUND_TRUTH)
    linked_report = bi.verify_ground_truth(config, repo_root=REPO_ROOT, dataset_root=linked)
    assert not linked_report.ok
    detail = " ".join(check.detail for check in linked_report.failures())
    assert "not a regular file" in detail


@requires_artifacts
def test_duplicate_rows_ids_and_invalid_joins_fail(
    config: bi.IntegrityConfig, tmp_path: Path
) -> None:
    original = pd.read_csv(GROUND_TRUTH, keep_default_na=False)

    duplicated_row = _shadow_dataset(tmp_path / "duplicate_row")
    _write_ground_truth(
        duplicated_row, pd.concat([original, original.iloc[[0]]], ignore_index=True)
    )
    row_report = bi.verify_ground_truth(config, repo_root=REPO_ROOT, dataset_root=duplicated_row)
    assert not row_report.ok
    assert "ground_truth:table" in _failure_names(row_report)

    duplicated_id = _shadow_dataset(tmp_path / "duplicate_id")
    altered = original.copy()
    altered.loc[altered.index[1], "source_id"] = altered.loc[altered.index[0], "source_id"]
    _write_ground_truth(duplicated_id, altered)
    id_report = bi.verify_ground_truth(config, repo_root=REPO_ROOT, dataset_root=duplicated_id)
    assert not id_report.ok
    assert "ground_truth:table" in _failure_names(id_report)

    invalid_join = _shadow_dataset(tmp_path / "invalid_join")
    lineage = original.copy()
    lineage.loc[lineage.index[0], "case_id"] = "case-does-not-exist"
    _write_ground_truth(invalid_join, lineage)
    join_report = bi.verify_ground_truth(config, repo_root=REPO_ROOT, dataset_root=invalid_join)
    assert not join_report.ok
    assert "ground_truth:semantic_joins" in _failure_names(join_report)


@requires_artifacts
def test_modified_or_deleted_pinned_modules_fail(
    config: bi.IntegrityConfig, tmp_path: Path
) -> None:
    def closure(root: Path) -> None:
        for dependency in config.dependencies:
            target = root / dependency.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(REPO_ROOT / dependency.path)

    deleted_root = tmp_path / "deleted"
    closure(deleted_root)
    assert bi.verify_dependency_closure(config, repo_root=deleted_root).ok
    (deleted_root / "src" / "captcha_quality" / "features.py").unlink()
    deleted_report = bi.verify_dependency_closure(config, repo_root=deleted_root)
    assert not deleted_report.ok
    assert "dependencies:hashes" in _failure_names(deleted_report)

    modified_root = tmp_path / "modified"
    closure(modified_root)
    assert bi.verify_dependency_closure(config, repo_root=modified_root).ok
    modified = modified_root / "src" / "captcha_quality" / "geometric.py"
    modified.unlink()
    original = (REPO_ROOT / "src" / "captcha_quality" / "geometric.py").read_bytes()
    modified.write_bytes(original + b"\n# tampered\n")
    modified_report = bi.verify_dependency_closure(config, repo_root=modified_root)
    assert not modified_report.ok
    assert "dependencies:hashes" in _failure_names(modified_report)


@requires_artifacts
def test_missing_and_unexpected_dependency_manifest_entries_fail(
    config: bi.IntegrityConfig,
) -> None:
    missing = replace(
        config,
        dependencies=tuple(item for item in config.dependencies if item.key != "quality_metrics"),
    )
    missing_report = bi.verify_dependency_closure(missing, repo_root=REPO_ROOT)
    assert not missing_report.ok
    assert "dependencies:membership" in _failure_names(missing_report)

    pilot = REPO_ROOT / "src" / "captcha_quality" / "pilot.py"
    unexpected = replace(
        config,
        dependencies=(
            *config.dependencies,
            bi.DependencyPin(
                key="unrelated_pilot",
                path="src/captcha_quality/pilot.py",
                sha256=sha256_file(pilot),
                role="not part of the M8C execution closure",
            ),
        ),
    )
    unexpected_report = bi.verify_dependency_closure(unexpected, repo_root=REPO_ROOT)
    assert not unexpected_report.ok
    assert "dependencies:membership" in _failure_names(unexpected_report)


@requires_artifacts
def test_dirty_tree_execution_is_rejected_before_creating_output(tmp_path: Path) -> None:
    root, config_path, commit = _shadow_repository(tmp_path)
    (root / "dirty.txt").write_text("uncommitted\n", encoding="utf-8")
    output = tmp_path / "future_dataset"
    called = False

    def action(_ticket: bi.ExecutionTicket) -> None:
        nonlocal called
        called = True
        output.mkdir()

    with pytest.raises(bi.IntegrityError, match="working tree is not clean"):
        bi.run_guarded_execution(
            repo_root=root,
            config_path=config_path,
            expected_commit=commit,
            output_paths=[output],
            action=action,
        )
    assert not called
    assert not output.exists()


@requires_artifacts
def test_clean_committed_execution_passes_preflight(tmp_path: Path) -> None:
    root, config_path, commit = _shadow_repository(tmp_path)
    output = tmp_path / "future_dataset"
    report = bi.verify_future_preflight(
        config_path,
        repo_root=root,
        expected_commit=commit,
        output_paths=[output],
    )
    assert report.ok
    assert "repository:clean" in {check.name for check in report.checks}
    assert "repository:outputs_absent" in {check.name for check in report.checks}

    def action(ticket: bi.ExecutionTicket) -> Path:
        assert ticket.revision == commit
        output.mkdir()
        return output

    result = bi.run_guarded_execution(
        repo_root=root,
        config_path=config_path,
        expected_commit=commit,
        output_paths=[output],
        action=action,
    )
    assert result == output
    assert output.is_dir()


@requires_artifacts
def test_duplicate_input_manifest_keys_are_rejected_for_future_runs(
    config: bi.IntegrityConfig, tmp_path: Path
) -> None:
    historical = bi.verify_historical_input_manifest(config, repo_root=REPO_ROOT)
    assert historical.ok
    keys = [check for check in historical.checks if check.name == "input_manifest:keys"]
    assert keys and "historical duplicate" in keys[0].detail

    future = bi.verify_future_input_manifest(INPUT_MANIFEST)
    assert not future.ok
    assert "input_manifest:keys" in _failure_names(future)

    frame = pd.read_csv(INPUT_MANIFEST, keep_default_na=False)
    unique = frame.drop_duplicates("key", keep="first")
    unique_path = tmp_path / "future_unique_manifest.csv"
    unique.to_csv(unique_path, index=False, lineterminator="\n")
    assert bi.verify_future_input_manifest(unique_path).ok


@requires_artifacts
def test_unrelated_shared_directory_additions_do_not_invalidate_scoped_verification(
    config: bi.IntegrityConfig, tmp_path: Path
) -> None:
    root = tmp_path / "closure"
    for dependency in config.dependencies:
        target = root / dependency.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(REPO_ROOT / dependency.path)
    (root / "src" / "captcha_quality" / "unrelated.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "unrelated.toml").write_text("value = 1\n", encoding="utf-8")
    report = bi.verify_dependency_closure(config, repo_root=root)
    assert report.ok
    membership = [check for check in report.checks if check.name == "dependencies:membership"]
    assert membership and "unrelated package files are out of scope" in membership[0].detail


@requires_artifacts
def test_historical_m8c_results_metadata_and_ground_truth_are_unchanged() -> None:
    for relative, expected in HISTORICAL_HASHES.items():
        if expected is not None:
            assert sha256_file(REPO_ROOT / relative) == expected

    assert sha256_file(GROUND_TRUTH) == (
        "426fa41949223ebc481e88b0c42a22770df80d30d3ed8f951fa0f52b92b39dfd"
    )
    assert (
        sha256_file(INPUT_MANIFEST)
        == HISTORICAL_HASHES["reports/matched_font_benchmark_v0_1_0/input_manifest.csv"]
    )
    assert (
        sha256_file(RUN_METADATA)
        == HISTORICAL_HASHES["reports/matched_font_benchmark_v0_1_0/run_metadata.json"]
    )
    assert (
        sha256_file(mfb.__file__)
        == HISTORICAL_HASHES["src/captcha_quality/matched_font_benchmark.py"]
    )

    predictions = pd.read_csv(REPORT / "predictions.csv")
    assert len(predictions) == 6300
    assert predictions["row_id"].is_unique
    assert len([path for path in REPORT.iterdir() if path.is_file()]) == 15
    assert len(list((DATASET / "images").rglob("*.png"))) == 6300

    tracked = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "ls-files",
            "--error-unmatch",
            str(GROUND_TRUTH.relative_to(REPO_ROOT)),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert tracked.returncode != 0, "ground_truth.csv must remain uncommitted"


@requires_artifacts
def test_existing_lifecycle_tamper_detection_still_fails_closed(tmp_path: Path) -> None:
    source = REPO_ROOT / "configs" / "benchmark_lifecycle_v0_3_0.toml"
    text = source.read_text(encoding="utf-8")
    tampered = tmp_path / "benchmark_lifecycle_tampered.toml"
    tampered.write_text(
        text.replace(
            "fa9a2c57f7aec6dda214a933f3b639160ef2f2f2996129d64b225f2e60f46f48",
            "0" * 64,
        ),
        encoding="utf-8",
    )
    report = bl.verify_lifecycle(tampered, state="post-execution")
    assert not report.ok
    assert "declared_inputs" in _failure_names(report)


@requires_artifacts
def test_executed_lifecycle_and_dependency_verification_remain_consistent(
    config: bi.IntegrityConfig,
) -> None:
    report = bl.verify_lifecycle(
        REPO_ROOT / "configs" / "benchmark_lifecycle_v0_3_0.toml",
        state="post-execution",
    )
    assert report.ok
    integrity = bi.verify_historical_evidence(CONFIG, repo_root=REPO_ROOT)
    assert integrity.ok
