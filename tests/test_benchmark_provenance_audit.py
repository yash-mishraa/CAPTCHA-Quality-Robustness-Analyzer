"""Focused M8D provenance and ground-truth-isolation regression tests.

These tests are read-only. They never train, tune, render, regenerate the
benchmark or call the frozen champion. The fake estimator used by the
isolation test only records the matrix it receives, so the test can prove
that the prediction path is a function of the eight image features alone.

The committed M8C artifacts are ignored by Git except for the files that were
force-added in the execution commit; the tests skip when those local artifacts
are not present. The provenance test reads the recorded execution evidence and
the committed module rather than reconstructing the run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from captcha_quality import matched_font_benchmark as mfb
from captcha_quality.benchmark_preflight import load_case_manifest, load_execution_protocol
from captcha_quality.features import FEATURE_NAMES
from captcha_quality.provenance import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPO_ROOT / "configs" / "cross_font_benchmark_protocol_v0_2_0.toml"
TEXT_MANIFEST = REPO_ROOT / "configs" / "cross_font_text_cases_v0_1_0.toml"
DATASET = REPO_ROOT / "data" / "processed" / "matched_font_benchmark_v0.1.0"
REPORT = REPO_ROOT / "reports" / "matched_font_benchmark_v0_1_0"
METADATA = REPORT / "run_metadata.json"
GROUND_TRUTH = DATASET / "ground_truth.csv"


def _outputs_present() -> bool:
    return (
        (DATASET / "manifest.csv").is_file()
        and (DATASET / "features.csv").is_file()
        and (REPORT / "predictions.csv").is_file()
        and METADATA.is_file()
    )


requires_outputs = pytest.mark.skipif(
    not _outputs_present(), reason="the committed M8C artifacts are not present"
)
requires_ground_truth = pytest.mark.skipif(
    not GROUND_TRUTH.is_file(), reason="the local ground_truth.csv is not present"
)


def _metadata() -> dict[str, object]:
    return json.loads(METADATA.read_text(encoding="utf-8"))


@requires_outputs
def test_execution_module_and_pinned_inputs_match_the_recorded_hashes() -> None:
    metadata = _metadata()
    module = REPO_ROOT / metadata["code"]["module"]
    assert sha256_file(module) == metadata["code"]["module_sha256"]

    inputs = pd.read_csv(REPORT / "input_manifest.csv", keep_default_na=False)
    assert len(inputs) == 36
    assert not inputs["matches"].eq(False).any()
    for row in inputs.itertuples(index=False):
        assert (REPO_ROOT / row.path).is_file()
        assert sha256_file(REPO_ROOT / row.path) == row.sha256
        if row.expected_sha256:
            assert row.sha256 == row.expected_sha256

    # The run honestly recorded a dirty working tree at the parent revision.
    assert metadata["code"]["revision"] == "75c8ef4-dirty"


@requires_ground_truth
def test_ground_truth_hash_matches_the_recorded_execution_evidence() -> None:
    metadata = _metadata()
    assert sha256_file(GROUND_TRUTH) == metadata["dataset"]["ground_truth_sha256"]


@requires_outputs
@requires_ground_truth
def test_ground_truth_never_enters_the_prediction_matrix() -> None:
    protocol = load_execution_protocol(PROTOCOL)
    bundle = mfb.load_dataset(DATASET)
    font_meta = mfb._font_meta(protocol)

    class RecordingEstimator:
        def __init__(self) -> None:
            self.matrices: list[np.ndarray] = []

        def predict(self, values: np.ndarray) -> np.ndarray:
            self.matrices.append(np.array(values, dtype=np.float64, copy=True))
            return np.clip(self.matrices[-1][:, 0], 0.0, 1.0)

    estimator = RecordingEstimator()
    baseline, checks = mfb._build_prediction_frame(bundle, estimator, font_meta)
    assert all(check.passed for check in checks)
    assert len(estimator.matrices) == 1

    matrix = estimator.matrices[0]
    assert matrix.shape == (6300, len(FEATURE_NAMES))
    merged = mfb._merge_features(bundle)
    assert np.array_equal(matrix, merged[list(FEATURE_NAMES)].to_numpy(dtype=np.float64))
    assert "ssim_target" not in FEATURE_NAMES
    assert "text" not in merged.columns

    tampered = bundle.ground_truth.copy()
    tampered.loc[tampered.index[0], "text"] = "ZZZZZ"
    second, _ = mfb._build_prediction_frame(
        replace(bundle, ground_truth=tampered), RecordingEstimator(), font_meta
    )
    assert np.array_equal(second["prediction"], baseline["prediction"])
    assert np.array_equal(second["ssim_target"], baseline["ssim_target"])


@requires_outputs
@requires_ground_truth
def test_source_table_verification_rejects_missing_altered_or_duplicated_ground_truth() -> None:
    protocol = load_execution_protocol(PROTOCOL)
    case_manifest = load_case_manifest(TEXT_MANIFEST)
    bundle = mfb.load_dataset(DATASET)

    mfb._verify_source_tables(bundle, protocol, case_manifest)

    altered = bundle.ground_truth.copy()
    altered.loc[altered.index[0], "text"] = "ZZZZZ"
    with pytest.raises(mfb.MatchedFontBenchmarkError, match="ground-truth text mismatch"):
        mfb._verify_source_tables(replace(bundle, ground_truth=altered), protocol, case_manifest)

    duplicated = pd.concat([bundle.ground_truth, bundle.ground_truth.iloc[[0]]], ignore_index=True)
    with pytest.raises(mfb.MatchedFontBenchmarkError, match="301 rows, expected 300"):
        mfb._verify_source_tables(replace(bundle, ground_truth=duplicated), protocol, case_manifest)

    replaced = bundle.ground_truth.copy()
    replaced.loc[replaced.index[1], "source_id"] = replaced.loc[0, "source_id"]
    with pytest.raises(mfb.MatchedFontBenchmarkError, match="do not match the plan"):
        mfb._verify_source_tables(replace(bundle, ground_truth=replaced), protocol, case_manifest)

    missing = bundle.ground_truth.iloc[1:].reset_index(drop=True)
    with pytest.raises(mfb.MatchedFontBenchmarkError, match="rows, expected"):
        mfb._verify_source_tables(replace(bundle, ground_truth=missing), protocol, case_manifest)

    wrong_lineage = bundle.ground_truth.copy()
    wrong_lineage.loc[wrong_lineage.index[0], "case_id"] = "case-does-not-exist"
    with pytest.raises(mfb.MatchedFontBenchmarkError, match="ground-truth lineage mismatch"):
        mfb._verify_source_tables(
            replace(bundle, ground_truth=wrong_lineage), protocol, case_manifest
        )


@requires_outputs
@requires_ground_truth
def test_missing_ground_truth_file_is_rejected_by_the_dataset_loader(tmp_path: Path) -> None:
    for name in (
        "manifest.csv",
        "sources.csv",
        "source_cases.csv",
        "features.csv",
        "dataset_summary.json",
    ):
        (tmp_path / name).symlink_to(DATASET / name)
    with pytest.raises(mfb.MatchedFontBenchmarkError, match="missing M8C ground_truth"):
        mfb.load_dataset(tmp_path)


@requires_outputs
def test_headline_metrics_recompute_from_the_committed_predictions() -> None:
    metadata = _metadata()
    predictions = pd.read_csv(REPORT / "predictions.csv")
    manifest = pd.read_csv(DATASET / "manifest.csv")

    assert len(predictions) == len(manifest) == 6300
    assert predictions["row_id"].is_unique
    assert predictions["sample_id"].is_unique
    assert set(predictions["row_id"]) == set(manifest["row_id"])
    joined = predictions.merge(
        manifest[["row_id", "sample_id", "ssim_target", "source_id", "font_id", "primary"]],
        on=["row_id", "sample_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_manifest"),
    )
    assert len(joined) == 6300
    assert np.array_equal(joined["ssim_target"], joined["ssim_target_manifest"])
    assert np.array_equal(joined["source_id"], joined["source_id_manifest"])
    assert np.array_equal(joined["font_id"], joined["font_id_manifest"])

    def metrics(frame: pd.DataFrame) -> tuple[float, float, float]:
        actual = frame["ssim_target"].to_numpy(dtype=np.float64)
        predicted = frame["prediction"].to_numpy(dtype=np.float64)
        residual = actual - predicted
        ss_res = float(np.sum(residual**2))
        ss_tot = float(np.sum((actual - actual.mean()) ** 2))
        return (
            float(np.mean(np.abs(residual))),
            float(np.sqrt(np.mean(residual**2))),
            1.0 - ss_res / ss_tot,
        )

    primary = predictions[predictions["protocol_scope"] == "primary"]
    assert len(primary) == 5700
    mae, rmse, r2 = metrics(primary)
    expected = metadata["metrics"]["overall_primary"]
    assert abs(mae - expected["mae"]) <= 1e-12
    assert abs(rmse - expected["rmse"]) <= 1e-12
    assert abs(r2 - expected["r2"]) <= 1e-12

    for key, group in (
        ("familiar_primary", "familiar"),
        ("m7b_held_out_primary", "m7b_held_out"),
        ("new_held_out_primary", "new_held_out"),
    ):
        subset = primary[primary["font_group"] == group]
        g_mae, g_rmse, g_r2 = metrics(subset)
        expected_group = metadata["metrics"][key]
        assert abs(g_mae - expected_group["mae"]) <= 1e-12
        assert abs(g_rmse - expected_group["rmse"]) <= 1e-12
        assert abs(g_r2 - expected_group["r2"]) <= 1e-12

    clean = predictions[predictions["variant_family"] == "clean"]
    assert np.unique(clean["ssim_target"]).tolist() == [1.0]
    assert metadata["metrics"]["clean_reference"]["r2"] is None
    assert metadata["metrics"]["clean_reference"]["r2_suppressed"] is True

    diagnostic = predictions[
        (predictions["variant_family"] == "elastic") & (predictions["severity_level"] == 1)
    ]
    assert float(diagnostic["ssim_target"].std(ddof=0)) < mfb.NEAR_DEGENERATE_SD
    assert metadata["metrics"]["diagnostic_0_5px"]["r2"] is None
    assert metadata["metrics"]["diagnostic_0_5px"]["r2_suppressed"] is True


@requires_outputs
def test_committed_report_and_dataset_hashes_match_the_run_metadata() -> None:
    metadata = _metadata()
    for name, expected in metadata["outputs"]["hashes"].items():
        assert sha256_file(REPORT / name) == expected

    dataset = metadata["dataset"]
    for filename, key in (
        ("manifest.csv", "manifest_sha256"),
        ("sources.csv", "sources_sha256"),
        ("source_cases.csv", "source_cases_sha256"),
        ("features.csv", "features_sha256"),
        ("dataset_summary.json", "dataset_summary_sha256"),
    ):
        assert sha256_file(DATASET / filename) == dataset[key]


@requires_outputs
@requires_ground_truth
def test_reordered_ground_truth_passes_semantic_verification_but_changes_the_file_hash() -> None:
    """Document the known hash-enforcement gap without changing verification.

    M8C's verifier validates ground-truth *content* against the frozen plan.
    It does not compare ``run_metadata.dataset.ground_truth_sha256`` with the
    current file, so a semantics-preserving rewrite is accepted while the
    recorded byte hash becomes stale. The audit records this limitation; this
    test keeps the exact behaviour visible to future maintainers.
    """

    protocol = load_execution_protocol(PROTOCOL)
    case_manifest = load_case_manifest(TEXT_MANIFEST)
    bundle = mfb.load_dataset(DATASET)
    reordered = bundle.ground_truth.iloc[::-1].reset_index(drop=True)
    assert reordered["source_id"].tolist() != bundle.ground_truth["source_id"].tolist()

    # Content-only verification accepts the reordered table ...
    mfb._verify_source_tables(replace(bundle, ground_truth=reordered), protocol, case_manifest)

    # ... although its byte hash no longer matches the recorded execution evidence.
    frame = reordered.to_csv(index=False).encode("utf-8")
    assert hashlib.sha256(frame).hexdigest() != _metadata()["dataset"]["ground_truth_sha256"]
