"""Tests for the M7A dataset-adequacy and evaluation audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("joblib")

from captcha_quality import dataset_audit as audit
from captcha_quality import robustness_benchmark as rb
from captcha_quality.corruptions import derive_noise_seed
from captcha_quality.features import FEATURE_NAMES
from captcha_quality.modelling import TARGET, bootstrap_metrics, regression_metrics
from captcha_quality.provenance import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "dataset_adequacy_audit_v0_1_0.toml"
DATA_ROOT = REPO_ROOT / "data" / "processed" / "captcha_quality_v0.1.0"

if not DATA_ROOT.exists():  # pragma: no cover - depends on generated artifacts
    pytest.skip(
        "frozen M7A inputs are not present; generate the upstream artifacts first",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def audit_bundle():
    """Load the frozen inputs once and run the deterministic audit core."""
    config = audit.load_audit_config(CONFIG_PATH)
    inputs = audit.load_audit_inputs(config, verify_images=False)
    result = audit.run_audit_core(inputs)
    return config, inputs, result


@pytest.fixture
def config() -> audit.AuditConfig:
    return audit.load_audit_config(CONFIG_PATH)


def _mutated_config(tmp_path: Path, old: str, new: str) -> Path:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "mutated_audit.toml"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Configuration and pinned inputs
# ---------------------------------------------------------------------------


def test_config_parses_frozen_schema_and_identity(config: audit.AuditConfig) -> None:
    assert config.run.name == "dataset-adequacy-audit"
    assert config.run.seed == 20261108
    assert config.run.n_bootstrap == 2000
    assert config.schema.feature_names == FEATURE_NAMES
    assert config.schema.target == TARGET
    assert config.schema.split_names == audit.SPLITS
    assert config.recommendation.selected_direction == "B"
    assert config.recommendation.directions == audit.DIRECTION_IDS
    assert config.config_sha256 == sha256_file(CONFIG_PATH)
    assert config.config_hash == audit.load_audit_config(CONFIG_PATH).config_hash
    assert len(config.config_hash) == 64


def test_config_rejects_unknown_keys_and_wrong_schema(tmp_path: Path) -> None:
    unknown = _mutated_config(tmp_path, "[audit]\n", "[audit]\nunknown_key = 1\n")
    with pytest.raises(audit.DatasetAuditConfigError, match="unknown key"):
        audit.load_audit_config(unknown)

    wrong_feature = _mutated_config(
        tmp_path,
        'feature_names = [\n  "intensity_mean",',
        'feature_names = [\n  "not_a_frozen_feature",',
    )
    with pytest.raises(audit.DatasetAuditConfigError, match="feature_names"):
        audit.load_audit_config(wrong_feature)

    missing = tmp_path / "missing.toml"
    with pytest.raises(audit.DatasetAuditConfigError, match="missing audit configuration"):
        audit.load_audit_config(missing)


def test_input_pin_mismatch_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    bad = _mutated_config(
        tmp_path,
        'manifest_sha256 = "a6a2cb8654eb51055e11e4aec6dbc94c2a59a6fdf8ba6b4946135bffc8b4cf37"',
        'manifest_sha256 = "0000000000000000000000000000000000000000000000000000000000000000"',
    )
    config = audit.load_audit_config(bad)
    with pytest.raises(audit.DatasetAuditError, match="hash mismatch"):
        audit.load_audit_inputs(config, verify_images=False)


def test_output_paths_cannot_target_frozen_assets() -> None:
    with pytest.raises(audit.DatasetAuditError, match="protected frozen path"):
        audit._safe_output_path("configs")
    with pytest.raises(audit.DatasetAuditError, match="protected frozen path"):
        audit._safe_output_path("reports/retention_aware_selection_v0_1_0")
    with pytest.raises(audit.DatasetAuditError, match="repository root"):
        audit._safe_output_path(REPO_ROOT)


# ---------------------------------------------------------------------------
# Dataset composition, splits and target slices
# ---------------------------------------------------------------------------


def test_inventory_and_split_counts(audit_bundle) -> None:
    config, inputs, result = audit_bundle
    inventory = result.inventory
    accounting = result.split_accounting

    assert len(inventory) == config.expectations.n_sources == 60
    assert inventory["source_id"].is_unique
    assert set(inventory["split"]) == set(audit.SPLITS)
    assert inventory.groupby("split")["source_id"].nunique().to_dict() == {
        "train": 36,
        "val": 12,
        "test": 12,
    }
    assert int(inventory["m3_samples"].sum()) == 540
    assert int(inventory["m5a_samples"].sum()) == 480
    assert int(inventory["m3_distinct_images"].sum()) == 540
    assert int(inventory["m5a_distinct_images"].sum()) == 360
    assert bool(inventory["clean_hash_matches"].all())

    m3 = accounting[accounting["dataset"] == "m3"].set_index("split")
    m5a = accounting[accounting["dataset"] == "m5a"].set_index("split")
    assert m3.loc[list(audit.SPLITS), "n_samples"].tolist() == [324, 108, 108]
    assert m5a.loc[list(audit.SPLITS), "n_samples"].tolist() == [288, 96, 96]
    assert bool(accounting["matches_expectation"].all())
    assert len(inputs.frames.m3_frame) == 540
    assert len(inputs.frames.m5a_frame) == 480


def test_sources_are_disjoint_across_splits(audit_bundle) -> None:
    _, inputs, result = audit_bundle
    inventory = result.inventory
    assert (inventory.groupby("source_id")["split"].nunique() == 1).all()

    m3_sources = inputs.frames.m3_sources
    m5a_sources = inputs.frames.m5a_sources
    assert set(m3_sources["source_id"]) == set(m5a_sources["source_id"])
    joined = m3_sources.merge(
        m5a_sources,
        on="source_id",
        suffixes=("_m3", "_m5a"),
        validate="one_to_one",
    )
    assert (joined["split_m3"] == joined["split_m5a"]).all()
    assert (joined["font_id_m3"] == joined["font_id_m5a"]).all()
    assert (joined["text_length_m3"] == joined["text_length_m5a"]).all()

    for frame in (m3_sources, m5a_sources):
        font_sets = {
            split: set(frame.loc[frame["split"] == split, "font_id"].astype(str))
            for split in audit.SPLITS
        }
        assert set.intersection(*font_sets.values()) == {
            "arial",
            "courier_new",
            "times_new_roman",
        }


def test_target_slice_accounting_and_degeneracy(audit_bundle) -> None:
    config, _, result = audit_bundle
    targets = result.target_distributions

    clean = targets[
        (targets["dataset"] == "m3")
        & (targets["split"] == "test")
        & (targets["slice"] == "clean_reference")
    ].iloc[0]
    diagnostic = targets[
        (targets["dataset"] == "m5a")
        & (targets["split"] == "test")
        & (targets["slice"] == "diagnostic_0_5")
    ].iloc[0]
    primary = targets[
        (targets["dataset"] == "m5a")
        & (targets["split"] == "test")
        & (targets["slice"] == "geometric_primary")
    ].iloc[0]

    assert float(clean["variance"]) == 0.0
    assert float(clean["sd"]) == 0.0
    assert bool(clean["near_degenerate"])
    assert float(diagnostic["sd"]) < config.audit.near_degenerate_sd
    assert bool(diagnostic["near_degenerate"])
    assert int(primary["n_samples"]) == 36
    assert int(primary["n_sources"]) == 12
    assert np.isfinite(float(primary["between_source_variance"]))
    assert np.isfinite(float(primary["within_source_variance"]))

    evaluation = result.split_accounting[result.split_accounting["dataset"] == "evaluation"]
    by_split = evaluation.set_index("split")
    assert int(by_split.loc["primary_test", "n_samples"]) == 36
    assert int(by_split.loc["diagnostic_0_5", "n_samples"]) == 12
    assert int(by_split.loc["m3_test", "n_samples"]) == 108
    assert int(by_split.loc["primary_test", "clean"]) == 0
    assert int(by_split.loc["primary_test", "primary_elastic"]) == 36


# ---------------------------------------------------------------------------
# Residual aggregation, leave-one-source-out and bootstrap
# ---------------------------------------------------------------------------


def test_per_source_aggregation_matches_manual(audit_bundle) -> None:
    _, inputs, result = audit_bundle
    scope = "geometric_primary"
    frame, predicted = audit._scope_predictions(inputs, scope)
    target = frame[TARGET].to_numpy(dtype=np.float64)
    source_id = sorted(frame["source_id"].unique())[0]
    mask = (frame["source_id"] == source_id).to_numpy()
    manual = regression_metrics(target[mask], predicted[mask])
    manual_sse = float(np.sum((target[mask] - predicted[mask]) ** 2))
    total_sse = float(np.sum((target - predicted) ** 2))

    row = result.source_metrics[
        (result.source_metrics["scope"] == scope)
        & (result.source_metrics["source_id"] == source_id)
    ].iloc[0]
    assert int(row["n_samples"]) == int(mask.sum())
    assert float(row["mae"]) == pytest.approx(float(manual["mae"]))
    assert float(row["rmse"]) == pytest.approx(float(manual["rmse"]))
    assert float(row["sse"]) == pytest.approx(manual_sse)
    assert float(row["sse_share"]) == pytest.approx(manual_sse / total_sse)

    for current_scope in audit.SCOPE_ORDER:
        subset = result.source_metrics[result.source_metrics["scope"] == current_scope]
        assert float(subset["sse_share"].sum()) == pytest.approx(1.0)


def test_leave_one_source_out_matches_manual(audit_bundle) -> None:
    _, inputs, result = audit_bundle
    scope = "geometric_primary"
    frame, predicted = audit._scope_predictions(inputs, scope)
    target = frame[TARGET].to_numpy(dtype=np.float64)
    source_id = sorted(frame["source_id"].unique())[1]
    keep = (frame["source_id"] != source_id).to_numpy()
    full = regression_metrics(target, predicted)
    without = regression_metrics(target[keep], predicted[keep])

    row = result.leave_one_source_out[
        (result.leave_one_source_out["scope"] == scope)
        & (result.leave_one_source_out["omitted_source_id"] == source_id)
    ].iloc[0]
    assert int(row["n_sources_remaining"]) == 11
    assert int(row["n_samples_remaining"]) == int(keep.sum())
    assert float(row["rmse_full"]) == pytest.approx(float(full["rmse"]))
    assert float(row["rmse_without"]) == pytest.approx(float(without["rmse"]))
    assert float(row["delta_rmse"]) == pytest.approx(float(without["rmse"]) - float(full["rmse"]))
    assert float(row["delta_mae"]) == pytest.approx(float(without["mae"]) - float(full["mae"]))


def test_grouped_bootstrap_is_deterministic_and_keeps_source_groups(audit_bundle) -> None:
    config, inputs, result = audit_bundle
    scope = "geometric_primary"
    frame, predicted = audit._scope_predictions(inputs, scope)
    seed = derive_noise_seed(config.run.seed, "dataset_audit_bootstrap", scope)
    first = bootstrap_metrics(
        frame,
        predicted,
        n_bootstrap=200,
        seed=seed,
        scope=scope,
    )
    second = bootstrap_metrics(
        frame,
        predicted,
        n_bootstrap=200,
        seed=seed,
        scope=scope,
    )
    pd.testing.assert_frame_equal(first, second)
    assert set(first["n_groups"]) == {12}
    assert set(first["n_samples"]) == {len(frame)}

    uncertainty = result.uncertainty
    assert set(uncertainty["scope"]) == set(config.audit.bootstrap_scopes)
    assert (uncertainty["n_groups"] == 12).all()
    assert (uncertainty["independent_groups"] == 12).all()
    assert (uncertainty["n_bootstrap"] == config.run.n_bootstrap).all()
    finite_interval = np.isfinite(uncertainty["ci_low"]) & np.isfinite(uncertainty["ci_high"])
    assert (
        uncertainty.loc[finite_interval, "ci_low"] <= uncertainty.loc[finite_interval, "ci_high"]
    ).all()


# ---------------------------------------------------------------------------
# Feature diagnostics, claims and recommendation
# ---------------------------------------------------------------------------


def test_feature_diagnostics_are_training_only(audit_bundle) -> None:
    config, inputs, result = audit_bundle
    summary = result.feature_summary
    importance = result.feature_importance
    diagnostics = result.feature_diagnostics

    assert len(summary) == len(FEATURE_NAMES) == 8
    assert not summary["constant"].any()
    assert (summary["duplicate_of"] == "").all()
    assert summary["n_samples"].eq(config.expectations.expanded_train_rows).all()
    assert diagnostics["n_rows"] == 432
    assert diagnostics["n_sources"] == 36
    assert diagnostics["n_features"] == 8
    assert diagnostics["max_abs_correlation"] > 0.99
    assert diagnostics["n_redundant_pairs"] >= 1
    assert importance["importance"].sum() == pytest.approx(1.0)
    assert list(importance["feature"]) != []

    # Training-only diagnostics must not contain test rows or target-derived
    # columns in the feature matrix.
    train = inputs.frames.expanded_train
    assert len(train) == 432
    assert set(train["split"]) == {"train"}
    assert set(FEATURE_NAMES).isdisjoint({"ssim_target", "psnr_target"})
    assert not any(name in inputs.m3_features.columns for name in ("ssim_target", "psnr_target"))

    means = result.feature_slice_means
    assert means["n_samples"].max() <= 432
    assert means["n_sources"].max() <= 36


def test_claim_schema_and_single_recommendation(audit_bundle) -> None:
    config, _, result = audit_bundle
    claims = result.claim_evidence
    directions = result.next_direction_comparison

    expected_columns = {
        "claim_id",
        "claim",
        "classification",
        "evidence",
        "limitations",
        "evidence_scope",
        "evidence_metric",
        "evidence_value",
        "benchmark_coverage",
    }
    assert set(claims.columns) == expected_columns
    assert claims["claim_id"].is_unique
    assert set(claims["classification"]).issubset(set(audit.CLASSIFICATIONS))
    assert set(audit.CLASSIFICATIONS).issubset(set(claims["classification"]))
    assert len(claims) == 12

    assert list(directions["direction"]) == ["A", "B", "C", "D"]
    recommended = directions[directions["recommended"].astype(bool)]
    assert len(recommended) == 1
    assert recommended.iloc[0]["direction"] == config.recommendation.selected_direction
    assert recommended.iloc[0]["direction"] == "B"
    assert result.summary["recommended_direction"] == "B"


# ---------------------------------------------------------------------------
# End-to-end run, verification, frozen immutability and tamper detection
# ---------------------------------------------------------------------------


def test_full_run_is_deterministic_and_verifiable(
    config: audit.AuditConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    before = rb._fingerprint_assets(audit.FROZEN_ASSET_PATHS)
    first_result, first_paths = audit.run_dataset_audit(
        config,
        output_reports=tmp_path / "first",
        verify_images=False,
    )
    after = rb._fingerprint_assets(audit.FROZEN_ASSET_PATHS)
    assert before == after

    second_result, second_paths = audit.run_dataset_audit(
        config,
        output_reports=tmp_path / "second",
        verify_images=False,
    )
    assert first_result.summary == second_result.summary

    for name in (*audit.OUTPUT_TABLE_NAMES, "report.md"):
        assert sha256_file(first_paths[name]) == sha256_file(second_paths[name])

    first_metadata = json.loads(first_paths["run_metadata.json"].read_text(encoding="utf-8"))
    second_metadata = json.loads(second_paths["run_metadata.json"].read_text(encoding="utf-8"))
    assert first_metadata["frozen_assets"]["unchanged"] is True
    assert (
        first_metadata["frozen_assets"]["before_sha256"]
        == first_metadata["frozen_assets"]["after_sha256"]
    )
    assert first_metadata["outputs"] == second_metadata["outputs"]
    assert first_metadata["report_sha256"] == second_metadata["report_sha256"]
    first_metadata.pop("runtime_seconds")
    second_metadata.pop("runtime_seconds")
    assert first_metadata == second_metadata

    verification = audit.verify_dataset_audit(
        config,
        output_reports=tmp_path / "first",
        verify_images=False,
    )
    assert verification["ok"] is True
    assert all(check.passed for check in verification["checks"])
    assert verification["metadata"]["frozen_assets"]["unchanged"] is True


def test_verification_detects_tampered_output(
    config: audit.AuditConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    audit.run_dataset_audit(
        config,
        output_reports=tmp_path / "tamper",
        verify_images=False,
    )
    claim_path = tmp_path / "tamper" / "claim_evidence.csv"
    claim_path.write_text(claim_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    verification = audit.verify_dataset_audit(
        config,
        output_reports=tmp_path / "tamper",
        verify_images=False,
    )
    assert verification["ok"] is False
    failed = {check.name for check in verification["checks"] if not check.passed}
    assert "output_hash_claim_evidence.csv" in failed


def test_cli_run_and_verify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    output = tmp_path / "cli"
    assert (
        audit.main(
            [
                "run",
                "--config",
                str(CONFIG_PATH),
                "--output-reports",
                str(output),
                "--skip-image-verification",
            ]
        )
        == 0
    )
    assert (output / "report.md").is_file()
    assert (output / "run_metadata.json").is_file()
    assert (
        audit.main(
            [
                "verify",
                "--config",
                str(CONFIG_PATH),
                "--output-reports",
                str(output),
                "--skip-image-verification",
            ]
        )
        == 0
    )
