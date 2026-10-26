from __future__ import annotations

import pandas as pd
import pytest

from app.lib import contracts, data
from app.lib.contracts import ImageIntegrityError
from app.tests.helpers import raises_integrity


def validate(
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    data.validate_core_frames(
        manifest,
        features,
        predictions,
        strict_release=False,
    )


def test_frozen_core_shape_and_composition(core: pd.DataFrame) -> None:
    assert len(core) == contracts.EXPECTED_ROWS
    assert core["sample_id"].is_unique
    assert core["protocol_scope"].value_counts().to_dict() == {
        "primary": contracts.EXPECTED_PRIMARY_ROWS,
        "diagnostic": contracts.EXPECTED_DIAGNOSTIC_ROWS,
    }
    assert core["font_id"].nunique() == contracts.EXPECTED_FONTS
    assert core["case_id"].nunique() == contracts.EXPECTED_CASES
    assert core["source_id"].nunique() == contracts.EXPECTED_SOURCES
    assert (core.groupby("source_id").size() == contracts.EXPECTED_ROWS_PER_SOURCE).all()
    assert not core.loc[:, ["image_sha256", "reference_sha256"]].isna().any().any()


def test_pinned_artifact_hashes() -> None:
    for name, expected in contracts.CORE_HASHES.items():
        path = (
            contracts.M8_REPORT_DIR / name
            if name == "predictions.csv"
            else contracts.DATA_ROOT / name
        )
        data.verify_file_hash(path, expected)

    with raises_integrity("hash mismatch"):
        data.verify_file_hash(
            contracts.DATA_ROOT / "manifest.csv",
            "0" * 64,
        )


@pytest.mark.parametrize("table", ["manifest", "features", "predictions"])
def test_duplicate_keys_are_rejected(
    frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    table: str,
) -> None:
    manifest, features, predictions = (frame.copy() for frame in frames)
    target = {"manifest": manifest, "features": features, "predictions": predictions}[table]
    duplicated = pd.concat([target, target.iloc[[0]]], ignore_index=True)

    with raises_integrity("duplicate sample_id"):
        if table == "manifest":
            validate(duplicated, features, predictions)
        elif table == "features":
            validate(manifest, duplicated, predictions)
        else:
            validate(manifest, features, duplicated)


def test_missing_join_key_is_rejected(
    frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    manifest, features, predictions = (frame.copy() for frame in frames)
    features = features.iloc[:-1].copy()

    with raises_integrity("features do not join"):
        validate(manifest, features, predictions)


def test_invalid_join_key_is_rejected(
    frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    manifest, features, predictions = (frame.copy() for frame in frames)
    features.loc[features.index[0], "sample_id"] = "not-a-frozen-sample"

    with raises_integrity("features do not join"):
        validate(manifest, features, predictions)


def test_missing_columns_are_rejected(
    frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    manifest, features, predictions = (frame.copy() for frame in frames)

    with raises_integrity("missing required columns"):
        validate(manifest, features, predictions.drop(columns=["prediction"]))


def test_missing_required_values_are_rejected(
    frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    manifest, features, predictions = (frame.copy() for frame in frames)
    predictions.loc[predictions.index[0], "prediction"] = float("nan")

    with raises_integrity("missing required values"):
        validate(manifest, features, predictions)


def test_manifest_and_prediction_targets_must_agree(
    frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    manifest, features, predictions = (frame.copy() for frame in frames)
    predictions.loc[predictions.index[0], "ssim_target"] += 0.1

    with raises_integrity("targets disagree"):
        validate(manifest, features, predictions)


def test_manifest_and_prediction_fonts_must_agree(
    frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    manifest, features, predictions = (frame.copy() for frame in frames)
    current = predictions.loc[predictions.index[0], "font_id"]
    predictions.loc[predictions.index[0], "font_id"] = "impact" if current != "impact" else "arial"

    with raises_integrity("font ids disagree"):
        validate(manifest, features, predictions)


def test_recorded_residual_is_target_minus_prediction(
    frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    manifest, features, predictions = (frame.copy() for frame in frames)
    error = (
        predictions["residual"] - (predictions["ssim_target"] - predictions["prediction"])
    ).abs()
    assert error.max() < 1e-12

    wrong = predictions.copy()
    row = wrong.index[0]
    wrong.loc[row, "residual"] = float(wrong.loc[row, "prediction"]) - float(
        wrong.loc[row, "ssim_target"]
    )
    with raises_integrity("residuals are internally inconsistent"):
        validate(manifest, features, wrong)


def test_manifest_dimensions_and_hashes_are_checked(
    frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    manifest, features, predictions = (frame.copy() for frame in frames)
    bad_dimensions = manifest.copy()
    bad_dimensions.loc[bad_dimensions.index[0], "height"] = 63
    with raises_integrity("unexpected image dimensions"):
        validate(bad_dimensions, features, predictions)

    bad_hash = manifest.copy()
    bad_hash.loc[bad_hash.index[0], "image_sha256"] = "not-a-sha256"
    with raises_integrity("invalid hashes"):
        validate(bad_hash, features, predictions)


def test_unknown_scope_and_family_are_rejected(
    frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    manifest, features, predictions = (frame.copy() for frame in frames)
    bad_scope = manifest.copy()
    bad_scope.loc[bad_scope.index[0], "protocol_scope"] = "public"
    with raises_integrity("unknown protocol scope"):
        validate(bad_scope, features, predictions)

    bad_family = manifest.copy()
    bad_family.loc[bad_family.index[0], "variant_family"] = "jpeg"
    with raises_integrity("unknown corruption family"):
        validate(bad_family, features, predictions)


def test_manifest_path_traversal_is_rejected(
    frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    manifest, features, predictions = (frame.copy() for frame in frames)
    manifest.loc[manifest.index[0], "image_path"] = "../ground_truth.csv"

    with pytest.raises(ImageIntegrityError):
        validate(manifest, features, predictions)


def test_report_table_allowlist_rejects_unknown_files() -> None:
    for kind, filename in (
        ("m8", "ground_truth.csv"),
        ("m9", "ground_truth.csv"),
        ("m8", "source_cases.csv"),
    ):
        with raises_integrity("not allowlisted"):
            data.read_report_table(kind, filename)

    with raises_integrity("unknown report collection"):
        data.read_report_table("other", "overall_metrics.csv")

    summary = data.read_report_table("m9", "summary_metrics.csv")
    assert "large_error_proportion" in summary.columns
    fonts = data.read_report_table("m9", "font_condition_metrics.csv")
    assert len(fonts.loc[fonts["scope"] == "interference"]) == 10
    assert "n_sources" in fonts.columns


def test_scope_and_record_helpers(core: pd.DataFrame) -> None:
    overall = data.read_report_table("m8", "overall_metrics.csv")
    row = data.require_scope_row(overall, "all_primary")
    assert float(row["mae"]) == pytest.approx(0.0596072060109396)

    duplicate = pd.concat([overall, overall.iloc[[0]]], ignore_index=True)
    with raises_integrity("duplicate published rows"):
        data.require_scope_row(duplicate, "all_primary")

    assert data.find_record(core, sample_id="not-a-sample") is None
    first = core.iloc[0]
    other_font = "impact" if first["font_id"] != "impact" else "arial"
    assert data.find_record(core, sample_id=first["sample_id"], font_id=other_font) is None
    with raises_integrity("did not resolve"):
        data.find_record(core, font_id=first["font_id"])


def test_composition_counts_are_realised_values() -> None:
    composition = data.read_report_table("m8", "dataset_composition.csv")
    assert data.composition_count(composition, "rows") == 6_300
    assert data.composition_count(composition, "primary_rows") == 5_700
    assert data.composition_count(composition, "diagnostic_rows") == 600
    assert data.composition_count(composition, "sources") == 300

    with raises_integrity("absent or duplicated"):
        data.composition_count(composition, "not-a-component")


def test_ground_truth_is_explicitly_forbidden() -> None:
    assert "ground_truth.csv" in contracts.FORBIDDEN_FILES
    with raises_integrity("not allowlisted"):
        data.read_report_table("m8", "ground_truth.csv")
