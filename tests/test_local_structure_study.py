"""Tests for the M6C local structural feature study."""

from __future__ import annotations

import ast
import inspect
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("joblib")
pytest.importorskip("skimage")

from captcha_quality import local_structure_features as lsf
from captcha_quality.features import FEATURE_NAMES
from captcha_quality.imaging import ImageContractError
from captcha_quality.local_structure_study import (
    GLOBAL_BASELINE,
    GLOBAL_PLUS_LOCAL,
    LOCAL_ONLY,
    LocalStructureConfigError,
    _forbidden_identifiers_in_feature_module,
    _severity_scopes,
    load_local_structure_config,
)
from captcha_quality.modelling import TARGET
from captcha_quality.provenance import sha256_file
from captcha_quality.spatial_study import select_ablation

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "local_structure_study_v0_1_0.toml"

EXPECTED_FEATURES = (
    "grad_mag_mean",
    "grad_mag_std",
    "grad_mag_median",
    "grad_mag_p90",
    "orient_hist_00",
    "orient_hist_01",
    "orient_hist_02",
    "orient_hist_03",
    "orient_hist_04",
    "orient_hist_05",
    "orient_hist_06",
    "orient_hist_07",
    "orient_coherence",
    "orient_entropy",
)


def _blank(height: int = 64, width: int = 192, value: int = 255) -> np.ndarray:
    return np.full((height, width), value, dtype=np.uint8)


def _stripes(vertical: bool = True) -> np.ndarray:
    image = _blank()
    if vertical:
        image[:, ::2] = 0
    else:
        image[::2, :] = 0
    return image


def _noise(seed: int = 20261024) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(64, 192), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Feature schema, ordering and definitions
# ---------------------------------------------------------------------------


def test_feature_schema_is_exact_and_ordered() -> None:
    spec = lsf.LocalStructureSpec()
    assert spec.families == (lsf.FAMILY_MAGNITUDE, lsf.FAMILY_ORIENTATION)
    assert spec.feature_names == EXPECTED_FEATURES
    assert len(spec.feature_names) == len(set(spec.feature_names)) == 14
    # Global names never collide with local names.
    assert not set(spec.feature_names) & set(FEATURE_NAMES)


def test_feature_manifest_matches_names_and_orientation_bins() -> None:
    spec = lsf.LocalStructureSpec()
    manifest = lsf.local_structure_manifest(spec)
    assert list(manifest["feature_name"]) == list(spec.feature_names)
    histogram = manifest[manifest["statistic"] == "histogram"]
    assert len(histogram) == spec.orient_bins
    assert list(histogram["bin_index"]) == list(range(spec.orient_bins))
    assert np.allclose(
        histogram["bin_lower_rad"].to_numpy(dtype=float),
        np.arange(spec.orient_bins) * np.pi / spec.orient_bins,
    )
    assert np.allclose(
        histogram["bin_upper_rad"].to_numpy(dtype=float),
        np.arange(1, spec.orient_bins + 1) * np.pi / spec.orient_bins,
    )
    magnitude = manifest[manifest["family"] == lsf.FAMILY_MAGNITUDE]
    assert list(magnitude["statistic"]) == ["mean", "std", "median", "p90"]


def test_spec_rejects_invalid_parameters() -> None:
    with pytest.raises(lsf.LocalStructureFeatureError):
        lsf.LocalStructureSpec(families=())
    with pytest.raises(lsf.LocalStructureFeatureError):
        lsf.LocalStructureSpec(families=("unknown",))
    with pytest.raises(lsf.LocalStructureFeatureError):
        lsf.LocalStructureSpec(orient_bins=1)
    with pytest.raises(lsf.LocalStructureFeatureError):
        lsf.LocalStructureSpec(orient_min_magnitude=-0.1)
    with pytest.raises(lsf.LocalStructureFeatureError):
        lsf.LocalStructureSpec(magnitude_quantile=1.0)


def test_declared_subset_families_produce_a_consistent_schema() -> None:
    magnitude = lsf.LocalStructureSpec(families=(lsf.FAMILY_MAGNITUDE,))
    assert magnitude.feature_names == EXPECTED_FEATURES[:4]
    values = lsf.extract_local_structure_features(_noise(), magnitude)
    assert tuple(values) == magnitude.feature_names

    orientation = lsf.LocalStructureSpec(families=(lsf.FAMILY_ORIENTATION,))
    assert orientation.feature_names == EXPECTED_FEATURES[4:]
    values = lsf.extract_local_structure_features(_noise(), orientation)
    assert tuple(values) == orientation.feature_names


# ---------------------------------------------------------------------------
# Extraction behaviour and degenerate inputs
# ---------------------------------------------------------------------------


def test_uniform_images_produce_defined_zeros() -> None:
    for value in (0, 1, 127, 255):
        features = lsf.extract_local_structure_features(_blank(value=value))
        assert all(quantity == 0.0 for quantity in features.values()), features


def test_structured_images_have_interpretable_orientation() -> None:
    vertical = lsf.extract_local_structure_features(_stripes(vertical=True))
    horizontal = lsf.extract_local_structure_features(_stripes(vertical=False))
    noise = lsf.extract_local_structure_features(_noise())

    assert vertical["orient_coherence"] == pytest.approx(1.0, abs=1e-9)
    assert horizontal["orient_coherence"] == pytest.approx(1.0, abs=1e-9)
    assert vertical["orient_entropy"] == pytest.approx(0.0, abs=1e-9)
    # Vertical stripes vary along x, so the gradient points along x and the
    # orientation is 0; horizontal stripes vary along y and give pi/2.
    assert vertical["orient_hist_00"] == pytest.approx(1.0, abs=1e-9)
    assert horizontal["orient_hist_04"] == pytest.approx(1.0, abs=1e-9)
    # Isotropic noise is incoherent and high-entropy.
    assert noise["orient_coherence"] < 0.2
    assert noise["orient_entropy"] > 0.9


def test_extraction_is_deterministic_and_does_not_mutate_the_input() -> None:
    image = _noise()
    before = image.copy()
    first = lsf.extract_local_structure_features(image)
    second = lsf.extract_local_structure_features(image)
    assert first == second
    assert np.array_equal(image, before)


def test_values_are_finite_and_in_range() -> None:
    for image in (_blank(), _stripes(), _noise(), _stripes(vertical=False)):
        features = lsf.extract_local_structure_features(image)
        assert all(np.isfinite(value) for value in features.values())
        for name in EXPECTED_FEATURES[:4]:
            assert 0.0 <= features[name] <= np.sqrt(2.0) + 1e-12
        for name in EXPECTED_FEATURES[4:]:
            assert 0.0 <= features[name] <= 1.0
        histogram_total = sum(
            features[name] for name in EXPECTED_FEATURES if name.startswith("orient_hist_")
        )
        # A blank image has no strong pixel and is the defined zero case.
        assert histogram_total == pytest.approx(1.0, abs=1e-9) or histogram_total == 0.0


def test_orientation_histogram_is_zero_only_when_no_pixel_is_strong() -> None:
    # A blank image has no strong pixel, so the histogram is explicitly zero.
    blank = lsf.extract_local_structure_features(_blank())
    assert sum(blank[name] for name in EXPECTED_FEATURES if name.startswith("orient_hist_")) == 0.0
    # A very faint image, below the declared weak-gradient threshold, is the
    # same defined zero rather than a division by zero.
    faint = np.full((64, 192), 128, dtype=np.uint8)
    faint[32, :] = 129
    features = lsf.extract_local_structure_features(faint)
    assert features["orient_coherence"] == 0.0
    assert features["orient_entropy"] == 0.0


def test_extraction_rejects_a_violating_image() -> None:
    with pytest.raises(ImageContractError):
        lsf.extract_local_structure_features(np.zeros((64, 192), dtype=np.float64))
    with pytest.raises(ImageContractError):
        lsf.extract_local_structure_features(np.zeros((64, 192, 3), dtype=np.uint8))
    with pytest.raises(ImageContractError):
        lsf.extract_local_structure_features(np.zeros((5, 5), dtype=np.uint8))


def test_batch_table_is_opaque_and_order_preserving() -> None:
    image = _noise()
    table = lsf.extract_local_structure_feature_table([("sample-b", image), ("sample-a", image)])
    assert tuple(table.columns) == ("sample_id", *EXPECTED_FEATURES)
    assert list(table["sample_id"]) == ["sample-b", "sample-a"]
    assert table.iloc[0, 1:].tolist() == table.iloc[1, 1:].tolist()

    with pytest.raises(lsf.LocalStructureFeatureError):
        lsf.extract_local_structure_feature_table([("same", image), ("same", image)])
    with pytest.raises(lsf.LocalStructureFeatureError):
        lsf.extract_local_structure_feature_table([("", image)])


# ---------------------------------------------------------------------------
# Strict no-reference boundary: structural checks, not column-name checks
# ---------------------------------------------------------------------------

_FORBIDDEN_CODE_IDENTIFIERS = {
    "ssim",
    "ssim_target",
    "psnr",
    "psnr_target",
    "target",
    "reference",
    "reference_image",
    "ref",
    "clean",
    "clean_image",
    "source_id",
    "severity",
    "severity_level",
    "corruption_type",
    "font_id",
    "split",
    "image_path",
    "metadata",
}


def _feature_module_tree() -> ast.Module:
    source = Path(lsf.__file__).read_text(encoding="utf-8")
    return ast.parse(source)


def test_extractor_signature_accepts_only_one_image_and_an_optional_spec() -> None:
    signature = inspect.signature(lsf.extract_local_structure_features)
    parameters = list(signature.parameters.values())
    assert tuple(parameter.name for parameter in parameters) == ("image", "spec")
    for parameter in parameters:
        assert parameter.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
    assert parameters[1].default is None
    required = [
        parameter for parameter in parameters if parameter.default is inspect.Parameter.empty
    ]
    assert [parameter.name for parameter in required] == ["image"]


def test_extractor_rejects_a_second_image_positionally() -> None:
    image = _blank()
    with pytest.raises(TypeError):
        lsf.extract_local_structure_features(image, image)  # type: ignore[arg-type]


def test_feature_module_code_contains_no_forbidden_identifier() -> None:
    tree = _feature_module_tree()
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.arg):
            used.add(node.arg)
        elif isinstance(node, ast.alias):
            used.add((node.asname or node.name).split(".")[-1])
    assert not (_FORBIDDEN_CODE_IDENTIFIERS & used)
    # The runtime check used by the study must agree with this independent scan.
    assert _forbidden_identifiers_in_feature_module() == []


def test_feature_module_cannot_reach_the_upstream_pipeline_or_the_filesystem() -> None:
    tree = _feature_module_tree()
    imported: set[str] = set()
    dynamic_or_io: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Name) and node.id in {
            "open",
            "eval",
            "exec",
            "__import__",
            "importlib",
        }:
            dynamic_or_io.add(node.id)
    assert not dynamic_or_io
    # The only repository module the feature code may import is the shared
    # image-contract validator, which inspects one array and nothing else.
    upstream = {module for module in imported if module.startswith("captcha_quality")}
    assert upstream == {"captcha_quality.imaging"}


def test_sample_id_never_affects_feature_values() -> None:
    image = _noise()
    first = lsf.extract_local_structure_feature_table([("alpha", image)])
    second = lsf.extract_local_structure_feature_table([("omega", image)])
    assert first.iloc[0, 1:].tolist() == second.iloc[0, 1:].tolist()


def test_feature_table_carries_no_target_or_metadata_columns() -> None:
    table = lsf.extract_local_structure_feature_table([("opaque", _noise())])
    assert tuple(table.columns) == ("sample_id", *EXPECTED_FEATURES)
    assert TARGET not in table.columns
    assert "reference_path" not in table.columns
    assert "severity_level" not in table.columns


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_shipped_config_declares_the_predeclared_protocol() -> None:
    config = load_local_structure_config(CONFIG_PATH)
    assert config.global_feature_names == tuple(FEATURE_NAMES)
    assert config.global_feature_set_version == "v1"
    assert config.target == TARGET
    assert config.features.feature_names == EXPECTED_FEATURES
    assert config.features.orient_bins == 8
    assert config.features.orient_min_magnitude == 0.10
    assert config.features.magnitude_quantile == 0.90
    assert config.ablations.names == (GLOBAL_BASELINE, LOCAL_ONLY, GLOBAL_PLUS_LOCAL)
    assert config.ablations.feature_sets == {
        GLOBAL_BASELINE: "global",
        LOCAL_ONLY: "local",
        GLOBAL_PLUS_LOCAL: "combined",
    }
    assert config.ablations.refit[GLOBAL_BASELINE] is False
    assert config.ablations.refit[LOCAL_ONLY] is True
    assert config.ablations.refit[GLOBAL_PLUS_LOCAL] is True
    assert config.model.family == "random_forest"
    assert config.model.n_estimators == 300
    assert config.model.seed == 798117391
    assert config.m6a.feature_set_version == "spatial-v1"
    assert config.selection.rule == "retention_gated_rmse_delta"
    assert config.selection.tolerance_clean == 0.50
    assert config.selection.tolerance_original == 0.10
    assert config.selection.tolerance_severity == 0.25
    assert config.protocol.primary_amplitudes_px == (1.0, 2.0, 3.0)
    assert config.protocol.diagnostic_amplitudes_px == (0.5,)
    assert len(config.config_hash) == 64
    assert len(config.config_sha256) == 64


def test_config_hash_is_stable_and_ignores_the_file_location() -> None:
    first = load_local_structure_config(CONFIG_PATH)
    second = load_local_structure_config(CONFIG_PATH)
    assert first.config_hash == second.config_hash
    assert first.identity_payload() == second.identity_payload()


def _broken_config(tmp_path: Path, mutate) -> Path:
    import tomllib

    payload = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    mutate(payload)
    broken = tmp_path / "broken_local_structure.toml"
    broken.write_text(_dump_toml(payload), encoding="utf-8")
    return broken


def _dump_toml(payload: dict) -> str:
    lines: list[str] = []
    for section, body in payload.items():
        lines.append(f"[{section}]")
        for key, value in body.items():
            if isinstance(value, dict):
                for inner_key, inner_value in value.items():
                    lines.append(f"{key}.{inner_key} = {_toml_value(inner_value)}")
            else:
                lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines)


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace('"', '\\"') + '"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return str(value)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["ablations"]["refit"].__setitem__(GLOBAL_BASELINE, True),
        lambda payload: payload["ablations"]["names"].pop(),
        lambda payload: payload["schema"]["global_feature_names"].pop(),
        lambda payload: payload["m6a"].__setitem__("feature_set_version", "spatial-v2"),
        lambda payload: payload["features"].__setitem__("orient_bins", 6),
        lambda payload: payload["features"].__setitem__("families", ["gradient_magnitude"]),
        lambda payload: payload["protocol"].__setitem__("primary_amplitudes_px", [0.5, 1.0, 2.0]),
        lambda payload: payload["selection"].__setitem__("rule", "other"),
        lambda payload: payload["run"].__setitem__("unexpected", 1),
    ],
)
def test_config_rejects_a_changed_protocol(tmp_path: Path, mutate) -> None:
    broken = _broken_config(tmp_path, mutate)
    with pytest.raises(LocalStructureConfigError):
        load_local_structure_config(broken)


# ---------------------------------------------------------------------------
# End-to-end study
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def study(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    from captcha_quality.local_structure_study import run_local_structure_study

    config = load_local_structure_config(CONFIG_PATH)
    base = tmp_path_factory.mktemp("m6c")
    reports = base / "reports"
    models = base / "models"
    result, paths = run_local_structure_study(
        config,
        output_reports=reports,
        output_models=models,
        verify_images=False,
    )
    return SimpleNamespace(
        config=config, result=result, paths=paths, reports=reports, models=models
    )


def test_end_to_end_run_persists_artifacts_and_verifies(study: SimpleNamespace) -> None:
    from captcha_quality.local_structure_study import verify_local_structure_study

    assert study.result.ok
    assert study.result.selected_name in {candidate.name for candidate in study.result.ablations}

    for name in (
        "validation_slice_metrics.csv",
        "selection_table.csv",
        "test_slice_metrics.csv",
        "uncertainty.csv",
        "paired_uncertainty.csv",
        "test_predictions.csv",
        "isolation_checks.csv",
        "local_feature_manifest.csv",
        "run_metadata.json",
        "report.md",
    ):
        assert (study.reports / name).is_file(), name
    for name in (LOCAL_ONLY, GLOBAL_PLUS_LOCAL):
        assert (study.models / f"{name}.joblib").is_file()
        assert (study.models / f"{name}_card.json").is_file()
    assert study.paths["report"].is_file()

    metadata = json.loads((study.reports / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["selection"]["selected_model"] == study.result.selected_name
    assert metadata["frozen_assets"]["all_unchanged"] is True
    assert metadata["schema"]["global_feature_names"] == list(FEATURE_NAMES)
    assert (
        metadata["schema"]["local_feature_set_version"] == lsf.LOCAL_STRUCTURE_FEATURE_SET_VERSION
    )
    assert metadata["schema"]["local_feature_names"] == list(EXPECTED_FEATURES)
    assert metadata["split"] == {
        "train_sources": 36,
        "val_sources": 12,
        "test_sources": 12,
        "train_samples": 432,
        "validation_samples": 144,
        "primary_test_samples": 36,
        "diagnostic_test_samples": 12,
        "m3_test_samples": 108,
    }

    for name in (LOCAL_ONLY, GLOBAL_PLUS_LOCAL):
        card = json.loads((study.models / f"{name}_card.json").read_text(encoding="utf-8"))
        assert card["target"] == TARGET
        assert card["artifact_sha256"] == sha256_file(study.models / f"{name}.joblib")
        assert card["selection"]["selected_on"] == "val"
        assert card["seed"] == study.config.model.seed
        assert card["local_feature_set_version"] == (lsf.LOCAL_STRUCTURE_FEATURE_SET_VERSION)
        assert card["feature_names"] == list(
            study.config.feature_names_for(study.config.ablations.feature_sets[name])
        )

    report = verify_local_structure_study(
        study.config, output_reports=study.reports, output_models=study.models
    )
    assert report["passed"], [check for check in report["checks"] if not check.passed]


def test_metadata_records_the_no_reference_boundary(study: SimpleNamespace) -> None:
    metadata = json.loads((study.reports / "run_metadata.json").read_text(encoding="utf-8"))
    boundary = metadata["no_reference_boundary"]
    assert boundary["clean_reference_used_for_features"] is False
    assert boundary["target_or_metadata_inputs"] == []
    assert boundary["required_inputs"] == ["image"]
    assert set(boundary["checks"]) == {
        "no_reference_signature",
        "no_reference_identifiers",
    }
    assert all(check["passed"] for check in boundary["checks"].values())

    checks = pd.read_csv(study.reports / "isolation_checks.csv")
    named = checks.set_index("check")
    for name in (
        "no_reference_signature",
        "no_reference_identifiers",
        "local_feature_schema",
        "local_feature_leakage",
    ):
        assert name in named.index
        assert bool(named.loc[name, "passed"])


def test_champion_reference_metrics_are_unchanged(study: SimpleNamespace) -> None:
    table = study.result.test_metrics
    expected = {
        "geometric_primary": (0.049275, 0.059006, 0.185531),
        "clean_reference": (0.001267, 0.002275, float("nan")),
        "original_blur_noise": (0.017410, 0.023328, 0.981409),
        "m3_test_overall": (0.015616, 0.022007, 0.987157),
    }
    for scope, (mae, rmse, r2) in expected.items():
        row = table[(table["model"] == GLOBAL_BASELINE) & (table["scope"] == scope)].iloc[0]
        assert float(row["mae"]) == pytest.approx(mae, abs=1e-5)
        assert float(row["rmse"]) == pytest.approx(rmse, abs=1e-5)
        if np.isfinite(r2):
            assert float(row["r2"]) == pytest.approx(r2, abs=1e-5)


def test_report_and_tables_are_deterministic(study: SimpleNamespace, tmp_path: Path) -> None:
    from captcha_quality.local_structure_study import run_local_structure_study

    second_reports = tmp_path / "reports"
    second_models = tmp_path / "models"
    run_local_structure_study(
        study.config,
        output_reports=second_reports,
        output_models=second_models,
        verify_images=False,
    )
    for name in (
        "validation_slice_metrics.csv",
        "selection_table.csv",
        "test_slice_metrics.csv",
        "uncertainty.csv",
        "paired_uncertainty.csv",
        "test_predictions.csv",
        "isolation_checks.csv",
        "local_feature_manifest.csv",
        "report.md",
    ):
        assert (study.reports / name).read_bytes() == (second_reports / name).read_bytes(), name

    # Model bundles are deterministic too.
    for name in (LOCAL_ONLY, GLOBAL_PLUS_LOCAL):
        assert (study.models / f"{name}.joblib").read_bytes() == (
            second_models / f"{name}.joblib"
        ).read_bytes()
        assert (study.models / f"{name}_card.json").read_bytes() == (
            second_models / f"{name}_card.json"
        ).read_bytes()

    # The only run-varying fields are the measured elapsed time and the
    # absolute output location recorded for each artifact.
    def normalise(metadata: dict) -> dict:
        metadata["runtime"].pop("elapsed_seconds_in_process")
        for entry in metadata.get("artifacts", {}).values():
            entry.pop("path", None)
        return metadata

    first = normalise(json.loads((study.reports / "run_metadata.json").read_text(encoding="utf-8")))
    second = normalise(
        json.loads((second_reports / "run_metadata.json").read_text(encoding="utf-8"))
    )
    assert first == second


def test_verification_detects_a_tampered_artifact(study: SimpleNamespace, tmp_path: Path) -> None:
    from captcha_quality.local_structure_study import verify_local_structure_study

    reports = tmp_path / "reports"
    models = tmp_path / "models"
    shutil.copytree(study.reports, reports)
    shutil.copytree(study.models, models)
    metrics_path = reports / "test_slice_metrics.csv"
    metrics_path.write_text(
        metrics_path.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8"
    )
    report = verify_local_structure_study(
        study.config, output_reports=reports, output_models=models
    )
    assert not report["passed"]
    assert any(not check.passed for check in report["checks"])


def test_source_groups_never_cross_splits(study: SimpleNamespace) -> None:
    retention = study.result.frames.retention
    for frame in (
        retention.expanded_train,
        retention.expanded_val,
        retention.primary_test,
        retention.diagnostic_test,
        retention.m3_test,
    ):
        assert frame.groupby("source_id")["split"].nunique().max() == 1


def test_diagnostic_rung_is_excluded_from_fit_and_primary_slices(
    study: SimpleNamespace,
) -> None:
    retention = study.result.frames.retention
    for frame in (
        retention.expanded_train,
        retention.expanded_val,
        retention.primary_test,
    ):
        m5a = frame[frame["dataset"] == "m5a"]
        if m5a.empty:
            continue
        assert not (m5a["amplitude_px"] == 0.5).any()
    assert (retention.diagnostic_test["amplitude_px"] == 0.5).all()


def test_selection_is_a_function_of_validation_metrics_only(
    study: SimpleNamespace,
) -> None:
    validation = study.result.validation_metrics
    assert set(validation["split"]) == {"val"}
    recomputed, table = select_ablation(
        validation,
        study.config.selection,
        severity_scopes=_severity_scopes(study.config),
    )
    assert recomputed == study.result.selected_name
    assert table["selected"].sum() == 1
    assert set(validation["scope"]) >= {
        "geometric_primary",
        "clean_reference",
        "original_blur_noise",
        "geometric_primary_severity_2",
        "geometric_primary_severity_3",
        "geometric_primary_severity_4",
    }


def test_selection_falls_back_to_the_champion_when_nothing_qualifies(
    study: SimpleNamespace,
) -> None:
    rows = [
        (GLOBAL_BASELINE, "geometric_primary", 0.05),
        (GLOBAL_BASELINE, "clean_reference", 0.002),
        (GLOBAL_BASELINE, "original_blur_noise", 0.020),
        (GLOBAL_BASELINE, "geometric_primary_severity_2", 0.05),
        (GLOBAL_BASELINE, "geometric_primary_severity_3", 0.025),
        (GLOBAL_BASELINE, "geometric_primary_severity_4", 0.06),
        ("worse", "geometric_primary", 0.08),
        ("worse", "clean_reference", 0.010),
        ("worse", "original_blur_noise", 0.030),
        ("worse", "geometric_primary_severity_2", 0.08),
        ("worse", "geometric_primary_severity_3", 0.030),
        ("worse", "geometric_primary_severity_4", 0.070),
    ]
    table = pd.DataFrame(
        [
            {
                "model": model,
                "feature_set": "combined",
                "n_features": 22,
                "scope": scope,
                "rmse": rmse,
            }
            for model, scope, rmse in rows
        ]
    )
    selected, scored = select_ablation(
        table,
        study.config.selection,
        severity_scopes=_severity_scopes(study.config),
        reference=GLOBAL_BASELINE,
    )
    assert selected == GLOBAL_BASELINE
    assert bool(scored[scored["model"] == GLOBAL_BASELINE].iloc[0]["selected"])


def test_model_bundles_reload_equivalently(study: SimpleNamespace) -> None:
    import joblib

    frame = study.result.frames.with_local(study.result.frames.retention.primary_test)
    for name in (LOCAL_ONLY, GLOBAL_PLUS_LOCAL):
        in_memory = next(ablation for ablation in study.result.ablations if ablation.name == name)
        reloaded = joblib.load(study.models / f"{name}.joblib")
        expected = in_memory.predict(frame)
        inputs = frame[list(in_memory.feature_names)].to_numpy(dtype=np.float64)
        actual = np.asarray(reloaded.predict(inputs), dtype=np.float64)
        assert np.array_equal(expected, actual)


def test_test_table_has_the_declared_columns(study: SimpleNamespace) -> None:
    table = pd.read_csv(study.reports / "test_slice_metrics.csv")
    assert list(table.columns) == [
        "model",
        "split",
        "scope",
        "mae",
        "rmse",
        "r2",
        "r2_defined",
        "mean_residual",
        "sd_residual",
        "n_samples",
        "n_sources",
    ]
    assert set(table["split"]) == {"test"}
    assert set(table["model"]) == {GLOBAL_BASELINE, LOCAL_ONLY, GLOBAL_PLUS_LOCAL}
