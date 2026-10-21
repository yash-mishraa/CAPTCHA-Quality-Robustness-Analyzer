"""Tests for the M6D multi-scale structure-tensor feature study."""

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

from captcha_quality import structure_tensor_features as stf
from captcha_quality.features import FEATURE_NAMES
from captcha_quality.imaging import ImageContractError
from captcha_quality.local_structure_study import load_local_structure_config
from captcha_quality.modelling import TARGET
from captcha_quality.provenance import sha256_file
from captcha_quality.spatial_study import select_ablation
from captcha_quality.structure_tensor_study import (
    GLOBAL_BASELINE,
    GLOBAL_PLUS_STRUCTURE,
    STRUCTURE_ONLY,
    StructureTensorConfigError,
    _forbidden_identifiers_in_feature_module,
    _severity_scopes,
    load_structure_tensor_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "structure_tensor_study_v0_1_0.toml"

EXPECTED_FEATURES = (
    "st_s1_coh_mean",
    "st_s1_coh_p90",
    "st_s1_aniso_mean",
    "st_s1_energy_p90",
    "st_s2_coh_mean",
    "st_s2_coh_p90",
    "st_s2_aniso_mean",
    "st_s2_energy_p90",
    "st_energy_mean",
)


def _blank(height: int = 64, width: int = 192, value: int = 255) -> np.ndarray:
    return np.full((height, width), value, dtype=np.uint8)


def _noise(seed: int = 20261107) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(64, 192), dtype=np.uint8)


def _grating(orientation: str = "vertical", period: float = 16.0) -> np.ndarray:
    x = np.arange(192, dtype=np.float64)
    y = np.arange(64, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    if orientation == "vertical":
        phase = xx
    elif orientation == "horizontal":
        phase = yy
    elif orientation == "diagonal":
        phase = xx + yy
    else:  # pragma: no cover - test helper guard
        raise ValueError(orientation)
    values = 127.5 + 100.0 * np.sin(2.0 * np.pi * phase / period)
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Feature schema, scales and definitions
# ---------------------------------------------------------------------------


def test_feature_schema_is_exact_ordered_and_uses_both_scales() -> None:
    spec = stf.StructureTensorSpec()
    assert spec.scales == (1.0, 3.0)
    assert spec.feature_names == EXPECTED_FEATURES
    assert len(spec.feature_names) == len(set(spec.feature_names)) == 9
    assert not set(spec.feature_names) & set(FEATURE_NAMES)
    assert any("_s1_" in name for name in spec.feature_names)
    assert any("_s2_" in name for name in spec.feature_names)


def test_manifest_matches_names_scales_statistics_and_quantiles() -> None:
    spec = stf.StructureTensorSpec()
    manifest = stf.structure_tensor_manifest(spec)
    assert list(manifest.columns) == [
        "feature_name",
        "family",
        "scale_index",
        "sigma_px",
        "statistic",
        "quantile",
    ]
    assert list(manifest["feature_name"]) == list(spec.feature_names)
    assert set(manifest["family"]) == {stf.FAMILY_STRUCTURE_TENSOR}
    first = manifest[manifest["scale_index"] == 1]
    second = manifest[manifest["scale_index"] == 2]
    assert np.allclose(first[first["feature_name"] != "st_energy_mean"]["sigma_px"], 1.0)
    assert np.allclose(second["sigma_px"], 3.0)
    assert set(first[first["feature_name"] != "st_energy_mean"]["statistic"]) == {
        "coh_mean",
        "coh_p90",
        "aniso_mean",
        "energy_p90",
    }
    assert set(second["statistic"]) == {
        "coh_mean",
        "coh_p90",
        "aniso_mean",
        "energy_p90",
    }
    assert list(manifest[manifest["feature_name"] == "st_energy_mean"]["statistic"]) == [
        "energy_mean"
    ]
    assert np.allclose(
        manifest[manifest["statistic"].isin(["coh_p90", "energy_p90"])]["quantile"],
        0.90,
    )


def test_spec_rejects_invalid_or_unfrozen_parameters() -> None:
    with pytest.raises(stf.StructureTensorFeatureError):
        stf.StructureTensorSpec(scales=())
    with pytest.raises(stf.StructureTensorFeatureError):
        stf.StructureTensorSpec(scales=(0.0, 3.0))
    with pytest.raises(stf.StructureTensorFeatureError):
        stf.StructureTensorSpec(scales=(1.0, 1.0))
    with pytest.raises(stf.StructureTensorFeatureError):
        stf.StructureTensorSpec(coherence_quantile=1.0)
    with pytest.raises(stf.StructureTensorFeatureError):
        stf.StructureTensorSpec(energy_quantile=0.0)
    with pytest.raises(stf.StructureTensorFeatureError):
        stf.StructureTensorSpec(epsilon=0.0)
    with pytest.raises(stf.StructureTensorFeatureError):
        stf.StructureTensorSpec(gaussian_truncate=0.0)
    with pytest.raises(stf.StructureTensorFeatureError):
        stf.StructureTensorSpec(gradient_operator="prewitt")
    with pytest.raises(stf.StructureTensorFeatureError):
        stf.StructureTensorSpec(smoothing_kernel="box")
    with pytest.raises(stf.StructureTensorFeatureError):
        stf.StructureTensorSpec(boundary="constant")


# ---------------------------------------------------------------------------
# Extraction behaviour, known orientations and degenerate inputs
# ---------------------------------------------------------------------------


def test_known_orientation_gratings_are_coherent_at_both_scales() -> None:
    for orientation in ("vertical", "horizontal", "diagonal"):
        features = stf.extract_structure_tensor_features(_grating(orientation))
        assert features["st_s1_coh_mean"] > 0.99
        assert features["st_s1_coh_p90"] > 0.99
        assert features["st_s1_aniso_mean"] > 0.99
        assert features["st_s2_coh_mean"] > 0.99
        assert features["st_s2_coh_p90"] > 0.99
        assert features["st_s2_aniso_mean"] > 0.99
        # Smoothing lowers the high-energy tail while preserving the mean.
        assert features["st_s1_energy_p90"] > features["st_s2_energy_p90"] > 0.0
        assert features["st_energy_mean"] > 0.0


def test_isotropic_noise_is_less_coherent_than_a_grating() -> None:
    noise = stf.extract_structure_tensor_features(_noise())
    grating = stf.extract_structure_tensor_features(_grating("vertical"))
    assert noise["st_s1_coh_mean"] < grating["st_s1_coh_mean"]
    assert noise["st_s2_coh_mean"] < grating["st_s2_coh_mean"]
    assert noise["st_s1_aniso_mean"] < grating["st_s1_aniso_mean"]
    assert 0.0 <= noise["st_s1_coh_mean"] <= 1.0


def test_blank_and_uniform_images_produce_defined_zeros() -> None:
    for value in (0, 1, 127, 255):
        features = stf.extract_structure_tensor_features(_blank(value=value))
        assert all(quantity == 0.0 for quantity in features.values()), features


def test_zero_energy_and_near_zero_tensor_cases_are_finite() -> None:
    blank = stf.extract_structure_tensor_features(_blank())
    assert all(value == 0.0 for value in blank.values())

    faint = _blank(value=128)
    faint[32, :] = 129
    features = stf.extract_structure_tensor_features(faint)
    assert all(np.isfinite(value) for value in features.values())
    assert 0.0 <= features["st_s1_coh_mean"] <= 1.0
    assert 0.0 <= features["st_s1_aniso_mean"] <= 1.0
    assert features["st_energy_mean"] > 0.0


def test_values_are_finite_and_respect_declared_ranges() -> None:
    for image in (_blank(), _grating(), _noise(), _grating("horizontal")):
        features = stf.extract_structure_tensor_features(image)
        assert tuple(features) == EXPECTED_FEATURES
        assert all(np.isfinite(value) for value in features.values())
        for name in EXPECTED_FEATURES:
            if "_coh_" in name or "_aniso_" in name:
                assert 0.0 <= features[name] <= 1.0
            if "_energy_" in name:
                assert features[name] >= 0.0


def test_extraction_is_deterministic_and_does_not_mutate_the_input() -> None:
    image = _noise()
    before = image.copy()
    first = stf.extract_structure_tensor_features(image)
    second = stf.extract_structure_tensor_features(image)
    assert first == second
    assert np.array_equal(image, before)


def test_extraction_rejects_a_violating_image() -> None:
    with pytest.raises(ImageContractError):
        stf.extract_structure_tensor_features(np.zeros((64, 192), dtype=np.float64))
    with pytest.raises(ImageContractError):
        stf.extract_structure_tensor_features(np.zeros((64, 192, 3), dtype=np.uint8))
    with pytest.raises(ImageContractError):
        stf.extract_structure_tensor_features(np.zeros((5, 5), dtype=np.uint8))


def test_batch_table_is_opaque_and_order_preserving() -> None:
    image = _noise()
    table = stf.extract_structure_tensor_feature_table([("sample-b", image), ("sample-a", image)])
    assert tuple(table.columns) == ("sample_id", *EXPECTED_FEATURES)
    assert list(table["sample_id"]) == ["sample-b", "sample-a"]
    assert table.iloc[0, 1:].tolist() == table.iloc[1, 1:].tolist()

    with pytest.raises(stf.StructureTensorFeatureError):
        stf.extract_structure_tensor_feature_table([("same", image), ("same", image)])
    with pytest.raises(stf.StructureTensorFeatureError):
        stf.extract_structure_tensor_feature_table([("", image)])


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
    source = Path(stf.__file__).read_text(encoding="utf-8")
    return ast.parse(source)


def test_extractor_signature_accepts_only_one_image_and_an_optional_spec() -> None:
    signature = inspect.signature(stf.extract_structure_tensor_features)
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
        stf.extract_structure_tensor_features(image, image)  # type: ignore[arg-type]


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
    upstream = {module for module in imported if module.startswith("captcha_quality")}
    assert upstream == {"captcha_quality.imaging"}


def test_sample_id_never_affects_feature_values() -> None:
    image = _noise()
    first = stf.extract_structure_tensor_feature_table([("alpha", image)])
    second = stf.extract_structure_tensor_feature_table([("omega", image)])
    assert first.iloc[0, 1:].tolist() == second.iloc[0, 1:].tolist()


def test_feature_table_carries_no_target_or_metadata_columns() -> None:
    table = stf.extract_structure_tensor_feature_table([("opaque", _noise())])
    assert tuple(table.columns) == ("sample_id", *EXPECTED_FEATURES)
    assert TARGET not in table.columns
    assert "reference_path" not in table.columns
    assert "severity_level" not in table.columns


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_shipped_config_declares_the_predeclared_protocol() -> None:
    config = load_structure_tensor_config(CONFIG_PATH)
    assert config.global_feature_names == tuple(FEATURE_NAMES)
    assert config.global_feature_set_version == "v1"
    assert config.target == TARGET
    assert config.features.scales == (1.0, 3.0)
    assert config.features.feature_names == EXPECTED_FEATURES
    assert config.features.coherence_quantile == 0.90
    assert config.features.energy_quantile == 0.90
    assert config.features.epsilon == 1e-12
    assert config.features.gaussian_truncate == 4.0
    assert config.features.gradient_operator == "sobel_3x3"
    assert config.features.smoothing_kernel == "gaussian"
    assert config.features.boundary == "reflect"
    assert config.ablations.names == (GLOBAL_BASELINE, STRUCTURE_ONLY, GLOBAL_PLUS_STRUCTURE)
    assert config.ablations.feature_sets == {
        GLOBAL_BASELINE: "global",
        STRUCTURE_ONLY: "structure",
        GLOBAL_PLUS_STRUCTURE: "combined",
    }
    assert config.ablations.refit[GLOBAL_BASELINE] is False
    assert config.ablations.refit[STRUCTURE_ONLY] is True
    assert config.ablations.refit[GLOBAL_PLUS_STRUCTURE] is True
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
    assert (
        config.m6c.config_sha256
        == "8ab83394bc07bcb3f372d71cad669dddab739dd18b0ae151175f4da789f61934"
    )
    assert (
        config.m6c.protocol_hash
        == "7c436dc5e179d9efb309bdb2b749ca1fd53fdc29e1bc4963765c7babb615c3d1"
    )


def test_config_hash_is_stable_and_ignores_the_file_location() -> None:
    first = load_structure_tensor_config(CONFIG_PATH)
    second = load_structure_tensor_config(CONFIG_PATH)
    assert first.config_hash == second.config_hash
    assert first.identity_payload() == second.identity_payload()


def test_m6c_pin_matches_the_frozen_local_study() -> None:
    config = load_structure_tensor_config(CONFIG_PATH)
    m6c = load_local_structure_config(config.m6c.config)
    assert sha256_file(config.m6c.config) == config.m6c.config_sha256
    assert m6c.config_hash == config.m6c.protocol_hash


def _broken_config(tmp_path: Path, mutate) -> Path:
    import tomllib

    payload = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    mutate(payload)
    broken = tmp_path / "broken_structure_tensor.toml"
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
        lambda payload: payload["features"].__setitem__("scales", [1.0, 1.1]),
        lambda payload: payload["features"].__setitem__("scales", [1.0]),
        lambda payload: payload["features"].__setitem__("coherence_quantile", 0.80),
        lambda payload: payload["features"].__setitem__("epsilon", 1e-9),
        lambda payload: payload["features"].__setitem__("gradient_operator", "prewitt"),
        lambda payload: payload["features"].__setitem__("boundary", "constant"),
        lambda payload: payload["protocol"].__setitem__("primary_amplitudes_px", [0.5, 1.0, 2.0]),
        lambda payload: payload["selection"].__setitem__("rule", "other"),
        lambda payload: payload["run"].__setitem__("unexpected", 1),
    ],
)
def test_config_rejects_a_changed_protocol(tmp_path: Path, mutate) -> None:
    broken = _broken_config(tmp_path, mutate)
    with pytest.raises(StructureTensorConfigError):
        load_structure_tensor_config(broken)


# ---------------------------------------------------------------------------
# End-to-end study
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def study(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    from captcha_quality.structure_tensor_study import run_structure_tensor_study

    config = load_structure_tensor_config(CONFIG_PATH)
    base = tmp_path_factory.mktemp("m6d")
    reports = base / "reports"
    models = base / "models"
    result, paths = run_structure_tensor_study(
        config,
        output_reports=reports,
        output_models=models,
        verify_images=False,
    )
    return SimpleNamespace(
        config=config, result=result, paths=paths, reports=reports, models=models
    )


def test_end_to_end_run_persists_artifacts_and_verifies(study: SimpleNamespace) -> None:
    from captcha_quality.structure_tensor_study import verify_structure_tensor_study

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
        "structure_tensor_manifest.csv",
        "run_metadata.json",
        "report.md",
    ):
        assert (study.reports / name).is_file(), name
    for name in (STRUCTURE_ONLY, GLOBAL_PLUS_STRUCTURE):
        assert (study.models / f"{name}.joblib").is_file()
        assert (study.models / f"{name}_card.json").is_file()
    assert study.paths["report"].is_file()

    metadata = json.loads((study.reports / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["selection"]["selected_model"] == study.result.selected_name
    assert metadata["frozen_assets"]["all_unchanged"] is True
    assert metadata["schema"]["global_feature_names"] == list(FEATURE_NAMES)
    assert (
        metadata["schema"]["structure_tensor_feature_set_version"]
        == stf.STRUCTURE_TENSOR_FEATURE_SET_VERSION
    )
    assert metadata["schema"]["structure_tensor_feature_names"] == list(EXPECTED_FEATURES)
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

    for name in (STRUCTURE_ONLY, GLOBAL_PLUS_STRUCTURE):
        card = json.loads((study.models / f"{name}_card.json").read_text(encoding="utf-8"))
        assert card["target"] == TARGET
        assert card["artifact_sha256"] == sha256_file(study.models / f"{name}.joblib")
        assert card["selection"]["selected_on"] == "val"
        assert card["seed"] == study.config.model.seed
        assert card["structure_tensor_feature_set_version"] == (
            stf.STRUCTURE_TENSOR_FEATURE_SET_VERSION
        )
        assert card["feature_names"] == list(
            study.config.feature_names_for(study.config.ablations.feature_sets[name])
        )

    report = verify_structure_tensor_study(
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
        "structure_tensor_feature_schema",
        "structure_tensor_feature_leakage",
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
    from captcha_quality.structure_tensor_study import run_structure_tensor_study

    second_reports = tmp_path / "reports"
    second_models = tmp_path / "models"
    run_structure_tensor_study(
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
        "structure_tensor_manifest.csv",
        "report.md",
    ):
        assert (study.reports / name).read_bytes() == (second_reports / name).read_bytes(), name

    for name in (STRUCTURE_ONLY, GLOBAL_PLUS_STRUCTURE):
        assert (study.models / f"{name}.joblib").read_bytes() == (
            second_models / f"{name}.joblib"
        ).read_bytes()
        assert (study.models / f"{name}_card.json").read_bytes() == (
            second_models / f"{name}_card.json"
        ).read_bytes()

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
    from captcha_quality.structure_tensor_study import verify_structure_tensor_study

    reports = tmp_path / "reports"
    models = tmp_path / "models"
    shutil.copytree(study.reports, reports)
    shutil.copytree(study.models, models)
    metrics_path = reports / "test_slice_metrics.csv"
    metrics_path.write_text(
        metrics_path.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8"
    )
    report = verify_structure_tensor_study(
        study.config, output_reports=reports, output_models=models
    )
    assert not report["passed"]
    assert any(not check.passed for check in report["checks"])


def test_frozen_artifacts_are_immutable_during_the_run(study: SimpleNamespace) -> None:
    metadata = json.loads((study.reports / "run_metadata.json").read_text(encoding="utf-8"))
    before = metadata["frozen_assets"]["before"]
    after = metadata["frozen_assets"]["after"]
    assert metadata["frozen_assets"]["all_unchanged"] is True
    assert set(before) == set(after)
    assert before == after


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
                "n_features": 17,
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

    frame = study.result.frames.with_structure(study.result.frames.retention.primary_test)
    for name in (STRUCTURE_ONLY, GLOBAL_PLUS_STRUCTURE):
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
    assert set(table["model"]) == {GLOBAL_BASELINE, STRUCTURE_ONLY, GLOBAL_PLUS_STRUCTURE}
