"""End-to-end tests for the pilot runner, configuration, and gate decisions."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from captcha_quality.corruptions import CorruptionError
from captcha_quality.gate import (
    FAIL,
    GateConfigurationError,
    GateThresholds,
    evaluate_gate,
)
from captcha_quality.pilot import PilotConfig, PilotConfigError, main, run_pilot


def test_pilot_end_to_end_writes_expected_artifacts(tiny_config: PilotConfig) -> None:
    result = run_pilot(tiny_config)

    out = tiny_config.output_dir
    for name in (
        "trials.csv",
        "sanity.csv",
        "summary.md",
        "gate_results.json",
        "levels.csv",
        "separation.csv",
        "examples_gaussian_blur.png",
        "examples_gaussian_noise.png",
    ):
        assert (out / name).is_file(), f"missing artifact: {name}"

    assert len(result.trials) == 2 * 3 * 2  # families x severities x trials
    assert result.sanity["identical"].all()
    assert result.sanity["ssim"].eq(1.0).all()
    assert result.trials.loc[result.trials["severity"] == 0, "identical"].all()

    payload = json.loads((out / "gate_results.json").read_text(encoding="utf-8"))
    assert {entry["verdict"] for entry in payload["verdicts"]} <= {
        "PASS",
        "CONDITIONAL",
        "FAIL",
    }
    assert len(payload["verdicts"]) == 4
    assert "ms_ssim" in payload["deferred"]
    assert payload["counts"]["trials"] == 2

    report = (out / "summary.md").read_text(encoding="utf-8")
    assert "## Gate verdicts" in report
    assert "ms_ssim" in report


def test_pilot_is_reproducible(tiny_config: PilotConfig) -> None:
    first = run_pilot(tiny_config, write_outputs=False)
    second = run_pilot(tiny_config, write_outputs=False)
    assert first.trials.equals(second.trials)
    assert first.sanity.equals(second.sanity)
    assert [(v.family, v.metric, v.verdict) for v in first.verdicts] == [
        (v.family, v.metric, v.verdict) for v in second.verdicts
    ]


def test_tiny_pilot_shows_negative_direction(tiny_config: PilotConfig) -> None:
    result = run_pilot(tiny_config, write_outputs=False)
    rho_table = result.summary["family_rho"]
    assert len(rho_table) == 4
    assert (rho_table["spearman_rho"] < 0).all()


def test_clean_sanity_failure_forces_a_fail_verdict(tiny_config: PilotConfig) -> None:
    result = run_pilot(tiny_config, write_outputs=False)
    broken_sanity = result.sanity.copy()
    broken_sanity["ssim"] = 0.99
    verdicts = evaluate_gate(result.summary, broken_sanity, tiny_config.gate, tiny_config.metrics)
    assert all(verdict.verdict == FAIL for verdict in verdicts)
    assert all("clean_sanity" in verdict.failed_criteria for verdict in verdicts)


def test_cli_runs_the_pilot(tmp_path: Path, font, pilot_config_writer, capsys) -> None:
    config_path = pilot_config_writer(
        output_dir=tmp_path / "ignored",
        font_key=font.key,
    )
    out = tmp_path / "cli-out"
    assert main(["--config", str(config_path), "--output-dir", str(out)]) == 0
    assert (out / "summary.md").is_file()
    captured = capsys.readouterr()
    assert "pilot 'test-pilot'" in captured.out
    assert "gaussian_blur" in captured.out


def test_cli_reports_configuration_errors(tmp_path: Path, capsys) -> None:
    assert main(["--config", str(tmp_path / "missing.toml")]) == 2
    assert "pilot config not found" in capsys.readouterr().err


def test_config_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(PilotConfigError, match="not found"):
        PilotConfig.from_toml(tmp_path / "absent.toml")


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ('metrics = ["ssim", "psnr"]', 'metrics = ["ssim", "bogus"]', "unsupported metric"),
        ('metrics = ["ssim", "psnr"]', 'metrics = ["ssim", "ms_ssim"]', "unsupported metric"),
        ('deferred_metrics = ["ms_ssim"]', 'deferred_metrics = ["bogus"]', "no recorded deferral"),
        ("min_required = 1", "min_required = 2", "must be between 1"),
        ("levels = [0.0, 1.0, 2.5]", "levels = [0.0, 2.5, 1.0]", "non-decreasing"),
    ],
)
def test_config_rejects_invalid_values(
    tmp_path: Path, font, pilot_config_writer, old: str, new: str, match: str
) -> None:
    config_path = pilot_config_writer(
        output_dir=tmp_path / "out",
        font_key=font.key,
    )
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8"
    )
    with pytest.raises((PilotConfigError, CorruptionError), match=match):
        PilotConfig.from_toml(config_path)


def test_config_rejects_unknown_gate_keys(tmp_path: Path, font, pilot_config_writer) -> None:
    config_path = pilot_config_writer(
        output_dir=tmp_path / "out",
        font_key=font.key,
    )
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "ink_correlation_max = 0.95", "ink_correlation_max = 0.95\nbogus = 1"
        ),
        encoding="utf-8",
    )
    with pytest.raises(GateConfigurationError, match="unknown key"):
        PilotConfig.from_toml(config_path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"spearman_max": 0.5},
        {"paired_effect_max": 0.1},
        {"violation_rate_max": 1.5},
        {"eta_squared_min": -0.1},
        {"ssim_range_min": 0.0},
        {"ink_correlation_max": 2.0},
    ],
)
def test_gate_thresholds_validate_their_range(overrides) -> None:
    base = dict(
        spearman_max=-0.9,
        paired_effect_max=-0.5,
        violation_rate_max=0.35,
        group_spearman_max=-0.8,
        eta_squared_min=0.5,
        ssim_range_min=0.1,
        psnr_range_min_db=3.0,
        ink_correlation_max=0.95,
    )
    with pytest.raises(GateConfigurationError):
        GateThresholds(**{**base, **overrides})


def test_gate_thresholds_require_all_keys() -> None:
    with pytest.raises(GateConfigurationError, match="missing key"):
        GateThresholds.from_mapping({"spearman_max": -0.9})


def test_sanity_table_is_deterministic_and_complete(tiny_config: PilotConfig) -> None:
    result = run_pilot(tiny_config, write_outputs=False)
    assert len(result.sanity) == len(result.trials["trial_id"].unique())
    assert isinstance(result.trials, pd.DataFrame)
    assert result.trials["clean_sha256"].nunique() == len(result.trials["trial_id"].unique())
