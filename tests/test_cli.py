"""End-to-end CLI behaviour and error handling."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from captcha_quality.cli import main


def test_generate_verify_and_features_commands(
    dataset_config_writer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = dataset_config_writer(output_dir=tmp_path / "d")
    assert main(["generate", "--config", str(config)]) == 0
    captured = capsys.readouterr()
    assert "generated 3 sources and 15 samples" in captured.out
    dataset = tmp_path / "d"
    assert (dataset / "manifest.csv").is_file()

    assert main(["verify", "--dataset", str(dataset)]) == 0
    assert "verification: PASS" in capsys.readouterr().out

    assert main(["features", "--dataset", str(dataset)]) == 0
    assert "rewrote" in capsys.readouterr().out


def test_generate_skip_features_then_rebuild(dataset_config_writer, tmp_path: Path) -> None:
    config = dataset_config_writer(output_dir=tmp_path / "d")
    assert main(["generate", "--config", str(config), "--skip-features"]) == 0
    features = pd.read_csv(tmp_path / "d" / "features.csv")
    assert features.empty
    assert main(["features", "--dataset", str(tmp_path / "d")]) == 0
    assert len(pd.read_csv(tmp_path / "d" / "features.csv")) == 15


def test_compare_command(dataset_config_writer, tmp_path: Path, capsys) -> None:
    config = dataset_config_writer(output_dir=tmp_path / "unused")
    assert main(["generate", "--config", str(config), "--output-dir", str(tmp_path / "a")]) == 0
    assert main(["generate", "--config", str(config), "--output-dir", str(tmp_path / "b")]) == 0
    assert main(["compare", "--a", str(tmp_path / "a"), "--b", str(tmp_path / "b")]) == 0
    assert "verification: PASS" in capsys.readouterr().out


def test_compare_command_detects_difference(dataset_config_writer, tmp_path: Path) -> None:
    config = dataset_config_writer(output_dir=tmp_path / "unused")
    main(["generate", "--config", str(config), "--output-dir", str(tmp_path / "a")])
    main(["generate", "--config", str(config), "--output-dir", str(tmp_path / "b")])
    manifest = pd.read_csv(tmp_path / "b" / "manifest.csv")
    manifest.loc[0, "ssim_target"] = 0.5
    manifest.to_csv(tmp_path / "b" / "manifest.csv", index=False, lineterminator="\n")
    assert main(["compare", "--a", str(tmp_path / "a"), "--b", str(tmp_path / "b")]) == 1


def test_diagnose_command(dataset_config_writer, tmp_path: Path, capsys) -> None:
    config = dataset_config_writer(output_dir=tmp_path / "d")
    main(["generate", "--config", str(config)])
    code = main(
        [
            "diagnose",
            "--dataset",
            str(tmp_path / "d"),
            "--config",
            str(config),
            "--output-dir",
            str(tmp_path / "reports"),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "verdict:" in out
    assert (tmp_path / "reports" / "diagnostics.md").is_file()


def test_missing_config_returns_error_code(tmp_path: Path, capsys) -> None:
    code = main(["generate", "--config", str(tmp_path / "absent.toml")])
    assert code == 2
    assert "not found" in capsys.readouterr().err


def test_existing_output_without_overwrite_returns_error_code(
    dataset_config_writer, tmp_path: Path, capsys
) -> None:
    config = dataset_config_writer(output_dir=tmp_path / "d")
    assert main(["generate", "--config", str(config)]) == 0
    assert main(["generate", "--config", str(config)]) == 2
    assert "already exists" in capsys.readouterr().err


def test_verify_tampered_dataset_returns_one(dataset_config_writer, tmp_path: Path) -> None:
    config = dataset_config_writer(output_dir=tmp_path / "d")
    main(["generate", "--config", str(config)])
    manifest = pd.read_csv(tmp_path / "d" / "manifest.csv")
    manifest.loc[0, "ssim_target"] = 0.0
    manifest.to_csv(tmp_path / "d" / "manifest.csv", index=False, lineterminator="\n")
    assert main(["verify", "--dataset", str(tmp_path / "d")]) == 1
