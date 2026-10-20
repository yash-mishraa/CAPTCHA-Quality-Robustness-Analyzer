"""Smoke tests for the project foundation.

These tests verify that the package is importable and that the repository
configuration is internally consistent. They intentionally do not test any
pipeline behaviour, which is not implemented at Milestone 0.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

import captcha_quality

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_package_exposes_version() -> None:
    assert isinstance(captcha_quality.__version__, str)
    assert re.fullmatch(r"\d+\.\d+\.\d+", captcha_quality.__version__)


def test_pyproject_is_valid_and_declares_supported_python() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)

    project = config["project"]
    assert project["name"] == "captcha-quality-analyzer"
    assert project["requires-python"] == ">=3.11"
    assert "numpy" in " ".join(project["dependencies"])
    assert config["build-system"]["build-backend"] == "hatchling.build"


def test_src_layout_matches_package_name() -> None:
    package_dir = PROJECT_ROOT / "src" / "captcha_quality"
    assert (package_dir / "__init__.py").is_file()
    assert Path(captcha_quality.__file__).resolve().parent == package_dir.resolve()


@pytest.mark.parametrize(
    "relative_path",
    [
        "app",
        "configs",
        "data/raw",
        "data/processed",
        "docs",
        "models",
        "notebooks",
        "reports",
        "scripts",
        "tests",
    ],
)
def test_expected_directories_exist(relative_path: str) -> None:
    assert (PROJECT_ROOT / relative_path).is_dir()


@pytest.mark.parametrize(
    "relative_path",
    [
        "README.md",
        "LICENSE",
        ".gitignore",
        "data/README.md",
        "docs/PROJECT_OVERVIEW.md",
        "docs/METHODOLOGY.md",
        "docs/EXPERIMENT_LOG.md",
    ],
)
def test_required_documents_exist(relative_path: str) -> None:
    assert (PROJECT_ROOT / relative_path).is_file()
