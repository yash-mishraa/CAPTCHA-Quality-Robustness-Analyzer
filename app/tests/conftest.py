from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from app import content
from app.lib import contracts, data


@pytest.fixture(scope="session")
def core() -> pd.DataFrame:
    return data.read_core()


@pytest.fixture(scope="session")
def frames(core: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = core.loc[:, list(contracts.MANIFEST_COLUMNS)].copy()
    features = core.loc[:, ["sample_id", *contracts.FEATURE_COLUMNS]].copy()
    predictions = core.loc[:, list(contracts.PREDICTION_COLUMNS)].copy()
    return manifest, features, predictions


@pytest.fixture(scope="session")
def specimen(core: pd.DataFrame) -> pd.Series:
    matches = core.loc[core["sample_id"] == content.ILLUSTRATIVE_SAMPLE_ID]
    assert len(matches) == 1
    return matches.iloc[0]


@pytest.fixture
def image_root(tmp_path: Path, specimen: pd.Series) -> Path:
    root = tmp_path / "dataset"
    for column in ("image_path", "reference_path"):
        relative = str(specimen[column])
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(contracts.DATA_ROOT / relative, destination)
    return root
