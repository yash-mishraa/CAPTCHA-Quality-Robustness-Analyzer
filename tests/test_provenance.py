"""Shared provenance and serialisation helpers."""

from __future__ import annotations

import math

import numpy as np

from captcha_quality.provenance import (
    canonical_json,
    code_revision,
    json_safe,
    package_version,
    sha256_bytes,
    sha256_file,
)


def test_canonical_json_is_key_order_independent() -> None:
    first = canonical_json({"b": 1, "a": [2, 3]})
    second = canonical_json({"a": [2, 3], "b": 1})
    assert first == second
    assert first == '{"a":[2,3],"b":1}'


def test_canonical_json_rejects_non_finite() -> None:
    try:
        canonical_json({"x": math.nan})
    except ValueError:
        pass
    else:  # pragma: no cover - the helper must reject NaN
        raise AssertionError("canonical_json accepted NaN")


def test_json_safe_converts_numpy_and_non_finite() -> None:
    payload = {
        "inf": math.inf,
        "nan": math.nan,
        "value": np.float64(1.5),
        "count": np.int64(3),
        "flag": np.bool_(True),
        "items": (1, 2),
    }
    converted = json_safe(payload)
    assert converted == {
        "inf": "inf",
        "nan": "nan",
        "value": 1.5,
        "count": 3,
        "flag": True,
        "items": [1, 2],
    }


def test_hash_helpers_agree(tmp_path) -> None:
    data = b"captcha-quality"
    path = tmp_path / "bytes.bin"
    path.write_bytes(data)
    assert sha256_bytes(data) == sha256_file(path)
    assert len(sha256_bytes(data)) == 64


def test_version_and_revision_are_strings() -> None:
    assert isinstance(package_version(), str) and package_version()
    revision = code_revision()
    assert isinstance(revision, str) and revision
