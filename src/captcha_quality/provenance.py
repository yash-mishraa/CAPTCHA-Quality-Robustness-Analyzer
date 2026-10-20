"""Reproducibility and serialisation helpers shared by experiment runners.

These helpers exist so that the Milestone 1 pilot and the Milestone 2 dataset
pipeline record provenance the same way. They are deliberately small: version
and revision lookup, hashing, canonical JSON, and JSON-safe conversion of
non-finite floats.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np


def package_version() -> str:
    """Return the installed ``captcha_quality`` version string."""
    from captcha_quality import __version__

    return __version__


def _git_output(*args: str) -> str | None:
    if shutil.which("git") is None:
        return None
    try:
        completed = subprocess.run(
            ["git", *args],
            capture_output=True,
            check=True,
            text=True,
            cwd=Path(__file__).resolve().parents[2],
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout


def code_revision() -> str:
    """Best-effort short Git revision for provenance; never fails the run.

    A ``-dirty`` suffix marks a working tree with uncommitted changes, so a
    generated summary cannot claim to come from a clean revision it does not
    match. Outside a Git checkout the value is ``unknown``.
    """
    revision = (_git_output("rev-parse", "--short", "HEAD") or "").strip()
    if not revision:
        return "unknown"
    status = _git_output("status", "--porcelain")
    if status is None:
        return revision
    return f"{revision}-dirty" if status.strip() else revision


def sha256_bytes(data: bytes) -> str:
    """Return the hexadecimal SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: object) -> str:
    """Serialise ``payload`` deterministically.

    Keys are sorted, separators are compact, and non-finite floats are rejected
    so that a configuration hash can never silently depend on ``nan`` equality.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def json_safe(value: object) -> object:
    """Replace non-finite floats with strings so JSON output is standards-compliant."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf" if value < 0 else "nan"
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return json_safe(float(value))
    if isinstance(value, np.bool_):
        return bool(value)
    return value
