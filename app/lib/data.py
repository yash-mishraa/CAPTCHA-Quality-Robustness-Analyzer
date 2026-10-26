"""Read-only loaders and integrity guards for the frozen research artifacts.

The functions in this module are deliberately boring: they read pinned CSV
files, validate their schemas and joins, and return private copies.  They never
write, train, fit, score, or generate data.  A failed contract raises
:class:`IntegrityError` with a generic user-facing message; the technical detail
is kept on the exception for tests and local debugging only.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError

from app.lib import contracts
from app.lib.contracts import ImageIntegrityError, IntegrityError

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class VerifiedImage:
    """A manifest-backed image whose bytes and dimensions have been checked."""

    data: bytes
    image: Image.Image
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise IntegrityError(f"missing pinned artifact: {path.name}")
    actual = sha256_file(path)
    if actual != expected:
        raise IntegrityError(f"hash mismatch for {path.name}")


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise IntegrityError(f"{label}: missing required columns {missing}")


def _require_values(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = [column for column in columns if frame[column].isna().any()]
    if missing:
        raise IntegrityError(f"{label}: missing required values in {missing}")


def _read_csv(
    path: Path, usecols: tuple[str, ...], dtype: Mapping[str, str] | None = None
) -> pd.DataFrame:
    if not path.is_file():
        raise IntegrityError(f"missing artifact: {path.name}")
    try:
        return pd.read_csv(path, usecols=list(usecols), dtype=dtype)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise IntegrityError(f"unreadable artifact {path.name}: {error}") from error


def resolve_image_path(relative: Any, data_root: Path = contracts.DATA_ROOT) -> Path:
    """Resolve a manifest image path inside the dataset root.

    Only relative ``images/**.png`` paths are accepted.  Absolute paths,
    traversal segments, empty parts, backslashes, unexpected suffixes, and
    symlinks are rejected before any file is opened.
    """

    if not isinstance(relative, str):
        raise ImageIntegrityError("image path is not a string", contracts.IMAGE_UNAVAILABLE_MESSAGE)
    if not relative or relative != relative.strip():
        raise ImageIntegrityError("empty or padded image path", contracts.IMAGE_UNAVAILABLE_MESSAGE)
    if "\x00" in relative or "\\" in relative:
        raise ImageIntegrityError(
            "control or backslash in image path", contracts.IMAGE_UNAVAILABLE_MESSAGE
        )

    pure = PurePosixPath(relative)
    if pure.is_absolute():
        raise ImageIntegrityError("absolute image path", contracts.IMAGE_UNAVAILABLE_MESSAGE)

    parts = relative.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ImageIntegrityError(
            "traversal or empty path segment", contracts.IMAGE_UNAVAILABLE_MESSAGE
        )
    if len(parts) < 2 or parts[0] != "images":
        raise ImageIntegrityError(
            "image path is outside images/", contracts.IMAGE_UNAVAILABLE_MESSAGE
        )
    if pure.suffix != ".png":
        raise ImageIntegrityError("unexpected image suffix", contracts.IMAGE_UNAVAILABLE_MESSAGE)

    root = data_root.resolve()
    images_root = (data_root / "images").resolve()
    candidate = root.joinpath(*parts)
    resolved = candidate.resolve()

    if not resolved.is_relative_to(root) or not resolved.is_relative_to(images_root):
        raise ImageIntegrityError(
            "resolved image path escapes dataset root", contracts.IMAGE_UNAVAILABLE_MESSAGE
        )
    if candidate.is_symlink():
        raise ImageIntegrityError("symlinked image path", contracts.IMAGE_UNAVAILABLE_MESSAGE)
    for parent in candidate.parents:
        if parent == root or parent == root.parent:
            break
        if parent.is_symlink():
            raise ImageIntegrityError(
                "symlinked image directory", contracts.IMAGE_UNAVAILABLE_MESSAGE
            )
    return resolved


def load_verified_image(
    row: Mapping[str, Any] | pd.Series,
    role: str,
    data_root: Path = contracts.DATA_ROOT,
) -> VerifiedImage:
    """Read, hash-check, and decode one selected manifest image.

    ``role`` is either ``image`` (the corrupted variant) or ``reference`` (the
    clean source image).  The manifest records a SHA-256 of the decoded
    grayscale pixel bytes, not of the PNG container, so the check is made
    against ``image.tobytes()``.  The image is only returned after the pixel
    hash and the expected 192x64 grayscale geometry both pass.
    """

    if role not in {"image", "reference"}:
        raise ValueError("role must be 'image' or 'reference'")

    path_column = "image_path" if role == "image" else "reference_path"
    hash_column = "image_sha256" if role == "image" else "reference_sha256"
    relative = row[path_column]
    expected = row[hash_column]
    path = resolve_image_path(relative, data_root)

    if not isinstance(expected, str) or not _HEX64.fullmatch(expected):
        raise ImageIntegrityError("invalid manifest hash", contracts.IMAGE_HASH_MESSAGE)
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ImageIntegrityError(
            f"image unavailable: {error}", contracts.IMAGE_UNAVAILABLE_MESSAGE
        ) from error

    try:
        with Image.open(BytesIO(data)) as opened:
            opened.load()
            if opened.format != "PNG":
                raise ImageIntegrityError("unexpected image format", contracts.IMAGE_HASH_MESSAGE)
            if opened.size != (contracts.EXPECTED_WIDTH, contracts.EXPECTED_HEIGHT):
                raise ImageIntegrityError(
                    "unexpected image dimensions", contracts.IMAGE_HASH_MESSAGE
                )
            if opened.mode != "L":
                raise ImageIntegrityError("unexpected image mode", contracts.IMAGE_HASH_MESSAGE)
            image = opened.copy()
    except ImageIntegrityError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ImageIntegrityError(
            f"unreadable image: {error}", contracts.IMAGE_HASH_MESSAGE
        ) from error

    digest = hashlib.sha256(image.tobytes()).hexdigest()
    if digest != expected:
        raise ImageIntegrityError(
            "image pixel hash does not match manifest", contracts.IMAGE_HASH_MESSAGE
        )

    return VerifiedImage(data=data, image=image, sha256=digest)


def _validate_manifest_paths(manifest: pd.DataFrame, data_root: Path) -> None:
    for column in ("image_path", "reference_path"):
        for value in manifest[column]:
            resolve_image_path(value, data_root)


def _validate_hashes(manifest: pd.DataFrame) -> None:
    for column in ("image_sha256", "reference_sha256"):
        invalid = [
            value
            for value in manifest[column]
            if not isinstance(value, str) or not _HEX64.fullmatch(value)
        ]
        if invalid:
            raise IntegrityError(f"manifest contains invalid hashes in {column}")


def _validate_family_severity(manifest: pd.DataFrame) -> None:
    allowed = {
        ("clean", "diagnostic"): {0},
        ("elastic", "diagnostic"): {1},
        ("elastic", "primary"): {2, 3, 4},
        ("gaussian_blur", "primary"): {1, 2, 3, 4},
        ("gaussian_noise", "primary"): {1, 2, 3, 4},
        ("thin_lines", "primary"): {1, 2, 3, 4},
        ("arc_strokes", "primary"): {1, 2, 3, 4},
    }
    for (family, scope), severities in allowed.items():
        selected = manifest[
            (manifest["variant_family"] == family) & (manifest["protocol_scope"] == scope)
        ]
        observed = set(int(value) for value in selected["severity_level"].dropna().unique())
        if not observed or not observed.issubset(severities):
            raise IntegrityError(f"unexpected severities for {family}/{scope}: {sorted(observed)}")
    unexpected = manifest[
        ~manifest.apply(
            lambda row: (row["variant_family"], row["protocol_scope"]) in allowed,
            axis=1,
        )
    ]
    if not unexpected.empty:
        raise IntegrityError("manifest contains an unknown family/scope combination")


def validate_core_frames(
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    data_root: Path = contracts.DATA_ROOT,
    strict_release: bool = True,
) -> None:
    """Validate the three core tables and their one-to-one join.

    Structural checks always run.  Release-composition checks (the exact M8C
    counts and roster) can be disabled by tests that intentionally mutate a
    small copy of the real tables.
    """

    _require_columns(manifest, contracts.MANIFEST_COLUMNS, "manifest")
    _require_columns(
        features,
        ("sample_id", *contracts.FEATURE_COLUMNS),
        "features",
    )
    _require_columns(predictions, contracts.PREDICTION_COLUMNS, "predictions")

    _require_values(
        manifest,
        tuple(column for column in contracts.MANIFEST_COLUMNS if column != "amplitude_px"),
        "manifest",
    )
    _require_values(features, ("sample_id", *contracts.FEATURE_COLUMNS), "features")
    _require_values(predictions, contracts.PREDICTION_COLUMNS, "predictions")

    if manifest["sample_id"].duplicated().any():
        raise IntegrityError("manifest contains duplicate sample_id values")
    if features["sample_id"].duplicated().any():
        raise IntegrityError("features contain duplicate sample_id values")
    if predictions["sample_id"].duplicated().any():
        raise IntegrityError("predictions contain duplicate sample_id values")

    manifest_ids = set(manifest["sample_id"])
    feature_ids = set(features["sample_id"])
    prediction_ids = set(predictions["sample_id"])
    if manifest_ids != feature_ids:
        raise IntegrityError("features do not join one-to-one to the manifest")
    if manifest_ids != prediction_ids:
        raise IntegrityError("predictions do not join one-to-one to the manifest")

    joined = manifest[["sample_id", "ssim_target", "font_id"]].merge(
        predictions[["sample_id", "ssim_target", "font_id"]],
        on="sample_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_manifest", "_prediction"),
    )
    if (joined["ssim_target_manifest"] - joined["ssim_target_prediction"]).abs().max() > 1e-9:
        raise IntegrityError("manifest and prediction targets disagree")
    if (joined["font_id_manifest"] != joined["font_id_prediction"]).any():
        raise IntegrityError("manifest and prediction font ids disagree")

    residual_error = (
        (predictions["residual"] - (predictions["ssim_target"] - predictions["prediction"]))
        .abs()
        .max()
    )
    absolute_error = (predictions["abs_residual"] - predictions["residual"].abs()).abs().max()
    if residual_error > 1e-9 or absolute_error > 1e-9:
        raise IntegrityError("precomputed residuals are internally inconsistent")

    if not set(manifest["protocol_scope"]).issubset(set(contracts.PROTOCOL_SCOPES)):
        raise IntegrityError("manifest contains an unknown protocol scope")
    if not set(manifest["variant_family"]).issubset(set(contracts.VARIANT_FAMILIES)):
        raise IntegrityError("manifest contains an unknown corruption family")

    if (manifest["height"] != contracts.EXPECTED_HEIGHT).any() or (
        manifest["width"] != contracts.EXPECTED_WIDTH
    ).any():
        raise IntegrityError("manifest contains unexpected image dimensions")

    _validate_hashes(manifest)
    _validate_manifest_paths(manifest, data_root)
    _validate_family_severity(manifest)

    if not strict_release:
        return

    if len(manifest) != contracts.EXPECTED_ROWS:
        raise IntegrityError("unexpected manifest row count")
    if len(features) != contracts.EXPECTED_ROWS or len(predictions) != contracts.EXPECTED_ROWS:
        raise IntegrityError("unexpected feature or prediction row count")

    primary = manifest["protocol_scope"] == "primary"
    if int(primary.sum()) != contracts.EXPECTED_PRIMARY_ROWS:
        raise IntegrityError("unexpected primary row count")
    if int((~primary).sum()) != contracts.EXPECTED_DIAGNOSTIC_ROWS:
        raise IntegrityError("unexpected diagnostic row count")

    if manifest["font_id"].nunique() != contracts.EXPECTED_FONTS:
        raise IntegrityError("unexpected font count")
    if manifest["case_id"].nunique() != contracts.EXPECTED_CASES:
        raise IntegrityError("unexpected case count")
    if manifest["source_id"].nunique() != contracts.EXPECTED_SOURCES:
        raise IntegrityError("unexpected source count")
    if set(manifest["font_id"]) != set(contracts.FONT_IDS):
        raise IntegrityError("unexpected font roster")

    per_source = manifest.groupby("source_id").size()
    if not (per_source == contracts.EXPECTED_ROWS_PER_SOURCE).all():
        raise IntegrityError("unexpected rows per source")

    if manifest.duplicated(
        subset=[
            "font_id",
            "case_id",
            "variant_family",
            "severity_level",
            "protocol_scope",
        ]
    ).any():
        raise IntegrityError("manifest is not unique on the record key")

    source_pairs = manifest.groupby(["font_id", "case_id"])["source_id"].nunique()
    if not (source_pairs == 1).all() or len(source_pairs) != contracts.EXPECTED_SOURCES:
        raise IntegrityError("expected exactly one source per font-case pair")


def read_core(
    data_root: Path = contracts.DATA_ROOT,
    report_dir: Path = contracts.M8_REPORT_DIR,
    *,
    verify_hashes: bool = True,
) -> pd.DataFrame:
    """Read and validate the manifest, features and prediction tables."""

    core_paths = {
        "manifest.csv": data_root / "manifest.csv",
        "features.csv": data_root / "features.csv",
        "predictions.csv": report_dir / "predictions.csv",
    }
    if verify_hashes:
        for name, path in core_paths.items():
            verify_file_hash(path, contracts.CORE_HASHES[name])

    string_dtypes = {column: "string" for column in contracts.CORE_STRING_COLUMNS}
    manifest = _read_csv(
        core_paths["manifest.csv"],
        contracts.MANIFEST_COLUMNS,
        string_dtypes,
    )
    features = _read_csv(
        core_paths["features.csv"],
        ("sample_id", *contracts.FEATURE_COLUMNS),
        {"sample_id": "string"},
    )
    predictions = _read_csv(
        core_paths["predictions.csv"],
        contracts.PREDICTION_COLUMNS,
        {"sample_id": "string", "font_id": "string", "font_group": "string"},
    )

    validate_core_frames(
        manifest,
        features,
        predictions,
        data_root=data_root,
        strict_release=True,
    )

    prediction_values = predictions.drop(columns=["ssim_target", "font_id"])
    joined = manifest.merge(features, on="sample_id", how="inner", validate="one_to_one")
    joined = joined.merge(prediction_values, on="sample_id", how="inner", validate="one_to_one")

    if len(joined) != contracts.EXPECTED_ROWS:
        raise IntegrityError("the validated core join changed the row count")
    if joined["sample_id"].duplicated().any():
        raise IntegrityError("the validated core join produced duplicate keys")

    return joined.reset_index(drop=True)


def read_report_table(
    kind: str,
    filename: str,
    *,
    verify_hashes: bool = True,
) -> pd.DataFrame:
    """Read one allowlisted report table and validate its required columns."""

    if kind not in contracts.REPORT_TABLES:
        raise IntegrityError(f"unknown report collection: {kind}")
    required = contracts.REPORT_TABLES[kind].get(filename)
    if required is None:
        raise IntegrityError(f"report table is not allowlisted: {filename}")

    path = contracts.REPORT_DIRS[kind] / filename
    if verify_hashes:
        expected = contracts.REPORT_HASHES.get((kind, filename))
        if expected is not None:
            verify_file_hash(path, expected)

    frame = _read_csv(path, required)
    _require_columns(frame, required, filename)
    if frame.empty:
        raise IntegrityError(f"required report table is empty: {filename}")
    return frame.copy()


def find_record(core: pd.DataFrame, **criteria: Any) -> pd.Series | None:
    """Return exactly one matching record, or ``None`` for an empty selection."""

    mask = pd.Series(True, index=core.index)
    for column, value in criteria.items():
        if column not in core.columns:
            raise IntegrityError(f"unknown record selector: {column}")
        mask &= core[column].eq(value)

    matches = core[mask]
    if matches.empty:
        return None
    if len(matches) > 1:
        raise IntegrityError("record selector did not resolve to one row")
    return matches.iloc[0]


def find_scope_row(
    frame: pd.DataFrame,
    scope: str,
    *,
    scope_column: str = "scope",
) -> pd.Series | None:
    """Return one published scope row, or ``None`` when a scope is absent."""

    if scope_column not in frame.columns:
        raise IntegrityError(f"missing scope column: {scope_column}")
    matches = frame[frame[scope_column] == scope]
    if matches.empty:
        return None
    if len(matches) > 1:
        raise IntegrityError(f"duplicate published rows for scope: {scope}")
    return matches.iloc[0]


def require_scope_row(
    frame: pd.DataFrame,
    scope: str,
    *,
    scope_column: str = "scope",
) -> pd.Series:
    row = find_scope_row(frame, scope, scope_column=scope_column)
    if row is None:
        raise IntegrityError(f"required scope is absent: {scope}")
    return row


def composition_count(composition: pd.DataFrame, component: str) -> int:
    """Read one validated realised count from the composition table."""

    _require_columns(composition, contracts.M8_TABLES["dataset_composition.csv"], "composition")
    matches = composition[composition["component"] == component]
    if len(matches) != 1:
        raise IntegrityError(f"composition component is absent or duplicated: {component}")
    if not bool(matches.iloc[0]["matches"]):
        raise IntegrityError(
            f"composition component does not match its expected value: {component}"
        )
    return int(matches.iloc[0]["realised"])


def _cache_data(**kwargs: Any):
    """Use Streamlit's bounded cache when available, without requiring it for tests."""

    try:
        import streamlit as st
    except Exception:  # pragma: no cover - only exercised without Streamlit installed
        return lambda function: function
    return st.cache_data(**kwargs)


@_cache_data(max_entries=1, show_spinner=False)
def load_core() -> pd.DataFrame:
    return read_core()


@_cache_data(max_entries=16, show_spinner=False)
def load_report_table(kind: str, filename: str) -> pd.DataFrame:
    return read_report_table(kind, filename)


@_cache_data(max_entries=16, show_spinner=False)
def load_selected_image(sample_id: str, role: str) -> bytes:
    """Load one selected image by manifest record, never by a user path."""

    core = load_core()
    matches = core[core["sample_id"] == sample_id]
    if len(matches) != 1:
        raise IntegrityError("selected image record is absent or duplicated")
    verified = load_verified_image(matches.iloc[0], role)
    return verified.data
