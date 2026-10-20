"""Spatially localized, deterministic image features for the M6A study.

The frozen Milestone 3 feature set is deliberately global: it summarises a
whole image with eight scalars. M5B, M5C and M5D all found that a
geometric-distortion model reaches only a near-zero between-source R2 on the
held-out geometric slice, and M5D concluded that the *representation* may be
the binding constraint rather than the objective. This module adds the
localized information the global features cannot express, without touching the
existing pipeline.

The eight frozen features and their extraction code are unchanged. The spatial
features live here, in a separate module with their own versioned names, so
prior model inputs and artefacts cannot be silently altered.

Design (frozen in ``configs/spatial_feature_study_v0_1_0.toml``):

* A fixed ``grid_rows`` x ``grid_cols`` grid divides the canonical image into
  cells, visited in row-major order.
* ``ink_occupancy`` is the fraction of pixels at or below the Otsu threshold
  of the **whole image**, counted within the cell. The threshold is computed
  once per image, so the cell values share one ink definition and are directly
  comparable with the global ``ink_fraction`` feature.
* ``edge_density`` is the fraction of Canny-marked pixels within the cell,
  using the repository's existing edge convention (``sigma=1.0``,
  ``low_threshold=0.1``, ``high_threshold=0.2`` on the image scaled to
  ``[0, 1]``). The edge map is computed once per image, so the cell values
  share one edge definition and are directly comparable with the global
  ``edge_density`` feature.

Regional intensity statistics were considered and rejected: on training rows
the per-cell mean intensity is largely redundant with the per-cell ink
occupancy, and a 4x2 grid was chosen over 2x4 because it has lower internal
redundancy among the spatial features at a comparable correlation with the
target. Both decisions were made from training rows only.

Every feature is computed from the degraded image alone. The extractor never
sees the clean reference, the SSIM/PSNR targets, the source identifier, the
corruption family, the severity, the file name or the split label, so it
cannot leak target or metadata information into a predictor.

Degenerate inputs are handled explicitly: an exactly uniform image has no
distinguishable ink class and no edges, so all its spatial features are zero.
A cell that would be empty (possible only for a grid finer than the image)
also yields zero rather than a division by zero.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from captcha_quality.imaging import validate_image

#: Version of the spatial feature definition. Changing the grid, the
#: thresholding rule or the edge convention requires bumping this so old
#: feature tables remain distinguishable.
SPATIAL_FEATURE_SET_VERSION: Final[str] = "spatial-v1"

#: Feature family names.
FAMILY_INK: Final[str] = "ink_occupancy"
FAMILY_EDGE: Final[str] = "edge_density"

#: Default grid, matching the declared M6A configuration.
DEFAULT_GRID_ROWS: Final[int] = 4
DEFAULT_GRID_COLS: Final[int] = 2

#: Canny settings on the image scaled to [0, 1]. These match the frozen global
#: ``edge_density`` feature so the two are directly comparable.
EDGE_SIGMA: Final[float] = 1.0
EDGE_LOW_THRESHOLD: Final[float] = 0.1
EDGE_HIGH_THRESHOLD: Final[float] = 0.2


class SpatialFeatureError(ValueError):
    """Raised when a spatial feature cannot be computed from the input."""


class SpatialFeatureDependencyError(RuntimeError):
    """Raised when the optional scikit-image dependency is unavailable."""


@dataclass(frozen=True, slots=True)
class SpatialFeatureSpec:
    """The versioned spatial feature design.

    Parameters
    ----------
    grid_rows, grid_cols:
        Grid dimensions. The image is divided into ``grid_rows * grid_cols``
        cells, visited in row-major order.
    families:
        Which feature families to compute, in output order.
    edge_sigma, edge_low_threshold, edge_high_threshold:
        Canny settings on the image scaled to ``[0, 1]``.
    """

    grid_rows: int = DEFAULT_GRID_ROWS
    grid_cols: int = DEFAULT_GRID_COLS
    families: tuple[str, ...] = (FAMILY_INK, FAMILY_EDGE)
    edge_sigma: float = EDGE_SIGMA
    edge_low_threshold: float = EDGE_LOW_THRESHOLD
    edge_high_threshold: float = EDGE_HIGH_THRESHOLD

    def __post_init__(self) -> None:
        if self.grid_rows < 1 or self.grid_cols < 1:
            raise SpatialFeatureError("grid dimensions must be at least 1")
        if not self.families:
            raise SpatialFeatureError("at least one feature family is required")
        unknown = sorted(set(self.families) - {FAMILY_INK, FAMILY_EDGE})
        if unknown:
            raise SpatialFeatureError(
                f"unknown feature family/families {unknown}; known families are "
                f"{sorted({FAMILY_INK, FAMILY_EDGE})}"
            )
        if len(set(self.families)) != len(self.families):
            raise SpatialFeatureError("feature families must be duplicate-free")
        if self.edge_sigma <= 0.0:
            raise SpatialFeatureError("edge_sigma must be strictly positive")
        for label, value in (
            ("edge_low_threshold", self.edge_low_threshold),
            ("edge_high_threshold", self.edge_high_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise SpatialFeatureError(f"{label} must lie in [0, 1], got {value}")

    @property
    def n_cells(self) -> int:
        return self.grid_rows * self.grid_cols

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return the exact feature names in their fixed output order."""
        names: list[str] = []
        for family in self.families:
            for row in range(self.grid_rows):
                for col in range(self.grid_cols):
                    names.append(f"{family}_r{row}_c{col}")
        return tuple(names)

    def identity_payload(self) -> dict[str, object]:
        """Return the declared design, for hashing into the study protocol."""
        return {
            "feature_set_version": SPATIAL_FEATURE_SET_VERSION,
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
            "families": list(self.families),
            "edge_sigma": self.edge_sigma,
            "edge_low_threshold": self.edge_low_threshold,
            "edge_high_threshold": self.edge_high_threshold,
            "feature_names": list(self.feature_names),
        }


def _threshold_otsu(image: np.ndarray) -> float:
    try:
        from skimage.filters import threshold_otsu
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise SpatialFeatureDependencyError(
            "scikit-image is required for Otsu thresholding. Install the image "
            "extra, e.g. 'uv sync --extra dev --extra image'."
        ) from exc
    return float(threshold_otsu(image))


def _canny(image: np.ndarray, spec: SpatialFeatureSpec) -> np.ndarray:
    try:
        from skimage.feature import canny
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise SpatialFeatureDependencyError(
            "scikit-image is required for Canny edge detection. Install the image "
            "extra, e.g. 'uv sync --extra dev --extra image'."
        ) from exc
    return canny(
        image,
        sigma=spec.edge_sigma,
        low_threshold=spec.edge_low_threshold,
        high_threshold=spec.edge_high_threshold,
    )


def _cell_bounds(height: int, width: int, rows: int, cols: int) -> list[tuple[int, int, int, int]]:
    """Return the ``(y0, y1, x0, x1)`` bounds of every cell in row-major order."""
    bounds: list[tuple[int, int, int, int]] = []
    for row in range(rows):
        y0 = (row * height) // rows
        y1 = ((row + 1) * height) // rows
        for col in range(cols):
            x0 = (col * width) // cols
            x1 = ((col + 1) * width) // cols
            bounds.append((y0, y1, x0, x1))
    return bounds


def extract_spatial_features(
    image: np.ndarray, spec: SpatialFeatureSpec | None = None
) -> dict[str, float]:
    """Compute the documented spatial feature vector for one grayscale image.

    Parameters
    ----------
    image:
        2-D ``uint8`` array satisfying the shared image contract.
    spec:
        The feature design. Defaults to the declared M6A design.

    Returns
    -------
    dict
        Feature name to ``float``, in ``spec.feature_names`` order.

    Raises
    ------
    ImageContractError
        If ``image`` violates the shared image contract.
    SpatialFeatureDependencyError
        If scikit-image is not installed.
    SpatialFeatureError
        If a computed value is unexpectedly non-finite.
    """
    validate_image(image, name="spatial feature image")
    spec = spec or SpatialFeatureSpec()

    height, width = image.shape
    bounds = _cell_bounds(height, width, spec.grid_rows, spec.grid_cols)
    names = spec.feature_names
    values: dict[str, float] = {name: 0.0 for name in names}

    if image.min() != image.max():
        if FAMILY_INK in spec.families:
            threshold = _threshold_otsu(image)
            ink = image <= threshold
            for index, (y0, y1, x0, x1) in enumerate(bounds):
                cell = ink[y0:y1, x0:x1]
                values[f"{FAMILY_INK}_r{index // spec.grid_cols}_c{index % spec.grid_cols}"] = (
                    float(cell.mean()) if cell.size else 0.0
                )
        if FAMILY_EDGE in spec.families:
            edges = _canny(image.astype(np.float64) / 255.0, spec)
            for index, (y0, y1, x0, x1) in enumerate(bounds):
                cell = edges[y0:y1, x0:x1]
                values[f"{FAMILY_EDGE}_r{index // spec.grid_cols}_c{index % spec.grid_cols}"] = (
                    float(cell.mean()) if cell.size else 0.0
                )

    non_finite = [name for name, value in values.items() if not math.isfinite(value)]
    if non_finite:
        raise SpatialFeatureError(f"non-finite spatial feature(s) computed: {non_finite}")
    if tuple(values) != names:  # pragma: no cover - structural guard
        raise SpatialFeatureError(f"spatial feature order {tuple(values)} does not match {names}")
    return {name: float(values[name]) for name in names}


def extract_spatial_feature_table(
    samples: Iterable[tuple[str, np.ndarray]],
    spec: SpatialFeatureSpec | None = None,
) -> pd.DataFrame:
    """Extract spatial features for a batch of ``(sample_id, image)`` pairs.

    The returned frame has ``sample_id`` followed by the spatial feature names
    in their fixed order, so batching cannot change column order or
    definitions. Duplicate or empty sample identifiers are rejected.
    """
    spec = spec or SpatialFeatureSpec()
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for sample_id, image in samples:
        if not isinstance(sample_id, str) or not sample_id:
            raise SpatialFeatureError(f"sample_id must be a non-empty string, got {sample_id!r}")
        if sample_id in seen:
            raise SpatialFeatureError(f"duplicate sample_id in feature batch: {sample_id!r}")
        seen.add(sample_id)
        row: dict[str, object] = {"sample_id": sample_id}
        row.update(extract_spatial_features(image, spec))
        rows.append(row)
    return pd.DataFrame(rows, columns=("sample_id", *spec.feature_names))


def spatial_feature_manifest(spec: SpatialFeatureSpec | None = None) -> pd.DataFrame:
    """Return the declared feature manifest: name, family and grid cell.

    The manifest is a pure function of the spec, so it can be published without
    running the extractor and can be compared between runs.
    """
    spec = spec or SpatialFeatureSpec()
    rows: list[dict[str, object]] = []
    for family in spec.families:
        for row in range(spec.grid_rows):
            for col in range(spec.grid_cols):
                rows.append(
                    {
                        "feature_name": f"{family}_r{row}_c{col}",
                        "family": family,
                        "grid_row": row,
                        "grid_col": col,
                    }
                )
    return pd.DataFrame(rows)


def describe_feature_set(names: Sequence[str], global_names: Sequence[str]) -> dict[str, object]:
    """Describe a feature set: which global and spatial columns it contains."""
    global_set = set(global_names)
    spatial_set = set(names) - global_set
    unknown = sorted(spatial_set - set(SpatialFeatureSpec().feature_names))
    if unknown:
        raise SpatialFeatureError(f"unrecognised feature name(s): {unknown}")
    return {
        "n_features": len(names),
        "n_global": len(global_set & set(names)),
        "n_spatial": len(spatial_set),
        "global_names": [name for name in names if name in global_set],
        "spatial_names": [name for name in names if name in spatial_set],
    }
