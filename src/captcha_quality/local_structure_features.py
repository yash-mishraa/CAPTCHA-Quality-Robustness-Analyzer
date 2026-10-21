"""Local structural image features computed from the corrupted image alone.

M6A added regional ink occupancy and edge density on a fixed 4x2 grid and found
they did not improve retention-aware geometric-distortion prediction. M6B varied
only the grid resolution, found that finer grids *do* resolve an elastic warp
better but still produced no demonstrable predictive gain, and concluded that
the blocker is the relevance of regional ink summaries to the other corruption
families rather than the resolution of the partition.

M6C tests a different hypothesis:

    Do local structural features computed from a corrupted image alone improve
    geometric-distortion SSIM prediction beyond the existing global features,
    while preserving clean-reference and original-task performance?

The feature families here describe *local edge and stroke structure* rather than
regional ink quantity. They are deliberately few, interpretable and cheap, and
they are computed with the repository's existing grayscale representation and
deterministic processing conventions.

**No-reference boundary.** Every feature is a pure function of one 2-D
``uint8`` grayscale image. The extractor has no parameter for, and no code
path to, the clean reference, the SSIM/PSNR targets, the source identifier, the
corruption family or severity, the font, the split label or the file path. The
clean reference is used only by the separate evaluation pipeline that computes
the SSIM label; it never enters this module. Tests enforce that boundary
structurally (the public extractor accepts a single image and an optional
parameter object, and the module's code contains no forbidden identifier).

Feature set ``local-structure-v1`` (frozen in
``configs/local_structure_study_v0_1_0.toml``):

``gradient_magnitude`` family (4 features)
    Sobel gradients are computed on the image scaled to ``[0, 1]`` with
    scikit-image's normalised 3x3 Sobel operator and reflected boundaries. The
    gradient magnitude is ``hypot(gx, gy)``. The family is summarised by the
    mean, the population standard deviation, the median and a fixed upper
    quantile (0.9 by default). These are whole-image summaries of a *local*
    operator: they describe how strong the local structure is, not only how
    much ink is present.

``gradient_orientation`` family (bins + 2 features)
    The local gradient orientation is ``arctan2(gy, gx)`` folded into
    ``[0, pi)``, because an edge orientation is undirected. Pixels whose
    gradient magnitude is below a fixed weak-gradient threshold are excluded;
    this convention is explicit and configured rather than implicit. The
    surviving orientations are binned into a fixed number of equal-width bins
    over ``[0, pi)`` and the counts are normalised to sum to one. Two scalar
    summaries accompany the histogram: the circular resultant length of the
    doubled angles (a coherence in ``[0, 1]``, where 1 means every strong edge
    shares one orientation) and the normalised Shannon entropy of the
    histogram (0 means one occupied bin, 1 means a uniform orientation
    distribution).

Degenerate inputs are handled explicitly. An exactly uniform image has zero
gradient everywhere, so all magnitude statistics are zero and, because no pixel
clears the weak-gradient threshold, the orientation histogram, coherence and
entropy are all zero. An empty strong-gradient set is therefore a defined value,
not a division by zero. Every output is finite and deterministic.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from captcha_quality.imaging import validate_image

#: Version of the local structural feature definition. Changing the operator,
#: the statistic set, the orientation convention or the bin count requires
#: bumping this so old feature tables remain distinguishable.
LOCAL_STRUCTURE_FEATURE_SET_VERSION: Final[str] = "local-structure-v1"

#: Feature family names.
FAMILY_MAGNITUDE: Final[str] = "gradient_magnitude"
FAMILY_ORIENTATION: Final[str] = "gradient_orientation"

#: Number of equal-width orientation bins over ``[0, pi)``.
DEFAULT_ORIENT_BINS: Final[int] = 8

#: Weak-gradient cut-off on the ``[0, 1]`` scale. Pixels below it do not carry
#: a reliable orientation and are excluded from the orientation family.
DEFAULT_ORIENT_MIN_MAGNITUDE: Final[float] = 0.10

#: Upper quantile reported by the magnitude family.
DEFAULT_MAGNITUDE_QUANTILE: Final[float] = 0.90

#: Magnitude statistic names in their fixed output order.
MAGNITUDE_STATISTICS: Final[tuple[str, ...]] = ("mean", "std", "median", "p90")

#: Scalar orientation summary names in their fixed output order.
ORIENTATION_SUMMARIES: Final[tuple[str, ...]] = ("coherence", "entropy")


class LocalStructureFeatureError(ValueError):
    """Raised when a local structural feature cannot be computed."""


class LocalStructureDependencyError(RuntimeError):
    """Raised when the optional scikit-image dependency is unavailable."""


@dataclass(frozen=True, slots=True)
class LocalStructureSpec:
    """The versioned local structural feature design.

    Parameters
    ----------
    families:
        Which feature families to compute, in output order.
    orient_bins:
        Number of equal-width orientation bins over ``[0, pi)``.
    orient_min_magnitude:
        Weak-gradient cut-off on the ``[0, 1]`` scale; pixels below it are
        excluded from the orientation family.
    magnitude_quantile:
        Upper quantile reported by the magnitude family.
    """

    families: tuple[str, ...] = (FAMILY_MAGNITUDE, FAMILY_ORIENTATION)
    orient_bins: int = DEFAULT_ORIENT_BINS
    orient_min_magnitude: float = DEFAULT_ORIENT_MIN_MAGNITUDE
    magnitude_quantile: float = DEFAULT_MAGNITUDE_QUANTILE

    def __post_init__(self) -> None:
        if not self.families:
            raise LocalStructureFeatureError("at least one feature family is required")
        unknown = sorted(set(self.families) - {FAMILY_MAGNITUDE, FAMILY_ORIENTATION})
        if unknown:
            raise LocalStructureFeatureError(
                f"unknown feature family/families {unknown}; known families are "
                f"{sorted({FAMILY_MAGNITUDE, FAMILY_ORIENTATION})}"
            )
        if len(set(self.families)) != len(self.families):
            raise LocalStructureFeatureError("feature families must be duplicate-free")
        if self.orient_bins < 2:
            raise LocalStructureFeatureError(
                "orient_bins must be at least 2 so orientation entropy is defined"
            )
        if not math.isfinite(self.orient_min_magnitude) or self.orient_min_magnitude < 0.0:
            raise LocalStructureFeatureError("orient_min_magnitude must be finite and non-negative")
        if not 0.0 < self.magnitude_quantile < 1.0:
            raise LocalStructureFeatureError("magnitude_quantile must lie strictly inside (0, 1)")

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return the exact feature names in their fixed output order."""
        names: list[str] = []
        for family in self.families:
            if family == FAMILY_MAGNITUDE:
                names.extend(f"grad_mag_{statistic}" for statistic in MAGNITUDE_STATISTICS)
            else:
                names.extend(f"orient_hist_{index:02d}" for index in range(self.orient_bins))
                names.extend(f"orient_{summary}" for summary in ORIENTATION_SUMMARIES)
        return tuple(names)

    def identity_payload(self) -> dict[str, object]:
        """Return the declared design, for hashing into the study protocol."""
        return {
            "feature_set_version": LOCAL_STRUCTURE_FEATURE_SET_VERSION,
            "families": list(self.families),
            "orient_bins": self.orient_bins,
            "orient_min_magnitude": self.orient_min_magnitude,
            "magnitude_quantile": self.magnitude_quantile,
            "feature_names": list(self.feature_names),
        }


def _sobel_gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from skimage.filters import sobel_h, sobel_v
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise LocalStructureDependencyError(
            "scikit-image is required for Sobel gradients. Install the image "
            "extra, e.g. 'uv sync --extra dev --extra image'."
        ) from exc
    values = image.astype(np.float64) / 255.0
    # sobel_v is the derivative along x (it responds to vertical edges) and
    # sobel_h is the derivative along y (it responds to horizontal edges).
    # Keep the conventional (gx, gy) order so arctan2(gy, gx) has the usual
    # meaning: a vertical edge has orientation 0 and a horizontal edge pi/2.
    return sobel_v(values), sobel_h(values)


def _magnitude_features(magnitude: np.ndarray, quantile: float) -> dict[str, float]:
    return {
        "grad_mag_mean": float(np.mean(magnitude)),
        "grad_mag_std": float(np.std(magnitude, ddof=0)),
        "grad_mag_median": float(np.median(magnitude)),
        "grad_mag_p90": float(np.quantile(magnitude, quantile)),
    }


def _orientation_features(
    gx: np.ndarray, gy: np.ndarray, magnitude: np.ndarray, spec: LocalStructureSpec
) -> dict[str, float]:
    bins = spec.orient_bins
    names = [f"orient_hist_{index:02d}" for index in range(bins)]
    values: dict[str, float] = {name: 0.0 for name in names}
    values["orient_coherence"] = 0.0
    values["orient_entropy"] = 0.0

    strong = magnitude >= spec.orient_min_magnitude
    count = int(np.count_nonzero(strong))
    if count == 0:
        # No reliable orientation: every summary is explicitly zero rather
        # than an undefined ratio.
        return values

    # Fold the undirected orientation into [0, pi).
    theta = np.mod(np.arctan2(gy[strong], gx[strong]), np.pi)
    counts, _ = np.histogram(theta, bins=bins, range=(0.0, np.pi))
    probabilities = counts.astype(np.float64) / float(count)
    for index, probability in enumerate(probabilities):
        values[f"orient_hist_{index:02d}"] = float(probability)

    nonzero = probabilities[probabilities > 0.0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)) / math.log(bins))
    # Normalise negative zero so serialised tables are unambiguous.
    values["orient_entropy"] = 0.0 if entropy == 0.0 else entropy

    doubled = 2.0 * theta
    coherence = float(np.hypot(np.mean(np.cos(doubled)), np.mean(np.sin(doubled))))
    values["orient_coherence"] = coherence
    return values


def extract_local_structure_features(
    image: np.ndarray, spec: LocalStructureSpec | None = None
) -> dict[str, float]:
    """Compute the local structural feature vector for one grayscale image.

    Parameters
    ----------
    image:
        2-D ``uint8`` array satisfying the shared image contract. This is the
        *corrupted* image the model would receive. No clean reference, target,
        identifier or metadata value is accepted.
    spec:
        The feature design. Defaults to the declared M6C design.

    Returns
    -------
    dict
        Feature name to ``float``, in ``spec.feature_names`` order.

    Raises
    ------
    ImageContractError
        If ``image`` violates the shared image contract.
    LocalStructureDependencyError
        If scikit-image is not installed.
    LocalStructureFeatureError
        If a computed value is unexpectedly non-finite.
    """
    validate_image(image, name="local structure image")
    if spec is not None and not isinstance(spec, LocalStructureSpec):
        raise TypeError(
            "spec must be a LocalStructureSpec or None; the extractor accepts no "
            "second image, reference or metadata object"
        )
    spec = spec or LocalStructureSpec()

    gx, gy = _sobel_gradients(image)
    magnitude = np.hypot(gx, gy)

    values: dict[str, float] = {}
    for family in spec.families:
        if family == FAMILY_MAGNITUDE:
            values.update(_magnitude_features(magnitude, spec.magnitude_quantile))
        else:
            values.update(_orientation_features(gx, gy, magnitude, spec))

    non_finite = [
        name for name in spec.feature_names if not math.isfinite(values.get(name, math.nan))
    ]
    if non_finite:
        raise LocalStructureFeatureError(
            f"non-finite local structural feature(s) computed: {non_finite}"
        )
    if tuple(values) != spec.feature_names:  # pragma: no cover - structural guard
        raise LocalStructureFeatureError(
            f"local structural feature order {tuple(values)} does not match {spec.feature_names}"
        )
    return {name: float(values[name]) for name in spec.feature_names}


def extract_local_structure_feature_table(
    samples: Iterable[tuple[str, np.ndarray]],
    spec: LocalStructureSpec | None = None,
) -> pd.DataFrame:
    """Extract local structural features for ``(sample_id, image)`` pairs.

    ``sample_id`` is an opaque join key. It is never read by the feature
    computation, which receives only the image array. The returned frame has
    ``sample_id`` followed by the declared feature names in their fixed order,
    so batching cannot change column order or definitions. Duplicate or empty
    sample identifiers are rejected.
    """
    spec = spec or LocalStructureSpec()
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for sample_id, image in samples:
        if not isinstance(sample_id, str) or not sample_id:
            raise LocalStructureFeatureError(
                f"sample_id must be a non-empty string, got {sample_id!r}"
            )
        if sample_id in seen:
            raise LocalStructureFeatureError(f"duplicate sample_id in feature batch: {sample_id!r}")
        seen.add(sample_id)
        row: dict[str, object] = {"sample_id": sample_id}
        row.update(extract_local_structure_features(image, spec))
        rows.append(row)
    return pd.DataFrame(rows, columns=("sample_id", *spec.feature_names))


def local_structure_manifest(spec: LocalStructureSpec | None = None) -> pd.DataFrame:
    """Return the declared manifest: name, family, statistic and orientation bin.

    The manifest is a pure function of the spec, so it can be published without
    running the extractor and compared between runs.
    """
    spec = spec or LocalStructureSpec()
    rows: list[dict[str, object]] = []
    for family in spec.families:
        if family == FAMILY_MAGNITUDE:
            for statistic in MAGNITUDE_STATISTICS:
                rows.append(
                    {
                        "feature_name": f"grad_mag_{statistic}",
                        "family": family,
                        "statistic": statistic,
                        "bin_index": pd.NA,
                        "bin_lower_rad": math.nan,
                        "bin_upper_rad": math.nan,
                    }
                )
        else:
            width = math.pi / spec.orient_bins
            for index in range(spec.orient_bins):
                rows.append(
                    {
                        "feature_name": f"orient_hist_{index:02d}",
                        "family": family,
                        "statistic": "histogram",
                        "bin_index": index,
                        "bin_lower_rad": index * width,
                        "bin_upper_rad": (index + 1) * width,
                    }
                )
            for summary in ORIENTATION_SUMMARIES:
                rows.append(
                    {
                        "feature_name": f"orient_{summary}",
                        "family": family,
                        "statistic": summary,
                        "bin_index": pd.NA,
                        "bin_lower_rad": math.nan,
                        "bin_upper_rad": math.nan,
                    }
                )
    return pd.DataFrame(rows)
