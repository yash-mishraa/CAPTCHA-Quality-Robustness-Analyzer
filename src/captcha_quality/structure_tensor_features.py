"""Multi-scale structure-tensor features from the corrupted image alone.

M6A tested regional ink-occupancy and edge-density features on a fixed grid and
found no retention-aware gain. M6B varied only the grid resolution, found that
finer grids resolve an elastic warp better but still produced no demonstrable
predictive gain, and concluded that the blocker was feature relevance rather
than partition resolution. M6C tested whole-image summaries of local gradient
magnitude and orientation and found that they did not qualify under the
predeclared retention rule: they were strongly redundant with the global eight
and traded clean-reference retention for a small, inconsistent geometric
effect.

M6D tests a distinct question:

    Do compact multi-scale structure-tensor features computed from the corrupted
    image alone improve geometric-distortion SSIM prediction while preserving
    clean-reference and original-task performance?

A structure tensor summarizes the local arrangement of gradients, not merely
their average strength. At each predeclared smoothing scale it yields two
eigenvalues whose ratio describes directional coherence and whose sum describes
local tensor energy. The resulting representation is still compact and
interpretable, and it is computed from one image with a fixed, documented
procedure.

**No-reference boundary.** Every feature is a pure function of one 2-D
``uint8`` grayscale image. The extractor has no parameter for, and no code path
to, the clean reference, the SSIM/PSNR targets, the source identifier, the
corruption family or severity, the font, the split label or the file path. The
clean reference is used only by the separate evaluation pipeline that computes
the SSIM label; it never enters this module. Tests enforce that boundary
structurally (the public extractor accepts a single image and an optional
parameter object, and the module's code contains no forbidden identifier).

Feature set ``structure-tensor-v1`` (frozen in
``configs/structure_tensor_study_v0_1_0.toml``):

At each of two predeclared Gaussian scales, the image is scaled to ``[0, 1]``
and horizontal/vertical gradients are computed with scikit-image's 3x3 Sobel
operator and reflected boundaries. The tensor components ``Jxx = gx*gx``,
``Jyy = gy*gy`` and ``Jxy = gx*gy`` are smoothed with a Gaussian of the declared
scale. The two non-negative eigenvalues of the symmetric 2x2 tensor are

    lambda_1 = (trace + sqrt((Jxx - Jyy)^2 + 4*Jxy^2)) / 2
    lambda_2 = (trace - sqrt((Jxx - Jyy)^2 + 4*Jxy^2)) / 2

with ``trace = Jxx + Jyy`` and ``lambda_2`` clipped to zero for round-off. The
four summaries per scale are:

``coherence``
    ``(lambda_1 - lambda_2) / (lambda_1 + lambda_2 + epsilon)``, in ``[0, 1]``.
    It is 1 for a perfectly line-like local tensor and 0 for an isotropic one.
    Zero-energy pixels contribute 0.
``anisotropy``
    ``1 - lambda_2 / (lambda_1 + epsilon)``, in ``[0, 1]``. It is the fraction
    of the dominant eigenvalue not matched by the minor eigenvalue. Zero-energy
    pixels contribute 0.
``energy``
    ``lambda_1 + lambda_2``, the local tensor trace. It is non-negative and
    unbounded above; its unit is squared gradient on the ``[0, 1]`` image scale.

The four statistics per scale are the mean coherence, the 0.90 quantile of
coherence, the mean anisotropy and the 0.90 quantile of energy. Mean tensor
energy is reported once as ``st_energy_mean`` at the finest scale: Gaussian
smoothing preserves the image mean, so repeating it at both scales would add a
duplicate column rather than a multi-scale measurement. Nine features result
across the two scales. The quantile method is NumPy's default linear
interpolation. ``epsilon`` is a fixed, documented numerical tolerance, so a
zero-energy tensor is a defined zero rather than a division by zero.

Degenerate inputs are explicit. An exactly uniform image has zero gradient
everywhere, so every tensor component, eigenvalue, coherence, anisotropy and
energy is zero. An all-zero tensor is therefore a defined zero. Every output is
finite and deterministic, and the bounded summaries are clipped to ``[0, 1]``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from captcha_quality.imaging import validate_image

#: Version of the structure-tensor feature definition. Changing the gradient
#: operator, the scales, the smoothing, the eigenvalue convention or the
#: statistic set requires bumping this so old feature tables remain
#: distinguishable.
STRUCTURE_TENSOR_FEATURE_SET_VERSION: Final[str] = "structure-tensor-v1"

#: Feature family name used in the manifest.
FAMILY_STRUCTURE_TENSOR: Final[str] = "structure_tensor"

#: The two predeclared Gaussian smoothing scales, in pixels. They are frozen in
#: the versioned configuration; the defaults here are the same values.
DEFAULT_SCALES: Final[tuple[float, ...]] = (1.0, 3.0)

#: Upper quantile reported for local coherence.
DEFAULT_COHERENCE_QUANTILE: Final[float] = 0.90

#: Upper quantile reported for local tensor energy.
DEFAULT_ENERGY_QUANTILE: Final[float] = 0.90

#: Numerical tolerance used in the coherence and anisotropy denominators.
DEFAULT_EPSILON: Final[float] = 1e-12

#: Gaussian truncation in standard deviations. Fixed so the smoothing footprint
#: is deterministic and documented.
DEFAULT_GAUSSIAN_TRUNCATE: Final[float] = 4.0

#: The four statistic suffixes reported at each scale, in fixed output order.
#: Mean energy is deliberately not repeated per scale: Gaussian smoothing
#: preserves the image mean, so a per-scale mean-energy column would be a
#: duplicate. It is reported once as ``st_energy_mean`` at the finest scale.
SCALE_STATISTICS: Final[tuple[str, ...]] = (
    "coh_mean",
    "coh_p90",
    "aniso_mean",
    "energy_p90",
)

#: The single global mean-energy feature name.
GLOBAL_ENERGY_MEAN: Final[str] = "st_energy_mean"


class StructureTensorFeatureError(ValueError):
    """Raised when a structure-tensor feature cannot be computed."""


class StructureTensorDependencyError(RuntimeError):
    """Raised when the optional scikit-image dependency is unavailable."""


@dataclass(frozen=True, slots=True)
class StructureTensorSpec:
    """The versioned multi-scale structure-tensor feature design.

    Parameters
    ----------
    scales:
        Gaussian smoothing scales in pixels, in output order. The study
        configuration requires exactly two, meaningfully different scales.
    coherence_quantile:
        Upper quantile reported for local coherence.
    energy_quantile:
        Upper quantile reported for local tensor energy.
    epsilon:
        Positive numerical tolerance in the coherence and anisotropy
        denominators. It makes a zero-energy tensor a defined zero.
    gaussian_truncate:
        Gaussian truncation in standard deviations.
    gradient_operator:
        Frozen gradient operator identifier. Only ``sobel_3x3`` is supported.
    smoothing_kernel:
        Frozen tensor smoothing kernel identifier. Only ``gaussian`` is
        supported.
    boundary:
        Frozen boundary mode for both operators. Only ``reflect`` is supported.
    """

    scales: tuple[float, ...] = DEFAULT_SCALES
    coherence_quantile: float = DEFAULT_COHERENCE_QUANTILE
    energy_quantile: float = DEFAULT_ENERGY_QUANTILE
    epsilon: float = DEFAULT_EPSILON
    gaussian_truncate: float = DEFAULT_GAUSSIAN_TRUNCATE
    gradient_operator: str = "sobel_3x3"
    smoothing_kernel: str = "gaussian"
    boundary: str = "reflect"

    def __post_init__(self) -> None:
        if not self.scales:
            raise StructureTensorFeatureError("at least one smoothing scale is required")
        if any(not math.isfinite(scale) or scale <= 0.0 for scale in self.scales):
            raise StructureTensorFeatureError("every smoothing scale must be finite and positive")
        if len(set(self.scales)) != len(self.scales):
            raise StructureTensorFeatureError("smoothing scales must be duplicate-free")
        if not 0.0 < self.coherence_quantile < 1.0:
            raise StructureTensorFeatureError("coherence_quantile must lie strictly inside (0, 1)")
        if not 0.0 < self.energy_quantile < 1.0:
            raise StructureTensorFeatureError("energy_quantile must lie strictly inside (0, 1)")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise StructureTensorFeatureError("epsilon must be finite and positive")
        if not math.isfinite(self.gaussian_truncate) or self.gaussian_truncate <= 0.0:
            raise StructureTensorFeatureError("gaussian_truncate must be finite and positive")
        if self.gradient_operator != "sobel_3x3":
            raise StructureTensorFeatureError(
                "only the frozen 'sobel_3x3' gradient operator is supported"
            )
        if self.smoothing_kernel != "gaussian":
            raise StructureTensorFeatureError("only the frozen 'gaussian' smoothing is supported")
        if self.boundary != "reflect":
            raise StructureTensorFeatureError("only the frozen 'reflect' boundary is supported")

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return the exact feature names in their fixed output order."""
        names = [
            f"st_s{index}_{statistic}"
            for index in range(1, len(self.scales) + 1)
            for statistic in SCALE_STATISTICS
        ]
        names.append(GLOBAL_ENERGY_MEAN)
        return tuple(names)

    def identity_payload(self) -> dict[str, object]:
        """Return the declared design, for hashing into the study protocol."""
        return {
            "feature_set_version": STRUCTURE_TENSOR_FEATURE_SET_VERSION,
            "scales": list(self.scales),
            "coherence_quantile": self.coherence_quantile,
            "energy_quantile": self.energy_quantile,
            "epsilon": self.epsilon,
            "gaussian_truncate": self.gaussian_truncate,
            "gradient_operator": self.gradient_operator,
            "gradient_implementation": "skimage.filters.sobel_h/sobel_v, 3x3",
            "gradient_input": "uint8 image scaled to [0, 1]",
            "smoothing_kernel": self.smoothing_kernel,
            "smoothing_implementation": "skimage.filters.gaussian",
            "boundary": self.boundary,
            "eigenvalue_convention": "lambda_1 >= lambda_2 >= 0 of the symmetric 2x2 tensor",
            "coherence": "(lambda_1 - lambda_2) / (lambda_1 + lambda_2 + epsilon)",
            "anisotropy": "1 - lambda_2 / (lambda_1 + epsilon)",
            "energy": "lambda_1 + lambda_2",
            "energy_mean_policy": (
                "reported once as st_energy_mean at the finest scale because Gaussian "
                "smoothing preserves the mean"
            ),
            "quantile_method": "numpy.quantile linear",
            "feature_names": list(self.feature_names),
        }


def _gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(gx, gy)`` from the fixed 3x3 Sobel operator."""
    try:
        from skimage.filters import sobel_h, sobel_v
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise StructureTensorDependencyError(
            "scikit-image is required for Sobel gradients. Install the image "
            "extra, e.g. 'uv sync --extra dev --extra image'."
        ) from exc
    values = image.astype(np.float64) / 255.0
    # sobel_v is the derivative along x (vertical edges); sobel_h is the
    # derivative along y (horizontal edges). This is the conventional
    # (gx, gy) order used by the tensor definition.
    return sobel_v(values), sobel_h(values)


def _smooth(values: np.ndarray, spec: StructureTensorSpec, sigma: float) -> np.ndarray:
    """Apply the fixed Gaussian smoothing at one declared scale."""
    try:
        from skimage.filters import gaussian
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise StructureTensorDependencyError(
            "scikit-image is required for Gaussian smoothing. Install the image "
            "extra, e.g. 'uv sync --extra dev --extra image'."
        ) from exc
    return np.asarray(
        gaussian(
            values,
            sigma=sigma,
            mode="reflect",
            truncate=spec.gaussian_truncate,
            preserve_range=True,
        ),
        dtype=np.float64,
    )


def _tensor_summaries(
    gx: np.ndarray, gy: np.ndarray, spec: StructureTensorSpec, sigma: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(energy, coherence, anisotropy)`` arrays at one scale."""
    jxx = _smooth(gx * gx, spec, sigma)
    jyy = _smooth(gy * gy, spec, sigma)
    jxy = _smooth(gx * gy, spec, sigma)

    trace = jxx + jyy
    delta = jxx - jyy
    root = np.sqrt(delta * delta + 4.0 * jxy * jxy)
    lambda_1 = 0.5 * (trace + root)
    lambda_2 = 0.5 * (trace - root)

    # Round-off can make the smaller eigenvalue slightly negative or slightly
    # larger than the larger one. Both are clamped so every derived quantity is
    # well defined and bounded.
    lambda_1 = np.maximum(lambda_1, 0.0)
    lambda_2 = np.maximum(lambda_2, 0.0)
    lambda_2 = np.minimum(lambda_2, lambda_1)
    energy = np.maximum(lambda_1 + lambda_2, 0.0)

    coherence = (lambda_1 - lambda_2) / (energy + spec.epsilon)
    coherence = np.clip(coherence, 0.0, 1.0)

    # The denominator uses lambda_1 rather than energy so anisotropy is a
    # distinct normalization from coherence. A zero-energy pixel is explicitly
    # zero, not 0/0.
    anisotropy = np.where(
        lambda_1 > spec.epsilon,
        1.0 - lambda_2 / (lambda_1 + spec.epsilon),
        0.0,
    )
    anisotropy = np.clip(anisotropy, 0.0, 1.0)
    return energy, coherence, anisotropy


def _scale_features(
    gx: np.ndarray,
    gy: np.ndarray,
    spec: StructureTensorSpec,
    *,
    index: int,
    sigma: float,
) -> tuple[dict[str, float], np.ndarray]:
    """Compute the four declared per-scale summaries and return the energy map."""
    energy, coherence, anisotropy = _tensor_summaries(gx, gy, spec, sigma)
    prefix = f"st_s{index}"
    features = {
        f"{prefix}_coh_mean": float(np.mean(coherence)),
        f"{prefix}_coh_p90": float(np.quantile(coherence, spec.coherence_quantile)),
        f"{prefix}_aniso_mean": float(np.mean(anisotropy)),
        f"{prefix}_energy_p90": float(np.quantile(energy, spec.energy_quantile)),
    }
    return features, energy


def extract_structure_tensor_features(
    image: np.ndarray, spec: StructureTensorSpec | None = None
) -> dict[str, float]:
    """Compute the multi-scale structure-tensor feature vector for one image.

    Parameters
    ----------
    image:
        2-D ``uint8`` array satisfying the shared image contract. This is the
        *corrupted* image the model would receive. No clean reference, target,
        identifier or metadata value is accepted.
    spec:
        The feature design. Defaults to the declared M6D design.

    Returns
    -------
    dict
        Feature name to ``float``, in ``spec.feature_names`` order.

    Raises
    ------
    ImageContractError
        If ``image`` violates the shared image contract.
    StructureTensorDependencyError
        If scikit-image is not installed.
    StructureTensorFeatureError
        If a computed value is unexpectedly non-finite.
    """
    validate_image(image, name="structure tensor image")
    if spec is not None and not isinstance(spec, StructureTensorSpec):
        raise TypeError(
            "spec must be a StructureTensorSpec or None; the extractor accepts no "
            "second image, reference or metadata object"
        )
    spec = spec or StructureTensorSpec()

    gx, gy = _gradients(image)
    values: dict[str, float] = {}
    finest_energy: np.ndarray | None = None
    for index, sigma in enumerate(spec.scales, start=1):
        features, energy = _scale_features(gx, gy, spec, index=index, sigma=sigma)
        values.update(features)
        if index == 1:
            finest_energy = energy
    # Gaussian smoothing preserves the mean, so mean tensor energy is a single
    # global summary appended once rather than a per-scale duplicate.
    if finest_energy is None:  # pragma: no cover - spec requires a scale
        raise StructureTensorFeatureError("no smoothing scale produced an energy map")
    values[GLOBAL_ENERGY_MEAN] = float(np.mean(finest_energy))

    non_finite = [
        name for name in spec.feature_names if not math.isfinite(values.get(name, math.nan))
    ]
    if non_finite:
        raise StructureTensorFeatureError(
            f"non-finite structure-tensor feature(s) computed: {non_finite}"
        )
    if tuple(values) != spec.feature_names:  # pragma: no cover - structural guard
        raise StructureTensorFeatureError(
            f"structure-tensor feature order {tuple(values)} does not match {spec.feature_names}"
        )
    return {name: float(values[name]) for name in spec.feature_names}


def extract_structure_tensor_feature_table(
    samples: Iterable[tuple[str, np.ndarray]],
    spec: StructureTensorSpec | None = None,
) -> pd.DataFrame:
    """Extract structure-tensor features for ``(sample_id, image)`` pairs.

    ``sample_id`` is an opaque join key. It is never read by the feature
    computation, which receives only the image array. The returned frame has
    ``sample_id`` followed by the declared feature names in their fixed order,
    so batching cannot change column order or definitions. Duplicate or empty
    sample identifiers are rejected.
    """
    spec = spec or StructureTensorSpec()
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for sample_id, image in samples:
        if not isinstance(sample_id, str) or not sample_id:
            raise StructureTensorFeatureError(
                f"sample_id must be a non-empty string, got {sample_id!r}"
            )
        if sample_id in seen:
            raise StructureTensorFeatureError(
                f"duplicate sample_id in feature batch: {sample_id!r}"
            )
        seen.add(sample_id)
        row: dict[str, object] = {"sample_id": sample_id}
        row.update(extract_structure_tensor_features(image, spec))
        rows.append(row)
    return pd.DataFrame(rows, columns=("sample_id", *spec.feature_names))


def structure_tensor_manifest(spec: StructureTensorSpec | None = None) -> pd.DataFrame:
    """Return the declared manifest: name, family, scale, statistic and quantile.

    The manifest is a pure function of the spec, so it can be published without
    running the extractor and compared between runs.
    """
    spec = spec or StructureTensorSpec()
    rows: list[dict[str, object]] = []
    for index, sigma in enumerate(spec.scales, start=1):
        for statistic in SCALE_STATISTICS:
            if statistic == "coh_p90":
                quantile = spec.coherence_quantile
            elif statistic == "energy_p90":
                quantile = spec.energy_quantile
            else:
                quantile = math.nan
            rows.append(
                {
                    "feature_name": f"st_s{index}_{statistic}",
                    "family": FAMILY_STRUCTURE_TENSOR,
                    "scale_index": index,
                    "sigma_px": float(sigma),
                    "statistic": statistic,
                    "quantile": quantile,
                }
            )
    rows.append(
        {
            "feature_name": GLOBAL_ENERGY_MEAN,
            "family": FAMILY_STRUCTURE_TENSOR,
            "scale_index": 1,
            "sigma_px": float(spec.scales[0]),
            "statistic": "energy_mean",
            "quantile": math.nan,
        }
    )
    return pd.DataFrame(rows)
