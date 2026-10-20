"""Reproducible synthetic dataset generation, verification and comparison.

The pipeline is deliberately closed-loop:

1. plan clean sources (deterministic text, length, font, seed, stable IDs),
2. assign grouped splits *before* generating pixels,
3. render clean references and every configured corruption variant,
4. compute SSIM/PSNR targets from the exact reference/variant pair,
5. extract handcrafted features from each degraded image,
6. write a public manifest plus internal ground truth and a summary.

Outputs go to a staging directory and are renamed into place only on success,
so an interrupted run cannot look like a complete dataset. Manifests and
summaries contain no timestamps, which makes repeated runs byte-comparable.
"""

from __future__ import annotations

import json
import math
import shutil
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from PIL import Image

from captcha_quality.config import FEATURE_SET_VERSION, DatasetConfig
from captcha_quality.corruptions import (
    FAMILY_BLUR,
    FAMILY_NOISE,
    CorruptionSpec,
    apply_corruption,
    derive_noise_seed,
)
from captcha_quality.features import FEATURE_NAMES, extract_feature_table
from captcha_quality.imaging import validate_image
from captcha_quality.provenance import (
    canonical_json,
    code_revision,
    package_version,
    sha256_bytes,
    sha256_file,
)
from captcha_quality.quality_metrics import compute_pair_metrics
from captcha_quality.render import (
    CleanRender,
    FontResolution,
    FontResolutionError,
    ResolvedFont,
    render_clean,
    resolve_fonts,
    sample_text,
)
from captcha_quality.splits import SplitError, assign_grouped_splits

#: Public manifest columns, in stable order. Raw text is deliberately absent.
MANIFEST_COLUMNS: Final[tuple[str, ...]] = (
    "sample_id",
    "source_id",
    "variant_id",
    "split_group_id",
    "split",
    "text_length",
    "font_id",
    "render_seed",
    "variant_seed",
    "corruption_type",
    "severity_level",
    "corruption_parameters",
    "image_path",
    "reference_path",
    "ssim_target",
    "psnr_target",
    "psnr_infinite",
    "height",
    "width",
    "dataset_version",
    "generation_config_hash",
    "text_sha256",
    "image_sha256",
    "reference_sha256",
)

#: Source-level table used for split and provenance checks.
SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "source_id",
    "split_group_id",
    "split",
    "font_id",
    "text_length",
    "render_seed",
    "background_level",
    "height",
    "width",
    "clean_image_path",
    "clean_image_sha256",
    "text_sha256",
    "generation_config_hash",
    "dataset_version",
)

#: Internal-only ground truth, separated from the public manifest.
GROUND_TRUTH_COLUMNS: Final[tuple[str, ...]] = ("source_id", "text")

CLEAN_VARIANT_ID: Final[str] = "clean_s0"
CLEAN_CORRUPTION_TYPE: Final[str] = "none"

#: Manifest parameter name for each supported family.
_PARAMETER_KEYS: Final[dict[str, str]] = {
    FAMILY_BLUR: "radius_px",
    FAMILY_NOISE: "sigma",
}


class DatasetError(RuntimeError):
    """Raised when a dataset cannot be generated or fails verification."""


class DatasetExistsError(DatasetError):
    """Raised when the target directory exists and overwriting was not requested."""


@dataclass(frozen=True, slots=True)
class SourcePlan:
    """A planned clean source, including its pre-assigned split."""

    index: int
    font: ResolvedFont
    text: str
    text_length: int
    render_seed: int
    source_id: str
    split_group_id: str
    split: str

    @property
    def text_sha256(self) -> str:
        """Hash of the raw text, stored publicly instead of the text itself."""
        return sha256_bytes(self.text.encode("utf-8"))


@dataclass(frozen=True, slots=True)
class DatasetResult:
    """In-memory view of a generated dataset."""

    root: Path
    config: DatasetConfig
    manifest: pd.DataFrame
    sources: pd.DataFrame
    features: pd.DataFrame
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One named verification check."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """Outcome of verifying or comparing a dataset."""

    dataset: str
    checks: tuple[CheckResult, ...]

    @property
    def ok(self) -> bool:
        """True when every check passed."""
        return all(check.passed for check in self.checks)

    def failures(self) -> tuple[CheckResult, ...]:
        """Return only the failed checks."""
        return tuple(check for check in self.checks if not check.passed)


def _stable_id(prefix: str, payload: Mapping[str, object]) -> str:
    digest = sha256_bytes(canonical_json(payload).encode("utf-8"))
    return f"{prefix}-{digest[:16]}"


def _variant_id(family: str, severity: int) -> str:
    if family == CLEAN_CORRUPTION_TYPE:
        return CLEAN_VARIANT_ID
    return f"{family}_s{severity}"


def _corruption_parameters(spec: CorruptionSpec) -> dict[str, float]:
    if spec.is_identity:
        return {}
    return {_PARAMETER_KEYS[spec.family]: spec.parameter}


def _relative(path: Path) -> str:
    return path.as_posix()


def plan_sources(config: DatasetConfig, resolution: FontResolution) -> tuple[SourcePlan, ...]:
    """Plan clean sources and assign grouped splits deterministically.

    Text length is drawn from the configured distribution, then text from the
    charset, using a generator seeded only by dataset seed, font and source
    index. Rendering uses a separate seed so changing jitter does not change the
    text, and vice versa.
    """
    groups: dict[str, str] = {}
    staged: list[tuple[int, ResolvedFont, str, int, int, str]] = []
    for font in resolution.available:
        for index in range(config.dataset.sources_per_font):
            text_rng = np.random.default_rng(
                derive_noise_seed(config.dataset.seed, "source", font.key, index)
            )
            weights = np.asarray(config.dataset.text_length_weights, dtype=np.float64)
            length = int(
                text_rng.choice(
                    np.asarray(config.dataset.text_lengths, dtype=np.int64),
                    p=weights / weights.sum(),
                )
            )
            render_config = config.render.to_render_config(
                text_length=length, charset=config.dataset.charset
            )
            text = sample_text(text_rng, render_config)
            render_seed = derive_noise_seed(config.dataset.seed, "render", font.key, index)
            source_id = _stable_id(
                "src",
                {
                    "generation_config_hash": config.generation_config_hash,
                    "font_id": font.key,
                    "index": index,
                    "text": text,
                    "text_length": length,
                    "render_seed": render_seed,
                },
            )
            if source_id in groups:
                raise DatasetError(
                    f"source_id collision for font={font.key!r}, index={index}; "
                    "this indicates a broken identifier scheme"
                )
            groups[source_id] = font.key
            staged.append((index, font, text, length, render_seed, source_id))

    try:
        assignment = assign_grouped_splits(groups, config.splits)
    except SplitError as exc:
        raise DatasetError(f"split assignment failed: {exc}") from exc

    return tuple(
        SourcePlan(
            index=index,
            font=font,
            text=text,
            text_length=length,
            render_seed=render_seed,
            source_id=source_id,
            split_group_id=source_id,
            split=assignment.assignments[source_id],
        )
        for index, font, text, length, render_seed, source_id in staged
    )


def _resolve_all_fonts(config: DatasetConfig):
    """Resolve every configured font; dataset generation never substitutes."""
    try:
        resolution = resolve_fonts(
            config.fonts.keys,
            min_required=config.fonts.min_required,
        )
    except FontResolutionError as exc:
        raise DatasetError(f"font resolution failed: {exc}") from exc
    if len(resolution.available) != len(config.fonts.keys):
        raise DatasetError(
            "dataset generation requires every configured font to be available; "
            f"unavailable -> {resolution.describe_unavailable()}"
        )
    return resolution


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, format="PNG")


def load_grayscale(path: Path) -> np.ndarray:
    """Load a single-channel 8-bit PNG and validate the shared image contract."""
    with Image.open(path) as handle:
        if handle.mode != "L":
            raise DatasetError(f"expected an 8-bit grayscale image, got mode {handle.mode}: {path}")
        image = np.array(handle, dtype=np.uint8).copy()
    validate_image(image, name=str(path))
    return image


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a deterministic CSV atomically (temp file then replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n", float_format="%.12g")
    temporary.replace(path)


def _write_json(payload: object, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _counts(values: Iterable[object]) -> dict[str, int]:
    counter = Counter(str(value) for value in values)
    return {key: counter[key] for key in sorted(counter)}


def generate_dataset(
    config: DatasetConfig,
    *,
    output_dir: Path | str | None = None,
    overwrite: bool = False,
    extract_features_from_images: bool = True,
) -> DatasetResult:
    """Generate the configured dataset.

    Parameters
    ----------
    config:
        Validated dataset configuration.
    output_dir:
        Optional override for ``dataset.output_dir`` (useful for reproducibility
        runs that must land in two separate directories).
    overwrite:
        Replace an existing non-empty target directory. Without this flag an
        existing target raises :class:`DatasetExistsError` instead of being
        silently overwritten.
    extract_features_from_images:
        Compute the feature table from the in-memory degraded images. If false,
        ``features.csv`` is written with headers only; ``write_feature_table``
        can rebuild it later from the images on disk.

    Raises
    ------
    DatasetExistsError
        If the target exists and ``overwrite`` is false.
    DatasetError
        If fonts are unavailable or split assignment is infeasible.
    """
    target = Path(output_dir) if output_dir is not None else config.dataset.output_dir
    target = target.expanduser()
    if target.exists() and not overwrite:
        raise DatasetExistsError(
            f"dataset directory already exists: {target}. "
            "Pass --overwrite to replace it, or choose a different output directory."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(target.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    try:
        resolution = _resolve_all_fonts(config)
        plans = plan_sources(config, resolution)
        manifest_rows: list[dict[str, object]] = []
        source_rows: list[dict[str, object]] = []
        ground_truth_rows: list[dict[str, str]] = []
        feature_rows: list[tuple[str, np.ndarray]] = []
        image_hashes: dict[str, str] = {}

        for plan in plans:
            render_config = config.render.to_render_config(
                text_length=plan.text_length, charset=config.dataset.charset
            )
            clean = render_clean(
                plan.font,
                plan.text,
                render_config,
                rng=np.random.default_rng(plan.render_seed),
            )
            clean_rel = Path("images") / "clean" / f"{plan.source_id}.png"
            _write_png(staging / clean_rel, clean.image)
            image_hashes[_relative(clean_rel)] = clean.sha256

            source_rows.append(
                {
                    "source_id": plan.source_id,
                    "split_group_id": plan.split_group_id,
                    "split": plan.split,
                    "font_id": plan.font.key,
                    "text_length": plan.text_length,
                    "render_seed": plan.render_seed,
                    "background_level": clean.background_level,
                    "height": int(clean.image.shape[0]),
                    "width": int(clean.image.shape[1]),
                    "clean_image_path": _relative(clean_rel),
                    "clean_image_sha256": clean.sha256,
                    "text_sha256": plan.text_sha256,
                    "generation_config_hash": config.generation_config_hash,
                    "dataset_version": config.dataset.version,
                }
            )
            ground_truth_rows.append({"source_id": plan.source_id, "text": plan.text})

            clean_sample_id = f"{plan.source_id}__{CLEAN_VARIANT_ID}"
            manifest_rows.append(
                _manifest_row(
                    config=config,
                    plan=plan,
                    sample_id=clean_sample_id,
                    variant_id=CLEAN_VARIANT_ID,
                    corruption_type=CLEAN_CORRUPTION_TYPE,
                    severity=0,
                    variant_seed=derive_noise_seed(
                        config.dataset.seed, "variant", plan.source_id, CLEAN_CORRUPTION_TYPE, 0
                    ),
                    parameters={},
                    image_rel=clean_rel,
                    reference_rel=clean_rel,
                    image=clean.image,
                    reference=clean.image,
                )
            )
            feature_rows.append((clean_sample_id, clean.image))

            for ladder in config.families.ladders:
                for severity in range(1, ladder.n_levels):
                    sample_id = f"{plan.source_id}__{_variant_id(ladder.family, severity)}"
                    variant_seed = derive_noise_seed(
                        config.dataset.seed, "variant", plan.source_id, ladder.family, severity
                    )
                    result = apply_corruption(
                        clean.image, ladder, severity, noise_seed=variant_seed
                    )
                    assert_no_identity_corruption(result.image, clean)
                    image_rel = Path("images") / ladder.family / f"{sample_id}.png"
                    _write_png(staging / image_rel, result.image)
                    image_hashes[_relative(image_rel)] = sha256_bytes(result.image.tobytes())
                    manifest_rows.append(
                        _manifest_row(
                            config=config,
                            plan=plan,
                            sample_id=sample_id,
                            variant_id=_variant_id(ladder.family, severity),
                            corruption_type=ladder.family,
                            severity=severity,
                            variant_seed=variant_seed,
                            parameters=_corruption_parameters(result.spec),
                            image_rel=image_rel,
                            reference_rel=clean_rel,
                            image=result.image,
                            reference=clean.image,
                        )
                    )
                    feature_rows.append((sample_id, result.image))

        manifest = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)
        manifest = _coerce_manifest_types(manifest)
        sources = pd.DataFrame(source_rows, columns=SOURCE_COLUMNS)
        sources = _coerce_source_types(sources)
        ground_truth = pd.DataFrame(ground_truth_rows, columns=GROUND_TRUTH_COLUMNS)
        if extract_features_from_images:
            features = extract_feature_table(feature_rows)
        else:  # pragma: no cover - exercised by CLI smoke tests
            features = pd.DataFrame(columns=("sample_id", *FEATURE_NAMES))

        _write_csv(manifest, staging / "manifest.csv")
        _write_csv(sources, staging / "sources.csv")
        _write_csv(ground_truth, staging / "ground_truth.csv")
        _write_csv(features, staging / "features.csv")

        summary = _build_summary(
            config=config,
            target=staging,
            manifest=manifest,
            sources=sources,
            features=features,
            ground_truth=ground_truth,
            resolution=resolution,
            image_hashes=image_hashes,
        )
        summary["counts"]["disk_usage_bytes"] = _disk_usage(staging)
        _write_json(summary, staging / "dataset_summary.json")

        if target.exists():
            shutil.rmtree(target)
        staging.rename(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return DatasetResult(
        root=target,
        config=config,
        manifest=manifest,
        sources=sources,
        features=features,
        summary=summary,
    )


def assert_no_identity_corruption(image: np.ndarray, clean: CleanRender) -> None:
    """Guard against a misconfigured ladder silently producing a clean variant."""
    if np.array_equal(image, clean.image):
        raise DatasetError(
            f"corruption produced an identical image for source {clean.sha256[:12]}; "
            "ladder parameters must be strictly increasing"
        )


def _manifest_row(
    *,
    config: DatasetConfig,
    plan: SourcePlan,
    sample_id: str,
    variant_id: str,
    corruption_type: str,
    severity: int,
    variant_seed: int,
    parameters: Mapping[str, float],
    image_rel: Path,
    reference_rel: Path,
    image: np.ndarray,
    reference: np.ndarray,
) -> dict[str, object]:
    metrics = compute_pair_metrics(reference, image, ("ssim", "psnr"))
    psnr_value = metrics["psnr"]
    psnr_infinite = math.isinf(psnr_value)
    return {
        "sample_id": sample_id,
        "source_id": plan.source_id,
        "variant_id": variant_id,
        "split_group_id": plan.split_group_id,
        "split": plan.split,
        "text_length": plan.text_length,
        "font_id": plan.font.key,
        "render_seed": plan.render_seed,
        "variant_seed": variant_seed,
        "corruption_type": corruption_type,
        "severity_level": severity,
        "corruption_parameters": canonical_json(dict(parameters)),
        "image_path": _relative(image_rel),
        "reference_path": _relative(reference_rel),
        "ssim_target": float(metrics["ssim"]),
        "psnr_target": None if psnr_infinite else float(psnr_value),
        "psnr_infinite": bool(psnr_infinite),
        "height": int(image.shape[0]),
        "width": int(image.shape[1]),
        "dataset_version": config.dataset.version,
        "generation_config_hash": config.generation_config_hash,
        "text_sha256": plan.text_sha256,
        "image_sha256": sha256_bytes(image.tobytes()),
        "reference_sha256": sha256_bytes(reference.tobytes()),
    }


def _coerce_manifest_types(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in (
        "text_length",
        "render_seed",
        "variant_seed",
        "severity_level",
        "height",
        "width",
    ):
        frame[column] = frame[column].astype("int64")
    frame["ssim_target"] = frame["ssim_target"].astype("float64")
    frame["psnr_target"] = frame["psnr_target"].astype("float64")
    frame["psnr_infinite"] = frame["psnr_infinite"].astype(bool)
    return frame


def _coerce_source_types(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in ("text_length", "render_seed", "background_level", "height", "width"):
        frame[column] = frame[column].astype("int64")
    return frame


def _disk_usage(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _build_summary(
    *,
    config: DatasetConfig,
    target: Path,
    manifest: pd.DataFrame,
    sources: pd.DataFrame,
    features: pd.DataFrame,
    ground_truth: pd.DataFrame,
    resolution: FontResolution,
    image_hashes: Mapping[str, str],
) -> dict[str, Any]:
    split_summary: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        split_sources = sources[sources["split"] == split]
        split_manifest = manifest[manifest["split"] == split]
        split_summary[split] = {
            "sources": int(len(split_sources)),
            "samples": int(len(split_manifest)),
            "fonts": _counts(split_sources["font_id"]),
            "families": _counts(split_manifest["corruption_type"]),
            "severities": _counts(split_manifest["severity_level"]),
            "text_lengths": _counts(split_sources["text_length"]),
            "background_levels": _counts(split_sources["background_level"]),
        }

    summary = {
        "dataset_name": config.dataset.name,
        "dataset_version": config.dataset.version,
        "generation_config_hash": config.generation_config_hash,
        "feature_set_version": FEATURE_SET_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "package_version": package_version(),
        "code_revision": code_revision(),
        "config": config.generation_config,
        "diagnostics_config": {
            "trivial_single_feature_r2": config.diagnostics.trivial_single_feature_r2,
            "strong_confound_r2": config.diagnostics.strong_confound_r2,
            "min_group_rows": config.diagnostics.min_group_rows,
        },
        "fonts": {
            "requested": list(config.fonts.keys),
            "min_required": config.fonts.min_required,
            "resolved": [
                {"key": font.key, "path": str(font.path)} for font in resolution.available
            ],
            "unavailable": [
                {"key": font.key, "reason": font.reason} for font in resolution.unavailable
            ],
        },
        "counts": {
            "sources": int(len(sources)),
            "samples": int(len(manifest)),
            "clean_samples": int((manifest["corruption_type"] == CLEAN_CORRUPTION_TYPE).sum()),
            "feature_rows": int(len(features)),
            "unique_texts": int(ground_truth["text"].nunique()),
            "by_split": _counts(manifest["split"]),
            "by_family": _counts(manifest["corruption_type"]),
            "by_severity": _counts(manifest["severity_level"]),
            "by_font": _counts(manifest["font_id"]),
            "by_text_length": _counts(manifest["text_length"]),
            "by_background_level": _counts(sources["background_level"]),
            "disk_usage_bytes": 0,  # filled after the staging directory is complete
        },
        "splits": split_summary,
        "split_policy": {
            "ratios": {name: ratio for name, ratio in config.splits.ratios},
            "seed": config.splits.seed,
            "min_sources_per_split": config.splits.min_sources_per_split,
            "stratify_by_font": config.splits.stratify_by_font,
            "assignment_sha256": sha256_bytes(
                canonical_json(
                    sorted(
                        zip(sources["source_id"].tolist(), sources["split"].tolist(), strict=True)
                    )
                ).encode("utf-8")
            ),
        },
        "text_policy": {
            "public_manifest_contains_text": False,
            "internal_ground_truth_file": "ground_truth.csv",
            "public_text_identifier": "text_sha256",
            "note": (
                "Raw text is kept only in the internal ground-truth file for reproduction "
                "and verification. It never appears in manifest.csv, sources.csv, or "
                "dataset_summary.json."
            ),
        },
        "artifacts": {
            "manifest_sha256": sha256_file(target / "manifest.csv"),
            "sources_sha256": sha256_file(target / "sources.csv"),
            "ground_truth_sha256": sha256_file(target / "ground_truth.csv"),
            "features_sha256": sha256_file(target / "features.csv"),
            "n_image_files": len(image_hashes),
            "image_pixel_hash_set": sha256_bytes(
                canonical_json(sorted(image_hashes.items())).encode("utf-8")
            ),
        },
    }
    return summary


def load_dataset(
    root: Path | str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load manifest, sources, features and summary from ``root``."""
    root = Path(root)
    missing = [
        name
        for name in (
            "manifest.csv",
            "sources.csv",
            "features.csv",
            "dataset_summary.json",
            "ground_truth.csv",
        )
        if not (root / name).is_file()
    ]
    if missing:
        raise DatasetError(f"dataset at {root} is missing required file(s): {missing}")
    manifest = pd.read_csv(root / "manifest.csv")
    sources = pd.read_csv(root / "sources.csv")
    features = pd.read_csv(root / "features.csv")
    summary = json.loads((root / "dataset_summary.json").read_text(encoding="utf-8"))
    return manifest, sources, features, summary


def write_feature_table(root: Path | str) -> pd.DataFrame:
    """Re-extract features from the images on disk and rewrite ``features.csv``.

    This is the supported way to rebuild features after a code change without
    regenerating pixels. The summary's feature hash is updated atomically.
    """
    root = Path(root)
    manifest, _, _, summary = load_dataset(root)
    samples = [
        (str(row.sample_id), load_grayscale(root / str(row.image_path)))
        for row in manifest.itertuples(index=False)
    ]
    features = extract_feature_table(samples)
    _write_csv(features, root / "features.csv")
    summary.setdefault("artifacts", {})["features_sha256"] = sha256_file(root / "features.csv")
    summary.setdefault("counts", {})["feature_rows"] = int(len(features))
    _write_json(summary, root / "dataset_summary.json")
    return features


def _check(checks: list[CheckResult], name: str, function: Any) -> None:
    try:
        detail = function()
    except Exception as exc:  # noqa: BLE001 - verification reports every failure
        checks.append(CheckResult(name=name, passed=False, detail=f"{type(exc).__name__}: {exc}"))
        return
    checks.append(CheckResult(name=name, passed=True, detail=str(detail)))


def verify_dataset(
    root: Path | str,
    *,
    recompute_targets: bool = True,
) -> VerificationReport:
    """Verify schema, paths, reference matching, targets, splits and features.

    Target verification recomputes SSIM and finite PSNR from the exact
    reference/variant pair referenced in each manifest row. Checks accumulate:
    a failure in one check does not hide the others.
    """
    root = Path(root)
    checks: list[CheckResult] = []

    required = (
        "manifest.csv",
        "sources.csv",
        "ground_truth.csv",
        "features.csv",
        "dataset_summary.json",
    )
    _check(
        checks,
        "required_artifacts",
        lambda: _require_files(root, required),
    )
    if any(not check.passed for check in checks):
        return VerificationReport(dataset=str(root), checks=tuple(checks))

    manifest = pd.read_csv(root / "manifest.csv")
    sources = pd.read_csv(root / "sources.csv")
    features = pd.read_csv(root / "features.csv")
    ground_truth = pd.read_csv(root / "ground_truth.csv")
    summary = json.loads((root / "dataset_summary.json").read_text(encoding="utf-8"))

    _check(checks, "manifest_schema", lambda: _check_schema(manifest, sources, ground_truth))
    _check(
        checks,
        "manifest_ids_unique",
        lambda: _require_unique(manifest, "sample_id"),
    )
    _check(
        checks,
        "manifest_paths_exist",
        lambda: _check_paths(root, manifest),
    )
    _check(
        checks,
        "reference_matches_source",
        lambda: _check_reference_matching(manifest, sources),
    )
    _check(
        checks,
        "conditional_psnr_targets",
        lambda: _check_psnr_condition(manifest),
    )
    _check(
        checks,
        "image_content_hashes",
        lambda: _check_image_hashes(root, manifest),
    )
    if recompute_targets:
        _check(
            checks,
            "targets_recomputed",
            lambda: _check_targets(root, manifest),
        )
    _check(checks, "split_isolation", lambda: _check_split_isolation(manifest, sources))
    _check(
        checks,
        "feature_table",
        lambda: _check_feature_table(features, manifest),
    )
    _check(
        checks,
        "summary_consistency",
        lambda: _check_summary(summary, manifest, sources, features),
    )
    return VerificationReport(dataset=str(root), checks=tuple(checks))


def _require_files(root: Path, names: Iterable[str]) -> str:
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise DatasetError(f"missing required file(s): {missing}")
    if not (root / "images").is_dir():
        raise DatasetError("missing required directory: images/")
    return f"found {len(list(names))} files plus images/"


def _check_schema(manifest: pd.DataFrame, sources: pd.DataFrame, ground_truth: pd.DataFrame) -> str:
    missing = [column for column in MANIFEST_COLUMNS if column not in manifest.columns]
    if missing:
        raise DatasetError(f"manifest is missing column(s): {missing}")
    missing_sources = [column for column in SOURCE_COLUMNS if column not in sources.columns]
    if missing_sources:
        raise DatasetError(f"sources.csv is missing column(s): {missing_sources}")
    missing_truth = [
        column for column in GROUND_TRUTH_COLUMNS if column not in ground_truth.columns
    ]
    if missing_truth:
        raise DatasetError(f"ground_truth.csv is missing column(s): {missing_truth}")
    if "text" in manifest.columns:
        raise DatasetError("public manifest must not contain raw text")
    if sources["source_id"].duplicated().any():
        raise DatasetError("sources.csv contains duplicate source_id values")
    return f"{len(manifest)} manifest rows, {len(sources)} sources, schema complete"


def _require_unique(frame: pd.DataFrame, column: str) -> str:
    if frame[column].isna().any():
        raise DatasetError(f"column {column!r} contains missing values")
    if frame[column].duplicated().any():
        duplicates = frame.loc[frame[column].duplicated(), column].head(5).tolist()
        raise DatasetError(f"column {column!r} is not unique, e.g. {duplicates}")
    return f"{len(frame)} unique {column} values"


def _check_paths(root: Path, manifest: pd.DataFrame) -> str:
    resolved_root = root.resolve()
    missing: list[str] = []
    escaped: list[str] = []
    for row in manifest.itertuples(index=False):
        for column in ("image_path", "reference_path"):
            relative = str(getattr(row, column))
            path = (root / relative).resolve()
            if not path.is_relative_to(resolved_root):
                escaped.append(relative)
            elif not path.is_file():
                missing.append(relative)
    if escaped:
        raise DatasetError(f"paths escape the dataset root: {sorted(set(escaped))[:5]}")
    if missing:
        raise DatasetError(f"missing image file(s): {sorted(set(missing))[:5]}")
    return "all image and reference paths exist inside the dataset root"


def _check_reference_matching(manifest: pd.DataFrame, sources: pd.DataFrame) -> str:
    source_index = sources.set_index("source_id")
    unmatched = 0
    for row in manifest.itertuples(index=False):
        if row.source_id not in source_index.index:
            raise DatasetError(
                f"manifest row {row.sample_id} references unknown source {row.source_id}"
            )
        source = source_index.loc[row.source_id]
        if row.split_group_id != row.source_id:
            raise DatasetError(f"{row.sample_id}: split_group_id must equal source_id")
        if row.split != source["split"]:
            raise DatasetError(f"{row.sample_id}: split disagrees with sources.csv")
        if row.reference_path != source["clean_image_path"]:
            raise DatasetError(f"{row.sample_id}: reference_path is not the source clean image")
        if row.reference_sha256 != source["clean_image_sha256"]:
            raise DatasetError(f"{row.sample_id}: reference hash is not the source clean hash")
        if row.text_sha256 != source["text_sha256"]:
            raise DatasetError(f"{row.sample_id}: text hash disagrees with sources.csv")
        unmatched += 1
    if set(sources["source_id"]) - set(manifest["source_id"]):
        raise DatasetError("some sources have no manifest rows")
    return f"{unmatched} rows reference the correct clean source"


def _check_psnr_condition(manifest: pd.DataFrame) -> str:
    infinite = manifest["psnr_infinite"].astype(bool)
    clean = manifest["corruption_type"] == CLEAN_CORRUPTION_TYPE
    if not (infinite == clean).all():
        raise DatasetError(
            "psnr_infinite must be true exactly for clean identity samples; "
            f"mismatches: {manifest.loc[infinite != clean, 'sample_id'].head(5).tolist()}"
        )
    finite = ~infinite
    if manifest.loc[finite, "psnr_target"].isna().any():
        raise DatasetError("finite PSNR rows must record psnr_target")
    if manifest.loc[infinite, "psnr_target"].notna().any():
        raise DatasetError("infinite PSNR rows must leave psnr_target empty")
    return f"{int(clean.sum())} clean rows use an explicit infinite-PSNR flag"


def _check_image_hashes(root: Path, manifest: pd.DataFrame) -> str:
    expected: dict[str, str] = {}
    for row in manifest.itertuples(index=False):
        relative = str(row.image_path)
        previous = expected.setdefault(relative, str(row.image_sha256))
        if previous != row.image_sha256:
            raise DatasetError(f"inconsistent image_sha256 for {relative}")
    mismatches = []
    for relative, digest in expected.items():
        image = load_grayscale(root / relative)
        if sha256_bytes(image.tobytes()) != digest:
            mismatches.append(relative)
    if mismatches:
        raise DatasetError(f"image pixel hash mismatch: {sorted(mismatches)[:5]}")
    return f"{len(expected)} image files match their recorded pixel hashes"


def _check_targets(root: Path, manifest: pd.DataFrame) -> str:
    references: dict[str, np.ndarray] = {}
    images: dict[str, np.ndarray] = {}
    mismatches: list[str] = []
    for row in manifest.itertuples(index=False):
        reference_path = str(row.reference_path)
        image_path = str(row.image_path)
        if reference_path not in references:
            references[reference_path] = load_grayscale(root / reference_path)
        if image_path not in images:
            images[image_path] = load_grayscale(root / image_path)
        metrics = compute_pair_metrics(
            references[reference_path], images[image_path], ("ssim", "psnr")
        )
        if abs(metrics["ssim"] - float(row.ssim_target)) > 1e-9:
            mismatches.append(f"{row.sample_id}:ssim")
        if bool(row.psnr_infinite):
            if not math.isinf(metrics["psnr"]):
                mismatches.append(f"{row.sample_id}:psnr-inf")
        elif abs(metrics["psnr"] - float(row.psnr_target)) > 1e-9:
            mismatches.append(f"{row.sample_id}:psnr")
    if mismatches:
        raise DatasetError(
            f"{len(mismatches)} target(s) disagree with the exact pair, e.g. {mismatches[:5]}"
        )
    return f"recomputed SSIM/PSNR targets for {len(manifest)} rows from their exact references"


def _check_split_isolation(manifest: pd.DataFrame, sources: pd.DataFrame) -> str:
    per_source = manifest.groupby("source_id")["split"].nunique()
    if (per_source > 1).any():
        offenders = per_source[per_source > 1].index[:5].tolist()
        raise DatasetError(f"source(s) span multiple splits: {offenders}")
    if set(manifest["split"]) != {"train", "val", "test"}:
        raise DatasetError(f"expected all three splits, found {sorted(set(manifest['split']))}")
    if sources["split_group_id"].duplicated().any():
        raise DatasetError("split_group_id must be unique per source")
    source_splits = sources.groupby("source_id")["split"].nunique()
    if (source_splits > 1).any():
        raise DatasetError("sources.csv assigns a source to multiple splits")
    counts = sources["split"].value_counts().to_dict()
    return f"no source crosses splits; source counts {counts}"


def _check_feature_table(features: pd.DataFrame, manifest: pd.DataFrame) -> str:
    expected = ("sample_id", *FEATURE_NAMES)
    if tuple(features.columns) != expected:
        raise DatasetError(
            f"feature columns must be exactly {expected}, got {tuple(features.columns)}"
        )
    if features["sample_id"].duplicated().any():
        raise DatasetError("features.csv contains duplicate sample_id values")
    if set(features["sample_id"]) != set(manifest["sample_id"]):
        raise DatasetError("features.csv sample IDs do not match the manifest")
    values = features[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise DatasetError("features.csv contains NaN or infinite values")
    forbidden = {"ssim", "psnr", "ms_ssim"}
    leaked = forbidden & {name.lower() for name in features.columns}
    if leaked:
        raise DatasetError(f"target metric(s) used as features: {sorted(leaked)}")
    return f"{len(features)} feature rows x {len(FEATURE_NAMES)} features are finite"


def _check_summary(
    summary: dict[str, Any],
    manifest: pd.DataFrame,
    sources: pd.DataFrame,
    features: pd.DataFrame,
) -> str:
    if summary.get("generation_config_hash") != manifest["generation_config_hash"].iloc[0]:
        raise DatasetError("summary generation_config_hash disagrees with the manifest")
    if set(manifest["generation_config_hash"]) != {summary.get("generation_config_hash")}:
        raise DatasetError("manifest contains multiple generation_config_hash values")
    if set(manifest["dataset_version"]) != {summary.get("dataset_version")}:
        raise DatasetError("manifest dataset_version disagrees with the summary")
    counts = summary.get("counts", {})
    if counts.get("sources") != len(sources):
        raise DatasetError("summary source count disagrees with sources.csv")
    if counts.get("samples") != len(manifest):
        raise DatasetError("summary sample count disagrees with the manifest")
    if counts.get("feature_rows") != len(features):
        raise DatasetError("summary feature count disagrees with features.csv")
    if summary.get("feature_names") != list(FEATURE_NAMES):
        raise DatasetError("summary feature order disagrees with this code version")
    split_policy = summary.get("split_policy", {})
    expected_split_hash = sha256_bytes(
        canonical_json(
            sorted(zip(sources["source_id"].tolist(), sources["split"].tolist(), strict=True))
        ).encode("utf-8")
    )
    if split_policy.get("assignment_sha256") != expected_split_hash:
        raise DatasetError("summary split assignment hash disagrees with sources.csv")
    if split_policy.get("seed") is None:
        raise DatasetError("summary is missing the split seed")
    return "summary counts, hashes, split assignment and feature names agree with the tables"


def compare_datasets(a: Path | str, b: Path | str) -> VerificationReport:
    """Compare two dataset directories for run-to-run reproducibility.

    Manifest, source, ground-truth and feature CSVs must be byte-identical.
    Image content is compared through the recorded pixel hashes, so PNG encoder
    differences cannot cause a false mismatch. Run metadata (counts, config
    hash, feature names, text policy) must also agree.
    """
    left, right = Path(a), Path(b)
    checks: list[CheckResult] = []

    def artifact(name: str) -> str:
        left_hash = sha256_file(left / name)
        right_hash = sha256_file(right / name)
        if left_hash != right_hash:
            raise DatasetError(f"{name} differs ({left_hash[:12]} != {right_hash[:12]})")
        return f"{name} identical ({left_hash[:12]})"

    for name in ("manifest.csv", "sources.csv", "ground_truth.csv", "features.csv"):
        _check(checks, name, lambda name=name: artifact(name))

    def metadata() -> str:
        left_summary = _read_summary(left)
        right_summary = _read_summary(right)
        keys = (
            "generation_config_hash",
            "dataset_name",
            "dataset_version",
            "feature_set_version",
            "feature_names",
            "counts",
            "text_policy",
        )
        differences = [key for key in keys if left_summary.get(key) != right_summary.get(key)]
        left_counts = {
            key: value
            for key, value in left_summary.get("counts", {}).items()
            if key != "disk_usage_bytes"
        }
        right_counts = {
            key: value
            for key, value in right_summary.get("counts", {}).items()
            if key != "disk_usage_bytes"
        }
        if left_counts != right_counts:
            differences.append("counts (excluding disk usage)")
        if differences:
            raise DatasetError(f"summary metadata differs for: {differences}")
        return "config hash, counts, feature names and text policy agree"

    _check(checks, "summary_metadata", metadata)

    def images() -> str:
        left_manifest = pd.read_csv(left / "manifest.csv")
        right_manifest = pd.read_csv(right / "manifest.csv")
        left_hashes = dict(
            zip(left_manifest["sample_id"], left_manifest["image_sha256"], strict=True)
        )
        right_hashes = dict(
            zip(right_manifest["sample_id"], right_manifest["image_sha256"], strict=True)
        )
        if left_hashes != right_hashes:
            differing = sorted(
                key
                for key in set(left_hashes) | set(right_hashes)
                if left_hashes.get(key) != right_hashes.get(key)
            )
            raise DatasetError(f"image pixel hashes differ for {len(differing)} sample(s)")
        return f"{len(left_hashes)} image pixel hashes identical"

    _check(checks, "image_content", images)
    return VerificationReport(dataset=f"{left} vs {right}", checks=tuple(checks))


def _read_summary(root: Path) -> dict[str, Any]:
    path = root / "dataset_summary.json"
    if not path.is_file():
        raise DatasetError(f"missing dataset_summary.json in {root}")
    return json.loads(path.read_text(encoding="utf-8"))
