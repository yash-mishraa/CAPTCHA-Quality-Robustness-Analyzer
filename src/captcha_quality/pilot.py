"""Milestone 1 pilot: validate reference-based image-quality targets.

The pilot renders a small deterministic corpus of synthetic text images, applies
two controlled corruption families at several predefined severities, scores each
degraded image against its own clean reference, and evaluates a predeclared
acceptance gate.

What this experiment does **not** do:

* It does not measure human readability or CAPTCHA solvability.
* It does not train a model.
* It does not build the Milestone 2 dataset pipeline.

Run it with::

    uv run captcha-pilot --config configs/pilot_target_validation.toml

Generated images and summaries are written to ``reports/pilot/`` (Git-ignored).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from captcha_quality.corruptions import (
    SUPPORTED_FAMILIES,
    CorruptionError,
    SeverityLadder,
    apply_corruption,
    derive_noise_seed,
)
from captcha_quality.gate import (
    GateConfigurationError,
    GateThresholds,
    MetricVerdict,
    deferred_metrics,
    evaluate_gate,
    render_markdown,
)
from captcha_quality.imaging import validate_pair
from captcha_quality.provenance import code_revision, json_safe, package_version
from captcha_quality.quality_metrics import SUPPORTED_METRICS, compute_pair_metrics
from captcha_quality.render import (
    FontResolution,
    FontResolutionError,
    RenderConfig,
    RenderError,
    ResolvedFont,
    render_clean,
    resolve_fonts,
    sample_text,
)
from captcha_quality.summarise import summarise

DEFAULT_CONFIG_PATH: Final[Path] = Path("configs/pilot_target_validation.toml")
SUPPORTED_GATE_KEYS: Final[tuple[str, ...]] = (
    "spearman_max",
    "paired_effect_max",
    "violation_rate_max",
    "group_spearman_max",
    "eta_squared_min",
    "ssim_range_min",
    "psnr_range_min_db",
    "ink_correlation_max",
)


class PilotConfigError(ValueError):
    """Raised when the pilot configuration is missing or invalid."""


@dataclass(frozen=True, slots=True)
class TrialPlan:
    """One clean source image plus the seeds used to distort it."""

    trial_id: str
    font: ResolvedFont
    text: str
    text_index: int
    repeat_seed: int
    jitter_seed: int


@dataclass(frozen=True, slots=True)
class PilotConfig:
    """Validated pilot configuration, loaded from TOML."""

    name: str
    seed: int
    repeat_seeds: tuple[int, ...]
    texts_per_font: int
    metrics: tuple[str, ...]
    deferred_metrics: tuple[str, ...]
    render: RenderConfig
    font_keys: tuple[str, ...]
    min_fonts_required: int
    ladders: tuple[SeverityLadder, ...]
    gate: GateThresholds
    output_dir: Path
    config_path: Path

    def ladder(self, family: str) -> SeverityLadder:
        """Return the ladder for ``family``, or raise if it is not configured."""
        for ladder in self.ladders:
            if ladder.family == family:
                return ladder
        raise PilotConfigError(f"no ladder configured for family {family!r}")

    @property
    def config_sha256(self) -> str:
        """Content hash of the configuration file used for this run."""
        return hashlib.sha256(self.config_path.read_bytes()).hexdigest()

    def provenance(self) -> dict[str, object]:
        """Return a JSON-serialisable description of the run configuration."""
        return {
            "name": self.name,
            "seed": self.seed,
            "repeat_seeds": list(self.repeat_seeds),
            "texts_per_font": self.texts_per_font,
            "metrics": list(self.metrics),
            "deferred_metrics": list(self.deferred_metrics),
            "canvas": list(self.render.canvas),
            "charset": self.render.charset,
            "text_length": self.render.text_length,
            "fit_fraction": self.render.fit_fraction,
            "vertical_padding": self.render.vertical_padding,
            "jitter_fraction": self.render.jitter_fraction,
            "fonts": list(self.font_keys),
            "min_fonts_required": self.min_fonts_required,
            "ladders": {
                ladder.family: {
                    "unit": ladder.unit,
                    "parameters": list(ladder.parameters),
                }
                for ladder in self.ladders
            },
            "gate": self.gate.as_dict(),
            "output_dir": str(self.output_dir),
            "config_path": str(self.config_path),
            "config_sha256": self.config_sha256,
        }

    @classmethod
    def from_toml(cls, path: Path | str) -> PilotConfig:
        """Load and validate a pilot configuration from TOML.

        Raises
        ------
        PilotConfigError
            If the file is missing, malformed, incomplete, or inconsistent.
        """
        config_path = Path(path)
        if not config_path.is_file():
            raise PilotConfigError(f"pilot config not found: {config_path}")
        try:
            with config_path.open("rb") as handle:
                data = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise PilotConfigError(f"invalid TOML in {config_path}: {exc}") from exc

        run = _table(data, "run", config_path)
        render_table = _table(data, "render", config_path)
        fonts = _table(data, "fonts", config_path)
        families = _table(data, "families", config_path)
        gate_table = _table(data, "gate", config_path)

        name = _string(run, "name", "run")
        seed = _integer(run, "seed", "run")
        repeat_seeds = tuple(_integer_list(run, "repeat_seeds", "run"))
        texts_per_font = _integer(run, "texts_per_font", "run")
        metrics = tuple(_string_list(run, "metrics", "run"))
        deferred = tuple(_string_list(run, "deferred_metrics", "run"))
        output_dir = Path(_string(run, "output_dir", "run"))

        if seed < 0:
            raise PilotConfigError("run.seed must be non-negative")
        if not repeat_seeds:
            raise PilotConfigError("run.repeat_seeds must not be empty")
        if len(set(repeat_seeds)) != len(repeat_seeds):
            raise PilotConfigError(f"run.repeat_seeds must be unique, got {repeat_seeds}")
        if any(value < 0 for value in repeat_seeds):
            raise PilotConfigError("run.repeat_seeds values must be non-negative")
        if texts_per_font < 1:
            raise PilotConfigError("run.texts_per_font must be >= 1")
        if not metrics:
            raise PilotConfigError("run.metrics must not be empty")
        unknown = [metric for metric in metrics if metric not in SUPPORTED_METRICS]
        if unknown:
            raise PilotConfigError(
                f"unsupported metric(s) {unknown}; this pilot computes {SUPPORTED_METRICS}. "
                "Record unimplemented metrics under run.deferred_metrics instead."
            )
        for metric in deferred:
            if metric not in deferred_metrics((metric,)):
                raise PilotConfigError(
                    f"metric {metric!r} is listed as deferred but has no recorded deferral "
                    f"reason; known deferrals: {sorted(deferred_metrics(SUPPORTED_METRICS + ('ms_ssim',)))}"
                )
            if metric in metrics:
                raise PilotConfigError(
                    f"metric {metric!r} is both computed and deferred; choose one"
                )

        render_config = RenderConfig(
            canvas=(
                _integer(render_table, "canvas_width", "render"),
                _integer(render_table, "canvas_height", "render"),
            ),
            charset=_string(render_table, "charset", "render"),
            text_length=_integer(render_table, "text_length", "render"),
            fit_fraction=_number(render_table, "fit_fraction", "render"),
            vertical_padding=_number(render_table, "vertical_padding", "render"),
            jitter_fraction=_number(render_table, "jitter_fraction", "render"),
        )

        min_fonts_required = _integer(fonts, "min_required", "fonts")
        font_keys = tuple(_string_list(fonts, "keys", "fonts"))
        if not font_keys:
            raise PilotConfigError("fonts.keys must not be empty")
        if not 1 <= min_fonts_required <= len(font_keys):
            raise PilotConfigError(
                "fonts.min_required must be between 1 and the number of requested fonts"
            )

        family_names = tuple(_string_list(families, "names", "families"))
        if not family_names:
            raise PilotConfigError("families.names must not be empty")
        if len(set(family_names)) != len(family_names):
            raise PilotConfigError("families.names must be unique")
        unsupported_families = [name_ for name_ in family_names if name_ not in SUPPORTED_FAMILIES]
        if unsupported_families:
            raise PilotConfigError(
                f"unsupported family/families {unsupported_families}; "
                f"this pilot implements {SUPPORTED_FAMILIES}"
            )
        ladders: list[SeverityLadder] = []
        for family in family_names:
            family_table = _table(families, family, f"families.{family}")
            levels = tuple(_number_list(family_table, "levels", f"families.{family}"))
            unit = _string(family_table, "unit", f"families.{family}")
            ladders.append(SeverityLadder(family=family, parameters=levels, unit=unit))

        gate_values = {key: gate_table[key] for key in SUPPORTED_GATE_KEYS if key in gate_table}
        unknown_gate_keys = sorted(set(gate_table) - set(SUPPORTED_GATE_KEYS))
        if unknown_gate_keys:
            raise GateConfigurationError(f"gate config has unknown key(s): {unknown_gate_keys}")
        gate = GateThresholds.from_mapping(gate_values)

        return cls(
            name=name,
            seed=seed,
            repeat_seeds=repeat_seeds,
            texts_per_font=texts_per_font,
            metrics=metrics,
            deferred_metrics=deferred,
            render=render_config,
            font_keys=font_keys,
            min_fonts_required=min_fonts_required,
            ladders=tuple(ladders),
            gate=gate,
            output_dir=output_dir,
            config_path=config_path,
        )


@dataclass(slots=True)
class PilotResult:
    """In-memory result of a pilot run."""

    config: PilotConfig
    font_resolution: FontResolution
    trials: pd.DataFrame
    sanity: pd.DataFrame
    summary: dict[str, pd.DataFrame]
    verdicts: tuple[MetricVerdict, ...]
    output_dir: Path
    report_path: Path


def _table(data: dict[str, object], key: str, context: Path | str) -> dict[str, object]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise PilotConfigError(f"missing or invalid [{key}] table in {context}")
    return value


def _string(table: dict[str, object], key: str, context: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise PilotConfigError(f"{context}.{key} must be a non-empty string")
    return value


def _integer(table: dict[str, object], key: str, context: str) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise PilotConfigError(f"{context}.{key} must be an integer")
    return value


def _number(table: dict[str, object], key: str, context: str) -> float:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PilotConfigError(f"{context}.{key} must be a number")
    return float(value)


def _string_list(table: dict[str, object], key: str, context: str) -> list[str]:
    value = table.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise PilotConfigError(f"{context}.{key} must be a list of strings")
    return value


def _integer_list(table: dict[str, object], key: str, context: str) -> list[int]:
    value = table.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise PilotConfigError(f"{context}.{key} must be a list of integers")
    return value


def _number_list(table: dict[str, object], key: str, context: str) -> list[float]:
    value = table.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, int | float) and not isinstance(item, bool) for item in value
    ):
        raise PilotConfigError(f"{context}.{key} must be a list of numbers")
    return [float(item) for item in value]


def plan_trials(config: PilotConfig, fonts: Sequence[ResolvedFont]) -> list[TrialPlan]:
    """Build the deterministic trial plan (font x text x repeat seed)."""
    plans: list[TrialPlan] = []
    for font in fonts:
        for text_index in range(config.texts_per_font):
            text_rng = np.random.default_rng(
                derive_noise_seed(config.seed, "text", font.key, text_index)
            )
            text = sample_text(text_rng, config.render)
            for repeat_seed in config.repeat_seeds:
                plans.append(
                    TrialPlan(
                        trial_id=f"{font.key}|t{text_index}|{text}|r{repeat_seed}",
                        font=font,
                        text=text,
                        text_index=text_index,
                        repeat_seed=repeat_seed,
                        jitter_seed=derive_noise_seed(
                            config.seed, "render", font.key, text_index, repeat_seed
                        ),
                    )
                )
    return plans


def run_pilot(
    config: PilotConfig,
    *,
    output_dir: Path | None = None,
    write_outputs: bool = True,
) -> PilotResult:
    """Run the pilot end to end and return its tables and gate verdicts.

    Parameters
    ----------
    config:
        Validated configuration.
    output_dir:
        Override for the configured output directory.
    write_outputs:
        When ``False``, compute results without writing files (used by tests).

    Raises
    ------
    FontResolutionError
        If too few requested fonts are available.
    """
    font_resolution = resolve_fonts(config.font_keys, min_required=config.min_fonts_required)
    plans = plan_trials(config, font_resolution.available)

    records: list[dict[str, object]] = []
    sanity_records: list[dict[str, object]] = []
    examples: dict[str, dict[str, dict[int, np.ndarray]]] = {
        ladder.family: {} for ladder in config.ladders
    }

    for plan in plans:
        clean = render_clean(
            plan.font,
            plan.text,
            config.render,
            rng=np.random.default_rng(plan.jitter_seed),
        )

        # Clean vs. itself: the deterministic sanity condition.
        self_metrics = compute_pair_metrics(clean.image, clean.image, config.metrics)
        sanity_records.append(
            {
                "trial_id": plan.trial_id,
                "font": plan.font.key,
                "text": plan.text,
                "seed": plan.repeat_seed,
                **self_metrics,
                "identical": True,
                "clean_sha256": clean.sha256,
            }
        )

        for ladder in config.ladders:
            if plan.text_index == 0 and plan.repeat_seed == config.repeat_seeds[0]:
                examples[ladder.family][plan.font.key] = {}

            for severity in range(ladder.n_levels):
                noise_seed = derive_noise_seed(
                    config.seed, "noise", plan.trial_id, ladder.family, severity
                )
                corruption = apply_corruption(
                    clean.image,
                    ladder,
                    severity,
                    noise_seed=noise_seed,
                )
                validate_pair(clean.image, corruption.image)
                metrics = compute_pair_metrics(clean.image, corruption.image, config.metrics)
                corrupted_sha = hashlib.sha256(corruption.image.tobytes()).hexdigest()

                records.append(
                    {
                        "trial_id": plan.trial_id,
                        "font": plan.font.key,
                        "text": plan.text,
                        "seed": plan.repeat_seed,
                        "family": ladder.family,
                        "severity": severity,
                        "parameter": corruption.spec.parameter,
                        "unit": ladder.unit,
                        **metrics,
                        "ink_fraction": clean.ink_fraction,
                        "clipped_fraction": corruption.clipped_fraction,
                        "realised_noise_std": corruption.realised_noise_std,
                        "identical": corrupted_sha == clean.sha256,
                        "clean_sha256": clean.sha256,
                        "corrupted_sha256": corrupted_sha,
                    }
                )

                if plan.text_index == 0 and plan.repeat_seed == config.repeat_seeds[0]:
                    examples[ladder.family][plan.font.key][severity] = corruption.image

        # Guard the invariant that every severity used the same clean reference.
        if hashlib.sha256(clean.image.tobytes()).hexdigest() != clean.sha256:
            raise RuntimeError(
                f"clean reference mutated during trial {plan.trial_id}; "
                "corruption code must never modify its input"
            )

    trials = pd.DataFrame.from_records(records)
    sanity = pd.DataFrame.from_records(sanity_records)
    summary = summarise(trials, config.metrics)
    verdicts = evaluate_gate(summary, sanity, config.gate, config.metrics)

    resolved_output = Path(output_dir) if output_dir is not None else config.output_dir
    result = PilotResult(
        config=config,
        font_resolution=font_resolution,
        trials=trials,
        sanity=sanity,
        summary=summary,
        verdicts=verdicts,
        output_dir=resolved_output,
        report_path=resolved_output / "summary.md",
    )

    if write_outputs:
        _write_outputs(result, examples)
    return result


def _write_outputs(
    result: PilotResult,
    examples: dict[str, dict[str, dict[int, np.ndarray]]],
) -> None:
    """Write tables, JSON, report, and example images to the output directory."""
    output_dir = result.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    result.trials.to_csv(output_dir / "trials.csv", index=False)
    result.sanity.to_csv(output_dir / "sanity.csv", index=False)
    for name, table in result.summary.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    payload = {
        "pilot": result.config.provenance(),
        "package_version": package_version(),
        "code_revision": code_revision(),
        "fonts": {
            "resolved": [
                {"key": font.key, "path": str(font.path)}
                for font in result.font_resolution.available
            ],
            "unavailable": [
                {"key": font.key, "reason": font.reason}
                for font in result.font_resolution.unavailable
            ],
        },
        "counts": {
            "trials": int(result.trials["trial_id"].nunique()),
            "trial_rows": int(len(result.trials)),
            "sanity_rows": int(len(result.sanity)),
            "images_evaluated": int(len(result.trials)),
            "unique_texts": int(result.trials["text"].nunique()),
        },
        "thresholds": result.config.gate.as_dict(),
        "verdicts": [
            {
                "family": verdict.family,
                "metric": verdict.metric,
                "verdict": verdict.verdict,
                "failed_criteria": list(verdict.failed_criteria),
                "criteria": [
                    {
                        "id": criterion.id,
                        "description": criterion.description,
                        "observed": criterion.observed,
                        "threshold": criterion.threshold,
                        "passed": criterion.passed,
                        "note": criterion.note,
                    }
                    for criterion in verdict.criteria
                ],
            }
            for verdict in result.verdicts
        ],
        "deferred": {
            metric: reason
            for metric in result.config.deferred_metrics
            for reason in [deferred_metrics((metric,))[metric]]
        },
    }
    (output_dir / "gate_results.json").write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    header_lines = _report_header(result)
    report = render_markdown(
        result.summary,
        result.verdicts,
        result.config.gate,
        header_lines=header_lines,
        deferred={
            metric: deferred_metrics((metric,))[metric] for metric in result.config.deferred_metrics
        },
    )
    result.report_path.write_text(report + "\n", encoding="utf-8")

    for family, by_font in examples.items():
        if not by_font:
            continue
        ladder = result.config.ladder(family)
        row_keys = sorted(by_font)
        cells = [
            [by_font[font_key][severity] for severity in range(ladder.n_levels)]
            for font_key in row_keys
        ]
        column_labels = [
            f"s{severity}: {parameter:g}{ladder.unit}"
            for severity, parameter in enumerate(ladder.parameters)
        ]
        _contact_sheet(
            output_dir / f"examples_{family}.png",
            row_labels=row_keys,
            column_labels=column_labels,
            cells=cells,
        )


def _report_header(result: PilotResult) -> list[str]:
    config = result.config
    available = ", ".join(font.key for font in result.font_resolution.available)
    unavailable = result.font_resolution.describe_unavailable() or "none"
    trials = int(result.trials["trial_id"].nunique())
    rows = len(result.trials)
    return [
        f"- **Config:** `{config.config_path}` (sha256 `{config.config_sha256[:16]}…`)",
        f"- **Run seed:** {config.seed}; repeat seeds: {list(config.repeat_seeds)}",
        f"- **Fonts:** {available} (unavailable: {unavailable})",
        f"- **Canvas:** {config.render.canvas[0]}x{config.render.canvas[1]} px, "
        f"text length {config.render.text_length}, {config.texts_per_font} texts per font",
        f"- **Trials:** {trials} clean images; {rows} corrupted-pair rows; "
        f"{len(result.sanity)} clean self-comparisons",
        "- **Scope:** synthetic image similarity to a clean reference. "
        "Not a measurement of human readability or CAPTCHA solvability.",
    ]


def _contact_sheet(
    path: Path,
    *,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    cells: Sequence[Sequence[np.ndarray]],
    padding: int = 6,
    label_margin_left: int = 110,
    label_margin_top: int = 18,
) -> None:
    """Write a labelled grid of example images for manual inspection."""
    if not cells or not cells[0]:
        return
    cell_height, cell_width = cells[0][0].shape
    n_rows = len(cells)
    n_columns = len(cells[0])
    width = label_margin_left + n_columns * (cell_width + padding) + padding
    height = label_margin_top + n_rows * (cell_height + padding) + padding

    sheet = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=11)

    for column, label in enumerate(column_labels):
        x = label_margin_left + column * (cell_width + padding)
        draw.text((x, 3), label, fill=0, font=font)

    for row_index, (label, row_cells) in enumerate(zip(row_labels, cells, strict=True)):
        y = label_margin_top + row_index * (cell_height + padding)
        draw.text((4, y + cell_height // 2 - 6), label, fill=0, font=font)
        for column, image in enumerate(row_cells):
            x = label_margin_left + column * (cell_width + padding)
            sheet.paste(Image.fromarray(image), (x, y))

    sheet.save(path, format="PNG")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the target-validation pilot."""
    parser = argparse.ArgumentParser(
        prog="captcha-pilot",
        description="Run the Milestone 1 reference-metric validation pilot.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"pilot configuration TOML (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="override the output directory from the configuration",
    )
    args = parser.parse_args(argv)

    try:
        config = PilotConfig.from_toml(args.config)
        result = run_pilot(config, output_dir=args.output_dir)
    except (
        PilotConfigError,
        FontResolutionError,
        CorruptionError,
        GateConfigurationError,
        RenderError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"pilot '{result.config.name}': {result.trials['trial_id'].nunique()} trials, "
        f"{len(result.trials)} corrupted-pair rows, {len(result.sanity)} sanity rows"
    )
    for verdict in result.verdicts:
        failed = (
            f" (failed: {', '.join(verdict.failed_criteria)})" if verdict.failed_criteria else ""
        )
        print(f"  {verdict.family:16s} {verdict.metric:5s} {verdict.verdict}{failed}")
    print(f"outputs: {result.output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via the CLI test
    raise SystemExit(main())
