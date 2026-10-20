"""Command-line interface for the Milestone 2 dataset pipeline.

Commands
--------
``generate``
    Generate a dataset from a TOML configuration.
``verify``
    Validate schema, paths, reference matching, recomputed targets, split
    isolation and the feature table.
``features``
    Re-extract ``features.csv`` from the images on disk.
``diagnose``
    Run grouped, leakage-safe feature/target diagnostics and write a report.
``compare``
    Compare two dataset directories for run-to-run reproducibility.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from captcha_quality.config import ConfigError, DatasetConfig, load_config
from captcha_quality.dataset import (
    DatasetError,
    compare_datasets,
    generate_dataset,
    verify_dataset,
    write_feature_table,
)
from captcha_quality.diagnostics import diagnose_dataset
from captcha_quality.features import FeatureDependencyError
from captcha_quality.imaging import ImageContractError
from captcha_quality.quality_metrics import MetricUnavailableError
from captcha_quality.render import FontResolutionError

_EXPECTED_ERRORS = (
    ConfigError,
    DatasetError,
    FontResolutionError,
    FeatureDependencyError,
    MetricUnavailableError,
    ImageContractError,
)


def _default_report_dir(config: DatasetConfig) -> Path:
    name = config.dataset.name.replace(" ", "_").replace("-", "_")
    return Path("reports") / f"{name}_{config.dataset.version}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="captcha-dataset",
        description="Generate, verify and diagnose a reproducible synthetic image dataset.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate a dataset from a config")
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, default=None)
    generate.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output directory (never the default)",
    )
    generate.add_argument(
        "--skip-features",
        action="store_true",
        help="write an empty features.csv; rebuild later with the 'features' command",
    )

    verify = subparsers.add_parser("verify", help="verify a generated dataset")
    verify.add_argument("--dataset", type=Path, required=True)
    verify.add_argument(
        "--skip-targets",
        action="store_true",
        help="skip recomputing SSIM/PSNR from every reference/variant pair",
    )

    features = subparsers.add_parser("features", help="re-extract features.csv from images")
    features.add_argument("--dataset", type=Path, required=True)

    diagnose = subparsers.add_parser("diagnose", help="run feature/target diagnostics")
    diagnose.add_argument("--dataset", type=Path, required=True)
    diagnose.add_argument(
        "--config",
        type=Path,
        required=True,
        help="config supplying the predeclared diagnostic thresholds",
    )
    diagnose.add_argument("--output-dir", type=Path, default=None)

    compare = subparsers.add_parser("compare", help="compare two dataset directories")
    compare.add_argument("--a", type=Path, required=True)
    compare.add_argument("--b", type=Path, required=True)
    return parser


def _print_report(report: object) -> None:
    checks = report.checks
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")
    ok = report.ok
    print("verification: PASS" if ok else "verification: FAIL")


def _command_generate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    print(f"config: {config.describe()}")
    started = time.perf_counter()
    result = generate_dataset(
        config,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        extract_features_from_images=not args.skip_features,
    )
    elapsed = time.perf_counter() - started
    summary = result.summary
    counts = summary["counts"]
    print(
        f"generated {counts['sources']} sources and {counts['samples']} samples "
        f"({counts['feature_rows']} feature rows) in {elapsed:.2f} s"
    )
    print(f"dataset: {result.root}")
    print(f"manifest: {result.root / 'manifest.csv'}")
    print(f"features: {result.root / 'features.csv'}")
    print(f"generation_config_hash: {summary['generation_config_hash']}")
    print(f"disk_usage_bytes: {counts['disk_usage_bytes']}")
    print(f"splits: {summary['splits']}")
    return 0


def _command_verify(args: argparse.Namespace) -> int:
    report = verify_dataset(args.dataset, recompute_targets=not args.skip_targets)
    _print_report(report)
    return 0 if report.ok else 1


def _command_features(args: argparse.Namespace) -> int:
    features = write_feature_table(args.dataset)
    print(f"rewrote {Path(args.dataset) / 'features.csv'} ({len(features)} rows)")
    return 0


def _command_diagnose(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    output_dir = args.output_dir or _default_report_dir(config)
    result = diagnose_dataset(args.dataset, config.diagnostics, output_dir=output_dir)
    for check in result.leakage_checks:
        print(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    print(
        f"best single feature: {result.best_single_feature} "
        f"(val R2={result.best_single_val_r2:.3f}, test R2={result.best_single_test_r2:.3f})"
    )
    print(f"verdict: {result.verdict}")
    print(f"report: {result.report_path}")
    return 0 if result.leakage_safe else 1


def _command_compare(args: argparse.Namespace) -> int:
    report = compare_datasets(args.a, args.b)
    _print_report(report)
    return 0 if report.ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dataset CLI; returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    commands = {
        "generate": _command_generate,
        "verify": _command_verify,
        "features": _command_features,
        "diagnose": _command_diagnose,
        "compare": _command_compare,
    }
    try:
        return commands[args.command](args)
    except _EXPECTED_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover - module invocation
    raise SystemExit(main())
