"""Pure transformations from published report tables into chart-ready frames.

These functions reshape recorded values only.  They do not estimate, smooth,
impute, or recompute a metric.
"""

from __future__ import annotations

import pandas as pd

from app.lib import contracts
from app.lib import format as fmt
from app.lib.contracts import IntegrityError

CONCENTRATION_SCOPES = ("gaussian_blur", "gaussian_noise", "thin_lines", "arc_strokes")
CALIBRATION_SCOPES = ("gaussian_blur", "gaussian_noise", "thin_lines", "arc_strokes")


def family_profile(family: pd.DataFrame, uncertainty: pd.DataFrame) -> pd.DataFrame:
    """Join published family metrics to their source-grouped MAE and R2 intervals."""

    mae_intervals = uncertainty[
        (uncertainty["scheme"] == "source_grouped") & (uncertainty["metric"] == "mae")
    ][["scope", "ci_low", "ci_high"]].rename(
        columns={"ci_low": "mae_ci_low", "ci_high": "mae_ci_high"}
    )
    r2_intervals = uncertainty[
        (uncertainty["scheme"] == "source_grouped") & (uncertainty["metric"] == "r2")
    ][["scope", "ci_low", "ci_high"]].rename(
        columns={"ci_low": "r2_ci_low", "ci_high": "r2_ci_high"}
    )

    primary = family[family["protocol_role"] == "primary"].copy()
    if primary.empty:
        raise IntegrityError("published primary family metrics are empty")
    if set(primary["variant_family"]) != set(fmt.PRIMARY_FAMILY_ORDER):
        raise IntegrityError("published primary family roster is incomplete")
    primary = primary.merge(mae_intervals, on="scope", how="left", validate="one_to_one")
    primary = primary.merge(r2_intervals, on="scope", how="left", validate="one_to_one")
    if len(primary) != len(fmt.PRIMARY_FAMILY_ORDER):
        raise IntegrityError("published family intervals are incomplete")
    primary["label"] = [fmt.family_label(value) for value in primary["variant_family"]]
    primary["order"] = [
        fmt.PRIMARY_FAMILY_ORDER.index(value) for value in primary["variant_family"]
    ]
    primary = primary.sort_values("order").reset_index(drop=True)

    if primary["mae_ci_low"].isna().any() or primary["r2_ci_low"].isna().any():
        raise IntegrityError("published family intervals are incomplete")
    return primary


def severity_long(severity: pd.DataFrame) -> pd.DataFrame:
    """Reshape M9A severity means into two series per family."""

    if severity.empty:
        raise IntegrityError("published severity metrics are empty")
    rows: list[dict[str, object]] = []
    for row in severity.itertuples(index=False):
        family = str(row.family)
        for series, value in (
            ("Target mean", row.target_mean),
            ("Prediction mean", row.prediction_mean),
        ):
            rows.append(
                {
                    "family": family,
                    "family_label": fmt.family_label(family),
                    "severity_level": int(row.severity_level),
                    "series": series,
                    "value": float(value),
                    "n_rows": int(row.n_rows),
                    "n_sources": int(row.n_sources),
                }
            )
    frame = pd.DataFrame(rows)
    order = {family: index for index, family in enumerate(fmt.PRIMARY_FAMILY_ORDER)}
    frame["order"] = frame["family"].map(order)
    return frame.sort_values(["order", "severity_level", "series"]).reset_index(drop=True)


def calibration(calibration_table: pd.DataFrame) -> pd.DataFrame:
    """Select the four primary family calibration summaries."""

    frame = calibration_table[calibration_table["scope"].isin(CALIBRATION_SCOPES)].copy()
    if len(frame) != len(CALIBRATION_SCOPES):
        raise IntegrityError("calibration table is missing a primary family")
    frame["label"] = [fmt.family_label(value) for value in frame["scope"]]
    frame["order"] = [CALIBRATION_SCOPES.index(value) for value in frame["scope"]]
    return frame.sort_values("order").reset_index(drop=True)


def concentration(concentration_table: pd.DataFrame) -> pd.DataFrame:
    """Reshape top 5/10/20% source SSE shares into a long frame."""

    frame = concentration_table[concentration_table["scope"].isin(CONCENTRATION_SCOPES)].copy()
    if len(frame) != len(CONCENTRATION_SCOPES):
        raise IntegrityError("concentration table is missing a primary family")
    rows: list[dict[str, object]] = []
    columns = (
        ("Top 5%", "top_5pct_source_sse_share"),
        ("Top 10%", "top_10pct_source_sse_share"),
        ("Top 20%", "top_20pct_source_sse_share"),
    )
    for row in frame.itertuples(index=False):
        for tier, column in columns:
            rows.append(
                {
                    "scope": str(row.scope),
                    "label": fmt.family_label(str(row.scope)),
                    "tier": tier,
                    "share": float(getattr(row, column)),
                    "n_sources": int(row.n_sources),
                }
            )
    long = pd.DataFrame(rows)
    order = {family: index for index, family in enumerate(CONCENTRATION_SCOPES)}
    long["order"] = long["scope"].map(order)
    return long.sort_values(["order", "tier"]).reset_index(drop=True)


def font_interference(font_table: pd.DataFrame) -> pd.DataFrame:
    """Select the ten descriptive font-by-interference rows."""

    frame = font_table[font_table["scope"] == "interference"].copy()
    if frame.empty:
        raise IntegrityError("interference font table is empty")
    if len(frame) != len(contracts.FONT_IDS):
        raise IntegrityError("interference font table does not contain the full roster")
    if set(frame["font_id"]) != set(contracts.FONT_IDS):
        raise IntegrityError("interference font table contains an unexpected font")
    frame["label"] = [fmt.font_label(value) for value in frame["font_id"]]
    frame["group_label"] = [fmt.group_label(value) for value in frame["font_group"]]
    frame = frame.sort_values("mae", ascending=True).reset_index(drop=True)
    if frame["mae_ci_low"].isna().any() or frame["mae_ci_high"].isna().any():
        raise IntegrityError("font interference intervals are incomplete")
    return frame
