"""Robustness & Error Analysis: an exploratory, non-causal drilldown."""

from __future__ import annotations

from typing import Any

import streamlit as st

from app import content
from app.lib import analysis, charts, data, ui
from app.lib import format as fmt
from app.lib.contracts import IntegrityError


def build_context() -> dict[str, Any]:
    """Assemble M9A exploratory tables and the M8C family profile."""

    family = data.load_report_table("m8", "family_metrics.csv")
    uncertainty = data.load_report_table("m8", "uncertainty.csv")
    severity = data.load_report_table("m9", "severity_metrics.csv")
    calibration = data.load_report_table("m9", "condition_calibration.csv")
    concentration = data.load_report_table("m9", "error_concentration.csv")
    font_table = data.load_report_table("m9", "font_condition_metrics.csv")
    summary = data.load_report_table("m9", "summary_metrics.csv")

    interference = data.require_scope_row(summary, "interference")
    profile = analysis.family_profile(family, uncertainty)
    severity_long = analysis.severity_long(severity)
    calibration_frame = analysis.calibration(calibration)
    concentration_frame = analysis.concentration(concentration)
    font_frame = analysis.font_interference(font_table)

    return {
        "summary": summary,
        "interference": interference,
        "family": profile,
        "severity": severity,
        "severity_long": severity_long,
        "calibration": calibration_frame,
        "concentration": concentration_frame,
        "fonts": font_frame,
    }


def _family_section(context: dict[str, Any]) -> None:
    ui.section("01", "Family error profile")
    ui.lede(
        "Negative R\u00b2 is displayed, not clipped. The intervals are source-grouped "
        "percentile intervals, not significance tests."
    )
    columns = st.columns([1.6, 1.0], gap="large")
    with columns[0]:
        st.altair_chart(charts.family_mae_chart(context["family"]), use_container_width=True)
    with columns[1]:
        st.altair_chart(charts.family_r2_chart(context["family"]), use_container_width=True)
    ui.note(content.FAMILY_READING_NOTE)
    ui.source_line(
        "Source: M8C `family_metrics.csv` and `uncertainty.csv` (source-grouped MAE "
        "and R\u00b2 intervals). n = 1,200 rows for blur, noise, thin and arc; 900 "
        "for elastic; 300 source groups each. "
        f"[M8C report]({content.M8_REPORT_URL})"
    )
    with st.expander("View family data"):
        ui.data_table(
            context["family"][
                [
                    "label",
                    "n_rows",
                    "n_sources",
                    "mae",
                    "mae_ci_low",
                    "mae_ci_high",
                    "rmse",
                    "r2",
                    "r2_ci_low",
                    "r2_ci_high",
                ]
            ].rename(
                columns={
                    "label": "Family",
                    "n_rows": "Rows",
                    "n_sources": "Sources",
                    "mae": "MAE",
                    "mae_ci_low": "MAE 95% low",
                    "mae_ci_high": "MAE 95% high",
                    "rmse": "RMSE",
                    "r2": "R\u00b2",
                    "r2_ci_low": "R\u00b2 95% low",
                    "r2_ci_high": "R\u00b2 95% high",
                }
            )
        )


def _severity_section(context: dict[str, Any]) -> None:
    ui.section("02", "Severity paths")
    ui.lede(
        "Target and prediction means by corruption severity. The lines connect "
        "recorded means at ordinal severities; they are not fitted response curves."
    )
    ui.legend(
        [
            ("Target mean", charts.TEAL, "line"),
            ("Prediction mean", charts.OCHRE, "line"),
        ]
    )
    long_frame = context["severity_long"]
    columns = st.columns(2, gap="large")
    for index, family in enumerate(fmt.PRIMARY_FAMILY_ORDER):
        panel = long_frame[long_frame["family"] == family]
        if panel.empty:
            continue
        with columns[index % 2]:
            ui.subhead(fmt.family_label(family))
            st.caption(
                f"{fmt.fmt_int(panel['severity_level'].nunique())} severities \u00b7 "
                f"{fmt.fmt_int(panel['n_rows'].max())} rows each \u00b7 "
                f"{fmt.fmt_int(panel['n_sources'].max())} sources"
            )
            st.altair_chart(
                charts.severity_panel(long_frame, family, show_y_title=(index % 2 == 0)),
                use_container_width=True,
            )
    ui.note(
        "Elastic begins at s2 in the primary protocol. Thin-line and arc-stroke s4 "
        "are annotated with the recorded means. Means describe observed rows; "
        "individual records vary."
    )
    severity_rows = context["severity"]
    ui.source_line(
        f"Source: M9A `severity_metrics.csv`; {fmt.fmt_int(len(severity_rows))} "
        "published scope rows across five primary families, with "
        f"{fmt.fmt_int(int(severity_rows['n_sources'].max()))} source groups per "
        f"primary severity. [M9A report]({content.M9_REPORT_URL})"
    )
    with st.expander("View severity data"):
        ui.data_table(
            context["severity"][
                [
                    "family",
                    "severity_level",
                    "scope",
                    "n_rows",
                    "n_sources",
                    "target_mean",
                    "prediction_mean",
                    "mae",
                    "large_error_proportion",
                ]
            ].rename(
                columns={
                    "family": "Family",
                    "severity_level": "Severity",
                    "scope": "Scope",
                    "n_rows": "Rows",
                    "n_sources": "Sources",
                    "target_mean": "Target mean",
                    "prediction_mean": "Prediction mean",
                    "mae": "MAE",
                    "large_error_proportion": "Large-error proportion",
                }
            )
        )


def _calibration_section(context: dict[str, Any]) -> None:
    ui.section("03", "Calibration attenuation")
    ui.lede(
        "Within-source spread and slope compare the model's response with the "
        "recorded target. A value of 1.0 means matched response; lower values mean "
        "the prediction compresses the target range."
    )
    columns = st.columns(2, gap="large")
    with columns[0]:
        ui.subhead("Prediction / target SD ratio")
        st.altair_chart(
            charts.calibration_chart(
                context["calibration"],
                "within_source_prediction_target_sd_ratio_median",
                title="Within-source SD ratio (1.0 = matched spread)",
            ),
            use_container_width=True,
        )
    with columns[1]:
        ui.subhead("Within-source median slope")
        st.altair_chart(
            charts.calibration_chart(
                context["calibration"],
                "within_source_calibration_slope_median",
                title="Within-source median slope (1.0 = matched response)",
            ),
            use_container_width=True,
        )
    ui.note(content.CALIBRATION_READING_NOTE)
    ui.source_line(
        "Source: M9A `condition_calibration.csv`; within-source medians for four "
        "primary families, 1,200 rows and 300 source groups each. Aggregate ratios "
        "and slopes are shown in the table, not charted. "
        f"[M9A report]({content.M9_REPORT_URL})"
    )
    with st.expander("View calibration data"):
        ui.data_table(
            context["calibration"][
                [
                    "label",
                    "n_rows",
                    "n_sources",
                    "prediction_target_sd_ratio",
                    "within_source_prediction_target_sd_ratio_median",
                    "calibration_slope",
                    "within_source_calibration_slope_median",
                ]
            ].rename(
                columns={
                    "label": "Family",
                    "n_rows": "Rows",
                    "n_sources": "Sources",
                    "prediction_target_sd_ratio": "Aggregate SD ratio",
                    "within_source_prediction_target_sd_ratio_median": "Within-source SD ratio (median)",
                    "calibration_slope": "Aggregate slope",
                    "within_source_calibration_slope_median": "Within-source slope (median)",
                }
            )
        )


def _concentration_section(context: dict[str, Any]) -> None:
    ui.section("04", "Source concentration")
    ui.lede(
        "How much of a family's total squared error comes from the top 5, 10, and "
        "20 percent of its 300 sources? Each family sums to 100 percent across all "
        "sources."
    )
    st.altair_chart(charts.concentration_chart(context["concentration"]), use_container_width=True)
    ui.note(content.CONCENTRATION_READING_NOTE)
    summary = context["summary"]
    interference_top5 = data.require_scope_row(summary, "interference")["top_5pct_source_sse_share"]
    blur_top5 = data.require_scope_row(summary, "gaussian_blur")["top_5pct_source_sse_share"]
    noise_top5 = data.require_scope_row(summary, "gaussian_noise")["top_5pct_source_sse_share"]
    ui.source_line(
        "Source: M9A `error_concentration.csv`; top source shares by sum of squared "
        "errors, 300 sources per family. Interference top-5% share is "
        f"{fmt.fmt_pct(interference_top5)}, compared with "
        f"{fmt.fmt_pct(blur_top5)} for blur and {fmt.fmt_pct(noise_top5)} for noise. "
        f"[M9A report]({content.M9_REPORT_URL})"
    )
    with st.expander("View concentration data"):
        pivot = (
            context["concentration"]
            .pivot(
                index=["scope", "label", "n_sources"],
                columns="tier",
                values="share",
            )
            .reset_index()
        )
        pivot.columns.name = None
        ui.data_table(
            pivot.rename(
                columns={
                    "scope": "Scope",
                    "label": "Family",
                    "n_sources": "Sources",
                    "Top 5%": "Top 5% SSE share",
                    "Top 10%": "Top 10% SSE share",
                    "Top 20%": "Top 20% SSE share",
                }
            )
        )


def _font_section(context: dict[str, Any]) -> None:
    ui.section("05", "Font differences")
    ui.lede(
        "Descriptive interference MAE by font, ordered by observed MAE. This is not "
        "a font-population estimate, a causal effect, or a recommendation."
    )
    st.altair_chart(charts.font_chart(context["fonts"]), use_container_width=True)
    ui.note(content.FONT_READING_NOTE)
    fonts = context["fonts"]
    low_font = fonts.iloc[0]
    high_font = fonts.iloc[-1]
    st.caption(
        f"Observed interference MAE range: {low_font['label']} "
        f"{fmt.fmt(low_font['mae'], 4)} to {high_font['label']} "
        f"{fmt.fmt(high_font['mae'], 4)} across the ten purposeful fonts. This "
        "ordering is descriptive."
    )
    ui.source_line(
        "Source: M9A `font_condition_metrics.csv`, scope `interference`; 10 fonts, "
        "240 rows per font, 30 source groups per font. Intervals are published "
        "font-level intervals. "
        f"[M9A report]({content.M9_REPORT_URL})"
    )
    with st.expander("View font data"):
        ui.data_table(
            context["fonts"][
                [
                    "label",
                    "group_label",
                    "n_rows",
                    "n_sources",
                    "mae",
                    "mae_ci_low",
                    "mae_ci_high",
                ]
            ].rename(
                columns={
                    "label": "Font",
                    "group_label": "Group",
                    "n_rows": "Rows",
                    "n_sources": "Sources",
                    "mae": "MAE",
                    "mae_ci_low": "MAE 95% low",
                    "mae_ci_high": "MAE 95% high",
                }
            )
        )


def render() -> None:
    try:
        context = build_context()
    except IntegrityError:
        ui.integrity_failure()
        st.stop()
        return

    ui.page_header(
        "Robustness & error analysis / M9A exploratory",
        "Error is not uniform.",
        "A descriptive look at where the frozen model's error concentrates across "
        "corruption family, severity, source, and font. M9A is exploratory; it does "
        "not test H1, H2, or H3.",
    )
    ui.exploratory("Exploratory analysis \u00b7 M9A", content.M9A_BANNER)

    interference = context["interference"]
    ui.ledger(
        [
            {
                "label": "Interference MAE",
                "value": fmt.fmt_metric("mae", interference["mae"]),
                "note": "Exploratory slice, not a new primary result",
            },
            {
                "label": "Interference R\u00b2",
                "value": fmt.fmt_metric("r2", interference["r2"]),
                "note": "Negative: worse than mean-target baseline",
                "tone": "negative" if float(interference["r2"]) < 0 else "",
            },
            {
                "label": "Large-error rate",
                "value": fmt.fmt_pct(interference["large_error_proportion"]),
                "note": "Fixed 0.20 SSIM threshold",
            },
            {
                "label": "Top-5% SSE share",
                "value": fmt.fmt_pct(interference["top_5pct_source_sse_share"]),
                "note": "Across 300 sources",
            },
        ],
        columns=4,
    )
    st.caption(
        f"M9A interference slice \u00b7 {fmt.fmt_int(interference['n_rows'])} rows "
        f"\u00b7 {fmt.fmt_int(interference['n_sources'])} source groups \u00b7 "
        "exploratory"
    )
    ui.source_line(
        "Source: M9A `summary_metrics.csv`, scope `interference`; descriptive "
        "aggregates only. "
        f"[M9A report]({content.M9_REPORT_URL})"
    )

    _family_section(context)
    _severity_section(context)
    _calibration_section(context)
    _concentration_section(context)
    _font_section(context)

    ui.section("06", "What this does not show")
    ui.caveat_list("Non-claims", content.ERROR_ANALYSIS_NON_CLAIMS)
    with st.expander("Analysis status & links"):
        for heading, body in content.LIMITATIONS:
            st.markdown(f"**{heading}.** {body}")
        st.markdown(
            "**Evidence links.** "
            f"[M9A report]({content.M9_REPORT_URL}) \u00b7 "
            f"[M8C report]({content.M8_REPORT_URL}) \u00b7 "
            f"[M7E licensing audit]({content.M7E_REPORT_URL}) \u00b7 "
            f"[research summary]({content.RESEARCH_SUMMARY_URL})"
        )

    ui.footer()
