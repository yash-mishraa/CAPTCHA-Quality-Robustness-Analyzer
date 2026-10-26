"""Research Overview: the question, the headline result, and the main failure."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app import content
from app.lib import analysis, charts, data, ui
from app.lib import format as fmt
from app.lib.contracts import ImageIntegrityError, IntegrityError


def build_context() -> dict[str, Any]:
    """Assemble the overview from frozen tables without recomputing metrics."""

    core = data.load_core()
    overall = data.load_report_table("m8", "overall_metrics.csv")
    uncertainty = data.load_report_table("m8", "uncertainty.csv")
    family = data.load_report_table("m8", "family_metrics.csv")
    composition = data.load_report_table("m8", "dataset_composition.csv")
    severity = data.load_report_table("m9", "severity_metrics.csv")

    overall_row = data.require_scope_row(overall, "all_primary")
    intervals: dict[str, tuple[float, float]] = {}
    for metric in ("mae", "rmse", "r2"):
        matches = uncertainty[
            (uncertainty["scheme"] == "source_grouped")
            & (uncertainty["scope"] == "all_primary")
            & (uncertainty["metric"] == metric)
        ]
        if len(matches) != 1:
            raise IntegrityError(f"missing source-grouped interval for {metric}")
        row = matches.iloc[0]
        intervals[metric] = (float(row["ci_low"]), float(row["ci_high"]))

    family_primary = analysis.family_profile(family, uncertainty)

    specimen = data.find_record(core, sample_id=content.ILLUSTRATIVE_SAMPLE_ID)
    if specimen is None:
        raise IntegrityError("illustrative specimen is absent from the frozen manifest")

    thin_s4 = data.find_scope_row(severity, "thin_lines_s4")
    arc_s4 = data.find_scope_row(severity, "arc_strokes_s4")

    return {
        "core": core,
        "overall": overall_row,
        "intervals": intervals,
        "family": family_primary,
        "specimen": specimen,
        "thin_s4": thin_s4,
        "arc_s4": arc_s4,
        "counts": {
            "rows": data.composition_count(composition, "rows"),
            "primary_rows": data.composition_count(composition, "primary_rows"),
            "diagnostic_rows": data.composition_count(composition, "diagnostic_rows"),
            "fonts": data.composition_count(composition, "fonts"),
            "cases": data.composition_count(composition, "cases"),
            "sources": data.composition_count(composition, "sources"),
        },
    }


def _specimen_panel(context: dict[str, Any]) -> None:
    specimen: pd.Series = context["specimen"]
    sample_id = str(specimen["sample_id"])
    font = fmt.font_label(str(specimen["font_id"]))
    case = fmt.case_label(str(specimen["case_id"]), specimen["text_length"])
    family = fmt.family_label(str(specimen["variant_family"]))
    severity = fmt.severity_label(
        str(specimen["variant_family"]),
        specimen["severity_level"],
        specimen["amplitude_px"],
    )

    ui.subhead("Illustrative record")
    columns = st.columns(2, gap="large")
    with columns[0]:
        ui.specimen_label("Reference")
        try:
            reference = data.load_selected_image(sample_id, "reference")
            st.image(
                reference,
                caption=f"Clean source-specific reference. {font}, {case}.",
                width="stretch",
            )
        except ImageIntegrityError as error:
            ui.empty_state(str(error))
    with columns[1]:
        ui.specimen_label("Corrupted")
        try:
            corrupted = data.load_selected_image(sample_id, "image")
            st.image(
                corrupted,
                caption=f"{family}, {severity}. {font}, {case}.",
                width="stretch",
            )
        except ImageIntegrityError as error:
            ui.empty_state(str(error))

    ui.specimen_label("Record")
    ui.ledger(
        [
            {
                "label": "Target SSIM",
                "value": fmt.fmt(specimen["ssim_target"], 3),
                "note": "Against the clean reference",
            },
            {
                "label": "Prediction",
                "value": fmt.fmt(specimen["prediction"], 3),
                "note": "Frozen M5C champion, precomputed",
            },
            {
                "label": "Residual",
                "value": fmt.fmt_residual(specimen["residual"]),
                "note": "Target minus prediction",
                "tone": "negative",
            },
            {
                "label": "Absolute error",
                "value": fmt.fmt(specimen["abs_residual"], 3),
                "note": "Displayed for convenience",
            },
        ],
        columns=4,
    )

    st.caption(
        "Illustrative case, not a random sample. The image pair and values are from "
        f"the frozen M8C dataset and predictions table. Sample ID: `{sample_id}`."
    )
    st.caption(
        "Residual = target \u2212 prediction (recorded convention). Negative values "
        "mean the prediction exceeded the target, so the model overestimated image "
        "quality. Target is not human judgment."
    )
    ui.chips(
        [
            ("Font", font),
            ("Group", fmt.group_label(str(specimen["font_group"]))),
            ("Family", family),
            ("Severity", severity),
            ("Scope", fmt.scope_label(str(specimen["protocol_scope"]))),
            ("Case", str(specimen["case_id"])),
        ]
    )


def render() -> None:
    try:
        context = build_context()
    except IntegrityError:
        ui.integrity_failure()
        st.stop()
        return

    counts = context["counts"]
    overall = context["overall"]
    intervals = context["intervals"]

    ui.page_header(
        "Research explorer / v1.0.0",
        "A measured look at how controlled degradation changes synthetic CAPTCHA-style images.",
        "A frozen image-quality model was evaluated zero-shot on 6,300 rendered "
        "variants across ten typefaces. The target is SSIM against each source's "
        "clean reference. This is not a solver, a security test, or a "
        "human-readability study.",
    )
    ui.rule()

    _specimen_panel(context)

    ui.section("01", "Primary result")
    ui.lede(
        f"Primary benchmark \u00b7 {fmt.fmt_int(counts['primary_rows'])} rows "
        f"\u00b7 {fmt.fmt_int(counts['sources'])} source groups"
    )
    ui.ledger(
        [
            {
                "label": "MAE",
                "value": fmt.fmt_metric("mae", overall["mae"]),
                "note": "95% source-grouped interval "
                + fmt.fmt_interval(*intervals["mae"], digits=4),
            },
            {
                "label": "RMSE",
                "value": fmt.fmt_metric("rmse", overall["rmse"]),
                "note": "95% source-grouped interval "
                + fmt.fmt_interval(*intervals["rmse"], digits=4),
            },
            {
                "label": "R\u00b2",
                "value": fmt.fmt_metric("r2", overall["r2"]),
                "note": "95% source-grouped interval "
                + fmt.fmt_interval(*intervals["r2"], digits=3),
            },
        ],
        columns=3,
    )
    st.caption(
        f"{fmt.fmt_int(counts['primary_rows'])} primary rows \u00b7 "
        f"{fmt.fmt_int(counts['sources'])} source groups \u00b7 "
        f"{fmt.fmt_int(counts['fonts'])} fonts \u00b7 "
        f"{fmt.fmt_int(counts['cases'])} cases"
    )
    ui.source_line(
        "Source: `overall_metrics.csv` (scope `all_primary`) and `uncertainty.csv` "
        "(scheme `source_grouped`). Intervals are descriptive percentile intervals, "
        "not significance tests or font-population estimates. "
        f"[M8C report]({content.M8_REPORT_URL})"
    )

    ui.section("02", "Where it fails")
    ui.lede(
        "Thin-line and arc-stroke interference are the weak point. The chart shows "
        "observed MAE and R\u00b2 by corruption family; negative R\u00b2 is not clipped."
    )
    chart_columns = st.columns([1.65, 1.0], gap="large")
    with chart_columns[0]:
        st.altair_chart(charts.family_mae_chart(context["family"]), use_container_width=True)
    with chart_columns[1]:
        st.altair_chart(charts.family_r2_chart(context["family"]), use_container_width=True)

    thin_s4 = context["thin_s4"]
    arc_s4 = context["arc_s4"]
    if thin_s4 is not None and arc_s4 is not None:
        ui.note(
            "At thin-line severity 4, the mean target is "
            f"{fmt.fmt(thin_s4['target_mean'], 3)} and the mean prediction is "
            f"{fmt.fmt(thin_s4['prediction_mean'], 3)}. At arc-stroke severity 4, the "
            f"means are {fmt.fmt(arc_s4['target_mean'], 3)} and "
            f"{fmt.fmt(arc_s4['prediction_mean'], 3)}. R\u00b2 is negative for both "
            "families: on these observed slices the model is worse than the mean-target "
            "predictor."
        )
    else:
        ui.note("Not reported for this scope.")
    ui.note(content.FAMILY_READING_NOTE)
    ui.source_line(
        "Source: `family_metrics.csv` and `uncertainty.csv` (source-grouped MAE and "
        "R\u00b2 intervals); severity means from M9A `severity_metrics.csv`. "
        "n = 1,200 rows for blur, noise, thin and arc; 900 for elastic; 300 source "
        f"groups each. [M8C report]({content.M8_REPORT_URL}) \u00b7 "
        f"[M9A report]({content.M9_REPORT_URL})"
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

    ui.section("03", "What this is not")
    ui.caveat_list("Non-claims", content.NON_CLAIMS)
    ui.lede(
        "This is a read-only presentation layer over frozen outputs. Predictions are "
        "precomputed; the model is not loaded or executed. Reference-based SSIM "
        "requires the known clean synthetic reference, so arbitrary upload scoring is "
        "not available."
    )

    with st.expander("Data, status & limitations"):
        for heading, body in content.LIMITATIONS:
            st.markdown(f"**{heading}.** {body}")
        st.markdown(
            "**Evidence links.** "
            f"[M8C benchmark report]({content.M8_REPORT_URL}) \u00b7 "
            f"[M9A exploratory report]({content.M9_REPORT_URL}) \u00b7 "
            f"[M7E licensing audit]({content.M7E_REPORT_URL}) \u00b7 "
            f"[research summary]({content.RESEARCH_SUMMARY_URL}) \u00b7 "
            f"[repository]({content.REPO_URL})"
        )

    ui.footer()
