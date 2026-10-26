"""Research Overview: the question, the headline result, and the main failure."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app import content
from app.lib import analysis, charts, contracts, data, ui
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

    if contracts.is_public_safe_mode():
        ui.subhead("Illustrative record exhibit (results-only)")
        ui.note(content.PUBLIC_SAFE_SPECIMEN_NOTE)
    else:
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
                "label": "Target SSIM (Actual)",
                "value": fmt.fmt(specimen["ssim_target"], 3),
                "note": "Actual clarity vs clean reference",
            },
            {
                "label": "Model Prediction",
                "value": fmt.fmt(specimen["prediction"], 3),
                "note": "Estimated visual clarity (precomputed)",
            },
            {
                "label": "Residual (Error)",
                "value": fmt.fmt_residual(specimen["residual"]),
                "note": "Target minus Prediction",
                "tone": "negative",
            },
            {
                "label": "Absolute Error",
                "value": fmt.fmt(specimen["abs_residual"], 3),
                "note": "Off by ~12.4 percentage points",
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
        "Research explorer · v1.0.0",
        "A measured look at how controlled degradation changes synthetic CAPTCHA-style images.",
        "A frozen image-quality model was evaluated zero-shot on 6,300 rendered "
        "variants across ten typefaces. The target is SSIM against each source's "
        "clean reference. This is not a solver, a security test, or a "
        "human-readability study.",
    )
    ui.rule()

    # Plain-English Executive Summary & Key Findings
    summary_data = content.PLAIN_ENGLISH_SUMMARY
    st.html(
        '<div class="executive-summary">'
        '<div class="eyebrow" style="color: var(--rust); margin-bottom: 0.35rem;">EXECUTIVE SUMMARY · PLAIN-ENGLISH OVERVIEW</div>'
        '<div style="font-family: var(--serif); font-size: 1.15rem; font-weight: 600; margin-bottom: 0.5rem; color: var(--ink);">'
        "What does this project do and what did it discover?"
        "</div>"
        f'<p style="font-size: 0.95rem; line-height: 1.6; margin-bottom: 0.85rem; color: var(--ink);">'
        f"{summary_data['what_is_this']}"
        "</p>"
        "</div>"
    )

    cols = st.columns(3, gap="medium")
    for col, (badge_cls, badge_text, title, body) in zip(
        cols, summary_data["takeaways"], strict=True
    ):
        with col:
            st.html(
                '<div class="takeaway-card">'
                f'<span class="takeaway-badge {badge_cls}">{badge_text}</span>'
                f'<div class="takeaway-title">{title}</div>'
                f'<div class="takeaway-body">{body}</div>'
                "</div>"
            )

    with st.expander("Primer: Understanding SSIM, MAE, and R² in Plain English"):
        st.markdown(
            "### 1. What is the SSIM Score?\n"
            "**SSIM (Structural Similarity Index)** is an industry-standard visual quality score ranging from **0.0 to 1.0**:\n"
            "- **1.0 = Pristine / Crisp:** The text image is completely clear and identical to the original.\n"
            "- **0.8 to 0.9 = Light Degradation:** Minor blur or light grain; text remains easily readable.\n"
            "- **Below 0.7 = Heavy Distortion:** Severe degradation or interference lines that challenge text recognition.\n\n"
            "### 2. What are the Headline Metrics?\n"
            "- **MAE (Mean Absolute Error — ~0.060):** The model's average prediction error. On a 0.0 to 1.0 scale, "
            "it is off by only **~6 percentage points** on average.\n"
            "- **R² (Accuracy Score — 0.733):** The model accounts for **73.3%** of visual quality variation across the primary benchmark.\n"
            "- **RMSE (Root Mean Squared Error — 0.089):** Similar to MAE, but penalizes rare large mistakes more heavily.\n\n"
            "### 3. How Does the Model Work?\n"
            "Instead of a complex black-box neural network, the model uses **8 handcrafted spatial measurements** "
            "(such as edge density, stroke area, and local contrast) to test how traditional computer vision math holds up against adversarial corruptions."
        )

    ui.rule()

    _specimen_panel(context)

    ui.section("01", "Primary benchmark results")
    ui.lede(
        f"Overall evaluation across {fmt.fmt_int(counts['primary_rows'])} primary test images "
        f"and {fmt.fmt_int(counts['sources'])} source text groups."
    )
    ui.ledger(
        [
            {
                "label": "MAE (Avg Error)",
                "value": fmt.fmt_metric("mae", overall["mae"]),
                "note": "Average error: off by only ~0.06 on a 0–1 scale ("
                + fmt.fmt_interval(*intervals["mae"], digits=4)
                + ")",
            },
            {
                "label": "R² (Accuracy)",
                "value": fmt.fmt_metric("r2", overall["r2"]),
                "note": "Explains 73.3% of quality variation ("
                + fmt.fmt_interval(*intervals["r2"], digits=3)
                + ")",
            },
            {
                "label": "RMSE",
                "value": fmt.fmt_metric("rmse", overall["rmse"]),
                "note": "Penalizes large errors ("
                + fmt.fmt_interval(*intervals["rmse"], digits=4)
                + ")",
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
        f"[Final Benchmark Report (Phase 8)]({content.M8_REPORT_URL})"
    )

    ui.section("02", "Where the model fails: Thin lines and arc strokes")
    ui.lede(
        "While standard blur and noise are easily predicted, line-cut interference exposes a major blindspot. "
        "The charts below compare observed MAE and R\u00b2 by corruption family; negative R\u00b2 is not clipped."
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
            "Why the model fails on line cuts: At severity 4, thin lines reduce actual clarity to "
            f"{fmt.fmt(thin_s4['target_mean'], 3)}, but the model overestimates clarity at "
            f"{fmt.fmt(thin_s4['prediction_mean'], 3)}. Similarly for arc strokes, actual clarity is "
            f"{fmt.fmt(arc_s4['target_mean'], 3)} while the model predicts {fmt.fmt(arc_s4['prediction_mean'], 3)}. "
            "Because R\u00b2 is negative on these slices, the model performs worse than simply guessing the average score."
        )
    else:
        ui.note("Not reported for this scope.")
    ui.note(content.FAMILY_READING_NOTE)
    ui.source_line(
        "Source: `family_metrics.csv` and `uncertainty.csv` (source-grouped MAE and "
        "R\u00b2 intervals); severity means from `severity_metrics.csv`. "
        "n = 1,200 rows for blur, noise, thin and arc; 900 for elastic; 300 source "
        f"groups each. [Final Benchmark Report (Phase 8)]({content.M8_REPORT_URL}) \u00b7 "
        f"[Error & Robustness Report (Phase 9)]({content.M9_REPORT_URL})"
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

    ui.section("03", "Research scope & key non-claims")
    ui.caveat_list("Key Disclaimers", content.NON_CLAIMS)
    ui.lede(
        "This dashboard is a read-only presentation layer over frozen research outputs. "
        "Predictions are precomputed; the model is not loaded or executed live. "
        "Reference-based SSIM requires the known clean synthetic reference, so arbitrary upload "
        "scoring is intentionally unavailable."
    )

    with st.expander("Data, status & limitations"):
        for heading, body in content.LIMITATIONS:
            st.markdown(f"**{heading}.** {body}")
        st.markdown(
            "**Research Documentation & Evidence Links.**  \n"
            f"- [Final Benchmark Report (Phase 8 M8C)]({content.M8_REPORT_URL})  \n"
            f"- [Robustness & Error Study (Phase 9 M9A)]({content.M9_REPORT_URL})  \n"
            f"- [Font Asset & Licensing Audit (Phase 7 M7E)]({content.M7E_REPORT_URL})  \n"
            f"- [Full Research Summary & Methodology]({content.RESEARCH_SUMMARY_URL})  \n"
            f"- [Project Source Code (GitHub)]({content.REPO_URL})"
        )

    ui.footer()
