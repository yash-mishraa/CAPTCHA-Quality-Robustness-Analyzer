"""Benchmark Explorer: inspect one real record and its provenance."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app import content
from app.lib import charts, data, ui
from app.lib import format as fmt
from app.lib.contracts import (
    DIAGNOSTIC_MESSAGE,
    ImageIntegrityError,
    IntegrityError,
)


def _filter_state(core: pd.DataFrame) -> dict[str, Any]:
    font_pairs = fmt.sort_font_options(core)
    font_ids = [font for font, _ in font_pairs]
    group_by_font = dict(font_pairs)

    case_order = (
        core[["case_id", "case_index"]]
        .drop_duplicates()
        .sort_values("case_index")["case_id"]
        .tolist()
    )

    include_diagnostics = bool(st.session_state.get("bench_diag", False))
    families = fmt.family_options(include_diagnostics)

    if st.session_state.get("bench_font") not in font_ids:
        st.session_state["bench_font"] = "arial" if "arial" in font_ids else font_ids[0]
    if st.session_state.get("bench_case") not in case_order:
        st.session_state["bench_case"] = "case-000" if "case-000" in case_order else case_order[0]
    if st.session_state.get("bench_family") not in families:
        st.session_state["bench_family"] = "gaussian_blur"

    font_id = st.session_state["bench_font"]
    case_id = st.session_state["bench_case"]

    lengths = (
        core[core["font_id"] == font_id]
        .drop_duplicates("case_id")
        .set_index("case_id")["text_length"]
        .to_dict()
    )

    columns = st.columns([1.2, 1.0, 1.0, 0.75], gap="large")
    with columns[0]:
        font_id = st.selectbox(
            "Font",
            font_ids,
            key="bench_font",
            format_func=lambda value: fmt.font_label_with_group(
                value, group_by_font.get(value, "")
            ),
        )
    with columns[1]:
        case_id = st.selectbox(
            "Case",
            case_order,
            key="bench_case",
            format_func=lambda value: fmt.case_label(value, lengths.get(value)),
        )
    with columns[2]:
        family = st.selectbox(
            "Family",
            families,
            key="bench_family",
            format_func=fmt.family_label,
        )
    sev_options = fmt.severity_options(
        core,
        font_id=font_id,
        case_id=case_id,
        family=family,
        include_diagnostics=include_diagnostics,
    )
    if not sev_options:
        with columns[3]:
            st.caption("Not reported for this selection.")
        severity = None
    else:
        if st.session_state.get("bench_severity") not in sev_options:
            st.session_state["bench_severity"] = fmt.default_severity(sev_options)
        with columns[3]:
            severity = st.selectbox(
                "Severity",
                sev_options,
                key="bench_severity",
                format_func=lambda value: fmt.severity_label(family, value),
            )

    include_diagnostics = st.checkbox(
        "Include diagnostic records",
        key="bench_diag",
        help="Clean and 0.5 px elastic records are diagnostic and excluded from primary metrics.",
    )
    st.caption(
        "Primary rows are the default. Diagnostic records are labelled and never mixed "
        "into primary aggregates."
    )

    return {
        "font_id": font_id,
        "case_id": case_id,
        "family": family,
        "severity": severity,
        "include_diagnostics": include_diagnostics,
    }


def _specimen(row: pd.Series) -> None:
    sample_id = str(row["sample_id"])
    font = fmt.font_label(str(row["font_id"]))
    case = fmt.case_label(str(row["case_id"]), row["text_length"])
    family = str(row["variant_family"])
    family_label = fmt.family_label(family)
    severity = fmt.severity_label(family, row["severity_level"], row["amplitude_px"])

    if family == "clean":
        columns = st.columns(2, gap="large")
        with columns[0]:
            ui.specimen_label("Clean reference (diagnostic)")
            try:
                reference = data.load_selected_image(sample_id, "reference")
                st.image(
                    reference,
                    caption=f"Clean reference. {font}, {case}.",
                    width="stretch",
                )
            except ImageIntegrityError as error:
                ui.empty_state(str(error))
        with columns[1]:
            ui.specimen_label("Corrupted")
            st.caption(
                "No corrupted variant is shown for a clean diagnostic row. The clean "
                "reference is shown once rather than duplicated."
            )
    else:
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
                    caption=f"{family_label}, {severity}. {font}, {case}.",
                    width="stretch",
                )
            except ImageIntegrityError as error:
                ui.empty_state(str(error))

    ui.specimen_label("Record")
    _record_ledger(row, columns=4)

    st.caption(
        "Residual = target \u2212 prediction (recorded convention). Negative values "
        "mean the prediction exceeded the target, so the model overestimated image "
        "quality. Target is not human judgment."
    )
    ui.chips(
        [
            ("Font", font),
            ("Group", fmt.group_label(str(row["font_group"]))),
            ("Family", family_label),
            ("Severity", severity),
            ("Scope", fmt.scope_label(str(row["protocol_scope"]))),
            ("Case", str(row["case_id"])),
        ]
    )
    ui.mono_id("Sample ID", sample_id)


def _record_ledger(row: pd.Series, columns: int = 4) -> None:
    ui.ledger(
        [
            {
                "label": "Target SSIM",
                "value": fmt.fmt(row["ssim_target"], 3),
                "note": "SSIM against the clean reference",
            },
            {
                "label": "Prediction",
                "value": fmt.fmt(row["prediction"], 3),
                "note": "Frozen M5C champion, precomputed",
            },
            {
                "label": "Residual",
                "value": fmt.fmt_residual(row["residual"]),
                "note": "Target minus prediction",
                "tone": "negative" if float(row["residual"]) < 0 else "",
            },
            {
                "label": "Absolute error",
                "value": fmt.fmt(row["abs_residual"], 3),
                "note": "Displayed for convenience",
            },
        ],
        columns=columns,
    )


def _feature_table(row: pd.Series) -> None:
    records = []
    for key in (
        "intensity_mean",
        "intensity_std",
        "contrast_rms",
        "ink_fraction",
        "laplacian_variance",
        "edge_density",
        "histogram_entropy",
        "noise_sigma_immerkaer",
    ):
        label, note = fmt.FEATURE_LABELS[key]
        records.append(
            {
                "Feature": label,
                "Value": fmt.fmt_feature(row[key]),
                "Plain-language note": note,
            }
        )
    st.dataframe(pd.DataFrame(records), hide_index=True, width="stretch")
    st.caption(
        "Values come from `features.csv`; the app does not recompute them. Features are "
        "computed on the degraded image only. The model did not receive font, severity, "
        "text, source, target, or PSNR."
    )


def _scatter(frame: pd.DataFrame, selected: pd.Series | None, diagnostic: bool) -> None:
    primary = frame[frame["protocol_scope"] == "primary"]
    st.altair_chart(
        charts.prediction_scatter(primary, None if diagnostic else selected),
        use_container_width=True,
    )
    if diagnostic:
        st.caption(
            "The selected record is diagnostic and is not highlighted in the primary-only scatter."
        )
    else:
        st.caption(
            "All primary rows. The highlighted point is the selected record. This is "
            "context, not a new metric."
        )
    with st.expander("View scatter context"):
        counts = (
            primary.groupby("variant_family")
            .size()
            .rename("Rows")
            .reset_index()
            .rename(columns={"variant_family": "Family"})
        )
        counts["Family"] = [fmt.family_label(value) for value in counts["Family"]]
        st.markdown("**Primary rows by family**")
        st.dataframe(counts, hide_index=True, width="stretch")
        if not diagnostic:
            st.markdown("**Selected record**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Sample ID": selected["sample_id"],
                            "Target SSIM": selected["ssim_target"],
                            "Prediction": selected["prediction"],
                            "Residual": selected["residual"],
                        }
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
    ui.source_line(
        "Source: `predictions.csv` joined one-to-one to `manifest.csv` and "
        "`features.csv`. Scatter: 5,700 primary rows; 300 source groups. "
        f"[M8C report]({content.M8_REPORT_URL})"
    )


def render() -> None:
    try:
        core = data.load_core()
    except IntegrityError:
        ui.integrity_failure()
        st.stop()
        return

    ui.page_header(
        "Benchmark explorer / M8C",
        "Inspect one case, not a leaderboard.",
        "Choose a font, case, corruption family, and severity to see the clean "
        "reference, the degraded image, the precomputed prediction, and the eight "
        "frozen image features. This is evidence and provenance, not a ranking.",
    )
    ui.rule()

    try:
        selection = _filter_state(core)
        row = data.find_record(
            core,
            font_id=selection["font_id"],
            case_id=selection["case_id"],
            variant_family=selection["family"],
            severity_level=selection["severity"],
        )
    except IntegrityError:
        ui.integrity_failure()
        st.stop()
        return

    if row is None:
        ui.empty_state()
        ui.footer()
        return

    diagnostic = str(row["protocol_scope"]) == "diagnostic"
    if diagnostic:
        ui.diagnostic_tag()
        st.caption(DIAGNOSTIC_MESSAGE)

    _specimen(row)

    ui.section("01", "What the model saw")
    _feature_table(row)

    ui.section("02", "Where this record sits")
    _scatter(core, row, diagnostic)

    ui.note(content.SSIM_EXPLANATION)

    with st.expander("Record provenance & integrity"):
        st.markdown(
            f"**Sample ID.** `{row['sample_id']}`  \n"
            f"**Source ID.** `{row['source_id']}`  \n"
            f"**Case.** `{row['case_id']}`  \n"
            f"**Font.** {fmt.font_label(str(row['font_id']))} "
            f"({fmt.group_label(str(row['font_group']))})  \n"
            f"**Family.** {fmt.family_label(str(row['variant_family']))}  \n"
            f"**Severity.** "
            f"{fmt.severity_label(str(row['variant_family']), row['severity_level'], row['amplitude_px'])}  \n"
            f"**Protocol scope.** {fmt.scope_label(str(row['protocol_scope']))}"
        )
        st.markdown(
            "The selected image and reference were path-checked inside the dataset "
            "root and verified against their manifest SHA-256 hashes before display. "
            "No raw text, font binary, or local path is shown. The app reads only "
            "allowlisted frozen artifacts."
        )
        st.markdown(
            f"[M8C benchmark report]({content.M8_REPORT_URL}) \u00b7 "
            f"[manifest on GitHub]({content.MANIFEST_URL}) \u00b7 "
            f"[M7E licensing audit]({content.M7E_REPORT_URL})"
        )

    ui.footer()
