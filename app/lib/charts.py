"""Altair chart specifications for the frozen benchmark.

Charts read their values from report tables and precomputed predictions.  They
do not recompute metrics, smooth lines, clip negative R-squared values, or use
dual axes.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

from app.lib import format as fmt

# The explorer deliberately plots all 5,700 primary rows. This is a small
# enough payload for a local research view; no server-side aggregation is used.
alt.data_transformers.disable_max_rows()

PAPER = "#F5F0E6"
INK = "#1B1B17"
INK_MUTED = "#5E5A52"
RULE = "#CFC5B5"
RUST = "#A6402F"
TEAL = "#1E5B57"
OCHRE = "#8A5519"
BLUE = "#3B6E8F"
UMBER = "#6C5B3A"
OLIVE = "#4F6B4A"
GRAY = "#6E6A61"

FAMILY_COLORS = {
    "gaussian_blur": BLUE,
    "gaussian_noise": UMBER,
    "thin_lines": RUST,
    "arc_strokes": OCHRE,
    "elastic": OLIVE,
    "clean": GRAY,
}

GROUP_COLORS = {
    "Familiar": TEAL,
    "M7B held-out": OCHRE,
    "New held-out": BLUE,
}

SERIES_COLORS = {"Target mean": TEAL, "Prediction mean": OCHRE}
TIER_COLORS = {"Top 5%": OCHRE, "Top 10%": TEAL, "Top 20%": UMBER}


def _configure(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure(background="transparent")
        .configure_view(stroke=None)
        .configure_axis(
            labelFont="IBM Plex Sans",
            titleFont="IBM Plex Sans",
            labelColor=INK,
            titleColor=INK_MUTED,
            gridColor=RULE,
            domainColor=RULE,
            tickColor=RULE,
            labelFontSize=10,
            titleFontSize=10,
            titlePadding=8,
            labelPadding=4,
        )
        .configure_legend(
            labelFont="IBM Plex Sans",
            titleFont="IBM Plex Sans",
            labelColor=INK,
            titleColor=INK_MUTED,
            labelFontSize=10,
            titleFontSize=10,
            orient="bottom",
            columns=3,
            symbolSize=70,
        )
        .configure_title(
            font="Source Serif 4",
            fontSize=13,
            color=INK,
            anchor="start",
        )
    )


def family_mae_chart(profile: pd.DataFrame) -> alt.Chart:
    """Horizontal MAE dot plot with published source-grouped intervals."""

    frame = profile.copy()
    labels = frame["label"].tolist()
    max_value = float(frame["mae_ci_high"].max())
    frame["label_text"] = [
        f"{fmt.fmt(mae, 3)} \u00b7 R\u00b2 {fmt.fmt(r2, 3)}"
        for mae, r2 in zip(frame["mae"], frame["r2"], strict=True)
    ]

    base = alt.Chart(frame)
    intervals = base.mark_rule(strokeWidth=1.2, opacity=0.8).encode(
        y=alt.Y("label:N", sort=list(reversed(labels)), title=None),
        x=alt.X(
            "mae_ci_low:Q",
            title="MAE (SSIM units, lower is better)",
            scale=alt.Scale(domain=[0, max_value * 1.7], zero=True),
            axis=alt.Axis(format=".2f"),
        ),
        x2="mae_ci_high:Q",
        color=alt.Color(
            "family:N",
            scale=alt.Scale(domain=list(FAMILY_COLORS), range=list(FAMILY_COLORS.values())),
            legend=None,
        ),
    )
    points = base.mark_point(filled=True, size=85, stroke=PAPER, strokeWidth=1.5).encode(
        y=alt.Y("label:N", sort=list(reversed(labels)), title=None),
        x=alt.X("mae:Q", scale=alt.Scale(domain=[0, max_value * 1.7], zero=True)),
        color=alt.Color(
            "family:N",
            scale=alt.Scale(domain=list(FAMILY_COLORS), range=list(FAMILY_COLORS.values())),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("label:N", title="Family"),
            alt.Tooltip("n_rows:Q", title="Rows", format=","),
            alt.Tooltip("mae:Q", title="MAE", format=".6f"),
            alt.Tooltip("mae_ci_low:Q", title="95% low", format=".6f"),
            alt.Tooltip("mae_ci_high:Q", title="95% high", format=".6f"),
            alt.Tooltip("r2:Q", title="R\u00b2", format=".6f"),
        ],
    )
    labels_layer = base.mark_text(
        align="left",
        dx=6,
        font="IBM Plex Sans",
        fontSize=9,
        color=INK,
    ).encode(
        y=alt.Y("label:N", sort=list(reversed(labels)), title=None),
        x=alt.X("mae_ci_high:Q", scale=alt.Scale(domain=[0, max_value * 1.7], zero=True)),
        text="label_text:N",
    )
    return _configure((intervals + points + labels_layer).properties(height=250))


def family_r2_chart(profile: pd.DataFrame) -> alt.Chart:
    """Companion R-squared strip that keeps negative values visible."""

    frame = profile.copy()
    labels = frame["label"].tolist()
    low = min(-0.75, float(frame["r2_ci_low"].min()) - 0.05)
    high = max(1.08, float(frame["r2_ci_high"].max()) + 0.05)
    frame["label_text"] = [fmt.fmt(value, 3) for value in frame["r2"]]

    base = alt.Chart(frame)
    zero = (
        alt.Chart(pd.DataFrame({"zero": [0.0]}))
        .mark_rule(color=INK, strokeDash=[3, 3], strokeWidth=1)
        .encode(x="zero:Q")
    )
    intervals = base.mark_rule(strokeWidth=1.2, opacity=0.8).encode(
        y=alt.Y("label:N", sort=list(reversed(labels)), title=None),
        x=alt.X(
            "r2_ci_low:Q",
            title="R\u00b2 (1 is best; negative is worse than mean predictor)",
            scale=alt.Scale(domain=[low, high], zero=False),
            axis=alt.Axis(format=".2f"),
        ),
        x2="r2_ci_high:Q",
        color=alt.Color(
            "family:N",
            scale=alt.Scale(domain=list(FAMILY_COLORS), range=list(FAMILY_COLORS.values())),
            legend=None,
        ),
    )
    points = base.mark_point(filled=True, size=85, stroke=PAPER, strokeWidth=1.5).encode(
        y=alt.Y("label:N", sort=list(reversed(labels)), title=None),
        x=alt.X("r2:Q", scale=alt.Scale(domain=[low, high], zero=False)),
        color=alt.Color(
            "family:N",
            scale=alt.Scale(domain=list(FAMILY_COLORS), range=list(FAMILY_COLORS.values())),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("label:N", title="Family"),
            alt.Tooltip("n_rows:Q", title="Rows", format=","),
            alt.Tooltip("r2:Q", title="R\u00b2", format=".6f"),
            alt.Tooltip("r2_status:N", title="Status"),
        ],
    )
    labels_layer = base.mark_text(
        align="left",
        dx=6,
        font="IBM Plex Sans",
        fontSize=9,
        color=INK,
    ).encode(
        y=alt.Y("label:N", sort=list(reversed(labels)), title=None),
        x=alt.X("r2_ci_high:Q", scale=alt.Scale(domain=[low, high], zero=False)),
        text="label_text:N",
    )
    return _configure((zero + intervals + points + labels_layer).properties(height=250))


def prediction_scatter(primary: pd.DataFrame, selected: pd.Series | None = None) -> alt.Chart:
    """All primary records, with the selected primary record ringed."""

    frame = primary[
        [
            "sample_id",
            "variant_family",
            "severity_level",
            "ssim_target",
            "prediction",
            "residual",
        ]
    ].copy()
    frame["family_label"] = [fmt.family_label(value) for value in frame["variant_family"]]
    frame["severity_text"] = [
        fmt.severity_label(family, severity)
        for family, severity in zip(frame["variant_family"], frame["severity_level"], strict=True)
    ]
    families = [fmt.family_label(value) for value in fmt.PRIMARY_FAMILY_ORDER]

    base = (
        alt.Chart(frame)
        .mark_circle(size=8, opacity=0.34)
        .encode(
            x=alt.X(
                "ssim_target:Q",
                title="Target SSIM (reference-based)",
                scale=alt.Scale(domain=[0, 1.02], zero=True),
            ),
            y=alt.Y(
                "prediction:Q",
                title="Predicted SSIM",
                scale=alt.Scale(domain=[0, 1.02], zero=True),
            ),
            color=alt.Color(
                "family_label:N",
                title="Corruption family",
                scale=alt.Scale(
                    domain=families, range=[FAMILY_COLORS[key] for key in fmt.PRIMARY_FAMILY_ORDER]
                ),
                legend=alt.Legend(orient="bottom", columns=3, title=None),
            ),
            tooltip=[
                alt.Tooltip("sample_id:N", title="Sample ID"),
                alt.Tooltip("family_label:N", title="Family"),
                alt.Tooltip("severity_text:N", title="Severity"),
                alt.Tooltip("ssim_target:Q", title="Target SSIM", format=".6f"),
                alt.Tooltip("prediction:Q", title="Prediction", format=".6f"),
                alt.Tooltip("residual:Q", title="Residual", format=".6f"),
            ],
        )
    )

    diagonal = (
        alt.Chart(pd.DataFrame({"value": [0.0, 1.0]}))
        .mark_line(color=INK_MUTED, strokeDash=[4, 4], strokeWidth=1)
        .encode(
            x=alt.X("value:Q", scale=alt.Scale(domain=[0, 1.02], zero=True)),
            y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1.02], zero=True)),
        )
    )

    layers: list[alt.Chart] = [base, diagonal]
    if selected is not None:
        point = pd.DataFrame(
            {
                "ssim_target": [float(selected["ssim_target"])],
                "prediction": [float(selected["prediction"])],
            }
        )
        ring = (
            alt.Chart(point)
            .mark_point(size=260, filled=False, stroke=RUST, strokeWidth=2)
            .encode(x="ssim_target:Q", y="prediction:Q")
        )
        label = (
            alt.Chart(point)
            .mark_text(
                text="Selected",
                align="left",
                dx=10,
                dy=-10,
                font="IBM Plex Sans",
                fontSize=10,
                color=RUST,
            )
            .encode(x="ssim_target:Q", y="prediction:Q")
        )
        layers.extend([ring, label])

    return _configure(alt.layer(*layers).properties(height=430).resolve_scale(color="shared"))


def severity_panel(
    long_frame: pd.DataFrame,
    family: str,
    *,
    show_y_title: bool = False,
) -> alt.Chart:
    """One small-multiple panel of target versus prediction means."""

    frame = long_frame[long_frame["family"] == family].copy()
    frame["severity_text"] = [f"s{int(value)}" for value in frame["severity_level"]]
    frame["value_text"] = [fmt.fmt(value, 3) for value in frame["value"]]
    order = frame["severity_text"].drop_duplicates().tolist()

    chart = (
        alt.Chart(frame)
        .mark_line(
            point=alt.OverlayMarkDef(size=45, filled=True, strokeWidth=0),
            strokeWidth=2,
        )
        .encode(
            x=alt.X("severity_text:N", sort=order, title="Severity", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "value:Q",
                title="Mean SSIM" if show_y_title else None,
                scale=alt.Scale(domain=[0, 1], zero=True),
                axis=alt.Axis(format=".2f"),
            ),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=list(SERIES_COLORS), range=list(SERIES_COLORS.values())),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("severity_text:N", title="Severity"),
                alt.Tooltip("value:Q", title="Mean SSIM", format=".6f"),
                alt.Tooltip("n_rows:Q", title="Rows", format=","),
            ],
        )
    )

    if family in {"thin_lines", "arc_strokes"}:
        last = frame["severity_text"].iloc[-1]
        annotations = frame[frame["severity_text"] == last].copy()
        text_layers = []
        for series, dy in (("Target mean", -10), ("Prediction mean", 14)):
            series_frame = annotations[annotations["series"] == series]
            text_layers.append(
                alt.Chart(series_frame)
                .mark_text(
                    align="right",
                    dx=-4,
                    dy=dy,
                    font="IBM Plex Sans",
                    fontSize=9,
                    color=INK,
                )
                .encode(
                    x=alt.X("severity_text:N", sort=order, title="Severity"),
                    y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1], zero=True)),
                    text="value_text:N",
                )
            )
        for text in text_layers:
            chart = chart + text

    return _configure(chart.properties(height=185))


def calibration_chart(
    frame: pd.DataFrame,
    metric: str,
    *,
    title: str,
    digits: int = 3,
) -> alt.Chart:
    """Horizontal dot plot with a reference line at 1.0."""

    data = frame.copy()
    labels = data["label"].tolist()
    values = data[metric].astype(float)
    high = max(1.25, float(values.max()) * 1.35)
    data["value_text"] = [fmt.fmt(value, digits) for value in values]

    reference = (
        alt.Chart(pd.DataFrame({"one": [1.0]}))
        .mark_rule(color=INK, strokeDash=[3, 3], strokeWidth=1)
        .encode(x="one:Q")
    )
    points = (
        alt.Chart(data)
        .mark_point(filled=True, size=90, color=TEAL, stroke=PAPER, strokeWidth=1.5)
        .encode(
            y=alt.Y("label:N", sort=list(reversed(labels)), title=None),
            x=alt.X(
                f"{metric}:Q",
                title=title,
                scale=alt.Scale(domain=[0, high], zero=True),
                axis=alt.Axis(format=".2f"),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Scope"),
                alt.Tooltip(f"{metric}:Q", title=title, format=".6f"),
                alt.Tooltip("n_rows:Q", title="Rows", format=","),
                alt.Tooltip("n_sources:Q", title="Sources", format=","),
            ],
        )
    )
    labels_layer = (
        alt.Chart(data)
        .mark_text(align="left", dx=6, font="IBM Plex Sans", fontSize=10, color=INK)
        .encode(
            y=alt.Y("label:N", sort=list(reversed(labels)), title=None),
            x=alt.X(f"{metric}:Q", scale=alt.Scale(domain=[0, high], zero=True)),
            text="value_text:N",
        )
    )
    return _configure((reference + points + labels_layer).properties(height=190))


def concentration_chart(frame: pd.DataFrame) -> alt.Chart:
    """Grouped horizontal bars for top 5/10/20% source SSE shares."""

    data = frame.copy()
    tiers = ["Top 5%", "Top 10%", "Top 20%"]
    labels = data["label"].tolist()
    maximum = float(data["share"].max())
    data["share_text"] = [fmt.fmt_pct(value, 2) for value in data["share"]]

    bars = (
        alt.Chart(data)
        .mark_bar(stroke=None)
        .encode(
            y=alt.Y("label:N", sort=list(reversed(labels)), title=None),
            yOffset=alt.YOffset("tier:N", sort=tiers),
            x=alt.X(
                "share:Q",
                title="Share of total source SSE",
                scale=alt.Scale(domain=[0, maximum * 1.35], zero=True),
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color(
                "tier:N",
                scale=alt.Scale(domain=tiers, range=[TIER_COLORS[t] for t in tiers]),
                legend=alt.Legend(title=None, orient="bottom", columns=3),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Family"),
                alt.Tooltip("tier:N", title="Source share"),
                alt.Tooltip("share:Q", title="SSE share", format=".6%"),
            ],
        )
    )
    labels_layer = (
        alt.Chart(data)
        .mark_text(align="left", dx=3, font="IBM Plex Sans", fontSize=9, color=INK)
        .encode(
            y=alt.Y("label:N", sort=list(reversed(labels)), title=None),
            yOffset=alt.YOffset("tier:N", sort=tiers),
            x=alt.X("share:Q", scale=alt.Scale(domain=[0, maximum * 1.35], zero=True)),
            text="share_text:N",
        )
    )
    return _configure((bars + labels_layer).properties(height=245))


def font_chart(frame: pd.DataFrame) -> alt.Chart:
    """Descriptive interference MAE by font with published intervals."""

    data = frame.copy().sort_values("mae", ascending=True).reset_index(drop=True)
    labels = data["label"].tolist()
    max_value = float(data["mae_ci_high"].max())
    data["label_text"] = [fmt.fmt(value, 4) for value in data["mae"]]
    groups = [fmt.group_label(value) for value in fmt.GROUP_ORDER]

    intervals = (
        alt.Chart(data)
        .mark_rule(strokeWidth=1.2, opacity=0.8)
        .encode(
            y=alt.Y("label:N", sort=list(reversed(labels)), title=None),
            x=alt.X(
                "mae_ci_low:Q",
                title="Interference MAE (SSIM units, lower is better)",
                scale=alt.Scale(domain=[0, max_value * 1.5], zero=True),
                axis=alt.Axis(format=".2f"),
            ),
            x2="mae_ci_high:Q",
            color=alt.Color(
                "group_label:N",
                scale=alt.Scale(domain=groups, range=[GROUP_COLORS[group] for group in groups]),
                legend=None,
            ),
        )
    )
    points = (
        alt.Chart(data)
        .mark_point(filled=True, size=75, stroke=PAPER, strokeWidth=1.5)
        .encode(
            y=alt.Y("label:N", sort=list(reversed(labels)), title=None),
            x=alt.X("mae:Q", scale=alt.Scale(domain=[0, max_value * 1.5], zero=True)),
            color=alt.Color(
                "group_label:N",
                title="Font group",
                scale=alt.Scale(domain=groups, range=[GROUP_COLORS[group] for group in groups]),
                legend=alt.Legend(orient="bottom", columns=3, title=None),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Font"),
                alt.Tooltip("group_label:N", title="Group"),
                alt.Tooltip("mae:Q", title="MAE", format=".6f"),
                alt.Tooltip("mae_ci_low:Q", title="95% low", format=".6f"),
                alt.Tooltip("mae_ci_high:Q", title="95% high", format=".6f"),
                alt.Tooltip("n_rows:Q", title="Rows", format=","),
            ],
        )
    )
    labels_layer = (
        alt.Chart(data)
        .mark_text(align="left", dx=6, font="IBM Plex Sans", fontSize=9, color=INK)
        .encode(
            y=alt.Y("label:N", sort=list(reversed(labels)), title=None),
            x=alt.X("mae_ci_high:Q", scale=alt.Scale(domain=[0, max_value * 1.5], zero=True)),
            text="label_text:N",
        )
    )
    return _configure((intervals + points + labels_layer).properties(height=330))
