"""Small editorial UI components built from static HTML and native Streamlit."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from html import escape

import pandas as pd
import streamlit as st

from app import content
from app.lib import contracts


def page_header(eyebrow: str, title: str, deck: str) -> None:
    st.html(
        '<div class="page-head">'
        f'<div class="eyebrow">{escape(eyebrow)}</div>'
        f'<h1 class="page-title">{escape(title)}</h1>'
        f'<p class="deck">{escape(deck)}</p>'
        "</div>"
    )


def section(index: str, title: str) -> None:
    st.html(
        '<div class="section-head">'
        f'<span class="section-index">{escape(index)}</span>'
        f'<h2 class="section-title">{escape(title)}</h2>'
        "</div>"
    )


def subhead(text: str) -> None:
    st.html(f'<div class="subhead">{escape(text)}</div>')


def lede(text: str) -> None:
    st.html(f'<p class="lede">{escape(text)}</p>')


def rule(strong: bool = False) -> None:
    st.html(f'<hr class="{"rule-strong" if strong else "rule"}" />')


def ledger(items: Sequence[dict[str, str]], columns: int = 3) -> None:
    """Render an editorial ledger, not a row of stock metric cards."""

    rendered = []
    for item in items:
        label = escape(item.get("label", ""))
        value = escape(item.get("value", "Not reported"))
        note = escape(item.get("note", ""))
        tone = " is-negative" if item.get("tone") == "negative" else ""
        note_html = f'<div class="ledger-note">{note}</div>' if note else ""
        rendered.append(
            '<div class="ledger-item">'
            f'<div class="ledger-label">{label}</div>'
            f'<div class="ledger-value{tone}">{value}</div>'
            f"{note_html}"
            "</div>"
        )
    st.html(f'<div class="ledger ledger--{int(columns)}">{"".join(rendered)}</div>')


def chips(items: Iterable[tuple[str, str]]) -> None:
    rendered = []
    for label, value in items:
        rendered.append(
            f'<span class="chip"><strong>{escape(str(label))}</strong> {escape(str(value))}</span>'
        )
    st.html(f'<div class="chips">{"".join(rendered)}</div>')


def note(text: str) -> None:
    st.html(f'<div class="note">{escape(text)}</div>')


def caveat(title: str, body: str) -> None:
    st.html(
        '<div class="caveat">'
        f'<div class="caveat-title">{escape(title)}</div>'
        f"<p>{escape(body)}</p>"
        "</div>"
    )


def caveat_list(title: str, items: Iterable[str]) -> None:
    rendered = "".join(f"<li>{escape(item)}</li>" for item in items)
    st.html(
        '<div class="caveat">'
        f'<div class="caveat-title">{escape(title)}</div>'
        f'<ul class="nonclaim-list">{rendered}</ul>'
        "</div>"
    )


def exploratory(title: str, body: str) -> None:
    st.html(
        '<div class="exploratory">'
        f'<div class="caveat-title">{escape(title)}</div>'
        f"<p>{escape(body)}</p>"
        "</div>"
    )


def source_line(text: str) -> None:
    st.caption(text)


def specimen_label(text: str) -> None:
    st.html(f'<div class="specimen-label">{escape(text)}</div>')


def diagnostic_tag() -> None:
    st.html(f'<div class="tag-diagnostic">{escape(contracts.DIAGNOSTIC_MESSAGE)}</div>')


def empty_state(message: str = contracts.NO_MATCH_MESSAGE) -> None:
    st.html(f'<div class="empty-state">{escape(message)}</div>')


def integrity_failure() -> None:
    st.html(
        '<div class="integrity-error">'
        f"<p>{escape(contracts.DATA_ERROR_MESSAGE)}</p>"
        f'<p class="micro"><a href="{escape(content.README_URL)}">Repository data '
        "and integrity notes</a></p>"
        "</div>"
    )


def legend(items: Iterable[tuple[str, str, str]]) -> None:
    """Render a text legend: (label, color, kind) where kind is dot or line."""

    rendered = []
    for label, color, kind in items:
        marker = (
            f'<span class="legend-dot" style="background:{escape(color)}"></span>'
            if kind == "dot"
            else f'<span class="legend-line" style="border-color:{escape(color)}"></span>'
        )
        rendered.append(f'<span class="legend-item">{marker}{escape(label)}</span>')
    st.html(f'<div class="legend">{"".join(rendered)}</div>')


def mono_id(label: str, value: str) -> None:
    st.html(f'<div class="specimen-meta">{escape(label)} <code>{escape(value)}</code></div>')


def data_table(frame: pd.DataFrame, *, columns: Sequence[str] | None = None) -> None:
    if columns is not None:
        frame = frame.loc[:, list(columns)]
    st.dataframe(frame, hide_index=True, width="stretch")


def footer() -> None:
    disclaimers = "".join(f"<p>{escape(text)}</p>" for text in content.GLOBAL_DISCLAIMERS)
    st.html(
        '<div class="footer">'
        "<p><strong>Research presentation layer.</strong> All figures are read from "
        "frozen v1.0.0 artifacts. Predictions are precomputed; the model is not loaded "
        "or run.</p>"
        f"{disclaimers}"
        f"<p>{escape(content.LICENSING_COPY)}</p>"
        f"<p>{escape(content.LOCAL_BUILD_NOTE)}</p>"
        '<p class="footer-links">'
        f'<a href="{escape(content.README_URL)}">Repository</a> \u00b7 '
        f'<a href="{escape(content.M8_REPORT_URL)}">M8C report</a> \u00b7 '
        f'<a href="{escape(content.M9_REPORT_URL)}">M9A report</a> \u00b7 '
        f'<a href="{escape(content.M7E_REPORT_URL)}">Licensing audit</a> \u00b7 '
        f'<a href="{escape(content.RESEARCH_SUMMARY_URL)}">Research summary</a>'
        "</p>"
        "</div>"
    )
