"""Local Streamlit entrypoint for the CAPTCHA Quality & Robustness Analyzer."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app import content  # noqa: E402
from app.lib import contracts, theme  # noqa: E402
from app.views import benchmark, error_analysis, overview  # noqa: E402

st.set_page_config(
    page_title="CAPTCHA Quality & Robustness Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

theme.inject()

is_public_safe = contracts.is_public_safe_mode()

pages = [
    st.Page(
        overview.render,
        title="Research overview",
        url_path="overview",
    ),
    st.Page(
        benchmark.render,
        title="Benchmark explorer",
        url_path="benchmark",
    ),
    st.Page(
        error_analysis.render,
        title="Robustness & error analysis",
        url_path="error-analysis",
    ),
]


def _enter_overview() -> None:
    """Keep the bare root URL as a redirect to the explicit overview route."""

    st.switch_page(pages[0])


pages.append(st.Page(_enter_overview, title="Home", default=True))
navigation = st.navigation(pages, position="sidebar")

with st.sidebar:
    st.html(
        '<div class="sidebar-wordmark">CAPTCHA Quality &amp; Robustness '
        "Analyzer</div>"
        '<div class="sidebar-release">v1.0.0 \u00b7 research release</div>'
    )
    if is_public_safe:
        st.html(
            '<div class="sidebar-badge public-safe-badge">Results-Only Profile · Public-Safe</div>'
        )
    st.html('<div class="sidebar-rule"></div>')
    if is_public_safe:
        st.html(
            '<div class="sidebar-meta">'
            "<strong>Frozen benchmark results</strong><br>"
            "5,700 primary evaluation rows<br>"
            "10 fonts \u00b7 30 cases \u00b7 300 sources<br>"
            "Results-only exhibit"
            "</div>"
        )
    else:
        st.html(
            '<div class="sidebar-meta">'
            "<strong>Frozen benchmark</strong><br>"
            "6,300 synthetic images<br>"
            "10 fonts \u00b7 30 cases \u00b7 300 sources<br>"
            "Evaluation only"
            "</div>"
        )
    st.markdown(
        f"[Repository]({content.REPO_URL})  \n"
        f"[M8C benchmark]({content.M8_REPORT_URL})  \n"
        f"[M9A exploratory analysis]({content.M9_REPORT_URL})  \n"
        f"[M7E licensing audit]({content.M7E_REPORT_URL})",
        unsafe_allow_html=False,
    )
    if is_public_safe:
        st.html(
            '<div class="sidebar-note">'
            "Public-safe results-only build. Communicates verified v1.0.0 research "
            "metrics and error analysis with zero benchmark CAPTCHA images, proprietary "
            "font binaries, or model checkpoints."
            "</div>"
        )
    else:
        st.html(
            '<div class="sidebar-note">'
            "Local/private review build. Public deployment and hosted image "
            "distribution are blocked pending an explicit public-display and "
            "licensing decision."
            "</div>"
        )

navigation.run()
