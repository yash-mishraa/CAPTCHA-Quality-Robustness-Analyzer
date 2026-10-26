"""Local Streamlit entrypoint for the CAPTCHA Quality & Robustness Analyzer."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app import content  # noqa: E402
from app.lib import theme  # noqa: E402
from app.views import benchmark, error_analysis, overview  # noqa: E402

st.set_page_config(
    page_title="CAPTCHA Quality & Robustness Analyzer",
    layout="wide",
    initial_sidebar_state="auto",
)

theme.inject()

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
    st.html('<div class="sidebar-rule"></div>')
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
    st.html(
        '<div class="sidebar-note">'
        "Local/private review build. Public deployment and hosted image "
        "distribution are blocked pending an explicit public-display and "
        "licensing decision."
        "</div>"
    )

navigation.run()
