"""Warm-paper editorial theme for the local research dashboard.

The CSS is deliberately static.  It uses documented Streamlit test ids and
data-baseweb attributes, never generated ``st-emotion-cache-*`` names.  The
three interface fonts are bundled locally as Latin WOFF2 files and embedded as
data URIs, so the app makes no runtime font request.
"""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "assets"
FONT_DIR = ASSETS / "fonts"


def _font_face(family: str, filename: str, weight: str) -> str:
    path = FONT_DIR / filename
    try:
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return ""
    return (
        "@font-face{"
        f'font-family:"{family}";'
        f'src:url(data:font/woff2;base64,{payload}) format("woff2");'
        f"font-weight:{weight};font-style:normal;font-display:swap;"
        "}"
    )


@lru_cache(maxsize=1)
def build_css() -> str:
    faces = "".join(
        [
            _font_face("Source Serif 4", "SourceSerif4-Variable.woff2", "400 700"),
            _font_face("IBM Plex Sans", "IBMPlexSans-Variable.woff2", "400 700"),
            _font_face("IBM Plex Mono", "IBMPlexMono-400.woff2", "400"),
            _font_face("IBM Plex Mono", "IBMPlexMono-500.woff2", "500"),
            _font_face("IBM Plex Mono", "IBMPlexMono-600.woff2", "600"),
        ]
    )

    return f"""
<style>
{faces}

:root {{
  --paper: #F5F0E6;
  --paper-alt: #EBE3D5;
  --ink: #1B1B17;
  --ink-muted: #5E5A52;
  --rule: #CFC5B5;
  --rust: #A6402F;
  --teal: #1E5B57;
  --ochre: #8A5519;
  --blue: #3B6E8F;
  --umber: #6C5B3A;
  --olive: #4F6B4A;
  --gray: #6E6A61;
  --serif: "Source Serif 4", Georgia, "Times New Roman", serif;
  --sans: "IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", "SFMono-Regular", Consolas, monospace;
}}

html, body, .stApp, [data-testid="stAppViewContainer"] {{
  background: var(--paper);
  color: var(--ink);
}}
.stApp {{
  font-family: var(--sans);
  font-size: 17px;
  line-height: 1.6;
}}
[data-testid="stHeader"] {{
  background: transparent;
  z-index: 100;
}}
[data-testid="stAppDeployButton"],
[data-testid="stToolbarActions"],
[data-testid="stHeaderActionElements"],
#MainMenu,
[data-testid="stDecoration"],
footer {{
  display: none !important;
}}
[data-testid="stToolbar"] {{
  background: transparent !important;
}}
[data-testid="stExpandSidebarButton"] {{
  display: flex !important;
  visibility: visible !important;
  opacity: 1 !important;
  color: var(--ink-muted) !important;
}}
[data-testid="stExpandSidebarButton"]:hover {{
  color: var(--rust) !important;
}}
[data-testid="stAppViewContainer"] .main .block-container,
.block-container {{
  max-width: 1180px;
  padding: 3.1rem 2rem 4.5rem;
}}
[data-testid="stSidebar"] {{
  background: var(--paper-alt);
  border-right: 1px solid var(--rule);
}}
[data-testid="stSidebar"] .block-container {{
  padding: 2rem 1.25rem 2rem;
}}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] a,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .sidebar-wordmark,
[data-testid="stSidebar"] .sidebar-release,
[data-testid="stSidebar"] .sidebar-badge,
[data-testid="stSidebar"] .sidebar-meta,
[data-testid="stSidebar"] .sidebar-note {{
  font-family: var(--sans);
}}
[data-testid="stSidebarCollapseButton"] span,
[data-testid="stExpandSidebarButton"] span,
button[data-testid="stSidebarCollapseButton"],
button[data-testid="stExpandSidebarButton"] {{
  font-family: "Material Symbols Rounded" !important;
}}
[data-testid="stSidebarCollapseButton"] {{
  color: var(--ink-muted) !important;
}}
[data-testid="stSidebarCollapseButton"]:hover {{
  color: var(--rust) !important;
}}

h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {{
  font-family: var(--serif);
  font-weight: 600;
  color: var(--ink);
  letter-spacing: -0.01em;
}}
p, li, label, .stMarkdown {{
  font-family: var(--sans);
}}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {{
  line-height: 1.65;
}}
code, kbd, pre, samp, .mono {{
  font-family: var(--mono);
  font-variant-numeric: tabular-nums;
}}
a {{
  color: var(--teal);
  text-decoration-thickness: 1px;
  text-underline-offset: 2px;
}}
a:hover {{
  color: var(--rust);
}}
a:focus-visible, button:focus-visible, input:focus-visible,
[role="button"]:focus-visible, summary:focus-visible {{
  outline: 2px solid var(--rust);
  outline-offset: 2px;
}}

/* Page frame ------------------------------------------------------------- */
.page-head {{
  border-top: 1px solid var(--ink);
  padding-top: 1rem;
}}
.eyebrow {{
  font-family: var(--mono);
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--rust);
  margin: 0 0 0.35rem;
}}
.page-title {{
  font-family: var(--serif);
  font-size: clamp(2.4rem, 5vw, 4.25rem);
  font-weight: 600;
  line-height: 1.02;
  letter-spacing: -0.02em;
  margin: 0.25rem 0 1rem;
  max-width: 19ch;
  text-wrap: balance;
}}
.deck {{
  font-family: var(--sans);
  font-size: 1.05rem;
  line-height: 1.65;
  color: var(--ink);
  max-width: 72ch;
  margin: 0;
}}
.rule {{
  border: 0;
  border-top: 1px solid var(--rule);
  margin: 2.4rem 0;
}}
.rule-strong {{
  border: 0;
  border-top: 1px solid var(--ink);
  margin: 3rem 0 1.5rem;
}}
.section-head {{
  display: flex;
  align-items: baseline;
  gap: 0.9rem;
  border-top: 1px solid var(--ink);
  padding-top: 0.85rem;
  margin: 2.8rem 0 1.1rem;
}}
.section-index {{
  font-family: var(--mono);
  font-size: 0.7rem;
  color: var(--ink-muted);
  letter-spacing: 0.08em;
}}
.section-title {{
  font-family: var(--serif);
  font-size: clamp(1.55rem, 2.4vw, 2rem);
  font-weight: 600;
  margin: 0;
  line-height: 1.15;
}}
.subhead {{
  font-family: var(--mono);
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-muted);
  margin: 0 0 0.5rem;
}}
.lede {{
  font-size: 1rem;
  line-height: 1.65;
  max-width: 72ch;
  margin: 0 0 0.75rem;
}}
.micro {{
  font-size: 0.8rem;
  line-height: 1.45;
  color: var(--ink-muted);
}}
.source-line {{
  font-family: var(--sans);
  font-size: 0.76rem;
  line-height: 1.5;
  color: var(--ink-muted);
  border-top: 1px solid var(--rule);
  padding-top: 0.55rem;
  margin-top: 0.35rem;
}}
.source-line code {{
  font-size: 0.74rem;
}}

/* Ledger ----------------------------------------------------------------- */
.ledger {{
  display: grid;
  gap: 0;
  border-top: 1px solid var(--ink);
  border-bottom: 1px solid var(--rule);
  margin: 0.5rem 0 1rem;
}}
.ledger--3 {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
.ledger--4 {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
.ledger--1 {{ grid-template-columns: 1fr; }}
.ledger-item {{
  padding: 0.9rem 1.1rem 0.95rem 0;
  border-right: 1px solid var(--rule);
  min-width: 0;
}}
.ledger-item:last-child {{
  border-right: 0;
  padding-right: 0;
}}
.ledger--1 .ledger-item {{
  border-right: 0;
  border-bottom: 1px solid var(--rule);
  padding: 0.65rem 0;
}}
.ledger--1 .ledger-item:last-child {{
  border-bottom: 0;
}}
.ledger-label {{
  font-family: var(--mono);
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.11em;
  text-transform: uppercase;
  color: var(--ink-muted);
  margin-bottom: 0.25rem;
}}
.ledger-value {{
  font-family: var(--mono);
  font-size: clamp(1.65rem, 3.4vw, 2.65rem);
  font-weight: 500;
  line-height: 1.05;
  letter-spacing: -0.03em;
  font-variant-numeric: tabular-nums;
  color: var(--ink);
  white-space: nowrap;
}}
.ledger--1 .ledger-value {{
  font-size: clamp(1.25rem, 2.2vw, 1.65rem);
}}
.ledger-note {{
  font-family: var(--sans);
  font-size: 0.74rem;
  line-height: 1.4;
  color: var(--ink-muted);
  margin-top: 0.25rem;
}}
.ledger-value.is-negative {{ color: var(--rust); }}

/* Specimen plate --------------------------------------------------------- */
.specimen-label {{
  font-family: var(--mono);
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-muted);
  border-bottom: 1px solid var(--rule);
  padding-bottom: 0.35rem;
  margin-bottom: 0.5rem;
}}
.specimen-caption {{
  font-size: 0.78rem;
  line-height: 1.45;
  color: var(--ink-muted);
  margin-top: 0.1rem;
}}
.specimen-meta {{
  font-family: var(--mono);
  font-size: 0.72rem;
  color: var(--ink-muted);
  word-break: break-word;
}}
[data-testid="stImage"] img {{
  image-rendering: pixelated;
  image-rendering: crisp-edges;
  border: 1px solid var(--rule);
  border-radius: 0;
  background: #ffffff;
  width: 100%;
}}
[data-testid="stImage"] {{
  margin-bottom: 0.15rem;
}}
[data-testid="stCaptionContainer"] {{
  color: var(--ink-muted);
  font-size: 0.8rem;
  line-height: 1.45;
  text-align: left;
}}
[data-testid="stCaptionContainer"] p {{
  font-size: 0.8rem;
  line-height: 1.45;
}}

/* Notes, caveats, chips -------------------------------------------------- */
.note {{
  background: transparent;
  border-left: 2px solid var(--teal);
  padding: 0.7rem 0 0.7rem 1rem;
  margin: 0.9rem 0;
  font-size: 0.95rem;
  line-height: 1.6;
  max-width: 76ch;
}}
.note strong {{
  font-weight: 600;
}}
.caveat {{
  background: var(--paper-alt);
  border-left: 2px solid var(--rust);
  padding: 0.85rem 1rem;
  margin: 1rem 0;
  font-size: 0.92rem;
  line-height: 1.6;
}}
.caveat-title {{
  font-family: var(--mono);
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--rust);
  margin-bottom: 0.35rem;
}}
.exploratory {{
  background: var(--paper-alt);
  border-left: 2px solid var(--ochre);
  padding: 0.85rem 1rem;
  margin: 1rem 0;
  font-size: 0.92rem;
  line-height: 1.6;
}}
.exploratory .caveat-title {{ color: var(--ochre); }}
.chips {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
  margin: 0.5rem 0 0.75rem;
}}
.chip {{
  border: 1px solid var(--rule);
  color: var(--ink-muted);
  font-family: var(--mono);
  font-size: 0.7rem;
  line-height: 1.35;
  padding: 0.22rem 0.48rem;
  white-space: nowrap;
}}
.chip strong {{
  color: var(--ink);
  font-weight: 500;
}}
.tag-diagnostic {{
  display: inline-block;
  border: 1px solid var(--ochre);
  color: var(--ochre);
  font-family: var(--mono);
  font-size: 0.68rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 0.2rem 0.45rem;
  margin-bottom: 0.5rem;
}}
.empty-state {{
  border: 1px dashed var(--rule);
  padding: 1.25rem 1rem;
  color: var(--ink-muted);
  font-size: 0.92rem;
}}
.integrity-error {{
  border-top: 2px solid var(--rust);
  border-bottom: 1px solid var(--rule);
  padding: 1rem 0;
  color: var(--ink);
  font-size: 0.95rem;
}}

/* Sidebar ---------------------------------------------------------------- */
.sidebar-wordmark {{
  font-family: var(--serif);
  font-size: 1.55rem;
  font-weight: 600;
  line-height: 1.06;
  letter-spacing: -0.02em;
  margin: 0.25rem 0 0.4rem;
}}
.sidebar-release {{
  font-family: var(--mono);
  font-size: 0.68rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--rust);
  margin-bottom: 0.5rem;
}}
.sidebar-badge {{
  display: inline-block;
  font-family: var(--mono);
  font-size: 0.65rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 0.2rem 0.45rem;
  margin-bottom: 0.75rem;
  border-radius: 2px;
}}
.public-safe-badge {{
  background: #eef7f6;
  color: var(--teal);
  border: 1px solid var(--teal);
}}
.sidebar-rule {{
  border-top: 1px solid var(--rule);
  margin: 1rem 0;
}}
.sidebar-meta {{
  font-family: var(--sans);
  font-size: 0.8rem;
  line-height: 1.55;
  color: var(--ink-muted);
}}
.sidebar-meta strong {{
  color: var(--ink);
  font-weight: 500;
}}
.sidebar-links {{
  font-size: 0.82rem;
  line-height: 1.9;
  margin-top: 0.5rem;
}}
.sidebar-links a {{
  color: var(--teal);
}}
.sidebar-note {{
  font-size: 0.74rem;
  line-height: 1.5;
  color: var(--ink-muted);
  border-top: 1px solid var(--rule);
  padding-top: 0.7rem;
  margin-top: 1rem;
}}
[data-testid="stSidebarNav"] {{
  padding-top: 0.35rem;
}}
[data-testid="stSidebarNav"] li:has(a[href$="/"]) {{
  display: none;
}}
[data-testid="stSidebarNav"] a {{
  font-family: var(--sans);
  font-size: 0.88rem;
  color: var(--ink);
}}
[data-testid="stSidebarNav"] a span {{
  font-size: 0.88rem;
}}

/* Charts and tables ------------------------------------------------------ */
.legend {{
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  font-family: var(--sans);
  font-size: 0.78rem;
  color: var(--ink-muted);
  margin: 0.25rem 0 0.5rem;
}}
.legend-item {{
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
}}
.legend-dot {{
  width: 0.55rem;
  height: 0.55rem;
  border-radius: 50%;
  display: inline-block;
}}
.legend-line {{
  width: 1.1rem;
  border-top: 2px solid;
  display: inline-block;
}}
[data-testid="stDataFrame"] {{
  border: 1px solid var(--rule);
}}
[data-testid="stDataFrame"] [role="columnheader"] {{
  font-family: var(--sans);
  font-size: 0.7rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}}
[data-testid="stExpander"] details {{
  border: 1px solid var(--rule);
  border-radius: 0;
  background: transparent;
}}
[data-testid="stExpander"] summary {{
  font-family: var(--sans);
  font-size: 0.85rem;
  color: var(--ink);
}}
[data-testid="stAlert"] {{
  border-radius: 0;
  border: 1px solid var(--rule);
  background: var(--paper-alt);
}}

/* Controls --------------------------------------------------------------- */
[data-testid="stWidgetLabel"] p {{
  font-family: var(--mono);
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.11em;
  text-transform: uppercase;
  color: var(--ink-muted);
}}
[data-baseweb="select"] > div {{
  background: var(--paper);
  border: 1px solid var(--rule);
  border-radius: 0;
  min-height: 42px;
}}
[data-baseweb="select"] > div:hover {{
  border-color: var(--ink-muted);
}}
[data-baseweb="select"] span {{
  font-family: var(--sans);
  font-size: 0.9rem;
  color: var(--ink);
}}
[data-baseweb="popover"] li {{
  font-family: var(--sans);
  font-size: 0.88rem;
}}
[data-testid="stCheckbox"] label p {{
  font-family: var(--sans);
  font-size: 0.88rem;
  letter-spacing: 0;
  text-transform: none;
  color: var(--ink);
}}
[data-testid="stCheckbox"] {{
  margin-top: 0.35rem;
}}

/* Footer ----------------------------------------------------------------- */
.footer {{
  border-top: 1px solid var(--ink);
  margin-top: 3.5rem;
  padding-top: 1rem;
  color: var(--ink-muted);
  font-size: 0.82rem;
  line-height: 1.55;
}}
.footer p {{
  margin: 0 0 0.45rem;
  max-width: 78ch;
}}
.footer strong {{
  color: var(--ink);
  font-weight: 600;
}}
.footer .footer-links {{
  margin-top: 0.55rem;
}}
.footer a {{
  color: var(--teal);
}}
.nonclaim-list {{
  margin: 0.35rem 0 0;
  padding-left: 1rem;
}}
.nonclaim-list li {{
  margin-bottom: 0.2rem;
}}

@media (max-width: 900px) {{
  .block-container {{
    padding: 2.2rem 1.4rem 3.5rem;
  }}
  .page-title {{
    font-size: clamp(2.25rem, 7.5vw, 3.2rem);
  }}
  .ledger--3, .ledger--4 {{
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }}
  .ledger--3 .ledger-item:nth-child(2n),
  .ledger--4 .ledger-item:nth-child(2n) {{
    border-right: 0;
    padding-right: 0;
  }}
  .ledger--3 .ledger-item:nth-child(-n+2),
  .ledger--4 .ledger-item:nth-child(-n+2) {{
    border-bottom: 1px solid var(--rule);
  }}
}}

@media (max-width: 640px) {{
  .block-container {{
    padding: 1.5rem 1rem 2.75rem;
  }}
  .page-title {{
    font-size: 2.25rem;
  }}
  .deck {{
    font-size: 1rem;
  }}
  .section-head {{
    margin-top: 2.2rem;
  }}
  .ledger-value {{
    font-size: 1.6rem;
  }}
}}
</style>
"""


def inject() -> None:
    """Inject the single static theme block into the current page."""

    import streamlit as st

    st.html(build_css())
