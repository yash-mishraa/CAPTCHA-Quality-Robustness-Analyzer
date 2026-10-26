"""Static copy, links and claims for the local research dashboard.

All numeric values are read from frozen tables at runtime.  Only labels, short
editorial copy, and provenance links live here.
"""

from __future__ import annotations

REPO_URL = "https://github.com/yash-mishraa/CAPTCHA-Quality-Robustness-Analyzer"
RELEASE_TAG = "v1.0.0"


def blob_url(path: str) -> str:
    return f"{REPO_URL}/blob/{RELEASE_TAG}/{path}"


def tree_url(path: str) -> str:
    return f"{REPO_URL}/tree/{RELEASE_TAG}/{path}"


README_URL = blob_url("README.md")
RESEARCH_SUMMARY_URL = blob_url("docs/RESEARCH_SUMMARY.md")
M8_REPORT_URL = blob_url("reports/matched_font_benchmark_v0_1_0/report.md")
M9_REPORT_URL = blob_url("reports/interference_error_analysis_v0_1_0/README.md")
M7E_REPORT_URL = blob_url("reports/font_roster_audit_v0_2_0/report.md")
LICENSE_URL = blob_url("LICENSE")
MANIFEST_URL = blob_url("data/processed/matched_font_benchmark_v0.1.0/manifest.csv")

ILLUSTRATIVE_SAMPLE_ID = "src-4c20a455606b18eb__thin_lines_s4"

SSIM_EXPLANATION = (
    "SSIM here means similarity to the exact clean source image. The benchmark "
    "pairs every corrupted image with its known reference. An arbitrary upload "
    "would not have that reference, and the frozen model is not deployed in this "
    "app, so upload scoring is intentionally unavailable."
)

M9A_BANNER = (
    "M9A is an exploratory, descriptive analysis of the frozen M8C predictions. "
    "It does not test H1, H2, or H3, and it does not identify a causal mechanism."
)

GLOBAL_DISCLAIMERS = (
    "Synthetic benchmark. Not a CAPTCHA solver. Not a security evaluation. "
    "Not a human-readability study.",
    "The target is reference-based SSIM, not human judgment or CAPTCHA difficulty.",
    "Rows within a source are not independent; the source group is the independent unit.",
    "The ten fonts are a purposeful roster, not a sample of fonts.",
)

LICENSING_COPY = (
    "Rendered benchmark images are displayed under the M7E audit conditions for "
    "the intended evaluation workflow. Font binaries are not distributed. The ten "
    "typefaces are proprietary; their names identify the rendered material only. "
    "Public deployment requires explicit clearance. This is not a legal opinion."
)

LOCAL_BUILD_NOTE = (
    "Local/private review build. Public deployment is blocked pending an explicit "
    "public-display and licensing decision."
)

PUBLIC_SAFE_BUILD_NOTE = (
    "Results-only public-safe profile. Exposes verified research metrics, interactive charts, "
    "and error analysis with zero benchmark CAPTCHA images, proprietary font binaries, or model checkpoints."
)

PUBLIC_SAFE_SPECIMEN_NOTE = (
    "Results-only public-safe exhibit: Benchmark CAPTCHA images and proprietary font "
    "bitmaps are omitted to ensure strict copyright and licensing compliance. Full precomputed "
    "metrics and handcrafted features are shown below."
)

NON_CLAIMS = (
    "No human labels, solver data, or production CAPTCHA evidence.",
    "No causal mechanism for the interference error.",
    "No confirmatory test. M9B is unexecuted; H1, H2, and H3 remain not_tested.",
)

ERROR_ANALYSIS_NON_CLAIMS = (
    "M9A is exploratory and descriptive.",
    "It does not test H1, H2, or H3.",
    "M9B is a protocol, not a result.",
    "No causal mechanism is identified.",
    "No human readability, solvability, security, or production robustness is measured.",
    "Font differences are not a font-population estimate.",
    "Large-error proportions are exploratory and use a fixed 0.20 SSIM threshold.",
    "Associations are not feature-selection results.",
)

LIMITATIONS = (
    (
        "Scope",
        "Synthetic 192\u00d764 grayscale images from a fixed renderer: ten purposeful "
        "fonts, thirty cases, five primary corruption families, evaluation only.",
    ),
    (
        "Target",
        "Reference-based SSIM against each source's clean synthetic reference. PSNR is "
        "diagnostic only and is not a target.",
    ),
    (
        "Independent unit",
        "The source group, not the individual image. Rows within a source are not independent.",
    ),
    (
        "Diagnostics",
        "Clean and 0.5 px elastic records are excluded from primary metrics and are "
        "labelled wherever they appear.",
    ),
    (
        "Baseline caveat",
        "The recorded M3 familiar-font baseline is not like-for-like with M8C. It is "
        "not charted here as an improvement or decline story.",
    ),
    (
        "M9A",
        "Exploratory, descriptive, and not causal. Subgroup large-error proportions use "
        "a fixed 0.20 SSIM threshold.",
    ),
    (
        "M9B and hypotheses",
        "M9B is preregistered and unexecuted. H1, H2, and H3 remain not_tested.",
    ),
    (
        "Licensing",
        LICENSING_COPY,
    ),
    (
        "Data availability",
        "The tracked M8C dataset is present in Git, including all 6,300 PNGs. The "
        "untracked ground-truth text table is never read by this app.",
    ),
)

FAMILY_READING_NOTE = (
    "Blur and noise are the strongest observed slices. Thin lines and arcs have "
    "negative R\u00b2: predictions are worse than a mean-target baseline on those "
    "observed slices. The error is larger and biased in these benchmark slices; the "
    "chart does not establish a causal mechanism."
)

SEVERITY_READING_NOTE = (
    "Means of observed rows. Individual records vary; these are not model calibration curves."
)

CALIBRATION_READING_NOTE = (
    "At severe interference the model compresses its response: the predicted spread "
    "is much smaller than the target spread. This is descriptive, not a causal "
    "mechanism."
)

CONCENTRATION_READING_NOTE = (
    "Blur and noise error is concentrated in a relatively small number of sources. "
    "Interference error is broad across sources, not carried by a handful of "
    "outliers."
)

FONT_READING_NOTE = (
    "Font-level error differs descriptively. Glyph shape, stroke width, target "
    "difficulty, rasterisation, and source composition are entangled. This is not a "
    "causal font effect and not a recommendation."
)
