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
    "SSIM (Structural Similarity Index) measures how visually close a degraded image is to "
    "its pristine, clean original on a 0.0 to 1.0 scale (1.0 = identical; lower = heavily distorted). "
    "Our model predicts this score without needing a human rater."
)

M9A_BANNER = (
    "This section provides an exploratory, descriptive breakdown of how model accuracy varies "
    "across fonts, distortion families, and degradation levels. It highlights empirical patterns in "
    "the benchmark rather than claiming a single biological or mathematical cause."
)

PLAIN_ENGLISH_SUMMARY = {
    "what_is_this": (
        "When text or CAPTCHAs are distorted (by blur, grain, or interference lines), how can a system "
        "automatically tell if the text is still legible without asking a human? This research evaluates "
        "whether automated computer vision models can accurately score image quality across 6,300 carefully "
        "degraded synthetic images covering ten typefaces and five corruption families."
    ),
    "what_is_ssim": (
        "SSIM (Structural Similarity) is a standard visual clarity score from 0.0 to 1.0. A score of 1.0 "
        "means crisp and pristine, while lower scores mean the image is degraded. Our model estimates this "
        "score using 8 handcrafted spatial measurements (such as edge sharpness, ink density, and contrast)."
    ),
    "takeaways": [
        (
            "🟢 What Works Well",
            "On smooth, natural distortions like **Gaussian Blur** and **Image Noise**, the model is highly accurate "
            "($R^2 > 0.88$). Its average prediction error is under 4%, accurately tracking how visual quality degrades.",
        ),
        (
            "🔴 Where It Fails",
            "When images are corrupted with **thin lines** or **arc strokes** crossing the letters (common in real-world CAPTCHAs), "
            "the model is fooled. At high severity, it overestimates clarity and performs worse than a simple baseline average.",
        ),
        (
            "💡 Why It Matters",
            "Automated quality tools cannot rely only on whole-image sharpness or contrast to evaluate CAPTCHAs. "
            "They need stroke-aware or character-aware models to detect line-cut distortions reliably.",
        ),
    ],
}

GLOBAL_DISCLAIMERS = (
    "Synthetic benchmark. Evaluates automated quality scoring, not CAPTCHA solving or security.",
    "The target is reference-based SSIM (structural similarity to clean reference), not human readability.",
    "Rows within a source text are grouped; the source text is the independent evaluation unit.",
    "The ten typefaces represent a purposeful evaluation set across serif, sans, and display fonts.",
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
    "Not a CAPTCHA solver: This research evaluates automated visual quality scoring, not bot solving or security bypass.",
    "Not a human-difficulty study: The target is mathematical image similarity (SSIM) to the clean reference, not human readability.",
    "Descriptive findings: Explores where spatial feature models fail (specifically thin lines and arcs) without claiming a single universal cause.",
)

ERROR_ANALYSIS_NON_CLAIMS = (
    "Exploratory analysis: Highlights empirical error patterns across corruption families, severities, and fonts.",
    "No human readability or security ratings: Does not measure human solve times or bot bypass resilience.",
    "Font differences are not a general population estimate: The ten typefaces represent a purposeful benchmark roster.",
    "Large-error threshold is fixed at 0.20 SSIM: Used as a convenient benchmark filter for severe quality mispredictions.",
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
