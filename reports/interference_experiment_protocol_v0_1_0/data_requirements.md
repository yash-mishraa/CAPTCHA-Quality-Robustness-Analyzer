# Data requirements

## 1. Existing inputs

These artifacts already exist and are frozen. They are used for provenance,
design context and exploratory comparison. They are not modified by M9B.

| Input | Path | SHA-256 |
| --- | --- | --- |
| M9A analysis protocol | `reports/interference_error_analysis_v0_1_0/analysis_protocol.json` | `9611a3ed48eca80dfa32ad27e33e16f06915e81625bd1eae35d5a7ffc6d9a4f1` |
| M9A hypotheses | `reports/interference_error_analysis_v0_1_0/hypotheses_and_followups.md` | `36bef0136f72f4e072f2da0d218883723750d7315969c5791037ee5771cbab24` |
| M9A README | `reports/interference_error_analysis_v0_1_0/README.md` | `22de39a32f23f501789f1a771d2c0c4f70eaa0f527988810ff1313f91de5a7c8` |
| M9A config | `configs/interference_error_analysis_v0_1_0.toml` | `2db5abf5237e3d247d3a358338cd07351a0b001c758632ef0ae8cbde4effd480` |
| M8C execution protocol | `configs/cross_font_benchmark_protocol_v0_2_0.toml` | `059388572f264fb2332f3427da774f1ab30f1339a27f19fd3e98e98163d4574c` |
| M8C lifecycle | `configs/benchmark_lifecycle_v0_3_0.toml` | `3339dc64472b1510cb297d9af0884bb9d3b79e1bf0a8e978f04b72cd64ece83d` |
| M8E integrity contract | `configs/benchmark_integrity_v0_1_0.toml` | `0b018004f41c2373fce2463eff78e11763ed53fd96f4436d1584ecd3e03707df` |
| M8C report | `reports/matched_font_benchmark_v0_1_0/report.md` | `ef266df7d0e3022b7eb0b6ccf08bb97782a8d2741e4f980f51facf94be7ad85f` |
| M8C predictions | `reports/matched_font_benchmark_v0_1_0/predictions.csv` | `e6c1f1ba83bbb05c18fed8a9b0dd1c87014a6f6c3667475971b2834dfedd4dd4` |
| M8C run metadata | `reports/matched_font_benchmark_v0_1_0/run_metadata.json` | `930ce8d8767a3e52919e5736ff8a9dfe4b74b008c54e2cc2107fd221d0944ebd` |
| M8C dataset summary | `data/processed/matched_font_benchmark_v0.1.0/dataset_summary.json` | `40158d7e94208e9eb342e3776d21b7f044b04dd1f3e851ce191e5009eea9795e` |
| M8C manifest | `data/processed/matched_font_benchmark_v0.1.0/manifest.csv` | `35ae405fd084070bf4d6c235638cbda6671573389e880dcca2a63d8b66049743` |
| M8C features | `data/processed/matched_font_benchmark_v0.1.0/features.csv` | `fa2d79c85d682b26545dc2c4e3ec5ece2a41d85aea7e1ea1070f7256ff8519bb` |
| M8C sources | `data/processed/matched_font_benchmark_v0.1.0/sources.csv` | `a6f721f8e849d44750a3aaeaec74cc1bd93a8122e243985173904b40116d5010` |
| M8C source cases | `data/processed/matched_font_benchmark_v0.1.0/source_cases.csv` | `5e34d6a826b5f4444675113c0030a138ace02d3369f085a6e197613bacf01b88` |
| M8C ground truth | `data/processed/matched_font_benchmark_v0.1.0/ground_truth.csv` | `426fa41949223ebc481e88b0c42a22770df80d30d3ed8f951fa0f52b92b39dfd` |
| M5C champion artifact | `models/retention_aware_selection_v0_1_0/selected_policy.joblib` | `c9b1625de3bb503f22fa14f9b6468b028e609b6c7febe7c307dafd0e3e1156e0` |
| M5C model card | `models/retention_aware_selection_v0_1_0/selected_policy_card.json` | `9a75725ba1ccf82ab42f009aa744a102cce86a6a9bb1d1aa86431362213507f8` |
| M8C text-case manifest | `configs/cross_font_text_cases_v0_1_0.toml` | `fa9a2c57f7aec6dda214a933f3b639160ef2f2f2996129d64b225f2e60f46f48` |
| Frozen feature extractor | `src/captcha_quality/features.py` | `ee345d09f3f0898d2b13daec85fa695a7c8f59693e318853cf3bdb20a88a1b30` |
| Frozen modelling and error metrics | `src/captcha_quality/modelling.py` | `c8c08596f1b60169e6e97c13ec1575911caa96372f6e192e6b33785dfafde682` |
| Grouped split implementation | `src/captcha_quality/splits.py` | `82b61f6048e09055fa65118313cb5b586ca769cd746df3f492b4eb2ea230ee3f` |
| M9A analysis code | `src/captcha_quality/interference_error_analysis.py` | `11e39fc14e55fe61f3d5aeb57f9e33a9bd3e6249c24e353ce52166e3af3dbc19` |
| Frozen renderer | `src/captcha_quality/render.py` | `729322c5edc4ba0734674778d12fb4295978e7e2c4279c40f51fe19ce517cd3e` |
| Frozen corruption ladder | `src/captcha_quality/corruptions.py` | `303136fcf65f92c6d17545d3b56233d827b2ffb98acbb2f2c4aa363789f316a3` |
| Frozen thin/arc interference | `src/captcha_quality/interference.py` | `dbc981a53a6a478bb339256aee603cf8a0f83d11a881576df8611d12acaf7578` |
| Frozen SSIM/PSNR metrics | `src/captcha_quality/quality_metrics.py` | `5b95a369761166f1b8172966485304143743b982c7b72c24eb5b3b66c267b40b` |

### Ground-truth handling

`ground_truth.csv` is Git-ignored and untracked. The protocol may check its
SHA-256 as an integrity input, but it does not read, print, join or write the
text column. No literal text from that file may appear in this package, in logs,
in a report table or in a committed artifact.

## 2. What the current data cannot provide

The existing M8C benchmark cannot serve as a confirmatory test set because:

- its outcomes have already been observed in M9A;
- M9A-derived hypotheses were generated from those outcomes;
- the M9C champion was already evaluated on it;
- ten purposive fonts are not a font population;
- it contains no independent new thin/thick font pairs for H3;
- it provides no untouched set for repeated model decisions.

Using M8C again is allowed only as an explicitly labelled exploratory analysis.
It cannot be relabelled as confirmatory by adding a new statistical test after
the fact.

## 3. Pilot P

Pilot P is a new, independently generated source set used only for planning. It
must not be used to adjudicate H1, H2 or H3.

### 3.1 Purpose

- estimate source-level and case-level variance for H1, H2 and H3;
- estimate matching yield and feature/target overlap for H2;
- estimate target-bin overlap and case-level variance for H3;
- determine the confirmatory sample size and attainable simultaneous-interval
  half-width;
- verify that the fixed renderer and corruption ladder produce valid images and
  features.

### 3.2 Requirements

- New cases, new render seeds and new corruption seeds.
- No source group, case or image reused from M8C.
- For H3 planning, any fonts used in P must be disjoint from the fonts used in
  confirmatory set C.
- The same 192x64 8-bit grayscale renderer and 21-row-per-source structure.
- Complete thin-line, arc-stroke, blur and noise ladders.
- A recorded pilot analysis plan that fixes the candidate sample-size grid,
  precision targets, matching feasibility checks and the rule for freezing the
  confirmatory sample size.
- Pilot outcomes are stored separately and are not merged into the confirmatory
  analysis.

### 3.3 Stopping rule

If the pilot cannot achieve the precision targets within a feasible dataset, or
if H2 feature/target overlap or H3 target overlap is inadequate, confirmatory
data collection does not proceed. The next step is a design revision, not a
weaker threshold.

## 4. Confirmatory set C

C is the only data regime that can adjudicate the three hypotheses. It must be
generated under a separate frozen execution protocol after this design package,
the pilot analysis plan, the sample size, the font roster and the analysis code
are frozen.

### 4.1 Source and case design

- New committed case manifest with text hashes disjoint from every M8C case and
  from pilot P.
- New render seeds and new corruption seeds under a new namespace.
- The same matched-case principle: the same text case is rendered in every
  predeclared font.
- The same 192x64 8-bit grayscale renderer and the same corruption ladder.
- 21 rows per source: 19 primary and 2 diagnostic. The clean reference and the
  0.5 px elastic diagnostic are excluded from primary estimands.
- All source groups are evaluation-only. No model fitting occurs.

### 4.2 Required row structure per source

| Rows | Condition | Role |
| --- | --- | --- |
| 1 | clean reference | diagnostic only |
| 4 | Gaussian blur s1-s4 | original reference |
| 4 | Gaussian noise s1-s4 | original reference |
| 4 | thin lines s1-s4 | interference |
| 4 | arc strokes s1-s4 | interference |
| 3 | elastic geometric s1, s2, s3 | primary diagnostic/context |
| 1 | elastic geometric 0.5 px | diagnostic only |

H1 uses the eight interference rows and the eight original blur/noise rows. H2
uses interference rows and benign blur/noise rows. H3 uses interference rows.
The geometric rows are secondary context unless a future protocol explicitly
promotes them.

### 4.3 Required fields

At minimum, C must provide or derive:

- `source_id`, `case_id`, `font_id`, `variant_family`, `severity_level`,
  `protocol_scope`, `sample_id`, `row_id`;
- `image_path`, `reference_path`, image and reference SHA-256;
- `ssim_target`, `psnr_target` (diagnostic only);
- the eight frozen feature values;
- the frozen champion prediction;
- split labels and split-group identifiers that keep all variants of a source
  together;
- generation, render and corruption seeds;
- font SHA-256, case text SHA-256, text length and generation-configuration
  hash.

The literal text column must remain in the Git-ignored ground-truth file.

### 4.4 Frozen champion evaluation

- Load the pinned M5C artifact and model card once.
- Verify the artifact SHA-256, card SHA-256, model family, target and exact
  eight-feature order.
- Extract the eight features with the frozen extractor.
- Predict every row once.
- Do not fit, finetune, calibrate, select, threshold-tune or change features.
- Do not use C outcomes to choose a later model.

### 4.5 H3 font requirements

H3 requires a new font roster that is independent of M5C training, M8C and pilot
P:

- at least two new thin-stroke fonts and at least two new thick-stroke fonts;
- licence eligibility recorded under the project font-audit process;
- pinned font-file SHA-256 values;
- complete glyph coverage for the project charset;
- a predeclared external stroke-width classification applied to clean
  renderings before outcomes are inspected;
- exact roster frozen after the pilot and before C generation.

If a font was used in M5C training or calibration, it cannot support H3 as a new
font. If a font identity is reused from M8C, it is exploratory only for H3.

### 4.6 Target overlap requirements

- H2 requires adequate overlap between interference and benign rows within
  source and target-SSIM bin. The pilot must establish feasible bin width and
  feature caliper.
- H3 requires adequate overlap between thin and thick fonts within case,
  family, severity and target-SSIM bin.
- If overlap is inadequate, the result is `not_testable`; the protocol does not
  relax the matching or threshold after seeing C.

## 5. What is deliberately not required

- No human readability, solvability or security labels.
- No new target metric; `ssim_target` remains the target.
- No new feature set; the frozen eight features remain the baseline.
- No model retraining or calibration artifact.
- No new benchmark that is used only to improve a model.
- No test-set-specific threshold or feature selection.

## 6. Data immutability and provenance

The confirmatory execution must record:

- the exact code revision and clean/dirty state;
- the protocol version and SHA-256;
- all input hashes;
- the model artifact and card hashes;
- the case manifest hash and text-hash set;
- the font roster and font-file hashes;
- the generation configuration hash;
- the bootstrap seed and replicate count;
- the complete analysis code revision;
- all verification outcomes.

The historical M8C, M8D, M8E and M9A files must remain byte-identical. If an
integrity verifier fails, the run stops; historical data is never repaired or
rewritten to make the new analysis pass.
