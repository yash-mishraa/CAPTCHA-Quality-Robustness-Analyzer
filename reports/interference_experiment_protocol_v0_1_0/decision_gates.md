# Decision gates

## 1. Principles

- The protocol is frozen before confirmatory set C is generated.
- Each hypothesis has one primary estimand and one primary decision interval.
- Support, falsification, inconclusive and not-testable are distinct states.
- A wide interval is never support. A failed match is never a null result.
- M9B does not authorise a model or feature change.
- M9C is not automatically a feature-engineering or model-improvement
  milestone. The next step depends on the evidence and requires a separate
  decision, a written prioritisation memo and a new protocol version.

## 2. Decision states

| State | Meaning | What may happen next |
| --- | --- | --- |
| `not_tested` | No confirmatory data have been evaluated. | Do not interpret as evidence. Complete the pilot and freeze C. |
| `support` | The predeclared support rule and all guards pass. | Plan a separately approved follow-up experiment. Do not change the model or features in M9B. |
| `falsify` | The predeclared falsification rule is met. | Deprioritise the mechanism. Do not build an intervention on it. |
| `inconclusive` | Neither support nor falsification is established, including a CI crossing the decision threshold or a missed precision target. | Revise precision, overlap or design in a new planning protocol. Do not treat as support. |
| `not_testable` | Required data, overlap, balance, integrity or completeness conditions cannot be met. | Stop. A new dataset, factor or design is required. Do not relax thresholds. |

## 3. Hypothesis-specific gates

### 3.1 H1 — attenuated response

| Result | Decision | Required next action |
| --- | --- | --- |
| Simultaneous CI for the paired slope deficit has lower bound > `0.10`, mean interference slope upper bound < `0.80`, and both family slopes < `1.0` | `support` | Write a separate controlled intervention-development protocol only if approved. It must use independent source-grouped development and evaluation data. No calibration or retraining is authorised by M9B. |
| Slope-deficit CI upper bound <= `0.00`, or mean interference slope lower bound >= `0.90` | `falsify` | Deprioritise attenuation. Do not calibrate or retrain on this basis. Redirect to another preregistered error source or stop. |
| Any other result, including a CI crossing `0.10` or an interference slope CI crossing `0.80`/`0.90` | `inconclusive` | Revise precision, target-range overlap or family-specific design. Do not reinterpret the point estimate as support. |
| Fewer than 90% of planned sources eligible, or precision target missed | `not_testable` | Do not adjudicate. Redesign the data or sample size. |

### 3.2 H2 — feature conflation

| Result | Decision | Required next action |
| --- | --- | --- |
| Simultaneous CI for matched absolute-residual excess has lower bound > `0.02`, all feature/target SMDs <= `0.10`, at least 80% of interference rows matched, at least 90% of sources contribute | `support` | A separately approved feature-diagnostic or development protocol may be written. It must not select features on C and must use independent source-grouped development and evaluation data. No feature addition is authorised by M9B. |
| Simultaneous CI upper bound <= `0.02` | `falsify` | Deprioritise the conflation mechanism. Do not add or replace features on this basis. |
| Any other result, including a CI crossing `0.02` | `inconclusive` | Improve overlap, matching resolution or precision in a new pilot. Do not treat unmatched rows as evidence. |
| Balance/coverage failure, fewer than 90% of sources, or precision target missed | `not_testable` | Do not adjudicate. Redesign the matching or data. |

### 3.3 H3 — font/glyph difficulty

| Result | Decision | Required next action |
| --- | --- | --- |
| Simultaneous case-cluster CI for the thin-minus-thick matched contrast has lower bound > `0.02`, and every predeclared per-font sign check holds | `support` | A separately approved new-font characterisation or font-stratified benchmark may be written. It must not claim a causal or font-population effect and must not change the frozen champion. |
| Simultaneous CI upper bound <= `0.02` | `falsify` | Deprioritise the font-specific modifier. Do not build a font-specific model or feature on this basis. |
| Any other result, including a CI crossing `0.02` or a sign-check inconsistency | `inconclusive` | Increase the matched font roster, improve target overlap or separate the renderer factor in a new pilot. Do not reinterpret a failed match as a null font effect. |
| Target-overlap failure, fewer than 90% of matched cases, or precision target missed | `not_testable` | Do not adjudicate. A new font roster or experimental factor is required. |

## 4. Cross-hypothesis rules

### 4.1 Multiple supports

If more than one hypothesis is supported:

1. Do not automatically start multiple development tracks.
2. Write a prioritisation memo that considers effect size, robustness, available
   independent data and scientific value.
3. Choose at most one next milestone unless a separate review explicitly
   authorises more.
4. Keep the frozen champion unchanged until a separate development protocol is
   approved.

### 4.2 Mixed results

| Pattern | Interpretation | Next action |
| --- | --- | --- |
| H1 support, H2/H3 not supported | Attenuation is the best-supported description; feature/font mechanisms are not established. | Consider a controlled intervention-development plan only after approval. |
| H2 support, H1/H3 not supported | Feature conflation is the best-supported description; attenuation is not established. | Consider a feature-diagnostic plan only after approval. |
| H3 support, H1/H2 not supported | A conditional font-group contrast is the best-supported description; mechanism is unresolved. | Consider a new-font characterisation or benchmark, not a model change. |
| All inconclusive | The design cannot resolve the hypotheses. | Revise the pilot, precision, overlap or data factor. Do not start model work. |
| All falsified | None of the three mechanisms is supported in this design. | Deprioritise all three and redirect to a new preregistered question or stop. |
| Any not_testable | The current data cannot identify the effect. | New dataset or experimental factor required; no model change. |

### 4.3 Protocol or integrity failure

Stop immediately and report the blocker if any of the following occurs:

- a historical artifact hash changes;
- an M8C/M8D/M8E/M9A verifier fails;
- C overlaps M8C or pilot P in source, case, text hash, image or corruption
  seed;
- a font used for H3 was used in M5C training, M8C or pilot P;
- the sample size or roster is changed after seeing C outcomes;
- a threshold, feature, caliper or endpoint is changed after seeing C outcomes;
- the frozen champion artifact or card hash changes;
- ground-truth text is exposed or committed;
- the primary analysis is run more than once with a different endpoint.

Do not repair historical data, relax a contract or bypass a guard to make the
analysis proceed.

## 5. What M9C may and may not be

M9C is **not** authorised by this package. If the gates are met, M9C may be:

- a controlled intervention-development planning milestone for H1;
- a feature-diagnostic or development planning milestone for H2;
- a new-font characterisation or font-stratified benchmark planning milestone
  for H3;
- a methods or pilot milestone if the results are inconclusive;
- a stop/reprioritisation milestone if the hypotheses are falsified.

M9C may **not** be automatically:

- a feature-engineering milestone;
- a model-retraining or calibration milestone;
- a replacement of the frozen champion;
- a test-set-specific improvement based on C;
- a new claim about human readability, solvability, security or production
  robustness.

Any of those requires a separate protocol, independent data, a new evaluation
split and explicit approval.

## 6. Gate checklist before any confirmatory run

- [ ] Protocol version frozen and committed.
- [ ] Pilot P completed under its own plan and not used for hypothesis
      decisions.
- [ ] Sample size fixed by the precision simulation.
- [ ] Confirmatory case manifest frozen and disjoint from M8C and P.
- [ ] H3 font roster frozen, licence-eligible, hash-pinned and unused by M5C
      training, M8C and P.
- [ ] Matching rules, bins and calipers frozen.
- [ ] Analysis code frozen and hash-pinned.
- [ ] Bootstrap seed and replicate count frozen.
- [ ] Historical M8C/M8D/M8E/M9A hashes verified.
- [ ] Frozen M5C artifact and card hashes verified.
- [ ] Ground truth remains Git-ignored and untracked.
- [ ] One-shot analysis lock recorded.
- [ ] No threshold, feature or endpoint may change after C is inspected.
