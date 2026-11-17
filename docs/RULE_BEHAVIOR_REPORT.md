# Rule Behavior Report

## Scope and limits

This report covers the existing deterministic hard rules on 100 fixed development images from the local BDD100K-derived manifest. It is a rule-trace diagnostic, not a judgment of real-world danger, accident probability, annotation quality, or model accuracy. No final holdout was read. The S4 source remains `local-unverified`, so this diagnostic cannot qualify a candidate or promote a rule/model change.

The internal `trace_hard_rules` path returns every matching stable rule ID in the exact production priority order. `apply_hard_rules` continues to return the existing `(risk_override, reason)` tuple and chooses the same first match. `tools/evaluate_rule_behavior.py` accepts baseline and candidate JSONL outputs, requires an explicit source namespace and development/stress/diagnostic role, rejects holdout input, records input/config/code hashes, and writes a bounded review queue containing model outputs only. Missing `reason_code` is classified as `legacy_unspecified` and counts as a processing failure. Incomplete feature dictionaries are not run through partial rules; they are counted separately.

## Development comparison

The paired diagnostic compares the legacy pipeline adapter and decoded pipeline using the same detector output for each of the same 100 images. This isolates the feature/rule response path. The two generated response manifests have identical SHA256, and all 100 sample IDs pair.

- Pipeline results: 32 `high` / `hard_rule_triggered`, 64 `unknown` / `no_rule_triggered`, 4 `unknown` / `no_supported_detections`.
- `unknown` here is kept separate from processing failures. The 68 unknown outcomes are not treated as false positives or evidence of safety. There were 0 processing failures.
- Feature coverage: 96 rows had all model features; 4 no-supported-detection rows correctly carried no feature payload. Only complete feature rows were passed to the rule trace.
- Output transitions: 0 changes across 100 paired outputs; there were no high-to-unknown or unknown-to-high changes.

### Trace counts

Counts below represent samples whose feature row matched each predicate. A sample may match more than one rule.

| Rule ID | Any match | First-match decisions |
| --- | ---: | ---: |
| `central_red_signal` | 0 | 0 |
| `near_person` | 0 | 0 |
| `person_forward_zone` | 3 | 3 |
| `vehicle_extremely_close` | 19 | 19 |
| `multiple_near_vehicles` | 24 | 10 |
| `center_blockage` | 2 | 0 |
| `high_congestion` | 0 | 0 |
| `substantial_vehicle_overlap` | 18 | 0 |

There were 23 rows with more than one matching rule. The trace keeps all matches and separately records which rule won by priority. This is useful for diagnosing rule overlap; it does not establish that any prediction is correct.

## Human review and promotion gates

`results/cpu-optimization/20261116-o5-rule-dev/audit-bounded/report.json` records hashes, role distributions, trace counts and output transitions. The matching `review_queue.jsonl` contains 100 items with `review_status=awaiting-human-review`; it contains no human labels and is not a reviewed sample. The queue is output-dependent and therefore diagnostic, not a natural-distribution accuracy sample. The evaluator caps each manifest at 2,000 rows and 100 MB, limits runtime to 1,800 seconds, and checks the deadline while reading and hashing inputs.

No second reviewer, annotation evidence, source archive evidence, or verified source-group mapping was available. The queue has not been reviewed, so visible-rule-premise agreement, missed-trigger rates, disagreements, and adjudication are unavailable. No paired bootstrap or quality-promotion test was run because O3's history, provenance, and annotation gates remain blocked. The candidate stays unpromoted and default rules remain unchanged.
