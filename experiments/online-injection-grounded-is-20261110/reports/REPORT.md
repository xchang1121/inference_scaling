# Grounded retrieval / IS: complete results

F1 is lexical overlap, not factual accuracy. Question means average seeds; regression and fresh cohorts remain separate.

| Cohort | Arm | Questions | F1 % | Work | Total generated | Seconds | PFLOPs | Empty | Forced close |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fresh | dynamic_anchor_sc | 12 | 13.729 | 2756.4 | 3245.9 | 98.65 | 1.475 | 0 | 1 |
| fresh | dynamic_legacy_sc | 12 | 11.199 | 4499.9 | 5011.9 | 119.88 | 1.586 | 0 | 4 |
| fresh | dynamic_verified_is | 12 | 14.069 | 5330.0 | 5330.0 | 112.00 | 1.224 | 0 | 3 |
| fresh | dynamic_verified_noise | 12 | 11.615 | 6103.8 | 6103.8 | 124.73 | 1.330 | 0 | 6 |
| fresh | static_anchor_sc | 12 | 14.750 | 2299.5 | 2787.6 | 83.68 | 1.385 | 0 | 3 |
| fresh | static_legacy_sc | 12 | 13.385 | 3909.8 | 4407.4 | 97.22 | 1.591 | 0 | 1 |
| fresh | static_verified_is | 12 | 14.713 | 4807.8 | 4807.8 | 97.31 | 1.149 | 0 | 4 |
| fresh | static_verified_noise | 12 | 11.015 | 5998.6 | 5998.6 | 117.02 | 1.334 | 0 | 7 |
| regression | dynamic_anchor_sc | 8 | 19.396 | 2791.8 | 2953.8 | 78.73 | 1.052 | 0 | 4 |
| regression | dynamic_legacy_sc | 8 | 14.029 | 3567.1 | 3758.6 | 83.43 | 1.013 | 0 | 1 |
| regression | dynamic_verified_is | 8 | 17.243 | 3206.8 | 3206.8 | 75.95 | 1.046 | 0 | 3 |
| regression | dynamic_verified_noise | 8 | 13.094 | 4776.6 | 4776.6 | 104.68 | 1.309 | 0 | 7 |
| regression | static_anchor_sc | 8 | 17.786 | 2415.2 | 2599.6 | 64.69 | 0.926 | 0 | 4 |
| regression | static_legacy_sc | 8 | 17.935 | 2926.1 | 3080.7 | 58.93 | 0.783 | 0 | 1 |
| regression | static_verified_is | 8 | 19.234 | 2676.6 | 2676.6 | 57.85 | 0.836 | 0 | 3 |
| regression | static_verified_noise | 8 | 13.405 | 4640.6 | 4640.6 | 96.89 | 1.212 | 0 | 7 |

## Realized token allocation

Per-condition means. Probe tokens include main drafts and retrieval auxiliaries, so this column overlaps retrieval auxiliaries; do not sum those two columns. Candidate answers and verifiers are paid inside verified-IS work.

| Cohort | Arm | Probe total | Search thinking | Candidate answers | Verifiers | Retrieval auxiliaries | Discarded confirmed prefix |
|---|---|---:|---:|---:|---:|---:|---:|
| fresh | dynamic_anchor_sc | 153.5 | 2602.8 | 0.0 | 0.0 | 69.5 | 29.2 |
| fresh | dynamic_legacy_sc | 189.8 | 4310.1 | 0.0 | 0.0 | 45.8 | 0.0 |
| fresh | dynamic_verified_is | 153.5 | 2057.5 | 2861.9 | 257.0 | 69.5 | 29.2 |
| fresh | dynamic_verified_noise | 153.5 | 2733.9 | 3045.7 | 170.7 | 69.5 | 29.2 |
| fresh | static_anchor_sc | 23.5 | 2276.0 | 0.0 | 0.0 | 23.5 | 0.0 |
| fresh | static_legacy_sc | 0.0 | 3909.8 | 0.0 | 0.0 | 0.0 | 0.0 |
| fresh | static_verified_is | 23.5 | 1809.6 | 2700.4 | 274.3 | 23.5 | 0.0 |
| fresh | static_verified_noise | 23.5 | 2740.4 | 3070.1 | 164.6 | 23.5 | 0.0 |
| regression | dynamic_anchor_sc | 144.0 | 2647.8 | 0.0 | 0.0 | 58.0 | 39.8 |
| regression | dynamic_legacy_sc | 196.0 | 3371.1 | 0.0 | 0.0 | 48.0 | 0.0 |
| regression | dynamic_verified_is | 144.0 | 1877.1 | 997.8 | 187.9 | 58.0 | 39.8 |
| regression | dynamic_verified_noise | 144.0 | 3075.1 | 1402.1 | 155.4 | 58.0 | 39.8 |
| regression | static_anchor_sc | 23.5 | 2391.7 | 0.0 | 0.0 | 23.5 | 0.0 |
| regression | static_legacy_sc | 0.0 | 2926.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| regression | static_verified_is | 23.5 | 1598.9 | 899.6 | 154.5 | 23.5 | 0.0 |
| regression | static_verified_noise | 23.5 | 3046.6 | 1415.5 | 155.0 | 23.5 | 0.0 |

## Paired question bootstrap

Differences and 95% intervals are F1 percentage points, descriptive and unadjusted.

- fresh: dynamic_verified_is − dynamic_legacy_sc: +2.871 [+0.735, +5.760], n=12.
- fresh: dynamic_verified_is − dynamic_anchor_sc: +0.341 [-0.792, +1.405], n=12.
- fresh: dynamic_verified_is − static_verified_is: -0.643 [-1.618, +0.215], n=12.
- fresh: dynamic_verified_is − static_legacy_sc: +0.684 [-4.304, +4.976], n=12.
- fresh: dynamic_anchor_sc − dynamic_legacy_sc: +2.530 [+0.214, +5.452], n=12.
- fresh: static_verified_is − static_anchor_sc: -0.037 [-0.883, +0.973], n=12.
- fresh: dynamic_verified_is − dynamic_verified_noise: +2.455 [+0.277, +5.404], n=12.
- fresh: static_verified_is − static_verified_noise: +3.698 [+0.976, +7.401], n=12.
- regression: dynamic_verified_is − dynamic_legacy_sc: +3.214 [+0.232, +6.647], n=8.
- regression: dynamic_verified_is − dynamic_anchor_sc: -2.153 [-3.363, -1.019], n=8.
- regression: dynamic_verified_is − static_verified_is: -1.990 [-6.144, +1.341], n=8.
- regression: dynamic_verified_is − static_legacy_sc: -0.692 [-5.547, +5.114], n=8.
- regression: dynamic_anchor_sc − dynamic_legacy_sc: +5.367 [+2.658, +8.385], n=8.
- regression: static_verified_is − static_anchor_sc: +1.448 [-1.597, +5.224], n=8.
- regression: dynamic_verified_is − dynamic_verified_noise: +4.149 [-0.042, +8.673], n=8.
- regression: static_verified_is − static_verified_noise: +5.828 [+0.127, +12.744], n=8.

## Stops in all executed calls

Each native execution unit is counted once, including unselected candidates. A first-token close in a later search call may follow a long existing prefix. The injected-root subset starts exactly at the immutable evidence/search boundary; it also includes static reference roots. Candidate-answer truncation is the deliberate answer cap, not thinking EOS. These are call diagnostics, not independent trials.

| Diagnostic | Count |
|---|---:|
| thinking_search_calls | 3437 |
| thinking_search_calls_with_eos | 1 |
| search_calls_first_close | 0 |
| search_calls_first_eos | 0 |
| injected_root_search_calls | 726 |
| injected_root_first_close | 0 |
| injected_root_first_eos | 0 |
| candidate_answer_calls | 792 |
| empty_candidate_answer_calls | 0 |
| truncated_candidate_answer_calls | 407 |

Audits passed: True; checks: 59561; unique execution units: 418.
