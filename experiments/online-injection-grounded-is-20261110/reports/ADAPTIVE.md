# Pre-registered gate-routed IS

Exact gate-based dispatch among paired arms, not extra independent trials; gates and shared probes are paid. Same question-level seed-averaged bootstrap as the primary analysis.

| Cohort | Policy | F1 % | Generated tokens | PFLOPs | Verification branch |
|---|---|---:|---:|---:|---:|
| fresh | dynamic_adaptive_is | 13.714 | 4653.0 | 1.413 | 58.3% |
| fresh | dynamic_adaptive_noise | 11.260 | 5426.8 | 1.519 | 58.3% |
| fresh | static_adaptive_is | 14.820 | 4468.8 | 1.275 | 75.0% |
| fresh | static_adaptive_noise | 11.122 | 5659.6 | 1.460 | 75.0% |
| regression | dynamic_adaptive_is | 17.643 | 3293.5 | 1.094 | 87.5% |
| regression | dynamic_adaptive_noise | 13.495 | 4863.2 | 1.357 | 87.5% |
| regression | static_adaptive_is | 18.764 | 2901.7 | 0.939 | 75.0% |
| regression | static_adaptive_noise | 12.936 | 4865.8 | 1.315 | 75.0% |

Paired question bootstrap, difference in F1 percentage points; descriptive, unadjusted.

- fresh: dynamic_adaptive_is − dynamic_legacy_sc: +2.515 [+0.245, +5.525], n=12.
- fresh: dynamic_adaptive_is − dynamic_anchor_sc: -0.014 [-1.010, +0.937], n=12.
- fresh: dynamic_adaptive_is − dynamic_verified_is: -0.355 [-0.865, +0.091], n=12.
- fresh: dynamic_adaptive_is − static_adaptive_is: -1.106 [-2.041, -0.137], n=12.
- fresh: dynamic_adaptive_is − dynamic_adaptive_noise: +2.455 [+0.298, +5.357], n=12.
- fresh: dynamic_adaptive_is − static_legacy_sc: +0.329 [-4.423, +4.582], n=12.
- fresh: static_adaptive_is − static_verified_is: +0.107 [-0.001, +0.264], n=12.
- fresh: static_adaptive_is − static_anchor_sc: +0.070 [-0.760, +1.048], n=12.
- regression: dynamic_adaptive_is − dynamic_legacy_sc: +3.614 [+0.717, +6.886], n=8.
- regression: dynamic_adaptive_is − dynamic_anchor_sc: -1.753 [-3.055, -0.591], n=8.
- regression: dynamic_adaptive_is − dynamic_verified_is: +0.400 [+0.000, +1.200], n=8.
- regression: dynamic_adaptive_is − static_adaptive_is: -1.121 [-5.496, +2.645], n=8.
- regression: dynamic_adaptive_is − dynamic_adaptive_noise: +4.149 [-0.167, +8.595], n=8.
- regression: dynamic_adaptive_is − static_legacy_sc: -0.292 [-5.186, +5.359], n=8.
- regression: static_adaptive_is − static_verified_is: -0.470 [-1.408, +0.000], n=8.
- regression: static_adaptive_is − static_anchor_sc: +0.978 [-1.802, +4.933], n=8.

Audit: True; 466 checks; no additional generation.
