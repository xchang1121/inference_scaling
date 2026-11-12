# IRCoT / ThinkRetrieve: frozen IS reader

2 questions, 1 seeds, 22 measurements.

F1 is word overlap on a 0–100 scale. All IS arms share a 65,536-token search/controller allowance and an 8,192-token retained-thinking cap; actual costs differ. All controller and discarded candidate generation is included. Direct uses one ordinary thinking trajectory capped at 8,192 tokens, with no retrieval or IS. Every arm has the same 16,384-token final-answer cap. Every latency is measured from a cold prefix cache; initial model/index loading and warm-up are excluded.

| Arm | F1 | Generated tokens | Retrieval seconds | End-to-end seconds | Latency / direct | Paired geometric ratio | Empty | Truncated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| direct | 13.287 | 1507.5 | 0.000 | 76.66 | 1.000 | 1.000 | 0 | 0 |
| closed_is | 13.910 | 1943.0 | 0.000 | 64.69 | 0.844 | 0.852 | 0 | 0 |
| static_facts | 22.599 | 1399.0 | 0.650 | 55.34 | 0.722 | 0.726 | 0 | 0 |
| static_facts_noise | 13.841 | 2355.5 | 0.569 | 84.48 | 1.102 | 1.096 | 0 | 0 |
| ircot | 22.926 | 1666.5 | 2.017 | 72.38 | 0.944 | 0.952 | 0 | 0 |
| ircot_noise | 15.737 | 2631.5 | 1.270 | 96.26 | 1.256 | 1.267 | 0 | 0 |
| static_examples | 31.439 | 5144.0 | 0.302 | 132.97 | 1.734 | 1.746 | 0 | 0 |
| static_examples_noise | 13.060 | 2837.5 | 0.294 | 88.94 | 1.160 | 1.159 | 0 | 0 |
| thinkretrieve | 0.000 | 2008.0 | 2.026 | 100.48 | 1.311 | 1.211 | 2 | 0 |
| thinkretrieve_noise | 0.000 | 1550.0 | 1.365 | 77.00 | 1.004 | 0.977 | 2 | 0 |
| rethink | 15.916 | 4757.0 | 0.000 | 212.48 | 2.772 | 2.623 | 0 | 0 |

## Generation and stop diagnostics

| Arm | Controller tokens | Checkpoint tokens | IS/direct tokens | Answer tokens | Input injection tokens | Thought EOS outputs | Intercepted close | Immediate IS suffix close | Forced close |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| direct | 0.0 | 0.0 | 694.0 | 813.5 | 0.0 | 0 | 0 | 0 | 0 |
| closed_is | 0.0 | 128.0 | 1078.5 | 736.5 | 0.0 | 0 | 0 | 0 | 0 |
| static_facts | 0.0 | 192.0 | 571.0 | 636.0 | 684.0 | 0 | 0 | 0 | 0 |
| static_facts_noise | 0.0 | 128.0 | 1371.5 | 856.0 | 684.0 | 0 | 0 | 0 | 0 |
| ircot | 256.0 | 192.0 | 593.0 | 625.5 | 633.5 | 0 | 0 | 0 | 0 |
| ircot_noise | 256.0 | 192.0 | 1425.5 | 758.0 | 633.5 | 0 | 0 | 0 | 0 |
| static_examples | 0.0 | 128.0 | 4584.0 | 432.0 | 683.5 | 0 | 0 | 0 | 0 |
| static_examples_noise | 0.0 | 128.0 | 1932.0 | 777.5 | 683.5 | 0 | 0 | 0 | 0 |
| thinkretrieve | 1542.5 | 0.0 | 465.5 | 0.0 | 493.5 | 2 | 3 | 0 | 0 |
| thinkretrieve_noise | 1105.0 | 0.0 | 445.0 | 0.0 | 337.0 | 2 | 2 | 0 | 0 |
| rethink | 703.0 | 0.0 | 3397.5 | 656.5 | 18.0 | 0 | 4 | 2 | 0 |

Immediate stops below are measured at actual insertion boundaries on the selected path. Several static packets form one insertion episode; ThinkRetrieve counts each sequential insertion separately. These differ from immediately closing an IS suffix after a checkpoint.

| Arm | Insertion episodes | Immediately close thinking | Immediate EOS | Mean thought tokens after last insertion |
|---|---:|---:|---:|---:|
| direct | 0 | 0 | 0 | — |
| closed_is | 0 | 0 | 0 | — |
| static_facts | 2 | 0 | 0 | 319.0 |
| static_facts_noise | 2 | 0 | 0 | 499.5 |
| ircot | 2 | 0 | 0 | 295.0 |
| ircot_noise | 2 | 0 | 0 | 528.5 |
| static_examples | 2 | 0 | 0 | 867.5 |
| static_examples_noise | 2 | 0 | 0 | 726.5 |
| thinkretrieve | 3 | 0 | 0 | 1.5 |
| thinkretrieve_noise | 2 | 0 | 0 | 7.0 |
| rethink | 4 | 3 | 0 | 0.0 |

## Paired F1 differences

Seeds are averaged within question; intervals are a descriptive 5,000-draw question-cluster bootstrap, without multiplicity adjustment.

- static_facts − closed_is: +8.689, 95% CI [+6.092, +11.286].
- static_facts_noise − closed_is: -0.068, 95% CI [-2.781, +2.645].
- ircot − closed_is: +9.016, 95% CI [+4.878, +13.154].
- ircot_noise − closed_is: +1.828, 95% CI [-0.619, +4.274].
- static_examples − closed_is: +17.530, 95% CI [+5.567, +29.493].
- static_examples_noise − closed_is: -0.850, 95% CI [-2.561, +0.861].
- thinkretrieve − closed_is: -13.910, 95% CI [-15.319, -12.500].
- thinkretrieve_noise − closed_is: -13.910, 95% CI [-15.319, -12.500].
- rethink − closed_is: +2.006, 95% CI [+1.162, +2.851].
- closed_is − direct: +0.622, 95% CI [-1.096, +2.341].
- ircot − static_facts: +0.327, 95% CI [-1.214, +1.868].
- thinkretrieve − static_examples: -31.439, 95% CI [-41.993, -20.886].
- thinkretrieve − rethink: -15.916, 95% CI [-16.481, -15.351].
- static_facts − static_facts_noise: +8.757, 95% CI [+8.642, +8.873].
- ircot − ircot_noise: +7.189, 95% CI [+5.497, +8.880].
- static_examples − static_examples_noise: +18.380, 95% CI [+8.128, +28.632].
- thinkretrieve − thinkretrieve_noise: +0.000, 95% CI [+0.000, +0.000].

Audit: 2115 checks passed. Exact-length noise pairs: 7/8.

IRCoT uses a separate fresh reader; its noise control corrupts reader evidence only. ThinkRetrieve retains and extends its trace; its noise control follows its own evolving trace with matched packet lengths. Their F1/cost differences therefore compare complete adapted pipelines, not an isolated placement change.

One-time index construction on a3: facts 37.41s; 136 complete worked examples 130.72s. These startup costs are excluded from per-question latency. Encoder: intfloat/e5-base-v2 at f52bf8ec8c7124536f0efb74aca902b2995e5bcd.

Total measured pipeline generation: 55,599 tokens; elapsed pipeline time: 0.590 hours. Native job allocation including startup and teardown: 0.696 device-hours.

## Offline retrieval diagnostic

Gold document IDs are used only here, after generation. Retrieving the right source document does not guarantee that the selected short window contains the answer. Noise arms retain retrieval metadata but do not expose the original contents.

- static_facts: retrieved a gold source document in 2/2 runs.
- ircot: retrieved a gold source document in 2/2 runs. Dynamic queries newly found the source in 0 runs.
