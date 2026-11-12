# IRCoT / ThinkRetrieve: frozen IS reader

2 questions, 1 seeds, 22 measurements.

F1 is word overlap on a 0–100 scale. Equal total generation allowances; actual costs differ. All controller and discarded candidate generation is included. Direct means one ordinary thinking trajectory, with no retrieval or IS. Every latency is measured from a cold prefix cache; initial model/index loading and warm-up are excluded.

| Arm | F1 | Generated tokens | Retrieval seconds | End-to-end seconds | Latency / direct | Paired geometric ratio | Empty | Truncated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| direct | 13.287 | 1507.5 | 0.000 | 81.37 | 1.000 | 1.000 | 0 | 0 |
| closed_is | 13.910 | 1943.0 | 0.000 | 75.65 | 0.930 | 0.946 | 0 | 0 |
| static_facts | 22.599 | 1399.0 | 0.667 | 62.47 | 0.768 | 0.781 | 0 | 0 |
| static_facts_noise | 13.841 | 2355.5 | 0.554 | 94.05 | 1.156 | 1.138 | 0 | 0 |
| ircot | 22.926 | 1666.5 | 2.007 | 86.79 | 1.067 | 1.086 | 0 | 0 |
| ircot_noise | 15.737 | 2631.5 | 1.232 | 110.93 | 1.363 | 1.383 | 0 | 0 |
| static_examples | 28.704 | 4997.0 | 0.302 | 133.80 | 1.644 | 1.553 | 0 | 0 |
| static_examples_noise | 14.568 | 3182.5 | 0.294 | 99.56 | 1.224 | 1.237 | 0 | 0 |
| thinkretrieve | 0.000 | 1104.0 | 1.345 | 62.23 | 0.765 | 0.778 | 2 | 0 |
| thinkretrieve_noise | 10.243 | 1375.0 | 1.346 | 75.17 | 0.924 | 0.932 | 1 | 0 |
| rethink | 15.916 | 4757.0 | 0.000 | 215.48 | 2.648 | 2.519 | 0 | 0 |

## Generation and stop diagnostics

| Arm | Controller tokens | Checkpoint tokens | IS/direct tokens | Answer tokens | Input injection tokens | Thought EOS outputs | Intercepted close | Immediate IS suffix close | Forced close |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| direct | 0.0 | 0.0 | 694.0 | 813.5 | 0.0 | 0 | 0 | 0 | 0 |
| closed_is | 0.0 | 128.0 | 1078.5 | 736.5 | 0.0 | 0 | 0 | 0 | 0 |
| static_facts | 0.0 | 192.0 | 571.0 | 636.0 | 684.0 | 0 | 0 | 0 | 0 |
| static_facts_noise | 0.0 | 128.0 | 1371.5 | 856.0 | 684.0 | 0 | 0 | 0 | 0 |
| ircot | 256.0 | 192.0 | 593.0 | 625.5 | 633.5 | 0 | 0 | 0 | 0 |
| ircot_noise | 256.0 | 192.0 | 1425.5 | 758.0 | 633.5 | 0 | 0 | 0 | 0 |
| static_examples | 0.0 | 128.0 | 4256.0 | 613.0 | 675.5 | 0 | 0 | 0 | 0 |
| static_examples_noise | 0.0 | 128.0 | 2314.0 | 740.5 | 675.5 | 0 | 0 | 0 | 0 |
| thinkretrieve | 1104.0 | 0.0 | 0.0 | 0.0 | 333.0 | 2 | 2 | 0 | 0 |
| thinkretrieve_noise | 1103.0 | 0.0 | 3.0 | 269.0 | 333.0 | 1 | 2 | 1 | 0 |
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
| static_examples | 2 | 0 | 0 | 821.5 |
| static_examples_noise | 2 | 0 | 0 | 711.5 |
| thinkretrieve | 2 | 0 | 2 | 0.0 |
| thinkretrieve_noise | 2 | 1 | 1 | 0.0 |
| rethink | 4 | 3 | 0 | 0.0 |

## Paired F1 differences

Seeds are averaged within question; intervals are a descriptive 5,000-draw question-cluster bootstrap, without multiplicity adjustment.

- static_facts − closed_is: +8.689, 95% CI [+6.092, +11.286].
- static_facts_noise − closed_is: -0.068, 95% CI [-2.781, +2.645].
- ircot − closed_is: +9.016, 95% CI [+4.878, +13.154].
- ircot_noise − closed_is: +1.828, 95% CI [-0.619, +4.274].
- static_examples − closed_is: +14.794, 95% CI [-0.723, +30.312].
- static_examples_noise − closed_is: +0.658, 95% CI [-0.637, +1.953].
- thinkretrieve − closed_is: -13.910, 95% CI [-15.319, -12.500].
- thinkretrieve_noise − closed_is: -3.667, 95% CI [-15.319, +7.985].
- rethink − closed_is: +2.006, 95% CI [+1.162, +2.851].
- closed_is − direct: +0.622, 95% CI [-1.096, +2.341].
- ircot − static_facts: +0.327, 95% CI [-1.214, +1.868].
- thinkretrieve − static_examples: -28.704, 95% CI [-42.812, -14.596].
- thinkretrieve − rethink: -15.916, 95% CI [-16.481, -15.351].
- static_facts − static_facts_noise: +8.757, 95% CI [+8.642, +8.873].
- ircot − ircot_noise: +7.189, 95% CI [+5.497, +8.880].
- static_examples − static_examples_noise: +14.136, 95% CI [-0.087, +28.358].
- thinkretrieve − thinkretrieve_noise: -10.243, 95% CI [-20.485, +0.000].

Audit: 2028 checks passed. Exact-length noise pairs: 8/8.

IRCoT uses a separate fresh reader; its noise control corrupts reader evidence only. ThinkRetrieve retains and extends its trace; its noise control follows its own evolving trace with matched packet lengths. Their F1/cost differences therefore compare complete adapted pipelines, not an isolated placement change.

One-time index construction on a3: facts 36.69s; 136 complete worked examples 153.09s. These startup costs are excluded from per-question latency. Encoder: intfloat/e5-base-v2 at f52bf8ec8c7124536f0efb74aca902b2995e5bcd.

Total measured pipeline generation: 53,837 tokens; elapsed pipeline time: 0.610 hours. Native job allocation including startup and teardown: 0.752 device-hours.

## Offline retrieval diagnostic

Gold document IDs are used only here, after generation. Retrieving the right source document does not guarantee that the selected short window contains the answer. Noise arms retain retrieval metadata but do not expose the original contents.

- static_facts: retrieved a gold source document in 2/2 runs.
- ircot: retrieved a gold source document in 2/2 runs. Dynamic queries newly found the source in 0 runs.
