# IRCoT / ThinkRetrieve: frozen IS reader

16 questions, 2 seeds, 352 measurements.

F1 is word overlap on a 0–100 scale. All IS arms share a 65,536-token search/controller allowance and an 8,192-token retained-thinking cap; actual costs differ. All controller and discarded candidate generation is included. Direct uses one ordinary thinking trajectory capped at 8,192 tokens, with no retrieval or IS. Every arm has the same 16,384-token final-answer cap. Every latency is measured from a cold prefix cache; initial model/index loading and warm-up are excluded.

| Arm | F1 | Generated tokens | Retrieval seconds | End-to-end seconds | Latency / direct | Paired geometric ratio | Empty | Truncated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| direct | 9.402 | 1465.9 | 0.000 | 74.98 | 1.000 | 1.000 | 0 | 0 |
| closed_is | 9.245 | 3027.8 | 0.000 | 92.45 | 1.233 | 1.229 | 0 | 0 |
| static_facts | 14.482 | 1813.9 | 0.605 | 60.85 | 0.812 | 0.793 | 0 | 0 |
| static_facts_noise | 9.070 | 2566.7 | 0.562 | 82.04 | 1.094 | 1.099 | 0 | 0 |
| ircot | 16.484 | 2298.8 | 1.314 | 80.78 | 1.077 | 1.070 | 0 | 0 |
| ircot_noise | 8.928 | 2795.0 | 1.279 | 95.80 | 1.278 | 1.298 | 0 | 0 |
| static_examples | 10.171 | 3085.9 | 0.348 | 91.52 | 1.221 | 1.181 | 0 | 0 |
| static_examples_noise | 9.040 | 3136.4 | 0.348 | 93.89 | 1.252 | 1.231 | 0 | 0 |
| thinkretrieve | 6.941 | 2514.3 | 2.307 | 120.41 | 1.606 | 1.554 | 13 | 0 |
| thinkretrieve_noise | 4.180 | 2147.0 | 1.885 | 104.67 | 1.396 | 1.356 | 20 | 0 |
| rethink | 8.788 | 1753.4 | 0.000 | 87.52 | 1.167 | 1.159 | 0 | 0 |

## Generation and stop diagnostics

| Arm | Controller tokens | Checkpoint tokens | IS/direct tokens | Answer tokens | Input injection tokens | Thought EOS outputs | Intercepted close | Immediate IS suffix close | Forced close |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| direct | 0.0 | 0.0 | 670.0 | 795.9 | 0.0 | 0 | 0 | 0 | 0 |
| closed_is | 0.0 | 136.0 | 2129.8 | 762.0 | 0.0 | 0 | 0 | 0 | 0 |
| static_facts | 0.0 | 142.7 | 1128.0 | 543.2 | 672.6 | 0 | 0 | 0 | 0 |
| static_facts_noise | 0.0 | 128.0 | 1736.2 | 702.5 | 672.6 | 0 | 0 | 0 | 0 |
| ircot | 254.1 | 138.2 | 1374.0 | 532.5 | 646.8 | 0 | 0 | 0 | 0 |
| ircot_noise | 254.1 | 128.0 | 1689.6 | 723.3 | 646.8 | 0 | 0 | 0 | 0 |
| static_examples | 0.0 | 128.0 | 2305.7 | 652.2 | 485.1 | 0 | 0 | 0 | 0 |
| static_examples_noise | 0.0 | 132.0 | 2304.5 | 699.9 | 485.1 | 0 | 0 | 0 | 0 |
| thinkretrieve | 1583.8 | 0.0 | 558.9 | 371.6 | 485.0 | 13 | 54 | 0 | 0 |
| thinkretrieve_noise | 1394.9 | 0.0 | 487.6 | 264.6 | 382.6 | 20 | 43 | 1 | 0 |
| rethink | 695.2 | 0.0 | 249.3 | 808.9 | 18.0 | 0 | 63 | 24 | 0 |

Immediate stops below are measured at actual insertion boundaries on the selected path. Several static packets form one insertion episode; ThinkRetrieve counts each sequential insertion separately. These differ from immediately closing an IS suffix after a checkpoint.

| Arm | Insertion episodes | Immediately close thinking | Immediate EOS | Mean thought tokens after last insertion |
|---|---:|---:|---:|---:|
| direct | 0 | 0 | 0 | — |
| closed_is | 0 | 0 | 0 | — |
| static_facts | 32 | 0 | 0 | 423.9 |
| static_facts_noise | 32 | 0 | 0 | 537.2 |
| ircot | 32 | 0 | 0 | 442.1 |
| ircot_noise | 32 | 0 | 0 | 511.4 |
| static_examples | 32 | 0 | 0 | 649.3 |
| static_examples_noise | 32 | 0 | 0 | 632.7 |
| thinkretrieve | 55 | 0 | 2 | 193.5 |
| thinkretrieve_noise | 44 | 1 | 3 | 63.4 |
| rethink | 64 | 47 | 0 | 66.8 |

## Paired F1 differences

Seeds are averaged within question; intervals are a descriptive 5,000-draw question-cluster bootstrap, without multiplicity adjustment.

- static_facts − closed_is: +5.237, 95% CI [+2.558, +8.363].
- static_facts_noise − closed_is: -0.175, 95% CI [-1.640, +0.884].
- ircot − closed_is: +7.239, 95% CI [+3.653, +11.188].
- ircot_noise − closed_is: -0.317, 95% CI [-1.712, +0.693].
- static_examples − closed_is: +0.927, 95% CI [+0.263, +1.590].
- static_examples_noise − closed_is: -0.205, 95% CI [-0.941, +0.353].
- thinkretrieve − closed_is: -2.304, 95% CI [-4.133, -0.530].
- thinkretrieve_noise − closed_is: -5.065, 95% CI [-7.294, -3.052].
- rethink − closed_is: -0.457, 95% CI [-1.486, +0.403].
- closed_is − direct: -0.157, 95% CI [-0.819, +0.452].
- ircot − static_facts: +2.002, 95% CI [-0.072, +4.289].
- thinkretrieve − static_examples: -3.230, 95% CI [-4.857, -1.700].
- thinkretrieve − rethink: -1.847, 95% CI [-3.667, +0.070].
- static_facts − static_facts_noise: +5.412, 95% CI [+2.202, +9.524].
- ircot − ircot_noise: +7.556, 95% CI [+3.702, +11.931].
- static_examples − static_examples_noise: +1.132, 95% CI [+0.268, +2.226].
- thinkretrieve − thinkretrieve_noise: +2.761, 95% CI [-0.153, +5.845].

Audit: 34518 checks passed. Exact-length noise pairs: 117/128.

IRCoT uses a separate fresh reader; its noise control corrupts reader evidence only. ThinkRetrieve retains and extends its trace; its noise control follows its own evolving trace with matched packet lengths. Their F1/cost differences therefore compare complete adapted pipelines, not an isolated placement change.

One-time index construction on a3: facts 36.36s; 136 complete worked examples 171.89s. These startup costs are excluded from per-question latency. Encoder: intfloat/e5-base-v2 at f52bf8ec8c7124536f0efb74aca902b2995e5bcd.

Total measured pipeline generation: 851,363 tokens; elapsed pipeline time: 8.755 hours. Native job allocation including startup and teardown: 8.877 device-hours.

## Offline retrieval diagnostic

Gold document IDs are used only here, after generation. Retrieving the right source document does not guarantee that the selected short window contains the answer. Noise arms retain retrieval metadata but do not expose the original contents.

- static_facts: retrieved a gold source document in 16/32 runs.
- ircot: retrieved a gold source document in 17/32 runs. Dynamic queries newly found the source in 5 runs.
