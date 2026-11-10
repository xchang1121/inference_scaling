# CoT guidance: paired F1 comparison

16 questions; 11 accepted evidence packets; 1024 selected-answer views.

F1 is lexical overlap with the reference answer. BoN selectors share the same candidate pool; single is candidate 0 with its logical token cost, not a separately timed run. All eight BoN answers were generated for selector/oracle diagnostics; selected readout cost and total physical readout cost are separate columns in scores.csv.

| Context | Method | F1 | Search tokens | Answer tokens | Empty | Forced thinking close |
|---|---|---:|---:|---:|---:|---:|
| inline_relevant | is_block_sc | 15.686 | 1368 | 600 | 0 | 0 |
| inline_relevant | is_met_sc | 15.601 | 1491 | 594 | 0 | 0 |
| inline_relevant | is_sir_sc | 15.385 | 1398 | 602 | 0 | 0 |
| inline_relevant | is_block_consilience | 15.042 | 1881 | 601 | 0 | 0 |
| inline_relevant | bon_logprob | 13.994 | 3002 | 616 | 0 | 0 |
| inline_relevant | single | 13.874 | 470 | 619 | 0 | 0 |
| inline_relevant | bon_sc | 13.521 | 3002 | 630 | 0 | 0 |
| inline_relevant | bon_consilience | 13.418 | 3002 | 671 | 0 | 0 |
| inline_unrelated | is_block_sc | 11.215 | 1351 | 638 | 0 | 0 |
| inline_unrelated | is_block_consilience | 11.187 | 1467 | 650 | 0 | 0 |
| inline_unrelated | is_met_sc | 11.147 | 1362 | 634 | 0 | 0 |
| inline_unrelated | is_sir_sc | 11.028 | 1451 | 644 | 0 | 0 |
| inline_unrelated | bon_sc | 10.770 | 3075 | 643 | 0 | 0 |
| inline_unrelated | single | 10.767 | 536 | 677 | 0 | 0 |
| inline_unrelated | bon_logprob | 10.633 | 3075 | 639 | 1 | 0 |
| inline_unrelated | bon_consilience | 9.394 | 3075 | 758 | 0 | 0 |
| static_relevant | is_block_consilience | 17.779 | 1289 | 511 | 0 | 0 |
| static_relevant | bon_logprob | 17.462 | 2739 | 525 | 0 | 0 |
| static_relevant | single | 17.325 | 444 | 537 | 0 | 0 |
| static_relevant | bon_consilience | 17.101 | 2739 | 548 | 0 | 0 |
| static_relevant | bon_sc | 17.011 | 2739 | 545 | 0 | 0 |
| static_relevant | is_block_sc | 16.976 | 1264 | 517 | 0 | 0 |
| static_relevant | is_met_sc | 16.891 | 1265 | 511 | 0 | 0 |
| static_relevant | is_sir_sc | 16.875 | 1263 | 535 | 0 | 0 |
| static_unrelated | is_block_consilience | 11.266 | 1720 | 648 | 0 | 0 |
| static_unrelated | is_block_sc | 11.171 | 1679 | 644 | 0 | 0 |
| static_unrelated | bon_sc | 11.154 | 3908 | 631 | 0 | 0 |
| static_unrelated | is_met_sc | 11.029 | 1674 | 650 | 0 | 0 |
| static_unrelated | is_sir_sc | 10.623 | 1812 | 650 | 0 | 0 |
| static_unrelated | single | 10.554 | 597 | 657 | 0 | 0 |
| static_unrelated | bon_logprob | 10.517 | 3908 | 658 | 0 | 0 |
| static_unrelated | bon_consilience | 10.350 | 3908 | 696 | 0 | 0 |

## Selected-path injection stops

An immediate thinking-close marker is distinct from EOS or an empty final answer. Counts below are correlated selector views; raw physical counts follow separately.

| Context | Method | Injected | Immediate thinking close | Immediate EOS | Mean thinking after injection |
|---|---|---:|---:|---:|---:|
| inline_relevant | is_block_sc | 22 | 15 | 0 | 125.0 |
| inline_relevant | is_met_sc | 22 | 15 | 0 | 125.0 |
| inline_relevant | is_sir_sc | 22 | 16 | 0 | 95.8 |
| inline_relevant | is_block_consilience | 22 | 6 | 0 | 388.0 |
| inline_relevant | bon_logprob | 22 | 10 | 0 | 318.2 |
| inline_relevant | single | 22 | 5 | 0 | 234.7 |
| inline_relevant | bon_sc | 22 | 16 | 0 | 150.0 |
| inline_relevant | bon_consilience | 22 | 5 | 0 | 514.1 |
| inline_unrelated | is_block_sc | 22 | 19 | 0 | 72.1 |
| inline_unrelated | is_block_consilience | 22 | 9 | 0 | 297.8 |
| inline_unrelated | is_met_sc | 22 | 19 | 0 | 74.5 |
| inline_unrelated | is_sir_sc | 22 | 19 | 0 | 73.5 |
| inline_unrelated | bon_sc | 22 | 17 | 0 | 132.7 |
| inline_unrelated | single | 22 | 8 | 0 | 330.5 |
| inline_unrelated | bon_logprob | 22 | 16 | 0 | 161.9 |
| inline_unrelated | bon_consilience | 22 | 4 | 0 | 582.6 |
| static_relevant | is_block_consilience | 22 | 0 | 0 | 284.2 |
| static_relevant | bon_logprob | 22 | 0 | 0 | 292.2 |
| static_relevant | single | 22 | 0 | 0 | 289.0 |
| static_relevant | bon_consilience | 22 | 0 | 0 | 284.0 |
| static_relevant | bon_sc | 22 | 0 | 0 | 288.5 |
| static_relevant | is_block_sc | 22 | 0 | 0 | 266.7 |
| static_relevant | is_met_sc | 22 | 0 | 0 | 266.7 |
| static_relevant | is_sir_sc | 22 | 0 | 0 | 267.2 |
| static_unrelated | is_block_consilience | 22 | 0 | 0 | 507.1 |
| static_unrelated | is_block_sc | 22 | 0 | 0 | 474.9 |
| static_unrelated | bon_sc | 22 | 0 | 0 | 484.5 |
| static_unrelated | is_met_sc | 22 | 0 | 0 | 460.3 |
| static_unrelated | is_sir_sc | 22 | 0 | 0 | 508.8 |
| static_unrelated | single | 22 | 0 | 0 | 523.5 |
| static_unrelated | bon_logprob | 22 | 0 | 0 | 477.9 |
| static_unrelated | bon_consilience | 22 | 0 | 0 | 606.1 |

Paired differences below average seeds within each question; bootstrap intervals are descriptive and unadjusted.

- bon_consilience, inline_relevant − static_relevant: -3.683 F1 points, 95% CI [-7.122, -1.241].
- bon_consilience, inline_relevant − inline_unrelated: +4.024 F1 points, 95% CI [+1.660, +6.822].
- bon_consilience, static_relevant − static_unrelated: +6.751 F1 points, 95% CI [+2.819, +11.962].
- bon_logprob, inline_relevant − static_relevant: -3.467 F1 points, 95% CI [-6.194, -1.351].
- bon_logprob, inline_relevant − inline_unrelated: +3.361 F1 points, 95% CI [+1.482, +5.472].
- bon_logprob, static_relevant − static_unrelated: +6.944 F1 points, 95% CI [+3.073, +11.966].
- bon_sc, inline_relevant − static_relevant: -3.490 F1 points, 95% CI [-6.697, -0.796].
- bon_sc, inline_relevant − inline_unrelated: +2.751 F1 points, 95% CI [+1.142, +4.409].
- bon_sc, static_relevant − static_unrelated: +5.857 F1 points, 95% CI [+2.540, +9.799].
- is_block_consilience, inline_relevant − static_relevant: -2.737 F1 points, 95% CI [-4.602, -1.159].
- is_block_consilience, inline_relevant − inline_unrelated: +3.855 F1 points, 95% CI [+1.629, +6.517].
- is_block_consilience, static_relevant − static_unrelated: +6.513 F1 points, 95% CI [+2.934, +11.035].
- is_block_sc, inline_relevant − static_relevant: -1.290 F1 points, 95% CI [-3.504, +0.758].
- is_block_sc, inline_relevant − inline_unrelated: +4.471 F1 points, 95% CI [+1.829, +7.719].
- is_block_sc, static_relevant − static_unrelated: +5.805 F1 points, 95% CI [+2.546, +10.204].
- is_met_sc, inline_relevant − static_relevant: -1.290 F1 points, 95% CI [-3.504, +0.758].
- is_met_sc, inline_relevant − inline_unrelated: +4.455 F1 points, 95% CI [+1.822, +7.703].
- is_met_sc, static_relevant − static_unrelated: +5.862 F1 points, 95% CI [+2.592, +10.287].
- is_sir_sc, inline_relevant − static_relevant: -1.490 F1 points, 95% CI [-3.791, +0.357].
- is_sir_sc, inline_relevant − inline_unrelated: +4.357 F1 points, 95% CI [+1.927, +7.406].
- is_sir_sc, static_relevant − static_unrelated: +6.252 F1 points, 95% CI [+2.868, +10.332].
- single, inline_relevant − static_relevant: -3.450 F1 points, 95% CI [-7.002, -0.780].
- single, inline_relevant − inline_unrelated: +3.108 F1 points, 95% CI [+1.244, +5.022].
- single, static_relevant − static_unrelated: +6.771 F1 points, 95% CI [+2.795, +12.097].
- inline_relevant, bon_consilience − bon_sc: -0.103 F1 points, 95% CI [-1.516, +1.308].
- inline_relevant, bon_logprob − bon_sc: +0.473 F1 points, 95% CI [-0.916, +2.297].
- inline_relevant, is_block_consilience − bon_sc: +1.521 F1 points, 95% CI [-0.325, +3.652].
- inline_relevant, is_block_sc − bon_sc: +2.165 F1 points, 95% CI [-0.060, +4.859].
- inline_relevant, is_met_sc − bon_sc: +2.080 F1 points, 95% CI [-0.149, +4.779].
- inline_relevant, is_sir_sc − bon_sc: +1.865 F1 points, 95% CI [-0.069, +4.331].
- inline_relevant, single − bon_sc: +0.354 F1 points, 95% CI [-0.599, +1.478].
- inline_unrelated, bon_consilience − bon_sc: -1.375 F1 points, 95% CI [-2.727, -0.359].
- inline_unrelated, bon_logprob − bon_sc: -0.136 F1 points, 95% CI [-1.128, +0.758].
- inline_unrelated, is_block_consilience − bon_sc: +0.417 F1 points, 95% CI [-0.443, +1.360].
- inline_unrelated, is_block_sc − bon_sc: +0.445 F1 points, 95% CI [-0.407, +1.221].
- inline_unrelated, is_met_sc − bon_sc: +0.377 F1 points, 95% CI [-0.451, +1.150].
- inline_unrelated, is_sir_sc − bon_sc: +0.258 F1 points, 95% CI [-0.599, +1.003].
- inline_unrelated, single − bon_sc: -0.003 F1 points, 95% CI [-0.459, +0.442].
- static_relevant, bon_consilience − bon_sc: +0.090 F1 points, 95% CI [-1.251, +1.901].
- static_relevant, bon_logprob − bon_sc: +0.451 F1 points, 95% CI [-0.727, +2.207].
- static_relevant, is_block_consilience − bon_sc: +0.768 F1 points, 95% CI [-0.557, +2.200].
- static_relevant, is_block_sc − bon_sc: -0.035 F1 points, 95% CI [-1.814, +1.641].
- static_relevant, is_met_sc − bon_sc: -0.120 F1 points, 95% CI [-1.891, +1.553].
- static_relevant, is_sir_sc − bon_sc: -0.136 F1 points, 95% CI [-1.497, +1.266].
- static_relevant, single − bon_sc: +0.314 F1 points, 95% CI [-1.183, +2.260].
- static_unrelated, bon_consilience − bon_sc: -0.803 F1 points, 95% CI [-1.305, -0.322].
- static_unrelated, bon_logprob − bon_sc: -0.636 F1 points, 95% CI [-1.096, -0.207].
- static_unrelated, is_block_consilience − bon_sc: +0.112 F1 points, 95% CI [-0.609, +1.027].
- static_unrelated, is_block_sc − bon_sc: +0.017 F1 points, 95% CI [-0.707, +0.761].
- static_unrelated, is_met_sc − bon_sc: -0.125 F1 points, 95% CI [-0.834, +0.614].
- static_unrelated, is_sir_sc − bon_sc: -0.531 F1 points, 95% CI [-1.054, +0.031].
- static_unrelated, single − bon_sc: -0.600 F1 points, 95% CI [-1.236, +0.030].

Raw stop diagnostics: `{"candidate_answers": 1176, "candidate_empty": 9, "candidate_truncated": 0, "immediate_injected_close": 428, "immediate_injected_eos": 4, "injected_root_calls": 924, "probe_calls": 81, "probe_thinking_eos": 0, "search_calls": 2487, "thinking_eos": 22}`.

Offline audit: 43588 checks passed. Same search cap does not imply equal realized FLOPs; complete-pool IS and BoN use different amounts of work. Final answer readout and evidence extraction are additional measured costs.
