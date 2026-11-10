# CoT guidance: paired F1 comparison

2 questions; 2 accepted evidence packets; 64 selected-answer views.

F1 is lexical overlap with the reference answer. BoN selectors share the same candidate pool; single is candidate 0 with its logical token cost, not a separately timed run. All eight BoN answers were generated for selector/oracle diagnostics; selected readout cost and total physical readout cost are separate columns in scores.csv.

| Context | Method | F1 | Search tokens | Answer tokens | Empty | Forced thinking close |
|---|---|---:|---:|---:|---:|---:|
| inline_relevant | is_block_consilience | 18.482 | 648 | 411 | 0 | 0 |
| inline_relevant | bon_sc | 18.281 | 1268 | 332 | 0 | 0 |
| inline_relevant | bon_logprob | 18.281 | 1268 | 332 | 0 | 0 |
| inline_relevant | is_sir_sc | 18.281 | 408 | 332 | 0 | 0 |
| inline_relevant | is_met_sc | 18.281 | 576 | 332 | 0 | 0 |
| inline_relevant | is_block_sc | 18.281 | 576 | 332 | 0 | 0 |
| inline_relevant | single | 14.542 | 233 | 688 | 0 | 0 |
| inline_relevant | bon_consilience | 10.584 | 1268 | 188 | 1 | 0 |
| inline_unrelated | single | 15.317 | 146 | 644 | 0 | 0 |
| inline_unrelated | is_block_consilience | 13.654 | 656 | 662 | 0 | 0 |
| inline_unrelated | bon_consilience | 13.246 | 748 | 694 | 0 | 0 |
| inline_unrelated | bon_sc | 12.816 | 748 | 850 | 0 | 0 |
| inline_unrelated | bon_logprob | 12.816 | 748 | 850 | 0 | 0 |
| inline_unrelated | is_sir_sc | 9.263 | 554 | 354 | 0 | 0 |
| inline_unrelated | is_met_sc | 9.263 | 624 | 354 | 0 | 0 |
| inline_unrelated | is_block_sc | 9.263 | 624 | 354 | 0 | 0 |
| static_relevant | bon_sc | 21.306 | 1036 | 440 | 0 | 0 |
| static_relevant | bon_logprob | 19.686 | 1036 | 416 | 0 | 0 |
| static_relevant | bon_consilience | 19.421 | 1036 | 426 | 0 | 0 |
| static_relevant | single | 19.311 | 279 | 462 | 0 | 0 |
| static_relevant | is_sir_sc | 18.592 | 522 | 464 | 0 | 0 |
| static_relevant | is_met_sc | 18.592 | 556 | 464 | 0 | 0 |
| static_relevant | is_block_sc | 18.592 | 556 | 464 | 0 | 0 |
| static_relevant | is_block_consilience | 18.592 | 556 | 464 | 0 | 0 |
| static_unrelated | bon_logprob | 16.691 | 3514 | 564 | 0 | 0 |
| static_unrelated | is_met_sc | 16.392 | 1438 | 572 | 0 | 0 |
| static_unrelated | is_block_sc | 16.392 | 1438 | 572 | 0 | 0 |
| static_unrelated | is_block_consilience | 16.392 | 1438 | 572 | 0 | 0 |
| static_unrelated | single | 15.419 | 624 | 684 | 0 | 0 |
| static_unrelated | is_sir_sc | 14.228 | 1510 | 785 | 0 | 0 |
| static_unrelated | bon_sc | 13.583 | 3514 | 692 | 0 | 0 |
| static_unrelated | bon_consilience | 11.744 | 3514 | 752 | 0 | 0 |

## Selected-path injection stops

An immediate thinking-close marker is distinct from EOS or an empty final answer. Counts below are correlated selector views; raw physical counts follow separately.

| Context | Method | Injected | Immediate thinking close | Immediate EOS | Mean thinking after injection |
|---|---|---:|---:|---:|---:|
| inline_relevant | is_block_consilience | 2 | 1 | 0 | 396.0 |
| inline_relevant | bon_sc | 2 | 2 | 0 | 0.0 |
| inline_relevant | bon_logprob | 2 | 2 | 0 | 0.0 |
| inline_relevant | is_sir_sc | 2 | 2 | 0 | 0.0 |
| inline_relevant | is_met_sc | 2 | 2 | 0 | 0.0 |
| inline_relevant | is_block_sc | 2 | 2 | 0 | 0.0 |
| inline_relevant | single | 2 | 1 | 0 | 104.0 |
| inline_relevant | bon_consilience | 2 | 0 | 0 | 442.0 |
| inline_unrelated | single | 2 | 1 | 0 | 17.5 |
| inline_unrelated | is_block_consilience | 2 | 0 | 0 | 467.0 |
| inline_unrelated | bon_consilience | 2 | 0 | 0 | 359.0 |
| inline_unrelated | bon_sc | 2 | 2 | 0 | 0.0 |
| inline_unrelated | bon_logprob | 2 | 2 | 0 | 0.0 |
| inline_unrelated | is_sir_sc | 2 | 2 | 0 | 0.0 |
| inline_unrelated | is_met_sc | 2 | 2 | 0 | 0.0 |
| inline_unrelated | is_block_sc | 2 | 2 | 0 | 0.0 |
| static_relevant | bon_sc | 2 | 0 | 0 | 226.0 |
| static_relevant | bon_logprob | 2 | 0 | 0 | 226.0 |
| static_relevant | bon_consilience | 2 | 0 | 0 | 229.0 |
| static_relevant | single | 2 | 0 | 0 | 226.5 |
| static_relevant | is_sir_sc | 2 | 0 | 0 | 240.0 |
| static_relevant | is_met_sc | 2 | 0 | 0 | 240.0 |
| static_relevant | is_block_sc | 2 | 0 | 0 | 240.0 |
| static_relevant | is_block_consilience | 2 | 0 | 0 | 240.0 |
| static_unrelated | bon_logprob | 2 | 0 | 0 | 548.0 |
| static_unrelated | is_met_sc | 2 | 0 | 0 | 582.0 |
| static_unrelated | is_block_sc | 2 | 0 | 0 | 582.0 |
| static_unrelated | is_block_consilience | 2 | 0 | 0 | 582.0 |
| static_unrelated | single | 2 | 0 | 0 | 603.0 |
| static_unrelated | is_sir_sc | 2 | 0 | 0 | 547.0 |
| static_unrelated | bon_sc | 2 | 0 | 0 | 531.5 |
| static_unrelated | bon_consilience | 2 | 0 | 0 | 649.0 |

Paired differences below average seeds within each question; bootstrap intervals are descriptive and unadjusted.

- bon_consilience, inline_relevant − static_relevant: -8.837 F1 points, 95% CI [-19.104, +1.431].
- bon_consilience, inline_relevant − inline_unrelated: -2.663 F1 points, 95% CI [-12.155, +6.830].
- bon_consilience, static_relevant − static_unrelated: +7.676 F1 points, 95% CI [+7.646, +7.707].
- bon_logprob, inline_relevant − static_relevant: -1.404 F1 points, 95% CI [-4.568, +1.759].
- bon_logprob, inline_relevant − inline_unrelated: +5.465 F1 points, 95% CI [-1.714, +12.644].
- bon_logprob, static_relevant − static_unrelated: +2.995 F1 points, 95% CI [+1.807, +4.183].
- bon_sc, inline_relevant − static_relevant: -3.025 F1 points, 95% CI [-4.568, -1.481].
- bon_sc, inline_relevant − inline_unrelated: +5.465 F1 points, 95% CI [-1.714, +12.644].
- bon_sc, static_relevant − static_unrelated: +7.723 F1 points, 95% CI [+4.955, +10.491].
- is_block_consilience, inline_relevant − static_relevant: -0.110 F1 points, 95% CI [-4.892, +4.672].
- is_block_consilience, inline_relevant − inline_unrelated: +4.828 F1 points, 95% CI [-2.258, +11.915].
- is_block_consilience, static_relevant − static_unrelated: +2.200 F1 points, 95% CI [+1.364, +3.036].
- is_block_sc, inline_relevant − static_relevant: -0.311 F1 points, 95% CI [-4.892, +4.271].
- is_block_sc, inline_relevant − inline_unrelated: +9.018 F1 points, 95% CI [-1.604, +19.640].
- is_block_sc, static_relevant − static_unrelated: +2.200 F1 points, 95% CI [+1.364, +3.036].
- is_met_sc, inline_relevant − static_relevant: -0.311 F1 points, 95% CI [-4.892, +4.271].
- is_met_sc, inline_relevant − inline_unrelated: +9.018 F1 points, 95% CI [-1.604, +19.640].
- is_met_sc, static_relevant − static_unrelated: +2.200 F1 points, 95% CI [+1.364, +3.036].
- is_sir_sc, inline_relevant − static_relevant: -0.311 F1 points, 95% CI [-4.892, +4.271].
- is_sir_sc, inline_relevant − inline_unrelated: +9.018 F1 points, 95% CI [-1.604, +19.640].
- is_sir_sc, static_relevant − static_unrelated: +4.364 F1 points, 95% CI [+4.050, +4.678].
- single, inline_relevant − static_relevant: -4.769 F1 points, 95% CI [-4.969, -4.568].
- single, inline_relevant − inline_unrelated: -0.775 F1 points, 95% CI [-1.714, +0.164].
- single, static_relevant − static_unrelated: +3.892 F1 points, 95% CI [+3.433, +4.350].
- inline_relevant, bon_consilience − bon_sc: -7.697 F1 points, 95% CI [-22.892, +7.497].
- inline_relevant, bon_logprob − bon_sc: +0.000 F1 points, 95% CI [+0.000, +0.000].
- inline_relevant, is_block_consilience − bon_sc: +0.201 F1 points, 95% CI [+0.000, +0.402].
- inline_relevant, is_block_sc − bon_sc: +0.000 F1 points, 95% CI [+0.000, +0.000].
- inline_relevant, is_met_sc − bon_sc: +0.000 F1 points, 95% CI [+0.000, +0.000].
- inline_relevant, is_sir_sc − bon_sc: +0.000 F1 points, 95% CI [+0.000, +0.000].
- inline_relevant, single − bon_sc: -3.739 F1 points, 95% CI [-7.479, +0.000].
- inline_unrelated, bon_consilience − bon_sc: +0.430 F1 points, 95% CI [-1.046, +1.907].
- inline_unrelated, bon_logprob − bon_sc: +0.000 F1 points, 95% CI [+0.000, +0.000].
- inline_unrelated, is_block_consilience − bon_sc: +0.838 F1 points, 95% CI [+0.545, +1.131].
- inline_unrelated, is_block_sc − bon_sc: -3.553 F1 points, 95% CI [-6.996, -0.110].
- inline_unrelated, is_met_sc − bon_sc: -3.553 F1 points, 95% CI [-6.996, -0.110].
- inline_unrelated, is_sir_sc − bon_sc: -3.553 F1 points, 95% CI [-6.996, -0.110].
- inline_unrelated, single − bon_sc: +2.501 F1 points, 95% CI [+0.000, +5.001].
- static_relevant, bon_consilience − bon_sc: -1.885 F1 points, 95% CI [-5.268, +1.498].
- static_relevant, bon_logprob − bon_sc: -1.620 F1 points, 95% CI [-3.241, +0.000].
- static_relevant, is_block_consilience − bon_sc: -2.714 F1 points, 95% CI [-5.752, +0.324].
- static_relevant, is_block_sc − bon_sc: -2.714 F1 points, 95% CI [-5.752, +0.324].
- static_relevant, is_met_sc − bon_sc: -2.714 F1 points, 95% CI [-5.752, +0.324].
- static_relevant, is_sir_sc − bon_sc: -2.714 F1 points, 95% CI [-5.752, +0.324].
- static_relevant, single − bon_sc: -1.995 F1 points, 95% CI [-3.991, +0.000].
- static_unrelated, bon_consilience − bon_sc: -1.839 F1 points, 95% CI [-2.423, -1.254].
- static_unrelated, bon_logprob − bon_sc: +3.108 F1 points, 95% CI [+3.067, +3.148].
- static_unrelated, is_block_consilience − bon_sc: +2.809 F1 points, 95% CI [+1.703, +3.915].
- static_unrelated, is_block_sc − bon_sc: +2.809 F1 points, 95% CI [+1.703, +3.915].
- static_unrelated, is_met_sc − bon_sc: +2.809 F1 points, 95% CI [+1.703, +3.915].
- static_unrelated, is_sir_sc − bon_sc: +0.645 F1 points, 95% CI [+0.061, +1.229].
- static_unrelated, single − bon_sc: +1.836 F1 points, 95% CI [+0.605, +3.067].

Raw stop diagnostics: `{"candidate_answers": 96, "candidate_empty": 1, "candidate_truncated": 0, "immediate_injected_close": 48, "immediate_injected_eos": 0, "injected_root_calls": 84, "probe_calls": 7, "probe_thinking_eos": 0, "search_calls": 194, "thinking_eos": 1}`.

Offline audit: 2524 checks passed. Same search cap does not imply equal realized FLOPs; complete-pool IS and BoN use different amounts of work. Final answer readout and evidence extraction are additional measured costs.
