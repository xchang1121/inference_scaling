# FLARE-gap step-boundary experiment

Completed conditions: 176/176. Structural audit failures: 0.

Work cap 1024 (all previews/checks included), answer cap 512, evidence cap 256. Token F1 is a generative overlap diagnostic, not official TechQA accuracy. One seed per question; cohort sizes are small.

## heldout

| Arm | n | F1 % | Injected | EOS in thinking | Empty | Immediate close | Work tokens | Aux tokens | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| closed.none | 8 | 9.95 | 0 | 0 | 0 | 0 | 568.4 | 0.0 | 54.4 |
| static.relevant | 8 | 14.06 | 8 | 0 | 0 | 0 | 537.1 | 0.0 | 47.8 |
| static.unrelated | 8 | 10.69 | 8 | 0 | 0 | 0 | 688.4 | 0.0 | 62.0 |
| legacy_sc.relevant | 8 | 11.09 | 7 | 0 | 0 | 6 | 168.4 | 0.0 | 30.9 |
| legacy_sc.unrelated | 8 | 8.53 | 7 | 0 | 0 | 7 | 155.4 | 0.0 | 24.6 |
| step_sc.relevant | 8 | 11.19 | 7 | 0 | 0 | 0 | 670.2 | 0.0 | 61.6 |
| step_sc.unrelated | 8 | 10.13 | 7 | 0 | 0 | 0 | 713.4 | 0.0 | 64.1 |
| flare_gap.relevant | 8 | 11.38 | 1 | 0 | 0 | 1 | 557.0 | 46.5 | 54.9 |
| flare_gap.unrelated | 8 | 9.56 | 1 | 0 | 0 | 1 | 557.0 | 46.5 | 55.2 |
| flare_gap_rebuild.relevant | 8 | 11.43 | 1 | 0 | 0 | 0 | 568.8 | 46.5 | 55.6 |
| flare_gap_rebuild.unrelated | 8 | 9.09 | 1 | 0 | 0 | 0 | 630.6 | 46.5 | 60.4 |

- flare_gap.relevant − static.relevant: -2.68 pp F1; paired question bootstrap 95% [-5.92, +0.87].
- flare_gap.relevant − legacy_sc.relevant: +0.29 pp F1; paired question bootstrap 95% [-2.20, +3.05].
- flare_gap_rebuild.relevant − step_sc.relevant: +0.24 pp F1; paired question bootstrap 95% [-4.69, +6.67].
- step_sc.relevant − legacy_sc.relevant: +0.11 pp F1; paired question bootstrap 95% [-3.94, +4.25].
- flare_gap_rebuild.relevant − flare_gap.relevant: +0.05 pp F1; paired question bootstrap 95% [+0.00, +0.16].
- flare_gap_rebuild.unrelated − flare_gap.unrelated: -0.48 pp F1; paired question bootstrap 95% [-1.43, +0.00].
- static.relevant − static.unrelated: +3.37 pp F1; paired question bootstrap 95% [-1.92, +10.66].
- legacy_sc.relevant − legacy_sc.unrelated: +2.56 pp F1; paired question bootstrap 95% [-0.38, +5.72].
- step_sc.relevant − step_sc.unrelated: +1.06 pp F1; paired question bootstrap 95% [-1.37, +3.82].
- flare_gap.relevant − flare_gap.unrelated: +1.82 pp F1; paired question bootstrap 95% [+0.00, +5.45].
- flare_gap_rebuild.relevant − flare_gap_rebuild.unrelated: +2.35 pp F1; paired question bootstrap 95% [+0.00, +7.05].

## regression

| Arm | n | F1 % | Injected | EOS in thinking | Empty | Immediate close | Work tokens | Aux tokens | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| closed.none | 8 | 15.38 | 0 | 0 | 0 | 0 | 630.4 | 0.0 | 56.8 |
| static.relevant | 8 | 15.56 | 8 | 0 | 0 | 0 | 551.5 | 0.0 | 51.2 |
| static.unrelated | 8 | 13.31 | 8 | 0 | 0 | 0 | 693.4 | 0.0 | 63.9 |
| legacy_sc.relevant | 8 | 13.81 | 8 | 1 | 1 | 3 | 424.9 | 0.0 | 47.1 |
| legacy_sc.unrelated | 8 | 8.77 | 8 | 0 | 0 | 5 | 207.1 | 0.0 | 28.7 |
| step_sc.relevant | 8 | 14.73 | 8 | 0 | 0 | 0 | 710.4 | 0.0 | 62.4 |
| step_sc.unrelated | 8 | 14.39 | 8 | 0 | 0 | 0 | 705.6 | 0.0 | 64.3 |
| flare_gap.relevant | 8 | 13.33 | 1 | 0 | 1 | 1 | 569.1 | 44.8 | 52.2 |
| flare_gap.unrelated | 8 | 15.03 | 1 | 0 | 0 | 1 | 569.1 | 44.8 | 55.7 |
| flare_gap_rebuild.relevant | 8 | 14.81 | 1 | 0 | 0 | 0 | 665.4 | 44.8 | 60.0 |
| flare_gap_rebuild.unrelated | 8 | 15.30 | 1 | 0 | 0 | 0 | 648.8 | 44.8 | 59.1 |

- flare_gap.relevant − static.relevant: -2.23 pp F1; paired question bootstrap 95% [-8.11, +2.28].
- flare_gap.relevant − legacy_sc.relevant: -0.48 pp F1; paired question bootstrap 95% [-5.41, +3.82].
- flare_gap_rebuild.relevant − step_sc.relevant: +0.09 pp F1; paired question bootstrap 95% [-3.13, +4.04].
- step_sc.relevant − legacy_sc.relevant: +0.92 pp F1; paired question bootstrap 95% [-2.14, +3.99].
- flare_gap_rebuild.relevant − flare_gap.relevant: +1.49 pp F1; paired question bootstrap 95% [+0.00, +4.46].
- flare_gap_rebuild.unrelated − flare_gap.unrelated: +0.26 pp F1; paired question bootstrap 95% [+0.00, +0.79].
- static.relevant − static.unrelated: +2.25 pp F1; paired question bootstrap 95% [-0.05, +4.65].
- legacy_sc.relevant − legacy_sc.unrelated: +5.03 pp F1; paired question bootstrap 95% [-0.74, +13.06].
- step_sc.relevant − step_sc.unrelated: +0.33 pp F1; paired question bootstrap 95% [-3.53, +3.58].
- flare_gap.relevant − flare_gap.unrelated: -1.70 pp F1; paired question bootstrap 95% [-5.11, +0.00].
- flare_gap_rebuild.relevant − flare_gap_rebuild.unrelated: -0.48 pp F1; paired question bootstrap 95% [-1.45, +0.00].

## Accounting and limits

Unique executed units: 180; measured unit time 6499.5s, native padded dense FLOP estimate 138.690 PFLOP. Warmup is separate. Model load, smoke, failed prep and container lifetime appear in provenance receipts.

Each condition's cost includes its full common stem plus cold branch prefill. Study totals deduplicate shared execution. A no-intervention real/noise pair reuses one result and is not an independent replicate. Unrelated bodies are substituted after the real-evidence support gate. Primary FLARE-gap appends after a complete assistant step. Its rebuild companion shares the same common stem/payload and moves references to the user region. Step SC also reconstructs the user region; legacy SC retains assistant-inline placement. Query, timing and gating still differ between trigger policies.

Source labels and wrappers stay identical in paired content controls. EOS, empty answers, capped outputs, parse failures and unsupported retrievals remain in the denominator. An empty answer may follow an immediate answer-phase EOS after a valid thinking close, as well as EOS inside thinking; these cases are separate in the data. Bootstrap intervals do not cover sampling-seed variability or generalization beyond this small sample.
