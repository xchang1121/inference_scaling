# Retrieval + inference scaling pilot

Completed: True; 120/120 conditions; 119 unique execution units; 13197 checks, 0 failures.

F1 is normalized answer token F1 (%), not exact-answer accuracy. Question counts: heldout=4, regression=4; one seed. Equal generated-token caps do not imply equal realized work or FLOPs. Normalized EM is in summary.csv.

Times cover measured execution units, including online dynamic lookup. Model startup, corpus-index construction and offline static-packet preparation are excluded. PFLOPs are native dense-padded graph estimates, not hardware-counter measurements of every operation.

| Cohort | Evidence / method | F1 % | Mean work | Mean s | Mean PFLOPs | Empty | Thinking EOS | Forced close |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| heldout | closed.none/block_met_sc | 7.446 | 3442.8 | 70.38 | 0.667 | 0 | 0 | 0 |
| heldout | closed.none/sc_bon | 9.797 | 4392.2 | 93.30 | 0.680 | 0 | 0 | 0 |
| heldout | closed.none/single | 10.363 | 528.2 | 42.88 | 0.371 | 0 | 0 | 0 |
| heldout | dynamic.relevant/block_met_sc | 10.345 | 3822.2 | 91.65 | 1.223 | 0 | 0 | 0 |
| heldout | dynamic.relevant/sc_bon | 11.605 | 3704.5 | 88.07 | 0.808 | 0 | 0 | 1 |
| heldout | dynamic.relevant/single | 10.089 | 705.2 | 48.40 | 0.415 | 0 | 0 | 0 |
| heldout | dynamic.unrelated/block_met_sc | 10.018 | 4135.5 | 95.07 | 1.218 | 0 | 0 | 0 |
| heldout | dynamic.unrelated/sc_bon | 10.244 | 4161.5 | 97.32 | 0.993 | 0 | 0 | 1 |
| heldout | dynamic.unrelated/single | 10.896 | 788.0 | 52.93 | 0.448 | 0 | 0 | 0 |
| heldout | static.relevant/block_met_sc | 10.906 | 2786.5 | 61.43 | 0.922 | 0 | 0 | 0 |
| heldout | static.relevant/sc_bon | 12.231 | 3282.2 | 66.78 | 0.651 | 0 | 0 | 0 |
| heldout | static.relevant/single | 10.712 | 443.0 | 33.40 | 0.322 | 0 | 0 | 0 |
| heldout | static.unrelated/block_met_sc | 9.988 | 4595.2 | 99.31 | 1.362 | 0 | 0 | 0 |
| heldout | static.unrelated/sc_bon | 10.855 | 4926.8 | 87.47 | 0.697 | 0 | 0 | 1 |
| heldout | static.unrelated/single | 11.602 | 538.0 | 42.87 | 0.519 | 0 | 0 | 0 |
| regression | closed.none/block_met_sc | 16.584 | 5660.0 | 106.00 | 1.108 | 0 | 0 | 2 |
| regression | closed.none/sc_bon | 14.628 | 6658.8 | 113.77 | 0.867 | 0 | 0 | 0 |
| regression | closed.none/single | 13.646 | 716.0 | 49.80 | 0.441 | 0 | 0 | 0 |
| regression | dynamic.relevant/block_met_sc | 20.192 | 4372.2 | 106.03 | 1.037 | 0 | 0 | 1 |
| regression | dynamic.relevant/sc_bon | 23.232 | 4641.8 | 112.85 | 0.934 | 0 | 0 | 1 |
| regression | dynamic.relevant/single | 17.108 | 776.0 | 57.06 | 0.504 | 0 | 0 | 0 |
| regression | dynamic.unrelated/block_met_sc | 19.154 | 5502.2 | 123.90 | 1.444 | 0 | 0 | 0 |
| regression | dynamic.unrelated/sc_bon | 18.909 | 5222.5 | 118.31 | 0.951 | 0 | 0 | 2 |
| regression | dynamic.unrelated/single | 15.995 | 880.8 | 63.83 | 0.639 | 0 | 0 | 0 |
| regression | static.relevant/block_met_sc | 22.429 | 3429.2 | 70.34 | 0.916 | 0 | 0 | 0 |
| regression | static.relevant/sc_bon | 26.641 | 3614.5 | 72.13 | 0.667 | 0 | 0 | 0 |
| regression | static.relevant/single | 24.015 | 562.2 | 40.03 | 0.294 | 0 | 0 | 0 |
| regression | static.unrelated/block_met_sc | 17.924 | 5197.0 | 102.44 | 1.355 | 0 | 0 | 0 |
| regression | static.unrelated/sc_bon | 14.176 | 6426.5 | 116.44 | 1.023 | 0 | 0 | 1 |
| regression | static.unrelated/single | 13.995 | 859.0 | 61.54 | 0.536 | 0 | 0 | 0 |

Paired question-bootstrap contrasts (percentage points):

- heldout: closed.none/sc_bon minus closed.none/single: -0.566, 95% [-3.894, +2.963], n=4.
- heldout: closed.none/block_met_sc minus closed.none/single: -2.917, 95% [-8.993, +2.902], n=4.
- heldout: static.relevant/sc_bon minus static.relevant/single: +1.519, 95% [-1.464, +5.357], n=4.
- heldout: static.relevant/block_met_sc minus static.relevant/single: +0.194, 95% [-1.835, +2.311], n=4.
- heldout: dynamic.relevant/sc_bon minus dynamic.relevant/single: +1.517, 95% [-0.578, +2.824], n=4.
- heldout: dynamic.relevant/block_met_sc minus dynamic.relevant/single: +0.256, 95% [-1.964, +2.476], n=4.
- heldout: dynamic.relevant/single minus static.relevant/single: -0.624, 95% [-3.055, +1.807], n=4.
- heldout: dynamic.relevant/sc_bon minus static.relevant/sc_bon: -0.626, 95% [-5.519, +3.637], n=4.
- heldout: dynamic.relevant/block_met_sc minus static.relevant/block_met_sc: -0.562, 95% [-4.284, +3.161], n=4.
- regression: closed.none/sc_bon minus closed.none/single: +0.981, 95% [-1.788, +3.655], n=4.
- regression: closed.none/block_met_sc minus closed.none/single: +2.938, 95% [-2.351, +9.180], n=4.
- regression: static.relevant/sc_bon minus static.relevant/single: +2.626, 95% [-1.722, +7.354], n=4.
- regression: static.relevant/block_met_sc minus static.relevant/single: -1.586, 95% [-2.890, +0.543], n=4.
- regression: dynamic.relevant/sc_bon minus dynamic.relevant/single: +6.124, 95% [+2.202, +10.046], n=4.
- regression: dynamic.relevant/block_met_sc minus dynamic.relevant/single: +3.084, 95% [+0.886, +5.020], n=4.
- regression: dynamic.relevant/single minus static.relevant/single: -6.907, 95% [-18.787, +3.552], n=4.
- regression: dynamic.relevant/sc_bon minus static.relevant/sc_bon: -3.409, 95% [-9.239, +2.653], n=4.
- regression: dynamic.relevant/block_met_sc minus static.relevant/block_met_sc: -2.237, 95% [-16.220, +10.261], n=4.
