# Algorithm map

The implementation follows the mathematical objects in `HW_share/inference_scaling_article.tex`.

| Framework name | Article label | Invariant to preserve |
|---|---|---|
| `mh` | `alg:main-power-mh` | suffix MH leaves the fixed-length power target invariant |
| `conditional-is` | `alg:main-onpolicy-is` | candidates and completions are sampled from the base model |
| `base-replay` | `alg:main-replay-is` | candidates remain base samples; replay only estimates conditional energy |
| `dynamic-is` | `alg:main-dynamic-is` | auxiliary candidates receive the outer base/proposal probability ratio |

The replay algorithms implement the document's data lifecycle literally:

1. design data may choose policies, variances, and integer budgets;
2. evaluation records stay hidden until the design is frozen and are consumed at most once;
3. every fresh rollout used by a current decision moves to design data;
4. only independent post-selection reserve rollouts may become future evaluation records.

The dynamic implementation will use the documented continuous allocation before deterministic integer rounding:
candidate-level probability ratios multiply both replay and fresh variance terms, and each source is divided by the
square root of its per-sample cost.

