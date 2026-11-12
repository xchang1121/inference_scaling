# Empty-answer termination diagnostics

33 of 352 native measurements have an empty final answer.

Empty means the final answer is an empty string after removing special tokens and surrounding whitespace. It does not mean that no thinking, retrieval or interim answer was generated. All empty answers receive zero F1 and retain their full costs.

| Arm | Observed terminal phase | Empty answers |
|---|---|---:|
| thinkretrieve | Retrieval/continuation controller thinking | 9 |
| thinkretrieve | Selected IS thinking trajectory | 4 |
| thinkretrieve_noise | Retrieval/continuation controller thinking | 11 |
| thinkretrieve_noise | Selected IS thinking trajectory | 9 |

## Interpretation

- This records where generation ended, not why the model selected EOS. It cannot by itself establish that a template, retrieved content or reward caused the stop.
- A controller EOS is handled before IS starts. The runner materializes an EOS sentinel for the common final-answer interface; that sentinel is not an IS-generated token. These cases therefore report zero selected-suffix generated tokens.
- A selected IS trajectory ending in EOS may follow substantial search, including discarded candidates. Its short selected suffix does not imply low total compute.
- Interim answers drive retrieval and are not substituted for a missing final answer.
- Terminal positions are zero-based within the observed terminal segment. The CSV and JSON retain per-run costs and actual insertion counts.
