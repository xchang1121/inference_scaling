# Retrieval + inference scaling results

The 120-condition native pilot completed with 13197 passing audits. On four
fresh TechQA questions, SC-BoN adds about 1.5 F1 points over each retrieval
policy's single-path baseline, with paired intervals crossing zero. Dynamic
SC-BoN reaches 11.605 versus static SC-BoN 12.231. Block-Met IS K=2 adds only
0.256 points to dynamic single-path F1 while using about 2.95 times its native
dense-padded graph estimate. This pilot does not establish a stable gain.

See the [full observations](../../experiments/online-injection-scaling-20261109/reports/OBSERVATIONS.md),
[numeric report](../../experiments/online-injection-scaling-20261109/reports/REPORT.md),
[package and reproduction](../../experiments/online-injection-scaling-20261109/README.md),
and [frozen design](RETRIEVAL_SCALING_PROTOCOL.md).

Generation source: `46244219de2d608ecf9228dcddf1fe3541957be7`. Fixed Qwen3.8-27B,
A physical NPU3 only, TP1. Eight questions (four fresh, four diagnostic
regressions), three methods, five evidence views, one seed. The search/aux cap
is 8192 generated tokens plus one answer capped at 512, rather than matched
realized FLOPs. No empty answers or thinking EOS occurred; ten logical views
hit the 1024-token candidate horizon (eight actual units). Real/noise bodies
are exactly length matched, with titles and wrappers preserved.

All 119 execution units decode correctly. The archive preserves every case,
unselected candidate, auxiliary decision, budget charge and native resource
record. Literal support quotations did not ensure semantic support: two plans
guessed the wrong product, and some accepted windows supplied symptoms or the
wrong license edition. The next useful intervention is entity/fact consistency
and evidence coverage, with SC-BoN as the simpler search candidate.
