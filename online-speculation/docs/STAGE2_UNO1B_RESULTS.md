# Stage 2 结果：RTX 3090 上的 Uno-1B checkpoint 复现

## 结论先行

本阶段得到两个肯定结论和两个明确的否定边界：

- **checkpoint/算法级复现通过。** 公开 Uno-1B adapter 在真实 0.9B base 上得到显著高于 1 的
  TPF；最好的 $B=8$ 为 1.401，paired-bootstrap 95% 区间为 $[1.341,1.432]$。
- **本机 Hugging Face KV-cache 回退 backend 的 wall-clock 加速通过。** $B=8$ 相对 AR 的配对
  decode speedup 中位数为 $1.352\times$，95% bootstrap 区间 $[1.250,1.386]$，10/10 对运行均
  更快。
- **官方 Nano-vLLM runtime 尚未执行。** 当前 Windows 缺 Triton/FlashAttention，且上游固定 Linux
  Python 3.10；`model_runner` 的实际 import 在 `tree_builder -> import triton` 处失败。
- **论文 H200 吞吐没有复现。** 单张 RTX 3090、batch 1、HF backend 的数字不能与论文的 H200
  批处理数字直接比较。

因此，准确表述是：

> 本机已经复现 Uno 的真实 checkpoint、接受机制和一个有统计余量的 1.35 倍 HF-backend
> batch-1 decode 加速；尚未复现作者的 fused runtime 或 H200 吞吐。

## 完整性检查

正式运行前完成以下 fail-closed 检查：

| 检查 | 结果 |
| --- | --- |
| base revision / SHA-256 | `ee770e...` / `6392cc67...d876c365`，通过 |
| adapter revision / SHA-256 | `b0d889...` / `5a499229...dedfe4e`，通过 |
| adapter tensor 映射 | 392 expected / 392 loaded / 0 missing |
| LoRA-A token hooks | 196，即 28 层 × 7 projections |
| clean/seed logits 与 base 最大差异 | 0.0 |
| noise logits 是否改变 | 是；mean abs 4.612，max abs 20.625 |
| 每轮 cache frontier | draft/rollback/verify 后均通过运行时断言 |
| 正式运行条数 | 5 methods × 10 repetitions = 50 |
| 每条输出长度 | 全部恰好 64 tokens |

公开 adapter 的 safetensors 使用 `model.layers...lora_A.weight` 格式，而 PEFT 对远程
`K2HorizonForCausalLM` 自动创建的参数是
`base_model.model.model.layers...lora_A.default.weight`。直接调用 `PeftModel.from_pretrained`
只发 warning，却会留下 392 个 missing keys。正式实现不接受这个 silent failure，而是显式逐键映射，
检查集合、形状和计数均完全一致后才开始推理。

## 实验设置

- GPU：NVIDIA RTX 3090 24 GiB，compute capability 8.6；
- PyTorch：2.13.0+cu130；Transformers：5.16.1；PEFT：0.20.0；
- backend：普通 Hugging Face SDPA + `DynamicCache`，batch 1；
- prompt：`Explain in three concise paragraphs why speculative decoding can be lossless.`；
- sampling：temperature 1.0、top-k 50、top-p 0.95、uniform noise；
- output：固定 64 tokens，忽略 stop token；
- block size：2、4、8、16；
- 每条路径先 warmup；正式 10 次，奇偶 repetition 反转方法顺序；
- 区间：对相同 repetition/seed 的 AR 与 Uno 做 50,000 次 paired percentile bootstrap；
- 计时：prefill 和 decode 分开，所有边界调用 `torch.cuda.synchronize()`。

## 结果

原始中位数与 IQR：

| 方法 | TPF median | spec accept median | decode tok/s median [IQR] | end-to-end tok/s | 峰值显存 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AR | 1.000 | — | 27.31 [26.73, 27.75] | 27.29 | 2.721 GB |
| Uno $B=2$ | 1.212 | 0.442 | 32.51 [31.61, 33.36] | 32.36 | 2.720 GB |
| Uno $B=4$ | 1.312 | 0.425 | 34.93 [32.61, 36.42] | 34.75 | 2.721 GB |
| Uno $B=8$ | **1.401** | 0.444 | **36.84 [34.39, 37.69]** | **36.54** | 2.722 GB |
| Uno $B=16$ | 1.370 | 0.450 | 36.00 [32.21, 37.59] | 35.80 | 2.726 GB |

配对 bootstrap：

| 方法 | TPF median [95% CI] | AR-relative decode speedup [95% CI] | TPS delta [95% CI] | wins |
| --- | ---: | ---: | ---: | ---: |
| $B=2$ | 1.212 [1.167, 1.260] | 1.194× [1.160, 1.250] | +5.28 [4.21, 6.72] | 10/10 |
| $B=4$ | 1.312 [1.260, 1.435] | 1.269× [1.206, 1.369] | +7.34 [5.35, 10.17] | 10/10 |
| $B=8$ | **1.401 [1.341, 1.432]** | **1.352× [1.250, 1.386]** | **+9.79 [6.79, 10.48]** | 10/10 |
| $B=16$ | 1.370 [1.286, 1.466] | 1.328× [1.209, 1.385] | +8.90 [5.41, 10.72] | 10/10 |

每个配置都是 10/10 配对胜出；描述性的双侧 exact sign-test 为 $p=0.001953$。这个 $p$ 值只描述
同一 prompt 下的十个运行对，不能替代跨 prompt/领域样本。

## 如何理解结果

TPF 不是逐 token acceptance rate。即使单个 examined speculative token 的接受率约 0.42--0.45，
每轮也总有一个免费 AR token，并在首次拒绝时提交 residual correction；若全接受还会多一个 lookahead。
$B=8$ 每两个 forward 中位数提交约 2.80 个 token，因此 TPF 约 1.40。

$B=16$ 没有继续改善，因为 accepted prefix 没有随 block 等比例增长；拒绝后的远端并行位置成为浪费，
而更长 block 的 forward 仍需付计算成本。这个结果支持后续 online controller 同时调 update stride 和
block size，而不是固定选择最大 $B$。

$B=8$ 的 TPF 1.401 高于实际 speedup 1.352，差异来自 draft 的 LoRA 开销、较长 block forward、
sampling/verifier Python 开销和 cache 管理。这再次说明不能用接受率或 TPF 代替真实 tokens/s。

## 限制与下一步

当前正式矩阵仍只有一个短 prompt、batch 1 和 64-token continuation。bootstrap 重采样的是十个
seed/repetition，只反映运行波动，不覆盖 prompt/domain shift。下一阶段先做可控 distribution-drift
仿真，比较 static、per-round、fixed-stride 与 adaptive-stride；之后在真实模型上引入 request-local
fast weights。任何 online 版本都要超过本阶段 $B=8$ 的 36.84 tok/s 静态基线，而不只是提高
acceptance。

原始数据见 `results/stage2_uno1b_rtx3090_hf.json`，配对分析见
`results/stage2_uno1b_rtx3090_hf_analysis.json`，官方 runtime 失败点见
`results/stage2_official_runtime_probe.json`。
