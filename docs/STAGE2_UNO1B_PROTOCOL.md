# Stage 2：Uno-1B checkpoint 级复现协议

## 问题拆分

“复现 Uno 加速”不是一个二元问题。本阶段将结论拆成三个彼此不能替代的层级：

1. **checkpoint/算法复现**：公开 adapter 能否按论文的 gated LoRA、uniform noise 和
   linear $\Psi$-Spec 工作，并得到 $\mathrm{TPF}>1$；
2. **本机系统复现**：在 RTX 3090 上，计入 draft 和 verify 后的实际 decode tokens/s 是否超过
   同一 checkpoint 的 AR；
3. **论文数字复现**：是否在作者的 H200、批处理和 fused runtime 条件下接近论文吞吐。

Windows Hugging Face 回退版可以回答 1，并提供一个明确标注 backend 的 2 的实验点；只有官方
Nano-vLLM runtime 才能回答 2 的正式版本。单卡 3090 不具备回答 3 的硬件条件。

## 锁定对象

| 对象 | revision | 文件 SHA-256 |
| --- | --- | --- |
| `IFM/K2-Horizon-0.9B` | `ee770e713760cf6350e4322cdbbff91a163b7d70` | `6392cc67...d876c365` |
| `IFM/K2-Horizon-0.9B-Uno` | `b0d8896a301a2f4bc755538b1234a35100da50d0` | `5a499229...dedfe4e` |
| `ifm-ai/uno` | `ed2ee36bb7a3aea8732ebc635b3f09490a032ea3` | Git commit |

完整 hash 位于 `references/stage2_environment.lock.json`，运行器默认拒绝 hash 不匹配的权重。

adapter 配置为 rank 128、$\alpha=2048$、dropout 0.05，并覆盖每层
`q/k/v/o/gate/up/down` 七个 projection，共应找到 196 个 LoRA-A hook 和 392 个 A/B tensor。

## 上游兼容性审计

官方 runtime 固定 Linux x86-64、Python 3.10、PyTorch 2.11.0+cu128、Triton 3.6 和
FlashAttention 2.8.3/3。当前主机是 Windows，且没有已配置 WSL 发行版，因此官方命令在 import
阶段之前就不满足平台前置条件。这不是显存不足：1B base、adapter 和 KV cache 均能放入 24 GiB。

另一个独立问题是，锁定 base 的 `config.json` 写有 `transformers_version=4.57.1`，但同一 revision
的远程 Python 配置导入 5.x 名称 `PreTrainedConfig`。在 4.57.1 上实测 `AutoConfig` 失败。官方
Nano-vLLM 的 `hf_compat.load_model_config` 直接读取原始 JSON，刻意绕过远程实现；完整 HF 回退模型
则使用 `transformers>=5.14,<6`。这两个环境必须分开记录，不能把版本差异静默修补掉。

## 回退实现与官方语义的对应

设进入一轮时，完整历史最后一个 token 为尚未缓存的 seed $s$，缓存长度为 $P$，block size 为
$B$。

### Draft forward

输入为

$$
[s,z_1,\ldots,z_{B-1}],\qquad z_i\sim\operatorname{Uniform}\{1,\ldots,64255\}.
$$

每层 LoRA-A 输出乘 token mask

$$
m=(0,1,\ldots,1).
$$

因此 seed 行严格使用 base AR，噪声行使用 base + Uno adapter。因 attention 是 causal，未来噪声
也不能影响 seed。draft 后缓存长度从 $P$ 变为 $P+B$，随即裁掉 $B-1$ 个噪声 KV，只保留 base
seed，回到 $P+1$。

### Verify forward

从 seed 后的 base cache 输入

$$
[y_0,y_1,\ldots,y_{B-1}],
$$

其中 $y_0$ 是 seed 行免费得到的 AR sample，$y_{1:B-1}$ 来自 noisy rows。verify 完全关闭 adapter。
对每个 speculative token 保存生成时的旧过滤分布 $q_t$，并使用

$$
a_i=\min\left(1,\frac{p_i(y_i)}{q_{t,i}(y_i)}\right).
$$

首次拒绝时从 $[p_i-q_{t,i}]_+$ 采 correction；全部接受时从最后一个 verify logit 采 lookahead。
在线更新后的 $q_{t+1}$ 永远不会传给当前 verifier。

verify 后只保留已提交序列除最后一个 token 之外的 KV。若本轮提交 $K$ 个 token，则从 verify cache
裁掉 $B+1-K$ 个 token。这一不变量由运行时断言检查。

### 采样

默认与官方自由推理入口一致：temperature 1.0、top-k 50、top-p 0.95。先在 BF16 logits 上取 top-k，
再把这 50 个值提升到 FP32 做 softmax 和 nucleus 截断。proposal 与 verifier 必须应用相同变换。

`mask_token_id=64256` 在 uniform-noise 模式只是开区间上界，不是送进 embedding 的 token；实际噪声
范围是 `[1,64256)`，所以不会越过模型合法 ID `[0,64256)`。

## 预注册实验矩阵

正式小规模矩阵：

| 因子 | 值 |
| --- | --- |
| backend | AR / HF Uno linear |
| block size | 2, 4, 8, 16 |
| sampling | temperature 1, top-k 50, top-p 0.95 |
| output budget | 64，忽略 EOS 保持配对长度 |
| batch | 1 |
| warmup | 每条路径至少一次 |
| repetitions | 10，奇偶重复反转方法顺序 |

每次记录 prefill/decode/end-to-end 时间、TPF、spec acceptance、lookahead、峰值显存和 token IDs。
报告 median、IQR、min/max。后续增加 prompt 数和 bootstrap CI，不用一次短生成下结论。

## 判定规则

- `routing_probe.clean_rows_max_abs_difference <= 1e-5` 且 noise 行差异非零，否则 checkpoint 路由失败；
- 任一 $B>1$ 的 median TPF $>1$，称为“Uno checkpoint 的算法加速机制已复现”；
- 只有 decode TPS 的配对/置信区间也超过 AR，才称为“该 backend 在本机实现 wall-clock 加速”；
- HF 回退版即便 TPS 更高，也必须带 `huggingface_pytorch_kv_cache_fallback` 限定语；
- 不能把本机结果外推为论文 H200/batch throughput。

## 官方 Linux 路线

在用户提供 Linux/WSL2 后，按锁定 commit 建 Python 3.10 环境，在 Ampere 上先选择 FA2 linear sampler，
不用 Hopper 专用 FA3 tree sampler。先运行上游单元测试，再用相同 prompt/采样矩阵分别运行 base AR 和
Uno。原始 JSON、环境 manifest、`nvidia-smi` 和 profiler trace 保留在本地，仓库只提交小型汇总。
