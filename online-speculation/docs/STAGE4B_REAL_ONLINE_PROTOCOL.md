# Stage 4B：K2-Horizon-0.9B-Uno 在线更新真机预注册协议

本协议在正式运行前冻结。Stage 4B 的问题不是再次证明 tabular learner 能适应 drift，而是检验真实公开
Uno checkpoint 上的 request-local fast residual 是否同时提高 TPF 和 RTX 3090 wall-clock throughput。

## 1. 工程 pilot 与正式数据的分界

以下运行只用于调通或优化实现，不进入正式统计：

- 32-token smoke 首先发现 PEFT `disable_adapter()` 退出时会恢复 392 个 LoRA tensor 的
  `requires_grad=True`；实现现于每次 base-only context 后强制重冻，并由回归测试覆盖；
- 第二次 smoke 发现 K2 remote code 接受 `output_hidden_states=True` 却返回 `None`；现从 frozen
  `lm_head` 的输入 hook 捕获最后 hidden，不依赖远程返回字段；
- 128-token、2-repetition pilot 确认真实 update/rollback 能运行，但序列太短，TPF 没有稳定收益；
- 512-token、2-repetition pilot 首次看到 stride-10 TPF ratio 约 1.053。逐 row 版本仍被 feedback/update
  吃掉收益，随后只做等价的批量化：每 block 一次 finite check/batched top-k，update padding variable
  support 后一次 low-rank einsum。复测 explicit online components 从约 6.0% 降到约 3.6%。

pilot 允许修 bug 和消除等价 Python/kernel 开销，但不允许据两对样本选择正式成功阈值。正式开始后不再改
学习率、loss、prompt、stride 或方法顺序；任何后续算法修改必须成为新阶段/新结果文件。

## 2. 锁定模型与 backend

| 对象 | revision / backend |
| --- | --- |
| base | `IFM/K2-Horizon-0.9B@ee770e713760cf6350e4322cdbbff91a163b7d70` |
| offline Uno | `IFM/K2-Horizon-0.9B-Uno@b0d8896a301a2f4bc755538b1234a35100da50d0` |
| runtime | Hugging Face SDPA + DynamicCache fallback |
| GPU | RTX 3090 24 GiB，BF16 |

运行器继续 fail-closed 校验 Stage 2 锁定的 base/adapter SHA-256 和 392/392 adapter tensor 映射。
官方 Nano-vLLM/Triton backend 仍未在 Windows 上运行，因此本阶段的 wall-clock 结论必须带
`HF fallback` 限定。

## 3. 冻结与 lossless 时序

每个 request 新建 rank-8 fast head：

$$
q_t=\operatorname{FilterSoftmax}
\left(\ell_{\theta_{AR}+\phi_{Uno}}+
\frac{8}{8}B_tA_th_t\right).
$$

$B_0=0$，故第一轮与 static Uno 一致。base 和公开 Uno adapter 永远 `requires_grad=False`，optimizer
parameter IDs 必须精确等于 $\{A_t,B_t\}$ 且与模型参数集合无交集。每轮严格执行：

$$
\boxed{
q_t\text{ sample}
\rightarrow
\text{save filtered }q_t
\rightarrow
\text{verify/correct with }q_t
\rightarrow
\text{materialize feedback}
\rightarrow
\delta_{t+1}
}
$$

更新后的 head 永远不能进入当前轮 acceptance denominator。模型 forward 全在 `inference_mode`；只有
detached hidden 上的 0.526M fast parameters 建 autograd graph。

## 4. 固定 online 配置

| 项目 | 值 |
| --- | --- |
| block size | 8（7 speculative rows） |
| sampling | temperature 1.0，top-k 50，top-p 0.95 |
| fast head | rank 8，alpha 8，zero-up initialization |
| optimizer | AdamW，LR $5\times10^{-3}$，无 weight decay |
| feedback | verifier/draft top-50 union，最多 100 IDs/row |
| supervision | on-policy，包含首次 rejection row |
| position weight | $0.97^i$ |
| loss | forward KL + 0.5 TV + 0.15 old-q KL + $10^{-6}\|B\|^2$ |
| validation | 每 5 items 取 1 个 held-out |
| reset | current validation 比 zero/static shadow 差 5% 时清零 |
| rollback | update 后 validation 恶化超过 1% 或非 finite |
| gradient clip | global norm 1.0 |
| decay | 1.0，本阶段不额外衰减 |

`online_s10` 是预注册主策略；`online_s20` 是成本更低的探索性对照。Stage 3 的 on-policy 策略在合成成本
下最好，而 TTS 给 stride 10 现实先验，因此不再把 full/discounted/stride-5 塞进本阶段正式矩阵。

## 5. Workload 与方法顺序

每条输出固定 384 tokens、batch 1、忽略 stop 以保持配对长度。三个 prompt 覆盖不同 token/推理形态：

1. English explanation：`Explain in three concise paragraphs why speculative decoding can be lossless.`
2. Python/code：`Implement a production-quality Python LRU cache from scratch. Explain the invariants, analyze complexity, and include tests for edge cases.`
3. 中文数学：`请严格推导为什么 Metropolis-Hastings 算法以目标分布为平稳分布，说明 detailed balance、不可约性与非周期性的作用，并给出一个离散状态空间例子。`

重复 5 次，seed 为 `20260905 + 1000 * repetition + prompt_index`，共 15 个 paired workloads、每种方法
5,760 output tokens。三种方法为 static、online_s10、online_s20；每个 prompt/repetition 按
`(repetition + prompt_index) mod 3` 做 cyclic Latin rotation，使每种方法在第一/第二/第三运行位置各出现
五次，减轻温度、时钟和 allocator 顺序偏差。正式方法前分别 warm static 路径和至少一次真实 online
backward；request-local optimizer 的实际首次状态分配仍计入每次请求。

## 6. 指标与统计

每条路径记录：

- exact output tokens、cycle/forward、TPF、spec acceptance；
- prefill/decode/end-to-end seconds 和 peak allocated VRAM；
- head forward、feedback materialization、transactional update seconds；
- feedback item、update/applied/rollback/reset 数、fast-weight norm；
- parameter isolation report 和逐 update train/validation/static-shadow loss。

相同 repetition/prompt 的 online/static 做配对比值。主统计为 15 个配对值的中位数，30,000 次 percentile
bootstrap 95% CI。该区间覆盖这三个固定 prompt 和 seeds 的运行变化，不代表开放域总体，也不把三个 prompt
当 15 个独立语义领域。

## 7. 预注册判定

对主策略 `online_s10`：

1. **安全门**：15/15 输出长度 384；hash/routing/cache/parameter isolation 全通过；无 NaN；
2. **真实模型学习门**：paired TPF ratio 的 95% CI 下界 $>1$；
3. **HF 系统门**：paired decode TPS ratio 的 95% CI 下界 $>1$；
4. 若 estimate $>1$ 但区间跨 1，只写“point estimate 有益、证据不足”；
5. `online_s20` 的任何最好结果均为 exploratory，不能替代主门。

即使第 3 门通过，也只能称“RTX 3090 Hugging Face fallback 上的 online fast-residual 加速”，不能声称
完整 diffusion LoRA 在线训练、官方 Nano-vLLM 或论文 H200 配置已经加速。

## 8. 正式命令

```powershell
.\.venv\Scripts\python -m online_speculation.hf_online_uno `
  --model-path ..\.tmp_k2_horizon_09b `
  --adapter-path ..\.tmp_k2_horizon_09b_uno `
  --block-size 8 --update-strides 10,20 `
  --max-new-tokens 384 --warmup-tokens 16 --repetitions 5 `
  --prompt "Explain in three concise paragraphs why speculative decoding can be lossless." `
  --prompt "Implement a production-quality Python LRU cache from scratch. Explain the invariants, analyze complexity, and include tests for edge cases." `
  --prompt "请严格推导为什么 Metropolis-Hastings 算法以目标分布为平稳分布，说明 detailed balance、不可约性与非周期性的作用，并给出一个离散状态空间例子。" `
  --feedback-top-k 50 --supervision on_policy --rank 8 --alpha 8 `
  --learning-rate 0.005 --bootstrap-samples 30000 --ignore-stop `
  --output .\online-speculation\results\stage4b_online_uno1b_rtx3090_hf.json
```
