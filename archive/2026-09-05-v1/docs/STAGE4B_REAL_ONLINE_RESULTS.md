# Stage 4B 结果：真实 Uno-1B request-local 在线更新

## 结论先行

Stage 4B 完成了真实 `K2-Horizon-0.9B-Uno` checkpoint 上的 post-verification 在线更新、GPU 计时和
预注册检验。结果是一个重要的**负结果**：

- **安全门通过**：45/45 条路径输出恰好 384 tokens，hash、392-tensor adapter 映射、tokenwise
  routing、cache frontier、数值有限性和 optimizer 参数隔离均通过；
- **预注册学习门未通过**：stride-10 的 paired TPF ratio 为 0.9796，95% CI
  $[0.9470,1.0000]$，没有稳定提高真实 filtered acceptance/推进量；
- **预注册 HF 系统门未通过，而且显著变慢**：paired decode TPS ratio 为 0.9624，95% CI
  $[0.9533,0.9819]$；15 对中只赢 2 对；
- **探索性 stride-20 不确定**：TPF ratio 1.0068，CI $[0.9574,1.0286]$；TPS ratio
  0.9860，CI $[0.9243,1.0075]$，两者都跨 1。

因此不能声称“Online Uno 真模型加速已经实现”。准确结论是：

> 一个严格 lossless、只训练 0.526M request-local 参数的真实 checkpoint 原型已经实现；它在部分长序列
> pilot 和部分正式 pair 上能提高 TPF，但预注册的跨 prompt 正式实验未能泛化，并在 HF fallback 上使
> 主策略显著减速。Stage 3 的 tabular 可行性并不足以证明任意神经 fast head 都会回本。

## 正式设置

- GPU：RTX 3090 24 GiB，BF16；PyTorch 2.13.0+cu130；
- 临时 checkpoint 环境：Transformers 5.16.1、PEFT 0.20.0；
- backend：HF SDPA + DynamicCache，不是官方 Nano-vLLM/Triton；
- block size 8，temperature 1、top-k 50、top-p 0.95；
- output 384 tokens、ignore stop、batch 1；
- 三个 prompt：English explanation、Python/code、中文数学；
- 5 repetitions，共 15 paired workloads、45 runs；
- cyclic Latin rotation 使 static/s10/s20 在运行位置上平衡；
- 主策略 s10，探索对照 s20；rank 8、LR 0.005、top-50 union、on-policy feedback；
- 30,000 次 paired median percentile bootstrap。

## 完整结果

### 绝对中位数

| 方法 | TPF | acceptance | decode tok/s | peak allocated VRAM |
| --- | ---: | ---: | ---: | ---: |
| static Uno | 1.3299 | 0.4017 | 33.87 | 2.752 GB |
| online s10 | 1.2939 | 0.3702 | 32.38 | 2.773 GB |
| online s20 | 1.3116 | 0.3882 | 33.80 | 2.773 GB |

### 配对结果

| 方法 | TPF ratio [95% CI] | acceptance delta [95% CI] | TPS ratio [95% CI] | speed wins |
| --- | ---: | ---: | ---: | ---: |
| **s10（预注册）** | 0.9796 [0.9470, 1.0000] | -0.0203 [-0.0509, 0.0026] | **0.9624 [0.9533, 0.9819]** | 2/15 |
| s20（探索） | 1.0068 [0.9574, 1.0286] | +0.0068 [-0.0382, 0.0237] | 0.9860 [0.9243, 1.0075] | 6/15 |

s10 的 speed exact sign-test 描述值为 $p=0.00739$，方向是**变慢**而不是变快。这个检验同样只覆盖
当前 15 对固定 workload，不能外推到所有 prompt。

online peak allocated VRAM 相对 static 增加约 20.6--20.9 MB；0.526M FP32 fast parameters、AdamW
state、feedback 和临时 buffer 在 24 GiB 3090 上不是容量瓶颈。

## 时间分解

| 方法 | head / decode | feedback / decode | update / decode | 三者合计 |
| --- | ---: | ---: | ---: | ---: |
| s10 | 0.104% | 1.194% | 2.174% | 3.479% |
| s20 | 0.107% | 1.232% | 1.689% | 3.021% |

批量化前的 512-token pilot 中，显式 online components 约占 6.0%；batched finite/top-k 和 padded
low-rank einsum 把它降至约 3.5%。所以系统负结果已不是单纯的逐行 Python 实现事故：s10 的主要问题是
真实 TPF 本身没有泛化改善，随后 3.5% online 开销进一步放大减速。

update 尝试的中位数是 s10 14 次、s20 7 次。s10 的 212 次总 update 中 25 次 rollback、20 次
static-shadow reset；s20 的 101 次中 12 次 rollback、1 次 reset。护栏确实在工作，但 same-buffer
held-out validation 无法保证 update 对**未来** verifier context 有益。

## Prompt 分解

每个 prompt 只有 5 对，因此以下是诊断而非新的确认性检验。

| 方法 / prompt | TPF ratio median | TPS ratio median | acceptance delta median |
| --- | ---: | ---: | ---: |
| s10 / English | 1.0068 | 0.9819 | +0.0026 |
| s10 / code | 0.9653 | 0.9499 | -0.0353 |
| s10 / 中文数学 | 0.9470 | 0.9576 | -0.0549 |
| s20 / English | 1.0268 | 1.0023 | +0.0237 |
| s20 / code | 1.0000 | 0.9605 | 0.0000 |
| s20 / 中文数学 | 0.9795 | 0.9717 | -0.0201 |

English continuation 对更稀疏的 s20 最友好；code 和中文数学暴露明显负迁移。因为 fast head 共享一组
position-agnostic low-rank mapping，来自过去少量 verifier row 的梯度会同时改变许多未来 context 的
常见 token logit。top-K union surrogate 在同一小 batch 上下降，并不等于下一轮 filtered TV 会下降。

## 为什么 2-pair pilot 看起来成功

优化后的 512-token、2-pair pilot 中，s10 TPF ratio 点估计为约 1.053，TPS ratio 约 1.049。正式实验
增加到 15 对并加入 code/中文 prompt 后反转。原因不是正式代码偷换：同一学习率、loss、stride、block 和
sampling 在协议冻结后未变。真正原因是：

1. 两对样本的区间没有领域覆盖，point estimate 方差很大；
2. 长序列中的 acceptance drift 依 prompt/seed 变化，不总朝同一方向；
3. same-buffer validation 只测局部拟合，不测 temporal generalization；
4. offline Uno 已经很强，无明显 drift 时在线更新容易把噪声当信号。

这正是预注册、多 prompt 和静态配对的价值：pilot 用来发现可能性，不能当最终结果。

## 安全实现得到验证

尽管性能门失败，本阶段把 Online Uno 的安全骨架落到了真实 checkpoint：

- 所有 30 条 online 路径均报告 526,336 个 fast trainable parameters；
- `trainable_base_parameter_tensors=0`、`base_optimizer_overlap=0` 在 30/30 路径成立；
- 公开 Uno adapter 路由仍为 196 hooks，clean row 最大差异 0，noise row mean/max 差异
  4.612/20.625；
- PEFT base-only context 退出后的 392 个 adapter tensor 会被重新冻结；
- K2 remote model 不返回 hidden states 时，由 lm-head pre-hook 捕获实际最后 hidden；
- proposal 使用 materialized `FilteredDistribution`；verify 完成前 learner 不可见 feedback；
- cache draft rollback 与 post-verify frontier 每轮断言；
- non-finite/validation regression 会 rollback，static shadow 显著更好时 reset。

lossless 指的是输出分布保持 AR target，不是每个 seed 与 static 产生同一 token 序列。proposal 改变后随机
耦合自然不同；正确性来自保存旧 $q_t$ 的 rejection/residual 机制以及 Stage 1 的枚举/Monte Carlo 证据。

## 下一版算法：延迟激活而非同批批准

正式负结果指向一个具体改造，而不是盲目扫更多学习率。Stage 5 的候选 update 不应立即替换 active head：

$$
\delta_t^{active}
\xrightarrow[\text{past feedback}]{\text{train}}
\tilde\delta_t^{candidate},
$$

在下一批**时间上真正未来的** verifier feedback 到来时，同时评估 active/candidate/zero-static：

$$
a^*=arg\min_{a\in\{active,candidate,zero\}}
\widehat{D}_{TV/KL}^{future}(a).
$$

只有 candidate 比 active 超过置信/成本 margin 才 promote；zero 更好则 reset。当前轮仍用 active 的旧
$q_t$ 完成 exact verification，因此延迟选择不破坏 lossless。这个 shadow-candidate controller 直接处理
本阶段的 temporal generalization 失败，并把动作定义为 `no update / promote / reset`，而不是假设每个
gradient step 都应该上线。

随后再比较：

- exact filtered overlap/TV 作为 promotion 指标，而非只看 raw union surrogate；
- per-position gate 或顶部层 rank-4 LoRA，降低跨 context 干扰；
- drift detector 只在 static acceptance 持续下降时开启训练；
- s20/cooldown/update budget，减少 optimizer 开销；
- 按 prompt/domain 持久化的 replay 与 request-local cold-start 的 break-even。

## 结论边界

- 本阶段确实更新了真实 Uno checkpoint 后的 fast head，但没有更新完整 rank-128 diffusion LoRA；
- wall-clock 是 RTX 3090 HF fallback 的真实计时，不是官方 Nano-vLLM；
- 15 pairs 覆盖三个固定 prompt，已经足以否定本次预注册主声明，但不足以断言所有在线 Uno 都失败；
- s20/English 的局部正信号是下一轮假设，不是已复现的系统加速。

原始结果见 `results/stage4b_online_uno1b_rtx3090_hf.json`，完整性、配对区间、prompt 分解和冻结判定见
`results/stage4b_online_uno1b_rtx3090_hf_analysis.json`。
