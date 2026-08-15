# rollout 生成、复用与验证

本文汇总 rollout 数据流。统计公式见[推理算法实现](ALGORITHMS.md)，执行与成本定义见
[推理基础设施实现](INFRASTRUCTURE.md)。

## 数据流

```text
历史 rollout ──┬──> replay store ──> 带 behavior probability 的 IS 估计
               └──> token tree ──> target 验证的 speculative decoding

过量提交 rollout ──> completion broker ──> 完整样本 / 可续跑部分状态

base 候选 ──> pilot ──> 冻结 evaluation 预算 ──> 独立 evaluation
                                                   ├──> 条件 IS
                                                   └──> SMC block 权重

MH 状态 ──┬──> base / replay-mixture 后缀 proposal
          ├──> accept/reject 分支预取
          └──> surrogate 第一阶段 ──> 精确奖励第二阶段
```

| 数据用途 | 概率要求 | 生命周期 |
| --- | --- | --- |
| IS/replay evaluation | 保存真实 behavior probability | 冻结后一次性消费 |
| variance/cost design | 保存 reward、概率和成本 | 持久 design pool |
| token-tree draft | 由 target verifier 校正 | 可复用 |
| broker partial | 保存 token、seed、behavior/reference 概率 | 完成后形成 trajectory |
| run-ahead draft | 计入后台成本 | 写入 token tree |

## 历史 token tree

`RolloutTokenTree` 保存“后缀 context → 下一 token 计数”。确定性模式提出最高频 token；随机模式从
经验分布 $`q_t`$ 提出 token $`a`$，接受概率为

$$
\min\left\{1,\frac{p_t(a)}{q_t(a)}\right\}.
$$

拒绝后从

$$
\frac{(p_t(v)-q_t(v))_+}{\sum_w(p_t(w)-q_t(w))_+}
$$

抽取替代 token。接受路径贡献 $`\min(p_t,q_t)`$，拒绝路径贡献
$`p_t-\min(p_t,q_t)`$，输出分布为 target $`p_t`$。

Transformers 一次验证 `prefix + drafts`，裁剪 `DynamicCache` 至已接受位置后继续生成。vLLM 使用
global suffix proposer 和原生 target verifier。报告记录 tree hit、draft acceptance、target
verification slots、墙钟与 cache build。

草稿长度可写成 active batch $`b`$ 的分段函数 $`K(b)`$：

```toml
[acceleration.speculation]
enabled = true
tiers = [[1, 8], [4, 0], [512, 0]]
min_context_tokens = 2
min_token_probability = 0.10
tree_max_context_tokens = 24
tree_max_contexts = 100000
dynamic_vllm = false
stochastic_tree = false
```

每个 tier 为 `[最大 active batch, K]`。固定 $`K`$ 与动态 $`K(b)`$ 分别构成实验臂。RTX 3090 结果中，
`batch=1` 门控用于限制低接受率草稿在大 batch 下的验证成本。

## 部分 rollout broker

`AsyncRolloutBroker` 将长请求拆成固定 token chunk。过量提交产生的部分轨迹保存以下字段：

- 原请求与已生成 token；
- behavior/reference token log-probability；
- continuation seed、优先级与分段数。

调度恢复使用 `original prefix + saved tokens`。完整轨迹触发 completion callback 并形成
`ReplayRecord`；部分轨迹保持为 broker 状态。Transformers 恢复包含 prefix prefill，vLLM 可通过 APC
命中 prefix block。成本字段为 saved tokens、resumed prefill、forward slots 与墙钟。

## pilot、evaluation 与流式执行

对候选前缀 $`s_i`$，条件能量为

$$
h(s_i)=\mathbb E_{z\sim p(\cdot\mid s_i)}
\left[\exp\!\left(\frac{r(s_i,z)}{\tau}\right)\right].
$$

pilot 估计单样本方差和成本；冻结 evaluation 数量 $`m_i`$ 后，最终估计使用独立样本：

$$
\widehat h_i=\frac1{m_i}\sum_{j=1}^{m_i}
\exp\!\left(\frac{r(s_i,z_{ij})}{\tau}\right).
$$

当 $`m_i\to\infty`$ 时，$`\widehat h_i\xrightarrow{p}h(s_i)`$。候选数有限时，归一化权重由连续映射
定理收敛。

`FrozenStreamingISEstimator` 在生成前冻结 fresh request id。completion 以任意顺序到达，每个 id
消费一次；固定样本 multiset 决定最终 log energy、ESS 和选择概率。`StreamingRewardEvaluator` 在
序列完成时提交 CPU/verifier 任务，使奖励计算与剩余 decode 重叠。

`LowPriorityRunAheadBackend` 在奖励等待期间生成历史草稿。请求按有界 chunk 执行，前台请求在当前
chunk 后取得调度权。后台 token、前台等待和 drain 独立计量。

## 奖励目标 MH 的执行路径

| 路径 | 执行 | 统计校正 | 主要成本字段 |
| --- | --- | --- | --- |
| proposal-tree 预取 | 奖励等待期间为接受/拒绝状态各生成下一 proposal | 普通 Hastings 判断消费对应分支 | used/unused proposal、FLOPs、墙钟 |
| delayed acceptance | surrogate 第一阶段早拒绝，精确奖励第二阶段 | 第二阶段补回 exact-surrogate 差 | surrogate/exact calls、early rejection |
| replay-mixture proposal | base 与冻结历史后缀混合 | 新旧后缀的混合概率进入 Hastings 比 | history hit、评分 slots、cache build |

预取以额外 proposal 隐藏奖励延迟。delayed acceptance 保持 proposal 计算量，减少精确奖励调用。
replay-mixture 将历史命中的自回归生成替换为 teacher-forced 批量评分。

## SMC rollout forest

粒子 $`s`$ 生成 block $`a`$ 后使用增量权重

$$
G(s,a)=\frac{\widehat h(sa)}{\widehat h(s)}.
$$

路径乘积满足

$$
\prod_tG(s_{t-1},a_t)=\widehat h(s_T).
$$

父 rollout 以 block $`a`$ 开头时，其余后缀是子前缀 $`sa`$ 下的条件 rollout。重采样产生多个相同
branch 时，reservoir 在副本间分桶，缺口由 fresh rollout 补足。有限粒子、branch factor 和 rollout
数形成 SMC 近似误差；诊断字段为 ESS、fresh/reused rollout 和 reservoir 命中。

## 后端能力

| 能力 | Transformers | vLLM |
| --- | --- | --- |
| 历史草稿 | 确定性/随机 token tree | global suffix proposer |
| KV 续算 | `DynamicCache` crop | 引擎内部 verifier/cache |
| active-batch $`K(b)`$ | 每批查询 | 动态 suffix 适配层 |
| 完成回调 | batch 返回后触发 | `AsyncLLM` 完成即触发 |
| broker 恢复 | token 状态 + prefill | token 状态 + APC |
| replay-mixture 评分 | Transformers 精确评分 | 原生或委托精确评分 |
| draft 指标 | 仓库计数 | vLLM metrics |

vLLM global suffix cache 收集同一引擎处理过的请求。其他模型产生的 off-policy 数据可进入 replay
estimator 或 replay-mixture MH；外部经验 token tree 的随机残差校正在 Transformers 路径执行。

## 测量入口

```powershell
$env:PYTHONPATH = "src;."
.\.venv\Scripts\python experiments\benchmark_rollout_infra.py `
  --backend transformers --dtype bfloat16 --section all `
  --output results\infra\rtx3090_transformers.json

.\.venv\Scripts\python experiments\benchmark_is_mh_reuse.py `
  --backend transformers --dtype bfloat16 --section all `
  --output results\infra\rtx3090_transformers_is_mh.json
```

```bash
export PYTHONPATH=src
python experiments/benchmark_rollout_infra.py \
  --backend vllm --dtype bfloat16 --section all \
  --output results/infra/rtx3090_vllm.json
```

后端比较固定模型、dtype、数据哈希、长度、预算和 GPU 数。报告将 cache build、在线路径、后台 drain、
主模型 slots、FLOPs 与墙钟分列。RTX 3090 Transformers 结果见
[推理执行与 rollout 复用实验](../reports/RTX3090_ROLLOUT_INFRA.md)。
