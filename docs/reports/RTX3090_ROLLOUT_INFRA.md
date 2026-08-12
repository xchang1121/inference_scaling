# RTX 3090 推理基础设施优化汇总

这份报告只讨论实现层优化如何改变墙钟时间、逻辑 FLOPs、吞吐和 rollout 复用，不承担方法准确率排序。
方法质量、pass@k、共享奖励目标和质量相关消融统一见
[GSM8K 方法效果与准确率](GSM8K_3090_ALIGNED_RESULTS.md)。本报告同时收拢早期完整网格中与 infra
直接相关的 replay、缓存、连续批处理结果，以及新增 rollout 加速栈的三随机种子消融。

## 计量口径

所有成对因子统一写为“优化路径 / 对照路径”：小于 1 表示减少，大于 1 表示增加。墙钟排除模型与数据
加载；FLOPs 按 `2 × 参数量 × 实际 forward token slots` 估算。该 FLOPs 是跨实现可核对的逻辑计算量，
不包含 attention 的长度二次项、逐元素 kernel、CPU token tree、tokenization 和调度开销，因而不能
替代墙钟。

报告包含两组不能直接混算的实验：

| 实验组 | 用途 | setting | 重复 |
| --- | --- | --- | --- |
| GSM8K 完整网格 | replay、动态候选、连续批处理和训练成本摊销 | 32 道固定 test 题，Qwen2.5-1.5B-Instruct，FP32，最长 192 token | 固定请求级随机数 |
| rollout infra 消融 | 历史 token tree、负载调度、progressive、run-ahead 与 SMC forest | 固定公开 test 第 1311 题，Qwen2.5-1.5B-Instruct，BF16，最长 64 token，16-token block | 3 个独立 seed |

第一组覆盖真实的 32 题算法调用，但部分 CUDA batch 形状会令 token trace 分叉；只有输出可比时才把
墙钟解释为相同 workload 的吞吐变化。第二组刻意缩小任务，只诊断 infra 因果关系，不使用单题 reward
排列方法质量。

## 影响总览

在当前硬件和 setting 上，能够明确观察到的正向结果有三类：

- 连续批处理把同一方法的墙钟降到逐 prompt 路径的 `0.206×–0.952×`，但逻辑 FLOPs 没有减少；
- warm replay 在缓存已经存在时，把 fresh-only 的在线 FLOPs 降到 `0.766×`、墙钟降到 `0.859×`，
  但包含建库的首次执行分别为 `2.341×` 和 `1.807×`；
- SMC forest 的条件后缀复用相对同一 SMC 不复用版，把墙钟降到 `0.856×`、主模型 FLOPs 降到
  `0.963×`，同时把 fresh rollout 均值从 35.3 降至 24.7。

其余机制在当前 workload 中主要表现为保护或负面消融：active-batch 调度避免静态草稿的严重退化，
但没有形成显著净加速；pilot/evaluation 分离保障统计流程，流式奖励与 run-ahead 用于隐藏昂贵 CPU
奖励空泡，它们在候选成本同质且 verifier 极便宜的任务上都更慢。下面逐项给出分母和适用边界。

## 完整 GSM8K 网格中的实现优化

### 连续批处理

8 个 prompt worker 共享一张 RTX 3090，并保留一次算法调用中的重复前缀组。墙钟因子的分母是同一方法
的逐 prompt 同步路径；FLOPs 因子为连续批处理 / 同步路径。

| 方法 | 同步墙钟 | 连续批处理墙钟 | 墙钟因子 | FLOPs 因子 | 数值答案一致 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 95.4 s | 19.7 s | 0.206× | 1.177× | 32/32 |
| Best-of-8 | 136.9 s | 67.9 s | 0.496× | 1.021× | 30/32 |
| 标准条件 IS | 427.1 s | 370.0 s | 0.866× | 1.008× | 32/32 |
| 0.5B proposal 条件 IS | 419.5 s | 399.5 s | 0.952× | 1.003× | 32/32 |

Base 的物理 batch 最容易填满，因此收益最大；IS 内部含多阶段依赖，跨 prompt 合并能够覆盖的串行段
更少。连续批处理提高的是 GPU 利用率，padding 和 batch 分叉甚至略微增加逻辑 slots，所以不能写成
算法 FLOPs 缩减。Best-of-8 的两条数值答案发生分叉，该行只支持相同配置 workload 的吞吐比较。

### warm rollout replay

对照固定为 8 个 base 候选、每候选 3 条总 rollout。warm 路径最多读取 2 条已评分历史记录，并始终
保留 1 条 fresh base rollout。

| 路径 | 推理 PFLOPs | 墙钟 | 相对 fresh-only FLOPs | 相对 fresh-only 墙钟 |
| --- | ---: | ---: | ---: | ---: |
| fresh-only | 1.3483 | 422.5 s | 1.000× | 1.000× |
| warm cache 在线阶段 | 1.0326 | 362.9 s | 0.766× | 0.859× |
| cache build + 首次 warm 查询 | 3.1563 | 763.4 s | 2.341× | 1.807× |

因此，23.4% 的在线 FLOPs 缩减和 14.1% 的在线墙钟缩减只适用于缓存已经由此前请求或独立异步资源
生产的阶段。把建库完整计入后，本轮 replay-key 覆盖率需要到第 7 次重复查询才同时在 FLOPs 和墙钟
上回本。warm 与 fresh-only 的准确率差为 -3.125 个百分点，配对区间跨 0；质量解释留在准确率报告。

### 动态候选、缓存与方差—成本预算

这组 oracle 诊断固定 8 个候选、48-token block、最长 192 token 和每个非终止候选 3 条 evaluation
rollout。三条路径依次加入 `base/0.5B` defensive proposal、候选层精确 IS、历史 evaluation 库，以及
独立 design rollout 驱动的方差—成本预算。

| 路径 | 实际复用率 | 稳态 PFLOPs | 稳态墙钟 | 一次性总 PFLOPs | 一次性总墙钟 |
| --- | ---: | ---: | ---: | ---: | ---: |
| base 候选 + 固定 fresh | 0% | 2.0973 | 705.7 s | 2.0973 | 705.7 s |
| 动态候选 + 固定 replay | 34.943% | 1.9312 | 704.0 s | 4.8764 | 1,401.5 s |
| 动态候选 + 方差—成本分配 | 5.707% | 2.3172 | 706.6 s | 7.2951 | 2,186.2 s |

动态固定组相对 base 固定组的稳态 FLOPs 因子为 `0.921×`，墙钟因子为 `0.998×`：小模型承担了一部分
工作，但未形成实质墙钟加速。计入 cache build 后，首次总 FLOPs 为 base 固定组的 `2.325×`。

方差—成本组相对动态固定组的稳态 FLOPs 因子为 `1.200×`、墙钟因子为 `1.004×`；计入 cache 和
design 后，一次性总 FLOPs 为 `1.496×`。当前每来源 2 条 design rollout 的估计噪声较大，分配器只
实际复用 5.707% rollout，平均最终 ESS 还略低，因此这一版本没有显示效率收益。它说明“可使用历史
数据”不等于“预算分配器会有效利用历史数据”。

### 训练与免训练推理的累计成本

GRPO 的一次训练成本为 15.646 PFLOPs、5,007,660 个前向等价 token slots 和 9,545.2 秒。只比较累计
FLOPs 时，verifier-MH、标准 verifier-IS 和 0.5B proposal verifier-IS 与“训练 + GRPO 推理”的交点
分别为 392、344 和 230 次查询。由于三种方法与 GRPO 的准确率绝对差均超过预设 5 个百分点阈值，
这些只是计算账本交点，不能写成“达到相同效果所需查询数”。

## rollout 加速栈的三随机种子消融

下图误差线是三个独立 seed 的样本标准差。cache build、在线关键路径和后台 drain 分列；主模型 FLOPs
覆盖 prefill、decode、评分及被拒绝草稿的 target verification slots。

![RTX 3090 rollout 基础设施消融](../assets/rtx3090_rollout_infra.svg)

### 历史 token tree 与负载感知调度

解码请求依次使用 active batch 4、2、1，模拟 rollout 尾部逐渐变稀。

| 路径 | 在线墙钟（s） | 输出 token/s | 主模型 PFLOPs | cache build（s） | 草稿接受率 | 墙钟因子 | FLOPs 因子 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 无草稿 | 3.838 ± 0.059 | 116.8 ± 1.8 | 0.00247 ± 0.00000 | 0.000 ± 0.000 | 0.0% | 1.000× | 1.000× |
| 历史树，始终草稿 | 8.297 ± 0.075 | 54.0 ± 0.5 | 0.00411 ± 0.00001 | 1.280 ± 0.027 | 13.1% | 2.162× | 1.660× |
| 历史树，负载感知 | 3.783 ± 0.067 | 118.5 ± 2.1 | 0.00249 ± 0.00001 | 1.335 ± 0.058 | 37.5% | 0.986× | 1.006× |

低命中历史树若始终草稿，target 会验证大量随后丢弃的 token，墙钟和 FLOPs 都明显退化。KV 裁剪避免
拒绝后重新 prefill；负载策略在 batch 4 和 2 保留普通批处理，只在 batch 1 长尾启用草稿，因而保护了
基线吞吐。`0.986×` 与三 seed 离散程度相比不能视为稳定加速，它首先是一条防退化策略。

### progressive、run-ahead 与 SMC rollout forest

算法层使用 3 个候选、每候选 2 条总 rollout 预算；SMC 使用 3 个粒子、每粒子 2 个分支。

| 路径 | 在线墙钟（s） | 在线主模型 PFLOPs | cache build（s） | 后台 drain（s） | fresh / reused rollout |
| --- | ---: | ---: | ---: | ---: | ---: |
| 固定 rollout 条件 IS | 3.442 ± 0.162 | 0.00809 ± 0.00067 | 1.248 ± 0.057 | 0.000 ± 0.000 | 0.0 / 0.0 |
| pilot/evaluation 分离 | 5.313 ± 0.231 | 0.01250 ± 0.00000 | 1.269 ± 0.100 | 0.000 ± 0.000 | 0.0 / 0.0 |
| 流式奖励 + run-ahead | 5.605 ± 0.958 | 0.01261 ± 0.00192 | 1.256 ± 0.068 | 7.763 ± 1.927 | 0.0 / 0.0 |
| SMC forest，不复用 | 2.985 ± 0.260 | 0.01271 ± 0.00038 | 1.250 ± 0.080 | 0.000 ± 0.000 | 35.3 / 0.0 |
| SMC forest，复用 | 2.571 ± 0.510 | 0.01226 ± 0.00206 | 1.263 ± 0.042 | 0.000 ± 0.000 | 24.7 / 13.3 |

pilot/evaluation 分离相对固定 rollout 的在线墙钟因子为 `1.544×`、FLOPs 因子为 `1.553×`。它的作用
是先用 pilot 冻结预算，再用独立 evaluation 样本形成最终估计；在候选成本近似相同的小 workload 上，
额外阶段没有可换取的效率收益。

流式奖励 + run-ahead 相对纯 progressive 的在线墙钟因子为 `1.051×`、主模型 FLOPs 因子约为
`1.009×`，另有 `7.763 ± 1.927 s` 后台 drain。本实验的正则数值 verifier 几乎没有 CPU 尾部，后台
工作没有空泡可隐藏。run-ahead 因此默认关闭，只有 reward、通信或 KV 管理存在可测空泡时才应启用。

SMC rollout forest 的复用版相对同 setting 不复用版为 `0.856×` 墙钟和 `0.963×` 主模型 FLOPs；
fresh rollout 均值减少约 30.0%。复用只继承与所选子 block 匹配的条件后缀，库存不足仍由 fresh base
rollout 补齐，因此该比较没有把不满足当前条件的旧后缀冒充有效样本。

## vLLM 复现实验状态

同一入口支持 `--backend vllm`：常驻 `AsyncLLM` 使用原生 global suffix tree，并读取 vLLM 原生
drafted/accepted 计数，把被拒绝的验证 token 计回主模型 FLOPs。active-batch 动态表只在专门的
load-aware arm 中启用；算法层默认使用静态 suffix，避免把实验性动态调度开销混入算法比较。

当前 RTX 3090 位于 Windows 主机且未安装 WSL，vLLM 不原生支持 Windows，因此报告不填造 vLLM
硬件数值。在 Linux/WSL2 上可生成相同 schema，再由同一汇总器处理：

```bash
export PYTHONPATH=src
python experiments/benchmark_rollout_infra.py \
  --output results/infra/rtx3090_vllm.json \
  --backend vllm --dtype bfloat16 --section all
```

## 部署结论

| 机制 | 当前建议 | 原因 |
| --- | --- | --- |
| 连续批处理 | 默认开启，并按方法分别标定并发度 | 明确降低墙钟，尤其适合独立 Base 请求 |
| warm replay | 有稳定重复 key 或异步建库时开启 | 在线 FLOPs 与墙钟均下降，但冷启动昂贵 |
| 历史 token tree | 必须搭配 active-batch/接受率门控 | 静态低命中草稿会严重退化 |
| pilot/evaluation 分离 | 需要自适应预算的统计路径中保留 | 保证估计流程清楚，不保证同质任务加速 |
| run-ahead | 默认关闭，测到 CPU/通信空泡后再开 | 廉价 verifier 上增加在线和 drain 成本 |
| SMC 条件后缀复用 | 当前最值得继续扩大 workload 验证 | 同算法成对比较同时减少墙钟、FLOPs 和 fresh rollout |
| 方差—成本分配 | 暂不作为加速默认项 | 当前 design 预算没有提高 ESS 或复用效率 |

机器可读来源分别为
[`results/gsm8k_3090/`](../../results/gsm8k_3090/) 中的 `compute`、`replay`、`dynamic_is` 与
`async_grouped` 汇总，以及 [`results/infra/`](../../results/infra/) 中的六份独立 seed 结果和
`rtx3090_transformers_summary.json`。
