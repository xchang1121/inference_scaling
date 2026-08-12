# RTX 3090 rollout 基础设施消融

## 实验设置

该实验只回答基础设施成本问题，不把单道题的准确率当作方法效果结论。三次运行使用固定的 Qwen2.5-1.5B-Instruct、BF16、公开且校验过哈希的 GSM8K test 第 1311 题、64 token 上限、16 token block，以及相同请求级随机数构造；仅改变被消融的基础设施。解码层依次提交 active batch 4、2、1，模拟 rollout 尾部逐渐变稀。算法层使用 3 个候选、每候选 2 条总 rollout 预算；SMC 使用 3 个粒子、每粒子 2 个分支。

主模型计算量按 `2 × 参数量 × 实际 target forward token slots` 估算；它覆盖 prefill、decode、评分以及被拒绝草稿的 target 验证，未包含 attention 的长度二次项、CPU token tree、采样和奖励解析。cache build、在线路径和后台 drain 分开报告。奖励使用标准答案 verifier 仅为稳定诊断算法关系，不能视为可部署 setting。

![RTX 3090 rollout 基础设施消融](../assets/rtx3090_rollout_infra.svg)

## 解码层结果

| 路径 | 在线墙钟时间（s） | 输出 token/s | 主模型 PFLOPs | cache build（s） | 草稿接受率 | 相对无草稿墙钟 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 无草稿 | 3.838 ± 0.059 | 116.8 ± 1.8 | 0.00247 ± 0.00000 | 0.000 ± 0.000 | 0.0% | 1.000× |
| 历史树，始终草稿 | 8.297 ± 0.075 | 54.0 ± 0.5 | 0.00411 ± 0.00001 | 1.280 ± 0.027 | 13.1% | 2.162× |
| 历史树，负载感知 | 3.783 ± 0.067 | 118.5 ± 2.1 | 0.00249 ± 0.00001 | 1.335 ± 0.058 | 37.5% | 0.986× |

静态草稿相对无草稿更慢，因为低接受率会让 target 一次验证多个随后丢弃的 token；这条对照说明不能把 speculative decoding 默认当作加速。KV 裁剪避免拒绝后重新 prefill，而负载策略在 batch 4 和 2 保留普通批处理、只在 batch 1 长尾启用草稿。其平均在线时间基本回到无草稿基线，同时只增加很少的 target FLOPs。BF16 下单请求验证与批量基线的数值路径不同，因此 token trace 不要求逐条完全相等；理论分布不因草稿改变，FP32 有限状态测试另外验证固定随机流的一致性。

## 算法层结果

| 路径 | 在线墙钟时间（s） | 在线主模型 PFLOPs | cache build（s） | 后台 drain（s） | fresh / reused rollout |
| --- | ---: | ---: | ---: | ---: | ---: |
| 固定 rollout 条件 IS | 3.442 ± 0.162 | 0.00809 ± 0.00067 | 1.248 ± 0.057 | 0.000 ± 0.000 | 0.0 / 0.0 |
| pilot/evaluation 分离 | 5.313 ± 0.231 | 0.01250 ± 0.00000 | 1.269 ± 0.100 | 0.000 ± 0.000 | 0.0 / 0.0 |
| 流式奖励 + run-ahead | 5.605 ± 0.958 | 0.01261 ± 0.00192 | 1.256 ± 0.068 | 7.763 ± 1.927 | 0.0 / 0.0 |
| SMC forest，不复用 | 2.985 ± 0.260 | 0.01271 ± 0.00038 | 1.250 ± 0.080 | 0.000 ± 0.000 | 35.3 / 0.0 |
| SMC forest，复用 | 2.571 ± 0.510 | 0.01226 ± 0.00206 | 1.263 ± 0.042 | 0.000 ± 0.000 | 24.7 / 13.3 |

pilot/evaluation 分离相对一次性固定 rollout 的在线墙钟因子为 `1.544×`，FLOPs 因子为 `1.553×`。两阶段必须先完成 pilot 再冻结预算，因此在候选成本近似相同、奖励解析很便宜的这个小 workload 上没有速度收益；它保留的价值是避免自适应预算直接读取 evaluation 值，并在异质 proposal、变长 rollout 或 replay 成本差异明显时重新分配预算。

流式奖励 + run-ahead 相对纯 progressive 的在线墙钟因子为 `1.051×`。本实验的正则数值 verifier 几乎没有 CPU 尾部，后台工作没有足够空泡可隐藏；其 drain 已单列，因此不能把预生成 token 写成免费。默认配置保持 run-ahead 关闭，只有实测 reward/KV 空泡足够大时才开启。

SMC rollout forest 的复用版相对不复用版墙钟因子为 `0.856×`，主模型 FLOPs 因子为 `0.963×`。这是直接同算法、同粒子和分支 setting 的加速比较；复用来自上一层 lookahead 中与所选子 block 匹配的条件后缀，缺少的粒子仍用 fresh base rollout 补齐。

## vLLM 复现实验状态

同一入口支持 `--backend vllm`：常驻 `AsyncLLM` 使用原生 global suffix tree，并从 vLLM 原生计数器读取 drafted/accepted token，把被拒绝的验证 token 加回主模型 FLOPs。active-batch 动态表只在专门的 load-aware arm 中显式启用；算法层默认使用静态 suffix，避免把 vLLM 0.25 的实验性动态调度开销混入算法比较。当前 RTX 3090 位于 Windows 主机，未安装 WSL；vLLM 不原生支持 Windows，因此这里没有伪造 vLLM 数值。安装 WSL2/Linux 环境后可用下方命令生成同 schema 的结果，再交给同一汇总器：

```bash
export PYTHONPATH=src
python experiments/benchmark_rollout_infra.py \
  --output results/infra/rtx3090_vllm.json \
  --backend vllm --dtype bfloat16 --section all
```

## 结论边界

- 历史序列进入 draft tree 不等于再次进入统计 estimator；base 会验证每个草稿 token。
- pilot 只冻结 evaluation 预算，pilot reward 不进入最终条件能量均值。
- 墙钟更短不一定意味着 FLOPs 更少：SMC 的大 batch 就可能以更高并行度换取较低墙钟。
- 单题三 seed 的结果用于 infra 消融，不支持质量排序；方法质量仍应读取完整 GSM8K 对照实验。
