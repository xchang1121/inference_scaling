# IS + retrieve：代码入口与验证边界

> 2026-11-17：当前 IS＋检索代码只保留 training-free 实现。学习型 probe/gate、监督 ridge/PCA 检索投影、训练标签采集及相关入口和测试已删除。历史测量、失败与成本记录保留。实验仍停止，本指南不是继续执行的授权。

当前实现从第一个思考 token 开始运行原 IS，在每条候选内部检索并追加资料。
主模型权重、IS 搜索与选择算法保持不变。检索控制器负责时机、查询、证据和续接，
不要求模型先生成 tool call。完整约定见 [原在线 IS 协议](reports/ONLINE_IS_PROTOCOL.md)。

这份指南对应当前研究分支的统一代码。实验已停止，独立配置选择与最终评估尚未开启；
“最新代码”不代表已选出质量最优策略。
[F1、分阶段时延与 Base 倍率图表](reports/RETRIEVAL_TUNING_STATUS.md)与[纠正后的调优协议](reports/RETRIEVAL_TUNING_PROTOCOL.md)
记录证据、失败及后续工作。通用服务的默认参数与本研究的固定参数分别管理。

## 应从哪个入口开始

| 用途 | 入口 | 说明 |
| --- | --- | --- |
| 原始在线检索及基线 | [benchmark_online_is.py](../scripts/benchmark_online_is.py) | direct、closed IS、静态/在线资料及等长无关资料对照 |
| 当前小批调优 | [benchmark_tuning_batch.py](../scripts/benchmark_tuning_batch.py) | 一批最多 12 次完整运行，仅接收 TRAIN tuning/probe_train 输入 |
| 固定历史六策略 | [benchmark_smart_online_is.py](../scripts/benchmark_smart_online_is.py) | 保留历史定义，供默认版与调优版对照 |
| 内部统计开关验证 | [check_internal_signals.py](../scripts/check_internal_signals.py) | 分布、最终隐藏状态、注意力；限定短请求证明 |
| 实际中间层验证 | [check_multilayer_signals.py](../scripts/check_multilayer_signals.py) | 16/32/48 层及最终层的 8 请求 OFF/ON 检查 |
| 设备租约与回收 | [leased_npu_job.py](../scripts/leased_npu_job.py) | 检查设备进程和容器占用、持锁、失败后回收 |

[六次运行的示例批次](../examples/retrieval/tuning-baselines.json)包含两道已用于调优的 TRAIN 题，
比较 closed IS、静态资料和原在线资料。它演示输入格式，未被指定为最佳配置。
在匹配的 Ascend/vLLM 环境中，安装本仓库后运行：

```bash
python scripts/benchmark_tuning_batch.py \
  --settings /path/to/frozen-model-settings.json \
  --prepared /path/to/prepared-smoke.json \
  --corpus /path/to/corpus.jsonl \
  --batch examples/retrieval/tuning-baselines.json \
  --output /path/to/new-run/measurements.jsonl
```

模型设置、文档语料、tokenizer 与 prepared 文件必须属于同一已验证环境。
`prepared` 提供历史等长无关证据 token；实际题目来自 batch 的 `cases`。
输出目录须存在，输出文件必须尚不存在。生产研究调度还应对输入分组、源码和短验证证明绑定哈希；
runner 的 TRAIN 字段检查不替代完整数据隔离清单。不要把 DEV/test 改名成 TRAIN。

## 控制器结构

| 组件 | 代码 | 当前职责 |
| --- | --- | --- |
| 原在线上下文与证据账本 | [online_is.py](../src/inference_scaling/retrieval/online_is.py) | 逻辑生成 token 与物理证据前缀分离，只恢复最终选中路径的证据 |
| 可配置时机、查询、格式 | [tuned_online_is.py](../src/inference_scaling/retrieval/tuned_online_is.py) | fixed/boundary/surprisal/lowprob/SC/rule/entropy/margin 等；问题约束、去重、正文包装、续接 |
| 文档与窗口排序 | [ranked_facts.py](../src/inference_scaling/retrieval/ranked_facts.py) | 文档/窗口融合、词项清理、每文档片段上限 |
| 辅助生成与表示分歧 | [auxiliary_online_is.py](../src/inference_scaling/retrieval/auxiliary_online_is.py) | FLARE/IRCoT/SeaKR 适配及独立辅助成本 |
| 支持检查与示例 | [gap_online_is.py](../src/inference_scaling/retrieval/gap_online_is.py)、[example_online_is.py](../src/inference_scaling/retrieval/example_online_is.py) | 证据缺口与真实训练示例；不伪造参考思考链 |

`arm="tuned_online"` 接受 `policy`、`fact_config`、`fact_ranking` 等配置。
辅助、支持检查、示例控制器一次只选一种。参数名与合法范围以对应 dataclass 为准；
修改检索参数不应改写 `GuidanceConfig`。本研究固定 M=3、K=2、block=512、SC 温度 0.25，
原 IS 从空生成状态开始，已固定块不会回滚。

证据缓存以当前完整逻辑前缀为键。不同候选不得继承兄弟分支或未来的证据、信号与决定。
模型生成 token 参与原 IS 统计，注入文本仅进入物理上下文。答案阶段只读选中路径，
自然 EOS、空答案、截断和被丢弃候选全部保留。后端按 offset 派生续接随机流，
改变请求分段或批次布局不保证与未分段生成逐 token 相同。

## 冻结模型的内部统计

保留概率分布、注意力与实际中间层采集，用于固定规则和 SeaKR 表示分歧统计。
中间层采集使用固定随机投影，不拟合权重，也不映射到 E5 语义空间。
`multilayer_auxiliary` 使用显式层选择的 SeaKR；普通多层诊断使用固定触发策略。
TRAIN 与 `probe_train` 是既有数据分组标识，保留以维持隔离边界，不表示当前代码进行参数训练。

## 审计与结果解释

[audit_tuned_online_is.py](../scripts/audit_tuned_online_is.py)分派给各控制器审计，
[audit_native_journal.py](../scripts/audit_native_journal.py)核对原生请求及成本。
每次批次保留 `measurements.jsonl` 与同目录 `measurements-native.jsonl`，
包括预热、辅助生成、最终答案及丢弃分支。源码/模型/输入哈希和设备租约日志一同保存。

常规因果审计通过、仪器 OFF/ON 等价通过、答案事实正确，是三个不同判断。
部分注意力完整对照仍有严格等价失败；结果保留，不能用几何正确或后续通过覆盖。
F1、来源命中及“非空”均不能替代事实审阅：当前已观察到版本号错误、无关片段、
自然结束的未完成答案，以及相同最终答案却显著不同的丢弃成本。

当前还没有统一选定的 IS+retrieve 最佳配置。后续按协议完成适用维度调优、独立 TRAIN 选择、
统一冻结后，才开启未见评估。历史结果与方法默认值均保留用于对照。

## CPU 回归

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

pytest 已包含 `src/` 与 `scripts/` 搜索路径。PyTorch/Transformers 后端测试还需要相应可选依赖；
真实 Ascend 信号及数值等价仍需上述原生检查，CPU 测试通过不代替它们。
