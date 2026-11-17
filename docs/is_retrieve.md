# IS + retrieve：代码入口与验证边界

> 2026-11-17：只保留统一的候选内 IS＋检索主线。训练代码、早期独立控制器及其跑分入口已删除。历史测量、失败与成本记录保留。实验仍停止。

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
| 统一小批入口 | [benchmark_tuning_batch.py](../scripts/benchmark_tuning_batch.py) | 统一调用 `run_online_is`；一批最多 12 次完整运行，仅接收 TRAIN tuning/probe_train 输入 |
| 内部统计开关验证 | [check_internal_signals.py](../scripts/check_internal_signals.py) | 分布、最终隐藏状态、注意力；限定短请求证明 |
| 实际中间层验证 | [check_multilayer_signals.py](../scripts/check_multilayer_signals.py) | 16/32/48 层及最终层的 8 请求 OFF/ON 检查 |
| 设备租约与回收 | [leased_npu_job.py](../scripts/leased_npu_job.py) | 检查设备进程和容器占用、持锁、失败后回收 |

[统一主线示例批次](../examples/retrieval/tuning-mainline.json)包含两道已用于调优的 TRAIN 题，
演示固定插入、锚定查询和 surprisal 触发。它只演示输入格式，未被指定为最佳配置。
在匹配的 Ascend/vLLM 环境中，安装本仓库后运行：

```bash
python scripts/benchmark_tuning_batch.py \
  --settings /path/to/frozen-model-settings.json \
  --corpus /path/to/corpus.jsonl \
  --batch examples/retrieval/tuning-mainline.json \
  --output /path/to/new-run/measurements.jsonl
```

模型设置、文档语料与 tokenizer 必须属于同一已验证环境；实际题目来自 batch 的 `cases`。
输出目录须存在，输出文件必须尚不存在。生产研究调度还应对输入分组、源码和短验证证明绑定哈希；
runner 的 TRAIN 字段检查不替代完整数据隔离清单。不要把 DEV/test 改名成 TRAIN。

## 控制器结构

| 组件 | 代码 | 当前职责 |
| --- | --- | --- |
| 统一候选账本与策略 | [online_is.py](../src/inference_scaling/retrieval/online_is.py) | `OnlineISLedger`、`TuningPolicy`、`run_online_is`；触发、查询、包装、续接、逻辑/物理前缀及分段采样均在同一实现 |
| 证据与示例检索 | [evidence.py](../src/inference_scaling/retrieval/evidence.py) | `FactBank` 共用窗口生成和封装，可选文档/窗口融合；`ExampleBank` 使用冻结 E5 检索完整示例 |
| 公共工具 | [common.py](../src/inference_scaling/retrieval/common.py) | token 哈希、边界识别、问题清理、检查器输出解析、原生成本快照 |
| 原 IS 接口与答案 | [cot_guidance.py](../src/inference_scaling/retrieval/cot_guidance.py) | 只保留完整池 block-SC IS、生成统计和选中轨迹的答案读出 |
| 辅助生成与表示分歧 | [auxiliary_online_is.py](../src/inference_scaling/retrieval/auxiliary_online_is.py) | 候选内 FLARE/SeaKR 适配及独立辅助成本 |
| 支持检查与示例 | [gap_online_is.py](../src/inference_scaling/retrieval/gap_online_is.py)、[example_online_is.py](../src/inference_scaling/retrieval/example_online_is.py) | 证据缺口与真实训练示例；不伪造参考思考链 |

`arm="tuned_online"` 接受 `policy`、`fact_config`、`fact_ranking` 等配置。
辅助、支持检查、示例控制器一次只选一种。参数名与合法范围以对应 dataclass 为准；
修改检索参数不应改写 `GuidanceConfig`。本研究固定 M=3、K=2、block=512、SC 温度 0.25，
原 IS 从空生成状态开始，已固定块不会回滚。

普通策略支持 fixed、boundary、surprisal、lowprob、scdrop、rule、hybrid、entropy、margin、risk、DRAGIN，
以及 `never` 关闭检索；查询支持问题、尾部、锚定融合、不确定 token、注意力词和完整句子。
IRCoT 风格的句子查询直接使用 `query="sentence"` 或 `sentence_anchored`。
六种旧 smart 预设不再拥有独立执行代码；相应行为通过统一策略字段表达。

已删除早期 context-rebuild/stepwise、预搜索 IRCoT/ThinkRetrieve、verified/grounded reward、
独立 BoN/Consilience 对照、direct/static/noise 检索跑分入口及其专用审计、汇总和绘图脚本。
通用推理服务和底层算法库保持原样。历史实现可从清理前提交
[`0063f45`](https://github.com/xchang1121/inference_scaling/tree/0063f45be3761c16654958fd0715cfab05e56b9b)查阅。

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
[retrieval_replay.py](../scripts/retrieval_replay.py)集中提供 IS 离线重放，
[retrieval_metrics.py](../scripts/retrieval_metrics.py)集中提供解码、F1 和 bootstrap。
每次批次保留 `measurements.jsonl` 与同目录 `measurements-native.jsonl`，
包括预热、辅助生成、最终答案及丢弃分支。源码/模型/输入哈希和设备租约日志一同保存。

常规因果审计通过、仪器 OFF/ON 等价通过、答案事实正确，是三个不同判断。
部分注意力完整对照仍有严格等价失败；结果保留，不能用几何正确或后续通过覆盖。
F1、来源命中及“非空”均不能替代事实审阅：当前已观察到版本号错误、无关片段、
自然结束的未完成答案，以及相同最终答案却显著不同的丢弃成本。

当前还没有统一选定的 IS+retrieve 最佳配置。后续按协议完成适用维度调优、独立 TRAIN 选择、
统一冻结后，才开启未见评估。历史报告保留原配置和结果，不代表相应旧入口仍存在。

## CPU 回归

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

pytest 已包含 `src/` 与 `scripts/` 搜索路径。PyTorch/Transformers 后端测试还需要相应可选依赖；
真实 Ascend 信号及数值等价仍需上述原生检查，CPU 测试通过不代替它们。
