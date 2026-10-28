# IS 加速尝试索引

范围：2026年10月18日至28日。结论见 [IS 加速研究总结](ASCEND_RESEARCH_SUMMARY.md)，逐阶段数据见 [昇腾条件 IS 报告](ASCEND_IS_INFRA.md)。本索引记录审计批次及主要结果文件的标识，便于核对已有尝试。

原始进度记录包含215个章节、50个证据目录，同一方向的调试、复测及失败运行分别留档。条目数不代表独立算法数量。以下目录名是仓外审计记录的标识；原始日志、临时脚本和本机路径不随报告提交。完整文件校验值列于末节。

## 算法与分布误差

效果、比较基线和反例见总结报告的[统计改进](ASCEND_RESEARCH_SUMMARY.md#统计改进)及[算法和统计预算](ASCEND_RESEARCH_SUMMARY.md#算法和统计预算)。

| 审计批次 | 内容及处理结果 |
| --- | --- |
| `ensemble-suffix-20261027` | 完整候选池后缀更新；基础分支已合入 |
| `stratified-pool-20261027` | 分层初始化；保留为可选项 |
| `persistent-strata-20261028` | 后续更新保留分层；保留为可选项 |
| `nonself-suffix-20261028` | 排除原轨迹并修正接受率；保留为可选项 |
| `prefix-selection-20261028` | 按候选块聚合权重并选择；保留为可选项 |
| `conditional-strata-20261028`、`coarse-conditional-strata-20261028` | 条件 IS 候选分层；多数有限模型误差改善，等待成本增加，未接入生产 |
| `prefix-strata-20261028`、`independent-strata-20261028` | 前缀和候选分层变体的统计检查；未作为新的独立优化合入 |
| `cut-strata-20261028`、`token-change-cuts-20261028`、`marginal-cuts-20261027` | 切点分配及提案概率检查；未替换当前切点策略 |
| `anytime-is-20261028` | 预算截止与多链返回；有限预算误差和未完成工作占比不利，未采用 |
| `anchored-length-budget-20261028` | 参考长度与预算控制的补充审计；未形成新的生产变更 |
| `group-stopping-20261028` | 相关结束随机数；等待减少伴随分布误差增加，未采用 |
| `lossless-weight-20261028` | 权重修正及提前判断检查；未形成稳定的同预算优势 |

## 调度、复用与完整入口

结果见[当前累计时延和成本](ASCEND_RESEARCH_SUMMARY.md#当前累计时延和成本)及[调度和状态复用](ASCEND_RESEARCH_SUMMARY.md#调度和状态复用)。

| 审计批次 | 内容及处理结果 |
| --- | --- |
| `general-scheduling-20261021` | 早期调度设计与筛查；保留和撤销范围见总结 |
| `candidate-race-20261021`、`rollout-race-20261021` | 候选内超发与完成配额；可选实现已合入，计算量和选择偏差另计 |
| `partial-prefix-20261021` | 部分 KV 块复用；局部收益未稳定转化为完整 IS 加速 |
| `prefix-coalescing-20261022` | 前缀去重与分叉可行性；未形成生产优化 |
| `abort-window-20261025` | 取消传播和停止后额外执行；清理与计量修复已保留 |
| `cache-admission-20261025`、`stage-admission-20261025` | 缓存与请求准入；收益和回退并存，未采用为通用优化 |
| `fork-cost-20261025` | 分叉与状态复制成本；局部算子收益未作为整题收益报告 |
| `fanout-reservation-20261028` | 为分叉预留并发容量；未形成通用收益 |
| `cached-prefix-opportunity-20261028` | 真实前缀重复机会审计；近期流程新增机会有限 |
| `bounded-pool-20261028`、`production-pipeline-20261028`、`integrated-pool-20261028` | 完整池跨轮流水线原型、接入及验收；最终实现已合入 |
| `fixed-work-20261028` | 固定请求长度和逻辑决定的执行对照；与自然生成成本分开 |
| `cumulative-is-20261028` | 修正概率后的原始固定、当前固定、当前快速 IS 累计对照 |
| `cross-question-20261028` | 既有跨题并发能力；整组吞吐改善，部分单题延迟增加 |

## 引擎与数值检查

详细范围见总结报告的[引擎和算子](ASCEND_RESEARCH_SUMMARY.md#引擎和算子)。除概率接口修复外，以下局部探针均未作为生产加速合入。

| 审计批次 | 内容及处理结果 |
| --- | --- |
| `probability-contract-20261028` | 温度、筛选和词表惩罚后的概率检查；接口修复已合入 |
| `batched-sampling-20261024`、`batched-random-20261028`、`buffered-random-20261028`、`buffered-model-20261028` | 原生异步、分块及缓存随机采样；完整模型收益较小，部分方式改变逐请求随机流 |
| `sparse-probabilities-20261028` | 稀疏概率归一化；替代路径更慢 |
| `cascade-attention-20261024`、`fia-sink-20261024` | 共享前缀及分页合并 attention；部分长前缀算子更快，较多配置回退 |
| `cascade-model-20261028` | 真实模型分页能力、回退和数值检查；尚无完整 IS 加速证据 |
| `compound-step-20261025` | 合并前向与采样调用；完整运行收益小且不稳定 |
| `dense-graphs-20261028` | 扩大执行图批量覆盖；填充减少，单卡耗时仅下降约0.27% |
| `decode-matrix-audit-20261025`、`linear-execution-20261025` | 解码计算段、布局和图执行检查；缺少稳定完整模型收益 |
| `linear-mv-20261025` | 直接矩阵向量乘法；慢于原生路径 |
| `grouped-linear-20261025`、`working-set-linear-20261025` | 分组线性层及大工作集复核；有效对照更慢，受干扰旧计时作废 |
| `hccl-path-20261025` | 双卡通信与等待分析；未形成稳定完整路径收益 |

## 主要结果文件校验

文件名对应仓外保存的完整结果；SHA-256 用于区分复测版本。以下均为已完成文件，未完成或受外部占用影响的测量不纳入累计收益。

| 文件标识 | SHA-256 |
| --- | --- |
| `cumulative-is-20261028/all-summary-r1.json` | `78fac9248c01ce9e596620248f5d2a4db080b65422459471da53a348798a9ada` |
| `integrated-pool-20261028/seven-r1.jsonl` | `ff2b892c664936c1869ee08cd378ba16a9992113a508135292e022c7f222b6a0` |
| `integrated-pool-20261028/pair-r1.jsonl` | `0b013d9d3e3e75bb5d80b4a8c107c49b202e1d0dbc9f0f7c620f9a9b3091f727` |
| `buffered-model-20261028/all-summary-r1.json` | `4eb0246d280a061e423bbe44e4872d9dd2a7785fd877bd67db3f412c32107e85` |
| `cascade-model-20261028/single-r3.jsonl` | `019e54a4b39d42627b6f2c127ff044fa8fc52b69a32c5103794879063d235c16` |
