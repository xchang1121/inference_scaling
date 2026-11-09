# 技术文档的在线检索与推理中注入

本分支实现外部控制的 context intervention：控制器读取已生成的推理，在检查点完成请求，检索本地文档，将证据 token 追加到原始 token 前缀，再提交续写。模型不接收工具定义，也不需要生成检索指令。入口为 `inference_scaling.retrieval.OnlineInjectionController`；基准脚本为 `scripts/benchmark_injection.py`。

## 首轮要回答的问题

1. 技术规则来自上下文后，回答是否比闭卷更正确？静态 RAG 与中途追加有什么差别？
2. 固定时机、采样概率、Self-Certainty 下降和文本规则触发，哪个在相同检索预算下更有效？
3. 同一批文档在推理开始、32/128 token 检查点、自然句界、关闭推理后追加，效果和开销如何变化？
4. BM25、字符 TF-IDF、两者的 reciprocal-rank fusion 对版本号、错误码和 API 名称有什么差异？
5. 实际复用了多少前缀计算？检索与更长上下文的代价，是否抵消生成质量的收益？

## 已实现的模块

| 维度 | 配置与行为 |
|---|---|
| 检索 | `bm25`、字符三元组 TF-IDF cosine (`char`)、RRF (`hybrid`)；同分按文档 ID 稳定排序 |
| Query | `question`、最近推理 `recent`、保留产品/版本范围的 `hybrid`；只使用原始问题与已生成文本 |
| 触发 | `never`、`periodic`、低概率 token 比例、平均 surprisal、SC 相对初始窗口下降、正则规则、组合 |
| 追加位置 | `user` 静态 RAG、`start` 在 assistant 推理起点、`inline` 当前检查点、`sentence` 下一句界、`pre_answer` 关闭推理后 |
| 预算 | 生成 token、推理 token、单次/累计证据 token、调用数、冷却间隔、句界等待、最大上下文分别限制 |
| 轨迹 | `generated`、`evidence`、`control` 三类片段；查询、来源、触发信号、实际注入位置与生成概率保留 |

`char` 是词形匹配基线，不能称作语义 dense retrieval。真实 embedding/reranker 可在后续接入检索接口；首轮不把尚未实现的模型计作已测方法。

### 序列、置信度与 KV 的约束

续写使用 `old_token_ids + evidence_token_ids`，不会把完整历史 decode 成字符串后重新分词。`user` 只在尚未开始生成时重建完整聊天模板。外部文本中的结构性角色/推理标记会被转义；资料被明确标注为引用数据。

证据和强制关闭推理的控制 token 不属于模型采样，不生成伪造的 log probability，也不参与生成置信度。SC 由已有后端按请求计算；低概率与 SC 都是触发启发式，不能当成正确性或知识掌握的证明。

检查点由已完成的 chunk 请求实现，默认每 32 个生成 token 检查一次。句界模式在触发后以单 token 请求等待下一个标点或换行，最多等 64 token；等待失败会记录并放弃该次注入。`pre_answer` 在同一推理结束标记之后追加证据。所有模式共用生成/推理预算。

vLLM 根据完全相同的 token 前缀自动复用仍驻留的缓存块。这是 prefix-cached continuation，不是引擎内部活动请求的 block-level fork/resume，更不是独立文档 KV 的直接拼接。混合 attention/recurrent 模型还受其状态 checkpoint 和缓存块对齐限制。报告必须展示实际 `shared_prefill_tokens_saved`、新 prefill、抢占及原生执行图位置，不能从追加序列这一事实推断已获得缓存加速。

各方法按同题种子配对，请求种子由 chunk 序号派生。vLLM 不保证跨不同 chunk 切分的逐 token 随机流完全一致。因此句界与普通检查点的对照同时包含请求切分差异。需要严格的同前缀反事实时，应固定一个 checkpoint 和后续随机流，单独做“有/无证据”的分叉实验。

### 控制器的请求流程

1. 用原始问题构造聊天模板并打开推理段。`user` 方法在这一阶段加入资料；`start` 方法在推理起点追加资料。其他方法先生成一个完整 chunk。
2. chunk 返回后，控制器检查结构性结束标记、EOS、生成预算与上下文预算，再从已有推理 token 的概率、统计量和文本计算触发信号。没有预测未来推理，也不访问 gold。
3. 触发时，用问题和最近推理构造 query。检索器排除已用文档，以稳定次序返回候选。格式化器按实际 tokenizer 输出分配标题、正文和包裹文本的预算，每份文档保留来源。
4. `inline` 立即追加；`sentence` 保存待注入资料并等待句界；`pre_answer` 保存资料，等待推理结束。若相应边界没有在预算内出现，记录跳过原因。检索调用数与实际注入数分别统计。
5. 以未改写的历史 token 前缀加证据 token 建立下一个请求。生成片段、证据片段和外部控制片段分别记录。耗尽推理预算时关闭推理段，继续用剩余生成预算回答；EOS、总预算或上下文预算结束整条轨迹。

首轮触发参数直接写在 `arms()` 和 `TriggerConfig` 中，不使用测试结果回调阈值：

| 信号 | 首轮判断 |
|---|---|
| 固定检查点 | 至少已生成 32 token；固定 32 或 128 token 触发 |
| 低概率 | 最近 32 个推理 token 中，采样概率低于 0.15 的比例至少 25% |
| Surprisal | 最近 32 token 平均负对数概率至少 1.5；首轮仅用于组合触发 |
| SC 下降 | 至少两个 32-token 窗口；当前窗口均值比第一个窗口低至少 5% |
| 文本规则 | 最近 800 字符命中预定的版本、重试、超时、不确定性等正则词项 |
| 组合 | 上述低概率、surprisal、SC 或文本规则任意命中；该方法未请求 SC，实际使用其余三个信号 |

间隔、冷却、概率窗口与单次检索预算是不同约束。首轮只允许一次检索，冷却的作用主要留给后续多次注入；接口与测试已覆盖两次检索的文档去重与累计预算。低概率或 SC 没有触发时，保留闭卷轨迹，不强行补一次检索。

混合 query 的问题部分保留产品和版本范围，近期推理部分提供新的错误码或需求词。三种检索器共享同一文档集和指纹：BM25 使用词项统计，字符检索使用三元组 TF-IDF，融合使用两者排序的 RRF。初版按整份文档索引，注入时只截取命中文档正文前部；TechQA 的长文档可能因此命中文档却未暴露答案片段。后续应在训练分区上先比较固定长度切块与段落切块，再冻结方式评测。

## 冻结的首轮 pilot

模型延用本项目的 Qwen3.8-27B 冻结 revision，BF16、无关键词惩罚。生成最多 512 token，其中推理最多 256；到达推理预算时显式追加结束标记，所有方法一致。检索最多一次、top-2、证据最多 1024 token；温度 1、完整支持分布。所有方法最终只提交一次回答。

用户在确认空闲卡后选定 `27.106.124.198` 的物理 3 号单卡，所有正式 GPU 测试统一使用该卡，TP=1 / vLLM 0.23。两个数据集在同一引擎中顺序执行，所有方法使用相同模型、并行设置与后端版本。此前 A、B 的输出归档为预运行，排除在正式统计之外。0–1 卡方案因其他任务占用而在零测量时放弃。自定义算子与既有验证容器对齐；初始化失败、资源等待和零测量时的数据修订均单独留档。共享卡被其他任务占用时等待释放。

数据分为两个独立部分，不合并正确率：

- **Official Document Rules Diagnostic：16 题。** 人工编写的规则应用题，来自 Stripe、Redis、PostgreSQL 16、Amazon SQS、Python 3.13 的 dict / lru_cache / asyncio，以及 RFC 9110 八份官方文档，每份两题。索引包含八条简短人工改述，保存原始网页快照、下载时间、URL 和 SHA256。判题要求最终 JSON 的 action 与 wait_seconds 同时正确。这不是已发布基准；候选动作、语料规模小，可能存在闭卷天花板效应。没有调用这些服务的 API。原先的 16 题虚构合同保存在 `synthetic-api`，仅作机制自测，不计入正式结果。
- **TechQA-RAG-Eval：8 题。** NVIDIA 发布版，revision `0b5bbc84b7f07d6d09d063130e90b716d8d4a32a`。仅从原始 DEV 分区的 158 个可回答样本按固定哈希抽样，使用其发布的全部 28,481 份技术文档。问题、答案、语料分开保存；索引只接收文档字段。使用回答 token F1/EM，属于生成式小样本 pilot，不能与 IBM 原抽取式、带不可回答题目的官方指标直接比较。

十三个预定方法：闭卷、静态 user RAG、推理起点 BM25、固定 32/128 token BM25、低概率 BM25、规则 BM25、组合触发 BM25、句界 BM25、回答前 BM25、固定 32 token 字符检索、固定 32 token 融合检索、SC 下降 BM25。所有阈值先冻结，不在这 24 题上调参。首轮总计 312 次最终回答。

每题轮换方法次序，方法开始前重置 prefix cache，避免前一个方法把下一方法的起始 prompt 预热；同一方法内部保留 continuation 缓存。模型加载、图编译与索引构建不计入每题延迟，单独记录；每题总延迟包含在线检索、格式化、追加和续写。成本同时保留逻辑前向估算和实际图执行含 padding 的 `2 × 参数量 × token 位置` 估算，后者也不是硬件 FLOPs 计数器。

主要统计：API 严格正确率、TechQA 回答 F1/EM、检索 Hit@1/2/3/5、触发/实际注入比例、生成和证据 token、端到端延迟、检索开销、缓存命中和总计算量。质量/成本比较使用相同已完成题目；置信区间以题为单位配对 bootstrap。失败/超长/未注入的输出保留。小样本结果只能用于发现现象与运行问题。

## 复现入口

```bash
PYTHONPATH=src python scripts/prepare_injection_benchmark.py --output /path/api --count 16
PYTHONPATH=src python scripts/prepare_official_api_diagnostic.py --output /path/official-api
PYTHONPATH=src python scripts/prepare_techqa_injection.py --raw /path/nvidia-files --output /path/techqa --count 8
PYTHONPATH=src python scripts/benchmark_injection.py --data /path/api --output /path/retrieval.jsonl --retrieval-only
PYTHONPATH=src python scripts/benchmark_injection.py --data /path/api --settings /path/settings.json --output /path/generation.jsonl
PYTHONPATH=src python scripts/benchmark_injection.py --data /path/api --output /path/generation.jsonl --grade-only
```

TechQA 原始文件为固定 revision 下的 `train.json` 和 `corpus.zip`，取自 [NVIDIA 发布页](https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval/tree/0b5bbc84b7f07d6d09d063130e90b716d8d4a32a)。准备器保存来源 revision、源文件和输出 SHA256。生成模式不读取 gold 文件；判题在生成完成后执行。

规则题来源：[Stripe 幂等请求](https://docs.stripe.com/api/idempotent_requests)、[Redis EXPIRE](https://redis.io/docs/latest/commands/expire/)、[PostgreSQL 16 事务隔离](https://www.postgresql.org/docs/16/transaction-iso.html)、[SQS 可见性超时](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-visibility-timeout.html)、[Python dict](https://docs.python.org/3.13/library/stdtypes.html#dict.setdefault)、[lru_cache](https://docs.python.org/3.13/library/functools.html#functools.lru_cache)、[asyncio.wait_for](https://docs.python.org/3.13/library/asyncio-task.html#asyncio.wait_for)、[RFC 9110 §9.2.2](https://www.rfc-editor.org/rfc/rfc9110.html#section-9.2.2)。冻结时已核对 Stripe API v1 与 v2 的不同重试规则，问题明确指定版本。

## 下一阶段的探索顺序

1. 在独立开发文档上校准置信度阈值，按产品/文档分组留出测试，避免同文档多问造成泄漏。扩展到全部可回答 DEV，随后单独增加不可回答题与拒答判分。
2. 固定检索器与 top-k，比较触发和位置；再固定最有希望的触发，比较 BM25/字符/embedding 与 reranker。补充随机无关资料、错误版本、同长度无信息文本，区分知识收益与单纯追加文字的效应。
3. 实现同 checkpoint 的有/无资料分叉，使用相同文档事实 probe 测量前后知识获取差值。Probe 答案不得反馈给生成轨迹或用作在线选择的隐藏测试。评估证据一致性与最终正确性，单纯 SC 增大不作为成功标准。
4. 接入多候选 IS/ETS 时，把外部检索视为显式、可记录的轨迹转移。模型概率只覆盖生成 token；先写清联合路径的 proposal/target 与选择校正，再接入原采样核。不能把含证据的 token 串和零占位概率直接交给现有置信度 reward，宣称仍是同一个正确的 IS 算法。
5. 先复用共同前缀上的检索和证据 prefill，再分叉；对分支独有证据维护独立来源状态。进一步优化缓存保留、部分块复用及混合模型 recurrent checkpoint。只有在实测开销证明必要时再研究文档 KV fusion。

### 对照扩展与进入下一阶段的条件

先保持文档集、检索次数和上下文上限不变，用独立开发分区校准阈值，再将产品或文档整体留出。位置比较需同时报告实际注入偏移、资料后继续生成的推理 token，以及最终回答是否有内容。若注入后模型立即关闭推理，正确率变化只能说明外部资料影响了后续回答，不能据此声称改善了后续推理过程。

知识获取量应通过独立 probe 定义：在相同 checkpoint 复制上下文，分别追加正确资料、无资料和同长度无信息资料，再询问不参与轨迹选择的事实 probe。比较 probe 的正确率或目标答案条件概率、证据一致性及最终任务分数。Probe 调用的 token 和前向计算另计；probe 的答案与分数不能成为在线触发器的隐藏 oracle。来源冲突或版本不匹配时，单独统计采用错误资料的比例。

缓存优化需满足两个条件：相同 token 前缀能在真实模型中得到与完整重算一致的续写结果，并且实测节省的 prefill 大于追加资料和重新提交请求的成本。本分支的真实小型因果模型 CPU 测试覆盖“生成前缀后追加资料”和“同 checkpoint 追加两份不同资料的兄弟分支”，核对 token、log probability 与缓存计数。它验证参考后端的状态隔离，不能替代昇腾混合模型的原生缓存测试。

接入 IS/ETS 时，状态应显式包含生成前缀、资料来源、触发历史和预算。把资料选择写为外部转移核 `K(e | history)`，模型生成写为 `p(token | context_with_evidence)`。若 proposal 与 target 共享完全相同的确定性检索/注入规则，外部转移在同一路径的概率比中可以约去；若检索、重排或分支筛选是随机的，必须记录相应概率与支持集。改变后缀后继续使用旧资料相当于条件化到另一种路径规则，需要单独定义目标，不能默认仍保持原来的估计性质。只有在这套路径定义、权重和回放测试通过后，才把在线注入接入现有选择核。

## 相关工作与边界

- [FLARE](https://arxiv.org/abs/2305.06983)：前瞻句子的低置信度触发检索与重生成；本分支不回删已生成 token，因此不是 FLARE 的完整复现。
- [DRAGIN](https://aclanthology.org/2024.acl-long.702/)：研究实时信息需求下的检索时机与查询构建；本分支规则与置信度触发未使用其完整 attention-based 信号。
- [ThinkRetrieve](https://arxiv.org/abs/2608.10928)：推理轨迹中检索已解示例；本实验主要注入技术事实与 API 合同，语料类型不同。
- [vLLM APC](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)：共同 token 前缀的缓存复用机制；本分支保留实际命中统计。
- [TechQA 原论文](https://aclanthology.org/2020.acl-main.117/)与 [NVIDIA RAG 版数据卡](https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval)：用于明确数据来源与任务协议。
