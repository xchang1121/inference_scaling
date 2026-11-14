# 智能在线检索与注入的探索协议

基于 `online-injection-full-is-metrics-20261113` 的逐题诊断，比较六种外部控制策略。
运行入口为 `scripts/benchmark_smart_online_is.py`，控制器为
`src/inference_scaling/retrieval/smart_online_is.py`。

## 配对设计

- 沿用固定16道DEV题、种子20261112和20261113，8组共256次全新运行。
- 静态事实、原在线注入均重新运行；另外六组为resume、anchored、boundary、adaptive、bridge、early。
- 按题分配A0、A1、A3，每题的两个种子和全部条件在同一卡执行。每组清空前缀缓存，顺序按题/种子轮换。
- 正式运行前，每卡完成2道TRAIN题×8组的完整性验证，共48次，不计算TRAIN F1。
- 此前的36次六组TRAIN试验显示命令式resume提示仍可导致EOS，因此保留原六组，并增加bridge与early比较提示形态及提前供证。该试验原样归档。
- 全部策略和阈值在查看本轮正式分数之前冻结，不根据中间指标选组或停跑。

## 保持不变

原分块Metropolized IS、SC统计、候选数M=3、rollout数K=2、block=512、温度0.25。
思考上限8192、搜索生成预算65536、答案上限16384、上下文上限32768。
原17个IS/评分文件与原 `online_is.py` 的文件哈希保持一致。

从第一个思考token开始运行IS，不额外预生成或固定一段推理。
各候选的证据由自身前缀决定，不继承兄弟候选的检索。
证据及继续推理提示属于输入，不进入IS逻辑长度、生成预算或SC统计。
不调用额外检索代理模型，不要求模型产生工具调用。

## 八组

| 组 | 触发与检索 | 注入 |
|---|---|---|
| static_facts | 原问题检索两份不同窗口 | 原用户消息中2×256token |
| online_is | 原问题+最近256思考token；生成128/512token后触发 | 原模板，每次256token |
| resume | 与原在线相同 | 同一256token证据后追加固定继续推理提示 |
| anchored | 128/512；问题查询与问题+当前思考查询各取8个窗口；权重0.75/0.25的倒数排名融合 | 同resume；排除与已注入同源窗口重叠超过50%的候选 |
| boundary | 同anchored；在早期64/96/128/160、后期320/352/.../544检查句末或换行 | 每阶段首个边界注入；未遇边界则160/544强制触发；每阶段最多一次 |
| adaptive | 同boundary；后期还需不确定性条件满足 | 最近32token平均惊讶度≥0.8 nats，或最近256token含固定不确定措辞才触发后期注入 |
| bridge | 原在线查询和128/512位置 | 原256token证据后以未完成第一人称句引导续写 |
| early | 同bridge，触发提前到32/256 | 与bridge完全相同的事实包和续写桥接格式 |

查询清理只移除问题尾部固定的答案格式指令，不读取参考答案。
排名融合为 `0.75/(20+question_rank) + 0.25/(20+branch_rank)`，缺失项为0。
同源窗口的交集长度超过较短窗口的50%时视为冗余；无可用非冗余窗口则跳过该次注入。
门控措辞包括unsure、uncertain、unknown、maybe、perhaps、possibly、might、could、not sure、need to check/verify。
门控只使用已完成前缀的原生logprob，不进行额外前向评分。

继续推理提示固定为：

> Continue reasoning about the original question. Treat the reference as external evidence, not your answer. Use relevant facts, ignore irrelevant material, then provide your final answer.

bridge与early替换为未完成句，不强制生成某个后续token，也不抑制EOS：

> I should check this external evidence against the original question and correct any unsupported assumptions before answering. The relevance of this reference to the question is

原256token事实包完整保留。上述提示增加输入长度，其实际token数由原模型tokenizer确定并记录。
边界检测是句末/换行优先，有明确的超时触发点，不声称严格语义句子切分。

## 终止、成本与审计

不抑制EOS、不重试空答案、不回滚后补答；空答案与截断均原样计入最终结果。
如果思考直接EOS，则沿用原读出规则记录空答案；正常思考结束后继续生成最终答案直至EOS或原上限。
候选池成本包含未选中候选和rollout；实际搜索次数、原生分段调用、输入长度及检索时间单独记录。

离线审计核对所有原生分段的前缀、种子及offset、拼接token与统计量，
重放原IS候选请求、评分和选择，并检查最终答案确实沿选中轨迹的证据上下文续写。
智能策略另核验每个前缀检查点的决定、实际句末、惊讶度、不确定措辞、查询融合和去重选择。

## 结论范围

这是已查看DEV题上的探索实验，不是新的独立测试。
F1为最终答案的词级重叠；对每题两个种子取平均后，按问题bootstrap 5000次。
置信区间不作多重比较校正，不能将最高点估计直接当作稳定优胜者。

边界方案改变原生分段及后端offset派生种子，不保证与原方案共享连续随机流。
各组为递进组合，查询、消息格式、位置、曝光量和分段可能共同影响结果；不能据此单独宣称某项改动的因果效果。
