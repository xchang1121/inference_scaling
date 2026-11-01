# 推理服务

一个模型对外提供 OpenAI Chat Completions 与 Anthropic Messages 两种接口；默认在思考段上按配置运行条件 IS
（Consilience 加权）选出一段思考，再从它正常生成一次答案。网关、agent、写代码的客户端按普通模型调用即可。

| 接口 | 用途 |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI 格式，支持 `tools`、`tool_choice=auto/none`、`stream`、`reasoning_effort` |
| `POST /v1/messages` | Anthropic 格式，支持 `tools`、`stream`、`thinking`、`output_config.effort` |
| `POST /v1/messages/count_tokens` | Anthropic 的 token 计数 |
| `GET /v1/models`、`GET /health` | 模型名与健康检查 |

## 一个请求的处理

1. 请求（含工具定义）转成 Qwen3.8 的 chat template 消息；多轮历史里的思考与工具调用按模板还原。
2. 使用 `ar.algorithms.is` 配置的探索调度、块长、候选数与补全数做思考段条件 IS；
   候选与补全写到 `</think>` 停止，按 Consilience 加权。单条输出上限和默认推理强度与批量评测共用。
3. 每轮选择后，向流式客户端交付已经固定的思考前缀；从最终选中的完整思考正常生成一次答案。无工具的答案使用增量交付。
4. Qwen3.8 的 `<tool_call><function=…><parameter=…>` 解析为 OpenAI `tool_calls` 或 Anthropic `tool_use`，参数按工具
   schema 取类型。

返回里额外带 `scaling`：生效推理强度、候选数、补全数与块长、IS 步数、停止原因（`eos`、`length`、`stop` 或显式启用时限后的 `deadline`）、耗时、`stopping_policy`（`logical/wall_clock`）与实际 `rollout_scheduler`（`streaming/batch`）。
`scaling.generation_budget` 记录配置上限、客户端上限、剩余上下文和实际生成上限。
`first_commit_seconds` 是首次固定非空前缀的时间，`first_thinking_publish_seconds` 是首次向服务交付思考文本的时间，
`thought_ready_seconds` 是完整思考就绪时间，均从请求开始推理起计。排队及网络延迟由客户端测量；
keep-alive、角色事件和内部候选的首 token 不计入用户可见 TTFT。
`first_content_publish_seconds` 记录首次交付最终内容的时间；最终续写保持原来的前缀、种子、采样及长度。
`cost` 分别记录搜索和最终内容生成的请求级计算量；`generated_tokens` 包含全部已选及未选候选，
与 `usage.completion_tokens` 中最终保留输出的长度分开。取消、评分计数缺失或期间发生引擎抢占时，
不完整的成本及生成量标为 `null`，避免跨客户端混算。

`ar.algorithms.is.exploration_schedule="two_stage"` 首轮使用 M/K/B，再按未选 rollout 的长度与权重决定
末轮候选数。末轮最多占用已配置的在途容量，并使用保留序列的随机长度区间。`fixed` 保留为 CPU 参考；
当前默认设置待 Qwen3.8-27B 配对验证后再调整。
`scaling.sampling_plans` 记录各轮实际 `(B, M, K)`，`final_pool_limit` 记录容量限制；
末轮 M=1 表示直接复用已有序列，无新请求。见[两阶段探索](../../../docs/methods/ALGORITHMS.md#is-two-stage)。

`selection_ready=true` 使用已有前缀统计近似未完成权重，在完整候选已满足选择条件时取消其余请求。
首轮保持预设 B，仅收齐原定补全的组可以入选；末轮允许直接选回完整参考序列。
关闭时等待全部有效候选。该近似存在选择偏差，详见[提前选择](../../../docs/methods/ALGORITHMS.md#is-selection-ready)；
仅读取缓存统计，额外模型评分调用数为零。

`scaling.engine_execution` 记录引擎生效的输出间隔、扁平概率格式、内部异步调度与 CUDA Graph 配置。
K=1 默认只向算法交付完成结果，内部仍保留首 token 的预填充进度通知；K>1 保留分支所需的前缀事件。

## 1. 安装（昇腾）

在已能用 vLLM + vllm-ascend 起 Qwen3.8-27B 的环境里：

```bash
git clone https://github.com/xchang1121/inference_scaling.git && cd inference_scaling
pip install -e ".[serve]"      # 只加装 fastapi、uvicorn；不要装 .[vllm]，它锁的是 CUDA 版 torch 与 vLLM
huggingface-cli download Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --local-dir /data/models/Qwen3.8-27B
```

非 root 容器的运行 UID 应有对应的 `/etc/passwd` 条目和可写用户目录。四卡验证中，缺少该条目曾导致
`hcclGetRootInfo` / `ra init` 错误；补齐容器内用户后，四卡通信和模型加载通过。部署时先检查设备可见性和
小张量 HCCL 通信，再加载模型。相关故障说明见 [vllm-ascend #9490](https://github.com/vllm-project/vllm-ascend/issues/9490)。

## 2. 配置

服务只读取 `settings/inference.json`：`ar` 提供模型、长度、模板、引擎、采样与 IS 参数，
`rewards.consilience` 提供奖励，`run.seed` 提供根种子，`serve` 提供 HTTP 参数。
字段说明见 [SETTINGS.md](../../../docs/SETTINGS.md#serve)。

`settings/inference.json` 需要按机器改的项：

| 字段 | 取值 | 说明 |
| --- | --- | --- |
| `ar.model.path` / `local_files_only` | 本地权重目录 / `true` | 启动时不联网 |
| `ar.engine.vllm.tensor_parallel_size` | 卡数，如 `2` | BF16 权重约 54 GB，64 GB 卡单卡几乎没有缓存空间 |
| `ar.engine.vllm.gpu_memory_utilization` | `0.9` 左右 | 按已跑通时的取值 |
| `ar.engine.vllm.fused_logprobs` | `false` | 融合 worker 只适配 GPU；Consilience 不需要它 |
| `ar.engine.vllm.engine_kwargs` | 删去所用 vLLM 不认识的键 | 如 `language_model_only` |
| `ar.engine.vllm.enable_prefix_caching` | 先 `true` | 混合模型的前缀缓存出错或输出异常时改 `false`，只影响速度 |
| `ar.algorithms.is` | `candidate_count = 4`、`rollout_count = 1`、`block_size = 1024`、`block_first = false` | 评测与服务共用；候选数、补全数与块长固定，块长在输出上限处截短 |
| `ar.engine.rollout_scheduler` | 默认开启，全局最多 `32` 条原生请求 | 候选前缀就绪后立即释放其余补全；按题公平准入、长度预测与老化排序、KV 水位和预填充预算限流；[调度说明](../../../docs/methods/ROLLOUT_SCHEDULING.md) |

`ar.max_new_tokens=null` 使用剩余上下文作为生成总长，`ar.engine.context_window=null` 读取模型/引擎配置；
Qwen3.8-27B 的原生上下文容量为 262144，不启用扩展。客户端 `max_tokens` / `max_completion_tokens` 限制
thinking 与最终内容的合计长度；同时传入时须相等，非法值返回 HTTP 400。

思考强度、预算及别名统一配置在 `ar.thinking`，按[官方规则](../../../docs/SETTINGS.md#arthinking)解析：
`low/minimal → 4096`、`medium → 16384`、`xhigh/high/max → 262144`；`none` 关闭思考。
档位与预算均未指定时为 `xhigh`、131072 thinking tokens。关闭思考时使用已有的全序列采样/评分回退，并在响应中记录。
也可传 `thinking_budget`（Anthropic 为 `thinking.budget_tokens`）直接设置思考预算，与显式 effort 互斥。
达到思考预算后结束思考段并继续生成内容；总长先耗尽时返回 `length`。原始预算与有效预算均写入
`scaling.generation_budget`，M/K/B、奖励和词表惩罚保持配置值。

`serve.max_seconds` 默认 `null`。显式设定后，到时不再开始新的 IS 步；至少完成第一步，进行中的一步与答案仍完成。
它不是整题总时长硬限，启用后可能改变有限步输出分布。单条长度上限同样不限制全部候选的总计算量。

## 3. 启动与验证

在仓库根目录启动（按相对路径读 `settings/`）：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 python -m inference_scaling.serve
```

```bash
curl localhost:8000/health
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model":"qwen3.8-27b-is","reasoning_effort":"low",
  "messages":[{"role":"user","content":"写一个判断素数的 Python 函数"}]}'
curl -N localhost:8000/v1/messages -H 'content-type: application/json' -d '{
  "model":"qwen3.8-27b-is","max_tokens":4096,"stream":true,
  "tools":[{"name":"read_file","description":"读文件","input_schema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}],
  "messages":[{"role":"user","content":"看一下 main.py 里有什么"}]}'
```

第二个请求应返回 `tool_use`，`input` 为 `{"path": "main.py"}`。启动时若报 vLLM 导入错误（如 `AsyncLLM`、
`vllm.v1.metrics.reader`），说明所用 vLLM 与本仓库适配的 0.25–0.26 不同，需要适配后端。

### 性能测量

[`benchmark_is_service.py`](../../../scripts/benchmark_is_service.py) 从客户端测量 MATH-500 或 GSM8K 请求，
报告思考/内容 TTFT、端到端延迟的 p50/p95、输出 TPS、全部生成 TPS、每卡输出 TPS、正确率和长度耗尽数。
TPS 以整批请求的墙钟时间为分母；预热、加载和答案评判不计入。失败请求保留，并使整批吞吐结果标为不完整。

```bash
python scripts/benchmark_is_service.py --urls http://127.0.0.1:8000 \
  --dataset math500 --questions 2 --repeats 2 --concurrency 1 --cards 2 \
  --effort low --max-tokens 8192 --label tp2-c1 --output results/tp2-c1.jsonl
```

这是显式 4096 思考预算、8192 总长的短负载示例；省略 `--effort` 和 `--max-tokens` 则继承服务配置。
对照实验固定题目、种子、M/K/B、奖励和长度，仅改变执行配置；跨卡数浮点误差可能改变轨迹，需要同时比较准确率和生成量。
客户端按 `run.seed`、题目 ID 和重复编号派生请求的 `seed`，预热使用独立种子，保持不同并发度和副本数下的种子对应。
服务返回实际 `sampling_seed`，不一致时测量标记失败。固定种子不保证不同批处理顺序、卡数下的浮点结果逐字相同。
`--urls` 可列出多个独立副本，`--cards` 填总卡数；两个 TP=2 副本的吞吐与单个 TP=4 实例分别报告。
`--deadline` 是客户端截止时间，超时后服务中的请求仍会继续，必须等待其结束或停止所属实验服务再启动下一组。

## 4. 接入网关与客户端

- 网关上游设为 `http://<机器>:8000`；鉴权、限流、计费由网关负责，服务不校验请求里的模型名。
- `serve.max_seconds` 默认 `null`；显式启用后只在完整 IS 步之间检查。网关与客户端超时需覆盖推理、答案生成及排队时间。
- 流式请求按 IS 轮次交付已固定思考。两次交付之间每 `keepalive_seconds`（默认 10 s）发送 keep-alive，网关空闲超时需大于该值。
- OpenAI SDK 与 agent 框架：`base_url = "https://<网关>/v1"`，`model = "qwen3.8-27b-is"`。
- Claude Code 等 Anthropic 客户端：`ANTHROPIC_BASE_URL=https://<网关>`，`ANTHROPIC_MODEL=qwen3.8-27b-is`，
  `ANTHROPIC_AUTH_TOKEN` 按网关要求填写；启用 `thinking` 后可收到逐轮固定的思考，工具调用在答案生成完成后交付。
- 扩容：一个进程占一组卡，多实例挂在网关后面，各用不同的 `ASCEND_RT_VISIBLE_DEVICES`；
  `server.max_concurrent_requests` 控制单实例同时推理的请求数，其余排队。

## 限制

- 思考前缀在本轮全部必要补全完成、选择确定后交付，尚未选中的候选保持内部可见。此改动保持采样、权重、随机种子与计算量不变；首轮仍有完整 rollout 的等待。
- Qwen 预填充的思考通道支持增量交付；未知或非预填充格式缓冲到完整思考选定。预填充通道中格式异常的文本仍作为思考返回，奖励的全序列回退规则不变。
- 流式后端逐步交付无工具的最终答案，工具调用保留完整解析后交付；全序列回退与非流式后端仍在完整生成后交付答案。TTFT 的降低不代表整体 TPS 同幅改善。
- 客户端断开或取消等待后，已启动的推理继续执行，并在实际结束后释放并发名额；服务正常关闭时等待这些请求完成。
- 工具选择支持 OpenAI 的 `auto`、`none` 和 Anthropic 的 `{type: auto}`、`{type: none}`。
  强制调用（`required`、`any`、指定工具）及额外工具选择选项返回 HTTP 400。
- 请求的 `temperature`、`top_p` 不生效：IS 在完整分布上采样，策略取自 `ar.sampling`。
- `seed` 可指定请求的非负 63 位整数种子（Anthropic 接口为本服务扩展）；省略时继续从 `run.seed` 和请求计数派生。
- 只支持文本内容，`n` 只能为 1。
- `thinking.budget_tokens` 设置独立思考上限；`max_tokens` 限制思考与最终内容的合计长度。预算与 effort 同时指定时返回 HTTP 400。
- 未写完的思考没有答案，作为思考返回，`finish_reason` 为 `length`（Anthropic 为 `max_tokens`）。
