# 推理服务

一个模型对外提供 OpenAI Chat Completions 与 Anthropic Messages 两种接口；每个请求只在思考段上做固定配置的条件 IS
（Consilience 加权）选出一段思考，再从它正常生成一次答案。网关、agent、写代码的客户端按普通模型调用即可。

| 接口 | 用途 |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI 格式，支持 `tools`、`tool_choice=auto/none`、`stream`、`reasoning_effort` |
| `POST /v1/messages` | Anthropic 格式，支持 `tools`、`stream`、`thinking`、`output_config.effort` |
| `POST /v1/messages/count_tokens` | Anthropic 的 token 计数 |
| `GET /v1/models`、`GET /health` | 模型名与健康检查 |

## 一个请求的处理

1. 请求（含工具定义）转成 Qwen3.8 的 chat template 消息；多轮历史里的思考与工具调用按模板还原。
2. 按推理强度选档位，使用 `ar.algorithms.is` 配置的固定块长、候选数与补全数做思考段条件 IS；
   候选与补全写到 `</think>` 停止，按 Consilience 加权。档位设置单条输出上限，默认只按逻辑停止条件结束。
3. 从选中的思考正常生成一次答案。
4. Qwen3.8 的 `<tool_call><function=…><parameter=…>` 解析为 OpenAI `tool_calls` 或 Anthropic `tool_use`，参数按工具
   schema 取类型。

返回里额外带 `scaling`：档位、固定的候选数、补全数与块长、IS 步数、停止原因（`eos`、`length`、`stop` 或显式启用时限后的 `deadline`）、耗时、`stopping_policy`（`logical/wall_clock`）与实际 `rollout_scheduler`（`streaming/batch`）。

`sibling_replay`、`completion_fraction` 与 `rollout_race` 分别记录同块补全复用、整轮完成配额和候选内超发的设置。超发最多将缺失补全数翻倍，每候选收齐 K 条后取消其余请求；完成时间筛选会引入偏差，默认关闭。配置为 `ar.algorithms.is.rollout_race`，需要流式后端。
默认 `sibling_replay=false`、`completion_fraction=1.0`、`rollout_race=false`。配额小于 1 会改变采样分布，详见[算法说明](../../../docs/methods/ALGORITHMS.md#is-completion-quota)。

`ar.algorithms.is.exploration_schedule` 默认 `fixed`。设为 `two_stage` 后，首轮使用配置的 M/K/B，
后续预测生成工作集中到一次完整后缀候选池，流式后端按现有并发容量限制末轮新候选数；与上述三项复用/取消选项互斥。
`scaling.exploration_schedule` 记录该选择，`scaling.sampling_plans` 按轮记录实际 `(B, M, K)`，
`scaling.final_pool_limit` 记录容量上限（未启用时为 `null`）。
原理与限制见[两阶段完整探索](../../../docs/methods/ALGORITHMS.md#is-two-stage)。

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

## 2. 配置

服务读取 `settings/inference.json` 的 `ar`（模型、引擎、采样、固定 IS 参数）和 `rewards.consilience`，以及
`settings/serve.json`。字段说明见 [SETTINGS.md](../../../docs/SETTINGS.md#settingsservejson)。

`settings/inference.json` 需要按机器改的项：

| 字段 | 取值 | 说明 |
| --- | --- | --- |
| `ar.model.path` / `local_files_only` | 本地权重目录 / `true` | 启动时不联网 |
| `ar.engine.vllm.tensor_parallel_size` | 卡数，如 `2` | BF16 权重约 54 GB，64 GB 卡单卡几乎没有缓存空间 |
| `ar.engine.vllm.gpu_memory_utilization` | `0.9` 左右 | 按已跑通时的取值 |
| `ar.engine.vllm.fused_logprobs` | `false` | 融合 worker 只适配 GPU；Consilience 不需要它 |
| `ar.engine.vllm.engine_kwargs` | 删去所用 vLLM 不认识的键 | 如 `language_model_only` |
| `ar.engine.vllm.enable_prefix_caching` | 先 `true` | 混合模型的前缀缓存出错或输出异常时改 `false`，只影响速度 |
| `ar.algorithms.is` | `candidate_count = 4`、`rollout_count = 1`、`block_size = 1024`、`block_first = false` | 所有档位共用；候选数、补全数与块长固定，块长在输出上限处截短 |
| `ar.engine.rollout_scheduler` | 默认开启，全局最多 `32` 条原生请求 | 候选前缀就绪后立即释放其余补全；按题公平准入、长度预测与老化排序、KV 水位和预填充预算限流；[调度说明](../../../docs/methods/ROLLOUT_SCHEDULING.md) |

`settings/serve.json` 的推理强度档位（请求的 `reasoning_effort` 或 `output_config.effort` 选择，默认 `medium`；
`none`、`minimal` 映射到 `low`，`xhigh`、`max` 映射到 `high`）：

| 档位 | 模板推理强度 | 单条输出上限 | 时长上限 |
| --- | --- | ---: | ---: |
| `low` | low | 8,192 | `null` |
| `medium` | medium | 32,768 | `null` |
| `high` | xhigh | 65,536 | `null` |

- 单条输出上限再受上下文（`max_model_len` 减提示）限制；请求的 `max_tokens` 只能调低它。
- 默认不设墙钟停止条件，使调度快慢不改变 IS 步数。显式设置 `max_seconds` 后，到时不再开始新的 IS 步；至少完成第一步，进行中的一步与答案仍完成，但不再保证调度前后的有限步输出分布相同。
- 这些是初值；上线后按返回的 `scaling.seconds` 与 `steps` 调整。

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

## 4. 接入网关与客户端

- 网关上游设为 `http://<机器>:8000`；鉴权、限流、计费由网关负责，服务不校验请求里的模型名。
- `max_seconds` 默认 `null`；显式启用后只在完整 IS 步之间检查。网关与客户端超时需覆盖推理、答案生成及排队时间。
- 建议用流式：搜索期间每 `keepalive_seconds`（默认 10 s）发一次 keep-alive，网关空闲超时需大于该值。
- OpenAI SDK 与 agent 框架：`base_url = "https://<网关>/v1"`，`model = "qwen3.8-27b-is"`。
- Claude Code 等 Anthropic 客户端：`ANTHROPIC_BASE_URL=https://<网关>`，`ANTHROPIC_MODEL=qwen3.8-27b-is`，
  `ANTHROPIC_AUTH_TOKEN` 按网关要求填写；每轮都要等 IS 选完，交互式写代码宜用 `low`。
- 扩容：一个进程占一组卡，多实例挂在网关后面，各用不同的 `ASCEND_RT_VISIBLE_DEVICES`；
  `server.max_concurrent_requests` 控制单实例同时推理的请求数，其余排队。

## 限制

- 思考要选定后才能发出：流式请求先收到开头事件，搜索期间收到 keep-alive，选定后一次收到思考、答案与工具调用。
- 客户端断开或取消等待后，已启动的推理继续执行，并在实际结束后释放并发名额；服务正常关闭时等待这些请求完成。
- 工具选择支持 OpenAI 的 `auto`、`none` 和 Anthropic 的 `{type: auto}`、`{type: none}`。
  强制调用（`required`、`any`、指定工具）及额外工具选择选项返回 HTTP 400。
- 请求的 `temperature`、`top_p` 不生效：IS 在完整分布上采样，策略取自 `ar.sampling`。
- 只支持文本内容，`n` 只能为 1。
- 未写完的思考没有答案，作为思考返回，`finish_reason` 为 `length`（Anthropic 为 `max_tokens`）。
