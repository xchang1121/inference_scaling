# 推理服务

一个模型对外提供 OpenAI Chat Completions 与 Anthropic Messages 两种接口；每个请求只在思考段上做联合预算 IS
（Consilience 加权）选出一段思考，再从它正常生成一次答案。网关、agent、写代码的客户端按普通模型调用即可。

| 接口 | 用途 |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI 格式，支持 `tools`、`tool_choice`、`stream`、`reasoning_effort` |
| `POST /v1/messages` | Anthropic 格式，支持 `tools`、`stream`、`thinking`、`output_config.effort` |
| `POST /v1/messages/count_tokens` | Anthropic 的 token 计数 |
| `GET /v1/models`、`GET /health` | 模型名与健康检查 |

## 一个请求的处理

1. 请求（含工具定义）转成 Qwen3.8 的 chat template 消息；多轮历史里的思考与工具调用按模板还原。
2. 按推理强度选档位，在档位的前向 token 预算与时长上限内做思考段联合预算 IS：块长、候选数、补全数由规划器选择，
   候选与补全写到 `</think>` 停止，按 Consilience 加权。
3. 从选中的思考正常生成一次答案。
4. Qwen3.8 的 `<tool_call><function=…><parameter=…>` 解析为 OpenAI `tool_calls` 或 Anthropic `tool_use`，参数按工具
   schema 取类型。

返回里额外带 `scaling`：档位、IS 步数、前向 token 数、停止原因（`eos`、`length` 或到时的 `deadline`）与耗时。

## 1. 安装（昇腾）

在已能用 vLLM + vllm-ascend 起 Qwen3.8-27B 的环境里：

```bash
git clone https://github.com/xchang1121/inference_scaling.git && cd inference_scaling
pip install -e ".[serve]"      # 只加装 fastapi、uvicorn；不要装 .[vllm]，它锁的是 CUDA 版 torch 与 vLLM
huggingface-cli download Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --local-dir /data/models/Qwen3.8-27B
```

## 2. 配置

服务读取 `settings/inference.json` 的 `ar`（模型、引擎、采样、IS 规划与网格）和 `rewards.consilience`，以及
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

`settings/serve.json` 的推理强度档位（请求的 `reasoning_effort` 或 `output_config.effort` 选择，默认 `medium`；
`none`、`minimal` 映射到 `low`，`xhigh`、`max` 映射到 `high`）：

| 档位 | 模板推理强度 | 单条输出上限 | 前向 token 预算 | 时长上限 |
| --- | --- | ---: | ---: | ---: |
| `low` | low | 8,192 | 32,768 | 180 s |
| `medium` | medium | 32,768 | 196,608 | 900 s |
| `high` | xhigh | 65,536 | 524,288 | 2400 s |

- 单条输出上限再受上下文（`max_model_len` 减提示）限制；请求的 `max_tokens` 只能调低它。
- 预算是一个请求所有序列（长度探测、试点、候选、补全）的前向 token 之和，约为 4、6、8 条满长序列；提示很长时自动
  抬到最低可行值（2 倍提示加 3 倍输出上限），请求不会被拒。
- 到时长上限后不再开始新的 IS 步，保留当前完整思考；第一步至少依次写完长度探测和候选，不受时长约束。
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
- 请求时长最长约为档位的 `max_seconds` 加答案生成，网关与客户端超时都要大于它。
- 建议用流式：搜索期间每 `keepalive_seconds`（默认 10 s）发一次 keep-alive，网关空闲超时需大于该值。
- OpenAI SDK 与 agent 框架：`base_url = "https://<网关>/v1"`，`model = "qwen3.8-27b-is"`。
- Claude Code 等 Anthropic 客户端：`ANTHROPIC_BASE_URL=https://<网关>`，`ANTHROPIC_MODEL=qwen3.8-27b-is`，
  `ANTHROPIC_AUTH_TOKEN` 按网关要求填写；每轮都要等 IS 选完，交互式写代码宜用 `low`。
- 扩容：一个进程占一组卡，多实例挂在网关后面，各用不同的 `ASCEND_RT_VISIBLE_DEVICES`；
  `server.max_concurrent_requests` 控制单实例同时推理的请求数，其余排队。

## 限制

- 思考要选定后才能发出：流式请求先收到开头事件，搜索期间收到 keep-alive，选定后一次收到思考、答案与工具调用。
- 客户端断开后服务端仍把该请求算完。
- 请求的 `temperature`、`top_p` 不生效：IS 在完整分布上采样，策略取自 `ar.sampling`。
- 只支持文本内容，`n` 只能为 1。
- 未写完的思考没有答案，作为思考返回，`finish_reason` 为 `length`（Anthropic 为 `max_tokens`）。
