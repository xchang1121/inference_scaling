# 运行与测量

运行位置为项目目录。模型、数据与结果位置由调用者传入。
命令中的环境变量表示本地选定的资源，个人配置保存在已忽略的 `local/` 或仓库外。

## 1. 环境与小模型检查

GPU 运行使用支持 CUDA 的 PyTorch；Windows 可在 WSL2 中运行。
安装与显卡驱动兼容的 PyTorch 后，执行：

```bash
python -m pip install -e '.[dev,text,hf,data]'
python -m pytest -q
python -m blockspec demo --device cpu
python scripts/train_dual_view.py demo --device cpu --output local/demo.pt
```

第一个闭环验证条件低秩蒸馏与在线更新，第二个验证双向起草训练、张量映射及精确恢复。
输出检查点使用新文件名。主运行环境采用项目依赖，外部参照可使用独立虚拟环境。
分词器配置和外部模型源码各自需要兼容的 Transformers 版本，运行时选择对应的虚拟环境。

## 2. 输入资源

| 参数 | 内容 |
|---|---|
| `--model` | 双视图权重目录及对应 tokenizer |
| `--base` | 自回归基座目录及配置 |
| `--adapter` | PEFT 目录或本项目的适配器检查点 |
| `--prompts`、`--learning-prompts` | 分离的评测、学习问题 JSONL |
| `--data`、`--validation` | 已分词的训练、验证 JSONL |
| `--output`、`--summary` | 调用者选定的结果或检查点位置 |

问题记录支持 `question` 或 `prompt` 字段；训练记录使用 `input_ids`。
模型维度、注意力头数、词表及特殊 token 从配置读取。
权重桥接校验受支持的架构、完整张量集合和形状；新增架构通过数值对齐后接入。

准备会话数据时，显式指定数据集、配置、划分和行偏移：

```bash
python -m blockspec prepare --base "$BASE_DIR" --output "$DATA_DIR" \
  --dataset "$DATASET_ID" --dataset-config "$DATASET_CONFIG" \
  --source-split train --offsets 0,100,200 --page-size 8
```

该入口读取含 `conversations` 的会话记录，按问题分组划分训练、验证和测试。
同一问题的不同回答归入同一划分。训练与评测入口同时检查数据重叠。

## 3. 固定双向起草

```bash
python scripts/dual_view.py benchmark --model "$DUAL_MODEL_DIR" \
  --prompts "$EVAL_PROMPTS" --requests 16 --tokens 2048 --blocks 32 \
  --empty-system --dtype bfloat16 --backend sdpa --temperature 0 \
  --repeats 2 --output "$RESULT_FILE"
```

`--thinking` 控制思考模板；省略时关闭。`--empty-system` 添加空 system 消息。
采样配置、聊天模板、EOS 规则与输出预算共同定义实验条件。
结果记录中的已有单轮测量使用 `--repeats 1`；独立复测宜增加重复轮次与问题数量。

三种概率执行的同轨迹对照：

```bash
python scripts/dual_view.py benchmark --model "$DUAL_MODEL_DIR" \
  --prompts "$EVAL_PROMPTS" --requests 16 --tokens 2048 --blocks 32 \
  --empty-system --temperature 1 --top-k 0 --top-p 1 \
  --sampling-executions scalar tensor graph --repeats 2 --output "$RESULT_FILE"
```

`scalar` 为逐项校正，`tensor` 为整块张量校正，`graph` 将相同张量操作录入 GPU 图。
输出给出各执行方式的 AR／投机 TPS、TPF、准备时间与配对区间。
剖析使用同一入口的 `profile` 模式，输出预算至多 256 token。

## 4. 在线学习

起草注意力后段续训：

```bash
python scripts/dual_continue.py --model "$DUAL_MODEL_DIR" \
  --prompts "$EVAL_PROMPTS" --learning-prompts "$LEARN_PROMPTS" \
  --thinking --prompt-offset 8 --requests 16 --tokens 512 --repeats 2 \
  --learn-requests 16 --learn-tokens 256 --last-layers 1 --stride 16 \
  --replay-blocks 1 --learning-rate 0.00001 --loss forward_kl \
  --temperature 1 --top-k 20 --top-p 0.8 --output "$RESULT_FILE"
```

每轮校正后保存教师反馈，在更新间隔到达时重放起草后段、反向并发布参数。
评测组包括 AR、原始固定、原起点在线、预学习后固定及相同学习起点继续更新。
在线 TPS 包含反馈与更新成本；独立预学习、状态初始化及执行器准备单列。

同前缀分布审计添加 `--audit-only --audit-requests 8 --audit-tokens 128`。
稀疏温度混合使用 `scripts/dual_online.py`，资源参数与独立学习／评测划分保持相同；
具体更新配置见各入口的 `--help`。

## 5. 离线训练与恢复

```bash
python scripts/train_dual_view.py train --base "$BASE_DIR" \
  --data "$TRAIN_TOKENS" --validation "$VALIDATION_TOKENS" \
  --block-size 32 --mask-token-id "$MASK_TOKEN_ID" --device cuda \
  --precision bf16 --steps 200 --stop-after 100 --output "$CHECKPOINT_FILE"

python scripts/train_dual_view.py resume --checkpoint "$CHECKPOINT_FILE" \
  --data "$TRAIN_TOKENS" --validation "$VALIDATION_TOKENS" \
  --device cuda --output "$NEXT_CHECKPOINT_FILE"
```

随机锚点定义多个隔离的起草块，干净 AR 视图提供完整教师分布。
检查点保存参数、优化器、随机数、数据顺序及学习率进度。
恢复沿用保存的总步数和调度，`--stop-after` 表示中间停止边界。

条件低秩训练使用 `python -m blockspec train --help`。
原适配器在线续训、PrefixRelay 条件头与稀疏接续混合分别由
`blockspec benchmark`、`scripts/prefix_relay.py` 和 `scripts/overlap_mix.py` 提供。

固定低秩适配器与 AR 的对照：

```bash
python scripts/benchmark_offline.py --base "$BASE_DIR" --adapter "$ADAPTER_DIR" \
  --data "$EVAL_TOKENS" --dtype bfloat16 --sampler linear --block-size 8 \
  --temperature 1 --sampling-top-k 50 --top-p 0.95 --repeats 2
```

PEFT 加载始终校验张量名称、形状、秩和缩放；可选的本地完整性参数由调用者提供。

## 6. 外部参照

独立运行读取权重张量。执行外部模型 Python 时，显式传入 `--reference-manifest "$REFERENCE_MANIFEST"`。
该本地文件提供被选中的代码、配置与权重的一致性校验，保存在仓库外或 `local/`。
结构为 `models.base`，包含 `weight_filename`、`weight_sha256` 和 `reference_lf_sha256`；
双视图参照另含 `entrypoint.file` 与 `entrypoint.class`。
`reference_transformers` 可指定外部环境版本。源码文件名及模型类均由本地配置选择。

```bash
python scripts/dual_view.py audit --model "$DUAL_MODEL_DIR" \
  --reference-manifest "$REFERENCE_MANIFEST" --blocks 4 32 --tokens 32

python scripts/audit_hf_reference.py --base "$BASE_DIR" \
  --reference-manifest "$REFERENCE_MANIFEST" --dtype bfloat16 --device cuda
```

`scripts/benchmark_reference.py` 额外使用本地清单的 `source.commit` 与 `models.adapter`，
在独立进程中运行外部引擎。CPU 数学参照入口
`scripts/audit_sampler_reference.py` 通过 `--source` 和 `--reference-revision` 选择代码。
共同管线迁移对照 `scripts/audit_pipeline.py` 通过 `--reference-commit` 选择本地 Git 参照。

## 7. 测量与提交

同一问题交错运行各方法，采用相同模型精度、模板、采样规则和输出预算。
记录总输出 token、请求耗时、TPS、解码 TPF、更新次数与更新耗时；
启动、预热、预学习和状态切换单列。每种方法使用自身的匹配 AR 计算加速比。

实验入口通过统一报告层写出方法设置、指标与校验结论。
文件地址、资源标识、内容摘要和原始提示／生成文本留在本地输入与运行状态中。
`ALGORITHM.md` 保存推导，`RESULTS.md` 保存有效结果；设计尝试的取舍直接简述于对应章节。
提交前汇报性能和测试结果，提交说明聚焦方法变化与对应测量。
