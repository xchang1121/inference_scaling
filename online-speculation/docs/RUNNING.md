# 当前原生推理优化与 Online Uno 的运行说明

## 环境

模型推理使用本机已安装的 WSL2 / Ubuntu 22.04，发行版名 `Ubuntu-22.04`、Linux 用户 `singm`。

| 内容 | 路径 / 版本 |
| --- | --- |
| Python 环境 | `/home/singm/.venvs/uno-cu128`，Python 3.10 |
| 官方 Uno | `/home/singm/online-speculation-work/uno` |
| base | `/home/singm/online-speculation-work/models/K2-Horizon-0.9B` |
| Uno adapter | `/home/singm/online-speculation-work/models/K2-Horizon-0.9B-Uno` |
| 完整 wheel 缓存 | `/home/singm/.cache/uno-wheels` |
| PyTorch / CUDA runtime | `2.11.0+cu128` / `12.8` |
| Triton / FlashAttention / Transformers | `3.6.0` / `2.8.3` / `4.55.0` |

源码 commit、模型 revision 和权重 SHA 见 [upstream.lock.json](../references/upstream.lock.json)。
基准启动前校验源码 commit、tracked clean 状态及两份权重 SHA。
当前代码不依赖已删除的 Windows HF 原型，也不依赖任何历史实验 JSON。

`bootstrap_uno_runtime.sh` 仅用于重建环境，不是每轮实验都要执行的下载脚本。
依次传入已有官方源码、项目、base、adapter 的来源目录和自检输出路径；
它将所需文件复制到 Linux 工作目录，校验 [wheel locks](../config/)，建立 venv 并运行 runtime smoke。
系统需已有 `python3.10-venv`、`git`、`rsync`、`curl`、`jq`；不会安装 Linux NVIDIA 显示驱动。
WSL、当前模型与官方源码、完整依赖包不是实验记录，清理不卸载这些运行依赖。

## 运行入口

以下在 WSL Bash 中执行：

```bash
cd /mnt/c/Users/singm/Desktop/hw/akg_related/inference_scaling/online-speculation
source /home/singm/.venvs/uno-cu128/bin/activate

python scripts/benchmark_native_uno.py \
  --source /home/singm/online-speculation-work/uno \
  --base /home/singm/online-speculation-work/models/K2-Horizon-0.9B \
  --adapter /home/singm/online-speculation-work/models/K2-Horizon-0.9B-Uno \
  --blocks 1,8 --fused-norm --fast-weights \
  --training-backend cuda_graph \
  --update-stride 16 --replay-blocks 4 --learning-rate 0.001 --rank 8 \
  --repetitions 2 --max-new-tokens 512 --warmup-tokens 128 \
  --output results/native_run.json

python scripts/analyze_native_uno.py \
  --input results/native_run.json --output results/native_run_audit.json
```

这是四个内置 prompts 的工程检查入口，不是完整论文复现或独立 held-out 评估。
`results/` 全部被 Git 忽略，没有默认提交例外。运行时使用新文件名，不覆盖已有结果。
源码树不保存旧成绩、失败轨迹或结果归档。

B=1 是原引擎 AR 对照；未启用 fast weights 时，B=8 是固定块长 Uno。
启用 `--fast-weights` 后自动交错测量四种方法：`1`、`8`（零增量分支控制）、
`plain8`（实际无新增分支的静态 Uno）、`fast8`（真正在线 LoRA）。
**完整净收益使用 `fast8/plain8`**；`fast8/8` 只分离训练增量成本，不能替代主比较。
同模型额外捕获无分支图，避免用不同进程、时间段的运行来估计微小分支开销。
每个 prompt 的相邻两次 repetition 使用正反成对顺序，每组在配对内的平均出场位次相同。
性能比较用偶数 repetitions；单次 smoke 仍可运行，审计会标记没有完整的顺序配对。
在上述命令基础上，去掉 `--fast-weights` 测不含新 LoRA 分支的融合版本；
再去掉 `--fused-norm` 测原生基线。三种配置使用同一 32-page KV 容量、同样 prompts/种子/预算。
`--audit-fast` 开启额外的重放 logits 检查，检查成本也计入 TPS，仅作功能验证。
初次 smoke 可以用 `--workloads english --repetitions 1 --max-new-tokens 128`。

新增评估输入为 `config/evaluation_prompts.json`，12 个事先固定的人工设计 prompts，
与四个开发 prompts 不重复；不是公开论文 benchmark，也不是模型预训练数据意义的 held-out。
使用 `--prompt-file config/evaluation_prompts.json --repetitions 2 --max-new-tokens 1024 --seed 20270909`。
未指定 `--workloads` 时运行整个所选 suite；可以按名称筛选。结果记录 prompt 文件 SHA 和代码 SHA。
该 suite 在首次评估后已见过，之后的重复测量不再宣称新的 held-out 验证。

默认更新 backend 为 `cuda_graph`；`--training-backend eager` 保留普通更新对照。
`--profile-update` 仅剖析一次预热后的更新并输出算子表，该运行不作为 TPS 证据，不生成 profiler trace 档案。

旧 `--online` 选项仅指块长统计控制器，需要 `--blocks 1,4,8,16`；
`--shadow` 是固定 B=8 的控制器包装对照。这两者不与 `--fast-weights` 混合评估。

基准使用 batch=1、BF16、FA2、预捕获 CUDA graph、temperature=0、ignore_eos=True。
完整生成计时包含新增参数/optimizer 重置、特征缓存、teacher 复制、backward、Adam、同步发布，
以及 wrapper 安装恢复、prefill、decode、detokenization；
模型初始化、共同 prompt 编码、GPU 快照和 JSON 写盘单独排除。
保存输出 token IDs、文本及反馈，不因输出不同就剔除样本。

官方 prefill 的第一个 token 不进入 decode stats，因此 `accepts=max_new_tokens−1`；
每个 decode cycle 的 `forwards` 增加 2。TPF 和 E2E TPS 的分母不能混用。
审计检查完整矩阵、冻结权重字节 hash、无抢占、合法时间及更新次数/提交统计对账。
少量 prompts 的聚类 bootstrap 只作描述，不能证明稳定额外收益。

## 接入自己的请求

按基准脚本导入官方 `LLM`，在构造模型之前安装 extension，才能让 CUDA graph 捕获新分支：

```python
from native_fast_weights import extended_runner, generate_fast

with extended_runner(fused_norm=True, fast_weights=True, rank=8, stride=16,
                     replay_blocks=4, lr=0.001, training_backend="cuda_graph"):
    engine = LLM(model=base_path, **config)  # config 见 benchmark_native_uno.py
output, diagnostics = generate_fast(engine, prompt_ids, params, budget=512)
```

引擎必须为单 GPU、batch=1、线性 XLLM Uno，预捕获 B=8，params 使用同一 B。
每个请求自动重置新增参数，不并发共享 engine/wrapper，不做异步训练。
CUDA 更新图在引擎初始化时捕获；其初始化成本与原生推理图捕获一样单独记录，未摊入稳态 TPS。
生产 API 的 `capture_plain=False` 不分配额外对照图；基准自动设为 True，
需要手动比较时可以传入 `capture_plain=True`，再在 idle 状态使用 `with plain_uno(engine):`。
它会切换真实 graph runner，退出作用域后恢复；不能在该作用域内调用 `generate_fast`。
请求内 reset 和每次更新全部计入 E2E；warmup / capture 后复位 Adam 状态，绝不带入预热学习。
R≤S：只收集每个更新间隔最后 R 轮，更新后清空；不同参数版本的 logits 不允许混用。
作为库可在项目目录 `pip install -e .`，仅安装当前模块；GPU 依赖仍由 WSL 环境提供。

## 单元测试

测试使用合成引擎/合成记录及小型张量，覆盖 mask、梯度、参数发布与重置；不依赖历史结果。
在父仓库 PowerShell 中：

```powershell
.venv/Scripts/python.exe -B -m pytest -p no:cacheprovider online-speculation/tests -q
.venv/Scripts/python.exe -m ruff check --no-cache online-speculation
```

WSL 中安装 `python -m pip install pytest` 后，运行 `python -m pytest tests -q` 可额外执行
CUDA 融合 norm 的四组数值测试。Windows 会跳过这些 Triton 测试。

实现原理、梯度/特征闭合、KV 隔离、分布保持条件和有限精度边界见 [ALGORITHM.md](ALGORITHM.md)。
