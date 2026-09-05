# 当前原生基线与在线控制器的运行说明

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
  --blocks 1,4,8,16 --online --shadow \
  --repetitions 2 --max-new-tokens 128 --warmup-tokens 128 \
  --output results/native_run.json

python scripts/analyze_native_uno.py \
  --input results/native_run.json --output results/native_run_audit.json
```

这是四个内置 prompts 的工程检查入口，不是完整论文复现或独立 held-out 评估。
`results/` 全部被 Git 忽略，没有默认提交例外。运行时使用新文件名，不覆盖已有结果。
源码树不保存旧成绩、失败轨迹或结果归档。

B=1 是原引擎 AR 对照；B=4/8/16 是固定块长 Uno；`--online` 使用当前块长控制器；
`--shadow` 让同一 wrapper 只选择 B=8，检查外围包装是否改变固定宽度行为。
**这里的 online 是策略统计更新，不是在线 LoRA 训练。**

基准使用 batch=1、BF16、FA2、预捕获 CUDA graph、temperature=0、ignore_eos=True。
完整生成计时包含控制器构造和更新、wrapper 安装恢复、prefill、decode、detokenization；
模型初始化、共同 prompt 编码、GPU 快照和 JSON 写盘单独排除。
保存输出 token IDs、文本及反馈，不因输出不同就剔除样本。

官方 prefill 的第一个 token 不进入 decode stats，因此 `accepts=max_new_tokens−1`；
每个 decode cycle 的 `forwards` 增加 2。TPF 和 E2E TPS 的分母不能混用。
审计检查完整矩阵、参数冻结、无抢占、合法时间及策略反馈/提交统计对账。
少量 prompts 的聚类 bootstrap 只作描述，不能证明稳定额外收益。

## 接入自己的请求

`NativeWidthPolicy` 与 `generate_online` 都位于 [native_online_policy.py](../scripts/native_online_policy.py)。
按基准脚本构造空闲 `LLM`、对应 chat formatter 的 prompt_ids 和 SamplingParams 后调用：

```python
from native_online_policy import NativeWidthPolicy, generate_online

output, diagnostics = generate_online(
    engine, prompt_ids, params, budget=128, policy=NativeWidthPolicy()
)
```

引擎必须为单 GPU、batch=1、线性 Uno，预捕获所有允许块长，最大块长至少为 16。
每个请求新建 policy，不复用状态，不并发共享同一个 engine/wrapper。
作为库可在项目目录 `pip install -e .`，只安装当前控制器模块，不带回旧 HF 原型命令。

## 单元测试

当前测试使用合成引擎/合成记录，不依赖历史结果，也不产生新的 GPU 实验成绩。
在父仓库 PowerShell 中：

```powershell
.venv/Scripts/python.exe -B -m pytest -p no:cacheprovider online-speculation/tests -q
.venv/Scripts/python.exe -m ruff check --no-cache online-speculation
```

实现原理、EMA 更新、数学分布保持条件和有限精度边界见 [ALGORITHM.md](ALGORITHM.md)。
