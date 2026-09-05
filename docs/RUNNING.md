# 运行与验证

数学和设计只有一份：[ALGORITHM.md](ALGORITHM.md)。这里保留可执行入口及其适用范围。

## 1. 本机环境

本机 RTX 3090 24GB，WSL2 发行版名 `Ubuntu-22.04`，Linux 用户 `singm`。
已有环境 `/home/singm/.venvs/uno-cu128`，PyTorch 2.11.0+cu128 / Python 3.10。
主体不需要作者引擎、FlashAttention 或 Triton 扩展；当前使用 PyTorch SDPA。
不重新安装 WSL，也不安装 Linux 显示驱动。

以下在 WSL Bash 中运行：

```bash
cd /mnt/c/Users/singm/Desktop/hw/akg_related/inference_scaling/online-speculation
source /home/singm/.venvs/uno-cu128/bin/activate
python -m pip install -e '.[dev,text,hf]'
python -m pytest -q
```

若暂不安装 editable 包，可在命令前加 `PYTHONPATH=src`；测试配置本身已加入 `src`。
`hf` 是可选数值测试参照，不作为生成后端。本次没有创建下载临时脚本。
现有模型和依赖不是失败实验产物，清理不会卸载它们。

## 2. 小模型全闭环

```bash
python -m blockspec demo --device cuda \
  --base-steps 120 --adapter-steps 900 --loss forward_kl \
  --block-size 4 --tokens 128 --checkpoint models/cycle.pt
```

输出只写 stdout。若指定检查点，只保存该文件，不生成结果文档。
`models/` 被 Git 忽略，已有检查点不覆盖。前向 KL 是这个合成启动检查的设置，不是论文默认配方；
真实离线入口可用反向 KL 热身接 L1。

顺序为：训练 AR 小基座、冻结基座、训练草稿适配器、验证存取、生成固定／在线结果。
检查基座指纹不变、贪心输出一致、在线参数变化。周期数据的难度和启动成本都不代表真实语言模型，
stdout 的 TPS 仅用于排查数量级。

## 3. 真实权重有限检查

本机已有基座 `/home/singm/online-speculation-work/models/K2-Horizon-0.9B`，
来源 revision 和权重 SHA 见 [upstream.lock.json](../references/upstream.lock.json)。
**不加载已经发布的草稿适配器。**

```bash
PYTHONPATH=src python scripts/check_local_model.py \
  --base /home/singm/online-speculation-work/models/K2-Horizon-0.9B \
  --dtype float32 --tokens 32 --train-steps 4
```

脚本经本项目 Transformer 生成少量训练轨迹，执行离线更新，再比较三种解码和基座指纹。
报告配对老师与普通老师的数值误差、贪心是否一致、更新次数和峰值显存，默认不写文件。
四步训练只能验证链路，不能称为“训练充分的适配器性能”。
可用 `--dtype bfloat16` 检查低精度路径，但需明确处理批形状导致的数值差异，不能预设逐 token 一致。
加 `--sampler tree` 检查独立目标路径树；它使用精确的目标分布遍历，不假装确定性 top-k 树来自原 q 的独立抽样。
小模型 `demo` 也支持同一个 `--sampler tree` 开关。

接口明确接入 dense K2-Horizon 和 Qwen3 结构；未知架构、MoE、量化、滑动窗口、部分旋转、门控注意力会拒绝。
不要改 `model_type` 绕过检查。新增架构先补数值参照测试。

## 4. 用自己的数据训练全适配器

要准备当前小规模公开数据子集，可运行：

```bash
python -m blockspec prepare \
  --base /home/singm/online-speculation-work/models/K2-Horizon-0.9B \
  --output /home/singm/online-speculation-work/data/blockspec_ot3_small --page-size 8
```

该目录在本机已存在，可以直接复用；重新准备须选择新目录，不覆盖。
`prepare` 需要 `.[data]` 依赖，使用本地聊天模板；保存 train／validation／test 三个数据文件及来源 manifest。
按问题分组防止同题不同答案跨集合。`test.jsonl` 暂不参与训练和参数选择。
这是常规可复用准备入口，不生成下载临时脚本或残留原始响应。

本地 JSONL 一行一个独立样本，二选一：

```json
{"input_ids": [10, 25, 37, 16, 8]}
{"text": "一段已经按实际部署格式整理好的足够长的文本……"}
```

编号必须属于基座词表。文本需 `--text-data`，仅解析 `tokenizer.json`，不执行远程代码、不自动套聊天模板。
部署的角色标记、思考标记和语言风格应在训练文本中明确匹配。样本长度至少为 `sequence-length - 1`；
训练随机取连续片段再加指定 BOS，不把不同样本拼在一起。

```bash
python -m blockspec train \
  --base /home/singm/online-speculation-work/models/K2-Horizon-0.9B \
  --data /path/outside/repo/train.jsonl --text-data \
  --output models/current-adapter.pt --device cuda --dtype bfloat16 \
  --rank 8 --steps 1000 --warmup-steps 100 --warmup-loss reverse_kl \
  --loss l1 --batch-size 1 --sequence-length 128 --blocks 2,4,6,8 \
  --learning-rate 0.0001 --bos-id 0 --seed 314159
```

这是 3090 上保守的起步配置，不承诺已收敛或最优。1000 步包含热身，不额外增加。
数据不默认上传、不入 Git。数据和噪声分别设随机种子。训练前后检查基座指纹；
适配器检查点含模型配置、基座指纹，错误基座／精度会被拒绝。

增加 `--validation-data /path/validation.jsonl --validation-every 100 --validation-batches 8`，
可在训练前及每 100 步检查固定窗口、固定噪声上的学生／老师差异。训练与验证问题或样本重叠会拒绝启动。
检查点记录训练配置和输入文件 SHA，完整训练时间包含验证开销。更换精度不是无损加载同一适配器的默认操作。

诊断数值差异可运行 `python scripts/audit_model_precision.py --base /path/to/base`；
默认仅打印摘要，`--trace` 逐层打印，也不写文件。该命令只作定位，不作为 TPS 基准。

## 5. 接进后续请求

```python
import torch
from blockspec.checkpoint import load_hf_base, load_checkpoint, save_checkpoint
from blockspec.decoding import generate_ar, generate_speculative
from blockspec.online import OnlineConfig, OnlineLearner
from blockspec.tokenizer import LocalTokenizer

base = "/home/singm/online-speculation-work/models/K2-Horizon-0.9B"
model = load_hf_base(base, rank=8, dtype=torch.bfloat16, device="cuda")
model, _ = load_checkpoint("models/current-adapter.pt", model=model, device="cuda")
tokenizer = LocalTokenizer(base)
learner = OnlineLearner(model, OnlineConfig(stride=16, replay_blocks=4, learning_rate=1e-4))
for text in ["Explain binary search.", "Now explain its loop invariant."]:
    prompt = torch.tensor([[0] + tokenizer.encode(text)], device="cuda")
    output = generate_speculative(model, prompt, 128, block_size=8, learner=learner)
    print(tokenizer.decode(output.tokens))
    print(output.summary())
save_checkpoint("models/continued-adapter.pt", model, adapter_only=True)
```

同一个 learner 保留权重、optimizer 和计数；请求结束释放重放 KV 和老师 logits。
重新构造 learner 新建 optimizer。磁盘当前保存权重，不恢复 Adam 动量。
不传 learner 是静态适配器；`generate_ar` 是每 token 一次前向的非投机基线。
树入口为 `from blockspec.tree import generate_tree`，接受同一个 learner，另有
`top_k` 和 `prefix_budget`（含根节点的预算）参数。树模式的 `proposed` 统计非根节点数，不是线性块长度。
默认贪心；随机采样传 `SamplingConfig(temperature=1, top_k=50, top_p=0.95)`。
EOS 要显式传本模型 `eos_id`；不传则按预算生成，不得把这一设置隐瞒成自然结束。

## 6. 同条件三路评测

当前 r=32 / FP32 离线起点在仓库外，已用训练集做 600 步训练；不能用上节 r=8 / BF16 的示例配置加载它。
开发评测使用验证集中的连续前缀，不是重新编写的指令任务；保留测试集不用于选配置。

```bash
python -m blockspec benchmark \
  --base /home/singm/online-speculation-work/models/K2-Horizon-0.9B \
  --adapter /home/singm/online-speculation-work/models/blockspec-r32-fp32-offline.pt \
  --data /home/singm/online-speculation-work/data/blockspec_ot3_small/validation.jsonl \
  --split-role validation --dtype float32 --prompts 4 --prompt-length 128 \
  --tokens 128 --block-size 4 --repeats 2 --update-stride 32 --replay-blocks 1
```

按文件顺序取前 4 个足够长记录的前 128 项作为输入。默认贪心、固定输出预算，**不按 EOS 提前结束**；
需要自然结束时显式加 `--eos-id 1`。`--sampler tree --prefix-budget 16 --top-k 4` 切换树路径。
重复次数必须为正偶数：AR／静态／在线，再在线／静态／AR。每个在线请求流从同一离线起点开始，
流内保留学习后的权重与 Adam 状态；静态版不学习。内核预热不改变正式起点，第一次真实 Adam 更新的分配成本仍计入。

输出汇总 TPS、每轮输出数、更新时间、含 learner 初始化的 TPS、逐请求贪心一致性、峰值显存和输入／实现 SHA。
输出不同会标为 `greedy_identical: false`，不会挑掉该样本再计算“等价加速”。`--progress` 可打印逐请求计数。
实验结束恢复传入模型的适配器，不把调参过程的在线权重发布回检查点；默认只写 stdout。
此处少量开发流用于选实现，不提供统计显著性或公开基准排名。

## 7. 完整计时与文件生命周期

`Generation.seconds` 从 prefill 前开始，包含生成、反馈、在线更新、清理本请求反馈和末尾 GPU 同步。
不含加载模型、离线训练、编码提示、存取检查点和转回文字；产品级延迟须另计这些项目。
`update_seconds` 仅为重放更新区域，不能代替净开销。learner 的一次初始化在请求前，冷启动评测须单列。

比较时固定基座、离线起点、采样、精度、输出预算、缓存条件。预热后正反顺序成对运行，报告总 tokens / 总秒数，
不是每条 TPS 的算术平均。AR 也使用相同优化。不要在另一项 GPU 训练同时运行时测速度。
合成模型单次结果不作为论文性能证据。

只提交当前代码、测试、必要配置、主报告和本说明。原始结果、下载中间文件、权重、日志不提交。
失败设计最多在主报告解释原因；无效实现删除，由 Git 历史追溯，不复制进 archive。
只清理已确认由本项目产生且无用的目标，不删除不明缓存或系统安装材料。
