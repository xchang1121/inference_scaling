# 运行与验证

数学推导与算法设计见 [ALGORITHM.md](ALGORITHM.md)。本说明提供环境配置、运行命令和测量方法。

## 1. 本机环境

本机 RTX 3090 24GB，WSL2 发行版名 `Ubuntu-22.04`，Linux 用户 `singm`。
已有环境 `/home/singm/.venvs/uno-cu128`，PyTorch 2.11.0+cu128 / Python 3.10。
注意力使用 PyTorch SDPA。

以下在 WSL Bash 中运行：

```bash
cd /mnt/c/Users/singm/Desktop/hw/akg_related/inference_scaling/online-speculation
source /home/singm/.venvs/uno-cu128/bin/activate
python -m pip install -e '.[dev,text,hf]'
python -m pytest -q
```

也可通过命令前的 `PYTHONPATH=src` 使用源码；测试配置已加入 `src`。
`hf` 提供可选数值参照。

## 2. 小模型全闭环

```bash
python -m blockspec demo --device cuda \
  --base-steps 120 --adapter-steps 900 --loss forward_kl \
  --block-size 4 --tokens 128 --checkpoint models/cycle.pt
```

结果写入 stdout；`--checkpoint` 指定保存位置，使用新文件名。`models/` 被 Git 忽略。
此周期序列示例使用前向 KL；真实离线入口默认反向 KL＋L1 联合热身，再接纯 L1。

顺序为：训练 AR 小基座、冻结基座、训练草稿适配器、验证存取、生成固定／在线结果。
检查基座指纹、贪心输出和在线参数更新，并打印各阶段耗时。

## 3. 真实权重检查

本机已有基座 `/home/singm/online-speculation-work/models/K2-Horizon-0.9B`，
来源 revision 和权重 SHA 见 [upstream.lock.json](../references/upstream.lock.json)。
草稿适配器由本项目训练。

```bash
PYTHONPATH=src python scripts/check_local_model.py \
  --base /home/singm/online-speculation-work/models/K2-Horizon-0.9B \
  --dtype float32 --tokens 32 --train-steps 4
```

脚本经本项目 Transformer 生成少量训练轨迹，执行离线更新，再比较三种解码和基座指纹。
stdout 报告配对／普通老师的数值误差、贪心一致性、更新次数和峰值显存。
四步训练用于集成检查；`--dtype bfloat16` 用于定位低精度和批形状带来的差异。
`--sampler tree` 切换为从目标分布逐节点抽样的树路径。
小模型 `demo` 也支持同一个 `--sampler tree` 开关。

权重接口接入 dense K2-Horizon 和 Qwen3，加载时校验模型结构。新增架构时配套添加数值参照测试。

完整基座外部数值参照使用隔离的依赖目录：

```bash
# 首次配置时安装；本机已有此目录，可直接运行下面的校验命令。
python -m pip install --target /home/singm/online-speculation-work/oracle-transformers515 \
  --only-binary=:all: --no-cache-dir transformers==5.15.0
PYTHONPATH=/home/singm/online-speculation-work/oracle-transformers515 \
  HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 python scripts/audit_hf_reference.py \
  --base /home/singm/online-speculation-work/models/K2-Horizon-0.9B \
  --data /home/singm/online-speculation-work/data/blockspec_ot3_small/validation.jsonl \
  --prompts 4 --prompt-length 128 --tokens 32 --execution cuda_graph --require-same-argmax
```

该脚本执行经过 hash 核对的本地基座作者模型，与本项目 `Decoder` 对照。
代码和配置按 LF 换行核对，权重逐字节核对，索引指向已校验的分片。结果写入 stdout。
默认 logits 最大误差阈值为 0.0005、TV 为 0.0001；上述命令还要求 argmax 全部相同，超出门槛时以错误码退出。
`--execution eager` 使用普通执行路径。移位短窗口检查大位置编号的计算，校验输入取自开发集。

## 4. 用自己的数据训练全适配器

要准备当前小规模公开数据子集，可运行：

```bash
python -m blockspec prepare \
  --base /home/singm/online-speculation-work/models/K2-Horizon-0.9B \
  --output /home/singm/online-speculation-work/data/blockspec_ot3_small --page-size 8
```

该目录在本机已存在，可以直接复用；重新准备时选择新目录。
`prepare` 需要 `.[data]` 依赖，使用本地聊天模板；保存 train／validation／test 三个数据文件及来源 manifest。
同题的所有答案归入同一集合。训练、参数选择、最终评测分别使用 train、validation、test。

本地 JSONL 一行一个独立样本，二选一：

```json
{"input_ids": [10, 25, 37, 16, 8]}
{"text": "一段已经按实际部署格式整理好的足够长的文本……"}
```

编号属于基座词表。文本输入加 `--text-data`，通过本地 `tokenizer.json` 编码；聊天模板由输入文本提供。
部署的角色标记、思考标记和语言风格应在训练文本中明确匹配。样本长度至少为 `sequence-length - 1`；
训练从每个独立样本随机取连续片段，再加指定 BOS。

```bash
python -m blockspec train \
  --base /home/singm/online-speculation-work/models/K2-Horizon-0.9B \
  --data /home/singm/online-speculation-work/data/blockspec_ot3_small/train.jsonl \
  --validation-data /home/singm/online-speculation-work/data/blockspec_ot3_small/validation.jsonl \
  --output models/current-adapter.pt --device cuda --dtype float32 \
  --rank 32 --steps 600 --warmup-steps 150 --warmup-loss reverse_kl_l1 \
  --loss l1 --batch-size 2 --sequence-length 128 --blocks 2,4 \
  --learning-rate 0.0003 --validation-every 100 --validation-batches 8 --bos-id 0 --seed 314159
```

上述配置已在本机 3090 执行，600 步由 150 步热身与 450 步纯 L1 训练组成。
纯文本 JSONL 通过 `--text-data` 启用文本编码；当前数据已经保存为 token 编号，直接使用上述命令。
数据保存在仓库外。数据和噪声分别设随机种子。训练前后检查基座指纹；
适配器检查点保存模型配置和基座指纹，加载器据此核对基座与精度。

增加 `--validation-data /path/validation.jsonl --validation-every 100 --validation-batches 8`，
可在训练前及每 100 步检查固定窗口、固定噪声上的学生／老师差异。启动时校验训练与验证集合的独立性。
检查点记录训练配置和输入文件 SHA，训练时间包含验证开销。加载时使用检查点对应的基座、秩和精度。

诊断数值差异可运行 `python scripts/audit_model_precision.py --base /path/to/base`；
默认打印摘要，`--trace` 逐层打印，用于定位最早出现数值差异的运算。

可选的固定公开实现契约对照：

```bash
python scripts/audit_sampler_reference.py \
  --source /mnt/c/Users/singm/Desktop/hw/akg_related/.tmp_uno_upstream
```

脚本读取固定 Git 对象，对照 CPU 树构建与给定目标 token 的遍历，向 stdout 打印节点和路径检查结果。

## 5. 接进后续请求

```python
import torch
from blockspec.checkpoint import load_hf_base, load_checkpoint, save_checkpoint
from blockspec.decoding import generate_ar, generate_speculative
from blockspec.online import OnlineConfig, OnlineLearner
from blockspec.tokenizer import LocalTokenizer

base = "/home/singm/online-speculation-work/models/K2-Horizon-0.9B"
model = load_hf_base(base, rank=32, dtype=torch.float32, device="cuda")
model, _ = load_checkpoint("/home/singm/online-speculation-work/models/blockspec-r32-fp32-paper.pt",
                           model=model, device="cuda")
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
`OnlineConfig(train_last_layers=4, stride=32, replay_blocks=1)` 提供末层续训配置，
续训最后 4 层中的适配器，并复用起草的前段特征；其余适配器以固定权重参与起草。
`train_last_layers=None` 对应全适配器续训。修改冻结前段后重新构造 learner；数学条件见主报告 9.3。
检查点保存权重，新建 learner 时初始化 Adam 状态。
`learner=None` 对应静态适配器；`generate_ar` 每 token 执行一次前向。
树入口为 `from blockspec.tree import generate_tree`，接受同一个 learner，另有
`top_k` 和 `prefix_budget`（含根节点的预算）参数。树模式的 `proposed` 统计根节点之外的候选节点总数。
默认贪心；随机采样传 `SamplingConfig(temperature=1, top_k=50, top_p=0.95)`。
`eos_id=None` 按固定预算生成；设置本模型的 `eos_id` 可在结束标记处停止。

## 6. 同条件三路评测

当前离线起点在仓库外，使用 r=32 / FP32 加载，训练配置为 600 步联合热身课程。
开发评测使用验证集中的连续前缀；test 集留给配置确定后的评测。

```bash
python -m blockspec benchmark \
  --base /home/singm/online-speculation-work/models/K2-Horizon-0.9B \
  --adapter /home/singm/online-speculation-work/models/blockspec-r32-fp32-paper.pt \
  --data /home/singm/online-speculation-work/data/blockspec_ot3_small/validation.jsonl \
  --split-role validation --dtype float32 --prompts 8 --prompt-length 256 \
  --tokens 256 --block-size 4 --repeats 2 --warmup-tokens 32 --update-stride 32 --replay-blocks 1
```

按文件顺序取前 8 个足够长记录的前 256 项作为输入。默认贪心、固定输出预算、`eos_id=None`；
`--eos-id 1` 启用结束标记。`--sampler tree --prefix-budget 16 --top-k 4` 切换树路径。
重复次数取正偶数，每对按 AR／静态／在线、在线／静态／AR 的顺序运行。每个在线请求流从同一离线起点开始，
在线流内保留学习后的权重与 Adam 状态，静态流保持离线权重。预热结束恢复正式起点，第一次 Adam 状态分配计入更新时间。

输出汇总 TPS、每轮输出数、更新时间、含 learner 初始化的 TPS、逐请求贪心一致性、峰值显存和输入／实现 SHA。
输出差异标为 `greedy_identical: false`。`--progress` 打印逐请求计数。
加 `--online-last-layers 4` 测相同离线起点的末 4 层续训；会另外报告实际可训练参数数。
全量续训使用默认设置，末层续训通过该参数选择可训练的层数。
重跑主报告当前表格：在上述命令增加 `--sampler tree --top-k 4 --prefix-budget 12 --online-last-layers 4`。
全适配器树基线对应上述命令增加 `--sampler tree --top-k 4 --prefix-budget 12`。

加 `--execution cuda_graph` 启用同一个独立固定形状执行器，AR、静态和在线都使用它。
本机当前表格使用此开关；默认 `eager` 使用普通执行路径。
图在预热和请求计时前准备，输出 `execution.setup_seconds_by_arm` 和 `tps_including_all_setup`，
分别报告每个方法单独部署所需的图准备时间，以及包含图准备和 learner 初始化的流级 TPS。
请求计时包括快照复制、prefix 传输和在线更新。图跨请求保留，适配器在固定存储中原地更新。
执行配置为 FP32 CUDA、batch=1、TF32 关闭。
长 prefill 普通执行，短查询按预先准备的形状和历史容量执行，入口校验查询尺寸。
实验结束恢复传入模型的适配器，结果写入 stdout。

## 7. 完整计时与文件生命周期

`Generation.seconds` 从 prefill 前开始，包含生成、反馈、在线更新、清理本请求反馈和末尾 GPU 同步。
`update_seconds` 记录重放更新区域。learner 初始化和图准备单列，并计算包含这两项的流级 TPS。
完整服务流程的延迟还包括模型加载、提示编码、检查点存取和文本解码；离线训练时间单独记录。

比较时固定基座、离线起点、采样、精度、输出预算、缓存条件。预热后正反顺序成对运行，报告总 tokens / 总秒数。
三路使用相同优化，测量期间独占实验 GPU。

仓库保存代码、测试、配置、主报告和运行说明。权重与数据存放于仓库外，结果通过 stdout 查看。
设计取舍在主报告相关章节简述，代码历史通过 Git 追溯。
