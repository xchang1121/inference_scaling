# 运行与验证

数学推导与算法设计见 [ALGORITHM.md](ALGORITHM.md)。本说明提供环境配置、运行命令和测量方法。

## 1. 本机环境

本机 RTX 3090 24GB，WSL2 发行版名 `Ubuntu-22.04`，Linux 用户 `singm`。
已有环境 `/home/singm/.venvs/uno-cu128`，PyTorch 2.11.0+cu128 / Python 3.10。
注意力提供 PyTorch SDPA 与分组短查询路径。
官方引擎参照使用已安装的 FlashAttention 2.8.3，独立实现继续使用自身的注意力入口。

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
当前性能对照采用公开适配器，版本与校验值列于来源清单。

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
  --prompts 4 --prompt-length 512 --tokens 64 --execution cuda_graph \
  --attention-backend grouped --require-same-argmax
```

该脚本执行经过 hash 核对的本地基座作者模型，与本项目 `Decoder` 对照。
代码和配置按 LF 换行核对，权重逐字节核对，索引指向已校验的分片。结果写入 stdout。
默认 logits 最大误差阈值为 0.0005、TV 为 0.0001；上述命令还要求 argmax 全部相同，超出门槛时以错误码退出。
`--attention-backend grouped` 在本项目模型中启用分组短查询，外部参照继续使用其原有注意力。
`--execution eager` 使用普通执行路径。移位短窗口检查大位置编号的计算，校验输入取自开发集。
BF16 外部审计添加 `--dtype bfloat16 --max-logit-error 0.5 --max-tv 0.02`；
数值验证的定义见[主报告](ALGORITHM.md#72-数值与缓存验证)，BF16 吞吐同时报告概率误差与最大项一致性。
`--trace` 在第一条共同 prefill 上比较各层；`--bf16-full-accumulation` 让两边 GEMM 采用完整累加精度，
结束后恢复调用方的全局设置。输出给出实际累加开关。

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
  --rank 32 --alpha 32 --steps 1200 --warmup-steps 300 --warmup-loss reverse_kl_l1 \
  --loss l1 --batch-size 2 --sequence-length 256 --blocks 2,4,6,8 \
  --learning-rate 0.0003 --validation-every 300 --validation-batches 8 --bos-id 0 --seed 314159
```

上述配置已在本机 3090 执行，1,200 步由 300 步热身与 900 步纯 L1 训练组成，四种块长各训练 300 步。
纯文本 JSONL 通过 `--text-data` 启用文本编码；当前数据已经保存为 token 编号，直接使用上述命令。
数据保存在仓库外。数据和噪声分别设随机种子。训练前后检查基座指纹；
适配器检查点保存模型配置和基座指纹，加载器据此核对基座与精度。
`--alpha` 控制低秩分支的缩放 $\alpha/r$，默认取 `--rank` 的值。
评测入口从检查点恢复秩与缩放；通过 Python 直接构造基座时，将相同的值传给 `load_hf_base(rank=..., alpha=...)`。

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
model, _ = load_checkpoint("/home/singm/online-speculation-work/models/blockspec-r32-fp32-curriculum8.pt",
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
`train_last_layers=None` 对应全适配器续训。修改冻结前段后重新构造 learner；数学条件见[冻结表示与条件修正](ALGORITHM.md#61-实际前缀的信息)。
词表投影按有效监督位置计算；Transformer 继续使用完整块的注意力关系，梯度等价推导同见附录 B。
`optimizer="auto"` 在 CUDA 上使用融合 AdamW，在 CPU 上使用标准版本；更新核对见 [test_online_execution.py](../tests/test_online_execution.py)。
`feedback_execution="windowed"` 按更新需求安排采集窗口，`all` 使用逐轮采集；参数更新对照见 [test_feedback_window.py](../tests/test_feedback_window.py)。
`update_policy="coverage"` 在检查时点读取窗口的路径覆盖标记：完整覆盖时保留参数与 Adam，含覆盖缺口时用全部有效位置训练。
`periodic` 为默认周期更新对照；门控及参数状态核验见 [test_update_policy.py](../tests/test_update_policy.py)。
`observe()` 接口逐次接收显式反馈；窗口化由线性／树解码器调度，参数与 Adam 状态跨请求续用。
显式反馈的 `fully_covered` 默认为 False；调用方设置 True 时，需要实际路径匹配全部带噪位置，并提供对应的全部老师行。
检查点保存权重，新建 learner 时初始化 Adam 状态。
`learner=None` 对应静态适配器；`generate_ar` 每 token 执行一次前向。
树入口为 `from blockspec.tree import generate_tree`，接受同一个 learner，另有
`top_k` 和 `prefix_budget`（含根节点的预算）参数。树模式的 `proposed` 统计根节点之外的候选节点总数。
默认贪心；随机采样传 `SamplingConfig(temperature=1, top_k=50, top_p=0.95)`。
`eos_id=None` 按固定预算生成；设置本模型的 `eos_id` 可在结束标记处停止。

末层训练图可在进入请求流前显式准备：

```python
from blockspec.replay_execution import SuffixReplayExecutor

model.set_attention_backend("grouped")
learner = OnlineLearner(model, OnlineConfig(train_last_layers=4, stride=16, replay_blocks=1,
                                          loss="forward_kl", learning_rate=0.0003,
                                          update_policy="coverage"))
replay = SuffixReplayExecutor(model, start_layer=learner.capture_layer, loss=learner.config.loss,
                              capacity=768, max_query=4)
replay.prepare([(b, m) for b in range(2, 5) for m in range(1, b)])
learner.replay_executor = replay
```

输入使用 FP32 CUDA、batch=1，历史长度在容量范围内；块长和有效监督行数通过 `prepare()` 显式准备。
准备好的 `replay` 可传给同一模型的新 `OnlineLearner(model, config, replay_executor=replay)`，
新 learner 使用相同的末层范围和损失，初始化自己的 Adam 状态。特征复用条件见[主报告](ALGORITHM.md#61-实际前缀的信息)。
`model.set_attention_backend("grouped")` 在创建 learner 和推理／训练图之前调用；`sdpa` 为按头对照。
分组路径处理至多 32 项的 FP32／FP64 查询，长 prefill 和低精度查询由 SDPA 计算，实现见 [attention.py](../src/blockspec/attention.py)。
结束执行器的使用时，先清理反馈、解除各 learner 的 `replay_executor` 引用，再释放调用方持有的执行器。
图槽与工作区随最后一个执行器引用释放；基座和仍在使用的 Adam 状态保持各自的生命周期。

## 6. 离线复现与同条件三路评测

当前优先使用独立两路入口。它逐请求交换先后顺序，固定 adapter，并分别核算 AR 与静态起草的图准备成本：

```bash
python scripts/benchmark_offline.py \
  --base /home/singm/online-speculation-work/models/K2-Horizon-0.9B \
  --adapter /home/singm/online-speculation-work/models/blockspec-r32-fp32-block4-context512.pt \
  --data /home/singm/online-speculation-work/data/blockspec_ot3_small/validation.jsonl \
  --prompts 17 --prompt-length 256 --tokens 512 --block-size 4 \
  --sampler tree --top-k 8 --prefix-budget 16 --repeats 2 --progress
```

公开权重对照使用同一入口，将 adapter 改为目录
`/home/singm/online-speculation-work/models/K2-Horizon-0.9B-Uno`，并添加
`--reference-sha256 5a499229d19ef4a69eb0b21884819d1b67cd983ba02b7ee2031ba8567dedfe4e`。
输出标明权重来源、文件 SHA、代码指纹、逐 token 比较、分轮吞吐和显存。
两路和三路入口均可添加 `--temperature 1 --sampling-top-k 0 --top-p 1`，运行完整词表的温度 1 采样。
目标分布的 top-k 使用 `--sampling-top-k`，树候选宽度继续使用 `--top-k`。
参数同时应用于 AR、静态、在线和各自预热；`config.sampling` 保存实际设置。
温度为零时输出 `comparison_mode: greedy_exact`；正温度时为 `stochastic_diagnostic`，
`greedy_identical: null`，逐请求的 token 比较作为观测信息。输出分布的推导见[概率校正](ALGORITHM.md#51-单位置的质量守恒)。
`tokens_per_decode_forward` 报告平均每次生成前向产出的 token 数，TPS 计时包含 prefill。
两路入口默认 FP32 CUDA，`--dtype bfloat16` 切换基座执行精度；本地检查点先按 FP32 来源验证，再转换基座，保留适配器主权重。
三路入口的 `--dtype` 指定检查点对应的基座精度，`--execution-dtype bfloat16` 在加载后执行同一转换。
`--noise-low 1` 对齐公开 1B 引擎的均匀噪声范围；`--noise-high` 为右侧开区间端点，默认取词表大小。
默认噪声为 `[0, vocab_size)`，沿用当前自训权重的约定。两路／三路的 `config.noise` 保存实际范围。
`--sampler linear --block-size 8` 提供公开入口默认线性块形状的本机对照。
数学与数值审计入口 `scripts/audit_decode_path.py` 接收相同的 `--base`、公开 `--adapter`、
`--reference-sha256` 和 `--data`，另用 `--request 16 --token-index 125 --seed 271844` 定位第 17 条提示的第 126 个输出。
它在共同历史上对齐树的祖先路径与 AR，输出目标 logits、前两项间隔和 TV；带观测的运行用于数值分析。
同一审计入口接受本地自训检查点。添加 `--online-stream --stream-prompts 17 --repeat-index 1 --stream-seed 271828`
时，先从离线起点重放前面的请求，再观测指定请求；默认在线配置与下面的三路命令一致。
本次在线路径观测使用当前自训 adapter、`--request 16 --token-index 125`，输出跨请求更新版本和实际请求种子。
单请求静态审计使用 `--seed`；在线流使用 `--stream-seed + repeat-index × stream-prompts + request`。
`--device cpu --execution eager --online-execution eager` 提供小模型的 CPU 审计测试。
离线续训在第 4 节训练命令中增加 `--initial-adapter /path/to/starting-adapter.pt`，同时传入该起点对应的 rank 与 alpha。
新检查点记录起点 SHA，Adam 重新初始化。验证文件用于选择配置，保留测试用于确定配置后的验收。

在线三路入口继续用于后续增量收益的比较：

当前离线起点在仓库外，使用 r=32、$\alpha=32$ / FP32 加载；先完成 1,200 步的 2→4→6→8 课程，
再以 B=4、2×512 窗口、L1 和学习率 $10^{-4}$ 续训 1,200 步。检查点元数据保存训练配置，评测输出记录文件 SHA。
开发评测使用验证集中的连续前缀；test 集留给配置确定后的评测。

```bash
python -m blockspec benchmark \
  --base /home/singm/online-speculation-work/models/K2-Horizon-0.9B \
  --adapter /home/singm/online-speculation-work/models/blockspec-r32-fp32-block4-context512.pt \
  --data /home/singm/online-speculation-work/data/blockspec_ot3_small/validation.jsonl \
  --split-role validation --dtype float32 --prompts 17 --prompt-length 256 \
  --tokens 512 --block-size 4 --repeats 2 --warmup-tokens 32 \
  --sampler tree --top-k 8 --prefix-budget 16 --execution cuda_graph \
  --online-last-layers 4 --update-stride 16 --replay-blocks 1 \
  --loss forward_kl --learning-rate 0.0003 --optimizer auto --feedback-execution windowed \
  --update-policy coverage --online-execution cuda_graph --attention-backend grouped
```

前期 BF16 适配器续训测量沿用上述命令，添加 `--execution-dtype bfloat16`，
并将 `--online-execution cuda_graph` 改为 `--online-execution eager`。检查点来源精度继续使用 `--dtype float32`。
该配置保留全部 FP32 适配器主权重，三路共同使用 BF16 基座；数值误差说明见[主报告](ALGORITHM.md#72-数值与缓存验证)。

按文件顺序取全部 17 个验证记录的前 256 项作为输入，包含 4 个代码请求和 13 个数学请求。默认贪心、固定输出预算、`eos_id=None`；
`--eos-id 1` 启用结束标记；`--sampler linear` 切换线性路径。
重复次数取正偶数，每对按 AR／静态／在线、在线／静态／AR 的顺序运行。每个在线请求流从同一离线起点开始，
在线流内保留学习后的权重与 Adam 状态，静态流保持离线权重。预热结束恢复正式起点，第一次 Adam 状态分配计入更新时间。

输出汇总 TPS、每轮输出数、更新时间、含 learner 初始化的 TPS、逐请求贪心一致性、峰值显存和输入／实现 SHA。
输出差异标为 `greedy_identical: false`。`--progress` 打印逐请求计数。
`trajectories` 返回按请求顺序排列的累计 TPS、每轮输出、学习耗时和适配器版本；图准备费用归入第一轮重复。
`--loss forward_kl` 使用老师到学生的 KL，`--loss l1` 使用概率差的绝对值总和。
`--optimizer auto` 自动选择执行后端，输出 `online_optimizer` 给出实际选择；`--optimizer standard` 提供标准 AdamW 对照。
`--feedback-execution all` 提供逐轮采集对照；`feedback_blocks` 给出实际保存的反馈块数，并计入累计轨迹。
`--update-policy periodic` 提供周期更新对照；`coverage_skips` 统计完整覆盖窗口保留状态的检查次数，
`fully_covered_rounds` 统计实际完整覆盖的验证轮次，`updates` 统计真正执行的训练次数。
`--online-last-layers 4` 选择末 4 层续训，输出报告实际可训练参数数。
`--online-execution cuda_graph` 准备末层前向／损失／梯度图，`eager` 使用普通重放路径。
`--attention-backend grouped` 对 AR、起草、验证和在线重放共同设置分组短查询；`sdpa` 提供按头对照。
结果的 `config.attention_backend` 记录本次选择，评测完成恢复调用方的注意力配置。
全适配器续训使用默认层范围和 `--online-execution eager`。

加 `--execution cuda_graph` 启用同一个独立固定形状执行器，AR、静态和在线都使用它。
本机当前表格使用此开关；默认 `eager` 使用普通执行路径。
推理图在预热和请求计时前准备，`execution.setup_seconds_by_arm` 报告各方法的推理图准备成本。
末层梯度图在正式请求流前准备，`online_execution.setup_seconds` 报告这部分准备耗时，
其成本归入在线方法；`tps_including_all_setup` 包含推理图、训练图和 learner 初始化。
请求计时包括快照复制、prefix 传输和在线更新。图跨请求保留，适配器在固定存储中原地更新。
末层窗口化采集使用带分界特征和普通起草两组图；自建执行器时在 `prepare()` 中准备这两组查询形状。
图内部按新增位置打包 KV，再与有效历史拼成独立快照；缓存关系见[历史缓存不变量](ALGORITHM.md#32-历史缓存不变量)。
推理图支持 FP32／BF16 CUDA、batch=1；末层梯度图使用 FP32，BF16 在线续训选择 `--online-execution eager`。
FP32 的测量保持 TF32 关闭，BF16 基座对应独立的精度配置；执行副本由 FP32 适配器主权重转换得到。
长 prefill 普通执行，短查询按预先准备的形状和历史容量执行，入口校验查询尺寸。
实验结束恢复传入模型的适配器，结果写入 stdout。

固定官方引擎的外部参照使用单独入口：

```bash
python scripts/benchmark_reference.py \
  --checkout /mnt/c/Users/singm/Desktop/hw/akg_related/.tmp_uno_upstream \
  --base /home/singm/online-speculation-work/models/K2-Horizon-0.9B \
  --adapter /home/singm/online-speculation-work/models/K2-Horizon-0.9B-Uno \
  --data /home/singm/online-speculation-work/data/blockspec_ot3_small/validation.jsonl \
  --cache /home/singm/online-speculation-work/hf_modules \
  --prompts 17 --prompt-length 256 --tokens 512 --block-size 8 --temperature 1 --repeats 2
```

源码 commit 和两份权重的 SHA 来自 `references/upstream.lock.json`，每个方法在新进程运行。
该入口调用官方 BF16／FA2 引擎的 B=1 串行对照和 B=8 线性投机，精确限制各请求输出长度，
报告 TPS、初始化、预热、接受计数、解码前向、显存和图计数。结果见[当前基线](ALGORITHM.md#73-当前基线)，计时口径见第 7.1 节。
自有实现的相近条件使用两路公开 adapter 命令，并添加
`--dtype bfloat16 --sampler linear --block-size 8 --temperature 1 --noise-low 1`。

## 7. 完整计时与文件生命周期

`Generation.seconds` 从 prefill 前开始，包含生成、反馈、在线更新、清理本请求反馈和末尾 GPU 同步。
`update_seconds` 记录重放更新区域。learner 初始化和图准备单列，并计算包含这两项的流级 TPS。
完整服务流程的延迟还包括模型加载、提示编码、检查点存取和文本解码；离线训练时间单独记录。

比较时固定基座、离线起点、采样、精度、输出预算、缓存条件。预热后正反顺序成对运行，报告总 tokens / 总秒数。
三路使用相同优化，测量期间独占实验 GPU。

仓库保存代码、测试、配置、主报告和运行说明。权重与数据存放于仓库外，结果通过 stdout 查看。
设计取舍在主报告相关章节简述，代码历史通过 Git 追溯。

## 8. PrefixRelay 训练和配对实验

当前入口使用官方扩散适配器目录，默认 BF16 自有 GPU 图骨干。
训练对象是新增的转移与置信度小头，基座及公开适配器保持冻结。
先运行第 6 节的官方权重两路基线，选择并锁定块长，再启动小头训练。

```bash
python scripts/prefix_relay.py train \
  --base /home/singm/online-speculation-work/models/K2-Horizon-0.9B \
  --adapter /home/singm/online-speculation-work/models/K2-Horizon-0.9B-Uno \
  --reference-sha256 5a499229d19ef4a69eb0b21884819d1b67cd983ba02b7ee2031ba8567dedfe4e \
  --train-data /home/singm/online-speculation-work/data/blockspec_ot3_medium/train.jsonl \
  --data /home/singm/online-speculation-work/data/blockspec_ot3_small/validation.jsonl \
  --head /home/singm/online-speculation-work/models/prefixrelay-official-r64-b8.pt \
  --backbone independent_graph --dtype bfloat16 --rank 64 --block-size 8 \
  --temperature 1 --sampling-top-k 50 --top-p .95 --noise-low 1 --noise-high 64256 \
  --train-requests 64 --train-tokens 128 --embedding-init base_projected \
  --interval 8 --lr .003 --confidence-lr .0001 --head-execution cuda_graph
```

输出路径采用独占创建；新训练使用新的检查点文件名。
保留相同路径参数，将子命令 `train` 改为 `benchmark`，并指定
`--prompts 17 --prompt-length 256 --tokens 256 --repeats 2`。
添加 `--online --interval 16 --lr .0003 --confidence-lr .00001`，
在同一进程中交错测量 AR、原并行草稿、固定小头和在线小头四路。
添加 `--scheduled --threshold .03` 可单独加入采样前截断诊断。
每个重复从相同检查点开始，在线头与 Adam 跨请求持续更新；其他方法每次使用固定参数。
stdout 输出配置、文件 SHA、实现指纹、总 TPS、图准备、训练开销和逐深度接受计数。
小头与采样合并为 GPU 图，准备时间计入对应方法；`--head-execution eager` 提供普通执行对照。
将子命令换成 `audit` 可测量相同前缀上的原始／修正 TV，并比较小头执行耗时。

`base_projected` 使用冻结基座嵌入的归一化高斯投影初始化转移向量，输出投影为零；
`random` 提供相同参数量的随机初始化对照。
`--audit-reference` 和 `--compare-head` 接受同一官方权重及配置下的其他小头检查点，
分别做共同前缀审计和交错吞吐比较。加载时同时校验权重、训练划分和完整推理配置。
训练与测量结束核验 `frozen_base_unchanged`、`frozen_adapter_unchanged`。
报告通过 `evaluation_sha256` 和 `config.evaluation_split` 标识使用的划分。

共享 HF SDPA 骨干对照使用 `--backbone hf_sdpa`，并在 WSL 命令前设置
`PYTHONPATH=/home/singm/online-speculation-work/oracle-transformers515`。
该隔离目录提供锁定的 Transformers 5.15.0，加载器核验基座 Python 源码、配置及权重 SHA。
两路 `benchmark_offline.py --sampler linear` 与四路 `prefix_relay.py` 均支持这个后端。
每个后端使用自己的 AR 基线，头检查点绑定训练时的后端及精度。

## 9. 稀疏概率混合

当前在线入口加载固定公开基座与适配器，更新对象为每个草稿深度的 5 个系数。
温度表为 `0.5 0.75 1 1.25 1.5`，默认每 8 个反馈块更新一次，初始份额全部分配给原表。

```bash
python scripts/overlap_mix.py benchmark \
  --base /home/singm/online-speculation-work/models/K2-Horizon-0.9B \
  --adapter /home/singm/online-speculation-work/models/K2-Horizon-0.9B-Uno \
  --reference-sha256 5a499229d19ef4a69eb0b21884819d1b67cd983ba02b7ee2031ba8567dedfe4e \
  --train-data /home/singm/online-speculation-work/data/blockspec_ot3_medium/train.jsonl \
  --data /home/singm/online-speculation-work/data/blockspec_ot3_small/validation.jsonl \
  --prompts 17 --prompt-length 256 --tokens 256 --repeats 2 --block-size 8 \
  --sampling-execution cuda_graph --compare-eager
```

默认采样为温度 1、top-k=50、top-p=0.95；骨干使用 BF16 分组注意力与固定形状 GPU 图。
`--sampling-execution cuda_graph` 让 AR、静态与在线共用概率变换和整块校正的 GPU 图入口。
`--compare-eager` 在同一轮增加原概率处理方式的 AR／静态对照，各版本分别计入自己的图准备成本。
`--sampling-execution eager` 使用原概率入口。`retention_pass` 按在线／静态配对区间下界是否达到 0.97 判定。
四路共享骨干执行器和采样设置。每个重复开始时创建新的混合系数，在线状态在该重复的请求流中持续更新。
逐请求交替执行方法顺序，每组方法使用相同请求种子；随机生成的接受与残差抽样会消耗各自的随机序列。
预热数据来自训练划分，正式划分与其按问题分组检查互斥。

将 `benchmark` 改为 `audit` 可在相同实际前缀上比较原表、各温度表与目标的 TV。
`--audit-online` 同时记录在线更新前的 TV；审计运行单列，吞吐测量使用关闭观测统计的入口。
输出包括权重指纹、配置、数据与实现 SHA、各路 TPS、每轮输出、更新时间和配对问题簇 bootstrap 区间。
`identity` 经过相同概率处理但保持原表，用于测量新增处理开销；`online` 包含系数更新。
加入 `--learn-requests 64 --learn-tokens 128 --audit-learned`，在独立训练请求上学习系数，
随后增加 `learned`（学习后冻结）与 `continued`（同一起点持续更新）两路验证。
`learning` 单列预学习的生成量、完整耗时与系数更新计时；验证 TPS 使用验证请求自身的耗时。
两路从同一份参数、累计梯度和更新步数恢复，`vs_learned` 报告继续学习相对冻结推理的净吞吐比。
`learned_audit` 在相同候选前缀上比较学习表与原表的 TV，审计运行独立于吞吐计时。
默认四路中的 `online` 仍从初始系数开始，`retention_pass` 对应这一冷启动对照。
`--method continuation` 选择历史续句的条件混合：每个请求重建提示／输出后缀索引，
系数按匹配长度和深度分组，初值为零。此选项沿用同一套六路评测、状态恢复与完整计时。
GPU 图以并行前缀扫描实现条件分支，历史匹配为空的轮次直接使用原草稿概率处理入口。
文件保存在现有源码目录，实验结果写入 stdout。
