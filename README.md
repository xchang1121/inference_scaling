# 分块起草、验证解码与在线续训

从已有自回归基座出发，独立实现条件低秩适配器、离线块去噪蒸馏、精确验证解码，以及利用推理反馈继续训练
**同一组适配器**。算法来源以 [Uno 原论文](https://arxiv.org/abs/2609.04010) 为起点，模型、训练器和推理流程实现在 `src/blockspec`。
通用张量运算使用 PyTorch，基座权重从本地 safetensors 接入。

当前管线包含线性验证、目标路径树、全适配器／末层在线续训和 GPU 图执行。
在线重放按监督位置计算词表投影，CUDA 上使用融合参数更新；评测沿请求流跟踪接受长度、累计耗时和参数版本。
末层训练可将前向、损失和梯度录制为 GPU 图，准备成本计入在线方法的完整吞吐。
分组短查询注意力直接复用共享 K/V，前向与梯度的重新排布同用于 AR、起草、验证和在线训练。
反馈按下一次更新所需的窗口采集，逐轮采集模式提供训练样本与参数更新对照。
覆盖触发策略根据实际验证路径安排训练：完整覆盖窗口保留参数与 Adam，含覆盖缺口的窗口使用全部有效软目标更新。
验证覆盖概率推导、梯度、KV 缓存、完整基座外部数值对照，以及 RTX 3090 上的三路 TPS 评测。
离线两路入口接入自训检查点和固定 SHA 的公开适配器，逐请求交错测量 AR 与静态起草，核算各自图准备成本。
两路与三路评测统一接收温度和目标 top-k/top-p，分别提供贪心逐项比较与随机采样观测，报告 TPS 和每次解码前向产出。
推理图支持 FP32／BF16，在线适配器保留 FP32 主权重；检查点验证和执行精度转换分开进行。
固定官方引擎的可选外部参照在本机 1B／BF16／FA2、温度 1 的线性负载得到 1.145× 吞吐比，配置与口径见主报告 13.5。

## 先读哪一份

- [算法主报告](docs/ALGORITHM.md)：从概率和矩阵运算基础开始，推导噪声、蒸馏、拒绝采样、缓存、在线更新和收益条件。
- [运行说明](docs/RUNNING.md)：本机 WSL 环境、小模型闭环、本地数据训练与真实权重检查。

实现变化直接更新算法报告和运行说明的相应章节。

## 当前结构

```text
src/blockspec/
  model.py           独立因果 Transformer、条件低秩层、KV、后段特征重放
  attention.py       共享 K/V 的分组短查询、显式 mask 与 softmax
  execution.py       固定形状 GPU 图、函数式 KV 快照、原地参数更新
  replay_execution.py 末层前向／损失／梯度图、反馈梯度累积
  diffusion.py       离散噪声与预测—校正数学核
  distillation.py    干净／带噪配对布局与蒸馏损失
  training.py        离线课程与 KL 热身
  sampling.py        概率变换、候选接受与残差抽样
  decoding.py        AR 基线、线性验证解码
  tree.py            前缀预算树、树注意力、精确目标路径遍历
  online.py          原适配器全量／末层子集续训、反向、更新与版本
  checkpoint.py     自有格式、基座指纹、本地权重桥接
  adapter_io.py      公开 PEFT 适配器的来源、形状、缩放与完整映射校验
  data.py            独立序列数据合同
  corpus.py          有界公开数据、问题分组划分及来源校验
  validation.py      固定窗口与噪声的独立验证
  benchmark.py       配对请求流、累计学习轨迹与包含续训成本的 TPS
  diagnostics.py     等价执行布局的逐层数值审计
  tokenizer.py       本地 tokenizer.json 解析
  cli.py             数据、训练、三路评测与小模型闭环入口
tests/               概率、梯度、缓存、端到端及外部数值参照
scripts/check_local_model.py  本地基座权重的集成检查
scripts/audit_sampler_reference.py  可选的固定公开实现 CPU 契约参照
scripts/audit_hf_reference.py  固定基座的完整外部数值参照
scripts/benchmark_offline.py  自训／公开适配器与 AR 的配对吞吐对照
scripts/benchmark_reference.py  固定官方引擎的外部性能参照
docs/ALGORITHM.md     持续更新的算法主报告
docs/RUNNING.md       运行与测量规范
```

## 最小运行

先安装与你的 GPU 匹配的 PyTorch，再在本目录执行：

```bash
python -m pip install -e '.[dev,text,hf]'
python -m pytest -q
python -m blockspec demo --device cpu
```

`demo` 会训练一个小型周期序列基座，再冻结基座训练适配器，最后检查 AR、固定适配器和在线续训的贪心输出。
加 `--checkpoint models/cycle.pt` 可检查保存与重新加载；检查点使用新文件名。

运行主体使用本项目的 `Decoder`；`hf` 提供数值测试参照，`text` 提供本地 tokenizer JSON 解析。

## 仓库约定

仓库保存实现、测试、配置、算法报告和运行说明；模型权重与实验数据保存在仓库外。
设计取舍直接写入相关章节，代码演进通过 Git 历史追溯。每个阶段验证后提交本目录并推送。
