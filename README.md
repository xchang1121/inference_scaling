# 分块起草、验证解码与在线续训

从已有自回归基座出发，独立实现条件低秩适配器、离线块去噪蒸馏、精确验证解码，以及利用推理反馈继续训练
**同一组适配器**。算法来源以 [Uno 原论文](https://arxiv.org/abs/2609.04010) 为起点，模型、训练器和推理流程实现在 `src/blockspec`。
通用张量运算使用 PyTorch，基座权重从本地 safetensors 接入。

当前管线包含线性验证、目标路径树、全适配器／末层在线续训和 GPU 图执行。
验证覆盖概率推导、梯度、KV 缓存、完整基座外部数值对照，以及 RTX 3090 上的三路 TPS 评测。

## 先读哪一份

- [算法主报告](docs/ALGORITHM.md)：从概率和矩阵运算基础开始，推导噪声、蒸馏、拒绝采样、缓存、在线更新和收益条件。
- [运行说明](docs/RUNNING.md)：本机 WSL 环境、小模型闭环、本地数据训练与真实权重检查。

实现变化直接更新算法报告和运行说明的相应章节。

## 当前结构

```text
src/blockspec/
  model.py           独立因果 Transformer、条件低秩层、KV、后段特征重放
  execution.py       固定形状 GPU 图、函数式 KV 快照、原地参数更新
  diffusion.py       离散噪声与预测—校正数学核
  distillation.py    干净／带噪配对布局与蒸馏损失
  training.py        离线课程与 KL 热身
  sampling.py        概率变换、拒绝接受与残差抽样
  decoding.py        AR 基线、线性验证解码
  tree.py            前缀预算树、树注意力、精确目标路径遍历
  online.py          原适配器全量／末层子集续训、反向、更新与版本
  checkpoint.py     自有格式、基座指纹、本地权重桥接
  data.py            独立序列数据合同
  corpus.py          有界公开数据、问题分组划分及来源校验
  validation.py      固定窗口与噪声的独立验证
  benchmark.py       配对请求流、输出对照与包含续训成本的 TPS
  diagnostics.py     等价执行布局的逐层数值审计
  tokenizer.py       只读本地 tokenizer.json
  cli.py             数据、训练、三路评测与小模型闭环入口
tests/               概率、梯度、缓存、端到端及外部数值参照
scripts/check_local_model.py  本地基座权重的集成检查
scripts/audit_sampler_reference.py  可选的固定公开实现 CPU 契约参照
scripts/audit_hf_reference.py  固定基座的完整外部数值参照
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
