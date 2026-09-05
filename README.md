# 分块起草、验证解码与在线续训

从已有自回归基座出发，独立实现条件低秩适配器、离线块去噪蒸馏、精确验证解码，以及利用推理反馈继续训练
**同一组适配器**。算法来源以 [Uno 原论文](https://arxiv.org/abs/2609.04010) 为起点；本目录不调用作者的模型、训练器或推理引擎完成主流程。
通用张量运算使用 PyTorch，基座权重可从本地 safetensors 接入。

这不是“已经复现全部论文性能”的声明。当前完成独立线性、目标路径树参考管线和基础验证；真实模型蒸馏收敛、
数值边界、推理优化和在线净收益仍需逐级实验。旧包装实现的速度数字不属于当前实现。

## 先读哪一份

- [算法主报告](docs/ALGORITHM.md)：从概率和矩阵运算基础开始，推导噪声、蒸馏、拒绝采样、缓存、在线更新和收益条件。
- [运行说明](docs/RUNNING.md)：本机 WSL 环境、小模型闭环、本地数据训练与真实权重检查。

只维护这一份算法报告和一份运行说明，不另建结果档案。实现变化直接更新相应章节。

## 当前结构

```text
src/blockspec/
  model.py           独立因果 Transformer、条件低秩层、KV
  diffusion.py       离散噪声与预测—校正数学核
  distillation.py    干净／带噪配对布局与蒸馏损失
  training.py        离线课程与 KL 热身
  sampling.py        概率变换、拒绝接受与残差抽样
  decoding.py        真正 AR 基线、线性验证解码
  tree.py            前缀预算树、树注意力、精确目标路径遍历
  online.py          全适配器重放、反向、更新与版本
  checkpoint.py     自有格式、基座指纹、本地权重桥接
  data.py            独立序列数据合同
  corpus.py          有界公开数据、问题分组划分及来源校验
  validation.py      固定窗口与噪声的独立验证
  diagnostics.py     等价执行布局的逐层数值审计
  tokenizer.py       只读本地 tokenizer.json
  cli.py             小模型与真实权重离线训练入口
tests/               概率、梯度、缓存、端到端及外部数值参照
scripts/check_local_model.py  本地真实权重的有限集成检查
docs/ALGORITHM.md     唯一持续维护的算法主报告
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
这是功能检查，不是自然语言加速证据。加 `--checkpoint models/cycle.pt` 可检查实际保存与重新加载；已有文件不会覆盖。

运行主体不依赖 Transformers 的模型实现；`hf` 额外依赖仅供测试数值参照。`text` 提供本地 tokenizer JSON 的解析。

## 仓库约定

只提交当前实现、测试、少量配置和上述两份文档。权重、训练数据、原始结果、日志和 profiler 产物不提交。
失败设计只留简短原因，不留一套弃用脚本和结果。每个可用阶段先验证，再仅提交本目录并推送，
不连带提交父仓库的其他工作。
