# 分块起草、验证解码与在线续训

当前实验固定 K2-Horizon-0.9B 基座和官方发布的扩散适配器，训练对象仅为
**PrefixRelay（前缀接力起草）** 的低秩转移与置信度小头。
四路对照为 AR、原并行草稿、固定 PrefixRelay、在线 PrefixRelay；基座与扩散适配器全程冻结。
检查点绑定两份权重、推理后端、精度、温度、top-k/top-p、噪声范围和块长。
训练及评测结束同时验证基座和扩散适配器的执行张量指纹。

算法以 [原论文](https://arxiv.org/abs/2609.04010) 为起点，独立实现逐 token 低秩路由、蒸馏、
采样、验证、KV 管理和在线反馈学习。自有 GPU 图执行器为推理优化主线；
固定源码与权重的 HF SDPA 桥接提供共享骨干参照，两种后端各自进行同条件配对测量。
前期适配器训练与续训模块保留为全管线复刻实现，当前实验采用公开适配器入口。

PrefixRelay 使用上一枚实际候选修正并行 logits，以采样前置信度选择验证前缀，
再从已有验证分布学习新增头。在线头与优化器跨请求持续更新，完整 TPS 计入反馈、更新和准备成本。
转移向量支持基座嵌入投影初始化，转移学习与置信度学习采用独立步长，小头和块内抽样合并为 GPU 图。
条件修正、在线学习与实测结果见[主报告第 6—8 节](docs/ALGORITHM.md#6-条件修正与在线学习)。

## 先读哪一份

- [算法主报告](docs/ALGORITHM.md)：从 NTP 的训练与生成实例开始，推导并行验证与概率校正，比较各类起草结构，并展开共享骨干起草、在线修正和吞吐条件。
- [运行说明](docs/RUNNING.md)：本机 WSL 环境、小模型闭环、本地数据训练与真实权重检查。

实现变化直接更新算法报告和运行说明的相应章节。

## 当前结构

```text
src/blockspec/
  model.py           独立因果 Transformer、条件低秩层、KV、后段特征重放
  attention.py       共享 K/V 的分组短查询、显式 mask 与 softmax
  execution.py       固定形状 GPU 图、函数式 KV 快照、原地参数更新
  hf_execution.py    固定 HF SDPA 骨干、冻结路由、函数式 KV 与末层特征桥接
  replay_execution.py 末层前向／损失／梯度图、反馈梯度累积
  diffusion.py       离散噪声与预测—校正数学核
  distillation.py    干净／带噪配对布局与蒸馏损失
  training.py        离线课程与 KL 热身
  sampling.py        概率变换、候选接受与残差抽样
  decoding.py        AR 基线、线性验证解码
  tree.py            前缀预算树、树注意力、精确目标路径遍历
  online.py          原适配器全量／末层子集续训、反向、更新与版本
  relay.py           PrefixRelay 条件修正、采样前截断、新增头在线更新与检查点
  relay_execution.py 块内小头 GPU 图、指数竞争抽样与单调准入
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
scripts/prefix_relay.py  PrefixRelay 训练、配对吞吐与在线续训评测
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
