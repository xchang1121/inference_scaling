# 分块起草、验证解码与在线续训

主线为共享 AR 骨干与历史 KV、并行扩散起草、AR 概率校正。
两条分支采用条件低秩增量／因果噪声块，以及独立注意力投影／双向掩码块，
共同使用初始化、候选验证、前缀提交和缓存管理。
低秩路径采用公开 K2-Horizon 权重；双视图路径独立加载公开 Qwen3-1.7B 权重。
来源版本、架构与权重摘要由 `references/upstream.lock.json` 固定。

条件低秩分支的在线模块复用实际验证反馈，支持适配器后段续训、轻量条件头和稀疏概率混合。
学习状态跨请求保留，参数更新在本轮校正完成后发布。
配对实验分别报告固定起草、学习后冻结与继续在线学习的完整成本。
共享骨干、分支推导与概率校正见[主报告](docs/ALGORITHM.md)，实测对照见[性能记录](docs/RESULTS.md)。

## 文档

- [算法主报告](docs/ALGORITHM.md)：从 NTP 的训练与生成实例开始，推导并行验证与概率校正，比较各类起草结构，并展开共享骨干起草、在线修正和吞吐条件。
- [性能记录](docs/RESULTS.md)：当前有效对照、测量配置、版本指纹与实验结论。
- [运行说明](docs/RUNNING.md)：本机 WSL 环境、小模型闭环、本地数据训练与真实权重检查。

实现变化更新算法报告和运行说明的相应章节，实验数据及结论集中更新性能记录。

## 当前工作范围

共同管线覆盖条件低秩与双向注意力起草，训练入口覆盖基座初始化、全分布蒸馏和断点恢复。
后续工作包括公开权重的推理执行优化，以及双向分支的在线学习：
以公开起草参数为起点，探索低开销概率修正与起草注意力子集续训；AR 基座始终冻结。
配对对照包含原始固定推理、学习后冻结推理和计入更新成本的在线推理，
评价端到端 TPS、接受长度、更新开销与目标分布校正。

## 当前结构

```text
src/blockspec/
  state.py           共同历史 KV、打包存储与提交边界
  parallel/
    backbone.py      独立 Qwen3 双视图骨干
    branches.py      两类起草的输入、位置和参数路径
    generation.py    共同初始化、验证与提交循环
    sampling.py      抽样／验证执行策略
    feedback.py      在线反馈与更新生命周期
    training.py      随机锚点、多块掩码、完整分布 KL
    fitting.py       索引数据、梯度累积、调度与精确断点恢复
    weights.py       严格公开权重映射、训练检查点
  model.py           独立因果 Transformer、条件低秩层、KV、后段特征重放
  attention.py       共享 K/V 的分组短查询、显式 mask 与 softmax
  execution.py       固定形状 GPU 图、函数式 KV 快照、原地参数更新
  hf_execution.py    固定 HF SDPA 骨干、冻结路由、函数式 KV 与末层特征桥接
  replay_execution.py 末层前向／损失／梯度图、反馈梯度累积
  diffusion.py       离散噪声与预测—校正数学核
  distillation.py    干净／带噪配对布局与蒸馏损失
  training.py        离线课程与 KL 热身
  sampling.py        概率变换、候选接受与残差抽样
  decoding.py        既有 AR／线性解码调用的兼容入口
  calibration.py     稀疏温度表、混合概率、反馈梯度与单纯形投影
  sampling_execution.py 概率变换、指数抽样与整块残差校正的 GPU 图
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
scripts/overlap_mix.py   冻结权重的在线混合、共同前缀审计与配对吞吐
scripts/dual_view.py    双视图公开权重数值对齐与配对吞吐
scripts/train_dual_view.py 双向起草训练、恢复与小型合成闭环
scripts/audit_pipeline.py  从指定 Git 版本对照重构前后的输出、计数与吞吐
docs/ALGORITHM.md     持续更新的算法主报告
docs/RESULTS.md       当前有效实验与复现记录
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

仓库保存实现、测试、配置、算法报告、精简性能记录和运行说明；模型权重、实验数据与原始测量文件保存在仓库外。
设计取舍直接写入相关章节，代码演进通过 Git 历史追溯。每个阶段验证后提交本目录并推送。
