# 分块起草、验证解码与在线续训

主线为共享 AR 骨干与历史 KV、并行扩散起草、AR 概率校正。
两条分支采用条件低秩增量／因果噪声块，以及独立注意力投影／双向掩码块，
共同使用初始化、候选验证、前缀提交和缓存管理。
两条路径从运行参数指定的本地权重加载，模型尺寸与张量布局由配置文件读取。

条件低秩分支的在线模块复用实际验证反馈，支持适配器后段续训、轻量条件头和稀疏概率混合。
双向注意力分支已接入逐位置的稀疏概率混合与共同采样执行器，覆盖首个待验证候选。
起草注意力后段支持冻结边界重放、FP32 主权重续训与跨请求状态恢复。
学习状态跨请求保留，参数更新在本轮校正完成后发布。
配对实验分别报告固定起草、学习后冻结与继续在线学习的完整成本。
共享骨干、分支推导与概率校正见[主报告](docs/ALGORITHM.md)，实测对照见[性能记录](docs/RESULTS.md)。

## 文档

- [算法主报告](docs/ALGORITHM.md)：从 NTP 的训练与生成实例开始，推导并行验证与概率校正，比较各类起草结构，并展开共享骨干起草、在线修正和吞吐条件。
- [性能记录](docs/RESULTS.md)：当前有效对照、测量配置与实验结论。
- [运行说明](docs/RUNNING.md)：环境准备、小模型闭环、参数配置与权重检查。

实现变化更新算法报告和运行说明的相应章节，实验数据及结论集中更新性能记录。

## 当前工作范围

共同管线覆盖条件低秩与双向注意力起草，训练入口覆盖基座初始化、全分布蒸馏和断点恢复。
当前推进公开权重的推理执行优化，以及双向分支的在线学习：
以公开起草参数为起点，探索低开销概率修正与起草注意力子集续训；AR 基座始终冻结。
配对对照包含原始固定推理、学习后冻结推理和计入更新成本的在线推理，
评价端到端 TPS、接受长度、更新开销与目标分布校正。

## 当前结构

```text
src/blockspec/
  state.py           共同历史 KV、打包存储与提交边界
  feedback.py        独立于起草结构的教师反馈与冻结边界协议
  parallel/
    backbone.py      独立双视图骨干
    branches.py      两类起草的输入、位置和参数路径
    generation.py    共同初始化、验证与提交循环
    sampling.py      抽样／验证执行策略
    feedback.py      在线反馈与更新生命周期
    training.py      随机锚点、多块掩码、完整分布 KL
    fitting.py       索引数据、梯度累积、调度与精确断点恢复
    online.py        起草注意力后段重放、在线更新与主权重恢复
    audit.py         同前缀分布对照与按问题重采样的比较区间
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
  checkpoint.py     自有格式、基座一致性校验、权重桥接
  adapter_io.py      PEFT 适配器的形状、缩放与完整映射校验
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
scripts/dual_online.py  双向分支的采样执行优化、固定／预学习／持续在线对照
scripts/dual_continue.py 双向起草注意力后段的公开权重续训对照
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

仓库保存实现、测试、通用配置、算法报告、精简性能记录和运行说明。
模型、数据、外部参照和输出位置通过参数传入；个人配置存于已忽略的 `local/` 或仓库外。
实验输出统一经过 `reporting.py`，保留方法设置、数值指标和校验结论。
检查点内部的一致性校验用于恢复训练，随本地训练产物保存。

每次提交前汇报性能与验证情况。提交说明采用“方法变化、对应结果”的格式；
算法推导更新主报告，实测数据更新性能记录，设计取舍写入相关章节。
