# RTX 3090 复现记录

## 实验范围

这是一次小规模真实模型的行为与系统检查，不用于稳定的准确率排序。`tests/` 中的精确有限状态
测试负责检查目标分布与 replay 估计器；本实验检查同一套实现能否在物理 GPU 上运行，并表现出预期的
定性行为。

已核验的 JSON 产物为 `results/rtx3090_reproduction.json`（FP32 算法与系统结果）和
`results/rtx3090_backend_bfloat16.json`（低精度诊断）。

## 环境

- GPU：NVIDIA GeForce RTX 3090，24 GiB；
- 驱动：596.49，`nvidia-smi` 显示最高支持 CUDA 13.2；
- PyTorch：2.13.0+cu130，自带 CUDA 13.0 运行时；
- Transformers：5.14.1；
- 模型：`Qwen2.5-0.5B-Instruct`，revision
  `7ae557604adf67be50417f59c2c2f167def9a775`；
- 模型权重 SHA-256：
  `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`；
- 操作系统：Windows 11；Python 3.12.5。

机器还安装了 CUDA Toolkit 11.8 与 12.6，当前 `PATH` 中的 `nvcc` 指向 11.8。这不会阻碍本实验使用
CUDA：普通 PyTorch 推理使用 wheel 自带的运行时，只有编译 CUDA 扩展时才依赖 `nvcc`。当前驱动
足以运行 cu130 wheel。

## 运行命令

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python experiments\rtx3090_reproduction.py `
  --model models\Qwen2.5-0.5B-Instruct `
  --dtype float32 `
  --output results\rtx3090_reproduction.json
```

BF16 诊断把 `--dtype` 改为 `bfloat16`，并写入第二个 JSON 文件。

## 后端效率

实验对 8 个具有相同 47-token prompt、各生成 24 个 token 的请求，分别逐个运行与组成一个 batch
运行。

| 指标 | 结果 |
| --- | ---: |
| 顺序生成 | 2.820 s |
| 批量生成 | 0.352 s |
| wall-time 加速 | 8.02x |
| 批量吞吐 | 545.7 generated tokens/s |
| 共享 prefix 避免的 prefill token | 329 |
| CUDA 峰值已分配显存 | 2.390 GB |
| 8 个并发调用形成的最大连续 batch | 8 |

连续调度器把 8 个同步单请求调用合并成一个物理模型 batch。共享 prefix 路径只计算一次 47-token
prefill，再复制 KV cache。这里的 8.02x 是“顺序 wall time / 批量 wall time”，表示硬件利用率提升；
它不表示算法所需 FLOPs 降低。

FP32 下，生成时 token log-probability 与随后完整序列重评分的平均差为 `5.33e-6`，最大差为
`1.09e-4`。BF16 下对应数值为 `4.93e-2` 与 `1.26`；虽然已分配显存降至 1.229 GB，但单次吞吐也
降至 486.0 tokens/s。因此，此环境下重要性权重默认使用 FP32。耗时比较只包含一次 warm run，不能
当作置信区间。

即使使用 FP32，顺序 batch 与合并 batch 的文本也不保证逐 bit 相同。batch 形状变化可能让某个
logit 改变几个浮点 ulp；固定均匀随机数可能因此落到 categorical CDF 边界另一侧，之后自回归上下文
就会分叉。请求级随机数流与数学 policy 不依赖调度，但不假设不同 batch 形状下 GPU 逐 bit 一致。

## 算法行为

### 后缀重采样 Metropolis--Hastings

四条固定长度链使用 `alpha=2`、总长度 16、block size 8，每个 block 进行 3 次 MH 更新。合并接受率
为 54.2%。直接采样的平均 base-model log-probability 为每 token `-1.761`，最终链状态为 `-0.971`，
每 token 提升 `0.790`，符合分布锐化的预期。四条链不足以估计分布级准确率；该性质由枚举测试检查。

### 条件重要性采样

检查使用四个固定算术 prompt、四个候选、每候选四条 rollout、block size 8 和二元最终答案奖励。
直接采样答对 3/4。固定 seed 下，on-policy rollout 与温度 0.7、带精确 completion likelihood ratio
的 off-policy rollout 都答对 4/4。

on-policy 平均绝对 log 修正为 `2.56e-6`，只反映 FP32 重算误差。off-policy 修正为非平凡的
`0.223`；最大为 4 的 completion ESS 平均为 `1.73`，on-policy 则为 `1.84`。该 seed 下相同输出说明
修正可以在 rollout policy 改变时保持决策，但四条 prompt 不是统计等价性证明。

### 离策略 rollout 回放

受控决策先为四个可复现的 base 候选分别生成两条温度 0.7 的历史 completion。replay 决策对每个
候选使用全部两条历史 completion，只补充一条新的 base completion；四个历史 ESS 都为 2.0。

决策前 evaluation pool 含 8 条记录。决策后 evaluation record 与 reserved record 均为 0，design
record 为 12：8 条已消费历史加 4 条 fresh completion。这验证了真实模型上的“设计阶段只读元数据”、
evaluation 单次使用与 fresh 数据生命周期。

### 动态候选重要性采样

defensive proposal 以相同比例混合 base candidate policy 与温度 0.7 的辅助 policy。在六次 2-token
决策中，两种分量各生成 24 个候选。每批 8 个候选的外层权重 ESS 范围为 2.20 至 8.0。实验正确
生成 `27 * 14 = 378`；固定 8-rollout 预算下，每个非终止候选都获得一条 fresh rollout。

## 解释与限制

这些测量说明 CUDA 执行、精确 policy 元数据、off-policy 修正、replay 消费、候选外层修正、KV
复用和连续批处理能在参考机器上共同运行。它们没有估计 benchmark 级置信区间、长链混合速度或
十亿参数模型的 scaling 行为。后续结论需要更多 prompt、多个 seed 与更大的模型，同时继续固定并
记录模型、奖励和 behavior-policy 元数据。
