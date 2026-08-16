# RTX 3090 复现记录

本记录覆盖真实 GPU 上的概率、KV、批处理、MH、IS 和 replay 路径。统计范围为工程检查。

## 环境

| 项目 | 版本 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3090，24 GiB |
| 驱动 | 596.49；最高 CUDA 13.2 |
| PyTorch | 2.13.0+cu130 |
| Transformers | 5.14.1 |
| Python / OS | Python 3.12.5 / Windows 11 |
| 模型 | `Qwen2.5-0.5B-Instruct`，revision `7ae557604adf67be50417f59c2c2f167def9a775` |
| 权重 SHA-256 | `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe` |

PyTorch wheel 提供 CUDA 13.0 运行时。系统 `nvcc` 指向 Toolkit 11.8；该工具链用于 CUDA 扩展编译。

## 命令与产物

```powershell
$env:PYTHONPATH = "src"
python experiments\rtx3090_reproduction.py `
  --model models\Qwen2.5-0.5B-Instruct `
  --dtype float32 `
  --output results\validation\rtx3090_reproduction.json
```

BF16 诊断使用 `--dtype bfloat16`，产物为
`results/validation/rtx3090_backend_bfloat16.json`。

## 后端结果

8 个请求共享 47-token prompt，每个请求生成 24 token。

| 指标 | 结果 |
| --- | ---: |
| 顺序生成 | 2.820 s |
| 单批生成 | 0.352 s |
| 顺序 / 单批墙钟 | 8.02× |
| 单批吞吐 | 545.7 generated tokens/s |
| 共享前缀节省 | 329 prefill token |
| FP32 CUDA 峰值已分配显存 | 2.390 GB |
| 最大连续 batch | 8 |

FP32 生成概率与完整序列重评分的平均绝对差为 `5.33e-6`，最大差为 `1.09e-4`。BF16 对应
`4.93e-2` 和 `1.26`，已分配显存为 1.229 GB，吞吐为 486.0 tokens/s。正式重要性权重实验据此采用
FP32。墙钟数据来自一次 warm run。

请求级随机数流固定；不同 batch 形状的 CUDA 数值差异可能改变 categorical CDF 边界。硬件结果同时
记录 token、共同前缀与最终数值。

## 算法结果

### 后缀 MH

设置：4 条固定长度链，`alpha=2`，总长度 16，block size 8，每个 block 更新 3 次。

| 指标 | 结果 |
| --- | ---: |
| 接受率 | 54.2% |
| 初始平均 base log-probability / token | -1.761 |
| 最终平均 base log-probability / token | -0.971 |
| 差值 | +0.790 |

### 条件 IS

设置：4 个固定算术 prompt，4 个候选，每候选 4 条 rollout，block size 8，二元终局奖励。

| 路径 | 正确数 / 4 | 平均绝对 log 修正 | 平均 completion ESS |
| --- | ---: | ---: | ---: |
| Base | 3 | — | — |
| on-policy 条件 IS | 4 | 2.56e-6 | 1.84 |
| 温度 0.7 off-policy 条件 IS | 4 | 0.223 | 1.73 |

### rollout replay

四个固定 base 候选各有两条温度 0.7 历史 completion；每个候选追加一条 fresh base completion。

| 时点 | evaluation | reserved | design |
| --- | ---: | ---: | ---: |
| 决策前 | 8 | 0 | 0 |
| 决策后 | 0 | 0 | 12 |

决策后的 design 集合由 8 条已消费历史记录和 4 条 fresh completion 组成；四个历史 ESS 均为 2.0。

### 动态候选 IS

候选 proposal 以 1:1 混合 base policy 与温度 0.7 辅助 policy。六次 2-token 决策中，两种分量各生成
24 个候选；每批 8 个候选的外层 ESS 为 2.20–8.00。固定 8-rollout 预算为每个非终止候选分配一条
fresh rollout。

## 适用范围

该记录验证单张 RTX 3090 上的执行路径和计数器。分布性质由有限状态测试覆盖；方法质量与置信区间见
[GSM8K 方法质量与计算量实验](../reports/GSM8K_3090_ALIGNED_RESULTS.md)。
