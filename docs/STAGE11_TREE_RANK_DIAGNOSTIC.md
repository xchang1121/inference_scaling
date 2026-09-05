# R3C 第二轮 pilot：完整保留的诊断失败

原始逐运行数据：[stage11_tree_rank_fp32_pilot.json](../results/stage11_tree_rank_fp32_pilot.json)。
这组没有完整结束，不能当作完成的性能评估。

两个问题必须分开：

1. 系统频率在同一 prompt 的配对方法之间切换，静态 TPS 在约 15–40 范围变化。
   因此不得从混合状态的 paired ratio 得出算法速度结论。
2. 后半组一次 fixed-tree 构建触发 rank sub-probability 检查。
   可复现的候选数值原因是 exp(top_logit - logsumexp(all_logits)) 的 FP32 消减误差：
   合成 logits [1000,999,0] 产生概率和 1.0000293，而稳定 softmax 为 1.0。
   原失败未保存导致检查失败的具体 logits，因此还不能把它的根因完全锁定为这一机制。
   实现改用稳定 full-vocabulary softmax 后 gather top-K，
   不采用“删去该 seed”或取消所有合法性检查的办法。

后续组使用新 seeds、修复的概率计算、显式进程 HighQoS。所有方法仍共享相同 target/adapter。
原始失败 payload 保留 completed=false；不补造缺失方法/seed 的输出。

树式方法的初步收益首先来自多候选覆盖，并不自动证明在线 rank 学习有效。
下一组包含 fixed tree、rank-only、budget-only、rank+budget 四种消融。
Uno 官方自身也已有静态 tree 模式；论文级在线收益必须相对于最优静态 tree 进一步证明，
不能只对照线性 Uno 后把全部收益归因于在线学习。
