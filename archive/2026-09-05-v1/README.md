# 旧研究归档（2026-09-05）

按用户要求清理主研究目录。本目录保存 Stage 3–8、10 的旧设计、试验与完整原始 JSON，
以及当时的项目 README/路线图。归档文件没有改写实验数值；Git 仍可定位每阶段提交。
原始文档中的命令/路径反映当时布局，重跑时把结果路径改到新的输出目录。

保留的关键教训：

| 路线 | 已得到的证据 | 对当前设计的影响 |
| --- | --- | --- |
| fast residual、deferred、stream、mixture | 多数试验未通过 held-out 性能门 | 在线梯度不放在默认单卡 decode 路径 |
| greedy residual | TPF +0.95%，TPS 区间跨 1 | 直接优化真实耗时 |
| exact-repeat verifier replay | HF repeated trajectory E2E 约 4.99× | 可作重复上界，不能代表开放请求 |
| causal within-request replay | 少量 TPF 收益，TPS 未确认 | 新主线不依赖检索命中 |

当前代码仍保留原组件以支持回归和消融；主入口与论文材料转到
[Recycling Uno](../../docs/RECYCLING_UNO_DESIGN_AND_PROOFS.md)。
如需恢复记录，从本目录复制回原位置或通过 Git 恢复即可。
