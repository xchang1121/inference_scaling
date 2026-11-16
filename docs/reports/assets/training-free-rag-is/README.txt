仅对已有本地记录重新汇总和作图，没有模型运行。
F1及区间直接来自原汇总；时延从全部原生measurement逐条读取，32次/组，核验均值与原汇总一致。
搜索计时包含materialization，拆分时先减去它，避免重复累计。检索段=materialization+static_retrieval。
其他开销=总时延-search-answer-static_retrieval；全部记录非负，各段加和与总时延相等。
搜索段仍含模型prefill/decode、SC统计、候选选择与编排，现有日志不支持将这些单独拆成互斥的硬件计时。
图为实际管线耗时分解，并非组件消融的因果增量。动态扩展的Base来自前一轮，仅作跨轮参照。
比值为均值之比，不是逐题比值的几何平均。加载、索引初始化和预热不计入单次时延。

F1和区间来源：仓库 experiments/online-injection-full-is-metrics-20261113/reports/summary.json 及 experiments/online-injection-smart-is-20261114/reports/summary.json。
chart-data.json.sources 记录仓库内原生归档路径、归档 SHA256 和其中 formal.jsonl 的未压缩 SHA256；不依赖原工作区绝对路径。
render_charts.py 从已验证的 chart-data.json 重绘，默认使用 Windows 微软雅黑；其他平台用 --font 指定中文字体。--output 可指定输出目录。
原绘图环境 matplotlib 3.11.2，字体或环境不同不保证文件逐字节相同。
