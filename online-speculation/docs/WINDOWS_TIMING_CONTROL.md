# Windows 吞吐测量的进程 QoS 控制

R3C 的第二轮 pilot 观察到 GPU memory clock 在 5001/9501 MHz 之间变化，
同模型、同静态方法的 TPS 会在约 15 与 40 之间切换。这样的跨状态配对不能直接用于
宣称算法 TPS 收益；TPF 与 greedy token 检验仍有工程价值。

只读检查显示实验进程的 power-throttling control_mask=0、state_mask=0，
表示未显式控制，不证明 Windows 没有自动降低进程 QoS。
根因目前只是候选假设，不能把频率变化全部归因于 EcoQoS。

下一次 pilot 可使用 --windows-disable-ecoqos：
通过 SetProcessInformation(ProcessPowerThrottling)，对**本次 benchmark 进程**
显式关闭 EXECUTION_SPEED throttling（control bit=1、state bit=0）。
不修改系统电源计划、GPU 时钟、驱动设置、其他进程或永久注册表。
进程结束后不遗留该设置。前后 API 返回值写入实验 host 元数据。

依据：[Microsoft SetProcessInformation documentation](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setprocessinformation)。
官方文档说明未显式设置时系统可以自动推断 QoS，并给出了显式 HighQoS 用法。

实验要求：所有比较方法在同一进程共享相同设置；全方法预热；前后 GPU 状态记录。
若仍有状态切换，不删异常 pairs，而是将整组归为工程诊断，重新设计隔离环境。
只有稳定状态的新评估才承担正向 TPS 结论。此设置不是算法创新，也不能将它带来的
运行时改进计为相对于采用不同 QoS 的 baseline 的“算法加速”。
