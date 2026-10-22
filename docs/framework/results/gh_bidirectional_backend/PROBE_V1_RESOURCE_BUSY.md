# probe_v1启动前资源阻塞记录

执行基准：`a46c6980c4abf58df02ca1ee199da799892dc81e`，分支`codex/gh-bidirectional-backend`。
本次仅获授权执行一次固定两条Train数值探针；A1/B两轮Smoke及训练未获授权。

## 结论

**RESOURCE_BUSY；GPU探针未启动，probe_v1身份未消费。**

| 检查 | 真实结果 |
| --- | --- |
| Git/源码/配置 | 冻结Commit正确、执行前工作树干净；CPU来源SHA通过 |
| CPU Preflight | GH_NATIVE_CPU_PREFLIGHT_PASS，退出码0 |
| CPU启动时序 | cpu_sources_verified→cpu_isolation_exited，无CUDA初始化 |
| 父恢复 | G60/H10模型及完整Adam、CPU RNG恢复一致 |
| GPU空闲显存 | 6385MiB；本次硬门槛10240MiB，FAIL |
| 共享计算进程 | PID2595393，/home/lzf/jmk/envs/foa_v5/bin/python，17656MiB；未干预 |
| GPU利用率 | 63%，仅记录；此次阻塞原因是显存不足 |
| 可用RAM | 121639403520字节，PASS |
| 可用磁盘 | 6179822403584字节，PASS |
| 可用inode | 240230656，PASS |
| 历史保护 | 151378项metadata，0不匹配；关键来源SHA由Preflight核验 |
| probe_v1 | 不存在；无Claim、pending、FAIL或完成回执 |

资源检查时间：2026-10-22 22:11:18 +08:00，只检查一次，没有等待或轮询。
未初始化CUDA，GPU模型加载、神经前向、JVP、Optimizer.step、完整图库刷新均0。
没有终止、暂停或修改其他GPU进程，没有降低显存门槛，没有创建其他Attempt。

## 本次没有获得的数值结果

Unicode新语境JVP/FD、NOOP梯度一致性、非NOOP梯度差、GPU allocated/reserved峰值均为NOT_RUN。
CUDA RNG仅在CPU保存原字节，不能据此声明设备恢复PASS。
双Adam已在CPU完整恢复：G所有step484080；H逐参数step20124–20370，完整游标20480 visits/20370更新。
没有执行参数更新。

现有224项CPU测试是前一阶段证据；本轮重新执行CPU Preflight，不冒称重跑224项回归。
由于资源阻塞，尚未满足后续A1/B完整Smoke的真实数值前提。

## 原始证据

目录：`/home/lzf/ldx/tmp/gh_bidirectional_probe_v1_gate_20261022/`。

- cpu_preflight.log及cpu_preflight.exit：真实CPU恢复、导入闭包及来源结果。
- history_before.json：151378项metadata检查结果。
- resources_before.json：即时资源与额外10GiB硬门槛判定。
- command.json：原CPU命令、冻结Commit及额外显存要求。

完整CPU结果、来源标识、日志SHA见[证据索引](PROBE_V1_RESOURCE_BUSY_EVIDENCE.json)。
冻结探针源码、配置和历史实验均未修改；本次仅新增资源阻塞报告并更新项目状态。
