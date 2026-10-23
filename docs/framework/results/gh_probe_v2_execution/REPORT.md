# G/H probe_v2执行前资源检查：RESOURCE_BUSY

执行基准：f9701c7ea924f734d83f6cfeeaa38b0b83c30f01。
执行树：/home/lzf/ldx/projects/idea-test-tbps-gh-probe-device-fix-20261023。
派生报告分支：codex/gh-probe-v2-results，不修改冻结执行树。

**本次没有执行GPU探针，没有创建probe_v2，也没有消费其Attempt。**

## 执行前核验

- Git HEAD、干净工作树、配置文件SHA8b6fce14…ea5e443正确。
- 140 producer及496运行来源SHA通过。
- source_key为0df9df75fac0c0615af26acf94e92bb8895255e24c1dbfa4a16445495821801b。
- 真实tbps-image CPU Preflight、完整G60/H10模型/Adam/CPU RNG恢复PASS，退出码0。
- G Adam484080，H逐参数20124–20370；CUDA RNG仅保留原字节，未在GPU恢复。
- G60/H10/IRRA父Checkpoint SHA通过。
- 151378历史metadata及原probe_v1/Watcher28份SHA/stat无变化。
- probe_v2与smoke_v2不存在，无相同探针/旧Watcher进程；未重启旧probe_v1。

## 即时资源

2026-10-23 12:03:11 +08:00，一次检查：

| 项目 | 真实值 | 门禁 |
| --- | --- | --- |
| GPU0 | RTX4090，总24564MiB | 固定GPU0 |
| 空闲显存 | 7505MiB | 需10240MiB，FAIL |
| 利用率 | 0% | 仅记录 |
| 计算进程 | Ollama PID3178899，16538MiB | 未干预 |
| 可用RAM | 117114478592字节 | PASS |
| 可用磁盘 | 6178879373312字节 | PASS |
| 可用inode | 240225066 | PASS |

尽管用户此前的11:23快照有18260MiB空闲，本次必须以12:03即时状态为准。
未降低门槛，未等待或轮询，未启动GPU、Watcher、A1/B、任何更新或评估。

## 数值交付边界

IRRA cuda:0真实加载、GPU Adam/CUDA RNG恢复、新Unicode语境JVP/FD、NOOP梯度差、
非NOOP梯度差及GPU allocated/reserved峰值均NOT_RUN，不能填0冒充数值结果。
本次真实神经前向/JVP/Optimizer.step/完整图库刷新均0，因为原native_cli probe入口尚未调用。
不存在completed回执；不能标记PROBE_PASS或PROBE_FAIL。
尚未满足A1/B两轮Smoke的真实数值前提。

控制及原始CPU/资源日志：/home/lzf/ldx/tmp/gh_probe_v2_execution_20261023/。
probe_v2预留路径：/home/lzf/ldx/outputs/tbps-framework-gh-bidirectional-backend/probe_v2/，仍不存在。
CPU Preflight再次验证隔离完整退出，startup_events只有cpu_sources_verified、cpu_isolation_exited。
完整日志SHA、父来源、CPU恢复及资源证明见[SOURCE_EVIDENCE.json](SOURCE_EVIDENCE.json)。

本次只新增派生报告及项目状态，不修改数值源码、配置、权重或任何已有FAIL。
停止等待下一次指令，不自动重试或创建probe_v3。
