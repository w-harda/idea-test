# C-PGC / SEP 取消固定显存门槛与隔离恢复

本轮仅改变资源准入和显式旧成功产物复用，不改变模型/攻击/数值定义。
实时确认旧监督器909831停在SEP ICFG IRRA Target之前，没有GPU子进程、没有该Target Attempt。
Ollama PID2749989占用19700MiB，旧诊断空闲4343MiB，其他主机资源达标；仅停止旧等待监督器，不干预Ollama。

runtime、活动调度器、CLI所用资源函数和Smoke峰值审计已删除固定显存否决；历史冻结版本保持不变。
独立profiling入口的固定20GiB要求也移除，但本轮不运行新的profiling实验。

历史6组CPGC Full+CPU审计、27个v3已完成Source/Target/Audit绑定通过只读Manifest/SHA/CSV-summary核验。
Smoke各单元额外校验payload SHA；历史Full复核已有完成/审计证据，不重新计算排名、不重新运行原审计。
旧成功产物在真实父配置身份下引用，不复制或重写Manifest。

新GPU验证将直接使用首个未完成的SEP ICFG IRRA Target Smoke及其CPU审计，不重复已有Source或其他完成Smoke。
正式Full没有因CPU PASS而被提前视为GPU通过。最终GPU结果与后台PID由后续独立启动证据登记。

## CPU验证完成

资源/算法/只读桥CPU回归56项PASS，缓存与调度回归19项PASS，共75个不同用例。
调整调度用例预期时出现一次括号语法错误，仅测试收集失败，原日志保留；修正后19项完整通过，不影响或重跑GPU。
8个数值/攻击模块target、victims、cache、sep_source、sep_bridge、official、dataset、verified_store与v3逐字节相同。
真实即时资源检查在空闲显存低于旧门槛时返回READY；所有主机门禁仍有效。

## 真实GPU与部署验收

2026-11-01T00:35:38.269306+08:00：新ICFG/IRRA Target Smoke真实PASS，独立CPU审计PASS，CUDA初始化/神经前向审计均0。
目标Source复用旧v3已完成产物，没有重跑。
本次真实GPU Target峰值allocated 434.59MiB、reserved 446.00MiB，耗时14.028秒。
这些是Smoke资源数据，不是正式Full攻击结论。

监督器PID/PGID/SID=2960820，PPID1，无TTY，stdin=/dev/null，stdout写本地launcher.log，两个flock互斥锁实际持有。
当前6/12完成，已自动进入sep_smoke_icfg_rasa_source，后续六份Smoke/审计就绪后串行执行六组SEP Full与各审计。
真实异常/OOM立即保存证据并停止，不重试失败Attempt。

新策略在4343MiB空闲、Ollama占用19700MiB时真实CPU资源准入已经返回READY。
GPU资源在代码/CPU来源检查期间自行回升；当前快照21047, 24564, 21（free MiB,total MiB,utilization %）。
没有人为释放、卸载、重启、暂停Ollama或其他进程，也没有因GPU显存等待。
旧v3控制文件的全部SHA与停止前记录一致。所有旧成功产物均引用原配置/Manifest，不重新标记为新SHA。

执行源码Commit：14f247fb5557d1d8eb40d71a470a4aecbe801655。
新配置SHA：d3a5bf917d2d4d28a19dee1b14f6c61d546670b039efadba727db7f6d5f347ad。
启动证据与本报告在另一文档工作树提交，不修改正在运行的源码树。

当前资源政策对本活动C-PGC/SEP执行链已移除所有固定显存门槛。历史冻结代码、配置和测试快照保留原值用于溯源，不是活动入口。
