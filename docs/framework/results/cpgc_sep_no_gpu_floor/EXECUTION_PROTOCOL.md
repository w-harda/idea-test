# C-PGC / SEP 无固定GPU空闲显存门槛恢复协议

## 固定边界
旧v3源码4c8eab66860d40b74a6f55116b510f53dc9d38de和配置a52e31c7ed3e455e14d1b5c17a4262e3c212f4d784681d39685a19615f45f903完全保留。
不编辑旧工作树、控制目录、FAIL/Claim或成功产物。仅定向停止已核实无GPU子进程的等待监督器909831。
官方算法、参数、Victim权重、原生精度、完整Top128/全Gallery排名及缓存优化不变。

## 当前运行政策
GPU空闲显存、GPU利用率、已有进程及实测峰值仅供诊断，不作任何统一最低显存否决。
不把门槛改成0或改成另一个数值；活动配置直接移除minimum_free_mib，运行时与审计中删除相应比较。
RAM仍96GiB、磁盘100GiB、inode80000。只有这些主机资源不足可每5分钟等待，累计最多12小时。
nvidia-smi状态读取故障仍作为真实基础设施错误停止，不伪造资源快照。
不会向Ollama或其他用户进程发送信号。真实CUDA OOM/算法/SHA/数值/审计错误首错停止，不重试、不调精度或攻击参数。

## 来源兼容
successful_step_reuse逐步绑定旧配置FileRef、真实根目录和completed Manifest SHA。
数据、权重、Tokenizer、全部evaluation数值和算法协议逐项相等；所有非资源策略Python producer必须与父版本逐字节相同。
仅runtime.py、audit.py、source_reuse.py及新step_reuse.py属于资源/只读版本接线允许变化。
旧Artifact的started/protocol/audit仍按其真实父配置SHA核验；不将其重标为v4。
已存在FAIL、pending、孤立Claim或部分Attempt必须拒绝，不覆盖或重复执行。

6组CPGC Full及CPU审计只读引用。SEP前4组Source/Target/Audit及ICFG/IRRA Source只读复用。
首个新GPU步骤为SEP ICFG IRRA Target Smoke，然后独立CPU审计。
随后SEP ICFG RaSa Source/Target Smoke及审计；六份Smoke+CPU审计都有效后串行执行六组SEP Full与各审计。
Smoke就绪检查接受有严格来源绑定的父版本成功回执，但仍要求真实正数GPU峰值记录；不再比较峰值+余量与固定显存门槛。

## 新身份
工作树：/home/lzf/ldx/projects/idea-test-tbps-cpgc-sep-no-gpu-floor-20261101
配置：configs/framework/cpgc_sep_tbps.no_gpu_floor.server.json
输出：/home/lzf/ldx/outputs/tbps-framework-cpgc-sep-no-gpu-floor-v4
控制：/home/lzf/ldx/tmp/cpgc_sep_no_gpu_floor_v4_20261101
新旧配置SHA、Source和Target真实生产版本分开记录。

## 后台启动
~~~bash
mkdir -p /home/lzf/ldx/tmp/cpgc_sep_no_gpu_floor_v4_20261101
nohup setsid bash /home/lzf/ldx/projects/idea-test-tbps-cpgc-sep-no-gpu-floor-20261101/operations/run_cpgc_sep_no_gpu_floor.sh >> /home/lzf/ldx/tmp/cpgc_sep_no_gpu_floor_v4_20261101/launcher.log 2>&1 < /dev/null &
~~~
持有原全局恢复锁和新process.lock；started永久一次性。后台须PPID1、无TTY并实际进入原生GPU步骤。
旧版本入口、配置及文档保留原门槛作为历史证据，当前唯一恢复入口是本文件上述新脚本；不再调用旧m.resources/m.wait_ready。

## 查询
~~~bash
cat /home/lzf/ldx/tmp/cpgc_sep_no_gpu_floor_v4_20261101/status.json
tail -n 40 /home/lzf/ldx/tmp/cpgc_sep_no_gpu_floor_v4_20261101/status.log
~~~
完成12组和所有独立CPU审计后才发布最终全量成功。Git报告发布失败不能触发任何重跑。
