# CUDA RNG真实验证PASS与A1/B两轮Smoke后台启动

## 当前结论

独立CUDA RNG验证真实PASS，完成原G60/H10的CPU/CUDA/sampler随机状态恢复、双向阶段隔离、序列重放和状态持久化读回。
随后已自动后台启动唯一smoke_v2，并真实进入A1-r1完整Train图库刷新。**当前是运行中，不是四轮及CPU审计最终PASS。**

部署核验快照：2026-10-23T15:11:02.319318+08:00。
最新记录：GH_BANK_PROGRESS A1 1 1040 2048。
G状态已从父5d48caab6a891a1afd3aaaa6f3594eb5f860ccd7b6090708d289224e6c819e47变为4b125683f6d4a1e07a6bf9cd4215a535b4c6909e6f5697a790aef48a1cb7528c，并有逐图不可变产物，证明已执行真实G更新，非仅创建后台PID。

## 独立资源修复与CPU测试

只修改operations和新增测试/文档，不修改冻结src/tbps/alternating、生产配置或原probe_v2回执。
RNG最低空闲显存独立设为2048MiB；原Smoke保持10240MiB、RAM32GiB、磁盘25GiB、inode80000。
新19项CPU测试全部PASS，2.50秒：覆盖原5项RNG组合流程、门槛不互相污染、当前probe已完成/Smoke未开始生命周期、既有Smoke任何状态拒绝、连续两次检查、资源回落不消费、独占锁、崩溃保护启动标记及失败后不重试。
旧准备态测试及其失败日志原样保留，不删除probe满足旧断言，也不声称所有历史阶段测试均在当前状态通过。

## 真实CUDA RNG结果

| 验证 | 结果 |
| --- | --- |
| CUDA_RNG_PASS | PASS |
| G/H父CUDA状态数/可见设备数 | 1/1/1，一致 |
| Python、NumPy、Torch CPU/CUDA、H sampler恢复/重放 | exact |
| G→H→G连续序列 | exact |
| H→G→H连续序列 | exact |
| 两阶段状态隔离 | PASS |
| 捕获→持久化→读回→再次恢复 | exact |
| G/H参数与双Adam状态 | 不变 |
| 调用者RNG最终恢复 | exact |
| CUDA随机Tensor调用 | 18，每次8个元素 |
| 神经前向/JVP/Optimizer.step | 0/0/0 |
| GPU peak allocated/reserved | 512/2097152 bytes |
| 进程Peak RSS | 822509568 bytes |
| CUDA初始化后验证及保护核验耗时 | 3.4063780307769775秒 |

CUDA验证调用原restore_rng/capture_rng以及NativeBackend.enter_phase_rng/leave_phase_rng，未重实现生产RNG数学。
完整父模型与Adam仅在CPU恢复，不加载IRRA/reader或调用神经前向。RNG报告内CPU_preflight.CUDA_RNG_restored=false是初始化CUDA之前的CPU快照，不代表随后真实CUDA验证失败。
独立CUDA证据的checks及CUDA_RNG_PASS记录随后完成的GPU验证，二者阶段不同。

独立回执：
- 产物目录：/home/lzf/ldx/outputs/tbps-framework-gh-bidirectional-backend/cuda_rng_v1/。
- Manifest SHA256：62b2748566c3c1eec61a286f48fa7846f5529e82877b5239d34a5dd3947e9905。
- 完整原始轻量结果：[CUDA_RNG_RESULT.json](CUDA_RNG_RESULT.json)。
- 状态持久化内容SHA：f23f5ea98ea833b8d1d7033d1e6b22015852a5e10aa00c099a451a1ab20553aa。
- 脚本SHA：4be1ff7d0036a94412d8a8c60a8f71b9c41102f9fcaf73adc33edfcd4c486550。

## 后台进程与真实Smoke

监督器PID/PGID/SID均3712068，PPID=1，TTY=false；单实例flock真实占用。
Smoke子进程PID3712463，与监督器同PGID/SID，已脱离SSH。
监督器没有libtorch/libcuda/libcudart映射，只进行资源、来源、进程和日志管理。

15:05:48启动监督器；首次GPU空闲24051MiB，满足门槛，按本轮明确允许的首检立即路径进入完整CPU Preflight。
15:05:58 READY；15:05:59创建唯一持久化启动记录并启动原native_cli。
原生smoke_v2 Claim已发布，身份已消费，不能自动或手动重复启动同一身份。
快照GPU空闲22413MiB，Smoke进程nvidia-smi占用1630MiB，利用率4%；该占用不是最终peak allocated/reserved。

原生固定协议：
- A1-r1→A1-r2→B-r1→B-r2。
- 同一G60/H10完整父状态、双Adam和阶段RNG，固定8条Train样本。
- A1用原文本条件训练G；B重算H攻击文本新语境条件/JVP；两组均完整刷新最新G的2048 Train图库后使用原H SCST。
- G更新108、H访问32/合法更新≤28、四次完整图库共8192图、B新JVP≤38。
- 原K/Loss/Reward、G(I)/H(T)、FP32与8/255不变，不做Validation/Test/Full。

只有四轮实际生成完成才调用原native_cli --action audit-smoke；随后额外核对历史保护和汇总既有数值。
最终分别记录smoke_control_flow、CPU_audit、feedback_effects.bidirectional_update_evidence_verified；运行成功不等于检索性能提升。
此刻这些最终四轮数值及审计结果仍PENDING，绝不预填PASS。

## 等待与单次消费

资源不够时每180秒检查，连续两次满足才进入完整门禁；初次快照已经合格时按授权直接检查启动。
WATCHING期间无smoke_started.json且不消费Smoke，资源回落重新等待。
通过来源/CPU/RNG/最终资源检查后才O_EXCL+fsync写smoke_started.json，之后失败/中断永不重试。
SIGTERM仅可取消尚未消费时的等待；运行后不会通过“取消等待”终止实验。
本次首次检查已合格，已经离开WATCHING，没有下一次等待检查时间；运行状态每10秒从进程和原生日志更新。

## 来源与保护

- 独立工具执行Commit：2f86593ca8226c238f972642cc71752322b4a5d8。
- 冻结生产HEAD：aafc7715ddafa78f9a30410b6a6ed8852e81780a，原数值执行代码：a7597f9c964dda0c8c48973e4678f0b9d3fa095a。
- 配置SHA：77ae41af56189ece991ee7cf21fd0ea0f3be5645202bd1cb7cf7e0b2ae735649。
- source_key：06b5c6a15061e635e49a9e13deb14022956fdbc272ad0ed378676e57ca5f826b。
- 原probe_v2 Manifest：d6e0b960e20a1ba5128f9e0b4edf497e7ea4b098fbf5cd3125799d2ad1136764，未覆盖/重跑。
- 140项producer、496项运行来源、固定8条与G60/H10完整双状态再次核验。
- 151378项历史metadata、28份旧FAIL/Watcher SHA/metadata、父Checkpoint和probe_v2原文件保持；启动后再次检查PASS。
- 源码索引、回执SHA及部署快照：[MONITOR_DEPLOYMENT.json](MONITOR_DEPLOYMENT.json)。
- 辅助monitor_deployment.json中source_key标签误填配置路径；该字段未用于门禁。原文件保留，控制目录deployment_identity.json记录正确身份；实际supervisor_identity、smoke_started和原生Claim均绑定正确source_key。

## 路径与查看

- 后台控制：/home/lzf/ldx/tmp/gh_smoke_wait_v1_20261023/。
- RNG日志：/home/lzf/ldx/tmp/gh_rng_smoke_monitor_v1_20261023/。
- Smoke产物：/home/lzf/ldx/outputs/tbps-framework-gh-bidirectional-backend/smoke_v2/。
- 状态status.json/phase，控制status.log，原生smoke.log，后续cpu_audit.log/final_evidence.json。

~~~bash
cat /home/lzf/ldx/tmp/gh_smoke_wait_v1_20261023/status.json
tail -f /home/lzf/ldx/tmp/gh_smoke_wait_v1_20261023/status.log /home/lzf/ldx/tmp/gh_smoke_wait_v1_20261023/smoke.log
~~~

启动核验已完成，Codex不持续等待四轮。后台正常情况下自动结束于Smoke及CPU审计；任何错误保留现场停止。
