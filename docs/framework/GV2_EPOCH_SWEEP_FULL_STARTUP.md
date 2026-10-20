# G-v2六节点三模型Full正式启动记录

快照时间：2026-10-20T20:40:31.775398+08:00。状态：**后台RUNNING，不能宣称Full PASS**。

## 来源与启动前核验

实际启动代码：6021f3bfdec181785d1baccd7ce4f80a1428b069；此前Smoke执行代码152498fd258572ad9c43c04b2ce8fba8b9e995b4。
分支codex/gv2-six-epoch-evaluation；启动前工作树干净、远端SHA一致。
配置SHA：66bdd1639b54508af1852f781b3951c637193ffa5069848b4b73145e9d9bf701。
Bash启动器SHA：0d3ea7a2c787e5710f42777dcfc2040803b011597b75b18498595f8368f7ff68。

107项producer、六Checkpoint、原训练来源、全部G架构strict load通过。
唯一smoke_v1和三份CPU审计有效，没有重复Smoke。
Full/控制/launcher日志启动前不存在，无重复Full进程。
即时GPU空闲18431MiB、利用率64%，RAM约113.72GiB、磁盘约5.69TiB、inode240434631；
满足6144MiB/32GiB/100GiB门禁。shared_observe_only保留其他任务PID1402779，不等待、不干预。
74184历史stat、289项SHA及六权重复核保持；本轮不改源码、配置、算法或任何旧产物。

## 实际后台执行

2026-10-20 20:35:12 +08:00启动；nohup+setsid。
监督器PID/PGID/SID均1545336，实测PPID1、TTY=?，父SSH会话退出后存活。
CPU预检GV2_SWEEP_CPU_PREFLIGHT_PASS，完整3074/6156、CUDA=false，
六G来源、Raw FP32输入、H7文本与三模型历史来源均通过。

已出现GV2_EPOCH10_FULL_REUSED irra并提交不可变只读结果引用，没有重新生成/编码/评分E10。
快照阶段：E10_REUSE_rasa。新Epoch图像尚未开始，这不是新Full结果。
原脚本继续：E10三模型引用/审计→E20/30/40/50/60分别Gallery生成→IRRA→CPU审计→RaSa→CPU审计→APTM→CPU审计。
不并行、不自动重试/恢复，失败保留现场。

## 输出与查看命令

正式输出：/home/lzf/ldx/outputs/tbps-framework-gv2-epoch-sweep/full/
控制：/home/lzf/ldx/tmp/gv2_epoch_sweep_full_v1/
启动日志：/home/lzf/ldx/tmp/gv2_epoch_sweep_full_launcher.log
阶段日志：控制目录中与phase同名的.log文件。

~~~bash
C=/home/lzf/ldx/tmp/gv2_epoch_sweep_full_v1
cat "$C/phase"
tail -n 20 "$C/status.log"
P=$(cat "$C/phase"); tail -n 30 "$C/$P.log"
ps -o pid,ppid,pgid,sid,tty,etime,args -p $(cat "$C/supervisor.pid")
tail -n 30 /home/lzf/ldx/tmp/gv2_epoch_sweep_full_launcher.log
~~~

G生成每32图及最终、图像编码每16图、原生评分按Query块记录进度；
CPU来源核验期间可能没有细粒度输出，不为日志修改冻结逻辑。
原generate_once、8/255、FP32/math/no_tf32、batch1、query chunk128、Full Top128不变；
H文本特征仅在同一Victim空间复用，目标结果不反馈G。

预计整链15–36小时，共享GPU与SHA I/O可能延长，不是完成承诺。
只有所有阶段和CPU审计通过，才能出现GV2_FIVE_NEW_EPOCHS_THREE_TARGET_FULL_AND_CPU_AUDIT。
六点表格、曲线和DR待真实Full成功后另行整理，不使用Smoke数据。
本轮启动核验后结束交互会话，不常驻监控，不创建轮询器。

[完整来源/资源/进程快照](results/gv2_epoch_sweep_full_startup/STARTUP_EVIDENCE.json)。
