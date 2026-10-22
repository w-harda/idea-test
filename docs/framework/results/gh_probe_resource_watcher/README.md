# G/H固定数值探针：资源轮询与单次后台执行

本轮仅部署资源等待器。源模型、父状态、算法、配置及探针都在冻结工作树执行：
/home/lzf/ldx/projects/idea-test-tbps-gh-bidirectional-backend-20261022。

执行源码基准a46c6980c4abf58df02ca1ee199da799892dc81e；
冻结工作树HEAD为仅增加资源阻塞文档的07a31edb4d1a4fc8e28ce8968a037b03b5681075。
新轮询分支codex/gh-probe-resource-watcher，不在冻结树新增Python，也不更新其producer清单。

## 实际控制协议

1. watcher只用Python标准库与nvidia-smi，不导入Torch、不加载GPU模型。
2. GPU0空闲显存至少10240MiB，RAM至少32GiB、磁盘至少25GiB、inode至少80000。
   利用率和其他计算进程仅记录，不终止、不抢占。
3. 每180秒查询一次。连续两次全部达标后，调用原CPU Preflight：
   --action preflight --check-parent-states，CUDA_VISIBLE_DEVICES为空。
4. 验证冻结HEAD、工作树、脚本/配置/producer SHA、父状态及151378项历史metadata。
   原CPU Preflight继续验证所有原数据、父Checkpoint、来源SHA、导入闭包和隔离退出。
5. 检查probe_v1不存在，以及/proc没有另一原生probe实例。任何已有目录/Claim/pending/FAIL/completed都停止。
6. 再次检查GPU、RAM、磁盘、inode。资源回落则清零连续次数、继续等待。
7. 以O_EXCL、fsync文件和目录写入probe.started.json，永久消费本次调用。
8. 仅调用原native_cli --action probe --allow-gpu-probe。原门禁、预算和来源检查均保留。
   冻结入口自身仍会再次进行CPU及原资源检查；若在此间资源回落导致拒绝，也不自动重新调用。
9. 正常退出后调用原verify_config/verify_probe验证不可变回执，核验历史metadata，记录PROBE_PASS。
   失败记录PROBE_FAIL并退出，不重试、不换样本、不创建第二Attempt。
10. 不执行A1/B、图库刷新、训练、Validation、Test或Full。

固定样本、上限及数值标准完全继承原EXECUTION_PROTOCOL.md。
G≤4、IRRA图像≤4、IRRA文本≤6、JVP≤1、reader≤1、Optimizer.step=0。
轮询器不包含攻击Loss、梯度、条件或数值验收实现。

## 互斥、崩溃与停止

- 独占watcher.lock的flock保证同一控制目录只能有一个活跃监督器。
- probe子进程继承锁描述符；即使父进程意外退出，子进程仍持锁。
- watcher.started.json阻止该部署自动重启。
- probe.started.json在Popen前持久化；即使Popen失败、进程崩溃且probe目录尚未出现，也不允许重新调用。
- 不能把先前RESOURCE_BUSY视为数值探针FAIL；本次新状态记录在独立控制目录。
- 取消命令仅在WATCHING且尚无probe.started时生效，通过transition.lock与启动消费操作互斥。
  它仅写cancel.request.json；监督器等待期间最多1秒发现并退出，不发送信号给其他进程。
- 禁止手动删启动标记以复用本次任务；任何后续恢复需人工审查和新的授权。

## 目录和状态

部署及状态目录：/home/lzf/ldx/tmp/gh_native_probe_wait_v1/。

- gh_probe_watch.py：部署脚本；deployment.json固定源码Commit、脚本及配置SHA。
- supervisor.pid、supervisor.pgid、supervisor.sid及watcher.started.json：真实进程信息和启动时间。
- status.json：WATCHING / PROBE_RUNNING / PROBE_PASS / PROBE_FAIL，检查次数、连续次数和状态说明。
- WATCHING + terminal=true + detail=CANCELLED：已安全取消等待。
- resources.latest.json：最近GPU0、进程、RAM、磁盘及inode。
- poll.log：轮询及状态，1MiB滚动、3份备份，避免长期无限增长。
- cpu_preflight.log及最多2份备份：合格后的CPU预检；失败即退出，失败日志不会被后续重跑覆盖。
- probe.log、probe.process.json、probe.exit.json：唯一原生探针调用和退出码。
- receipt_check.log、probe.pass.json / failure.json：只读回执核验及最终状态。
- history_before.json、history_after.json：历史metadata结果。
- launcher.log：nohup启动/异常，轮询进度写poll.log。

CPU回归：39项PASS，全部资源/控制流Fixture，不执行GPU或真实模型。
真实原CPU来源预检和部署信息见DEPLOYMENT.json；不能用Fixture通过代替实际探针PASS。
原GPU峰值、JVP、梯度、FD及CUDA RNG恢复必须依据真实探针报告。
若原报告没有CUDA RNG恢复证据，记录NOT_REPORTED，不能据此宣称CUDA RNG已验证。

## 用户查看和取消命令

以下命令在Ubuntu SSH终端执行，不会发起实验：

~~~bash
cat /home/lzf/ldx/tmp/gh_native_probe_wait_v1/status.json
cat /home/lzf/ldx/tmp/gh_native_probe_wait_v1/resources.latest.json
tail -n 30 /home/lzf/ldx/tmp/gh_native_probe_wait_v1/poll.log
tail -n 50 /home/lzf/ldx/tmp/gh_native_probe_wait_v1/probe.log
ps -o pid,ppid,pgid,sid,tty,stat,etime,args -p "$(cat /home/lzf/ldx/tmp/gh_native_probe_wait_v1/supervisor.pid)"
~~~

安全取消等待（仅WATCHING允许，不能停止已经运行的探针）：

~~~bash
/usr/bin/python3 -B /home/lzf/ldx/tmp/gh_native_probe_wait_v1/gh_probe_watch.py --cancel
~~~

本轮由Codex部署并通过nohup setsid启动；用户无需再次启动。
最终状态不能提前声称为GH_NATIVE_NUMERIC_PROBE_PASS，亦不自动接续两轮Smoke。

## 部署实测快照

2026-10-23 00:08:32 +08:00启动，PID/PGID/SID1349747、PPID1、无TTY、flock已持有。
首轮WATCHING，空闲6501MiB；原probe_v1与smoke_v1均未创建。
39项CPU测试及真实冻结来源预检PASS；151378历史metadata无变化。
代码Commit为3b2ef7391e792a9c8599f350c2b2be5752aed763。
[部署版本、脚本SHA及实际进程证据](DEPLOYMENT.json)。后续动态状态以控制目录status.json为准。
