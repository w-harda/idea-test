# RaSa Full一次性GPU等待器

脚本：`src/tbps/engines/wait_rasa_full.sh`。仅增加Bash执行控制，不修改模型、算法、数据、配置或现有评估入口。

默认每60秒查询GPU0显存、利用率及compute-apps。必须连续3次满足空闲≥8192MiB、利用率≤10%、无其他计算进程；Xorg/Xwayland图形进程忽略。任一不满足或查询失败均清零连续计数并继续等。不干预Ollama，不抢占、不卸载其他任务。

满足后调用原rasa_eval.py与materialized_rasa.server.json执行Full四组，成功且完成标志存在后独立CPU审计。Full或审计失败即退出，不自动重试。

flock防止同一等待器重复运行；启动及触发前检查已有Full进程/正式输出。triggered通过noclobber一次性发布，永久保留。已存在触发记录、输出目录或未退出Full时拒绝再次启动；不覆盖原实验。取消等待只停止本脚本及它自己的sleep，不操作其他进程。GPU条件仅为采样时刻状态，不构成GPU预约。

## 运行位置与日志

控制目录：`/home/lzf/ldx/tmp/rasa_full_wait/`

- wait.pid：独立后台监督器PID。
- phase：当前WAITING/RUNNING_FULL/RUNNING_CPU_AUDIT/SUCCESS/FAILED/REFUSED/CANCELLED。
- poll.log：60秒GPU快照与连续计数。
- launch.log：单次Full启动、退出码及最终成功。
- status.log：阶段变化、失败及取消记录。
- full_gpu.log：原Full全部进度和完整stdout/stderr。
- cpu_audit.log：独立CPU审计日志。
- nohup.log：等待监督器自身stderr/stdout。
- triggered：一次性触发记录，失败后不得删除来重试。

正式结果仍为 `/home/lzf/ldx/outputs/tbps-framework-rasa-materialized-evaluation/full/`，等待器不提前创建这个目录。成功标志：GPU RASA_MATERIALIZED_EVALUATION_PASS full、CPU RASA_MATERIALIZED_CPU_AUDIT_PASS full、监督器RASA_FULL_AND_CPU_AUDIT_SUCCESS。

## 查看与取消

```bash
cat /home/lzf/ldx/tmp/rasa_full_wait/phase
cat /home/lzf/ldx/tmp/rasa_full_wait/status.log
tail -n 20 /home/lzf/ldx/tmp/rasa_full_wait/poll.log
tail -n 20 /home/lzf/ldx/tmp/rasa_full_wait/launch.log
tail -n 30 /home/lzf/ldx/tmp/rasa_full_wait/full_gpu.log
tail -n 20 /home/lzf/ldx/tmp/rasa_full_wait/cpu_audit.log
ps -p "$(cat /home/lzf/ldx/tmp/rasa_full_wait/wait.pid)" -o pid,ppid,sid,stat,args
```

仅取消等待：

```bash
if [ "$(cat /home/lzf/ldx/tmp/rasa_full_wait/phase)" = WAITING ]; then
  kill -TERM "$(cat /home/lzf/ldx/tmp/rasa_full_wait/wait.pid)"
else
  echo "已不在等待阶段，未发送信号。"
fi
```

需要首次手动启动时，原命令如下；本轮已明确授权Codex使用nohup脱离会话启动，因此不要重复执行：

```bash
mkdir -p /home/lzf/ldx/tmp/rasa_full_wait
nohup bash /home/lzf/ldx/projects/idea-test-tbps-rasa-materialized-eval-20261017/src/tbps/engines/wait_rasa_full.sh  > /home/lzf/ldx/tmp/rasa_full_wait/nohup.log 2>&1 < /dev/null &
echo $!
```

本轮启动额外使用独立session，确保不依赖SSH。启动后只做一次进程/首条等待日志确认，Codex不持续轮询或监控。实际后台PID及状态由PROJECT_STATE和控制目录记录。

## CPU验证

bash -n通过。10项合成CPU测试通过（0.57秒）：所有nvidia-smi/模型命令使用Fixture，未运行真实GPU或Full；验证阈值边界、连续计数重置、查询失败、图形例外、已有输出/Full、flock、失败不重试和取消自身sleep。评估算法及已锁定producer SHA保持不变。

## 本轮部署记录

代码Commit：8e4a4ab3d53df1fa940a509e6937b346e5cda91a。2026-10-17 23:45:13+0800启动PID3297416；一次性确认PPID=1、SID=3297416、TTY=?、stdin=/dev/null，stdout/stderr为nohup.log。首条检查WAITING 0/3，GPU空闲3865MiB、利用率100%、Ollama PID2436556；Full目录与triggered尚不存在。此记录仅为启动快照，不代表Full完成；后续由独立后台进程执行，Codex不持续监控。
