# APTM一次性后台实验链

脚本：src/tbps/engines/wait_aptm_full_chain.sh。基于e7d28861cf5a91a801e47b699d58ac9389900f4b，仅新增Bash控制层及CPU测试，不改APTM算法/模型/配置，不改RaSa监督器、G/H或历史结果。

用户本轮明确授权Codex直接nohup部署；后台满足条件后自动Smoke → Smoke CPU audit → 重新等待GPU → Full → Full CPU audit。Codex确认一次独立会话和等待状态后退出，不前台等待或持续监控。

## 门槛与RaSa证据兼容

每60秒查询GPU0：空闲≥20000MiB、利用率≤10%、无其他计算进程（包括Ollama；Xorg/Xwayland图形进程例外）。RaSa完整成功且进程退出后，连续3次资源满足才启动Smoke；任一失败、查询错误或RaSa未完成清零计数。Full前重新从0累计3次，不能沿用Smoke前计数。

RaSa现有监督器实际成功行为：
- phase=SUCCESS。
- launch.log为SUCCESS | RASA_FULL_AND_CPU_AUDIT_SUCCESS（兼容无_SUCCESS尾缀的同义标志）。
- full_gpu.log包含RASA_MATERIALIZED_EVALUATION_PASS full。
- cpu_audit.log包含RASA_MATERIALIZED_CPU_AUDIT_PASS full。
- 正式completed Manifest/完成回执/summary及最新execution Manifest/完成回执/report都合法；核验声明SHA、3074/6156、四组完整、status=PASS且model_frozen。
- RaSa评估/审计进程退出，GPU也无其他计算进程。

旧CPU审计入口没有单独JSON审计回执，成功日志加已知监督器SUCCESS是原始完成证据。本脚本不修改或补造RaSa回执；仅在自己的控制目录写rasa_verified.json，明确标记not_a_new_rasa_audit，保存原完成/执行/审计日志SHA及对规范SUCCESS | RASA_FULL_AND_CPU_AUDIT的显式对应关系。只读确认不加载特征、不重算排名、不重复运行RaSa审计。

RaSa phase为FAIL/FAILED/REFUSED/CANCELLED或成功来源损坏，APTM进入FAIL并退出。RaSa尚未完成则等待，不抢占或干预任何外部进程。

## 原有CLI与一次性保护

原工作树：/home/lzf/ldx/projects/idea-test-tbps-aptm-materialized-eval-20261018。
原Python：/home/lzf/ldx/envs/tbps-image/bin/python。
使用src/tbps/aptm_eval.py、configs/framework/materialized_aptm.server.json；参数完全沿用原CLI：
- --action run --scope smoke --allow-smoke。
- --action audit --scope smoke，CUDA_VISIBLE_DEVICES为空。
- --action run --scope full --allow-full-evaluation。
- --action audit --scope full，CUDA_VISIBLE_DEVICES为空。

每一步必须退出码0且对应精确PASS行存在，GPU步骤还须有正式完成回执。任一步失败/OOM/缺标志立即FAIL，保留所有输出，不自动重试、恢复、删除或覆盖。最终状态和launch.log写SUCCESS | APTM_FULL_AND_CPU_AUDIT。

flock防止同一控制目录多监督器；启动、等待期间及触发边界检查APTM进程/Smoke与Full输出。triggered在Smoke前以noclobber发布，full_triggered在Full前发布，均永久保留。既有输出或触发记录拒绝整条链重入；CPU Fixture允许路径环境覆盖，实际部署清除所有APTM_*覆盖变量以使用固定路径。

资源检查是快照而非GPU预约。运行期间沿用原CLI GPU状态/异常记录；不终止外部任务，CUDA OOM或非零退出停止实验链。没有绕过等待门槛直接启动GPU，没有--resume或自动重试。

## 日志、状态与取消

全部控制日志：/home/lzf/ldx/tmp/aptm_full_wait/。

- wait.pid：独立后台PID；phase：WAITING_RASA/WAITING_GPU/RUNNING_SMOKE/RUNNING_SMOKE_AUDIT/RUNNING_FULL/RUNNING_FULL_AUDIT/SUCCESS/FAIL。
- poll.log：RaSa准备状态及每阶段连续计数；gpu.log：GPU0资源/计算进程及查询错误。
- status.log：状态变化、取消和错误；launch.log：阶段启动/退出码及最终结论。
- smoke.log、smoke_audit.log、full.log、full_audit.log：原CLI完整stdout/stderr。
- nohup.log：控制器输出；rasa_verified.json：只读证据确认，非新的RaSa审计。
- triggered、full_triggered：已消费许可；失败后不得删除用来重试。

正式APTM结果保持/home/lzf/ldx/outputs/tbps-framework-aptm-materialized-evaluation/{smoke,full}/，不混入控制目录。

查看：
```bash
cat /home/lzf/ldx/tmp/aptm_full_wait/phase
tail -n 20 /home/lzf/ldx/tmp/aptm_full_wait/poll.log
tail -n 10 /home/lzf/ldx/tmp/aptm_full_wait/gpu.log
tail -n 20 /home/lzf/ldx/tmp/aptm_full_wait/status.log
tail -n 30 /home/lzf/ldx/tmp/aptm_full_wait/smoke.log
tail -n 30 /home/lzf/ldx/tmp/aptm_full_wait/full.log
tail -n 20 /home/lzf/ldx/tmp/aptm_full_wait/full_audit.log
```

阶段日志仅在相应阶段启动后出现。仅取消等待，不影响RaSa或其他用户进程：
```bash
case "$(cat /home/lzf/ldx/tmp/aptm_full_wait/phase)" in
  WAITING_RASA|WAITING_GPU)
    pid=$(cat /home/lzf/ldx/tmp/aptm_full_wait/wait.pid)
    if ps -p "$pid" -o args= | grep -q '/wait_aptm_full_chain.sh'; then
      kill -TERM "$pid"
    fi ;;
  *) echo "当前不在等待阶段，未发送信号。" ;;
esac
```

取消记FAIL及cancelled原因（退出码0），保留已消费许可和已完成Smoke。运行阶段不接受此取消方式，TERM/INT仅记录忽略，避免误伤GPU子进程。

本轮由Codex使用nohup及独立session实际部署，不需要用户再次启动。禁止重复执行启动命令。后台PID/部署快照登记PROJECT_STATE，之后以运行日志为准。

## CPU验证

21项新增真实Bash流程Fixture全部通过；模型、nvidia-smi、sleep使用Mock，回执由合成JSON和真实SHA校验。覆盖全部四阶段失败、缺标志、RaSa失败/未完成、坏回执/错误规模/进程未退出、两种显式成功标志、GPU阈值/连续3次重置/图形例外/查询失败、第二次GPU门槛、重复输出/进程/flock、取消及RaSa源文件SHA/mtime保持。

bash -n通过；另回归36项APTM接口与10项RaSa等待器测试。新脚本不改变13项已锁定APTM运行源码及原配置SHA；不以等待器部署状态声称Smoke或Full已完成。
