# G-v2 Epoch11→60正式后台训练启动管理

冻结训练工作树：/home/lzf/ldx/projects/idea-test-tbps-gv2-epoch60-ablation-20261020；
训练Commit固定635471b6d8a722616f87df16c300a8358861cd10。
本工作树idea-test-tbps-gv2-epoch60-launch-20261020仅管理启动、来源记录和训练成功后的CPU曲线/报告。
不改冻结工作树、原G/IRRA/Adam/RNG/缓存、训练算法或配置SHA。

## 执行链

1. 本轮前台真实tbps-image CPU Preflight必须exit0；控制cpu_preflight.log及preflight.exit保留。
2. 一次即时GPU/RAM/磁盘检查、固定Git/SHA、无重复进程/正式目录检查。
3. nohup+setsid启动一次监督器，started独占创建；无自动重试或恢复。
4. 原gv2_epochs.py --action train --allow-training连续执行绝对Epoch11–60。
5. 原train-audit独立CPU进程，仅成功后生成Epoch1–60原精度CSV、PNG/SVG曲线、六点SHA与报告，
   再提交推送本管理分支。绘图仅只读复用已有apattack Python的matplotlib，
   CUDA_VISIBLE_DEVICES为空、无模型加载，不安装依赖或修改任何训练环境。
6. 训练或审计失败立即停止；未提交Claim/pending按原规则保留，不自动恢复。
   正常TERM传给训练子进程，等待原安全Epoch边界退出，监督器标记STOPPED_EPOCH_BOUNDARY。
7. 没有攻击Gallery生成、G/H修改或任何IRRA/RaSa/APTM Full。

控制目录：/home/lzf/ldx/tmp/gv2_epoch60_train_v1/。
正式输出：/home/lzf/ldx/outputs/tbps-framework-gv2-epoch60-ablation/train/。
监督器保存supervisor.pid/pgid/sid、process.txt、child.pid、launcher.commit和管理源码SHA。
phase/status.log记录阶段；train.log原生GV2_TRAIN_PROGRESS记录绝对Epoch、样本位置与Adam step，
GV2_EPOCH_COMPLETE携带已提交Checkpoint SHA。没有GPU轮询/等待器。

CPU检查：冻结实现158项已验收；新增启动/曲线检查7项PASS（包括真实无GPU绘图）。
启动时间、真实PID和样本进度见后续启动证据；不能把启动成功写成50轮训练完成。
原正式E10、H7和所有既有Full输出均受原训练代码只读保护检查。

## 查看与停止

~~~bash
C=/home/lzf/ldx/tmp/gv2_epoch60_train_v1
cat "$C/phase"
cat "$C/process.txt"
tail -n 10 "$C/status.log"
grep -E '^GV2_(TRAIN_PROGRESS|EPOCH_COMPLETE|SAFE_EPOCH_BOUNDARY_STOP|EPOCH60_TRAINING_PASS)' "$C/train.log" | tail -n 5
tail -n 25 "$C/cpu_audit.log" 2>/dev/null
~~~

实时训练日志：

~~~bash
tail -f /home/lzf/ldx/tmp/gv2_epoch60_train_v1/train.log
~~~

正常停止：只给监督器TERM，监督器转发给训练子进程，等待当前Epoch完整发布。
不要对进程组或训练子进程使用SIGKILL；不要删除Claim/日志或自动重启。

~~~bash
kill -TERM "$(cat /home/lzf/ldx/tmp/gv2_epoch60_train_v1/supervisor.pid)"
~~~

完整成功必须同时有原训练完成回执、原CPU训练审计PASS及控制标志：
SUCCESS | GV2_EPOCH60_TRAINING_AND_CPU_AUDIT。
派生报告自动提交到codex/gv2-epoch60-launch的docs/framework/results/gv2_epoch60_training/；
completion.commit记录该次发布Commit。期间不运行模型评估。
