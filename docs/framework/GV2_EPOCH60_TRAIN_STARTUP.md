# G-v2 Epoch11→60正式后台启动核验

核验时间：2026-10-20T13:49:51.396151+08:00。状态：真实Epoch11训练已开始，完整50轮及最终CPU审计仍PENDING。

- 训练固定Commit：635471b6d8a722616f87df16c300a8358861cd10，原工作树干净。
- 外层启动管理Commit：14f2833be06c75b62e08473886c5c81268e53fd3；仅管理运行、原CPU审计与最终报告，不改原训练入口。
- CPU Preflight：真实tbps-image，exit0，父Checkpoint/来源/2176缓存和Epoch11顺序通过。
- 新增7项启动与标量汇总CPU测试PASS；未重跑GPU Smoke。
- 监督器PID/PGID/SID：454418/454418/454418；PPID=1，TTY=?。
- 训练子进程PID：454474。nohup+setsid，独立会话，SSH退出后继续。
- 最近原生日志：GV2_TRAIN_PROGRESS 11 544 / 2048 Adam [82834]。
- 核验时已有550/2048条Epoch11样本完成并fsync记录，
  对应累计Adam=82856；训练仍在继续，此数值是带时间戳快照。
- 首条严格为绝对Epoch11顺序中的cuhk:211:0，K4；父Adam80680，未从Smoke80685开始。
- 初始G/完整Adam/Python+NumPy+CPU+CUDA RNG恢复exact；学习率2e-4、betas(.5,.999)。
- 最近完整已提交Checkpoint仍是父Epoch10；当前Epoch11未完成，不能认定已发布。
- 启动资源：GPU24051MiB空闲、利用率0%、无计算进程；RAM约113.96GiB、磁盘约5871.3GiB。
  没有轮询等待、终止或抢占其他任务。
- 无FAIL/pending；完整Epoch/最终Adam484080/独立CPU训练审计尚未完成，不预先宣称成功。

控制目录：/home/lzf/ldx/tmp/gv2_epoch60_train_v1；输出：/home/lzf/ldx/outputs/tbps-framework-gv2-epoch60-ablation/train。
初始来源与实际控制状态见[证据索引](GV2_EPOCH60_TRAIN_STARTUP_EVIDENCE.json)；
[查询、正常Epoch边界停止和完成后处理](GV2_EPOCH60_TRAIN_LAUNCH.md)。

实际启动命令：

~~~bash
nohup setsid bash /home/lzf/ldx/projects/idea-test-tbps-gv2-epoch60-launch-20261020/src/tbps/ablations/run_gv2_epoch60_train.sh --allow-training >> /home/lzf/ldx/tmp/gv2_epoch60_train_v1/launcher.log 2>&1 < /dev/null &
~~~

任务顺序：原训练11–60 → 原train-audit独立CPU → 原精度Loss CSV/曲线/六点来源 → 本管理分支提交推送。
不自动恢复或重试。TERM请求等待原完整Epoch安全边界。后续Gallery生成或三模型Full未获授权，也没有入口调用。
正式训练预估10–14小时，随资源和I/O变化；确认启动后结束交互核验，不持续等待训练。
