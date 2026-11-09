# GH-Cooperative-v1后台启动交接

科学执行Commit：`ca209a1d4417aa05f26514e4c358c006e137c2de`；分支`codex/gh-cooperative-v1`，启动前远端SHA一致。29项CPU测试、Smoke/Train完整CPU来源预检PASS。

唯一后台Supervisor PID/PGID/SID均为`2137488`、无TTY；首个子进程PID`2137490`，启动阶段`01_smoke`。启动核验只读取started/child回执并确认进程存在，没有等待或查看实验进度。权重、训练账本、原FAIL及其他实验均未修改。

计划：真实Smoke/独立CPU审计→Joint Smoke/独立CPU审计→从零60轮，每10轮保存本次配对G/H、CPU前缀审计、3074 Gallery/6156 Query IRRA Full Joint Test及CPU审计→六组表格/R@1与mAP曲线自动发布。真实错误停止，无重试。

控制目录：`/home/lzf/ldx/tmp/gh_cooperative_v1_20261109`。

输出目录：`/home/lzf/ldx/outputs/tbps-framework-gh-cooperative-v1-20261109`。

用户在PowerShell按需执行以下一条命令，即可查看当前阶段、训练轮次/游标、当前日志末尾和已通过CPU审计的正式Test结果：

```powershell
ssh tbps-server "/home/lzf/ldx/envs/tbps-image/bin/python -B /home/lzf/ldx/projects/idea-test-tbps-gh-cooperative-v1-20261109/operations/gh_cooperative_v1_status.py"
```

该命令执行一次即退出。最终成功阶段为`SUCCESS_60_ROUNDS_SIX_JOINT_TESTS_PUBLISHED`；真实失败为`FAIL_STOPPED_NO_RETRY`。源码及方法见[方法协议](GH_COOPERATIVE_V1.md)，封存启动身份见GH_COOPERATIVE_V1_LAUNCH.json。
