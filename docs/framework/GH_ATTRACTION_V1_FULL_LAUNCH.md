# 双向靠近KL版本Full启动回执

独立后台Supervisor及首个真实Smoke子进程已成功启动。核验仅覆盖启动身份，不包含训练、Smoke或正式Test结果。

- 科学执行Commit：`ec197618e639b169ebece0c317f5e36e9f5bd141`，分支`codex/gh-attraction-v1`。
- 配置SHA256：`b80e2d7393a00e34e365e5aec9568d1798089ff5640e4a5564ad9ab2cb5f39df`。
- Supervisor PID/PGID/SID：`3043588`，PPID1，无TTY。
- 启动时首子进程：`3043590`，阶段`01_smoke`。
- 工作树：`/home/lzf/ldx/projects/idea-test-tbps-gh-attraction-v1-20261109`。
- 输出：`/home/lzf/ldx/outputs/tbps-framework-gh-attraction-v1-20261109`。
- 控制/日志：`/home/lzf/ldx/tmp/gh_attraction_v1_20261109`。
- 服务器原始启动回执：`/home/lzf/ldx/tmp/gh_attraction_v1_20261109/launch_receipt.json`。

计划从零60轮，每10轮配对G/H并执行完整CUHK-PEDES × IRRA Joint Test及CPU独立审计；先真实训练Smoke和Joint Smoke审计。失败停止不自动重试。图像严格8/255，文本原Natural-K、每原词最多1Unicode字符。启动前30项CPU测试及两scope来源预检PASS。

助手核验启动后停止查看实验进度，由用户执行一次性只读命令：

```powershell
ssh tbps-server "/home/lzf/ldx/envs/tbps-image/bin/python -B /home/lzf/ldx/projects/idea-test-tbps-gh-attraction-v1-20261109/operations/gh_attraction_v1_status.py"
```

`SUCCESS_60_ROUNDS_SIX_JOINT_TESTS_PUBLISHED`表示全流程完成并发布；`FAIL_STOPPED_NO_RETRY`表示真实失败停止。当前回执不宣称任何科学阶段PASS。历史Cooperative和其他实验未改动。
