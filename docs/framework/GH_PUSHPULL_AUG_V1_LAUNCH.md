# Push-pull三视图增强后台启动回执

独立Supervisor和首个真实Smoke子进程已启动。仅核验启动身份，未持续查看训练进度或宣称GPU阶段/正式效果PASS。

- 执行Commit：7ebf8684227f54fa2fc4e462e058e7144c6cd361，分支codex/gh-pushpull-augmentation-v1。
- 配置SHA256：7d80b03cd638a1718eae3ac5121e6156c56567160ec339abd4cfd9052b26cc1d。
- Supervisor PID/PGID/SID：1388274，PPID1，无TTY。
- 首子进程：1388276，阶段01_smoke。
- 工作树：/home/lzf/ldx/projects/idea-test-tbps-gh-pushpull-augmentation-v1-20261111。
- 输出：/home/lzf/ldx/outputs/tbps-framework-gh-pushpull-aug-v1-20261111。
- 控制与日志：/home/lzf/ldx/tmp/gh_pushpull_aug_v1_20261111。
- 原始启动回执：/home/lzf/ldx/tmp/gh_pushpull_aug_v1_20261111/launch_receipt.json。

29项CPU测试及Smoke/Train完整来源预检PASS。流程为真实训练Smoke/独立CPU审计、Joint Smoke/独立CPU审计，通过后从零60轮，每10轮对应配对Checkpoint完整IRRA Joint Test及CPU审计，最后自动发布六节点表格及曲线。三视图训练6144视图/2048原Train，G/H更新数保持；Test原图严格8/255，原文本Natural-K每词最多1Unicode字符。详见[方法与来源](GH_PUSHPULL_AUG_V1.md)。

用户执行一次性只读检查：

```powershell
ssh tbps-server "/home/lzf/ldx/envs/tbps-image/bin/python -B /home/lzf/ldx/projects/idea-test-tbps-gh-pushpull-augmentation-v1-20261111/operations/gh_pushpull_aug_v1_status.py"
```

最终成功SUCCESS_60_ROUNDS_SIX_JOINT_TESTS_PUBLISHED；真实失败FAIL_STOPPED_NO_RETRY。失败现场保留，不自动重试。旧实验源码、模型和产物保持，助手启动核验后停止监督，由用户查看后续进度。
