# 跨模型迁移流水线启动记录

2026年11月12日20:05（北京时间）已启动独立后处理流水线。科学执行Commit `0795609f47ea15107bf9085d803141d7f736b179`；配置SHA `00a7ace233616eca64943000669d5634d8acaa210fba98533f25d8a84bb26fe7`。

启动后一次核对：Supervisor PID/PGID/SID均2148305、PPID1、无TTY；子进程2148326正常执行`inputs_cooperative`完整CPU输入审计。启动时无FAIL。本记录是启动快照，不表示CPU全量输入、GPU Smoke或Full已通过。

75项所选CPU回归通过；1项固定旧APTM工作树路径的配置用例不适用于新树，已排除并在CPU_EVIDENCE记录。两个目标真实导入/官方Tokenizer预检通过；两版Round60历史完成/审计回执和Checkpoint内部状态核验通过，CUDA未初始化。

后续自动完成两版完整CPU输入审计，等待已有增强版Supervisor退出，再执行两版×两目标四组Smoke及审计，全部通过后四组Full及独立审计，最终提交发布结果。固定Round60，3074 Gallery／6156 Query，Clean/Joint，原8/255与Natural-K每原词1字符；所有真实失败停止且不自动重试。旧训练与原FAIL不修改。

```powershell
ssh tbps-server "/home/lzf/ldx/envs/tbps-image/bin/python -B /home/lzf/ldx/projects/idea-test-tbps-gh-cross-model-v1-20261112/operations/gh_cross_model_v1_status.py"
```

只核对了脱离终端的启动身份，不持续监控。实时阶段、失败与正式结果由用户执行上面的只读命令检查；最终状态见[执行协议](GH_CROSS_MODEL_V1.md)。
