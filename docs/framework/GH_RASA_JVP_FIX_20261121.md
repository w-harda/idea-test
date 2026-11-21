# RaSa 源增强 Push-pull 方向导数校验修复

原实验正式第 1 轮的 G 阶段完成 1360/2048 条训练样本、5697 次 G 更新，在处理 `cuhk:9169:0` 时停止。原控制目录 `/home/lzf/ldx/tmp/gh_rasa_source_20261121` 保留。该轮没有完成回执或可恢复的 G/H Checkpoint，H 尚未更新。

用冻结的 RaSa 和原始初始 H 复现。触发属性的方向导数范数约为 0.001291。固定步长 0.01 的中心差分需要相减两个极接近的 FP32 输出，舍入使相对差异达到 2.445%。把步长改为 0.04、0.08、0.16 时，中心差分的相对差异分别为 0.505%、0.355%、0.103%。

原步长和 2% 检查门槛保留。首次比较超限时使用固定多步长序列和 Richardson 外推，要求相邻两组估算的差异在 1% 内，再与生产 JVP 比较，仍须满足 2%。对不稳定、非有限或稳定但超限的结果报错停止。独立 CPU 审计验证记录的步长、两项门槛和原有零扰动检查。训练继续使用原始自动微分结果。

- 16 项既有 CPU 回归、5 项新增回归、原生 RaSa CPU 探针及准备产物来源预检均通过。
- 真实 RTX 4090 回归覆盖 64 条训练样本及 201 项 JVP，最大校验差异为 1.659%。原故障属性的复核差异为 0.476%，相邻估算的差异为 0.769%。回归中 G/H 更新均为 0。
- 具体验证回执位于 `/home/lzf/ldx/tmp/gh_rasa_jvp_fix_20261121/real_gpu_regression.json`。固定配置 SHA256 为 `db162b1fb81d33e2d18f625d1408fb3bd6b20bed98de95fe5c9bfc01b7b1988b`。

修复版正式作业待启动。独立入口使用新控制与输出目录，从第 1 轮开始，每 10 轮保存配对权重，并按既定协议评估 RaSa、IRRA 和 APTM。

在服务器启动

```bash
/home/lzf/ldx/envs/tbps-image/bin/python -B /home/lzf/ldx/projects/idea-test-tbps-gh-rasa-jvp-fix-20261121/operations/gh_native_aug.py launch
```

检查状态

```bash
/home/lzf/ldx/envs/tbps-image/bin/python -B /home/lzf/ldx/projects/idea-test-tbps-gh-rasa-jvp-fix-20261121/operations/gh_native_aug.py status
```
