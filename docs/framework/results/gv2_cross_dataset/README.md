# G-v2 E60跨数据集迁移执行

固定配置：configs/framework/gv2_cross_dataset.server.json。服务器Python使用tbps-image，源码冻结、任务独立。G为原fp32/math/no_tf32，目标IRRA沿用C-PGC/SEP已核验的各域官方convert_weights原生参数与FP32检索特征/打分。所有Caption保持原样，完整PID正例。无G/H训练、微调、Backward、Optimizer或JVP。

## 授权执行

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-gv2-cross-dataset-20261102
nohup setsid /home/lzf/ldx/envs/tbps-image/bin/python -B -u operations/run_gv2_cross_dataset.py --allow-smoke-and-auto-full > /home/lzf/ldx/tmp/gv2_cross_dataset_20261102/supervisor.log 2>&1 < /dev/null &
```

单次Attempt，禁止重复启动。两目标真实Smoke与独立CPU审计均PASS后自动进入完整Full，不再次询问。任何真实阶段失败保留现场并停止，禁止自动重试。

## 一条命令查看进度

```bash
/home/lzf/ldx/envs/tbps-image/bin/python -B /home/lzf/ldx/projects/idea-test-tbps-gv2-cross-dataset-20261102/operations/run_gv2_cross_dataset.py --status
```

同时显示Supervisor PID/PPID/PGID/SID、真实阶段/当前Dataset、生成/编码/Query评分或审计进度、已完成Full组数及当前日志。控制根/home/lzf/ldx/tmp/gv2_cross_dataset_20261102；输出根/home/lzf/ldx/outputs/tbps-framework-gv2-cross-dataset-20261102。

共享GPU观察模式，无最低空闲显存门槛。实际OOM/RAM/磁盘/inode/数值错误仍拒绝；不修改其他任务进程或运行资产。

最终REPORT/METRICS/METRICS_2DP/SOURCE_EVIDENCE仅在两个Full独立审计通过后生成，完整大型资产留在服务器输出根。独立CPU审计从保存的完整分数重新计算官方PID排名与指标，逐图重读原图核对FP32来源/8/255预算，核对G/Target checkpoint和Tokenizer身份及历史stat。
