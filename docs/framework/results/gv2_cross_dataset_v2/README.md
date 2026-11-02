# v2后台执行与状态

本配置属于独立v2；旧v1失败Attempt永久保留，不运行旧命令。已授权单次nohup+setsid启动v2，CPU Preflight → RSTP Smoke/审计 → ICFG Smoke/审计 → RSTP Full/审计 → ICFG Full/审计 → 报告。全成功前不发布正式指标。

状态命令：

```bash
/home/lzf/ldx/envs/tbps-image/bin/python -B /home/lzf/ldx/projects/idea-test-tbps-gv2-cross-dataset-v2-20261102/operations/run_gv2_cross_dataset.py --status
```

控制/status.json与supervisor.log位于/home/lzf/ldx/tmp/gv2_cross_dataset_v2_20261102。输出为/home/lzf/ldx/outputs/tbps-framework-gv2-cross-dataset-v2-20261102。真实失败禁止自动重试或覆盖，模型/预算/Batch/精度不降级，显存只观察无最低门槛。

## 审计后自动发布

独立operations/finalize_gv2_cross_dataset.py仅等待主监督器SUCCESS，重新核验两Smoke/两Full审计、4行正式指标、旧FAIL保护后提交/推送报告并核对远端SHA；不启动、恢复或重跑实验。publication_status.json位于同一v2控制目录。失败只记录并停止，不重试。其新增文件不属于运行中的冻结模型/评分依赖，执行Commit仍为618cc19。
