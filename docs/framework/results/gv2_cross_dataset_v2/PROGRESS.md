# v2真实Smoke回执与Full启动快照

这是执行中的快照，不是最终SUCCESS报告。完整结果仅在两个Full独立CPU审计PASS后由REPORT.md/METRICS.csv/METRICS_2DP.csv/SOURCE_EVIDENCE.json发布。

审计边界修复Commit618cc19f6e7f42522b9057ce5e3834a7dab81bc8，原112+新增29=141项相关CPU回归通过。另9个纯报告发布门禁用例通过。G/原generate_once/预算/原生IRRA/Tokenizer/排名数值源码未改；FP32、8/255与Batch保持。旧v1 FAIL及全部文件保留。

## 新的两个真实Smoke

RSTPReid与ICFG-PEDES各10 Gallery / 10原Query。两组真实G生成、各自官方IRRA Clean/Image-only评分及独立CPU审计全部PASS。G每组10次真实前向；IRRA每组20 image/10 text样本；Backward/Optimizer/JVP均0。各组完整不可变审计及SHA、预算和真实计数见SMOKE_RECEIPTS.json；这些小样本值不填入Full表格。

## 已完成RSTP Full

完整1000 Gallery / 2000 Query，独立CPU原像素复核、预算、完整PID排名重算审计PASS。正式指标单位%。

| Dataset | Victim | Mode | R@1 | R@5 | R@10 | mAP |
|---|---|---|---:|---:|---:|---:|
| RSTPReid | IRRA | Clean | 60.25 | 81.55 | 88.20 | 47.19 |
| RSTPReid | IRRA | Image-only (G-v2 E60) | 28.10 | 53.15 | 64.10 | 18.83 |

## ICFG后台运行

快照 1793605914.9465601 Unix time：3400/19848 Gallery，阶段GENERATE_GALLERY。未完成Full评分/审计，不报告正式ICFG指标，不宣称整项SUCCESS。

Supervisor PID/PGID/SID=989202、PPID1、无TTY；仅报告收尾进程PID1021509，PPID1，等待主任务SUCCESS与全部审计证据，不启动/重试实验。完成后自动提交小型报告并核对GitHub远端SHA；真实失败保留并停止，不重试。无固定VRAM门槛，不干预外部任务。

控制根 `/home/lzf/ldx/tmp/gv2_cross_dataset_v2_20261102`；主状态 `status.json`，报告发布状态 `publication_status.json`，日志 `supervisor.log` / `publication.log`。原始大型产物仅服务器outputs。

```bash
/home/lzf/ldx/envs/tbps-image/bin/python -B /home/lzf/ldx/projects/idea-test-tbps-gv2-cross-dataset-v2-20261102/operations/run_gv2_cross_dataset.py --status
```
