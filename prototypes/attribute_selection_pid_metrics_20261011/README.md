# 固定100句 PID-mAP 补算

仅在服务器运行。复用 `attribute_tta_selection_value_20261005/outputs/diagnostic`，原输入只读，独立输出；不修改 AttributeInstance / S_raw / Top-K / JVP / TTA。

## 评测协议

原诊断每句都使用34054张clean train图库，仅替换自己的paired image。所有同PID图库图像均为正例，无camera过滤，复用正式 `text_to_image_metrics` 计算AP/CMC/mINP。

旧输出没有保存对抗图像/最终图像特征，无法构造共享adversarial gallery。对IRRA可以通过原fp32 singleton text forward重建全部clean scores，并把paired位置回填旧final paired cosine，恢复该query原来的完整score vector。对clean位置也回填旧I0 cosine，保留原singleton image encode与批量cache的数值差。用旧paired rank、ties、soft-rank逐句核验；不重跑image forward、攻击、JVP或scoring/backward。

这是固定100句的query-conditioned paired replacement PID-mAP诊断，**不是完整test、不是共享对抗图库**。gallery_audit.json记录重复源图、覆盖及局限。

## 指标

输出IRRA clean/adv mAP、单模型下降率。当前M=1，aAP=mAP、mDR=单模型下降率。未来聚合：
- aAP = 多victim mAP平均；
- mDR = (aAP_clean-aAP_adv)/aAP_clean ×100；
- mean_per_victim_DR另列，仅辅助，不混作mDR。

原AP原生脚本使用mean-per-target-DR，因此与本入口论文公式不同；比较时注明口径。

`aggregate.py`接受未来真实victim_summary.json，要求query、gallery layout、attack集合和协议一致，并拒绝重复victim。当前只支持IRRA分数重建；其它victim需要真实对抗图像再编码，不能从IRRA cosine推算。

## 命令

以下命令由用户手动运行；默认preflight只CPU，不执行100句推理。

```bash
cd /home/lzf/ldx/projects/idea-TBPS-test1
PYTHONDONTWRITEBYTECODE=1 /home/lzf/ldx/envs/tbps-image/bin/python prototypes/attribute_selection_pid_metrics_20261011/run.py --mode preflight

tmux new-session -d -s tbps-selection-pid-map 'cd /home/lzf/ldx/projects/idea-TBPS-test1 && PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 /home/lzf/ldx/envs/tbps-image/bin/python -u prototypes/attribute_selection_pid_metrics_20261011/run.py --mode diagnostic --allow-diagnostic > prototypes/attribute_selection_pid_metrics_20261011/outputs/diagnostic.log 2>&1'
```

结果：`outputs/diagnostic/metrics.csv`、`summary.md`、`summary.json`、`victim_summary.json`、`per_query_metrics.jsonl`、`reconstruction_checks.json`、`gallery_audit.json`、`completed.json`。拒绝覆盖已有目录。

完成标志：日志COMPLETE，completed.json method_audit_passed=true/case_count=100/paired_rank_and_ties_exact=true。无需重新跑约2.7小时的攻击；耗时主要为冻结SHA校验、模型加载及100次text forward。
