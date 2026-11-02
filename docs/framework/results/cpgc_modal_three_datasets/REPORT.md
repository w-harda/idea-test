# C-PGC三数据集单模态消融扩展：执行准备

本报告是启动准备记录，不是四组新增Full完成报告。最终24行CSV和报告仅在四组评分及独立CPU审计全部PASS后发布到codex/cpgc-modal-three-datasets-results。

## 后台启动核验（快照）

执行源码Commit `969a6d0fe0292042b86cb27653a0060b8168b154`，计划SHA `ac32e1884c2839f3c9bef2e05cebf351f0e4bbd033711fb4a58af3e2df2852fd`。
监督器PID/PGID/SID均46866，PPID=1、无TTY，stdin=/dev/null，flock真实持有，已脱离SSH。
最新快照阶段：`RUNNING_FULL`，当前组合：`rstp/rasa`，新增完成`1/4`。
CPU Inventory已PASS；每组仍须完整Preflight、真实评分及独立CPU审计。启动不等于最终完成。
SEP原2960820/3041291保持存活，未发送信号或修改执行链。
[完整启动证据](LAUNCH_EVIDENCE.json)。

## 已核实状态

- CUHK两Victim四模式Full及独立CPU审计均PASS；真实历史结果只读复用，不再运行。原结果报告Commit b6b913943f874d7d8ce54f2f32d7a3c64d9c63a5。
- 4组目标历史Full和CPU审计回执存在：RSTP/IRRA、RSTP/RaSa、ICFG/IRRA=v2，ICFG/RaSa=v3。来源配置分别独立绑定，不伪造同一SHA。
- 72项不同CPU测试PASS。补充CUHK权重来源索引后，受影响的2项报告测试再次PASS；不计为74个不同测试。
- 与7973128相比，评测模块仅泛化数据集、Full尺寸及独立输出路径（30行新增、20行替换）；原Matched输入桥、特征读取、Top128记录、CSV、SHA/Claim检查函数AST一致。
- cache.py、victims.py、EvaluationEngine、ranking、VerifiedArtifactStore等数值和保存实现逐字节不变。

## 固定四模式

| Mode | 图像 | 文本 |
| --- | --- | --- |
| Matched Clean | clean_gallery | matched_text |
| Image-only | adv_gallery | matched_text |
| Text-only | clean_gallery | adv_text |
| Joint | adv_gallery | adv_text |

沿用独立Matched Manifest及原SHA绑定；Engine的clean_text输入明确绑定matched_text特征，不使用Raw Caption。官方UAP、12/255浮点语义、历史缩放/裁剪、官方单词替换不变。新的攻击生成、图文编码、Backward和Optimizer均0。

## 串行执行与真实门禁

1. CPU inventory完整核验已有CUHK消融所有单元，并核验四组历史Full/审计完成回执。
2. 每组在GPU之前独立CPU Preflight，对该组历史Target、Source、Audit所有单元执行Manifest、payload文件SHA、来源SHA和保护元数据检查。缺失/冲突立即FAIL，不自动补生成。
3. 按RSTP/IRRA→RSTP/RaSa→ICFG/IRRA→ICFG/RaSa顺序执行。各组先重放历史Matched Clean与Joint前32 Query，逐元素exact；作为接线Smoke，不重跑历史Full。
4. 新增Image-only和Text-only完成完整Gallery/Query评分，再由独立CPU进程执行审计。通过后才进入下一组。
5. 最终只读加入CUHK 8行，发布24行METRICS.csv、METRICS_2DP.csv、REPORT.md及SOURCE_EVIDENCE.json。

| Dataset | Gallery | Query | Victim | 原配置 |
| --- | ---: | ---: | --- | --- |
| RSTPReid | 1000 | 2000 | IRRA | v2 |
| RSTPReid | 1000 | 2000 | RaSa | v2 |
| ICFG-PEDES | 19848 | 19848 | IRRA | v2 |
| ICFG-PEDES | 19848 | 19848 | RaSa | v3 |

IRRA使用原Checkpoint、Tokenizer及权重/特征精度；RaSa使用原FP32全Gallery全局Top128候选与ITM正类logit，非候选保留原-100 sentinel，不宣称执行了全Gallery ITM。Query Chunk32、Fusion Batch16不变。原Matched Clean和Joint分数只读重算指标核对；新模式保存完整矩阵及逐Chunk、Top128索引。

CPU局部Token缓存固定有界8GiB，按需读取，不将全部ICFG局部Token放入GPU或RAM。所有原Full、Checkpoint、CUHK消融及SEP后台输出只读。

## 后台与资源

用户明确允许共享GPU。SEP监督器2960820及其Source子进程3041291在检查时仍存活，本任务不发信号、不修改其执行链。GPU显存/利用率仅诊断，不设置固定最低空闲显存门槛；保留各原配置RAM、磁盘、inode和全部来源检查。

新工作树：/home/lzf/ldx/projects/idea-test-tbps-cpgc-modal-three-datasets-20261102。
输出：/home/lzf/ldx/outputs/tbps-framework-cpgc-modal-three-datasets-v1/。
控制：/home/lzf/ldx/tmp/cpgc_modal_three_datasets_v1_20261102/。

入口operations/run_cpgc_modal_three_datasets.py --allow-shared-gpu-full，nohup+setsid后台运行，互斥锁、started不可覆盖。任何真实OOM、SHA、数值或CPU审计异常首错停止，不重试、不调批量或精度、不删除现场。

预期最终标志：CPGC_RSTP_ICFG_FOUR_FULL_MODAL_ABLATIONS_AND_CPU_AUDIT_PASS。未出现该标志且四份独立CPU审计未PASS时，不宣布完成。

```bash
cat /home/lzf/ldx/tmp/cpgc_modal_three_datasets_v1_20261102/status.json
tail -f /home/lzf/ldx/tmp/cpgc_modal_three_datasets_v1_20261102/status.log
```

评分进度使用原NATIVE_EXACT_SCORE_PROGRESS日志。独立CPU SHA预检未提供逐文件实时计数，不为增加进度而修改冻结数值函数。
