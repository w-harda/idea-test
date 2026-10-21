# 正式Full结果完整性只读核验

结论：**READ_ONLY_RESULT_INTEGRITY_PASS**。本次未调用原CPU audit入口，不反序列化Checkpoint或Tensor、不执行模型/评分/排名、不使用GPU。
原有15份独立CPU审计均已PASS，本轮只复核其回执、来源和日志，不把本次核验冒充新审计运行。

| 核验对象 | 结果 |
| --- | --- |
| 官方Test Gallery/Query | 3074/6156，原annotation逐成员/Caption/配对/PID/顺序exact |
| 六Checkpoint | 全部实际文件SHA与预设一致，见SOURCE_EVIDENCE.json |
| E10 | 3组结果/审计来源引用PASS，6条指标与历史Full exact，无重新生成/评分 |
| 新E20/30/40/50/60 | 各3074图，共15370；各一次G恢复，3074次前向、无跳过 |
| Claim与生成证明 | 15370个Claim逐一绑定同索引发布图像，无孤立Claim |
| 新Target Full/独立CPU审计 | 15/15 PASS；每组两模式、3074图/6156Query完整 |
| CSV/summary/Engine | 全部公开144个指标逐值exact；两位小数CSV格式一致 |
| 原生评分 | 原ModelSpace/Tokenizer/预处理SHA保持；FP32/batch1/query128，RaSa/APTM Top128 |
| 模型冻结 | 所有执行报告参数及buffers前后摘要相同，H/训练/目标反馈均0 |
| 不可变产物及SHA | 34721个新artifact；94058个独立文件SHA，约64.16GiB按字节读取 |
| pending/FAIL/缺失/孤立Claim | 0；新Full阶段日志无Traceback或FAIL |
| 只读保护 | 128810份使用文件stat前后exact；74184历史stat/289 SHA保持 |
| 正式总耗时 | 22小时32分15秒，含来源预检/生成/评分/审计 |

本轮文件SHA核验只证明现存字节与回执一致；像素数值/预算及Score Tensor形状、有限性等运行时检查沿用已完成独立CPU审计证据，未重新加载张量计算。
原artifact中的Claim永久保留是正常审计记录，“无未完成Claim”不表示删除这些Claim。

完成区间：2026-10-20T20:35:12+08:00 至 2026-10-21T19:07:27+08:00。
最终标志：SUCCESS | GV2_FIVE_NEW_EPOCHS_THREE_TARGET_FULL_AND_CPU_AUDIT。
完整逐文件SHA与stat清单留在服务器/home/lzf/ldx/tmp/gv2_sweep_analysis_20261021/；Git仅收录关键来源及清单digest，避免提交缓存或日志。
