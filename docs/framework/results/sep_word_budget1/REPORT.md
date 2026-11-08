# SEP-TBPS-TextOnly-WordBudget1

执行 Commit `7fd8a742c3c2b2c9a17872860d417e4a392e40d4`，当前状态 `SUCCESS`。
控制目录 `/home/lzf/ldx/tmp/sep_word_budget1_20261105`。输出目录 `/home/lzf/ldx/outputs/tbps-framework-sep-word-budget1-20261105`。

只执行Text-only。Source为CUHK IRRA与完整Train参考图库，固定Source伪PID。S1原100随机权重/DPP调用保留，N/E=1/1，最终权重[1]。CUHK IRRA为Source可访问模型。

原官方AST仅适配max_change_num=1、att_succ_num=1，禁用整词截断；重要性探针由删词改为基于原文的单位置[UNK]替换。其他原始单词、空白及标点位置保持，实际输出仅替换原始一个词跨度。单一可选位置直接保留softmax概率1，多位置常量/非有限重要性仍显式报错停止。T=10、patience=5、候选top50、tau gamma=.02保持。

生成过程和每次Source评分前检查全部候选、搜索状态和最终文本，超预算立即失败。无合法同义词时记录NO_LEGAL_SYNONYM并原文NOOP。Source-only伪标签及单代理属于既有检索适配，本版属于固定单词预算SEP-inspired变体。

## 实际阶段

|阶段|状态|秒|
|---|---|---:|
|reference|PASS|1.85|
|smoke-source-cuhk|PASS|75.74|
|smoke-target-cuhk-irra|PASS|10.32|
|smoke-audit-cuhk-irra|PASS|4.03|
|smoke-target-cuhk-rasa|PASS|20.29|
|smoke-audit-cuhk-rasa|PASS|4.15|
|smoke-source-rstp|PASS|72.73|
|smoke-target-rstp-irra|PASS|11.00|
|smoke-audit-rstp-irra|PASS|3.94|
|smoke-target-rstp-rasa|PASS|19.98|
|smoke-audit-rstp-rasa|PASS|4.00|
|smoke-source-icfg|PASS|69.22|
|smoke-target-icfg-irra|PASS|10.25|
|smoke-audit-icfg-irra|PASS|4.69|
|smoke-target-icfg-rasa|PASS|20.21|
|smoke-audit-icfg-rasa|PASS|4.60|
|full-source-cuhk|PASS|49884.48|
|full-target-cuhk-irra|PASS|240.16|
|full-audit-cuhk-irra|PASS|33.91|
|full-target-cuhk-rasa|PASS|2321.10|
|full-audit-cuhk-rasa|PASS|33.85|
|full-source-rstp|PASS|16454.01|
|full-target-rstp-irra|PASS|70.23|
|full-audit-rstp-irra|PASS|13.38|
|full-target-rstp-rasa|PASS|746.96|
|full-audit-rstp-rasa|PASS|13.39|
|full-source-icfg|PASS|182139.51|
|full-target-icfg-irra|PASS|1567.23|
|full-audit-icfg-irra|PASS|288.24|
|full-target-icfg-rasa|PASS|9721.49|
|full-audit-icfg-rasa|PASS|287.51|

## 正式Full结果

|Dataset|Victim|Attack|Budget|R@1|R@5|R@10|mAP|
|---|---|---|---|---:|---:|---:|---:|
|CUHK-PEDES|IRRA|SEP Text-only|1 Word|53.48|73.31|81.08|50.02|
|CUHK-PEDES|RaSa|SEP Text-only|1 Word|57.81|75.76|82.13|53.90|
|RSTPReid|IRRA|SEP Text-only|1 Word|47.95|72.15|80.40|39.45|
|RSTPReid|RaSa|SEP Text-only|1 Word|53.55|74.95|82.45|42.55|
|ICFG-PEDES|IRRA|SEP Text-only|1 Word|53.09|70.71|77.19|32.02|
|ICFG-PEDES|RaSa|SEP Text-only|1 Word|53.80|69.31|74.71|33.59|

## 覆盖与预算

|Dataset|Victim|Query|1词修改|NOOP|违规|Source前向|Source秒|Target秒|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|CUHK-PEDES|irra|6156|6147|9|0|398496|49454.49|229.70|
|CUHK-PEDES|rasa|6156|6147|9|0|398496|49454.49|2310.53|
|RSTPReid|irra|2000|1994|6|0|129870|16357.47|66.00|
|RSTPReid|rasa|2000|1994|6|0|129870|16357.47|742.57|
|ICFG-PEDES|irra|19848|19829|19|0|1425075|179696.73|1519.45|
|ICFG-PEDES|rasa|19848|19829|19|0|1425075|179696.73|9645.93|

逐组Checkpoint、NOOP原因、配置SHA、Seed及产物SHA见SOURCES.json。
