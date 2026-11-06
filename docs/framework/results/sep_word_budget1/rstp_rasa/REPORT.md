# RSTPReid × RaSa · SEP Text-only

正式Full及独立CPU审计PASS。协议为SEP-TBPS-TextOnly-WordBudget1。

| Dataset | Victim | Attack | Budget | Query | Gallery | R@1 | R@5 | R@10 | mAP |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RSTPReid | RaSa | SEP Text-only | 1 Word | 2000 | 1000 | 53.55 | 74.95 | 82.45 | 42.55 |

指标单位为%。完整精度值为 {"R@1": 53.54999923706055, "R@10": 82.45000457763672, "R@5": 74.94999694824219, "mAP": 42.55080032348633}。

## 覆盖、编辑和NOOP

- 完整Query覆盖 2000/2000。
- 实际替换1词 1994，占 99.7000%。
- NOOP 6，占 0.3000%。
- Word Budget违规 0。
- NOOP原因 {"NO_DISTINCT_OFFICIAL_TAU_CANDIDATE": 6}。

## 来源与原生评分

- Source Checkpoint `/home/lzf/Attack/IRRA/logs/CUHK-PEDES/best.pth`，SHA256 `9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009`。
- Victim Checkpoint `/home/lzf/ldx/checkpoints/cpgc_sep_official_20261028/rasa_rstp_checkpoint.pth`，SHA256 `7e62ca69bd0e334a7fdcbd65ad675c6858e3fbe2d8a5ab406063a4ab5c948ff6`。
- 攻击配置 `/home/lzf/ldx/projects/idea-test-tbps-sep-word-budget1-20261105/configs/framework/sep_text_only_full.server.json`，SHA256 `6bc8e0e232904ce18d148bdf6814b2280579de9fbc227172d109f144609334e0`。
- Seed规则 `uint32(first8hex(SHA256(dataset + NUL + query_id)))`。
- 冻结CUHK IRRA与完整CUHK Train参考图库，固定Source伪PID。生成不读取Target Test PID或评测反馈。单代理N/E=1/1、DPP权重[1]。CUHK IRRA也作为Source，该行是Source可访问评测。
- Gallery为原始图像。IRRA原生完整Gallery检索；RaSa完整Gallery全局Top128及ITM融合，fusion_batch16。

## 实际耗时

- Source文本生成累计 16357.470 秒，平均每Query 8.178735 秒。
- Source文本前向 129870 次。
- 本组Target加载、特征编码与评分累计 742.566 秒。
- 同一数据集两Victim共用Source文本，Source耗时不应在两行之间重复累计。

## 独立审计及产物

- CPU审计 PASS，`/home/lzf/ldx/outputs/tbps-framework-sep-word-budget1-20261105/full/audit/rstp/rasa/completed`。
- 审计Manifest SHA256 `b0287016d9a8e184a2f546274d8eaafb431920aec594625cc3f65a2e289ea6fe`。
- Target产物及执行Manifest `/home/lzf/ldx/outputs/tbps-framework-sep-word-budget1-20261105/full/target/rstp/rasa`。
- Target完成Manifest SHA256 `f340466eb43220a4d07ae5d6aef40caf8c8ef4a63544c2a7abe3bc08a75f8b44`。
- 正式Full评分矩阵SHA256 `3db2b5f5646bd8f152726ca89a376dd5cdb556c4c38d064fb00141417ff3cd3a`。
- 冻结文本及逐词编辑记录 `/home/lzf/ldx/outputs/tbps-framework-sep-word-budget1-20261105/source/rstp`。
- Source完成Manifest SHA256 `ccca3e223fd40fadf7a5d9f44f96cd1c430750b3b41959ab6346eab7756b9d3e`。
- 发布器读取已提交的CPU回执和小型指标元数据，核验SHA、完整覆盖与字段一致性。完整评分矩阵及原始候选的数值核对沿用已完成的独立CPU审计。
