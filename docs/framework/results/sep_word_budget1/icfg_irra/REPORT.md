# ICFG-PEDES × IRRA · SEP Text-only

正式Full及独立CPU审计PASS。协议为SEP-TBPS-TextOnly-WordBudget1。

| Dataset | Victim | Attack | Budget | Query | Gallery | R@1 | R@5 | R@10 | mAP |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ICFG-PEDES | IRRA | SEP Text-only | 1 Word | 19848 | 19848 | 53.09 | 70.71 | 77.19 | 32.02 |

指标单位为%。完整精度值为 {"R@1": 53.08847427368164, "R@10": 77.1866226196289, "R@5": 70.71240997314453, "mAP": 32.01571273803711}。

## 覆盖、编辑和NOOP

- 完整Query覆盖 19848/19848。
- 实际替换1词 19829，占 99.9043%。
- NOOP 19，占 0.0957%。
- Word Budget违规 0。
- NOOP原因 {"NO_DISTINCT_OFFICIAL_TAU_CANDIDATE": 19}。

## 来源与原生评分

- Source Checkpoint `/home/lzf/Attack/IRRA/logs/CUHK-PEDES/best.pth`，SHA256 `9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009`。
- Victim Checkpoint `/home/lzf/Attack/IRRA/logs/ICFG-PEDES/best.pth`，SHA256 `a6b90aecee40307b240efcc2404b956f4f3e5a7a7373b91e2bc88dc298e74c38`。
- 攻击配置 `/home/lzf/ldx/projects/idea-test-tbps-sep-word-budget1-20261105/configs/framework/sep_text_only_full.server.json`，SHA256 `6bc8e0e232904ce18d148bdf6814b2280579de9fbc227172d109f144609334e0`。
- Seed规则 `uint32(first8hex(SHA256(dataset + NUL + query_id)))`。
- 冻结CUHK IRRA与完整CUHK Train参考图库，固定Source伪PID。生成不读取Target Test PID或评测反馈。单代理N/E=1/1、DPP权重[1]。CUHK IRRA也作为Source，该行是Source可访问评测。
- Gallery为原始图像。IRRA原生完整Gallery检索；RaSa完整Gallery全局Top128及ITM融合，fusion_batch16。

## 实际耗时

- Source文本生成累计 179696.726 秒，平均每Query 9.053644 秒。
- Source文本前向 1425075 次。
- 本组Target加载、特征编码与评分累计 1519.450 秒。
- 同一数据集两Victim共用Source文本，Source耗时不应在两行之间重复累计。

## 独立审计及产物

- CPU审计 PASS，`/home/lzf/ldx/outputs/tbps-framework-sep-word-budget1-20261105/full/audit/icfg/irra/completed`。
- 审计Manifest SHA256 `ad539fec6014dcd99cc3ee687f043f3f69da5423882df73dc95f888af76ceb00`。
- Target产物及执行Manifest `/home/lzf/ldx/outputs/tbps-framework-sep-word-budget1-20261105/full/target/icfg/irra`。
- Target完成Manifest SHA256 `079bc4552715775e8b54174a18f4a5cb22fb17a32adc6719255f3ba95bd03a6d`。
- 正式Full评分矩阵SHA256 `dcff5542220c7b0595dd734327bb275fc377f5fc7fe0358443e420c2c16a524b`。
- 冻结文本及逐词编辑记录 `/home/lzf/ldx/outputs/tbps-framework-sep-word-budget1-20261105/source/icfg`。
- Source完成Manifest SHA256 `900167f17640892046f7b0f2f5a56432a4b19be3f698230852c40e70b5fac901`。
- 发布器读取已提交的CPU回执和小型指标元数据，核验SHA、完整覆盖与字段一致性。完整评分矩阵及原始候选的数值核对沿用已完成的独立CPU审计。
