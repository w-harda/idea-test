# CUHK-PEDES × RaSa · SEP Text-only

正式Full及独立CPU审计PASS。协议为SEP-TBPS-TextOnly-WordBudget1。

| Dataset | Victim | Attack | Budget | Query | Gallery | R@1 | R@5 | R@10 | mAP |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUHK-PEDES | RaSa | SEP Text-only | 1 Word | 6156 | 3074 | 57.81 | 75.76 | 82.13 | 53.90 |

指标单位为%。完整精度值为 {"R@1": 57.813514709472656, "R@10": 82.13125610351562, "R@5": 75.76348114013672, "mAP": 53.898651123046875}。

## 覆盖、编辑和NOOP

- 完整Query覆盖 6156/6156。
- 实际替换1词 6147，占 99.8538%。
- NOOP 9，占 0.1462%。
- Word Budget违规 0。
- NOOP原因 {"NO_DISTINCT_OFFICIAL_TAU_CANDIDATE": 9}。

## 来源与原生评分

- Source Checkpoint `/home/lzf/Attack/IRRA/logs/CUHK-PEDES/best.pth`，SHA256 `9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009`。
- Victim Checkpoint `/home/lzf/ldx/checkpoints/RaSa/rasa_cuhk_checkpoint.pth`，SHA256 `bc85da09c2991d5de503c2c2ac4f032e20cc536fec5094f0acff45e93d0104d2`。
- 攻击配置 `/home/lzf/ldx/projects/idea-test-tbps-sep-word-budget1-20261105/configs/framework/sep_text_only_full.server.json`，SHA256 `6bc8e0e232904ce18d148bdf6814b2280579de9fbc227172d109f144609334e0`。
- Seed规则 `uint32(first8hex(SHA256(dataset + NUL + query_id)))`。
- 冻结CUHK IRRA与完整CUHK Train参考图库，固定Source伪PID。生成不读取Target Test PID或评测反馈。单代理N/E=1/1、DPP权重[1]。CUHK IRRA也作为Source，该行是Source可访问评测。
- Gallery为原始图像。IRRA原生完整Gallery检索；RaSa完整Gallery全局Top128及ITM融合，fusion_batch16。

## 实际耗时

- Source文本生成累计 49454.487 秒，平均每Query 8.033542 秒。
- Source文本前向 398496 次。
- 本组Target加载、特征编码与评分累计 2310.535 秒。
- 同一数据集两Victim共用Source文本，Source耗时不应在两行之间重复累计。

## 独立审计及产物

- CPU审计 PASS，`/home/lzf/ldx/outputs/tbps-framework-sep-word-budget1-20261105/full/audit/cuhk/rasa/completed`。
- 审计Manifest SHA256 `22c3393185d10f91c32f11d514cbf8eb433ba84efe8ce1d60a98c4781f412967`。
- Target产物及执行Manifest `/home/lzf/ldx/outputs/tbps-framework-sep-word-budget1-20261105/full/target/cuhk/rasa`。
- Target完成Manifest SHA256 `2b1694489996fddab031c45dfb88f16f559939be96331b59ac9f315420a3af84`。
- 正式Full评分矩阵SHA256 `796577e87ac1905e09041a972046180c28ed371f1bafcaf517f90c1e2029e6e6`。
- 冻结文本及逐词编辑记录 `/home/lzf/ldx/outputs/tbps-framework-sep-word-budget1-20261105/source/cuhk`。
- Source完成Manifest SHA256 `6781b8cb35bf088443e1189ab22391abb732bec643c58e947d60f40078022335`。
- 发布器读取已提交的CPU回执和小型指标元数据，核验SHA、完整覆盖与字段一致性。完整评分矩阵及原始候选的数值核对沿用已完成的独立CPU审计。
