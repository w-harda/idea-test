# CUHK-PEDES × IRRA · SEP Text-only

正式Full及独立CPU审计PASS。协议为SEP-TBPS-TextOnly-WordBudget1。

| Dataset | Victim | Attack | Budget | Query | Gallery | R@1 | R@5 | R@10 | mAP |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUHK-PEDES | IRRA | SEP Text-only | 1 Word | 6156 | 3074 | 53.48 | 73.31 | 81.08 | 50.02 |

指标单位为%。完整精度值为 {"R@1": 53.476287841796875, "R@10": 81.07537841796875, "R@5": 73.31059265136719, "mAP": 50.016231536865234}。

## 覆盖、编辑和NOOP

- 完整Query覆盖 6156/6156。
- 实际替换1词 6147，占 99.8538%。
- NOOP 9，占 0.1462%。
- Word Budget违规 0。
- NOOP原因 {"NO_DISTINCT_OFFICIAL_TAU_CANDIDATE": 9}。

## 来源与原生评分

- Source Checkpoint `/home/lzf/Attack/IRRA/logs/CUHK-PEDES/best.pth`，SHA256 `9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009`。
- Victim Checkpoint `/home/lzf/Attack/IRRA/logs/CUHK-PEDES/best.pth`，SHA256 `9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009`。
- 攻击配置 `/home/lzf/ldx/projects/idea-test-tbps-sep-word-budget1-20261105/configs/framework/sep_text_only_full.server.json`，SHA256 `6bc8e0e232904ce18d148bdf6814b2280579de9fbc227172d109f144609334e0`。
- Seed规则 `uint32(first8hex(SHA256(dataset + NUL + query_id)))`。
- 冻结CUHK IRRA与完整CUHK Train参考图库，固定Source伪PID。生成不读取Target Test PID或评测反馈。单代理N/E=1/1、DPP权重[1]。CUHK IRRA也作为Source，该行是Source可访问评测。
- Gallery为原始图像。IRRA原生完整Gallery检索；RaSa完整Gallery全局Top128及ITM融合，fusion_batch16。

## 实际耗时

- Source文本生成累计 49454.487 秒，平均每Query 8.033542 秒。
- Source文本前向 398496 次。
- 本组Target加载、特征编码与评分累计 229.698 秒。
- 同一数据集两Victim共用Source文本，Source耗时不应在两行之间重复累计。

## 独立审计及产物

- CPU审计 PASS，`/home/lzf/ldx/outputs/tbps-framework-sep-word-budget1-20261105/full/audit/cuhk/irra/completed`。
- 审计Manifest SHA256 `51460c79f6e1dd2fcb2c5bddba87b9d54de80df43b5cbe1357136738e289ada2`。
- Target产物及执行Manifest `/home/lzf/ldx/outputs/tbps-framework-sep-word-budget1-20261105/full/target/cuhk/irra`。
- Target完成Manifest SHA256 `b3019feebea4510f2bf1aad76a7f80aa9c42f973b26fd748b1332bcfe00b68c3`。
- 正式Full评分矩阵SHA256 `589eebae0966b11dc57331ffac85eb8cdb0ff12b4637410474bf6786d53771c0`。
- 冻结文本及逐词编辑记录 `/home/lzf/ldx/outputs/tbps-framework-sep-word-budget1-20261105/source/cuhk`。
- Source完成Manifest SHA256 `6781b8cb35bf088443e1189ab22391abb732bec643c58e947d60f40078022335`。
- 发布器读取已提交的CPU回执和小型指标元数据，核验SHA、完整覆盖与字段一致性。完整评分矩阵及原始候选的数值核对沿用已完成的独立CPU审计。
