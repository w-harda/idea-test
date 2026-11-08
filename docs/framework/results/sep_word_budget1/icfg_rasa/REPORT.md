# ICFG-PEDES × RaSa · SEP Text-only

正式Full及独立CPU审计PASS。协议为SEP-TBPS-TextOnly-WordBudget1。

| Dataset | Victim | Attack | Budget | Query | Gallery | R@1 | R@5 | R@10 | mAP |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ICFG-PEDES | RaSa | SEP Text-only | 1 Word | 19848 | 19848 | 53.80 | 69.31 | 74.71 | 33.59 |

指标单位为%。完整精度值为 {"R@1": 53.80390930175781, "R@10": 74.71282196044922, "R@5": 69.30673217773438, "mAP": 33.58690643310547}。

## 覆盖、编辑和NOOP

- 完整Query覆盖 19848/19848。
- 实际替换1词 19829，占 99.9043%。
- NOOP 19，占 0.0957%。
- Word Budget违规 0。
- NOOP原因 {"NO_DISTINCT_OFFICIAL_TAU_CANDIDATE": 19}。

## 来源与原生评分

- Source Checkpoint `/home/lzf/Attack/IRRA/logs/CUHK-PEDES/best.pth`，SHA256 `9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009`。
- Victim Checkpoint `/home/lzf/ldx/checkpoints/cpgc_sep_official_20261028/rasa_icfg_checkpoint.pth`，SHA256 `91531bb30090923c8fb97b4a83662a9c610510ce4cd0ee208e82661798b62497`。
- 攻击配置 `/home/lzf/ldx/projects/idea-test-tbps-sep-word-budget1-20261105/configs/framework/sep_text_only_full.server.json`，SHA256 `6bc8e0e232904ce18d148bdf6814b2280579de9fbc227172d109f144609334e0`。
- Seed规则 `uint32(first8hex(SHA256(dataset + NUL + query_id)))`。
- 冻结CUHK IRRA与完整CUHK Train参考图库，固定Source伪PID。生成不读取Target Test PID或评测反馈。单代理N/E=1/1、DPP权重[1]。CUHK IRRA也作为Source，该行是Source可访问评测。
- Gallery为原始图像。IRRA原生完整Gallery检索；RaSa完整Gallery全局Top128及ITM融合，fusion_batch16。

## 实际耗时

- Source文本生成累计 179696.726 秒，平均每Query 9.053644 秒。
- Source文本前向 1425075 次。
- 本组Target加载、特征编码与评分累计 9645.931 秒。
- 同一数据集两Victim共用Source文本，Source耗时不应在两行之间重复累计。

## 独立审计及产物

- CPU审计 PASS，`/home/lzf/ldx/outputs/tbps-framework-sep-word-budget1-20261105/full/audit/icfg/rasa/completed`。
- 审计Manifest SHA256 `b239a528247d1f8c48e5332e5894246cba9561f1c8bdbf05010574a53d891076`。
- Target产物及执行Manifest `/home/lzf/ldx/outputs/tbps-framework-sep-word-budget1-20261105/full/target/icfg/rasa`。
- Target完成Manifest SHA256 `5067dae964bd9ecf142629440026483db4f67d709baac057d22dc71c0e2d7119`。
- 正式Full评分矩阵SHA256 `44bf46662e85bbcca37eef845793dede33d150f4ae57695360451f49660c85ab`。
- 冻结文本及逐词编辑记录 `/home/lzf/ldx/outputs/tbps-framework-sep-word-budget1-20261105/source/icfg`。
- Source完成Manifest SHA256 `900167f17640892046f7b0f2f5a56432a4b19be3f698230852c40e70b5fac901`。
- 发布器读取已提交的CPU回执和小型指标元数据，核验SHA、完整覆盖与字段一致性。完整评分矩阵及原始候选的数值核对沿用已完成的独立CPU审计。
