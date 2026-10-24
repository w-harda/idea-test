# G60→H60真实基线身份审查

本报告只读取真实Checkpoint、训练协议、完成记录、已有CPU审计和已有结果；没有重新训练、评分或执行原CPU审计。原精度与来源见[BASELINE_AUDIT.json](BASELINE_AUDIT.json)。

## 核心结论

| 项目 | 真实身份与结论 |
| --- | --- |
| G60 | 原G-v2 Image-only G(I)，60个绝对Epoch，累计484080次Adam更新 |
| H父状态 | 原H10，20480 visits / 20370次合法Adam更新；不是H60 |
| 适配H60 | H10之后新增Epoch11–60，102400 visits / 101850次合法更新 |
| 最终H60累计 | 122880 visits / 122220次合法Adam更新 |
| H适配使用的图像 | 固定G60生成的一份完整2048张Train图库；没有交替更新G |
| H60训练与原CPU审计 | HV1_G60_TRAINING_PASS / HV1_G60_TRAINING_CPU_AUDIT_PASS |
| 已找到的G60/H60检索结果 | 128张内部Validation Gallery / 128条Query，IRRA |
| G60/H60官方Test Full | 在已审查的适配输出中未找到；不能用G60/H7 Full代替 |
| 历史G60 Full | Epoch Sweep的正式结果绑定H7，不是这里的适配H60 |

H的前10轮沿原G10奖励环境训练，后50轮切换为固定G60奖励图库。因此这里的“H60”不是从头在G60上训练60轮，也不是H7 Selection或原H10的别名。

## Checkpoint与协议

| 模型 | 绝对路径 | SHA-256 |
| --- | --- | --- |
| G60 | /home/lzf/ldx/outputs/tbps-framework-gv2-epoch60-ablation/train/checkpoints/epoch_0060/checkpoint.pt | 9148a6ac06178b2eb474194ef6892c50182c5623b4e88bdcf88faa358aa46049 |
| H10父状态 | /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/formal_v1/training/checkpoints/visit_00020480/checkpoint.pt | 4ae964d51ea94840de71d9597d90a2234a36104e5d649cfee96577f59f94bacd |
| 适配H60 | /home/lzf/ldx/outputs/tbps-framework-hv1-g60-adaptation-epoch60/training/checkpoints/visit_00122880/checkpoint.pt | c0de26658c5f36083aaeb04dde3948cdd5bf941d30b1b5c97f7bb2c796404094 |

H60的checkpoint.protocol与原training/protocol.json内容及摘要一致，父Checkpoint为上述H10；checkpoint.json、steps.jsonl摘要和最终累计游标一致。旧审计记录50轮，每轮2048访问、2037合法更新、11个固定资格NOOP；400份周期/边界Checkpoint由原审计记录确认，本次未重跑原审计。

H的“合法Adam更新累计数”不意味着每个参数的step始终相同。真实H10恢复的参数step范围为20124–20370；正式恢复保持完整动量和各参数原step，而非统一改写成20370。

## 已有128条内部Validation结果

以下全部是旧结果，非本轮新评估、非官方Test。

| Attack Method | R@1 | R@5 | R@10 | mAP |
| --- | ---: | ---: | ---: | ---: |
| Clean | 100.00 | 100.00 | 100.00 | 100.00 |
| G60 Image-only | 39.84 | 59.38 | 65.63 | 48.81 |
| H60 Text-only | 98.44 | 100.00 | 100.00 | 99.22 |
| G60 + H60 Joint | 35.16 | 51.56 | 55.47 | 42.87 |

原精度Joint mAP=42.874080657958984，Image-only mAP=48.80861282348633，额外下降5.934532165527344个百分点。不得将这组128条结果与3074/6156的历史G60/H7 Test结果混为一表。

## 推理与评估口径

G仅接收图像，按冻结generate_once执行单次生成；FP32 RGB [0,1]、L∞≤8/255。H仅接收原Caption及caption-only Canonical；沿原Natural-K gamma3/beta1、原Unicode合法候选、每属性/共享词最多一次编辑；不强制用满K。

IRRA官方Checkpoint SHA为9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009。Tokenizer为IRRA原生CLIP Byte-BPE，图像沿原PIL RGB→bilinear Resize384×128→FP32输入；Adv RGB不重新量化。原R@K/mAP来自原生图文评分与全部同PID正例排名。

本轮另外实际读取官方Test Manifest，确认3074 Gallery、6156 Query、1000 PID；Manifest SHA：
a338f5429b64393b8062e0a7127773d72adb5dd6832eeb89d24c2c63c2f4d9eb。
它与历史Epoch60 IRRA正式Engine的dataset_manifest逐项一致。

原H60结果未提供官方Test ASR。新统一入口定义条件ASR@R1：分母为Clean Top1正确的Query数，成功为相同Query在攻击后Top1不再命中任何同PID正例；分母为0时记null。它通常不等于100−Attack R@1。

已确认G50轮扩展训练耗时16998.3820秒，旧H50轮适配耗时7127.0924秒。B还增加新语境JVP和每轮图库刷新，不能沿用旧H训练时长作为整链预测。
