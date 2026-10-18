# AP-Attack B3与G-v2：三受害者正式Full对比

本报告仅提取用户已完成的正式Full结果与独立CPU审计证据，读取基准Commit为2047fb4c61f6ecfde32a148a593e87fe268462de。未执行GPU、模型加载、攻击生成、检索排名重算或新的完整CPU缓存审计。

## 正式结果

CUHK-PEDES官方Test：3074 Gallery、6156 Query。仅攻击Gallery，全部Text Query保持原文；多Caption、同PID全部正例、官方顺序不变。指标单位为百分数，展示两位小数；数值越低表示本设置下攻击越强。

| Target Model | Attack Method | R@1 | R@5 | R@10 | mAP |
| --- | --- | --- | --- | --- | --- |
| IRRA | Clean | 73.15 | 89.88 | 93.71 | 66.10 |
| IRRA | AP-Attack B3 | 46.35 | 68.18 | 76.51 | 40.43 |
| IRRA | G-v2 Image-only | 37.44 | 60.77 | 70.08 | 32.28 |
| RaSa | Clean | 76.22 | 90.29 | 94.09 | 69.18 |
| RaSa | AP-Attack B3 | 46.86 | 66.60 | 74.43 | 39.99 |
| RaSa | G-v2 Image-only | 42.33 | 61.71 | 69.56 | 35.05 |
| APTM | Clean | 74.61 | 89.26 | 93.18 | 64.13 |
| APTM | AP-Attack B3 | 52.44 | 73.02 | 80.26 | 42.39 |
| APTM | G-v2 Image-only | 50.19 | 70.21 | 77.13 | 39.66 |

metrics_2dp.csv可用于论文/WPS/PPT；full_precision.csv保留四项指标的原文件精度。未输出其他检索指标，未计算相对下降率。

## 对比结论与限制

- 三个受害者上，AP-Attack B3与G-v2 Image-only的四项指标均低于各自Clean，说明两种冻结图像攻击在本次Full设置下均造成检索性能下降。
- G-v2在三个模型上的R@1、R@5、R@10和mAP均低于B3，本次四项指标没有发现G-v2弱于B3的模型。
- IRRA：G-v2相较B3进一步降低R@1 **8.90个百分点**、mAP **8.14个百分点**。这是IRRA训练来源模型上的同模型攻击评估。
- RaSa：G-v2相较B3进一步降低R@1 **4.53个百分点**、mAP **4.95个百分点**。这是IRRA训练得到的冻结攻击产物向目标模型的跨模型迁移评估。
- APTM：G-v2相较B3进一步降低R@1 **2.24个百分点**、mAP **2.72个百分点**。这是IRRA训练得到的冻结攻击产物向目标模型的跨模型迁移评估。

- G-v2相对B3的R@1/mAP优势在IRRA最大，在APTM最小；这只是当前固定权重、数据与协议的观察，不能据此推断所有目标模型上的迁移规律。
- G-v2的优势是本次同模型及两个跨模型目标上均取得更低的检索指标；尚未证明的是等训练预算下的算法优势、独立属性模块贡献及多seed统计稳定性。这里没有训练预算对齐消融，不能用结果单独证明因果机制。
- AP-Attack B3使用34054张Train图像/11003 PID、60 Epoch、seed1234，ReID/Semantic权重10/10；不使用Caption或IRRA文本监督，另有DukeMTMC-reID语义骨干与CUHK语义反演来源。固定最后Epoch60，不按本次Test选择权重。
- G-v2使用2048张内部Train图像、128条PID隔离内部Validation、10 Epoch、seed42；训练利用完整Caption与属性/JVP监督，并按方案A逐属性及联合步更新，固定Epoch10。其评分参考图库还涉及34054张clean train图像（含内部Validation图片）；不能把2048简单解释为全部数据访问范围，也不能把两种方法视为相同信息访问权限。
- 因训练规模、Epoch、每样本更新次数、监督信息、辅助模型和网络结构均不同，本表是已冻结方法在同一测试与受害者协议下的实证比较，**不是完全等训练预算比较**。本轮没有测量/比较训练效率。

## 输入与评分一致性

同一受害者内，B3和历史G-v2 Full的Dataset Manifest（包含完整有序Gallery/Query、PID、Caption及配对）、ModelSpace（Checkpoint、Tokenizer、预处理、精度、评分协议）逐字段exact；三个模型的Dataset Manifest也exact。两个方法的图像预算均为原RGB空间L∞≤8/255，文本完全不攻击。

B3复用既有3074张/97批原生FP32攻击像素。其原生尺寸256×128：按既有像素传递实现，对(adv256−clean256)作bilinear尺寸映射后加到同一clean384×128；G-v2直接使用generation_v3的384×128物化像素。此差异保留原算法并明确记录，不是重新生成或针对目标优化。B3完整像素集合回执记录最大L∞=0.031372129917144775≤8/255；本轮引用并核对该回执，未重新反序列化全部像素。

IRRA保留原生归一化双塔评分；RaSa/APTM保留各自原生Tokenizer、预处理、候选融合与评分规则，未套用IRRA特征空间。跨模型目标没有给攻击器梯度、损失或排名反馈。Clean是统一物化输入下各目标原生基线，不直接等同论文原始图片输入设置的Clean。

**Clean和G-v2 Image-only不是此次重新编码的控制组**：程序显式复用已独立审计的同模型历史Full指标及原Clean文本表示；本轮逐项核验24个控制组数值与历史summary完全一致，未发生Clean漂移。只有B3新增目标图像编码与评分，不应把复用控制组写成三组均重新推理。

## Full与独立CPU审计

| Target Model | Full执行 | 独立CPU审计 | 参数/buffers | 新增图像样本编码 | 新增文本样本编码 |
| --- | --- | --- | --- | --- | --- |
| IRRA | PASS | PASS | 前后exact | 3074 | 0（复用Clean文本） |
| RaSa | PASS | PASS | 前后exact | 3074 | 0（复用Clean文本） |
| APTM | PASS | PASS | 前后exact | 3074 | 0（复用Clean文本） |

三个目标各加载受害者模型1次；G/H新加载与前向、Backward、Optimizer、JVP全部为0。模型保持eval、requires_grad=false、参数梯度为空。RaSa/APTM各记录787968融合pair；这些是原生候选融合计算，不是额外Gallery成员。

最终状态来自/home/lzf/ldx/tmp/b3_shared_run/phase=SUCCESS及status.log的SUCCESS | B3_THREE_MODEL_FULL_AND_CPU_AUDIT。日志记录完成时间2026-10-18 14:38:41 +08:00。

每模型真实执行证据：
- /home/lzf/ldx/outputs/tbps-framework-b3-shared-evaluation/full/<model>/completed/{metrics.csv,metrics_2dp.csv,summary.json,manifest.json,completed.json}
- /home/lzf/ldx/outputs/tbps-framework-b3-shared-evaluation/full/<model>/execution/{report.json,manifest.json,completed.json}
- /home/lzf/ldx/tmp/b3_shared_run/logs/<model>_full.log：B3_EVALUATION_PASS full <model>。
- /home/lzf/ldx/tmp/b3_shared_run/logs/<model>_full_audit.log：B3_CPU_AUDIT_PASS full <model>。
- <model>依次为irra、rasa、aptm；本报告未使用Smoke指标。

历史控制组来源：
- /home/lzf/ldx/outputs/tbps-framework-irra-materialized-evaluation/full/completed/summary.json
- /home/lzf/ldx/outputs/tbps-framework-rasa-materialized-evaluation/full/completed/summary.json
- /home/lzf/ldx/outputs/tbps-framework-aptm-materialized-evaluation/full/completed/summary.json

## 冻结Checkpoint及训练来源

以下5份Checkpoint的实际文件SHA-256本轮以普通文件流读取核验，没有torch.load或模型构造。

| 角色 | SHA-256 |
| --- | --- |
| AP-Attack B3 E60 | c1c637e2e5346a3011f23006bb77bad60e4bd4bc28eb746d5f6bfcd881ba4ebb |
| G-v2 E10 | f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e |
| IRRA | 9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009 |
| RaSa | bc85da09c2991d5de503c2c2ac4f032e20cc536fec5094f0acff45e93d0104d2 |
| APTM | 521c9bf784e7e821ade471deaafa2f25e2501950591e574dc60ca177e9be5a2c |

实际路径：
- AP-Attack B3 E60：/home/lzf/ldx/outputs/AP-Attack/cuhk_irra_semantic_10_10/G_CUHK_IRRA_AP.pth.tar
- G-v2 E10：/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0010/checkpoint.pt
- IRRA：/home/lzf/Attack/IRRA/logs/CUHK-PEDES/best.pth
- RaSa：/home/lzf/ldx/checkpoints/RaSa/rasa_cuhk_checkpoint.pth
- APTM：/home/lzf/ldx/checkpoints/APTM/aptm_cuhk_checkpoint_best.pth

B3训练配置与Checkpoint sidecar位于/home/lzf/ldx/outputs/AP-Attack/cuhk_irra_semantic_10_10/的config.json、G_CUHK_IRRA_AP.pth.tar.json；原训练来源摘要见[来源审计](../../B3_SOURCE_AUDIT.json)（早期PENDING为历史快照）。G-v2训练配置来自其outputs/train_v2/training_protocol.json，本轮SHA核验通过。完整关键来源路径与实际SHA见evidence.json。

## 本轮CPU核验与发布范围

- 标准库CPU核对9行/36项正式指标、两位小数CSV与summary一致性；历史控制组24项exact。
- 核验三个模型完整有序Manifest、ModelSpace、完成/执行Manifest及完成回执SHA，五份Checkpoint实际SHA及声明的历史来源SHA。
- 本轮实际读取的82份来源文件，前后SHA/inode/size/mtime保持不变；未对未读取的大型特征缓存宣称新审计。

- 独立CPU审计PASS是用户已经执行完成的真实日志；本轮只读核对，没有重新启动审计、模型推理或排名。
- 仅发布报告、两个四指标CSV、派生来源evidence.json及统一状态索引，不提交Checkpoint、像素、特征、原日志或正式outputs。
- 当前没有发现结果整理阻断项。未覆盖等训练预算对比、多seed置信区间、跨数据集及其他Baseline；本轮不扩展这些实验。
