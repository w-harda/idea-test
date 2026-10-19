# SGA官方算法（IRRA Source）三模型正式Full结果

日期：2026-10-19。本报告仅从已完成的正式Full原始结果生成，代码执行版本为f8d3431b4f51d4fb117cf031f6d827bb89fab69c，启动登记版本3caacf0ac124ac532febb587a290721c2d4d6608。报告提交SHA由PROJECT_STATE后续登记。

**结论：Source Full、IRRA/RaSa/APTM Target Full、独立CPU Audit均PASS；正式实验已完成。** 完成时间2026-10-19 06:07:53 +08:00，最终标志：
SUCCESS | SGA_OFFICIAL_IRRA_SOURCE_THREE_TARGET_FULL_AND_CPU_AUDIT

本轮只用标准库进行文件读取、SHA/身份/标量核验及报告生成；模型/Tensor反序列化、GPU、攻击、完整排名重算均0，未重新执行已完成的CPU审计。

## 1. 实验配置与算法来源

| 项目 | 固定设置 |
|---|---|
| Dataset / split | CUHK-PEDES官方Test |
| 完整成员 | 3074 Gallery、6156 Query；官方原序、多Caption及全部同PID正例 |
| Source | 官方CUHK IRRA；唯一攻击源 |
| Target | IRRA同源；RaSa/APTM黑盒迁移 |
| 官方SGA Commit | f49e6926cd13148e49bdbc2c3e537660e096230c |
| 实际攻击入口 | 原SGAttacker.attack、ImageAttacker、TextAttacker |
| 图文流程 | 原Caption文本攻击T1 → T1引导图像攻击 → 从原Caption重新执行文本攻击T2 |
| 图像预算 / 轮次 / 步长 | 官方FP32 2/255 / 10 / 0.5/255 |
| 多尺度 | 原图＋0.5/0.75/1.25/1.5，原高斯输入增强及BICUBIC处理 |
| batch / seed | 8 / 全局42；最后batch保留2图 |
| 文本 | 原BERT MLM候选、KL重要性、图像引导负匹配选择；每段最多替换1词，最终从原文重启 |
| Source输入 | PIL RGB→bilinear384×128→ToTensor；官方pre_caption(raw,30) |
| Source模型接口 | 原生IRRA单位全局特征；官方BERT30 WordPiece→文本→IRRA Byte-BPE77桥 |
| 优化权限 | 只用IRRA输入梯度及Test原图/Caption配对；不更新模型、不访问RaSa/APTM反馈 |
| 评价 | 三Target各自原生预处理、Tokenizer及评分，统一全部同PID正例排名，无camera过滤 |
| 指标 | R@1、R@5、R@10、mAP，百分数；表格保留两位小数 |

原attacker.py SHA-256：79384bb914824c59c2a7c65c9b3b565cebac9c3b469a7a15d8f21b5ebf110a30。[固定官方源码](https://github.com/Zoky-2020/SGA/tree/f49e6926cd13148e49bdbc2c3e537660e096230c)及运行producer当前SHA均与协议一致，官方Git HEAD正确且工作树干净。没有新增ITM/Push-Pull/属性Loss、扩大预算、替换更新策略或使用旧白盒优化循环。

本结果是**原SGA算法在IRRA Source上的移植**，不是原论文ALBEF权重/数据设置的完全复现，也不能称与R²TUA的SGA适配协议完全一致。BERT与IRRA源Tokenizer不同的必要I/O差异见[此前接口报告](../../SGA_IRRA_SOURCE.md)及[ADR-0010](../../../decisions/0010-sga-official-irra-source.md)。

Source独立环境Torch2.0.0+cu118、torchvision0.15.0+cu118、Transformers4.28.1；Target继续使用已有原生环境。目标模型不共享IRRA Embedding，不反传攻击。

| 角色 | Checkpoint SHA-256 |
|---|---|
| IRRA Source / Target | 9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009 |
| RaSa Target | bc85da09c2991d5de503c2c2ac4f032e20cc536fec5094f0acff45e93d0104d2 |
| APTM Target | 521c9bf784e7e821ade471deaafa2f25e2501950591e574dc60ca177e9be5a2c |
| BERT MLM | 68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3 |


## 2. 正式论文结果表

**以下Clean均为本实验Matched Clean；所有数值来自本次Full的summary与metrics_2dp.csv，未混入两图Smoke或旧SGA-TBPS White-box。**

| Target Model | Attack Method | R@1 | R@5 | R@10 | mAP |
|---|---|---:|---:|---:|---:|
| IRRA | Clean | 69.41 | 87.23 | 92.02 | 63.01 |
| IRRA | SGA Image-only | 5.52 | 13.71 | 19.61 | 4.27 |
| IRRA | SGA Text-only | 44.96 | 67.41 | 77.47 | 42.80 |
| IRRA | SGA Joint | 2.70 | 7.57 | 11.65 | 2.29 |
| RaSa | Clean | 74.45 | 89.23 | 93.26 | 67.45 |
| RaSa | SGA Image-only | 59.93 | 79.37 | 85.59 | 53.79 |
| RaSa | SGA Text-only | 53.25 | 73.21 | 80.46 | 49.15 |
| RaSa | SGA Joint | 39.13 | 60.38 | 69.48 | 35.64 |
| APTM | Clean | 73.55 | 88.82 | 92.84 | 63.38 |
| APTM | SGA Image-only | 62.82 | 81.08 | 87.10 | 53.98 |
| APTM | SGA Text-only | 54.21 | 74.64 | 81.82 | 47.51 |
| APTM | SGA Joint | 42.59 | 63.78 | 72.66 | 37.59 |


SGA Joint是Baseline主结果。Image-only/Text-only是**同一组官方Joint攻击产物的评估组合**：Image-only用Adv图＋Matched Clean文本，Text-only用Clean图＋最终T2文本；并非分别重新运行的独立单模态优化。特别是Adv图已受T1指导，T2已受Adv图指导。

原始精度保留于同目录SGA_OFFICIAL_IRRA_FULL_EVIDENCE.json的raw_precision_metrics，论文/WPS导入使用SGA_OFFICIAL_IRRA_FULL_METRICS_2DP.csv。

## 3. SGA效果分析

所有百分点差均用原始精度相减后保留两位小数，可能与表内已取两位小数直接相减存在0.01的尾差。

- IRRA同源：Joint降至R@1=2.70、mAP=2.29，相对自己的Matched Clean绝对降低66.72、60.73个百分点。图像分支在同源模型上影响较大（Image-only mAP4.27，Text-only42.80）；Joint进一步降低。
- RaSa迁移：Joint为R@1=39.13、mAP=35.64，相对自己的Matched Clean降低35.32、31.82个百分点。Text-only比Image-only更低，Joint低于两者。
- APTM迁移：Joint为R@1=42.59、mAP=37.59，相对自己的Matched Clean降低30.96、25.79个百分点。Text-only比Image-only更低，Joint低于两者。
- 三个Target的Joint四项指标都低于各自单模态组合；相对Image-only额外降低mAP分别为1.98、18.15、16.39个百分点。这是固定产物组合的结果差异，不是独立机制因果消融。
- 当前IRRA同源降幅最大；在两个迁移Target中，RaSa的Matched Clean→Joint绝对下降更大。不同模型原生评分和Clean水平不同，不据此推导普遍的鲁棒性排序或跨数据集结论。

本文只报告绝对指标及百分点差，不计算DR，也不使用历史Raw Clean计算本次SGA下降率。

## 4. 与G-v2 + H-v1 Joint的描述性比较

历史主方法数据重新只读核对原三模型正式Full summary和已有CPU审计日志，模型Checkpoint与本次相同；与用户提供的两位小数一致。

| Target Model | SGA Joint R@1 | G-v2 + H-v1 Joint R@1 | SGA Joint mAP | G-v2 + H-v1 Joint mAP |
|---|---:|---:|---:|---:|
| IRRA | 2.70 | 26.53 | 2.29 | 23.55 |
| RaSa | 39.13 | 22.87 | 35.64 | 17.20 |
| APTM | 42.59 | 38.82 | 37.59 | 30.49 |


- IRRA：SGA Joint最终R@1、mAP分别更低23.83、21.27个百分点。
- RaSa：G-v2 + H-v1 Joint最终R@1、mAP分别更低16.26、18.43个百分点。
- APTM：G-v2 + H-v1 Joint最终R@1、mAP分别更低3.77、7.10个百分点。

因此本次结果不是“SGA在所有模型更强”或“主方法在所有模型更强”：SGA同源最终指标更低，主方法在RaSa/APTM迁移上的最终指标更低。上述仅为两个既有协议的描述性数值比较，不能归因为单独某个模块或等预算优势。

### 必须保留的协议差异

| 维度 | 本次SGA Joint | G-v2 + H-v1 Joint |
|---|---|---|
| Clean | Matched Clean，pre_caption30与源输入桥 | 历史Raw Clean |
| 图像预算 | 官方FP32 2/255 | 8/255 |
| 文本编辑 | 官方BERT MLM词替换，最终至多1个词槽替换 | H-v1 Unicode字符替换，Dynamic-K及每属性词至多1字符 |
| Test配对访问 | 可见原图及其Caption对应关系，并用于迭代源攻击 | 冻结G只接图像；H Caption-only，不用正确图像/PID/Test反馈 |
| 推理成本 | 每batch十轮图像梯度、两段文本候选/重要性评估 | 冻结生成器/策略前向 |
| 受害者权限 | IRRA源可微；RaSa/APTM仅黑盒评分 | IRRA训练来源；RaSa/APTM消费冻结产物 |

Clean实际数值也不同：

| Target Model | SGA Matched Clean R@1 | 历史Raw Clean R@1 | SGA Matched Clean mAP | 历史Raw Clean mAP |
|---|---:|---:|---:|---:|
| IRRA | 69.41 | 73.15 | 63.01 | 66.10 |
| RaSa | 74.45 | 76.22 | 67.45 | 69.18 |
| APTM | 73.55 | 74.61 | 63.38 | 64.13 |


不能将不同Clean、预算、编辑语义和信息访问权限下的最终数值写成严格等条件公平比较；不把预处理造成的Clean差异当作攻击效果。本轮不为对齐协议修改任何已有产物。

## 5. 执行时间、真实调用与资源

全过程：2026-10-19 00:51:36至06:07:53 +08:00，**5小时16分17秒**（18977秒），包括CPU预检、全部模型加载、Source、三Target、最终审计及进程外围开销。Source各批攻击段耗时合计1234.2041816599667秒；不能把它等同全流程耗时。

| 会话 | 执行报告耗时（秒） | GPU allocated峰值（GiB） | reserved峰值（GiB） | CPU Peak RSS（GiB） |
|---|---:|---:|---:|---:|
| IRRA Source + BERT MLM | 1359.75 | 6.3714 | 10.6270 | 2.2822 |
| IRRA Target | 278.55 | 0.9008 | 0.9785 | 2.2075 |
| RaSa Target | 9088.98 | 1.8555 | 2.0332 | 25.0871 |
| APTM Target | 8146.56 | 0.9703 | 1.0488 | 6.7039 |


独立CPU审计阶段06:06:48→06:07:53，约65秒。资源为各执行进程报告峰值，不能把不同会话峰值相加当作同时使用，也不是整台服务器所有进程总内存。

Source IRRA与MLM各加载1次，生成385批（384批8图＋末批2图），3074张图各一份最终Adv，6156句各一份最终文本。源图像样本前向159849=52×3074+1（原图/最终图、十轮五尺度、额外输入梯度检查1）；源文本396849包含遮词、候选和中间文本，不是独立Query数。MLM调用2376次/41800样本。Optimizer与G/H加载0。

每个Target仅加载1次，6148图像/12312文本样本前向（Clean与Adv各一遍）；RaSa/APTM各3151872融合pair。三个Target的source_regeneration均0，协议ranking_feedback_to_attack=false，Source target_models_in_generation为空。IRRA源、MLM和三个Target参数/buffers前后摘要exact、eval、requires_grad=False、参数grad None。

## 6. 产物、预算和文本完整性

- 官方annotation逐项核对3074 Gallery /6156 Query的原始顺序、PID、原Caption、caption_index、paired_image_id及原图文件SHA；不zip配对、不筛去失败Query。
- Source 385批/385 Claim顺序完整、无重复/漏项；每批T1→image→T2记录完整，T2从原Matched Clean重新开始，T1监督绑定通过。
- 6145句最终发生词替换、11句NOOP，最终动作6145次；每条至多1个实际候选替换动作。NOOP全部保留，编辑率不能当作攻击成功率。
- 原pre_caption本身改变6137条原始Caption，这些是输入处理，不算攻击编辑。396849行桥全部WordPiece往返exact，新增IRRA77截断0；其中90677行为官方BERT30窗口截断，统计含候选/遮词，并非90677条Query。
- 三个Target绑定同一Source completed Manifest：54a9969c66f80cbf9c5f1a727eccbd6fa47f45f92ac6b3ae9a97bc8a40ebec7f；Adv图像和Adv文本的有序内容SHA序列跨Target exact，ModelSpace/缓存仍各自独立。
- 官方FP32投影上下界在全部提交单元中通过，最大实数L∞=0.007843166589736938，数学2/255=0.00784313725490196，舍入高出2.9334834977801094e-08。没有二次投影；**不能标注严格数学上界PASS**，这是沿用已批准的官方FP32语义，不是本轮新增异常。
- 本轮不加载原像素/Score Tensor重新计算预算或排名；上述数值引用已完成Source/CPU审计记录，并核对对应文件SHA和完整来源链。

## 7. 最终CPU审计及本轮只读核验结论

已有独立审计回执：/home/lzf/ldx/outputs/tbps-framework-sga-irra-source/full/audit/completed/audit.json，status=PASS、GPU=false、3074/6156、IRRA+BERT_MLM源、target_feedback=false；日志SGA_OFFICIAL_IRRA_CPU_AUDIT_PASS full。四个Source/Target执行报告均status=PASS且scope=full。

本轮标准库核验：
1. 4647份产物Manifest/完成回执及全部声明payload/source SHA一致，无pending或缺失Claim。
2. 21335份原来源文件在读取前后size/mtime/inode保持；完整保护索引留服务器tmp，Git只发布关键来源索引。
3. 48项summary数值与各原CSV两位小数字符串逐项一致；汇总CSV再读回核对，原始精度留JSON。
4. 三Target官方全量身份/顺序与同一冻结Adv输入绑定一致；原生模型Checkpoint SHA有效。
5. 既有独立审计PASS，仅只读核实，没有再次调用audit CLI、GPU/神经前向、攻击生成或完整排名。
6. 初次v1路径错误FAIL独立保留，不重写为成功；旧Smoke和额外SGA-TBPS White-box数据未用于本报告任何SGA指标。

可正式标记：**SGA官方算法（IRRA Source）三模型Full完成，独立CPU审计通过。** 该结论限定于本次任务移植协议，不表示原论文ALBEF数据表完全复现、严格等权限主方法比较或其他硬件/数据集验证。

## 8. 文件与证据入口

- 本报告：SGA_OFFICIAL_IRRA_FULL_REPORT.md
- 两位小数论文/WPS表格：SGA_OFFICIAL_IRRA_FULL_METRICS_2DP.csv
- 原始精度、模型SHA、运行报告、共享产物及完成证据：SGA_OFFICIAL_IRRA_FULL_EVIDENCE.json
- 正式结果根：/home/lzf/ldx/outputs/tbps-framework-sga-irra-source/full/
- 控制目录：/home/lzf/ldx/tmp/sga_official_irra_full_v2/
- 本轮只读核验日志：/home/lzf/ldx/tmp/sga_official_irra_full_readonly_extract_20261019.log
- 全来源保护索引：/home/lzf/ldx/tmp/sga_official_irra_full_result_protection_complete_20261019.json
- 主方法历史结果：../three_model_full_20261018/REPORT.md，实际原summary路径与审计SHA在本次证据JSON中。

本轮只提交派生报告、CSV、证据索引及统一状态文档；不提交模型、像素、特征、原日志或缓存。不自动开始VLPTAttack。
