# 以引用R²TUA Baseline为目标的评测环境核验

日期2026-10-26；独立分支codex/r2tua-aligned-evaluation，基准c2062e430c664d626c07782490128fc3316dfa61。

## 核心结论

**可以不重新复现全部算法，直接引用R²TUA Table1的已发表数值作为跨论文文献参考。** 应标记† author-reported、引用数值真正来源、保留本项目自己的Clean，并把目标攻击、攻击面和已知/未知权限差异写清楚。这个结论不要求所有方法采用相同架构或Loss。

现有证据尚不足以将这些数值称为“已核实统一8/255、同Checkpoint、同Source和同一评测代码的受控公平比较”。这不等于禁止引用，也不等于必须重跑全部算法。IRRA/CUHK/Untargeted是最接近且成本最低的起点；RaSa/APTM还要注明本项目为IRRA源迁移、对方Source未确认，并处理已确认的Clean输入路径差异。

## 一、八项环境核查

| 问题 | 已核实内容 | 对论文实际运行仍未确认 |
| --- | --- | --- |
| Test划分 | 作者称官方split；本地官方annotation实际3074 Gallery/6156 Query/1000PID | 作者成员Manifest、排序及具体PID列表 |
| IRRA版本/权重 | 实际加载副本11文件与固定官方Git对象逐字节相同；唯一权重SHA重新核验 | 作者具体Commit与Checkpoint SHA |
| 图像输入 | 官方IRRA 384×128/PIL bilinear/CLIP归一化，本地3固定样本像素一致 | 作者库版本和任何附加输入桥 |
| 文本输入 | 官方Byte-BPE77；本项目6156条Raw Caption逐条Token ID一致，15条截断 | 作者是否额外清洗、Tokenizer版本 |
| 评分 | 官方单位化图文点积、全部同PID正例、完整Text→Image R@K/mAP；本项目定义一致 | 作者实际评分脚本、数值精度与排序细节 |
| 攻击位置 | R²TUA描述输出扰动图像；本项目G仅Gallery、Joint还修改Query | 其他Baseline在R²TUA适配中是否实际也输出Adv Query |
| 8/255空间 | 作者自身名义最终8/255明确；本项目归一化前FP32 RGB硬投影 | 作者具体张量空间、量化与其他Baseline逐行epsilon |
| 所有Baseline是否统一 | 同表共同Clean和指标已确认 | 没有公开代码证明统一预处理/预算/Source/评分程序 |

本轮既未发现可归属作者的R²TUA公共实现/配置，也不把“未检索到”写成“作者一定未公开”。精确范围见[未核实设置](UNVERIFIED_SETTINGS.md)。
Source未知的Baseline保持UNKNOWN，不能因为Target列写IRRA就推断其Source统一IRRA。

## 二、Clean差异及已确认的具体接口差异

| Target | Ours R@1 | Paper R@1 | 差值pp | Ours mAP | Paper mAP | 差值pp |
| --- | --- | --- | --- | --- | --- | --- |
| IRRA | 73.148148 | 73.38 | -0.231852 | 66.097443 | 66.13 | -0.032557 |
| RaSa | 76.218323 | 76.51 | -0.291677 | 69.184067 | 69.38 | -0.195933 |
| APTM | 74.610138 | 76.53 | -1.919862 | 64.126839 | 66.91 | -2.783161 |

以上差值为百分点。本轮没有重新执行Clean、模型编码、排名或独立实验审计。

1. **IRRA的已确认差异是数值加载路径。** 官方构建调用convert_weights，适用层采用混合参数精度；本项目显式model.float并采用FP32/math/no_tf32。两者返回的检索特征都可以是FP32，不能因此误以为编码数学相同。R²TUA的实际dtype未公开，所以这只是相对官方参考实现的差异，尚非Clean差距的实验证实原因。
2. **RaSa/APTM存在已确认的Clean图像输入链差异。** 本项目共用物化桥先将原图bilinear到384×128，RaSa再Tensor bicubic到384²；APTM在共同桥上接收384×128。官方两者分别从原PIL直接bicubic到最终尺寸。固定前3图的RGB max_abs：RaSa约0.05635–0.11402；APTM约0.04706–0.05882。这是Clean路径比较，不是攻击扰动预算。未测试其对R@1/mAP的贡献，不能把APTM全部差距归因于Resize。
3. **IRRA文本/像素抽查没有发现本轮对应差异。** 全6156条Raw Token ID一致，前3图的IRRA像素差0。该结论不外推到全体神经特征、权重等价或作者实际环境。
4. 论文Clean与三个模型原始公开参考数字对应；这不证明R²TUA直接复制数字或未跑实验。Clean接近也不能证明攻击侧权限相同。

逐文件SHA、函数和行锚点见[SOURCE_EVIDENCE.json](SOURCE_EVIDENCE.json)、[IRRA官方对象检查](IRRA_OFFICIAL_SOURCE_CHECK.json)、[当前/原生处理源码索引](NATIVE_SOURCE_INDEX.json)。

## 三、论文已发表Baseline清单

下表为第一优先级CUHK/IRRA；完整Table1含三数据集、三模型的99条记录在[CSV](BASELINE_TABLE.csv)和[Markdown](BASELINE_TABLE.md)。
类别1=可直接作近似同协议跨论文对比；2=可以引用但说明差异；3=不适合当前同一公平比较表（仍可作为背景文献引用）。

| Method | 类型 | 实际模态 | epsilon | R@1 | R@5 | R@10 | mAP | G引用类 | Joint引用类 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Clean (None*) | Clean | None | N/A | 73.38 | 89.93 | 93.71 | 66.13 | 2 | 2 |
| SGA | Untargeted | NOT_REPORTED | NOT_REPORTED | 15.55 | 35.40 | 46.22 | 12.25 | 2 | 2 |
| VLPTAttack | Untargeted | NOT_REPORTED | NOT_REPORTED | 7.52 | 20.78 | 28.62 | 6.78 | 2 | 2 |
| MFHA | Untargeted | NOT_REPORTED | NOT_REPORTED | 20.53 | 38.76 | 48.16 | 15.69 | 2 | 2 |
| AnyAttack | Untargeted | NOT_REPORTED | NOT_REPORTED | 33.56 | 55.18 | 63.39 | 26.26 | 2 | 2 |
| MisRanking | Untargeted | Image-only | NOT_REPORTED | 15.62 | 25.68 | 30.23 | 11.60 | 2 | 3 |
| MetaAttack | Untargeted | Image-only | NOT_REPORTED | 8.79 | 17.95 | 24.14 | 7.51 | 2 | 3 |
| MTGA | Untargeted | Image-only | NOT_REPORTED | 7.49 | 13.66 | 16.93 | 5.44 | 2 | 3 |
| R²TUA | Untargeted | Prompt-conditioned Image-only | 8/255 (R²TUA自身设置，正文p4-5) | 0.58 | 2.71 | 4.56 | 1.07 | 2 | 3 |
| AnyAttack (Targeted) | Targeted | NOT_REPORTED | NOT_REPORTED | 12.39 | 33.34 | 46.39 | 9.00 | 3 | 3 |
| R²TUA (Targeted) | Targeted | Prompt-conditioned Image-only | 8/255 (R²TUA自身设置，正文p4-5) | 84.10 | 98.00 | 99.10 | 52.16 | 3 | 3 |

本轮没有一行攻击结果能无条件升级为类别1：不是因为模型架构或Loss不同，而是Table1缺少各Baseline实际epsilon/Source/适配权限以及作者运行身份。针对G Image-only，当前CUHK的Untargeted行均可按类别2放入**明确标注为文献参考的分组表**；表内不得统一宣称所有方法8/255。
已确认Image-only的方法与我们的Joint不是同攻击面，不能给Joint一栏贴“同威胁模型公平比较”。模态未报告的行保持未报告，不擅自假定它们是Joint。
Targeted指标方向为攻击目标成功↑，不可和本项目Untargeted下降↓争优劣；ICFG/RSTP属于另外的数据集，不混CUHK。

## 四、后续最低成本路线（仅方案，本轮不执行）

1. 保留现有FP32 Raw结果及其Clean不变；立即可以采用本报告的†文献引用和脚注，不需要实现所有Baseline。
2. 若进一步接近公开官方IRRA参考，实现并预注册独立PUBLIC_IRRA_REFERENCE评测Profile，固定一个官方权重/全部输入与指标，保留官方convert_weights语义，先做一次Clean核验。它尚未被当前Loader支持，不能只改表头就宣称完成对齐。
3. 现有G固定Adv像素在身份/预算匹配时可复用，只需未来按新Profile编码评分；不必重训G、不必重跑所有攻击。不得根据接近论文Clean与否挑Checkpoint/精度/样本。
4. 对RaSa/APTM应先单独登记原图直接bicubic参考与历史共同物化桥的协议差异。改输入桥会改变攻击施加/评分的空间，不能覆盖历史结果或悄悄混用。
5. 如果论文要求“同环境、等预算、等权限的严格优劣”，优先向作者获取各Baseline运行配置及权重SHA；无法获得时保留文献参考的结论范围，而不是承诺凭公开数字完成严格配对实验。

明确标准见[R2TUA_ALIGNED_EVALUATION_PROTOCOL.md](R2TUA_ALIGNED_EVALUATION_PROTOCOL.md)。
可直接复用的中文/英文表注见[CITATION_GUIDE.md](CITATION_GUIDE.md)。

## 五、本轮验证和保护

- 20项CPU数据/文档一致性检查PASS；不是神经模型回归测试，也不是旧Full审计重跑。
- Table1共396个已发表指标逐值复核，99条CSV均保留原两位出版精度。
- 6156条Tokenizer及3张固定Clean像素测试；CUDA初始化、模型加载、神经前向、训练、攻击生成、真实排名均0。
- 三个Victim Checkpoint和50份指定已读历史来源SHA/stat保持；未声称本轮重扫151378历史文件。

只新增派生文档、CSV和轻量证据，并更新项目状态索引；模型加载器、运行配置、Checkpoint、原metrics/summary/Manifest和已完成实验未修改。
新的评测Profile仍是设计方案，不启动任何GPU或Baseline复现。

## 一手来源

- [R²TUA正文](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_R2TUA_Reconstruction-residual_Based_Targeted_and_Untargeted_Attack_Against_Text-Image_Person_CVPR_2026_paper.pdf)：PDF p4–5自身约束/设置，p6（印刷22862）Table1；p7/Table3另属不同迁移协议。
- [补充材料](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Wang_R2TUA_Reconstruction-residual_Based_CVPR_2026_supplemental.pdf)：SCF及可视化讨论；未补齐逐Baseline实际运行身份。
- [IRRA官方固定Commit](https://github.com/anosorae/IRRA/tree/cfc23785545ead82dee2749fa0fc0ad948c96430)：读取器、预处理、Tokenizer、build_model与指标。
- [RaSa官方仓库](https://github.com/Flame-Chasers/RaSa)、[APTM官方仓库](https://github.com/Shuyu-XJTU/APTM)、[APTM作者原论文](https://www.zdzheng.xyz/files/MM23_Yang.pdf)：原生参考设置与Clean。
