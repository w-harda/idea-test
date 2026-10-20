# VLPTAttack（DRA）官方算法／IRRA Source正式Full结果报告

整理日期：2026-10-20。仅使用v2真实Full，三Target及独立CPU审计均PASS。
本轮只读读取、来源/身份/CSV核对与派生报告，不运行GPU、攻击、模型前向、完整排名或审计程序。
统计数值单位均为百分数，论文表保留两位小数；百分点差由原始精度相减后再取两位。
本报告不使用Smoke或历史SGA-TBPS White-box数据，不计算DR，不添加其他检索指标。

## 1. 论文主表

固定顺序IRRA→RaSa→APTM；每个模型内Clean（Matched Clean）→Text-only→Image-only→Joint。

| Target Model | Attack Method | R@1 | R@5 | R@10 | mAP |
|---|---|---:|---:|---:|---:|
| IRRA | Clean (Matched Clean) | 69.41 | 87.23 | 92.02 | 63.01 |
| IRRA | VLPTAttack Text-only | 43.40 | 66.05 | 76.32 | 41.67 |
| IRRA | VLPTAttack Image-only | 0.02 | 0.06 | 0.10 | 0.10 |
| IRRA | VLPTAttack Joint | 0.00 | 0.08 | 0.15 | 0.10 |
| RaSa | Clean (Matched Clean) | 74.45 | 89.23 | 93.26 | 67.45 |
| RaSa | VLPTAttack Text-only | 52.37 | 72.61 | 80.25 | 48.50 |
| RaSa | VLPTAttack Image-only | 29.13 | 48.39 | 57.39 | 22.77 |
| RaSa | VLPTAttack Joint | 16.34 | 31.12 | 39.33 | 13.03 |
| APTM | Clean (Matched Clean) | 73.55 | 88.82 | 92.84 | 63.38 |
| APTM | VLPTAttack Text-only | 52.86 | 73.80 | 81.06 | 46.74 |
| APTM | VLPTAttack Image-only | 40.04 | 60.95 | 69.17 | 31.80 |
| APTM | VLPTAttack Joint | 24.42 | 42.66 | 51.66 | 19.85 |

VLPTAttack Joint是该Baseline主要结果。两种单模态行是同一份Joint攻击产物的评估组合，
并非额外重新优化的image-only/text-only攻击：Adv图像受中间T1文本指导，最终T2文本受图像三组监督。
Image-only用Adv图＋Matched Clean文本，Text-only用Clean图＋最终T2文本，全部Query均保留。

## 2. 效果分析

- IRRA为唯一Source及同源Target。Image-only已把检索压到很低；
  Joint R@1为0.00、mAP为0.10，不表示mAP严格为零。
  Joint与Image-only的mAP显示均为0.10，但原始精度下Joint略高，差异小于0.01个百分点；
  R@5/R@10也略高，因此不能写成“Joint所有指标均优于Image-only”。
- RaSa为跨模型黑盒迁移Target。Joint为R@1=16.34、mAP=13.03，
  四项指标均低于两个单模态组合；相对Image-only，mAP进一步降低9.74个百分点。
- APTM为跨模型黑盒迁移Target。Joint为R@1=24.42、mAP=19.85，
  四项指标均低于两个单模态组合；相对Image-only，mAP进一步降低11.95个百分点。
- 相对各自Matched Clean，Joint的mAP绝对降低：IRRA62.91、RaSa54.42、APTM43.53个百分点。
  IRRA是可访问梯度的源模型，不能称其为跨模型黑盒迁移；RaSa/APTM未参与生成或反馈。
- 三个模型上Image-only都比Text-only产生更低的最终R@1和mAP。
  当前DRA在RaSa上的迁移最终指标低于APTM，但不能仅凭这一实验推导所有攻击下的模型鲁棒性排序。

## 3. 与历史主方法及官方SGA的描述性比较

历史结果重新从各自正式Full summary读取，与既有结果证据SHA、完成链和CPU审计日志核对。
各模型Checkpoint、Tokenizer、图像预处理与原生评分身份一致；无新评分。
主方法固定为G-v2 + H-v1 Joint，SGA只用官方算法＋IRRA Source结果。

| Target Model | Attack Method | R@1 | R@5 | R@10 | mAP |
|---|---|---:|---:|---:|---:|
| IRRA | G-v2 + H-v1 Joint | 26.53 | 46.83 | 57.20 | 23.55 |
| IRRA | SGA Joint | 2.70 | 7.57 | 11.65 | 2.29 |
| IRRA | VLPTAttack Joint | 0.00 | 0.08 | 0.15 | 0.10 |
| RaSa | G-v2 + H-v1 Joint | 22.87 | 34.68 | 40.12 | 17.20 |
| RaSa | SGA Joint | 39.13 | 60.38 | 69.48 | 35.64 |
| RaSa | VLPTAttack Joint | 16.34 | 31.12 | 39.33 | 13.03 |
| APTM | G-v2 + H-v1 Joint | 38.82 | 58.74 | 66.60 | 30.49 |
| APTM | SGA Joint | 42.59 | 63.78 | 72.66 | 37.59 |
| APTM | VLPTAttack Joint | 24.42 | 42.66 | 51.66 | 19.85 |

VLPTAttack Joint的最终R@1和mAP在本次三个模型上均低于这两个已完成方法，差值如下；
这是各自原生协议下的结果对照，不能单独归因为某个模块的贡献。

| Target | 比SGA Joint低的R@1百分点 | 比SGA Joint低的mAP百分点 | 比历史G/H Joint低的R@1百分点 | 比历史G/H Joint低的mAP百分点 |
|---|---:|---:|---:|---:|
| IRRA | 2.70 | 2.19 | 26.53 | 23.45 |
| RaSa | 22.79 | 22.60 | 6.53 | 4.17 |
| APTM | 18.18 | 17.74 | 14.41 | 10.64 |

### 3.1 Clean与攻击协议差异

本次DRA和官方SGA使用相同Matched Manifest；三个Target的Matched Clean四项指标在原始精度下逐值一致。
历史G/H使用Raw Clean，数值单独列出，不将Raw Clean代替本次Clean或用来计算DRA攻击下降率。

| Target | DRA/SGA Matched Clean R@1 | Matched Clean mAP | 历史Raw Clean R@1 | Raw Clean mAP |
|---|---:|---:|---:|---:|
| IRRA | 69.41 | 63.01 | 73.15 | 66.10 |
| RaSa | 74.45 | 67.45 | 76.22 | 69.18 |
| APTM | 73.55 | 63.38 | 74.61 | 64.13 |

| 维度 | VLPTAttack（DRA） | 官方SGA＋IRRA Source | G-v2 + H-v1 Joint |
|---|---|---|---|
| 图像预算 | 官方FP32 8/255 | 官方FP32 2/255 | 8/255 |
| 图像优化 | 10轮、2/255步长、每轮5区域候选（首轮不搜索）、原整数三角采样与五尺度 | 10轮、0.5/255步长、官方五尺度 | 冻结G单次图像前向，无Test迭代优化 |
| 文本攻击 | BERT MLM、KL重要性、最终至多1个原始词槽替换；完整T1→image→T2 | 官方MLM词替换与T1→image→T2，最终至多1个词槽替换 | 学习的Unicode字符策略，Dynamic-K、每合法属性词最多1字符 |
| Source文本上下文 | 官方攻击BERT max_length=77，cls=False；IRRA Byte-BPE77 | 已冻结SGA攻击BERT max_length=30；IRRA Byte-BPE77 | H的Caption-only及冻结IRRA训练侧Tokenizer约束 |
| Clean输入桥 | 原PIL RGB bilinear384×128与pre_caption(...,30)保存输入 | 同一Matched输入桥，实际Clean四指标完全一致 | 历史generation_v3统一物化Raw输入协议 |
| Test信息权限 | 原图及其全部正确配对Caption用于IRRA源攻击 | 同样可见源图文配对并使用IRRA梯度 | 冻结G只接图，H只接Caption及caption-only属性，不用Test正确配对/PID/受害者反馈 |
| Source/Target | IRRA生成一次；RaSa/APTM只进行最终原生评分 | IRRA生成一次；RaSa/APTM只进行最终原生评分 | IRRA训练所得冻结攻击器；RaSa/APTM消费同一冻结物化产物 |
| 测试计算成本 | 385批迭代图像梯度、区域搜索与两轮文本候选评估 | 385批官方迭代图文攻击 | 冻结生成器/策略推理，不在Test重新训练或优化 |

可将三者作为方法级Baseline结果展示，但应保留上述协议说明；
DRA与SGA的图像预算、Source文本窗口及计算过程不同，DRA与主方法的Test访问权限、文本扰动语义和成本不同。
本结果不证明严格等预算、等权限、等成本的单因素优势，也不因这些差异否定各方法原生配置结果的比较意义。
不为对齐协议修改任何已生成产物。

## 4. 官方算法、Source和数据协议

- 官方仓库：SensenGao/VLPTransferAttack。
- 固定Commit：91f78325e7cf0c497c9bbb5be2215be652d8cd9d。
- RAttacker.py SHA-256：
  75518cb9465d9f8e994ea2f59c86b89fd4c39e3fe74f66b7a1d0a090eec2d313。
- 实际执行代码：34097c338997561fd7556e43c472d2e5625c05f0；
  原启动登记Commit：c97c2bf6c5bcbf0a0d70560fbc54d339adb577e6。
- 直接调用原RAttacker.Attacker.attack()；官方源码保持原字节，未复制攻击算法或改Loss、区域采样、
  梯度更新/归一化、候选选择、精度、原FP32投影。
- Source=官方IRRA，BERT MLM冻结；eval，无Optimizer或参数更新。
  官方候选评分需要Source参数梯度，因此运行中允许requires_grad=True，
  不应误写成整个Source过程参数均requires_grad=False；结束后梯度清空，IRRA/MLM参数及buffers前后exact。
- Batch8，385批（384×8＋最后2图），全任务只设置一次seed42；每批全部原图配对Caption，
  包括cuhk:38920:0的10 Caption，不扩展成同PID其他图像Caption。
- 官方8/255、10轮、2/255、5候选、原三角区域整数采样、原图加0.5/0.75/1.25/1.5尺度，
  BERT MLM、最多1词槽及最终0.6/0.2/0.2三组图像监督全部保留。
- 原始词槽允许官方候选展开为多WordPiece短语，未按空格词数新增过滤。
- 同PID全部图库正例、全部Query、无camera过滤，复用原EvaluationEngine与原生Adapter。

| Target | 原生图像尺寸H×W | 原生Tokenizer | 原生评分 |
|---|---|---|---|
| IRRA | 384×128 | Byte-BPE，77 tokens | L2特征点积/余弦 |
| RaSa | 384×384 | BERT WordPiece，50 tokens | 全局召回Top-128，再ITM融合分数 |
| APTM | 384×128 | BERT WordPiece，56 tokens | Top-128 ITM与全局分数，原生完整Query集合归一化 |

不同Target的表示和融合评分缓存各自隔离，未将IRRA embedding当作RaSa/APTM特征。
Source源输入桥、Target原生处理和Matched Clean协议是三个独立层次。

### Checkpoint来源

| 模型 | Checkpoint SHA-256 |
|---|---|
| IRRA | 9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009 |
| RaSa | bc85da09c2991d5de503c2c2ac4f032e20cc536fec5094f0acff45e93d0104d2 |
| APTM | 521c9bf784e7e821ade471deaafa2f25e2501950591e574dc60ca177e9be5a2c |
| G-v2 E10（历史比较） | f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e |
| H-v1 E7（历史比较） | 487f3061b6cba2ddabccdd49745a7b99c8c549f4541781b8814b299ef5f279df |

BERT MLM及其余完整文件来源见v2配置和本目录证据JSON。
G/H权重只作为历史比较来源，本次DRA执行G/H加载与前向为0。

## 5. 全量覆盖、文本与预算记录

- 官方annotation逐成员核对3074 Gallery /6156 Query，顺序、PID、原Caption、caption_index、
  paired_image_id完整，无遗漏、重复或错误PID绑定；不是把6156 Query当作6156张独立图像。
- Source完成385批/385 Claim，各批保存RGB/effective_delta、原文/攻击文、T1和完整动作；
  所有批次记录text1/image/text2，原始词槽动作与最终字符串一致。
- 最终6153句有编辑、3句NOOP，全部保留；6153次最终词槽编辑。编辑数量不是攻击成功率。
- pre_caption自身改变6137条原始Caption，属于Matched输入处理，不算攻击编辑。
- Source391818个文本样本/桥接记录，官方BERT截断记录0；
  该计数包含候选与遮词，不能当成独立Query数。既有CPU审计逐桥检查通过。
- Source攻击385次，区域采样17325次，图像backward38500次；
  原图像样本前向439582，文本391818；MLM调用2221次/38944个样本。
  全局RNG元数据链连续，实际无恢复重跑，不宣称任意中断可无损恢复。
- 每个Target只加载一次，6148图像样本、12312文本样本（Clean/Adv各一遍）；
  RaSa/APTM各3151872融合pair。Target backward/Optimizer/重新Source生成均0。
- 三Target绑定同一Source完成Manifest：
  a5978dc834b14a34d4c6932e0c2d463fb052105a551c5bd9137d647d3639df18。
  四组有序输入内容SHA跨Target逐项相同，官方顺序和完整数据Manifest一致，
  Target特征和Score则保留各自模型空间，不跨模型误用缓存。
- Source、MLM、三个Target参数/buffers前后摘要均exact；Target保持eval/无参数梯度。
- 全部提交批次记录官方FP32投影边界通过，最大实数L∞=0.03137257695198059，
  高于数学8/255约2.793237274811844e-8；未添加二次投影，
  不标注严格数学上界PASS。该口径与原Smoke/正式启动授权一致。
- 无pending，所有385个图像批次与6156文本身份均进入Source完成集合。
  本轮不加载像素Tensor重算预算，预算/有限性引用原Source及已完成审计记录。

## 6. 完整耗时与资源

开始：2026-10-19 16:38:15 +08:00；结束：2026-10-20 09:51:44 +08:00。
**整链17小时13分29秒，62009秒**，含两环境CPU预检、全部Source/Target加载及评分、最终CPU审计与外围开销。
Source各批纯攻击段合计43523.58300687745秒，不能将其当作三模型全过程耗时。

| 阶段 | 开始（北京时间） | 结束（北京时间） | 墙钟耗时 |
|---|---|---|---|
| CPU_PREFLIGHT (Target) | 2026-10-19T16:38:15+08:00 | 2026-10-19T16:38:26+08:00 | 0:00:11 |
| CPU_PREFLIGHT (Source) | 2026-10-19T16:38:26+08:00 | 2026-10-19T16:38:31+08:00 | 0:00:05 |
| RUNNING_SOURCE | 2026-10-19T16:38:32+08:00 | 2026-10-20T04:47:18+08:00 | 12:08:46 |
| RUNNING_IRRA | 2026-10-20T04:47:18+08:00 | 2026-10-20T04:52:03+08:00 | 0:04:45 |
| RUNNING_RASA | 2026-10-20T04:52:03+08:00 | 2026-10-20T07:32:40+08:00 | 2:40:37 |
| RUNNING_APTM | 2026-10-20T07:32:40+08:00 | 2026-10-20T09:48:20+08:00 | 2:15:40 |
| RUNNING_CPU_AUDIT | 2026-10-20T09:48:20+08:00 | 2026-10-20T09:51:44+08:00 | 0:03:24 |

| 会话 | 执行报告耗时（秒） | GPU allocated峰值（GiB） | GPU reserved峰值（GiB） | CPU Peak RSS（GiB） |
|---|---:|---:|---:|---:|
| IRRA Source + BERT MLM | 43717.21 | 7.24 | 9.43 | 2.28 |
| IRRA Target | 272.37 | 0.90 | 0.98 | 2.23 |
| RaSa Target | 9621.93 | 1.86 | 2.03 | 25.82 |
| APTM Target | 8112.69 | 0.97 | 1.05 | 6.59 |

峰值按各独立进程报告，不能相加视为并发占用；也不是整台服务器所有进程的内存。
Source约12小时8分，RaSa约2小时40分，APTM约2小时15分，完整运行快于原先18–22小时估计。
本轮没有为提速更改算法、Batch、模型、精度或预算。

## 7. 最终审计与本轮只读核对范围

控制目录phase=SUCCESS，Source与三个Target执行报告均PASS。
原独立CPU审计status=PASS、GPU=false、3074/6156/385、target_feedback=false，
391818条桥接记录；原audit.log包含三个DRA_FULL_CPU_TARGET_AUDIT_PASS及
DRA_INDEPENDENT_CPU_AUDIT_PASS full。
最终标志：
SUCCESS | VLPTATTACK_OFFICIAL_IRRA_SOURCE_THREE_TARGET_FULL_AND_CPU_AUDIT。

本轮只读核对：
1. 原精度metrics.csv、summary、metrics_2dp.csv共48个数值一致；生成的CSV再逐格读回核对。
2. 审计完成回执按SHA绑定Source与三Target的同一completed Manifest；
   这些Manifest绑定本次CSV、summary和Score，未混入Smoke、v1 FAIL或旧白盒结果。
   audit.json本身不重复存放指标，本轮依据其完成链确认其核验的是当前结果。
3. 385批元数据/Claim、原annotation完整身份、原始/最终文本、词槽动作和RNG摘要连续性；
   三Target数据与四组内容SHA一致。
4. 当前官方核心文件、运行来源、Checkpoint和关键结果/元数据SHA一致。
5. 原文件stat与使用过的元数据SHA受保护；没有重新反序列化Tensor、
   重新编码/攻击/计算完整分数或排名，也未再次调用任何audit入口。

**可正式标记：VLPTAttack（官方DRA算法、IRRA Source、CUHK-PEDES）三Target Full完成，独立CPU审计通过。**
该结论是本固定移植协议的实际结果，不冒充原论文原始Source/数据集完全复现。
v1预检环境隔离FAIL及18份证据永久保留，v2新身份成功不回填v1。

## 8. 结果文件与证据路径

- VLPTATTACK_OFFICIAL_IRRA_FULL_REPORT.md：本报告。
- VLPTATTACK_OFFICIAL_IRRA_FULL_METRICS_2DP.csv：论文/WPS主表。
- VLPTATTACK_OFFICIAL_IRRA_FULL_METRICS_RAW.csv：源CSV原始精度字符串。
- JOINT_METHOD_COMPARISON_2DP.csv：三个Joint方法的描述性结果对照。
- VLPTATTACK_OFFICIAL_IRRA_FULL_EVIDENCE.json：数据、SHA、配置、Checkpoint、运行、审计及历史比较来源。
- 正式根：/home/lzf/ldx/outputs/tbps-framework-vlpt-official-irra-v2/full/
- 控制根：/home/lzf/ldx/tmp/vlpt_official_irra_full_v2/
- 既有审计：full/audit/completed/audit.json与full/audit.log。
- 本轮只读提取：/home/lzf/ldx/tmp/vlpt_full_results_20261020/read_results.log。
- 本轮原来源保护索引保留服务器tmp，不提交Tensor、模型、缓存、日志或原产物副本。

本轮仅提交派生报告、CSV、必要来源索引和统一状态更新，不修改实验代码及任何正式实验产物。
