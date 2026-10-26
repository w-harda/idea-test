# 本项目与R²TUA及全部对比方法：实验配置一致性独立审计

审计日期：2026-10-26；本项目分析基准a820278eff5ccd79ba4b90de833e77bdbab022d3。
本轮只做CPU读取、来源SHA核验、论文/源码研究和派生表格；未初始化CUDA，未调用模型、训练、攻击、完整排名或已PASS审计入口。
这是事实审计，不调整任何方法，不为匹配论文数值调参。

## 先回答三个问题

**1. 我们目前与R²TUA配置一样吗？不完全一样，且若干关键项尚不能确认。**
共同点是CUHK-PEDES官方Test意图、IRRA/RaSa/APTM模型家族、非目标检索评价，以及G Image-only和R²TUA Table1最终8/255。
已知不同包括：图像单输入G(I)与图像+adversarial prompt生成器、Joint是否改Query、Table3源池与元学习机制、16/255迁移预算，以及我们已固定的训练过程。
对方真实权重SHA、输入桥、实际Train图片/PID数量及各Baseline适配代码不公开，不能填成相同，也不能填成已知不同。

**2. 什么影响直接比较，什么只是实现/方法差异？**
关键实验控制量是受害者权重、完整测试身份及多正例指标、输入桥/Clean、攻击模态/预算、Source与Test配对/梯度权限、训练数据与暴露量。改变它们可能改变R@1/mAP，必须披露。
架构、Loss、硬/软投影、prompt条件、课程训练和单步/迭代成本是方法设计差异；它们可以被比较，不要求所有算法采用同一Loss。
但在攻击面或权限不同、评测来源未对齐时，跨论文分数不能证明某个设计独立造成差距。
日志位置、后台nohup方式、原子保存布局等管理细节，在不改变Tensor/顺序/数学时不构成数值协议差异；FP32、resize、Tokenizer、TopK、截断和排序绝不是纯管理细节。

**3. 优先控制哪三个变量，研究G的同模型效果？**
①先固定同一IRRA权重、Raw输入桥、完整Manifest/排名，复现同一Clean，排除评价侧差异。
②固定威胁模型：Image-only或Joint分别比较；RGB epsilon、Test配对/梯度/prompt权限一致。先将G60 Image-only与同权限生成式Image-only放在同组，不拿Joint对Image-only归因。
③固定实际训练数据与有效优化预算：2048图/1861PID、访问数/Adam数/源模型调用量，而非只比较“30/60 Epoch”。
这些控制完成后，再把排序对齐Loss、投影饱和或生成器容量作为独立研究变量；本轮不修改、不执行。

## 1. 真实模型身份与官方Test结果

G10/G60及H7/H10/H60五个文件实际重新计算SHA，并在CPU上读取完整Checkpoint的配置/Optimizer头部；没有构造网络或应用CUDA RNG。三个Victim大权重也逐字节重新hash。
| 身份 | 绝对Epoch/visits | 合法更新/Adam | SHA-256 |
| --- | --- | --- | --- |
| G10 | Epoch10 | 80680 | f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e |
| G60 | Epoch60 | 484080 | 9148a6ac06178b2eb474194ef6892c50182c5623b4e88bdcf88faa358aa46049 |
| H7 | visit14336 | 14259 | 487f3061b6cba2ddabccdd49745a7b99c8c549f4541781b8814b299ef5f279df |
| H10 | visit20480 | 20370 | 4ae964d51ea94840de71d9597d90a2234a36104e5d649cfee96577f59f94bacd |
| H60 | visit122880 | 122220 | c0de26658c5f36083aaeb04dde3948cdd5bf941d30b1b5c97f7bb2c796404094 |

H60从H10继续新增50轮，前10轮使用G10、后50轮固定G60奖励图库，不是从头使用G60训练60轮。
H参数并非每次访问均参与梯度，因此每参数Adam step不全相等；CPU读出的实际step集合保存在OUR_CONFIG，未统一改写。
当前G60+H60官方Test Full在已搜索范围内NOT_FOUND；仅有128内部Validation（Joint mAP42.874080657958984），不进入Test表。
本次增量搜索与2026-10-24较宽检索互补，已找到G60三模型正式Full，但其Joint明确绑定H7。
B框架在基准分支已完成CPU准备；本次pilot_v1、train_v1和统一评估输出均不存在，无可报告的B正式Full指标。
旧smoke_v2 GPU执行完成/严格exact审计FAIL属于历史机制Smoke，不能改签B效果实验。

下表是实际已有官方Test值；百分数两位，原精度在OUR_CONFIG。G60+H7特意单列为身份参考，不能标为H60。
| Target | 方法 | R@1 | R@5 | R@10 | mAP |
| --- | --- | --- | --- | --- | --- |
| IRRA | G10 Image-only | 37.44 | 60.77 | 70.08 | 32.28 |
| IRRA | G10+H7 Joint | 26.53 | 46.83 | 57.20 | 23.55 |
| IRRA | G60 Image-only | 21.05 | 37.88 | 45.52 | 17.51 |
| IRRA | G60+H7 Joint（身份参考，非H60） | 15.72 | 28.56 | 36.39 | 13.08 |
| RaSa | G10 Image-only | 42.33 | 61.71 | 69.56 | 35.05 |
| RaSa | G10+H7 Joint | 22.87 | 34.68 | 40.12 | 17.20 |
| RaSa | G60 Image-only | 33.12 | 50.49 | 58.46 | 26.47 |
| RaSa | G60+H7 Joint（身份参考，非H60） | 17.53 | 27.73 | 32.26 | 12.57 |
| APTM | G10 Image-only | 50.19 | 70.21 | 77.13 | 39.66 |
| APTM | G10+H7 Joint | 38.82 | 58.74 | 66.60 | 30.49 |
| APTM | G60 Image-only | 42.72 | 62.00 | 69.95 | 32.11 |
| APTM | G60+H7 Joint（身份参考，非H60） | 31.89 | 50.05 | 57.12 | 23.74 |

## 2. 我们的训练、攻击与评价协议

### G：训练监督丰富，推理只接图像
实际Train=2048图/1861PID、内部Validation=128图/121PID；本轮CPU实算Train/内部Validation/官方Test的PID交集均0；每图固定Caption index0。原始官方Train有34054图/11003PID，并不等于我们的优化数据量。
官方Test实际3074图/6156句/1000PID，官方Val3078图/6158句/1000PID；从reid_raw.json重新统计。
G为width32/4 residual blocks、1377411参数的ImageOnlyGenerator，不含FiLM。
每个训练样本按S_raw属性顺序做K次更新，输入上一轮detach图、相对原图投影；最终再从原图做一次G内部Joint更新。推理只调用一次generate_once，不进行K+1次Test优化。
训练每轮6020属性步+2048最终步=8068 Adam更新；K平均6020/2048=2.939453125，无cap4，实际0–7。6020是更新账本，不是6020个不同词/区域。
Natural-K使用正重要性三次幂归一化后的熵有效数再向上取整，beta1；离线S_raw来自配对软排名梯度及属性方向投影。H的重要性是策略网络softplus预测，含义不能混同。
Global为整句文本特征点积；Local为当前JVP点积；Retain为前一属性分数升高的ReLU，参考detach；权重均1。最终内部Joint用S_raw加权JVP，不等于G/H联合攻击。
没有新增TV、扰动正则、ITM、Push-Pull或Scheduler。Adam2e-4/(.5,.999)，batch1，seed42。
离线S_raw竞争图库含34054官方Train图（包括内部Val图）；因此不能宣称整个重要性构造gallery-disjoint。G优化器内没有显式PID负例InfoNCE；不能据此否认离线竞争信息。

### H：字符策略与SCST
UTF8 Byte Transformer宽64、2层4头、154594参数。Attribute/Position/Replacement三头仅使用Caption及caption-only Canonical。
Natural-K由detach的softplus分数决定；每属性和共享词至多1字符、Unicode合法候选/Tokenizer约束、允许STOP与Position NOOP，不能等同SGA/DRA的至多1词。
训练奖励使用配对相似度下降与best-positive减max-negative的margin下降，lambda_sim=lambda_rank=1；SCST采样减贪心优势detach，只更新H，Adam1e-4/(.9,.999)、batch1。
PID和正确图文配对在训练Reward中可见，Test H(T)不读图像/PID/源梯度；G(I)亦不读Test Caption。
G/H原始训练没有B的新语境双向刷新，不能把B开发状态归入既有G60/H60效果。

### 冻结评价器
统一物化RGB FP32，先PIL RGB/bilinear384×128；攻击后不再PIL/JPEG量化，8/255在归一化前相对原像素硬投影，端点向内舍入。
IRRA384×128/CLIP Byte-BPE77、CLIP mean/std、单位特征点积。
RaSa由物化Tensor bicubic antialias到384×384，WordPiece50，原生Top128后ITM分数；不额外添加全局分数。
APTM384×128、自身mean/std、WordPiece56，Top128 ITM+global及全Query原生归一化。
因此“原生Target评分”不等于“每个模型从原JPEG独立读入的原论文Clean”；共同物化图像桥必须保留说明。IRRA官方datasets/build.py本地也调用默认Resize，未证实其与我们发生bicubic/bilinear差异，不能捏造该原因。
全6156条Query/3074 Gallery、全部同PID正例、无camera过滤；R@k为前k任一正例，AP为全部正例rank precision均值，所有Query FP32求均值；torch.argsort默认非stable。无模型反馈调整G。

关键源码函数/行号/SHA见SOURCE_EVIDENCE source_code；路径指向本次读过的真实文件，非README推断。
## 3. R²TUA正文与补充材料逐页核查

正文10页、补充5页已读取；Table1/2在PDF第6页、Table3/4在第7页另作页面视觉核对。正文与补充SHA见来源索引。
- 正文p1–2：问题、动机及相关攻击；不能从摘要推断所有Baseline均同一威胁模型。
- p3：图像与描述误导目标的adversarial prompt进入ViT/BERT/TGMF重建器，残差生成扰动；“多模态条件”不等于同时改Query文本。
- p4：SCF及SSIM、图文对批交换prompt。训练用真实配对，但Test正确Caption是否必需、非目标prompt选择规则未完整报告。
- p5：PPLs推开真配对/拉近误导prompt；RaSa/APTM匹配CE含hard negative、Top128。设置为6层ViT/BERT/TGMF、eta.9、alpha1、30epochs、paired N16、AdamW1e-5，epsilon24/255每轮乘.9到8/255。paired N16不能直接当作每step总图像数16。
- p6：Table1三个TBPS数据集/三个模型的非目标与目标结果；IRRA仅使用PPLs，不能说IRRA使用其融合Matching Loss。Table2是Market1501图像ReID，不是CUHK文本Query。
- p7：Table3/4、Sec4.3。明确采用可见模型集合的元学习：RaSa目标的源池为APTM+MARS；IRRA目标为Ham+MLLM4Reid。MARS此处是TBPS模型（ref10），不是同名视频数据集。Table3未给APTM unseen-target。
- p8：效率/训练成本Table5等；SGA/VLPT/MFHA无离线训练，其他有离线成本；不能把本服务器成本与论文计时直接等同。
- p9–10：参考文献。MetaAttack[31]对应TPAMI扩展版，而非旧清单AAAI2021前身。
- 补充p1–2：SCF推导/玩具数值；p3：稳定性、隐式梯度/溢出保护讨论；p4：不同epsilon可视化/实际服务；p5：防御实验。未补出Baseline source路由、实际训练数和Victim SHA。

Table1可称正文中的模型可访问比较情境；但作者未公开各Baseline具体Source/是否每Target重新生成/文本输出组合/epsilon/步数，不能仅根据每行Target断定其原始适配详情。
Table3则明确是集合元学习迁移；不允许把表内VLPTAttack说成我们IRRA Source同一协议。其“对所有攻击使用元学习”也说明至少存在作者适配设置，具体是否改官方核心算法仍UNKNOWN。

## 4. 原始Baseline与R²TUA重新实现分离

| 方法 | 原始类型/是否训练G | 原始Test配对Caption/源梯度 | 原始图像预算/迭代 | 文本攻击 | R²TUA实际适配证据 |
| --- | --- | --- | --- | --- | --- |
| SGA | iterative image+text; G=False | True / True | 2/255; steps=10 | BERT MLM <=1 original word slot; candidates W10; T1 -> image -> T2 | NOT_REPORTED |
| VLPTAttack | iterative image+text DRA; G=False | True / True | 8/255; steps=10 | BERT MLM <=1 original word slot; 5 region candidates; triangular sampling; final image supervision .6/.2/.2 | Table3 all-method meta-learning adaptation stated; exact implementation NOT_REPORTED |
| MFHA | iterative image+text; feature heterogeneity / spatial-frequency augmentation / momentum; G=False | True / True | 2/255; steps=10 | BERT-Attack1 word; candidate list W10; 4 augmentations | NOT_REPORTED |
| AnyAttack | trained feature-conditioned decoder, targeted image generation; G=True | False / False | 16/255; steps=0 | none; target-image conditioning, not Query text rewriting | paper reports both targeted/untargeted and Table3 meta-learning; exact adaptation NOT_REPORTED |
| MisRanking | trained image generator + discriminator / mis-ranking / differentiable pixel sampling; G=True | False / False | UNKNOWN; steps=0 | none | image-only TI-ReID adaptation stated; loss/weight/code details NOT_REPORTED |
| MetaAttack | universal additive + functional color perturbation, virtual-guided meta-learning; G=False | False / False | official runAttackMar.sh MAX_EPS=8; joint color+delta bound meaning NOT_VERIFIED; steps=0 | none | paper calls UAP; whether additive-only or color+delta adaptation UNKNOWN |
| MTGA | meta-learned image generator/GAN, perturbation random erasing + Normalization Mix; G=True | False / False | 8/255; steps=0 | none | Table3 adapts set-based scheme; exact TBPS training code NOT_REPORTED |
| R2TUA | trained image+adversarial-prompt reconstructor/residual generator; G=True | UNKNOWN / False | Table1 8/255; Table3 16/255 except star8/255; steps=0 | adversarial prompt conditioning; no final query text-edit budget reported | author own method; supplementary SCF numerics do not supply adapted baseline code |

SGA/DRA本项目已有官方算法字节保持及真实Full证据；这支持“官方算法在IRRA上的适配Baseline”表述，不支持“已复现R²TUA同名行”。
MFHA声明代码本次仍404，仅以作者原文确认参数。MetaAttack正确仓库为WJJLL/Meta-Attack-Defense；AAAI21的FlyingRoastDuck仓库单独保留，不混用。
AnyAttack原始16/255、LAION400M预训练等不能直接填进R²TUA的AnyAttack行；同理不能把SGA原2/255当作R²TUA实际SGA预算。
没有找到R²TUA官方适配代码/配置；CVF页面无code入口，GitHub按R2TUA查询返回0仓库，仅是已检查公开范围的结果，不证明任何地方绝对未公开。
详尽场景及原始论文数值在COMPARISON.csv/COMPARISON.md和PAPER_TABLES.json。

## 5. Clean核对与数值差距

下面论文值均为Table1、CUHK-PEDES、各模型未受攻击；差为本项目Raw Clean减论文Clean，单位百分点。
| Target | 本项目R@1 | 论文R@1 | 差值pp | 本项目mAP | 论文mAP | 差值pp |
| --- | --- | --- | --- | --- | --- | --- |
| IRRA | 73.148148 | 73.38 | -0.231852 | 66.097443 | 66.13 | -0.032557 |
| RaSa | 76.218323 | 76.51 | -0.291677 | 69.184067 | 69.38 | -0.195933 |
| APTM | 74.610138 | 76.53 | -1.919862 | 64.126839 | 66.91 | -2.783161 |

IRRA差虽小，也不足以确认权重/输入完全相同；APTM差较大，提示应优先核对，但不能仅凭差值认定是哪一个预处理或Checkpoint造成。
本项目官方SGA/DRA Matched Clean另为IRRA69.41/63.01、RaSa74.45/67.45、APTM73.55/63.38（R@1/mAP），因为官方pre_caption30及BERT→IRRA接口；与Raw Clean有已知输入协议差异。不能拿论文或Raw Clean当作这些攻击自己的Clean来计算下降。

作为带标签的结果描述：本项目G60→IRRA Image-only Test R@1=21.05、mAP=17.51；R²TUA Table1 CUHK IRRA Untargeted/8/255报告0.58、1.07。
本项目G10→IRRA Image-only为37.44、32.28；G10+H7 Joint为26.53、23.55。它们分别是图像单模态及图文联合，不能合并为同一方法身份。
这些最终数值差是真实公开/已有结果差，不是统一评测下重新计算的因果比较。不能把Table3 16/255迁移分数或Targeted↑分数掺入上述比较。



### Table3预算与Source的直接证据

下列均为论文Table3、CUHK-PEDES、Untargeted；数值顺序R@1/mAP（百分数），不是本项目重新评测。

| Unseen Target | 可见Source集合 | R²TUA*，8/255 | R²TUA，16/255 |
| --- | --- | --- | --- |
| RaSa | APTM + MARS模型 | 38.79 / 30.63 | 5.70 / 5.25 |
| IRRA | Ham + MLLM4Reid | 31.74 / 24.33 | 4.17 / 3.57 |

因此不能只选Table1或Table3较低的一个数字，忽略白盒情境与unseen-target迁移的区别。原MTGA论文SecIV写8/255，而R²TUA Table3明确统一16/255（星号除外）；它们属于不同实验场景。

同名Baseline的差距也不自动证明我们做了算法优化：
本项目SGA/IRRA同源Joint为2.70/2.29，R²TUA Table1 CUHK/IRRA/Untargeted的SGA为15.55/12.25；
本项目VLPTAttack/IRRA同源Joint为0.00/0.10，论文对应VLPTAttack为7.52/6.78。
本项目有固定官方源码SHA，论文两行的Source、epsilon及作者适配代码未核实，不能反推差距来自某个优化。
本项目RaSa/APTM仅迁移，而Table1论文适配路由未知；不得按Target行直接宣称相同权限。

## 6. 逐项公平性结论

| 编号 | 场景 | 结论 | 解释 |
| --- | --- | --- | --- |
| A | G60→IRRA vs R²TUA Table1 IRRA untargeted | PARTIAL | 同为IRRA同源、Image-only、最终8/255；但对方权重/图像桥/实际Train规模、Test prompt构造不公开。不能把21.05/17.51与0.58/1.07解释为等条件纯算法差。算法架构/Loss不同本身不是不公平理由。 |
| B | G+H Joint vs R²TUA image-only untargeted | NOT_MATCHED | 我们修改Gallery与Query，R²TUA图文条件生成图像、未证明修改Query；Unicode预算与prompt权限不同。可做方法级结果展示，不能称等攻击面比较。Untargeted不是Image-only同义词。 |
| C | IRRA→RaSa/APTM vs R²TUA Table3 | NOT_MATCHED | 对方RaSa来源APTM+MARS，IRRA来源Ham+MLLM4Reid，均为集合元学习；没有APTM unseen-target结果。不是固定IRRA训练后迁移。 |
| D | 实际2048图/1861PID vs 对方训练规模 | UNKNOWN | 我们的实际规模已核实；R²TUA仅说采用官方划分，未给实际攻击训练图片/PID数。不能把34054官方Train总数当作对方实际用量。 |
| E | Clean差异的具体原因 | UNKNOWN | 数值不同已确认；原因可能涉及权重、输入桥、tokenization、排名等，但论文无SHA/代码，不能从Clean差异直接锁定某一原因。直接分数比较仅PARTIAL。 |
| F | 本项目8/255迁移 vs Table3未加星16/255 | NOT_MATCHED | 最终图像预算相差2倍。星号8/255只对齐epsilon，仍未对齐源集合/训练权限及输入协议。 |
| G | SGA/VLPT/MFHA与冻结G/H的测试权限 | PARTIAL | 原算法需Test图文配对和源梯度，我们无Test优化；R²TUA具体适配Source/文本输出协议未知，不能自动宣称它们使用我们相同权限。 |
| H | 本项目官方SGA/DRA移植 vs R²TUA同名表格行 | UNKNOWN | 本项目官方字节SHA和IRRA Source明确；对方作者适配diff/Source/预算等未公开。算法名相同不能证明实验身份相同。 |
| I | G10 vs G60 Image-only 本项目内部 | MATCHED | 同Raw输入、同Manifest、同三Victim权重/评分、同8/255与生成方式；预设轮数消融可比较。训练轮数是有意变化量。 |
| J | G60+H60官方Test vs 任一论文数值 | UNKNOWN | 未找到指定G60+H60官方Test Full；128内部Validation不能混入。 |
| K | Table2 Market1501 vs 本项目CUHK文本检索 | NOT_MATCHED | 数据集、查询模态、受害者及评价集合不同；不能抽取其R@1/mAP替代TBPS。 |
| L | Targeted↑ vs 我们Untargeted↓ | NOT_MATCHED | 成功语义及指标方向不同，不能比较数值大小判断攻击强弱。 |

## 7. 性能差距候选因素：事实与假设分层

| 候选因素 | 已核实事实 | 解释等级 |
| --- | --- | --- |
| 训练数据规模 | 我们2048/1861确认；对方实际攻击训练规模NOT_REPORTED | 可能影响性能，尚需实验；不能先写成34054vs2048的已证实差异 |
| Test单次G与迭代Baseline | G一次；原SGA/DRA/MFHA需多步源梯度/配对文本 | 已证实原算法权限/成本差异；R²TUA各适配细节未知；不能用此解释G与同为生成式R²TUA的全部差距 |
| Prompt条件 | G(I)不接文本，R²TUA图像+adversarial prompt | 已证实方法差异；影响大小尚需控制实验，不能称确定根因 |
| 负例/PID与Push-Pull | G点积/JVP；H有正负PID奖励；R²TUA有PPLs/配对批负例 | 已证实目标差异；不等于我们全流程没有负例；是否更对齐排序需实验 |
| Fusion Matching Loss | R²TUA RaSa/APTM有，IRRA无；我们的G只针对IRRA全局/属性 | 已证实算法/Source差异；不得将IRRA攻击差距归因其ITM Loss |
| 投影与epsilon课程 | 我们硬投影始终8/255；R²TUA SCF与24→8课程 | 已证实方法差异，可能影响梯度；未验证它是本项目性能根因 |
| G串行训练与单次推理 | Train K+1链式、末步原图；Test一次G | 已证实训练/推理机制；是否造成不足尚需消融，不能直接称Bug |
| Loss与实际排名 | G点积/JVP非完整R@1/mAP可微目标；H margin SCST也非直接mAP | 已证实数学差异；非目标函数不相同不自动等于实现错误 |
| 容量与轮数 | 我们1.377M、10/60轮；对方三套6层Transformer、30轮 | 架构层数/轮数已证实；精确容量与等计算量未核实，Epoch数不可单独比较 |
| Checkpoint与输入桥 | 我们8个权重SHA及输入桥确定；对方SHA/resize/token设置未公开 | 未知混杂因素，必须先对齐；Clean差异不能定位唯一原因 |

## 8. 最低成本的后续对照（本轮未执行）

先做CPU事实表与作者询证，询问Table1每Baseline源码/Source/epsilon/文本输出、完整权重、输入变换、Test图文对应及Table3元学习配置。
若获得对方攻击Tensor/文本和相同Victim权重，可直接复用攻击产物在一个统一原生评分器评测；只需编码/排名，不需重训。若缺模型来源，先复现统一Clean再比较。
现有G10/G60图像与H7文本可以保留；H60目前缺6156 Test文本，补齐需新的攻击推理/评估授权，不能只重写报告解决。
仅改变评分桥/同权重打分可能可复用RGB产物；若像素尺寸/预算定义变化，必须先核验预算与语义，不可重新缩放后仍宣称原攻击等价。
源模型或威胁模型、prompt访问、最终epsilon改变，通常需独立重生成；若要求针对不同Victim重新白盒训练，则训练型攻击需新训练，不能把固定IRRA迁移改名白盒。
严格等训练预算方案须公开Train身份并匹配访问/更新/源调用量，保留各方法原Loss；不要求一次重训全部Baseline。分阶段成本与停止条件见NEXT_EXPERIMENTS。
本轮没有任何新Validation、GPU、训练、攻击或评分。

## 9. 核验范围与局限

本轮重新hash221份此前原始结果/训练协议/日志引用、5份G/H及3份Victim权重，读取原始训练Manifest并统计PID/K和三类PID隔离；结果、Checkpoint身份与先前报告一致。
原始151378历史文件保护结论引用原训练/CPU审计回执，本轮没有重跑该全量审计，不能冒充本轮151378重算。
增量结果搜索范围、异常数、B输出存在状态完整存于SOURCE_EVIDENCE；与此前宽检索覆盖边界并列，不宣称扫描服务器所有外部磁盘。
论文作者实际攻击训练规模、Victim Checkpoint、完整适配算法、图像实际编辑范围及Test标签用法未公开处均保留UNKNOWN/NOT_REPORTED。
没有多种子统一重测，跨论文数值不能用于统计显著性或单因素因果推断。

全部对照可以作为公开方法结果的带协议标签展示。当前不能声称与R²TUA严格同配置复现，也不能仅凭差距断言我们对Baseline做了优化。


## 一手来源入口

- [R²TUA正式正文](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_R2TUA_Reconstruction-residual_Based_Targeted_and_Untargeted_Attack_Against_Text-Image_Person_CVPR_2026_paper.pdf)，PDFp3–8为方法/设置/表格。
- [R²TUA补充材料](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Wang_R2TUA_Reconstruction-residual_Based_CVPR_2026_supplemental.pdf)，PDFp1–5。
- [SGA官方固定源码](https://github.com/Zoky-2020/SGA/tree/f49e6926cd13148e49bdbc2c3e537660e096230c)。
- [DRA官方固定源码](https://github.com/SensenGao/VLPTransferAttack/tree/91f78325e7cf0c497c9bbb5be2215be652d8cd9d)。
- [MFHA作者原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11873190/)，Adversarial attack settings。
- [AnyAttack官方代码](https://github.com/jiamingzhang94/AnyAttack)，本轮固定Commit见PAPER_CONFIG及证据索引。
- [MisRanking官方代码](https://github.com/whj363636/Adversarial-attack-on-Person-ReID-With-Deep-Mis-Ranking)。
- [MetaAttack TPAMI作者稿](https://iris.unitn.it/bitstream/11572/377289/1/PAMI22-Towards_Robust_Person_Re-Identification_by_Defending_Against_Universal_Attackers.pdf)及[正确官方仓库](https://github.com/WJJLL/Meta-Attack-Defense)。
- [MTGA作者论文](https://arxiv.org/abs/2409.04208)及[作者代码](https://github.com/yuanbianGit/MTGA)。

完整字段及证据链接见[CSV矩阵](COMPARISON.csv)、[Markdown矩阵](COMPARISON.md)、[本项目配置](OUR_CONFIG.json)、[论文配置](PAPER_CONFIG.json)、[来源索引](SOURCE_EVIDENCE.json)、[可比性审计](FAIRNESS_AUDIT.md)、[后续最小方案](NEXT_EXPERIMENTS.md)。
