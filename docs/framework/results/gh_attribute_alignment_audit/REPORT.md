# G-v2/H-v1属性选择、动态K预算与双模态交互机制审计

日期2026-10-22；分析基准23fd02daebdcd28bc2e77eb18de1e6fd04432518。
独立CPU只读分析与研究方案，未改变G/H、K、Reward、Selection或历史产物。

## 核心结论

| Validation候选 | 平均K | 平均选中 | 平均编辑 | Position NOOP槽 | STOP未用槽 | G选择/H编辑Jaccard | Top1一致* | 同集Spearman** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H9 | 2.171875 | 2.171875 | 2.171875 | 0 | 0 | 0.247638 | 13/127 | -0.506345 |
| H20 | 2.234375 | 2.234375 | 1.898438 | 43 | 0 | 0.211099 | 10/127 | -0.672066 |
| H60 | 2.179688 | 2.156250 | 1.859375 | 38 | 3 | 0.259449 | 10/127 | -0.502401 |


*Top1分母127个两侧非空样本，可含不同候选集合；不能冒充完整排名一致性。
**Spearman只比较严格同候选集、共同合法子集，至少2属性、并列平均秩；Validation有效105条。
Jaccard分母127，排除两侧都空的1条，不伪造空集Jaccard=1。

| 核心问题 | 审计结论 |
| --- | --- |
| G/H平均选择数？ | G Train K2.939453、中位3；H9/20/60 Validation K2.171875/2.234375/2.179688，编辑2.171875/1.898438/1.859375 |
| K差异来源？ | 公式同为gamma3/beta1；同候选分数分布是主要代数差异，STOP/NOOP只解释K之后编辑减少 |
| 最重要属性重合？ | 全127非空样本13/10/10；严格同集105条仅7/4/4，平均Spearman为负 |
| 约2次编辑是Bug？ | 已有K/动作/回放账本成立，未发现代码计数错误；兼有自然K、位置NOOP和少量STOP |
| 是否修改NOOP耗槽？ | 可以受控研究，无收益证明；属于算法/动作语义变化，不能当纯修复 |
| 是否属性错位？ | 排序/集合明显不同，但H60重合更高而Joint更弱，不能据此判定错位有害 |
| 原文→H文本→G？ | trace映射可行；重提取会丢失属性，旧JVP不能充当新语境JVP |
| 如何双向交替？ | B建立H文本→G新条件与G图库→H奖励；C再研究统一Joint/交互目标 |
| 最小下一实验？ | 两轮A1对B匹配控制验证H→G反向通路；不立刻扩K或60轮重训 |

G2.94来自Train，H2.17来自Validation，不能直接相减。相同128 Validation的G K为2.8515625。
H Train全部2048的已解析候选均值5.0283203125；用户约5.04涉及只统计2037可更新样本的分母。

| 范围 | H | n | G原候选 | G正分候选 | G K | H已解析 | H合法* | H K | 实际编辑 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Train | 9 | 2048 | 5.116211 | 4.894043 | 2.939453 | 5.028320 | 5.008301 | 2.225098 | 2.221680 |
| Train | 20 | 2048 | 5.116211 | 4.894043 | 2.939453 | 5.028320 | 5.008301 | 2.246094 | 1.878906 |
| Train | 60 | 2048 | 5.116211 | 4.894043 | 2.939453 | 5.028320 | 5.008301 | 2.107910 | 1.856934 |
| Validation | 9 | 128 | 4.898438 | 4.648438 | 2.851562 | 4.781250 | 4.781250 | 2.171875 | 2.171875 |
| Validation | 20 | 128 | 4.898438 | 4.648438 | 2.851562 | 4.781250 | 4.781250 | 2.234375 | 1.898438 |
| Validation | 60 | 128 | 4.898438 | 4.648438 | 2.851562 | 4.781250 | 4.781250 | 2.179688 | 1.859375 |


*H合法列为进入策略动作的初始支持集，Train11条不可训练样本保留visit、固定NOOP。
原G候选包含unresolved/不支持表示，H候选已经解析，不能把这两列直接当同类数量。
Validation无Optimizer更新，表中G K只描述旧监督条件，K+1不是Validation真实更新次数。

## 1. 来源与验证范围

2176条原G条件、2048/128 H资格记录。H9/H20/H60各完整2048 Train visit动作，共6144条；
每个候选冻结128 Validation动作，共384条。
Train是Epoch内逐次更新过程的greedy快照，不是用Epoch末权重重新对Train推理。
H9使用G10奖励图库，H20/H60来自H10后的G60适配，不是单因素时长曲线。

逐Query AP/Rank-1等读取基准23fd02d已核验的128×128分数派生CSV，本轮未重新排名/评分/编码。
原128 Caption规则重提取与资格属性exact；384条旧攻击文本仅规则重提取，无Tokenizer/模型执行。
113冻结producer、G7个prepare producer及实际输入SHA核对；151378历史文件和18617相关产物stat保持。
主解析65.772秒、Peak RSS 1.661GiB；无Checkpoint/Tensor反序列化为模型。

正式H11–60训练和既有CPU审计仍PASS，旧发布器hashlib.file_digest FAIL原样保留。
H7 Selection未变，无Test选模、新Full、GPU或交替训练。
code_evidence.csv提供下列全部公式的实际源码绝对路径、函数、行号与完整SHA。

## 2. G的真实属性选择与训练链

### 2.1 Caption、Canonical、Span

/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_unit_v55_interface_20261003/attribute_interface.py:55–103逐Stage01 provenance mention转换，
保留semantic_axis、region、canonical_value、raw_text、原字符Span、linked_object及unresolved原因。
不字符串搜索迁移Span，不合并同词不同位置。G ID含row_id/旧slot/linked_object，
H ID只哈希六字段，不同ID不能用来判断属性不同。

必须区分：原候选→有表示/原Span/semantic basis rank>0可评分候选→正S_raw候选→Top-K。
本Train可评分10023项，均为正值；H没有同样的reader/basis资格要求。
G每样本K属性轮+1原图联合更新，不等于对K个词或K个互不重叠图像区域各攻击一次。
G推理仅G(I)，一次generate_once与原投影，不读文本/属性/JVP，也没有推理时FiLM。

### 2.2 S_raw数学及来源权限

令原Caption block8状态H_T，最终归一化编码f(H_T)，图库特征v_j，配对索引p：

~~~text
s_j = <f(H_T), v_j>
F = 1 + sum_(j != p) sigmoid((s_j-s_p)/0.02)
D_T = dF/dH_T
g_i = sum_l w_i,l * D_T,l
S_raw,i = || Q_i^T g_i ||_2
~~~

D_T是IRRA隐藏状态检索梯度，非生成器G。w_i由冻结reader/区域pooling给出，单位L2；
Q_i为冻结正交语义子空间basis。见/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_importance_step51_20261003/importance.py:15–21、
/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_importance_step51_20261003/importance.py:24–35及paired_soft_rank源码索引。

F针对指定paired图，其他同PID图仍列入竞争项；H奖励则使用全部同PID正例的最佳正例Margin。
历史S_raw用官方train clean参考图库，34054张且含内部Validation图；
不是H的2048 Train奖励bank，不声称参考图库与内部Validation图像完全隔离。
本轮保持历史协议，不以旧S_raw参考设置冒充新的Train-only联合奖励。

S_raw是连续子空间一阶敏感度，不是Unicode候选真实收益或类别概率。
basis维数/reader定位/Caption梯度尺度会影响它；原算法不做sqrt(rank)归一化。
本轮不改变排名，也不把高S_raw等同于更强Joint收益。

### 2.3 自然K与截断

/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_tta_step74_dynamic_k_20261004/selection_audit.py:47–72按正S_raw降序，tie保留原输入顺序：

~~~text
p_i = S_i^3 / sum_j S_j^3
E = -sum_i p_i log p_i
N_eff = exp(E)
K_G = min(m, ceil(beta*N_eff)), beta=1
m=0 => K_G=0
~~~

没有执行cap4。payload.capped_k/truncated为旧capped对照元数据，实际selected_attributes使用natural_k前缀。
prepare_uncapped按连续chunk复用旧验证器，不删超过4的项，见/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_tta_step74_dynamic_k_20261004/paired_prefix.py:11–26。
Train实际最大7；同公式合成9个均分候选允许K9，证明不是硬上限7。

| K | Train样本/2048 | 比例(%) |
| --- | --- | --- |
| 0 | 5 | 0.2441 |
| 1 | 34 | 1.6602 |
| 2 | 639 | 31.2012 |
| 3 | 842 | 41.1133 |
| 4 | 463 | 22.6074 |
| 5 | 61 | 2.9785 |
| 6 | 3 | 0.1465 |
| 7 | 1 | 0.0488 |


均值2.939453125、中位3；K>5为4/2048，K>4为65/2048。
sum K=6020，sum(K+1)=8068；6020是更新账本，非不同词/区域数。
Validation原K均值2.8515625，绝不执行训练Optimizer。

S_raw n10023，均值2.129364251、median0.144939333、p95 4.609741516、max2421.937997455；
入选n6020，均值3.169883346、median0.339174973、p95 6.612051997。
长尾明显，均值不能代表普通Caption尺度，原精度见g_attribute_scores.csv。

### 2.4 JVP、逐属性更新与最终Joint

~~~text
B_i = w_i outer unit(projected_context_i)
v_i(T) = d f_T(H_T + t*B_i)/dt |_(t=0)
alpha_i = S_i / sum_(selected) S_i
v_local = sum_i alpha_i * v_i(T)
z_T = normalized original whole-caption feature
~~~

S_raw的Q投影梯度方向与这里的projected_context方向不同，不能混写。
JVP映射原block8→后续block→ln_final/projection/EOT/L2；
不再单位归一化，聚合直接加权。见/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_tta_step72_20261003/attribute_tta.py:63–71、
/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_tta_step72_20261003/attribute_tta.py:91–103。本轮只验已存SHA，不新算JVP。

各属性轮从上一图像detach继续、相对原图投影，立即Adam更新：

~~~text
L_global = <f_I(I_adv), z_T>
L_local,i = <f_I(I_adv), v_i>
L_retain,i = ReLU(<f_I(I_adv), v_(i-1)> - stopgrad(reference_(i-1)))
L_i = L_global + L_local,i + L_retain,i
~~~

第一轮无Retain，只保紧邻上一轮。K轮后重新输入原图，最后一次更新
L_joint=<f_I(I_adv),z_T>+<f_I(I_adv),v_local>，无Retain；正dot最小化，各权重1。
这里joint是G属性聚合步，不是G/H联合训练；无TV/扰动正则/FiLM。
见/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/serial_training.py:116–165、/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/losses.py:38–44。
最终推理见/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/inference_once.py:9–14，只有1次G。

## 3. H的真实属性、K与三个Head

### 3.1 重要性与K梯度

CaptionFrontend.convert只读Caption规则，不输入图/PID/S_raw。
Byte Transformer字符向量在属性Span内平均，再加axis/region embedding；
Attribute Head产生l_i=w_A^T h_i+b_A，predicted_importance=softplus(l_i)。
没有S_raw回归监督，数值不能叫G属性重要性的估计值。

/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/policy.py:104–159先计算各属性合法Unicode/Tokenizer候选，
排除初始不可编辑属性，再在importance.detach()的Python float上计算同gamma3/beta1自然K。
K本身不接收Reward梯度；同Head的logits仍通过SCST动作log-prob更新，K可以随参数间接变化。
gamma=3是幂强化，不是改3个字符；beta=1是有效数系数，不是只选1项。

额外数学性质：属性softmax（含同Head的STOP）对logits共同平移不敏感，
softplus分数比和K却可能变化。CPU Fixture保持同样softmax概率，K可从2变3。
这说明预算参数化没有S_raw式绝对校准，不证明它导致当前退化。
K无硬上限2/4；实际H20 Train出现K6、H60 Validation出现K5。

### 3.2 动作顺序

1. Attribute Head在当前合法属性+STOP中采样/贪心；STOP始终合法，选择即退出。
2. 已选属性移出available，保存selected ID。
3. Position Head对该原Span合法字符与额外NOOP位置评分。
4. Position NOOP不改字，但消耗外层1槽、该属性不能重选；并不标记该词已编辑。
5. Replacement Head只在原字符白名单支持中选，每属性最多1字符、共享原词最多1次。
6. 根据原坐标回放全部编辑，保存原/新文本、Span、码位、UTF-8、决策/掩码与Tokenizer审计。

见/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/constraints.py:73–89、/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/constraints.py:52–71。
无白名单候选、Tokenizer不变或产生新截断的候选先被mask；
原文本截断/超长/无合法编辑按既有资格固定处理，不补抽、不删Query。
编辑后共享词或Tokenizer支持可能变化；Train的mask移除计数是事件，不是等量K损失。
三组Validation的额外mask移除事件均0，实际未用槽由下面账本完整解释。

### 3.3 为什么约2次编辑

~~~text
K_H - edits = Position_NOOP次数 + (K_H - 非STOP属性选择次数)
            = Position_NOOP槽 + STOP剩余未用槽
~~~

H9 Validation：278槽→278选中→278编辑，NOOP/STOP未用均0。
H20：286→286→243，43个Position NOOP、STOP未用0。
H60：279→276→238，38个Position NOOP、1次STOP留下3槽。
H20未用率43/286=15.03%，H60 41/279=14.70%，账本与字符串回放均成立。
H9约2.17次主要由自然K限制；H20/H60还有约0.34/0.32槽未用。

Train各2048完整；H9/20/60预算4557/4600/4317，
编辑4550/3848/3803，Position NOOP0/729/501、STOP未用槽7/23/13。
因此不能直接判为Bug，也不能一律叫无效动作浪费：NOOP允许主动拒绝低收益编辑。
取消耗槽/强制用满K/加NOOP惩罚会改变策略语义，是新算法变量，不是修复。

80条Validation至少5合法候选，H9中0条只编辑0/1次，H20为17条、H60为13条。
支持研究预算利用，但现有数据不能证明强制多编辑会更强。

## 4. G/H严格匹配与排名一致性

跨样本先核对Caption SHA/原图身份/PID/role，属性以六个Span/语义字段严格匹配，
不用ID字符串或fuzzy。重叠/重复词不同Span不能合并，一对多标AMBIGUOUS。

| 范围 | 样本 | G全部候选 | G可用Span语义 | H已解析 | 原候选全等样本 | G-only键 | H-only键 | 歧义键 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2048 | 10478 | 10405 | 10298 | 1956 | 106 | 0 | 1 |
| validation | 128 | 627 | 623 | 612 | 120 | 11 | 0 | 0 |


Train73个G原候选无可用严格Span/语义，Validation4个。
Train106个G-only键和1个歧义，Validation11个G-only键；H-only原候选键均0。
歧义cuhk:8643:0不强行对齐。原候选全集相等1956/2048、120/128；
进一步G正分候选与H已解析集合相等1776/2048、112/128。
attribute_alignment.csv以G正分候选作左侧，H_ONLY不等于原提取器新建了属性；
原提取层的覆盖必须看raw_candidate_coverage.csv。

| 范围 | H | 同集≥2样本 | 有效Spearman | 其中2属性 | G并列样本 | H并列样本 | 平均Spearman | 同集Top1一致 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Train | 9 | 1736 | 1734 | 83 | 187 | 0 | -0.533802 | 100/1736 |
| Train | 20 | 1736 | 1734 | 83 | 187 | 0 | -0.698357 | 27/1736 |
| Train | 60 | 1736 | 1734 | 83 | 187 | 0 | -0.501819 | 95/1736 |
| Validation | 9 | 105 | 105 | 4 | 8 | 0 | -0.506345 | 7/105 |
| Validation | 20 | 105 | 105 | 4 | 8 | 0 | -0.672066 | 4/105 |
| Validation | 60 | 105 | 105 | 4 | 8 | 0 | -0.502401 | 4/105 |


Train2条常量G排名无相关，故1734/1736有效；不伪造0。
原主表corr helper最低3点，strict_rank_comparison.csv补充至少2点并单列2属性数量；
本报告排名以补充完整口径为准。对候选集不同的样本不强行计算全排序相关。

直接结论：G敏感的连续语义子空间通常不是H最偏好的受约束字符属性。
H没有复现S_raw排名的监督，因此负相关不是Bug证据。
G Train选择garment_type上/下装3502/6020=58.17%，color上/下装1951/6020=32.41%；
H Validation gender/person编辑占比H9 110/278=39.57%、H20 77/243=31.69%、H60 56/238=23.53%。
这些是文本语义标签，不代表已测得的图像空间区域归因。

G选中而H未编辑平均为H9 2.109375、H20 2.164062、H60 1.984375项/Caption；
H编辑但G未选平均1.429688/1.210938/0.992188。
差异确实存在，但重合率不能等同互补效果，不能强制H追随S_raw就宣称改进。

## 5. K差异的可核验代数分解

令C为两侧严格共有的合法候选，在已有分数上计算：

~~~text
K_G(fullG)-K_H(fullH)
 = [K_G(fullG)-K_G(C)] + [K_G(C)-K_H(C)] + [K_H(C)-K_H(fullH)]
~~~

前后项是候选集影响，中项是同候选分数形状影响。
只做标量公式重算，绝非改变K后实际攻击；分解顺序固定，不叫唯一因果贡献。

| 范围 | H | n* | G→共同候选 | 共同集分数形状 | 共同→H候选 | G K−H K |
| --- | --- | --- | --- | --- | --- | --- |
| Train | 9 | 2037 | 0.001964 | 0.735395 | -0.027491 | 0.709867 |
| Train | 20 | 2037 | 0.001964 | 0.754050 | -0.067256 | 0.688758 |
| Train | 60 | 2037 | 0.001964 | 0.854197 | -0.028473 | 0.827688 |
| Validation | 9 | 127 | 0.000000 | 0.700787 | -0.015748 | 0.685039 |
| Validation | 20 | 127 | 0.000000 | 0.669291 | -0.047244 | 0.622047 |
| Validation | 60 | 127 | 0.000000 | 0.708661 | -0.031496 | 0.677165 |


*仅有真实importance的2037 Train/127 Validation，不给11/1个固定不可训练样本补假分数。
Validation的G→共同候选项均值0，分数形状项约0.67–0.71槽，
H额外候选再抵消约0.016–0.047槽。
H合法候选4.78125反而多于G正分4.6484375，因此不能说主要因H候选少导致K少。
STOP/NOOP只解释K之后编辑减少，不解释K_G与K_H本身。

## 6. 原文到攻击文本与G监督

| H | 原属性 | 重提取属性* | 原语义/坐标找回 | 未找回 | 其中自身编辑 | 其中未编辑 | UTF8起点移动 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 612 | 319 | 308 | 304 | 278 | 26 | 472 |
| 20 | 612 | 370 | 363 | 249 | 243 | 6 | 396 |
| 60 | 612 | 369 | 362 | 250 | 238 | 12 | 374 |


完整见[原文/攻击文本映射](original_adversarial_mapping_analysis.md)和[Schema草案](proposed_attribute_schema.json)。
字符坐标保持，UTF-8偏移变化；规则重提取会丢失大量原属性；
继承原Canonical锚点与trace才是可靠来源链。
原JVP可继续作为原语义参照，但不能冒充攻击语境JVP。若G只读旧参照，仍没有H→G反馈。

## 7. K更大或重合更多是否值得研究

| H | 变量对 | n | Spearman | Pearson |
| --- | --- | --- | --- | --- |
| 9 | G_selected_H_edited_jaccard__AP_delta_vs_image_only_pp | 127 | -0.004486 | 0.088254 |
| 9 | unused_K__AP_delta_vs_image_only_pp | 128 | — | — |
| 9 | candidate_count__unused_K | 128 | — | — |
| 20 | G_selected_H_edited_jaccard__AP_delta_vs_image_only_pp | 127 | 0.190478 | 0.195915 |
| 20 | unused_K__AP_delta_vs_image_only_pp | 128 | 0.074392 | 0.158845 |
| 20 | candidate_count__unused_K | 128 | 0.026014 | 0.014069 |
| 60 | G_selected_H_edited_jaccard__AP_delta_vs_image_only_pp | 127 | 0.198957 | 0.251729 |
| 60 | unused_K__AP_delta_vs_image_only_pp | 128 | 0.108130 | 0.163676 |
| 60 | candidate_count__unused_K | 128 | -0.256454 | -0.257511 |


AP_delta=Joint AP−Image-only AP，负值代表文本带来更强额外攻击。
Jaccard与AP_delta的Spearman H9≈−0.0045、H20≈+0.1905、H60≈+0.1990，
没有“越重合越强”的稳定负关联。
H60 Jaccard0.259449高于H20 0.211099，Joint mAP却42.874081高于40.427635，攻击更弱。
单轨迹/128重复Validation且Caption难度/类别/K混杂，不能据此做因果判断。

H20→H60未用槽变化与AP变化Pearson≈0.202、Spearman≈0.164，较弱；
候选数量与未用槽Spearman H20≈0.026、H60≈−0.256，不支持属性越多必然越浪费。
没有真实K=3/4/5干预结果，绝不推算这些预算的mAP。

最小未来K设计：同固定H、G bank、Caption及候选白名单，预设自然K与K1/2/3/4；
实际支持不足则受合法候选数限制，保留STOP/NOOP和每词一次规则。
“NOOP不耗槽/强制用满”应单独作另一变量，不与K一起改。
先Train诊断/独立Validation，不能Test选K。本轮未运行。

## 8. 交替训练基础、缺口与最小方案

可复用原G单样本K+1训练、H SCST单步、Caption-only前端/动作、
Train权限/正负例、不可变存储/Checkpoint与独立Validation。
缺少H→G文本条件桥、新语境JVP身份、round级双模型/双Adam/RNG完整恢复和动态bank重绑定。

原H奖励：
~~~text
R_sim = sim(I_G[paired],T)-sim(I_G[paired],T_adv)
M_G(T) = max_samePID sim(I_G,T)-max_differentPID sim(I_G,T)
R_rank = M_G(T)-M_G(T_adv)
R_total = R_sim+R_rank
L_SCST = -detach(R_sampled-R_greedy) * sum log pi(sampled actions)
~~~
synergy_increment=Adv配对相似度下降−Clean配对下降，仅诊断，不进SCST。
见/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/rewards.py:63–82、/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/scst.py:6–14。
目标不是完整多正例AP，也不显式优化相对单模态交互。
G训练读取固定原文监督，H读取固定G图库，是单向条件链，不是端到端联合优化。

[交替训练方案](alternating_training_design.md)区分A0/A1/B/C，
包含缓存失效、两轮最小原型、更新/编码/JVP预算、真实信息流验证及失败恢复。
B产生双向条件依赖，C进一步形成目标层协调；仅交替两个旧训练器不够。
未核对MSI定义，不作与MSI机制等价或创新成立的声称。

只允许下一项新实验时，优先A1对B两轮匹配控制验证H→G：
同G父状态、图像、原属性，原文/旧H文本应产生真实可追踪的新z/JVP和G梯度差，
NOOP作阴性控制；然后同H动作比较旧/新G bank奖励和H更新差。
不要求G变化在H未更新前直接改变Caption-only推理；那反而违反接口。
本轮仅设计，不启动，也不改K、原Reward或Selection。

## 9. 交付与局限

g_attribute_statistics.csv含G候选/K/实际训练更新；Validation实际更新为0，theoretical_Kplus1只是理论计数。
h_attribute_statistics.csv/k_budget_comparison.csv含6144 Train与384 Validation真实动作，
attribute_alignment.csv/strict_rank_comparison.csv/attribute_overlap_statistics.csv含严格比较和分母。
k_decomposition.csv是代数而非干预；attribute_span_mapping.csv/canonical_reextraction*.csv保留位置和重提取证据。
8张16:9 PNG/SVG及原精度CSV见FIGURES.md，所有源SHA/行号见code_evidence.csv和SOURCE_EVIDENCE.json。

没有测得G像素的属性空间因果归因、没有新语境JVP数值、没有K/NOOP变更后的攻击结果、
没有多seed或独立验证。不能把已有H60退化归因于单一属性错位、K不足或NOOP。
Train动作不是固定末态H重推；所有缺失都保持缺失，没有伪造概率或新mAP。

## 附录：关键函数路径、行号与SHA

| 路径 | 函数 | 行号 | SHA-256 |
| --- | --- | --- | --- |
| /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/src/attributes/text_extractor.py | extract_with_provenance | 438–440 | 47deccbf7c86ac6d2d6000ecb7997b8f7f2aff0ebb3c96b814e1cd7c665253b3 |
| /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/src/attributes/text_extractor.py | extract_with_provenance | 184–187 | 47deccbf7c86ac6d2d6000ecb7997b8f7f2aff0ebb3c96b814e1cd7c665253b3 |
| /home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_unit_v55_interface_20261003/attribute_interface.py | convert_record | 55–103 | 5ec8d0c9e941b9583deae721ce59c9ba9cf2fa8045fc4f3e041b6b59cf0fdcca |
| /home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_unit_v55_interface_20261003/attribute_interface.py | text_candidates | 106–112 | 5ec8d0c9e941b9583deae721ce59c9ba9cf2fa8045fc4f3e041b6b59cf0fdcca |
| /home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_unit_v55_interface_20261003/attribute_interface.py | instance_id | 49–51 | 5ec8d0c9e941b9583deae721ce59c9ba9cf2fa8045fc4f3e041b6b59cf0fdcca |
| /home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_importance_step51_20261003/importance.py | baseline_gradient | 15–21 | a6dc10128e16a21db0c7fb8ad239506f827b2d44fef3bb1557c1a162fd922eba |
| /home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_importance_step51_20261003/importance.py | projected_score | 24–35 | a6dc10128e16a21db0c7fb8ad239506f827b2d44fef3bb1557c1a162fd922eba |
| /home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_importance_step51_20261003/importance.py | score_instances | 45–63 | a6dc10128e16a21db0c7fb8ad239506f827b2d44fef3bb1557c1a162fd922eba |
| /home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_importance_step51_20261003/importance.py | attach | 112–155 | a6dc10128e16a21db0c7fb8ad239506f827b2d44fef3bb1557c1a162fd922eba |
| /home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_tta_step74_dynamic_k_20261004/selection_audit.py | selection | 47–72 | 46213fc3f0e09cdb9c37b2c1abb8880fac94d4989739aaa21e51e4385c68a409 |
| /home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_tta_step74_dynamic_k_20261004/paired_prefix.py | prepare_uncapped | 11–26 | 8c4f37cc7a92d6ad96269468577f86feccf751acc3367b20b7a729b62689ed05 |
| /home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_tta_step72_20261003/attribute_tta.py | attribute_delta | 63–71 | 169dc1676d66f172de7079d9b083a23e09ced35dd0ba583b7f2fac86b8ff4051 |
| /home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_tta_step72_20261003/attribute_tta.py | TextTransport | 74–103 | 169dc1676d66f172de7079d9b083a23e09ced35dd0ba583b7f2fac86b8ff4051 |
| /home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_tta_step72_20261003/attribute_tta.py | read | 91–103 | 169dc1676d66f172de7079d9b083a23e09ced35dd0ba583b7f2fac86b8ff4051 |
| /home/lzf/ldx/outputs/idea-TBPS-test1/stage03-attribute-unit-v2-20261002/code/src/attributes/irra_soft_rank_gradient.py | paired_soft_rank | 18–32 | 284ecfc897b6f0a901398babf2c95fe8c991796a708158c448504496616c8ba8 |
| /home/lzf/ldx/outputs/idea-TBPS-test1/stage03-attribute-unit-v2-20261002/code/src/attributes/irra_attribute_unit_prototype.py | objective | 105–107 | 33998e7ab6a251a5fa92ec0c3e8d151ff09a17ecd9c95dc85030116a6f784f62 |
| /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/prep_text_frontend.py | __call__ | 34–55 | ef28b99f8fbf62fab6efbe29a89a511b8414c59c8bb97763419c48c483e64b40 |
| /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/serial_training.py | _update | 69–114 | e42fa8bd6ea933530bda8e2e67d5149e7059151f4b2caa0b662dbaaa856e105d |
| /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/serial_training.py | train_single_sample | 116–165 | e42fa8bd6ea933530bda8e2e67d5149e7059151f4b2caa0b662dbaaa856e105d |
| /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/losses.py | retain_loss | 24–36 | d415dfe326cc2f580d202d0220d5bb0c5895ed626115b0828e516b598df8f0b2 |
| /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/losses.py | attribute_losses | 38–44 | d415dfe326cc2f580d202d0220d5bb0c5895ed626115b0828e516b598df8f0b2 |
| /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/losses.py | joint_losses | 46–56 | d415dfe326cc2f580d202d0220d5bb0c5895ed626115b0828e516b598df8f0b2 |
| /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/inference_once.py | generate_once | 9–14 | 83d38a6c782581acd53c3d8c5272c24c89ae85a32f5c7da9230edff1a8f97e6a |
| /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/epoch_training.py | epoch_order | 20–23 | 0f567c02d4a891dc0fff109abb03aee5942dc97322c6940f92fdef17dfffba4e |
| /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/epoch_training.py | train_epoch | 35–57 | 0f567c02d4a891dc0fff109abb03aee5942dc97322c6940f92fdef17dfffba4e |
| /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/image_only_generator.py | forward | 20–21 | 4745e6947308fdc4f5d5496b6a99c9a68699abef8131f2a211e1bd13470eec85 |
| /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/image_only_generator.py | forward | 38–47 | 4745e6947308fdc4f5d5496b6a99c9a68699abef8131f2a211e1bd13470eec85 |
| /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/text_frontend.py | convert | 28–50 | f9fc38d5bdcb40d77f5a0bfa350ff7a9540f2d51c50f4dc4bea84b1d208e3b6d |
| /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/policy.py | natural_k | 10–22 | 0bbcd430e02eb61c38857b7a812b87f163d1bd0a09f0b147cede23e5712fbed3 |
| /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/policy.py | forward | 78–102 | 0bbcd430e02eb61c38857b7a812b87f163d1bd0a09f0b147cede23e5712fbed3 |
| /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/policy.py | rollout | 104–159 | 0bbcd430e02eb61c38857b7a812b87f163d1bd0a09f0b147cede23e5712fbed3 |
| /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/constraints.py | make_edit | 36–50 | 7b9202d2885165302aa1ec475feaefe0150869b12660aa40d4552a0f29759264 |
| /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/constraints.py | replay | 52–71 | 7b9202d2885165302aa1ec475feaefe0150869b12660aa40d4552a0f29759264 |
| /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/constraints.py | candidates | 73–89 | 7b9202d2885165302aa1ec475feaefe0150869b12660aa40d4552a0f29759264 |
| /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/contracts.py | word_at | 63–67 | 2dc90ad865dcdc9b72b5ebb3fe4772a5b0c4c082f91694d1da244a07b72de0e9 |
| /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/contracts.py | attribute_id | 30–33 | 2dc90ad865dcdc9b72b5ebb3fe4772a5b0c4c082f91694d1da244a07b72de0e9 |
| /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/phase4_training.py | train_step | 38–79 | 649ae6b0e9ac63245c9d77f8cfcbe653b000ec9f466070e79990ece210d0466e |
| /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/rewards.py | evaluate_many | 63–82 | 4591d7fbb993eb2fcb5470931ebca4b24b88f92a22e118660b822a9c5fb769df |
| /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/scst.py | scst_loss | 6–14 | 3edb22ac84791f2cd451a5ef93ea87fc75e88827a3703b5648a51b302b988c36 |
| /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/formal_training/runtime.py | __init__ | 17–41 | d68bed7ec961dbf056aa16f3eb4330565588d71041eff36a9ccf8d6835d34ab6 |
| /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/formal_training/runtime.py | __init__ | 67–72 | d68bed7ec961dbf056aa16f3eb4330565588d71041eff36a9ccf8d6835d34ab6 |
| /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/formal_training/runtime.py | bind | 74–87 | d68bed7ec961dbf056aa16f3eb4330565588d71041eff36a9ccf8d6835d34ab6 |
| /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/formal_training/runtime.py | __init__ | 90–93 | d68bed7ec961dbf056aa16f3eb4330565588d71041eff36a9ccf8d6835d34ab6 |
| /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/formal_training/runtime.py | evaluate_many | 101–116 | d68bed7ec961dbf056aa16f3eb4330565588d71041eff36a9ccf8d6835d34ab6 |


## 附录：一次真实H20动作及原G属性对照

固定规则：按原Validation顺序，首个既有Position NOOP且实际编辑至少1字符的样本；不读取指标选例。

row_id=cuhk:17663:0，原始记录：/home/lzf/ldx/outputs/tbps-framework-hv1-g60-adaptation-epoch60/validation/epoch_0020/query-002/record.json，SHA=aa039a84ca7ca540a0b951ae28078973e6f223d2ae46904816f5c869c2e94650。

原文："The woman is wearing a white shirt with a black mini skirt and white shoes.She is also wearing a grey hat"

攻击文本："The wоman is wearing a white shirt with a black mini skirt and white shoes.She is also wearing a grey hаt"

| 原词/Span | axis/region | G S_raw | H importance | G选中 | H选中 | H编辑 |
| --- | --- | --- | --- | --- | --- | --- |
| black/42:47 | color/lower | 0.015550303243025199 | 1.5165247917175293 | 1 | 1 | 0 |
| white/23:28 | color/upper | 0.015193100164382632 | 0.012784487567842007 | 1 | 0 | 0 |
| skirt/53:58 | garment_type/lower | 0.025922892673363056 | 0.0004449640109669417 | 1 | 0 | 0 |
| shirt/29:34 | garment_type/upper | 0.02074278730283516 | 0.00033593192347325385 | 1 | 0 | 0 |
| woman/4:9 | gender/person | 0.00033676041770915803 | 3.5050861835479736 | 0 | 1 | 1 |
| hat/102:105 | hat/head | 0.001555055142238579 | 4.5223493576049805 | 0 | 1 | 1 |

K=3，属性选择顺序=["e01ff9677acf3c311cc69826", "09c5a1520d173efb19948b0a", "9f06003071349f2f186bf383"]。

完整decisions与掩码、Unicode编辑/码位、Tokenizer旧审计见TRACE_EXAMPLE.json；此处未调用H。
原候选歧义cuhk:8643:0的两G来源/linked_object及唯一H锚点见AMBIGUOUS_EXAMPLES.json，未强配。

## 附录：类别层级重合（不是同一原文属性）

类别只取semantic_axis/region；不同词、Span、canonical即使类别相同，也不改标为EXACT。

| 范围 | H | n | 平均类别Jaccard | H编辑类别在G选中类别中的次数/全部编辑 |
| --- | --- | --- | --- | --- |
| Train | H9 | 2048 | 0.231169 | 1596/4550 |
| Train | H20 | 2048 | 0.210231 | 1408/3848 |
| Train | H60 | 2048 | 0.257923 | 1736/3803 |
| Validation | H9 | 128 | 0.251837 | 96/278 |
| Validation | H20 | 128 | 0.219235 | 90/243 |
| Validation | H60 | 128 | 0.266404 | 112/238 |

## 最终CPU交付核验

26项不同CPU测试PASS，包含空属性样本保留及陌生成员拒绝；6528条类别派生行与当前代码逐条一致。
首次类别派生辅助器因未保留空属性样本被覆盖数断言拒绝；只修本轮解析器，原失败日志保留，无实验或神经推理重跑。
Train全部2048候选平均5.0283203125；2037条可训练样本候选平均5.038782523318606，四舍五入约5.04。
初始交付检查24项、最终26项分别留证；原旧训练/audit/发布FAIL均未改写。
