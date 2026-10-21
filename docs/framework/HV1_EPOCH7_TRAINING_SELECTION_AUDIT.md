# H-v1 Epoch7训练充分性、Checkpoint选优与G60联合攻击适配性只读审计

审计基准：20f61ae5471ae68081b6b80419baba8703fd1744，分支codex/gv2-six-epoch-evaluation。
本次仅读取历史源码、元数据、已保存逐visit标量/Validation结果及文件SHA，新增本报告和派生证据/CSV。
没有导入Torch/模型、反序列化Checkpoint、执行GPU、训练、生成、评分、排名或历史CPU audit入口；
历史原件不改写。[证据索引](results/hv1_epoch7_training_selection_audit/SOURCE_EVIDENCE.json)。

## 1. 五个核心问题

**Q1：H为什么只训练到Epoch7？**

这一前提不成立。正式H训练计划及实际完成均为10 Epoch：
20480 visits、20370次Adam step，最终Checkpoint为E10/visit20480。
E7/visit14336是事后独立内部Validation选中的发布权重，不是训练提前终止或Early Stopping。
H7之后还有24份Checkpoint，包括已验证的E8/E9/E10及周期版本。
H与G的Epoch也不是相同训练量：H每轮2048 visits/2037次Adam step；
G每轮8068次逐属性+联合Adam step。不能仅凭“7对60”决定是否需要补齐训练轮数。

**Q2：历史证据足以证明H7是合理选择吗？**

**是，限定于固定G10、IRRA和128内部Validation上的10个Epoch边界候选。**
全部E1–10均有完成结果并包含在Selection中，H7的Joint mAP=67.49383544921875，唯一最低。
选择准则是Joint mAP，不是Text-only mAP。没有证据证明它优于所有81份已保存权重、
任意更晚训练或未知数据，也没有证据足以确认优化已经收敛。

**Q3：H7是否是在G10条件下选优？**

**是。**Train奖励图库与Validation图库均绑定G-v2 E10文件SHA
f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e。
H的SCST奖励使用该固定Adv Gallery，Validation Joint使用同一G10来源。
G/IRRA冻结、只有H参数更新；这不是G/H共同反向传播或交替联合训练。

**Q4：H7搭配G60是否可能限制Joint性能？现有证据能验证吗？**

可以把“G10训练/选优未覆盖G60”列为待验证的适配问题，但**不能据此认定H7限制了G60**。
原Selection不等于G60+H7已被独立Validation选优确认。
正式Full证明H7在G60上仍有总体额外mAP下降；额外下降大小变化不证明欠训练、冲突或样本级互补性。
本轮未找到G60×H候选的独立Validation比较证据，也未进行新的验证。

**Q5：现在需要继续训练H吗？最小方案是什么？**

不建议立即续训，更不建议机械训练到60 Epoch。
先复用已有E1–10候选和128 Validation特征，**另行授权一次固定G60的内部Validation接线**：
仅新增128张Validation的G60/IRRA图像特征，复用已有10×128 H文本特征及Clean控制，
比较G10/G60两行×10个H候选的Joint与Text-only，不使用Test选模。
这比先训练新H或开展6×10×3个Full组合更小，也能直接检验原H7选择对G变化是否稳定。

## 2. 真实来源与只读核验范围

正式H根目录：
/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015

正式训练根：
/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/formal_v1/

已核验：

- 81份Checkpoint实际文件SHA、元数据、父子链、关联steps.jsonl SHA和连续visit/Adam计数。
- 10个Epoch边界候选、各128条Query的元数据/特征文件SHA、完成回执和Selection逐项对应。
- 训练/Validation图库完成记录与所声明文件SHA，G10/IRRA实际权重SHA及冻结来源源码SHA。
- 原qualification/训练/Validation身份与官方annotation、Train/Validation PID隔离、与官方Test PID无交集。
- 既有日志含FORMAL_CPU_AUDIT validation PASS及VALIDATION_PIPELINE_SUCCESS；
  本轮不重新调用这些审计程序。
- 14811份原文件stat、14802份实际文件SHA读取核验，前后不变。
  本轮不加载Checkpoint内部Tensor重验Adam数值；其内部恢复正确性引用原attempt中restored_H_Adam_RNG_exact=true，
  并核验原文件SHA，不能将其说成本次新执行的恢复测试。

关键SHA：

- H7 Checkpoint：487f3061b6cba2ddabccdd49745a7b99c8c549f4541781b8814b299ef5f279df。
- Selection文件：584e4ba2afc7b17fea3bdfb8501716498b84731fa216558daa2e8e4bb353f757。
- H训练协议canonical digest：bec28bbcc7b194e44b6cbe351b2c2615418c93d12b5afdb7cf3c4da975a041a8。
- G10：f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e。
- IRRA：9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009。
- E10 H最终Checkpoint：4ae964d51ea94840de71d9597d90a2234a36104e5d649cfee96577f59f94bacd。

Selection实际文件：
/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/formal_v1/selections/30f573098d28b6af6afe146330823986f724378fc7bd4df9d257fdd2c6f20dd1.json

训练/Validation协议、所有候选绝对路径和SHA详见证据JSON、候选CSV和附表。
文件名中的Selection canonical digest与文件SHA是两种不同身份，不能混用。

## 3. H具体学习什么

### 3.1 网络和编辑策略

正式配置：UTF-8 Byte Embedding、位置Embedding、属性类型/区域Embedding，
width64、2层、4 attention head的轻量Transformer Encoder；dropout=0，无大型预训练语言模型。
Byte上下文按字符聚合，再结合Caption-only Canonical属性及准确原Span，使用三个Head：

1. Attribute Head：为合法属性预测logit及softplus重要性，沿原gamma3/beta1 Natural-K计算K；
   同一Head对pooled特征产生STOP logit，采样时带mask无放回选择，贪心时选最大合法概率。
2. Position Head：在所选原属性Span内选择字符位置，也可选本属性不替换的NOOP槽。
3. Replacement Head：在Greek/Cyrillic视觉相似Unicode白名单中选择合法替换。

每属性最多一字符、共享词最多一编辑、总编辑≤K，保持原Span/字符位置/码位/UTF-8偏移可追踪。
STOP可提前结束，NOOP合法；原文过长/无合法候选等资格失败不会被静默删除。
候选必须让冻结IRRA Byte-BPE token发生变化且不引入截断；推理时仍使用固定训练侧Tokenizer约束，
不根据RaSa/APTM Tokenizer或目标评分重新优化。
H学习的是受约束动作概率，不是自然语言重写、配对图像属性评分、S_raw模仿或图像条件网络。

### 3.2 SCST、训练奖励和信息权限

每条合资格Caption由当前H产生Sampled与Greedy两种结果；
冻结IRRA编码原文/Sampled/Greedy三条文本，奖励脱离计算图：

R_sim = sim(G10(I_paired),T) − sim(G10(I_paired),T_adv)

Margin(T) = max同PID正例 sim(G10(I),T) − max负例 sim(G10(I),T)

R_rank = Margin(T) − Margin(T_adv)

R_total = R_sim + R_rank；两项权重均1。

SCST Loss = −detach(R_sampled − R_greedy) × sum(log π_sampled)。

只有H执行Adam(lr=1e−4, betas=(0.9,0.999), eps=1e−8, weight_decay=0)，batch1、seed42。
不通过真实Unicode替换或Tokenizer反传，不更新G/IRRA；H在CPU上做策略梯度，IRRA只给冻结文本奖励。
实际训练使用缓存Clean与G10 Adv两套图库，但**优化奖励的R_sim/R_rank均来自Adv Gallery**。
Clean相似度下降、Clean margin下降和synergy_increment只作诊断，
没有额外Text-only Loss、辅助协同Loss或synergy奖励权重。
synergy_increment=Adv正确配对相似度下降−Clean正确配对相似度下降，不要求正值。

训练奖励绑定层可以使用真实训练Caption/图像配对、PID、同PID正例/负例、IRRA反馈；
这些信息不是H.forward或attack的输入。H测试只有Caption及其独立提取的Canonical/type/Span，
无正确Gallery、PID、paired index、测试标签、评分或梯度。
官方Test不参与训练奖励或Checkpoint选择。

best-positive margin只约束最优正例和最难负例，不等于多正例mAP；
SCST优化期望奖励也不等于直接优化完整检索mAP。
因此训练R_total上升与Validation mAP变差可以并存，不能据此擅自改奖励公式。

## 4. Epoch、Visit、Adam step和真实停止原因

训练集合2048条固定图像/Caption身份，每轮全部遍历；
第e个Epoch使用torch.randperm(seed=42+(e−1))，预先保存完整query_order。
2048 Train和128内部Validation都来自official train，分别1861/121 PID，PID交集0，
与official Test PID交集0；全部使用对应图像的caption_index=0。
内部Validation不是official val，亦不是正式6156 Query的全Caption集合。
祖先split.json含54246/13700条Train/Validation Caption记录；H的2048/128身份是对应角色的子集，
与G-v2 conditions_full固定子集逐成员完全一致，不是祖先划分的全部成员。

visit是访问样本计数，batch1时一个visit对应一个样本；Adam step只在合资格visit上执行。
资格账本有2037 eligible、11 ineligible；后者每轮仍保留visit/NOOP和跳过原因，奖励/Adam均0。

- H7：7×2048=14336 visits，7×2037=14259 Adam steps，差77次。
- 全程：10×2048=20480 visits，10×2037=20370 Adam steps，差110次。
- 其中5条无Canonical、5条原Byte-BPE已截断，另1条有属性但无合法token-changing edit；
  三类都可能记录no_legal_token_changing_edit，总共11条，不能把原因计数相加成21。
- 合资格样本即使采样/贪心同奖励、出现零梯度，也仍按代码执行Adam step；
  不是资格跳过。Adam动量可导致该步参数继续变化。

每256个更新或Epoch边界保存完整不可变Checkpoint，外加seed42初始版本，
共81份，不是每个visit保存一次。
之后24份权重在H7之后，其中E8/E9/E10三个Epoch候选已全部完成Validation；
其余21份后期周期版本未进入这10候选比较。

completed.json、唯一训练attempt(complete=true)、全部81份父子元数据与20480条日志连续一致；
最后版本reason=epoch_complete，最终Adam step20370，无pending、failure或abandoned恢复记录。
原记录支持“按既定10 Epoch正常完成”，而不是E7被中断或自动Early Stopping。
本轮未在formal_v1/logs找到独立训练stdout原文件；不编造其原进程退出码或训练耗时。
正常完成结论来自上述正式完成和训练attempt证据，而非猜测日志缺失原因。

### 4.1 固定不更新样本

| row_id | 每个Epoch跳过 | 原因 |
| --- | --- | --- |
| cuhk:10822:0 | 1 | original_BPE_truncated；no_legal_token_changing_edit |
| cuhk:11364:0 | 1 | no_canonical_attributes；no_legal_token_changing_edit |
| cuhk:13924:0 | 1 | original_BPE_truncated；no_legal_token_changing_edit |
| cuhk:26530:0 | 1 | no_canonical_attributes；no_legal_token_changing_edit |
| cuhk:28228:0 | 1 | no_canonical_attributes；no_legal_token_changing_edit |
| cuhk:31553:0 | 1 | original_BPE_truncated；no_legal_token_changing_edit |
| cuhk:31873:0 | 1 | no_canonical_attributes；no_legal_token_changing_edit |
| cuhk:31942:0 | 1 | no_canonical_attributes；no_legal_token_changing_edit |
| cuhk:4532:0 | 1 | original_BPE_truncated；no_legal_token_changing_edit |
| cuhk:4611:0 | 1 | original_BPE_truncated；no_legal_token_changing_edit |
| cuhk:7586:0 | 1 | no_legal_token_changing_edit |


## 5. 全部历史Validation候选（原精度）

“Train Loss/Reward”是本次从已提交逐visit标量日志汇总的**该Epoch在线训练均值**，
不是对该Checkpoint重新评估得到的固定策略奖励，也不是训练结束后计算的新Loss。
合资格更新数每轮2037；记录均存在，没有用“原记录缺失”填补数值。
Validation每个候选128 Gallery/128 Query，全部保留，127编辑+1固定NOOP；
官方IRRA FP32归一化512维特征与官方rank，全同PID正例、无camera过滤。
所有候选的Clean mAP均100.0，G10 Image-only mAP均77.11044311523438。

| Candidate | Epoch | Visit | Adam step | Train SCST Loss均值 | Train sampled R均值 | Validation Text-only mAP | Validation Joint mAP | 选中 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H1 | 1 | 2048 | 2037 | -0.055089364034685735 | 0.02145191780659546 | 97.36328125 | 69.56414031982422 | 否 |
| H2 | 2 | 4096 | 4074 | -0.04831398501359857 | 0.03667101538138876 | 97.36328125 | 70.49264526367188 | 否 |
| H3 | 3 | 6144 | 6111 | -0.03627376534945376 | 0.04204251401324811 | 95.86588287353516 | 70.29902648925781 | 否 |
| H4 | 4 | 8192 | 8148 | -0.03067668712351935 | 0.046281565424003655 | 96.0403060913086 | 69.54497528076172 | 否 |
| H5 | 5 | 10240 | 10185 | -0.02475928492517237 | 0.04586396669163371 | 96.36582946777344 | 69.38693237304688 | 否 |
| H6 | 6 | 12288 | 12222 | -0.023416898912076038 | 0.04762060014086615 | 96.77734375 | 68.69178009033203 | 否 |
| H7 | 7 | 14336 | 14259 | -0.029624600380163582 | 0.04930738798536042 | 96.32160949707031 | 67.49383544921875 | 是 |
| H8 | 8 | 16384 | 16296 | -0.034940146162056034 | 0.05293554033367187 | 96.26303100585938 | 68.67459869384766 | 否 |
| H9 | 9 | 18432 | 18333 | -0.023161645972759843 | 0.05634713139651077 | 95.9375 | 68.97724151611328 | 否 |
| H10 | 10 | 20480 | 20370 | -0.020572395805177123 | 0.05713917396456976 | 96.97265625 | 69.76632690429688 | 否 |


四项检索指标原精度：

| Epoch | Text R@1 | Text R@5 | Text R@10 | Text mAP | Joint R@1 | Joint R@5 | Joint R@10 | Joint mAP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 95.3125 | 99.21875 | 99.21875 | 97.36328125 | 57.03125 | 85.9375 | 89.0625 | 69.56414031982422 |
| 2 | 95.3125 | 99.21875 | 99.21875 | 97.36328125 | 57.8125 | 85.15625 | 90.625 | 70.49264526367188 |
| 3 | 92.96875 | 99.21875 | 99.21875 | 95.86588287353516 | 57.8125 | 84.375 | 88.28125 | 70.29902648925781 |
| 4 | 93.75 | 99.21875 | 99.21875 | 96.0403060913086 | 57.8125 | 82.8125 | 87.5 | 69.54497528076172 |
| 5 | 93.75 | 99.21875 | 99.21875 | 96.36582946777344 | 57.8125 | 82.03125 | 88.28125 | 69.38693237304688 |
| 6 | 94.53125 | 99.21875 | 99.21875 | 96.77734375 | 56.25 | 83.59375 | 89.0625 | 68.69178009033203 |
| 7 | 94.53125 | 99.21875 | 99.21875 | 96.32160949707031 | 56.25 | 82.03125 | 86.71875 | 67.49383544921875 |
| 8 | 94.53125 | 99.21875 | 99.21875 | 96.26303100585938 | 56.25 | 82.03125 | 88.28125 | 68.67459869384766 |
| 9 | 93.75 | 99.21875 | 99.21875 | 95.9375 | 56.25 | 82.8125 | 86.71875 | 68.97724151611328 |
| 10 | 95.3125 | 99.21875 | 99.21875 | 96.97265625 | 57.8125 | 83.59375 | 88.28125 | 69.76632690429688 |


候选checkpoint路径统一为：
/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/formal_v1/training/checkpoints/visit_XXXXXXXX/checkpoint.pt

候选Validation结果路径统一为：
/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/formal_v1/validation/<Checkpoint SHA>/summary.json

具体10条路径及所有原始五项指标（包括原文件已有mINP）存于
[validation_candidates_raw.csv](results/hv1_epoch7_training_selection_audit/validation_candidates_raw.csv)；
没有新定义或重算排名。

### 5.1 候选SHA附表

| Epoch | Checkpoint SHA-256 | Validation summary文件SHA-256 |
| --- | --- | --- |
| 1 | eb4cbbeaf4f5aa71591fbda03bee552766aca69241831b370c56af5f803f7783 | b72688e1c0d1f660498161206e0463b3c510dcbdfd0a42ba23efe2b3ba49f51a |
| 2 | adad480049a3b223c0a94a3c56dce15b6bfbeba7863fe562e2b08877a12e7f99 | 8fa57734131cc66281b2b0c3c6e416e50c899fb48fe69592c029f9992c349c38 |
| 3 | 0d3f8482cf93d9ac301d8fdab8c4725ee5fd9197974fd443cd150908eb79eb5d | 343e482a48d3a7d5d32c6cf2a63fcc334edd3c927af5b82ae7a85ece209d5016 |
| 4 | de8de5320d87261aa6fc4ef73ca4340048506a21eccab315a4cb0afdc57cc3ac | 87febfb4057c69e52af611c4e444ab95e156bb500f62c20e2d852154823b8d93 |
| 5 | 107a021305a748766f24b93f2e25f99587a77e295c964c700de98b1e76ec30eb | 17bbc18204f218e3d32a47c8d6c40f968cdd9cf80ed535f78c841eb34f57464d |
| 6 | 14a83a877009d2b6b28d5b379bb75f64b5c653f4cfd36aadcbe0006237139734 | 8e62b189201b6a17a43cc362a570fecae5c07208ef0113768e989c409e50a628 |
| 7 | 487f3061b6cba2ddabccdd49745a7b99c8c549f4541781b8814b299ef5f279df | 9b63d7c7417aad678fcbf9036c0434ca52a4adb8d3faa11cbe044ff75501e2c0 |
| 8 | 6fcb655e1ef799348e7a7d22128fa75c12fcf89acc4763c6ce0f5cf4ac38fc6b | 9e0eca2474d040f17874205afdd6492d174e699b8320f160d0468339f1860519 |
| 9 | ca3f47ca4d9fb9da82a44b0252251539aa48feaac49dee8328dbd609fd026a79 | e29de0e3b2cda1621744fa4d14542c88579d296dbe7f0e6441a655608bd5e4ce |
| 10 | 4ae964d51ea94840de71d9597d90a2234a36104e5d649cfee96577f59f94bacd | dc30e2b137f14161b12290e9069207c071ad9d68d4e2adb2bef5a83d73d9a608 |


## 6. Selection规则、覆盖范围与训练充分性

原函数select_best取min((Joint mAP, committed visit))；
Joint mAP并列时取最早visit，但当前10候选的Joint最低值唯一，实际没有触发并列规则。
Selection包含E1–10全部10个Epoch边界候选，故不是只比较到E7就停止。
它没有覆盖所有81个已保存权重（另71个周期/初始版本没有Validation结果），也不覆盖本正式训练目录没有保存的E11以后权重。

H7在既定目标上有明确选优依据：
67.49383544921875低于E8=68.67459869384766、E9=68.97724151611328、E10=69.76632690429688。
但H7不是Text-only最优：H3 Text-only mAP=95.86588287353516，
低于H7的96.32160949707031，而其Joint mAP=70.29902648925781更高。
E4、E8、E9的Text-only mAP也低于H7，但Joint均更差。
因此不能把“Joint选出的文本生成器”自动称为独立文本攻击最优。

### 6.1 训练标量趋势（原精度）

| Epoch | Sampled R_sim | Sampled R_rank | Sampled R_total | Greedy R_total | Greedy synergy | 平均梯度范数 | 零梯度次数/2037 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.005362830186517846 | 0.016089087620077616 | 0.02145191780659546 | 0.030676237627302438 | -0.030694413347472426 | 0.8758770073197926 | 232/2037 |
| 2 | 0.013655951596836746 | 0.02301506378455202 | 0.03667101538138876 | 0.04521137050471406 | -0.026120540924383092 | 0.652797376261426 | 547/2037 |
| 3 | 0.017479572391432833 | 0.024562941621815274 | 0.04204251401324811 | 0.05059832929179555 | -0.02335004520268309 | 0.5328063765165351 | 731/2037 |
| 4 | 0.019522146380583725 | 0.02675941904341993 | 0.046281565424003655 | 0.05212024207102116 | -0.02635750327891157 | 0.5334003808428094 | 837/2037 |
| 5 | 0.019451164118360377 | 0.02641280257327333 | 0.04586396669163371 | 0.05245995827208497 | -0.025279943934364255 | 0.5019242805322055 | 880/2037 |
| 6 | 0.021417674355009533 | 0.026202925785856618 | 0.04762060014086615 | 0.05412477314829731 | -0.021527322377940063 | 0.47741904595489765 | 935/2037 |
| 7 | 0.022317456559058157 | 0.026989931426302263 | 0.04930738798536042 | 0.05654758661980825 | -0.020558569833835624 | 0.47050830995551457 | 965/2037 |
| 8 | 0.02410835189498855 | 0.028827188438683322 | 0.05293554033367187 | 0.060656048612148815 | -0.02192162737482587 | 0.47955372803990726 | 986/2037 |
| 9 | 0.026109758152081927 | 0.030237373244428847 | 0.05634713139651077 | 0.06111120472414804 | -0.0214116393056299 | 0.40339186017088957 | 1096/2037 |
| 10 | 0.026544956552348464 | 0.0305942174122213 | 0.05713917396456976 | 0.062017510457866895 | -0.020352980808003324 | 0.34623392293302024 | 1174/2037 |


E7→E10，训练Sampled奖励0.04930738798536042→0.05713917396456976，
Greedy奖励0.05654758661980825→0.062017510457866895，平均梯度范数仍非零；
但G10 Joint Validation反而由67.49升至69.77（攻击变弱）。
SCST loss含有随策略变化的log-prob/Greedy baseline和有符号advantage，不是应单调下降的监督Loss，
不能拿它不趋零或变化方向直接判定未收敛。

E10有1174/2037个零梯度步，E7为965/2037；
记录中这些Epoch的零parameter_delta步均0。零梯度/相同奖励既可能来自动作重合、有限动作空间，
也可能有其他原因；本轮没有逐样本归因，不能直接把这个比例当作收敛判据。
Greedy synergy均值为负，但它不是奖励项，也不能据此推翻Joint Full的总体收益。

**有证据证明：**训练按计划完成、后续三个Epoch已比较、H7为既定10候选Joint最优。
**可合理提出的假设：**奖励代理与mAP目标不完全一致、固定小Validation的波动或泛化差异，可能解释后期训练奖励与Validation不一致。
**证据不足：**训练已经充分收敛、延长训练肯定有效、已过拟合、H7对G60造成瓶颈或具体机制。
单seed/单轨迹、128规模Validation、缺少多seed置信区间和更长训练，均限制判断；
没有证据支持“继续训练到60轮必然增强”。

## 7. H与G在训练、Validation和Test时的关系

| 阶段 | G来源/状态 | H来源/输入 | 优化/选择依据 |
| --- | --- | --- | --- |
| Train图库准备 | 固定G-v2 E10；每图一次，IRRA生成Clean/Adv特征 | 无H训练参与图像生成 | 固定Train缓存 |
| H训练 | 读取G10 Adv特征，G/IRRA参数冻结，图像前向0 | seed42初始化H、Caption-only；仅H Adam | Train R_sim+R_rank、SCST |
| H选优 | 固定G10的128 Validation Adv图库 | 10个H候选，Caption-only greedy | 最低IRRA Joint Validation mAP |
| 历史G10+H7 Full | 固定G10与选中H7 | 冻结产物、Raw Caption控制 | Test只评估，不选模 |
| G60+H7 Full | 新G60冻结图像；H7历史文本不变 | 无H重新训练/生成 | 预设G Epoch消融，不构成新H选优 |

原Train、Validation协议中只有G10身份，未找到H训练/选优曾使用G60的证据；
G60参数变化不等于已经证实产生有害的特定分布偏移。
G60+H7可以合法作为固定组件组合的Test结果，但不是已被原G10 Selection确认最优的G60组合。

## 8. 当前Full对H7的支持与局限

以下为已封存Full的只读引用，单位百分数、展示两位小数；不是本轮新评估。

| Victim | Raw Clean mAP | H7 Text-only | G10 Image-only | G10+H7 Joint | G60 Image-only | G60+H7 Joint |
| --- | --- | --- | --- | --- | --- | --- |
| IRRA | 66.10 | 51.47 | 32.28 | 23.55 | 17.51 | 13.08 |
| RaSa | 69.18 | 32.86 | 35.05 | 17.20 | 26.47 | 12.57 |
| APTM | 64.13 | 51.80 | 39.66 | 30.49 | 32.11 | 23.74 |


| Victim | G10额外下降(pp) | G60额外下降(pp) | H7 Text-only相对Clean下降(pp) |
| --- | --- | --- | --- |
| IRRA | 8.72906685 | 4.42803288 | 14.62588120 |
| RaSa | 17.84432602 | 13.89636993 | 36.32547379 |
| APTM | 9.17168427 | 8.36816216 | 12.32493591 |


令M为完整检索mAP，固定H7的额外效果为：
Δ(G)=M(G(I),T)−M(G(I),H7(T))。
G发生变化后，图文匹配和排名会同时变化，M是非线性排序指标，因此固定H不意味着Δ(G)固定。
它也不是训练Reward中R_sim或synergy_increment的同义词。

现有证据支持：三Victim上G60+H7的总体mAP低于G60 Image-only，H7仍有补充效果。
现有证据不支持：差值缩小必然由H欠训练造成、样本级攻击互补/重叠已得到证明、
或二者产生某种已验证的优化冲突。

可能的替代解释仅作为假设：

- mAP有0下界，Image-only更强后可再下降的绝对空间也改变；
  Δ(G)≤Image-only mAP是简单上界，不证明实际缩小由下界主导。
- 两种攻击影响的成功/失败Query集合可能改变、重叠增加或重排；
  **没有逐Query集合和排序对照，不能从总体差值判断是哪一种。**
- H7的G10奖励/选优条件与G60输入不同，可能有适配效应；
  需要独立Validation实验证明，不能用当前Test差值确认。

H7 Text-only相对Clean的mAP下降在三个Victim上明显不同（表中给出绝对百分点）；
RaSa较大、IRRA/APTM较小值得研究。原生Tokenizer/Unicode处理、模型表示与评分不同是候选解释，
但本轮没有额外Tokenizer探针或因果实验，不能给出确定归因。
128 Validation的Text-only mAP95–97与3074图库Full不能直接横向推断训练不足。

原Full结果保存了Score缓存/来源，但本轮未加载或重新排序这些矩阵，也未核验逐Query AP/R@1成功集合。
若要验证互补/重叠，需另行授权从既有Score/Rank记录派生逐Query对照；
可能涉及CPU重新排序，不能把其算作本轮已经执行的只读标量分析，更不能因此重跑GPU。

## 9. 是否继续训练与最小额外实验（只设计，不执行）

| 问题 | 本次建议及理由 |
| --- | --- |
| H7是否有合理选优依据 | 有，10候选/G10/IRRA/128 Validation/Joint mAP，范围明确 |
| 是否有充分训练/收敛证据 | 有完成既定10 Epoch的证据，没有证明已收敛或更长训练无益的证据 |
| 现在是否继续训练H | 暂不优先；先利用已存在E8–10和全部旧Validation数据 |
| 是否做H Epoch消融 | 已有E1–10的G10 Validation曲线数据可直接分析；值得补充G60下的同候选Validation，暂不做Test选优或全量H-Test sweep |
| 是否做G Epoch×H Epoch网格 | 先做G10/G60×现有10个H候选的内部Validation；不建议立即6×10×3个Test Full |
| 最小额外计算 | 新G60内部Validation图像特征128张，复用10×128旧文本特征；不训练H，不用Test反馈 |

### 阶段1：已有证据优先，本轮已完成

全部10个候选已比较，Text-only和Joint均存在，所以“只有Joint Validation、没有Text-only”的假设不成立。
无需为了检查E8–10而继续训练；同样不能把只保存未验证的周期权重说成已知优于H7。
如确需增密候选，应事先固定周期候选范围，先在内部Validation进行；新增H策略/IRRA文本编码需要单独授权。

### 阶段2：最小G60 Validation接线，等待授权

固定已有128内部Validation图像、原始Caption和PID；G60按预设研究计划固定，不根据Test数值调参。
图像输入严格复现旧Validation的PIL RGB→bilinear Resize384×128→ToTensor，验证原图SHA；
G60输出保持FP32且不转回PIL，不使用官方Test的物化Gallery代替Validation。
新独立输出目录/配置/SHA和协议引用原G10缓存，不覆盖旧Selection、H7或G60。

可严格复用的资产：

- 原Clean图像特征128条和原文文本特征128条。
- 10候选各128条H Validation文本/动作/已编码IRRA特征。
  H的Caption-only推理与G无关，所以在模型/Tokenizer/身份/SHA一致时无需再次运行H或文本编码。
- 原G10 Joint及Text-only结果，用于两行比较。

必要新增量：G60图像前向128次＋IRRA Adv图像编码128次；
10组128×128 Joint相似度约163840个标量位置，外加一次Image-only控制及CPU排名。
H前向=0、新文本编码=0、训练/Optimizer=0。
512维FP32新图像特征纯数据约256KiB；10候选文本特征纯数据约2.5MiB，metadata/SHA另计。
依现有IRRA/G实测，顺序加载自身显存约1–2GiB量级、含恢复/SHA/文件I/O预计分钟量级，
仅是执行预估；实际门禁、代码接线和资源需在后续授权时核验，不能作为容量保证。
本轮未实现或执行该接线，也不提供可误触发GPU的运行命令。

先报告10候选在G60下的完整Text-only/Joint四指标、选择是否稳定和差异量；
同一128 Validation重复使用的选择偏差需如实披露，不把它等同独立新数据验证。
暂不新增RaSa/APTM Validation编码；若后续研究目标要求其参与，需各自原生编码，绝不能借用IRRA特征空间。

### 阶段3：只有必要时再讨论H续训

若独立Validation仍有预设、可检验的学习长度问题，再先审计H10完整Adam/RNG恢复；
正式CLI绑定epochs=10，直接--resume --epochs60会被拒绝，不能改旧协议或跳过SHA。
严格连续延长优先父H10，而非把H7后的已完成轨迹隐藏掉。
新扩展身份、父SHA、允许变更字段、样本顺序及恢复验证均需单独设计和授权。

若把奖励图库由G10换成G60，是训练条件变化/再适配研究，不再是单一H训练轮数延长；
应单独标注协议，保留原奖励数学定义与旧资产，不把两条轨迹拼成“原H连续训练”。

### 阶段4：正式Test只作最终冻结比较

在预先确定的Validation选择规则下固定H，再另行申请一次最终三Victim比较。
当前G60+H7 Full永久保留，不反复用其Test mAP调整H训练长度，不用Test选择新Checkpoint。
不存在当前已获授权的新训练/GPU/Full任务。

## 10. 源码与来源索引

| 源码（本次SHA与Git快照逐字节一致） | 证明的机制 |
| --- | --- |
| [policy.py:59–164](https://github.com/w-harda/idea-test/blob/20f61ae5471ae68081b6b80419baba8703fd1744/prototypes/attribute_text_policy_h_v1_20261015/policy.py#L59) | Byte Transformer、三Head、Dynamic-K/STOP/NOOP、greedy测试 |
| [text_frontend.py:28–50](https://github.com/w-harda/idea-test/blob/20f61ae5471ae68081b6b80419baba8703fd1744/prototypes/attribute_text_policy_h_v1_20261015/text_frontend.py#L28) | Caption-only Canonical/原Span |
| [constraints.py:6–89](https://github.com/w-harda/idea-test/blob/20f61ae5471ae68081b6b80419baba8703fd1744/prototypes/attribute_text_policy_h_v1_20261015/constraints.py#L6) | 视觉相似Unicode、每属性/共享词一次、Tokenizer合法候选 |
| [scst.py:6–14](https://github.com/w-harda/idea-test/blob/20f61ae5471ae68081b6b80419baba8703fd1744/prototypes/attribute_text_policy_h_v1_20261015/scst.py#L6) | SCST=-detach(R_sample−R_greedy)*log_prob |
| [rewards.py:62–82](https://github.com/w-harda/idea-test/blob/20f61ae5471ae68081b6b80419baba8703fd1744/prototypes/attribute_text_policy_h_v1_20261015/rewards.py#L62) | Adv Gallery R_sim/R_rank与Clean诊断 |
| [phase4_training.py:38–79](https://github.com/w-harda/idea-test/blob/20f61ae5471ae68081b6b80419baba8703fd1744/prototypes/attribute_text_policy_h_v1_20261015/phase4_training.py#L38) | 当前Sampled/Greedy、H-only Adam及逐步日志 |
| [formal_training/training.py:16–43](https://github.com/w-harda/idea-test/blob/20f61ae5471ae68081b6b80419baba8703fd1744/prototypes/attribute_text_policy_h_v1_20261015/formal_training/training.py#L16) | 10 Epoch配置、2048成员、seed+epoch顺序与反馈隔离 |
| [formal_training/training.py:79–130](https://github.com/w-harda/idea-test/blob/20f61ae5471ae68081b6b80419baba8703fd1744/prototypes/attribute_text_policy_h_v1_20261015/formal_training/training.py#L79) | visit/step/skip、完成条件、Adam/RNG恢复证据 |
| [formal_training/runtime.py:65–122](https://github.com/w-harda/idea-test/blob/20f61ae5471ae68081b6b80419baba8703fd1744/prototypes/attribute_text_policy_h_v1_20261015/formal_training/runtime.py#L65) | 同PID正例、精确配对、冻结Adv奖励与三文本实时编码 |
| [formal_training/validation.py:16–32](https://github.com/w-harda/idea-test/blob/20f61ae5471ae68081b6b80419baba8703fd1744/prototypes/attribute_text_policy_h_v1_20261015/formal_training/validation.py#L16) | 官方rank与固定128图文四组 |
| [formal_training/validation.py:72–114](https://github.com/w-harda/idea-test/blob/20f61ae5471ae68081b6b80419baba8703fd1744/prototypes/attribute_text_policy_h_v1_20261015/formal_training/validation.py#L72) | Validation来源、所有Query保留、H冻结 |
| [formal_training/validation.py:146–155](https://github.com/w-harda/idea-test/blob/20f61ae5471ae68081b6b80419baba8703fd1744/prototypes/attribute_text_policy_h_v1_20261015/formal_training/validation.py#L146) | 最低Joint mAP、最早visit并列规则 |
| [formal_training/cli.py:188–250](https://github.com/w-harda/idea-test/blob/20f61ae5471ae68081b6b80419baba8703fd1744/prototypes/attribute_text_policy_h_v1_20261015/formal_training/cli.py#L188) | 训练正常完成回执、全部Epoch候选与选优入口 |
| [formal_training/checkpoint.py:45–69](https://github.com/w-harda/idea-test/blob/20f61ae5471ae68081b6b80419baba8703fd1744/prototypes/attribute_text_policy_h_v1_20261015/formal_training/checkpoint.py#L45) | 不可变周期Checkpoint保存policy/Adam/RNG/来源 |


派生文件：

- [全部候选原精度CSV](results/hv1_epoch7_training_selection_audit/validation_candidates_raw.csv)；
- [逐Epoch日志标量原精度](results/hv1_epoch7_training_selection_audit/epoch_training_statistics_raw.csv)；
- [81份Checkpoint清单](results/hv1_epoch7_training_selection_audit/checkpoint_inventory.csv)；
- [固定跳过样本](results/hv1_epoch7_training_selection_audit/skipped_samples.csv)；
- [既有Full引用](results/hv1_epoch7_training_selection_audit/full_reference_raw.csv)、
  [额外下降原精度](results/hv1_epoch7_training_selection_audit/full_increment_raw.csv)；
- [来源/Selection/完成/冻结/保护证据](results/hv1_epoch7_training_selection_audit/SOURCE_EVIDENCE.json)。

本次只提交新增派生报告与证据索引，不修改任何旧训练/Validation/Selection/Test资产或历史框架源码，
不重新执行历史CPU审计；完成后停止，等待用户决定下一阶段。
