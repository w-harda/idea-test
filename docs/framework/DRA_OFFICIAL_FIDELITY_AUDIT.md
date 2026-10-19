# DRA官方原始算法保真审计

## 1. 官方来源与完整性

- 仓库：https://github.com/SensenGao/VLPTransferAttack
- 固定Commit：91f78325e7cf0c497c9bbb5be2215be652d8cd9d
- 本地只读副本：/home/lzf/ldx/projects/VLPTransferAttack-official
- RAttacker.py SHA-256：75518cb9465d9f8e994ea2f59c86b89fd4c39e3fe74f66b7a1d0a090eec2d313
- 开发前、各会话预检、成功Source结束及最终核验均一致。官方仓库git diff/status为空。
- 未复制攻击算法到项目；未修改官方RAttacker.py；未继承、覆盖或Monkey Patch官方Attacker/ImageAttacker/TextAttacker任何方法。

## 2. 真实调用路径

dra_irra.source_run → modules加载上述SHA文件 → official.Attacker(source,image,text) → attacker.attack。
实际入口前以__func__ is official.Attacker.attack等检查三个原绑定方法身份。
官方Attacker内部仍按text1→image→text2调用，Runner没有另一套攻击循环。
成功Source Commit 8513a49b6f53c63ea98a8f6795b4e659b8fb88de与来源路径完整保留；随后只修Target协议和CPU审计import，不重新生成攻击。

## 3. 外部Wrapper和观测器

| 接线 | 功能 | 攻击计算 |
|---|---|---|
| IRRADRASource | 继承项目已验证的IRRASGASource I/O，开启原生IRRA参数梯度 | 无Loss、迭代、采样或候选决策；不是继承官方攻击类 |
| IRRASGASource.inference_image | 原生encode_image + F.normalize | 同原官方双塔接口的单位全局特征语义 |
| IRRASGASource.inference_text / AuditedBert / decode_bert | 官方BERT输入保真解码，再原生IRRA Byte-BPE/encode_text/F.normalize | 不改MLM候选集合和分数；额外上下文截断显式拒绝 |
| PassiveTrace | sys.settrace只读官方frame，记录函数进入/返回、官方局部计数、标量梯度与内存 | 不替换任何函数、参数或返回值，不消费随机数，不保留历史Tensor图 |
| MLM forward_pre_hook | 仅增加实际调用/样本整数计数 | 不修改输入输出或梯度 |
| ArtifactStore / FeatureCache | 紧凑Tensor clone、Claim、来源SHA、发布与读回 | 攻击结束后的存储 |
| 原生Target Adapter/Scorer | 独立模型评估冻结产物 | 不进入Source生成 |

继承的IRRA I/O模块保持原SHA。仅复用sga_irra.manifest_from_dict恢复数据类；不调用SGA Source Run或其OfficialObservation。
CPU开/关PassiveTrace后，真实官方合成ImageAttacker输出Tensor与Python/NumPy/Torch RNG exact；追踪异常退出也恢复sys.gettrace。
此项是CPU等价证据，不宣称GPU双运行bitwise复测。

## 4. 官方eval.py与实际参数

| 项目 | 官方实际调用 | 本次实际 |
|---|---|---|
| 入口 | RAttacker.Attacker.attack | 同一原函数 |
| eps / steps / step_size | 8/255 / 10 / 2/255 | 相同 |
| sample_numbers | 5 | 5 |
| 区域比例 | 原rand3Num整数三元组，总和100 | 原方法实际调用45次 |
| 多尺度 | 原图、.5/.75/1.25/1.5；原高斯噪声/BICUBIC | 原get_scaled_imgs实际调用10次 |
| Batch / Seed | 8 / 42 | 8 / 42；不按样本重设 |
| cls | False | False |
| 文本候选 | MLM topk10、threshold .3、原过滤/词重要性 | 原方法，未新增规则 |
| number_perturbation | 1 | 1个原词槽位；不强加空格词数不变 |
| text_ratios | clean / final / previous=.6/.2/.2 | 相同 |
| pre_caption | max_words30 | 相同官方函数 |
| TextAttacker及Attacker max_length | ALBEF/TCL30、CLIP类77 | IRRA按对齐模型接口固定77 |
| 图像特征/文本特征 | 各Source单位全局特征 | IRRA原生512维单位全局特征 |
| Source精度 | model.float，eval | FP32/eval，无AMP；IRRA约定关闭TF32 |
| 更新/投影/候选选择 | 官方代码 | 原样，无额外投影/STE/近似 |

ImageAttacker构造默认2/255不是正式入口的8/255，本次未误用构造默认。
第一轮因无历史轨迹不执行区域采样，后9轮各5候选，共45；没有为了“每轮5”去改官方首轮。
原候选评分的二次backward全部保留；全批共100次图像backward，不删除看似冗余的反向。

## 5. 自动求导与模型不更新

CPU真实调用官方ImageAttacker：将Source所有参数requires_grad=False会在候选评分反向阶段失败。
IRRADRASource仅把backend参数requires_grad设True，保持eval。官方model.zero_grad/loss.backward原样执行，无Optimizer或更新。
成功GPU Source观察55个非零且有限图像梯度，152份参数有梯度；官方攻击结束后统一清理梯度、关闭参数开关。
源IRRA及MLM的全部参数/buffers前后SHA exact。参数参与求导不等于参数被训练。
区域选择后allocated约1.535–1.538GiB，无9轮持续累积；不据此承诺全量所有批次稳定。
三Target各自独立会话eval/requires_grad=False、无Backward/Optimizer，前后参数/buffers exact。

## 6. 文本候选和特殊Token

官方Tokenizer文件与已验证SGA版本逐字节相同，SHA 7723a9acc1a44bfe0eb435cb9d49a36677700e87cf572e7347329763c41cdfbf。
保留真正内容中的[UNK]/候选Token，仅移除实际开头CLS和attention padding；逐行检查WordPiece decode→encode exact。
BERT77与IRRA77分开检查；不把BERT ID当IRRA ID，不静默截断或过滤额外候选。
成功运行1026桥接行全部exact、BERT截断0、IRRA额外截断0；756行有原词重要性[UNK]。
实际内容中内部CLS/SEP/PAD/MASK均0。官方原CLIP Wrapper会字符串删除CLS/SEP/PAD并允许CLIP截断；本IRRA桥显式保存内容、遇额外截断拒绝。这一I/O差异已披露，当前Smoke未触发内部特殊Token/额外截断差异；未来遇到不能往返的候选必须停止。

## 7. 原1词预算的准确含义

官方get_bpe_substitues可以把一个原词槽位换成含空格短语。真实样本cuhk:37134:1：leggings→green and wearing；原change计数1。
不能把这条合法候选删掉来满足新加的词数约束。
最初外部审计错误地拒绝空格词数变化，v2 FAIL完整保留。派生修复只在原final_adverse.append前读取change、words、final_words。
最终16条change=1，每条只改1个原槽位；1条空格词数增加2。验证原槽位与最终槽位后join恰为官方返回文本。
这是纠正外部审计的解释错误，不是放宽或修改官方候选规则。

## 8. 完整真实阶段和固定产物

成功Source：Attacker.attack1、img_guided_attack2、txt_guided_attack1、rand3Num45、get_scaled_imgs10、image backward100。
T1编码结果与图像攻击收到的监督逐值exact。最终文本调用收到原图、最终Adv、上一轮Adv三组真实特征，原.6/.2/.2在官方函数内部执行。
实际8图/16文，1144图像样本/1026文本样本，MLM4调用/80样本。
FP32原RGB、Adv RGB、effective_delta保存并bitwise读回。官方FP32投影上界通过，实数超8/255约2.793237274811844e-8，未做额外投影，不声称严格实数预算PASS。
同一已完成Source Collection被三个Target引用，来源SHA/order/content绑定；RaSa/APTM未参与攻击生成、无反馈、无重新攻击。

## 9. 相对原论文原实验的必要差异

- Source改为官方IRRA/CUHK权重；不是原论文ALBEF/TCL/CLIP权重的数值复现。
- 数据改为CUHK Test3074/6156，多Caption数量可变；只保留每图annotation配对，不扩大同PID集合。
- Source像素为IRRA原生PIL bilinear384×128；不是原CLIP方形crop。官方后续增强/Loss/更新不变。
- 使用IRRA原生Byte-BPE和512维单位特征，BERT MLM及原词替换算法不变。
- Target评分为三个模型原生TBPS协议，保留同PID多正例并计算mAP；不照搬Flickr单图匹配ASR。
- Smoke只有8图，因此max_rank8且R@10不适用；Full引擎门禁没有放宽。
- Python被动追踪有显著CPU开销；没有为加速改算法或近似计算。

这些差异属于已授权Source/Data/Evaluation I/O移植；不能称原论文原模型/原数据集数值复现。
VLPTAttack与G-v2+H-v1 Joint可作为各自原生配置下的方法级Baseline比较；披露配对访问、词/Unicode预算、训练及推理开销，不为对齐而修改原方法。

## 10. 结论、失败记录与限制

OFFICIAL_ALGORITHM_FIDELITY_PASS：原RAttacker.py字节、类方法与核心数学流程未改变，仅通过外部接口接入IRRA/CUHK。
该状态限定固定8/16 Smoke，不代表Full或所有Token边界通过。原论文性能、全量上下文、10-Caption batch GPU峰值、长期资源趋势和3074/6156 Full指标均PENDING。
完整CPU审计PASS、三Target评分PASS，未启动Full。旧准备/保存/目标协议/CPU import FAIL均保留，不改写成PASS。
最初失败Source官方攻击1次，最终成功Source官方攻击1次，总共2次；只有最终产物用于三Target。详见DRA_SMOKE_REPORT.md与DRA_SMOKE_EVIDENCE.json。
