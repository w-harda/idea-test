# SGA官方算法复现审查与最小CUHK-PEDES接线方案

## 结论与历史归类更正

此前SGA-TBPS White-box的Full数值和缓存审计有效，但**不是固定官方SGA算法的严格复现**。复用官方get_scaled_imgs不足以构成完整SGA；目标权限、图文流程、部分损失、预算及数值执行方式已经改变。它应保留为“额外白盒适配实验”，不再占据正式原始SGA Baseline位置。原源码、6f89b799922ac54f041dabd3cd8e9d444735b146、原Full、报告/CSV、FAIL与独立CPU审计均原样保留。

本项目的主方法为**G-v2 + H-v1 Joint**。新官方SGA的目标是算法忠实、可解释的源模型黑盒迁移，不是进一步降低旧白盒指标。当前官方SGA状态：**NOT_REPRODUCED / 静态审查完成，接口与验证尚未实施**。既有白盒PASS不得继承为官方算法PASS。

固定官方源码：Zoky-2020/SGA，Commit f49e6926cd13148e49bdbc2c3e537660e096230c；服务器/home/lzf/ldx/projects/SGA-official，HEAD一致、工作树干净。本文以该Commit实际调用路径为准，结合[原论文](https://arxiv.org/html/2307.14061v1)的攻击设置核对。

本轮仅源码/配置/文件SHA、模型文件头及只读annotation审查、公开链接HEAD探测、文档归类更新；没有单元测试、模型恢复、前向、CUDA初始化、GPU、Full或权重下载。模型可加载性和算法运行等价尚未验收。

## 1. 官方完整真实调用链

[attacker.py:17–38](https://github.com/Zoky-2020/SGA/blob/f49e6926cd13148e49bdbc2c3e537660e096230c/attacker.py#L17)的SGAttacker.attack依次执行：

1. 原图经过source的normalization及inference_image，按txt2img取每条原Caption对应的image_feat。
2. 第一次TextAttacker.img_guided_attack(source, 原Caption, 原图监督)，得到临时攻击文本T1。
3. 用source文本接口编码T1，以其text_feat监督官方ImageAttacker.txt_guided_attack，产生攻击图I_adv。
4. 编码I_adv，再次调用TextAttacker.img_guided_attack(source, **原Caption**，攻击图监督)，返回最终T_adv。
5. 返回I_adv与最终T_adv；后续目标评估只消费这些固定产物。

关键：最后文本阶段从原Caption重新开始，不从T1累计编辑；number_perturbation=1约束最终文本最多替换一个官方分词意义下的词，不是两次累积两词。原始SGAttacker调用的是img_guided_attack两次，不能以另一个TextAttacker.attack函数替代；后者候选评分路径不同。

原图/Caption配对关系是SGA的必要输入。目标模型黑盒不等于数据配对未知：SGA使用Test原图对应Caption指导文本攻击；H-v1测试Caption-only不使用正确配对图像。这一权限差异必须保留在与Joint的比较说明中，不能悄悄把SGA改成Caption-only。

## 2. 官方算法、损失与实际参数

### 图像分支

[ImageAttacker:43–106](https://github.com/Zoky-2020/SGA/blob/f49e6926cd13148e49bdbc2c3e537660e096230c/attacker.py#L43)：
- eps=2/255，steps=10，step_size=0.5/255；[实际ALBEF脚本:43–47](https://github.com/Zoky-2020/SGA/blob/f49e6926cd13148e49bdbc2c3e537660e096230c/eval_albef2tcl_flickr.py#L43)明确采用这些值，不能只看YAML字段后自行改参。
- NumPy uniform随机起点并clamp到[0,1]；每轮按各尺度累计原loss，loss.backward，取adv_imgs.grad。
- loss为source原生image_feat/text_feat的匹配dot之负值：每图正Caption求和，再对图像batch求mean；不是ITM分类logit。
- 梯度按每张图mean(abs(grad))归一化，sign上升，投影到原图±eps，再clamp[0,1]。源模型不执行optimizer.step。

[get_scaled_imgs:110–132](https://github.com/Zoky-2020/SGA/blob/f49e6926cd13148e49bdbc2c3e537660e096230c/attacker.py#L110)和[官方脚本默认scales](https://github.com/Zoky-2020/SGA/blob/f49e6926cd13148e49bdbc2c3e537660e096230c/eval_albef2tcl_flickr.py#L356)：附加0.5/0.75/1.25/1.5及原图1.0共五尺度；附加分支Gaussian std0.05、BICUBIC缩放、clamp、BICUBIC逆缩放。不得改成普通PGD、删噪声或加额外Loss/动量。

官方ALBEF/TCL源模型的inference_image/inference_text返回其自己的单位化投影全局特征，参见models/model_retrieval.py:166–178。**融合模型也不代表SGA攻击必须使用ITM Loss**：原算法的对齐攻击目标与最终融合检索评分是两层不同职责。

### 文本分支

[TextAttacker:165–265、382–403](https://github.com/Zoky-2020/SGA/blob/f49e6926cd13148e49bdbc2c3e537660e096230c/attacker.py#L165)：
- 官方评估实例cls=False，number_perturbation=1，topk=10，threshold_pred_score=0.3，候选与词重要性批大小默认32。
- ref_net为bert-base-uncased的BertForMaskedLM；MLM logits产生候选。阈值0.3比较的是原始logit，不应擅自解释成概率阈值。
- 逐词[UNK]遮蔽，使用source文本特征分布的KL变化排序词重要性；候选替换按与对应图像特征的负dot选argmax。
- 原stopword/filter_words、##过滤、原词候选、BPE候选合成及PPL排序全部保留；代码中注释掉的反义词/语义过滤不能作为已启用约束，也不能自行加入。
- 这是词替换策略，不是H-v1视觉相似Unicode字符替换；不得以H的候选集、属性Head或Dynamic-K替代。

ALBEF源TextAttacker和中间文本编码长度30。CLIP源TextAttacker配置长度77，但官方脚本传给SGAttacker的是拼错的max_lemgth=77，被**kwargs吸收，SGAttacker的max_length仍为默认30。这是已确认的固定Commit实际行为：**若选择CLIP，不能无声明修成77**；“忠实该Commit”与“纠正上游疑似笔误”必须分开。推荐先选ALBEF避免此分支差异，未修改上游文件。

官方脚本默认image batch=8、seed=42，按完整数据原顺序、一次设置Torch/NumPy/Python全局seed；新复现不沿用旧实现的每图哈希重置seed或逐尺度microbatch。官方脚本source.float().eval()，不更新源权重；未来应冻结原依赖环境/实际运行设置，不能把旧TBPS环境强行标为原始环境。

## 3. 与此前白盒适配的逐项差异

| 项目 | 固定官方实现 | 历史SGA-TBPS White-box | 判断 |
|---|---|---|---|
| 模型角色 | ALBEF/TCL/CLIP源模型生成，再迁移目标 | IRRA/RaSa/APTM各自生成 | 攻击权限/实验设置改变 |
| 完整流程 | 原文→T1→I_adv→T_adv | Clean Caption监督，仅图像攻击 | 算法流程改变 |
| 图像监督文本 | 第一阶段攻击文本T1 | 固定Clean文本缓存 | 跨模态引导改变 |
| 图像Loss | source全局feature匹配负dot，batch mean | IRRA负dot；RaSa/APTM新增ITM负logit | 后两者目标已改变；IRRA同型也不等于完整SGA |
| 文本Loss/策略 | KL词重要性＋图像引导MLM候选词替换 | 未执行TextAttacker | 缺失原算法组成 |
| eps | 2/255 | 8/255 | 攻击预算改变 |
| steps/step_size | 10 / 0.5/255 | 10 / 0.5/255 | 数值相同，但不足以证明算法一致 |
| 五尺度/噪声 | 官方get_scaled_imgs | 直接复用该函数 | 这部分保留 |
| 更新代码 | 官方loss.backward和原投影 | 自写autograd.grad/增强VJP、FP64边界内舍入 | 数值执行路径已改写，不属于纯格式转换 |
| batch/RNG | 脚本batch8、全局seed42推进 | batch1、每图hash派生seed并恢复外层RNG | 调度与随机序列不同 |
| 图像工作空间 | ALBEF384×384；CLIP原生Resize/CenterCrop | 既有384×128物化RGB | 预处理/预算空间不同 |
| Caption处理 | pre_caption；原词替换 | Clean原文始终不改 | 输入及文本预算不同 |
| 最终指标 | Flickr/COCO检索与基于clean正确集合的ASR | TBPS全Query、多PID正例的四指标 | 必要任务评估转换，需明确标注，不能假称论文原表 |
| 目标模型反馈 | 只在最终黑盒评估时读取目标Score | 梯度用于目标专属攻击 | 不能把历史结果重标成Transfer |

历史对应位置：src/tbps/baselines/sga_attack.py:13、26–35、45–80，src/tbps/models/sga_surrogate.py:6–9、38–55，以及sga_whitebox.py::generate/run。旧源文件在其原工作树保持不变；本表是分类说明，不是对旧文件补丁。

## 4. 官方源模型资源可用性

| 资源 | 实际核验 | 当前状态 |
|---|---|---|
| 固定SGA源码 | 本地HEAD=f49e692…、Git干净，93个tracked文件SHA记录 | 已具备，不导入执行 |
| CLIP ViT-B/16 | /home/lzf/.cache/clip/ViT-B-16.pt，350837078 bytes，SHA与官方下载URL内值完全一致 | 本地字节已核验，模型恢复/运行PENDING |
| bert-base-uncased | 本地snapshot86b5e093…含config/vocab/tokenizer与model.safetensors；config为BertForMaskedLM，文件头含MLM预测Head，共206 Tensor | 本地文件/头部具备，加载兼容性PENDING |
| ALBEF Flickr30k Retrieval | 官方直链HTTP HEAD 200，Content-Length838449862；限定资源目录未发现本地文件 | 公开资源可获得，未下载/未加载，SHA尚UNKNOWN |
| TCL Flickr Retrieval | 官方README给出Google Drive发布页；限定目录未发现本地权重，服务器HEAD超时 | 下载可用性未确认，不阻断只选ALBEF的最小方案 |
| 原requirements环境 | 当前tbps-image为Torch2.5.1、Torchvision0.20.1、Transformers4.44.2、timm0.9.8、ftfy6.3.1；未找到ruamel_yaml发行元数据/模块候选 | 不等同官方环境，须单独解决后再验证 |

已核验CLIP SHA-256：
5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f。
BERT model.safetensors SHA-256：
68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3。
BERT仅解析safetensors JSON头并读取文件SHA，没有模型反序列化或前向。HF在线config HEAD超时不否定本地已存在的snapshot；未自动重试。

官方资源：
- [ALBEF官方README](https://github.com/salesforce/ALBEF#download)及[公开Flickr30k检索权重](https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/flickr30k.pth)。
- [TCL官方README](https://github.com/uta-smile/TCL#pre-trained-checkpoint)及其Drive发布页。
- [SGA自带CLIP下载表](https://github.com/Zoky-2020/SGA/blob/f49e6926cd13148e49bdbc2c3e537660e096230c/models/clip_model/clip.py#L29)。

应使用SGA仓库自带的模型I/O实现。不能用IRRA的CLIP分支、RaSa训练权重，或HF通用CLIPModel替代后称其与官方SGA完全一致。现有官方ViT-B/16文件可经原clip.load绝对路径只读加载；不必改权重或触发默认缓存下载。

推荐源模型：**ALBEF＋官方Flickr30k检索Checkpoint**，在看CUHK攻击指标之前固定。它不裁去行人上下区域，也没有CLIP分支中间max_length歧义。备选CLIP只基于资源和接口条件选择，不能根据三个Test攻击结果挑源或挑权重。本轮不下载/安装；ALBEF后续建议独立保存至/home/lzf/ldx/checkpoints/SGA-official/albef_retrieval_flickr.pth（路径尚未创建）。

## 5. CUHK-PEDES能否直接接入：可接线，不能直接照搬旧Full入口

### Caption集合与身份

官方dataset.py::paired_dataset接收每图任意长度caption列表，collate后构造txt2img，没有硬编码每图五Caption。CUHK Test实际3073张图各2 Caption，另1张cuhk:38920:0有10 Caption，共3074/6156。可以只转换annotation字段image/caption、原路径及顺序；保持全部Caption、全部Query，不能补齐到5、只取首句、按PID扩充集合或按效果筛选。

生成一次source攻击后，每张Gallery仅一张Adv、每条Query仅一个最终T_adv；三个目标共享同一套source产物。PID及全部同PID正例只用于最终TBPS指标；SGA必要的txt2img保持原图Caption配对，不能被同PID标签替代。

### 文本预处理是实质输入变化

[dataset.py::pre_caption](https://github.com/Zoky-2020/SGA/blob/f49e6926cd13148e49bdbc2c3e537660e096230c/dataset.py#L12)会小写、去部分标点、拆连字符/斜杠、清理空格、最多保留30词。纯JSON/regex审查发现：6156条中6137条字符串会改变，876条发生词级截断；这些不是Tokenizer前向测试。

必须保存raw Caption、pre_caption后的source clean文本、T1（如可无副作用观测）与最终T_adv，区分预处理差异与MLM词替换。直接把返回的归一化/截断字符串与此前raw Clean指标比较，会混入非攻击因素。不得无声明取消pre_caption、把长尾补回、以原Span重写文本或让target Tokenizer参与候选选择。

### 图像与目标输入桥需要预先冻结

ALBEF官方source为PIL RGB→BICUBIC Resize384×384→ToTensor；CLIP为原生Resize→CenterCrop224→ToTensor。两者均不同于旧generation_v3的384×128 RGB。CLIP裁剪可能丢失行人上下文，故本方案优先ALBEF整图源输入。

官方不同transfer脚本的target桥也不同：ALBEF→TCL含ToPILImage/Resize/ToTensor，CLIP→CLIP可直接Tensor Resize/CenterCrop；不存在可凭空套用给三个TBPS模型的唯一官方桥。IRRA/APTM需要384×128，RaSa需要384×384。新接线须在生成前固定三目标原生预处理及映射，并对Clean/Adv使用**同一条桥**。

保留source原始FP32 clean/adv及2/255来源预算；额外记录target桥后像素/内容身份、插值/量化及实际差异。BICUBIC与量化不保证源空间L∞在目标空间不变；不能静默二次投影、改成8/255或把feature SHA冒充像素SHA。旧G/H MaterializationPlan不应为兼容SGA而改签。

若为共享历史Raw Clean而新增“只映射delta再加回旧原图”、改成bilinear或回填截断尾部，这都是额外输出传递规则，**不是零差异格式转换**；需单独说明和批准，不能自动加入本复现。

### Clean及Joint比较

最忠实的最小验证先采用官方source文本/图像处理，目标对source clean与source adv执行相同冻结桥，并只把“实际词替换/扰动”作为攻击增量。这一Matched Clean可能不同于历史Joint的Raw Clean，因此**本阶段不能承诺复用历史66.0974等Clean或直接把新数值拼入旧表**。

与G-v2＋H-v1 Joint做主表之前，须冻结共同输入和Clean约定；若不能证明两条输入链等价，先分协议报告Matched Clean与Joint历史Raw Clean，不冒充公平横比。预处理-only检查用于排除虚假收益，不用于选择更强的source/预算/候选。已有Joint结果及物化产物保持不变。

官方Flickr/COCO的std_eval_idx不能迁移到CUHK。使用已有原生TBPS Engine计算完整6156 Query的R@1/R@5/R@10/mAP，是允许的任务评价转换；不照搬原ASR的clean正确子集筛选，不把新TBPS表称为论文原数据表复现。

## 6. 允许的接口接线与禁止的算法改变

允许：annotation字段/索引转换、原生source模型接口/Tokenizer数据结构、Checkpoint路径绑定、不可变输出和目标原生预处理/评估桥。任何预处理影响都需显式来源和Matched Clean。

禁止：新增ITM/Push-Pull/属性Loss；替换ImageAttacker/TextAttacker循环；删除两段文本；修改eps/步长/轮数/词预算；引入Unicode/属性Head；改变candidate过滤；偷偷修CLIP max_length或词索引；重设每图RNG；针对目标分数/梯度选字符、选源模型或更新攻击。融合target评分不能反传进source攻击。

仅增加一个薄source调用入口和一个CUHK/产物桥即可，无需重建框架。未来代码调用原SGAttacker/ImageAttacker/TextAttacker本体，不调用旧sga_attack.attack_image或SGASurrogate。ArtifactStore和三目标原生Adapter/Engine可复用，旧G/H Store及旧白盒Store不改；Source攻击会话与target评估会话隔离，避免legacy模块冲突及目标反馈。

## 7. 最小可行复现与验收顺序（尚未执行）

1. **先冻结方案**：源ALBEF、公开Flickr30k权重、原source preprocess、原pre_caption、batch8、seed42、上述官方参数及独立输出命名；目标IRRA/RaSa/APTM仅评估。另明确Matched Clean及未来Joint同协议比较办法。
2. **准备资源**：后续授权后只下载公开ALBEF权重，保存实际SHA及发布来源；复用本地BERT snapshot，建立隔离依赖环境，不改现有TBPS运行环境。原attacker.py及model/tokenizer算法文件SHA保持。
3. **只写薄I/O**：将官方Test有序成员送入SGAttacker.attack，保留source模型原feature、MLM及所有默认策略；保存source RGB、最终文本及raw/preprocessed身份，三个目标共享引用。
4. **确认后CPU测试**：比较官方原生调用与薄入口的真实调用顺序/参数、两段img_guided_attack、T1指导图像、T2从原文本开始、可变2/10 Caption组、所有成员和预算声明；验证目标不得反馈及旧资产不变。不能仅证明get_scaled_imgs一致就判SGA一致。
5. **另行授权最小GPU Smoke**：建议固定已有两个不同PID样本cuhk:37132:0、cuhk:37147:0及各自两条Caption，共2 Gallery/4 Query；DataLoader batch8在两图子集自然形成尾批2，不为Full改batch。source仅恢复一次，对原官方路径与薄入口作固定RNG的独立输出对照，再将同一产物送三个冻结target原生评估器。实际MLM/候选文本前向很多，不能把4 Query误报成4次文本前向。
6. 记录原图/Adv、词替换、归一化与截断、模型/Tokenizer/依赖SHA、真实调用和资源；任何不一致或OOM停止，不调Loss、预算、候选或batch来通过验收。
7. **当前不提供可冒充已实现的Full命令**；所有一致性验收通过、输入桥和共同Clean确认后，用户再授权正式3074/6156运行。生成一次，迁移三目标，不重新训练任何网络。

## 8. 与主方法Joint的必要说明

| 维度 | 官方SGA-on-CUHK计划 | G-v2 + H-v1 Joint |
|---|---|---|
| 模态 | 原图文交替攻击 | 图像生成器＋字符策略 |
| 源/训练模型 | 固定公开ALBEF；或另行声明CLIP/TCL | IRRA训练来源 |
| 目标权限 | 三目标黑盒，生成时零目标反馈 | 三目标消费冻结产物；IRRA同来源，另两目标迁移 |
| Test配对访问 | 文本阶段使用对应原图/Caption | H测试Caption-only，无正确图像/PID反馈 |
| 图像预算 | 官方2/255，source canvas | 8/255，既有RGB工作空间 |
| 文本预算 | 每次img_guided最多1词，最终从原文本重启 | Dynamic-K、每属性词至多1个Unicode字符 |
| 在线成本 | 迭代source梯度及大量MLM/候选编码 | 冻结G/H生成，目标评估不优化 |
| 数据/模型训练 | 不重新训练source，公开VLP权重 | 已有CUHK Train及内部Validation训练/选优 |

二者可作为图文攻击方法比较，但必须声明预算、配对信息、训练来源、编辑规则、输入桥和成本差异；不能通过扩大SGA eps、放宽词预算或新Loss来追求最低mAP。目标是可信复现，不是抹平原方法与本研究的设计差别。

## 9. 本轮保护与停止状态

93份官方tracked文件、旧白盒执行源码/配置、既有Full关键协议/结果及日志、本地CLIP/BERT与annotation共145份读取来源SHA/stat登记；原Full和首次FAIL共67380份文件只读stat清单保持。详见SGA_OFFICIAL_RESOURCE_AUDIT.json；这不是全量Tensor内容重审。

原官方、旧白盒实现、旧Full报告分支及初始FAIL源码工作树保持原HEAD和干净状态；旧结果文件不重新标记或覆盖，只在新PROJECT_STATE/EXPERIMENT_REGISTRY/ADR更正当前分类。没有生成官方SGA实验输出，未开始VLPTAttack。

本轮到此停止，等待源模型、输入桥/Matched Clean及算法一致性方案审查；不运行CPU单元测试、真实模型、GPU或Full。
