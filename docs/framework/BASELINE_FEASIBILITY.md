# 七种候选Baseline复现可行性（2026-10-18）

本轮只查资料与本机资源，没有运行以下方法、下载训练数据或启动训练。优先级依据接线工作量和权限可比性，不根据其他论文结果推断本项目效果。仅将实际已核验的B3作为本轮可执行Baseline。

| 方法 | 论文与官方来源 | 原始攻击类型/权限 | 本机代码与权重 | TBPS白盒适配、资源与建议 |
|---|---|---|---|---|
| SGA | ICCV2023，Set-level Guidance Attack；[论文](https://arxiv.org/abs/2307.14061)、[作者代码](https://github.com/Zoky-2020/SGA) | 访问代理模型梯度的逐样本图文迭代攻击，重点验证黑盒迁移；不等于天然的三个TBPS模型白盒攻击 | 项目/Checkpoint目录中未发现对应独立资源；官方入口依赖ALBEF/TCL/CLIP权重，不需预训练攻击Generator | **优先1**。接原生可微图像/文本表示及多Caption集合；若访问每个目标模型梯度，应标注SGA的TBPS白盒适配。需固定单图如何聚合多个Caption，不能生成多个query-conditioned Gallery后冒充共享图库。代价为每样本多轮、多尺度前后向；先10图测显存/时间再估Full。 |
| VLPTAttack | 用户已确认VLPTAttack对应ECCV2024 [VLPTransferAttack](https://github.com/SensenGao/VLPTransferAttack)、[论文](https://arxiv.org/abs/2403.12445)，作者将方法命名为DRA；本项目实验表保留VLPTAttack | 代理梯度、文本引导的轨迹区域多样化与图文迁移攻击 | 未发现本机资源；官方仓库提供ALBEF/TCL/CLIP权重链接，部分TCL链接已失效并提供替代HF入口 | **优先2**。不能静默用VLAttack或Co-Attack替代该名称。该方法原设置8/255，仓库明确SGA原设置2/255；比较时预算统一且标记重设。多轨迹采样比单次Generator成本高，白盒TBPS需要受害者可微表示接线。 |
| MFHA | Scientific Reports2025，Boosting adversarial transferability in vision-language models via multimodal feature heterogeneity；[作者全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11873190/)、[论文声明的代码](https://github.com/doyoudooo/MFHA) | 模态一致/差异特征、局部互信息、空间/频域增强与动量；访问源模型梯度，评估源端及黑盒迁移 | 本机未发现；本次访问论文声明的GitHub链接返回404，代码/权重可用性**未就绪** | **优先5**。需作者可用代码或存档；原论文含2/255、10轮及多增强，统一8/255是新适配设置。局部Token/全局特征接口和频域增强增加显存与多轮成本，不能只实现一个PGD后冠名MFHA。 |
| AnyAttack | CVPR2025；[官方代码](https://github.com/jiamingzhang94/AnyAttack) | 自监督生成式图像攻击，官方示例含目标图像输入；大规模预训练/下游微调与迁移，不能直接当作无训练的untargeted白盒PGD | 本机未发现；官方README提供Generator下载链接，是否可下载及与CUHK兼容尚未实测 | **优先3：先核验公开权重**。复用权重推理成本低；从头复现大规模预训练成本很高（官方涉及LAION400M、COCO/Flickr）。必须先冻结非目标使用方式、目标参考来源、预算映射；若改为对IRRA/RaSa/APTM训练攻击器，应单列白盒适配训练，不能沿用原论文权限表述。 |
| MisRanking | CVPR2020，Transferable, Controllable, and Inconspicuous Adversarial Attacks on Person Re-identification With Deep Mis-Ranking；[官方代码](https://github.com/whj363636/Adversarial-attack-on-Person-ReID-With-Deep-Mis-Ranking)、[论文](https://arxiv.org/abs/2004.04199) | 生成式ReID图像攻击；源模型训练访问和跨模型转移。原ReID查询/图库与camera协议不是TBPS原生协议 | 已有/home/lzf/ldx/projects/DeepMisRanking；/home/lzf/ldx/checkpoints/DeepMisRanking下IDE/LSRO/Aligned资源是ReID模型目录，未证明存在可直接用的CUHK/IRRA攻击Generator | **优先4**。代码较易定位，但原README针对很旧Torch。TBPS无原ReID分类头，排序/分类监督需明确适配，可能需要授权重新训练Generator；预算与Gallery-only也需单列，不能把已有ReID权重当作攻击Generator。 |
| MetaAttack | AAAI2021，Learning to Attack Real-World Models for Person Re-identification via Virtual-Guided Meta-Learning；[作者代码](https://github.com/FlyingRoastDuck/MetaAttack_AAAI21)、[论文](https://ojs.aaai.org/index.php/AAAI/article/download/16422/16229) | 用真实源域+虚拟PersonX进行元学习的通用扰动迁移；不是简单逐样本白盒攻击 | 本机未发现；README提供ReID模型/PersonX资源下载，不代表已发布本任务可用扰动 | **优先6**。需要PersonX、多模型与元训练；迁移到TBPS应重建正确的图文排序目标。现有冻结G/H不能充当该方法产物。额外训练/多模型资源明显高于SGA。 |
| MTGA | Learning to Learn Transferable Generative Attack for Person Re-Identification；[作者论文](https://arxiv.org/abs/2409.04208)、[作者代码](https://github.com/yuanbianGit/MTGA) | 元学习生成式攻击，模拟跨模型/数据/测试域的黑盒迁移，含扰动随机擦除与Normalization Mix | 本机未发现；官方仓库有训练/测试代码，简短README未给出已核验的CUHK攻击器Checkpoint | **优先7**。需要多源ReID模型/域与元训练。将其改成针对每个TBPS目标的训练是白盒训练适配，并非原生现成白盒接口。先明确训练资源与公开Generator，再授权实验。 |

上表资源成本是基于算法结构的定性判断，不是本服务器运行实测。除B3外，没有为七种方法承诺具体GB、分钟或正式Full结果。

白盒对比必须另行冻结：源/目标权重、攻击侧可见Caption/标签/梯度、训练与Test的权限边界、Gallery-only或图文联合、8/255像素空间、文本编辑预算、共享图库生成规则、优化次数。尤其RaSa/APTM的Top-K候选选择与全局归一化不能无说明地替换成IRRA余弦Loss；可微攻击目标与正式原生评测需要分别声明。所有攻击最终输出像素/文本后交给既有评测器，受害者Adapter不绑定攻击算法。

推荐顺序：先完成本轮B3三模型公平测试协议对照，再做SGA小样本白盒适配；并行准备VLPTAttack官方代码适配与AnyAttack公开权重。MFHA代码链接和三个ReID方法的TBPS训练契约先解决后再安排GPU。
