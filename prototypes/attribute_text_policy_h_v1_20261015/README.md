# H-v1 Phase 1：图文协同字符策略与SCST CPU接口

基准G-v2 Commit：4e40c1ac877aeaa07a54f55740b5e4034d449da6。本目录完全隔离，不导入G-v1/G-v2模型或integration，不修改既有训练/评测/资产。当前没有训练好的H；仅神经网络、受约束动作、奖励接口和合成CPU验证。没有真实H训练入口、GPU或Test评测入口。

## 输入、网络与推理隔离

`TextExample(caption, attributes)`只包含原Caption及从该Caption独立提取的Canonical/type/source-span字段。每条AttributeSpan有semantic_axis、canonical_value、region、raw_text、char_start/end；使用固定12 axes，原文切片必须逐字符exact。未知/未解析属性应由文本前端排除，不用字符串搜索重新定位。重复原词必须携带各自真实Span。attribute_id只由文本字段生成；PID、图像、正确配对、S_raw、JVP、数据集row_id不能进入该API。

默认UTF-8 Byte Embedding＋位置Embedding＋2层Transformer Encoder，width64/heads4/FF128/dropout0，共154594参数，从零初始化。多字节原字符的Byte hidden求均值得到字符位置表示，属性源Span池化加axis/region embedding。Canonical值保留作可追踪metadata，属性表示来自原词上下文及类型。

三个Head：Attribute预测重要性与STOP；Position预测Span内部位置及NOOP；Replacement在受限Unicode词表输出分布。STOP/NOOP复用对应Head，不增加整句生成器。当前明确CPU执行，max_bytes=1024，超限报错，不能悄悄截断或改变Span。

推理`policy.attack(example)`只有上述文本输入，贪心/eval/inference；不接收训练奖励上下文。当前未实现真实Caption提取接线，调用方必须提供已核对的文本源属性；不能把原图文Top-K的选中集合伪装为未知Query的文本输入。

## Dynamic-K与采样

对有合法字符候选的属性使用H预测正数softplus(logit)，沿用gamma3/beta1的有效数公式：

p_i = score_i³ / Σ score_j³；K = min(ceil(exp(-Σ p_i log p_i)), m)。

这里的score是H从文本预测的数，不是原配对S_raw。不改动冻结评分/选择源码。无cap=4，CPU覆盖K=9。K及排序是离散选择，不宣称对ceil/Top-K反传；Attribute Head通过所选动作的log-prob学习。

训练rollout按masked categorical无放回采样，最多K个属性选择槽；贪心按预测logit排序并可STOP。选中属性最多编辑一次，Position NOOP消耗该选择槽但不编辑；STOP提前终止。没有可用候选时K0、原文不变、log-prob为可backward的0。随机策略用显式torch.Generator可复现。

## Unicode动作与Tokenizer

仅人工白名单Greek/Cyrillic形近单字符，禁止插入、删除、空白、控制字符或任意词替换；视觉近似依赖字体，不保证人类语义完全不受影响。共享同一原词的多个属性也只能总计编辑一个字符。

每次编辑保留属性ID、Canonical/type/region、原Span与raw、原文绝对字符位置、Span内位置、原UTF-8字节偏移、原/新字符、U+码位及替换Unicode名称。输出同时保存全部原属性Canonical/Span映射，即使NOOP也不丢失对应关系。位置均基于原文Unicode code point；替换后UTF-8字节数可能增加，不能将字节偏移当字符位置。

TokenizerProbe复现官方77 token的SOT/EOT/padding/truncate，加载官方SimpleTokenizer及BPE前验证SHA。真实ftfy/html/lowercase/Byte-BPE清洗可能使Unicode修改无效，故Action Mask拒绝模型输入Token不变或截断的候选。原Caption已被IRRA截断时也不编辑并记录状态。最终记录原/新content tokens、77维model tokens、长度、UTF-8长度与截断标志。没有Token可微近似。

## SCST与训练专用奖励

RewardEvaluator通过外部encode_text((original, sampled, greedy))回调取得冻结、单位化CPU [N,512]特征。TrainGallery同时保存clean/adv gallery特征、训练positive_mask、paired_index、query/gallery split和role；只允许official train且内部role=train，拒绝Test和内部Validation反馈。该上下文绝不传入policy.forward/attack。

R_sim = sim(I_adv,T) - sim(I_adv,T_adv)
R_rank = Margin_adv(T) - Margin_adv(T_adv)
Margin = max(全部同PID训练正例分数) - max(训练负例分数)
R = lambda_sim R_sim + lambda_rank R_rank，默认两项均1。

另记录Clean图上的配对相似度下降、Clean margin下降及synergy_increment=adv_sim_drop-clean_sim_drop；不将该诊断差值另加进优化目标，也不以其单个数值宣称协同贡献。

贪心输出作self-critical baseline，同一参数下sample和greedy分别获奖：
L_SCST = -(R_sample-R_greedy).detach() × Σ log π(a_t|T,mask_t)。

真实字符串、Tokenizer、奖励全部不可微；只通过策略log-prob更新H。synthetic_scst_step只提供单次CPU接口，无epoch/dataset/真实模型加载；optimizer必须仅含H参数，检查Loss/梯度/参数有限。真实数据来源身份、pair/negative绑定仍须下一阶段的独立接线验证；仅填写train字符串不等于真实来源审计。

## 文件

| 文件 | 职责 |
| --- | --- |
| contracts.py | 文本专用输入、Canonical/Span与词区间 |
| tokenizer_probe.py | 官方Byte-BPE只读CPU检查 |
| constraints.py | 白名单、掩码候选、编辑记录与严格回放 |
| policy.py | Byte Transformer、三个Head、采样/贪心、Natural-K |
| rewards.py | train-only特征上下文与协同奖励 |
| scst.py | stop-gradient SCST及合成单步更新 |
| config.json | 阶段配置，未冻结真实H训练超参数 |
| test_policy.py / test_scst.py / test_tokenizer_real.py | CPU测试 |

## CPU验证命令

```bash
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -m unittest -v test_policy test_scst test_tokenizer_real
```

68新增CPU tests通过；G-v2原289＋评测68回归通过，共425项。默认网络另用3张合成512维训练图库特征与合成字符串完成2次模拟SCST step，参数更新、最终Loss4.0893330574035645、advantage1.4000000953674316、梯度范数24.232439095105683。该奖励人为构造，不能当攻击效果。无CUDA初始化，无真实训练样本、G/IRRA模型前向或H权重保存。首次截断测试夹具差1 byte，修正边界后通过，没有放宽过滤规则。

## 下一阶段真实接线方案（本轮未执行）

1. 新增本目录的text_frontend.py：调用只读src/attributes/text_extractor.py::extract_with_provenance；以Caption派生临时ID适配冻结attribute_interface.convert_record，再text_candidates(require_representation=False)。只取已解析固定12 axes的原始六字段，去重同字段，交给from_text_spans；不attach reader/basis，不依据配对score/Top-K筛候选。测试也运行同一caption-only流程。
2. 新增training_inputs.py：先以G-v2固定2048 role=train样本做接线，独立核验Caption、图像SHA、PID、row_id和split/role；同PID全部正例与负例mask只在训练端创建。128内部Validation完全隔离，不传给SCST接口；不读取官方Test反馈。
3. 新增frozen_feedback.py：只读加载Epoch10固定G-v2 SHA f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e，复用inference_once.generate_once返回的adversarial（不是把G输出delta当图像）。每训练Gallery图一次G，原图8/255投影，冻结官方IRRA编码clean/adv图像后存H独立缓存。G/IRRA eval、requires_grad=False、inference及完整参数/buffers SHA核对。
4. IRRA文本回调按本探针的实际官方77-token输入，encode_text→float→L2 normalize→detach.cpu.clone；可去重original/sample/greedy字符串，保留回填顺序。首轮接线H仍CPU、G/IRRA可在后续获批的小规模GPU中运行；返回特征给RewardEvaluator，图文/标签不进入H前向。
5. 真实trainer只调用当前SCST接口，奖励后端和训练来源另行门禁；缓存/配置/Checkpoint均在本目录隔离。需先经授权少量真实train验证、奖励符号/正负例绑定与冻结审计，再确定真实H训练超参数。当前没有这些运行入口。
6. 测试部署接口仍为H(T, caption-only canonical/type/spans)，G Gallery独立一次生成；不把正确图像、PID、S_raw或Test标签交给H，不用Test真实匹配选动作或Checkpoint。

尚未验证：真实H奖励、真实训练收敛、未知Caption泛化、视觉可读性人工评估、相对普通文本攻击的协同贡献。没有开展消融、迁移或新的算法实验。本阶段完成后停止等待审查。
