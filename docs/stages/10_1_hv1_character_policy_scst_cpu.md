# H-v1 Phase 1：受约束字符Transformer与SCST接口报告

基准：G-v2 Commit 4e40c1ac877aeaa07a54f55740b5e4034d449da6；分支codex/gv2-per-attribute-cache。独立目录prototypes/attribute_text_policy_h_v1_20261015/。本阶段完成CPU可测试的神经策略及训练接口，不做真实H训练。

## 实现结果

默认Byte Transformer width64/2层/4heads/FF128/dropout0，154594参数；Attribute/Position/Replacement三Head，包含STOP/NOOP及严格Action Mask。H仅接收原Caption和文本独立导出的Canonical/type/source spans，无图像、PID、配对、S_raw、JVP输入。

Attribute预测分数经softplus，沿用Natural-K gamma3/beta1公式，无固定K4；采样为masked无放回，贪心按预测重要性。K为离散预算，不宣称可微Top-K。每选中属性最多一处编辑、全局最多K，并避免共享词被多个属性重复编辑；可空候选、K0和不替换。

编辑使用限定Greek/Cyrillic视觉近似字符，记录Canonical到原Span、原/新字符及U+码位、原文位置、UTF-8偏移和Unicode名称；回放逐字段校验。官方IRRA Byte-BPE探针只读加载Tokenizer和BPE，检查真实Token变化、77长度及截断；无效和截断候选直接Mask。人类可读性尚未人工验证，不保证所有字体形近或语义绝对不变。

## 奖励与SCST

R_sim为G攻击图像上的额外配对相似度下降。R_rank为受攻击训练Gallery的Margin下降，Margin明确为最佳同PID正例−最大负例；默认权重均1。另记录Clean/Adv文本增量及synergy_increment=Adv增量−Clean增量，不额外加入目标。

L=−stopgrad(R_sample−R_greedy)×序列动作log-prob总和。真实Unicode/Tokenizer不可微，通过策略梯度更新H；无大型预训练模型、Targeted或额外算法。训练上下文只接受official train/internal role=train，拒绝Test及内部Validation；policy模块不导入奖励或G模型。真实来源与正确配对的审计须下一阶段接线实施，本轮没有真实数据读取作策略训练。

## 源码清单

新增6模块：contracts.py、tokenizer_probe.py、constraints.py、policy.py、rewards.py、scst.py；新增3测试：test_policy.py、test_scst.py、test_tokenizer_real.py；新增config.json、README.md及本中文报告。共12文件，全部新建，不改G-v2/IRRA源码、Checkpoint、Full输出或冻结属性前端。

## 验证

- 68项H-v1 CPU tests通过，覆盖输入隔离、原Span/多字节偏移、预算/共享词、非法Mask、K0与K9、采样/贪心、三Head/Byte Embedding梯度、SCST符号与stop-gradient、H-only optimizer、train/validation/test反馈隔离。
- 官方SimpleTokenizer/Byte-BPE实际CPU验证通过；探针在普通、Unicode及截断字符串上与官方datasets.bases.tokenize的77维Token逐值一致。
- 原G-v2 289项CPU＋评测68项回归通过，累计425项。
- 默认154594参数网络使用合成3图512维特征，2次模拟SCST step完成参数更新。最终Loss4.0893330574035645、advantage1.4000000953674316、梯度范数24.232439095105683，均有限。这些是合成奖励数据，不是攻击效果。
- 首轮截断边界测试夹具少1 byte，改为真实超限边界后通过；没有放宽规则。失败日志保留H outputs，不发布。
- 无CUDA初始化，无真实G/IRRA模型前向，无真实H训练/100/Full/官方Test反馈，无H权重保存。G-v2 Full结果仅作资产保护SHA/stat核对，不作训练数据。

阶段证据在独立H目录outputs/cpu_final.log、gv2_regression.log、evaluation_regression.log、synthetic_summary.json；仅服务器留存，不提交outputs。

## 阅读与复用依据

- src/attributes/text_extractor.py：文本独立Canonical及provenance来源，可用于下一阶段Caption前端。
- attribute_unit_v55_interface_20261003/attribute_interface.py：固定12 axes、convert_record、text_candidates与原Span exact规则；本轮不调用reader/representation构造。
- attribute_tta_step74_dynamic_k_20261004/selection_audit.py：正式gamma3/beta1 Natural-K；H复用数学但输入为自身预测分数，不修改原S_raw。
- G-v2 inference_once.py：独立image-only一次生成，未来只读使用adversarial返回值；本轮未加载。
- 官方IRRA datasets/bases.py::tokenize与utils/simple_tokenizer.py：ftfy/html/lowercase、Byte-BPE及77长度；仅CPU Tokenizer被调用。

## 下一阶段具体接口

完整步骤见README。拟新增text_frontend.py、training_inputs.py、frozen_feedback.py及真实训练runner，当前均未实现：

1. Caption→extract_with_provenance→冻结Canonical adapter→仅文本源Span→TextExample。禁止以原配对Top-K筛测试输入，禁止重定位span。
2. 训练数据先核验固定2048 role=train；训练端用PID构造全部同PID正例mask/paired_index，内部Validation和Test隔离。标签只用于奖励。
3. 只读冻结G-v2 Epoch10与IRRA，G一次生成训练Gallery对抗图；独立缓存clean/adv单位图像特征。核验原图预算、参数/buffers和来源SHA。
4. 回调encode_text(tuple[str,...])只接字符串，返回官方IRRA单位CPU [N,512]特征；RewardEvaluator接训练上下文，SCST接口更新H，网络前向看不到图像和标签。
5. 经用户审查、后续小规模真实接线授权后才能进入GPU与真实训练；正式训练超参数、H Checkpoint恢复及数据缓存协议仍待实施。

## 边界与尚未完成

尚无训练好的H、真实奖励接线、真实H训练调度、未知Query真实攻击测试、协同消融或可读性人工验证；不宣称已增强G-v2正式Full结果。推理会对超1024 UTF-8 bytes报错、对原IRRA已截断Caption不编辑，避免静默Span变化。真实数据/模型只读接线须下一阶段单独授权。

CPU验证及资产保护核对通过后，仅提交源码/config/tests/README/中文报告，自动push；不提交outputs、数据、缓存或模型。本阶段交付后停止等待审查。
