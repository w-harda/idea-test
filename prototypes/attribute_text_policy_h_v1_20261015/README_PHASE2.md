# H-v1 Phase 2：真实属性前端与训练奖励接口

本阶段仅实现 Caption-only 前端、G-v2 原训练身份绑定、独立图库缓存契约和 CPU Mock 奖励接线。第一阶段模型、Dynamic-K、动作约束、SCST 和奖励公式全部保留。没有真实 G/IRRA 模型加载器或训练命令。

## 可用接口

- text_frontend.CaptionFrontend()(caption) → TextExample。输入严格为字符串；复用冻结 extract_with_provenance → v5.5 convert_record/text_candidates。只读取规则源码及 YAML，不加载 reader、basis、S_raw、JVP、图像、PID或标签。原始 Span 按 Python Unicode 字符索引记录；不同位置的重复属性保留。unresolved 原因通过 convert() 的第二个返回值记录。
- training_inputs.load_training_catalog()：只读验证原 conditions_full 的固定 Manifest、Protocol、7个 producer SHA、2176条 metadata/condition 文件 SHA、原 annotation 与内部 PID 划分；不反序列化条件 Tensor，不向 H 传递历史 K/S_raw/JVP。train 为2048条、validation 为128条。require_train/get_train 拒绝验证集、Test和伪造身份；verify_image 在实际使用前验证图像 SHA。
- feedback_backend.FrozenMockOps：只能注入带 h_cpu_mock=True 的 CPU 合成模块。冻结参数/eval/no_grad，检查参数及 buffers SHA。采用冻结 G-v2 预算投影；每图一次 Mock G、两次 Mock image encoder。正式模型前向计数恒为0。
- feature_cache.GalleryFeatureCache：只接受 cpu_mock 协议；逐图紧凑 clone 保存 clean/adversarial [1,512]。协议包含训练目录指纹、样本身份、源图 SHA、Mock状态 SHA、相关实现 SHA、官方Tokenizer/BPE SHA。Future real contract 中的官方权重 SHA 是下一阶段目标，不表示本阶段已编码真实图像。
- cache.bind_reward(query, gallery_samples, ops, frontend, budget_k)：按同一顺序从校验缓存构建图库；全部同 PID 图像为正例，其他 PID 为负例，paired_index 绑定 query 自身图像。只允许 train；需要至少一个正例和一个负例。evaluate_many(original, variants) 先校验原 Caption、受约束 Unicode 编辑、预算和 Tokenizer，再调用未修改的 RewardEvaluator。budget_k 应来自当前 H 的预测；同一原文的 sampled/greedy 在固定参数、无 dropout 下具有相同初始 K。

政策调用端只接收 frontend(sample.caption)；样本身份、PID和图像只在训练奖励绑定层存在。测试入口仍然只接受 Caption，不构造训练目录或奖励后端。

## 保持不变的奖励

R_sim = sim(I_adv,T) − sim(I_adv,T_adv)。
Margin(T) = max 同PID正例相似度 − max 负例相似度。
R_rank = Margin(T) − Margin(T_adv)。
Reward = R_sim + R_rank，两个权重仍为1。
synergy_increment = adv_sim_drop − clean_sim_drop，只作诊断，不加入优化目标。
SCST仍为 −stop_gradient(R_sample−R_greedy) × log_prob_sample。

真实Unicode文本经官方 Byte-BPE 生成77个模型token后交给注入的 TextMock；没有真实IRRA文本前向。发生截断、Tokenizer等价或非法替换时拒绝反馈。

best-positive Margin不等同于多正例mAP。例如正例分数0.9/0.8、负例0.7变成正例0.9/0.6、负例0.7时，Margin仍为0.2，AP却从1变为5/6。后续可讨论平均/最低正例Margin、正负pairwise目标或列表式soft-rank；本阶段均未实现、未调整奖励和权重。

## CPU验证命令

以下命令只运行CPU测试；不启动真实模型或实验。服务器目录和Python环境为实际路径。

~~~bash
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/lzf/ldx/envs/tbps-image/bin/python -m unittest test_phase2 test_policy test_scst test_tokenizer_real -v
~~~

最多16条真实train Caption的只读检查（现有证据不覆盖，请使用新输出名）：

~~~bash
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/lzf/ldx/envs/tbps-image/bin/python inspect_phase2.py --samples 16 \
--output "outputs/phase2/inspection_$(date +%Y%m%d_%H%M%S).json"
~~~

该检查会做2176条训练来源的文件SHA/身份核验，仅对16条Caption做属性/Tokenizer/未训练H的CPU诊断，并只核验所选16张原图SHA。它不编码图像、不运行G/IRRA、不生成正式奖励，不进行全量文本编码或检索。

## 缓存中断与隔离

新缓存单位在独占 pending 目录写入意图后计算，保存并校验SHA，再发布单位和回执。有效已有单位只读复用，不重复调用G。失败留下 pending；单位已发布但无回执时拒绝使用，不静默重算或删除。Phase 2不提供自动修复/压缩历史缓存入口。Mock与将来的真实缓存必须使用不同目录及backend协议，不能迁移为真实特征。

## 下一阶段真实接线（尚未执行）

1. 新增显式授权的真实加载器，严格验证 G-v2 Epoch10 f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e 与官方 IRRA 9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009，冻结/eval，核验状态无变化。
2. 只读使用 catalog.train：每张图 verify_image 后以原G-v2 PIL Resize(384,128)+ToTensor载入[0,1]，一次G(I)后复用原投影；用原IRRA适配器的mean/std和L2归一化编码Clean/Adv。每个唯一训练图库图像仅一次G；128 validation不进入奖励图库。
3. 在独立real缓存协议中锁定真实权重、预处理、实际模型/Tokenizer源码、训练目录及每图SHA。可复用经这些协议和样本身份逐项核验的Clean/Adv特征，不直接复用正式Test或旧混合图库。
4. 原文/采样/贪心Unicode Caption通过相同官方Tokenizer与冻结IRRA encode_text编码到同一检索空间。只根据train PID建立所有正例与负例；保持当前奖励和synergy诊断。原文特征可在严格来源键下缓存，采样/贪心不能使用错误文本缓存。
5. 首次真实GPU奖励链路、显存/耗时、图文特征空间一致性、正式H训练调度及Checkpoint均待独立审查与后续授权。本阶段不提供GPU/训练/检索命令。
