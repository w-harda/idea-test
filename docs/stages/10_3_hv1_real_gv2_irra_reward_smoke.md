# H-v1 Phase 3：真实G-v2 / IRRA GPU奖励链路验证报告

日期：2026-10-16。基准Commit：c6122f0eeb68f32df0910c1913d3df625e542007。分支：codex/gv2-per-attribute-cache。发布Commit以Git记录为准。

本阶段真实RTX 4090 Smoke一次通过，完成10 Train Gallery + 10 Train Query的图像生成、图文编码及sampled/greedy奖励计算。没有训练H、没有optimizer/backward，没有正式检索指标。

## 实现与保护边界

新增且只新增以下文件，不修改G-v1/G-v2、IRRA、H前两阶段源码、Checkpoint或历史结果：

| 文件 | 内容 |
|---|---|
| phase3_inputs.py | 默认CPU预检、冻结来源及确定性5 PID×2图选择 |
| real_feedback.py | 冻结真实模型加载、G单次推理、官方图文编码、实际前向计数与冻结核验 |
| real_feature_cache.py | 独立Real协议、紧凑特征、SHA/身份绑定、原子发布及回执 |
| smoke_phase3.py | 显式授权10+10 Smoke、停止边界、失败留证、CPU完成证据核验 |
| test_phase3.py | 42项新增CPU接口测试 |
| config_phase3.json | 54项既有源码/配置SHA与本阶段范围声明 |
| README_PHASE3.md | 接口、实际命令、证据、限制及下一阶段建议 |
| docs/stages/10_3_hv1_real_gv2_irra_reward_smoke.md | 本中文报告（Windows/服务器同步） |

前7项位于prototypes/attribute_text_policy_h_v1_20261015/。发布不含数据、模型、条件/特征缓存、outputs或日志。

129项历史保护文件SHA精确保持：包含G-v1/G-v2/H既有已跟踪源码、H前两阶段证据、G-v2 Epoch1–10 Checkpoint、条件Manifest/Protocol、Full关键完成结果及官方IRRA权重。没有重复运行G-v2 Full审计，没有访问Test样本进行反馈。

## 模型、数据与计算链路

| 项目 | 固定值 |
|---|---|
| G-v2 Epoch10文件SHA-256 | f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e |
| 官方IRRA权重SHA-256 | 9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009 |
| 训练来源 | 原2048 Train/1861 PID；128内部Validation/121 PID；PID交集0 |
| 训练目录指纹 | a4736eacd7bbf70e91e4a09504e24a009dc9828e0c04e45cf9859ff4682fb8dd |
| Smoke选择 | SHA256(42\|PID)排序选5个有至少两图的train PID，各按SHA256(42\|row_id)选2图；Query为这10条原训练Caption |
| 正负样本 | 每Query自身paired图存在，同PID2个正例，其他PID8个负例 |
| 图像预算 | 原始[0,1]像素空间，相对原图L∞≤8/255 |
| 图像预处理 | 原G-v2 PIL RGB Resize(384,128)+ToTensor；IRRA mean/std与L2归一化保持 |
| 文本预处理 | 官方Byte-BPE，长度77，保留原Unicode，不额外转写；截断拒绝 |
| G/IRRA状态 | eval、requires_grad=False、所有前向inference_mode |
| H状态 | 默认CharacterPolicy，seed42，未训练，CPU eval/inference；本次H也冻结 |

实际复用G-v2 inference_once.load_generator/generate_once和irra_image_adapter.load_official；官方同一个IRRA实例同时提供图像与文本最终512维检索特征。没有重新实现G、预算投影、IRRA模型或奖励。

H只收到由CaptionFrontend生成的TextExample，内容为Caption与Canonical/type/原Span。图像、PID、paired_index及正负集合只存在训练奖励绑定层；没有将它们传给H。旧S_raw、JVP和配对Top-K不进入H。内部Validation和官方Test均不进入本次图库或Query。

## 实际前向、特征与预算

| 调用 | 实际调用次数 | 编码/处理样本数 |
|---|---:|---:|
| G forward | 10 | 10 Gallery |
| IRRA Clean encode_image | 10 | 10 Gallery |
| IRRA Adv encode_image | 10 | 10 Gallery |
| IRRA encode_text | 10 | 30个文本向量（10原文+10sampled+10greedy） |
| H CPU forward | 20 | 10sampled+10greedy |
| optimizer/backward/JVP | 0 | 0 |

G与IRRA计数由实际调用处的hook/包装函数观察；每张Gallery恰好一次G，无重新生成。30个文本向量属于10条Train Query的原文与变体，不是30条独立Query。包括一次采样NOOP，仍在对应三文本bundle中编码。官方构造器打印Training Model任务名仅为初始化提示，实际冻结/inference状态及零更新均核验。

Clean、Adv图库特征均为[10,512]，每条Query文本bundle为[3,512]。图像特征范数范围0.9999999403953552–1.0，文本范数范围0.9999998807907104–1.0，均有限且脱离计算图。图像像素范围合法。

最大实际L∞=0.0313725471496582，严格≤8/255，没有额外容差。

| 状态指纹（包含参数与buffers的名称/形状/dtype/数值） | 前后均为 |
|---|---|
| G | 1047822bcf2608631aefce24cb7e5f92ae187657284c2a8e1c33179d028ee3e9 |
| IRRA适配器及模型 | 5e87d09c5c3336e74d3656174f66bffd643d172a755aba753dfcb8f4e815981b |
| H | 9d607ba8b6b348a8783b3dad9b79d0206dd23147edec5fa794b7e932c5a2853f |

这些是运行时状态指纹，不是上文Checkpoint文件SHA。所有参数grad为None；H、G与IRRA均无更新。

## H动作与Tokenizer

10条Query预测K全部为2。sampled共11次单字符编辑、9条有编辑、1条NOOP；greedy共17次编辑、10条均有编辑。28次实际编辑全部在原Canonical Span内，遵守每属性/共享词最多一次、总编辑数≤K。原/新字符、码位、属性类型、原Span、原字符位置及UTF-8偏移均保存在summary对应actions中。

示例：cuhk:17793:0的sampled动作，gender=male、region=person、原Span[4,7)、source_raw=man、字符位置5、属性内部位置1、UTF-8偏移5，将a（U+0061）替换为Cyrillic а（U+0430）。这是真实H输出，不是手工诊断。

9个sampled编辑输出与10个greedy编辑输出均改变模型tokens；NOOP的tokens保持不变。变体总Token长度23–39（含SOT/EOT），未发生截断。H未全部NOOP，因此没有执行额外单字符诊断；legal_diagnostic_separate为null。

## 奖励原始数据

R_sim=sim(I_adv,T)−sim(I_adv,T_adv)。R_rank=Margin(T)−Margin(T_adv)，Margin=max同PID正例−max负例。总奖励R_sim+R_rank，权重仍1/1。synergy_increment=adv_sim_drop−clean_sim_drop，只诊断。SCST只记录sampled−greedy的advantage，未计算训练Loss或反传。

下表直接来自本次summary，不含任何mAP/ASR：

| Query | Mode | K | 编辑 | R_sim | R_rank | Total | synergy_increment |
|---|---|---:|---:|---:|---:|---:|---:|
| cuhk:17793:0 | sampled | 2 | 1 | -0.002528734505176544 | 0.07571184635162354 | 0.07318311184644699 | -0.013745494186878204 |
| cuhk:17793:0 | greedy | 2 | 2 | -0.04449385404586792 | 0.0635409876704216 | 0.01904713362455368 | -0.031726986169815063 |
| cuhk:17792:0 | sampled | 2 | 2 | -0.06168675422668457 | -0.006250426173210144 | -0.06793718039989471 | -0.05007633566856384 |
| cuhk:17792:0 | greedy | 2 | 2 | 0.0860765129327774 | 0.0701267272233963 | 0.1562032401561737 | 0.06695468723773956 |
| cuhk:33254:0 | sampled | 2 | 1 | -0.019678354263305664 | 0.026191264390945435 | 0.0065129101276397705 | -0.022835254669189453 |
| cuhk:33254:0 | greedy | 2 | 2 | -0.03136557340621948 | 0.09576058387756348 | 0.064395010471344 | -0.06470459699630737 |
| cuhk:33256:0 | sampled | 2 | 1 | -0.004933565855026245 | 0.02003338932991028 | 0.015099823474884033 | -0.011946439743041992 |
| cuhk:33256:0 | greedy | 2 | 2 | 0.010642021894454956 | -0.025184154510498047 | -0.01454213261604309 | 0.033300191164016724 |
| cuhk:10405:0 | sampled | 2 | 1 | 0.0019440427422523499 | 0.0076901912689208984 | 0.009634234011173248 | -0.008800528943538666 |
| cuhk:10405:0 | greedy | 2 | 1 | 0.0019440427422523499 | 0.0076901912689208984 | 0.009634234011173248 | -0.008800528943538666 |
| cuhk:10406:0 | sampled | 2 | 1 | 0.0005228221416473389 | 0.04698731005191803 | 0.04751013219356537 | -0.007934808731079102 |
| cuhk:10406:0 | greedy | 2 | 2 | -0.03702077269554138 | 0.05176883935928345 | 0.014748066663742065 | -0.02113834023475647 |
| cuhk:27219:0 | sampled | 2 | 1 | 0.0020837336778640747 | -0.008535027503967285 | -0.0064512938261032104 | 0.011852219700813293 |
| cuhk:27219:0 | greedy | 2 | 2 | 0.03567850589752197 | 0.008768200874328613 | 0.044446706771850586 | 0.03196188807487488 |
| cuhk:27220:0 | sampled | 2 | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| cuhk:27220:0 | greedy | 2 | 1 | -0.017183810472488403 | -0.008979648351669312 | -0.026163458824157715 | -0.005420118570327759 |
| cuhk:4323:0 | sampled | 2 | 1 | 0.0062386393547058105 | -0.002648979425430298 | 0.0035896599292755127 | 0.010568737983703613 |
| cuhk:4323:0 | greedy | 2 | 1 | 0.0062386393547058105 | -0.002648979425430298 | 0.0035896599292755127 | 0.010568737983703613 |
| cuhk:4322:0 | sampled | 2 | 2 | 0.04497147351503372 | 0.024540886282920837 | 0.06951235979795456 | 0.007073350250720978 |
| cuhk:4322:0 | greedy | 2 | 2 | -0.07217508554458618 | -0.019889503717422485 | -0.09206458926200867 | -0.1059410572052002 |

| 均值（10条Query） | R_sim | R_rank | Total | synergy_increment |
|---|---:|---:|---:|---:|
| sampled | -0.003306669741868973 | 0.01837204545736313 | 0.015065375715494156 | -0.008584455400705338 |
| greedy | -0.006165937334299087 | 0.02409532442688942 | 0.017929387092590333 | -0.009494612365961075 |

总奖励正负均有；平均R_sim与synergy_increment均为负。正的总奖励均值主要来自R_rank，不能据此宣称额外配对相似度下降、协同收益或正式检索改善。H未训练、小图库仅2正例/8负例、best-positive Margin与多正例mAP不同；本阶段只确认数值链路可用。

## 性能、缓存和异常

| 项目 | 实测 |
|---|---:|
| GPU | NVIDIA GeForce RTX 4090 |
| peak allocated MiB | 805.740234375 |
| peak reserved MiB | 886.0 |
| 模型加载阶段秒 | 7.078916180878878 |
| Smoke阶段总秒（不含CPU preflight） | 11.01696678251028 |
| 本次运行内CPU preflight秒 | 8.748803734779358 |
| 图像逐条阶段计时合计秒 | 0.9843740351498127 |
| Query逐条阶段计时合计秒 | 1.298962328583002 |

计时包含相应CPU检查/H动作/缓存IO和同步，不等于纯GPU kernel时间。运行前40MiB/0%利用率；结束后无GPU计算PID，显存回到40MiB。

Real缓存与Mock/Test严格隔离：10个图像单位+10个三文本bundle；每单位包含intent/features/metadata和发布回执，加协议共81个文件SHA由summary锁定。保存/读取均clone，单图特征紧凑[1,512]。协议绑定模型文件SHA、预处理、训练样本及图像SHA、来源和特征SHA。

无NaN、OOM、预算越界、缺失/损坏单位、pending、未提交回执、异常或中断。只有attempt_0001，无GPU重试。运行后独立CPU进程仅核对完成文件/源码/权重/缓存SHA，HV1_PHASE3_CPU_EVIDENCE_VERIFIED通过；没有重新编码或计算奖励/检索指标。

## CPU测试与证据

42项Phase3新增 + 56项Phase2 + 68项Phase1 = 166项CPU测试全部通过。覆盖真实接口核心的单次G、实际计数、inference/冻结/version/完整SHA、Unicode token差异、正负绑定、训练隔离、预算、缓存损坏/紧凑Storage/回执缺失/pending拒绝/恢复不重复G、默认CPU及显式GPU授权。真实GPU Smoke验证同一接口的官方实现；CPU fixture不冒充真实模型结果。

本阶段未运行无关G-v2历史Full审计。已有阶段证据完整保留。首次CPU fixture使用系统默认/tmp，临时文件由测试清理；收尾将TMPDIR显式限定为/home/lzf/ldx/tmp后重新运行全部166项CPU测试并通过，未重跑GPU。CPU命令已补充该约束。

服务器证据根目录：

/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/phase3/

关键文件：cpu_initial.log、cpu_h_final.log、preflight.log、gpu_smoke.log、cpu_evidence_verify.log、cpu_h_project_tmp.log、protected_before.json、stage_completed_final.json；真实产物在smoke10/protocol.json、summary.json、completed.json、attempts/attempt_0001/execution.json、real_features/。

| 证据 | SHA-256 |
|---|---|
| Protocol canonical digest | 987cb33e2e4ab174e6d878b45b31dd93656b6e700a6fb0b75568611ff0c3467b |
| protocol.json文件 | 9ee09af026042ab55809618148ca16cee5ad545717db0cb266b199ac1123465d |
| summary.json | 157fcc0d043960023b88670d04c4c8c64475fab4bbde94d439c813c297ce98af |
| completed.json | 10696268820b832a02d4e1a08b74e80a2b5526c58811eab3cd83e7f507b5cb08 |
| attempt execution.json | 1e7139b4232c5a99fc88899daf8140323804f4128d9dd63a6f7ba1e4befa0d6f |
| gpu_smoke.log | b76087512f3f211b6fe2546d1da7d7c16dce082b1995f166563ab63f4286583a |

canonical digest与文件SHA因JSON序列化不同而不同，不可混用。运行命令、查看进度及停止边界详见本目录README_PHASE3.md。已有输出会拒绝再次GPU运行，不提供自动失败恢复，不删除pending/回执来重试。

## 下一阶段最小实现与建议（尚未授权/执行）

1. 独立正式Real训练缓存：2048 Train全图库Clean/Adv，逐图图像SHA与一次G回执；128 Validation和Test不得用于训练奖励。若复用Smoke10单位，须另行建立严格来源兼容证明，不能直接混合协议。
2. 真实SCST调度：解冻且仅优化H，sample/greedy来自同一原文与当前H；G/IRRA始终冻结/inference，奖励stop-gradient。保留H CPU、奖励GPU即可先验证，不必修改网络或第一阶段设备限制。
3. 确定性采样、动作/奖励/advantage/梯度日志、来源锁定、H/Adam/RNG事务Checkpoint及恢复测试；没有这些机制前不启动长训练。
4. 建议首轮候选配置为Adam lr=1e-4、betas=(0.9,0.999)、单Query batch=1、seed42、λ_sim=λ_rank=1，原Dynamic-K/字符预算不变。先获得≤32次受控SCST更新授权，确认梯度、日志与恢复，再决定正式Epoch数。不加入额外奖励或正则；建议不是本阶段已冻结/执行的配置。
5. 文本可读性、未知Query泛化和图像攻击感知的额外协同贡献仍需后续训练与专门消融；当前随机H的均值不能回答这些问题。

本阶段完成后停止，等待用户与ChatGPT独立审查。没有启动真实H训练、全量Real缓存、Validation/Test反馈、100+100/Full或跨模型实验。
