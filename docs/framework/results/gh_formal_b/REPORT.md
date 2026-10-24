# B双向交替正式训练准备与G60→H60统一评估

本轮目标是把既有B闭环推进为可保存、恢复及统一评估的独立实现；不继续NOOP kernel调查，不修订旧Smoke exact断言。所有历史PASS/FAIL均保留。

## 结论

| 项目 | 状态 |
| --- | --- |
| 旧G60/H10/H60身份、真实训练预算 | PASS，见BASELINE_AUDIT |
| 全2048 Train的G/H严格Canonical映射 | PASS，6020个选中属性，无AMBIGUOUS回退 |
| Pilot和正式Train两个真实环境CPU Preflight | PASS，权重/双Adam/CPU RNG恢复、完整来源及历史保护通过 |
| 新B调度、Round恢复、缓存/图库绑定及统一IRRA接口 | 已实现，CPU Fixture验证通过 |
| 新真实B Pilot | NOT_STARTED / RESOURCE_BUSY，无Attempt |
| 正式长时间B训练 | BLOCKED，未启动 |
| 新Validation / Test / Full | 未执行 |
| 新B攻击性能 | NOT_VERIFIED，不能宣称改善mAP |

最近一次即时资源门禁仅空闲1368MiB，低于原Smoke/Pilot的10240MiB门槛。RAM、磁盘、inode通过；没有Watcher、等待循环、抢占或其他用户进程干预。

## 基线身份与比较范围

真实基线是固定G60 + 适配H60 visit122880。H60来自H10新增50轮，在固定G60 Train奖励图库上训练。它有128条内部Validation结果，当前已审查资产中没有对应官方Test Full。历史G60/H7 Full不能代替G60/H60。

新B从同一完整G60/H10父状态开始，拟执行50个交替Round。这样匹配旧H的新增数据访问与Adam更新预算，但B额外训练G并刷新图库，不是等训练计算量对照。历史基线保留为顺序训练参照；若要单独归因“双向条件反馈”，后续可另行授权相同G/H更新、相同刷新次数的A1控制，当前未开展。

## 实际信息通路

每个Round按以下顺序执行：

1. 当前H在本Round的G训练顺序上产生caption-only攻击轨迹，保留原Canonical属性锚点。
2. NOOP显式复用已核验原条件；非NOOP调用冻结condition/reader/TextTransport，重算新语境整句特征、Byte-BPE/EOT定位、属性方向和真实JVP/FD检查，绝不重标旧JVP。
3. 当前G使用这些条件调用原train_single_sample，逐属性更新并最后从原图执行Joint更新；保持原K+1、Global/Local/Retain权重、Adam及投影。
4. 更新后的G调用冻结refresh_bank，一次完整生成2048张Train图库，逐图Claim、三FP32 Tensor、特征、SHA读回；BankIdentity绑定当前G状态。
5. 安装新RewardBank，当前H调用原SCST train_step；R_sim/R_rank及权重均不变，记录同文本在前/后图库的Reward差和Advantage差。
6. 保存当前G/H、双Adam、分开的G_RNG/H_RNG和游标。下一Round沿这些更新后的状态前进，不重载G60/H10。

G(I)、H(T)的推理接口不变。Train阶段允许真实配对和PID进入监督/Reward；推理时G只接图像，H只接Caption及其自身属性，PID只进入检索指标。

## 实现边界

新增formal_sources/backend/state/cli/audit/evaluation。FormalBackend继承原NativeBackend，仅增加调度、日志与边界恢复；没有覆盖condition、g_update、h_update或refresh_bank的训练数学。

原140份框架producer逐字节保持；496项运行来源继续核验。新增146项完整框架清单、条件空间和两种scope的配置身份正常绑定，不冒用旧source_key。

新调度不调用原固定A1/B两轮Prototype，也不消费原probe_v2/smoke_v2授权票据。旧probe/RNG作为冻结数学及父状态证据引用；新Pilot与正式训练使用自己的范围授权和不可变输出。旧Legacy Preflight仅作为明确标记的来源投影检查；完整2048映射、Round预算及原绝对Epoch排序另行实际核验。

源模型、reader、G/H损失、H奖励、K和Unicode规则均未改；没有FiLM、新Joint Loss或共享属性强制选择。

## 正式预算

| 项目 | 旧G60→H60适配阶段 | 拟定B 50 Round |
| --- | ---: | ---: |
| 初始G / H | G60 / H10 | G60 / H10 |
| 新G样本访问 | 0 | 102400 |
| 新G Adam更新 | 0 | 403400 |
| 新H访问 | 102400 | 102400 |
| 新H合法Adam更新 | 101850 | 101850 |
| G60/新G Train图库生成 | 固定一份2048张 | 50份，共102400张 |
| 新语境JVP | 0 | 上限301000 |
| 最终G累计更新 | 484080 | 887480 |
| 最终H累计visits / updates | 122880 / 122220 | 122880 / 122220 |

每Round：2048 G访问、8068 G更新、2048 H访问、2037合法H更新。G采用原排序函数的绝对Epoch61–110；H采用原排序函数的绝对Epoch11–60，seed42。两侧顺序规则分别复用原实现，不强行改成同一个排列。

Full配置保留GPU空闲≥10240MiB、RAM≥32GiB；为50份不可变图库预留磁盘250GiB和200万inode。三RGB Tensor纯数据约168.75GiB，另有模型、条件、特征、Claim/Manifest及目录inode。不能只看模型参数体积估算资源。

旧Smoke四Round耗时1824.9946秒、峰值reserved1158MiB；单份2048图库约420–449秒。结合旧G新增50轮约4.72小时、H约1.98小时及新增JVP，整链暂估20–36小时，受共享GPU与大量SHA/fsync影响。该值是规划估计，不是新正式实测；完整2048条件/plan常驻内存和长程行为仍未GPU验证。

## CPU验证与真实来源

最终相关回归154项PASS（其中新B用例27项），5项可选旧服务器配置用例SKIP；真实Pilot/Train CPU预检另外实际执行PASS。冻结原目录的IRRA专属CPU用例额外1项PASS。完整计数、来源及原始日志见SOURCE_EVIDENCE.json，不将SKIP伪报为PASS。

覆盖：双Adam/RNG完整边界状态、连续两轮沿更新态前进、CPU连续/恢复一致、不可变发布和损坏拒绝、后续Claim阻止重播、错误父状态/重复启动拒绝、预设训练预算校验、每轮阶段进度、完整新审计控制流及Fixture篡改拒绝。

统一评估的CPU Fixture实际执行FeatureCache→原EvaluationEngine→四模式排名→CSV/ASR；另测试独立G(I)/H(T)推理接线、原子产物、固定最终B Checkpoint及禁止未完成训练时准备Test协议。没有加载真实IRRA进行评估。

准备期真实预检曾暴露两项新接线问题：src下__main__触发冻结legacy命名空间保护；历史证据索引包含metadata字段不能直接解包FileRef。现已分别使用独立operations入口、显式path/sha256构造解决；原门禁和失败日志均保留。另在CPU Fixture中修正了新物化Claim与ArtifactStore保留文件名的冲突，没有污染GPU产物。

探索性旧IRRA配置测试在新工作树因绝对producer路径不同被严格拒绝；旧配置和断言不改，在原冻结工作树单独执行该CPU用例PASS。新统一入口有独立来源清单与真实Preflight。

## Round恢复与失败处理

Checkpoint由原ArtifactStore fsync/原子提交，含完整双模型、双Adam、双RNG、父来源、绝对累计游标和Bank Manifest引用。

仅显式恢复最新完整Round；后续Round Claim、未完成事务、后续IO或bank目录均阻止回退。先验证状态、全部已提交图库文件和来源，再恢复最新图库为下一轮的旧奖励参考；不能回退使用初始G60 bank。

正常信号只请求完整Round边界停止。失败不自动重试、不删除Claim/pending、不重放已提交更新。显式边界恢复记录旧FAIL的SHA并保留原文件；新审计区分已登记旧失败与当前未处理失败。CUDA未来轨迹不承诺bitwise一致。

## 统一IRRA评估

两方法使用同一official Test Manifest，CPU已确认与历史Epoch60 IRRA Engine逐项相同，3074 Gallery/6156 Query/1000 PID。统一原PIL RGB→Resize384×128→FP32；Adv RGB保持浮点，不回PIL/JPEG。

先各自生成固定攻击输入，再原生IRRA编码并保存FeatureCache，最后同一EvaluationEngine按Query chunk128、全部同PID正例及原排序实现计算Clean/Image-only/Text-only/Joint。训练不接触评分结果。

公开表仅R@1、R@5、R@10、mAP和条件ASR@R1。ASR分母为Clean Top1正确的Query，成功为攻击后Top1错误；分母0时null，不能用100−R@1替代。CSV同时保存原精度与两位小数。

训练配置中的Test评估保持关闭。B正式训练和独立审计完成后，用独立CPU配置准备器冻结新的比较协议，绑定最终Round50状态、原训练配置和审计；不改训练配置或旧资产，再由后续明确授权执行评估。

## 当前仍未验证

新B Pilot的实际GPU执行、两轮新模型状态及独立审计；完整2048样本的新语境定位/JVP是否全部可用；50轮资源、恢复后的真实CUDA轨迹；新B及G60/H60官方Test效果，均为NOT_VERIFIED。

本轮没有新mAP/R@1，不能从信息流或训练Loss推断攻击收益。正式训练READY条件仍不满足，当前仅CPU_READY；下一步先完成独立B两轮Pilot及CPU审计，随后再授权长训练和统一Test。

本次正式配置SHA-256：

e0220fe92bdd4480bdb751e5b67a6da8a26a74a543694871b28ea1627c97bf74
