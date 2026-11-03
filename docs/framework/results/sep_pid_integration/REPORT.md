# SEP A类PID优化集成与公平迁移协议设计（2026-11-03）

**A类生产接线已完成CPU验收：原144项+新增32项共176项通过，原/新生产入口未出现位差、候选选择或RNG差异。** 生产只修改sep_bridge.py和sep_source.py。未初始化CUDA运行时、加载真实模型、运行真实Smoke/Full或重放旧44条/第45条。公平协议仅为设计，未修改代理、标签目标、oracle、DPP或正式配置。

基准Commit `769648d0b8330e7a9bf93a096abac1397fb13940`；独立Worktree `/home/lzf/ldx/projects/idea-test-tbps-sep-pid-integration-20261103`，分支`codex/sep-pid-integration`。本轮不把CPU局部Exact等同于公平威胁模型，也不把CPU微基准等同完整攻击提速。

## A. PID优化集成结果及Exact证据

### 实际修改范围

| 生产文件 | 本轮修改 | 保持不变 |
|---|---|---|
| sep_bridge.py | 新增冻结dataclass PIDPlan；固定Gallery一次构建原顺序PID/成员索引 | 原pid_logits兼容函数、make_ensemble、文本预处理/语义标记均原样 |
| sep_source.py | Classifier构造时建立独立self.pid_plan；cache miss改调用其aggregate | 候选顺序、字符串缓存键/8192容量/清空时机、RNG、模型加载/评分/Top128、原trace、attack调用和oracle都不改 |

每个Classifier有自己的Plan，即使两个模型的PID成员相同也不共享Plan；不同Gallery独立构建，不设全局缓存。正式DatasetManifest/Gallery不可变，Plan的order/members也为tuple；每次保留原宽度/维度校验。计划依赖固定Gallery，不能跨相同长度的不同Gallery复用。

Plan构建对PersonID/str为O(G+P)，之后取消每候选O(PG)的相等扫描；仍用同一`scores[:, list(indices)].max(dim=1).values`及`torch.stack(..., dim=1)`。PID按首次出现顺序，组内按原Gallery位置；未用scatter、index_select或新归约。非正式类型沿用原相等分组，空组/非有限输入保留原异常或传播行为，没有“修复”算法边界。

完整补丁见[PRODUCTION_DIFF.patch](PRODUCTION_DIFF.patch)。SHA完整记录在[SOURCE_EVIDENCE.json](SOURCE_EVIDENCE.json)：

| 文件 | 基准SHA256前缀 | 集成SHA256前缀 |
|---|---|---|
| sep_bridge.py | 9308443443630343 | 9346207dba66e6fc |
| sep_source.py | 71137fdd7a962dfc | d5932335b57bcbf2 |

新src不依赖tests原型。官方SEP_Attack.py/SEP_utils.py、其余生产源码和configs未改；未实施批量编码、缓存策略、Trace或设备搬运等B/C类优化。

### 新旧生产入口的CPU对照

基准函数直接从指定Git对象提取原AST编译，不手写“近似旧实现”。新入口是真实生产PIDClassifier.forward；编码器/NativeScorer注入**固定合成分数替身**，没有真实模型/Query。比较包括：

- FP32/FP64、原PID和成员顺序、signed zero/NaN/Inf/极端有限值、非连续输入及原异常类型/消息。
- 原/新实际Classifier缓存命中/8192条淘汰、编码与评分调用序列、给定矩阵输出原始字节、输入不变与RNG。
- 3个Seed（0/1/7）的官方原候选流程合成输入：最终文本、捕获候选全序列、oracle序列、缓存和Python/NumPy/Torch RNG完全相同。
- 每Classifier仅构建一次、两模型/两Gallery分别拥有Plan、Plan不可修改；生产CPU双线程入口也单独Exact验证。
- 精确生产diff白名单：sep_source仅导入/构造/调用Plan三处，sep_bridge仅新增Plan及两个导入；原有函数体均保持。

| CPU回归 | 通过 |
|---|---:|
| 原test_cpgc_sep.py | 28 |
| 原test_cpgc_sep_throughput.py | 19 |
| 原test_sep_cpu_optimization_review.py | 97 |
| 新test_sep_pid_integration.py | 32 |
| 合计 | **176** |

无skip/xfail/删除用例。原144项中只有`test_source_files_still_frozen`的阶段性冻结预期按授权更新：旧执行树仍严格锁定基准，未授权三文件新旧相同；本轮两文件的精确修改由新增白名单测试额外约束。不能把它描述为“144项测试文件完全未改”。

全部比较采用字节对照而非allclose；未发现数值不一致，不存在近似容忍或静默数值fallback。CPU_TESTS.json保存逐例、日志/JUnit位置及SHA。原有throughput回归只读核对过小型Smoke缓存；没有重新扫描大型Full Tensor缓存。

### 实际生产入口CPU微基准

同一固定合成分数、同一Fake编码/评分I/O，新旧各自真实Classifier.forward，每次清空分数缓存以强制相同miss。单线程、绑CPU0、i9-14900K、Torch2.5.1+cu124、FP32、Seed20261103；预热后交替7次（ICFG5次/trace3次），下表为中位数。

| 形状 G/P/每次文本数 | 原生产入口 ms | 新生产入口 ms | 新Classifier构造 ms（含Plan） | 本环节实测比值 |
|---|---:|---:|---:|---:|
| RSTP形状1000/200/1 | 29.508 | 1.787 | .606 | 16.52× |
| CUHK形状3074/1000/1 | 423.055 | 8.633 | 1.942 | 49.00× |
| ICFG形状19848/1000/1 | 2833.166 | 10.180 | 12.361 | 278.31× |
| CUHK形状+原式trace模式 | 1363.787 | 9.107 | 1.837 | 149.75× |

全部分数bitwise exact、RNG不变。这里测的是**合成模型I/O下的生产聚合入口**，没有真实IRRA/RaSa编码、融合或GPU同步，不能推算104分钟真实Query/Full会加速多少。trace代码没有修改，速度差来自相同trace模式下消除大量PID相等比较。

[BENCHMARK.json](BENCHMARK.json)保存每次计时与环境，[BENCHMARK.csv](BENCHMARK.csv)提供汇总。现有锁定执行配置与producer来源未改，**不能拿新代码直接恢复旧Partial**；后续新Attempt须另行授权并锁定新代码SHA。

## B. 原SEP与Idea的威胁模型差异

[原论文§4.1.2/4.1.5](https://arxiv.org/html/2605.24958v1#S4.SS1)使用BERT/ALBERT/RoBERTa/LSTM/TextCNN，攻击其中一个时用其余四个代理，N/E=4/4。固定官方main也排除目标模型。原分类任务具有正确标签y；它不是无标签TBPS生成协议。原实现还有Victim Clean门禁及末端成功统计，不能把“代理优化不使用目标反馈”等同于实现全程没有目标查询。

当前已停止的TBPS适配：

- 每域均加载该域IRRA+RaSa，两模型都参与置信度与tau评分；当前Victim恰好是其中之一。
- 按Test Gallery真实PID聚合，Query真实PID作为y；Victim Clean错误会提前NOOP。整体是Victim权重/标签可访问、score-based无梯度攻击，不是独立Target黑盒迁移。
- 三数据集分别用本域代理/GT重新生成，并不是“CUHK冻结Source→其他域”。即便本轮PID数值Exact，以上权限仍完全不变。

Idea冻结G(I)/H(T)在生成时不接受目标Test PID/配对或Target反馈。冻结Source模型供SEP在线搜索，与冻结生成器一次前向，也不是同样计算预算。比较至少要分别交代：可访问信息、源模型数量、训练数据、文本/图像模态、编辑单位和生成成本。

**没有GT PID时不能声称保持原正确身份条件目标。** 函数可以接受整数伪标签，但伪标签不等于真实y；Source预测也无法保证逐Query等于GT。若再把原Test Gallery换成Source训练参考图库，标签空间和优化语义都变了。保留官方函数体/DPP/候选搜索只能说明机制复用，不能说明原攻击定义或数值结果不变。

## C. 推荐的公平SEP-TBPS协议（设计候选，未采纳/实现）

### 必须满足的共同合同

1. 固定CUHK训练的IRRA为指定主Source；Source权重与参考资产提前冻结，不按目标域换用RSTP/ICFG模型。RaSa/APTM及跨域IRRA的Target权重只在独立评测进程加载，不进入生成器配置、分数、筛选或候选选择。
2. 生成只接收原Caption与存储用不透明ID；不向策略传Query的Test PID、正确paired image或Test Gallery PID分组。参考资产只来自批准的Source训练数据；不从目标测试注释补标签。
3. 生成所有预注册Query，不能用Target Clean正确性早退或Target ASR选候选/选Source checkpoint。Source自身的自一致性统计与Target攻击成功指标分开。
4. Source-only的tau排序一次确定最终文本，封存文本、来源、参数和SHA后，Target对同一文本独立评分。Target失败也不能触发重新生成/重选文本。
5. 每个Target使用其原生Tokenizer/Adapter/Checkpoint；全test captions×固定Gallery、PID-level正式评测。文本缓存可跨Target共享，IRRA特征不能冒充RaSa/APTM特征。

### 建议优先审查：Source参考PID＋Clean伪标签

候选名称可为`SEP-TBPS-SRefPseudoPID`，不能直接标为作者原始SEP。以下是明确的设计建议，**未选定或写入任何可执行目标函数/代理配置**。

- R_S：固定、可审计的CUHK **train** 参考图库；只使用Source训练身份c_j，禁止混入任何目标Test图像/标签/正确配对。具体完整图库或子集及SHA仍OPEN，不能按Target效果筛选。
- S_m：批准的冻结CUHK Source模型。所有模型共享R_S的Source类别顺序，各自用自己的原生编码/评分得到z_m。
- 主Source S_0在clean Caption上产生一次伪标签c0，之后保持固定。并列沿原顺序argmax first-hit，不能每次编辑重算“当前最佳”标签。

```text
z_m(c | T) = max_{j: SourceTrainPID(I_j)=c} score_m(T, I_j)
p_m(c | T) = softmax_c(z_m(c | T))
c0(T) = argmax_c p_0(c | T)                 # clean主Source预测，固定一次
h_m(T'; T) = max(p_m(c0 | T') - max_{c!=c0} p_m(c | T'), -k)
F_w(T'; T) = sum_m w_m * h_m(T'; T)
```

在这个候选中，官方同义词/POS、DPP和tau搜索机制可以保留，PID max仍可复用本轮Plan。**算法上的必要变化**是：正确Query Test身份y→Source clean伪身份c0，目标Test Gallery→Source train参考库，目标参与的oracle→只读Source伪预测/自一致性oracle；Target成功统计全部移到冻结文本之后。

原attack仍需要callable时，未来包装可以提供仅Source的oracle，而不能把Target参数删掉后留下隐式Target调用。clean伪标签来自同一主Source，其初始自一致性不代表真实身份正确；末端Source标签翻转也不是Target ASR。该目标攻击的是Source相对自身clean预测的稳定性，**不能再宣称优化真实正确PID或明确配对图排名**，也不保证Target检索效果。

只用Source训练标签并不违反“禁止目标Test真实PID”，但这仍是新的Source条件目标。Source参考库的在线评分也是额外Source计算；与H(T)的比较必须报告成本，不能称等计算预算。

备选`SRefPseudoImage`：完全不用PID分组，以Source参考图像ID为类别，固定Source clean Top1图像作为伪标签。它连Source训练PID也不需要，但类空间和损失语义变化更大，本轮不选。**正式采用PID版还是Image版、参考库、主Source checkpoint及代理清单均保持OPEN。** 本文建议优先审查PID版是因为无需新增分类头且可复用既有聚合，不代表已经批准该方法。

### 两代理、四代理与论文基线的区分

| 设置 | Source/Target隔离 | 标签与N/E | 可如何报告 |
|---|---|---|---|
| 作者原分类SEP | Victim排除 | 真分类y，4/4异构代理 | 原分类任务复现，不能直接冒充无标签TBPS协议 |
| 历史TBPS 2代理 | 当前Victim在IRRA/RaSa中 | Test GT PID，2/2 | 额外目标/标签信息可访问参考；不是公平独立Target基线，且当前0组Full |
| 候选SRefPseudoPID-S1 | 仅CUHK IRRA主Source，独立RaSa/APTM | 伪标签，1/1；DPP退化 | 若授权/验证，可作为**Source信息匹配的SEP-inspired TBPS基线**，必须标变体，不能声称论文4代理复现 |
| 候选SRefPseudoPID-S2/S4 | 两个/四个合格CUHK Source变体，所有Target排除 | 共享Source类空间与c0，2/2或4/4 | Source ensemble消融/参考；相对Idea单IRRA更多Source信息，不能自动称完全公平 |

S2/S4清单必须来自已有、来源合格的冻结Source资产；本轮不选、更不训练任何新模型。四个IRRA变体也不等于论文四种异构分类器。只把RaSa从Target挪成Source不能解决与IRRA-only信息预算不匹配的问题；若RaSa本身仍作Target，更不能让它参与生成。

**可解释的论文比较建议**：把严格Source-only、无目标Test标签的获批变体单列，先与Idea **Text-only H**对照，明确名称、伪标签目标、文本预算/语义限制和在线Source查询成本。两/四Source作为不同访问预算消融。历史含Victim/GT设置只能单列强信息参考。G-v2 Image-only或G+H Joint不是与纯文本SEP模态相同的对照，不混称等预算。

APTM应作为独立Target补充覆盖；同一源文本同时评测RaSa/APTM。跨数据集IRRA则是训练域转移，非跨模型：Source IRRA始终CUHK训练，Target IRRA可为RSTP/ICFG原生权重，生成只读取该域原Caption，不能替换Source权重或读取该域PID/配对。分别记录attack训练域、Source权重训练域、生成输入域、参考库域、Target权重域和评测域。

## D. 后续真实Smoke计划（NOT_RUN）

下列均须另行明确授权，当前没有可执行GPU配置或后台任务。

### D1：只验证A优化，不混入新公平方法

- 建立新版本Worktree/config/输出/control和producer SHA；旧44条、RNG及第45条Claim完整冻结，不在原Attempt恢复/重播。现有配置仍绑定旧生产来源，不能悄悄绕过SHA门禁。
- 预注册至少128张Gallery以覆盖RaSa完整Top128路径，2–10条新审查Query，排除旧44条和第45条身份；固定原Batch、原生精度、T/patience/DPP/预算和旧强信息协议。该Smoke只证明A接线，不证明公平。
- 同一原生分数请求序列只编码/评分一次，捕获分数带供原/新CPU生产入口重放；逐候选核验score位模式、PID成员/并列、softmax/margin、tau、所选文本、cache事件、所有RNG和冻结参数/buffer。避免把两次GPU本身非确定性误算成PID优化差异。
- 不一致立即FAIL保存首差异，不近似放行、不变更算法、不重试。独立CPU审计完成才能讨论任何后续Full。预算/非有限/超时也停止保留；外部超时是实验失败，不能转成NOOP或跳样本掩盖零向量缺陷。

### D2：公平变体单独审查与Smoke

先由用户确定变体/伪标签定义、Source参考库和N，另文档冻结协议后才实施。建议与旧GT适配分开模块/协议名，**不是本轮两个文件里顺手切换目标**。

CPU门禁先证明：参考类别至少2组且成员非空；生成API没有目标Test PID/paired字段；Target加载/评分函数在生成阶段一旦被调用即拒绝；伪标签只在clean主Source固定一次且所有代理共用同一Source类顺序；参考库为合格Source train且与目标Test隔离；候选选择和NOOP不依赖任何Target输出。

获批后先封存一个Source文本集，再在独立进程加载RaSa/APTM或跨域IRRA；Target只能消费冻结产物，不能反向传回源选择。独立审计核验来源/身份/文本SHA与正式全图库指标，Source伪标签成功与Target PID-mAP分别记录。

**本轮仍未决定未来代理集合或正式目标函数，未修零向量循环，未触碰B/C优化，未初始化CUDA运行时或启动Smoke/Full。** 测试、微基准和保护证据见[CPU_TESTS.json](CPU_TESTS.json)、[BENCHMARK.json](BENCHMARK.json)、[SOURCE_EVIDENCE.json](SOURCE_EVIDENCE.json)。
