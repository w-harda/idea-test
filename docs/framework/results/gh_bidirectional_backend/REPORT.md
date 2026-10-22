# G/H双向交替真实后端：CPU准备报告

开发基准：8ae623d3b6cb01f96c86e15e167d7dd7b1f56734。
独立分支：codex/gh-bidirectional-backend。
工作树：/home/lzf/ldx/projects/idea-test-tbps-gh-bidirectional-backend-20261022。

**本轮没有初始化CUDA，没有真实G/H/IRRA/reader前向、JVP、参数更新、图库生成或两轮Smoke。**
真实G60/H10的Checkpoint在CPU完成只读反序列化和完整状态恢复；真实reader权重仅用于无前向CPU组装检查。
CPU合成模型的梯度/Adam测试与真实模型执行严格区分。

## 核心结论

| 项目 | 实现与验证边界 |
| --- | --- |
| H→G | 原Canonical锚点与真实Unicode动作回放后，调用原FrozenLocalization.attach和TextTransport，重算当前文本z、w/e、JVP/FD，再调用原G单样本K+1训练函数。真实数值仍PENDING。 |
| G→H | 每次G更新阶段结束后，以该G版本流式刷新全部2048 Train图像，创建新RewardBank；原H train_step从新图库取得原Reward/SCST。真实刷新及更新仍PENDING。 |
| A1/B | A1用原G条件+最新图库；B增加H文本反馈。两组同完整父状态、样本顺序、原Loss/Reward/K/预算。 |
| 父恢复 | G60/H10权重、完整Adam及CPU RNG读回一致；CUDA RNG字节完整保留，尚未在CUDA设备恢复。 |
| 执行门禁 | 默认preflight；probe与smoke使用不同显式参数。smoke还必须核验同一源码/协议的不可变数值探针PASS回执。 |
| 原算法 | 不修改历史prototypes；不增加共享属性、强制K、新Loss或新Reward；推理保持G(I)、H(T)。 |
| 当前状态 | CPU后端准备；真实数值、CUDA Adam/RNG恢复、两轮双向更新与运行成本均待新授权。 |

最终测试数量、真实来源检查、资源快照及保护结果见SOURCE_EVIDENCE.json与CPU_PREFLIGHT.json。
开发期间CPU失败日志保留；未创建任何GPU Attempt。

## 1. 文件与原版复用

| 新文件 | 职责 |
| --- | --- |
| src/tbps/alternating/native_sources.py | 固定来源、真实传递导入闭包、版本检查、命名空间恢复、同一模块对象复用与精确源码编译。 |
| parents.py | G60/H10模型/完整Adam/RNG恢复；禁止CPU恢复阶段前向、求导及CUDA；设备迁移后重新绑定Adam对象并读回全部状态。 |
| token_bridge.py | 原生Byte-BPE/EOT检查；精确字节到字符Span接口，保留原mapper诊断。 |
| native_backend.py | 真实模型组装、原条件/训练函数、图库刷新、RewardBank、冻结核验、计数、不可变动作与条件Tensor。 |
| native_numeric.py | 用原投影、原Loss及原权重计算第一步G梯度；不执行Optimizer.step。 |
| native_gate.py | 显式probe/smoke授权和数值探针回执门禁。 |
| native_cli.py | CPU预检、资源快照、探针、两轮Smoke与独立CPU审计入口。 |

仅对已有runner.py作必要扩展：合成CPU后端仍可执行；真实后端必须持有有效ExecutionGrant，且通过数值探针。
旧“未授权GPU后端拒绝”测试和所有旧断言保留。没有更改历史训练器、缓存、Checkpoint或Selection。

G训练直接调用serial_training.train_single_sample；H直接调用phase4_training.train_step。
Global/Local/Retain、K+1顺序更新、最后原图Joint步、原SCST和R_sim+R_rank均不重新实现。
数值探针另用原attribute_losses/joint_losses和原_weighted计算梯度，只验证第一步目标；它不是新的训练算法。

## 2. 完整父状态

- G60：9148a6ac06178b2eb474194ef6892c50182c5623b4e88bdcf88faa358aa46049。
- H10：4ae964d51ea94840de71d9597d90a2234a36104e5d649cfee96577f59f94bacd。
- IRRA：9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009。

CPU读回G Adam全部step=484080；H各参数Adam step范围20124–20370，原完整训练游标为20480 visits/20370更新。
不同Head并非每次都有梯度，因此不能要求所有H参数的Adam step都等于20370。

G权重状态摘要：5d48caab6a891a1afd3aaaa6f3594eb5f860ccd7b6090708d289224e6c819e47。
H权重状态摘要：1b1e3888ee8dc0e4dff39af3438ec48987cd472746cbfde71ad4d10f562ad24d。
完整Adam/RNG摘要见CPU_PREFLIGHT.json的parent_restore。

先移动G，再创建绑定当前Parameter对象的原Adam，并读回全部父状态；这不等于重置动量。
CPU测试模拟Parameter换对象，验证新Adam拥有正确参数且完整状态摘要不变；真正CUDA迁移仍PENDING。
H保持CPU。两个父Checkpoint的全局RNG拆成G_RNG/H_RNG阶段域，阶段开始恢复、结束捕获；不把两套CUDA状态混成一套。
模型构造、reader加载均在训练阶段RNG恢复之前完成。

## 3. H攻击文本如何改变G

固定原Caption T、原Canonical、原G选中属性顺序、S_raw及权重α。H从原T产生T̃，不递归攻击上一轮T̃。

A1：z=z(T)，v_i=J_T[w_i(T)⊗unit(e_i(T))]。
B：z=z(T̃)，v_i=J_T̃[w_i(T̃)⊗unit(e_i(T̃))]。

B逐步执行：

1. 原constraints.replay校验完整原生EditRecord，保存原/攻击文本、原码位、新码位、位置、原生动作。
2. 用原Caption六字段属性锚点严格对应G所选属性，不靠G/H ID字符串、不重新抽取Unicode文本属性。
3. 字符数和字符Span保持；新UTF-8字节Span重新计算。更新当前raw_text和linked_object文本，原Canonical不变。
4. 原Byte-BPE重新编码，检查77-token、SOT/EOT、EOT位置及截断；不沿用旧token位置。
5. 原TextTransport获得新整句特征和block8隐藏状态。
6. 原FrozenLocalization在新隐藏状态上计算reader定位w、上下文e；原basis Q可复用，旧w/e不能当新值。
7. 原TextTransport.read逐原G属性重算真实JVP，检查zero exact及原central FD相对误差≤0.02。
8. 原condition_schema.build生成兼容载荷，以新外层身份发布；原S_raw仍明确是原文评分，不伪称新评分。
9. 把新z/JVP交给原G训练器。第一属性步的原Global+Local梯度随监督变化；后续Retain及原图Joint步继续执行原逻辑。

新条件空间另外绑定IRRA、reader、basis、字节接口及producer，不能只用IRRA表示维度充当条件身份。
实际condition.pt与动作均原子发布，独立保留来源、内容SHA和新语境证据。

### 必须审查的字节接口差异

原irra_token_alignment对“一个Unicode字符被BPE拆成多个字节片”保守返回utf8_character_split。
新ByteSpanBridge仅对可证明的该类字节区间，恢复其覆盖的原字符/UTF-8坐标；不改token ID、文本、reader或JVP数学。
保留native_alignment_status/reasons与逐token byte_span_proof，不能把新接口说成旧mapper已原生支持。

其他cleaning歧义、lowercase扩张、token重放失败、截断、跨属性边界及缺失覆盖仍拒绝。
真实CPU测试验证原生BPE ID不变、EOT定位、Unicode字节片及原Span边界拒绝。
这仅证明坐标接线；新语境reader/JVP的真实有效性尚未验证。

NOOP时完整复用原条件，新增文本/JVP为0；不把这种复用推广到任何非NOOP文本。

## 4. G变化如何反馈给H

每个G阶段结束，冻结刚更新的G，用原PIL RGB→bilinear Resize384×128→ToTensor输入桥逐图调用原generate_once。
图像在模型内保持FP32 RGB，不经JPEG/PIL二次量化。每图严格核验8/255，保存原图、Adv、effective_delta和IRRA特征。

必须完成同一G状态的全部2048有序Train成员，才发布bank。
不能使用8张新图+2040张旧图，也不接受跨臂、旧G或错误ModelSpace的bank。

使用原RewardBank新实例，旧Query context失效。H每次按原流程生成Sampled/Greedy，真实编码三条文本，原SCST更新H。
PID/paired index只进入Train奖励绑定层，不进入H.forward或H.attack。

对同一组已编码的当前文本，额外用原RewardEvaluator分别计算刷新前后bank的Reward，记录Reward及Advantage差；
该诊断不增加文本编码、不参与策略选择，也不增加新的Reward项。
即使Reward或Advantage差为0，也不强制动作、更换样本或修改奖励。
流水线审计与“可测的双向更新证据”分别标记，不能仅因程序跑完就宣称双向机制已被证明。

旧G60 bank内容SHA继续使用其原g60_h_bridge.tensor_sha定义；新联合状态使用tree_sha，两种摘要不会混用。
旧2048 Clean/G60特征已通过真实CPU读取、原审计内容及完整有序成员核对；这不是重新生成图库或重跑旧审计。

## 5. A1/B两轮协议与计算量

固定8条Train，顺序不变：
cuhk:26530:0、cuhk:26747:0、cuhk:12746:0、cuhk:9955:0、
cuhk:29253:0、cuhk:21540:0、cuhk:3127:0、cuhk:8385:0。

原G K为0/1/2/3/4/3/3/3；首条是固定NOOP资格控制，其余7条合法。
每臂独立复制同一完整G60/H10、双Adam/RNG；两轮沿各自状态继续，不继承另一臂。

| 阶段 | A1 | B |
| --- | --- | --- |
| H greedy | 原Caption、当前H | 相同接口 |
| G条件 | 原文条件 | 非NOOP重算新语境；NOOP复用 |
| G更新 | 原8条、每轮27个Adam step | 相同原K和原Loss |
| 奖励图库 | 当前G完整刷新2048张 | 相同要求 |
| H更新 | 原8 visits、7次合法更新、最新bank | 相同原SCST/Reward |
| 提交 | 完整round状态及来源 | 同样不可变提交 |

两臂两轮总上限：G训练108次更新；H32 visits/28次合法更新；4次完整bank刷新。
G图像样本前向及IRRA图像样本前向各8300=8192+108。
B最多14个非NOOP语境/38个属性JVP；IRRA文本样本上限226，包括新语境/JVP/FD及28×3个SCST文本。
H整策略样本前向上限84，reader调用上限14；这些计数不是网络内部每个Layer的调用数。

单臂完成两轮的游标为G Adam484134、H20496 visits/20384更新。
这不是完整G61或H11 Epoch，更不是正式60轮联合训练。

每个bank三张RGB Tensor约3.375GiB，四个bank约13.5GiB；加特征、状态、元数据和回执预计14–16GiB。
包含每图Claim和独立artifact，约需7.4万个inode，正式资源门禁见执行协议。
运行时间尚未实测：探针预计数十秒至数分钟；完整Smoke暂估15–60分钟，SHA/fsync及共享GPU会显著影响速度。

## 6. 恢复、冻结与失败保护

- 新IRRA加载在legacy会话之外；CPUStartupGate检查CPU隔离退出后，才允许CUDA初始化。
- G/H旧模块以核验后的同一模块对象重装，避免TextExample类型身份变化和H动态import冲突。
- Kornia传递导入有CPU辅助对象构造：独立禁止权重、前向、求导、CUDA后导入；严格旧闭包阶段继续禁止神经构造。
- 精确核验当前CLI的__main__/__mp_main__别名；不放行任意legacy命名空间。
- IRRA/reader始终eval、requires_grad=False、无参数梯度，并核对Tensor版本及最终完整state摘要。
- G阶段只允许G更新，H阶段只允许H更新；原K、原Adam超参数和原函数保持。
- round开始先Claim，图库每单元先Claim，再前向/原子发布/读回；动作和新条件亦不可变保存。
- 未完成Claim、pending、FAIL及后续Claim阻止自动重跑、回退或恢复；不提供自动恢复入口。
- 完整round边界保存双模型/双Adam/两RNG与父来源；只提供明确完整边界读回接口，CUDA后续轨迹不声称bitwise等价。
- 任何来源/FD/预算/冻结/数值/资源或OOM失败均停止，保留现场，不改阈值、不换样本、不重试。

## 7. 本轮测试与仍未验证项

测试包含原162项回归及新增真实后端CPU检查：
授权与探针回执、SHA/来源隔离、命名空间成功/异常恢复、真实G60/H10完整CPU恢复、
Parameter设备绑定、原生BPE/Unicode/EOT、固定8条属性、原Clean/G60 bank读取、
2048全顺序Mock流式刷新、Claim及失败停止、原数值探针CPU控制流与零Optimizer更新。

真实CUDA上的Adam/RNG恢复、新reader定位、JVP/FD、非NOOP梯度差、NOOP梯度数值、
完整新bank刷新、H真实Reward/Advantage差、两轮完整恢复与显存/耗时均为PENDING。
合成CPU通过与真实数值探针通过不是同一验收状态。

继承原S_raw的34054张official Train参考图库含内部Validation图像；本轮不重算、不宣称gallery-disjoint。新奖励bank严格限定2048 Train，不使用新Validation/Test反馈。

执行命令及明确预算见EXECUTION_PROTOCOL.md。**本轮到此停止，等待新的GPU授权。**
