# GH-Alternating-v1：WordBudget1双向交替训练与两轮真实Pilot

日期：2026-11-06。**实现、两轮真实GPU Pilot及独立CPU审计PASS；正式长训练和官方Test/Full均NOT_RUN。**

执行源码Commit `9080d1accf51da7a9b9776b1d051ef24f62b5d9f`，分支 `codex/gh-alternating-v1`。配置SHA `31d77d919fbccadf31999764caabf42714fc658ba41e15c3e06a9817f9bf7c7a`。本报告读取既有新产物，不重跑GPU或审计。

## 1. 复用与新增范围

原146份框架producer、496项运行来源与G/H prototypes逐字节保持。复用NativeBackend的条件重建、reader/真实JVP、原G `train_single_sample`、完整Bank刷新、原H `phase4_training.train_step`/SCST及FormalJournal不可变发布。没有重写G、H、IRRA、Loss或Reward。

| 新文件 | 关键能力 |
|---|---|
| `word_budget.py` | `check_words/candidate_guard/install`：调用原rollout，仅增加候选支持集和输出预算 |
| `configured_sources.py` | 新配置、完整producer/运行来源、原146源码保护、CPU门禁与父恢复 |
| `configured_training.py` | `ConfiguredBackend/run_rounds`：多pass、独立Scheduler、阶段隔离、当前Bank和完整状态 |
| `configured_cli.py` | CPU默认入口、显式Pilot/Train授权、共享GPU观察、边界恢复与资源日志 |
| `configured_audit.py` | 独立CPU审计：状态链、原Unicode回放、预算、真实JVP与完整Bank |
| `operations/gh_alternating*.py` | 显式配置冻结、命令入口、单次Supervisor |
| `test_gh_alternating_v1.py` | 23项新定向CPU用例 |

## 2. 实际每轮数据流与权限

当前H冻结、greedy生成T_adv → 保留原Canonical锚点/S_raw/K/顺序，非NOOP重算整句z_t、reader定位和真实JVP → 仅G执行原K+1更新 → 最新G一次生成完整2048张Train Bank → 仅H执行原sampled/greedy SCST → 保存双模型/Adam/Scheduler/RNG、Round游标和Bank引用。

G推理仍为单次G(I)，H推理仍只接原Caption及Caption-only属性。训练配对/PID只在Source Train绑定层；没有Target Test反馈。历史S_raw参考34054张官方Train图且含内部Validation，沿原协议保留，不冒充新2048 RewardBank。

## 3. 配置、预算与初始化

完整配置及启动/停止/恢复说明见[运行说明](../../GH_ALTERNATING_V1.md)。正式候选rounds=50、g_passes_per_round=h_passes_per_round=1、G/H LR=2e-4/1e-4、seed42；独立Scheduler默认gamma1保持常数，支持显式StepLR。Checkpoint间隔1，可配置正整数里程碑；每个完整Round始终保存恢复状态，里程碑仅发布引用不重复Tensor。

从独立G60/H10完整恢复双Adam动量、各参数历史step、G/H阶段Python/NumPy/Torch CPU/CUDA及采样RNG。配置LR显式记录，新Scheduler从Round0开始；不伪称此前存在旧Scheduler连续状态。不是从旧严格审计失败Smoke续训。

新WordBudget1按原文连续非空白片段的位置计数，保留分隔符/标点/连字符；维持原每属性/共享词最多一次字符替换，所以每条最多1词且该词最多1字符。Natural-K仍独立决定属性选择槽，不改为K=1；候选在首个实际编辑后仅允许STOP/NOOP。超预算拒绝，没有后处理截断。旧H7/H60不能改标为WordBudget1。

## 4. 两轮真实GPU Pilot

原8条Train样本参与更新，每轮完整2048成员Bank。Round1正常提交后退出3，新Python进程显式恢复Round1再执行Round2；这不是失败重试。

| Round | G更新 | H访问/更新 | 新JVP | 新Bank | 非零当前H梯度 | Round秒 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 27 | 8/7 | 19 | 2048 | 3/7 | 391.911 |
| 2 | 27 | 8/7 | 19 | 2048 | 5/7 | 403.515 |

总计54次G更新、16次H访问/14次H更新、38个新JVP、4096张新图；最终游标G484134/H visits20496/H updates20384。实际计数G/IRRA image各4150、H42、IRRA text184、reader14。没有额外G重编码或Target评分。

每轮G/H参数均改变。G阶段H及H Adam指纹exact、H梯度None；H阶段G及G Adam指纹exact、G梯度None；IRRA/reader始终eval/冻结，版本与最终完整SHA断言通过。

14个非NOOP rollout均实际重建条件，未重标旧JVP；原zero/2% FD门禁通过。新G的两份Bank分别绑定生成它的完整G state.pt、G SHA及2048成员顺序，所有14个有效H步使用当前Bank。同一文本在刷新前/后Bank的Reward在14/14步变化。

**证据边界：**Round2确实读取更新后的H参数，但本固定8样本的greedy文本变化为0/8，条件Tensor也保持相同。CPU定向干预已有H Head能改变合法文本，CPU连续两轮测试验证更新态传递；本Pilot证明接线、更新及状态链，不证明H已经学出新的greedy动作或攻击性能提高。8/14步有非零当前H梯度，其余零梯度步的参数变化可由已有Adam动量产生，不能统称为新反馈学习。

## 5. 预算与恢复审计

独立CPU审计对全部4096张持久化图像检查FP32/[0,1]/effective_delta/严格L∞，两轮最大均0.0313725471496582≤8/255。44份动作输出（含rollout及sampled/greedy，非44个独立Caption）逐项核对WordBudget1；42份实际编辑、2份固定NOOP。原Unicode白名单、Canonical Span、码位/UTF8偏移及完整EditRecord通过原replay核验。

完整状态含双模型、双Adam、双Scheduler、双阶段全RNG、绝对更新游标和配置。Round2从新进程恢复，原恢复函数逐值检查G/H/Adam，阶段开始检查CPU/CUDA/采样RNG，新增Scheduler状态exact。恢复前核对全部Round1 Bank文件，再安装该Bank，未回退初始G60奖励。CPU连续/边界恢复全状态exact；未执行GPU连续两轮对照，不承诺CUDA未来轨迹bitwise一致。

独立审计状态为 `GH_ALTERNATING_V1_PILOT_CPU_AUDIT_PASS`，READY仅为后续另行授权准备，不自动训练。原旧 `first NOOP control differs` SHA保持，不能改签旧严格A1/B实验。

## 6. Loss、奖励与相互抵消诊断

以下是Train诊断，不是TBPS正式指标。完整原精度见PILOT_ROUNDS.csv及FOUR_MODE_DIAGNOSTICS.csv。

| Round | G Global均值 | G Local均值 | G Retain均值 | G Total均值 | H sampled reward | H greedy reward |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.159295689 | 0.000037311 | 0.000083113 | 0.159416113 | -0.029530432 | -0.003431595 |
| 2 | 0.109465081 | -0.000405211 | 0.000046906 | 0.109106775 | -0.000714853 | -0.003041640 |

| Round / H | Clean相似度 | Image-only | Text-only | Joint | Joint高于Image-only |
|---|---:|---:|---:|---:|---:|
| 1 / sampled | 0.505352 | 0.131165 | 0.499703 | 0.146143 | 5/7 |
| 1 / greedy | 0.505352 | 0.131165 | 0.490527 | 0.130106 | 4/7 |
| 2 / sampled | 0.505352 | 0.096679 | 0.494368 | 0.097210 | 4/7 |
| 2 / greedy | 0.505352 | 0.096679 | 0.490527 | 0.095451 | 4/7 |

存在局部抵消：若Joint高于Image-only，当前Unicode动作提高了该配对相似度。SCST优化sampled−greedy的R_sim+R_rank，既不保证每个动作奖励为正，也不保证多正例mAP下降；零advantage时Adam动量仍可更新参数。以上能解释优化目标与观察量的区别，不能确定唯一退化原因。未添加Push-Pull、synergy reward或其他新Loss。

## 7. CPU/GPU资源、耗时与正式预算估算

GPU仅观察，无固定空闲VRAM floor，精度未变。H策略CPU，G/IRRA GPU；外部进程未干预。CPU百分比按1核心=100%计算，源于同一进程阶段累计CPU时间差，GPU利用率为阶段快照、非持续采样均值。

| Round | G及条件秒 | Bank秒 | H秒 | CPU秒/平均占核% | peak RSS MiB | GPU allocated/reserved MiB |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 5.211 | 386.135 | 0.565 | 120.88/30.83% | 2310.75 | 1034.98/1080.00 |
| 2 | 3.009 | 399.980 | 0.525 | 121.14/30.00% | 2329.65 | 1034.98/1080.00 |

两轮Round段合计795.426秒；含两个进程预检、加载、恢复及独立CPU审计的Supervisor总计963.611秒（16.06分钟）。每轮完整Bank约386–400秒，是Pilot主要耗时。资源原始阶段记录在服务器control/resource_stages.jsonl。

正式50轮候选配置：新增403400次G更新、102400次H访问，按当前资格101850次H更新，50份完整Bank/102400张图，新JVP上限301000。图库三RGB Tensor纯数据约168.75GiB，另有状态/来源；磁盘配置预留250GiB。额外G训练及刷新使其不等于固定G训练H的计算成本。

按本Pilot的G更新/语境规模比例（约256–317倍）与H有效访问比例（2037/7）、每轮相同完整Bank外推，50轮粗估约18–32小时，另需完整审计/报告时间；完整2048动态新语境、长期资源和最终性能未实测，该值不是运行承诺。

## 8. 验证、保护、运行与未执行项

23项新增CPU＋27项已有直接相关用例，共50个不同用例通过。覆盖真实原H三Head反传/样本及贪心预算/参数与Optimizer对象保持、H状态干预、独立多pass、双Scheduler和CPU完整边界恢复、Claim拒绝、完整新审计与损坏拒绝、0空闲VRAM仍允许观察策略。真实Pilot/Train CPU预检均PASS；没有重跑无关大回归。

开发期临时_config_path比较问题在未消费GPU Attempt时修正；原失败日志/配置修订保留。早期测试中随机Head干预未必改变greedy、合成CPU RNG缺少CUDA结构字段，均修正测试构造，未放宽真实门禁。真实GPU Attempt一次，没有失败重试。

原146源码、当前151 producer、496运行来源及G60/H7/H10/H60文件SHA复核exact；历史151378 metadata/28旧FAIL与Watcher保护通过来自本轮CPU预检。未修改SEP/C-PGC运行树、控制器、配置或产物。

运行入口、后台Supervisor、状态/正常停止/显式边界恢复与完整参数见[运行说明](../../GH_ALTERNATING_V1.md)。官方评测将复用原生Engine，但本轮没有训练长Full、选择新Checkpoint、生成Test攻击输入、计算Test排名或加载RaSa/APTM。后续比较应使用同WordBudget1及可说明的父状态/访问预算；旧多词H7/H60仅作冻结历史参照。

证据索引：[SOURCE_EVIDENCE.json](SOURCE_EVIDENCE.json)。Checkpoint、Tensor、原图、原Caption缓存和日志仅保留服务器，不提交Git。
