# 项目当前状态（唯一权威入口）

状态来自Git、实际代码与不可变实验记录；旧阶段报告与旧Commit中的PENDING/等待状态是历史快照。

## 当前阶段：G60 × H1–H10内部Validation探索性适配诊断PASS

基准be30206f，冻结执行048fafc15040f90f7152f78a3b4d48728bd300ac；独立codex/g60-h-validation。
32新/受影响CPU用例、129既有通过用例及1原路径冻结用例，共162个不同用例验证PASS；原失败日志保留。
一次128图GPU运行：前2图Smoke/CPU读回PASS并保留计入总量；G60/IRRA各加载1次、图像各128前向，
H/新文本编码/Backward/Optimizer/JVP均0，模型参数buffers exact，严格8/255。
Image-only与10组Joint新评分、271个产物的独立CPU审计PASS；150417历史文件stat及6564消费来源SHA/stat不变。
执行段39.319秒，GPU reserved866MiB、Peak RSS2342.36MiB；共享外部任务未干预。
G60 Image-only mAP48.80861%；Joint最低H9 40.42548%，H7 41.59334%排第4，差1.16786pp。
G10最低仍为历史H7；仅重复128内部Validation探索性诊断，不覆盖正式H7 Selection、不使用Test选模。
[完整报告/10候选/原精度CSV/3图](framework/results/g60_h1_h10_validation/REPORT.md)；
[实际执行及来源证据](framework/results/g60_h1_h10_validation/SOURCE_EVIDENCE.json)。
完成后停止，未启动H60训练、新奖励、官方Test或跨模型验证。

## 已完成：G-v2六节点三模型Full及只读分析


冻结执行6021f3bfdec181785d1baccd7ce4f80a1428b069，启动5b187151432dacf5618f7759ed10a211dbf23e2e。
2026-10-20 20:35:12→2026-10-21 19:07:27 +08:00，总耗时22小时32分15秒。
最终GV2_FIVE_NEW_EPOCHS_THREE_TARGET_FULL_AND_CPU_AUDIT；五个新Epoch各3074图、15组Target Full及15份既有独立CPU审计PASS。
E10三模型已审计历史结果/审计来源只读引用；H7/6156文本固定、不重新生成。

本轮仅标准库CPU读取和派生绘图：34721 artifact、94058独立文件SHA/约64.16GiB字节核验，
15370个Claim与已发布图像配对，无pending/孤立Claim/失败/缺失；36攻击记录+6固定控制、168指标原精度exact。
107 producer及六Checkpoint保持，74184历史stat/289 SHA及128810份读取文件stat保护通过。
没有GPU、模型加载/张量反序列化、攻击、训练、评分/排名或原CPU audit入口重跑。

六节点Image-only mAP IRRA32.28→17.51、RaSa35.05→26.47、APTM39.66→32.11；
Joint23.55→13.08、17.20→12.57、30.49→23.74；全部mAP序列严格下降。
APTM Image-only R@1在E20→30小幅回升约0.065百分点，不能称全部指标逐区间单调。
六节点最低mAP均为E60，仅单seed/单轨迹描述，不作Test选模、显著性或过拟合因果结论。
[正式报告/全部表格/六图](framework/results/gv2_epoch_sweep_full_analysis/REPORT.md)、
[完整性](framework/results/gv2_epoch_sweep_full_analysis/INTEGRITY_REPORT.md)、
[来源证据](framework/results/gv2_epoch_sweep_full_analysis/SOURCE_EVIDENCE.json)。
CSV原精度/两位小数、DR_mAP/DR_R1/百分点、六图PNG/SVG及统计JSON均在同目录。
旧RUNNING/PENDING和未授权语句为当时快照；本轮只提交派生文档，不改旧实验或算法，整理后停止等待讨论。


## 已完成的前置阶段：G-v2六节点准备的唯一真实Smoke及三CPU审计PASS，Full等待最终授权

执行代码152498fd258572ad9c43c04b2ce8fba8b9e995b4；同一独立codex/gv2-six-epoch-evaluation工作树。
用户明确授权共享GPU；仅resources准入8192MiB/利用率≤10%改为6144MiB且利用率仅监控，
对应producer SHA正规更新，其余运行函数AST exact，Target/Checkpoint/模型/预算/精度/输入不变。
适用CPU234 PASS（43当前含9资源用例+99公共+92原生）；旧APTM绝对路径配置单项仍按基准排除。

2026-10-20 19:57:41→20:13:31 +08:00，整条命令950.148秒，唯一smoke_v1。
G10/G60各加载1次、10次前向；G10对历史Adv RGB最大误差0，
原/Adv/effective_delta的30项内容SHA exact；G60实数L∞0.0313725471496582≤8/255。
IRRA/RaSa/APTM各加载1次/编码10图/0文本，RaSa/APTM各200融合pair；
H加载/前向0，Backward/Optimizer/JVP0，全部模型参数/buffers前后exact。
三份独立CPU审计PASS后发布GV2_SIX_NODE_SMOKE_PASS，没有重试。
自身GPU最大reserved2082MiB，最大Worker RSS4.075GiB；共享其他任务一直保留，无进程干预。

74184份相关历史stat、289项SHA及六Checkpoint不变，107项运行来源复核通过；无pending/残留Worker。
原0dba135资源阻塞/CPU准备记录保留，当前Source报告与旧快照分别索引，不回填旧证据。
[实际Smoke报告](framework/GV2_EPOCH_SWEEP_SHARED_SMOKE.md)、
[原精度/完成回执/源SHA证据](framework/results/gv2_epoch_sweep_shared_smoke/SMOKE_EVIDENCE.json)、
[正式计划及命令](framework/GV2_EPOCH_SWEEP.md)。
技术Smoke门禁已满足，但**正式五节点Full仍未授权/启动**，Top10不能宣称Full Top128新验证。
真实报告/CSV/证据内容Commit：5dfc0834034a77cb9b53ac2ebc7fe158321e9496；执行代码152498f已推送。
本次后续提交仅登记发布索引；提交推送后停止等待最终Full授权。

## 已完成：G-v2 Epoch11→60训练及CPU审计PASS，Git派生报告格式修复

冻结训练635471b，50个新增Epoch/403400更新完成，Adam最终484080；
Epoch11–60全部50份Checkpoint及关联日志/完成回执已由原生独立CPU审计PASS。
本轮仅只读SHA/元数据核对，没有重新运行审计、GPU、训练或Full。
CSV650字段与原摘要exact、Markdown60行一致，六点Checkpoint SHA核验通过。

原监督器在PUBLISHING_REPORT的git diff --cached --check因CSV的61个CRLF行尾、
Matplotlib SVG的1117处行尾空格退出，phase=FAIL exit=2 line=83。
只修派生CSV/SVG空白格式，不修改生成器或任何训练源码；CSV单元格和SVG图形语义exact、PNG不变。
原20份控制文件及567份正式训练文件SHA/mtime保持；原FAIL事件/phase/日志不改写。
修复后的报告内容Commit：e036c1eb29339441e30b643ed1c4a77a1d4590e6；发布凭证在控制publication_repair_v1/publication_success.json；
不能将后续发布成功冒充原监督器整链成功。后续Gallery生成/三模型Full仍待授权。
[正式训练报告与修复说明](framework/results/gv2_epoch60_training/REPORT.md)；
[曲线](framework/results/gv2_epoch60_training/loss_curves.png)；
[原始精度CSV](framework/results/gv2_epoch60_training/loss_raw.csv)；
[发布修复证据](framework/results/gv2_epoch60_training/PUBLICATION_REPAIR.json)。


## 已完成：VLPTAttack（DRA）v2三模型Full及CPU审计完成，正式结果只读汇总

实际执行34097c338997561fd7556e43c472d2e5625c05f0，启动登记c97c2bf6c5bcbf0a0d70560fbc54d339adb577e6。
2026-10-19 16:38:15→2026-10-20 09:51:44 +08:00，整链17小时13分29秒；
Source、IRRA、RaSa、APTM与既有独立CPU审计全部PASS，最终VLPTATTACK_OFFICIAL_IRRA_SOURCE_THREE_TARGET_FULL_AND_CPU_AUDIT。
385批覆盖3074图/6156文，IRRA与MLM各加载一次，6153句编辑/3句NOOP；三个Target共享同一冻结内容SHA。
Joint R@1/mAP：IRRA0.00/0.10、RaSa16.34/13.03、APTM24.42/19.85。
IRRA原精度下Joint mAP略高于Image-only，不能称Joint所有指标均优于单模态。
官方RAttacker SHA75518cb9…保持，源/MLM/Target参数buffers exact；FP32预算保留已批准舍入语义，无二次投影。
DRA/官方SGA的Matched Clean四指标原精度一致；历史G/H为Raw Clean，比较披露预算、文本窗口/语义、Test权限及成本差异。
本轮独立结果分支codex/tbps-vlpt-full-results；只读核对48个指标、385批身份/动作/RNG摘要和完成链，
未导入模型/反序列化Tensor、GPU、攻击、评分、排名或audit入口；正式审计为原已完成审计。
原50163份实验/来源文件stat及5551份使用元数据/来源SHA读取后保护；不改旧产物或实验代码。
[正式中文报告与CSV](framework/results/vlpt_official_irra_full_20261020/VLPTATTACK_OFFICIAL_IRRA_FULL_REPORT.md)，同目录保留原精度及SHA证据。
报告/CSV/证据内容Commit：467b54193f4e1110006df0f9a4202dc26c7bf534（codex/tbps-vlpt-full-results，已推送）；本次仅补发布索引。完成后停止，下一Baseline等待授权。

## 历史启动快照：VLPTAttack Full v2已真实后台运行Source，首批8图16文提交通过

基准055fea880214c95f3b8eee6282bb9de91ea1b742；
分支codex/tbps-vlpt-full-v2，独立工作树idea-test-tbps-vlpt-full-v2-20261019。
仅dra_full.py拆分Source/Target预检；Source主流程除stage关键字外AST exact，
官方RAttacker、Wrapper、原生Target评分、模型和算法参数不改。
两套真实Python CPU预检均exit0；Source47项、Target153项CPU测试通过（环境专用用例各跳过3，Target历史配置排除2）。3074/6156/385、10 Caption成员、全部SHA检查保留，
model_loads=0/CUDA=false。v1全部18份FAIL证据SHA/mtime/size保持。
v2控制vlpt_official_irra_full_v2，输出tbps-framework-vlpt-official-irra-v2/full；
未复用v1身份。本轮授权资源通过后启动一次，失败不重试，不创建v3。
实际运行代码34097c338997561fd7556e43c472d2e5625c05f0；
16:38:15启动，后台两套预检PASS，16:38:32进入RUNNING_SOURCE。
监督器PID/PGID/SID1307026，PPID1、TTY=?，Source子进程1307611。
16:42:11核验已提交1/385批、8图/16文，第二批开始；首批原官方text1/image/text2完整，
45次区域采样、100次图像backward、SHA和读回通过。无Target版本错误。
Full和最终CPU审计仍PENDING，不宣称Full成功。按授权确认启动后退出，不持续监控。
[启动报告](framework/DRA_FULL_V2_STARTUP_REPORT.md)、[快照证据](framework/DRA_FULL_V2_STARTUP_EVIDENCE.json)。
[修复与执行说明](framework/DRA_FULL_V2_FIX_REPORT.md)。

## 历史FAIL：VLPTAttack官方IRRA Source Full首次启动FAIL，未加载模型/生成产物

基准889a57c，独立分支codex/tbps-vlpt-official-full、工作树idea-test-tbps-vlpt-full-20261019。
只新增Full外层调度、385批像素/文本/RNG证据、三模型原生评分、独立CPU审计和后台监督器；
官方RAttacker与既有Smoke/Wrapper逐字节保留。42项DRA/Full CPU、148项适用公共回归PASS，
两项旧绝对路径配置测试显式排除，详见执行文档。真实Full CPU预检3074/6156/385通过，
10-Caption成员cuhk:38920:0、最后2图；无GPU加载。资源通过后按本轮授权启动一次，不自动重试。
输出/home/lzf/ldx/outputs/tbps-framework-vlpt-official-irra/full/，
控制/home/lzf/ldx/tmp/vlpt_official_irra_full_v1/。
实际执行代码fde8f6b23af09e18fc1ad2a8da9aacc241b89cb2。13:54:14启动，
监督器PID/PGID/SID870320，实测PPID1/无TTY。Target环境CPU预检及资源检查通过；
13:54:31 Source重复完整预检时把RaSa依赖校验带入独立Source环境，
触发RaSa runtime dependency version mismatch。监督器已退出，phase=FAIL；
IRRA/MLM加载、攻击/Claim/图文产物均0；未重试，原控制与source.log保留。
需后续最小CPU修复阶段化预检，并分别在两环境验证，再另行授权新Attempt。
不能宣称Source开始处理或Full成功。RNG可追溯，不支持自动恢复。
[启动失败报告](framework/DRA_FULL_LAUNCH_REPORT.md)、[证据](framework/DRA_FULL_LAUNCH_EVIDENCE.json)。
[执行说明](framework/DRA_FULL_EXECUTION.md)。

## 已完成：VLPTAttack（DRA）官方IRRA Source最小接线与8图Smoke完成

基准715f636；发布分支codex/tbps-vlpt-official-complete。原官方RAttacker.py固定91f78325… / SHA75518cb9…，字节及方法保持原样；没有重新实现攻击或改Loss/候选/梯度。初始失败代码230ff9e、成功Source8513a49、成功Target d455f3a各自保留在独立工作树；最终CPU审计仅补json import。本阶段代码、保真审计与结果报告提交：1fac414e2edf605d44194f949d301fc12e236582；后续仅登记发布索引。

固定Test前8图/全部16 Caption，完整原三阶段、Batch8、8/255、10轮、2/255、原5候选/整数三角采样/五尺度、BERT77/IRRA77/pre_caption30。
成功Source实际45次区域采样、100次图像backward、1144图像/1026文本样本，IRRA/MLM各加载1次；源参数允许求导但无Optimizer，参数/buffers前后exact。16条最终原词槽位替换、NOOP0；1条官方多WordPiece候选展开为短语，未过滤。全部桥接exact、无额外截断。

Source133.38秒（完整攻击含被动追踪122.22秒；图像子段28.59秒）、allocated7.2386/reserved9.1914GiB，RSS2.2778GiB；9轮选择时allocated无持续累积。官方FP32预算通过，实数超8/255约2.793e-8公开保留，未二次投影。三Target使用同一Source Collection，各16图像/32文本样本；RaSa/APTM各512融合pair，权重前后exact，无Target反馈。

三个Target评分及独立CPU审计PASS；8图R@10不适用。Joint Smoke mAP IRRA29.11、RaSa93.58、APTM98.96，仅链路验证，不判断方法效果。26新增+180适用回归PASS；1项旧APTM绝对路径测试在原基线也失败，显式排除，不改旧文件。历史18924文件stat、两个早期FAIL19文件、源/旧IRRA FAIL42文件、全部GPU产物174文件保护通过（集合有交集）。

所有失败保留：缺safetensors、外部空格词数误判、IRRA会话缺标志、CPU审计缺import；前三者分别独立Attempt/目标会话，不覆盖或把旧FAIL改PASS。全任务Source恢复3次、MLM2次、官方完整攻击2次，仅最后一批提交；IRRA Target恢复2次，RaSa/APTM各1次。没有隐瞒重试，也没有全量生成/Full/训练。

[阶段报告](framework/DRA_SMOKE_REPORT.md)、[保真审计](framework/DRA_OFFICIAL_FIDELITY_AUDIT.md)、[命令](framework/DRA_IRRA_SOURCE.md)、[原精度证据](framework/DRA_SMOKE_EVIDENCE.json)。
OFFICIAL_ALGORITHM_FIDELITY_PASS限定本固定环境/小样本移植；Full范围尚未实现/授权，保留原Batch8后估计整链18–22小时，后续10-Caption GPU峰值/Token边界/长期趋势仍待正式运行验证。
方法比较按各自原生配置进行，主方法为G-v2+H-v1 Joint；披露权限/预算/输入差异，不据此否定方法级Baseline比较。完成后停止，不自动Full或新开发。

## 已完成：官方SGA（IRRA Source）三模型Full完成，独立CPU审计PASS，结果只读汇总完成

正式执行Commit f8d3431b4f51d4fb117cf031f6d827bb89fab69c，启动登记3caacf0ac124ac532febb587a290721c2d4d6608。独立结果分支codex/tbps-sga-official-full-results，基准3caacf0；结果报告提交：b680dfda020395dfebe56c771c883a6b2f61db2a；当前文档提交补充发布索引。

2026-10-19 00:51:36至06:07:53 +08:00，总计5小时16分17秒；Source、IRRA、RaSa、APTM及独立CPU审计均PASS，控制目录v2 phase=SUCCESS，最终SGA_OFFICIAL_IRRA_SOURCE_THREE_TARGET_FULL_AND_CPU_AUDIT标志存在。原v1路径检查FAIL及所有旧实验保留，不回填成功。

唯一IRRA Source使用固定官方SGA f49e6926…完整三阶段、2/255、10轮、0.5/255、batch8/seed42/五尺度；385批覆盖3074图/6156句。三个Target绑定同一冻结图文内容，不重新攻击、不用RaSa/APTM反馈；参数buffers前后exact。6145句词替换、11句NOOP全部保留。官方FP32边界通过，实数舍入高出2/255约2.93348e-8，不声称严格数学上界通过。

主要SGA Joint R@1/mAP：IRRA 2.70/2.29、RaSa 39.13/35.64、APTM 42.59/37.59；来自正式Full，非Smoke/旧白盒。SGA同源最终指标低于历史G-v2+H-v1 Joint；历史主方法在RaSa/APTM迁移最终指标更低。Matched Clean/Raw Clean、2/255与8/255、MLM词替换/Unicode字符、Test配对权限及推理成本不同，不作严格公平优劣结论，不用Raw Clean计算SGA下降率。

[正式报告与论文CSV](framework/results/sga_official_irra_full_20261019/SGA_OFFICIAL_IRRA_FULL_REPORT.md)，同目录两位小数CSV、原始精度及来源SHA证据JSON。方法顺序Clean→SGA Image-only→SGA Text-only→SGA Joint；模型IRRA→RaSa→APTM；仅R@1/5/10/mAP。

本轮标准库只读核验4647份产物Manifest/回执及声明文件SHA、21335份来源stat、完整官方身份/顺序/Claim、共享Adv内容和48项CSV/summary数值。原独立CPU审计已执行并核实，本轮不重跑审计、Tensor/模型加载、GPU、攻击或Full排名。完整来源保护索引仅留服务器tmp，原实验文件未改。完成后停止，不自动开始VLPTAttack。

## 已完成：官方SGA Batch8容量测试PASS与Full后台入口准备

独立分支codex/tbps-sga-batch8-capacity，基准9e15291a6d6601a72cdb3deff247b0e2066d26a6；代码及实际验证Commit：28677244c98057c7ae93460b7364af0c0c8341f0；本次文档提交登记。

一次实际Source-only容量Attempt：官方Test前8图/全部16 Caption，完整原SGAttacker三阶段/五尺度/10轮/2/255/步长0.5/255/global seed42/batch8，IRRA与MLM各恢复1次。攻击9.02253秒，会话19.15762秒；allocated6.3714GiB/reserved10.6270GiB、CPU Peak RSS2.2802GiB，无OOM。模型及buffers前后exact；官方FP32预算边界PASS，实数差2.93348e-8公开报告，未改投影。没有目标评分或Full。

容量入口仅限定原source_run的数据范围和独立输出；原Full及算法字节未改。新增Bash薄监督器和原生编码标量进度，记录PID/PGID/phase/日志、一次启动/失败即停，仍调用原Source和三个Target与CPU audit。32项必要CPU tests PASS，后台链仅用CPU Fixture验证。正式Full输出及控制目录未创建；没有部署nohup。

详见[实测及可授权启动命令](framework/SGA_BATCH8_CAPACITY.md)、[实际证据](framework/SGA_BATCH8_CAPACITY_EVIDENCE.json)。维持原Full20GiB空闲显存门禁，不轮询或结束其他进程；预计6–8小时，后续batch/外部任务仍有资源波动风险。Matched Clean单列，主方法为G-v2+H-v1 Joint；旧白盒结果不冒充官方SGA。完成后停止，等待Full授权。

## 已完成：官方SGA算法/IRRA Source最小移植与三目标Smoke，Full未执行

独立分支codex/tbps-sga-irra-source，基准04bddc407a59763118cdd3b1200231ec6c2d8e8e。代码及验证提交：2e1359034eb63a2db7cf9c21db095ea99f23dc11；当前文档提交登记该不可变代码版本。

按用户最新授权固定IRRA为唯一Source，直接调用官方f49e6926…的文本→图像→文本；原Loss、MLM、KL、增强、2/255、10轮、0.5/255、batch8/seed42保留。新增BERT→IRRA Byte-BPE接口，Source隔离原核心版本环境；目标环境与旧源码不改。ALBEF推荐仅是前次静态审查快照，不再是当前执行方案。

固定Test两图四文：源攻击PASS；初始IRRA目标在编码后被R@10至少10图门禁拒绝，原FAIL保留。CPU复用完整IRRA特征、独立两图评分桥后，IRRA/RaSa/APTM四组原生Score及独立CPU审计PASS，无攻击/IRRA重跑。两图R@5/R@10不适用。IRRA Joint R@1=50/mAP75，其余组100，仅链路测试。

实际官方FP32投影边界通过；实数L∞0.007843166589736938，比数学2/255多2.93348e-8，不声称严格实数上界通过，没有二次投影。SourceIRRA/MLM与RaSa/APTM前后冻结摘要exact；初次IRRA目标异常前未写出最终冻结摘要/实际前向计数，状态未完整记录。

18+9+119=146项适用CPU测试通过；2项旧工作树绝对路径锁定测试显式排除，不改旧证据。Full仅CPU3074/6156 preflight通过，无Full模型前向、排名或后台任务。Full前须确认浮点预算口径、Matched Clean比较方式及batch8容量；尚未授权。

详见[实施报告](framework/SGA_IRRA_SOURCE_SMOKE_REPORT.md)、[命令](framework/SGA_IRRA_SOURCE.md)、[来源与实际结果](framework/SGA_IRRA_SOURCE_SMOKE_EVIDENCE.json)、[ADR-0010](decisions/0010-sga-official-irra-source.md)。旧白盒/Joint/G/H/B3与原FAIL均保留。

## 历史额外实验：SGA-TBPS White-box三模型Full及CPU审计完成

冻结实现6f89b799922ac54f041dabd3cd8e9d444735b146，结果整理3a4ab6f40bbf66451e803d9b1d0651a4dd1f4daa。原[四方法表与CSV](framework/results/sga_three_model_full_20261018/REPORT.md)内容保持；其中SGA-TBPS White-box不能改名为SGA原算法复现，也不再作为官方Baseline主表依据。原Full与独立审计PASS、48值核验及历史保护是该额外实验的执行证据。

## 已完成：AP-Attack B3三模型Full及CPU审计完成，结果已只读汇总

报告与CSV内容Commit：0da6d30caf7a1224f9c734b3d607ea0843fb749d（后续仅补充发布索引）。

基准2047fb4c61f6ecfde32a148a593e87fe268462de；独立文档分支codex/tbps-b3-full-results。用户已手动执行共享GPU链，三个Full/独立CPU审计PASS，status.log最终SUCCESS | B3_THREE_MODEL_FULL_AND_CPU_AUDIT（2026-10-18 14:38:41 +08:00）。此前B3_STAGE_REPORT/B3_SHARED_RUN中的PENDING/尚未启动属于历史快照。

正式报告：[B3与G-v2三模型对比](framework/results/b3_three_model_full_20261018/REPORT.md)，同目录metrics_2dp.csv、full_precision.csv、evidence.json。方法顺序固定Clean → AP-Attack B3 → G-v2 Image-only，仅R@1/5/10/mAP，百分数两位小数，不新增指标或下降率。

mAP按上述方法顺序：IRRA 66.10/40.43/32.28，RaSa 69.18/39.99/35.05，APTM 64.13/42.39/39.66。G-v2四指标均低于B3；IRRA是同模型，RaSa/APTM是冻结IRRA训练攻击器产物的跨模型迁移。训练规模与监督来源不同，不能宣称等训练预算下优势。

本次仅标准库CPU读取与整理：9行36指标/24项历史控制数值exact、完整Manifest/ModelSpace、完成回执与五份Checkpoint SHA核验。82份实际读取文件SHA/inode/size/mtime不变；未启动GPU、重新生成、排名或重复独立CPU审计。Clean/G-v2为历史已审计Full显式复用，B3每模型实际新增3074图像编码，文本编码0。

旧源码、原共享运行树2047fb4、Checkpoint、generation_v3、B3像素及全部正式Full不变。等待用户审阅结果，不自行启动下一Baseline或新实验。

## 已完成：三受害者正式Full及独立CPU审计结果整理

2026-10-18，IRRA、RaSa、APTM的CUHK-PEDES Test正式Full及各自独立CPU审计均已完成。当前任务只读提取、来源验证、标量统计及文档发布，无新GPU/模型/攻击生成/排名/独立审计运行。

报告内容Commit：505942fcc1287ac07afc6400342628c6f9cb8ba0。

结果报告与原精度数据唯一索引：[三模型正式Full报告](framework/results/three_model_full_20261018/REPORT.md)。所有新导出固定模型IRRA→RaSa→APTM；方法固定Clean→H-v1 Text-only→G-v2 Image-only→G-v2 + H-v1 Joint，不改变原文件的历史保存顺序。

- IRRA：同训练来源模型评估，Full目录/home/lzf/ldx/outputs/tbps-framework-irra-materialized-evaluation/full/；GPU与CPU审计PASS，监督日志SUCCESS | IRRA_FULL_AND_CPU_AUDIT。
- RaSa：冻结IRRA训练攻击器的跨模型迁移，Full目录/home/lzf/ldx/outputs/tbps-framework-rasa-materialized-evaluation/full/；phase=SUCCESS、GPU/CPU审计PASS，监督器最终RASA_FULL_AND_CPU_AUDIT_SUCCESS。
- APTM：冻结IRRA训练攻击器的跨模型迁移，Full目录/home/lzf/ldx/outputs/tbps-framework-aptm-materialized-evaluation/full/；phase=SUCCESS、GPU/CPU审计PASS，监督日志SUCCESS | APTM_FULL_AND_CPU_AUDIT。
- 每模型Full为首次正式Attempt，参数/buffers摘要保持；G/H新加载/前向、Backward/Optimizer/JVP为0。独立CPU审计状态引用已完成真实日志，本轮没有重跑。
- 同一generation_v3物化输入3074图/6156文、G10/H7及Plan不变；三组Manifest、PID配对、有序成员、四类输入内容SHA和攻击来源exact。IRRA特征不跨受害者复用，RaSa/APTM保留原生融合评分。

当前结果：Joint五指标均低于各模型的单模态；相较G-v2 Image-only，Joint mAP额外下降IRRA 8.7291、RaSa 17.8443、APTM 9.1717个百分点。IRRA不能称作跨模型黑盒迁移。不同模型Clean是统一物化输入下的原生目标基线，不等同论文原始输入Clean；当前不推导跨数据集、多seed或因果性模块贡献。

## 本轮数据验证与发布边界

独立文档分支codex/tbps-three-model-full-results，基准525488a1a7de7f3258d71e195019c2bdc0a79e8f；工作树/home/lzf/ldx/projects/idea-test-tbps-three-model-results-20261018。

标准库CPU核验12行/60项指标、45项DR、三个完整有序Manifest及5份Checkpoint实际SHA。CSV和summary/result metrics一致；DR最大尾差1.4210854715202004e-14个百分点。2682个已读取来源文件SHA/size/mtime/inode保护核验通过；完整保护清单仅留服务器tmp，Git发布关键来源evidence.json和派生报告/CSV，不提交旧日志、模型或特征缓存。

导出metrics_4dp.csv、dr_4dp.csv、full_precision.csv和evidence.json；原实验文件、Checkpoint、模型/生成/评估源码均不修改。报告与导出排序/原精度/四位小数逐项检查通过。当前无阻塞；等待用户审阅结果，未授权新的训练/消融/模型或实验。

## 实现与历史索引

- 冻结Stage4D1 Plan及generation_v3 Image/Joint封存：见framework/STAGE4D1.md、framework/GENERATION_V3_GPU_SHARING.md及已完成生成/CPU审计回执。
- IRRA物化评估：b51d9e55190e96b6422f0c87952471ad6ac12bed，见framework/IRRA_MATERIALIZED_EVALUATION.md。
- RaSa实现及等待器：174eb51f4c346a6d74ace3f61f1ea170aeccf00b，见framework/RASA_MATERIALIZED_EVALUATION.md、framework/RASA_GPU_WAIT.md。
- APTM实现：2f21e910d9e560aa96ae29adcdb3d712c38dd4d6；自动链实现568eba1fd08407b7d6daabba8e47f1735c9cd98d、部署文档525488a1a7de7f3258d71e195019c2bdc0a79e8f，见framework/APTM_MATERIALIZED_EVALUATION.md、framework/APTM_GPU_WAIT.md。其中“尚未执行”保留原阶段快照属性，最新完成状态以本文及正式结果报告为准。
- Stage3B/4C/4D旧FAIL证据与历史源码/缓存永久保留，本轮没有改签或清理。
