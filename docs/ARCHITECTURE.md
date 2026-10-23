# 架构边界与迁移约定

## 双向Smoke只读复核与严格控制边界

冻结native_cli原audit-smoke在首NOOP exact控制失败后不发布审计回执；不能以完成四轮或存在梯度差替代前置控制。
独立operations只读复核使用新输出身份：验证已保存结构/内容/SHA，保留严格协议FAIL。
H原生state_sha与框架tree_sha的编码不同，必须调用各自锁定摘要函数对应Ledger；ConditionIdentity从JSON无损恢复tuple，不改变内容身份。
本固定Smoke四图库与双状态完整，但首非NOOP前已承接NOOP数值分化，A1/B因果归因未验证。
Reward/Advantage/非零梯度/参数变化分别统计，尤其零梯度Adam动量更新不等于新反馈学习。
见framework/results/gh_bidirectional_smoke_v2_review/REPORT.md；不改训练、Loss、Reward或原审计。


## 独立RNG门槛与Smoke单次后台监督

operations层把仅随机数CUDA验证的显存门槛设2048MiB，原Smoke10240MiB/RAM/磁盘/inode不变；不改生产producer/配置或probe回执。
现有restore/capture与生产阶段切换方法已真实GPU验证exact，独立Manifest绑定脚本SHA。
标准库监督器WATCHING不导入Torch/CUDA且不消费身份；连续两次资源通过（首次已合格可直接检查）后重做CPU/来源/RNG与即时资源门禁。
仅真正Popen前O_EXCL+fsync消费启动身份，持有flock，不自动重试；四轮结束自动执行原audit-smoke。
当前smoke_v2已实际后台运行，原生Claim/每Round/图库事务不变。见framework/results/gh_rng_smoke_launch/CUDA_RNG_AND_SMOKE_DEPLOYMENT.md。


## 独立CUDA RNG与Smoke执行管理

新增operations层调用冻结生产树，不改变producer或旧probe回执。RNG验证通过原restore/capture和阶段切换方法执行，单独Claim及脚本SHA；Smoke仍由原native_cli授权门禁执行。
单次flock/持久化启动标记、来源/唯一Attempt及原CPU门禁不放宽；完成后自动调用原CPU审计。
当前执行因资源不足在CUDA/Claim前停止，真实CUDA验证与Smoke仍PENDING。见framework/results/gh_rng_smoke_launch/REPORT.md。


## Probe专用资源派生与数值验证边界

resource_protocol(c,action)只对显式Probe覆盖6144MiB，Smoke继续使用原prototype的10240MiB。
verify_config仍核验完整producer/来源，派生配置重算source_key及ConditionIdentity；未消费的probe_v2首次绑定新身份。
native_cli仅改资源协议选择，numerical_probe与f970 AST一致，未改原算法。
唯一GPU探针已验证新语境JVP/FD和H→G第一步梯度，不执行Adam或完整图库刷新。
CUDA RNG恢复读回未被该冻结探针调用覆盖，完整双向反馈仍待单独授权验证。
见[真实结果](framework/results/gh_probe_shared/REPORT.md)；下面设备修复/CPU准备条目保留各自阶段边界。


## probe_v2设备参数与来源派生

原IRRA loader严格要求cpu或cuda:index；真实后端只修正from_source设备参数为cuda:0，不改校验与数学。
新独立配置重建完整producer/managed_roots及ConditionIdentity来源，保留原父状态/legacy源码。
probe_v2与smoke_v2隔离；新Smoke只认同源码probe_v2回执。10GiB门槛进入新配置，不依赖旧Watcher。
本轮只有CPU验证，原v1 FAIL与Watcher不动；新GPU执行仍待授权。见framework/results/gh_probe_device_fix/REPORT.md。

## 单次探针外部资源轮询器

operations/gh_probe_watch.py是独立标准库监督程序，运行部署副本，不从新树导入冻结后端。
固定调用原工作树native_cli，原producer/父来源/CPUStartupGate/唯一Attempt门禁不变。
10240MiB、180秒、连续两次，CPU预检后再查资源；资源回落继续等待。
flock+子进程继承锁，O_EXCL/fsync启动标记在Popen前持久化；崩溃/失败不自动重试。
轮询器不包含攻击数学，不补写未报告的CUDA RNG恢复PASS；部署和结果证据独立记录。

## G/H真实后端接线（CPU准备）

alternating/native_*与parents/token_bridge为原CPU原型提供受控真实后端；
只复用原G/H训练、FrozenLocalization、TextTransport与IRRA，不新增统一训练算法。
native_cli默认CPU；显式probe禁止Optimizer.step，smoke要求同源码/协议的数值探针不可变PASS回执。
CPUStartupGate必须在旧命名空间/CPU隔离完全退出后才允许CUDA；IRRA模型在legacy会话之外加载。
G迁移设备后重新绑定Adam对象并读回全部状态，双阶段RNG单独恢复。
B新语境重算z/w/e/JVP，A1及固定NOOP显式引用原条件；reader/basis/字节接口绑定独立条件空间。
G变化必须使整个2048 Train bank失效，完整刷新后创建原RewardBank新实例，再执行原H SCST。
动作、条件Tensor、逐图Claim/产物和完整round状态均使用既有ArtifactStore不可变提交；不自动重放。
精确字节Span桥对旧mapper的utf8_character_split作可证明坐标接线，保留原诊断，其他错误不放宽。
本轮只有CPU恢复/结构/控制流验证，真实JVP与双向更新尚未运行。见ADR-0016及
[执行协议](framework/results/gh_bidirectional_backend/EXECUTION_PROTOCOL.md)。


## G-v2固定Epoch评估变体

gv2_epoch_eval + ablations/gv2_sweep(_target)只连接原生成器/单步投影与原生缓存/Engine，
不包含训练或攻击算法。原G10 MaterializationPlan不放宽；新Checkpoint使用独立RGB variant，
显式引用旧Plan、原FP32 RGB、原H7文本和新G权重。先Claim、一次G、紧凑三Tensor及SHA读回，
未提交Claim/pending/failure拒绝自动重做，已完成单元仅在明确恢复授权下验证复用。
文本缓存只在同一Victim ModelSpace复用；新图像必须重新编码、ITM Top-K和完整集合归一化重新计算。
IRRA评分复用原通过审计的native cache协议作为原生空间证明，同时新Score绑定新Epoch图像/协议。
目标反馈不进入G，G与Victim分进程；H不运行。

当前六点CPU权重/来源与唯一G10/G60→三Victim真实Smoke及三独立CPU审计通过。
资源策略经用户显式授权改为shared_observe_only/6144MiB，利用率仅记录；所有非资源函数AST保持。
新Schema装载、原G10像素复现及新G60产物通路已验证，不推导Top128 Full已完成。
2026-10-20获最终授权后已用同一冻结入口后台启动五节点串行Full；缺Smoke/三审计仍拒绝。
nohup+setsid监督器不新增轮询器，不改变生成或评分。
2026-10-21五节点15组原生Full及15份独立CPU审计完成；本轮只读核对原产物/回执/SHA并派生CSV/六图，
没有变更Engine、Adapter、G/H或恢复协议。E10历史引用与新Epoch正式评分分开标记，Raw Clean控制不重复计为独立观测。
详见[运行说明](framework/GV2_EPOCH_SWEEP.md)、ADR-0013。

## VLPTAttack Full执行接线

新增dra_full/dra_full_target/run_dra_full，仅外层385批调度与存储。复用原Wrapper/官方调用、ArtifactStore、三模型原生评分与EvaluationEngine；保留完整RNG链，不支持自动重试/恢复。Source完成验证后才进入Targets，无目标反馈。详见framework/DRA_FULL_EXECUTION.md。v1因Source重复执行Target依赖预检在模型加载前FAIL并保留。v2仅通过明确stage拆分预检：Source严格检查自身依赖，Target保留RaSa/APTM原生依赖检查；公共SHA和数据检查都保留，官方算法不变。

本固定v2已完成385批Source、三Target Full及既有独立CPU审计；本次仅文档分支只读提取正式结果，不变更运行接口。详见[Full结果报告](framework/results/vlpt_official_irra_full_20261020/VLPTATTACK_OFFICIAL_IRRA_FULL_REPORT.md)。

## 当前结构

| 路径 | 职责及保护 |
| --- | --- |
| src/attributes、scripts、ontology | 旧属性/AP/TTA/IRRA/数据工具；仍有间接依赖，不删除 |
| prototypes/attribute_image_generator_20261014 | G-v1条件生成基线，全目录冻结 |
| prototypes/attribute_image_generator_v2_20261014 | G-v2 image-only、K+1训练、Epoch恢复、IRRA Full；全部冻结 |
| prototypes/attribute_text_policy_h_v1_20261015 | H/SCST/正式训练/Selection/Joint/消融；全部冻结 |
| 其他prototypes及_frozen_sources | reader/Canonical/JVP/选择及历史研究闭包；UNKNOWN也保留 |
| src/tbps | 新框架稳定职责模块，Stage 1契约/配置/存储/只读桥；Stage 2官方Dataset/IRRA Adapter；Stage 3A冻结Attack Plugin |
| configs/framework、tests/framework、docs/framework | 框架专属配置/CPU tests/使用与迁移说明 |

H CaptionFrontend仍依赖text_extractor→v55→v54→v53规则。G2正式训练不导入G1，但缓存来源/JVP准备有历史依赖。Joint复用G2 Full特征，消融复用Joint；不能按目录年代判废弃。

## 新框架契约

- DatasetManifest保留有序Gallery/Query、多Caption、多正例、原split/内部role、dataset命名空间PID。
- PublicTextInput不携带PID/正确Gallery/配对/奖励；EvaluationIdentity、TrainRewardContext与策略输入分开。
- ModelSpec声明图像尺寸/Tokenizer/表示维度（可无独立embedding）、原生cosine或fusion评分及能力。
- AttackSpec声明离线训练/测试优化/模态/信息访问；不要求所有方法实现train。
- 攻击产物、模型表示和Score分型；预算须有单位和像素/文本空间。
- Stage 1解析配置、核验来源和索引历史，不实例化模型/插件，不提供训练或GPU评测引擎。

## 历史兼容

旧文件/协议digest/路径均保留。新bridge_manifest只记录原路径、SHA、验证范围、来源和复用边界；不移动/复制历史资产、不加载Checkpoint张量、不重新排名。

IRRA特征不能跨RaSa/APTM复用；G2 Full未保存对抗像素，新victim可能需要经授权的新生成/编码。G2 Epoch恢复、H周期visit/update恢复及AP多阶段/状态机制不合并成同一恢复承诺。

未来新增src/tbps/datasets、models、attacks与engines。先契约/只读桥→官方Adapter→薄插件→缓存评测等价→训练基础设施→Baseline，不一次搬迁旧算法。详见ADR-0001/0002。

## 历史文档索引

README.md、docs/PROJECT_STATUS.md、docs/stages/*.md、prototypes/*/README*均为原阶段快照，不删除或重写。Windows E:\Git\idea-TBPS-test1\AGENTS.md及其PROJECT_STATUS是旧本地管理上下文；新Git版AGENTS/PROJECT_STATE取代其当前状态入口职责，不修改原文。

两份只读审计：E:\Git\idea-TBPS-test1-audit-20261016\TBPS_全仓审计与可插拔框架方案_20261016.md及TBPS_逐文件静态索引_20261016.md。Git基准40c7476；319 Python/43,754行。新文档只收录必要结论，不复制完整索引。

## Stage 1落地边界

contracts/config/artifacts/compat.legacy_assets均为独立模块；模块导入不导入Torch或历史prototype。只有显式CPU Tensor存储方法惰性导入Torch；没有模型工厂。配置解析成功与资产就绪、插件可执行是三个不同状态。

- 身份：DatasetManifest→公开输入投影/独立EvaluationIdentity；训练奖励单独绑定TrainRewardContext。
- 存储：只写tbps-framework*新命名空间，Manifest/完成回执/相关producer与来源SHA绑定；旧Store完全不改。
- 旧桥：Git固定catalog→只读文件SHA/JSON关系→新bridge_manifest；不加载Tensor、不重算排名。
- Full/消融保留原父/子协议digest；H Selection绑定已有Validation候选结果。没有把旧目录SHA门禁放宽成“任意源码可变”。

详细API、CPU命令、输出及Stage 2建议见[框架README](framework/README.md)。当前实现仅接受JSON配置和all_same_pid数据相关性；未来Adapter扩展其他规则时须明确声明。新存储锁基于Linux flock，Stage 1不承诺Windows写入后端。

## Stage 2 Adapter落地

src/tbps/datasets/cuhk_pedes读取原始有序annotation并输出DatasetManifest/公开输入；historical_membership验证G/H实际成员并提供受限训练奖励绑定。official val与attack_validation不混用；native PID保留原值，分类id-1重编号不进入检索身份。

src/tbps/models/base以原生score为公共契约，允许Fusion/local tokens及不同维度。IRRA专属source/io/view模块提供严格离线来源、官方预处理/Tokenizer、原生编码以及冻结Inference/输入可微Surrogate。模型不在import/dry-run中加载，真正加载要求显式授权及独立进程避免model包碰撞。

缓存按ModelSpace与InputBinding隔离：模型/官方来源/4个相关producer/运行协议，加上Dataset Manifest、有序ID与实际输入SHA。RepresentationRecord/ScoreRecord只存元数据，没有旧缓存重写。旧特征需未来显式桥接验证，不能按512维猜测兼容。

配置仍无Attack Plugin/统一Evaluation Engine可执行；Joint权限为G/H组件并集，图像梯度不错误套到H Unicode。当前未实现RaSa/APTM、ICFG/RSTP。接口、已验证/待验证边界与最小GPU授权方案见framework/STAGE2.md及ADR-0003。

历史根.gitignore的datasets/会忽略新源码；src/tbps/.gitignore仅局部开放Dataset Adapter目录，根规则及真实数据忽略边界不改。

## Stage 2.5真实等价性验收

独立smoke_irra入口默认CPU，仅显式授权时共用一个冻结IRRA Backend，对照官方独立函数路径及Adapter View。固定2图/2文本、总5图像/4文本样本前向、1次输入导数和1e-6阈值；单次attempt及Manifest/回执只写新tbps-framework-stage25命名空间，不提供训练/排名/攻击入口。

本轮数值逐值exact、冻结与非零输入梯度通过；仅覆盖固定Train样本和FP32/math/no_tf32设置。Adapter公共接口未改；旧ModelSpace/缓存不能据此自动重标或跨模型复用。命令与证据见framework/STAGE25.md、STAGE25_REPORT.md。

## Stage 3A攻击侧落地

src/tbps/attacks的base/registry/configuration负责描述、分阶段能力、公开输入/成员绑定和配置；provenance/legacy_runtime负责固定来源、只读条件索引与显式隔离加载；gv2/hv1直接调用原实现；joint只组合同一G/H；storage用新ArtifactStore保存像素/文本与完整动作。

插件不输出IRRA Embedding、不持有victim评分回调。H约束Tokenizer属于冻结训练侧策略，目标Tokenizer不改动动作。training_kind描述原方法而不授权训练；training_executable/offline_preparation_executable均False。原统一config仍拒绝执行，无Evaluation Engine。

新插件可在独立模块注册，无需修改G/H；当前冻结生成配置不会把迭代攻击或未实现的RaSa/APTM/Dataset误标可执行。真实G/H权重加载与等价仍未运行；CPU结果和Stage 3B建议见framework/STAGE3A_REPORT.md、ADR-0004。前述Stage 1/2“无Attack Plugin”描述保留当时阶段边界，当前状态以PROJECT_STATE为准。

## Stage 3B一次性验证与当前阻断

smoke_attacks提供单次attempt、权重/实际前向计数、完整H动作对照和Joint三模式计划。真实尝试在G加载后因像素读取器间接导入未锁定condition_schema失败，G/H输出均未生成；不改变Plugin等价状态或算法。原部分CPU预检漏查此路径，不能作为完整运行许可。

新增check_attack_runtime是无GPU选项的严格CPU闭包检查，覆盖实际G像素读取与H状态助手链；当前必须拒绝缺项。原执行脚本及来源清单保留，后续修复须单独审查/授权，不能覆盖失败attempt或自动重试。见framework/STAGE3B_REPORT.md。

## Stage 3B-R CPU导入闭包修复

新增runtime_closure，按真实加载工厂AST核对全部7个旧导入入口，解析11个符号并检查CaptionFrontend直接文件加载链。SourceFileLoader在执行前验证来源；整个CPU预检硬性阻断权重反序列化、神经构造/前向、CUDA初始化和导数。旧算法/工厂/Smoke脚本不改。

Stage 3A原配置及Stage 3B FAIL证据保持原样；新派生配置只增condition_schema单项，另有parent/derived SHA与精确增量证明。75项来源核验、33份实际源码执行检查通过，但真实输出等价仍PENDING。默认无GPU执行选项；任何GPU新尝试需要另行授权和新attempt。见framework/STAGE3BR.md及报告。

## Stage 3B-2严格门禁与固定条件等价验收

新增smoke_attacks_gated：先执行Stage 3B-R完整CPU闭包检查，捕获同一实际preflight结果，绑定门禁来源与报告，再交给原smoke_attacks.run_once。默认CPU，唯一授权输出equivalence_v2，已用目录拒绝；原数值比较/工厂/算法不改。

固定2条Train样本、G10/H7、FP32/math/no_tf32下G像素/delta及H完整动作exact，Joint三模式共享组件通过；实际G/H/IRRA前向4/2/0。对应固定环境状态VERIFIED，不改变通用execution_kind保守标签，也不授权全量或跨模型执行。原FAIL及CPU修复证据永久保留。见framework/STAGE3B2_REPORT.md。

## Stage 4A统一只读评测层

engines/{metrics,provenance,evaluation,legacy}消费预先准备的有序表示和来源，原生Adapter负责score，Engine只处理四模式、矩阵/身份与检索指标。没有攻击回调/模型工厂，不把PID、正确配对或目标分数回流给冻结G/H。

Feature/InputBinding→原生ScoreBatch→ScoreRecord→百分数指标/DR→新ArtifactStore原子回执。允许原生局部Token/Fusion、任意维度及显式高/低分方向；分块要求原生Query选择能力和Query独立性，不切分Gallery排名。tie保留官方默认argsort，FP32统一reduce。

旧Joint桥仅metadata索引，feature字节/回执重放未验收。历史Full缺少逐张对抗RGB SHA，不允许用feature SHA替代；后续需单独批准分型历史重放身份，IRRA特征不跨Victim。真实cached scorer/Full加载器未实现，原实验配置继续dry-run。见ADR-0005及framework/STAGE4A_REPORT.md。

## Stage 4B-1显式历史特征重放

historical_contracts提供独立ReplayIdentity/FeatureBinding/Representation，不继承或放宽严格materialized InputBinding/BoundRepresentation。历史对抗像素SHA=UNKNOWN；身份只对原IRRA/CUHK官方Test、固定G10/H7、原协议/Selection/完成回执/文件SHA成立，不能作为新RGB或新攻击产物。

historical_sources独立核对annotation有序成员、原来源与全部文件声明并显式映射ModelSpace；historical_loader只读Joint四family，每批detach().clone/原生H动作审计。无模型加载，权重只SHA。irra_cached实现原IRRA FP32归一化点积，evaluation显式历史分支复用同一排名/指标循环，正常materialized分支保持。

状态分METADATA_VERIFIED、FEATURES_VERIFIED、METRICS_REPLAY_VERIFIED。本轮真实四family仅到FEATURES_VERIFIED，真实分数/排名0；默认Scorer和核验CLI拒绝真实评分。Stage4B-2环境/资源/阈值冻结但未授权执行。跨Victim仍需独立Adapter/编码，不重标IRRA特征。见[接口](framework/STAGE4B1.md)、[报告](framework/STAGE4B1_REPORT.md)、ADR-0006。

## Stage4B-2受控CPU重放验收

replay_cpu默认无评分预检；固定旧FEATURES_VERIFIED回执和完整来源、冻结Torch环境/条件后，独立显式授权只可Claim一次replay_v1。replay_control监视当前进程RSS/耗时、恢复信号/线程并保留失败证据；配置execution_authorized=false原样保持。

replay_oracle在Engine原ranking调用处做数值仪器：独立原128块点积、独立官方rank及完整索引exact比较，原Stage4A排名/指标循环不变。四组原生评分/指标统一publish_result/load_result；历史Score读回补齐Tensor/shape/FP32/有限性检查。临时官方Score/Ranking逐模式释放，不另建Full算法。

唯一正式Attempt通过：METRICS_REPLAY_VERIFIED仅同IRRA历史特征重放，UNKNOWN像素身份/严格materialized边界保持。实际资源和完整误差见[报告](framework/STAGE4B2_REPORT.md)，执行许可已用完；不扩展成端到端攻击或跨模型等价。

## Stage4C-1模型无关物化层

src/tbps/materialization的contracts/store/pipeline/generation/legacy_text/audit_sources提供冻结计划、FP32 RGB/Unicode单元、前置生成Claim、独立Gallery/Query集合与只读H来源。复用原G/H插件和ArtifactStore，无新网络/Loss/投影/字符规则，没有真实生成CLI。

真实G2生成不读训练条件缓存；3074原图SHA READY。历史H7完整文本/动作可TEXT_SOURCE_VERIFIED复用，不读Embedding。G保存原/Adv/delta及文件/像素SHA，H保存UTF-8原字节和完整action；Joint只引用已有两类产物。计划无目标Model/Score，ModelSpace只在后续Representation/Score绑定引入。

中断前置Claim避免前向后未发布被静默重做；完整单元可验证后跳过，未完成Claim/回执须人工审查。来源摘要缓存以完整SHA及inode/size/mtime/ctime变更拒绝支撑，不放宽materialized身份。

真实RGB/端到端尚未验证，历史UNKNOWN像素身份不被升级。RaSa/APTM原生Adapter及GPU Runner留待单独授权，不能重标IRRA特征。见framework/STAGE4C1.md及报告、ADR-0007。

## Stage4C-2受控物化Smoke与当前FAIL边界

新增materialization/smoke.py与smoke_support.py，父进程先严格CPU闭包/来源校验，显式授权后创建唯一Attempt，G、历史文本、IRRA使用独立子进程。私有阶段需父回执/令牌/前阶段PASS；保存单元和生成证明，不弱化完整集合seal契约。Model Adapter只消费已保存RGB/Unicode，不向G/H传目标评分。

本次两图/两文真实物化验证通过；IRRA child提前初始化CUDA后进入CPU历史动作检查，原cpu_only拒绝，最终FAIL。保持原Runner来源和失败回执，不事后绕过门禁；后续需独立CPU-only调整初始化时序并再次申请小样本GPU许可。特征/Score等价仍PENDING，不能升级为全量或跨模型验证。

## Stage4C-2R：CPU启动隔离与原Plan只读引用

irra_startup在实际Runner中集中CPU准备，保留原Torch权重/神经/CUDA门禁，仅允许固定缓存Tensor；退出后验证函数、命名空间与legacy锁已恢复，才允许初始化CUDA/加载IRRA。真实child控制流程用Fixture/设备Mock验收，未运行真实模型。

retained_inputs用FrozenPlanReference显式只读原FAIL中完整提交的四单位；原protocol canonical digest、原工作树producer fingerprint、原FileRef/生成证明不改。复用原MaterializedStore.load验证逻辑，但没有publish/generate方法，不将旧单位挂到新工作树Plan。audit_retained仅CPU来源/小样本Tensor/动作与checkpoint_layout词法资源检查，没有GPU/Attempt入口。

资源估计仍不能证明现有IRRA eager加载满足2GiB；不修改Model Adapter、不降低门禁。下一步IRRA-only新验证需要先解决资源依据及独立授权。原特征/Score等价PENDING，原Stage4C-2整体FAIL保持。

## Stage4C-3：IRRA-only物化输入等价PASS

新增materialization/irra_only.py默认CPU预检，新独立Attempt只恢复官方IRRA；通过RetainedInputs原Plan只读引用与CPUStartupGate保证隔离退出后才初始化CUDA。stage_irra仅增加显式store_factory/result_store/on_cpu_ready注入，编码/归一化/比较循环复用，不重新实现攻击或IRRA编码。观察器核验实际收到的像素与已保存Tensor逐值exact；Tokenizer ID与历史before/after审计exact。

新representation有IRRA ModelSpace、旧物化单元/历史参考批次SHA和独立新Manifest，结果回执绑定四family特征Manifest。旧Plan、原四攻击单位和整体FAIL不重签。IRRA两图/两文8样本前向、参数冻结及1e-6等价已在固定环境验证；没有Full排名或跨模型证明。

用户本轮取消旧2GiB RSS硬限制；独立新资源策略监视系统可用RAM与RSS/耗时，不修改旧资源协议或官方加载器。唯一Attempt已完成并拒绝重入，后续全量/其他Victim仍需独立授权。

## Stage4D-1：流式全量物化与历史文本集合

materialization/full_prepare绑定同一个3074 Gallery/6156 Query完整Plan，默认CPU预检，显式历史文本物化没有GPU选项。full_runner只调度原插件与原生成事务，不复制G/H算法；G流式Runner须由未来单独授权的入口注入冻结工厂和资源策略。

bulk_session新增StreamingMaterializedStore成员索引及有限作用域来源缓存，不改原ArtifactStore/MaterializedStore。首、集合强制边界、尾完整SHA；直接使用来源每次stat、全来源stat最多1秒间隔，输出/未知文件仍完整SHA。原子fsync、Claim/pending、内容SHA和不可变回执保留。真实H仅复用旧Joint Unicode/action；全量语义TEXT_FULL_MATERIALIZED明确不等于Image/Joint完成。

full_audit独立只读逐条UTF8/完整动作/trace/官方顺序与固定历史摘要，并检查历史保护清单；无模型、Embedding或排名。后续不同Victim只消费同一物化RGB/Unicode，不能反馈目标Tokenizer/Score改变已冻结产物。固定Plan绑定本轮producer路径，跨工作树不得直接重新计算Plan并重签已有H单位。接口/命令见framework/STAGE4D1.md。

## Stage4D-2：冻结库执行桥与资源阻断

新增full_gpu/gpu_safety/full_gpu_audit，生产绝对脚本的PYTHONPATH仅指向Stage4D1冻结库；运行时核验实际模块路径与原Git HEAD/clean，保持原MaterializationPlan，新增入口SHA另行绑定。旧图文Store只允许追加，既有H文本单位与回执不改签。

父CPU预检→单次G GPU子进程→GPU退出后CPU文本再核验/Joint引用封存→独立CPU验收。实际G Hook/逐图日志/Claim/Proof相互绑定；只允许一次G权重反序列化，禁止其他模型与优化。资源使用系统实际可用RAM、磁盘和GPU冲突检查及7200秒时限，无进程RSS硬上限。

本轮预检被外部GPU进程和空闲显存不足阻断；正式Attempt未创建、G全量/Joint封存未执行。CPU Fixture通过不能升级为真实执行验证。冻结库桥已实现，但新GPU执行路径仍PENDING，详见framework/STAGE4D2_REPORT.md。

## Stage4D-2正式Attempt实测FAIL

资源就绪后唯一generation_v1在CUDA初始化前失败。新入口先安装no_optimization/g_load_only，改变CPUStartupGate快照中的autograd/torch.load函数，随后initialize_cuda严格检查拒绝。G恢复/前向0，没有图像单元或Joint；原门禁、算法、Plan及源码不改。

建议后续CPU-only调整为“CPU退出→gate初始化CUDA→GPU安全包装→模型加载/前向”，保留资源guard并补实际child组合时序测试。本轮未授权自动修复/重试，失败源码/配置必须冻结。当前611项CPU测试不覆盖该组合，详见framework/STAGE4D2_EXECUTION_REPORT.md。

## Stage4D-2R：CPU启动顺序修复与派生Attempt

独立分支仅调整full_gpu.child的初始化/安全包装嵌套：CPUStartupGate快照和锁恢复后先initialize_cuda，再立即安装no_optimization与g_load_only，两项成功才允许G工厂。资源guard仍覆盖全程；原Gate/监督器/冻结Stage4D1/算法不改。

新树入口固定generation_v2；新派生配置仅改output及entry_sources，原v1/config/Plan/文本保持。实际child组合CPU Fixture通过，真实GPU未运行。原FAIL报告完整收录新分支但旧工作树脏文档不改；CPU通过不升级正式Image/Joint状态。见framework/STAGE4D2R_REPORT.md。

## generation_v2 FAIL后的受限预处理与首图Claim恢复

gpu_safety将图像读取限定为原_LegacyG/pixels_from_pil及显式绑定输入；只有该作用域内原Resize的PIL RGB/bilinear384×128前向可通过。ToTensor仍由原函数执行，无通用非G网络白名单。G Hook、单次恢复、推理冻结和优化阻断保持。

claim_recovery是独立零前向证据契约，不修改Stage4D1 generation/full_runner：先验证原v2 FAIL和唯一首图旧Claim，在原generate_image_once的单线程局部_prepare中允许一次恢复，原_finish发布正常Proof；独立Start/Proof记录依据。同一G工厂保持打开，其余图仍用冻结stream_images。新公开计数包含恢复首图，另记录原Runner内部跳过该刚完成首图1次。

新入口generation_v3绑定独立源码/证据/预处理来源SHA，原Plan及6156文本不重签。当前仅CPU代码与恢复资格，未生成真实图像。见framework/STAGE4D2R2_REPORT.md。

## generation_v3固定身份GPU共享（CPU实现）

当前Runner的gpu_safety独立提供共享资源检查，不修改冻结Stage4D1通用GPU门禁或生成算法。仅配置明确绑定的Ollama PID2436556可与本GPU子进程共存；proc启动/boot/命令行/服务父身份、可读二进制SHA和GPU报告路径联合验证。启动3584MiB/≤10%，运行1024MiB安全余量；子进程FullGuard每秒采样，异常转为安全停止。父进程不把自己创建的GPU子进程误当外部任务。

配置和入口SHA属于独立执行层，原Plan、文本、Claim恢复及旧FAIL不改签。GPU共享运行尚未验证；具体命令及限制见framework/GENERATION_V3_GPU_SHARING.md。

## 封存物化输入的官方IRRA评估接线

engines/materialized_irra.py是独立绝对脚本，生产PYTHONPATH指向冻结Stage4D1，保持原Plan路径和Store来源。CPU阶段校验generation_v3封存/独立审计、完整官方Manifest及历史Unicode，退出隔离后才允许IRRA恢复。原FP32 RGB直接执行IRRA normalize_pixels，文本送官方Tokenizer，不生成攻击。

materialized_cache提供每64行的ArtifactStore紧凑缓存与无模型CachedIRRA评分薄层；四family独立有序，ModelSpace/输入内容/攻击来源分别绑定。四组复用原EvaluationEngine/Ranking/publish_result/load_result，不增加通用评测框架。正式Full的Clean采用原历史0.001pp一致性门槛；新指标来自实际Score。

本轮CPU测试与真实来源预检通过，真实GPU评估PENDING。原生成结果/源码不修改；后续其他Victim只能另建ModelSpace，不能复用IRRA特征。详见framework/IRRA_MATERIALIZED_EVALUATION.md。

## IRRA物化评估的共享GPU修订与Smoke

仅当前IRRA执行入口采用shared_observe_only并记录GPU状态，不要求GPU独占或固定空闲显存；实际CUDA OOM按正常异常保存完整堆栈与计数，不自动重试。冻结Stage4D生成层的资源政策不变，也不引入新的进程身份绑定。FP32、batch1、原IRRA及Engine/Ranking均保持。

唯一真实10+10 Smoke通过：IRRA加载1次、图像/文本样本前向20/20，G/H与优化操作0，模型摘要exact。当前新入口Full仍PENDING，未执行额外真实测试。证据与资源数据见framework/IRRA_MATERIALIZED_SHARED_GPU_SMOKE.md。

## RaSa原生融合接入与可复用边界

models/rasa.py持有官方加载、原生Tokenizer/预处理、全局256维候选与局部Token ITM评分，不包含攻击器或PID。ModelSpec表示fusion、representation_dim=None。Engine增加materialized_native显式执行类别，要求已验证Adapter/Checkpoint/官方身份；仍无攻击生成或模型加载工厂。

materialized_inputs只读封存集合并复用现有读取/身份接线；rasa_eval生产入口固定两个冻结attacks/materialization包路径，保持Stage4D原Plan，其余Adapter/Engine来自新树。NativeFeatureCache存Tensor字典，ChunkCachedScorer按Query块保存原生Score；不固定特征维度/目标模型或攻击ID。AttackOrigin描述显式组件和单模态预算，Engine按模态检查，不把未知ID当成已实现攻击插件。

RaSa官方384×384 BICUBIC、BERT50、Top-128最终ITM保持。现有FP32工作空间已是384×128，因此采用Tensor resize避免量化，不能宣称与从原始文件直接PIL处理相同。Full报告须标注固定物化Clean协议。历史10+10 Smoke仅覆盖10候选；当前RaSa与APTM的正式Full及CPU审计均已完成。新增模型仍需独立Adapter/来源与薄入口，复用输入桥、缓存和Engine；新增攻击样本不改RaSa。见framework/RASA_MATERIALIZED_EVALUATION.md。

## 轻量RaSa GPU等待执行层

wait_rasa_full.sh为独立Bash监督器，不修改任何Adapter/Engine/模型或配置。60秒GPU快照、连续3次资源门槛、flock与一次性triggered避免重复启动；满足后只调用已有Full及CPU审计，失败不重试。由用户明确授权nohup独立session部署，Codex不保持会话或后台监控。运行日志与正式结果隔离，见framework/RASA_GPU_WAIT.md。

## APTM原生评分接入（正式Full与CPU审计已验证）

models/aptm只负责官方来源、Swin/BERT编码、CUHK预处理和原始ITM/global；不依赖G/H、PID、目标奖励。engines/materialized_aptm复用物化encode/bound_inputs、NativeFeatureCache、EvaluationEngine和结果存储，生产入口aptm_eval只把attacks/materialization包指向冻结Stage4D1以保留Plan。

APTM原始Score归一化依赖完整Query集合，因此Adapter明确query_independent_scores=False。APTMSetScorer先按128 Query缓存原始ITM/global（来源绑定完整Query/Gallery），再完整集合归一化，之后才提供固定上下文的Score切片给原Engine。最终缓存绑定所有原始Score Manifest，不逐块minmax或复用RaSa分数；单模式额外最终矩阵缓存避免每块重复读全矩阵。

不同受害者ModelSpace/Token/Score独立，仍消费相同FP32 RGB与Unicode。当前输入桥绑定generation_v3；后续攻击方法替换物化输入桥，不修改APTM原生评分。原CPU阶段不代表真实GPU；当前APTM正式Full与独立CPU审计已完成，来源及结果见framework/results/three_model_full_20261018/REPORT.md，仅覆盖固定模型/输入/协议，不推广到其他组合。

## APTM自动执行控制层

wait_aptm_full_chain.sh只编排原APTM CLI：RaSa完成证据→GPU连续3次空闲→Smoke→CPU audit→独立重新等待3次→Full→CPU audit。与Model Adapter/Engine/配置完全分离，不引入训练、攻击生成或其他模型。RaSa旧成功标志/日志回执通过显式只读确认桥兼容，确认文件只写APTM控制目录。

flock、输出/进程检查和不可重入trigger限制单链；各阶段退出码和精确PASS共同决定推进。控制日志不等于正式评估结果；不修改旧Rasa监督器，不重算旧指标。用户明确授权后台链，Codex部署确认后退出。见framework/APTM_GPU_WAIT.md。

## B3独立Baseline桥

src/tbps/baselines只读接入B3历史像素，复用MaterializedStore原图与严格image_checks、ArtifactStore、三个原生Adapter和EvaluationEngine。旧MaterializationPlan固定gv2/hv1，不为B3放宽旧类型；B3Pixels为独立目标无关FP32物化薄桥。Full引用严格同空间/同顺序的既有Clean文本表示和Clean/G-v2已审计指标，新增B3图像表示及原生评分。攻击器不接收目标反馈，不复制检索算法。对外结果投影为四项指标，固定Clean/B3/G-v2顺序。launch在严格CPU启动快照前导入torchvision，GPU资源拒绝不会创建Attempt；当前真实GPU验证PENDING。见framework/AP_B3_COMPARISON.md。

## B3共享后台链修订

comparison.py新增一次性shared_observe_only启动快照：记录利用率/外部进程而不据此阻塞，保留3072MiB内存底限与B3重复执行检查。脚本将快照传给各子进程，不增加GPU轮询。run_b3_full.sh仅编排原CLI：三个Smoke/审计全部PASS后再Full/审计；不可重入标记和独立会话控制一次启动及安全停止。算法/Adapter/Ranking不改，新来源配置和输出与旧097证据隔离；详见framework/B3_SHARED_RUN.md。

## SGA目标专属白盒薄适配

baselines/sga_attack调用固定官方ImageAttacker的五尺度增强，持有集合目标、迭代和统一像素预算；models/sga_surrogate只提供原生可微图像路径/ITM融合目标。sga_whitebox负责来源CPU门禁、逐图Claim/FP32物化、实际Hook/梯度计数及原生评估编排，不重新实现Ranking。SGAPixels独立绑定victim Checkpoint和像素来源；旧固定G/H MaterializationPlan不放宽、不改签。

复用原Dataset输入桥、B3的只读FullReference/模型加载工具、三个Model Adapter、ArtifactStore和原生缓存/Score/EvaluationEngine。只调用公共工具，不使用B3攻击像素或Generator。各目标单独白盒生成，Clean Query保持原文；Full控制指标与Clean文本仅引用同空间已审计历史缓存。新输出tbps-framework-sga-whitebox-v2与原失败/所有正式Full隔离。

三模型10+10 GPU Smoke/CPU审计通过；随后三模型正式3074/6156 Full及独立CPU审计均完成。正式来源与四方法表见framework/results/sga_three_model_full_20261018/REPORT.md。原SGA文本攻击、原环境和APTM最终排名全链求导不在当前验证范围。原生目标与适配差异、资源及一次性手动Full链见framework/SGA_SMOKE_REPORT.md、ADR-0008。CPU通过不等于SGA原论文全流程复现。

## 官方SGA审查与旧白盒适配隔离（ADR-0009）

旧sga_attack/SGASurrogate/Full保持冻结，归类额外白盒适配。官方SGA后续只允许薄I/O入口调用固定官方SGAttacker本体，保留两次img_guided_attack和原ImageAttacker Loss/更新；不复用旧自写攻击循环或ITM目标。官方源模型与三个target职责隔离，source产物生成一次，target仅执行原生预处理/评分。

现有ArtifactStore/Model Adapter/Engine可复用，旧G/H MaterializationPlan不改签。官方source canvas及pre_caption不同于历史物化空间，必须另行冻结输入桥/Matched Clean后才能与主方法Joint比较。当前仅静态审查，无官方攻击新代码或执行状态升级；见framework/SGA_OFFICIAL_REPRODUCTION_REVIEW.md。

## 官方SGA / IRRA Source最小移植（ADR-0010）

source wrapper只转换原生IRRA输入/特征；官方SGAttacker/ImageAttacker/TextAttacker保持只读并实际调用。Source与Target分别子进程，冻结产物隔离目标反馈。Matched Clean绑定pre_caption30及384×128源RGB，不复用旧Raw Clean数值。BERT定制Tokenizer→文本→IRRA Byte-BPE逐行保真，额外截断拒绝。

Full仍复用原EvaluationEngine/原生融合Scorer。固定两图Smoke另用限定评分桥和官方rank(max_rank2)，不放宽Full R@10门禁；保留原IRRA排名FAIL、CPU只读复用其完整特征。算法/预算/预处理差异及资源见framework/SGA_IRRA_SOURCE_SMOKE_REPORT.md。官方FP32边界舍入超实数epsilon公开记录，未引入二次投影。

## SGA官方Batch8容量与Full执行管理（不改攻击）

sga_batch8_capacity只在独立作用域限定Dataset为官方前8图/全部Caption，委托冻结source_run；新输出tbps-framework-sga-irra-capacity不污染Full。原算法/Full源码和配置不改。run_sga_irra_full.sh及sga_full_execution只增加一次性监督、PID/PGID/phase与编码标量进度，委托原Source/Target/Audit，不改原生评分。容量实测PASS，后台全链仅CPU Fixture验证；Full未启动。见framework/SGA_BATCH8_CAPACITY.md。

## 官方SGA / IRRA Source Full实际验收完成

此前“Full未启动”均为阶段快照。正式Full现已完成：唯一Source调用官方SGA原三阶段，3074/6156冻结产物给三种原生Target，不复用跨模型Embedding或目标反馈；所有阶段与独立CPU审计PASS。最终结果及SHA见results/sga_official_irra_full_20261019/SGA_OFFICIAL_IRRA_FULL_REPORT.md。本轮只追加派生结果与索引，没有改架构、攻击或评估算法。Matched Clean与主方法Raw Clean及其他预算/权限差异继续分协议报告。

## VLPTAttack（DRA）IRRA Source薄接线

dra_official仅来源/数据/投影验收/被动追踪；dra_irra_source继承已验证IRRA I/O，仅管理参数梯度；dra_irra只调原RAttacker.Attacker.attack并存储，dra_target复用原生模型/Cache/Engine校验和官方rank。没有共用攻击Loss或自写DRA更新。
源代码冻结后修Target协议与CPU import均用派生工作树/配置，避免改变已有Source/Target回执依赖。不同阶段不是新算法。
官方number_perturbation按原词槽位计数，多WordPiece替换可能形成短语，见ADR-0011；不能加空格词数过滤。
固定8/16三Target和CPU审计已PASS，不开放Full，不把CPU/Smoke等价推广到原论文模型或全量结果。


## G-v2独立扩展训练边界（2026-10-20）

src/tbps/ablations/gv2_epoch_state.py管理只读父契约、完整G/Adam/RNG恢复、日志SHA和新事务Checkpoint；
gv2_epochs.py是外部调度/观测，直接调用冻结G-v2 train_epoch/train_single_sample/validate_epoch，
不把旧G改造成新攻击插件，不修改原训练源码或缓存。有效配置唯一变化为训练上限10→60，绝对Epoch编号不变。
原contract和新extension-contract分型；单样本Smoke用独立probe schema，不能冒充完整Epoch或作为正式恢复入口。
生产边界拒绝pending/损坏/未完成Claim自动重播；不提供Step级恢复。正式执行仍需独立授权。
详细决定见ADR-0012，当前完成状态以PROJECT_STATE为准。

## G-v2正式续训的外层执行隔离

冻结635471b工作树只运行原gv2_epochs入口；独立管理工作树的run_gv2_epoch60_train.sh只监督一次启动、
原生CPU训练审计及成功后的标量报告/发布，不进入训练计算图或修改配置、来源和算法。
正常TERM只转发训练子进程、等待原Epoch边界；失败不重试/恢复。
gv2_epoch60_results.py只读已完成回执和标量，生成原精度CSV/曲线及六点Checkpoint来源，不加载模型、不评分。
图形绘制只读复用现有matplotlib CPU环境，训练环境不安装/更新依赖。当前完成状态见PROJECT_STATE。

## G60与历史H候选的内部Validation只读桥

g60_h_bridge为128内部Validation建立独立来源，不改历史ModelSpace或H7 Selection。
g60_h_validation仅复用原G推理和IRRAAdapter/官方rank；新图像产物与旧文本特征通过完整身份顺序/内容SHA绑定。
前2张Smoke保留在128预算内，CPU读回及冻结核验通过后继续；最终独立CPU审计。
此入口不是统一训练引擎，也不提供Test、跨模型或自动候选选优执行。

本固定128图执行已通过：同一次GPU运行内先2图真实Smoke及CPU读回门禁、再126图，
不因各H候选增加G/IRRA前向。最终独立CPU进程复核11组128×128原生评分/官方指标；
保留旧Selection和G10缓存，新增G60 Gallery有独立内容SHA及128生成证明。
结果仅为重复内部Validation的探索性诊断；不新增Test自动选模能力。

## H10父状态的G60奖励适配Extension

hv1_g60_state只读核验父状态并建立新契约，hv1_g60_adaptation为原H SCST/奖励/事务Checkpoint的调度层。
不调用旧正式execute/rollback写历史目录；新链以H10的cursor20480/updates20370作为已验证父边界。
独立G60 Train图库只生成一次，训练/Validation TextRuntime不持有G且禁止图像前向。
新节点Validation复用原128 G60 Gallery，原Clean/文本及候选只读引用；不新增统一训练引擎。
协议差异与恢复边界见ADR-0014。

## H60审计器版本隔离

原113 producer运行树保持冻结。修复审计器只改局部变量；
独立入口先验证旧完整来源、精确修复差异及新入口来源，再使用旧包加载上下文执行新审计。
audit_repair_v1额外绑定新审计器与旧产物/旧FAIL，不能改签旧执行协议。
训练继续调用原入口；新监督器仅消费版本化审计并把新报告输出到修复树。
详见framework/HV1_G60_AUDIT_REPAIR.md。

## H60只读事后分析边界

hv1_g60_posthoc及diagnostics/plot/report/validate脚本只解析既存日志、回执和128×128评分，
不导入旧模型或训练入口；新分析目录与失败发布目录隔离。
官方rank函数以AST提取作独立数值对照；正式Full只读摘要，绝不加载其完整Score重新排名。
SCST优化R_sim+R_rank，synergy仅诊断；G→H为固定特征奖励条件引导，并非G/H端到端联合训练。
研究建议与实际实现分开登记，不自动改变奖励、Selection或模型。


## G/H属性审计与交替研究边界

gh_attribute_audit/supplement只消费旧Canonical/动作/Score派生数据；规则重提取无神经前向。
G S_raw连续敏感度与H策略importance分开，跨模态按Caption+Span+语义匹配，保留歧义。
原文锚点Schema与A/B/C交替仅为研究草案，未修改现有contracts、策略、Reward或缓存门禁。
未来若使用H攻击文本指导G，应显式新建语境条件身份，不将原z/JVP改签；
最新G的bank需新绑定RewardBank上下文，不能靠切换时间顺序宣称双向交互。
见[审计与设计](framework/results/gh_attribute_alignment_audit/REPORT.md)。

## G/H双向条件交替CPU原型

src/tbps/alternating集中承载本次两轮研究原型；不是通用Training Engine。
原Canonical+完整H动作→新语境定位/JVP→原G单样本训练；
G更新→完整2048 Adv bank→原H RewardBank/SCST。
A1不消费新文本条件，B消费，二者均刷新G→H反馈。Loss/Reward/K函数和独立推理接口保持。
新条件身份包含原Caption锚点与当前语境，旧condition/JVP不重标。
双Adam与G/H阶段RNG分别存储，原子round、不可变Claim与完整边界恢复；未完成事务拒绝自动重播。
当前仅CPU Fixture执行，真实后端组装与GPU数值验收PENDING；CLI只有只读预检，无训练动作。
协议/接口/缓存失效及预算见framework/results/gh_bidirectional_prototype/和ADR-0015。
