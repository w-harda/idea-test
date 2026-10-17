# 架构边界与迁移约定

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
