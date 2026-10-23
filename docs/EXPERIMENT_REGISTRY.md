# 历史实验注册表

## G/H probe_v2设备字符串修复（CPU-only，2026-10-23）

基准7ba66d95；独立codex/gh-probe-device-fix。
一行cuda→cuda:0；276 CPU测试及真实来源/父状态Preflight PASS，140 producer/496 legacy来源。
独立probe_v2/smoke_v2身份，10GiB门槛；旧FAIL/28文件与151378历史metadata保持。
尚未执行任何新GPU探针、Watcher、Smoke或训练，数值仍PENDING。
[修复与完整证据](framework/results/gh_probe_device_fix/REPORT.md)。

## G/H probe_v1：真实启动在IRRA加载前FAIL（2026-10-23）

02:41:45自动调用，02:41:55退出码1；已消费Attempt，不得重用。
设备字符串cuda与加载器cuda:index约束不一致；IRRA/reader权重未恢复、神经前向/JVP/更新0。
新数值与GPU峰值未产生；原FAIL/Claim/控制日志不改写。
[只读核验报告](framework/results/gh_probe_resource_watcher/PROBE_V1_FAILURE_READONLY.md)。

## G/H数值探针：独立资源轮询已部署（2026-10-23快照）

39项CPU测试与冻结后端来源Preflight PASS；已部署PID1349747，PPID1/独立会话/flock通过。
部署时WATCHING/空闲6501MiB，probe_v1不存在，GPU探针结果仍PENDING；后续只读status.json确认。
只允许原probe_v1；10GiB连续两次/180秒，独立持久化单次启动保护，不启动其他实验。
[协议、CPU及部署证据](framework/results/gh_probe_resource_watcher/README.md)。

## G/H数值探针启动前资源阻塞（2026-10-22）

基准a46c6980；CPU Preflight及151378历史metadata保护PASS。
空闲显存6385MiB低于本次10240MiB硬门槛，RESOURCE_BUSY；probe_v1未创建，未初始化CUDA。
没有GPU模型加载、神经前向、JVP或更新；不属于GPU探针FAIL Attempt，数值仍PENDING。
[报告与来源](framework/results/gh_bidirectional_backend/PROBE_V1_RESOURCE_BUSY.md)。


## G/H双向真实后端CPU准备（2026-10-22，无GPU实验）

开发父8ae623d3；codex/gh-bidirectional-backend；实现Commit为`656ede2dc25c87ab21c559ad2b49aedb90b92de0`。
224项CPU测试+3项真实CLI拒绝检查PASS；真实CPU完整恢复G60/H10权重、Adam及CPU RNG。
G60 SHA9148a6ac…46049；H10 SHA4ae964d5…bacd；IRRA SHA9ff62095…5009。
原2048 Clean/G60特征只读消费并核验审计内容/顺序；151378历史及10918相关来源metadata保持。
CPU启动门禁事件完整，CUDA RNG未恢复；真实G/H/IRRA/reader前向、JVP、更新为0。
新输出probe_v1/smoke_v1未创建，不存在真实数值探针或双向Smoke PASS声明。
一次资源快照不构成执行授权；未来先固定两条Train数值探针，再另行审查两轮A1/B。
[报告](framework/results/gh_bidirectional_backend/REPORT.md)、
[SHA与来源](framework/results/gh_bidirectional_backend/SOURCE_EVIDENCE.json)、
[执行方案](framework/results/gh_bidirectional_backend/EXECUTION_PROTOCOL.md)。



## G-v2六节点三模型Full消融（2026-10-21完成，结果整理）

执行6021f3b，启动5b18715；正式Full22小时32分15秒，2026-10-21 19:07:27最终成功。
E10历史6条引用+五新Epoch30条真实Full，共36条；独立Clean/H-only控制6条。
15份新独立CPU审计及3份E10审计来源PASS；3074/6156与官方顺序/PID/Caption exact。
各新Epoch一次3074图/一次G恢复，三Target共享固定像素；原生文本特征复用，H0、Backward/Optimizer/JVP0。
本轮不重新运行audit，仅复核SHA/回执/来源和168指标；34721个artifact，原文件无改动。
六节点mAP均下降、最低为E60：Image-only17.51/26.47/32.11；Joint13.08/12.57/23.74（IRRA/RaSa/APTM）。
属于预设Test消融描述，不是独立Validation选模，不追加训练或其他实验。
[正式报告与原精度CSV/六图](framework/results/gv2_epoch_sweep_full_analysis/REPORT.md)、
[证据](framework/results/gv2_epoch_sweep_full_analysis/SOURCE_EVIDENCE.json)。
下面RUNNING条目保留启动时历史快照属性。


## G-v2六节点三模型Full（2026-10-20正式启动，RUNNING）

运行代码6021f3bfdec181785d1baccd7ce4f80a1428b069，算法/配置/输入不变。
20:35:12后台启动，监督器PID/PGID/SID1545336、PPID1/无TTY；CPU_PREFLIGHT PASS。
E10 IRRA旧结果已只读引用，快照E10_REUSE_rasa；新Epoch生成/完整评分/最终审计PENDING。
E10不重算，E20/30/40/50/60各固定Gallery共享三Victim，H7/Raw文本和原生空间不变。
Full输出tbps-framework-gv2-epoch-sweep/full；控制gv2_epoch_sweep_full_v1。
无轮询、无自动重试，不发布Smoke充当Full指标。
[启动报告](framework/GV2_EPOCH_SWEEP_FULL_STARTUP.md)、[完整快照](framework/results/gv2_epoch_sweep_full_startup/STARTUP_EVIDENCE.json)。

## G-v2 Epoch Sweep唯一共享GPU Smoke（2026-10-20，PASS）

代码152498fd258572ad9c43c04b2ce8fba8b9e995b4；用户显式许可共享，资源函数之外AST exact。
smoke_v1只创建一次，G10/G60各10图，同一G60 Gallery供三个目标；H文本及原生特征只读复用。
三Target及三新CPU审计全PASS，标志GV2_SIX_NODE_SMOKE_PASS。
G10原/Adv/effective_delta内容SHA exact；G60 L∞0.0313725471496582，全部模型冻结。
G共2恢复/20前向，三Victim各1恢复/10图/0文，RaSa/APTM各200 ITM pair，H/Backward/Optimizer/JVP0。
无正式五节点Full、无训练或重试；仅固定10+10/Top10通路，不作为训练轮数效果结果。
[真实报告](framework/GV2_EPOCH_SWEEP_SHARED_SMOKE.md)、
[回执/SHA/资源/原精度](framework/results/gv2_epoch_sweep_shared_smoke/SMOKE_EVIDENCE.json)。
旧准备PENDING/RESOURCE_BUSY为历史阶段快照，未改写；本轮完成后等待最终Full授权。


## G-v2六节点三模型消融准备（2026-10-20）

状态CPU_PREPARED_GPU_SMOKE_PENDING；不是新的Full实验结果。
六节点来源固定10/20/30/40/50/60，原训练完成报告6187eed96ed9eff63b8da37f5d98b5adddf75300。
H7/visit14336与原H文本不变，三目标仍为IRRA/RaSa/APTM，各自原生模型空间。
原三Full的E10 Image-only/Joint及Raw Clean/H Text-only已只读核验48项数值与来源；不重算排名。
新Epoch像素、各Target编码及评分仍未执行；GPU利用率76%时停止CPU入口，实际GPU调用0。
227项适用CPU通过，旧APTM绝对路径用例基准复现失败并排除1项；相关历史stat/SHA不变。
[准备报告](framework/GV2_EPOCH_SWEEP_PREPARATION_REPORT.md)、
[六点SHA/来源/资源/测试证据](framework/results/gv2_epoch_sweep_preparation/evidence.json)、
[E10原精度引用](framework/results/gv2_epoch_sweep_preparation/epoch10_reused_raw.csv)。
不得把此CPU状态提升成真实Smoke或五节点Full完成。


## VLPTAttack官方DRA／IRRA Source v2 Full（完成，独立CPU审计PASS）

执行34097c3，启动c97c2bf；2026-10-19 16:38:15至10-20 09:51:44 +08:00，
62009秒；Source及三个Target Full/既有独立审计全部PASS。
根目录/home/lzf/ldx/outputs/tbps-framework-vlpt-official-irra-v2/full/，
控制/home/lzf/ldx/tmp/vlpt_official_irra_full_v2/。
Source completed SHA a5978dc834b14a34d4c6932e0c2d463fb052105a551c5bd9137d647d3639df18，
385批/3074图/6156文；6153编辑、3 NOOP，三Target共享冻结输入，权重buffers不变。
IRRA/RaSa/APTM Joint mAP=0.10/13.03/19.85；只引用v2 Full，不混入Smoke/v1/旧白盒。
[正式报告](framework/results/vlpt_official_irra_full_20261020/VLPTATTACK_OFFICIAL_IRRA_FULL_REPORT.md)、
[来源索引](framework/results/vlpt_official_irra_full_20261020/VLPTATTACK_OFFICIAL_IRRA_FULL_EVIDENCE.json)。
正式报告/CSV/证据内容Commit：467b54193f4e1110006df0f9a4202dc26c7bf534，分支codex/tbps-vlpt-full-results。本次只读提取不重跑审计或完整排名；下方RUNNING/PENDING均为原启动快照，已完成状态以本条为准。


## VLPTAttack官方IRRA Source Full v2（RUNNING_SOURCE，首批提交）

基准055fea8，v2独立分支codex/tbps-vlpt-full-v2。
两环境真实CPU预检PASS；Source不执行Target运行时检查，Target原生检查保留。
新输出tbps-framework-vlpt-official-irra-v2/full，控制vlpt_official_irra_full_v2；
v1的18个FAIL证据保持。官方SHA、模型、参数、385批/3074图/6156文协议不变。
运行代码34097c338997561fd7556e43c472d2e5625c05f0，监督器1307026（PPID1/无TTY），
16:42:11核验首批8图16文已提交、官方三阶段完整，第二批开始。Full与最终CPU审计PENDING。
[v2说明](framework/DRA_FULL_V2_FIX_REPORT.md)、[启动报告](framework/DRA_FULL_V2_STARTUP_REPORT.md)。


## VLPTAttack官方IRRA Source Full（首次启动FAIL，0产物）

独立输出tbps-framework-vlpt-official-irra/full；旧v3 Smoke和全部FAIL不覆盖。官方91f78325/RAttacker SHA75518cb9、8/255/10轮/2/255/batch8/seed42；3074图6156句。CPU预检/测试通过，首次实际启动fde8f6b在Source模型加载前因跨环境Target依赖校验失败。
控制vlpt_official_irra_full_v1保留FAIL；Source/MLM加载0、攻击0、Target评估0，无Full结果，
未自动重试。[失败证据](framework/DRA_FULL_LAUNCH_EVIDENCE.json)。

这里只索引已有不可变记录，不启动实验。发布/实验树别名：
- RELEASE=/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014
- OLD=/home/lzf/ldx/projects/idea-TBPS-test1
- G2=RELEASE/prototypes/attribute_image_generator_v2_20261014
- H1=RELEASE/prototypes/attribute_text_policy_h_v1_20261015

| 实验 | 不可变证据 | 状态/范围 |
| --- | --- | --- |
| G-v2 Epoch10 | G2/outputs/train_v2/training_protocol.json；checkpoints/epoch_0010/checkpoint.pt和checkpoint.json | 原10 Epoch完成；本阶段只读sidecar/协议/文件SHA，不恢复模型 |
| H-v1 Epoch7 Selection | H1/outputs/formal_v1/selections/30f573098d28b6af6afe146330823986f724378fc7bd4df9d257fdd2c6f20dd1.json | 独立Validation选模，非Test；visit14336、updates14259 |
| H-v1选中Checkpoint | H1/outputs/formal_v1/training/checkpoints/visit_00014336/checkpoint.pt/.json；training/protocol.json | 不继承32Step权重；Checkpoint/Adam/RNG由历史任务保存 |
| G-v2 Image-only Full | G2/outputs/evaluation_phase29a/full/{protocol,summary,completed,clean_verified}.json及metrics.csv | 官方IRRA/CUHK 3074/6156；含限定Storage兼容证据 |
| Joint Full | H1/outputs/phase5_joint/full/{protocol,summary,completed,baseline_verified}.json及metrics.csv | 原Full及独立CPU审计已由用户完成；文本/图库四组 |
| Joint Ablation Full | H1/outputs/joint_ablation_v1/full/{protocol,summary,completed}.json、metrics.csv、aggregate.csv及variants | completed.method_audit_passed=true；独立日志双成功标志已绑定，见下文 |
| AP B1/B2/B3 Full | /home/lzf/ldx/outputs/idea-TBPS-test1/baseline-ap-{duke-irra,cuhk-irra,cuhk-irra-surrogate}/evaluations/full/{summary,metadata}.json | 独立Baseline；Stage 1仅metadata级来源索引，不宣称全缓存新审计 |
| G-v1及属性/TTA历史 | OLD/prototypes/attribute_image_generator_20261014/outputs及各旧prototype | 保留历史基线；本阶段不迁移、不重验全部资产 |

核心SHA：
- G-v2 E10：f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e
- H选中权重：487f3061b6cba2ddabccdd49745a7b99c8c549f4541781b8814b299ef5f279df
- IRRA：9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009
- Selection文件：584e4ba2afc7b17fea3bdfb8501716498b84731fa216558daa2e8e4bb353f757

开发Commit索引：G2完整训练2a3f166；G2 Storage修复4e40c1a；H正式系统a668b1c；Joint d501f11；消融40c7476。指标/当前证据以原文件为准，不依据阶段README的旧CURRENT。

新桥必须区分元数据/直接证据已核验与全缓存/训练轨迹未重验。缺失路径返回NOT_READY，SHA/schema/来源冲突拒绝，不编造通过。

## Stage 1只读桥证据

Git目录：[legacy_assets.server.json](../configs/framework/legacy_assets.server.json)。8个索引：gv2_epoch10、hv1_epoch7_selection、gv2_image_only_full、joint_full、joint_ablation_full、ap_b1/b2/b3_metadata。

消融独立审计/管道日志：
H1/outputs/joint_ablation_v1/logs/ablation_20261016_163457.log。
SHA-256：ab69946e159054b263bc55619b74c9c6985d77efbc66f70b67fd57ddcb6b0872。
实际存在精确行ABLATION_CPU_AUDIT PASS和ABLATION_PIPELINE_SUCCESS。本轮只读取，未重新运行该审计。

只读桥输出：
/home/lzf/ldx/outputs/tbps-framework-stage1/bridges/readonly-history-v3/bridge_manifest.json。
8个索引均METADATA_VERIFIED，绑定192个唯一文件；H候选Validation和消融trial均保留原协议层次。新目录Manifest/回执另绑定catalog和3个实际producer SHA，不锁整个src/tbps目录。

此处通过范围仅为固定文件字节SHA、schema/关系及已有完成/审计证据。正式Full特征、Checkpoint内部状态与整个训练历史未重新审计。AP只核验metadata/summary关联；不据此宣称所有Baseline与Joint公平同协议，仍需按原pixel mapping、训练来源等逐项判断。旧IRRA特征不自动授权给新模型使用。

## Stage 2来源与身份核验

本阶段没有新攻击实验。固定目录configs/framework/stage2_sources.server.json绑定annotation、G/H内部来源、三份Full协议及IRRA官方来源。

- Test 3074/6156完整有序身份与G-v2/Joint/消融协议一致，未读取Full特征或重算指标。
- 原2048 Train/128内部Validation实际成员一致，PID交集0；128来自official train，不是official val。
- 7991个保护文件SHA/mtime不变；历史Checkpoint、Selection、结果及8个资产索引全部保留。
- Stage 2 CPU Model I/O检查只使用合成数据、Byte-BPE和小型排名Fixture；真实IRRA前向/加载尚未执行。
- 本阶段日志位于/home/lzf/ldx/tmp/tbps-framework-stage2，属于开发CPU证据，不是新的攻击效果实验，不提交Git。

当前实现及测试Commit统一由PROJECT_STATE登记。

## Stage 2.5 IRRA最小等价性（不是攻击效果实验）

一次受控RTX 4090验证，仅内部Train cuhk:9955:0、cuhk:29253:0。一个冻结IRRA Backend，官方独立路径与Adapter预处理/Token/图文特征/2×2分数逐值exact；5图像/4文本样本前向，1次输入导数，optimizer=0。

证据：/home/lzf/ldx/outputs/tbps-framework-stage25/irra_equivalence_v1/{attempt,result}/；result/report.json SHA b27212f576573f13d5129fcb644effb0dcb535bf13bea259ed2e55a0d09897d9；Manifest/回执CPU核验通过。7991份历史保护文件SHA/mtime不变；不是Full缓存全量复审。详细数据与未验证范围见framework/STAGE25_REPORT.md，Commit由PROJECT_STATE登记。

## Stage 3A插件CPU验收（没有新攻击实验）

复用注册表中固定G10、H7/Selection、IRRA训练来源；未加载模型。独立分支codex/tbps-framework-stage3a，Commit由PROJECT_STATE登记。285框架+36历史基础=321 CPU tests通过；原前端/Tokenizer只做合成Caption检查，真实G/H/IRRA前向0。

证据：/home/lzf/ldx/outputs/tbps-framework-stage3a/{cpu_verified.log,source_verified.log,frontend_synthetic_cpu_final.log,protection_result.json}。7991份历史保护文件SHA/mtime不变，Stage 2.5回执与关联来源只读核验通过。所有正式G/H/Joint/消融/AP实验及8个资产索引保持原样；不是Full指标或全缓存重新审计。详见framework/STAGE3A_REPORT.md。

## Stage 3B单次真实尝试：FAIL，未生成等价对照

固定内部Train cuhk:9955:0（可编辑资格）及cuhk:26530:0（固定NOOP），不按攻击效果选择。G10恢复1次、H7未恢复，因缺少condition_schema运行来源锁定停止；真实G/H/IRRA前向均0，无训练/排名。原G/H等价状态保持PENDING。

证据：/home/lzf/ldx/outputs/tbps-framework-stage3b/equivalence_v1/{attempt,result}/；FAIL report SHA e72a65118cb0c216e2fbb0877152d11524b0254f73cc9f2c8547e91900c7a79c。回执及2358来源SHA有效、无pending，7991历史保护文件不变；完成回执不表示验证PASS。GPU尝试仅1次，未重试。详细未完成项和修复建议见framework/STAGE3B_REPORT.md；Commit由PROJECT_STATE登记。

## Stage 3B-R CPU-only修复验收（不是GPU实验）

基准0fb90cd，独立分支codex/tbps-framework-stage3br；实现Commit由PROJECT_STATE登记。仅新增派生来源清单中condition_schema一项，原配置和全部历史实验不变。

CPU证据：/home/lzf/ldx/outputs/tbps-framework-stage3br/cpu_closure_v1/result/。
report SHA b974a016a61876fa0c4ee40df8faff6d9b83413afc092cf4d1a978be252d4401；Manifest SHA 19434bdcdbda7255ba582190fa5c003b8a0fe98a58fc27e0878cbc8392217f34。323框架+36历史基础CPU测试通过；75项来源、33份实际执行源码与11个符号核验通过，GPU尝试0。

7991历史文件SHA/mtime和原Stage 3B输出17文件不变；FAIL回执与2358来源仍有效。此PASS仅为CPU加载导入闭包，真实G/H/Joint等价仍PENDING；不覆盖或更名原equivalence_v1。完整边界见framework/STAGE3BR_REPORT.md。

## Stage 3B-2：固定G/H/Joint真实等价PASS（非攻击效果评测）

一次新受控尝试；内部Train cuhk:9955:0/cuhk:26530:0，固定G10/H7，RTX4090-G/CPU-H，FP32/math/no_tf32。两路径G输入/delta/像素/effective_delta逐值exact；H完整动作exact，K2双编辑及K0 NOOP；Joint三模式共享组件通过。G/H各加载1次，真实前向4/2，IRRA0，无训练/JVP/排名。

新证据：/home/lzf/ldx/outputs/tbps-framework-stage3b/equivalence_v2/{attempt,result}/。
report SHA 50afe401bf57ad244d74a014221825b58a1feb24f07201a42c501404dffecbb6；attempt Manifest 731e99aa2b8f19cf30ec1d8cdfad9ee78c1bca12d649d45e204269188bc8ec85；result Manifest 26185c4cad3f964c121a4d25d189d8f12dd8db3bfb5bd76dd734bde5c99d1f43。
日志和只读保护审计：/home/lzf/ldx/outputs/tbps-framework-stage3b2/。

332框架+36历史基础CPU tests通过；7991保护文件和原FAIL的17输出SHA/mtime不变。新2362/旧2358来源与回执有效，无pending。只在本固定环境标记VERIFIED；原equivalence_v1保持FAIL，正式Full/消融等资产不变。实现Commit见PROJECT_STATE；详细边界见framework/STAGE3B2_REPORT.md。

## Stage 4A CPU评测基础设施验收（无新攻击实验）

基准0d90e79，独立分支codex/tbps-framework-stage4a，实现Commit见PROJECT_STATE。383框架+36历史基础=419 CPU tests通过，51新增；独立官方rank合成小矩阵索引/五指标exact。未加载真实模型或feature Tensor，未重算Full。

新证据：/home/lzf/ldx/outputs/tbps-framework-stage4a/。
Joint metadata索引对应旧协议59ffb61a575cbeebf38e06bee5154b96f8967bb165b9b22305fad6c9cc967b48；3074/6156有序身份已对照annotation，仅声明3472 feature文件索引，实际字节仍PENDING。

原Full未记录逐张对抗RGB SHA；metadata桥不能伪造该证据或自动授权新ModelSpace重放。历史G-v2/Joint/消融/AP指标与回执保持原样，7991保护文件SHA/mtime不变。Stage 4B仅有条件预案，尚未执行；详细限制及固定等价阈值见framework/STAGE4A_REPORT.md。

## Stage 4B-1历史Joint特征接入验收（非新攻击实验）

基准32f55827，独立codex/tbps-framework-stage4b1，实现Commit见PROJECT_STATE。412框架+36历史基础=448 CPU tests通过，29新增；合成历史路径与正常路径score/五指标/DR exact。

只读原Joint Full四family：clean_gallery49批/3074，image_gallery49/3074，clean_text385/6156，adv_text385/6156。3472文件SHA/回执/结构/身份和6156条原生H动作合法性核验通过。METADATA_VERIFIED、FEATURES_VERIFIED；METRICS_REPLAY_VERIFIED=PENDING，真实Full scores/rank/metrics和模型前向均0。

证据：/home/lzf/ldx/outputs/tbps-framework-stage4b1/features_final/result/；
report SHA7d8dd8364df3537471658a0d3ce3c30f5e4609f700bb75a68e2f68e0fdc979bf；
Manifest SHA5cd8dcf0e8ccbc61b58adc7dae37e26e51f361369e15f6fefa77c9465bfc1370，绑定6720唯一来源。首轮features_v1核验也通过且保留，最终版补全新检查证据来源绑定后再次只读验证。

7991历史保护文件、3472Joint缓存SHA/mtime不变；原Stage3B FAIL、Stage3B-2 PASS、Stage4A回执继续有效。缺失对抗RGB SHA仍UNKNOWN；不冒充新攻击、不同Victim特征或完成Full指标重放。下阶段固定环境及阈值见Stage4B1报告，须另行授权。

## Stage4B-2：历史Joint Full CPU指标重放PASS（非新攻击实验）

基准a4c9387；独立codex/tbps-framework-stage4b2，实现Commit见PROJECT_STATE。唯一正式Attempt/replay_v1退出码0、METRICS_REPLAY_VERIFIED，旧Full/G/H/Selection及原Stage4B1回执未改。

原Joint四family868批、3472文件独立内容exact；全部3074 Gallery/6156 Query。新/旧评分各196次，新/官方Ranking各4次，score误差0/索引exact/五指标对原summary误差0，DR最大误差1.4210854715202004e-14个百分点。mAP Clean66.09744262695312、Image-only32.28228759765625、Text-only51.471561431884766、Joint23.553220748901367，来自本次真实计算。

证据：/home/lzf/ldx/outputs/tbps-framework-stage4b2/replay_v1/{attempt,evaluation,result}/；result Manifest f11b7af27e4c87d4c4f9be4e30f908d8e0afbd68f3db547759ce191f6cb0a39a，report SHA8c13994ab049ad7f76dbe380b6647887387b9be9e8be78f16382bfacf8a96f89。每套Manifest绑定6740来源，统一Score/指标读回通过。

471 CPU tests通过；7991历史保护文件及3472缓存SHA/mtime保持。峰值RSS1862.4453125MiB、总发布用时98.56915779039264秒；模型加载/前向/GPU/训练0。没有新RGB证明，像素SHA仍UNKNOWN；不是RaSa/APTM或新端到端攻击实验。原FAIL及旧验收证据保留，唯一许可已用完。完整五指标/DR和验收范围见framework/STAGE4B2_REPORT.md。

## Stage4C-1物化基础设施与历史来源检查（非新攻击实验）

基准c09754e，独立codex/tbps-framework-stage4c1；实现Commit见PROJECT_STATE。514 CPU tests（43新增）通过，真实模型加载/前向/GPU/JVP/排名/真实单元发布全部0。

G2 Test纯G(I)无需条件，3074原图SHA就绪；2176训练监督无Test成员。旧G1 3074条件仅只读校验，不供G2使用。H7原Joint Full 6156字符串/完整动作、385批1155元数据文件通过；6116编辑/40 NOOP/13537字符替换，固定可复用文本，不重新生成。

证据：/home/lzf/ldx/outputs/tbps-framework-stage4c1/source_audit_v2/result/；
report SHA37d244e6ec250a5d536e4a4e06832005a9e896233cc9b908f88e9b83365bf6e7；
Manifest SHA484605d0501d94dbb4a21315ba0f61c76499e17d283fd2a51c6eb7a1d96c9ce7，10570项来源。
文本动作摘要afe9b2a7302a4c135984768aa07a33d175ed18d3200dc08f970c137ae555f868。

7991历史文件和3472Joint特征SHA/mtime不变，原4B1/4B2回执有效。真实像素NOT_MATERIALIZED，历史文本TEXT_SOURCE_VERIFIED不等于新完整产物已发布；不能据Fixture宣称Test端到端或跨模型攻击通过。下一阶段仅有提案，未执行。

## Stage4C-2：固定Test物化Smoke（FAIL，未重试）

- 基准4d3bc025a7da593368ebd22cc05db1ebe787b446，独立codex/tbps-framework-stage4c2。
- 两图cuhk:37132:0、cuhk:37147:0真实G10物化PASS；两条Query cuhk:37132:0编辑、cuhk:37147:1 NOOP历史H7复用PASS。
- G10加载1/前向2，H加载0/前向0，IRRA加载0/图文前向0；Backward/Optimizer/JVP=0。L∞0.0313725471496582；原/Adv/delta读回bitwise exact。
- IRRA会话提前初始化CUDA后进入历史CPU校验，错误isolated CPU process required；最终FAIL，特征与2×2 Score对照PENDING，不是新Full检索结果。
- 证据/home/lzf/ldx/outputs/tbps-framework-stage4c2/materialized_smoke_v1/result/manifest.json SHA c1922bdf32935623dc9e85b1aeffe8a817cef2ab42b3ba767f62eda90e45155b。
- 537 CPU tests；7991历史文件+3472Joint缓存SHA/mtime不变，无pending，旧FAIL/PASS/重放回执保持。后置CPU保护与最终CPU验收分别post_cpu_protection_v1、cpu_verified_v1。
- [完整数据、来源SHA、失败调用链与未验证边界](framework/STAGE4C2_REPORT.md)。再次GPU需要独立修复审查与新授权。

## Stage4C-2R：CPU-only修复/旧产物审计（无新模型实验）

- 基准fcf097d3，独立codex/tbps-framework-stage4c2r；554 CPU tests，17新增，完整实际Runner时序Fixture通过。
- 原Stage4C-2两图RGB/delta、两条Unicode/action、原Plan/Claim/生成证明/权重来源严格只读有效；70原Attempt文件SHA/mtime不变，整体FAIL不提升。
- 新旧Plan明确分开：旧7c0c00da...保留，新e94013ed...只作为新树身份；旧单位不复制、不重签。
- CPU审计/home/lzf/ldx/outputs/tbps-framework-stage4c2r/cpu_audit_v1/result/，Manifest df0a94bea0bb8b454ba44d9a1e4b02ead859137327a0a6cce5e73250b9f78bf5。
- 来源75/旧执行33/官方执行5通过，实际CPU准备完成并退出隔离；本轮CUDA、真实模型加载/前向、GPU Attempt、Full排名0。
- 现有IRRA完整Storage1259.41MiB+模型419.80MiB，再加运行时有明显2GiB风险；保守2.3548GiB，非实测峰值。GPU申请暂停，阈值不变。
- 7991历史文件+3472Joint缓存SHA/mtime保持，原Stage3B FAIL/Stage3B2 PASS/B2重放/C2 FAIL不变。
- 真实特征与四组Score等价PENDING；[完整报告与资源/IRRA-only提案](framework/STAGE4C2R_REPORT.md)。

## Stage4C-3：固定两图两文IRRA物化端到端PASS

基准9f1f943e，独立codex/tbps-framework-stage4c3。唯一irra_only_v1退出0、IRRA_MATERIALIZED_SMOKE_VERIFIED，原Stage4C-2整体FAIL保持。只读原RGB/Unicode，不生成G/H、不做排名/Full或训练。

IRRA恢复1、图像4/文本4；G/H加载/前向0，Backward/Optimizer/JVP0。原/Adv Gallery特征bitwise exact；原/Adv文本最大误差1.9371509552001953e-07。Score Clean/Image-only/Text-only/Joint误差分别7.450580596923828e-08/1.0151416063308716e-07/8.940696716308594e-08/1.0151416063308716e-07，均≤固定1e-6；模型状态exact。

证据/home/lzf/ldx/outputs/tbps-framework-stage4c3/irra_only_v1/，result Manifest e6aa5f89e9fce861bfd900c5160629b6f54bc196a858b43ee341116539062f70，report c85ffc5a315dcfe233a2c44459400f2489059bedcc2ead6ca6aaf728436de98f。原Plan 7c0c00da...和70旧文件不变。

568 CPU tests，7991历史文件/3472Joint缓存SHA/mtime exact，后置CPU审计通过。新授权取消旧RSS上限，实测RSS2.350719451904297GiB、GPU allocated922.380859375/reserved1002MiB，执行段24.73483756557107秒，未重试。仅该固定样本/环境VERIFIED，Full/跨模型仍未覆盖；[完整原始Score与资源](framework/STAGE4C3_REPORT.md)。

## Stage4D-1：6156历史H7文本全量物化（非新攻击效果实验）

基准a5a6f6a，独立codex/tbps-framework-stage4d1，实现Commit由PROJECT_STATE登记。唯一正式CPU物化退出0、TEXT_FULL_MATERIALIZED；6156 Query原顺序，6116编辑/40 NOOP/13537字符替换，H/G/IRRA模型加载/前向/GPU/训练/排名0。G流式Runner仅16图CPU Mock，未恢复G10、未生成3074真实图像，Image/Joint全量PENDING。

证据/home/lzf/ldx/outputs/tbps-framework-stage4d1/materialized_full_v1/：result Manifest a149a57fc7e8c87a389ede80c7e2eaedafec9d3ffc81dff9a7275915040c615f，text-full/text_collection.json内容SHA9ab8881dab9f82bd31cc31c0a7cc478e3beca9b2f9b940443603efa54d21069b。Plan7ac4fefd0abaa10df085be3ec3ed56c4b38c782a21bfc44473f40b90965447d5绑定C1/C3及原模型/Selection/数据来源。完整原文与Unicode/action独立物化，不是新H推理或旧Embedding转RGB。

独立full_audit_v1/result/ Manifest 913b87ecf0620730e537a33f96858e4dff41fbb6de8fa233b085da077fbbc9a2、report114857db1ca8c29b7a44cf592ab11693feeab0c823dc3f87bac6a3543143866f，TEXT_FULL_CPU_AUDIT_PASS。6156全文/action exact，有序摘要afe9b2a7302a4c135984768aa07a33d175ed18d3200dc08f970c137ae555f868；30797新文件、7991历史文件、3472Joint缓存、70原FAIL文件SHA/mtime保持，8个保护工作树HEAD/clean未变。

591 CPU tests通过；全量记录879.7607615217566秒/RSS607948800 bytes，独立审计110.86877850443125秒/RSS607436800 bytes。原Stage4C2整体FAIL不提升，原C3小样本PASS保持；没有新mAP/Score或跨模型结果。详细SHA、吞吐限制与Stage4D-2待授权方案见framework/STAGE4D1_REPORT.md、STAGE4D1.md。

## Stage4D-2：G全量入口CPU开发/资源阻断（无新GPU实验）

基准82fe88a，独立codex/tbps-framework-stage4d2，实现Commit由PROJECT_STATE登记。预检最终因GPU已有PID2209695/显存20164MiB、空闲3879MiB被拒绝；退出1，无CUDA/G/H/IRRA恢复，无正式generation_v1、GPU Attempt0。没有等待后自动重试或抢占；Image/Joint全量PENDING。

611 CPU tests通过（52.47秒，20新增）；完整源检查与历史保护单独CPU通过：75来源/33旧执行文件、3074原图/原顺序、6156 H文本/action、6116编辑/40 NOOP/13537替换均核验；原30797文件快照SHA/mtime、7991历史文件、3472缓存、70旧FAIL文件保持。冻结Stage4D1库/原Plan不改，旧两图像素来源只读有效，新生成对照未执行。

证据/home/lzf/ldx/outputs/tbps-framework-stage4d2/cpu_source_verified_v1/result/，Manifest SHA5e5d7ac9a22992fbb1f0559d85acb390596a111a56e1293286459d7adb3984e1，report SHA55870922aebc9d02ebd4190c232ce27ef911e8a0ac2574b93fd8945f991e1729；CPU_SOURCES_AND_HISTORY_VERIFIED_GPU_RESOURCE_BLOCKED。原资源拒绝日志/home/lzf/ldx/tmp/stage4d2_cpu_preflight_v1.log SHA b1f9cacd38a46b83ee1a315a26e5b22d25019b8effa355b00e69f6f3d289185e。

未产生新G像素、Image/Joint完成回执、mAP/Score或GPU成本数据；原H文本完成/原C3小样本PASS及旧FAIL保持。GPU资源就绪后须用户指示继续，不能据CPU Fixture/源核验自动启动。见framework/STAGE4D2_REPORT.md、STAGE4D2.md。

## Stage4D-2资源就绪后唯一正式Attempt：FAIL

基准883c4be59284b79f475412eb40ea6d96e8b1742a，复用原代码和配置。资源及CPU来源预检通过；generation_v1/gpu-start已发布，随后因安全包装改变CPUStartupGate快照而在CUDA初始化前拒绝。G加载尝试/恢复/前向0，H/IRRA及优化0；未发布任何图像/逐图Claim/Proof或Image/Joint集合。原H6156完成保持，不能宣称Joint就绪。

证据/home/lzf/ldx/outputs/tbps-framework-stage4d2/generation_v1/：attempt Manifest97208f29d78ef656069cab51aba9a2f1f64f2c51400fe256783233d4eba4889d，gpu-result/report SHA1fcc0b7cf487043d0b4b5762057ad6380308dd753f231673da555b2ed9388c26，result Manifestde017a15233cc57184108c5a10aaa6effe34f49e7afb736f7b76de3833ec8174。整体/子结果FAIL，正式Attempt1，不得重跑原目录。

failed_attempt_cpu_audit_v1/result/ Manifest86bbeda8a4e24aa840ec0343d9e2eafebc7b658628f045ae931b700452667420，FAILED_ATTEMPT_READONLY_PROTECTION_VERIFIED：30797旧文件、7991历史文件、3472缓存、70旧FAIL文件及16新失败文件SHA/mtime不变，无pending。611 CPU tests仍通过（51.03秒），不替代真实GPU验收；无新像素/mAP/Score。报告framework/STAGE4D2_EXECUTION_REPORT.md。

本次按“全部通过后提交推送”的条件未commit/push，必要Markdown为工作副本更新；源码HEAD883c4be不变。旧资源阻断记录、旧H回执及此前全部FAIL/PASS证据保留。

## Stage4D-2R：CPU-only启动修复（非GPU实验）

基准883c4be，独立codex/tbps-framework-stage4d2r。626 CPU tests通过（611原+15新增），实际child组合门禁/包装/Hook/Claim/Fixture封存及故障恢复验证。75来源/33导入闭包PASS；30797原物化文件、7991历史文件、3472缓存、70旧FAIL及16新FAIL SHA/mtime不变。

证据/home/lzf/ldx/outputs/tbps-framework-stage4d2r/cpu_audit_v1/result/，Manifest SHA24a39edf72666ae9279009e4e97751d51c268787a29a0a35f5944d987a690461，report SHA381f1434f5e0df5a49fc5a4ab58a6cb95726ca4dcf63fe142c620b7fb0246682；STAGE4D2R_CPU_REPAIR_SOURCES_AND_PROTECTION_VERIFIED。原执行报告字节SHA dca03f00ccc8333e1c13eec310f24d5601fa37eb9c00f45a9adc12cff1a9c143完整收录，原树文件SHA/mtime与未提交状态保留。

generation_v1仍FAIL；generation_v2尚不存在，真实G/H/IRRA加载/前向及CUDA初始化0。原Text完成、Image/Joint PENDING，无新攻击效果或跨模型结果；下一次GPU必须单独授权。完整边界见framework/STAGE4D2R_REPORT.md。

## generation_v2：预处理误拦截FAIL及CPU修复

原执行来源dd0a213f，G10恢复1、实际前向0/0、新图像0；torchvision Resize在原像素读取时被非G保护拒绝。原Store首图cuhk:37132:0 Claim1/Proof0/pending0，H6156不变。原v2证据/home/lzf/ldx/outputs/tbps-framework-stage4d2/generation_v2/；result Manifest9b2dae20980ce9e1943fcaada80e2d73539fd09fb9ea3d4ada106e077d79efa9，gpu-report176ae52ffc1851a3d710c445268569a987baaed912d95696941ec7e73c052d13。原Claim Manifest21ed14c935fdde887cfc6afcbdb67b6e6c6687085f6b8a46e8c3aed5ad5ea1e0。

独立codex/tbps-framework-stage4d2r2仅CPU修复/647测试：受限原预处理、零前向首图恢复、正常与独立恢复Proof、实际child组合及3074顺序Mock。正式generation_v3未创建，真实模型/前向/CUDA/新图像0，原v1/v2 FAIL不提升，Image/Joint仍PENDING。CPU只读证据及边界见framework/STAGE4D2R2_REPORT.md。

CPU只读资格证据/home/lzf/ldx/outputs/tbps-framework-stage4d2r2/cpu_audit_v2/result/，Manifest73d895b065cac4d3abe9316478811fa5a6cc1fa6eb9414b2f8532668d54e61fc、report030bbdedf5f51becdbaec613219a556a923f3442ee061643da03deca41e4ca7c。原30797＋3 Claim、7991历史文件、3472缓存、70旧FAIL、v1/v2各16文件SHA/mtime保持；75来源/33导入PASS。首次CPU清单结构拒绝及修正记录保留，无新增GPU尝试。

## generation_v3共享许可CPU开发（无新GPU实验）

基准029a081，独立codex/tbps-generation-v3-gpu-sharing。仅当前Runner允许已绑定Ollama PID2436556共存，启动3584MiB/≤10%、运行1024MiB安全余量，身份变化或未知计算进程立即拒绝。本轮无正式GPU Attempt、模型加载/前向或新图像；generation_v3仍未创建，Image/Joint不提升。

687 CPU回归/88.78秒＋最终40项共享复核/2.54秒通过。CPU证据/home/lzf/ldx/outputs/tbps-generation-v3-sharing-cpu/；原R2树580文件、旧FAIL与Claim35文件SHA/mtime和原Store30800成员/size/mtime保持，冻结Stage4D1 clean。完整历史内容审核引用R2既有CPU证据，不伪称本轮重跑。用户手动执行说明见framework/GENERATION_V3_GPU_SHARING.md。

## generation_v3正式全量物化及独立审计（用户执行完成）

实际核对generation_v3/result/report.json=JOINT_FULL_MATERIALIZED、generation_v3_cpu_audit_v1/result/report.json=JOINT_FULL_CPU_AUDIT_PASS。3074图/6156文、G真实前向3074/恢复1、最大L∞0.0313725471496582，H新加载/前向0；6116编辑/40 NOOP/13537替换。原7991历史文件、3472缓存及70旧FAIL的保护由此已完成审计记录，不是新的检索指标。

生成结果Manifest fb696a1c0c3235018202118a2d945fa728f6fc06be39738f04a26e6a39fe7634；独立审计Manifest 5a693b3877a35c6778e79de7c3f9e555f7772191bbd2e3af5624fcef268bdce5。原Plan/Store和v1/v2 FAIL保持不变。

## 官方IRRA物化四组评估准备（无新GPU结果）

独立codex/tbps-irra-materialized-evaluation，默认CPU来源检查，显式用户参数才编码。25项新测试、704项新树回归及原生产树2项来源测试通过；真实CPU Full预检PASS、两个RGB读回、完整身份/Unicode一致，真实模型/GPU/Full排名0。CPU证据/home/lzf/ldx/outputs/tbps-framework-irra-eval-development-cpu/。

四组真实Full及新mAP尚未执行；不将历史23.5532等数值填作本次结果。执行与审计方案见framework/IRRA_MATERIALIZED_EVALUATION.md。当前不接入RaSa/APTM。

## 物化IRRA共享GPU Smoke（10 Gallery + 10 Query）

现有评估工作树的首次attempt-000001通过，未复用或覆盖其他Attempt；此前GPU独占门禁失败未创建输出。读取generation_v3固定RGB/Unicode，IRRA原权重/FP32/math/no_tf32/batch1，真实20图像与20文本样本前向，G/H0。四组小样本mAP依次100.0/93.0/99.16666412353516/90.95238494873047，仅验证链路。

证据：/home/lzf/ldx/outputs/tbps-framework-irra-materialized-evaluation/smoke/{protocol,execution-000001,completed}/；summary SHA d9499227e2f785b96af640f8b9bf6ec88285b0c5aac6f2664a5c68c827067f28，execution report SHA 6a4752556488ec311f9856e1d07311cc287430ef811f792110a9e550fca1a84d。参数/buffers exact、无OOM/重试，完整日志/home/lzf/ldx/tmp/irra_materialized_smoke_shared.log。正式Full未启动，资源共享不代表其他硬件/批量均验证。见framework/IRRA_MATERIALIZED_SHARED_GPU_SMOKE.md。

## IRRA Materialized Full完成（用户执行，独立CPU审计PASS）

Git b51d9e55190e96b6422f0c87952471ad6ac12bed；固定generation_v3 RGB/Unicode，3074/6156四组原生IRRA重新编码。四组mAP66.09744262695312/32.28228759765625/51.471561431884766/23.553220748901367，历史Joint Full20项指标exact。IRRA加载1次、图像6148/文本12312，G/H0。证据/home/lzf/ldx/outputs/tbps-framework-irra-materialized-evaluation/full/completed/summary.json，SHA4530f3f97156d6e9f65c07b0775588324c39679feb0f16ae66b356692b017eb0；/home/lzf/ldx/tmp/irra_materialized_full_status.log含SUCCESS | IRRA_FULL_AND_CPU_AUDIT，audit日志含MATERIALIZED_IRRA_CPU_AUDIT_PASS full。本轮只读引用，未重跑。

## RaSa固定攻击产物迁移Smoke（非正式Full结论）

官方RaSa代码bd16aa1、CUHK权重bc85da09...，独立原生空间/缓存。同一generation_v3前10 Gallery/前10正例覆盖Query，20图像/20文本/400融合pair，模型加载1，G/H/IRRA0；唯一attempt-000001 PASS。四组mAP100.0/95.88888549804688/99.16666412353516/91.44444274902344，不解释为正式迁移成功率。

证据/home/lzf/ldx/outputs/tbps-framework-rasa-materialized-evaluation/smoke/；summary SHA036ca6df1ffc793dfb39e36cc7b38294b12655f0f23975dc005822829c04aa61，execution SHA8a9b7bcc0b73017f3cc85a8809b530c4a21e55ca071ab0242291f39d7c989995。参数/buffers exact、无OOM/重试。Full保持Top-128，当前仅10候选Smoke；固定384×128物化RGB到384×384 Tensor预处理区别于官方原始图片PIL基线。正式Full尚未执行，等待用户手动运行。见framework/RASA_SMOKE_REPORT.md。

## RaSa Full GPU等待器部署授权

用户明确授权独立nohup进程等待GPU0资源：每60秒、连续3次≥8192MiB/≤10%/无其他计算进程后单次Full，成功后CPU审计。等待状态不表示Full已执行；真实启动/结束以/home/lzf/ldx/tmp/rasa_full_wait/{poll,status,launch}.log及正式Full回执为准。开发仅10项CPU Fixture通过，不新增真实GPU测试。现有RaSa Smoke、IRRA Full及generation_v3保持。

## APTM迁移评估代码准备（非真实GPU实验）

官方Commit b20025445471eb4b743164a6faec29a9eccfb293；官方CUHK checkpoint SHA521c9bf784e7e821ade471deaafa2f25e2501950591e574dc60ca177e9be5a2c。冻结generation_v3 3074/6156输入原样复用；APTM独立模型空间，Swin384×128、BERT56、完整集合归一化Top128 ITM＋.002 global。

CPU官方模型629状态strict匹配，GPU未初始化；140项必要CPU测试、完整身份/来源预检与10+10成员预检通过。证据/home/lzf/ldx/tmp/aptm_development/{cpu_tests_final,full_cpu_preflight,smoke_cpu_preflight}.log。预期正式输出/home/lzf/ldx/outputs/tbps-framework-aptm-materialized-evaluation/{smoke,full}/，本轮未创建，无APTM真实mAP或迁移结论。

RaSa后台任务不由本任务监控/修改。先由用户确认资源与RaSa任务结束，再执行APTM Smoke；Full需Smoke验收后手动运行。源码/CPU通过不授予新GPU许可。

## APTM一次性自动链部署授权（尚非GPU实验结果）

基准e7d28861cf5a91a801e47b699d58ac9389900f4b。用户授权独立nohup链等待RaSa Full/CPU审计/进程退出及GPU≥20000MiB/≤10%/无计算进程连续3次后执行APTM Smoke；Smoke CPU审计通过后独立重新等3次，再Full与CPU审计。失败不重试，保留现场，不改G/H/RaSa/IRRA。

21项新控制流程CPU测试与46项相关回归共67通过，模型命令/资源均Fixture；部署阶段不声称真实APTM mAP或Full完成。实际阶段以/home/lzf/ldx/tmp/aptm_full_wait/{phase,status.log,launch.log}与正式APTM输出为准。旧RaSa没有独立JSON CPU审计回执，显式核对旧成功行及CPU PASS日志、正式完成/执行SHA后仅在新控制目录写只读确认。


## 三模型物化Full完成与只读结果汇总（2026-10-18）

IRRA、RaSa、APTM正式Full与独立CPU审计均PASS；每模型首次正式Attempt，6148图像/12312文本样本前向，受害者加载1次，RaSa/APTM融合pair各3151872。G/H新加载/前向及优化0。报告：[三模型正式结果](framework/results/three_model_full_20261018/REPORT.md)，原精度full_precision.csv及来源evidence.json同目录。

以下mAP均按Clean → H-v1 Text-only → G-v2 Image-only → G-v2 + H-v1 Joint排列：
- IRRA：66.09744262695312 / 51.471561431884766 / 32.28228759765625 / 23.553220748901367；同模型评估。
- RaSa：69.18406677246094 / 32.85859298706055 / 35.04695129394531 / 17.202625274658203；跨模型迁移。
- APTM：64.12683868408203 / 51.801902770996094 / 39.662025451660156 / 30.490341186523438；跨模型迁移。

三者同一3074 Gallery/6156 Query及generation_v3内容，官方身份/排序/配对和G10/H7来源一致；各自原生预处理/Tokenizer/评分与ModelSpace独立。本轮只读提取和标量统计，不重跑审计或排名。关键证据：
- IRRA：/home/lzf/ldx/outputs/tbps-framework-irra-materialized-evaluation/full/completed/summary.json，SHA 4530f3f97156d6e9f65c07b0775588324c39679feb0f16ae66b356692b017eb0；CPU审计/home/lzf/ldx/tmp/irra_materialized_full_audit.log，精确PASS标志存在。
- RaSa：/home/lzf/ldx/outputs/tbps-framework-rasa-materialized-evaluation/full/completed/summary.json，SHA 12b3a90ca761c632bdf7781e7d52b43b753ddb5e7bdf36ad53e65515d7fc37ac；CPU审计/home/lzf/ldx/tmp/rasa_full_wait/cpu_audit.log，精确PASS标志存在。
- APTM：/home/lzf/ldx/outputs/tbps-framework-aptm-materialized-evaluation/full/completed/summary.json，SHA 8944ea0cd29e5c9f4410bf9ad7ccc65f67e3c26ad8a794e2a8af6915bca36d18；CPU审计/home/lzf/ldx/tmp/aptm_full_wait/full_audit.log，精确PASS标志存在。

本轮12行/60指标/45DR与完整来源核对通过，5份Checkpoint实际SHA匹配；原2682已读取文件保持，报告不包含Smoke数值。尚未开展新的训练、跨数据集或额外方法实验。

## AP-Attack B3三模型对比准备（2026-10-18）

- 冻结B3 E60：/home/lzf/ldx/outputs/AP-Attack/cuhk_irra_semantic_10_10/G_CUHK_IRRA_AP.pth.tar，SHA c1c637e2e5346a3011f23006bb77bad60e4bd4bc28eb746d5f6bfcd881ba4ebb；只索引B3。
- 历史3074图/97批缓存：/home/lzf/ldx/outputs/idea-TBPS-test1/baseline-ap-cuhk-irra-surrogate/adversarial-gallery/，SHA/顺序核验通过；现有IRRA旧Full不是本轮新结果。
- 新CPU物化10图：/home/lzf/ldx/outputs/tbps-framework-b3-comparison-ready/smoke/pixels/，最大L∞ .031372129917144775，G/H前向0。
- 新三模型GPU Smoke及正式Full均PENDING；ready仅资源拒绝、无模型Attempt。初次CPU门禁FAIL保留在/home/lzf/ldx/outputs/tbps-framework-b3-comparison/smoke/irra/execution/。
- Full CPU预检PASS，来源摘要framework/B3_SOURCE_AUDIT.json，阶段报告framework/B3_STAGE_REPORT.md。既有三个受害者Full及generation_v3仍不可变，不更新其指标。

## B3共享GPU正式链准备（2026-10-18，尚未启动）

基准097c96d，codex/tbps-b3-shared-run。49项B3 CPU/149项相关回归PASS，Full CPU预检3074/6156、97历史批、CUDA=false。旧B3来源与正式三模型结果保持只读；新正式输出预定/home/lzf/ldx/outputs/tbps-framework-b3-shared-evaluation，当前不存在。用户手动nohup+setsid启动后依次Smoke/审计/Full/审计，共享Ollama，单次资源快照、无轮询、不重试。监督目录预定/home/lzf/ldx/tmp/b3_shared_run，当前未部署，无新的GPU实验结果。入口/日志/停止命令见framework/B3_SHARED_RUN.md。

## AP-Attack B3三模型Full结果（用户执行完成，本轮只读汇总）

正式输出/home/lzf/ldx/outputs/tbps-framework-b3-shared-evaluation/full/{irra,rasa,aptm}/completed/；三模型Full和独立CPU审计PASS，监督status.log含SUCCESS | B3_THREE_MODEL_FULL_AND_CPU_AUDIT。冻结B3 E60 SHA c1c637e2e5346a3011f23006bb77bad60e4bd4bc28eb746d5f6bfcd881ba4ebb，G-v2 E10及三个受害者来源不变。

仅四指标、Clean→AP-Attack B3→G-v2 Image-only：IRRA mAP66.10/40.43/32.28、RaSa69.18/39.99/35.05、APTM64.13/42.39/39.66。控制组与历史Full原精度exact；新增B3目标编码，不重新生成攻击。IRRA同模型、RaSa/APTM迁移；训练预算/文本监督/辅助来源未对齐，不标作等预算对比。

[正式报告与CSV/证据](framework/results/b3_three_model_full_20261018/REPORT.md)。本轮9行36指标、完成回执、完整身份/ModelSpace及五Checkpoint实际SHA只读核验，82读取文件前后不变，无GPU/排名/新审计。上文准备阶段PENDING保留历史快照属性，当前以该完成证据为准。

## SGA-TBPS三目标白盒真实Smoke（2026-10-18，非Full）

实现与报告Commit：d6e719a2cccc5a2d3e01f2465600464358a013fa；分支codex/tbps-sga-whitebox-fixed。

官方SGA f49e6926cd13148e49bdbc2c3e537660e096230c；IRRA/RaSa/APTM分别使用自己的冻结Checkpoint和输入梯度，10张Test Gallery的全部20 Caption作集合，评估10条Clean Query。每目标独立生成10图；预算8/255、10轮、5尺度。三个Smoke与独立CPU审计PASS，Full未运行。

正式Smoke证据/home/lzf/ldx/outputs/tbps-framework-sga-whitebox-v2/smoke/{irra,rasa,aptm}/；审计/home/lzf/ldx/tmp/sga_fixed_smoke_<model>_audit.log。SGA四项小指标见[报告](framework/SGA_SMOKE_REPORT.md)，原始来源SHA索引SGA_SMOKE_EVIDENCE.json。各目标受害者加载1、视觉510、文本0、融合pair0/1200/1200；模型摘要exact、无G/H/Optimizer/属性JVP。小指标不得放入正式Full比较表。

原初始IRRA Smoke FAIL另存tbps-framework-sga-whitebox/smoke/irra/，冻结源码80361ec1fdb7c4c96e4ec68fa256742cc8188566。Claim1/图像0，异常未持久化Hook计数，实际视觉数UNKNOWN（5..50）。预算精确边界派生修复后使用新v2输出，不覆盖旧FAIL。保护2491 SHA/stat＋4工作树＋18FAIL＋12203历史stat均不变；121相关CPU tests通过。

当前适配为image-only，不等同原多模态SGA；RaSa/APTM用原生ITM可微代理，最终评估使用原生排名。Full预定tbps-framework-sga-whitebox-v2/full/{irra,rasa,aptm}/，仅提供用户手动nohup命令，无后台部署或Full结果。

## SGA-TBPS三模型正式Full与独立CPU审计完成（2026-10-18）

正式报告与CSV内容Commit：a3a33be0cbb6c06fda7bdbdaf7a529b11aa583e6；结果分支codex/tbps-sga-full-results。

冻结执行Commit6f89b799922ac54f041dabd3cd8e9d444735b146，官方SGA f49e6926cd13148e49bdbc2c3e537660e096230c。正式输出/home/lzf/ldx/outputs/tbps-framework-sga-whitebox-v2/full/{irra,rasa,aptm}/，各completed及execution回执PASS，监督器status.log最终2026-10-18T22:19:11+08:00 SUCCESS | SGA_THREE_MODEL_FULL_AND_CPU_AUDIT。各独立审计/home/lzf/ldx/tmp/sga_whitebox_full_v2/<model>-full-audit.log，SGA_TBPS_CPU_AUDIT_PASS full <model>。

三个目标分别白盒优化完整3074 Gallery，6156 Query原文不变，L∞≤8/255、10轮5尺度。SGA R@1/mAP：IRRA0.49/0.54、RaSa1.61/1.46、APTM1.71/1.39。当前image-only TBPS适配及APTM局部ITM代理边界不变；SGA对白盒目标的权限不同于冻结G/B3迁移，不能解释为等预算公平优势。

四方法正式表、原精度CSV和关键来源：[报告](framework/results/sga_three_model_full_20261018/REPORT.md)。模型IRRA→RaSa→APTM；方法Clean→AP-Attack B3→G-v2 Image-only→SGA-TBPS White-box；仅四指标。SGA指标来自实际Full；Clean/B3/G-v2复用此前已审计正式Full且24项控制值exact。上文开发/Smoke阶段的Full PENDING保持历史快照属性。

本轮仅标准库CPU核验12行48值、完整身份/ModelSpace及9222单元/Claim元数据；58711实际读取文件SHA/stat不变。未运行新GPU、模型、排名或完整Tensor审计，不修改旧实验、原FAIL、Checkpoint或攻击器。下一Baseline未开始。

## SGA实验归类纠正：官方Baseline尚未复现（ADR-0009）

此前SGA-TBPS White-box全部结果、源码与审计保留为额外白盒适配，不能作为原论文SGA Baseline。旧表数值未改，旧Full执行PASS仅证明该适配实验完整。官方SGA固定f49e6926cd13148e49bdbc2c3e537660e096230c，当前NOT_REPRODUCED/静态审查阶段，无新CPU单元测试、模型/GPU/Full或下载。

主方法明确G-v2 + H-v1 Joint。源ALBEF建议、CLIP/BERT实际资源SHA、原文本→图像→文本与2/255参数、CUHK2/10 Caption组、输入预处理/Matched Clean及权限差异见[审查报告](framework/SGA_OFFICIAL_REPRODUCTION_REVIEW.md)。新官方结果不得引用旧白盒指标填表，不自动继承其PASS；本轮尚无官方SGA-on-CUHK实验指标。

## 官方SGA算法 / IRRA Source固定两图四文Smoke（2026-10-19，非Full）

唯一攻击源IRRA，固定官方SGA f49e6926…原完整流程及2/255/10轮/0.5/255/五尺度，G/H均未使用。Source/MLM各加载1次，实际源图像105/文本265（含一次额外输入梯度检查）；2张对抗图、4条最终词编辑。IRRA同源评分，RaSa/APTM同一产物黑盒原生评分，各32融合pair。原IRRA目标编码后R@10门禁FAIL保留，CPU复用缓存恢复R@1/mAP；IRRA目标最终冻结摘要及实际forward日志未完整记录。

独立两图Score CPU审计及真实像素/词编辑CPU审计PASS；SourceIRRA/MLM及RaSa/APTM参数buffers摘要exact。原pipeline exit1没有改签。两图R@5/R@10为N/A，实数L∞高于数学2/255约2.93e-8、官方FP32投影边界exact，不宣称严格实数预算PASS或正式攻击效果。

产物根/home/lzf/ldx/outputs/tbps-framework-sga-irra-source/smoke/；正式完成证据分source、irra_smoke_scores、rasa、aptm、smoke_audit、source_cpu_audit。原irra/execution FAIL和全部旧白盒/Full证据保留。详情framework/SGA_IRRA_SOURCE_SMOKE_EVIDENCE.json及SGA_IRRA_SOURCE_SMOKE_REPORT.md。

Full仅CPU预检3074/6156通过；未授权或执行Full。当前主方法仍G-v2+H-v1 Joint。原论文ALBEF实验、R²TUA同协议复现、batch8 Full容量及Matched Clean与历史Raw Clean数值等价均未验证。

## 官方SGA IRRA Source Batch8容量测试（非Full，2026-10-19）

一次Source-only实际Attempt，前8张官方Test图cuhk:37132:0至cuhk:37139:0与全部16句。batch8及全部原官方参数/流程不变；IRRA/MLM各加载1次、实际图像417/文本1026，目标评分0。SGA_BATCH8_CAPACITY_PASS，独立CPU读回PASS；攻击9.02253秒，GPU allocated6.3714GiB/reserved10.6270GiB、Peak RSS2.2802GiB，无OOM。参数buffers前后exact，官方FP32预算边界PASS，实数舍入差2.93348e-8，未声称严格数学上界通过。

证据根/home/lzf/ldx/outputs/tbps-framework-sga-irra-capacity/batch8_v1/source/；日志/home/lzf/ldx/tmp/sga_batch8_capacity_v1.log。SHA/实测数据见framework/SGA_BATCH8_CAPACITY_EVIDENCE.json。164旧Smoke文件与历史来源/结果保护，不重写旧完成回执。Full监督器/进度入口已完成CPU测试但未运行；正式Full输出、started及后台任务均未创建。

## SGA官方IRRA Source Full v1启动FAIL与v2修复

v1控制目录/home/lzf/ldx/tmp/sga_official_irra_full_v1/，CPU预检PASS后df不存在路径失败，Source/Target均未执行，正式Full输出无。全部13个证据文件保留，不能改签成功。v2仅修Bash父目录检查、独立ROOT/CONTROL；8项CPU测试PASS，算法/模型/配置字节不改；实际启动核验见framework/SGA_FULL_LAUNCH_V2.md，不能把启动成功当成Full完成。

v2实际启动核验（2026-10-19 00:51:54 +08:00）：修复代码f8d3431b4f51d4fb117cf031f6d827bb89fab69c，独立nohup/setsid监督器PID/PGID/SID=3001120、PPID1/无TTY。CPU预检PASS，当前RUNNING_SOURCE、IRRA初始化已有日志；尚非Full完成。资源及startup_verified证据路径见SGA_FULL_LAUNCH_V2.md。v1十三文件保护未变。Codex未持续监控。

## 官方SGA算法（IRRA Source）三模型正式Full与独立CPU审计完成（2026-10-19）

执行f8d3431b4f51d4fb117cf031f6d827bb89fab69c，官方算法f49e6926…，唯一IRRA源。385批/3074图/6156句一次生成，IRRA同源及RaSa/APTM同产物迁移。四种模式Clean（Matched）/SGA Image-only/SGA Text-only/SGA Joint；Joint R@1/mAP分别2.70/2.29、39.13/35.64、42.59/37.59。不要与旧SGA-TBPS White-box表混用。

Source/三Target/CPU审计PASS，v2最终SUCCESS标志及phase=SUCCESS；00:51:36→06:07:53，共18977秒。所有模型冻结，Target source_regeneration0，6145最终词替换/11NOOP均保留。FP32边界通过但不宣称严格数学2/255上界通过，已批准约2.93e-8舍入语义。

[正式报告](framework/results/sga_official_irra_full_20261019/SGA_OFFICIAL_IRRA_FULL_REPORT.md)，同目录SGA_OFFICIAL_IRRA_FULL_METRICS_2DP.csv和SGA_OFFICIAL_IRRA_FULL_EVIDENCE.json。本轮只读核验4647产物、21335来源、48数值与官方身份及共享内容，无新GPU/Tensor/模型/排名。与历史G-v2+H-v1 Joint只作跨协议描述性比较。v1 FAIL/旧Smoke/白盒/全部历史结果原样保留。

## VLPTAttack（DRA）官方IRRA Source Smoke

- 官方91f78325e7cf0c497c9bbb5be2215be652d8cd9d；RAttacker SHA75518cb9465d9f8e994ea2f59c86b89fd4c39e3fe74f66b7a1d0a090eec2d313。
- 源成功代码8513a49b6f53c63ea98a8f6795b4e659b8fb88de；目标成功代码d455f3a05797451dd4d2ac145ad03b58090d3fa2。
- /home/lzf/ldx/outputs/tbps-framework-vlpt-irra-source-v3/smoke/source/：前8图/16句完整原算法生成、权重冻结、FP32预算语义及来源回执PASS。
- 同目录irra-session-v2、rasa、aptm：同一Source Collection四组原生评分PASS；audit/completed/audit.json独立CPU审计PASS。
- 旧两个Source输出、旧irra及初次CPU审计FAIL全部保留，参见framework/DRA_SMOKE_REPORT.md。
- framework/DRA_OFFICIAL_FIDELITY_AUDIT.md的OFFICIAL_ALGORITHM_FIDELITY_PASS只针对固定Source/Data I/O移植与8/16，不代表Full或原论文性能复现。
- 无Full结果、无后台任务，不引用旧SGA White-box冒充本方法。

本阶段代码/报告发布Commit：1fac414e2edf605d44194f949d301fc12e236582；远端分支codex/tbps-vlpt-official-complete，后续文档提交登记。


## G-v2 Epoch11→60扩展验证（2026-10-20）

父G10 SHA f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e。
唯一GPU Smoke与独立CPU审计PASS；仅固定Train cuhk:211:0的K4+Joint共5次更新，
另1条内部Validation无更新；不属于正式Epoch11或新Full攻击结果。
运行证据：/home/lzf/ldx/outputs/tbps-framework-gv2-epoch60-ablation/resume_smoke_v1/；
protocol、initial_restore、probe_state、probe_result、completed和cpu_audit均不可变。
源码及SHA索引见[续训验证报告](framework/GV2_EPOCH_EXTENSION_VALIDATION_REPORT.md)
与[证据清单](framework/GV2_EPOCH_EXTENSION_EVIDENCE.json)。
原G10、H7及G/H/B3/SGA/DRA Full资产只读保护；后续正式50轮和三模型消融PENDING。

## G-v2 Epoch11→60正式训练启动快照（2026-10-20）

训练635471b，管理14f2833be06c75b62e08473886c5c81268e53fd3；父G10 SHA f59bd3b5…658e。
监督器PID/PGID/SID454418，PPID1/无TTY；已真实Epoch11更新，核验时550样本、Adam82856。
CPU Preflight通过，父G/Adam/RNG exact；正式完整训练及CPU终审PENDING，不是新攻击Full结果。
[启动证据](framework/GV2_EPOCH60_TRAIN_STARTUP.md)。当前状态以PROJECT_STATE和不可变完成回执为准。

## G-v2 Epoch11→60正式训练完成

50个新增Epoch与原生独立CPU审计PASS，最终Adam484080；六点Checkpoint及Loss曲线见[训练报告](framework/results/gv2_epoch60_training/REPORT.md)。没有新Gallery生成或三模型评估。


发布状态补充：原监督器在Git暂存空白检查阶段FAIL（CSV CRLF、SVG尾空格），训练和既有CPU审计均已PASS。
后续只规范派生报告格式，数值/图形不变，单独提交推送；原FAIL日志保持，不改写成原运行SUCCESS。
见结果目录PUBLICATION_REPAIR.json；没有重新训练、重跑CPU审计或进行模型评估。

正式训练报告与格式修复内容Commit：e036c1eb29339441e30b643ed1c4a77a1d4590e6（codex/gv2-epoch60-launch）。

## G60 × H1–H10内部Validation探索性诊断

基准be30206f，G60 SHA9148a6ac…60949、原IRRA SHA9ff62095…009。
固定原128内部Validation及H1–H10历史缓存，G10指标只读对照，H7 Selection保持。
独立输出tbps-framework-g60-h-validation；CPU来源/回归已通过，实际GPU/最终审计以新报告为准。
不属于新Test结果或正式H候选Selection，协议见framework/G60_H_VALIDATION.md。

### 本轮实际执行完成（2026-10-21）

冻结执行048fafc15040f90f7152f78a3b4d48728bd300ac，一次validation_v1：
G60/IRRA图像128/128，H/文本/优化0；前2图Smoke保留、完整新审计PASS。
G60 Image-only mAP48.80861282348633，Joint H9最低40.425479888916016；H7为41.59333801269531。
原正式H7 Selection不变。输出/home/lzf/ldx/outputs/tbps-framework-g60-h-validation/{validation_v1,audit_v1}/。
完整来源/四指标/排名/限制见framework/results/g60_h1_h10_validation/REPORT.md；不是Test或跨模型新结果。

## H10→H60的G60适配续训准备

基准807fdc6，独立分支codex/hv1-g60-adaptation-epoch60。
父H10 SHA4ae964d5…4bacd，G60 SHA9148a6ac…60949；同一IRRA、原2048 Train/128内部Validation。
CPU完整H/Adam/RNG恢复已验证；真实图库/Smoke/正式训练以新输出tbps-framework-hv1-g60-adaptation-epoch60回执为准。
这不是全部60 Epoch从零G60训练，不覆盖旧Selection/Checkpoint/实验产物。

### H适配前置运行实际状态：正式训练未启动

9765a53执行G60 Train图库2048图与新图库CPU审计PASS；独立4-step真实H Smoke PASS。
新独立Smoke CPU审计因局部stat命名遮蔽失败，现场保留，没有GPU重试或正式H11–H60。
源码/原配置/Gallery/Smoke均保持原身份；详见framework/HV1_G60_PRETRAIN_GATE_FAILURE.md。
不能把临时cursor20484、Adam20374冒充正式Epoch11或H60训练完成。

## H60独立CPU审计修复及正式适配训练启动（2026-10-22）

修复执行135df869；原执行9765a536/1b16d653失败证据保持。
73个不同CPU用例与真实既有4-step Smoke独立CPU审计PASS；
151378历史stat、113旧producer SHA、14358份图库/Smoke文件不变。
正式监督器3410067（PPID1、PGID/SID同PID、无TTY），10:43:16进入H11–60。
H10完整20480/20370恢复exact，已见Epoch11真实更新与首不可变Checkpoint。
2048图库只读复用，无新的G/图像编码或Smoke。最终训练及六节点新结果PENDING。
来源/日志/进度见framework/HV1_G60_AUDIT_REPAIR.md及results/hv1_g60_audit_repair_startup/EVIDENCE.json。

## H60适配训练完成及CPU事后科研分析（2026-10-22）

原执行9765a536/修复135df869/启动cb51cd3；H60训练与版本化独立CPU审计PASS，原报告发布FAIL保留。
本轮仅读取：历史20480+新增102400 visits，400新/81旧Checkpoint SHA和父子元数据；
15个固定G60候选、21份既存128×128分数，2688条逐Query记录及PID聚类Bootstrap。
H20新增候选最低40.427635%，历史H9为40.425480%；H60为42.874081%，不发布新Selection。
奖励/梯度/动作全50轮统计、12图PNG/SVG及中文研究结论位于framework/results/hv1_g60_posthoc_analysis/。
10项CPU统计测试和交付检查PASS。原151378历史文件与18617实验/控制/失败现场元数据不变，113 producer SHA不变。
没有GPU、训练、神经编码、Full排名或原CPU审计重跑。历史三Victim Full仅作固定H7背景。
此条不把原监督器FAIL改写为SUCCESS，不授权任何后续训练或Test选模。


## G/H属性与K只读机制审计（非新训练/攻击实验）

基准23fd02da；codex/gh-attribute-alignment-audit。
范围：原2048/128、H9/H20/H60既有动作；26 CPU tests，严格同集排名/预算分解及512次纯规则转换。
保护151378历史stat、18617相关产物和冻结来源；没有GPU/新攻击/新排名或Selection。
原H7/H60及G60结果仍以各原完成/审计回执为准，本次不重发旧回执。
[报告](framework/results/gh_attribute_alignment_audit/REPORT.md)、
[证据](framework/results/gh_attribute_alignment_audit/SOURCE_EVIDENCE.json)、
[后续方案草案](framework/results/gh_attribute_alignment_audit/alternating_training_design.md)。

## G/H双向条件交替原型：仅CPU准备

新身份：gh-alternating-prototype-1，codex/gh-bidirectional-prototype。
只读父G60 SHA9148a6ac…60949与H10 SHA4ae964d5…4bacd；H60及H7 Selection保持。
固定8 Train更新样本、原2048奖励图库、两轮A1/B；不含新Validation/Test结果。
模型、Adam、RNG的CPU合成恢复和来源门禁通过不等于真实GPU Smoke通过。
真实G/H/IRRA加载、JVP和更新数本轮均0；计划输出smoke_v1尚未创建。
[设计](framework/results/gh_bidirectional_prototype/REPORT.md)、
[实际CPU/资源/保护证据](framework/results/gh_bidirectional_prototype/SOURCE_EVIDENCE.json)。
不重发任何历史完成回执，不自动启动后续训练。
