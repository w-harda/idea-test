# 历史实验注册表

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
