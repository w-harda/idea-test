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
