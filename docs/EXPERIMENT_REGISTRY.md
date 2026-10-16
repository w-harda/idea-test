# 历史实验注册表

这里只索引已有不可变记录，不启动实验。发布/实验树别名：
- RELEASE=/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014
- OLD=/home/lzf/ldx/projects/idea-TBPS-test1
- G2=RELEASE/prototypes/attribute_image_generator_v2_20261014
- H1=RELEASE/prototypes/attribute_text_policy_h_v1_20261015

| 实验 | 不可变证据 | 状态/范围 |
| --- | --- | --- |
| G-v2 Epoch10 | G2/outputs/train_v2/training_protocol.json；checkpoints/epoch_0010/checkpoint.pt/.json | 原10 Epoch完成；本阶段只读sidecar/协议/文件SHA，不恢复模型 |
| H-v1 Epoch7 Selection | H1/outputs/formal_v1/selections/30f573098d28b6af6afe146330823986f724378fc7bd4df9d257fdd2c6f20dd1.json | 独立Validation选模，非Test；visit14336、updates14259 |
| H-v1选中Checkpoint | H1/outputs/formal_v1/training/checkpoints/visit_00014336/checkpoint.pt/.json；training/protocol.json | 不继承32Step权重；Checkpoint/Adam/RNG由历史任务保存 |
| G-v2 Image-only Full | G2/outputs/evaluation_phase29a/full/{protocol,summary,completed,clean_verified}.json及metrics.csv | 官方IRRA/CUHK 3074/6156；含限定Storage兼容证据 |
| Joint Full | H1/outputs/phase5_joint/full/{protocol,summary,completed,baseline_verified}.json及metrics.csv | 原Full及独立CPU审计已由用户完成；文本/图库四组 |
| Joint Ablation Full | H1/outputs/joint_ablation_v1/full/{protocol,summary,completed}.json、metrics.csv、aggregate.csv及variants | completed.method_audit_passed=true；用户确认两项成功标志，待本阶段绑定日志 |
| AP B1/B2/B3 Full | /home/lzf/ldx/outputs/idea-TBPS-test1/baseline-ap-{duke-irra,cuhk-irra,cuhk-irra-surrogate}/evaluations/full/{summary,metadata}.json | 独立Baseline；Stage 1仅metadata级来源索引，不宣称全缓存新审计 |
| G-v1及属性/TTA历史 | OLD/prototypes/attribute_image_generator_20261014/outputs及各旧prototype | 保留历史基线；本阶段不迁移、不重验全部资产 |

核心SHA：
- G-v2 E10：f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e
- H选中权重：487f3061b6cba2ddabccdd49745a7b99c8c549f4541781b8814b299ef5f279df
- IRRA：9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009
- Selection文件：584e4ba2afc7b17fea3bdfb8501716498b84731fa216558daa2e8e4bb353f757

开发Commit索引：G2完整训练2a3f166；G2 Storage修复4e40c1a；H正式系统a668b1c；Joint d501f11；消融40c7476。指标/当前证据以原文件为准，不依据阶段README的旧CURRENT。

新桥必须区分元数据/直接证据已核验与全缓存/训练轨迹未重验。缺失路径返回NOT_READY，SHA/schema/来源冲突拒绝，不编造通过。
