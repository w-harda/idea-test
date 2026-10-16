# 架构边界与迁移约定

## 当前结构

| 路径 | 职责及保护 |
| --- | --- |
| src/attributes、scripts、ontology | 旧属性/AP/TTA/IRRA/数据工具；仍有间接依赖，不删除 |
| prototypes/attribute_image_generator_20261014 | G-v1条件生成基线，全目录冻结 |
| prototypes/attribute_image_generator_v2_20261014 | G-v2 image-only、K+1训练、Epoch恢复、IRRA Full；全部冻结 |
| prototypes/attribute_text_policy_h_v1_20261015 | H/SCST/正式训练/Selection/Joint/消融；全部冻结 |
| 其他prototypes及_frozen_sources | reader/Canonical/JVP/选择及历史研究闭包；UNKNOWN也保留 |
| src/tbps | 新框架稳定职责模块，Stage 1契约/配置/存储/只读桥；Stage 2官方Dataset/IRRA Adapter |
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
