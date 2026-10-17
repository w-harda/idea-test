# 项目当前状态（唯一权威入口）

当前状态来自Git、源码及不可变证据；历史报告是当时快照。架构、实验与决策见ARCHITECTURE、EXPERIMENT_REGISTRY及ADR。

## 当前任务：generation_v3物化输入的官方IRRA评估准备（CPU-only）

基准0f5b8838b9ef3d0032f656e5169c74054ddd78cf；独立codex/tbps-irra-materialized-evaluation，工作树/home/lzf/ldx/projects/idea-test-tbps-irra-materialized-eval-20261017。实现Commit：bc710e5baf807bb8347a51c6071d03b43b2c0245；本后续文档提交登记。

Stage4D generation_v3已由用户完成GPU生成/Joint封存及独立CPU审计，实际文件确认JOINT_FULL_MATERIALIZED和JOINT_FULL_CPU_AUDIT_PASS。3074图/6156文，同一Plan7ac4fefd...；G恢复1、前向3074，最大L∞0.0313725471496582；6116编辑/40 NOOP/13537替换。原v1/v2 FAIL和首图Claim保持，不改签旧文本或结果。

本轮只新增官方IRRA物化评估入口、紧凑特征缓存/只读评分薄层、配置及测试。复用冻结Stage4D1 Plan/Store/Unicode校验、官方Dataset Adapter、IRRAAdapter、EvaluationEngine及原Ranking；不调用G/H、不改变攻击或指标。默认CPU预检，真实GPU需显式参数。

新增25项最终CPU测试通过；新树704项回归通过，2项绝对生产路径来源检查在原生成工作树通过，未改签旧配置。真实CPU预检退出0：官方3074/6156身份、全部Unicode与两个RGB读回通过，CUDA和真实模型加载0。原生成/审计26文件SHA/mtime、61555物化文件成员/size/mtime保持，冻结生产树clean；新正式输出仍未创建。完整保护结果及命令见[评估说明](framework/IRRA_MATERIALIZED_EVALUATION.md)。

## 当前产物与验证边界

- Joint真实攻击数据已就绪；当前新IRRA四组评估仍PENDING，没有新GPU编码或正式mAP。
- CPU Mock执行四组完整接线，10x10合成Score与独立官方rank指标exact；不能作为真实攻击效果。
- generation_v3/result Manifest fb696a1c0c3235018202118a2d945fa728f6fc06be39738f04a26e6a39fe7634。
- 独立Stage4D审计Manifest 5a693b3877a35c6778e79de7c3f9e555f7772191bbd2e3af5624fcef268bdce5。
- 新正式评估输出预定/home/lzf/ldx/outputs/tbps-framework-irra-materialized-evaluation/，本轮未创建；CPU开发证据独立于此目录。

用户下一步可先授权/手动10+10 Smoke，再正式Full四组；本轮不执行GPU、不监控、不自动恢复、不重训、不进入RaSa/APTM。IRRA入口仍采用原独占≥8192MiB门禁，G阶段专属共享许可没有自动扩展。

## 历史索引

- generation_v2：原FAIL结果Manifest SHA9b2dae20980ce9e1943fcaada80e2d73539fd09fb9ea3d4ada106e077d79efa9；gpu-result/report SHA176ae52ffc1851a3d710c445268569a987baaed912d95696941ec7e73c052d13。
- 原首图Claim Manifest SHA21ed14c935fdde887cfc6afcbdb67b6e6c6687085f6b8a46e8c3aed5ad5ea1e0，unit_id50b19770bb01c5c7e52237a4af73e4186342e77ccafe06204a8c3c91ecbfe3b7。
- Stage4D2R：dd0a213f2ac1b787412b755372c378ad49f31f9c，启动顺序CPU修复/626测试通过，报告是当时v2尚未执行的历史快照。
- generation_v1：整体FAIL、G前向0，原报告STAGE4D2_EXECUTION_REPORT.md及旧原树未提交文档保持。
- Stage4D1：82fe88a5a74955838fc6341c1ed9a46921c94d7c，6156历史文本物化/独立审计通过。
- Stage4C3：a5a6f6ae8c9200482ae0c70d228e1873d3ca4dd2，固定两图两文IRRA物化端到端等价PASS；不代表Full或跨模型。
