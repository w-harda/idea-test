# 项目当前状态（唯一权威入口）

当前状态来自Git、源码及不可变证据；历史报告是当时快照。架构、实验与决策见ARCHITECTURE、EXPERIMENT_REGISTRY及ADR。

## 当前任务：物化IRRA共享GPU 10+10 Smoke已通过

现有独立分支 `codex/tbps-irra-materialized-evaluation`，工作树 `/home/lzf/ldx/projects/idea-test-tbps-irra-materialized-eval-20261017`；本轮基准 `01dd2ac199b6357473959e63012ef2e265dbd325`。原评估实现Commit为bc710e5baf807bb8347a51c6071d03b43b2c0245；共享修订实现Commit：a6875e53c7581cf747e02eda1882259b7df7de6f；本后续文档提交登记。

用户授权只修改本评估入口为shared_observe_only，取消独占及固定8192MiB门槛，记录GPU状态、保留正常OOM与完整堆栈，不干预Ollama或自动重试。Stage4D1及G/H源码、权重、Plan、数据、FP32/math/no_tf32、batch1和指标均不变。

必要CPU回归102项通过（43.42秒）。唯一真实Smoke退出0，MATERIALIZED_IRRA_EVALUATION_PASS smoke；原输出此前不存在，本次仅首次attempt-000001。10 Gallery/10 Query，IRRA恢复1、图像样本前向20、文本样本前向20，G/H及Backward/Optimizer/JVP均0。参数/buffers前后exact，四类特征及四组指标真实生成。

显存峰值allocated922.37890625MiB、reserved1002MiB，Peak RSS约2.239GiB；执行段36.317618秒、含来源预检完整进程121.658754秒。退出后空闲3865MiB，仅原Ollama占用20178MiB。当前余量支持相同batch1流式Full的资源预期，但后续外部占用变化仍可能OOM。没有启动Full或额外GPU测试。

## 当前产物与验证边界

- generation_v3 Joint真实攻击数据：3074图/6156文，同一Plan7ac4fefd0abaa10df085be3ec3ed56c4b38c782a21bfc44473f40b90965447d5；6116编辑/40 NOOP/13537替换。原图文产物仅只读消费。
- 新入口Smoke四组mAP：Clean100.0、Image-only93.0、Text-only99.16666412353516、Joint90.95238494873047。仅小样本链路结果，不能代替正式Full。
- Smoke证据：/home/lzf/ldx/outputs/tbps-framework-irra-materialized-evaluation/smoke/；完成摘要SHA d9499227e2f785b96af640f8b9bf6ec88285b0c5aac6f2664a5c68c827067f28。
- 日志：/home/lzf/ldx/tmp/irra_materialized_smoke_shared.log。详情见[共享Smoke报告](framework/IRRA_MATERIALIZED_SHARED_GPU_SMOKE.md)。
- Stage4D生成/审计26文件SHA/mtime、61555物化文件成员/size/mtime及两冻结生产树clean状态不变。
- 新入口正式Full仍PENDING，full目录不存在；未运行独立Smoke CPU审计CLI、历史Full重审、RaSa/APTM或训练。

后续用户可按[评估说明](framework/IRRA_MATERIALIZED_EVALUATION.md)手动执行Full及完成后CPU只读审计。本轮完成即停止，无后台监控、自动恢复或正式Full执行。

## 历史索引

- generation_v2：原FAIL结果Manifest SHA9b2dae20980ce9e1943fcaada80e2d73539fd09fb9ea3d4ada106e077d79efa9；gpu-result/report SHA176ae52ffc1851a3d710c445268569a987baaed912d95696941ec7e73c052d13。
- 原首图Claim Manifest SHA21ed14c935fdde887cfc6afcbdb67b6e6c6687085f6b8a46e8c3aed5ad5ea1e0，unit_id50b19770bb01c5c7e52237a4af73e4186342e77ccafe06204a8c3c91ecbfe3b7。
- Stage4D2R：dd0a213f2ac1b787412b755372c378ad49f31f9c，启动顺序CPU修复/626测试通过，报告是当时v2尚未执行的历史快照。
- generation_v1：整体FAIL、G前向0，原报告STAGE4D2_EXECUTION_REPORT.md及旧原树未提交文档保持。
- Stage4D1：82fe88a5a74955838fc6341c1ed9a46921c94d7c，6156历史文本物化/独立审计通过。
- Stage4C3：a5a6f6ae8c9200482ae0c70d228e1873d3ca4dd2，固定两图两文IRRA物化端到端等价PASS；不代表Full或跨模型。
