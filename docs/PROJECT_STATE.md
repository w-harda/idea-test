# 项目当前状态（唯一权威入口）

当前状态来自Git、源码及不可变证据；历史阶段报告仅为当时快照。架构、实验与决策见ARCHITECTURE、EXPERIMENT_REGISTRY及ADR。

## 当前阶段：generation_v2 FAIL后的CPU最小修复

基准dd0a213f2ac1b787412b755372c378ad49f31f9c；独立codex/tbps-framework-stage4d2r2，工作树/home/lzf/ldx/projects/idea-test-tbps-framework-stage4d2r2-20261017。代码实现Commit：cd6c222fc6751e4dbefc44bb7869a9c08d93a680；本后续文档提交登记。

generation_v2已FAIL：G10恢复1次、实际前向started/completed=0/0、新图像0。原_LegacyG像素读取中的torchvision Resize被非G模块保护误拒绝。原Store有唯一首图Claim、Proof0、pending0；两次generation_v1/v2 FAIL永久保留。

## 本轮实现

- GObservation仅在经过来源/图像身份核验的原像素读取作用域中允许必要Resize，维持原RGB、bilinear384×128、ToTensor。作用域外Resize、其他网络、错误读取器/来源/图像及异常参数继续拒绝；G Hook和优化禁令保留。
- FirstClaimRecovery绑定generation_v2完整FAIL及来源、零实际前向、唯一cuhk:37132:0 Claim、原30797文件快照＋3 Claim文件。其他Claim、Proof、图像、pending或未知新增/篡改拒绝。
- CPU预检、fresh_images、原generate_image_once局部恢复和冻结stream_images的调度衔接全部覆盖。旧Claim不删除或覆盖，新Attempt额外绑定恢复Start/Proof；原正常生成Proof不改。
- 新入口固定generation_v3；独立配置绑定4项执行源码、恢复证据与3项预处理来源。原Stage4D1源码/Plan/6156文本、G10/H7、预算和全部攻击算法不改。
- 已完成647 CPU tests，84.36秒、退出0：原626项＋新增21项。含实际child/门禁/真实Resize-ToTensor/CPU Mock G/首图事务及3074顺序调度。
- 真实CPU来源/恢复资格审计PASS：75来源、33导入文件；30797原发布文件＋3 Claim、7991历史文件、3472缓存、70旧FAIL及v1/v2各16文件SHA/mtime保持。
- 证据/home/lzf/ldx/outputs/tbps-framework-stage4d2r2/cpu_audit_v2/result/，Manifest SHA73d895b065cac4d3abe9316478811fa5a6cc1fa6eb9414b2f8532668d54e61fc，report SHA030bbdedf5f51becdbaec613219a556a923f3442ee061643da03deca41e4ca7c。
- 首次CPU核验拒绝新清单缺path字段；补正新清单并新增拒绝测试后第二次通过，失败日志保留。本次不查询GPU、不运行正式CPU预检入口或GPU Attempt。

[执行说明与未来命令](framework/STAGE4D2R2.md)；[完整修复与证据报告](framework/STAGE4D2R2_REPORT.md)。

## 当前真实产物与授权

原H Text 6156/6156保持TEXT_FULL_MATERIALIZED（6116编辑/40 NOOP/13537替换）。Image0/3074、Joint全量PENDING。原首图Claim的3文件保留，尚未实际恢复生成。

本轮真实G/H/IRRA加载/前向、CUDA初始化、GPU Attempt和新图像均0；generation_v3尚未创建。CPU资格验证不等于FIRST_CLAIM_RECOVERED或Image/Joint完成。

下一步仅在用户审查并另行授权后，通过原资源门限和完整CPU预检启动唯一generation_v3。无需人工清理旧Claim或旧文本；失败保存现场，不自动重试。禁止自动IRRA Full、RaSa/APTM、训练或其他阶段。

## 历史索引

- generation_v2：原FAIL结果Manifest SHA9b2dae20980ce9e1943fcaada80e2d73539fd09fb9ea3d4ada106e077d79efa9；gpu-result/report SHA176ae52ffc1851a3d710c445268569a987baaed912d95696941ec7e73c052d13。
- 原首图Claim Manifest SHA21ed14c935fdde887cfc6afcbdb67b6e6c6687085f6b8a46e8c3aed5ad5ea1e0，unit_id50b19770bb01c5c7e52237a4af73e4186342e77ccafe06204a8c3c91ecbfe3b7。
- Stage4D2R：dd0a213f2ac1b787412b755372c378ad49f31f9c，启动顺序CPU修复/626测试通过，报告是当时v2尚未执行的历史快照。
- generation_v1：整体FAIL、G前向0，原报告STAGE4D2_EXECUTION_REPORT.md及旧原树未提交文档保持。
- Stage4D1：82fe88a5a74955838fc6341c1ed9a46921c94d7c，6156历史文本物化/独立审计通过。
- Stage4C3：a5a6f6ae8c9200482ae0c70d228e1873d3ca4dd2，固定两图两文IRRA物化端到端等价PASS；不代表Full或跨模型。
