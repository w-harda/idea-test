# 项目当前状态（唯一权威入口）

当前状态来自Git、源码及不可变证据；历史阶段报告为当时快照。架构、实验与决策见ARCHITECTURE、EXPERIMENT_REGISTRY及ADR。

## 当前任务：generation_v3固定Ollama身份的GPU共享许可（CPU-only）

基准029a0817f764b8241ca1790bbf335fdd144ed9a9；独立codex/tbps-generation-v3-gpu-sharing，工作树/home/lzf/ldx/projects/idea-test-tbps-generation-v3-sharing-20261017。代码Commit在后续文档提交登记。

仅当前full_gpu/gpu_safety及独立树配置调整。启动空闲显存3584MiB、利用率≤10%；共享仅固定Ollama PID2436556，绑定boot、start_ticks、UID/GID、完整命令行SHA、cgroup、父进程和二进制SHA。运行每秒监测，仅允许该Ollama与当前GPU子进程，空闲低于1024MiB或身份/来源/未知进程异常即停止，无自动重试。

原Stage4D1源码/Plan/6156文本、G10/H7、8/255、图像预处理及FirstClaimRecovery代码保持不变。原v1/v2 FAIL及首图旧Claim永久保留；generation_v3尚未创建，不另建v4。完整CPU来源/历史保护预检仍在用户正式入口运行时执行，本轮不运行该入口或GPU。

CPU回归687项通过/88.78秒；最终共享安全停止补充40项复核通过/2.54秒，原定向57项通过。原R2树580文件、v1/v2和Claim35文件SHA/mtime相同；原Store30800文件成员/size/mtime相同，冻结Stage4D1 HEAD/clean不变。本轮不重复整个Store内容SHA或7991/3472完整历史审计，引用原R2证据。

证据/home/lzf/ldx/outputs/tbps-generation-v3-sharing-cpu/。详细身份、来源SHA、边界及用户nohup/预检/审计命令见[共享执行说明](framework/GENERATION_V3_GPU_SHARING.md)。

## 当前真实产物与授权

H历史Text6156保持TEXT_FULL_MATERIALIZED（6116编辑/40 NOOP/13537替换）；Image0/3074、Joint全量PENDING，旧首图Claim尚未实际恢复。

本轮真实模型恢复/前向、CUDA初始化、GPU Attempt及新图像均0。只读一次资源查询与CPU Mock测试不等于GPU执行通过；没有轮询或等待资源。实际GPU共享稳定性待用户手动唯一generation_v3验证。

下一步用户审查后自行运行；PID退出/重启、共享身份或显存不满足即拒绝，不自动替换PID或降低门限。禁止自动生成、IRRA Full、RaSa/APTM或训练。

## 历史索引

- generation_v2：原FAIL结果Manifest SHA9b2dae20980ce9e1943fcaada80e2d73539fd09fb9ea3d4ada106e077d79efa9；gpu-result/report SHA176ae52ffc1851a3d710c445268569a987baaed912d95696941ec7e73c052d13。
- 原首图Claim Manifest SHA21ed14c935fdde887cfc6afcbdb67b6e6c6687085f6b8a46e8c3aed5ad5ea1e0，unit_id50b19770bb01c5c7e52237a4af73e4186342e77ccafe06204a8c3c91ecbfe3b7。
- Stage4D2R：dd0a213f2ac1b787412b755372c378ad49f31f9c，启动顺序CPU修复/626测试通过，报告是当时v2尚未执行的历史快照。
- generation_v1：整体FAIL、G前向0，原报告STAGE4D2_EXECUTION_REPORT.md及旧原树未提交文档保持。
- Stage4D1：82fe88a5a74955838fc6341c1ed9a46921c94d7c，6156历史文本物化/独立审计通过。
- Stage4C3：a5a6f6ae8c9200482ae0c70d228e1873d3ca4dd2，固定两图两文IRRA物化端到端等价PASS；不代表Full或跨模型。
