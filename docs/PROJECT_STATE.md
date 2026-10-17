# 项目当前状态（唯一权威入口）

状态以Git、实际代码和不可变证据为准。历史README/报告为阶段快照；架构与实验来源见ARCHITECTURE、EXPERIMENT_REGISTRY及ADR。

## 当前阶段：Stage4D-2 CPU开发完成，GPU资源阻断

**尚未执行3074图真实G生成，Image/Joint全量PENDING。** 资源预检最终退出1：GPU has existing compute processes; safe stop。未初始化CUDA、未恢复G/H/IRRA、未创建正式generation_v1，GPU Attempt为0。没有等待资源后自动重试、抢占、终止外部进程或降低门禁。

- 基准82fe88a5a74955838fc6341c1ed9a46921c94d7c。
- 独立分支codex/tbps-framework-stage4d2。
- 工作树/home/lzf/ldx/projects/idea-test-tbps-framework-stage4d2-20261017。
- 实现Commit由本轮代码提交后的文档提交登记。
- [阶段报告](framework/STAGE4D2_REPORT.md)、[实际命令与停止策略](framework/STAGE4D2.md)。

检查时RTX4090空闲3879MiB/已用20212MiB，已有PID2209695占用20164MiB；GPU利用率0%不代表可抢占。固定启动条件为无外部计算进程且空闲至少8192MiB。本轮不再查询等待GPU释放。

## 已完成代码与CPU验证

新增full_gpu.py、gpu_safety.py、full_gpu_audit.py及配置/20项CPU tests。生产必须显式从冻结Stage4D1库导入stream_images、Store和原G插件，核对实际模块路径、旧Git身份与原Plan；新入口源码另行SHA绑定，不改变旧Plan/文本产物。

执行链设计为CPU预检→独立一次G GPU子进程→GPU退出后CPU H再核验/Joint引用封存→独立CPU验收。真实G根模块Hook、单次权重加载、逐图日志及Claim/Proof独立计数；旧两图SHA不一致停止并保留证据；只管理本入口创建的子进程。7200秒预算与系统可用RAM/磁盘监测，无2GiB进程RSS硬上限。

最终611 CPU tests通过，52.47秒；新增20项覆盖路径隔离、单次加载/实际前向、顺序/预算/Claim/Proof、历史像素异常停止、资源/中断与只读保护。CPU Fixture不是全量GPU路径已验证。

## 独立CPU来源与保护证据

/home/lzf/ldx/outputs/tbps-framework-stage4d2/cpu_source_verified_v1/result/：

- status=CPU_SOURCES_AND_HISTORY_VERIFIED_GPU_RESOURCE_BLOCKED。
- Manifest SHA5e5d7ac9a22992fbb1f0559d85acb390596a111a56e1293286459d7adb3984e1。
- report SHA55870922aebc9d02ebd4190c232ce27ef911e8a0ac2574b93fd8945f991e1729。
- 原Stage4D1 30797文件快照digest ed2c18d9e64715419d488ea0ac764586486e719aa9e39b4c38e743e5740dd163，SHA/mtime不变。
- 6156 H文本及完整动作/顺序：6116编辑、40 NOOP、13537替换，动作摘要与C1一致；没有新H推理。
- 严格闭包PASS，75来源/33旧执行文件；3074原图SHA及原官方顺序通过。
- 7991历史文件、3472Joint缓存、70原FAIL文件保持；Stage4D1冻结库HEAD/clean和原八保护工作树保持。
- 原Stage4C2两图RGB/Adv/effective_delta及生成证明只读有效；本轮新生成像素对照PENDING。
- G/H/IRRA加载、前向、GPU、Backward、Optimizer、JVP均0。

原预检失败日志/home/lzf/ldx/tmp/stage4d2_cpu_preflight_v1.log，SHA b1f9cacd38a46b83ee1a315a26e5b22d25019b8effa355b00e69f6f3d289185e。CPU来源保护日志stage4d2_cpu_source_protection.log、全套测试stage4d2_final_cpu_tests.log均在/home/lzf/ldx/tmp/，不入Git。

## 原已完成资产保持

Stage4D1 H文本仍TEXT_FULL_MATERIALIZED及TEXT_FULL_CPU_AUDIT_PASS。固定Plan7ac4fefd0abaa10df085be3ec3ed56c4b38c782a21bfc44473f40b90965447d5；原Store/home/lzf/ldx/outputs/tbps-framework-stage4d1/materialized_full_v1/保持，未追加图像或Joint。

旧result Manifest a149a57fc7e8c87a389ede80c7e2eaedafec9d3ffc81dff9a7275915040c615f；text_collection SHA9ab8881dab9f82bd31cc31c0a7cc478e3beca9b2f9b940443603efa54d21069b；原独立CPU审计Manifest913b87ecf0620730e537a33f96858e4dff41fbb6de8fa233b085da077fbbc9a2。

G10/H7/IRRA权重、Stage3B FAIL、Stage3B2 PASS、Stage4B2指标重放、Stage4C2整体FAIL及Stage4C3两图两文PASS均不改。本阶段未完成任何新的攻击效果评测。

## 下一步与权限

资源就绪后由用户明确指示继续，使用同一入口重新完成全部预检。正式Attempt未消费，无需删除/转换任何缓存、Claim或回执。不得自动等待、重试或后台启动。

真实G全量生成、旧两图新输出对照、Image/Joint封存和独立CPU全量验收仍待执行；IRRA Full、RaSa/APTM、训练均未授权。当前代码提交推送后停止等待审查。

## 既往阶段索引

- Stage4D1：82fe88a5a74955838fc6341c1ed9a46921c94d7c，[完整H物化/CPU报告](framework/STAGE4D1_REPORT.md)。
- Stage4C3：a5a6f6ae8c9200482ae0c70d228e1873d3ca4dd2，[固定两图两文IRRA端到端PASS](framework/STAGE4C3_REPORT.md)。
- Stage4C2R：9f1f943ed8486a8c2889ab272764e5707a431552，[CPU修复](framework/STAGE4C2R_REPORT.md)。
- Stage4C2：fcf097d3ab31785381947289a680e5f5ccbf9d42，[原整体FAIL/局部G-H PASS](framework/STAGE4C2_REPORT.md)。
- Stage4B2：c09754e6afc8cf53a6c0c6e8715b6d4eace80946，METRICS_REPLAY_VERIFIED仅旧IRRA特征重放。
