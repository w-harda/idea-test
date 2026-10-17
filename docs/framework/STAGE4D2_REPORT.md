# Stage4D-2：G全量入口开发与GPU资源阻断报告

基准82fe88a5a74955838fc6341c1ed9a46921c94d7c；独立分支codex/tbps-framework-stage4d2，工作树/home/lzf/ldx/projects/idea-test-tbps-framework-stage4d2-20261017。实现Commit由后续文档提交在PROJECT_STATE登记。

**本轮没有执行3074图GPU生成。** 严格预检在GPU资源门禁拒绝：GPU has existing compute processes; safe stop。正式generation_v1未创建，GPU Attempt、G/H/IRRA加载/前向均0；Image/Joint全量PENDING。没有重试GPU检查、抢占或终止外部进程，也没有降低资源条件。其后仅完成CPU测试、来源保护和文档。

## 一次性资源检查与停止原因

开始开发时RTX4090空闲24051MiB/无计算进程；完整CPU预检期间出现其他计算任务。预检最终退出1。随后一次只读诊断取得：

| 项目 | 实际读取 |
| --- | ---: |
| GPU | NVIDIA GeForce RTX 4090 |
| 空闲/已用显存MiB | 3879 / 20212 |
| GPU利用率 | 0% |
| 已有计算PID | 2209695 |
| 该进程显存MiB | 20164 |

即使GPU利用率为0，已有计算进程及空闲显存不足8192MiB仍触发拒绝，不能推断可抢占。未再次查询等待资源释放，不自动重试或创建新Attempt。原失败预检日志/home/lzf/ldx/tmp/stage4d2_cpu_preflight_v1.log保留。

CPU源码与资产检查不依赖GPU，可在资源阻断后继续完成；该独立证据明确区分CPU_SOURCES_AND_HISTORY_VERIFIED_GPU_RESOURCE_BLOCKED与完整GPU预检PASS，不伪造GPU许可。

## 新增文件与真实复用

| 文件 | 职责 |
| --- | --- |
| src/tbps/materialization/full_gpu.py | 默认CPU预检、冻结库路径门禁、独立一次GPU子进程与退出后CPU Joint封存 |
| src/tbps/materialization/gpu_safety.py | 系统RAM/时间/磁盘监督、实际G Hook/顺序、单次权重加载、非G模型/优化阻断 |
| src/tbps/materialization/full_gpu_audit.py | GPU结束后的独立只读像素/Claim/Proof/日志/文本/来源验收；目前未运行完整验收 |
| configs/framework/stage4d2_generation.server.json | 固定Plan/3074+6156、batch1、FP32/seed42、7200秒、资源与新入口源码SHA |
| tests/framework/test_stage4d2_gpu_full.py | 20项CPU Fixture/控制流程Mock |
| docs/framework/STAGE4D2.md | 可复制服务器命令、停止/失败策略与后续消费接口 |

Stage4D1的所有源码、config、6156文本单位、原Collection及回执完全不修改。生产入口作为绝对脚本运行，PYTHONPATH显式指向冻结Stage4D1/src；检查artifacts、full_prepare、full_runner、legacy_runtime实际文件路径及原Git HEAD/clean。

使用冻结Stage4D1 prepare()构造原计划，ID保持7ac4fefd0abaa10df085be3ec3ed56c4b38c782a21bfc44473f40b90965447d5。新入口/监督器/审计文件单独SHA绑定，不向旧Plan新增路径，不复制/改签H单位。只有通过全部门禁后才允许调用冻结stream_images()及原GV2Plugin/load_legacy_plugins；没有复制网络、投影或字符替换算法。

## CPU验证与来源保护

最终611 CPU tests通过，52.47秒。独立CPU来源保护退出0，状态CPU_SOURCES_AND_HISTORY_VERIFIED_GPU_RESOURCE_BLOCKED；它不等于完整资源预检或GPU Attempt通过。

CPU来源核验时间2026-10-17T09:00:09.970614+00:00至09:02:35.734622+00:00；75项固定来源/33旧执行文件、3074原图及顺序、6156 H字符串/动作/顺序均通过。原Stage4D1 30797文件SHA/mtime保持，有序文件快照digest ed2c18d9e64715419d488ea0ac764586486e719aa9e39b4c38e743e5740dd163；7991历史文件、3472Joint缓存、70旧FAIL文件不变。原G/H/IRRA、原Plan与完成回执不改。

新证据/home/lzf/ldx/outputs/tbps-framework-stage4d2/cpu_source_verified_v1/result/：
- Manifest SHA5e5d7ac9a22992fbb1f0559d85acb390596a111a56e1293286459d7adb3984e1。
- report SHA55870922aebc9d02ebd4190c232ce27ef911e8a0ac2574b93fd8945f991e1729。
- old_files.json SHAed2c18d9e64715419d488ea0ac764586486e719aa9e39b4c38e743e5740dd163。
- 原资源拒绝日志SHAb1f9cacd38a46b83ee1a315a26e5b22d25019b8effa355b00e69f6f3d289185e。

原两图历史Adv像素SHA为cuhk:37132:0 → 54c594950652698fb62925eee8e50bf87c3505f55b66850bdc332b2572cf4be6，cuhk:37147:0 → bad51abb2f0e46affbb7ec745f9f436b1f63a9fcb61a976c9c439a6caffcaabe；均原linf0.0313725471496582、FP32[1,3,384,128]，读取验证不等于本轮生成对照已完成。本轮初始全套608 tests通过（48.33秒），最后监督器补强后20项新增回归通过（1.97秒）；没有GPU Fixture或真实权重加载。

新增测试覆盖：默认CLI不能启动GPU、误用新树公共库被拒绝、Claim/pending在模型前拒绝、G Hook真实顺序/计数、旧像素SHA不一致停止、只加载一次G且拒绝其他Checkpoint、只允许G子模块前向、禁止Backward/Optimizer/JVP、实际RAM与7200秒门限、RSS超过2GiB不会误拒、外部GPU进程冲突、旧文件只追加不覆盖、独立CPU图像读回/Claim/Proof、日志计数异常拒绝、父进程中断只停止自己创建的子进程而不重试。

Stage4D1 6156条历史H文本复用来源已在CPU预检真实读取；本轮没有重新生成文本。原6116编辑/40 NOOP/13537字符替换、原UTF8与完整action保持。原Stage4C2两图的RGB/Adv/effective_delta、生成证明和原整体FAIL通过只读引用；本轮新生成像素对照仍PENDING。

## GPU执行与封存契约（代码具备，真实执行未验证）

1. 默认仅CPU预检；完成全部来源、原图SHA/顺序、H全文及GPU资源门禁后，才能创建generation_v1一次Attempt。
2. GPU子进程再次核验旧30797文件与Plan/来源、CPU隔离退出，单次gpu-start Claim之后才初始化CUDA。
3. G10 SHA f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e，恢复一次；H7/IRRA不加载。batch1按3074 Gallery顺序调用原G，PIL RGB/bilinear384×128/ToTensor，FP32/math/no_tf32，原8/255投影不改。
4. 每图原生成事务先Claim再G，再保存原RGB/Adv/effective_delta并立即读回。根模型Hook记录实际started/completed，逐图日志绑定文件/内容SHA、预算、耗时和资源。
5. cuhk:37132:0、cuhk:37147:0额外比较原/Adv/effective_delta内容SHA；不一致保存独立失败像素证据并停止，不调权重/精度/预算。
6. 只允许G10权重一次反序列化及已声明pixels.pt读回；禁止其他模型前向和Backward/Optimizer/JVP。模型参数/buffers摘要前后exact才可通过。
7. GPU子进程退出后，CPU再次核验H全文并引用同一图文单位封存Joint；旧TEXT_FULL_MATERIALIZED不改。新IMAGE_FULL_MATERIALIZED/JOINT_FULL_MATERIALIZED只能在真正全量完成后发布。
8. 独立CPU验收对照GPU Hook、全部逐图日志、3074三Tensor预算/内容与文件SHA、Claim/Proof、Gallery/Query顺序、6156 Unicode/action及历史保护。没有真实GPU数据时不能执行或宣称该验收通过。

## 冻结资源与成本边界

起始磁盘≥10GiB、inode≥100000、系统可用RAM≥8GiB、GPU空闲≥8192MiB且无其他计算进程。运行时RAM安全余量2GiB、磁盘安全余量2GiB，记录进程RSS而不设硬上限；父/子进程总时间按7200秒剩余预算管理。父进程中断/超时只停止所创建子进程，保留Claim/pending/日志，不自动重试。

Stage4D1测得的G Mock存储调度量级约25分钟，真实G全量时间、峰值显存/RSS和最终磁盘消耗本轮**尚无实测**。纯三FP32 Tensor约5.066GiB，预算预留10GiB。已有两图G历史显存数据不能替代本轮3074图实测。

## 当前状态与后续

| 项目 | 当前状态 |
| --- | --- |
| G3074真实图像 | 未执行，PENDING |
| H6156历史文本 | 原TEXT_FULL_MATERIALIZED保持；本轮仅只读来源复核 |
| Joint全量 | 未封存，PENDING |
| 本轮G/H/IRRA实际加载/前向 | 全0 |
| 正式GPU Attempt | 未创建，0次 |
| 新旧两图生成数值对照 | 原像素来源核验完成；本轮新输出对照PENDING |
| 独立CPU Joint全量验收 | 未执行；不能用Fixture或旧审计代替 |

待用户确认GPU资源可用后，按STAGE4D2.md的同一显式入口重新开展完整预检；不得自动等待资源、后台启动或恢复。正式Attempt仍未消费，不需要处理旧缓存/Claim或删除任何文件。若将来执行后失败，必须保留现场并经审查另行授权恢复Attempt。

后续IRRA/RaSa/APTM只消费同一物化FP32 RGB/Unicode，各自原生预处理/Tokenizer并创建隔离ModelSpace表示/Score；禁止把IRRA Embedding当跨模型输入或让目标反馈改变攻击产物。当前尚无全量RGB，因此这些评测仍未就绪、未授权；本轮不执行IRRA Full或任何跨模型/训练。
