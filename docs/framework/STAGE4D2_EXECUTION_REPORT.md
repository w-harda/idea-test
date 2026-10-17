# Stage4D-2正式执行报告：唯一Attempt FAIL，G恢复/前向为0

执行基准883c4be59284b79f475412eb40ea6d96e8b1742a；复用codex/tbps-framework-stage4d2及原工作树、原入口、原配置。没有修改任何Python、配置、权重、Plan、旧文本或回执。此报告是正式执行证据，原STAGE4D2_REPORT.md的资源阻断记录保留为历史快照。

## 结论与完成范围

**generation_v1唯一正式Attempt已消费，最终FAIL，不能自动重试。** 本次资源和完整CPU预检均通过，失败发生在CUDA初始化门禁前。G恢复尝试/恢复/实际前向全部0，H/IRRA/RaSa/APTM加载/前向0，Backward/Optimizer/JVP0。未新增任何Gallery像素单元或逐图Claim/Proof。

Image 0/3074，IMAGE_FULL_MATERIALIZED未发布；原Text 6156/6156保持TEXT_FULL_MATERIALIZED；Joint未封存。旧两图新输出对照和独立CPU Joint全量验收均PENDING，不能用原两图或Fixture冒充完成。

## 资源与预检

最初只读检查RTX4090：无计算进程、空闲24051MiB/已用40MiB、利用率0%；系统可用RAM124427784192 bytes，磁盘6487394717696 bytes，inode240915476。满足原8192MiB/8GiB RAM/10GiB磁盘/100000 inode条件。

原默认CPU命令执行退出0，出现GH_RUNTIME_PREFLIGHT_PASS和G_FULL_CPU_PREFLIGHT_PASS；正式入口自身再次预检也通过。固定75项来源/33份实际导入文件、原Plan、3074原图SHA、6156文本/action及完整官方顺序通过。原Plan ID保持7ac4fefd0abaa10df085be3ec3ed56c4b38c782a21bfc44473f40b90965447d5。

日志/home/lzf/ldx/tmp/stage4d2_cpu_preflight_v2.log，SHA980edd84fc40e7d81152592e0a692a3c9b7b6b9d7864c1981c5cfed4bf52da40。正式命令完全沿用STAGE4D2.md的--allow-gpu-generation，PYTHONPATH指向冻结Stage4D1/src，未更改7200秒预算或精度。

## 已确认的失败原因

报错：CPU contexts must exit before CUDA initialization。

- 新入口src/tbps/materialization/full_gpu.py:176先进入no_optimization()、g_load_only()。
- gpu_safety.py:55–58替换torch.autograd.grad/backward等函数；:77替换torch.load。
- :179随后调用gate.initialize_cuda()。
- 冻结Stage4D1的irra_startup.py:9–11把torch.load和autograd函数纳入isolation_snapshot；:43要求此时快照仍与CPU准备前一致。
- 包装已改变这些函数身份，足以触发拒绝；异常发生在:47 torch.cuda.reset_peak_memory_stats()之前。因此没有CUDA初始化、权重恢复或模型前向。

这是新执行入口的启动顺序问题，不是G权重、图像预算、模型数值、资源不足或攻击算法问题。旧门禁正确拒绝了已改变的快照，不能删掉或绕过检查。

现有611项CPU tests仍通过，但未覆盖真实child控制流中“CPUStartupGate + 两个GPU安全包装”的组合时序。CPU测试通过不能替代这次失败的实际执行。

## 实际计数、资源与耗时

| 项目 | 实际结果 |
| --- | --- |
| 正式Attempt | 1，FAIL，gpu-start Claim已发布 |
| G恢复尝试/恢复/前向started/completed | 0 / 0 / 0 / 0 |
| H/IRRA及其他目标模型加载/前向 | 0 |
| Backward / Optimizer / JVP | 0 / 0 / 0 |
| 新图像单位/逐图Claim/Proof | 0 / 0 / 0 |
| 最大新扰动 / 两图新输出对照 | 不适用 / 未执行 |
| CUDA显存峰值 | 不适用，CUDA初始化被门禁提前拒绝 |
| 子进程资源监督段 | 0.013803698122501373秒 |
| 子进程CPU Peak RSS | 643301376 bytes（613.5MiB） |
| 父监督段（不含入口预检） | 19.92894448712468秒 |
| 父CPU Peak RSS | 747520000 bytes（712.890625MiB） |
| 正式失败Attempt文件 | 16个，13694845内容bytes，13746176已分配blocks bytes |

持久化control.started_unix=2026-10-17T09:23:12.250036+00:00，result/report.json的mtime为09:26:00.368091+00:00；文件时间差约168.118秒，包含入口CPU预检与失败收尾，并非GPU生成耗时。无RSS硬上限、OOM、超时或磁盘安全余量异常。失败后一次只读检查确认无GPU计算进程、无存活Runner。

## 不可变失败证据

根目录/home/lzf/ldx/outputs/tbps-framework-stage4d2/generation_v1/，整体及子结果均FAIL，原失败文件不重写：

| 证据 | SHA-256 |
| --- | --- |
| attempt/manifest.json | 97208f29d78ef656069cab51aba9a2f1f64f2c51400fe256783233d4eba4889d |
| gpu-start/manifest.json | d14e86c6921d927c09eec7e4240acc48f6ca10c47ac65ead7e05debb00a9c4d1 |
| gpu-result/report.json | 1fcc0b7cf487043d0b4b5762057ad6380308dd753f231673da555b2ed9388c26 |
| gpu-result/manifest.json | 19f4ec556e62ac67b2579d6d8cdbf5eb1e9261d70193adf589f7eed1ad637631 |
| result/manifest.json | de017a15233cc57184108c5a10aaa6effe34f49e7afb736f7b76de3833ec8174 |

主日志/home/lzf/ldx/tmp/stage4d2_generation_v1.log，SHA891bb99be4d7718f7012e8cc190acad8c50512a11e34989d22059631dfb99b02；子日志generation_v1/gpu.log仅FAIL，详细异常在gpu-result/report.json。

## 只读保护与CPU回归

失败后的独立CPU审计退出0，FAILED_ATTEMPT_READONLY_PROTECTION_VERIFIED。它只证明失败现场与旧资产保护，不是Joint全量验收PASS。

- 原Stage4D1 30797文件SHA/mtime exact，整个原Store与Attempt保存的old_files.json快照一致，未追加单位。
- 原H 6156条、6116编辑/40 NOOP/13537字符替换保留，未重新推理。
- 7991历史文件、3472Joint缓存、70原Stage4C2 FAIL文件SHA/mtime不变。
- 新失败Attempt16文件SHA/mtime在审计前后不变，回执和来源SHA有效。
- 无pending；原Stage4D1冻结库Git HEAD/clean不变；旧FAIL/PASS/指标重放证据保持。

审计/home/lzf/ldx/outputs/tbps-framework-stage4d2/failed_attempt_cpu_audit_v1/result/：
Manifest SHA86bbeda8a4e24aa840ec0343d9e2eafebc7b658628f045ae931b700452667420；
report SHA8889d0456ba75eb45bd62e91b08c7e4bdf0f6c0897b20813d503361edaa5f048。

失败后611 CPU tests再次通过，51.03秒，exit0。日志/home/lzf/ldx/tmp/stage4d2_after_attempt_cpu_tests.log，SHA01d7060b6c6e4e81254b84c3069d589de14987610037ecf31cd970961b2b1830。

## 最小后续修复建议（本轮未实施）

保持冻结门禁与所有G/H算法不变：资源guard继续在外层，CPU隔离退出后先gate.initialize_cuda()，之后才进入no_optimization()/g_load_only()；这两个GPU安全包装仍必须早于G权重加载及任何前向。

新增CPU控制流回归应执行实际child控制路径、真实CPUStartupGate和真实安全上下文，仅Mock CUDA调用与模型加载入口，覆盖错误顺序拒绝、正确顺序以及来源异常在加载前停止，不能仅用AST/两个步骤Mock代替。

原generation_v1已消费，即使G前向为0也不得重跑原命令或删Claim。修复必须在新工作树保留旧源码/配置来源，新增派生配置并经用户审查；新的GPU Attempt需单独授权、独立输出，不能覆盖此FAIL。当前无需清理或转换任何旧文本/缓存。

本轮遵守用户“失败保留现场、不自动重试、原代码不修改”。因未满足“全部通过后提交推送”，源码Commit保持883c4be59284b79f475412eb40ea6d96e8b1742a；本次仅更新必要Markdown工作副本，未commit/push。等待用户决定CPU-only修复及新的单次GPU授权，不进入IRRA Full/RaSa/APTM/训练。
