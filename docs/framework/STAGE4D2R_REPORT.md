# Stage4D-2R：GPU启动顺序最小修复与CPU验收报告

基准883c4be59284b79f475412eb40ea6d96e8b1742a；独立codex/tbps-framework-stage4d2r，工作树/home/lzf/ldx/projects/idea-test-tbps-framework-stage4d2r-20261017。实现Commit由后续文档提交登记。

**CPU修复与验收通过；真实GPU仍未重试。** 本轮真实G/H/IRRA加载及前向、CUDA初始化、新GPU Attempt、新图像生成均0。generation_v1永久FAIL；generation_v2尚不存在；Image/Joint全量仍PENDING。

## 原FAIL与原工作树保护

原工作树HEAD仍883c4be，4份已修改Markdown与未跟踪STAGE4D2_EXECUTION_REPORT.md保持原SHA/mtime及Git状态，未reset、clean、覆盖或提交原树。

原执行报告完整内容以字节exact方式收录到新分支docs/framework/STAGE4D2_EXECUTION_REPORT.md。原文件SHA：

dca03f00ccc8333e1c13eec310f24d5601fa37eb9c00f45a9adc12cff1a9c143

原失败根/home/lzf/ldx/outputs/tbps-framework-stage4d2/generation_v1/保留全部16文件，整体/子结果均FAIL；原G加载尝试/恢复/前向0，新图像/逐图Claim/Proof0。本轮没有清理、重新签名、重写旧失败回执或恢复该Attempt。

| 原证据 | SHA-256 |
| --- | --- |
| attempt/manifest.json | 97208f29d78ef656069cab51aba9a2f1f64f2c51400fe256783233d4eba4889d |
| gpu-start/manifest.json | d14e86c6921d927c09eec7e4240acc48f6ca10c47ac65ead7e05debb00a9c4d1 |
| gpu-result/report.json | 1fcc0b7cf487043d0b4b5762057ad6380308dd753f231673da555b2ed9388c26 |
| result/manifest.json | de017a15233cc57184108c5a10aaa6effe34f49e7afb736f7b76de3833ec8174 |

## 最小运行代码diff

原child在no_optimization/g_load_only已经改变autograd及torch.load之后才调用initialize_cuda，违反CPUStartupGate快照要求。原门禁正确拒绝，不是权重、预算或模型数值问题。

```python
# 原顺序
with guard, no_optimization() as blocked, g_load_only(...):
    # 原线程、seed、精度配置
    gate.initialize_cuda()
    # G工厂、Hook、流式生成

# 修复后
with guard:
    # 原线程、seed、精度配置
    gate.initialize_cuda()
    with no_optimization() as blocked, g_load_only(...):
        # 原G工厂、Hook、流式生成及验收代码原样嵌套
```

CPU来源/Plan检查、cpu_only退出、checked/exit_checked、资源查询仍在前。两个安全上下文均成功后才可执行G工厂；初始化或安装失败均停止。原CPUStartupGate、gpu_safety.py、full_gpu_audit.py、Stage4D1及全部历史算法不修改。

另将新树入口OUTPUT固定为generation_v2，防止误用已经消费的generation_v1。新增派生config与来源证明，只改变output与entry_sources；冻结Plan、Stage4D1来源路径、预算、预处理、3074/6156、seed42、7200秒及全部资源门限不变。原stage4d2_generation.server.json原样保留，不能通过新入口启动。

| 新来源 | SHA-256 |
| --- | --- |
| 修复后full_gpu.py | ac327585a14d6f0732423ecb891ed93bd0b2f7aacfab0b06d3e229b649901530 |
| stage4d2r_generation.server.json | d4b802d9962baf4061f7ffe1ec2c4127746b7c3721793e19d07e3eab3869cbca |
| stage4d2r_derivation.server.json | e3659f5a9f6a50351c4426a497686674b62ce3489de726e16712bc119a1f76fc |

## 实际组合时序CPU测试

626项全部通过，65.47秒，退出0：既有611项+新增15项。新增测试内另有独立临时Fixture子案例覆盖5类启动失败、7类禁止操作、5类CPU/资源异常。不是三个Context Manager分别测试。

实际调用child，保留真实CPUStartupGate、cpu_only、source_execution_audit、两项安全包装、GObservation、stream_images、Claim/Proof、Store读回。仅设备/CUDA reset与权重入口Mock，G是合成CPU Mock模型，不加载真实网络/权重；CUDA init、lazy init和native init硬拒绝。

成功进入Mock G路径的事件顺序：

CPU_read_start → CPU_sources_checked → CPU_context_exit → cpu_sources_verified → cpu_isolation_exited → cuda_initialization_call → CUDA_mock → no_opt_ready → load_guard_ready → G_load_entry → G_weight_mock → Claim_before_G:g0 → Claim_before_G:g1

验证结果：

- reset Mock调用时CPU准备已完成，隔离已退出，完整函数/导入快照exact、legacy锁释放，CUDA未初始化。
- 原错误组合顺序仍被原门禁拒绝，尚未调用reset；提前初始化也拒绝。
- 来源、Plan、pending、导入路径、锁、RAM、资源异常阻止G入口。
- CUDA Mock、任一包装安装、工厂、前向失败均停止，不发布Fixture Image完成状态。
- 包装生效时禁止其他权重、Tensor/Autograd Backward、grad、两种JVP、Optimizer；真实观察器拒绝非G模块前向。
- 两图Fixture真实Hook started/completed=2/2，合成权重入口=1，逐图Claim先于G、Proof后于保存、读回与模型状态检查通过。
- 两图不能冒充3074：真实child最终由原固定3074总数检查拒绝，此为预期测试断言，未放宽生产作用域。
- 在临时合成Store实际执行Image与Joint封存路径及独立像素审计；Joint引用同两图与4文本，旧Fixture文本SHA/mtime保持，组合新增G/H前向0。
- 所有异常后Torch函数、导入设置、原模块对象、Hook与legacy锁恢复；默认入口和已消费Attempt拒绝回归通过。

测试日志/home/lzf/ldx/outputs/tbps-framework-stage4d2r/cpu_tests.log，
SHA fde79405a4b95129575f4f467acb7dd3dc9521b2903df8426aa95da56a3ef301。

## 真实来源与历史保护只读验收

独立CPU审计退出0，STAGE4D2R_CPU_REPAIR_SOURCES_AND_PROTECTION_VERIFIED。未调用GPU查询或初始化、模型加载、真实神经前向及排名。

- GH_RUNTIME_PREFLIGHT_PASS：75固定来源、33实际导入文件，forbidden_attempts为空。
- 官方3074 Gallery原图SHA/顺序、原Plan及完整6156历史文本核验通过；原H 6116编辑/40 NOOP/13537替换不变。
- 原Plan保持7ac4fefd0abaa10df085be3ec3ed56c4b38c782a21bfc44473f40b90965447d5。
- 30797个Stage4D1已发布文件、7991历史文件、3472历史缓存、70旧Stage4C2 FAIL文件SHA/mtime保持。
- 16个Stage4D2 FAIL文件SHA/mtime保持，旧FAIL快照digest 4e27aa15103b5b6a329d06622a2c4f39ea59ed558ccbbb0d00c358bae233ad9a。
- 原Stage4D1 Store完整快照仍ed2c18d9e64715419d488ea0ac764586486e719aa9e39b4c38e743e5740dd163；无图像单位、逐图Claim、pending或新Image/Joint回执。
- 原Stage4D2三个入口来源SHA另行核验通过；原工作树未提交报告及4份脏文档SHA/mtime/Git状态保持。
- CPU gate退出后的函数、导入与锁恢复，通过；CUDA_initialized=false，GPU资源检查明确NOT_EXECUTED_CPU_ONLY。

总只读审计142.45755676925182秒；其中全文/历史审计110.85626298189163秒，记录CPU Peak RSS611729408 bytes。不是GPU生成耗时或正式GPU资源预检。

证据/home/lzf/ldx/outputs/tbps-framework-stage4d2r/cpu_audit_v1/result/：

| 文件 | SHA-256 |
| --- | --- |
| manifest.json | 24a39edf72666ae9279009e4e97751d51c268787a29a0a35f5944d987a690461 |
| report.json | 381f1434f5e0df5a49fc5a4ab58a6cb95726ca4dcf63fe142c620b7fb0246682 |
| closure.json | d3937fbada0f6a95c5c7f5cff364c0f15898c4141a91fbf52e20e968b448aa86 |

## 后续建议与停止状态

已具备申请generation_v2 GPU尝试的CPU代码与来源依据，**不等于GPU执行许可或GPU路径已实测通过**。当前没有查询GPU、没有创建v2、没有新G RGB或正式Joint集合。

下一次须用户另行授权，使用派生配置及修复Runner，重新完整来源/历史保护/资源预检后，只开展一个新Attempt；仍复用冻结Stage4D1 Plan与6156文本。资源不足即停，失败保留现场不重试。准确的未来预检、显式GPU入口与完成后CPU审计命令见[执行说明](STAGE4D2R.md)。

无需恢复生成进度或人工清理文本/旧Claim。保留generation_v1 FAIL，禁止复用其身份。不进入IRRA Full、RaSa/APTM、训练或其他阶段。
