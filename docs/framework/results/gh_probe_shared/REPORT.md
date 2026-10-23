# G/H probe_v2：Probe专用6144MiB协议与真实GPU结果

## 核心结论

2026-10-23完成唯一一次probe_v2，原生结果为 **GH_NATIVE_NUMERIC_PROBE_PASS**，退出码0；后续CPU只读回执、内容SHA、来源及历史保护核验PASS。未重试、未部署Watcher、未运行A1/B、训练或评测。

真实Unicode新语境使G第一步原Loss的梯度发生变化；固定NOOP条件完全复用，梯度差小于既定容差。这只验证H→G训练条件和第一步梯度通路，不代表G/H参数已经更新，也不代表完整双向训练验证完成。

**覆盖限制：冻结探针没有调用GPU阶段RNG恢复/读回对照，CUDA RNG恢复仍为PENDING；G梯度L2范数也未保存在原回执中。** 本次不追加GPU计算补齐，不修改已提交原结果，不宣称所有A1/B启动前提均已验证。

## 执行与来源

| 项目 | 实际值 |
| --- | --- |
| 基准Commit | f9701c7ea924f734d83f6cfeeaa38b0b83c30f01 |
| 执行源码Commit | a7597f9c964dda0c8c48973e4678f0b9d3fa095a |
| 分支 | codex/gh-probe-shared |
| 工作树 | /home/lzf/ldx/projects/idea-test-tbps-gh-probe-shared-20261023 |
| 原生结果目录 | /home/lzf/ldx/outputs/tbps-framework-gh-bidirectional-backend/probe_v2 |
| 独立控制/日志目录 | /home/lzf/ldx/tmp/gh_probe_v2_shared_execution_20261023 |
| 执行时间 | 2026-10-23 12:24:24至12:24:47 +08:00 |
| 启动器PID | 3273589，任务已退出，无持续GPU后台任务 |
| 配置SHA256 | 77ae41af56189ece991ee7cf21fd0ea0f3be5645202bd1cb7cf7e0b2ae735649 |
| source_key | 06b5c6a15061e635e49a9e13deb14022956fdbc272ad0ed378676e57ca5f826b |
| ConditionIdentity | bb9ba7d89eb09a9a68bedcfdf01889766178dc8e081585390bc129dc416bd9bd |
| completed Manifest SHA256 | d6e0b960e20a1ba5128f9e0b4edf497e7ea4b098fbf5cd3125799d2ad1136764 |

派生配置为configs/framework/gh_bidirectional_backend.probe_v2.shared.server.json；原f970配置和SHA保持不变。原probe_v2此前尚未创建或消费，本次首次Claim绑定派生配置与新source_key，因此继续使用唯一probe_v2目录。现在该身份已消费，禁止重复调用。

140项框架producer、496项旧运行来源及G60/H10/IRRA父Checkpoint均重新核验。执行Commit之后仅提交派生证据与文档，不更改本次执行代码。完整路径、源码SHA、回执、日志SHA见[SOURCE_EVIDENCE.json](SOURCE_EVIDENCE.json)及[CPU_RECEIPT_AUDIT.json](CPU_RECEIPT_AUDIT.json)。

## 最小资源修改与CPU验证

生产Python只修改native_sources.py与native_cli.py：
1. resource_protocol(c,action)仅对action=probe返回6144MiB门槛的派生资源字典，不原地修改prototype。
2. run_gpu在创建Claim/CUDA之前使用该action对应资源协议。
3. verify_config继续检查全部来源，并严格拒绝未经授权的override字段/数值。

Smoke仍为10240MiB；RAM、磁盘、inode和训练协议不变。数值函数numerical_probe的AST与f970逐项一致；native_numeric、native_backend、原G/H/JVP/Loss/Reward未修改。

13项新资源契约测试+276项原回归：**289 passed，0 skip，67.71秒**。覆盖Probe/Smoke分离、不变性、6143/浮点数/额外字段拒绝及实际run_gpu入口在资源拒绝时尚未创建Attempt/CUDA。真实tbps-image环境CPU Preflight和G/H完整父状态恢复PASS，导入环境/CPU隔离正常退出。

即时快照：RTX4090空闲21957MiB，总24564MiB，利用率89%；其他PID3255229占2086MiB，未干预。RAM可用122773651456 bytes、磁盘可用6178506125312 bytes、可用inode240217919，全部门禁通过。利用率只记录，未要求独占或等待。实际启动显存也高于旧门槛，此结果不能单独证明在恰好6144MiB余量下必然安全。

## 真实数值

| 验证项 | 原始结果 | 判定 |
| --- | --- | --- |
| 固定NOOP | cuhk:26530:0，K=0，条件内容SHA完全一致 | PASS |
| NOOP G梯度最大绝对差 | 4.1443854570388794e-08 | ≤1e-6，PASS；非bitwise exact |
| 非NOOP | cuhk:26747:0，K=1，一次Unicode编辑 | PASS |
| 非NOOP G梯度最大绝对差 | 0.005286501720547676 | >0，PASS |
| 原Caption独立编码与旧z_t最大差 | 0.0 | exact |
| 新JVP | 1，旧JVP未重命名复用 | PASS |
| JVP与中心FD相对误差（比值） | 0.0005382464902543462 | 固定门槛内 |
| JVP与FD相对误差（百分数） | 0.05382464902543% | PASS |
| JVP方向范数 | 0.03990959137026973 | 有限、非零 |
| JVP径向点积 | -1.4551915228366852e-11 | 原核验通过 |

原文本条件/新文本条件Loss：NOOP均0.3706851899623871；非NOOP为0.47378572821617126/0.4904400408267975。该Loss变化是条件改变的数值记录，不是训练优化效果或检索指标。原生只保留梯度SHA和max_abs差，未保存完整梯度或L2范数；不得从SHA推导范数。

非NOOP原文：
> A man wearing a red and white cheerleader outfit is wearing a blond wig and has his arms up in the air

攻击后：
> A mаn wearing a red and white cheerleader outfit is wearing a blond wig and has his arms up in the air

零起始字符位置3：a（U+0061）→а（U+0430）。Canonical锚点male/gender/person、原Span[2,5]保持；H属性a19834c40e72bf5597661d76映射G属性3f284f87e4e11cfbcfa9fb6f。UTF-8字节长度102→103，Byte-BPE内容Token22→24，总Token24→26，EOT位置23→25，无截断。通过新语境整句编码、原锚点reader定位及真实JVP/FD；本样本原字节对齐已合法，不能声称额外的utf8_character_split修复路径也经此次GPU覆盖。

## 模型、状态和计算预算

| 项目 | 实际 |
| --- | --- |
| 模型恢复 | G=1、H=1、IRRA=1、reader=1 |
| 神经调用 | G=4、H=1、IRRA image=4、IRRA text=6、reader=1 |
| 新JVP | 1 |
| Optimizer.step | 0 |
| 完整Train图库刷新 | 0 |
| Validation/Test/检索评分 | 0 |
| G/H权重和双Adam前后 | 原生tree_sha断言一致 |
| IRRA/reader权重、buffers、eval及冻结 | 原生版本/最终SHA断言通过 |
| GPU Adam迁移 | 原生重新绑定/状态读回检查通过 |
| CPU RNG父状态 | 恢复exact |
| CUDA RNG | 父CUDA状态保存；本探针未进行GPU恢复读回验证，PENDING |

CPU恢复G Adam step=484080；H参数step范围20124–20370（不同参数并非每visit都有更新），H父visits=20480、合法updates=20370。没有继承任何Smoke更新，也没有执行新更新。

边界说明：original_gradient临时采用原训练第一步的G.train()以保持训练语义，参数及buffers仍SHA不变；IRRA/reader始终eval冻结。探针只比较K+1串行训练的第一步，不模拟完整K+1更新，也没有验证G→最新图库→H Reward通路。

源码依据：native_cli.py:43 numerical_probe、native_numeric.py:6 original_gradient/:29 compare_gradients、native_backend.py:162 enter_phase_rng/:193 condition/:338 loaded_backend。探针没有调用enter_phase_rng或restore_complete；不能把CPU预检的CUDA_RNG_preserved=true混同于CUDA_RNG_restored=true。

## 资源与时间

- GPU peak allocated：1083664896 bytes（1033.463379MiB）。
- GPU peak reserved：1105199104 bytes（1054.000000MiB）。
- 进程Peak RSS：2425774080 bytes（2.259178GiB）。
- 原生GPU执行范围耗时：14.204803325235844秒。
- 含启动、CPU准备与退出的进程wall time：22.70秒，GNU time退出0；无OOM。
- 共享GPU可能影响时间；PyTorch峰值为本进程分配器统计，不是全卡占用。

## 只读审计与历史保护

CPU审计逐单元验证claim、两份rollout、两份condition、completed的Manifest/文件SHA/回执；两份小条件Tensor内容SHA再次核对，检查无FAIL/pending或额外单元，审计前后所有原始artifact SHA及metadata不变。该审计未初始化CUDA，未调用模型前向。

151378项历史文件size/mtime_ns/inode无变化；旧probe_v1/Watcher的28份SHA及metadata不变；G60/H10/IRRA重新SHA与原来源一致。此处不宣称对151378文件做全量内容SHA重扫。原probe_v1 FAIL和Watcher日志保持，Smoke输出仍不存在。

执行器原exit0及日志保留；后续只读核验单独追加CPU_READONLY_RECEIPT_SOURCE_PROTECTION_PASS事件，控制phase为NATIVE_NUMERIC_PASS_CUDA_RNG_PENDING，没有改写原生completed报告。

## 下一阶段结论

H→G的真实Unicode语境、JVP/FD及原Loss梯度变化已验证；NOOP阴性控制和模型/Adam不更新已验证。原生数值回执PASS，但CUDA RNG真实恢复读回、G梯度L2记录及完整G→H Reward反馈尚未覆盖，不能宣称A1/B全部技术前提均已验证。

本次结束。任何补充GPU验证或A1/B两轮Smoke均需新的明确授权；不自动运行、不追加训练、不使用Test或改变原K/Loss/Reward。
