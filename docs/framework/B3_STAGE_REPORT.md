# AP-Attack B3三模型Baseline开发报告

2026-10-18，基准7bff515185efabcabff7a3e21a1002878178ff9c，独立分支codex/tbps-baseline-b3，工作树/home/lzf/ldx/projects/idea-test-tbps-baseline-b3-20261018。

## 结论

代码、CPU测试、B3来源审计及Full CPU预检完成。**三模型真实GPU Smoke尚未执行成功，当前为PENDING_RESOURCE_GATE**；没有本轮B3检索指标，不能标记正式Full就绪已验收。用户确认Full前，必须先完成三模型Smoke及各自CPU审计，Full脚本强制检查此条件。

GPU在修复启动顺序后的资源检查时空闲3865MiB、利用率22%；随后一次只读检查为92%，Ollama PID726931占用20178MiB。未终止/卸载Ollama，未部署等待器或后台实验。

## B3来源与CPU结果

- Generator实际SHA：c1c637e2e5346a3011f23006bb77bad60e4bd4bc28eb746d5f6bfcd881ba4ebb。
- config/sidecar/progress一致，Epoch60，ReID/Semantic权重10/10，固定最后Epoch，无caption/IRRA文本训练。
- 原annotation重算train为34054图、11003 PID，train与val/test PID交集均0。训练记录SHA重建ee9687e57b90ad710aefaa55563f73ad2be7164e430310959b1c0457158a589a，与原训练来源exact。
- B3训练module/runner实际SHA分别02cfb28d7125fc6256e3c36bcd428a3dd53582be0572b0d19a5ec7e85ef8f1b8、a7e9dbf151691bbab0571b7eec2d323509a508f2ee4e225d60aac8b4bf469a89，与sidecar exact。
- AP训练来源21文件SHA全部匹配；Duke语义骨干实际SHA d61a156497e73e451dba43697eaedd5e4cf6b714e0ce9e34b4ababc1313e3c53。仅只读哈希，没有加载语义骨干或其他历史Baseline权重。
- B3历史3074图/97批FP32像素缓存可复用，整批文件SHA、官方Gallery顺序、源图SHA与Checkpoint来源通过。
- Full CPU预检exit0，B3_CPU_PREFLIGHT_PASS full，官方3074/6156，三受害者既有Full协议、完整顺序及ModelSpace来源检查通过；没有Full排名、编码或GPU初始化。
- 固定10图完成CPU像素传递、不可变发布和逐值读回；最大L∞=0.031372129917144775，严格小于8/255。G/H权重加载/前向0，不是新的Generator推理。
- 历史B3/三模型正式资产12203个文件的inode/大小/mtime核验无变化；已读取的声明来源与97批实际SHA匹配。既有tracked代码diff为空，Stage4D/三个受害者源码、权重、原Plan与Full结果未改。

## 测试与失败记录

- 新增32项CPU测试，包括独立进程的真实torchvision/旧像素模块导入、Resize/ToTensor、CPUStartupGate→Mock CUDA→Mock加载时序。
- 修复后相关B3/IRRA/RaSa/APTM/Engine回归132项通过，6项因-k筛选未执行。首次广泛框架回归798通过、4项旧配置绝对路径门禁失败；这4项在未修改的基准工作树同样失败，未改旧配置规避。
- bash -n通过；Full脚本缺少--allow-full时exit2，且缺少任一成功Smoke/审计时禁止Full。
- 首次启动失败：新像素读取器首次导入torchvision增加sys.path，被原CPUStartupGate正确拒绝。G/H/IRRA/RaSa/APTM加载、CUDA初始化、神经前向全0。保留原FAIL及原代码/配置SHA；新增launch在门禁之前完成合法CPU依赖导入，原门禁及旧攻击算法不改。
- 修复后的ready资源门禁拒绝发生在Attempt创建之前，因此ready/smoke/irra/attempt不存在；未执行RaSa/APTM Smoke，无OOM或真实模型数值结果可报告。

证据：

- /home/lzf/ldx/tmp/b3_preflight.log：初始严格来源预检。
- /home/lzf/ldx/tmp/b3_cpu_tests.log、b3_base_regression.log、b3_final_cpu_tests.log：CPU回归。
- /home/lzf/ldx/tmp/b3_startup_cpu.log：真实CPU快照诊断，定位sys.path变化。
- /home/lzf/ldx/outputs/tbps-framework-b3-comparison/smoke/irra/execution/report.json：原启动FAIL。
- /home/lzf/ldx/tmp/b3_ready_smoke_irra.log：资源门禁拒绝，无新Attempt。
- /home/lzf/ldx/tmp/b3_ready_full_preflight.log：Full CPU预检PASS。
- /home/lzf/ldx/outputs/tbps-framework-b3-comparison-ready/smoke/pixels/collection/collection.json：10图CPU产物及预算。
- [可发布来源摘要](B3_SOURCE_AUDIT.json)；不提交Tensor、Checkpoint、缓存或原运行日志。

## 正式对比与后续

[运行文档](AP_B3_COMPARISON.md)提供可直接复制的CPU预检、10+10 Smoke、独立审计及一键nohup Full命令。方法顺序固定Clean → AP-Attack B3 → G-v2 Image-only，只有R@1/R@5/R@10/mAP，百分数两位小数，另存四项原始精度；不发布额外指标或下降率表。

Full只新增B3 Gallery编码/评分，复用已有同受害者Clean Query表示与已审计Clean/G-v2 Full指标，保持同一generation_v3原图、顺序、PID与原生模型协议。RaSa/APTM不使用IRRA Embedding。攻击侧G/H加载/前向始终0，目标模型不给B3像素任何反馈。

七种候选Baseline的官方来源/权限/资源/优先级见[可行性报告](BASELINE_FEASIBILITY.md)。用户已确认VLPTAttack就是Gao等ECCV2024 VLPTransferAttack/DRA；后续用官方代码，实验表仍称VLPTAttack。本轮没有接入或训练这七种方法。

待办：资源空闲后完成三模型真实Smoke与独立CPU审计；用户审查确认后手动启动Full。本轮停止，不执行正式Full或其他Baseline训练。
