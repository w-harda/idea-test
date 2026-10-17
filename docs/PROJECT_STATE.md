# 项目当前状态（唯一权威入口）

状态来自Git、实际代码及不可变证据；旧报告是历史快照。架构见ARCHITECTURE，实验见EXPERIMENT_REGISTRY，重要边界见ADR。

## 当前阶段

**Stage4C-2R CPU-only启动顺序修复及旧产物只读审计完成。** 554 CPU tests通过。旧G两图/H两文可严格只读引用；原Stage4C-2整体FAIL保持。IRRA真实特征与四组2×2 Score等价仍PENDING。

**暂停申请GPU：现有加载器无法证明满足固定2GiB RSS门限。** 静态Storage布局与本次CPU准备峰值的保守叠加2.3548354730010033GiB；不是实测超限或严格下界。Stage2.5未记录CPU RSS，不能消除此风险。本轮未提高门限、未改变模型加载器、未执行GPU或创建新的正式Attempt。

- 基准fcf097d3ab31785381947289a680e5f5ccbf9d42。
- 分支codex/tbps-framework-stage4c2r。
- 工作树/home/lzf/ldx/projects/idea-test-tbps-framework-stage4c2r-20261017。
- 实现Commit由后续文档提交登记。
- [修复diff、时序测试、只读复用、资源与后续方案](framework/STAGE4C2R_REPORT.md)。

## 已完成与验证

smoke.py仅调整IRRA启动顺序：CPU历史特征/图文产物/动作与来源校验→全部隔离退出并核验Torch函数/导入路径/旧模块锁恢复→CUDA峰值初始化→IRRA加载入口。原cpu_only、feature_only_cpu、native_action_validator门禁保留；G/H阶段函数与原基准AST完全一致，原配置、模型、预算、数值阈值不变。

新增CPU测试执行实际child→stage_irra→prepare_inputs，用真实CPU Fixture文件和原门禁记录五阶段事件；正确顺序进入Mock加载/编码，错误来源/动作/特征或未退出隔离均在加载前拒绝。518框架+36历史基础=554项通过，17新增。

真实只读CPU准备亦通过：75项G/H固定来源、33个旧实际执行文件、5个官方IRRA执行文件闭包；限定两行四family FP32特征读取/动作验证完成后退出隔离，没有初始化CUDA或模型。

RetainedInputs/FrozenPlanReference明确引用原Plan 7c0c00daf022bfcba94d6e3583c6de26c89b58c0e9c6696449cdbadb1ab7072d；新树普通Plan e94013ed0986b3aacb780b611df247081af40e4c381cc7d4f7610dca033e6b78不同，不替换原id、不复制/重签旧单位。两图L∞0.0313725471496582，文本2编辑/固定NOOP及完整动作exact；原G生成Claim/证明、G/H权重来源均有效。原Attempt70文件SHA/mtime保持FAIL。

## CPU资源结论

IRRA Checkpoint共1320591330 bytes Storage，其中模型440197110、其他880394220；现有加载流程先build模型，再完整torch.load包含optimizer的payload，重叠估计1760788440 bytes，2GiB预算只剩386695208 bytes运行时空间。CPU准备峰值767696896 bytes，保守合计2528485336 bytes。仅ZIP/pickletools词法扫描，无pickle执行、模型反序列化或Storage内容读取。

Stage2.5 PASS及其930.52MiB GPU峰值不等于CPU RSS通过；其CPU RSS无记录。资源状态GPU_APPLICATION_BLOCKED_RESOURCE_BUDGET_UNPROVEN。后续需单独批准低内存加载方案的CPU调查或明确修订资源预算，不能自行GPU试跑。

## 不可变证据与保护

/home/lzf/ldx/outputs/tbps-framework-stage4c2r/：

- cpu_audit_v1/result/manifest.json：df0a94bea0bb8b454ba44d9a1e4b02ead859137327a0a6cce5e73250b9f78bf5。
- cpu_audit_v1/result/report.json：7b2b6cdf71fed7b203e896867b15f3c87462112ab9bb98ea3e9ad4e524ba8a33。
- protection_v1/result/manifest.json：da6626c11b36a4b53d929d7ac5b94abb912f2c60e536e5818c9f3da252ca4423。
- cpu_verified_v1/result/：最终源码/测试/日志/保护校验绑定，Manifest SHA8fbf97ccfe373f80f64c1f5c3bcc9be7ad5ee6744c1857def242ac60eba06e70。
- 无新正式Attempt、无GPU、真实G/H/IRRA加载/前向、Full排名或训练。缓存/权重/outputs不提交。

7991项历史保护文件、3472个Joint Full文件SHA/mtime保持；原Stage4C-2/C1/B2/Stage3B FAIL/Stage3B2 PASS/发布树HEAD及clean保持。原Stage4C-2 FAIL Manifest c1922bdf32935623dc9e85b1aeffe8a817cef2ab42b3ba767f62eda90e45155b不变。

## 下一步与权限

等待审查。先解决CPU资源预检依据，再申请新的IRRA-only两图/两文验证：只读引用原单位与原Plan，G/H新恢复和前向0；IRRA恢复1、图像4/文本4、FP32/no_tf32/1e-6，不排名、不重生成。新结果只能标记IRRA验证，不能修改原整体FAIL。

本轮未添加可启动新IRRA-only Attempt的GPU CLI；旧入口仍拒绝已用materialized_smoke_v1。没有授权全量物化、Stage5、RaSa/APTM、Full或训练。

## 既往阶段索引

- Stage4C-2：fcf097d3ab31785381947289a680e5f5ccbf9d42，[唯一Attempt FAIL报告](framework/STAGE4C2_REPORT.md)，G两图/H两文本物化局部PASS。
- Stage4C-1：4d3bc025a7da593368ebd22cc05db1ebe787b446，[来源与存储验收](framework/STAGE4C1_REPORT.md)。
- Stage4B-2：c09754e6afc8cf53a6c0c6e8715b6d4eace80946，历史特征METRICS_REPLAY_VERIFIED，不代表真实像素端到端通过。
