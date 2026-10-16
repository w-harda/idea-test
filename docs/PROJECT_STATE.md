# 项目当前状态（唯一权威入口）

状态来自Git、代码和不可变证据；历史文档是阶段快照。架构见[ARCHITECTURE](ARCHITECTURE.md)，历史实验见[EXPERIMENT_REGISTRY](EXPERIMENT_REGISTRY.md)。

## 当前阶段

**Stage 3B-R CPU-only修复完成：真实G/H加载导入闭包CPU预检PASS。** G/H/Joint真实输出等价仍PENDING/legacy_unverified，未进入Stage 4，未执行新的GPU尝试。

- 基准：`0fb90cd1b280da886aec08834dc13d28eacd5848`（Stage 3B FAIL发布）。
- 独立分支：`codex/tbps-framework-stage3br`。
- 独立树：`/home/lzf/ldx/projects/idea-test-tbps-framework-stage3br-20261016`。
- 实现Commit：7acadf64b2d9956c87eb1d26f8f39a88563c1db0（本条由后续文档提交登记）。
- [文件/CPU命令/停止边界](framework/STAGE3BR.md)；[修复与保护报告](framework/STAGE3BR_REPORT.md)。

## 本轮完成与验证

- 新建派生来源配置，不覆盖Stage 3A原配置；原74项及全部其他字段保持，新增加condition_schema.py原件SHA `f13932c2688cab4e135d063eed6f066016f6ba26e2316cbc201f19b246b298b4`，总计75项。
- 派生证明严格绑定parent/derived SHA及唯一允许增量；不关闭来源检查。
- 严格CPU入口覆盖实际加载工厂全部7个旧模块入口、11个函数/类符号及前端动态文件导入；33份实际执行源码、75项来源核验通过，无其他缺项。
- 整个预检硬性禁止torch.load/jit.load、神经构造/前向、CUDA初始化及导数；实际禁用操作尝试为空。CaptionFrontend与Byte-BPE仅CPU规则检查，固定2条Train资格记录保持。
- **323项框架CPU测试（298原有+25新增）+36项历史基础回归=359项不同测试通过。** 含合成依赖逐项缺失/篡改拒绝、动态加载SHA、派生证明、正向通过及命名空间/锁恢复。
- 无权重反序列化、真实G/H/IRRA前向、GPU尝试、训练、Full排名或新条件准备。

## 证据与保护

新CPU输出仅在 `/home/lzf/ldx/outputs/tbps-framework-stage3br/`，不提交Git。

不可变证据：`cpu_closure_v1/result/`，report SHA `b974a016a61876fa0c4ee40df8faff6d9b83413afc092cf4d1a978be252d4401`；Manifest SHA `19434bdcdbda7255ba582190fa5c003b8a0fe98a58fc27e0878cbc8392217f34`。

7991份历史文件SHA/mtime不变；原Stage 3B输出17份文件SHA/mtime不变。FAIL attempt/result回执及2358项来源有效，无pending。原实验树40c7476、Stage 2.5树b7aac69、Stage 3A树ef835a6、Stage 3B树0fb90cd均保持原HEAD且clean。未修改历史prototypes、Checkpoint、Tokenizer、预算、缓存、协议、日志或结果。

## 原Stage 3B FAIL永久保留

发布Commit `0fb90cd1b280da886aec08834dc13d28eacd5848`，实现0891e26525483dc22dfab589c3f8d81965f23812。唯一真实尝试在G10恢复一次后因G像素读取链缺少condition_schema来源pin停止；H未恢复，真实G/H/IRRA前向均0。

输出 `/home/lzf/ldx/outputs/tbps-framework-stage3b/equivalence_v1/`；FAIL report SHA `e72a65118cb0c216e2fbb0877152d11524b0254f73cc9f2c8547e91900c7a79c`。没有输出误差、H动作或Joint真实组合验收，不能将其改为PASS。[FAIL报告](framework/STAGE3B_REPORT.md)。原配置缺项拒绝的CPU回归继续保留。

## 已完成与冻结历史

Stage 1发布61471db；Stage 2发布4e2a583；Stage 2.5发布b7aac69，真实IRRA最小等价Smoke通过。Stage 3A实现487584a/发布ef835a6，G/H/Joint薄插件CPU验收完成。

Stage 2官方Test3074/6156身份及2048 Train/128内部Validation成员已核验。G-v2完整训练/Full、H正式SCST及Epoch7 Selection、Joint Full、消融/AP实验与8个历史资产索引全部冻结。历史IRRA Embedding不得跨模型重标或复用。

## 下一步与权限

CPU来源阻断已解除，具备提交审查、重新申请受控GPU等价Smoke的前置条件。Checkpoint payload、真实模型构造/设备迁移、G像素输出、H完整动作及Joint真实组合均尚未验证；本轮PASS不将等价状态升级为VERIFIED。

本轮仅CPU授权。任何GPU新尝试须用户重新明确批准、先执行严格CPU入口、使用新的不可变attempt。不得恢复/覆盖原equivalence_v1；不得自动重试、Full、训练或Stage 4。提交推送后停止等待审查。
