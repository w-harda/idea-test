# 项目当前状态（唯一权威入口）

状态来自Git、实际代码和不可变证据；历史文档不用于推断当前授权。架构见[ARCHITECTURE](ARCHITECTURE.md)，实验来源见[EXPERIMENT_REGISTRY](EXPERIMENT_REGISTRY.md)。

## 当前阶段

Stage 1已原则通过审查。Stage 2 CUHK-PEDES Dataset Adapter与IRRA Model Adapter已实现并完成CPU验收，等待用户源码审查；真实GPU等价性尚未授权，未进入Stage 3。

- Stage 2基准：61471db976508411aafe58e97113322c14d6bbc7。
- 分支：codex/tbps-framework-stage2。
- 开发树：/home/lzf/ldx/projects/idea-test-tbps-framework-stage2-20261016。
- 实现Commit：STAGE2_IMPLEMENTATION_PENDING（由后续文档提交登记）。
- 接口、CPU命令及下一步授权方案：[Stage 2说明](framework/STAGE2.md)。
- 验收范围及证据：[Stage 2报告](framework/STAGE2_REPORT.md)。

## 本轮完成

- CUHK官方annotation保持原文、多Caption、原顺序、native PID；Test 3074 Gallery/6156 Query与G-v2、Joint、消融三份协议完整身份一致。
- G/H实际2048 Train（1861 PID）及128内部Validation（121 PID）绑定annotation/internal split/G metadata/H资格/Gallery/调度；PID交集0。内部Validation来自official train，不是official val。
- PublicImage/Text无PID/配对/奖励。只有已校验成员工厂可创建AttackPartitions；奖励绑定拒绝角色伪造、未入选Caption及Validation/Test成员。
- 官方IRRA来源严格SHA；RGB预处理、77-token Byte-BPE、原生编码及冻结Inference/输入可微Surrogate接口。
- 通用ModelAdapter支持原生score/Fusion Mock，不固定维度或双塔；新缓存身份绑定模型、相关producer、数据、有序成员和实际输入内容。
- Joint训练权限补齐G/H组件并集，图像梯度只要求G组件所需能力。“8类资产”更正为“8个资产索引”。

## 验证结果

- 183项框架CPU tests（Stage 1的114项+Stage 2新增69项）全部通过。
- 36项历史数据/Top-K/生成器/Joint基础CPU回归通过；合计219项，不是全部prototype tests重跑。
- 10种合成文本Token ID与官方tokenize逐值一致；3组合成PIL预处理误差0；4×12合成排名验证官方定义。
- 真实来源/成员CPU preflight通过，读取2176份G metadata；未加载condition Tensor或Full特征。
- 7991份保护文件SHA/mtime_ns不变。原实验树仍40c7476且干净；Stage 1工作树仍61471db且干净。
- 无真实IRRA模型构造/图像文本前向、GPU、G/H训练、Full排名重算或后台监控。

## 历史完成与下一步

G-v2完整训练/Full、H-v1正式SCST/Validation Epoch7 Selection、Joint Full和消融均保持冻结，8个历史资产索引保留；Stage 1实现ffd18dd、最终发布61471db。

下一项建议仅授权2张Train图+2条Caption的真实IRRA等价Smoke（最多5个图像样本前向、4个文本样本前向，含1次输入梯度检查），具体阈值及边界见Stage 2说明。当前不授权执行。

没有G/H Attack Plugin、统一Evaluation Engine、RaSa/APTM或ICFG/RSTP Adapter；配置仍不能执行攻击/评测。旧特征不能静默标记为新ModelSpace，真正数值/冻结状态需后续真实Smoke验证。
