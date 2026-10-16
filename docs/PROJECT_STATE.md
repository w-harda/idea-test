# 项目当前状态（唯一权威入口）

本文件记录Git/源码/不可变证据支持的当前状态。历史入口及来源见EXPERIMENT_REGISTRY；不以旧文档CURRENT推断事实。

## 当前开发

- Stage 1基础契约、配置dry-run、旧资产只读桥及独立资产管理：进行中。
- 分支codex/tbps-framework-stage1；独立服务器树/home/lzf/ldx/projects/idea-test-tbps-framework-stage1-20261016。
- 基准Commit：40c7476f92d6bc2458d86029cf7bf8317da0a086。
- 本轮允许：新增框架/管理文档、CPU fixture测试、历史资产CPU只读描述、独立分支commit/push。
- 本轮禁止：历史代码/资产改动，GPU/模型前向/Full指标重算，真实训练/评测入口，自动Stage 2。

## 已完成事实

- G-v2 10 Epoch、官方IRRA/CUHK Full；H-v1正式SCST、独立Validation Selection；Joint Full已完成。
- Joint消融Full completed存在，3074 Gallery/6156 Query。用户确认ABLATION_CPU_AUDIT PASS及ABLATION_PIPELINE_SUCCESS；本阶段将补充对应日志绑定。
- 全仓审计基准40c7476，412 tracked/319 Python；两份审计报告留在Windows独立审计目录，设计摘要纳入ARCHITECTURE及ADR。
- 已有研究实现和资产全部冻结，详情见EXPERIMENT_REGISTRY。此处不复制指标或日志。

## 下一阶段与阻塞

- 下一阶段候选：Stage 2官方Dataset/Model Adapter，尚未授权。
- Stage 1不提供可执行Attack/Model/Dataset插件；任何真实执行配置均应被拒绝。
- 待核：消融独立CPU审计/管道成功日志路径；新模型权重/上游commit及VLPTAttack方法身份等留待未来。
- 结束时将补充真实测试、保护核验、实现Commit和当前限制。
