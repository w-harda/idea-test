# ADR-0003：实际成员权限与原生Model View

状态：接受（Stage 2实现；Stage 2.5已按单独授权完成2个Train样本的真实GPU数值/输入梯度验收，范围见../framework/STAGE25_REPORT.md）。

背景：role字段不能证明样本属于历史攻击Train；官方val不同于official train内部Validation。模型缓存仅按维度/Checkpoint命名也无法区分输入编辑、顺序及Tokenizer变化。融合模型不必提供独立global embedding。

决定：

- Dataset保留原文、原顺序及native PID；训练奖励通过annotation、内部split、G/H来源共同校验后的成员绑定。公开输入不携带评测身份。
- 通用ModelAdapter约定原生score及能力，编码方法为IRRA专属能力；不强制双塔。
- IRRA提供冻结Inference与输入可微Surrogate View，显式真实加载，默认无模型副作用。
- ModelSpace绑定实际相关producer与运行协议；InputBinding另外绑定Dataset、有序成员和实际输入内容，持久化Record不得仅依赖维度或Query ID。
- Joint训练权限按G/H组件合并；图像梯度权限属于G组件，不错误要求H Unicode可微。

理由：防止Validation/Test反馈混入训练，保持原IRRA协议，避免跨模型/跨输入误用缓存。显式View使冻结参数与输入梯度可以同时成立。

影响：历史代码/缓存不修改，新空间不能静默标记旧特征为可用。配置仍无Attack Plugin/统一Engine可执行；真实GPU等价验证单独申请。未改变任何攻击算法或奖励公式。
