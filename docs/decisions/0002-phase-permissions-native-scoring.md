# ADR-0002：按阶段隔离信息，保留原生评分

状态：接受。背景：H测试必须Caption-only；模型既有双塔也有融合评分，维度、Tokenizer和图像尺寸不同。

决定：PublicImage/Text与EvaluationIdentity/TrainRewardContext分型，配置显式区分代理/受害者、模型训练来源与攻击训练来源。Attack/ModelSpec声明可选能力和访问权限；未知/未实现能力拒绝执行。表示、攻击产物、Score分型，不强制512/77/cosine。

理由：通用接口不能泄露PID/正确配对，不能把融合模型伪装成余弦，也不能统一SCST/K轮/Meta等算法。

影响：合法配置不等于可执行配置。Stage 1只验证及dry-run，所有真实插件状态仍未实现；未来Adapter逐个以官方协议验收。
