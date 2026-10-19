# ADR-0011：DRA官方Source参数梯度与原词槽位预算

状态：本阶段固定8图/16文真实Smoke及CPU审计PASS。

官方RAttacker保持91f78325…原字节。候选试探评分有额外backward；Source权重固定不等于禁用参数梯度。允许Wrapper仅开启参数求导，保持eval、无Optimizer，完整攻击后清梯度并检查所有参数/buffers摘要；不改官方反向或添加梯度技巧。

官方number_perturbation=1限制原words列表的一个槽位。get_bpe_substitues可返回含空格短语，因此空格分词数量不变不是原算法约束。只读记录官方change、words、final_words，在原槽位上审计，并核对join与返回文本exact；不新增候选过滤。
真实leggings→green and wearing属于一个槽位替换，必须保留。早期外部误判FAIL保留，不能重写为PASS。

Source/Data/Tokenizer/Target原生评分属于必要I/O移植。所有算法参数/流程/源码不变；固定环境Smoke保真PASS不代表原论文原模型、Full、所有Token边界或严格实数投影上界已验证。
方法级Baseline比较保留各自原生机制，披露访问权限、预算、文本规则与Matched Clean；不为降低指标重新设计DRA，也不因这些配置差异否定方法比较。
