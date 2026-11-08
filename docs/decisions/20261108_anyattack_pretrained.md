# ADR：固定官方AnyAttack Decoder的免训练TBPS迁移

状态：采纳。用户明确禁止TBPS下游微调；不替换原CLIP特征空间，保留官方Decoder与16/255，采用可CPU exact重放的噪声尺寸桥接。固定异PID参考图，仅映射访问PID，无Victim反馈。代价：与R²TUA未公开适配不保证一致，与Idea/SGA预算和信息条件不同。三Victim评分严格复用既有原生实现。详见[完整协议](../framework/ANYATTACK_PRETRAINED.md)。
