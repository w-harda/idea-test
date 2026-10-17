# ADR-0007：攻击产物先物化，受害者编码与评分后置

状态：接受（Stage4C-1 CPU基础设施；真实生成和端到端GPU尚未执行）。

背景：历史IRRA特征不足以重建或证明对抗RGB。G-v2正式推理是image-only，不需要Test条件/JVP；已有H7 Test文本和完整动作可证明原始来源。未来RaSa/APTM应消费相同冻结样本，不允许目标反馈改变策略。

决定：

- 新materialization命名空间复用G/H薄插件和ArtifactStore，不复制网络/投影/字符替换算法。G保存FP32原RGB、Adv RGB及effective_delta；分别记录原图文件SHA、Tensor内容SHA、序列化文件SHA，不能用JPEG或Embedding冒充。
- H以UTF-8原字节SHA绑定原/攻击Caption，保存完整Unicode动作；历史H文本走显式只读来源桥和原生规则校验，不伪装成重新运行H。
- MaterializationPlan没有目标ModelSpace/Tokenizer/Score字段；训练代理/历史IRRA信息仅作为来源。Public输入没有PID/正确配对/奖励。Joint只索引已有两类产物，Gallery/Query独立有序。
- 可恢复入口先发布不可变生成Claim，再调用插件；前向后未发布或回执不完整时拒绝自动重复。已完整发布并校验的单元可跳过；pending与失败证据不删除、不回滚。
- 同一计划的不可变Manifest摘要只计算一次。会话首/集合边界完整SHA核验，期间复用已验证SHA并检查inode/size/mtime/ctime；变化即拒绝，不静默接受新来源。
- 目标Adapter后续只读取同一FP32 RGB/Unicode，用自身原生预处理/Tokenizer/评分；Representation/Score缓存按目标ModelSpace与实际输入SHA隔离，不重用IRRA特征，也不回传反馈给攻击器。

影响：本轮只完成CPU来源、Fixture和存储验证。真实RGB仍NOT_MATERIALIZED，H来源TEXT_SOURCE_VERIFIED不等于新文本单元已发布。Stage4C-2和全量物化须另行授权，现有历史特征重放状态不替代端到端验收。
