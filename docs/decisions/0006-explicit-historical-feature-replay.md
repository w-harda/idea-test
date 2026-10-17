# ADR-0006：历史特征重放使用独立身份，不放宽像素产物契约

状态：接受（Stage 4B-1）；真实Full指标重放未授权。

背景：ADR-0005确认旧Joint Full没有逐图对抗RGB内容SHA。已有特征、原输入身份、冻结G/H/IRRA权重、Selection、协议和完成回执可核验，但不能据此伪造像素身份。用户批准只读接入该历史IRRA特征，并要求与materialized攻击输入隔离。

决定：

- 新HistoricalReplayIdentity/FeatureBinding/Representation与InputBinding/BoundRepresentation分型；UNKNOWN像素身份只允许出现在历史重放契约，严格materialized输入继续要求实际内容SHA。
- 限定原IRRA/CUHK官方Test及固定G10/H7；原协议/完成回执/Selection/权重、完整有序成员、特征文件SHA和实际原/攻击文本SHA绑定新ModelSpace。
- 历史载入器只读原Joint紧凑缓存，读取即detach().clone；源SHA、批次身份/结构、H原文动作/Unicode/Tokenizer证据逐项核验。模型不恢复、神经动作不重新生成。
- 新Engine显式分支接入，原IRRA cached scorer负责原生点积，同一官方Ranking/指标循环负责检索。产物明确historical_feature_replay、UNKNOWN像素及not_new_attack，不能冒充RGB/新攻击。
- METADATA_VERIFIED、FEATURES_VERIFIED、METRICS_REPLAY_VERIFIED分开；特征通过不意味着真实指标已重放。默认真实score拒绝，核验CLI没有Full评分选项。
- Stage 4B-2资源/环境/数值条件预先冻结，不因差异事后改容差或Ranking。

理由：忠实表达历史证据能证明和不能证明的范围，保留新框架强身份约束，避免历史特征跨Victim/Tokenizer混用及目标反馈回流攻击。

影响：本阶段可CPU检查真实特征和小型合成排名。真实Full排名另行授权；RaSa/APTM需要独立原生编码和评分，不能利用本桥重标IRRA特征。旧缓存/协议/源码保持原样，Stage 4A文档作为当时快照保留。
