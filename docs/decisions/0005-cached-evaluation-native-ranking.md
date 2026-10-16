# ADR-0005：准备产物与原生评分分离、历史重放不冒充像素来源

状态：接受（Stage 4A只完成CPU开发，Stage 4B未授权）。

背景：Gallery与Query数量不同，Caption/同PID正例不应zip配对。模型可能是双塔或融合结构，分数可能越小越好。原G/H生成策略不能获得目标评分。历史Full缓存只保存图像特征与原图身份，没有逐张对抗RGB内容SHA。

决定：

- Evaluation Engine只消费已准备、具有ModelSpace/InputBinding/实际payload摘要及缓存来源的表示，不加载/调用攻击策略或模型工厂。原生Adapter负责score及原生分块选择；PID和配对仅留在最后的检索指标层。
- 四模式选用两套Gallery和两套Query，保留Manifest原顺序及全部成员；不以配对关系筛选Query。native payload可为不同维度Tensor或局部Token/结构，不强制512维或余弦。
- 当前ranking复现官方IRRA all-same-PID/no-camera定义、CPU FP32、默认torch.argsort tie行为，指标百分数。lower_better须与Adapter声明一致，排序时取负；不按ID二次排序。
- 分块仅沿Query方向，每块覆盖完整Gallery，分数拼回完整有序矩阵后统一reduce；Adapter必须声明Query独立性和原生slice能力，不推测不透明payload的索引规则。
- 新输出使用原ArtifactStore不可变Manifest/回执，绑定特征/攻击/模型来源、原有序身份、ScoreRecord及指标协议。保存和读回score均clone，禁止复用底层大Storage。
- 旧Joint桥只核验元数据、原顺序和原生IRRA空间来源，明确feature SHA尚未核验，不重标新ModelSpace。真实重放需另行授权和特征/回执检查。
- 历史特征SHA不能替代缺失的对抗RGB SHA。Stage 4B若仅重放历史特征，必须单独批准并实现显式历史重放身份，记录pixel SHA未知及用途边界；不能填假hash或调用G补齐。

理由：保持原评测定义，阻断跨Victim/Tokenizer/内容的缓存混用，避免目标反馈进入攻击，忠实表达历史证据缺口。

影响：Stage 4A可执行小型合成CPU Fixture；目前没有生产级model-free IRRA cached scorer/历史Full加载器或统一真实执行CLI。原实验配置继续dry-run/拒绝直接执行。RaSa/APTM需独立原生Adapter和受审查能力接入，不能只改model_id复用IRRA特征。
