# ADR-0004：冻结攻击器薄插件与独立评分边界

状态：接受（Stage 3A范围，真实G/H等价待Stage 3B另行授权）。

背景：原G/H算法、权重与正式结果已冻结。G训练监督依赖原IRRA配对评分/JVP；H测试只允许Caption及文本可提取属性。旧Joint评测后端直接输出IRRA特征，不能据此迁移到其他模型。

决定：

- 插件复用原模型/推理/资格和Unicode函数，不复制网络、训练循环或Loss。
- 插件描述与实例化分开；发现无模型副作用。训练和准备能力默认不可执行，不要求所有攻击具备train()/Optimizer。
- 公开输入只含图像或Caption，Manifest成员绑定独立；标签、配对、奖励不进入生成后端。
- G输出原RGB工作空间像素/扰动；H输出Unicode及完整动作。Joint只组合同一G/H组件，Gallery/Query处理独立。
- H固定原训练侧IRRA Tokenizer约束，未来victim Tokenizer只测量不优化动作；目标评分没有反馈通道。
- 绝对import通过显式、非重入隔离会话复用，只读锁定实际依赖和producer。真实权重加载默认拒绝，缺失/SHA/来源冲突不能绕过。
- 旧特征不重写、不重标为新模型空间；新攻击产物用新ArtifactStore独立不可变发布。

理由：保持方法忠实性、来源审计与Caption-only信息隔离；为不同维度、Tokenizer、尺寸和原生融合评分保留边界。

影响：Stage 3A只有CPU Fixture/只读检查；不能据此宣称真实G/H完全等价、可执行Full或跨模型效果。原统一配置仍无Evaluation Engine。未来迭代攻击新增独立插件/阶段权限，不更改G/H。

实现、命令和验证边界见[Stage 3A](../framework/STAGE3A.md)及[报告](../framework/STAGE3A_REPORT.md)。

Stage 3B记录：单次真实尝试在运行来源缺项处失败，未生成输出等价数据，状态仍PENDING。未改本ADR决定、Plugin算法或源清单；严格CPU预检须覆盖实际加载助手的传递导入。见[FAIL报告](../framework/STAGE3B_REPORT.md)。
