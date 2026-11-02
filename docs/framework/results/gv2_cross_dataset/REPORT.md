# G-v2 E60跨数据集迁移：执行准备

状态：CPU源码与接口回归已通过，真实CPU Preflight、两目标Smoke、独立CPU审计与Full尚待单次Supervisor执行。此准备报告不是实验成功报告，最终由仅接受完整审计PASS的报告器更新。

范围：固定CUHK训练的G-v2 E60，RSTPReid/ICFG-PEDES各自官方IRRA，原始Caption，只有Clean/Image-only两模式。参考Commit 6021f3bfdec181785d1baccd7ce4f80a1428b069。G网络、generate_once、8/255投影、原生图像预处理和完整PID排名源码字节保持；新代码仅泛化数据身份、输出和执行调度。

## CPU验证

112个不同CPU用例PASS：新24个与相关旧88个。新工作树主回归111 passed、1 deselected；唯一依赖原工作树绝对路径的历史missing-runtime-producer测试，在其原冻结工作树单独1 passed，未修改历史测试或其配置。原始运行日志保存在独立控制目录。

新增覆盖：官方顺序/Caption/PID身份漂移拒绝、固定Smoke正例覆盖、完整两Smoke审计门禁、CPU禁CUDA/神经前向/Backward、零空闲VRAM不拒绝、历史stat改变拒绝、模式/预算/Epoch/Batch/Victim错误拒绝，以及真实E60严格CPU恢复。真实G/IRRA GPU Smoke仍PENDING，不用CPU结果替代。

## 执行方式

单次nohup+setsid、独立工作树/配置/控制/outputs。CPU_PREFLIGHT → 两目标SMOKE/SMOKE_AUDIT → RSTP Full/审计 → ICFG Full/审计 → 最终报告。失败保留且停止，不自动重试或降精度/Batch Size，不干扰外部C-PGC/SEP。

命令和状态见README.md。大型FP32原图/对抗图/完整矩阵/排名索引仅保留在服务器；Git发布代码/配置、报告、小型指标和来源/审计摘要，不发布数据或权重。
