# SEP Text-only 三数据集 × 两 Victim

本轮执行协议为 SEP-TBPS-SRefPseudoPID-S1。Source 为固定 CUHK-PEDES IRRA 和完整 34,054 张官方 Train 参考图像，按 Train PID 聚合，对每条 Caption 固定 Source Clean Top1 伪 PID。

复用既有 PIDPlan 生产实现、官方同义词/POS、DPP、候选优化、tau 选择与原生评分。176 项历史 CPU 测试和真实模型矩阵 Exact 已有通过证据，本轮只验证受修改接线。既有 Source-only Smoke 在 Landlock 下初始化 CUDA 失败，本轮改用 Caption-only 输入投影与生成进程文件审计钩子，拒绝读取测试标注、评测身份产物及其他模型权重。

单代理 N/E=1/1，官方 100 个随机权重和 DPP 调用照常执行，权重退化为 [1]，没有多代理多样性。本协议属于 SEP-inspired 检索适配。CUHK IRRA 同时也是 Source，该行属于 Source 可访问模型评测，RaSa 与跨数据集 IRRA 均在攻击文本冻结后独立评分。

每 Caption 使用其不透明键派生的固定 Seed，便于合法复用本轮 Smoke 的冻结产物。官方常量/非有限重要性缩放循环明确抛错停止，不添加 NOOP 或替代候选。正常长耗时没有停止门槛。

三个数据集按文本长度与原顺序固定选 8 条 Smoke，选样不访问模型效果。每组攻击生成一次，IRRA 与 RaSa 共用文本。Smoke 六组评分与独立 CPU 完整性检查均通过后自动进入 Full。

Full 尺寸为 CUHK 3074 Gallery / 6156 Query、RSTP 1000 / 2000、ICFG 19848 / 19848。评分仅 Text-only，原生 RaSa 全 Gallery Top128 和 ITM fusion_batch16、query_chunk32、image_batch1、text_batch8保持。Source 内部 Clean Caption 预测只用于冻结伪 PID，未执行 Target Clean 或 Matched Clean 评测。

共享 GPU 显存和利用率只记录，无最低空闲显存门槛。真实 OOM、数值、算法、来源错误首错停止，不自动重试。新旧现场独立。

运行状态以新控制目录回执为准，正式 Full 结果只有在完整 Query 覆盖及独立 CPU 指标核对通过后才填入。

输出目录为 `/home/lzf/ldx/outputs/tbps-framework-sep-text-only-full-20261105`。
控制目录为 `/home/lzf/ldx/tmp/sep_text_only_full_20261105`。
