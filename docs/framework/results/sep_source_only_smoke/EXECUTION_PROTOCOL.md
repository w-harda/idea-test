# SEP-TBPS-SRefPseudoPID-S1 执行协议

固定独立 Smoke，基准 `6db21efa95cc70d738681335949597456de49844`。旧 SEP 两代理执行目录、44 条已发布文本和第 45 条 Claim 只读。本协议独立于旧两代理含 Victim 和 Test GT 的适配。

## A 类真实分数验证

按配置冻结 CUHK 两条真实 Caption 及 128 张测试图片，从两条原文与第一条首尾词删除构造四个固定文本。每模型只计算一份 4 × 128 的真实矩阵，RaSa 每个文本执行原 Top128 和既有融合 batch 16。独立 CPU 子进程将同一份 FP32 分数送入旧 Git 定义和现行生产 classifier，校验 PID 顺序、成员、聚合、softmax、官方 margin、DPP、候选排序和缓存访问顺序。只接受逐位相同，记录第一个差异并停止。真实矩阵是固定候选排序试验，CPU 原算法回归另覆盖实际候选搜索。旧 IRRA Loader 的原生混合参数保持，图像输入、输出特征和评分维持既有 FP32 路径。

## Source-only

- 参考图库固定为 CUHK-PEDES 官方 Train，按官方标注顺序保留全部 34,054 张图、11,003 个 Train PID。逐图 SHA、标注来源和有序清单摘要在 CPU Preflight 写入不可变输入，GPU 读取时再核对图片 SHA。
- 生成器只接收每条原始 Caption 和不携带身份信息的不透明键，以及冻结 Source IRRA 和 Train 参考库。Python 生成子进程进入 Linux Landlock，针对 Train 图片逐文件授权，授权目录不包含目标标注、权重、测试图或评分输出。真实资产拒绝探针在独立 CPU 进程执行。预备与评测进程拥有各自的元数据，生成代码既不接收这些字段也无法打开其文件。参考隔离审计覆盖本机文件访问边界。
- 原作者 Caption 清洗用于 SEP 优化。原生 IRRA 对完整 Source 库评分，按 Train PID 取 max，再以 Clean Top1 的 first-seen PID 索引固定伪标签。原 SEP 自检接口仅查询同一 Source 的 argmax，一开始必定与伪标签相同，后续调用只用于原代码的 Source 自检统计。最终文本来自原 gamma=.02 分支的第一个 tau 候选。无编辑时原始 Caption 逐字保留。评测的 Clean 为原始 Caption，有编辑时 Text-only 为 SEP 输出，含既有清洗差异。
- 单代理 N/E=1/1，原 100 个随机权重归一化与 DPP 调用保留，最终只有权重 `[1]`，没有代理多样性。原 POS、top50 同义词、RNG、替换与删除、剪裁概率 margin、T=10、patience=5、gamma=.02、初始预算公式和最终 0.15 词预算仍由官方函数执行。固定 Source Train 伪 PID 及单代理均是协议差异。该协议是 SEP-inspired TBPS 变体。 [SEP 官方代码](https://github.com/ZevineXu/SEP-Attack/tree/3a6d848a48097aee79be0299da8959e78db543c7) · [SEP 论文](https://arxiv.org/html/2605.24958v1)
- 每目标两条真实 Caption，选样在 GPU 前锁定，只使用文字长度、POS 数量、词汇可用性和旧 Query 排除。较短 Caption 覆盖低预算边界，自然 NOOP 是否出现由运行结果决定。独立 CPU 回归已通过原官方搜索验证编辑与 NOOP 两种路径；真实超时直接终止并保留 Claim。未引入零重要性回退。每条上限 1800 秒，真实前向按原 batch 运行。

## Target 和审计

Source 全部 6 条发布并冻结后才允许四个 Target 评分子进程读取文本。同一份 CUHK 产物交给 RaSa、APTM，RSTPReid 与 ICFG-PEDES 使用各自原生 IRRA。每目标 2 Query × 固定 128 Gallery，Clean、Text-only 独立评分，APTM 各模式分别运行原生全矩阵归一化。独立 CPU 审计重核来源、词预算、Tokenizer、伪标签、缓存流和指标。样本规模仅用于链路验证。

## 执行边界

Supervisor 串行执行真实分数采集、CPU Exact、Source 生成、四组 Target 评分、CPU 审计。共享 GPU 资源只记录，无固定显存门槛。任何真实错误停止，保留新旧现场。运行及发布目录见 `configs/framework/sep_source_only_smoke.server.json`。没有 Full 入口。
