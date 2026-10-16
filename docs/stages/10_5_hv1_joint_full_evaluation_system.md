# H-v1 Phase 5：官方 Joint Full Test 评测系统开发报告

基准 a668b1c7149830f385b92224b896cbde1645d2a3，分支 codex/gv2-per-attribute-cache。本阶段只新增独立评测系统与CPU测试，不运行真实GPU、Test缓存生成、正式Full或真实Full CPU指标重算；不监控服务器正在执行的Phase4 Validation，不读取或猜测实际最佳H，不改动formal_training或其冻结来源。

## 交付文件

新增 prototypes/attribute_text_policy_h_v1_20261015/phase5_joint/：
- __init__.py：独立包。
- config.json：基准、62项既有来源SHA、固定G/IRRA SHA与历史mAP参考。
- common.py：固定隔离输出、源码/证据校验、复用旧G-v2官方评测模块。
- protocol.py：运行时验证Selection、Checkpoint链、Validation证据、官方Test身份及历史特征来源。
- storage.py：紧凑Storage、身份/计算意图/metadata/特征/回执SHA、原子提交与保留式恢复。
- backend.py：Caption-only H贪心输出、Unicode追踪、冻结IRRA文本编码与标量计数。
- evaluate.py：四组指标、基线检查、完整回执/执行记录、用户运行门、CPU独立审计。
- test_phase5.py：57项新增CPU tests。
- README.md：实际服务器预检、Selection解析、Full、恢复和审计完整命令。

另新增本中文阶段报告。Windows同步报告，服务器源码统一提交；任何缓存、模型、Checkpoint、outputs或日志均不提交。既有tracked源码逐项保持，未git pull或切换工作树。

## 协议与信息隔离

官方CUHK-PEDES Test固定3074唯一Gallery / 6156 Query，复用原官方reader/layout与IRRA rank。多Caption对应同一图，不按6156 Query重复生成Gallery。同PID全部正例，无camera过滤；FP32单位检索特征、384×128官方图像流程、77-token官方Byte-BPE不变。

比较Clean、Image-only、Text-only、Joint，输出完整精度mAP、R@1/5/10、mINP、DR_mAP和Joint−Image-only差。DR=(mAP_Clean−mAP_group)/mAP_Clean×100%，允许负收益。

全部Query保留，包括无属性、无合法字符、原文截断、H输入超长及H主动STOP/NOOP。资格失败固定原文NOOP；原文截断按官方77-token处理，禁止新增截断。NOOP特征逐值复用原文，不按攻击结果筛样本。

H接口只接受字符串Caption，Canonical/type/准确Span仅从Caption提取。PID/paired_index/Gallery不进入H，原文feature也不进入H，仅由外层用于NOOP复用。动作与IRRA反馈之间没有策略更新，指标只在全部文本固定后计算；无Test奖励、S_raw/JVP、targeted分支或按Test指标选Checkpoint。

Dynamic-K、选中属性IDs/Canonical类型、各字符Unicode码位/位置/UTF8偏移、Token变化、编辑次数、STOP、NOOP等有逐Query记录及总统计。资格固定NOOP的有效K0和H实际预测K在H_forward/eligible字段中区分。

## 动态Selection与冻结模型

不预先指定H Epoch。用户正式执行时：
1. 验证Phase4训练已完整提交、完成回执和全部Epoch Checkpoint链/来源/H/Adam进度。
2. 核对全部训练Epoch的独立128 Validation完成证据与来源；要求最终selection覆盖这些Epoch。
3. 验证selection内容摘要对应不可变文件名、声明未使用Test、最低Validation Joint mAP及最早visit并列规则。
4. 找到对应发布Checkpoint，核对SHA、schema、训练协议与架构；加载H权重后逐值/类型状态SHA一致。
5. Selection未完成、记录歧义或不匹配时显式失败，不等待、不轮询、不猜模型。

固定G-v2 E10 SHA：
f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e

固定官方IRRA SHA：
9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009

H与IRRA均eval/requires_grad=False/inference_mode，前后参数/buffers SHA一致；无optimizer/backward。首批H前向前设置与Phase4 Validation相同的FP32/attention后端开关，避免IRRA懒加载改变首批H的执行路径；配置开关不初始化GPU。Torch/Python/CPU线程数进入协议。

## 特征复用、预算与回执

只读复用已完成G-v2 Full：
- SHA锁定原Clean/Text bulk参考，核对官方全部身份与原文。
- 3074条Epoch10对抗图库feature，逐批核对原summary SHA、metadata、计算回执、Storage修复producer与严格L∞≤8/255。
- 保留全部原protocol、兼容记录、缓存和旧约90GiB大Storage文件，绝不覆盖或转换；避免逐条加载旧Clean/Text大view。

因此当前评测 **G新前向=0、IRRA image新前向=0**。历史E10 G=3074、Adv image=3074；Clean参考3074、原文text参考6156另列。两种文本组共享相应的Clean/Adv Gallery。仅H实际合法改变的文本新编码IRRA text，最多6156条，NOOP直接复用原文feature。

在加载H/IRRA及新文本编码前，先用官方指标核对历史Clean/Image-only五项。mAP参考66.09744262695312 / 32.28228759765625，沿用原0.001百分点阈值，失配立即停止，不擅自放宽。当前没有正式Text-only或Joint结果。

独立固定输出：
/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/phase5_joint/full/

每批features显式clone，load后也clone；计算意图、特征、metadata、receipt同目录fsync后原子提交。已发布单元不可覆盖，损坏/身份错误/未知pending拒绝。正常SIGINT/SIGTERM在批次或指标组边界安全停止，同命令恢复不重复已发布H/文本编码。

硬中断的同协议未提交批次，仅用户显式--recover-pending才整段归档至abandoned并保存文件SHA，随后允许无状态文本/参考重算；不涉及G新前向。已提交批次不重复，未知来源仍拒绝。失败/中断/成功attempt记录全部绑定completed SHA，区分累计硬件尝试和最终已提交编码数，保留所有证据。

summary与CSV完整精度一致；completed绑定summary、CSV、基线回执及所有attempt文件。CPU audit只读核验全部源/模型/身份/回执/动作/Tokenizer/历史feature逐值一致，并用缓存重算四组官方指标，无任何模型前向。

## CPU验证

最终合计 **193项CPU tests通过**：
- Phase5新增57项 + H-v1原核心68项。
- G-v2已验证评测与Storage兼容68项。

均设置CUDA_VISIBLE_DEVICES=""、OMP/MKL线程数2、PYTHONDONTWRITEBYTECODE=1，临时目录在/home/lzf/ldx/tmp。测试用合成12 Gallery / 24 Query和CPU动作/编码器fixture；新增H加载测试只保存并读取独立合成CPU H权重。不是正式Test特征、GPU Smoke或真实H/G/IRRA前向。

覆盖Selection全Epoch/最低mAP/并列/错误来源/禁止Test反馈，H文件SHA/schema/加载一致性与首批attention设置，Caption-only输入、准确重复Span/Unicode回放、非法编辑/Token变更、截断/超长NOOP，完整Query保留，3074/6156角色区分，四组指标与DR/负收益、CSV精度，基线异常在H前停止，紧凑Storage/回执/意图/身份损坏，安全停止/恢复/显式保留pending，完成重入无新模型、CPU只读审计SHA/mtime不变、attempt篡改拒绝及GPU门禁。

阶段日志仅位于/home/lzf/ldx/tmp/hphase5_release_tests.log、hphase5_legacy_tests.log，不提交。没有执行真实Selection解析、正式Full CPU preflight或Full审计，因为Selection可能仍在生成；当前选中模型未知。

## 用户命令与成本

[完整复制命令及停止恢复说明](https://github.com/w-harda/idea-test/blob/codex/gv2-per-attribute-cache/prototypes/attribute_text_policy_h_v1_20261015/phase5_joint/README.md)

按顺序由用户手动执行：
1. 可选无Selection源码CPU preflight，默认入口绝不GPU。
2. Phase4全部Validation/选优完成后 resolve-selection，再携带实际不可变文件做Full CPU preflight。
3. --action run --user-run-full，固定3074/6156，默认text-batch16。
4. 正常停止后同命令恢复；硬中断审查证据后才--recover-pending。
5. --action audit，CUDA_VISIBLE_DEVICES=""，独立CPU只读复算。

成功标志分别为PHASE5_CPU_SOURCE_PREFLIGHT、JOINT_FULL_CPU_PREFLIGHT、JOINT_FULL_EVALUATION_COMPLETE、JOINT_FULL_CPU_AUDIT PASS。最终CPU指标阶段GPU利用率0%可正常，日志明确打印每组JOINT_METRIC_COMPLETE。

成本仅估计：G/IRRA image新计算0，H CPU最多6156次、IRRA新text≤6156；粗略预留10–30分钟，batch16 GPU约1–3GiB，CPU RAM4–8GiB。四套原始feature约36MiB，动作/元数据另计，预留0.5–1GiB磁盘。没有本阶段GPU耗时、显存实测或攻击效果结论。

提交源码、测试和文档后停止，等待独立审查。全部真实GPU评测和Full CPU审计由用户亲自启动，不监控、不自动重试或调整正在运行的实验。
