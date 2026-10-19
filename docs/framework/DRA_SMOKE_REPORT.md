# VLPTAttack（DRA）IRRA Source最小接线与真实Smoke报告

## 结论与边界

官方DRA完整攻击、固定8图/16句三模型原生评分及独立CPU审计PASS；OFFICIAL_ALGORITHM_FIDELITY_PASS仅覆盖固定环境和该小样本。
本阶段没有3074图Full、训练、新Loss或Target反馈。Source生成一次已提交攻击产物，三Target共享；不能把此Smoke指标当正式Full效果。

开发基准715f636cba0c9454b199052cd7cc2444a66a8691。
初始接口/FAIL代码230ff9eeca5619ac04d151d83b6f02fdfeb33ecc；成功Source代码8513a49b6f53c63ea98a8f6795b4e659b8fb88de；成功Target代码d455f3a05797451dd4d2ac145ad03b58090d3fa2。
最终CPU审计仅在派生树补json导入，所有已执行Source/Target文件与配置路径保持原样。当前发布分支codex/tbps-vlpt-official-complete；具体提交在PROJECT_STATE登记。

## 接线与算法保留

直接导入/home/lzf/ldx/projects/VLPTransferAttack-official/RAttacker.py，Commit 91f78325e7cf0c497c9bbb5be2215be652d8cd9d，SHA 75518cb9465d9f8e994ea2f59c86b89fd4c39e3fe74f66b7a1d0a090eec2d313。
原文件及Git diff均未改；不复制、覆盖、继承或Monkey Patch官方攻击函数。
原三阶段、8/255、10轮、2/255步长、整数三角采样、5候选、五尺度、MLM/KL、单原词槽位替换及.6/.2/.2全保留。详见DRA_OFFICIAL_FIDELITY_AUDIT.md。

Source Wrapper仅复用IRRA原生单位特征及已验证BERT→文本→Byte-BPE桥，开启参数梯度满足官方候选二次backward；eval、无Optimizer、参数/buffers前后exact，攻击结束清理梯度。CPU关闭参数梯度反例真实复现原backward错误；开启后真实官方流程通过。

## 数据与Tokenizer

CUHK官方Test完整CPU核验3074/6156，3073图各2 Caption，特殊cuhk:38920:0有10 Caption，原配对与顺序完整。不按PID扩展攻击Caption集合。
GPU样本为前8 Gallery cuhk:37132:0至cuhk:37139:0及全部16 Caption，运行前锁定，不按效果选取。
pre_caption30、BERT77、IRRA77、cls=False分别固定；1026行桥接全部WordPiece往返exact，BERT截断0、IRRA额外截断0，756行包含词重要性[UNK]。实际可见内容中的额外PAD/CLS/SEP/MASK均0。
数据预处理造成的Raw/Matched差异与攻击替换分开，Target Clean使用Matched输入。

最终16条均替换1个原词槽位，NOOP0；1条原词leggings替换成官方候选green and wearing，空格词数增加2。这是官方多WordPiece候选行为，没有过滤、改写或重新挑选。完整原槽位/最终槽位/action保存在Source metadata。

## 实际调用、资源与预算

已提交Source：IRRA加载1、MLM加载1，image_samples1144、text_samples1026，MLM4次/80样本；官方Attacker.attack1、文本调用2、图像调用1、区域rand3Num45、图像backward100、图像梯度观察55次均有限且非零。
T1实际特征与图像优化监督逐值一致；最终文本监督三组齐全，官方权重.6/.2/.2。
每个成功Target加载1次、图像16样本、文本32样本；RaSa/APTM各512次融合pair，原生评分不替换为IRRA余弦。
Source区域候选选择时allocated约1.535–1.538GiB，未出现随9轮持续增长；reserved预热后稳定。仅一批，不能推广为全量长期稳定结论。

| Stage | seconds | allocated GiB | reserved GiB | peak RSS GiB |
|---|---:|---:|---:|---:|
| Source | 133.383791 | 7.2386 | 9.1914 | 2.2778 |
| irra | 21.643366 | 0.9008 | 0.9785 | 2.0696 |
| rasa | 26.563948 | 1.8555 | 2.0332 | 3.8912 |
| aptm | 18.051904 | 0.9703 | 1.0488 | 2.1614 |

完整攻击含被动追踪122.218629秒；官方ImageAttacker外层计时28.592292秒；整个Source会话133.383791秒。无OOM。
最大实数L∞=0.03137257695198059；数学8/255=0.03137254901960784；差2.793237274811844e-08。
官方FP32上下界检查通过，不声称严格实数上界通过，也没有增加二次投影。像素为FP32[0,1]，original/adv/effective_delta紧凑保存并bitwise读回。

## 真实Smoke评分（百分数，仅链路诊断）

| Target | Mode | R@1 | R@5 | R@10 | mAP |
|---|---|---:|---:|---:|---:|
| IRRA | Clean | 100.00 | 100.00 | — | 100.00 |
| IRRA | Text-only | 100.00 | 100.00 | — | 98.78 |
| IRRA | Image-only | 0.00 | 50.00 | — | 27.35 |
| IRRA | Joint | 0.00 | 68.75 | — | 29.11 |
| RASA | Clean | 100.00 | 100.00 | — | 98.96 |
| RASA | Text-only | 100.00 | 100.00 | — | 100.00 |
| RASA | Image-only | 100.00 | 100.00 | — | 96.70 |
| RASA | Joint | 87.50 | 100.00 | — | 93.58 |
| APTM | Clean | 100.00 | 100.00 | — | 100.00 |
| APTM | Text-only | 100.00 | 100.00 | — | 100.00 |
| APTM | Image-only | 100.00 | 100.00 | — | 98.96 |
| APTM | Joint | 100.00 | 100.00 | — | 98.96 |

仅8张Gallery，R@10不适用，不补图、不改通用Full门禁。IRRA Smoke中Joint mAP高于Image-only，不据小样本预设联合一定更强，也不据此修改算法。
RaSa/APTM使用原生融合评分、独立ModelSpace和缓存；三者绑定同一Source Collection内容及原始成员。

## 失败与修复记录（全部保留）

1. 新环境缺少safetensors，Source第一次会话IRRA恢复1，MLM0、官方攻击0；补齐权重读取依赖后使用独立v2输出。
2. v2原官方攻击完整执行，但外部保存审计把单原词槽位替换误判为“空格词数必须不变”，拒绝发布；无已提交像素。保存原代码和FAIL，在派生树只读取官方change/final_words来审计，不修改任何攻击决策。v3为唯一已封存Source。
3. v3第一次IRRA已完成四组编码，因会话缺少G_H_generation/ranking_feedback_to_attack字段被原缓存门禁拒绝。旧FAIL/特征保留，独立irra-session-v2补协议并重新完成原生IRRA编码/冻结核验；Source不重跑。
4. 独立CPU审计初次因缺json导入退出；派生树仅补import，原Source/Target代码/配置不改，再次CPU审计PASS，没有新的模型前向。

全任务累计Source IRRA恢复3次、MLM2次、官方完整攻击调用2次；只有最后一批8图/16句提交并用于全部Target。
IRRA Target累计恢复2次，图像32/文本64；RaSa/APTM各1次。早期失败Source未落盘全部神经样本计数，不将最终成功计数冒充全任务总数。

## CPU与保护

26项新增DRA CPU测试、180项共用组件回归通过。1项旧APTM固定绝对路径配置测试在原715f636基线树也同样失败，显式排除；未修改旧测试/配置或关闭新DRA来源检查。
CPU独立审计DRA_INDEPENDENT_CPU_AUDIT_PASS smoke，校验全部Score与summary、CSV、源像素预算/有效delta、顺序、多正例、来源和无pending。
18924个原历史源码/SGA Full及控制文件stat保持；19个早期FAIL文件、42个已完成Source/首次IRRA FAIL文件及174个全部GPU产物分别SHA/size/mtime/inode保持（集合有交集，不求和冒充唯一文件数）。
官方DRA及SGA仓库HEAD/diff保持；旧G/H/B3/SGA/三模型Full不修改。本轮GPU进程已全部退出，无后台任务。

## Full准备与实测成本

最小算法接线已具备，正式Full尚未授权且CLI没有Full选项。后续仅增加范围/调度，不改DRA。
推荐官方Batch8、FP32、10轮、8/255、2/255、5候选、seed42；Source只初始化一次，各Target串行独立进程。
保留至少20GiB空闲GPU、48GiB可用RAM、35GiB磁盘。三个FP32图像Tensor的纯数据约5.066GiB，另有特征/Score及日志。
按本次含详细被动追踪单批122.22秒线性估计Source约13小时；参考已完成原生Target Full约5小时，整链暂预留18–22小时，非6–9小时保证。
首8图Caption偏长、WordPiece候选和Python只读追踪影响耗时；没有为速度近似计算或删官方步骤。10-Caption batch显存、后续长候选上下文、长期内存与Full指标尚未验证。
不自动开始Full或新开发。

## 证据

- Source：/home/lzf/ldx/outputs/tbps-framework-vlpt-irra-source-v3/smoke/source/
- Target：/home/lzf/ldx/outputs/tbps-framework-vlpt-irra-source-v3/smoke/irra-session-v2/、rasa/、aptm/
- CPU审计：/home/lzf/ldx/outputs/tbps-framework-vlpt-irra-source-v3/smoke/audit/completed/audit.json
- 日志：/home/lzf/ldx/tmp/dra_official_source_smoke_v3.log、dra_native_*_final_smoke.log、dra_independent_cpu_audit_final.log
- 来源SHA、原精度Smoke数据及冻结摘要：DRA_SMOKE_EVIDENCE.json。
