# G60/H60官方Test结果找回与身份核验

核验日期：2026-10-24；参考Commit：a820278eff5ccd79ba4b90de833e77bdbab022d3。

## 核心结论

**检索范围内未找到指定G60+H60的官方Test Full。**
已找到G60/H60训练完成及独立CPU审计、IRRA的G60+H60内部Validation、三模型G60 Image-only正式Test。
Epoch Sweep三模型Joint Full明确绑定H7，而非指定H60；不能改名充当G60+H60。缺少的是H60 Test攻击文本及评分/审计的可验证产物链，不只是一个漏写的汇总报告。

本轮仅CPU读取、检查既有Manifest/CSV/summary/Engine指标并生成派生文档；没有CUDA、神经前向、训练、攻击生成、排名重算或重跑已有CPU审计。被新任务替代的Pilot等待器未部署、Pilot未启动。

## 1. Checkpoint身份与原训练协议

| 模型 | 真实身份 | SHA-256 |
| --- | --- | --- |
| G60 | completed_epoch60、next_epoch61；累计Adam484080 | 9148a6ac06178b2eb474194ef6892c50182c5623b4e88bdcf88faa358aa46049 |
| H60 | 固定G60奖励图库适配；visit122880、累计Adam122220 | c0de26658c5f36083aaeb04dde3948cdd5bf941d30b1b5c97f7bb2c796404094 |

G60：/home/lzf/ldx/outputs/tbps-framework-gv2-epoch60-ablation/train/checkpoints/epoch_0060/checkpoint.pt

H60：/home/lzf/ldx/outputs/tbps-framework-hv1-g60-adaptation-epoch60/training/checkpoints/visit_00122880/checkpoint.pt

G60由原Epoch10继续训练11–60，父Checkpoint SHA为f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e。保留Image-only G(I)、原K+1 Global/Local/Retain、Adam2e-4/(0.5,0.999)和8/255投影；已有训练CPU审计PASS。
H60由H10 visit20480/20370更新恢复，在固定G60的2048 Train奖励图库上新增50 Epoch、102400 visits、101850更新；不是从头在G60上训练60 Epoch。原Adam1e-4/(0.9,0.999)、SCST R_sim+R_rank、Natural-K及Unicode规则不变；已有HV1_G60_TRAINING_CPU_AUDIT_PASS。
本轮重新计算两个文件SHA，并仅用CPU加载Checkpoint头部，没有创建模型或应用CUDA RNG。

H60原监督器在报告写作阶段保留FAIL exit=1 line=50；其训练完成JSON及训练CPU审计均PASS。该旧控制FAIL不等于训练失败，也不是Test Full的状态。本轮不修改或重新发布旧日志。

## 2. 官方Test主表：真实数值与缺失状态

单位百分数，保留两位小数。—表示未找到指定H60正式Test指标，不代表0、NOOP或推算值。

| Target | Method | R@1 | R@5 | R@10 | mAP |
| --- | --- | --- | --- | --- | --- |
| IRRA | Clean | 73.15 | 89.88 | 93.71 | 66.10 |
| IRRA | G60 Image-only | 21.05 | 37.88 | 45.52 | 17.51 |
| IRRA | H60 Text-only | — | — | — | — |
| IRRA | G60+H60 Joint | — | — | — | — |
| RaSa | Clean | 76.22 | 90.29 | 94.09 | 69.18 |
| RaSa | G60 Image-only | 33.12 | 50.49 | 58.46 | 26.47 |
| RaSa | H60 Text-only | — | — | — | — |
| RaSa | G60+H60 Joint | — | — | — | — |
| APTM | Clean | 74.61 | 89.26 | 93.18 | 64.13 |
| APTM | G60 Image-only | 42.72 | 62.00 | 69.95 | 32.11 |
| APTM | H60 Text-only | — | — | — | — |
| APTM | G60+H60 Joint | — | — | — | — |

三模型正式Test Manifest逐项一致：3074 Gallery、6156 Query、1000 PID，official_split=test。IRRA为IRRA Source同模型；RaSa/APTM为冻结G60跨模型迁移。
Image-only不调用H，因此可以从Epoch Sweep独立提取；其Joint行绑定H7，不能改签H60。
主METRICS两个CSV只含6条真实Clean/Image-only结果，RESULT_STATUS.csv记录全部12条请求状态。原精度CSV、summary与对应Engine指标逐值一致。

正式来源根目录：/home/lzf/ldx/outputs/tbps-framework-gv2-epoch-sweep/full/epoch_0060/
每个Target读取completed/metrics.csv、completed/summary.json、result/metrics.json、execution/report.json、cpu_audit/report.json及其Manifest。三份已有执行与CPU审计均PASS，均编码3074张，未生成新G/H攻击或重编码文本。
Clean来自各模型历史materialized-evaluation/full，与Epoch Sweep只读引用精确相等；对应原后台成功日志存在。

图像预算8/255，FP32 RGB不重新PIL量化，Image Batch1、Query Chunk128、fp32/math/no_tf32；保留原生Tokenizer、预处理和评分，RaSa/APTM使用Full有效Top128融合流程。
受害者Checkpoint、Tokenizer、预处理及评分SHA在SOURCE_EVIDENCE各model.protocol.model_space中。

## 3. G60 Image-only相对各自Clean的下降

百分点=Clean−Attack；相对下降率=(Clean−Attack)/Clean×100%。全部从原精度计算，DROPS.csv保留四指标结果。

| Target | R@1下降百分点 | R@1相对下降% | mAP下降百分点 | mAP相对下降% |
| --- | --- | --- | --- | --- |
| IRRA | 52.10 | 71.22 | 48.59 | 73.51 |
| RaSa | 43.10 | 56.54 | 42.72 | 61.74 |
| APTM | 31.89 | 42.74 | 32.02 | 49.93 |

IRRA mAP由66.10%至17.51%，RaSa由69.18%至26.47%，APTM由64.13%至32.11%。这些只支持G60 Image-only效果，不证明H60的Test攻击能力。
H60 Test Text-only/Joint缺失，无法计算其下降或Joint相对单模态的额外下降。

## 4. 真正找到的G60+H60指标：仅128内部Validation

来源：/home/lzf/ldx/outputs/tbps-framework-hv1-g60-adaptation-epoch60/validation/epoch_0060/completed/summary.json

| Target | Method | R@1 | R@5 | R@10 | mAP |
| --- | --- | --- | --- | --- | --- |
| IRRA | Clean | 100.00 | 100.00 | 100.00 | 100.00 |
| IRRA | G60 Image-only | 39.84 | 59.38 | 65.62 | 48.81 |
| IRRA | H60 Text-only | 98.44 | 100.00 | 100.00 | 99.22 |
| IRRA | G60+H60 Joint | 35.16 | 51.56 | 55.47 | 42.87 |

Gallery=128、Query=128，属于训练集内部Validation，不是官方Test；没有对应RaSa/APTM评估。
Joint R@1原值35.15625、mAP原值42.874080657958984；相对Image-only额外下降mAP5.93个百分点，相对Text-only额外下降56.34个百分点，仅对本Validation成立。
另存VALIDATION_REFERENCE.csv，绝不混入主Test CSV，也不能据此推断官方Test或迁移表现。

## 5. 容易混淆的历史Full与有限可比项

| Target | G60+H7 Joint R@1 | G60+H7 Joint mAP | 身份 |
| --- | --- | --- | --- |
| IRRA | 15.72 | 13.08 | G60 + H7 visit14336，非H60 |
| RaSa | 17.53 | 12.57 | G60 + H7 visit14336，非H60 |
| APTM | 31.89 | 23.74 | G60 + H7 visit14336，非H60 |

H7 SHA为487f3061b6cba2ddabccdd49745a7b99c8c549f4541781b8814b299ef5f279df，与H60不同。历史G10+H7 Full也不是本次目标。
仅就独立于H的Image-only部分，Manifest、Raw Clean、原生模型输入/评分协议一致，可以描述G10与G60的变化：

| Target | G10 Image-only mAP | G60 Image-only mAP | 额外下降百分点 |
| --- | --- | --- | --- |
| IRRA | 32.28 | 17.51 | 14.77 |
| RaSa | 35.05 | 26.47 | 8.58 |
| APTM | 39.66 | 32.11 | 7.55 |

这是图像单模态比较，不是G60+H60与G10+H7 Joint的比较。历史身份参考表另存HISTORICAL_IDENTITY_REFERENCE.csv。

## 6. 检索范围与来源边界

搜索了/home/lzf/ldx/outputs/、/home/lzf/ldx/projects/、/home/lzf/ldx/tmp/，包括现有结果CSV、summary、配置、协议、报告与控制日志；以H60精确SHA、visit_00122880及G60/H60字符串追溯。
结果导向遍历10270个目录、读取55332份文本、索引662个metrics/summary候选；读取异常0项，详情在SOURCE_EVIDENCE。
剪枝依赖、逐图/逐文本Tensor单元、内容寻址二进制缓存及本次自身派生目录；大文本列表保留，不据未读取的缓存成员宣称全盘不存在。此前较宽枚举进入旧Baseline逐图单元后已停止本轮自身CPU检索进程，未干预任何GPU或实验进程。
全部相关Git结果对象、日志提交和分支已核查；GitHub实时70个分支与本地origin引用全部一致，未发现未获取的新结果分支。H60相关已有提交对应训练、Validation与事后分析，没有找到另一份已提交而未合并的Test Full报告。

结论为检索范围内NOT_FOUND，不证明服务器任意外部位置绝对不存在。路径、SHA、远端分支快照与候选结果索引见SOURCE_EVIDENCE。

## 7. 若后续补齐，具体缺少什么（本轮均未执行）

1. 核验/物化指定H60在官方Test全部6156条原Caption上的Unicode攻击文本、完整动作、来源及合法性回执；本轮未找到完成集合。
2. 复用已封存的G60官方Test Gallery，在同一Victim原生协议下进行H60 Text-only及G60+H60 Joint Full评分。
3. 完成对应CPU审计及封存；严格匹配的Clean/G60 Image-only可继续只读复用。

这些步骤需要新的明确授权。本轮不补跑、不训练、不重新生成或评估。

## 8. 派生文件与保护

METRICS.csv保留6条已核实Test原精度；METRICS_2DP.csv为其两位小数展示。RESULT_STATUS、DROPS、VALIDATION_REFERENCE及HISTORICAL_IDENTITY_REFERENCE分别记录缺失项、合法下降统计、内部Validation和历史身份。
SOURCE_EVIDENCE保存两个Checkpoint、训练协议、已有审计、结果/日志SHA以及检索范围；不提交Checkpoint、Tensor、原Caption数据或缓存。

本轮检查已消费来源文件的SHA、size、mtime保持不变。151378项历史保护PASS来自原H训练CPU审计，本轮未重跑该全量审计，不能冒充本轮新验证。
