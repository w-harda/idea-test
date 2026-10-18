# 项目当前状态（唯一权威入口）

状态来自Git、实际代码与不可变实验记录；旧阶段报告与旧Commit中的PENDING/等待状态是历史快照。

## 当前阶段：SGA三模型正式Full及独立CPU审计完成，四方法结果已只读整理

报告与CSV内容Commit：a3a33be0cbb6c06fda7bdbdaf7a529b11aa583e6；本次文档提交补充发布索引。

冻结执行Commit6f89b799922ac54f041dabd3cd8e9d444735b146保持；三模型正式SGA Full和独立CPU审计PASS，监督器最终SUCCESS | SGA_THREE_MODEL_FULL_AND_CPU_AUDIT（2026-10-18 22:19:11 +08:00）。旧SGA开发/Smoke报告中的Full PENDING是历史快照。SGA/B3/G-v2/IRRA/RaSa/APTM算法、权重和旧结果均未修改。

独立结果分支codex/tbps-sga-full-results，基准6f89b79；报告与两个CSV唯一入口：[四方法三模型Full对比](framework/results/sga_three_model_full_20261018/REPORT.md)，同目录metrics_2dp.csv、full_precision.csv、evidence.json。模型IRRA→RaSa→APTM，方法Clean→AP-Attack B3→G-v2 Image-only→SGA-TBPS White-box，仅R@1/R@5/R@10/mAP，百分数两位小数。

按上述方法顺序mAP：IRRA66.10/40.43/32.28/0.54，RaSa69.18/39.99/35.05/1.46，APTM64.13/42.39/39.66/1.39。SGA四指标均最低，但它是对每个目标独立优化的白盒攻击；B3/G-v2在RaSa/APTM上是冻结IRRA训练攻击器迁移。不能标作等权限/等成本公平比较。SGA仍为image-only TBPS适配，APTM局部ITM代理不等于最终排名全链导数。

CPU标准库结果核验：12行48值、24历史控制值exact，完整3074/6156官方顺序/身份/Caption及三目标ModelSpace匹配；9222个SGA图像/Claim元数据与全部轮次记录通过。58711份实际读取文件SHA/stat不变。只读确认已有审计PASS，没有新GPU、模型、排名或重复Tensor缓存审计。

实际Full从17:27:02到22:19:11，墙钟4小时52分09秒；每目标156774视觉样本前向、文本0、模型加载1，模型摘要exact。Full来源tbps-framework-sga-whitebox-v2/full/{irra,rasa,aptm}/，原FAIL和历史资产保留。

当前任务完成；等待审查和下一Baseline VLPTAttack（ECCV2024/DRA）的独立授权，不自动开始适配或实验。

## 已完成：AP-Attack B3三模型Full及CPU审计完成，结果已只读汇总

报告与CSV内容Commit：0da6d30caf7a1224f9c734b3d607ea0843fb749d（后续仅补充发布索引）。

基准2047fb4c61f6ecfde32a148a593e87fe268462de；独立文档分支codex/tbps-b3-full-results。用户已手动执行共享GPU链，三个Full/独立CPU审计PASS，status.log最终SUCCESS | B3_THREE_MODEL_FULL_AND_CPU_AUDIT（2026-10-18 14:38:41 +08:00）。此前B3_STAGE_REPORT/B3_SHARED_RUN中的PENDING/尚未启动属于历史快照。

正式报告：[B3与G-v2三模型对比](framework/results/b3_three_model_full_20261018/REPORT.md)，同目录metrics_2dp.csv、full_precision.csv、evidence.json。方法顺序固定Clean → AP-Attack B3 → G-v2 Image-only，仅R@1/5/10/mAP，百分数两位小数，不新增指标或下降率。

mAP按上述方法顺序：IRRA 66.10/40.43/32.28，RaSa 69.18/39.99/35.05，APTM 64.13/42.39/39.66。G-v2四指标均低于B3；IRRA是同模型，RaSa/APTM是冻结IRRA训练攻击器产物的跨模型迁移。训练规模与监督来源不同，不能宣称等训练预算下优势。

本次仅标准库CPU读取与整理：9行36指标/24项历史控制数值exact、完整Manifest/ModelSpace、完成回执与五份Checkpoint SHA核验。82份实际读取文件SHA/inode/size/mtime不变；未启动GPU、重新生成、排名或重复独立CPU审计。Clean/G-v2为历史已审计Full显式复用，B3每模型实际新增3074图像编码，文本编码0。

旧源码、原共享运行树2047fb4、Checkpoint、generation_v3、B3像素及全部正式Full不变。等待用户审阅结果，不自行启动下一Baseline或新实验。

## 已完成：三受害者正式Full及独立CPU审计结果整理

2026-10-18，IRRA、RaSa、APTM的CUHK-PEDES Test正式Full及各自独立CPU审计均已完成。当前任务只读提取、来源验证、标量统计及文档发布，无新GPU/模型/攻击生成/排名/独立审计运行。

报告内容Commit：505942fcc1287ac07afc6400342628c6f9cb8ba0。

结果报告与原精度数据唯一索引：[三模型正式Full报告](framework/results/three_model_full_20261018/REPORT.md)。所有新导出固定模型IRRA→RaSa→APTM；方法固定Clean→H-v1 Text-only→G-v2 Image-only→G-v2 + H-v1 Joint，不改变原文件的历史保存顺序。

- IRRA：同训练来源模型评估，Full目录/home/lzf/ldx/outputs/tbps-framework-irra-materialized-evaluation/full/；GPU与CPU审计PASS，监督日志SUCCESS | IRRA_FULL_AND_CPU_AUDIT。
- RaSa：冻结IRRA训练攻击器的跨模型迁移，Full目录/home/lzf/ldx/outputs/tbps-framework-rasa-materialized-evaluation/full/；phase=SUCCESS、GPU/CPU审计PASS，监督器最终RASA_FULL_AND_CPU_AUDIT_SUCCESS。
- APTM：冻结IRRA训练攻击器的跨模型迁移，Full目录/home/lzf/ldx/outputs/tbps-framework-aptm-materialized-evaluation/full/；phase=SUCCESS、GPU/CPU审计PASS，监督日志SUCCESS | APTM_FULL_AND_CPU_AUDIT。
- 每模型Full为首次正式Attempt，参数/buffers摘要保持；G/H新加载/前向、Backward/Optimizer/JVP为0。独立CPU审计状态引用已完成真实日志，本轮没有重跑。
- 同一generation_v3物化输入3074图/6156文、G10/H7及Plan不变；三组Manifest、PID配对、有序成员、四类输入内容SHA和攻击来源exact。IRRA特征不跨受害者复用，RaSa/APTM保留原生融合评分。

当前结果：Joint五指标均低于各模型的单模态；相较G-v2 Image-only，Joint mAP额外下降IRRA 8.7291、RaSa 17.8443、APTM 9.1717个百分点。IRRA不能称作跨模型黑盒迁移。不同模型Clean是统一物化输入下的原生目标基线，不等同论文原始输入Clean；当前不推导跨数据集、多seed或因果性模块贡献。

## 本轮数据验证与发布边界

独立文档分支codex/tbps-three-model-full-results，基准525488a1a7de7f3258d71e195019c2bdc0a79e8f；工作树/home/lzf/ldx/projects/idea-test-tbps-three-model-results-20261018。

标准库CPU核验12行/60项指标、45项DR、三个完整有序Manifest及5份Checkpoint实际SHA。CSV和summary/result metrics一致；DR最大尾差1.4210854715202004e-14个百分点。2682个已读取来源文件SHA/size/mtime/inode保护核验通过；完整保护清单仅留服务器tmp，Git发布关键来源evidence.json和派生报告/CSV，不提交旧日志、模型或特征缓存。

导出metrics_4dp.csv、dr_4dp.csv、full_precision.csv和evidence.json；原实验文件、Checkpoint、模型/生成/评估源码均不修改。报告与导出排序/原精度/四位小数逐项检查通过。当前无阻塞；等待用户审阅结果，未授权新的训练/消融/模型或实验。

## 实现与历史索引

- 冻结Stage4D1 Plan及generation_v3 Image/Joint封存：见framework/STAGE4D1.md、framework/GENERATION_V3_GPU_SHARING.md及已完成生成/CPU审计回执。
- IRRA物化评估：b51d9e55190e96b6422f0c87952471ad6ac12bed，见framework/IRRA_MATERIALIZED_EVALUATION.md。
- RaSa实现及等待器：174eb51f4c346a6d74ace3f61f1ea170aeccf00b，见framework/RASA_MATERIALIZED_EVALUATION.md、framework/RASA_GPU_WAIT.md。
- APTM实现：2f21e910d9e560aa96ae29adcdb3d712c38dd4d6；自动链实现568eba1fd08407b7d6daabba8e47f1735c9cd98d、部署文档525488a1a7de7f3258d71e195019c2bdc0a79e8f，见framework/APTM_MATERIALIZED_EVALUATION.md、framework/APTM_GPU_WAIT.md。其中“尚未执行”保留原阶段快照属性，最新完成状态以本文及正式结果报告为准。
- Stage3B/4C/4D旧FAIL证据与历史源码/缓存永久保留，本轮没有改签或清理。
