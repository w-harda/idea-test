# 项目当前状态（唯一权威入口）

状态来自Git、实际代码与不可变实验记录；旧阶段报告与旧Commit中的PENDING/等待状态是历史快照。

## 当前阶段：B3共享GPU后台链已准备，等待用户手动启动

实现及CPU验证Commit：d229e6ea5e9c2f5a764d4fbb297f8397c48e9fed（后续仅补充此发布索引）。

2026-10-18，基准097c96d9deec68c1bfc71b74a2fba5f98e3e81c1，独立分支codex/tbps-b3-shared-run，工作树/home/lzf/ldx/projects/idea-test-tbps-b3-shared-run-20261018。

用户已授权手动一次性启动Smoke→审计→Full→审计，无需在Smoke后再次确认Full；Codex本轮仅修改代码与CPU验证，未部署nohup、未查询真实GPU状态、未执行模型或生成攻击。当前正式运行状态PENDING_MANUAL_START，不是GPU轮询等待。

shared_observe_only：去掉瞬时利用率和Ollama存在限制；保留3072MiB最低启动空闲显存检查。全链只做一次GPU/其他B3进程快照；后续真实CUDA OOM/异常记录并停止，不重试、不干预Ollama。原run_b3_full.sh依次执行三个10+10 Smoke及审计，全部PASS后才执行三模型Full及审计；flock/一次性started/既有输出检查防重入，setsid独立进程组支持完整停止。

49项B3 CPU测试（17新增+32原有）和149项相关回归通过，6项旧配置路径测试按既定筛选未执行。Full严格CPU来源预检PASS，3074 Gallery/6156 Query、97批历史B3、CUDA=false。运行文档：[B3共享后台链](framework/B3_SHARED_RUN.md)。

只读复用原B3历史图像、generation_v3及三模型对照；G/H加载/前向0，无训练或重新生成扰动。新配置绑定新源码/Shell SHA，新结果将写tbps-framework-b3-shared-evaluation，旧097源码树、FAIL/ready产物不改。四项结果固定Clean → AP-Attack B3 → G-v2 Image-only，R@1/5/10/mAP百分数两位小数。

旧B3开发、训练来源与CPU产物证据见framework/B3_STAGE_REPORT.md、framework/B3_SOURCE_AUDIT.json；其中资源门禁PENDING为原阶段快照，本次运行政策以B3_SHARED_RUN.md为准。VLPTAttack官方身份已确认，本轮不扩展其他Baseline。

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
