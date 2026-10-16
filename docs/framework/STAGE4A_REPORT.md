# Stage 4A统一评测引擎CPU验收报告

Stage 4A开发/CPU验收完成；**不代表正式Full重放已通过**。基准0d90e79c08bf313b9085e75531a4226b3b266698，独立分支codex/tbps-framework-stage4a、工作树/home/lzf/ldx/projects/idea-test-tbps-framework-stage4a-20261016。实现Commit由PROJECT_STATE后续文档提交登记。

## 真实旧路径与逐项映射

| 历史实际源码/职责 | 新框架对应 |
| --- | --- |
| 官方utils/metrics.py::rank：默认torch.argsort降序，PID等值，多正例、无camera | engines.metrics.ranking；排序索引及FP32定义独立对照 |
| 官方Evaluator：文本/图库分开收集、各自L2归一化、qfeats@gfeats.T | 原生Model Adapter负责表示及评分；Engine分别消费Query/Gallery，不zip |
| G-v2 evaluation_phase29a/eval_metrics.py::metrics：每128 Query乘完整Gallery，再官方rank | Engine仅Query分块，Adapter原生score，拼完整分数后排名 |
| G-v2 eval_metrics.py::drops | drops，(Clean-Adv)/Clean×100，Clean=0时None |
| G-v2 evaluate.py：clean、epoch1/3/10共享原文 | 模式独立绑定相同clean_text，可分别使用固定攻击缓存；不加载/选择权重 |
| G-v2 eval_storage.py：逐条SHA、旧Storage兼容clone、gather | 本轮不调用旧Store/gather；新score保存/读取clone，旧桥仅metadata |
| H phase5_joint/evaluate.py::cached_results/baseline_metrics | Clean/Image-only/Text-only/Joint选用四个固定family，再统一排名 |
| H phase5_joint/storage.py::Store：Gallery64/文本16、intent/receipt/SHA、actions | 新ArtifactStore原子发布/回执；旧回执需Stage 4B独立只读检查，不改写 |
| H phase5_joint/protocol.py及common.py：原IRRA来源/预处理、原G2参考、H Selection | ModelSpace/InputBinding/AttackOrigin与明确旧协议索引，不重标cache |
| H Backend/TextAttacker | 不进入Evaluation Engine；策略输出先冻结，目标评分没有回流接口 |

官方ranking源码SHA为94910252a9e42f0820d49c11203b31d3b67951f58624f66abff4bf662500d5fb。测试从该官方文件独立加载rank，仅输入≤7×19合成矩阵；没有把新ranking调用两次伪装官方对照。源码pin由stage4a_legacy配置固定。

## 统一执行协议与能力

四模式按原Manifest有序成员独立组合，不用正确配对筛选Query。所有同PID图像为正例；paired_image仅验证原始来源，不限制检索正例集合。保留多Caption/NOOP与全部Query。PID及配对只在指标层，不传给原生评分器或策略。

Native Adapter负责实际score和Query切片，Engine不规定512维、77 tokens或余弦。CPU测试包括1/7/33维及无global维度的Fusion本地Token结构。higher/lower方向显式绑定Adapter声明；未知或冲突拒绝。当前生产read-only入口只接受CUHK/IRRA及受核验纯缓存Adapter标志，未实现RaSa/APTM，不能仅修改model_id执行。

所有指标为百分数；DR=(Clean-Adv)/Clean×100，Clean=0→null，允许负DR。mAP为全部正例精度平均后Query平均；mINP为正例数量/最后正例排名的Query平均；R@K为前K是否命中。拒绝无正例、错误PID命名空间、Gallery<10、非有限/非FP32/错误shape。

只沿Query分块，每块覆盖完整Gallery；全部分数拼接后一次统一FP32 reduce，不平均chunk指标。tie保持原torch.argsort默认行为，不改stable或加入ID排序；依赖相同Torch版本，不声称跨版本tie排序稳定。

## 绑定、存储与安全

- Dataset Manifest摘要包含官方split、原顺序、PID命名空间、Caption和配对；有序输入SHA独立绑定实际内容。
- ModelSpace包含Victim、Checkpoint、native代码来源、Tokenizer、预处理、评分协议和精度。
- BoundRepresentation记录独立内存payload摘要、原特征文件FileRef、原/对抗输入摘要域、G/H权重/协议/预算及代理来源。
- ScoreRecord绑定模型/评分方向、Query/Gallery顺序及两侧InputBinding；指标产物再绑定协议、身份和score摘要。
- 自动原子发布、不可变Manifest/回执、来源校验、pending保留；保存/读回真正clone，异常/篡改拒绝。
- Engine不加载模型、Tokenizer、攻击器、Checkpoint或旧Store；模块import不导入Torch。只有显式评分/新score存储调用惰性使用Torch。
- 本轮仅小型CPU Fixture执行；没有真实IRRA/G/H前向、GPU、训练、攻击数据生成或Full排名。

代码/API、CPU命令及输出布局见STAGE4A.md；相关决定见ADR-0005。

## CPU验收

**383项框架CPU测试=332历史+51新增；另36历史基础回归通过，合计419项。**

覆盖全部四模式、多Caption/多正例/独立数量与顺序、mAP/R@K/mINP/DR、任意维度/Fusion、低分优先、来源/内容/Tokenizer/ModelSpace隔离、缺项/错误维度/错误score顺序/无正例拒绝、分块/tie exact、无策略反馈、Safe-stop、不可变发布、中断pending、Storage紧凑clone和import无模型副作用。

随机小矩阵、非均匀多正例和全部tie三类与独立官方rank的索引及五项指标均**bitwise/数值exact**。不因历史Full目标数值调整排序、精度或容差。

日志：/home/lzf/ldx/outputs/tbps-framework-stage4a/{cpu_framework_final.log,cpu_legacy.log}。
历史保护：7991个既有文件SHA/mtime不变，原FAIL/成功Smoke/CPU修复回执及来源有效，原实验树及Stage 3B/3BR/3B2树HEAD/clean保持。没有修改prototypes、原始Dataset、Checkpoint、缓存、协议、训练或正式结果。新源码/文档之外输出均不提交Git。

## 历史Full桥实际检查与缺口

本轮仅读取G-v2/Joint协议、摘要/完成JSON及3条必要批次metadata示例；未读取features.pt。使用annotation独立恢复官方Test成员顺序，核对Joint的3074 Gallery/6156 Query、全部Caption/PID/paired_index及固定IRRA预处理/Tokenizer/BPE/ranking来源。

元数据索引：/home/lzf/ldx/outputs/tbps-framework-stage4a/legacy_metadata_index.json。
旧Joint协议digest：59ffb61a575cbeebf38e06bee5154b96f8967bb165b9b22305fad6c9cc967b48。
索引声明3472个feature相关文件，仅验证其索引SHA格式/路径及摘要绑定，**没有核验这些文件的实际字节**；status=METADATA_VERIFIED_FEATURES_PENDING、features_loaded=0、ranking_executed=false、replay_executable=false。不能据此宣称全缓存已再次审计。

重要缺口：G-v2 epoch10单条metadata只有text_samples/G_samples/IRRA_image_samples/linfs；Joint image_gallery只有reference_reused等来源统计，没有逐张对抗RGB内容SHA，原Full也未保存对抗像素。Feature SHA不是像素SHA，无法从feature可靠恢复像素内容身份。本轮未调用G补齐，也未伪造hash。

因此旧Joint特征**有条件适用于同一IRRA的历史特征重放**，但不能直接伪装成新引擎严格materialized攻击输入，更不能给RaSa/APTM使用。H缓存的真实adv Caption/Unicode动作可在未来授权后读取并计算其UTF-8 SHA；那仍不是新H生成。G-v2旧Text/Clean存在大Storage历史文件；不优先加载它们，不改写旧缓存。

## Stage 4B预案（未授权、未执行）

1. 首先由用户批准历史feature-only重放类型：显式记录adversarial_pixel_sha256未知，身份改以原图SHA+冻结G/H权重+原协议/计算回执+feature SHA绑定。它与严格RGB攻击输入分型，不能流入跨Victim编码器。当前require_replay继续拒绝。
2. 优先使用已clone紧凑保存的Joint Full四个family：clean_gallery、image_gallery、clean_text、adv_text。对应原协议、Selection、完成记录及本轮固定metadata FileRef；G-v2 Full仅作Image-only/Clean来源交叉核对，避免读取约90GiB旧Text/Clean大Storage。
3. 在新工作树增加纯CPU、model-free IRRA cached scorer及显式桥接加载器；逐批核验3472文件实际SHA、intent/receipt、rows顺序、shape=[N,512]、FP32/L2/有限性和原source协议。原Storage兼容记录/计算回执只读；任何pending/损坏停止，不删除或恢复。
4. 全部features.pt读取后立即detach().clone，按原Manifest构建3074/6156顺序。原gallery64/text16批次只负责读取；评分固定128 Query×完整Gallery，和旧eval_metrics路径一致。不加载IRRA/G/H/Tokenizer神经模型，不生成任何攻击。
5. 原生cached scorer归Model Adapter层，Engine仅评分矩阵/身份/指标；旧→新空间需显式重放映射和回执，不按512维直接认同ModelSpace。
6. 固定运行环境为历史Torch2.5.1+cu124、CPU2线程、FP32；四组逐组处理。18,923,544个分数约72.19MiB/组，int64排名约144.38MiB/组；4个feature表约36.06MiB。预先预算RSS≤2GiB、耗时≤30分钟，达到资源/停止边界保留新证据并停止，不改算法。CPU初估，非本轮实测。
7. **执行前固定阈值**：feature逐值exact；新/原相同128-Query评分路径max_abs_error≤1e-6；相同score矩阵官方/新rank索引必须exact；各组五项指标对历史对应summary绝对误差≤1e-5个百分点，DR绝对误差≤1e-5个百分点。遇tie/版本差异或任何超阈值停止，不临时改stable、阈值、舍入或排序。
8. 仅新框架命名空间保存重放协议、来源、score和指标回执；原completed/summary/缓存不改。该预案不是执行授权，本轮无Full重放命令。

阈值针对重放对应历史summary，而非更早其他精度参考结果。不能通过历史指标硬编码新输出；结果只能由新Engine计算。预算须在Stage 4B正式授权时再次确认，但不能根据输出误差事后放宽阈值。

## RaSa/APTM扩展边界

新增独立原生Adapter与经过审查的执行能力接入，支持局部Token/跨模态融合、原生score方向及Query切片语义；不需要改四模式和排名算法。真实支持登记前依旧拒绝执行，不把Fusion Mock成功称为RaSa/APTM已实现。

跨模型必须使用已冻结的真实对抗RGB/Unicode产物，再交给目标Tokenizer/预处理/score；目标分数不进入G/H策略。旧IRRA特征只能留在IRRA空间，不能作为目标特征。未来模型不同相关性/排序协议需要新版本声明，不能静默套用IRRA定义。

Stage 4A完成后提交推送并停止，等待审查；未开始Stage 4B或Stage 5。

本轮不可变CPU验收：/home/lzf/ldx/outputs/tbps-framework-stage4a/cpu_verified_v1/result/；Manifest SHA c489c1accc8ea08097dcf60229ff0db84d45cab799b4c5cc0e967eced80e6a95，保护记录SHA 162735d019ce739f9dc669b2376a420c66c4f94fd3f3332df18143bc7a2ac82e。仅CPU证据，不是Full重放完成回执。
