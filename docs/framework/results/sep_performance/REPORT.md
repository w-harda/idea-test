# SEP WordBudget1：CPU/GPU性能诊断与独立无损优化

2026-11-06。分支codex/sep-performance，基于逐组报告快照Commit d5550d55f288b2e3529476912c02c3fce552121c。本轮只诊断、独立实现与小规模真实GPU验证，不替换当前六组Full。

## 结论

已确认主要瓶颈是CPU逐PID的小Tensor归约，不是GPU显存不足、模型未加载GPU或其他用户竞争。独立实现PackedPIDPlan后，六条固定真实Caption的完整攻击函数平均 **8.582 → 1.270 秒，6.76×**。原生编码/评分、候选、最终文本及RNG通过Exact验证。当前Full仍使用原版，结果不混用。

Batch=8编码原型出现浮点差异，未接入攻击。没有修改WordBudget1、官方SEP、DPP、T、patience、词语选择、伪PID目标、Checkpoint、Tokenizer、精度、Target Top128/ITM或Batch Size。

## 1. 当前正式进程的60秒只读采样

从/proc及nvidia-smi连续采样60.048秒，共61点，监控当时的RSTP Source PID3006245；独立GPU测试在采样结束后才开始。

| 指标 | 实测 |
| --- | ---: |
| 总逻辑核心数 | 32 |
| SEP进程CPU平均／峰值 | 99.37% / 100.99% |
| 主线程CPU平均 | 99.32% |
| SEP平均占用核心 | 0.994 |
| 整机CPU平均繁忙比例 | 4.62% |
| 平均低于5%使用率的核心数 | 29.1 |
| GPU-Util平均／峰值 | 3.16% / 5.00% |
| GPU-Util完全为0的采样比例 | 14.75% |
| GPU设备显存均值／峰值 | 1033.8 / 1034 MiB |
| Source进程GPU显存 | 984–986 MiB |
| 平均GPU功率／SM频率 | 61.06 W / 2520 MHz |
| 进程RSS峰值 | 2.533 GiB |
| 实际磁盘读／写增量 | 0 / 880640 bytes |
| 主线程自愿／非自愿上下文切换增量 | 100 / 153 |
| 整机Swap换入／换出页数增量 | 0 / 0 |

Linux进程100%约代表一个逻辑核心；SEP约99.37%不能解释为整机99.37%。它仅相当于32核容量的约3.11%，整机仍大量空闲。进程68线程不意味着68核并行，实际工作集中于主线程。

GPU-Util平均3.16%的补数约96.84%，是采样窗的非忙估计；与“完全0%的采样点占14.75%”是不同口径，不能视作连续空闲时长。可见低而间歇的GPU活动，没有持续高负载。61点GPU进程清单只有该SEP Source；不是Ollama或外部GPU竞争。RAM约113 GiB可用，Swap虽有历史占用，但本窗口没有换页。设备频率稳定，低功率与低计算占空比一致。

## 2. 生产调用链与主要耗时

### 2.1 完整Query与CPU小操作

首条固定真实CUHK Caption的独立cProfile：8.456秒，157次文本评分请求，64次未命中缓存，93次命中。每次miss按顺序调用encode_texts([text])、NativeScorer、PIDPlan.aggregate。后者保留PID预计算，但仍执行11003次Python索引/max。

| 路径 | 单Query累计耗时 | 解释 |
| --- | ---: | --- |
| sep_bridge.PIDPlan.aggregate | 6.901 s，81.6% | 64×11003=704192次小型max，主要在CPU |
| victims.encode_texts | 0.845 s，10.0% | 64次原生单文本编码，含冻结状态检查与CPU返回 |
| NativeScorer._compute | 0.433 s，5.1% | GPU完整Gallery矩阵乘、图库上传及结果返回 |
| 其他函数及胶水 | 约0.276 s，3.3% | 原搜索、Trace、POS、预算、DPP等；含重叠口径限制 |

这些是同次cProfile的非嵌套大项，Profiler会增加耗时，不能直接当作Full平均。另一轮统一预热后的首Query阶段计时如下，包含独立计数器开销：

| 阶段 | 原版 | Packed |
| --- | ---: | ---: |
| 原生文本编码64次 | 0.600711 s | 0.604695 s |
| 完整Gallery评分64次 | 0.377716 s | 0.358639 s |
| PID聚合64次 | 6.806946 s | 0.032993 s |
| Query总耗时 | 8.025978 s | 1.155788 s |
| Python层CPU→GPU实际设备切换调用 | 389 | 389 |
| Python层GPU→CPU实际设备切换调用 | 159 | 159 |
| CPU→GPU字节量 | 4470606912 | 4470606912 |
| GPU→CPU字节量 | 8892992 | 8892992 |

此处设备切换数是Tensor.to/cpu/cuda的Python API实测，不覆盖C++内部的所有memcpy。另一次冷启动诊断原版额外12次CPU→GPU、284592 bytes，恰为12层77×77×4的attention mask首次初始化；已保留冷启动证据，并追加双方预热后的公平计数，没有将初始化差异宣传为优化收益。

### 2.2 GPU计算、传输与等待

独立Torch Profiler的同一真实单评分窗口：原版11003次aten::max，Packed仅1次，原生GPU数值路径保持不变。两者均记录12次cudaMemcpyAsync（HtoD5、DtoH4、DtoD3）。模型编码仍Batch=1，没有隐藏的批量参数变更。

| Profiler时间线 | 原版 | Packed |
| --- | ---: | ---: |
| 带Profiler的评分窗口 | 400.377 ms | 14.697 ms |
| CUDA kernel区间并集 | 0.767 ms | 0.765 ms |
| GPU memcpy区间并集 | 5.563 ms | 6.393 ms |
| GPU活动区间并集/窗口 | 1.58% | 48.71% |
| host cudaStreamSynchronize区间并集 | 0.116 ms | 0.116 ms |

这是独立进程中带Profiler的窗口，原版大量小操作的采集开销很大；400.377/14.697不能作为真实吞吐量加速比。CUDA与CPU可重叠，上述时间不能简单相加成为CPU/GPU总时间分解。Kernel约0.77ms而CPU调度和归约远长于它，说明GPU主要等待下一次CPU提交；消除CPU小操作后GPU能更频繁获得有效工作。

34,054×512的FP32图库常驻CPU。NativeScorer每次miss重新将约69.74MB传到GPU，query编码又返回CPU后再上GPU，评分矩阵返回CPU，PID logits最终再上GPU。首Query累计约4.47GB的CPU→GPU传输已实测。本轮不改变这一路径，因此Packed加速不是复制次数减少造成的。冻结图库设备缓存、进一步异步搬运可作为后续独立优化，尚未实现或证明完整链Exact。

### 2.3 Trace、日志与串行算法

WordBudget1的全局sys.settrace即便对无关帧返回None，仍收到大量Python call回调。独立计时包裹的首Query有602811次回调，回调累计0.336秒，Query共1.415秒，mutation_checks=3626。该数包含计时包裹自身开销，是诊断量级，不是关闭Trace的纯净收益。cProfile不会完整记录Trace回调内的所有调用，不能凭其budget_check短耗时判定Trace免费。

所有Trace、Forward Pre-Hook、候选历史及预算检查保持原样。没有通过取消检查提速。缩小Trace范围需要另证等效覆盖，本轮未实施。

正式CUHK Source的逐Query耗时总和49454.487秒，生成阶段49873.526秒，差约419.039秒（0.84%），包含产物发布、SHA、日志及其他循环开销，不能全部归因于I/O。结合60秒实际磁盘读取为0，不支持“模型重复加载或磁盘阻塞是主要瓶颈”。generate在Query循环外只加载一次Source模型和固定参考特征；但评分时确实重复上传图库。

SEP官方get_important_order及find_max_synonym本来向ensemble发送文本列表，现有classifier.forward却按文本逐条编码和评分。批量编码有优化空间，但已实测非Exact，见下文。T=10、patience=5及依赖前一步选择的串行搜索保持不变。

### 2.4 RaSa原生评分

复用已审计CUHK Smoke的8条冻结Query及136图缓存，仅独立分析原生NativeScorer；没有重新生成文本、重新编码图像或发布新指标。全局Top128、fusion_batch16和全部1024个ITM配对保持原样。

带Profiler总计2.620秒，8次融合累计1.807秒；Token缓存18次miss、1014次hit，加载约242.45MB、耗时0.102秒。输出与原冻结Smoke矩阵逐位一致，max_abs_diff=0，模型状态不变。该Smoke性能不能替代Full估计。已完成CUHK RaSa Full原链阶段2321.098秒（含加载、编码、评分等），其评分与Source生成分属不同进程；当前Source只有IRRA，RaSa ITM不是Source八秒/Query的原因。

## 3. 实际优化实现

新增src/tbps/baselines/cpgc_sep/sep_packed_pid.py，提供PackedPIDPlan及显式enable_packed_pid(classifier)。不改现有sep_source、sep_bridge或NativeScorer默认入口。

1. 绑定该Classifier原有PIDPlan，完整保留PID顺序和各组Gallery成员顺序。
2. 按原顺序构造二维索引，以各组首成员在尾部补齐到最大组长，不使用-inf哨兵。当前Source矩阵11003×15。
3. 一次批量索引及沿成员维max替代11003次Python循环。补齐项均位于原成员之后，首个并列最大值及signed-zero优先位置保持。
4. 空组、零组或过度不均衡导致填充超过8倍Gallery时回退原PIDPlan.aggregate，保留原异常行为；无静默数值近似。
5. 设备索引在该Plan内按device缓存，不跨Gallery或Classifier共享。没有RNG、候选或文本cache更改。

这属于已通过本轮输入范围Exact验证的工程优化；实现默认不启用，不会自动替换当前Full。

## 4. 真实GPU A/B

预注册在preregister.json：三个数据集各取官方Caption-only顺序前两条，六条在查看模型效果前固定。固定CUHK Source IRRA、同一Train参考图库、相同逐Query种子，原版先运行、Packed后运行。每臂统一预热、逐Query清空cache，两臂均保留预算Trace及事件日志。未使用Target Test PID或Target评分选择样本、候选或攻击文本。

| 指标 | 原版 | Packed |
| --- | ---: | ---: |
| 平均进程CPU（100%=一个逻辑核心） | 101.18% | 191.06% |
| CPU峰值／平均占用核心 | 109.92% / 1.01 | 209.81% / 1.91 |
| 设备GPU-Util均值／峰值 | 6.49% / 12.00% | 25.76% / 32.00% |
| 设备GPU-Util=0采样比例 | 2.47% | 0.00% |
| 设备峰值显存 | 2026 MiB | 2026 MiB |
| 本进程CUDA allocated峰值 | 504.148 MiB | 504.148 MiB |
| 单Query平均耗时 | 8.582386 s | 1.269530 s |
| P50／P95（6条插值统计） | 8.351756 / 9.436862 s | 1.249068 / 1.382443 s |
| 总文本编码前向／原classifier调用 | 394 / 1176 | 394 / 1176 |
| 缓存命中 | 782 | 782 |
| 吞吐量 | 0.116518 Query/s | 0.787693 Query/s |
| Word Budget违规 | 0 | 0 |
| 相对原版加速 | 1.00× | 6.7603× |

A/B每约200ms读取一次设备指标，NVML底层窗口可能重复，采样点不相互独立；进程CPU包含诊断采样线程的约4%开销。设备GPU均值6.49%→25.76%包含同卡原Full Source；不是独占GPU或单进程SM利用率。A/B原始进程清单、线程及采样记录留存，不能将背景GPU活动算作本优化的纯收益。主吞吐量数据来自本进程六条真实完整attack_caption，未使用Profiler时钟。

IRRA沿用官方原生参数类型及FP32输出/相似度，没有强制修改为另一种参数精度、启用AMP或调整TF32设置。未改变文本Batch=1、64级别miss数或原生GPU计算量。进程CPU从约1核变为约1.9核，是相同torch.set_num_threads(2)下批量CPU算子真正使用第二个线程，不是改变线程配置。设备GPU利用率提高来自更快提交有效工作。

这是六条样本的A/B，未随机化臂顺序，P95基于六条插值；GPU频率稳定但存在共享资源波动。平均时间包含词法、预算、候选、DPP、原生推理及事件日志，不含模型启动或逐Query ArtifactStore发布；没有据此承诺长文本、所有Query或Full的固定加速倍数。

## 5. Exact、预算与局限

- 新增20项CPU测试全部PASS（0.98秒），不重复未受修改影响的176项历史测试。覆盖多PID、非均衡、多正例、FP32/FP64、并列、signed zero、NaN/Inf及有限极值、空组原异常、RNG、softmax、margin、排序及cache/counter对象保留。
- 首次CPU收集因测试夹具PersonID缺少namespace参数报错，只修测试输入后通过；初次log/XML保留，未隐藏或跳过用例。
- 六条完整真实GPU攻击：全部非时间输出字段相同，包括最终文本、候选及tau/顺序、每次请求顺序、原统计、DPP、伪PID、预算检查次数；394个cache值、6个clean logits逐字节相同，cache键及顺序相同，Python/NumPy/Torch CPU/CUDA RNG起止状态相同。
- 六条另行逐入口核验：编码特征、完整1×34054原生矩阵、1×11003 PID logits、GPU softmax和margin全部bitwise exact。相同真实冻结矩阵的Packed CPU/GPU归约均与原CPU归约逐位一致。Exact使用uint8逐字节比较，没有allclose。
- 每次评分请求、全部选定候选及最终文本都再次检查≤1个原始空格词位置；违规0。模型状态SHA、冻结/无梯度状态保持。
- RaSa 8条原生评分1024 ITM配对与冻结Smoke位级一致。没有改Top128、融合、精度或其默认路径。
- 同一真实冻结分数的12次CPU微基准中位：89.614ms→0.245ms，约365.27×，仅PID聚合局部值；不能用作整条攻击加速。
- 当前固定有限分数和这些边界输入通过，不声称所有设备、Torch版本、NaN payload或无限规模都已穷举。新增Plan为未来独立版本提供显式入口。

## 6. 未采用的优化与未来限制

| 方案 | 本轮证据 | 决定 |
| --- | --- | --- |
| Packed PID批量索引/归约 | CPU20项、真实六Query全链及真实矩阵Exact | 可供未来独立版本采用，当前不部署 |
| GPU端Packed归约 | 单份真实完整矩阵Exact | 未接入NativeScorer，不能宣称端到端已消除CPU传输 |
| Batch=8文本编码 | encoder最大差0.000276834；由两份特征经同一评分入口所得矩阵最大差0.000433505 | 非Exact原型，不接入攻击或当前Full |
| 冻结图库常驻GPU、异步搬运 | 已确认每Query约4.47GB重复上传，未做新实现/完整验证 | 后续候选，不能宣称已有加速 |
| 缩小Trace或缓存AST | Trace有实测成本，尚未证明预算覆盖及命名空间/RNG等价 | 本轮不改 |
| RaSa批处理/Top128修改、精度或算法参数调整 | 不属于本轮已证Exact的优化 | 不实施 |

Batch=8的两个encoder输出不再位级一致；没有进一步据Target效果筛选或改变原SEP选择。最终攻击文本是否会因此改变尚未验证，不能因某个argmin碰巧相同就称无损。

Packed后的首Query中PID仅约2.85%。在该已测分解下，假设完全消除剩余PID耗时且其他成本不变，额外理论上界约1.03×，因此继续只优化PID收益有限。下一瓶颈是单文本编码、反复图库上传及原搜索/Trace的串行依赖。批处理可提升吞吐量但本轮已证非Exact；无法据小样本给出整个Full进一步加速的可靠上限。

## 7. 当前Full保护与交付范围

原Supervisor682941、发布器2919691及其子进程一直运行，未发送任何停止或重启信号。保护核验覆盖原工作树9份生产源码/配置、两份已发布CUHK报告，与原Git对象逐字节一致；原工作树干净。未覆盖Source、Target、Audit、Manifest或日志，也未重跑已完成Full或旧44条/第45条Claim。

独立工作树：/home/lzf/ldx/projects/idea-test-tbps-sep-performance-20261106。原始诊断输出：/home/lzf/ldx/outputs/tbps-sep-performance-20261106。控制、Profiler、测试与错误日志：/home/lzf/ldx/tmp/sep_performance_20261106。本轮短时GPU测试均退出，无新的持续GPU调度或自动Full。收尾核验时原链已自然进入ICFG Source，CUHK/RSTP四组Full报告均由原发布器发布。

未来适用性：Packed可用于下一次显式授权、独立版本的Full，需固定新代码/环境及协议记录；不应在当前六组中途启用。已有Full报告保持原方法版本，不能混入本轮诊断指标。尚未执行优化版Full。

必要脚本：monitor、diagnose、ab、verify_native、stages和rasa_profile。GPU入口都有独占启动回执，重复路径拒绝重跑；实际失败保留证据，不自动重试。全部原始证据留服务器，Git仅提交源码、针对性测试、本文及小型[EVIDENCE.json](EVIDENCE.json)。

诊断产物的SHA清单见[RUN_MANIFEST.json](RUN_MANIFEST.json)，原始对应run_manifest.json保留在独立输出目录。SHA覆盖原始Profiler/日志、预注册输入、测试结果和独立Tensor；Git不包含这些大型原始文件。
