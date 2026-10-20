# G-v2 Epoch Sweep：共享GPU准入修复与唯一真实Smoke报告

**结论：GV2_SIX_NODE_SMOKE_PASS。** 234项适用CPU测试、完整来源检查、G10/G60生成、
IRRA/RaSa/APTM原生评分及三份独立CPU审计均PASS。没有重试、训练或正式Full。
原技术Smoke门禁已满足，仍等待用户最终授权五节点三模型Full。

## 来源与最小修改

- 资源修复基准：0dba135a3bd0a59fdf3ab220acfd6066a6102c5a。
- 实际执行代码：152498fd258572ad9c43c04b2ce8fba8b9e995b4，运行前已提交并推送。
- 分支/工作树：codex/gv2-six-epoch-evaluation；
  /home/lzf/ldx/projects/idea-test-tbps-gv2-epoch-sweep-20261020。
- 只改gv2_sweep.py::resources()、其配置资源字段及对应producer SHA、相关CPU测试；
  Target Runner及resources以外运行函数AST exact，所有非资源实验字段保持一致。
- 旧策略：空闲≥8192MiB且利用率≤10%；新策略：空闲≥6144MiB，利用率/其他进程只记录。
  原RAM≥32GiB、Smoke磁盘≥10GiB/Full≥100GiB、来源校验保留；错误即停止，无轮询或抢占。
- 模型/Checkpoint、FP32/math/no_tf32、batch1、384×128原RGB输入、Query chunk128、
  原生评分/Top-K和8/255预算均不改；没有改Source contract或历史准备证据。

| 文件 | 原SHA-256 | 新SHA-256 |
| --- | --- | --- |
| gv2_sweep.py | `a3eeecdf087576ac18c824c68480c0ba625f0da8f65feb6d6ee444050b093058` | `13a551c89a5ceff962a5d2020b2de2f91ab3e95cfe49fdf9890b691a081499ba` |
| gv2_epoch_sweep.server.json | `b7b5fa34cdfd3c5ff4c8568b91cc3f5804268cf210ebd36215f4390eb714b7bd` | `66bdd1639b54508af1852f781b3951c637193ffa5069848b4b73145e9d9bf701` |

[逐项资源配置差异](results/gv2_epoch_sweep_shared_smoke/resource_policy.json)。
producers集合仍是完整107项，只更新resources所在文件的实际SHA，没有关闭或跳过来源门禁。

## CPU验证

资源9场景覆盖74%/18433MiB放行、90%放行、6144边界、高利用率不否决、6143拒绝、
RAM/磁盘不足拒绝、GPU状态读取失败在Attempt创建前停止、不调用kill/killpg/进程控制/CUDA、
显式shared_observe_only策略。既有SHA缺项/篡改拒绝和不可变Attempt保护继续通过。

当前模块43项+公共99项+原生适用92项=234 PASS。
旧APTM绝对producer路径用例仍沿用先前基准复现的单项排除，没有修改旧测试/配置。
无真实模型CUDA执行的CPU Fixture与实际GPU结果分开记录。

## 唯一Attempt及原生流程

输出：/home/lzf/ldx/outputs/tbps-framework-gv2-epoch-sweep/smoke_v1/
父日志：/home/lzf/ldx/tmp/gv2_epoch_sweep_shared_smoke/gpu_smoke.log
子日志：上述smoke_v1/logs/{G10,G60,irra,rasa,aptm,irra-audit,rasa-audit,aptm-audit}.log。
没有旧Attempt，唯一smoke_v1不可变创建一次，没有v2或自动重试。

固定Gallery为官方0–8及15，Query为官方前9条及cuhk:37147:1（固定历史NOOP），共10+10。
H7和原始/攻击文本特征仅按各Victim原ModelSpace只读复用，H恢复/前向和文本神经编码均0。

| 阶段 | 权重GPU恢复 | 图像样本前向 | 文本样本前向 | 融合pair |
| --- | ---: | ---: | ---: | ---: |
| G10 | 1 | 10 | 0 | 0 |
| G60 | 1 | 10 | 0 | 0 |
| IRRA | 1 | 10 | 0 | 0 |
| RaSa | 1 | 10 | 0 | 200 |
| APTM | 1 | 10 | 0 | 200 |

两次G恢复分别为两个冻结Epoch，不是重复生成同一Epoch。
IRRA原dot、RaSa原生ITM、APTM原生ITM/global和完整本次Query集合归一化均复用旧实现。
RaSa/APTM的200个融合pair是两个模式各10 Query×Top10 Gallery；
这不等于文本特征重新编码，也不等于完整Top128 Full路径已验收。
所有Backward/Optimizer/JVP/新属性评分为0。没有借用历史Full指标冒充Smoke。

## G像素、预算与冻结

G10对历史generation_v3的10张对抗RGB最大绝对误差 **0.0**；
独立只读补核原RGB、Adv RGB和effective_delta的30项内容SHA全部exact。
两轮生成均使用原generate_once，三Tensor保存后立即读回，内容bitwise exact、紧凑FP32；
没有PIL重新量化或额外投影。

G10/G60最大实数L∞均为 **0.0313725471496582 ≤ 8/255**，像素合法、有限。
G10参数/buffers整体SHA d5681683475cb292199c203022358ecf9172c7f1cffa8127d17197a03cd723c6；
G60为5d48caab6a891a1afd3aaaa6f3594eb5f860ccd7b6090708d289224e6c819e47，前后exact。
三个Victim的参数/buffers独立前后摘要exact，eval、requires_grad=False、parameter_grads_none=True。
实际摘要/Checkpoint及全部完成Manifest SHA见[原始精度证据](results/gv2_epoch_sweep_shared_smoke/SMOKE_EVIDENCE.json)。

## G60小样本真实指标

仅是10 Gallery/10 Query、Top10的链路结果，不是正式Full或训练轮数效果结论；
特别是10张Gallery下R@10本身不具备Full区分力。

| Target Model | Attack Method | R@1 | R@5 | R@10 | mAP |
| --- | --- | ---: | ---: | ---: | ---: |
| IRRA | G-v2 Image-only | 90.00 | 100.00 | 100.00 | 87.89 |
| IRRA | G-v2 + H-v1 Joint | 80.00 | 100.00 | 100.00 | 84.29 |
| RaSa | G-v2 Image-only | 90.00 | 100.00 | 100.00 | 90.33 |
| RaSa | G-v2 + H-v1 Joint | 80.00 | 100.00 | 100.00 | 81.11 |
| APTM | G-v2 Image-only | 80.00 | 100.00 | 100.00 | 89.11 |
| APTM | G-v2 + H-v1 Joint | 80.00 | 90.00 | 100.00 | 86.79 |

[原精度CSV](results/gv2_epoch_sweep_shared_smoke/SMOKE_METRICS_RAW.csv)；
[两位小数CSV](results/gv2_epoch_sweep_shared_smoke/SMOKE_METRICS_2DP.csv)。
CSV/summary/既有Engine结果逐项一致，不重新评分来整理报告。

## 共享资源、时间和存储

启动2026-10-20 19:57:41.424323 +08:00；完成回执20:13:31.571892 +08:00。
整条命令至回执950.148秒（15分50秒），包括CPU来源预检、模型恢复、生成/评分和三次CPU审计；
父Runner计时824.457秒从初始CPU检查之后开始。下面子阶段计时不含各自前置CPU来源预检，
包括加载、实际推理、绑定校验和发布，不能称为纯CUDA执行耗时。

启动GPU空闲18433MiB、利用率58%；其他Python PID1402779占5610MiB。
后续IRRA/APTM进入时利用率94%仍按新策略放行。
没有终止、暂停或改变其他用户进程；执行结束外部进程仍在，且本任务无残留Worker。
共享负载可能影响耗时，快照不保证后续无OOM；本次无OOM/预算/数值/来源异常。

| 阶段 | peak allocated MiB | peak reserved MiB | 单进程peak RSS GiB | 阶段秒 |
| --- | ---: | ---: | ---: | ---: |
| G10 | 41.82 | 52.00 | 1.161 | 11.438 |
| G60 | 41.82 | 52.00 | 1.162 | 8.310 |
| IRRA | 922.10 | 1002.00 | 2.228 | 19.180 |
| RaSa | 1900.06 | 2082.00 | 4.075 | 23.214 |
| APTM | 993.63 | 1074.00 | 2.345 | 12.895 |

GPU峰值为本任务各进程自身PyTorch allocator统计，不包括其他用户显存；
RSS为各Worker进程峰值，不冒充同时刻整机总内存。
新增产物278文件，逻辑55442097字节（约52.87MiB），实际分配56340480字节（约53.73MiB），
包含两Epoch图像、原生表示/Score、回执和日志，不复制旧H文本/特征。

## 三份独立CPU审计与历史保护

三个独立新进程均CUDA_VISIBLE_DEVICES为空，source/像素/预算、官方顺序与PID、
两模式缓存/Score shape/FP32/有限性/SHA、模型冻结、CSV/summary及Engine结果一致性通过；
没有重算排序、运行模型或重新生成图文。
完成日志为GV2_SWEEP_CPU_AUDIT_PASS smoke 60 irra/rasa/aptm，
三者全部通过后父Runner才发布GV2_SIX_NODE_SMOKE_PASS。

74184个相关历史文件size/mtime/inode未变，289项来源/Checkpoint/prototype SHA未变；
六点Checkpoint、原H/Full、原训练contract均只读，107个现行producer SHA复核通过。
没有pending、损坏或失败单元，没有训练或Full目录，没有重复审计已PASS的历史Full。
旧RESOURCE_BUSY与旧准备证据仍保留，不回填其状态。

## 正式Full条件及成本更新

技术Smoke、G10再现、新schema G60生成及三Target/三CPU审计门禁已满足；
**仍需用户最终Full授权，当前没有启动正式Full。**
固定五个新Epoch20/30/40/50/60分别生成3074图、共享给三Victim的Image-only/Joint，
共15370次G，H文本与可严格核验的历史Raw控制/E10结果继续只读复用。
不基于本次小样本指标修改节点、模型或参数。

实际G reserved约52MiB、三个Target最大reserved约2.03GiB，支持保留6144MiB启动门槛；
不代表Full Top128或其他共享负载下必定不OOM。完整RaSa原历史RSS约24.43GiB，
Full可用RAM≥32GiB门槛保留，不能据Smoke的4.07GiB调低。

本次新像素/原生特征实际字节与既有维度一致，五节点像素约25.33GiB，
加三模型Adv图缓存、两模式完整Score/原生中间Score和回执，继续预估65–80GiB，保留≥100GiB空闲。
历史三模型完整四模式计时按两模式×五节点约12.5小时主要评分；
加生成、CPU来源SHA、发布及审计，低干扰条件仍估15–20小时。
本次只有200个ITM pair，不能将小矩阵时间线性乘7879.68冒充Top128实测。
共享GPU持续高负载时可能显著延长，规划可留24–36小时窗口；这是调度余量，非实测承诺。
正式入口和后续授权命令见[运行说明](GV2_EPOCH_SWEEP.md)，本轮不部署后台任务。

本次来源及算法保真、唯一GPU Smoke、三CPU审计与历史保护全部PASS；
结论仅限固定环境和两Epoch小样本，不声明六点Full、Top128或训练轮数效果已验证。
