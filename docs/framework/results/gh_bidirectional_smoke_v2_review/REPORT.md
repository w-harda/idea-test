# G/H A1/B两轮Smoke：正式产物复核与CPU审计失败分析

## 核心结论

| 层级 | 本轮结论 |
| --- | --- |
| GPU原生生成 | PASS：GH_NATIVE_TWO_ROUND_SMOKE_GENERATED，退出0 |
| 四轮/四份2048图库、双状态及SHA完整性 | 独立CPU只读复核PASS |
| 原独立CPU审计 | FAIL：first NOOP control differs |
| H→G新条件/新JVP实际通路 | 真实执行；但本Smoke严格A/B因果归因NOT_VERIFIED |
| G→H图库→Reward/Advantage→H更新通路 | 有实际数值证据 |
| 完整严格双向A/B验证 | **未通过**，不可将原FAIL改成PASS |
| 检索性能/跨模型效果 | 未评测，不存在可报告的mAP或Rank-1 |

真实执行代码a7597f9c964dda0c8c48973e4678f0b9d3fa095a；部署1de92e512585f332cc28966c5f5870bb6f6627d8。
只比较固定8条内部Train样本、A1/B各两轮；未重新运行probe、GPU Smoke、训练、评测或原audit-smoke入口。

## 1. 失败根因与精确位置

异常为ValueError: first NOOP control differs。
调用链：native_cli.py:204 main → audit_smoke():178–179 → contracts.py:8 require。
监督器gh_smoke_watch.py:128的child()随后收到audit-smoke退出1，并在15:37:19记录SMOKE_FAIL。

[原审计断言](https://github.com/w-harda/idea-test/blob/a7597f9c964dda0c8c48973e4678f0b9d3fa095a/src/tbps/alternating/native_cli.py#L178)同时要求：
首NOOP G_before相同、G_after相同、全部gradient_observations逐字段相同。
实际G_before及条件内容相同，但G_after/梯度SHA/梯度范数不同。
失败属于**严格数值对照语义不一致**；没有发现原审计路径解析或变量遮蔽错误，不是缺失/SHA损坏导致。
原Traceback、退出码和监督状态均保留：[原始Traceback](ORIGINAL_CPU_AUDIT_TRACEBACK.txt)。

| 首条NOOP：cuhk:26530:0 | A1-r1 | B-r1 |
| --- | --- | --- |
| K / 更新次数 | 0 / 1 | 0 / 1 |
| Loss | 0.3706851899623871 | 0.3706851899623871 |
| 原记录梯度L2范数 | 0.5205852405058475 | 0.5205852190515232 |
| G_before | 5d48caab6a891a1afd3aaaa6f3594eb5f860ccd7b6090708d289224e6c819e47 | 5d48caab6a891a1afd3aaaa6f3594eb5f860ccd7b6090708d289224e6c819e47 |
| 梯度SHA | 8381442ed23f4608e912a137b1f63bfa81db0c4160184e0ae25dfab853558b39 | 6b85d28129f0feb3949925bf672aceb3beef847f29d5fc583741fe14c9026e81 |
| G_after | aff751d548935c8d1c07ce73786e66bd534d076076c2543829af5cc2866107b7 | 27c6ef00c943e68de05272e1abc236062c7a76b25dc6484ca2b572cf9c752e07 |

两个梯度**范数的差**为2.1454324317105034e-08，不是梯度差向量的范数，也不能作为max_abs误差。
原始完整梯度和首步参数Tensor未保存，故首步逐值误差NOT_VERIFIED；不得由SHA估计差异大小。
条件payload内容SHA exact、Loss相同并不能使不同的G_after/梯度SHA满足原严格断言。
FP32/math/no_tf32与RNG状态恢复不自动保证CUDA核bitwise确定性；原fp32_runtime关闭TF32/benchmark，但未强制deterministic algorithms。具体产生差异的算子或原因仍NOT_VERIFIED，不能仅据小范数差断言是某个GPU规约算子。

后续native_cli.py:180还要求首个非NOOP的G_before相同；实际亦不相同，因为已继承首NOOP的分化。
该断言原来未执行到，独立复核现已明确记录为false。不能仅修第178行就宣布完整控制有效。

## 2. 实验配置与四轮完整性

A1：原Caption条件训练G，最新G图库训练H。
B：H攻击文本的新语境条件/真实JVP训练G，最新G图库训练H。
两组完整父状态SHA均为d9b3593f31493766e6b90633ecd5e8bf2d20fc57aa2977e2b5ec3ba39f2614e6，包含G60/H10、双Adam、双RNG及游标；两臂初始状态相同。
固定样本顺序：cuhk:26530:0, cuhk:26747:0, cuhk:12746:0, cuhk:9955:0, cuhk:29253:0, cuhk:21540:0, cuhk:3127:0, cuhk:8385:0。

| Round | G Adam更新 | H访问/Adam更新 | 图库完整性 | 同文本Reward变化 | 同文本Advantage变化 | 非零H梯度 |
| --- | --- | --- | --- | --- | --- | --- |
| A1-r1 | 27 | 8/7 | 2048/2048 | 7/7 | 2/7 | 2/7 |
| A1-r2 | 27 | 8/7 | 2048/2048 | 7/7 | 4/7 | 4/7 |
| B-r1 | 27 | 8/7 | 2048/2048 | 7/7 | 2/7 | 2/7 |
| B-r2 | 27 | 8/7 | 2048/2048 | 7/7 | 4/7 | 4/7 |
| 合计 | 108 | 32/28 | 8192/8192 | 28/28 | 12/28 | 12/28 |

每轮1条固定NOOP不更新H；变化比例分母7是实际合法H更新数，访问数为8。
所有图库官方内部Train有序成员、逐图Claim、FP32三Tensor、feature、文件SHA/内容SHA、[0,1]及严格8/255预算检查通过。
四份最大L∞均0.0313725471496582；全部特征为FP32、有限且单位L2。
本次验证16465个Artifact单元、57637个原文件，未发现pending、孤立未完成Claim或原生FAIL；原Smoke下audit目录不存在，因为原审计在发布之前抛错。

每臂r1游标G484107/H visits20488/H updates20377；r2为G484134/H visits20496/H updates20384。
G Adam所有step分别484107/484134；H各参数step分别[20131,20255,20377]/[20138,20262,20384]，不要求不同参数历史step人为相等。
原父Checkpoint实际读取校验、双Adam所有状态有限/组参数不变，G逐步SHA链、H原生SHA链及Round最终状态对应通过。
G/H RNG保存完整Python/NumPy/Torch CPU/CUDA/sampler与单设备CUDA字节，并与完整状态Manifest绑定；本次没有在GPU重放新Round RNG。历史CUDA父状态重放PASS来自独立cuda_rng_v1，不冒充此次CPU重放。

## 3. H→G：确实执行了新条件，但严格控制失败

B实际生成38个新JVP、14次新语境reader定位，G的K/属性顺序不强制共享；共108次原K+1更新。
第一条非NOOPcuhk:26747:0的原属性步：

| 项目 | A1-r1 | B-r1 |
| --- | --- | --- |
| 原记录梯度L2范数 | 0.5494481355916896 | 0.48542884594413266 |
| Global Loss | 0.47330576181411743 | 0.48912498354911804 |
| Local Loss | -0.0025320835411548615 | -0.0013111564330756664 |
| Total Loss | 0.4707736670970917 | 0.4878138303756714 |

梯度和G_after SHA不同，但该样本G_before已不同，不能据本Smoke将差异完全归因为H文本。
第二轮仍有状态差异，且已累积先前差异，不是新的同父状态控制。

| Round | Global均值 | Local均值 | Retain均值 | Total均值 | H Advantage均值（7条） |
| --- | --- | --- | --- | --- | --- |
| A1-r1 | 0.1630510612 | 0.0001413595 | 0.0000595873 | 0.1632520076 | 0.0181943285 |
| A1-r2 | 0.1085354453 | -0.0002645141 | 0.0000025763 | 0.1082735070 | 0.0259867587 |
| B-r1 | 0.1704780197 | 0.0006182992 | 0.0000403310 | 0.1711366504 | 0.0146044344 |
| B-r2 | 0.1236222482 | 0.0001701191 | 0.0000343310 | 0.1238266996 | 0.0206516385 |

均值来自各Round实际27次G更新，阶段/样本不同，不能把A1/B不同目标的Loss直接当作攻击性能优劣。
完整108条梯度范数、Loss、属性ID、K、步骤和SHA见[G_UPDATE_DETAILS.json](G_UPDATE_DETAILS.json)。

## 4. G→H：区分Reward、Advantage、梯度和参数变化

28次有效访问中，同文本旧/新图库Reward均变化；12次Advantage变化且产生非零H梯度。
28次H步骤均出现参数状态变化，其中16次梯度范数为0。原H10全部34个Adam参数状态含非零exp_avg/exp_avg_sq；原step不跳过零梯度。
这种零梯度仍移动参数与Adam保留动量的机制一致，不能把16次变化全部归为当前SCST反馈的新学习信号。

强证据样本cuhk:9955:0（r1第四条）：
两臂H_before相同、sampled/greedy Caption相同、Token SHA和IRRA文本特征SHA均相同；图库不同。

| 项目 | A1-r1 | B-r1 |
| --- | --- | --- |
| Sampled Reward | 0.12087792158126831 | 0.124656543135643 |
| Greedy Reward | -0.007815133780241013 | 0.015169557183980942 |
| Advantage | 0.12869305908679962 | 0.10948698222637177 |
| H梯度范数 | 2.6328039794091835 | 2.23988581222645 |
| SCST Loss | 0.8923116326332092 | 0.759143590927124 |

更新后H状态分化，说明不同图库经原Reward/Advantage影响实际H更新。
在同一更新内，原程序也对相同文本特征计算了旧图库反事实奖励；这些是原日志数值，本轮没有重新调用RewardEvaluator。
Advantage变化方向并不总为正：A1-r2有4条正变化，B-r2有4条负变化。不能因此直接判定检索攻击增强或退化。
全部32条访问的原精度奖励分量、Advantage、梯度/参数增量、动作和SHA见[H_VISIT_DETAILS.json](H_VISIT_DETAILS.json)。

## 5. 轮末A1/B状态数值距离

以下距离由实际保存的state Tensor在CPU上逐值计算，不从SHA推测；包含state中的参数/缓冲Tensor。

| 时点 | G max_abs | G L2距离 | H max_abs | H L2距离 |
| --- | --- | --- | --- | --- |
| r1末 | 0.005090191960 | 1.048986679316 | 0.000594258308 | 0.005735669081 |
| r2末 | 0.006344459951 | 1.437921487400 | 0.000971317291 | 0.012915111348 |

最终G/H及双Adam均分化，但这是观测到的训练轨迹分化，不是mAP提升，也不能消除首NOOP控制问题。

四份图库特征内容SHA：
| 图库 | SHA256 |
| --- | --- |
| A1-r1 | 4377f24fb586dc5c987040ad6289cb68752ea2410ec036f9c63e5c47140a7a6f |
| A1-r2 | a5f5944be29f56f266ce91db8be337edc2b759f2feb5a4ac7c5e764e84b7f7b1 |
| B-r1 | 61345752dc2a355e212477dfb5fd030832500e6216ca676ac0abd1e77b019347 |
| B-r2 | 2bb6f3dcb8a66def353adbc12469f55810d7d167620a3ea868a6856af2d9b995 |

全部图库成员顺序摘要相同：f50ca55e17b4f0c88d68b4ccf59fb3336c86332d8bb71405eb8bb10b2362a445。
每份均绑定对应Round G版本；完整状态/Adam/RNG/Manifest SHA见[RESULTS.json](RESULTS.json)和[SOURCE_EVIDENCE.json](SOURCE_EVIDENCE.json)。

## 6. feedback_effects必须分层解释

原审计在178行抛错，**没有发布feedback_effects或审计PASS回执**。
独立按181–186行原表达式只读重算：
- first_nonNOOP_gradient_diff=true。
- first_nonNOOP_G_update_diff=true。
- same_text_Reward_changed=true。
- same_text_Advantage_changed=true。
- 该四项all表达式=true。

但原前置NOOP exact=false、首非NOOP同G_before=false，因此综合严格bidirectional_update_evidence_verified=false。
RESULTS.json将derived_original_expression与严格结论分开；不能把派生true伪装成原审计成功，或把存在差异等同于严格归因成立。
H→G的同G数值基础曾由probe_v2独立验证；那不替代本Smoke失败的严格对照。

## 7. CPU复核版本与保护

未修生产审计断言、未修改数值阈值，原审计仍FAIL。
独立CPU工具保留两次解析诊断失败：
1. readonly_v1把框架tree_sha与H原生feedback_backend.state_sha直接比较；两者编码不同，修为调用锁定原H摘要函数。
2. readonly_v2未把JSON数组无损恢复成ConditionIdentity要求的tuple，修为只还原类型且保持顺序/数值。
两者自身Claim/FAIL、源码和日志均保留，未覆盖或重签。上述问题属于本轮补充解析工具，与原G NOOP断言失败无关。
修正后readonly_v3完整只读复核结束，退出2明确表示严格协议未通过，而不是把工具执行完写成审计PASS。

共12项CPU测试通过（8基础、2摘要空间、2真实32条Condition JSON往返），未跳过或放宽断言。
未在原smoke_v2中创建audit输出；独立回执位于/home/lzf/ldx/outputs/tbps-framework-gh-smoke-review/readonly_v3/completed/。
57,637份原Smoke文件及原控制文件metadata前后不变；151378项历史metadata、28份旧FAIL/Watcher SHA、父Checkpoint/probe_v2保护检查通过。
全部文件内容经原ArtifactStore逐单元SHA验证；完整核验单元索引摘要2aa92504403bad2c3b1c3516603a9685ca1dbc5f3ca3d7fb1b876bc3875c6157。
本轮CUDA初始化、神经前向、新梯度/奖励计算、训练及检索评分均0。

## 8. 资源、来源与下一步

原生执行范围（加载、更新、图库生成）1824.994606591761秒；进程wall1836.3012017048895秒。
GPU peak allocated=1086753792 bytes（1036.409179688MiB）；
reserved=1214251008 bytes（1158.000MiB）。
Peak RSS=2427342848 bytes（2.260639GiB）。
G/IRRA image各8300=8192图库+108训练；H84、IRRA text226、reader14、JVP38。
IRRA/reader冻结由原执行的版本/完整状态摘要断言通过及权重SHA不变支持，本次不重新加载模型验证。

本结果支持“新语境条件与新图库反馈接口均实际参与训练”，但不满足严格A1/B阴性控制，不能作为完整因果验证或性能结论。
下一步建议仅设计并另行授权最小数值可复现性诊断：同父G/Adam/RNG、同NOOP输入，保存完整梯度/首步参数及运行时确定性设置，定位产生bitwise差异的算子。
如未来采用浮点容差，必须在新的实验协议中预先规定并重新验证对照；**本轮不能事后改阈值转PASS**。
保持K/Loss/Reward不变，严格控制成立后再讨论更长交替训练或Validation；本轮不自动执行。

来源及绝对路径均在SOURCE_EVIDENCE.json；正式原生结果保留在/home/lzf/ldx/outputs/tbps-framework-gh-bidirectional-backend/smoke_v2，原控制日志保留在/home/lzf/ldx/tmp/gh_smoke_wait_v1_20261023。
