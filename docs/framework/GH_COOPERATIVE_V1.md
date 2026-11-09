# GH-Cooperative-v1：双向协同目标与六节点Joint实验

用户确认从零训练60轮；每10轮保存配对G/H、独立CPU前缀审计、CUHK-PEDES × IRRA完整Joint Test及独立CPU Test审计，再恢复下一十轮。启动后台后由用户查看进度。

## 方法

记s00、s10、s01、s11分别为Clean、Image-only、Text-only、Joint的配对归一化相似度，S=s10+s01−s11−s00。

1. G保留原串行K个属性更新和原图联合更新。联合步增加−0.1S，s10和s11都对当前G可微，clean图像及文本监督detach。
2. H在原R_sim+R_rank上增加0.1S；当前完整2048 Train图库固定并绑定本轮G状态。每个实际编辑的贡献为完整采样文本奖励减去仅撤销该编辑后的奖励。每组attribute/position/replacement的log-prob乘detach贡献，取编辑均值后乘−0.1，与SCST共同执行一次Optimizer更新。STOP/NOOP没有虚构编辑贡献。
3. G在首个属性步已有特征上计算各原JVP单位方向的残余匹配证据，将其标准化后与原score_weights的log先验相加做softmax。局部权重为0.5+0.5K·softmax，均值1并detach；仅改变原K个属性步的Local系数，原属性顺序、K、Canonical、JVP计算规则和Retain保持。
4. 同图多描述目标使用主描述原文、当前H主描述、第一条不同且原生77-token未截断的官方Train描述及其当前H改写。特征全部冻结；温度0.1的log-mean-exp匹配分数以0.1权重加入G联合步，优先压低尚能匹配的描述。没有可用第二描述时记录原因，仅使用主描述两项。额外描述不触发新S评分、属性重选或额外JVP。

以上权重是训练前固定的初始研究设置，没有根据Test调参。四项一起组成新方法；本次六节点比较表示这一组合随轮次变化。

## 数据、预算和推理

2048官方Train固定成员与顺序，原G每轮8068次更新、H每轮2048次访问/2037次更新。60轮预期G484080、H访问122880/H更新122220。G和H seed42从零初始化，Adam空状态、无Scheduler；IRRA/Reader/basis/原条件缓存/clean训练图库只读冻结。

图像实际FP32 RGB L∞≤8/255，使用原向内舍入投影。文本沿用原Dynamic Natural-K：可改多个词，但每个合法原词位最多替换1个Unicode字符，每属性最多1次，共享词最多1次。没有全Query一个词的上限。推理仍为G(I)、H(T)，额外描述及图像/PID配对仅用于训练监督与奖励绑定。

六组分别使用本次G10/H10、G20/H20、G30/H30、G40/H40、G50/H50、G60/H60，完整3074 Gallery/6156 Query，IRRA原生完整Gallery检索，仅Joint，报告R@1/R@5/R@10/mAP。Test结果不进入训练或选择参数。

## 工程与审计

独立src/tbps/cooperative包临时扩展原损失/SCST调用并在finally还原函数绑定；原prototypes、alternating模块和历史资产不修改。新配置绑定全部当前src/tbps Python SHA、四个新operation SHA、原固定资源及运行依赖。

每轮账本G/H/conditions数组拆成≤4MiB、≤32记录的严格JSON分片，Manifest覆盖全部分片；CPU逐分片读取并核对成员、连续状态、原JVP/FD证据、每个Unicode编辑、各协同项及更新总数。全局read_json的64MiB限制保持。最新完整G/H/双Adam/双阶段RNG/当前Bank用于边界恢复，旧Tensor仅在后继完整提交后在本次输出树内退休；全部分片、SHA、Claim/退休证据和十轮评测权重保留。

单次监督器先执行真实八样本Smoke及独立审计，再执行十图/十Query Joint链路Smoke及审计。之后每十轮暂停于完整边界，先CPU审计，再正式Test/CPU审计，最后继续训练。退出码3只在预设10/20/30/40/50轮边界接受；其他错误或末轮非零退出停止且不重试，保留原失败。

输出：`/home/lzf/ldx/outputs/tbps-framework-gh-cooperative-v1-20261109`。

控制：`/home/lzf/ldx/tmp/gh_cooperative_v1_20261109`。阶段日志由status.json及child.json定位，results_so_far.json只汇总已经通过独立CPU审计的正式节点。六组全部通过后自动生成CSV、R@1/mAP曲线和GitHub报告。

## 当前验证

新增CPU测试验证四模式两分支梯度、K0/1/3/7与原K+1、逐编辑多词撤销及单次SCST更新、停止/NOOP、函数异常还原、超过64MiB总账本分片读取和篡改拒绝、十轮Checkpoint和完整边界恢复。最终CPU回归29项通过（16项新增、13项原机制回归）；4项旧配置专属用例未运行，新方法由独立Smoke/Train预检核验完整2048身份和全部来源。预检均PASS、无CUDA初始化。来源见GH_COOPERATIVE_V1_CPU_EVIDENCE.json；真实Smoke及Full结果由后台独立审计产生。

用户一键检查：`ssh tbps-server "/home/lzf/ldx/envs/tbps-image/bin/python -B /home/lzf/ldx/projects/idea-test-tbps-gh-cooperative-v1-20261109/operations/gh_cooperative_v1_status.py"`。该命令只读当前状态与日志，不会启动或重试实验。
