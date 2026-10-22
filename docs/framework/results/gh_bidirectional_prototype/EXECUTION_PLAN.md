# 待明确授权的GPU执行方案（本轮不执行）

## 执行前还需完成的最小事项

CPU原型不能直接以修改配置true的方式启动GPU。当前CLI只有来源预检，Prototype硬拒绝非CPU Fixture后端。
下一轮明确授权后，组装冻结IRRA/reader、原G/H加载与完整父状态恢复、原train_single_sample/train_step、
原generate_once和原RewardBank的生产后端。优先复用既有legacy_imports、CPUStartupGate和来源闭包检查；
CPU隔离上下文全部退出后才初始化CUDA，不复制既往启动时序错误。
新后端及执行入口必须重新绑定其源码SHA，保留本准备版本；不改任何历史producer或旧contract。

真实验证重点尚为PENDING：

1. 新Unicode语境下reader的token/linked-object定位及新e/w；
2. TextTransport真实zero exact、非零/有限JVP及原FD相对误差≤.02；
3. 完整G60/H10双Adam与阶段CUDA RNG的真实读回；
4. 源模型/reader完整参数及buffers前后不变；
5. 首轮NOOP及非NOOP的真实G梯度差、旧/新图库的真实H Advantage差。

遇到不支持的Tokenizer/定位，不改原规则，不补旧JVP，不更换固定样本，保存FAIL停止。

## 计划执行顺序

1. 来源预检、保护快照、单次即时资源检查；未授权前不创建smoke_v1。
2. 新独立Attempt锁定配置和代码版本；两臂完整恢复证据。
3. A1 round1：固定H输出→原G条件→27步G更新→2048完整bank→7合法H更新/1 skip→round提交。
4. A1 round2：相同规则继续，不重置Adam/RNG。
5. B从相同父状态开始；round1/round2改为真实新语境条件，其余与A1相同。
6. 独立CPU审计四个round、bank、条件、梯度/Reward探针、父子恢复链和历史保护。
7. 仅报告信息流验证状态和实际计数，不运行Validation、Test、RaSa/APTM或Full。
8. 任一阶段失败立即停止，不自动重试/换样/恢复，不启动60轮训练。

输出计划：
/home/lzf/ldx/outputs/tbps-framework-gh-bidirectional-prototype/smoke_v1/{A1,B}/
每臂两个round；新父组合协议独立，历史G60/H60与H7 Selection保持。

## 两臂×两轮合计预算（未实测）

| 项目 | 计划数量/上限 | 说明 |
| --- | --- | --- |
| G训练Adam更新/G前向/IRRA图像编码 | 各108 | 每轮Σ(K+1)=27，4轮；原K+1不变 |
| 完整奖励bank刷新 | 4份 | 每份2048唯一Train图、同一G版本 |
| bank G生成/IRRA Adv图像编码 | 各8192 | batch1逐图generate_once，原Clean可只读引用 |
| G前向合计 | 8300 | 108训练+8192生成；不含任何未经登记的诊断重放 |
| IRRA图像前向合计 | 8300 | 与上述对应，无新Clean编码 |
| H visits / 合法Adam更新 | 32 / 最多28 | 固定1条不合格NOOP每轮保留skip |
| H SCST文本样本 | 最多84 | 每合法更新3条；不缓存过期H输出 |
| H策略rollout | 最多84个合格样本rollout | 28反馈greedy+56训练sampled/greedy；不合格由原资格规则处理 |
| B新语境条件 | 最多14条 | 两轮各7个合格样本，实际NOOP进一步减少 |
| 新属性JVP | 最多38个 | 2×Σ选定8条原K=38，仅B非NOOP |
| JVP内部autograd.grad | 按实际hook记录，原实现通常2/JVP | 约76，明确属于自动求导，不能称“无反传” |
| 新语境IRRA文本编码 | 预计最多142样本 | 若原TextTransport每语境2次初始化，每JVP 1次primal+2次FD；实际计数为准 |
| IRRA文本合计 | 预计最多226样本 | 84+142；reader前向独立登记，最多14 |
| H/G更新诊断 | 尽量复用已有梯度和三元组特征 | 如必须额外前向，先在冻结预算中明确，不能未经计数执行 |

G生成是推理；G训练需要图像梯度；JVP需要原自动求导；H只有SCST更新。
IRRA和reader的参数不更新，不意味着没有输入梯度或JVP。
旧/新bank的Reward诊断复用同一组已编码文本特征，原Reward数学不变，不增加目标反馈。
本轮不安排额外GPU中等规模试跑；下一轮先在固定样本序列的第一个真实条件检查处停止于任何异常。

## 资源与风险

拟定共享资源门槛：空闲显存≥6144MiB、可用RAM≥32GiB、磁盘≥25GiB、空闲inode≥20000；
GPU利用率仅观察，不终止或修改其他用户进程。此门槛借鉴既有工作，不是新JVP峰值实测。
本轮一次资源快照见SOURCE_EVIDENCE.json；可用显存只略高于门槛时，其他任务显存增长会带来OOM风险。
真正执行前仍需即时重检并明确授权；当前资源合格不等于已经授权或保证不会OOM。

三Tensor FP32图库纯数据：
8192 × 3 Tensor × 3 channel × 384 × 128 × 4 byte = 13.5GiB。
若只保存Adv而引用原图可减少重复，但本计划按较保守完整物化估计，不在本轮实施存储优化。
另加新条件、特征、双Adam/双模型、逐阶段日志及SHA回执，磁盘建议至少25GiB。
bank特征可CPU驻留，不应在内存累积8192图Tensor；逐图提交/读回后释放。
历史G训练reserved约1.1GiB、H约0.8GiB，仅参考，不能据此保证新JVP/reader同时驻留峰值。
GPU阶段可顺序释放重载IRRA会话，但必须登记加载次数，并保持阶段RNG/权重精确来源。
计划时间约15–90分钟，主要不确定性为共享负载、完整图库I/O/SHA与初始化；
这只是预算，不能以8条更新规模误报“数秒Smoke”。

OOM/来源冲突/映射或冻结失败：保留当前Claim、pending、日志及已完成bank，不自动重试。
不降低FP32、不改变分辨率、batch、K、Loss或预算绕过问题。

## 完整边界和记录细节

每臂两轮预期新增G更新54、H visits16/合法更新14；
若全部正常，累计G Adam484134、H visits20496/Adam20384。
这只是两个短更新块，不发布为完整G61或H11 Epoch。
原G单步返回的临时图不进入逐步JSON，LegacySteps只保留标量/动作记录；
最终Adv Gallery在G块结束后用原generate_once重新生成。H单步的非JSON返回对象不进入日志。
RoundStore在嵌套输出路径使用显式new_base=/home/lzf/ldx/outputs/tbps-framework-gh-bidirectional-prototype；
不得把smoke_v1/A1误当成ArtifactStore允许的顶层namespace。
