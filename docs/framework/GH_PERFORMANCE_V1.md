# GH工程加速v1：实现、验证和边界部署

本版本用于既有GH-Attribute-Response训练的工程提速，继续原60轮轨迹，不新增攻击损失或模型模块。

## 实际优化

1. 原图CPU缓存：按原始图像SHA验证，缓存384×128 FP32预处理结果；返回副本，防止调用者改坏缓存。
2. 四个CPU工作线程、最多16项视图预取：沿用原row/round确定性增强recipe，在CPU准备原图/局部/全局三视图，然后上传GPU。
   消除原路径CPU→GPU→CPU增强→GPU的往返。原图缓存最多2048张，约1.13GiB；增强预取有界。
3. H完整合法候选KL向量化：IRRA文本编码仍保持原batch上限32；奖励按128个候选一组在GPU计算全部三视图/2048参考候选分布。
   所有合法候选、概率、NOOP、顺序与Unicode预算保留，不截断候选。
4. CPU收益评价器复用相同历史：同一次预测中，相同历史和版本只运行一次GRU历史编码；动作/状态、后续trunk和输出头仍逐行计算。
   评价器训练forward/Adam、历史写入及Replay逻辑不变。

没有改逐属性K+1 Adam、H三头SCST、Push-pull、Local/Retain、前瞻频率或分支数。
没有改变原文本编码batch32、FP32/math/no-TF32、每10轮Checkpoint及Joint Full协议。
每步模型/梯度SHA、预算与冻结检查、原子发布、独立CPU审计保持。
本轮没有实现异步发布或删减审计日志，以保留原提交语义。

## 来源兼容和审计

运行时从原工作树导入原模块，原213份producer及科学配置SHA继续由原预检验证；没有编辑旧源码文件。
独立兼容桥明确安装新的输入准备、候选奖励和评价器预测实现，不把新实现伪装为原冻结实现。
configs/framework/gh_performance_v1.json锁定9份新运行工具、9份原接线源码及原科学配置SHA。
运行工具和原接线SHA不匹配即拒绝；CPU预检仍在CUDA初始化之前。

每个新训练Round的Ledger新增runtime_optimization来源标记，绑定独立runtime_tools/performance_v1不可变Manifest。
每节点先运行原完整CPU数值/预算/状态链审计，再运行额外的优化来源覆盖审计；
两项都完成后才执行正式Test。旧Round及10/20/30结果不重写。
GPU KL与原CPU KL存在微小浮点舍入差别，保持原独立数值审计容差，不承诺优化前后完整CUDA轨迹逐位一致。

## 用户数值验收口径

用户最新要求：工程数值对照绝对误差不超过0.001即可，不再为更小差异追加优化。当前实现已满足这一要求；原历史CPU审计容差因现有结果也已通过而保持，预算和SHA身份仍严格验证。

## 已完成验证

28项CPU用例PASS，包括真实Train图像384×128、两轮recipe、预取像素逐值一致；
完整候选覆盖、奖励/条件优势/梯度对照、非有限值/错误类型拒绝、缓存失效；
已学习非零评价器的历史复用输出逐位一致、RNG与模型/Optimizer/记忆不变；
只向本项目准确PID发停止信号、真实失败不交接、未提交claim和状态SHA冲突拒绝。

真实Train特征、Round30训练好的评价器及克隆H/完整Adam对照：

| 验证环节 | 原耗时 | 优化耗时 | 结果 |
|---|---:|---:|---|
| 18条真实候选的KL计算 | 10.10ms | 0.49ms | 约20.63倍，最大差9.54e-7 |
| 64行重复历史的评价器预测 | 11.47ms | 2.72ms | 约4.22倍，输出逐位一致 |
| 一条真实Train文本的克隆H更新，三次交替计时中位数 | 323.33ms | 306.12ms | 耗时减少约5.3%，吞吐约1.056倍 |

H更新样本cuhk:22257:0，3次实际编辑、13条反事实文本；
采样/贪心动作、K、一次H更新及评价器状态一致。
最大H梯度差1.04e-9，更新后参数最大差7.45e-9。
该对照不调用前瞻，原R_sim/R_rank辅助诊断使用双方相同的零fixture，两者均不参与当前优化目标；
正式接线另以原RewardBank验证。以上为局部/单样本性能，不是整轮加速或正式攻击指标。

真实运行时接线验证也已PASS：克隆G执行4次原属性/联合更新，克隆H使用原完整RewardBank执行1次更新，双方非活动模块保持精确冻结；像素预算通过、IRRA/Reader冻结通过、历史JVP重算0。验证文件与正式训练目录分开，克隆更新全部丢弃。

首个开发期GPU诊断因先初始化CUDA再调用CPU预检，被原CPU隔离门禁正确拒绝；
已修正验证入口顺序，保留失败日志。正式训练未受影响。
没有重算历史JVP、没有覆盖训练Ledger或权重；对照更新仅发生在克隆状态。

## 部署流程

新工作树：

    /home/lzf/ldx/projects/idea-test-tbps-gh-performance-v1-20261119

分支：codex/gh-performance-v1。

新控制目录：

    /home/lzf/ldx/tmp/gh_performance_v1_20261119

部署器先核验测试/基准与运行源码身份，启动原来源CPU预检。
随后只终止本项目旧恢复Supervisor，并向其训练子进程发送SIGTERM；
原训练器处理信号后完成当前Round、原子保存完整状态并正常退出。
部署器等待该完整边界，核对Manifest/SHA和无后续claim后，恢复G/H、双Adam、RNG、评价器/记忆/Replay。
不回退或重做正在完成的Round，不操作外部任务，不按估算空闲显存等待。

接续到40/50/60节点，每节点仍为训练审计→完整CUHK-PEDES×IRRA Joint Test→独立CPU审计；
最后发布六组结果与曲线。各科学阶段独立进程与日志。真实失败停止，不自动重试。

服务器检查：

    /home/lzf/ldx/envs/tbps-image/bin/python -B /home/lzf/ldx/projects/idea-test-tbps-gh-performance-v1-20261119/operations/gh_performance_supervisor.py status

WAITING_COMMITTED_ROUND_BOUNDARY表示旧训练器正在完成当前整轮，新优化训练还未开始；
此阶段训练快照中的旧PID仍应存活。child.json可能仍指向已经完成的来源CPU预检进程。
train_to_40/50/60表示新的优化训练阶段，真实轮次读取训练快照。
FAIL_STOPPED_NO_RETRY表示失败停止，应保留现场。

本说明实现时部署尚未启动；最终启动核验另附在末节。
