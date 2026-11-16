# 属性条件的双向更新收益引导

2026-11-16后续授权：用户已要求启动Full；新执行范围及状态入口见[启动交接](GH_ATTRIBUTE_RESPONSE_LAUNCH.md)。下文CPU验收段落保留为实现阶段快照。

本版本名为 GH-Attribute-Response-v1。以原IRRA Push-pull＋三视图增强版本为基础，新增G侧一步前瞻属性重加权、H侧反事实位置/字符优势、双向收益评价器、语义属性历史记忆和可选同G梯度协调。当前阶段只授权实现与CPU验证；未启动GPU、Smoke或正式训练，没有新攻击效果指标。

## 研究目标与来源

Top-K仍决定原属性集合；新模块学习把更新力度分配给哪些属性，以及哪些字符动作能帮助对方后续更新。全局攻击图文的关系仍由Push-pull规定，不直接优化旧Cooperative四模式相似度项。

借鉴及迁移边界：

- [Learning to Reweight Examples，ICML 2018](https://proceedings.mlr.press/v80/ren18a)：以一次更新后的效果评价权重。这里把样本权重改成属性步权重，并以有限分支探测及回归蒸馏适配离散H；不声称直接实现完整可微双层元梯度。
- [Meta-Weight-Net，NeurIPS 2019官方代码](https://github.com/xjtushujun/meta-weight-net)：学习权重映射。新评价器接收语义属性、当前证据与历史响应，输出属性收益预测。
- [COMA，AAAI 2018](https://arxiv.org/abs/1705.08926)、[作者团队PyMARL](https://github.com/oxwhirl/pymarl)：借鉴反事实期望基线。这里对固定属性和其余编辑的局部合法位置/字符/NOOP分布计算基线；不是原COMA环境或原整序列策略的无偏性复现。
- [L2T-DLN，NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/hash/8667f264f88c7938a73a53ab01eb1327-Abstract-Conference.html)：借鉴以历史经验指导动态教学。新模块用有界语义属性事件和GRU编码，不复制论文完整动态损失网络。
- [PCGrad官方代码](https://github.com/tianheyu927/PCGrad)：可选投影冲突的G属性梯度。默认关闭；不比较G和H不同参数空间的梯度。

全部为本项目的适配实现，不是这些论文在TBPS上的复现结果。

## 保持的主线

- G仍是原image-only生成器，H仍是原Byte Transformer及属性/位置/替换三头。推理不加载评价器、奖励图库或配对标签。
- 使用原Natural-K、原Top-K成员及顺序、原Unicode形近替换白名单。每个原词最多修改1字符，不是整句只允许1个词；STOP和NOOP保留。
- G每条数据仍执行原K+1正式Adam更新，Local/Retain只作用在原始视图。新模块改变属性步Local系数，均值归一化为1。
- 原图/局部/全局三视图，原增强recipe及上游来源SHA保持。图像扰动相对各干净视图投影到严格8/255；正式Test对原图攻击。
- 全局KL使用固定有序2048张干净Train图像作为共同锚点。Push远离自身干净表示，Pull靠近当前对方攻击表示。原R_sim/R_rank不混入正式H主奖励；排名Margin只参与前瞻评价。
- 每个正式H阶段仍绑定当前G完整重建的2048行三视图库，拒绝旧G权重的Bank。

## 属性收益评价器与历史

CPU小网络由语义摘要Embedding、GRU历史编码、共享MLP及G→H/H→G两个输出头组成。语义键只含semantic_axis/canonical_value/region，不含PID、row_id、图像路径或原Span坐标；相同语义跨样本共享经验。

12维状态包含原分数/先验、属性数量、JVP强度及残余方向、干净/当前表示关系、轮次与已有响应。动作特征含字符位置和码位等公开动作信息。PID只在Train奖励绑定中确定正负例，不进入策略或评价器的语义键。

默认每语义键保留8条事件、最多1024个语义键、Replay最多512条；模型、完整Adam、历史、Replay、独立采样器和绝对更新计数均参与Checkpoint。历史事件保存模型轮次，拒绝逆版本写入。未知属性由共享编码器预测，空历史正常处理。

## G侧前瞻重加权

每32条正式G访问，最多探测2个当前Top-K属性：

1. 从相同G/H、完整双Adam及RNG出发，建立均匀权重基准和单属性增强分支。
2. 在克隆G上执行当前样本的完整K+1步，始终使用当前真实H语境JVP。
3. 克隆H在该候选G图像条件下执行一次原Push-pull SCST更新。
4. 在另一个合格Train样本上，比较H更新前后攻击效用的变化。
5. 相对基准分支的收益形成G→H标签，训练评价器。正式G使用评价器预测值与原属性先验生成正值、均值为1的权重。

这里的前瞻Bank是明确记录的近似：冻结完整Train参考Bank，仅替换support/probe两行。它不冒充克隆G重新生成的完整图库；正式H使用的全量当前G Bank不受影响。

## H侧反事实编辑优势及H→G标签

正式H先生成原greedy/sample，计算原Push-pull SCST。随后对每个实际编辑：

1. 固定属性和其他实际编辑。
2. 重新核验所有合法字符位置、每个位置的合法替换，以及该属性NOOP。
3. 用位置头概率乘替换头条件概率，计算完整局部条件分布；不以Top-N截断替代期望。
4. 每个候选的值为真实多视图Push-pull奖励＋预测的下一步G收益。
5. 用选中值减条件期望，形成位置/替换头的优势；属性头另外接受可编辑机会的蒸馏监督。三头的原SCST仍保留。

每次最多2048个条件候选，超出即停止，不静默截断支持集。文本编码按32条分批。原Unicode、BPE及共享词规则逐候选检查。

每32条正式H访问，最多对2个实际编辑执行H→G探测：克隆G执行一个全局Push-pull Adam步，再在另一个Train样本上评价收益。此探测明确为Global-only，不套用新文本的旧属性JVP、不生成伪JVP标签。

探测效用使用固定合法轨迹支持集的条件期望：先固定greedy及若干采样轨迹，更新前后重新计算其概率及相同检索效用。它是有限支持近似，不等于完整策略期望；相比只看greedy变化，能观察参数小幅更新带来的概率变化。效用由多视图KL奖励及Train最强负例减最强正例Margin组成。

## 隔离、保存及审计

虚拟分支克隆模型和完整Adam，不向正式权重回写。退出时严格恢复Python/NumPy/CPU/CUDA及H sampler RNG，核对正式模型、Adam、梯度、training/requires_grad均未变化。异常也执行隔离核验。虚拟G/H前向、虚拟更新和额外图文编码单独记账。

每轮保存完整可恢复边界。每10轮额外保存配对G.pt/H.pt和独立teacher.pt/runtime.json；教师导出包含完整训练状态。只在完整边界后退休本Attempt旧大Tensor；原历史实验不改动。

H全部条件候选的文本及特征按不可变Artifact保存，在退休round_io前封存进Ledger。账本数组沿用4MiB严格分片；不修改全局JSON 64MiB限制。

独立CPU审计验证：

- 原数据成员、顺序、状态链、原K+1正式更新、三视图预算和最新Bank；
- FP64重算原Push-pull、条件概率、所有候选奖励、反事实基线及完整H损失；
- 重建探测的局部Bank替换，核对每条轨迹的KL/检索Margin、更新收益和教师回归目标；
- 前瞻分支同起点/RNG隔离、教师状态链及Checkpoint身份；
- 分别核对正式与虚拟计数。不重跑G/H/Victim前向、JVP或原训练梯度。

CPU测试中的图像编码器及文本数值fixture明确为合成数据。真实BPE、H三头、Unicode动作与Adam机制可以在CPU验证，但不能由此宣称真实IRRA攻击效果或GPU可运行性已验证。

## 执行入口

当前服务器工作树：/home/lzf/ldx/projects/idea-test-tbps-gh-attribute-response-20261116

分支：codex/gh-attribute-response

默认冻结配置：configs/framework/gh_attribute_response.server.json，GPU/new_Test/automatic_training均为false。

CPU来源预检：

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-gh-attribute-response-20261116
/home/lzf/ldx/envs/tbps-image/bin/python -B operations/gh_attribute_response.py --config configs/framework/gh_attribute_response.server.json --action preflight --scope smoke
```

用户后续明确授权后，使用新未消费Attempt重新冻结授权配置，再调用独立Supervisor。完整代码保留从零60轮、10/20/30/40/50/60节点CPU审计、原生Joint Full及CPU审计、表格与曲线。当前没有执行这些命令。

按需查看状态：

```bash
/home/lzf/ldx/envs/tbps-image/bin/python -B /home/lzf/ldx/projects/idea-test-tbps-gh-attribute-response-20261116/operations/gh_attribute_response_status.py
```

GPU只观察实际分配，不设置固定空闲显存门槛、不干预外部进程；任何真实OOM/数值失败停止并保留证据，不自动重试。
