# G/H双向交替设计：仅方案，未执行

## A/B/C的机制区别

| 方案 | G监督 | H反馈 | 意义与风险 |
| --- | --- | --- | --- |
| A0仅时间交替 | 原文本固定z/JVP/S_raw | 旧冻结bank | 仅调度，不构成新交互 |
| A1单向刷新控制 | 原文本固定z/JVP/S_raw | 每轮最新G bank，原Reward | G→H成立，H→G缺失 |
| B双向条件交替 | 上轮H文本的新语境z/JVP，原语义锚点 | 最新G bank，原Reward | 建立双向依赖；仍非端到端同时反传，不保证互补 |
| C显式Joint/交互目标 | 新Joint检索Loss与受控交互项 | 与之协调的新Reward | 目标层交互更明确，但新Loss/Reward是额外研究变量，收益未证 |

建议先A1对B做匹配控制，再考虑C，不能同时改K、编辑空间、Loss和调度后把收益归因于交替训练。
仅最大化Joint相对单模态的差值可能靠削弱单模态投机，必须锚定绝对Joint检索破坏。

## G阶段

H冻结，从原Caption生成T_adv，禁止递归攻击上轮T_adv造成输入漂移。
语义以原Canonical/Span为锚点，通过完整trace得到edited_span及字节位置。
若G Global改读T_adv，必须真实重编码z_Tadv；Local若在新语境则重算token定位、reader w/e、方向和JVP。
basis Q/ontology在来源一致时可复用；不能因维度相同复用原token权重或原JVP。
首原型固定原S_raw排序与选中集合，隔离H→G文本条件变量。
“让G关注Joint尚未攻击成功属性”另需Train-only新归因/评分或目标，属于新增变量，当前没有已验证方案。

保留G(I)结构、原K+1更新、Global/Local/Retain函数和最后从原图开始的Joint步。
但传入z/JVP改变仍改变实际监督目标，必须称条件适配，不能宣称原训练轨迹不变。
建立新schema/来源协议，不绕过旧clean==old_z_t及selection/caption/SHA门禁。
解析不出可靠新语境映射则停止，不能用原向量或零向量假装补齐。

## H阶段与缓存

RewardBank初始化gather静态特征、按Query缓存TrainGallery context；
应每轮新建不可变bank和RewardBank实例/round绑定，不原地替换旧context。
H仍仅接原Caption，SCST公式和两个权重可在B保持不变，PID/配对只在Train上下文。

| 变化 | 可复用 | 必须失效/重新验证 |
| --- | --- | --- |
| G权重改变 | 原图/Clean IRRA特征、原文资格、原语义basis | Adv RGB/Adv特征/RewardBank context及对应reward |
| H权重改变 | 固定G图库和Clean特征、原Caption资格 | 当前H动作/新文本编码；G若消费新文本则对应条件 |
| Caption改变 | 原语义锚点、trace、同源basis | Token IDs/EOT/激活/reader w/e/z/JVP |
| IRRA/Tokenizer/预处理改变 | 原始RGB与文本可引用 | 对应表示/Score及reader相关缓存，不跨空间重标 |
| 部分图像改变 | 逐单元内容SHA完全相同且来源合法时可复用该单元特征 | 未刷新2040图不能冒充全部来自新G |

G改变通常要完整刷新2048 Train Adv bank；Clean bank可只读复用。
不使用混合G版本图库伪装同一Checkpoint。Validation128单独保存，不进入Reward或梯度，Test不参与。
G不变时可复用同SHA Adv bank；H文字变了不要求重算图像，但必须重编码当前文本。

## 最小两轮原型（未来需授权）

冻结完整2048 Train/128 Validation身份及PID隔离、原图/Canonical来源。
起点用明确的G/H完整Checkpoint（例如G60与H10作为新的组合身份），含两Adam和全部RNG；不继承旧Smoke更新。
A1、B各自独立状态及输出，匹配样本顺序、实际G步数、H visits、合法更新、数值/预算/随机起点。
差异仅为H→G条件反馈；不要把B称为只改调度。

预先按资格锁定8条Train更新样本及顺序，含可编辑、NOOP和不同K，不看攻击收益。
仍使用完整2048奖励图库，避免新增负例池变量；固定128 Validation身份，
最小链路可只用其中固定8条，不选Checkpoint、不报告总体性能；完整128评估另行授权。

轮0：核验父状态、Clean bank、原条件和隔离身份，无更新。
轮1：
1. H0冻结，取得8条原文的greedy动作/T_adv；A1使用原文目标，B生成新语境条件。
2. 仅G更新8条，每条原K_i+1步；H、IRRA、reader/basis冻结。
3. 固定G1，为全部2048图各生成一次Adv RGB并真实IRRA编码，发布独立bank。
4. 仅H更新固定8条Train，原SCST sampled/greedy三元组；G/IRRA冻结。
5. 发布整个round边界，执行固定8条Validation链路核验，无反馈/选优。
轮2：H1→G2→H2同样步骤，不重置Adam/RNG；第三轮只能事先固定，不能看效果加轮。

## 预算

每臂两轮G训练步数=2×sum_(8样本)(K_i+1)，按Train均值约63，实际按锁定样本登记。
H最多16合法更新；不可编辑保留visit/skip，不补抽。
每臂两次全库刷新：4096次G生成、4096次IRRA Adv图像编码，Clean编码0。
H SCST文本最多48条；G新语境z/JVP与定位的额外文本前向/自动求导要单列，
JVP内部autograd.grad不能叫无反传。两个实验臂再乘2，不隐藏全库刷新成本。
这比先做60轮重训小得多，但不等于只跑16次H。

已有历史G训练约1.1GiB、IRRA/H训练reserved约824MiB只是单任务量级参考；
新JVP图/同时驻留可能更高，不能直接保证峰值。授权后先验证一个新语境条件再冻结资源门槛。
规划预留十几分钟至一小时，不是实测耗时；全库SHA、I/O和重复加载可主导时间。
两轮每臂完整三Tensor图库纯数据约6.75GiB，两臂约13.5GiB；
连同G/H/两Adam、条件、日志及审计，建议独立可用磁盘至少25GiB并实际预检。
本轮没有执行任何上述GPU或训练。

## 证明真实双向信息流

H→G：同G父状态/图像/RNG/固定属性，比较原文本与H文本产生的真实z/JVP SHA、
G loss/梯度/单步参数差。H为NOOP时条件应相同，作为阴性控制。
把已记录文本置回原文，差异应随输入条件改变；不能只换Checkpoint标签或日志。
若IRRA特征恰相同/映射失败，应如实记录该固定样本无效，不能换样或关闭检查。

G→H：先固定H和同一组sampled/greedy文本，比较旧/新bank的R_sim、R_rank、Advantage。
再用相同RNG做H真实更新，记录梯度/参数差及后续固定Caption rollout。
同H权重、同Caption下G变了不会直接改变H输出；这是Caption-only约束的正确行为，
应通过奖励和H更新体现影响，不能给H.forward塞图像/PID来“证明协调”。
输出改变不保证攻击更强，不以正收益作为链路PASS。

## 稳定性、恢复、失败

一阶段仅更新一侧；两Adam/样本游标/Python/NumPy/Torch/CUDA/H采样Generator RNG分别保存。
G每Epoch8068与H2037更新不等计算量，另报前向/IRRA编码/自动求导/时间。
先短块G→完整bank刷新→短块H，块长以后可单独研究；不直接指定60轮交替。
固定原文、固定Validation，报告各单模态及Joint，防止双方抵消或靠单模态退化放大交互差。

round Manifest绑定G/H/两Adam/RNG、父链、阶段、bank及文本条件/JVP SHA、有序成员、实际计数。
所有子事务完成并审计后才提交round；未完成Claim/pending禁止自动重放。
来源/映射/预算/冻结/有限性失败即停止，保留旧现场。
恢复/回滚必须单独授权与新Attempt，不删pending、不静默重跑已发生的更新。
