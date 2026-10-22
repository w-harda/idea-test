# G/H双向交替协同最小原型：CPU准备与待授权GPU方案

基准：0b87fa55caf04e7b636bd8d0bf74b13ef3a759ff。分支：codex/gh-bidirectional-prototype。
本轮仅实现CPU可执行控制流、薄接口、合成测试及真实来源检查。没有运行真实G/H/IRRA、JVP或训练。
真实GPU运行器的模型组装、导入隔离与数值验收仍待后续授权；不能把本报告当成GPU Smoke PASS。

## 核心结论

| 问题 | 本轮决定 |
| --- | --- |
| H如何影响G？ | 把当前H从原Caption生成的攻击文本作为新线性化语境，重算IRRA整句特征、原Canonical锚点下的定位w/上下文e及JVP；把这些真实新向量交给原G训练函数。 |
| G如何影响H？ | G更新后生成并编码全部2048 Train奖励图像，发布单一G版本的新bank；新RewardBank提供原Reward与SCST Advantage，进而改变H更新。 |
| 是否把文本输入G.forward？ | 否。G.forward仍为G(I)；文本只影响训练监督。H.forward仍为H(T)，PID/配对只在Train奖励绑定层。 |
| 是否增加Loss/Reward/K规则？ | 否。保留K+1逐属性/原图Joint更新、原Global/Local/Retain、SCST及R_sim+R_rank；固定原G的S_raw排序/K，不强制H选择与G重合。 |
| A1对照 | G用原文本条件，H用最新G bank。只存在G→H反馈。 |
| B实验臂 | G用H文本的新语境条件，H用最新G bank。验证两个方向的信息传递，不以mAP下降作为本轮成功条件。 |
| GPU状态 | 本轮只有一次nvidia-smi即时资源快照；真实执行未授权、未启动。详见SOURCE_EVIDENCE.json。 |

## 1. 原版复用与实现边界

新增src/tbps/alternating/，不改任何prototypes或历史训练器。

- contracts.py：Caption-only Trace、Canonical锚点、条件身份、完整图库身份及缓存依赖。
- conditions.py：先调用原constraints.replay校验完整EditRecord，再转换轨迹；A1/NOOP复用门禁；
  CanonicalLocator调用原FrozenLocalization.attach；LegacyContextKernel调用原TextTransport.read和condition_schema.build；
  LegacySteps仅转发原train_single_sample与train_step。
- state.py：复用ArtifactStore及既有tree_hash；保存双模型/双Adam、两套阶段RNG、父来源及完整round游标；
  原子发布、SHA核验、不可变Claim和显式完整边界读取。
- runner.py：8条更新样本、两臂×两轮、完整2048 bank、单侧更新/模型空间/来源/顺序检查的真实控制流；
  当前只接受synthetic_cpu_fixture后端。没有真实GPU命令或隐式启动通道。
- preflight.py：标准库读取SHA、成员/PID/原Caption/原图、父元数据与旧协议；可选一次资源快照，不反序列化权重。
- tests/framework/test_gh_alternating.py：调用上述控制流及原Loss/Reward纯函数，用合成CPU状态/特征验证。

已有工具直接复用：ArtifactStore、原条件schema、原投影和G单样本训练、
原H规则/动作回放/SCST、IRRA原生编码、TrainGallery/RewardEvaluator/RewardBank。
本轮不引入统一Training Engine，不重写损失/JVP算法，不修改H7 Selection。
定位与新语境JVP接口已有可测试实现；其真实Tokenizer/reader实例和导入闭包尚未在本新协议中执行。
因此可称“CPU原型准备完成”，不能称“真实双向训练链路已通过”。

## 2. 为什么监督真的会改变下一次G更新

令原文本为T，当前H的greedy结果为T̃。原属性锚点a_i及原S_raw排序、权重α_i固定。

A1：z=z(T)，v_i=J_T B_i(T)。
B：z=z(T̃)，v_i=J_T̃ B_i(T̃)。

z是IRRA归一化整句特征。B_i=w_i⊗unit(e_i)，其中w/e须在当前T̃语境中通过原冻结reader、
原basis及映射后的字符Span重新取得。J仍由原TextTransport的block8→输出编码链和原JVP计算。
原Q/basis/ontology可引用；原token权重w、上下文e、隐藏激活、EOT/Tokenizer和JVP不可因维度相同复用。

每个原G属性步最小化原函数：
L = <f_I(I_adv),z> + <f_I(I_adv),v_i>
    + ReLU(<f_I(I_adv),v_(i-1)> − stopgrad(reference_(i-1)))。
第一属性无Retain；最后从原I重新开始，以Σα_i v_i作原Joint Local监督，不新增项。
因此，即使G(I)没有文本输入，z/v改变也会改变传回图像及G参数的梯度。
“向量发生变化”不保证Adam后参数可分辨，更不保证攻击更强，必须实测并分别记录。

新外层ConditionIdentity绑定原文SHA、当前语境SHA、固定G属性ID/原S_raw、模型/reader来源与映射SHA。
传入旧build的字典只是原训练函数兼容载荷；发布使用新外层身份，绝不重写旧condition.pt或规避旧Frontend的clean==old_z_t检查。
S_raw保留“原文语境原评分”的含义，不把它改名为T̃的重评分。

## 3. 原Canonical到Unicode文本的映射

先用原constraints.replay核验完整原生EditRecord，包括原/新码位、原UTF-8偏移、原Span、
预算、每属性/共享词限制；再保存全部native_action摘要和原生动作。
匹配依据原Caption及六字段semantic_axis/canonical_value/region/raw_text/char_start/char_end，
不依据G/H ID字符串，不做模糊匹配或重新解析Unicode文本。

单码位替换保持字符数及字符Span；UTF-8字节Span按攻击文本重新计算。
原Canonical和语义ID作为稳定锚点。raw_text和linked_object的当前文本片段随新语境更新；
linked_object改变可导致旧实现instance_id改变，因此分开保存原anchor/G ID与context_instance_id。
新SelectedAttribute只承载原方向对象所需数据；不把它声明成原clean prepared_plan，也不伪造新S_raw。
缺失、一对多、Token对齐不支持、截断、无有效e、零/非有限JVP、FD门禁失败均停止，不补零、不回退旧JVP。

NOOP且原文本、完整原条件和来源完全相同：B显式标记NOOP_identity_reuse，编码/JVP新增0。
A1无论H如何编辑均使用原条件。这不是把NOOP扩展为“Unicode文本可以复用旧JVP”。

## 4. 固定父状态与样本协议

新组合父状态选择G60+原H10，沿用上一阶段两轮方案，不使用H60或Validation最低候选重新选模。
两者原训练条件不同，故本协议是新的组合实验身份，不宣称延续某一条历史联合训练轨迹。
父SHA、源文件和所有2048成员锁在configs/framework/gh_alternating_prototype.server.json。
H60、G60、H7 Selection及已完成训练/Validation产物全部只读。

- G60：9148a6ac06178b2eb474194ef6892c50182c5623b4e88bdcf88faa358aa46049。
- H10：4ae964d51ea94840de71d9597d90a2234a36104e5d649cfee96577f59f94bacd，20480 visits/20370 updates。
- IRRA：9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009。

| 更新顺序 | row_id | 原G K | H资格 |
| --- | --- | --- | --- |
| 1 | cuhk:26530:0 | 0 | 固定NOOP，不补抽 |
| 2 | cuhk:26747:0 | 1 | 合法 |
| 3 | cuhk:12746:0 | 2 | 合法 |
| 4 | cuhk:9955:0 | 3 | 合法 |
| 5 | cuhk:29253:0 | 4 | 合法 |
| 6 | cuhk:21540:0 | 3 | 合法 |
| 7 | cuhk:3127:0 | 3 | 合法 |
| 8 | cuhk:8385:0 | 3 | 合法 |

选择规则：原资格顺序首个不可编辑K0；各K1/2/3/4首个合法且所选G锚点严格可映射样本；
再按原资格顺序补足8条。未读取攻击收益或AP。每轮同序，不按结果换样。
原2048 Train/128内部Validation及Test PID隔离核验；本Smoke不运行Validation或Test评估。
完整Train图库保留全部2048张，包括H不合格样本作为奖励图库成员。

## 5. 两轮控制流

每臂分别从完整G60/H10、两Adam和各父RNG的独立副本开始，不继承另一臂或旧Smoke更新。

每轮固定：

1. 冻结H，从原Caption做greedy回放；不把上一轮攻击文本递归送回H。
2. A1取原条件；B对非NOOP建立新语境条件。IRRA/reader始终eval、参数冻结。
3. 只训练G：按上述8条执行原K+1步，H不变；每轮27次G Adam更新。
4. 固定刚更新的G，对原2048 Train图逐图generate_once并原生编码；
   原图FP32输入桥保持，严格8/255。全部通过后原子提交新bank。
5. 新建RewardBank实例，清空旧Query上下文；只训练H，保留原资格skip规则，
   每轮8 visits、最多7次Adam更新。每次真实编码原文/sampled/greedy。
6. 核验单侧冻结、字节SHA、真实计数、阶段RNG、完整bank及各日志，再提交整个round边界。

第二轮从该臂上一轮完整状态继续。A1/B第二轮H文本可以因第一轮反馈而不同，这是处理效应，不强求动作一致。
两臂匹配G更新数、H visits/资格更新数、预算、原Loss/Reward、初始化和样本顺序。
实际K/编辑/STOP/NOOP全部记录；不强迫消耗预算。

## 6. G→H的真实路径

新G权重使I_adv与IRRA Adv Gallery特征改变，再使同一文本在新图库中的paired score和
正负margin改变。原RewardEvaluator计算R_sim、R_rank及Sampled−Greedy Advantage；
原SCST使用 −detach(Advantage)×logπ(sampled)更新H。

同一H权重、Caption和采样RNG下，换G不应直接改变H输出。
证明方式是在同一组已经编码的sampled/greedy文本特征上，对旧/新合法Train bank分别调用原RewardEvaluator，
再确认真正用于H更新的是新bank，并记录原SCST梯度/参数变化。不得给H增加图像或PID输入。
synergy_increment只诊断，不增加新交互Reward。

## 7. 因果接线验收，不做性能选优

- NOOP阴性：同父状态、输入、RNG及原条件下应相同。首轮首样本置于两臂第一位便于直接对照；
  第二轮两臂G可能已不同，不能要求其梯度天然相同。
- H→G：至少一个固定非NOOP样本产生真实新z/JVP，且在同G状态下产生可分辨梯度及后续参数差；
  保存首个可编辑样本两臂更新前状态、梯度向量SHA/差值、原条件和新条件。
  原G observer读取已有p.grad，不重写训练函数，不为了显示差异调整样本或步长。
- G→H：固定同文本特征，旧/新bank的Reward及Advantage差异真实存在；新bank用于后续原H更新。
- 如果文本特征相同、梯度舍入后不变、两条文本相同或Advantage仍0，分开报告“数据链存在”与“更新影响未证实”；
  不能靠增加Loss、强制编辑或换样使其PASS。
- NOOP差异、来源失效、非有限、预算/冻结/顺序异常则FAIL。负Reward/负synergy本身不是FAIL。
- 不要求攻击mAP变好，不计算新Test排名，不输出虚构协同增益。

## 8. 缓存失效

| 变化 | 允许复用 | 必须新建/拒绝旧缓存 |
| --- | --- | --- |
| G权重 | 原RGB、Clean特征、原Canonical和固定原条件 | Adv RGB、Adv图像特征、RewardBank及每Query context |
| H权重 | 同G图库、原Caption资格、原语义basis | 当前H动作；B消费的文本条件/文本特征/JVP须按内容重新绑定 |
| Caption内容 | 原Canonical、合法完整trace、同源basis Q | Token IDs/EOT、block8、reader w/e、新z/JVP |
| IRRA/Tokenizer/预处理 | 原始像素/UTF-8输入 | 对应ModelSpace、特征与评分，不跨模型重标 |
| 同一输出文本再次出现 | 只有完整条件key及来源一致才可显式引用 | 不能仅凭H epoch或向量维度复用 |

图库身份绑定G状态SHA、全部2048有序身份、固定Clean来源、Adv内容和IRRA ModelSpace。
不是8张新图+2040张旧图；也不能将RaSa/APTM特征装入IRRA奖励bank。
condition使用完整ConditionIdentity.key；reward缓存还必须区分正例mask/paired index，不能只按Caption字符串缓存。

## 9. 双Adam/RNG与失败恢复

G使用原Adam2e-4/betas(.5,.999)；H使用原H10配置和完整Adam，不新建空动量。
父Checkpoint两套全局RNG不可能同时覆盖同一进程，因此分成G_RNG/H_RNG两个阶段域；
每域保存Python、NumPy、Torch CPU、CUDA及H sampler状态。
阶段开始恢复对应域，结束捕获；所有模型构造/加载应在恢复阶段RNG前完成。
两臂复制相同两个域，JVP/诊断必须不意外推进训练域；无需随机的特征生成仍记录域前后摘要。

round提交包含双模型/双Adam/两RNG/原父来源、当前round/实际更新数、
有序样本、条件SHA、bank SHA、日志及阶段计数。保存前克隆CPU Tensor，避免大Storage保留。
原子Claim先于任何更新；已提交版本不可覆盖。未完成Claim、pending或更晚Claim存在时拒绝自动恢复，
不静默回退上一轮重做已发生的更新；保留FAIL现场，恢复需新的明确授权和人工核对。
完整已提交边界可在无后续Claim且来源一致时显式读回恢复；本轮CPU测试证明合成下一步状态exact，
不宣称未来CUDA数值轨迹bitwise等价。

## 10. 下一步

GPU预算、执行步骤、资源风险及停止条件见EXECUTION_PLAN.md。
测试与只读来源结果见TEST_MATRIX.md和SOURCE_EVIDENCE.json。
共享属性选择、K预算消融、新Joint Loss/Reward、60轮联合训练均不在本阶段范围。

## CPU复核命令

以下只运行新CPU测试与只读预检，不启动任何真实模型：

~~~bash
cd /home/lzf/ldx/projects/idea-test-tbps-gh-bidirectional-prototype-20261022
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TMPDIR=/home/lzf/ldx/tmp
export PYTHONPATH=/home/lzf/ldx/projects/idea-test-tbps-gh-bidirectional-prototype-20261022/src
/home/lzf/ldx/envs/tbps-image/bin/python -B -m pytest -q tests/framework/test_gh_alternating.py
/home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.alternating.preflight --config /home/lzf/ldx/projects/idea-test-tbps-gh-bidirectional-prototype-20261022/configs/framework/gh_alternating_prototype.server.json
~~~

真实GPU入口尚未开放，本轮不提供可意外启动训练的命令。
GPU执行方案审核后才组装并锁定生产后端。不能把修改gpu_execution_authorized字段当作授权。

## 本轮实际验收结果

- 61项新增CPU原型测试，连同26项前置属性审计及75项契约/配置/存储回归，共162项PASS，退出码0。
- 真实来源预检PASS：49关键来源、113旧producer、132当前producer、7项G prepare、
  4450 upstream、76项H训练来源、8192个原Clean图库关联文件SHA核验。
- 151378份历史文件及10918份上一审计来源metadata保持，父权重与H60/H7 Selection关键SHA保持。
- 唯一资源快照：RTX4090空闲6501MiB、利用率49%；外部PID2595393占17656MiB，未干预。
- 未加载/前向真实G/H/IRRA，新增文本编码、JVP、真实G/H更新均0；smoke_v1未创建。
- 新语境数值等价、真实双向更新差与GPU独立审计仍为PENDING；完整日志和SHA见SOURCE_EVIDENCE.json。

## 继承来源的研究限制

原G的S_raw缓存来源仍为旧协议的34054张official Train竞争图库，其中包含内部Validation图像；
本轮不重算或改写这批S_raw，也不把它描述成gallery-disjoint监督。
A1/B使用完全相同的这一固定来源。新更新样本与新H奖励bank严格限定2048 Train，
内部Validation仅做身份/PID隔离核验，不产生新反馈或选优结果。
