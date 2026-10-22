"""独立中文科研报告生成；只读本轮派生统计，不修复旧发布器或实验产物。"""
from pathlib import Path
from collections import Counter
import ast,csv,json,hashlib
ROOT=Path(__file__).resolve().parents[3];O=ROOT/"docs/framework/results/hv1_g60_posthoc_analysis"
s=json.loads((O/"statistics.json").read_text());f=json.loads((O/"supplementary_statistics.json").read_text())
t={x["epoch"]:x for x in s["training_epochs"]};v={x["H_epoch"]:x for x in s["validation"]}
a=list(csv.DictReader((O/"action_epoch_statistics.csv").open()))
va={int(x["H_epoch"]):x for x in a if x["scope"]=="Validation"}
hist=list(csv.DictReader((O/"action_distribution.csv").open()))
q=list(csv.DictReader((O/"per_query_analysis.csv").open()))
qr={(x["candidate"],x["group"],int(x["query_index"])):x for x in q}
vr={x["candidate"]:x for x in f["validation_reward"]}
def num(x,n=6):return f"{float(x):.{n}f}"
def pc(x,n=2):return num(100*float(x),n)
def table(headers,rows):
    text="| "+" | ".join(headers)+" |\n| "+" | ".join("---" for _ in headers)+" |\n"
    for row in rows:text+="| "+" | ".join(str(x).replace("|","\\|").replace("\n"," ") for x in row)+" |\n"
    return text
def hfrac(scope,e,dim,value):
    rr=[x for x in hist if x["scope"]==scope and int(x["H_epoch"])==e and x["rollout"]=="greedy" and x["dimension"]==dim and x["value"]==value]
    return float(rr[0]["fraction"]) if rr else 0.
def hcount(scope,e,dim,value):
    rr=[x for x in hist if x["scope"]==scope and int(x["H_epoch"])==e and x["rollout"]=="greedy" and x["dimension"]==dim and x["value"]==value]
    return int(rr[0]["count"]) if rr else 0
def save(name,text):(O/name).write_text(text.rstrip()+"\n",encoding="utf-8")
p9=s["query_comparisons"]["H9_vs_H20"];p60=s["query_comparisons"]["H20_vs_H60"]
core=table(["H Epoch","训练图像引导","Text-only R@1 (%)","Text-only mAP (%)","Joint R@1 (%)","Joint mAP (%)","Joint额外下降 (pp)"],
    [[e,"G10" if e<=10 else "H10后G60适配",num(v[e]["Text-only R@1"],2),num(v[e]["Text-only mAP"]),
       num(v[e]["Joint R@1"],2),num(v[e]["Joint mAP"]),num(v[e]["Joint_extra_drop_pp"])] for e in [7,9,10,20,30,40,50,60]])
allval=table(["H Epoch","训练图像引导","Text-only R@1","Text-only mAP","Joint R@1","Joint mAP","额外下降 pp"],
    [[e,"G10" if e<=10 else "G60适配",num(x["Text-only R@1"],2),num(x["Text-only mAP"]),num(x["Joint R@1"],2),
      num(x["Joint mAP"]),num(x["Joint_extra_drop_pp"])] for e,x in v.items()])
train=table(["Epoch","Sampled R","Greedy R","R_sim (sampled)","R_rank (sampled)","Loss均值","零梯度 %","同文本 %","Greedy编辑数","参数ΔL2均值"],
    [[e,num(t[e]["sampled_total_mean"]),num(t[e]["greedy_total_mean"]),num(t[e]["sampled_R_sim_mean"]),num(t[e]["sampled_R_rank_mean"]),
      num(t[e]["SCST_loss_mean"]),pc(t[e]["zero_gradient_rate"]),pc(t[e]["same_text_rate"]),num(t[e]["greedy_mean_edits"],3),num(t[e]["parameter_delta_l2_mean"])] for e in [10,11,20,30,40,50,60]])
alltrain=table(["Epoch","Sampled R","Greedy R","Loss均值","零梯度 %","Greedy平均编辑","Greedy NOOP %","Greedy STOP %"],
    [[e,num(t[e]["sampled_total_mean"]),num(t[e]["greedy_total_mean"]),num(t[e]["SCST_loss_mean"]),pc(t[e]["zero_gradient_rate"]),
      num(t[e]["greedy_mean_edits"],3),pc(t[e]["greedy_NOOP_rate"]),pc(t[e]["greedy_STOP_rollout_rate"])] for e in range(11,61)])
behavior=table(["H候选","总编辑/128","平均K","平均实际编辑","0编辑数","1编辑数","2编辑数","≥3编辑数","位置NOOP/位置决策","STOP Rollout"],
    [[e,int(va[e]["edits"]),num(va[e]["mean_budget"],3),num(va[e]["mean_edits"],3),
      hcount("Validation",e,"edits","0"),hcount("Validation",e,"edits","1"),hcount("Validation",e,"edits","2"),
      128-sum(hcount("Validation",e,"edits",str(i)) for i in range(3)),
      str(va[e]["position_NOOP_decisions"])+"/"+str(va[e]["position_decisions"]),va[e]["STOP_rollouts"]] for e in [7,9,10,20,30,40,50,60]])
flip9=table(["Query index","row_id","正例数","H9→H20 Rank-1","最佳正例排名","AP H9→H20","Margin H9→H20"],
    [[x["query_index"],x["row_id"],qr[("H9","Joint",x["query_index"])]["positives"],
      str(x["rank1_a"])+"→"+str(x["rank1_b"]),str(x["best_rank_a"])+"→"+str(x["best_rank_b"]),
      num(x["AP_a"],4)+"→"+num(x["AP_b"],4),num(x["margin_a"],6)+"→"+num(x["margin_b"],6)] for x in p9["rank1_flips"]])
flip60=table(["Query index","row_id","AP H20→H60","最佳正例排名","编辑数H20→H60","Margin H20→H60"],
    [[x["query_index"],x["row_id"],num(x["AP_a"],4)+"→"+num(x["AP_b"],4),str(x["best_rank_a"])+"→"+str(x["best_rank_b"]),
      str(x["edits_a"])+"→"+str(x["edits_b"]),num(x["margin_a"])+"→"+num(x["margin_b"])] for x in p60["rank1_flips"]])
bootstrap=table(["比较（后者−前者）","官方summary mAP差 pp","逐Query FP64差 pp","PID Bootstrap mAP 95%CI pp","R@1差 pp","R@1 95%CI pp"],
    [[name,num(v[b]["Joint mAP"]-v[a0]["Joint mAP"]),num(p["bootstrap_AP"]["difference_b_minus_a_pp"]),
      "["+num(p["bootstrap_AP"]["CI95_lower_pp"],3)+", "+num(p["bootstrap_AP"]["CI95_upper_pp"],3)+"]",
      num(p["bootstrap_R1"]["difference_b_minus_a_pp"],4),
      "["+num(p["bootstrap_R1"]["CI95_lower_pp"],3)+", "+num(p["bootstrap_R1"]["CI95_upper_pp"],3)+"]"]
     for name,a0,b,p in [("H20−H9",9,20,p9),("H60−H20",20,60,p60)]])
attrtypes=["gender/person","color/upper","color/lower","garment_type/upper","garment_type/lower","backpack/carried","bag/carried","hair_length/head"]
attrs=table(["原始semantic_axis/region","E11 Greedy编辑占比 %","E20占比 %","E60占比 %"],
    [[name,*[pc(hfrac("G60 adaptation",e,"axis_region",name)) for e in [11,20,60]]] for name in attrtypes])
chars=["U+0061→U+0430","U+0065→U+0435","U+006F→U+043E","U+0075→U+03C5","U+0069→U+0456"]
charmap=table(["原→新Unicode码位","E11占比 %","E20占比 %","E60占比 %"],
    [[name,*[pc(hfrac("G60 adaptation",e,"character",name)) for e in [11,20,60]]] for name in chars])
rewardval=table(["候选","派生R_sim均值","派生R_rank均值","派生R_total均值","Joint mAP","R>0但AP上升Query数","R<0但AP下降Query数"],
    [[key,num(vr[key]["R_sim_mean"]),num(vr[key]["R_rank_mean"]),num(vr[key]["R_total_mean"]),num(v[int(key[1:])]["Joint mAP"]),
      vr[key]["positive_R_but_AP_increased"],vr[key]["negative_R_but_AP_decreased"]] for key in ["H9","H20","H40","H60"]])
hypotheses=[
 ["A. SCST有效信号变弱","同文本→同Reward→Advantage=0→Loss梯度0。",
  "E20零梯度64.46%→E60 82.18%；各50轮同文本比例等于零梯度比例。",
  "支持信号变稀疏；不支持所有非零梯度趋零：条件非零梯度均值0.995→1.827，全部合法Adam步仍改变H。",
  "需要受控探索/奖励对照才能判断是否为退化原因，当前不能归因。"],
 ["B. Sampled与Greedy趋同","更集中的策略更常产生相同最终文本。",
  "相同文本64.46%→82.18%，完全相同编辑/决策轨迹52.92%→71.18%；Advantage标准差0.04116→0.02508。",
  "支持输出趋同；不支持100%确定或已证明模式坍塌。平均Reward差未单调缩为0。",
  "需要动作概率/熵日志或多seed/受控实验，当前未新增。"],
 ["C. 字符策略更保守","少编辑、位置NOOP或STOP使攻击空间利用下降。",
  "H9/H20/H60验证编辑278/243/238；位置NOOP 0/43/38；总NOOP 1/5/6。",
  "相对历史H9部分支持，但H20→H60不单调；STOP仅0/0/1次，反对大量提前STOP的说法。",
  "需固定预算/动作支持的反事实实验；不能把编辑少直接认定为唯一原因。"],
 ["D. Reward与Joint mAP目标不一致","正确配对点积与最佳正例/最大负例Margin不覆盖完整多正例排名。",
  "同一128图库中H40派生R=0.044051>H20 0.040003，但Joint mAP41.223633>40.427635；存在正Reward且AP升高的Query。",
  "直接支持代理目标非等价；不证明它是全部后期退化的唯一原因。",
  "可另行预注册直接Train AP或多正例listwise奖励对照。"],
 ["E. 适应Train但不能稳定泛化","固定G60反馈下可学到Train特有的属性/字符模式。",
  "E20→60 Train Sampled R 0.031844→0.042965，Validation Joint mAP40.427635→42.874081。",
  "支持训练与验证趋势分离；证据不足以把原因命名为G60特有过拟合。",
  "缺G10延长训练控制、独立确认Validation和多seed。"],
 ["F. 字符空间存在能力上限","形近字符、每词一次、BPE变化与不截断等限制可能限制可达效果。",
  "平均K约2，合法跨度/替换白名单固定；E60 a/e两种唯一候选编辑合计约67.84%。",
  "限制存在是事实，但没有穷举最优/更丰富空间对照，不能证明已触顶。",
  "须固定预算的合法空间/属性策略对照；不因当前结果自动扩预算。"],
 ["G. 单seed、小Validation与选择偏差","少量Query和反复比较候选影响最小值稳定性。",
  "128 Query/121 PID；H20−H9 mAP CI跨0；H60−H20探索性CI不跨0，但H20经过同集选优。",
  "支持对小差异保持谨慎；不能把所有变化一概归为噪声，也不能宣称独立显著。",
  "新PID独立确认、冻结候选与多seed；禁止用正式Test选新H。"]
]
hypdoc="# H60研究假设逐项评估\n\n本表区分事实、机制解释与尚未证实的因果。\n\n"+table(
 ["假设","理论机制","已有实测证据","判断","尚需验证"],hypotheses)+"""
## 判定边界

A/B/C是部分被数据支持的行为变化，不能自动拼成“已证明过拟合/模式坍塌”。
D有同一固定Validation分数矩阵的直接不一致证据；E有Train/Validation趋势分离，但成因未识别。
F的算法约束已经核实，空间上限没有被证明。G的区间是固定已选候选的探索性重采样结果，
不包含重新选优的不确定性，也不涵盖训练随机种子变化。

原始值、全部50轮与逐Query结果见REPORT.md及CSV；当前不实现任何新Loss或新训练。
"""
save("hypothesis_assessment.md",hypdoc)
nextsteps=[
 ["A（优先1）","固定G10图库，从同一H10完整状态续训至60；同样六节点、绝对Epoch顺序、Adam和预算。",
  "区分延长训练本身与把奖励图库换成G60的影响；当前这两个因素同时变化。",
  "一次匹配控制分支，复用已核验G10 Train图库；预先冻结Validation规则与终点。",
  "粗略同本轮约2小时训练量级，加Validation/CPU审计；不需要重新生成G图，需数GiB日志/Checkpoint。",
  "代码接线较低；是必要控制实验，通常不是独立算法创新。",
  "单分支仍有随机性，重复Validation仍有选择偏差，结果不保证支持G60适配。"],
 ["E（确认性配套）","对冻结H9/H20/H60在新增PID隔离确认集验证，并明确多seed策略。",
  "判断近似持平和后期变弱是否能在未反复使用的数据上重现。",
  "独立official-train holdout，禁止Test调参；未来独立训练seed须新协议，不能伪称原轨迹续训。",
  "冻结候选只需一次Gallery特征及少量文本编码；多seed训练成本按分支数量增长。",
  "低至中；提升证据可信度，不是新攻击算法创新。",
  "确认集难度不同、单次仍可能波动，需冻结抽样和分析计划。"],
 ["B（控制后）","让奖励更贴近固定Adv Gallery上的多正例检索失败，如Train AP下降或listwise surrogate。",
  "解决已观察到Reward增加而AP改善/Joint mAP变差的目标差距。",
  "只改一个奖励因素，与现R_sim+R_rank作固定预算、相同父状态和数据对照。",
  "可复用2048图特征；排序/多正例计算增加CPU开销，具体需另行测量，不能承诺速度。",
  "中；可能成为方法模块，但已有相关方法多，须另做新颖性核验及消融。",
  "离散AP奖励稀疏、方差和负池依赖；更直接的目标也未必带来泛化。"],
 ["C（研究性）","在明确Joint失败主目标下加入可辨识的图文交互差分诊断/奖励。",
  "当前synergy仅记录，训练并未显式奖励相对单模态的交互增益。",
  "预注册Joint主目标与交互定义，和仅换排序奖励的B分开比较。",
  "Clean/Adv特征已有，未必增加神经前向；但奖励尺度/方差和完整对照增加开发成本。",
  "中至高；有潜在方法贡献，不代表新颖性或有效性已成立。",
  "只优化差分可能通过削弱Text-only而刷高“交互”，却不改善Joint。需绝对Joint锚点。"],
 ["D（较后）","在同一编辑预算下分析位置NOOP、属性类型与字符空间选择。",
  "H后期偏向颜色且出现位置NOOP，少数Query对替换身份敏感。",
  "明确固定预算的策略/合法空间对照，并保留可读性与原BPE规则的独立检查。",
  "中；更大动作空间会增加候选检查和方差，资源需另行测量。",
  "中至高；是否构成创新取决于新机制及独立证据，不是加大K。",
  "更激进编辑可能破坏语义/公平预算；少编辑未必是失败，禁止无授权强制编辑。"]
]
save("research_next_steps.md","# 下一阶段研究决策（建议，未执行）\n\n"+
 """**首选一个新实验：A，G10固定奖励图库的H10→H60匹配控制。**
它最直接补齐本轮的因果识别缺口。当前数据只能说“延长训练并切换G60后出现这些变化”，
还不能说“G60导致变化”，也不能据此决定应设计新交互Loss。
如资源仅允许一次实验，先完成这条控制轨迹；新增独立确认集E应作为后续性能结论的前提。
它不是通过Test最低点选模，不改变历史H7 Selection。

""" + table(["方案","具体设置","已观察问题/假设","新增实验","成本","复杂度/创新定位","风险"],nextsteps)+"""
## 数学候选仅作讨论

现有奖励已经用Adv Gallery计算条件增量，但没有直接优化AP。
SCST允许非可微奖励；因此Train-only的完整AP下降在接口上可以作为未来候选，
无需把Unicode或排序改成可微操作。这只是可行性分析，不等于授权替换当前奖励。
若采用listwise近似，应同时说明正负池、多个正例、温度和尺度的变化，不能默认等价AP。

交互量可考虑“Adv Gallery下AP下降 − Clean Gallery下AP下降”等差分，但必须保留明确的Joint失败主目标。
单独最大化交互差分，可能通过使Clean文本攻击变差而增大差分；这不能保证Joint更强。
当前的synergy_increment是配对相似度差分，不等同于该AP差分，也没有参与现训练奖励。

任何新方案均需另行授权、预注册固定对照、控制预算和独立来源；本次没有执行A–E中的任何新实验。
""")
report=f"""# H-v1 G60适配续训：完整训练动态、逐Query证据与研究决策

日期：2026-10-22。性质：CPU只读事后分析。原实验与失败发布现场保持不变。

## 核心结果总表

指标单位为百分数；mAP与额外下降保留6位以显露H9/H20的细微差别。额外下降=同一G60 Image-only mAP−Joint mAP，单位为百分点。

{core}

固定128 Gallery/128 Query、121 PID；Clean mAP=100%，G60 Image-only mAP={s["controls"]["G60_Image_only"]["mAP"]}%。
H1–H10训练使用G10奖励图库；H20–H60从原H10状态切换G60适配，**不是相同条件下完整单因素训练曲线**。
H20是五个新增预设候选中最低Joint mAP；全部15个候选的最低仍是历史H9。
这些只是重复内部Validation上的描述性最小值，未发布新Selection，未用Test选模。

**核心结论：**训练奖励确实提高，后期有效SCST梯度更稀疏，动作出现部分保守化；
但不能把它们直接当作Joint退化的唯一原因。最强的直接证据是：现奖励与完整多正例AP并不等价，
同一Validation上也存在Reward更大而Joint mAP更高的候选。H20与H9几乎持平来自样本级正负变化抵消，
不是全部Query表现相同。当前训练是G→H的单向条件引导，不是端到端双模态联合训练。

## 1. 原始实验完成与失败状态核验

| 项目 | 真实状态 |
| --- | --- |
| 正式H11–H60 | HV1_G60_TRAINING_PASS；新增102400 visits/101850更新 |
| 累计账本 | 122880 visits/122220 Adam更新；50轮均2048/2037/11 |
| 新Checkpoint | 400份周期/边界版本，含50份完整Epoch边界，链/关联日志/字节SHA核对 |
| 独立CPU审计 | 已有HV1_G60_VERSIONED_CPU_AUDIT_PASS；本轮只读验证回执，不重跑入口 |
| G/IRRA | G60固定图库2048张；训练G加载/图像前向0，IRRA参数buffers前后exact |
| 原控制器 | FAIL exit=1，WRITING_REPORT阶段Python3.10缺hashlib.file_digest |
| 本轮行为 | 不修复旧发布器、不重训、不编码、不使用GPU、不执行Full排名 |
| 保护 | 151378份历史stat及18617份实验/控制/失败报告文件状态保持；113原producer SHA通过 |

正式输出：/home/lzf/ldx/outputs/tbps-framework-hv1-g60-adaptation-epoch60/。
控制目录：/home/lzf/ldx/tmp/hv1_g60_adaptation_epoch60_v1/。
原执行代码9765a536a11a9a0e9111f2f834230c53d41bf2d8；
独立审计修复135df869b81d728261a625ca6e4a2c6b8b7f2ae5；启动cb51cd3c0f11041fadb072c2f0853b222a649421。
没有把原FAIL改写为监督器SUCCESS；训练/审计成功与原Git发布失败分别记录。

原H10完整H/Adam/Python/NumPy/CPU/CUDA RNG恢复exact，起点20480/20370，不继承4-step Smoke。
累计Adam更新是优化器调用账本，不要求每个参数的Adam step相同；原参数动量和不同step原样恢复。
内部Train/Validation均属于official train，2048/128身份、PID及官方Test PID隔离已核验。
路径含test_query不代表official Test，角色以annotation和成员身份为准。

本轮流式读取历史20480及新增102400条Visit，合计122880条；没有把全部动作日志驻留内存。
读取核验10690份输入、约{5094734131/2**30:.3f}GiB，670个新式Artifact；31? 评分数量以实际21组矩阵为准：
1组G60 Image-only、15组Joint、5组新增Text-only，均为已有128×128 FP32分数。
这些矩阵的原生Ranking索引和四项汇总与独立提取的官方rank函数逐值一致。
本轮统计执行{ s["analysis_seconds"]:.3f}秒，Peak RSS约{s["peak_RSS_bytes"]/2**20:.1f}MiB；GPU/模型/编码/旧审计入口调用均0。

## 2. 实际训练目标与权限

以同一Query的原文t、攻击文本t'，固定原图图库I及G60图库I_g，归一化IRRA评分s为例：

```text
R_sim(t') = s(I_g[paired], t) - s(I_g[paired], t')
M_g(t) = max_{{same PID}} s(I_g, t) - max_{{different PID}} s(I_g, t)
R_rank(t') = M_g(t) - M_g(t')
R_total(t') = R_sim(t') + R_rank(t')
Advantage = detach(R_total(sampled) - R_total(greedy))
L_SCST = -Advantage * sum(log pi_H(sampled actions | caption))
synergy_increment = [s(I_g[paired],t)-s(I_g[paired],t')]
                    -[s(I[paired],t)-s(I[paired],t')]
```

H通过SCST最大化期望R_total，IRRA/Unicode/Tokenizer不反传，只有H的动作log-prob参与梯度。
两个奖励权重均1；**synergy_increment、clean_sim_drop、clean_margin_drop只记录，不是额外Loss或Reward项**。
负synergy不等于Joint无效；其量纲为正确配对相似度差分，不是mAP增益。

正例是2048 Train Gallery的全部同PID图像，负例是不同PID图像；R_sim使用真实配对的训练图像索引。
PID/配对/图库仅在RewardBank绑定层，H.forward/attack仅接Caption与其Canonical/type/Span。
Train/Validation/Test权限没有混用。H推理不接收图像、PID、检索评分或测试标签。

源码证据：rewards.py的RewardEvaluator.evaluate_many；scst.py的scst_loss；
phase4_training.py的train_step；formal_training/runtime.py的RewardBank.bind、FreshReward.evaluate_many。
实际行号、文件SHA与路径见code_evidence.csv。

### 2.1 为什么奖励不是mAP

R_sim只约束一个配对正例分数，R_rank只约束最佳正例与最高分负例；AP依赖全部同PID正例在全部Gallery中的排名。
保持最优正例/最难负例分数不变，其他正例或负例的顺序仍可改变AP；
分数有变化但未越过排名边界时，Reward也可能变而AP不变。
SCST将这一非等价目标的期望推高，不会自动补上AP目标。

Validation有114条单正例Query和14条双正例Query，后者来自7个双图PID；不能简化成唯一正确图像。
下文的逐Query AP均保留多正例、官方sort/tie行为且无camera过滤。

## 3. 全部50轮训练动态

下表是在线训练期间均值，不是重新评估该冻结Checkpoint的奖励。每轮奖励/梯度分母2037，
另外11条资格失败仍计入2048 visits但不产生奖励/Adam更新，不能与“合法零梯度更新”混为一谈。

{train}

### 3.1 奖励提高，但后期增益放缓

E11→E60 Sampled R {t[11]["sampled_total_mean"]:.6f}→{t[60]["sampled_total_mean"]:.6f}，
R_sim {t[11]["sampled_R_sim_mean"]:.6f}→{t[60]["sampled_R_sim_mean"]:.6f}，
R_rank {t[11]["sampled_R_rank_mean"]:.6f}→{t[60]["sampled_R_rank_mean"]:.6f}，两项都贡献后期奖励。
E41–50与E51–60的Sampled十轮均值为0.042717与0.043619；最后十轮线性斜率约{f["late_E51_E60_sampled_reward_slope_per_epoch"]:.8f}/Epoch。
这支持“后期波动、边际提升放缓”的描述，**不证明达到理论奖励上限或优化已收敛**。
H10的0.057139来自G10图库，不能拿它与切换G60后的E11直接比较来判断模型突然变坏。

### 3.2 Sampled/Greedy趋同与梯度稀疏

E20→E60同文本比例64.46%→82.18%，同编辑/决策轨迹52.92%→71.18%；
Advantage标准差0.041159→0.025084。全部50轮同文本比例与零梯度比例完全相等，
符合相同文本→相同奖励→Advantage零→SCST梯度零的源码机制。
但平均Advantage从−0.001556到−0.001768并非单调逼近零，不能说所有奖励差异消失。

反对“梯度整体消失”的证据也存在：E20/E60平均梯度0.353605/0.325537；
只在非零梯度上条件均值0.994881→1.826773。信号发生更少，不代表每次非零信号更小。
E60仍有363/2037次非零梯度更新。Replacement Head更稀疏：E20为318/2037，E60为121/2037，
但没有真实动作概率/熵记录，不能把这些频率当作策略熵或宣布模式坍塌。

### 3.3 参数仍在变化，SCST Loss不能当攻击指标

合法101850个Adam步的parameter_delta_l2均非零；E20/E60均值为0.005383/0.004955。
E60的1674个零当前梯度步仍有参数变化，符合原Adam动量机制；没有重初始化Optimizer。
Loss中位数均为0，均值趋近0不能等同最优攻击；大量零Advantage直接使Loss为0。
Loss、梯度、Reward的均值/std/min/p05/p25/median/p75/p95/max与正/负/零计数完整保存在training_metric_distributions.csv。

所有50轮均列于文末附表和training_epoch_statistics.csv；逐轮资源时间另见training_resource_epoch_statistics.csv。
资格固定、模型冻结及没有NaN/中断的结论来自原完成/独立审计回执及本次连续日志检查，而非只凭曲线猜测。

![训练奖励](figures/training_rewards.png)
![Loss与梯度](figures/training_loss_gradient.png)
![实际参数更新](figures/parameter_updates.png)

## 4. 全部15个H候选的固定G60 Validation

{allval}

Raw Clean和Image-only固定不重算。H1–H10是旧G10引导训练，H20–H60为适配分支；
10→20之间未保存逐Epoch Validation点，不插值生成观测，也不声称这是一条纯训练长度单因素曲线。

H20比H10低{v[10]["Joint mAP"]-v[20]["Joint mAP"]:.6f}pp，说明这个已选内部集合上有观察到的适配改进；
但它比历史H9高{v[20]["Joint mAP"]-v[9]["Joint mAP"]:.6f}pp，不能称超越历史最低值。
H20→30变弱，H30→40改善约0.631pp，H40→50→60又变弱，**不是每个后续节点都持续恶化**。

Text-only也非单调变弱：H20→H40 mAP从99.283859降至98.470047，独立文本攻击反而增强，
之后H60回升到99.218750；H60相对H20仍略低0.065109pp。
准确说法是：新增候选总体比历史H9更接近Clean，不能说“训练越晚Text-only越弱”。

![Validation四项摘要](figures/validation_metrics.png)
![G10/G60背景与额外下降](figures/G10_G60_validation_context.png)

## 5. 逐Query：H9与H20为何近似持平

H9→H20有38条AP升高（攻击变弱）、36条AP下降（攻击变强）、54条不变。
正向AP变化贡献+{p9["positive_AP_contribution_pp"]:.6f}pp，负向贡献{p9["negative_AP_contribution_pp"]:.6f}pp，几乎抵消。
这不是全部Query相同，也不能仅由“净少2个Rank-1命中”推出mAP更低。

### 5.1 Rank-1净少2条实际是3条新失败、1条恢复

{flip9}

因此没有唯一的“两条Query名单”：必须同时报告3个丢失命中与1个恢复命中。
Query13有两个PID正例，Rank-1正确时AP仍是0.833333而非1，说明多正例实现确实被保留。

把“攻击成功”限定为Image-only本来Rank-1正确、Joint后错误，H9/H20成功集合分别11/12条，
交集10、并集13、Jaccard={p9["new_attack_success"]["jaccard"]:.4f}。
这与“Joint全部错误Query集合”不同，不能把Image-only已经错误的样本都算作H新增成功。

### 5.2 H60相对H20主要在哪里变弱

H20→H60有29条AP升高、20条下降、79条不变；5条Rank-1恢复正确，没有反方向的新Rank-1失败。
它们是：

{flip60}

五条的净mAP贡献为{sum(x["delta_AP_pp"] for x in p60["rank1_flips"])/128:.6f}pp，
约占总净上升{sum(x["delta_AP_pp"] for x in p60["rank1_flips"])/128/p60["mean_AP_delta_pp"]*100:.2f}%。
绝对AP变化最大的前5/10条占全部绝对变化的{p60["top5_absolute_share"]*100:.2f}%/{p60["top10_absolute_share"]*100:.2f}%；
其余Query仍有正负抵消，不能说只有这5条发生变化。

其中2条编辑数不变、1条编辑更多、2条编辑更少；**退化不能仅用“编辑字符少了”解释**。
Image-only正确Query中的新增打错数H20=12、H60=9，原本错误却被H修复的数H20=1、H60=3，合计净命中恢复5条。
具体AP/最佳正例排名/正负分数/Margin均见per_query_analysis.csv及per_query_comparisons.csv。

### 5.3 探索性PID聚类Bootstrap

{bootstrap}

使用121个PID作为重采样单位，每次保留该PID的全部Query，20000次、seed20261022、配对percentile区间。
H20−H9区间跨零，不能确认一个在总体上更好。
H60−H20区间在这次固定比较下不跨零，但H20来自重复Validation选优，
区间不覆盖重新选优的不确定性、训练seed变化或新数据分布；**不是独立确认性显著性验证**。

原summary采用Torch FP32均值；逐Query配对统计对原生FP32 AP作FP64平均，差异小于1e−5pp。
因此官方summary差和Bootstrap点估计末几位不同，二者均保留，不改原结果来消除规约尾差。

![逐Query变化](figures/per_query_AP_changes.png)
![逐Query相对Image-only](figures/per_query_AP_heatmap.png)
![配对区间](figures/paired_PID_bootstrap.png)

## 6. 字符策略究竟发生了什么

### 6.1 更少编辑主要是位置NOOP，不是大量提前STOP

{behavior}

H9总预算278且全用；H20预算286、实际243，H60预算279、实际238。
平均K并没有相对H9下降，主要是选中属性后允许“不改此属性”的位置NOOP。
H20/H60总NOOP为5/6条，均包含1条固定资格NOOP；不能全算为学到的STOP。
H20没有STOP、H60仅1条。后期实际编辑并非单调下降，H30为222、H40又为237、H60为238。

训练Greedy NOOP在E20和E60均22/2037≈1.08%；若把11条固定跳过也计入全部Visit，则为33/2048≈1.61%。
Train STOP比例0.49%左右（E60），不足以支持“大量提前STOP”。
低编辑数与Joint弱化可同时出现，但上面逐Query证明二者不是单一对应。

每个Validation候选有相同612个Canonical Span，来自同一前端、同一128 Caption；
没有因为攻击失败改变样本集或属性提取器。平均有效属性数4.78125，K预算与使用次数分别统计，
不存在把6156 Query误当Gallery或换一套分辨率/Tokenizer的情况。

![训练策略行为](figures/training_policy.png)
![验证行为分布](figures/validation_behavior.png)

### 6.2 属性与替换字符分布

以下标签直接来自真实semantic_axis/region，不是按词义重新猜类别。占比分母为实际编辑次数。

{attrs}

E11性别标签约44.17%，E60约23.17%；上下装颜色合计由约11.29%升至38.92%。
这是明确的选择分布变化，不足以证明颜色选择导致泛化失败；没有针对其他属性强制替换的反事实比较。

{charmap}

E60的a→а、e→е、o→о合计约90.01%，E11约89.81%，没有证据表明后期才突然集中到这三种字符。
构成发生变化（a/e上升、o下降），不等于总集中度持续增加。
按源码白名单，a/e只有一个合法同形字候选；二者约占E60编辑67.84%，这些位置的Replacement选择本身没有多候选分辨空间，
但属性/字符位置仍可选择，不能由此推断整个策略已确定。

所有实际非NOOP都在既有审计中满足合法Span、每属性/共享词最多一次、Unicode白名单及无新增截断；
本轮再次回放原字符/码位/Span到最终字符串并统计。
127条可攻击Validation的Tokenizer记录均保留；H20/H60实际编辑123/122条，均token改变，未新增截断。
Token长度总增加/编辑数：H9约{float(va[9]["token_delta_sum"])/float(va[9]["edits"]):.3f}，
H20约{float(va[20]["token_delta_sum"])/float(va[20]["edits"]):.3f}，
H60约{float(va[60]["token_delta_sum"])/float(va[60]["edits"]):.3f}。
没有“每次编辑造成的Token长度变化持续变小”的证据；
Token改变或变长也不能直接代表嵌入更远/攻击更强。本轮没有重新编码来验证这种因果。

![字符与属性](figures/replacement_attribute_distribution.png)

## 7. 为什么Text-only较弱，但Joint增量仍较大

H20在Clean图库的mAP下降只有{100-v[20]["Text-only mAP"]:.6f}pp，
在G60图库额外下降{v[20]["Joint_extra_drop_pp"]:.6f}pp。
同一个文本扰动处在不同图像评分结构中，排名效果可以不同；这不是必须让Text-only最强的训练目标。

直接的分数证据：H20 Text-only正负Margin中位数{vr["H20"]["Text_only_margin_median"]:.6f}，
Joint Margin中位数{vr["H20"]["Joint_margin_median"]:.6f}。
G60 Image-only原本正确的51条Query，其Margin中位数{f["image_only_correct_margin_median"]:.6f}，
其中3条低于0.01。说明图库攻击后存在较脆弱的排序边界，但不能概括成全部正确Query都只差极小分数。
语义看似类似的字符替换可能改变正负样本相对分数，图像已经受攻击时更容易造成额外排名变化；
这是受数据支持的解释，不证明H显式学会了“互补”或某种唯一机制。

H9 Text-only mAP95.9375、H20 99.283859，独立文本效果差{v[20]["Text-only mAP"]-v[9]["Text-only mAP"]:.6f}pp，
Joint却近似持平。这直接否定“Text-only越强则Joint必然越强”的排序等价假设。
但不能据此把128小图库的Text-only数值与正式Full的H7数值相减作为提升/退化。

## 8. Reward与排名差距：同一个Validation上也能观察到

下表只使用已有Image-only/Joint矩阵按同一R_sim+R_rank数学定义派生诊断，
不送回训练或H策略。分母128；负池128而非Train2048，不能与在线Train均值直接作泛化误差相减。
paired分数、Margin差均来自FP32；汇总按Python/NumPy浮点执行，微小规约差异见方法说明。

{rewardval}

H40的派生R_sim、R_rank和R_total都高于H20，但Joint mAP反而高约0.796pp。
H20有4条“正奖励但AP升高”的Query及13条“负奖励但AP下降”的Query；
H60对应7/16条。这些是**代理奖励并不等价AP**的直接数据证据，
而不只是“Train与Validation样本不同”的笼统解释。

E20→E60 Train Sampled奖励提高，但Joint mAP上升{v[60]["Joint mAP"]-v[20]["Joint mAP"]:.6f}pp。
仅五个新节点上，在线Reward与Joint mAP的Pearson为{f["train_reward_Joint_mAP_Pearson_n5"]:.3f}
（mAP越高攻击越弱），Spearman约0.800；这是n=5、同轨迹、事后描述，不能据此推断因果或泛化规律。

训练Greedy synergy均值E20为−0.022062、E60为−0.016649，变得较不负但Joint更弱；
诊断量本身不应被当作优化目标或成功判据。若未来要引入交互奖励，需独立授权和控制，见research_next_steps.md。

![训练奖励与Validation](figures/reward_validation_alignment.png)

## 9. 对“为何H20以后没有继续改善”的证据分级

{table(["假设","直接证据/机制","评估"],[[x[0],x[2],x[3]] for x in hypotheses])}

**事实**：训练奖励/动作/梯度和逐Query变化如上述；H20不是后续每一轮的观测最低点，只是五个预设Validation节点中的最低。
**合理解释**：奖励代理差距、输出趋同与小集合中脆弱排序共同限制了继续延长训练的可见收益。
**尚未证实**：哪一个是主因、G60是否导致过拟合、字符空间是否到上限，以及新交互奖励能否改善。
因此当前最准确的回答不是“已证明过拟合”，而是“更长适配训练没有在这条轨迹的预设内部Validation点持续改善Joint”。

完整假设机制、支持/反对及新实验需求见[hypothesis_assessment.md](hypothesis_assessment.md)。

## 10. G与H的联系，以及尚未实现的交互

G-v2是ImageOnlyGenerator：推理只有G(I)，没有FiLM或文本属性输入。
训练由Caption提取属性、Dynamic Top-K/JVP方向与整句z_t提供监督；
单样本逐属性K次Adam更新，下一轮输入上一轮detach图，最后从原图再做一次联合更新。
Global最小化图像特征与z_t的正dot，Local使用当前属性JVP，Retain惩罚上一属性分数回升；
最后Joint步用聚合v_local。这里的G训练“Joint步”是原图联合监督，不是G/H端到端联合训练。

H通过冻结G的Adv Gallery评分学习Caption-only字符策略，因此已有图像攻击感知的条件奖励。
当前联系为：
Caption属性→G训练监督；冻结G60图像特征→H训练奖励；最终固定G(I)和H(T)在检索输入层组合。
没有H→G的参数训练反馈，没有共享可微图文攻击生成图，没有两生成器共同优化的交互目标。
所以“有条件引导/检索层联合收益”成立，“端到端双模态联合训练”不成立。
本轮没有新核对MSI-Attack论文定义，不将当前机制与其等同。

G10固定图库时H7为历史最低67.493835%；相同H1–10换成G60图库时H9最低40.425480%。
排序确实变化，但G条件也改变了检索问题的基线和难度；
不能拿这两个绝对mAP差直接定义H适配收益或证明某模块因果。

## 11. 历史三Victim Full仅作背景

下表H始终为历史H7，不是本次H60。固定3074 Gallery/6156 Query、Raw Clean及各Victim原生评分。

"""
fullrows=list(csv.DictReader((O/"historical_full_context.csv").open()))
flookup={(x["epoch"],x["target"],x["method"]):x for x in fullrows}
report+=table(["Victim","G Epoch","Image-only mAP","H7 Joint mAP","H7额外下降 pp"],
  [[model,e,num(float(flookup[(str(e),model,"G-v2 Image-only")]["mAP"]),4),
    num(float(flookup[(str(e),model,"G-v2 + H-v1 Joint")]["mAP"]),4),
    num(float(flookup[(str(e),model,"G-v2 Image-only")]["mAP"])-float(flookup[(str(e),model,"G-v2 + H-v1 Joint")]["mAP"]),4)]
   for model in ["IRRA","RaSa","APTM"] for e in [10,60]])
report+="""
G增加训练轮数后的三模型Image-only/固定H7 Joint总体更强，是此前正式Full的结果；
它不能证明新H60也会在三模型Full更强，本轮没有这些新结果。
Full的Clean/H7 Text-only、全部R@1/5/10/mAP及原始来源在historical_full_context.csv中保留。
IRRA为同来源模型，RaSa/APTM为冻结IRRA训练产物的迁移；
128内部Validation与Full绝对mAP不可直接比较，Raw Clean也不能替换成其他Baseline的Matched Clean。

## 12. 下一阶段决策

**优先一个实验：A，固定G10奖励图库，按相同H10父状态、训练长度、数据顺序和预算续训至60的匹配控制。**
理由是当前同时延长训练并切换图像奖励条件，尚不能区分两者。
不要先修改Loss再解释这条轨迹；新的奖励/交互设计应建立在控制与独立确认之上。
E（新PID确认及多seed）是可靠报告小差异的配套需求。
B/C/D可作为后续独立研究，不能承诺提升，也不能把常见代理奖励替换直接当作已成立创新。

[具体方案、资源、创新定位与失败风险](research_next_steps.md)。

## 13. 资源、缺失项与可复现边界

"""
report+=f"""
训练会话attempt_elapsed_seconds={s["training_completed"]["attempt_elapsed_seconds"]:.6f}，
训练目录计时elapsed_seconds={s["training_completed"]["elapsed_seconds"]:.6f}（约{s["training_completed"]["elapsed_seconds"]/3600:.3f}小时）。
逐Step记录合计{f["total_recorded_step_seconds"]:.3f}秒；Checkpoint、Validation及部分日志I/O不计入Step计时，
不能把两者差额全部归因于某一个因素。
GPU allocated/reserved峰值约761.311/824MiB；训练进程Peak RSS约{s["training_completed"]["runtime"]["peak_RSS_bytes"]/2**20:.1f}MiB。
共享GPU，未终止其他进程，不把时间当独占吞吐。

训练IRRA加载1次，文本调用102465次、样本306165条；
其中SCST合法更新需要101850×3=305550条，剩余615条对应五次Validation改变的Caption编码。
本轮动作统计的五候选非NOOP总数同为615，来源计数一致。
训练期间G/图像新前向0，缓存准备原2048图属于之前独立阶段，不重新计算。

明确缺失/限制：

- H1–H10历史Text-only完整分数矩阵未保存，只有审计通过的汇总、文本特征与动作；本轮不从汇总mAP推导逐Query Text-only。
- 历史G60+H9 Joint完整矩阵可用，所需核心逐Query对比不缺失。
- 没有动作概率/策略熵，也没有针对候选空间的穷举最优性证明。
- 单训练轨迹、单seed、重复128 Validation，没有新的独立确认或G10延长训练控制。
- 新H20–60没有官方Test或RaSa/APTM结果；历史Full全是固定H7，不混入当前候选选择。
- 人类语义可读性没有人工实验；合法同形字不自动等于语义/视觉无损。
- 原报告失败目录原样保留；本轮新目录与之不同，没有修复或覆盖hashlib.file_digest原件。

新增统计CPU单元测试10项PASS，覆盖多正例AP、原生tie排序、PID Bootstrap、动作记录与来源篡改拒绝。
附加Full标量读取初次遇到“长方法名/短模式名”字段差异而停止，随后仅修独立分析读取映射；
原实验无错误、未重跑模型，初次分析日志保留。最终读取与全部原始Full摘要原精度一致。
这是本轮分析器格式兼容，不是训练/评测重试或修改结果。

所有CPU分析代码位于src/tbps/ablations/hv1_g60_posthoc*.py及plot_hv1_g60_posthoc.py，
源码SHA、输入清单与运行环境见SOURCE_EVIDENCE.json和INPUT_MANIFEST.json。
完整输入5.094GB只读核验不等于全部151378历史文件重新算SHA；后者是元数据保护循环，二者明确区分。

## 附表A：全部50轮在线统计

{alltrain}

## 附表B：输出索引

| 文件 | 内容 |
| --- | --- |
| training_epoch_statistics.csv | 新50轮全部数值与分布摘要，原始精度 |
| historical_training_epoch_statistics.csv | H1–10同口径流式解析 |
| training_metric_distributions.csv | 全60轮各指标分布分位数、正负零计数 |
| training_resource_epoch_statistics.csv | 新50轮逐Step时间与GPU峰值 |
| training_phase_statistics.csv | 每10轮描述性聚合 |
| validation_comparison.csv | 固定G60下全部15候选主表原精度 |
| validation_metrics_raw.csv | Text-only/Joint四指标原精度 |
| per_query_analysis.csv | 2688条逐Query AP/排名/相似度/Margin与派生诊断 |
| per_query_comparisons.csv | 5对候选逐Query变化 |
| validation_reward_diagnostics.csv | 同Validation代理奖励与AP不一致性统计 |
| action_distribution.csv | 类型、字符、预算/编辑数与Token变化直方图 |
| action_epoch_statistics.csv | 训练Sampled/Greedy及Validation的STOP/NOOP/编辑统计 |
| checkpoint_inventory.csv | 历史81+新增400版本元数据/SHA，不含权重 |
| historical_full_context.csv | 历史三模型Full背景及控制组，未重算 |
| hypothesis_assessment.md | A–G机制、事实、判断与待验证 |
| research_next_steps.md | A–E方案、成本、风险与优先级 |
| code_evidence.csv | 实际源码函数、行号、SHA与作用 |
| figures/ | 12张PNG及12张SVG，真实节点无虚构插值 |

## 附录C：代表性Query文本与Unicode动作

以下按Rank-1翻转及绝对AP变化选取，只作描述，不据此选样或修改攻击策略。
完整动作包括原Span、码位、位置与Tokenizer记录保留在统计JSON的examples字段。
"""
for ex in s["examples"][:4]+[x for x in s["examples"] if x["pair"]=="H20_vs_H60"][:3]:
    i=ex["effect"]["query_index"];row=ex["effect"]
    report+=f"\n### {ex['pair']} / Query {i} / {ex['identity']['row_id']}\n\n"
    report+="原文："+json.dumps(ex["original"],ensure_ascii=False)+"\n\n"
    for key,action in ex["actions"].items():
        edited=action.get("adversarial",ex["original"])
        trace="；".join(e["semantic_axis"]+"/"+e["region"]+" "+e["original_codepoint"]+"→"+e["replacement_codepoint"]+" @"+str(e["position"]) for e in action.get("edits",[])) or "NOOP"
        report+=key+"："+json.dumps(edited,ensure_ascii=False)+"\n\n动作："+trace+"。\n\n"
    report+=f"AP {row['AP_a']:.6f}→{row['AP_b']:.6f}，最佳正例排名{row['best_rank_a']}→{row['best_rank_b']}；分数/Margin见逐Query CSV。\n"
report=report.replace("31? 评分数量以实际21组矩阵为准：","评分仅21组已存小矩阵：")
save("REPORT.md",report)

# 源码行号索引使用AST，只读源文件，不导入算法。
history=Path("/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes")
files={
 "attribute_text_policy_h_v1_20261015/policy.py":["natural_k","rollout","forward","attack"],
 "attribute_text_policy_h_v1_20261015/rewards.py":["evaluate_many"],
 "attribute_text_policy_h_v1_20261015/scst.py":["scst_loss"],
 "attribute_text_policy_h_v1_20261015/phase4_training.py":["action_stats","train_step"],
 "attribute_text_policy_h_v1_20261015/formal_training/runtime.py":["bind","evaluate_many"],
 "attribute_text_policy_h_v1_20261015/text_frontend.py":["convert"],
 "attribute_text_policy_h_v1_20261015/constraints.py":["make_edit","replay","candidates"],
 "attribute_image_generator_v2_20261014/serial_training.py":["train_single_sample","_update"],
 "attribute_image_generator_v2_20261014/losses.py":["attribute_losses","joint_losses","retain_loss"]}
code=[]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(2**20),b""):h.update(b)
    return h.hexdigest()
for path,names in files.items():
    p=history/path
    for n in ast.walk(ast.parse(p.read_text())):
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names:
            code.append(dict(path=str(p),function=n.name,line=n.lineno,end_line=n.end_lineno,sha256=sha(p)))
with (O/"code_evidence.csv").open("x",newline="",encoding="utf-8") as fp:
    w=csv.DictWriter(fp,list(code[0]),lineterminator="\n");w.writeheader();w.writerows(code)
# 完整四指标原精度另有主数据，此处新增两位展示版便于WPS粘贴。
rows=list(csv.DictReader((O/"validation_comparison.csv").open()))
with (O/"validation_comparison_2dp.csv").open("x",newline="",encoding="utf-8") as fp:
    w=csv.DictWriter(fp,list(rows[0]),lineterminator="\n");w.writeheader()
    for row in rows:
        w.writerow({k:(format(float(val),".2f") if val and k not in ("H_epoch","training_guide") else val) for k,val in row.items()})
print("H60_POSTHOC_REPORT_PASS")
