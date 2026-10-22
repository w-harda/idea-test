"""由既有CPU统计生成属性审计和交替设计，不运行实验。"""
from pathlib import Path
import ast,csv,json
from tbps.ablations import gh_attribute_audit as a
D=a.OUT
def csvrows(n):return list(csv.DictReader((D/n).open()))
def f(x,n=6):return "—" if x is None else format(x,"."+str(n)+"f")
def table(head,rows):
 return "| "+" | ".join(head)+" |\n| "+" | ".join(["---"]*len(head))+" |\n"+"\n".join("| "+" | ".join(map(str,r))+" |" for r in rows)+"\n"
def main():
 s=json.loads((D/"statistics.json").read_text());u=json.loads((D/"supplementary_statistics.json").read_text())
 sources={
 a.OLD/"src/attributes/text_extractor.py":["extract_with_provenance"],
 a.LEGACY/"prototypes/attribute_unit_v55_interface_20261003/attribute_interface.py":["convert_record","text_candidates","instance_id"],
 a.LEGACY/"prototypes/attribute_importance_step51_20261003/importance.py":["baseline_gradient","projected_score","score_instances","attach"],
 a.LEGACY/"prototypes/attribute_tta_step74_dynamic_k_20261004/selection_audit.py":["selection"],
 a.LEGACY/"prototypes/attribute_tta_step74_dynamic_k_20261004/paired_prefix.py":["prepare_uncapped"],
 a.LEGACY/"prototypes/attribute_tta_step72_20261003/attribute_tta.py":["attribute_delta","TextTransport","read"],
 Path("/home/lzf/ldx/outputs/idea-TBPS-test1/stage03-attribute-unit-v2-20261002/code/src/attributes/irra_soft_rank_gradient.py"):["paired_soft_rank"],
 Path("/home/lzf/ldx/outputs/idea-TBPS-test1/stage03-attribute-unit-v2-20261002/code/src/attributes/irra_attribute_unit_prototype.py"):["objective"],
 a.G/"prep_text_frontend.py":["__call__"],a.G/"serial_training.py":["_update","train_single_sample"],
 a.G/"losses.py":["attribute_losses","retain_loss","joint_losses"],a.G/"inference_once.py":["generate_once"],
 a.G/"epoch_training.py":["epoch_order","train_epoch"],a.G/"image_only_generator.py":["forward"],
 a.H/"text_frontend.py":["convert"],a.H/"policy.py":["natural_k","forward","rollout"],
 a.H/"constraints.py":["candidates","replay","make_edit"],a.H/"contracts.py":["attribute_id","word_at"],
 a.H/"phase4_training.py":["train_step"],a.H/"rewards.py":["evaluate_many"],
 a.H/"scst.py":["scst_loss"],a.H/"formal_training/runtime.py":["__init__","bind","evaluate_many"]}
 refs=[]
 for p,names in sources.items():
  tree=ast.parse(p.read_text());h=a.sha(p)
  for n in ast.walk(tree):
   if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names:refs.append(dict(path=str(p),function=n.name,line=n.lineno,end_line=n.end_lineno,sha256=h))
 a.writec("code_evidence.csv",refs)
 def ref(fragment,name):
  x=next(x for x in refs if fragment in x["path"] and x["function"]==name)
  return x["path"]+":"+str(x["line"])+"–"+str(x["end_line"])
 core=[]
 for ep in a.EPOCHS:
  z=s["H"][f"Validation_H{ep}"];v=u["budget_decomposition_and_rank"][f"Validation_H{ep}"]
  core.append([f"H{ep}",f(z["H_K"]["mean"]),f(z["selected_count"]["mean"]),f(z["edit_count"]["mean"]),
    z["position_NOOP_sum"],z["STOP_unused_slots_sum"],f(z["G_selected_H_edited_jaccard"]["mean"]),
    str(z["G_H_top1_exact_any"]["count"])+"/127",f(v["spearman"]["mean"])])
 coretab=table(["Validation候选","平均K","平均选中","平均编辑","Position NOOP槽","STOP未用槽","G选择/H编辑Jaccard","Top1一致*","同集Spearman**"],core)
 stages=[]
 for scope,role in (("Train","train"),("Validation","validation")):
  for ep in a.EPOCHS:
   z=s["H"][f"{scope}_H{ep}"];g=s["G"][role]
   stages.append([scope,ep,z["samples"],f(g["candidates"]["mean"]),f(g["positive"]["mean"]),f(g["K"]["mean"]),
    f(z["candidate_count"]["mean"]),f(z["legal_count"]["mean"]),f(z["H_K"]["mean"]),f(z["edit_count"]["mean"])])
 ranktab=[];decomp=[]
 for scope in ("Train","Validation"):
  for ep in a.EPOCHS:
   v=u["budget_decomposition_and_rank"][f"{scope}_H{ep}"];top=v["rank_comparable_top1"]
   ranktab.append([scope,ep,v["ranking_samples"],v["spearman"]["n"],v["two_candidate_samples"],v["G_ties"],v["H_ties"],
    f(v["spearman"]["mean"]),str(top["count"])+"/"+str(top["denominator"])])
   decomp.append([scope,ep,v["decomposition_samples"]]+[f(v[k]["mean"]) for k in ("candidate_component_G_to_common","importance_component_common","candidate_component_common_to_H","total_G_minus_H")])
 rawtab=table(["范围","样本","G全部候选","G可用Span语义","H已解析","原候选全等样本","G-only键","H-only键","歧义键"],[
 [r,u["raw_coverage"][r]["samples"],u["raw_coverage"][r]["G_original_candidates"],u["raw_coverage"][r]["G_span_semantic_valid"],
 u["raw_coverage"][r]["H_resolved"],u["raw_coverage"][r]["complete_raw_candidate_match"],u["raw_coverage"][r]["G_only"],
 u["raw_coverage"][r]["H_only"],u["raw_coverage"][r]["ambiguous"]] for r in ("train","validation")])
 mappingtab=table(["H","原属性","重提取属性*","原语义/坐标找回","未找回","其中自身编辑","其中未编辑","UTF8起点移动"],[
 [ep,u["canonical_reextraction"][str(ep)]["original_attributes"],u["canonical_reextraction"][str(ep)]["reextracted_attributes"],
 u["canonical_reextraction"][str(ep)]["semantic_refound"],u["canonical_reextraction"][str(ep)]["not_refound"],
 u["canonical_reextraction"][str(ep)]["own_edit_not_refound"],u["canonical_reextraction"][str(ep)]["unedited_not_refound"],
 u["canonical_reextraction"][str(ep)]["UTF8_start_shifted_attributes"]] for ep in a.EPOCHS])
 mapping=f"""# 原文—Unicode攻击文本—G属性监督映射

本轮只处理既有原文/攻击文本。原128条Caption经冻结CaptionFrontend规则重提取，逐条与原资格属性exact；
另对既有H9/H20/H60共384条攻击文本重提取，总规则转换512次。
不执行Tokenizer、IRRA/G/H神经前向或新攻击。

{mappingtab}

*重提取可能新建/重绑定其他属性，因此其总数不等于原属性找回数。
逐原属性以semantic_axis、region、canonical_value、原字符起止坐标严格匹配；多候选标AMBIGUOUS。
本Validation重提取歧义均0，不能与Train原始G/H候选的1个歧义混为一谈。

## 字符坐标、字节与Tokenizer

全部实际替换为1个code point→1个code point，所以字符串字符数量、半开字符Span和后续字符坐标不变。
本白名单由ASCII变Greek/Cyrillic，UTF-8由1字节变2字节，每个编辑增加1字节；
H9/H20/H60整组增加278/243/238字节。必须重算新UTF-8偏移，不能沿用original_utf8_offset。
612个属性中，新字节起点移动472/396/374个。

现有EditRecord保存attribute_id、Canonical、region、原Span/raw、position、position_in_attribute、
original_utf8_offset、原/新字符、码位和Unicode名。Tokenizer前后IDs、长度、新截断等是旧已存审计，
本轮只读；字符Span可保持不代表BPE token坐标或EOT保持。

## 重新提取的实际风险

H9/H20/H60所有自身被编辑属性278/243/238项，都未被原词法规则按原语义/Span重新找回。
还有26/6/12项未被直接编辑的属性消失，提示衣物/对象词改变还可能影响颜色绑定。
这不证明人类语义或神经理解一定丢失；它证明直接重新提取再匹配旧属性不可靠。

## ID与统一Schema

G ID包含row_id、旧slot、linked_object等；H ID只哈希六个语义/Span字段，且不含Caption/PID。
建议新anchor_id绑定original_caption_sha256+六字段，同时保留原G/H ID列表和linked_object；
不要替换旧ID，不要把相同词不同位置强配。

proposed_attribute_schema.json是本轮草案，不是已接入训练的接口。
含original_caption、adversarial_caption及各自SHA、original_span、edited_span、原/新字节Span、
semantic_axis、region、canonical_value、attribute_id、character_edit、unicode_codepoint、
G_selection、H_selection、linearization_caption_sha256及来源。
G/H选择和H实际编辑分开；UNRESOLVED/AMBIGUOUS保留列表，不能伪造唯一映射。
PID/配对只在独立training_binding_ref，不能进入H公开策略输入。

## 旧JVP的含义

v_i(T)是在原Caption的block8激活H_T上，按原token定位w_i和语境方向e_i计算的线性化。
Unicode改变后，即使本属性未编辑，整句BPE/激活/reader定位都可能改变。
原z_T不能冒充z_Tadv；旧v_i(T)不能改签为v_i(Tadv)。维度都是512或77不能证明可复用。

可保留Q/basis和原Canonical作为固定语义锚点，新语境token映射、reader w/e、z和JVP要按新来源重算。
若仍保留旧JVP作原文语义参照，必须如此命名；该参照本身不构成H→G反馈。
现有prep_text_frontend.clean==old_z_t及原caption/selection门禁会拒绝直接改Caption；
未来只能建立独立新条件协议，不能关闭旧门禁。

例如原blue的字符Span保持，H输出bluе可通过编辑trace精确回放；
保存edited_raw=bluе并继承原Canonical blue锚点，不要求词法规则重新认出bluе。
可追踪性由既有动作证明，新语境JVP数值仍需未来授权验证。
"""
 (D/"original_adversarial_mapping_analysis.md").write_text(mapping)
 schema={"$schema":"https://json-schema.org/draft/2020-12/schema","title":"G/H原文锚点映射草案，尚未接入训练","type":"object",
 "required":["schema","original_caption","adversarial_caption","original_caption_sha256","adversarial_caption_sha256","attribute_id","original_span","edited_span","semantic_axis","region","canonical_value","character_edit","unicode_codepoint","G_selection","H_selection","sources"],
 "properties":{
  "schema":{"const":"gh-original-anchor-mapping-proposal-1"},"original_caption":{"type":"string"},"adversarial_caption":{"type":"string"},
  "original_caption_sha256":{"type":"string","pattern":"^[0-9a-f]{64}$"},"adversarial_caption_sha256":{"type":"string","pattern":"^[0-9a-f]{64}$"},
  "attribute_id":{"type":"string","description":"anchor_id哈希原Caption SHA及六字段，不冒充旧G/H ID"},
  "g_legacy_instance_ids":{"type":"array","items":{"type":"string"}},"h_legacy_attribute_id":{"type":["string","null"]},
  "semantic_axis":{"type":"string"},"region":{"type":"string"},"canonical_value":{"type":"string"},
  "original_span":{"$ref":"#/$defs/span"},"edited_span":{"$ref":"#/$defs/span"},
  "original_utf8_span":{"$ref":"#/$defs/bytes"},"edited_utf8_span":{"$ref":"#/$defs/bytes"},
  "linked_object_original_span":{"type":["object","null"]},"linked_object_edited_span":{"type":["object","null"]},
  "alignment_status":{"enum":["EXACT","G_ONLY","H_ONLY","AMBIGUOUS","UNRESOLVED"]},
  "character_edit":{"type":"array","items":{"type":"object","required":["position","original_char","replacement_char","original_codepoint","replacement_codepoint","original_utf8_offset","edited_utf8_offset"]}},
  "unicode_codepoint":{"type":"array","items":{"type":"object","required":["original","replacement"]}},
  "G_selection":{"type":"object","required":["selected","selection_rank","S_raw","K","condition_source","linearization_caption_sha256","JVP_source"],"description":"不可用值显式null并说明原因；不重标旧JVP"},
  "H_selection":{"type":"object","required":["selected","actual_edited","predicted_importance","K","position_NOOP","STOP","checkpoint_sha256"]},
  "sources":{"type":"array","items":{"type":"object","required":["path","sha256"]}},
  "tokenizer_audit_ref":{"type":"object"},"training_binding_ref":{"type":["object","null"],"description":"独立PID/配对训练权限，不属于H策略输入"},
  "dependency_identity":{"type":"object","description":"IRRA/Tokenizer/reader/basis/source/caption SHA、round；图库另绑定G SHA"}},
 "$defs":{"span":{"type":"object","required":["start","end","raw","unit"],"properties":{"start":{"type":"integer","minimum":0},"end":{"type":"integer","minimum":1},"raw":{"type":"string"},"unit":{"const":"unicode_codepoint_half_open"}}},
  "bytes":{"type":"object","required":["start","end","unit"],"properties":{"start":{"type":"integer","minimum":0},"end":{"type":"integer","minimum":1},"unit":{"const":"utf8_byte_half_open"}}}},
 "$comment":"只定义结构；精确回放、SHA、跨字段/权限/歧义规则仍需未来验证器，不宣称已实现。"}
 a.writej("proposed_attribute_schema.json",schema)
 design="""# G/H双向交替设计：仅方案，未执行

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
"""
 (D/"alternating_training_design.md").write_text(design)
 gK=table(["K","Train样本/2048","比例(%)"],[[k,s["G"]["train"]["K_counts"].get(str(k),0),f(100*s["G"]["train"]["K_counts"].get(str(k),0)/2048,4)] for k in range(8)])
 corrt=[]
 for ep in a.EPOCHS:
  for k in ("G_selected_H_edited_jaccard__AP_delta_vs_image_only_pp","unused_K__AP_delta_vs_image_only_pp","candidate_count__unused_K"):
   z=s["H"][f"Validation_H{ep}"]["correlations"][k];corrt.append([ep,k,z["n"],f(z["spearman"]),f(z["pearson"])])
 report=f"""# G-v2/H-v1属性选择、动态K预算与双模态交互机制审计

日期2026-10-22；分析基准23fd02daebdcd28bc2e77eb18de1e6fd04432518。
独立CPU只读分析与研究方案，未改变G/H、K、Reward、Selection或历史产物。

## 核心结论

{coretab}

*Top1分母127个两侧非空样本，可含不同候选集合；不能冒充完整排名一致性。
**Spearman只比较严格同候选集、共同合法子集，至少2属性、并列平均秩；Validation有效105条。
Jaccard分母127，排除两侧都空的1条，不伪造空集Jaccard=1。

| 核心问题 | 审计结论 |
| --- | --- |
| G/H平均选择数？ | G Train K2.939453、中位3；H9/20/60 Validation K2.171875/2.234375/2.179688，编辑2.171875/1.898438/1.859375 |
| K差异来源？ | 公式同为gamma3/beta1；同候选分数分布是主要代数差异，STOP/NOOP只解释K之后编辑减少 |
| 最重要属性重合？ | 全127非空样本13/10/10；严格同集105条仅7/4/4，平均Spearman为负 |
| 约2次编辑是Bug？ | 已有K/动作/回放账本成立，未发现代码计数错误；兼有自然K、位置NOOP和少量STOP |
| 是否修改NOOP耗槽？ | 可以受控研究，无收益证明；属于算法/动作语义变化，不能当纯修复 |
| 是否属性错位？ | 排序/集合明显不同，但H60重合更高而Joint更弱，不能据此判定错位有害 |
| 原文→H文本→G？ | trace映射可行；重提取会丢失属性，旧JVP不能充当新语境JVP |
| 如何双向交替？ | B建立H文本→G新条件与G图库→H奖励；C再研究统一Joint/交互目标 |
| 最小下一实验？ | 两轮A1对B匹配控制验证H→G反向通路；不立刻扩K或60轮重训 |

G2.94来自Train，H2.17来自Validation，不能直接相减。相同128 Validation的G K为2.8515625。
H Train全部2048的已解析候选均值5.0283203125；用户约5.04涉及只统计2037可更新样本的分母。

{table(["范围","H","n","G原候选","G正分候选","G K","H已解析","H合法*","H K","实际编辑"],stages)}

*H合法列为进入策略动作的初始支持集，Train11条不可训练样本保留visit、固定NOOP。
原G候选包含unresolved/不支持表示，H候选已经解析，不能把这两列直接当同类数量。
Validation无Optimizer更新，表中G K只描述旧监督条件，K+1不是Validation真实更新次数。

## 1. 来源与验证范围

2176条原G条件、2048/128 H资格记录。H9/H20/H60各完整2048 Train visit动作，共6144条；
每个候选冻结128 Validation动作，共384条。
Train是Epoch内逐次更新过程的greedy快照，不是用Epoch末权重重新对Train推理。
H9使用G10奖励图库，H20/H60来自H10后的G60适配，不是单因素时长曲线。

逐Query AP/Rank-1等读取基准23fd02d已核验的128×128分数派生CSV，本轮未重新排名/评分/编码。
原128 Caption规则重提取与资格属性exact；384条旧攻击文本仅规则重提取，无Tokenizer/模型执行。
113冻结producer、G7个prepare producer及实际输入SHA核对；151378历史文件和18617相关产物stat保持。
主解析{s["analysis_seconds"]:.3f}秒、Peak RSS {s["peak_RSS_bytes"]/1024**3:.3f}GiB；无Checkpoint/Tensor反序列化为模型。

正式H11–60训练和既有CPU审计仍PASS，旧发布器hashlib.file_digest FAIL原样保留。
H7 Selection未变，无Test选模、新Full、GPU或交替训练。
code_evidence.csv提供下列全部公式的实际源码绝对路径、函数、行号与完整SHA。

## 2. G的真实属性选择与训练链

### 2.1 Caption、Canonical、Span

{ref("attribute_interface.py","convert_record")}逐Stage01 provenance mention转换，
保留semantic_axis、region、canonical_value、raw_text、原字符Span、linked_object及unresolved原因。
不字符串搜索迁移Span，不合并同词不同位置。G ID含row_id/旧slot/linked_object，
H ID只哈希六字段，不同ID不能用来判断属性不同。

必须区分：原候选→有表示/原Span/semantic basis rank>0可评分候选→正S_raw候选→Top-K。
本Train可评分10023项，均为正值；H没有同样的reader/basis资格要求。
G每样本K属性轮+1原图联合更新，不等于对K个词或K个互不重叠图像区域各攻击一次。
G推理仅G(I)，一次generate_once与原投影，不读文本/属性/JVP，也没有推理时FiLM。

### 2.2 S_raw数学及来源权限

令原Caption block8状态H_T，最终归一化编码f(H_T)，图库特征v_j，配对索引p：

~~~text
s_j = <f(H_T), v_j>
F = 1 + sum_(j != p) sigmoid((s_j-s_p)/0.02)
D_T = dF/dH_T
g_i = sum_l w_i,l * D_T,l
S_raw,i = || Q_i^T g_i ||_2
~~~

D_T是IRRA隐藏状态检索梯度，非生成器G。w_i由冻结reader/区域pooling给出，单位L2；
Q_i为冻结正交语义子空间basis。见{ref("importance.py","baseline_gradient")}、
{ref("importance.py","projected_score")}及paired_soft_rank源码索引。

F针对指定paired图，其他同PID图仍列入竞争项；H奖励则使用全部同PID正例的最佳正例Margin。
历史S_raw用官方train clean参考图库，34054张且含内部Validation图；
不是H的2048 Train奖励bank，不声称参考图库与内部Validation图像完全隔离。
本轮保持历史协议，不以旧S_raw参考设置冒充新的Train-only联合奖励。

S_raw是连续子空间一阶敏感度，不是Unicode候选真实收益或类别概率。
basis维数/reader定位/Caption梯度尺度会影响它；原算法不做sqrt(rank)归一化。
本轮不改变排名，也不把高S_raw等同于更强Joint收益。

### 2.3 自然K与截断

{ref("selection_audit.py","selection")}按正S_raw降序，tie保留原输入顺序：

~~~text
p_i = S_i^3 / sum_j S_j^3
E = -sum_i p_i log p_i
N_eff = exp(E)
K_G = min(m, ceil(beta*N_eff)), beta=1
m=0 => K_G=0
~~~

没有执行cap4。payload.capped_k/truncated为旧capped对照元数据，实际selected_attributes使用natural_k前缀。
prepare_uncapped按连续chunk复用旧验证器，不删超过4的项，见{ref("paired_prefix.py","prepare_uncapped")}。
Train实际最大7；同公式合成9个均分候选允许K9，证明不是硬上限7。

{gK}

均值2.939453125、中位3；K>5为4/2048，K>4为65/2048。
sum K=6020，sum(K+1)=8068；6020是更新账本，非不同词/区域数。
Validation原K均值2.8515625，绝不执行训练Optimizer。

S_raw n10023，均值2.129364251、median0.144939333、p95 4.609741516、max2421.937997455；
入选n6020，均值3.169883346、median0.339174973、p95 6.612051997。
长尾明显，均值不能代表普通Caption尺度，原精度见g_attribute_scores.csv。

### 2.4 JVP、逐属性更新与最终Joint

~~~text
B_i = w_i outer unit(projected_context_i)
v_i(T) = d f_T(H_T + t*B_i)/dt |_(t=0)
alpha_i = S_i / sum_(selected) S_i
v_local = sum_i alpha_i * v_i(T)
z_T = normalized original whole-caption feature
~~~

S_raw的Q投影梯度方向与这里的projected_context方向不同，不能混写。
JVP映射原block8→后续block→ln_final/projection/EOT/L2；
不再单位归一化，聚合直接加权。见{ref("attribute_tta.py","attribute_delta")}、
{ref("attribute_tta.py","read")}。本轮只验已存SHA，不新算JVP。

各属性轮从上一图像detach继续、相对原图投影，立即Adam更新：

~~~text
L_global = <f_I(I_adv), z_T>
L_local,i = <f_I(I_adv), v_i>
L_retain,i = ReLU(<f_I(I_adv), v_(i-1)> - stopgrad(reference_(i-1)))
L_i = L_global + L_local,i + L_retain,i
~~~

第一轮无Retain，只保紧邻上一轮。K轮后重新输入原图，最后一次更新
L_joint=<f_I(I_adv),z_T>+<f_I(I_adv),v_local>，无Retain；正dot最小化，各权重1。
这里joint是G属性聚合步，不是G/H联合训练；无TV/扰动正则/FiLM。
见{ref("serial_training.py","train_single_sample")}、{ref("losses.py","attribute_losses")}。
最终推理见{ref("inference_once.py","generate_once")}，只有1次G。

## 3. H的真实属性、K与三个Head

### 3.1 重要性与K梯度

CaptionFrontend.convert只读Caption规则，不输入图/PID/S_raw。
Byte Transformer字符向量在属性Span内平均，再加axis/region embedding；
Attribute Head产生l_i=w_A^T h_i+b_A，predicted_importance=softplus(l_i)。
没有S_raw回归监督，数值不能叫G属性重要性的估计值。

{ref("policy.py","rollout")}先计算各属性合法Unicode/Tokenizer候选，
排除初始不可编辑属性，再在importance.detach()的Python float上计算同gamma3/beta1自然K。
K本身不接收Reward梯度；同Head的logits仍通过SCST动作log-prob更新，K可以随参数间接变化。
gamma=3是幂强化，不是改3个字符；beta=1是有效数系数，不是只选1项。

额外数学性质：属性softmax（含同Head的STOP）对logits共同平移不敏感，
softplus分数比和K却可能变化。CPU Fixture保持同样softmax概率，K可从2变3。
这说明预算参数化没有S_raw式绝对校准，不证明它导致当前退化。
K无硬上限2/4；实际H20 Train出现K6、H60 Validation出现K5。

### 3.2 动作顺序

1. Attribute Head在当前合法属性+STOP中采样/贪心；STOP始终合法，选择即退出。
2. 已选属性移出available，保存selected ID。
3. Position Head对该原Span合法字符与额外NOOP位置评分。
4. Position NOOP不改字，但消耗外层1槽、该属性不能重选；并不标记该词已编辑。
5. Replacement Head只在原字符白名单支持中选，每属性最多1字符、共享原词最多1次。
6. 根据原坐标回放全部编辑，保存原/新文本、Span、码位、UTF-8、决策/掩码与Tokenizer审计。

见{ref("constraints.py","candidates")}、{ref("constraints.py","replay")}。
无白名单候选、Tokenizer不变或产生新截断的候选先被mask；
原文本截断/超长/无合法编辑按既有资格固定处理，不补抽、不删Query。
编辑后共享词或Tokenizer支持可能变化；Train的mask移除计数是事件，不是等量K损失。
三组Validation的额外mask移除事件均0，实际未用槽由下面账本完整解释。

### 3.3 为什么约2次编辑

~~~text
K_H - edits = Position_NOOP次数 + (K_H - 非STOP属性选择次数)
            = Position_NOOP槽 + STOP剩余未用槽
~~~

H9 Validation：278槽→278选中→278编辑，NOOP/STOP未用均0。
H20：286→286→243，43个Position NOOP、STOP未用0。
H60：279→276→238，38个Position NOOP、1次STOP留下3槽。
H20未用率43/286=15.03%，H60 41/279=14.70%，账本与字符串回放均成立。
H9约2.17次主要由自然K限制；H20/H60还有约0.34/0.32槽未用。

Train各2048完整；H9/20/60预算4557/4600/4317，
编辑4550/3848/3803，Position NOOP0/729/501、STOP未用槽7/23/13。
因此不能直接判为Bug，也不能一律叫无效动作浪费：NOOP允许主动拒绝低收益编辑。
取消耗槽/强制用满K/加NOOP惩罚会改变策略语义，是新算法变量，不是修复。

80条Validation至少5合法候选，H9中0条只编辑0/1次，H20为17条、H60为13条。
支持研究预算利用，但现有数据不能证明强制多编辑会更强。

## 4. G/H严格匹配与排名一致性

跨样本先核对Caption SHA/原图身份/PID/role，属性以六个Span/语义字段严格匹配，
不用ID字符串或fuzzy。重叠/重复词不同Span不能合并，一对多标AMBIGUOUS。

{rawtab}

Train73个G原候选无可用严格Span/语义，Validation4个。
Train106个G-only键和1个歧义，Validation11个G-only键；H-only原候选键均0。
歧义cuhk:8643:0不强行对齐。原候选全集相等1956/2048、120/128；
进一步G正分候选与H已解析集合相等1776/2048、112/128。
attribute_alignment.csv以G正分候选作左侧，H_ONLY不等于原提取器新建了属性；
原提取层的覆盖必须看raw_candidate_coverage.csv。

{table(["范围","H","同集≥2样本","有效Spearman","其中2属性","G并列样本","H并列样本","平均Spearman","同集Top1一致"],ranktab)}

Train2条常量G排名无相关，故1734/1736有效；不伪造0。
原主表corr helper最低3点，strict_rank_comparison.csv补充至少2点并单列2属性数量；
本报告排名以补充完整口径为准。对候选集不同的样本不强行计算全排序相关。

直接结论：G敏感的连续语义子空间通常不是H最偏好的受约束字符属性。
H没有复现S_raw排名的监督，因此负相关不是Bug证据。
G Train选择garment_type上/下装3502/6020=58.17%，color上/下装1951/6020=32.41%；
H Validation gender/person编辑占比H9 110/278=39.57%、H20 77/243=31.69%、H60 56/238=23.53%。
这些是文本语义标签，不代表已测得的图像空间区域归因。

G选中而H未编辑平均为H9 2.109375、H20 2.164062、H60 1.984375项/Caption；
H编辑但G未选平均1.429688/1.210938/0.992188。
差异确实存在，但重合率不能等同互补效果，不能强制H追随S_raw就宣称改进。

## 5. K差异的可核验代数分解

令C为两侧严格共有的合法候选，在已有分数上计算：

~~~text
K_G(fullG)-K_H(fullH)
 = [K_G(fullG)-K_G(C)] + [K_G(C)-K_H(C)] + [K_H(C)-K_H(fullH)]
~~~

前后项是候选集影响，中项是同候选分数形状影响。
只做标量公式重算，绝非改变K后实际攻击；分解顺序固定，不叫唯一因果贡献。

{table(["范围","H","n*","G→共同候选","共同集分数形状","共同→H候选","G K−H K"],decomp)}

*仅有真实importance的2037 Train/127 Validation，不给11/1个固定不可训练样本补假分数。
Validation的G→共同候选项均值0，分数形状项约0.67–0.71槽，
H额外候选再抵消约0.016–0.047槽。
H合法候选4.78125反而多于G正分4.6484375，因此不能说主要因H候选少导致K少。
STOP/NOOP只解释K之后编辑减少，不解释K_G与K_H本身。

## 6. 原文到攻击文本与G监督

{mappingtab}

完整见[原文/攻击文本映射](original_adversarial_mapping_analysis.md)和[Schema草案](proposed_attribute_schema.json)。
字符坐标保持，UTF-8偏移变化；规则重提取会丢失大量原属性；
继承原Canonical锚点与trace才是可靠来源链。
原JVP可继续作为原语义参照，但不能冒充攻击语境JVP。若G只读旧参照，仍没有H→G反馈。

## 7. K更大或重合更多是否值得研究

{table(["H","变量对","n","Spearman","Pearson"],corrt)}

AP_delta=Joint AP−Image-only AP，负值代表文本带来更强额外攻击。
Jaccard与AP_delta的Spearman H9≈−0.0045、H20≈+0.1905、H60≈+0.1990，
没有“越重合越强”的稳定负关联。
H60 Jaccard0.259449高于H20 0.211099，Joint mAP却42.874081高于40.427635，攻击更弱。
单轨迹/128重复Validation且Caption难度/类别/K混杂，不能据此做因果判断。

H20→H60未用槽变化与AP变化Pearson≈0.202、Spearman≈0.164，较弱；
候选数量与未用槽Spearman H20≈0.026、H60≈−0.256，不支持属性越多必然越浪费。
没有真实K=3/4/5干预结果，绝不推算这些预算的mAP。

最小未来K设计：同固定H、G bank、Caption及候选白名单，预设自然K与K1/2/3/4；
实际支持不足则受合法候选数限制，保留STOP/NOOP和每词一次规则。
“NOOP不耗槽/强制用满”应单独作另一变量，不与K一起改。
先Train诊断/独立Validation，不能Test选K。本轮未运行。

## 8. 交替训练基础、缺口与最小方案

可复用原G单样本K+1训练、H SCST单步、Caption-only前端/动作、
Train权限/正负例、不可变存储/Checkpoint与独立Validation。
缺少H→G文本条件桥、新语境JVP身份、round级双模型/双Adam/RNG完整恢复和动态bank重绑定。

原H奖励：
~~~text
R_sim = sim(I_G[paired],T)-sim(I_G[paired],T_adv)
M_G(T) = max_samePID sim(I_G,T)-max_differentPID sim(I_G,T)
R_rank = M_G(T)-M_G(T_adv)
R_total = R_sim+R_rank
L_SCST = -detach(R_sampled-R_greedy) * sum log pi(sampled actions)
~~~
synergy_increment=Adv配对相似度下降−Clean配对下降，仅诊断，不进SCST。
见{ref("rewards.py","evaluate_many")}、{ref("scst.py","scst_loss")}。
目标不是完整多正例AP，也不显式优化相对单模态交互。
G训练读取固定原文监督，H读取固定G图库，是单向条件链，不是端到端联合优化。

[交替训练方案](alternating_training_design.md)区分A0/A1/B/C，
包含缓存失效、两轮最小原型、更新/编码/JVP预算、真实信息流验证及失败恢复。
B产生双向条件依赖，C进一步形成目标层协调；仅交替两个旧训练器不够。
未核对MSI定义，不作与MSI机制等价或创新成立的声称。

只允许下一项新实验时，优先A1对B两轮匹配控制验证H→G：
同G父状态、图像、原属性，原文/旧H文本应产生真实可追踪的新z/JVP和G梯度差，
NOOP作阴性控制；然后同H动作比较旧/新G bank奖励和H更新差。
不要求G变化在H未更新前直接改变Caption-only推理；那反而违反接口。
本轮仅设计，不启动，也不改K、原Reward或Selection。

## 9. 交付与局限

g_attribute_statistics.csv含G候选/K/实际训练更新；Validation实际更新为0，theoretical_Kplus1只是理论计数。
h_attribute_statistics.csv/k_budget_comparison.csv含6144 Train与384 Validation真实动作，
attribute_alignment.csv/strict_rank_comparison.csv/attribute_overlap_statistics.csv含严格比较和分母。
k_decomposition.csv是代数而非干预；attribute_span_mapping.csv/canonical_reextraction*.csv保留位置和重提取证据。
8张16:9 PNG/SVG及原精度CSV见FIGURES.md，所有源SHA/行号见code_evidence.csv和SOURCE_EVIDENCE.json。

没有测得G像素的属性空间因果归因、没有新语境JVP数值、没有K/NOOP变更后的攻击结果、
没有多seed或独立验证。不能把已有H60退化归因于单一属性错位、K不足或NOOP。
Train动作不是固定末态H重推；所有缺失都保持缺失，没有伪造概率或新mAP。

## 附录：关键函数路径、行号与SHA

{table(["路径","函数","行号","SHA-256"],[[x["path"],x["function"],str(x["line"])+"–"+str(x["end_line"]),x["sha256"]] for x in refs])}
"""
 (D/"REPORT.md").write_text(report)
 figs=sorted((D/"figures").glob("*.png"))
 (D/"FIGURES.md").write_text("# CPU派生图表：16:9 PNG/SVG\n\n"+"\n".join("- ["+p.stem+" PNG](figures/"+p.name+") / [SVG](figures/"+p.stem+".svg)" for p in figs)+"\n")
 print("GH_ATTRIBUTE_REPORT_PASS",len(report))
if __name__=="__main__":main()
