import json
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
def load(n):return json.loads((OUT/(n+'.json')).read_text())
def fmt(d):return ', '.join(f'{k}:{v}' for k,v in sorted(d.items())) or '无'
def stat(d):return f"{d.get('mentions',0)} / {d.get('distinct_captions',0)} / {d.get('distinct_pids',0)}"
def ratio(x):return '无候选' if x is None else f'{100*x:.3f}%'
def table(rows):return '\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)
def write(n,lines):(OUT/n).write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
inv=load('garment_inventory'); lc=load('length_classification'); lm=load('length_mentions'); ca=load('candidate_axes'); sl=load('short_long_conflict'); pro=load('protocol')
a=['# Stage03 v5.4：Fine-grained Ontology Axis Audit（2026-10-03，未提交）',
'''结论：trousers_shorts / skirt_dress 应在下一版prototype拆开表达，但父类、子类、构造修饰和未知信息须保留，不能把全部词视为互斥类别。原clothing_length混合袖长、衣身长、裤腿长、裙摆相对长度，并有hair证据错绑定。Upper sleeve、lower leg、skirt/dress相对hem extent有明确两侧数据；outer garment body extent基本够用但与type耦合。精确landmark、大部分full-body subtype缺对照。本轮是数据/ontology可行性，不证明subspace已选择性解耦，不冻结正式方法。''',
'## 1. 固定协议与统计单位',
'''固定v4 54246 train/13700 validation captions，8779/2195 PID，互不重叠。原9句及重复文本PID按原split留出。只读原文、Stage01 labels/provenance/linked_object、真实ontology/alias/extractor及既有H/Z/region cache。训练、IRRA forward、重提feature、retrieval/intervention为零；原13槽、Stage01/Stage03/Top-K/G未修改，无暂存/commit/push。

原始census按longest non-overlapping lexical nominal phrase，不按collapsed canonical统计。pants/trousers仅同义合并；shirt/t-shirt、jacket/coat、hoodie/sweater分开。dress shirt/pants/shoes保留compound；jean jacket/skirt的jean是material_modifier；one-piece是construction qualifier；spatial top不是garment。携带/否定scope风险单列。文本可包含叠穿、多人物或未被Stage01选中的服装；census不等于目标人物可靠监督。

候选axis按caption/region/axis一个value：upper/lower优先原primary type provenance；primary null允许明确raw regional head；原alias未覆盖fullbody可raw-only。弱binding/多value冲突保留unresolved，不修原标签/span。

读取rank之前固定覆盖门槛：每value至少50 train captions/10 train PID、10 validation captions/5 validation PID；至少2个合格value。只是工程覆盖条件，不是显著性/语义纯度结论，父子重叠/缺可靠负类可否决纳入。unresolved分母仅该axis已识别候选caption，没证据的不当负例；axis不明ambiguous池另列，因此0%不等于原数据100%可解析。''',
'## 2. 完整原始garment census与现行mapping',
'下表M/C/PID=mentions/distinct captions/distinct PID。head列排除material/location/construction；canonical mapping按原type provenance重叠mention计数。raw未映射实例仍保留。']
for region in sorted({d['region'] for d in inv.values()}):
 rows=[['lexical family','train全部M/C/PID','validation全部M/C/PID','train head M/C/PID','validation head M/C/PID','train canonical mapping','validation canonical mapping'],['---']*7]
 for v,d in sorted(inv.items()):
  if d['region']!=region:continue
  t=d.get('train',{});q=d.get('validation',{})
  rows.append([v,stat(t),stat(q),stat(t.get('head_only',{})),stat(q.get('head_only',{})),fmt(t.get('canonical_mapping',{})),fmt(q.get('canonical_mapping',{}))])
 a += [f'### {region}',table(rows)]
rows=[['lexical family','train raw forms','validation raw forms','train / val roles','train / val scope-risk mentions'],['---']*5]
for v,d in sorted(inv.items()):
 t=d.get('train',{});q=d.get('validation',{})
 rows.append([v,fmt(t.get('raw_forms',{})),fmt(q.get('raw_forms',{})),fmt(t.get('roles',{}))+' / '+fmt(q.get('roles',{})),str(t.get('scope_risk_mentions',0))+' / '+str(q.get('scope_risk_mentions',0))])
a += ['### Alias/角色/scope完整表',table(rows),
'''trousers_shorts覆盖pants/trousers、jeans、shorts、slacks/leggings/tights/capri/khakis；skirt_dress混lower skirt与fullbody dress。原alias漏掉gown/overalls等；jump suit的suit有upper jacket_coat映射7/3 mentions；sweat shirt有t_shirt_shirt映射35/10；dress另有2 train mentions映到trousers_shorts。均记录不修改。

普通pants不说明非jeans，普通shirt不说明非t-shirt，普通dress不说明非gown/one-piece。capri为腿cut，khakis兼具材质/风格/颜色义，one-piece为构造，不能做dress互斥类别。''',
'## 3. 所有原clothing-length provenance',
f"原upper/lower clothing_length共{lc['original_clothing_mentions']} mentions（train11372/validation2853），全部计入。原slot不变，依据text/evidence/object/邻近名词分类。",
'''显式sleeve/hair名词优先；lower裤装长短是leg extent，skirt/dress/gown是hem extent；直接outer extent或明确body/hem描述是body length。裸long/short shirt/sweater/top不强判袖/衣身。binding独立记录：sleeve evidence误绑pants不能进有效sleeve corpus。''']
rows=[['axis','train M/C/PID','validation M/C/PID','train value分布','validation value分布','valid binding train / val M','原source slot train / val'],['---']*7]
for ax,d in sorted(lc['by_axis'].items()):
 t=d['train'];q=d['validation'];rows.append([ax,stat(t),stat(q),fmt(t['values']),fmt(q['values']),str(t['valid_binding']['mentions'])+' / '+str(q['valid_binding']['mentions']),fmt(t['source_slots'])+' / '+fmt(q['source_slots'])])
a += [table(rows),'有效binding13234；其余991/14225=6.97%，含681 axis ambiguous、102 hair→clothing错绑定、208 sleeve binding风险。']
for ax in sorted(lc['by_axis']):
 rows=[['split','value','garment mention分布'],['---']*3]
 for part in ['train','validation']:
  rr=[x for x in lm if x['source_slot'] in ['upper_clothing_length','lower_clothing_length'] and x['axis']==ax and x['split']==part]
  for val in sorted({x['canonical_value'] for x in rr}):rows.append([part,val,fmt(Counter(x['garment'] for x in rr if x['canonical_value']==val))])
 a += [f'### {ax} garment/value分布',table(rows)]
a += ['## 4. short/long跨轴冲突','按(caption,evidence start,canonical)去重，同一个evidence跨slot只算一次；含未解决binding，不能当清洗监督样本。']
rows=[['canonical','split','mentions / unique anchors /重复','axis unique anchors','hair原文value aliases'],['---']*5]
for val,d in sl.items():
 for part,q in d.items():rows.append([val,part,f"{q['provenance_mentions']} / {q['unique_evidence_anchors']} / {q['duplicated_anchor_count']}",fmt(q['unique_anchor_axis_counts']),fmt(q['hair_raw_value_counts'])])
a += [table(rows),'''hem extent按region再拆：short下裙train421/val119、fullbody250/76；long下裙274/61、fullbody255/45。Canonical hair long含真实long3181/788、medium216/61、shoulder_length745/189。相对hair length medium和hair landmark不能继续单一long解释。short/long在多个axis复用，不应共享单一clothing_length语义定义。''',
'## 5. 每候选axis覆盖、rank与singular spectrum',
'''train cached原512D vectors按value求mean，value centroid等权中心化，float64 SVD，容差沿用v5 max(C.shape)*eps64*smax。全部observed values报告，不按spectrum择rank；validation不参与centroid。稀有value也可产生非零rank，因此rank不等于足够样本/选择性。

Upper/lower使用v5.3 region Z（原三个frozen v4 pooled slot Z均值），head用原hair Z；fullbody优先原object-uniform region Z，对新增noun仅从已有H同样object-uniform pooling，无新encoder features。细length轴未有专用reader，以下仅现有定位下数据条件。''',
'Cache pooling audit：`'+json.dumps(load('cache_pooling_audit'),ensure_ascii=False)+'`。']
rows=[['region:axis','value数','effective rank','达覆盖门槛values','train unresolved / candidate','validation unresolved / candidate'],['---']*6]
for key,d in sorted(ca.items()):
 t=d['unresolved']['train'];q=d['unresolved']['validation']
 rows.append([key,d['observed_value_count'],d['effective_rank'],', '.join(d['eligible_values']) or '无',f"{t['unresolved_captions']}/{t['candidate_captions']} ({ratio(t['unresolved_ratio'])})",f"{q['unresolved_captions']}/{q['candidate_captions']} ({ratio(q['unresolved_ratio'])})"])
a += [table(rows)]
for key,d in sorted(ca.items()):
 rows=[['value','train captions / PID','validation captions / PID','coverage ready'],['---']*4]
 for val,v in sorted(d['values'].items()):
  t=v['train'];q=v['validation'];rows.append([val,f"{t['distinct_captions']} / {t['distinct_pids']}",f"{q['distinct_captions']} / {q['distinct_pids']}",'是' if v['coverage_ready'] else '否'])
 spec=d['singular_values']; spec='无' if spec is None else ', '.join(f'{x:.9g}' for x in spec)
 a += [f'### {key}',table(rows),'完整singular spectrum：`'+spec+'`。']
a += ['## 6. 最小hierarchical ontology proposal（只在报告中）',
'''Region → Semantic Axis → Value，garment泛称保留partial parent label，显式子类保留细词；unknown不是负类。

- head → hair_length → short/medium/long（rank2）；shoulder_length保留hair_landmark观察，bald保留hair_state观察，均无可靠互斥多值对照。
- upper → garment_type → shirt_family（显式shirt/t-shirt/blouse/polo），outerwear（jacket/coat/blazer/overcoat），knit/sweat family（hoodie/sweater/sweatshirt/cardigan/pullover），vest/tank_top/tunic等明确名词。16 lexical values达门槛，全部22值rank21；父子label不强制互斥。
- upper → sleeve_length → short/long（4241/2471 train、1079/633 val，rank1）；upper → garment_body_length → short/long（50/353、15/73，rank1），仅明确outer/body证据，short边薄且jacket/coat type耦合，暂列谨慎支线。
- lower → garment_type → trousers parent（显式jeans/slacks子类）、shorts、skirt、leggings/tights等。9 lexical values全达门槛、rank8；capri/khakis作为cut/style/material qualifiers，不能与裤装子类强制互斥。
- lower → lower_garment_length → short/long（842/1281、232/302，rank1）；lower → hem_extent → short/long（421/274、119/61，rank1），主要skirt，和腿extent分轴。
- full_body → garment_type → dress_family（dress泛称，gown显式但可能ceremonial robe）、bib_overalls谨慎候选。原dress/gown/overalls3418/82/66 train、872/20/16 val，全5词rank4。dress-family与明确bib-overalls基本具备小规模数据条件，但overalls可bib-skirt、gown可叠穿/robe，不宣称成熟互斥类别；jumpsuit/romper不足。
- full_body → hem_extent → short/long（250/255、76/45，rank1）；fullbody sleeve long32/7不足，不纳入完整袖长对照。
- 各clothing region → color继续已有canonical color和可靠region绑定；本轮未重新审计颜色频次/构造新颜色方法。

Landmark轴每轴通常只有knee_length或shoulder_length一个value足够，其余ankle/calf/floor/thigh/hip/waist稀少，虽rank2–4也暂不纳入当前contrast。one_piece仅17/4 captions，单正类rank0；未提construction不当负类。suit/uniform/robe region未知、裸short/long上衣axis未知、跨对象/多对象冲突全部保留unresolved。''',
'## 7. 固定样本人工审读与限制',
'''固定样本在读rank之前按axis/binding和numeric row_id选取，不按几何删改。cuhk:96:0 short length jean pants被保守规则判cross-object/weak，是漏检；136:1 longgray shirt维持axis ambiguous；116:1 shortred jacket withlongsleeves可明确拆衣身/袖；538:1 clothing slot取shortblackhair是绑定错误；36:0/3910:0 sleeve错绑shorts/pants。

raw-only landmark仍有compound误报：504:1 blackdress coat可能是upper coat而非fullbodydress；gown可ceremonial robe/叠穿，5962:0 skirt withoveralls可bibskirt。规则不是人工gold，coverage-ready并不消除这些错误。不按rank结果更改规则/删除实例。完整固定原文/provenance见配套axis examples文档。''',
'## 8. 验证、产物与追溯',
'''19项针对性tests通过；14225原clothing-length provenance全计入；41项既有输入/checkpoint/cache/source SHA前后不变，prototype source/classification frozen不变，正式tracked/staged diff为空。无训练、重提IRRA/retrieval、正式ontology修改/commit/push。

独立服务器目录：`/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_unit_v54_axis_audit_20261003/`。源码axis_audit.py/test_axis_audit.py；outputs有garment_inventory/mentions、length_classification/mentions、short_long_conflict、candidate_frames/axes、centroid_rank（完整centroids/Q/spectrum）、fixed_audit_sample、protocol/source_audit/cache_pooling_audit。Windows只同步Markdown摘要/可读样本，不复制源码/原始JSON/cache。''',
'```bash\ncd '+str(ROOT)+'\nPYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 /home/lzf/ldx/envs/tbps-image/bin/python -m unittest -v test_axis_audit\nPYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 /home/lzf/ldx/envs/tbps-image/bin/python axis_audit.py\nPYTHONDONTWRITEBYTECODE=1 /home/lzf/ldx/envs/tbps-image/bin/python report.py\n```',
'来源审计：`'+json.dumps(load('source_audit'),ensure_ascii=False)+'`。',
'### Prototype source SHA256',table([['file','SHA256'],['---','---']]+list(pro['source_sha256'].items())),
'### 原41项来源SHA256',table([['输入服务器绝对路径','SHA256'],['---','---']]+list(pro['old_sha256'].items()))]
write('summary.md',a)
b=['# Stage03 v5.4 固定原文/绑定审计样本','按axis/binding和numeric row_id确定性选择，读取rank之前冻结；原文/标签/span不修改。以下为写入Markdown的小型可读记录，不同步原始JSON产物。']
for x in load('fixed_audit_sample'):
 v=x['record'];b += [f"## {v['row_id']} · {v['axis']} / {v['binding']} · {v['split']}",x['caption'],f"原slot `{v['source_slot']}={v['canonical_value']}`；candidate `{v['region']} / {v['axis']} / {v['value']}`；garment `{v['garment']}`；规则 `{v['reason']}`。",'原provenance：\n```json\n'+json.dumps(v['mention'],ensure_ascii=False,indent=2)+'\n```','原13-slot：\n```json\n'+json.dumps(x['original_attributes'],ensure_ascii=False,indent=2)+'\n```']
b += ['## 人工审读边界',a[-8] if False else '96:0保守false-unresolved；136:1裸longshirt轴未知；116:1衣身/袖分轴；538:1hair→clothing、36:0/3910:0sleeve→pants错绑。额外rawlandmark504:1dresscoat compound、5962:0bib-skirt overalls、gown/robe叠穿歧义仍需保留。没有按几何结果修标签或删样本。']
write('axis_audit_examples.md',b)
print('Markdown reports written',len(a),len(b))
