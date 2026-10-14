"""v4.3 fixed semantic direction diagnostics and Chinese research summary."""
import json,itertools,statistics,sys,subprocess
from pathlib import Path
import torch
from semantic import ROOT,OUT,V42,digest
sys.path.insert(0,str(ROOT.parent/'attribute_unit_v3_query_20261002'))
from query_reader import cosine,distribution

def run():
    d=json.loads((OUT/'diagnostic.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text())
    prev=json.loads((V42/'outputs/D/diagnostic.json').read_text());by={r['row_id']:r for r in prev['results']}
    assert all(digest(Path(p))==h for p,h in protocol['preserved_baseline_sha256'].items())
    assert digest(ROOT/'diagnose_semantic.py')==protocol['source_sha256'] and digest(ROOT/'semantic.py')==protocol['semantic_source_sha256']
    rows=d['semantic_directions'];directions={(r['slot'],r['canonical']):torch.tensor(r['d_512'],dtype=torch.float64) for r in rows}
    attrs=[a for r in d['results'] for a in r['attributes']];assert len(attrs)==45
    used=sorted({(a['slot'],a['canonical']) for a in attrs})
    same_value=[]
    for l,r in itertools.combinations(used,2):
        if l[0]!=r[0] and l[1]==r[1]:same_value.append({'left':list(l),'right':list(r),'cosine':cosine(directions[l],directions[r])})
    same_slot=[]
    slots=list(dict.fromkeys(r['slot'] for r in rows))
    for slot in slots:
        keys=[k for k in directions if k[0]==slot]
        for l,r in itertools.combinations(keys,2):same_slot.append({'slot':slot,'left_value':l[1],'right_value':r[1],'cosine':cosine(directions[l],directions[r])})
    same_slot_stats={slot:distribution([p['cosine'] for p in same_slot if p['slot']==slot]) for slot in slots}
    critical=[];rankings=[];pred_regression=0
    for r in d['results']:
        old=by[r['row_id']];oldattrs={a['slot']:a for a in old['attributes']}
        for a in r['attributes']:
            assert a['S_subspace']==oldattrs[a['slot']]['S_projection'] and a['g_raw']==oldattrs[a['slot']]['g_raw'] and a['w']==oldattrs[a['slot']]['w']
            assert a['semantic_direction']==directions[(a['slot'],a['canonical'])].tolist()
        for read,oread in zip(r['all_slot_reads'],old['all_slot_reads']):
            assert read==oread;pred_regression+=1
        old_order=sorted(r['attributes'],key=lambda a:-a['S_subspace']);new_order=sorted(r['attributes'],key=lambda a:-a['S_value'])
        rankings.append({'row_id':r['row_id'],'subspace_order':[a['slot'] for a in old_order],'value_order':[a['slot'] for a in new_order],'changed':[a['slot'] for a in old_order]!=[a['slot'] for a in new_order]})
        if r['row_id'] in ('cuhk:99:1','cuhk:326:0','cuhk:3828:0'):
            left,right=('upper_clothing_color','lower_clothing_color') if r['row_id']=='cuhk:3828:0' else ('hair_length','upper_clothing_length')
            lookup={a['slot']:a for a in r['attributes']};pair=next(p for p in r['pairs'] if {p['left'],p['right']}=={left,right})
            record={'caption':r['row_id'],'left':left,'right':right,'semantic_cosine':pair['cos_semantic'],'old_subspace_cosine':pair['cos_subspace']}
            for key,slot in [('left',left),('right',right)]:
                a=lookup[slot];i=a['interventions'][0];record[key]={'canonical':a['canonical'],'r':a['r'],'S_value':a['S_value'],'S_subspace':a['S_subspace'],'ratio':a['value_subspace_ratio'],'F_change':i['F_change'],'prediction':i['first_order_prediction'],'prediction_error':i['relative_prediction_error'],'paired_cosine_before':i['paired_cosine_before'],'paired_cosine_after':i['paired_cosine_after']}
            critical.append(record)
    assert pred_regression==117
    signs={str(sign):sum(a['sign']==sign for a in attrs) for sign in (-1,0,1)}
    ratios=distribution([a['value_subspace_ratio'] for a in attrs]);readout=distribution([a['S_value'] for a in attrs])
    same_caption=[{'caption':r['row_id'],**p} for r in d['results'] for p in r['pairs'] if next(a['canonical'] for a in r['attributes'] if a['slot']==p['left'])==next(a['canonical'] for a in r['attributes'] if a['slot']==p['right'])]
    all_pairs=distribution([p['cos_semantic'] for r in d['results'] for p in r['pairs']]);shared=d['distributions']['shared_object']['value_semantic']
    sensitivity=[]
    for eps in d['epsilons']:
        items=[i for a in attrs for i in a['interventions'] if i['epsilon']==eps];valid=[i for i in items if not i['zero_readout']]
        sensitivity.append({**next(s for s in d['intervention_summary'] if s['epsilon']==eps),'n':len(items),'zero_readout_n':sum(i['zero_readout'] for i in items),
          'absolute_prediction_error_median':statistics.median(i['absolute_prediction_error'] for i in items),'absolute_prediction_error_max':max(i['absolute_prediction_error'] for i in items),
          'nonpositive_F_changes':[{'caption':r['row_id'],'slot':a['slot'],'S':a['S_value'],**i} for r in d['results'] for a in r['attributes'] for i in a['interventions'] if i['epsilon']==eps and i['F_change']<=0]})
    result={'direction_count_all_nonnull':len(rows),'unique_current_attribute_units':len(used),'same_value_different_slot_unique_current_pairs':same_value,'same_value_distribution':distribution([p['cosine'] for p in same_value]),
      'same_caption_same_value_distribution':distribution([p['cos_semantic'] for p in same_caption]),'same_slot_different_value':same_slot,'same_slot_distributions':same_slot_stats,
      'critical':critical,'ratios':ratios,'S_value_distribution':readout,'sign_counts':signs,'rankings':rankings,'all_caption_pair_distribution':all_pairs,'shared_object_distribution':shared,
      'semantic_norm_range':[min(r['d_norm'] for r in rows),max(r['d_norm'] for r in rows)],'classifier_and_reader_regression':117,'sensitivity':sensitivity}
    (OUT/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    f=lambda x:f'{x:.9g}'
    lines=['# Attribute Unit v4.3 Value-specific Semantic Direction 验证（2026-10-03）','','独立feasibility validation：只读取v4.2 D epoch20 checkpoint，不新增/训练模型，不修改正式Stage1–3/Top-K/generator，不commit/push。原9句45属性、34054图clean train gallery、block8 fp32、paired soft rank tau=.02及epsilon=.01/.05/.1均不变。','','## A. Semantic direction construction','','真实classifier为13个nn.Linear(64,C_s)，logits=W_s z_s+b_s。中心mu=所有class rows均值（包括null），c=W_s[v]-mu，q=c/||c||2，d512=B_s q。bias未参与方向；null仅参与中心，不作为本轮目标属性。采用原记录canonical value，即使classifier预测null也不删除。','','w=alpha/||alpha||2，graw=w^T G，r=<graw,d512>，S_value=abs(r)，delta=epsilon*sign(r)*w*d512^T。固定semantic direction不随caption/gradient改变，干预正负号单独记录；cosine一律比较未乘sign的固定512D方向，不比较64D坐标。','','c/q/d由fp32冻结权重计算；norm、dot和误差统计用double读取fp32向量，IRRA graph/intervention仍为fp32。没有再对d做额外归一化。','','中心化具有class-weight共同平移不变性；方向在同步basis/classifier坐标旋转下保持不变。零中心化direction显式报错，无fallback；若r=0，遵循sign(0)=0并记录零干预，不能声称此时仍有epsilon预算。',
      '',f"已计算{len(rows)}个ontology非null directions，当前45属性对应{len(used)}个唯一(slot,value)。d norm范围{result['semantic_norm_range'][0]:.12g}–{result['semantic_norm_range'][1]:.12g}；45个r符号计数={signs}。",'','## B. Direction diagnostics','','### Same value / different slot','','只对当前45属性覆盖的唯一(slot,value)配对，避免重复caption把固定direction重复计权。无预设cosine阈值。','','| left | right | cosine |','| --- | --- | ---: |']
    for p in same_value:lines.append(f"| {'='.join(p['left'])} | {'='.join(p['right'])} | {f(p['cosine'])} |")
    sv=result['same_value_distribution'];lines+=['',f"unique pair n={sv['n']}；mean/median={f(sv['mean'])}/{f(sv['median'])}，range={f(sv['min'])}–{f(sv['max'])}。同caption重复计数的分布另存comparison.json，不混用。",'','### Same slot / different value','','来自冻结classifier的全部非null canonical values，未新增caption；每个slot的均值/中位数/范围及全部pair如下。','','| slot | pair n | mean | median | min | max |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for slot,s in same_slot_stats.items():lines.append(f"| {slot} | {s['n']} | {f(s['mean'])} | {f(s['median'])} | {f(s['min'])} | {f(s['max'])} |")
    lines+=['','| slot | left value | right value | cosine |','| --- | --- | --- | ---: |']
    for p in same_slot:lines.append(f"| {p['slot']} | {p['left_value']} | {p['right_value']} | {f(p['cosine'])} |")
    lines+=['','## C. Critical cases','','99/326为hair_length=short及upper_clothing_length=short；3828为upper/lower color=black。326的原binding疑点保留。epsilon=.01。','','| caption | fixed semantic cosine | old D gradient-optimized cosine | S_value left/right | signed r left/right | ΔF left/right |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for k in critical:
        l,r=k['left'],k['right'];lines.append(f"| {k['caption']} | {f(k['semantic_cosine'])} | {f(k['old_subspace_cosine'])} | {f(l['S_value'])} / {f(r['S_value'])} | {f(l['r'])} / {f(r['r'])} | {f(l['F_change'])} / {f(r['F_change'])} |")
    lines+=['','相同(slot,value)在99/326使用逐值相同的固定d，cosine也相同；S、r及干预仍随caption改变。117 alpha及classifier读数、45 w/graw/S_subspace与D逐值回归，attention未修改。','','## D. Subspace vs value-specific sensitivity','','S_subspace=||B_s^T graw||2；ratio=S_value/S_subspace是长度投影比例，不是方差解释率或语义正确率。理论S_value<=S_subspace，浮点允许1e-6相对误差，无人工clamp。',
      '',f"45属性ratio mean/median={f(ratios['mean'])}/{f(ratios['median'])}，range={f(ratios['min'])}–{f(ratios['max'])}；明显越界n={sum(a['S_value']>a['S_subspace']*(1+1e-6) for a in attrs)}。",'','| caption | slot=value | S_subspace | signed r | S_value | ratio | ΔF epsilon=.01 |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in d['results']:
        for a in r['attributes']:lines.append(f"| {r['row_id']} | {a['slot']}={a['canonical']} | {f(a['S_subspace'])} | {f(a['r'])} | {f(a['S_value'])} | {f(a['value_subspace_ratio'])} | {f(a['interventions'][0]['F_change'])} |")
    lines+=['','每句排序（只记录变化，不评价质量）：','','| caption | D subspace order | value-specific order | changed |','| --- | --- | --- | --- |']
    for r in rankings:lines.append(f"| {r['row_id']} | {' > '.join(r['subspace_order'])} | {' > '.join(r['value_order'])} | {r['changed']} |")
    lines+=['','## E. Mathematical validity','','| epsilon | actual norm min/max | max relative budget error | first-order relative error median/max | absolute error median/max | central relative error median/max | positive ΔF |','| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for x in sensitivity:lines.append(f"| {x['epsilon']} | {x['norm_min']:.12g} / {x['norm_max']:.12g} | {f(x['budget_max_relative_error'])} | {f(x['prediction_median_relative_error'])} / {f(x['prediction_max_relative_error'])} | {f(x['absolute_prediction_error_median'])} / {f(x['absolute_prediction_error_max'])} | {f(x['central_median_relative_error'])} / {f(x['central_max_relative_error'])} | {x['positive_F_change']}/45 |")
    lines+=['','表中误差为比例，不是百分数。相对误差分母是epsilon*S_value；S较小时需结合绝对误差及central difference，不为结果调epsilon或过滤属性。全部原始positive/minus intervention、paired cosine before/after、effective G·delta保留服务器JSON。',
      '',f"zero-readout counts={[x['zero_readout_n'] for x in sensitivity]}；zero intervention exact，IRRA/reader/projection/classifier全部冻结且无参数梯度，reader/projection state逐值不变；special-token delta=0，baseline F/paired cosine不变。Gram最大元素误差={f(d['geometry']['max_orthogonality_abs_error'])}。",'','## F. Conclusion','','待完整结果核查后填写，不调整模型、公式、样本或epsilon。','','## 追溯与边界','',f"服务器独立目录：`{ROOT}`。所有源码/tests/JSON/logs/direction vectors只在服务器；Windows只同步本Markdown研究摘要。4项针对性测试通过，无训练/新checkpoint、无暂存/commit/push。旧模型/源文件/结果SHA保持。",'','| item | SHA256 |','| --- | --- |']
    for path in [ROOT/'semantic.py',ROOT/'test_semantic.py',ROOT/'diagnose_semantic.py',Path(__file__),OUT/'protocol.json',OUT/'diagnostic.json',OUT/'comparison.json']:
        lines.append(f'| {path.relative_to(ROOT)} | `{digest(path)}` |')
    lines+=['',f"D checkpoint SHA：`{digest(V42/'outputs/D_final.pt')}`；D old diagnostic SHA：`{digest(V42/'outputs/D/diagnostic.json')}`；完整旧产物hash见protocol.json。"]
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    compact={k:v for k,v in result.items() if k not in ('same_slot_different_value',)}
    for k in ['ratios','same_value_distribution','same_caption_same_value_distribution','S_value_distribution','all_caption_pair_distribution','shared_object_distribution']:compact[k]={n:v for n,v in compact[k].items() if n!='sorted_values'}
    compact['same_slot_distributions']={s:{n:v for n,v in x.items() if n!='sorted_values'} for s,x in same_slot_stats.items()}
    print(json.dumps(compact,ensure_ascii=False,indent=2))

if __name__=='__main__':run()
