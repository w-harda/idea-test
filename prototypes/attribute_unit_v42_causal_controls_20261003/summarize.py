"""Aggregate fixed v4.2 controls; no source or raw output copied to Windows."""
import json,statistics,sys
from pathlib import Path
from controls import ROOT,OUT,V41,V4,digest
sys.path.insert(0,str(ROOT.parent/'attribute_unit_v3_query_20261002'))
from query_reader import distribution
KEY=[('upper_clothing_color','upper_clothing_type'),('upper_clothing_color','upper_clothing_length'),('upper_clothing_type','upper_clothing_length'),('lower_clothing_color','lower_clothing_type')]

def run():
    training=json.loads((OUT/'training.json').read_text());A=json.loads((V41/'outputs/training.json').read_text());V=json.loads((V4/'outputs/training.json').read_text())
    ts={'v4':V,'A':A,**training['conditions']}
    ds={'A':json.loads((V41/'outputs/diagnostic.json').read_text()),**{m:json.loads((OUT/m/'diagnostic.json').read_text()) for m in ('B','C','D')}}
    assert all(digest(Path(p))==h for p,h in training['protocol']['preserved_baseline_sha256'].items())
    assert all(d['caption_count']==9 and d['attribute_count']==45 and d['gallery_count']==34054 and d['tau']==.02 and d['epsilons']==[.01,.05,.1] and d['rank']==64 for d in ds.values())
    assert all(d['attention_alpha_exact_regression_count']==117 and d['v4_weight_raw_gradient_exact_regression_count']==45 for d in ds.values())
    pairs={m:{(r['row_id'],p['left'],p['right']):p for r in d['results'] for p in r['pairs']} for m,d in ds.items()}
    assert all(set(p)==set(pairs['A']) for p in pairs.values());assert len(pairs['A'])==92
    shared=[k for k,p in pairs['A'].items() if p['shared_object_indices']];assert len(shared)==18
    shared_dist={m:distribution([pairs['A'][k][field] for k in shared]) for m,field in [('Evidence','cos_e'),('v4','cos_learned')]}
    shared_dist.update({m:distribution([pairs[m][k]['cos_projection'] for k in shared]) for m in ds})
    geometry={}
    for m in ds:
        initial={(p['left'],p['right']):p['O'] for p in ts[m]['initial_geometry']['overlap']};final={(p['left'],p['right']):p['O'] for p in ts[m]['final_geometry']['overlap']}
        geometry[m]=[{'left':l,'right':r,'initial':initial.get((l,r),initial.get((r,l))),'final':final.get((l,r),final.get((r,l)))} for l,r in KEY]
    critical=[]
    for rid,left,right in [('cuhk:99:1','hair_length','upper_clothing_length'),('cuhk:326:0','hair_length','upper_clothing_length'),('cuhk:3828:0','upper_clothing_color','lower_clothing_color')]:
        for m,d in ds.items():
            r=next(x for x in d['results'] if x['row_id']==rid);lookup={a['slot']:a for a in r['attributes']};a,b=lookup[left],lookup[right]
            pair=next(p for p in r['pairs'] if {p['left'],p['right']}=={left,right})
            critical.append({'caption':rid,'condition':m,'left':left,'right':right,'cosine':pair['cos_projection'],'S_left':a['S_projection'],'S_right':b['S_projection'],'delta_F_left':a['interventions'][0]['F_change'],'delta_F_right':b['interventions'][0]['F_change'],'prediction_left':a['prediction_projection'],'prediction_right':b['prediction_projection']})
    classification={m:ts[m]['final_validation']['slots'] for m in ts}
    compact={'classification':classification,'geometry':geometry,'D_projector_distance':ts['D']['projector_distance_from_B0'],'shared_object':shared_dist,'critical':critical,'sensitivity':{m:ds[m]['intervention_summary'] for m in ds},'orthogonality':{m:ds[m]['geometry']['max_orthogonality_abs_error'] for m in ds},'baseline_hashes_preserved':True}
    (OUT/'comparison.json').write_text(json.dumps(compact,ensure_ascii=False,indent=2))
    lines=['# Attribute Unit v4.2 Low-rank Projection 因果对照（2026-10-03）','','独立prototype；本轮只区分独立随机子空间几何和Stage01 pseudo-label监督训练的作用，不接入正式Stage03。','','## 固定协议与实施','','A：已有v4.1 learned independent，直接复用checkpoint、classification及诊断，未重新训练/干预。B：精确复现A seed42初始化，冻结独立basis，只训练classifier。C：所有槽共用B第一槽的随机basis并冻结，只训练各槽classifier。D：从C同一B0开始，各槽basis与classifier独立训练。','','B/C/D classifier初始state逐值相同，20轮minibatch顺序逐值相同；B初始geometry逐值复现A；C/D初始basis逐值一致。C forward仅返回一个共同basis的expand view，未引入slot-specific几何。B/C frozen raw basis与实际B逐值保持初始值。','','rank64、seed42、20轮Adam lr=.001、batch512、weight_decay0，最终epoch20 checkpoint固定；train54246句/8779PID、validation13700句/2195PID与v4/v4.1不变。原9句所属7PID及22个重复文本PID排除。仅普通sum_s CE，无image/retrieval supervision、provenance attention supervision或额外loss；不训练reader，无搜索/重复seed。','','直接只读复用既有v4.1 pooled z cache；frozen v4 alpha、原caption、block8 fp32和原Stage01 ontology/provenance不变。完整34054图clean train gallery；paired soft rank tau=.02，epsilon=.01/.05/.1；paired cosine仅诊断。','','## A. Classification','','衡量Stage01 pseudo-label reproduction。每槽validation N=13700；单元格为overall% / non-null%，未按诊断结果筛选属性。','','| slot | non-null N | v4 | A learned independent | B random independent | C random shared | D shared init learned |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    slots=[r['slot'] for r in classification['A']]
    for idx,slot in enumerate(slots):
        rr=[classification[m][idx] for m in ts];assert all(r['slot']==slot and r['sample_count']==13700 and r['non_null_count']==rr[0]['non_null_count'] for r in rr)
        lines.append('| '+slot+' | '+str(rr[0]['non_null_count'])+' | '+' | '.join(f"{r['accuracy']*100:.3f} / {r['non_null_accuracy']*100:.3f}" for r in rr)+' |')
    lines+=['','## B. Projection geometry','','O(s,t)=||B_s^T B_t||F²/64，以下为initial → final；投影方向在共同512D比较。','','| pair | A | B | C | D |','| --- | ---: | ---: | ---: | ---: |']
    for idx,(l,r) in enumerate(KEY):lines.append('| '+l+' / '+r+' | '+' | '.join(f"{geometry[m][idx]['initial']:.6f} → {geometry[m][idx]['final']:.6f}" for m in ds)+' |')
    lines+=['','D的projector distance Ds=||Bs Bs^T-B0 B0^T||F；避免basis旋转造成假分离。','','| slot | final Ds |','| --- | ---: |']
    for s,d in ts['D']['projector_distance_from_B0'].items():lines.append(f'| {s} | {d:.6f} |')
    lines+=['','D逐轮关键overlap：','','| epoch | upper color/type | upper color/length | upper type/length | lower color/type |','| ---: | ---: | ---: | ---: | ---: |']
    for epoch in ts['D']['history']:
        o={(p['left'],p['right']):p['O'] for p in epoch['orthogonality']['overlap']}
        lines.append('| '+str(epoch['epoch'])+' | '+' | '.join(f"{o.get((l,r),o.get((r,l))):.6f}" for l,r in KEY)+' |')
    lines+=['','各条件全部78对initial/final/每epoch overlap与13槽projector distance原始记录保留服务器training JSON；未仅选择关键pair。','','## C. Shared-object direction cosine','','原18对不变，方向 d512=B(gsub/||gsub||)；跨slot的64D坐标不用于cosine。','','| method | n | mean | median | min | max |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for m,s in shared_dist.items():lines.append(f"| {m} | {s['n']} | {s['mean']:.6f} | {s['median']:.6f} | {s['min']:.6f} | {s['max']:.6f} |")
    lines+=['','18对逐项对照：','','| caption | pair | Evidence | v4 | A | B | C | D |','| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for k in shared:lines.append('| '+k[0]+' | '+k[1]+' / '+k[2]+' | '+f"{pairs['A'][k]['cos_e']:.6f} | {pairs['A'][k]['cos_learned']:.6f}"+' | '+' | '.join(f"{pairs[m][k]['cos_projection']:.6f}" for m in ds)+' |')
    lines+=['','## D. Critical held-out cases','','99/326的left/right为hair_length/upper_clothing_length，3828为upper_clothing_color/lower_clothing_color。326的原binding存疑，保留作机制诊断。epsilon=.01。','','| caption | condition | cosine | S left / right | ΔF left / right | prediction left / right |','| --- | --- | ---: | ---: | ---: | --- |']
    for k in critical:lines.append(f"| {k['caption']} | {k['condition']} | {k['cosine']:.6f} | {k['S_left']:.9g} / {k['S_right']:.9g} | {k['delta_F_left']:.9g} / {k['delta_F_right']:.9g} | {k['prediction_left']} / {k['prediction_right']} |")
    lines+=['','每条件117个alpha、45个w/g_raw均逐值回归v4；attention定位未改变。shared evidence未修改，所有45属性均进入诊断，即使classifier预测null。','','## E. Sensitivity validity','','graw=w^T G，gsub=B^T graw，S=||gsub||2；d512=B(gsub/S)，delta=epsilon*w*d512^T，无额外归一化或人工方向。','','| condition | max abs Gram error |','| --- | ---: |']
    for m,d in ds.items():lines.append(f"| {m} | {d['geometry']['max_orthogonality_abs_error']:.9g} |")
    lines+=['','| condition | epsilon | norm min / max | max relative budget error | first-order relative error median / max | central relative error median / max | positive ΔF |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for m,d in ds.items():
        for s in d['intervention_summary']:lines.append(f"| {m} | {s['epsilon']} | {s['norm_min']:.12g} / {s['norm_max']:.12g} | {s['budget_max_relative_error']:.9g} | {s['prediction_median_relative_error']:.9g} / {s['prediction_max_relative_error']:.9g} | {s['central_median_relative_error']:.9g} / {s['central_max_relative_error']:.9g} | {s['positive_F_change']}/45 |")
    lines+=['','表中误差均为比例，不是百分数。IRRA/reader/projection/classifier在sensitivity阶段全部frozen、无参数梯度、state逐值不变；zero intervention exact，special-token delta为0。baseline F/paired cosine与v4逐值相同，全部paired cosine before/after在服务器JSON。A数学检查直接复用既有报告，B/C/D分别真实执行。','','## F. 因果结论','','待基于以上完整对照填写；不得据结果更改seed、样本、训练或协议。','','## 追溯与边界','',f'服务器独立目录：`{ROOT}`。3项针对性测试通过；正式tracked源码未改，旧产物全部hash不变，无暂存/commit/push。Windows仅同步本Markdown摘要。','','| item | SHA256 |','| --- | --- |']
    for path in [ROOT/'controls.py',ROOT/'test_controls.py',ROOT/'diagnose_controls.py',Path(__file__),OUT/'protocol.json',OUT/'training.json',OUT/'comparison.json',*[OUT/f'{m}_final.pt' for m in ('B','C','D')],*[OUT/m/'diagnostic.json' for m in ('B','C','D')]]:
        lines.append(f'| {path.relative_to(ROOT)} | `{digest(path)}` |')
    lines+=['',f"frozen pooled cache SHA：`{training['protocol']['pooled_cache_sha256']}`；既有A/v4/source/split/checkpoint/diagnostic的完整hash见服务器protocol.json。"]
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Compact readout; raw vectors/confusion matrices remain server-only.
    printed={**compact,'classification':{m:[{k:v for k,v in r.items() if k not in ('confusion','values')} for r in rr] for m,rr in classification.items()},'shared_object':{m:{k:v for k,v in d.items() if k!='sorted_values'} for m,d in shared_dist.items()}}
    print(json.dumps(printed,ensure_ascii=False,indent=2))

if __name__=='__main__':run()
