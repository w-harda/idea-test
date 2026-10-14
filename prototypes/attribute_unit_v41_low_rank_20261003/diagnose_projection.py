"""v4.1 diagnostic: only compare directions mapped to common IRRA 512D."""
import sys,json,statistics
from pathlib import Path
import torch
from projection import ROOT,OUT,V4,REPO,CODE,RAW,SlotReader,LowRankProjection,digest,setup,geometry,load_victim
sys.path.insert(0,str(REPO/'prototypes/attribute_unit_v3_query_20261002'))
from query_reader import AttributeUnitV2Probe,uniform_unit_gradient,mass_and_top,cosine,distribution

KEY_PAIRS=[('upper_clothing_color','upper_clothing_type'),('upper_clothing_color','upper_clothing_length'),('upper_clothing_type','upper_clothing_length'),('lower_clothing_color','lower_clothing_type')]

def summary(report,training,previous,split):
    f=lambda x:f'{x:.6g}'
    lines=['# Attribute Unit v4.1 Slot-aware Low-rank Projection（2026-10-03）','',
      '独立feasibility prototype；冻结IRRA和v4全部slot query，只训练每槽512×64 raw basis（经可微thin QR得到正交B）及新的64D Linear value/null classifier。r=64固定，不引入cross-slot orthogonality loss。',
      '', 'B=Q×sign(diag(R))，QR在每次forward中严格参数化正交列，不用penalty。训练为sum_s普通CE，z64=B_sᵀΣalpha_sj h_j；线性pool/projection可交换。训练仅使用原frozen文本状态和Stage01 canonical labels，不使用image/retrieval objective/gradient/attack/provenance/linked_object supervision。',
      '', f"直接只读复用v4 split和block8 fp32 cache；train {split['train_caption_count']}句/{split['train_pid_count']}PID，validation {split['validation_caption_count']}句/{split['validation_pid_count']}PID，seed42；原7个诊断PID及22个跨PID重复文本PID全部排除。Adam lr=.001、batch512、weight_decay=0，固定20 epochs和最终epoch20 checkpoint，未按9句结果调参/重训。",
      '', f"本轮raw basis参数{training['projection_parameters']}、classifier参数{training['classifier_parameters']}；v4 reader全部冻结。预计算z512仅缓存原alpha pooling，不改变attention。",
      '', '## Validation与v4并列（Stage01 pseudo-label reproduction）','',
      '| slot | overall v4 | overall v4.1 | non-null N | non-null v4 | non-null v4.1 |',
      '| --- | ---: | ---: | ---: | ---: | ---: |']
    old={r['slot']:r for r in previous['final_validation']['slots']}
    for r in training['final_validation']['slots']:
        a=old[r['slot']];assert r['sample_count']==a['sample_count']==13700 and r['non_null_count']==a['non_null_count']
        lines.append(f"| {r['slot']} | {a['accuracy']*100:.3f}% | {r['accuracy']*100:.3f}% | {r['non_null_count']} | {a['non_null_accuracy']*100:.3f}% | {r['non_null_accuracy']*100:.3f}% |")
    lines+=['','## 训练曲线与正交误差','', '| epoch | train CE | validation CE | max abs BᵀB-I |', '| ---: | ---: | ---: | ---: |']
    for e in training['history']:lines.append(f"| {e['epoch']} | {f(e['train_CE_sum'])} | {f(e['validation']['CE_sum'])} | {f(e['orthogonality']['max_orthogonality_abs_error'])} |")
    lines+=['','## 共同512D方向分布','', '| group | method | n | mean | median | min | max |', '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for group,methods in report['distributions'].items():
        for method,s in methods.items():lines.append(f"| {group} | {method} | {s['n']} | {f(s['mean'])} | {f(s['median'])} | {f(s['min'])} | {f(s['max'])} |")
    lines+=['','## 三个关键案例（方向比较均为映射回512D）','',
      '| caption | left/right | Evidence cosine | v4 cosine | v4.1 cosine | S left/right | ΔF at ε=.01 left/right |',
      '| --- | --- | ---: | ---: | ---: | --- | --- |']
    for r in report['results']:
        lookup={a['slot']:a for a in r['attributes']}
        for p in r['pairs']:
            if p['identical_evidence'] or (r['row_id']=='cuhk:99:1' and {p['left'],p['right']}=={'hair_length','upper_clothing_length'}):
                a=lookup[p['left']];b=lookup[p['right']]
                lines.append(f"| {r['row_id']} | {p['left']} / {p['right']} | {f(p['cos_e'])} | {f(p['cos_learned'])} | {f(p['cos_projection'])} | {f(a['S_projection'])} / {f(b['S_projection'])} | {f(a['interventions'][0]['F_change'])} / {f(b['interventions'][0]['F_change'])} |")
    lines+=['','117个slot attention alpha、原45属性w/read_pattern、g_raw与v4全部逐值回归；token localization完全保持原v4。326:0原binding存疑，只作机制诊断。',
      '', '## Subspace overlap O=||B_sᵀB_t||F²/64','',
      '以下初始化与训练后overlap仅作几何诊断，不新增retrieval对照方法。独立低秩初始化本身已产生不同子空间；cosine下降不能单独证明学到了语义维度，也不意味着跨slot被要求互相正交。',
      '', '| left | right | initialization O | trained O |', '| --- | --- | ---: | ---: |']
    initial={(p['left'],p['right']):p['O'] for p in training['initial_geometry']['overlap']}
    for p in report['geometry']['overlap']:
        lines.append(f"| {p['left']} | {p['right']} | {f(initial[(p['left'],p['right'])])} | {f(p['O'])} |")
    lines+=['',f"final max abs Gram error={report['geometry']['max_orthogonality_abs_error']:.9g}；max Frobenius Gram error={report['geometry']['max_orthogonality_F_error']:.9g}。", 
      '', '## 原45属性S及mapped direction规范','',
      'g_raw=wᵀG；g_sub=Bᵀg_raw；S=||g_sub||2；d512=B(g_sub/S)。delta=εw d512ᵀ；没有额外normalize或人工方向。不同槽64D坐标不用于cosine比较。',
      '', '| caption | slot=value | S_v4 | S_projection | ||d512|| | prediction_v4 | prediction_projection |', '| --- | --- | ---: | ---: | ---: | --- | --- |']
    for r in report['results']:
        for a in r['attributes']:
            lines.append(f"| {r['row_id']} | {a['slot']}={a['canonical']} | {f(a['S_v4'])} | {f(a['S_projection'])} | {f(a['mapped_direction_norm'])} | {a['prediction_v4']} | {a['prediction_projection']} |")
    lines+=['','## 等预算与一阶预测','', '| ε | actual norm min/max | budget max relative error | prediction median/max relative error | central median/max relative error | positive ΔF |',
      '| ---: | --- | ---: | --- | --- | ---: |']
    for s in report['intervention_summary']:
        lines.append(f"| {s['epsilon']} | {s['norm_min']:.12g} / {s['norm_max']:.12g} | {f(s['budget_max_relative_error'])} | {f(s['prediction_median_relative_error'])} / {f(s['prediction_max_relative_error'])} | {f(s['central_median_relative_error'])} / {f(s['central_max_relative_error'])} | {s['positive_F_change']}/45 |")
    lines+=['','误差为比例，不是百分数。epsilon=.01/.05/.1、34054图clean train gallery、paired soft rank τ=.02、原9句45属性及block8/fp32完全不变。paired cosine仅诊断，全部before/after在服务器JSON。',
      '', '## 原92对方向逐项对照','', '| caption | left/right | Evidence | v4 | v4.1 mapped512 | shared object | identical evidence |',
      '| --- | --- | ---: | ---: | ---: | --- | --- |']
    for r in report['results']:
        for p in r['pairs']:lines.append(f"| {r['row_id']} | {p['left']} / {p['right']} | {f(p['cos_e'])} | {f(p['cos_learned'])} | {f(p['cos_projection'])} | {p['shared_object_indices']} | {p['identical_evidence']} |")
    lines+=['','## 回归与追溯','',
      'zero intervention exact；IRRA/reader/projection/classifier在sensitivity阶段全部冻结且无参数梯度；reader与projection state逐值不变。原baseline F/paired cosine、117 alpha、45 w/g_raw回归通过；实际delta只进入content位置。无正式源码修改，无暂存/commit/push。',
      '', f"服务器目录：`{ROOT}`；原始JSON、checkpoint、cache、tests均只在服务器，Windows只同步Markdown研究摘要。",
      '', f"training SHA：`{report['training_sha256']}`；projection checkpoint：`{training['checkpoint_sha256']}`；v4 diagnostic：`{report['v4_sha256']}`；训练源码：`{training['protocol']['source_sha256']}`；诊断源码：`{digest(__file__)}`。"]
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def run():
    setup();target=OUT/'diagnostic.json';assert not target.exists()
    training=json.loads((OUT/'training.json').read_text());previous=json.loads((V4/'outputs/training.json').read_text());split=json.loads((V4/'outputs/split.json').read_text())
    assert training['training_completed_before_diagnostic'] and training['history'][-1]['epoch']==20
    assert digest(ROOT/'projection.py')==training['protocol']['source_sha256']
    assert digest(OUT/'projection_final.pt')==training['checkpoint_sha256']
    assert digest(V4/'outputs/reader_final.pt')==training['protocol']['v4_checkpoint_sha256']
    assert digest(V4/'outputs/split.json')==training['protocol']['split_sha256']
    state=torch.load(OUT/'projection_final.pt',map_location='cuda',weights_only=True)
    slots=state['slots'];values=state['values'];model=LowRankProjection([len(v) for v in values]).to('cuda')
    model.load_state_dict(state['state_dict']);model.eval().requires_grad_(False)
    with torch.no_grad():B=model.basis().detach()
    assert torch.equal(B,state['B']) and B.shape==(13,512,64)
    frozen_projection={k:v.clone() for k,v in model.state_dict().items()}
    rs=torch.load(V4/'outputs/reader_final.pt',map_location='cuda',weights_only=True)
    reader=SlotReader([len(v) for v in values]).to('cuda');reader.load_state_dict(rs['state_dict']);reader.eval().requires_grad_(False)
    frozen_reader={k:v.clone() for k,v in reader.state_dict().items()}
    v4path=V4/'outputs/diagnostic.json';assert digest(v4path)=='3cfc5f2bdc2bc40f7a9453a517bf06cd02f8ad16c69df6d50cdaf2f4c4c7629e'
    prior=json.loads(v4path.read_text());assert prior['epsilons']==[.01,.05,.1]
    full=json.loads(Path('/home/lzf/ldx/outputs/idea-TBPS-test1/stage03-attribute-unit-v2-20261002/prototype.json').read_text());fullby={r['row_id']:r for r in full['results']}
    cache=Path('/home/lzf/ldx/outputs/idea-TBPS-test1/stage03-v2-smoke-20261001/clean-train-gallery.pt')
    assert digest(cache)==full['gallery_cache_sha256']
    payload=torch.load(cache,map_location='cpu',weights_only=True);spec=payload['spec']
    assert spec==previous['protocol']['IRRA_spec'] and len(payload['features'])==34054 and str(torch.__version__)==spec['torch_version']
    assert digest('/home/lzf/Attack/IRRA/logs/CUHK-PEDES/best.pth')==spec['checkpoint_sha256']
    assert digest(CODE/'configs/irra_cuhk_official.yaml')==spec['config_sha256']
    for name,h in spec['source_sha256'].items():assert digest(Path('/home/lzf/Attack/IRRA-main/IRRA-main')/name)==h
    gallery=payload['features'].to('cuda');victim=load_victim();results=[];ge=geometry(B,slots)
    assert ge['max_orthogonality_abs_error']<2e-6
    for old in prior['results']:
        assert old['pid'] in split['held_out_pids'] and old['pid'] not in split['train_pids'] and old['pid'] not in split['validation_pids']
        probe=AttributeUnitV2Probe(victim,gallery,fullby[old['row_id']]['paired_gallery_index'],tau=.02)
        with torch.no_grad():original=victim.encode_texts([old['caption']])
        feature,H,z=probe.encode(old['caption'],8);assert torch.equal(original,feature.detach()) and not z.count_nonzero()
        with torch.no_grad():zf,_,az=probe.encode(old['caption'],8,torch.zeros_like(H))
        assert torch.equal(original,zf) and not az.count_nonzero()
        F0,c0=probe.objective(feature);G,=torch.autograd.grad(F0,H);baseline=float(F0);cosine0=float(c0)
        assert baseline==old['F'] and cosine0==old['paired_cosine']
        tokens=old['token_alignment']['content_tokens'];ids=[t['token_index'] for t in tokens]
        mask=torch.zeros(1,H.shape[0],device=H.device,dtype=torch.bool);mask[0,ids]=True
        with torch.no_grad():
            alpha,Z,_=reader(H.detach().permute(1,0,2),mask);_,logits=model(Z,B)
        alpha=alpha[0].detach();reads=[];rows=[]
        for s,slot in enumerate(slots):
            saved=old['all_slot_reads'][s];assert saved['slot']==slot
            assert torch.equal(alpha[s].cpu(),torch.tensor(saved['alpha'],dtype=torch.float32))
            assert torch.equal(Z[0,s].cpu(),torch.tensor(saved['z'],dtype=torch.float32))
            reads.append({'slot':slot,'alpha':alpha[s].cpu().tolist(),'read_pattern':saved['read_pattern'],
              'prediction_v4':saved['prediction'],'prediction_projection':values[s][int(logits[s].argmax())],
              'class_probabilities_projection':logits[s].softmax(-1)[0].cpu().tolist()})
        for a in old['attributes']:
            s=slots.index(a['slot']);ae=alpha[s];w=(ae/ae.norm()).detach();raw=(w[:,None]*G[:,0]).sum(0);Sraw=float(raw.double().norm())
            assert Sraw==a['S_learned'] and torch.equal(raw.cpu(),torch.tensor(a['g_learned'],dtype=torch.float32))
            assert torch.equal(w.cpu(),torch.tensor(a['w_learned'],dtype=torch.float32))
            assert mass_and_top(ae,a,tokens)==a['read_pattern']
            sub=B[s].T@raw;S=float(sub.double().norm());d=B[s]@(sub/S)
            assert 0<S<=Sraw*(1+1e-6) and d.numel()==512 and not d.requires_grad
            row={k:a[k] for k in ['slot','canonical','evidence_token_indices','object_token_indices','support_token_indices']}
            row.update(S_v4=Sraw,S_projection=S,g_raw=raw.cpu().tolist(),g_sub=sub.cpu().tolist(),mapped_direction=d.cpu().tolist(),
              mapped_direction_norm=float(d.double().norm()),w=w.cpu().tolist(),alpha=ae.cpu().tolist(),read_pattern=a['read_pattern'],
              prediction_v4=a['prediction'],prediction_projection=reads[s]['prediction_projection'],interventions=[])
            for eps in prior['epsilons']:
                delta=eps*w[:,None,None]*d[None,None,:]
                with torch.no_grad():
                    fp,_,actual=probe.encode(old['caption'],8,delta);fm,_,minus=probe.encode(old['caption'],8,-delta)
                    Fp,cp=probe.objective(fp);Fm,cm=probe.objective(fm)
                outside=actual.clone();outside[ids]=0;assert not outside.count_nonzero()
                norm=float(actual.norm());pred=eps*S;change=float(Fp)-baseline;negative=float(Fm)-baseline
                row['interventions'].append({'epsilon':eps,'constructed_delta_norm':float(delta.double().norm()),'effective_delta_norm':norm,
                  'effective_budget_relative_error':abs(norm-eps)/eps,'minus_effective_delta_norm':float(minus.norm()),
                  'F_before':baseline,'F_after':float(Fp),'F_change':change,'F_minus_change':negative,'first_order_prediction':pred,
                  'first_order_effective_prediction':float((G.double()*actual).sum()),'relative_prediction_error':abs(change-pred)/pred,
                  'central_change':(change-negative)/2,'central_relative_error':abs((change-negative)/2-pred)/pred,
                  'paired_cosine_before':cosine0,'paired_cosine_after':float(cp),'paired_cosine_minus':float(cm)})
                assert row['interventions'][-1]['effective_budget_relative_error']<1e-4
            rows.append(row)
        lookup={a['slot']:a for a in rows};pairs=[]
        for p in old['pairs']:
            l=lookup[p['left']];r=lookup[p['right']]
            dl=torch.tensor(l['mapped_direction'],dtype=torch.float64);dr=torch.tensor(r['mapped_direction'],dtype=torch.float64)
            pairs.append({**{k:p[k] for k in ['left','right','shared_object_indices','identical_evidence','cos_e','cos_learned']},
              'cos_projection':cosine(dl,dr),'comparison_space':'common IRRA 512D; never cross-slot 64D coordinates'})
        probe.assert_frozen()
        assert all(not p.requires_grad and p.grad is None for p in reader.parameters())
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        assert all(torch.equal(v,reader.state_dict()[k]) for k,v in frozen_reader.items())
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in frozen_projection.items())
        results.append({'row_id':old['row_id'],'pid':old['pid'],'caption':old['caption'],'F':baseline,'paired_cosine':cosine0,
          'attributes':rows,'all_slot_reads':reads,'pairs':pairs,'token_alignment':old['token_alignment']})
        print('DIAGNOSTIC',old['row_id'],flush=True)
        for p in pairs:
            if p['identical_evidence'] or (old['row_id']=='cuhk:99:1' and {p['left'],p['right']}=={'hair_length','upper_clothing_length'}):print('KEY_PAIR',json.dumps(p),flush=True)
    assert len(results)==9 and sum(len(r['attributes']) for r in results)==45
    pairs=[p for r in results for p in r['pairs']];dist={}
    for name,selected in [('all',pairs),('shared_object',[p for p in pairs if p['shared_object_indices']]),('identical_evidence',[p for p in pairs if p['identical_evidence']])]:
        dist[name]={method:distribution([p[key] for p in selected]) for method,key in [('Evidence','cos_e'),('v4','cos_learned'),('v4.1','cos_projection')]}
    assert dist['shared_object']['v4.1']['n']==18
    stats=[]
    for eps in prior['epsilons']:
        items=[i for r in results for a in r['attributes'] for i in a['interventions'] if i['epsilon']==eps]
        stats.append({'epsilon':eps,'norm_min':min(i['effective_delta_norm'] for i in items),'norm_max':max(i['effective_delta_norm'] for i in items),
          'budget_max_relative_error':max(i['effective_budget_relative_error'] for i in items),
          'prediction_median_relative_error':statistics.median(i['relative_prediction_error'] for i in items),
          'prediction_max_relative_error':max(i['relative_prediction_error'] for i in items),
          'central_median_relative_error':statistics.median(i['central_relative_error'] for i in items),
          'central_max_relative_error':max(i['central_relative_error'] for i in items),'positive_F_change':sum(i['F_change']>0 for i in items)})
    report={'caption_count':9,'attribute_count':45,'gallery_count':34054,'layer':8,'rank':64,'precision':'fp32','tau':.02,'epsilons':prior['epsilons'],
      'training_sha256':digest(OUT/'training.json'),'v4_sha256':digest(v4path),'results':results,'distributions':dist,'geometry':ge,'intervention_summary':stats,
      'attention_alpha_exact_regression_count':117,'v4_weight_raw_gradient_exact_regression_count':45,
      'IRRA_frozen':True,'reader_frozen':True,'projection_frozen':True,'zero_intervention_exact':True,'reader_projection_state_exact_unchanged':True,
      'special_token_delta_zero':True,'common_direction_space':512}
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');summary(report,training,previous,split)
    print('SHARED_OBJECT',json.dumps({k:{f:v for f,v in d.items() if f!='sorted_values'} for k,d in dist['shared_object'].items()}),flush=True)
    print('BUDGET',json.dumps(stats),flush=True)
if __name__=='__main__':run()
