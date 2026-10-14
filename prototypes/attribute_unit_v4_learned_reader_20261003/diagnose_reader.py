"""冻结训练后的reader，只在原9句计算历史对照和IRRA敏感性。"""
import sys,json,statistics
from pathlib import Path
import torch
from learned_reader import ROOT,REPO,OUT,DIAG,CODE,RAW,SlotReader,digest,setup,load_victim
sys.path.insert(0,str(REPO/'prototypes/attribute_unit_v3_query_20261002'))
from query_reader import AttributeUnitV2Probe,uniform_unit_gradient,unit_delta,mass_and_top,cosine,distribution

def write_summary(report,training,split):
    fmt=lambda x:f'{x:.6g}'
    lines=['# Attribute Unit v4 Learnable Slot Reader（2026-10-03）','',
      '独立feasibility prototype；只训练13个slot query和13个线性value/null classifier。IRRA全部frozen，训练只使用caption block8 fp32 states与既有Stage01 canonical labels，无image/retrieval objective/gradient/attack/provenance attention supervision；没有value prompt、MLP、projection或linked_object embedding。',
      '', 'attention为q_sᵀh_j/sqrt(512)、content-only softmax，z_s=Σalpha_sj h_j；普通sum_s CE。sensitivity阶段reader全部冻结，baseline alpha/w detach固定；w=alpha/||alpha||2、g=wᵀG、S=||g||2、delta=εw(g/S)ᵀ，干预进入原IRRA后续forward。',
      '', '## 固定训练协议与划分','',
      f"CUHK原train共{split['source_train_caption_count']}句；诊断9句属于{len(split['original_diagnostic_pids'])}个原PID；另排除{len(split['duplicate_text_excluded_pids'])}个含完全相同诊断文本的PID，整个PID从train/validation都排除（{split['excluded_caption_count']}句）。seed=42，剩余PID固定80/20划分：train {split['train_caption_count']}句/{split['train_pid_count']} PID；validation {split['validation_caption_count']}句/{split['validation_pid_count']} PID。",
      '', 'Adam lr=.001、weight_decay=0、batch512、固定20 epochs，取最终第20轮checkpoint，未按9句诊断/validation择优改模型、数据或超参。query normal std=.02，classifier默认Linear初始化。',
      '', f"query参数{training['protocol']['query_parameters']}，classifier参数{training['protocol']['classifier_parameters']}；训练/验证中{split['truncated_caption_count']}句超75个content BPE，按原IRRA截断且不改标签；这些样本未根据结果剔除。",
      '', '## Validation classification（13槽）','',
      'accuracy衡量复现Stage01 pseudo labels，不代表人工语义ground truth；majority baseline取train最多类。non-null分母单独报告。',
      '', '| slot | 样本数 | overall accuracy | non-null数 | non-null accuracy | train majority | majority accuracy |',
      '| --- | ---: | ---: | ---: | ---: | --- | ---: |']
    for r in training['final_validation']['slots']:
        nn='NA' if r['non_null_accuracy'] is None else f"{r['non_null_accuracy']*100:.3f}%"
        lines.append(f"| {r['slot']} | {r['sample_count']} | {r['accuracy']*100:.3f}% | {r['non_null_count']} | {nn} | {r['majority_train_value']} | {r['majority_baseline_accuracy']*100:.3f}% |")
    lines+=['', '## 训练曲线','', '| epoch | train CE sum | validation CE sum |', '| ---: | ---: | ---: |']
    for e in training['history']:lines.append(f"| {e['epoch']} | {fmt(e['train_CE_sum'])} | {fmt(e['validation']['CE_sum'])} |")
    lines+=['','## 历史方向对照','', '| group | method | n | mean | median | min | max |', '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for group,methods in report['distributions'].items():
        for method,d in methods.items():lines.append(f"| {group} | {method} | {d['n']} | {fmt(d['mean'])} | {fmt(d['median'])} | {fmt(d['min'])} | {fmt(d['max'])} |")
    lines+=['','## 关键3个held-out案例','', '| caption | pair | Evidence cosine | Value cosine | Learned cosine | learned w cosine | learned w L2差 | learned S left/right |',
      '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for r in report['results']:
        for p in r['pairs']:
            if p['identical_evidence'] or (r['row_id']=='cuhk:99:1' and {p['left'],p['right']}=={'hair_length','upper_clothing_length'}):
                lookup={a['slot']:a for a in r['attributes']}
                lines.append(f"| {r['row_id']} | {p['left']} / {p['right']} | {fmt(p['cos_e'])} | {fmt(p['cos_value'])} | {fmt(p['cos_learned'])} | {fmt(p['learned_weight_cosine'])} | {fmt(p['learned_weight_l2_difference'])} | {fmt(lookup[p['left']]['S_learned'])} / {fmt(lookup[p['right']]['S_learned'])} |")
    lines+=['','326:0原Stage01衣物绑定存疑，只作机制诊断；hair无linked_object记录时hair词位于other context，不补造绑定。不设cosine硬阈值，不作小样本显著性结论。',
      '', '## 原9句×13槽 reader行为（117项）','',
      'top5包含原content标点/subword，不人工过滤。mass为概率alpha；context是evidence/object并集外的content。null槽没有已有evidence/object时两种mass=0、context≈1，仅用于完整审计；不作为实际非null属性sensitivity。',
      '', '| caption | slot | label | prediction | evidence mass | object mass | context mass | top5 index:piece:alpha |',
      '| --- | --- | --- | --- | ---: | ---: | ---: | --- |']
    for r in report['results']:
        lines+=['',f"### {r['row_id']}",'',r['caption'],'',
          '| caption | slot | label | prediction | evidence mass | object mass | context mass | top5 index:piece:alpha |',
          '| --- | --- | --- | --- | ---: | ---: | ---: | --- |']
        for a in r['all_slot_reads']:
            p=a['read_pattern'];top=', '.join(f"{t['token_index']}:{t['bpe_piece']}:{t['alpha']:.5f}" for t in p['top5'])
            lines.append(f"| {r['row_id']} | {a['slot']} | {a['label']} | {a['prediction']} | {fmt(p['evidence_mass'])} | {fmt(p['object_mass'])} | {fmt(p['context_mass'])} | {top} |")
    lines+=['','## 原45属性敏感度','', '| caption | slot=value | S_e | S_value | S_learned |', '| --- | --- | ---: | ---: | ---: |']
    for r in report['results']:
        for a in r['attributes']:lines.append(f"| {r['row_id']} | {a['slot']}={a['canonical']} | {fmt(a['S_e'])} | {fmt(a['S_value'])} | {fmt(a['S_learned'])} |")
    lines+=['','## 等预算与一阶预测','', '| ε | actual norm min/max | budget max relative error | prediction median/max relative error | central median/max relative error | positive ΔF |',
      '| ---: | --- | ---: | --- | --- | ---: |']
    for s in report['intervention_summary']:
        lines.append(f"| {s['epsilon']} | {s['norm_min']:.12g} / {s['norm_max']:.12g} | {fmt(s['budget_max_relative_error'])} | {fmt(s['prediction_median_relative_error'])} / {fmt(s['prediction_max_relative_error'])} | {fmt(s['central_median_relative_error'])} / {fmt(s['central_max_relative_error'])} | {s['positive_F_change']}/45 |")
    lines+=['','误差为比例，不是百分数。实际fp32加法有舍入，不能称逐位等ε。原始JSON保存135次paired cosine/F before/after和负向干预/有效内积预测。',
      '', '## 完整同句方向对照（92对）','', '| caption | left/right | Evidence cosine | Value cosine | Learned cosine | shared object | identical evidence |',
      '| --- | --- | ---: | ---: | ---: | --- | --- |']
    for r in report['results']:
        for p in r['pairs']:lines.append(f"| {r['row_id']} | {p['left']} / {p['right']} | {fmt(p['cos_e'])} | {fmt(p['cos_value'])} | {fmt(p['cos_learned'])} | {p['shared_object_indices']} | {p['identical_evidence']} |")
    lines+=['','## 实现与回归','',
      '所有IRRA参数及reader参数在sensitivity阶段requires_grad=False/grad=None；reader state逐值保持不变。zero intervention exact、baseline F/paired cosine与旧报告逐值相同、45个Evidence/Value g/S逐值回归；SOT/EOT/pad实际delta=0。',
      '', f"服务器源码/tests/缓存/原始outputs根目录：`{ROOT}`。Windows只同步本摘要。未改正式pipeline/Stage1–3/Top-K/generator，未暂存/commit/push。",
      '', f"training SHA256：`{report['training_sha256']}`；reader checkpoint：`{training['checkpoint_sha256']}`；diagnostic源码：`{digest(__file__)}`；training源码：`{training['protocol']['source_sha256']}`。",
      '', f"Evidence旧结果：`{report['evidence_sha256']}`；Value旧结果：`{report['value_sha256']}`。"]
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def run():
    setup();target=OUT/'diagnostic.json';assert not target.exists()
    training=json.loads((OUT/'training.json').read_text());split=json.loads((OUT/'split.json').read_text())
    assert training['training_completed_before_diagnostic'] and training['history'][-1]['epoch']==20
    assert digest(OUT/'reader_final.pt')==training['checkpoint_sha256'] and digest(ROOT/'learned_reader.py')==training['protocol']['source_sha256']
    state=torch.load(OUT/'reader_final.pt',map_location='cuda',weights_only=True)
    slots=state['slots'];values=state['values'];reader=SlotReader([len(v) for v in values]).to('cuda')
    reader.load_state_dict(state['state_dict'],strict=True);reader.eval().requires_grad_(False)
    frozen_state={k:v.clone() for k,v in reader.state_dict().items()}
    assert all(not p.requires_grad and p.grad is None for p in reader.parameters())
    prior=json.loads(DIAG.read_text());assert digest(DIAG)=='5de3594ade4be5267c019b1c0e2e74bb681985916121c963f35598bf2b87fe0b'
    evpath=REPO/'prototypes/attribute_unit_v21_evidence_20261002/outputs/evidence.json'
    assert digest(evpath)=='c77307003057df8f0514f85312434f1e5756f2fc1b0692b3693a60642ae91f1f'
    ev={r['row_id']:r for r in json.loads(evpath.read_text())['results']}
    full=json.loads(Path('/home/lzf/ldx/outputs/idea-TBPS-test1/stage03-attribute-unit-v2-20261002/prototype.json').read_text())
    fullby={r['row_id']:r for r in full['results']}
    cache=Path('/home/lzf/ldx/outputs/idea-TBPS-test1/stage03-v2-smoke-20261001/clean-train-gallery.pt')
    assert digest(cache)==prior['gallery_sha256']
    payload=torch.load(cache,map_location='cpu',weights_only=True)
    assert payload['spec']==training['protocol']['IRRA_spec'] and len(payload['features'])==34054
    assert str(torch.__version__)==payload['spec']['torch_version']
    gallery=payload['features'].to('cuda');victim=load_victim();raw=json.loads(RAW.read_text());results=[]
    for old in prior['results']:
        pid=raw[int(old['row_id'].split(':')[1])]['id']
        assert pid in split['held_out_pids'] and pid not in split['train_pids'] and pid not in split['validation_pids']
        probe=AttributeUnitV2Probe(victim,gallery,fullby[old['row_id']]['paired_gallery_index'],tau=.02)
        with torch.no_grad():original=victim.encode_texts([old['caption']])
        feature,H,z=probe.encode(old['caption'],8)
        assert torch.equal(original,feature.detach()) and not z.count_nonzero()
        with torch.no_grad():zf,_,az=probe.encode(old['caption'],8,torch.zeros_like(H))
        assert torch.equal(original,zf) and not az.count_nonzero()
        F0,c0=probe.objective(feature);G,=torch.autograd.grad(F0,H);baseline=float(F0);cosine0=float(c0)
        assert baseline==old['F'] and cosine0==old['paired_cosine']
        tokens=old['token_alignment']['content_tokens'];ids=[t['token_index'] for t in tokens]
        mask=torch.zeros(1,H.shape[0],device=H.device,dtype=torch.bool);mask[0,ids]=True
        with torch.no_grad():alpha,zslot,logits=reader(H.detach().permute(1,0,2),mask)
        alpha=alpha[0].detach();reads=[];rows=[];attrs={a['slot']:a for a in old['attributes']}
        for s,slot in enumerate(slots):
            a=attrs.get(slot,{'evidence_token_indices':[],'object_token_indices':[]})
            pattern=mass_and_top(alpha[s],a,tokens)
            reads.append({'slot':slot,'label':a.get('canonical','null'),'prediction':values[s][int(logits[s].argmax())],
              'class_probabilities':logits[s].softmax(-1)[0].cpu().tolist(),'alpha':alpha[s].cpu().tolist(),'z':zslot[0,s].cpu().tolist(),'read_pattern':pattern})
        eva={a['slot']:a for a in ev[old['row_id']]['attributes']}
        for a in old['attributes']:
            s=slots.index(a['slot']);ae=alpha[s];w=(ae/ae.norm()).detach();g=(w[:,None]*G[:,0]).sum(0);score=float(g.double().norm())
            assert score>0 and not w.requires_grad and abs(float(w.double().norm())-1)<1e-6
            _,ge,_,se=uniform_unit_gradient(G,a['evidence_token_indices'])
            assert se==a['S_e'] and torch.equal(ge.cpu(),torch.tensor(eva[a['slot']]['g_e'],dtype=torch.float32))
            oldw=torch.tensor(a['w_value'],device=H.device,dtype=torch.float32);oldg=(oldw[:,None]*G[:,0]).sum(0)
            assert float(oldg.double().norm())==a['S_value'] and torch.equal(oldg.cpu(),torch.tensor(a['g_value'],dtype=torch.float32))
            row={k:a[k] for k in ['slot','canonical','evidence_token_indices','object_token_indices','support_token_indices']}
            row.update(S_e=se,S_value=a['S_value'],S_learned=score,g_learned=g.cpu().tolist(),
              d_learned=(g.double()/score).cpu().tolist(),w_learned=w.cpu().tolist(),alpha_learned=ae.cpu().tolist(),
              read_pattern=reads[s]['read_pattern'],prediction=reads[s]['prediction'],interventions=[])
            for eps in prior['epsilons']:
                delta=unit_delta(w,g,eps)
                with torch.no_grad():
                    fp,_,actual=probe.encode(old['caption'],8,delta);fm,_,minus=probe.encode(old['caption'],8,-delta)
                    Fp,cp=probe.objective(fp);Fm,cm=probe.objective(fm)
                outside=actual.clone();outside[ids]=0;assert not outside.count_nonzero()
                norm=float(actual.norm());pred=eps*score;change=float(Fp)-baseline;negative=float(Fm)-baseline
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
            gl=torch.tensor(l['g_learned'],dtype=torch.float64);gr=torch.tensor(r['g_learned'],dtype=torch.float64)
            wl=torch.tensor(l['w_learned'],dtype=torch.float64);wr=torch.tensor(r['w_learned'],dtype=torch.float64)
            # 只保留本轮允许的Evidence / Value / Learned对照字段。
            pairs.append({**{k:p[k] for k in ['left','right','shared_object_indices','identical_evidence','cos_e','cos_value']},
              'cos_learned':cosine(gl,gr),'learned_weight_cosine':cosine(wl,wr),'learned_weight_l2_difference':float((wl-wr).norm()),
              'learned_g_l2_difference':float((gl-gr).norm()),'learned_S_difference':r['S_learned']-l['S_learned'],
              'weights_exact_equal':torch.equal(wl,wr),'gradients_exact_equal':torch.equal(gl,gr)})
        probe.assert_frozen();assert all(not p.requires_grad and p.grad is None for p in reader.parameters())
        assert all(torch.equal(v,reader.state_dict()[k]) for k,v in frozen_state.items())
        results.append({'row_id':old['row_id'],'pid':pid,'caption':old['caption'],'F':baseline,'paired_cosine':cosine0,
          'attributes':rows,'all_slot_reads':reads,'pairs':pairs,'token_alignment':old['token_alignment']})
        print('DIAGNOSTIC',old['row_id'],flush=True)
        for p in pairs:
            if p['identical_evidence'] or (old['row_id']=='cuhk:99:1' and {p['left'],p['right']}=={'hair_length','upper_clothing_length'}):print('KEY_PAIR',json.dumps(p),flush=True)
    assert len(results)==9 and sum(len(r['attributes']) for r in results)==45
    assert sum(len(r['all_slot_reads']) for r in results)==117
    pairs=[p for r in results for p in r['pairs']];dist={}
    for name,selected in [('all',pairs),('shared_object',[p for p in pairs if p['shared_object_indices']]),('identical_evidence',[p for p in pairs if p['identical_evidence']])]:
        dist[name]={method:distribution([p[key] for p in selected]) for method,key in [('Evidence','cos_e'),('Value','cos_value'),('Learned','cos_learned')]}
    stats=[]
    for eps in prior['epsilons']:
        items=[i for r in results for a in r['attributes'] for i in a['interventions'] if i['epsilon']==eps]
        stats.append({'epsilon':eps,'norm_min':min(i['effective_delta_norm'] for i in items),'norm_max':max(i['effective_delta_norm'] for i in items),
          'budget_max_relative_error':max(i['effective_budget_relative_error'] for i in items),
          'prediction_median_relative_error':statistics.median(i['relative_prediction_error'] for i in items),
          'prediction_max_relative_error':max(i['relative_prediction_error'] for i in items),
          'central_median_relative_error':statistics.median(i['central_relative_error'] for i in items),
          'central_max_relative_error':max(i['central_relative_error'] for i in items),'positive_F_change':sum(i['F_change']>0 for i in items)})
    report={'caption_count':9,'attribute_count':45,'gallery_count':34054,'layer':8,'precision':'fp32','tau':.02,'epsilons':prior['epsilons'],
      'training_sha256':digest(OUT/'training.json'),'value_sha256':digest(DIAG),'evidence_sha256':digest(evpath),'results':results,'distributions':dist,
      'intervention_summary':stats,'IRRA_frozen':True,'reader_frozen':True,'zero_intervention_exact':True,'reader_state_exact_unchanged':True,
      'historical_gradient_exact_regression':True,'special_token_delta_zero':True}
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');write_summary(report,training,split)
    print('DISTRIBUTIONS',json.dumps(dist),flush=True);print('BUDGET',json.dumps(stats),flush=True)
if __name__=='__main__':run()
