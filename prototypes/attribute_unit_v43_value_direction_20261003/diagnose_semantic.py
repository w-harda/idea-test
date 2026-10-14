"""v4.3 diagnostics: existing D model, fixed semantic directions, same graph."""
import sys,json,statistics
from pathlib import Path
import torch
from semantic import ROOT,OUT,V42,V4,V41,REPO,ControlProjection,digest,setup,geometry,extract_directions,retrieval_readout
from projection import CODE,RAW,SlotReader,load_victim
MODE="D";BASEOUT=V42/"outputs"
sys.path.insert(0,str(REPO/'prototypes/attribute_unit_v3_query_20261002'))
from query_reader import AttributeUnitV2Probe,uniform_unit_gradient,mass_and_top,cosine,distribution

KEY_PAIRS=[('upper_clothing_color','upper_clothing_type'),('upper_clothing_color','upper_clothing_length'),('upper_clothing_type','upper_clothing_length'),('lower_clothing_color','lower_clothing_type')]

def run():
    setup();target=OUT/'diagnostic.json';assert not target.exists()
    all_training=json.loads((BASEOUT/'training.json').read_text());training={**all_training['conditions'][MODE],'protocol':all_training['protocol'],'training_completed_before_diagnostic':all_training['all_training_completed_before_diagnostic']};previous=json.loads((V4/'outputs/training.json').read_text());split=json.loads((V4/'outputs/split.json').read_text())
    assert training['training_completed_before_diagnostic'] and training['history'][-1]['epoch']==20
    assert digest(V42/'controls.py')==training['protocol']['source_sha256']
    assert digest(BASEOUT/f'{MODE}_final.pt')==training['checkpoint_sha256']
    assert digest(V4/'outputs/reader_final.pt')==training['protocol']['v4_checkpoint_sha256']
    assert digest(V4/'outputs/split.json')==training['protocol']['split_sha256']
    state=torch.load(BASEOUT/f'{MODE}_final.pt',map_location='cuda',weights_only=True)
    slots=state['slots'];values=state['values'];model=ControlProjection([len(v) for v in values],MODE).to('cuda')
    model.load_state_dict(state['state_dict']);model.eval().requires_grad_(False)
    with torch.no_grad():B=model.basis().detach()
    assert torch.equal(B,state['B']) and B.shape==(13,512,64)
    frozen_projection={k:v.clone() for k,v in model.state_dict().items()}
    rs=torch.load(V4/'outputs/reader_final.pt',map_location='cuda',weights_only=True)
    reader=SlotReader([len(v) for v in values]).to('cuda');reader.load_state_dict(rs['state_dict']);reader.eval().requires_grad_(False)
    frozen_reader={k:v.clone() for k,v in reader.state_dict().items()}
    v4path=V4/'outputs/diagnostic.json';assert digest(v4path)=='3cfc5f2bdc2bc40f7a9453a517bf06cd02f8ad16c69df6d50cdaf2f4c4c7629e'
    prior=json.loads(v4path.read_text());assert prior['epsilons']==[.01,.05,.1]
    old_D=json.loads((BASEOUT/'D/diagnostic.json').read_text());old_D_by={r['row_id']:r for r in old_D['results']}
    preserved={**all_training['protocol']['preserved_baseline_sha256'],**{str(p):digest(p) for p in [BASEOUT/'training.json',BASEOUT/'D_final.pt',BASEOUT/'D/diagnostic.json',V42/'controls.py',V42/'diagnose_controls.py',V41/'projection.py']}}
    assert digest(BASEOUT/'D_final.pt')=='12dc360f814efbd9ac0858d71f2d791f283c4f815819fb0b4aa6c7dd7c7a9715'
    protocol={'model':'v4.2 D epoch20; reused frozen','trained_this_round':False,'source_sha256':digest(__file__),'semantic_source_sha256':digest(ROOT/'semantic.py'),'layer':8,'rank':64,'precision':'fp32','tau':.02,'epsilons':prior['epsilons'],'center':'mean all class rows including null','null_targets':False,'preserved_baseline_sha256':preserved}
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    directions,direction_rows=extract_directions(model,B,slots,values)
    assert max(abs(r['d_norm']-1) for r in direction_rows)<2e-6
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
        D_old=old_D_by[old['row_id']];D_attrs={a['slot']:a for a in D_old['attributes']}
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
            sub=B[s].T@raw;S_subspace=float(sub.double().norm())
            assert S_subspace==D_attrs[a['slot']]['S_projection']
            assert torch.equal(sub.cpu(),torch.tensor(D_attrs[a['slot']]['g_sub'],dtype=torch.float32))
            d=directions[(a['slot'],a['canonical'])];r,S,sign=retrieval_readout(raw,d,S_subspace)
            assert 0<=S<=S_subspace*(1+1e-6) and d.numel()==512 and not d.requires_grad
            row={k:a[k] for k in ['slot','canonical','evidence_token_indices','object_token_indices','support_token_indices']}
            row.update(S_v4=Sraw,S_subspace=S_subspace,S_value=S,r=r,sign=sign,value_subspace_ratio=S/S_subspace,g_raw=raw.cpu().tolist(),g_sub=sub.cpu().tolist(),semantic_direction=d.cpu().tolist(),
              semantic_direction_norm=float(d.double().norm()),w=w.cpu().tolist(),alpha=ae.cpu().tolist(),read_pattern=a['read_pattern'],
              prediction_v4=a['prediction'],prediction_projection=reads[s]['prediction_projection'],interventions=[])
            for eps in prior['epsilons']:
                delta=eps*sign*w[:,None,None]*d[None,None,:]
                with torch.no_grad():
                    fp,_,actual=probe.encode(old['caption'],8,delta);fm,_,minus=probe.encode(old['caption'],8,-delta)
                    Fp,cp=probe.objective(fp);Fm,cm=probe.objective(fm)
                outside=actual.clone();outside[ids]=0;assert not outside.count_nonzero()
                norm=float(actual.norm());pred=eps*S;change=float(Fp)-baseline;negative=float(Fm)-baseline
                row['interventions'].append({'epsilon':eps,'constructed_delta_norm':float(delta.double().norm()),'effective_delta_norm':norm,
                  'effective_budget_relative_error':abs(norm-eps)/eps if sign else None,'zero_readout':sign==0,'minus_effective_delta_norm':float(minus.norm()),
                  'F_before':baseline,'F_after':float(Fp),'F_change':change,'F_minus_change':negative,'first_order_prediction':pred,
                  'first_order_effective_prediction':float((G.double()*actual).sum()),'relative_prediction_error':abs(change-pred)/pred if pred else None,'absolute_prediction_error':abs(change-pred),
                  'central_change':(change-negative)/2,'central_relative_error':abs((change-negative)/2-pred)/pred if pred else None,
                  'paired_cosine_before':cosine0,'paired_cosine_after':float(cp),'paired_cosine_minus':float(cm)})
                if sign:assert row['interventions'][-1]['effective_budget_relative_error']<1e-4
                else:assert not actual.count_nonzero() and change==0
            rows.append(row)
        lookup={a['slot']:a for a in rows};pairs=[]
        for p in old['pairs']:
            l=lookup[p['left']];r=lookup[p['right']]
            dl=torch.tensor(l['semantic_direction'],dtype=torch.float64);dr=torch.tensor(r['semantic_direction'],dtype=torch.float64)
            pairs.append({**{k:p[k] for k in ['left','right','shared_object_indices','identical_evidence','cos_e','cos_learned']},
              'cos_semantic':cosine(dl,dr),'cos_subspace':next(x['cos_projection'] for x in D_old['pairs'] if x['left']==p['left'] and x['right']==p['right']),'comparison_space':'common IRRA 512D; never cross-slot 64D coordinates'})
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
        dist[name]={method:distribution([p[key] for p in selected]) for method,key in [('Evidence','cos_e'),('v4','cos_learned'),('D_subspace','cos_subspace'),('value_semantic','cos_semantic')]}
    assert dist['shared_object']['value_semantic']['n']==18
    stats=[]
    for eps in prior['epsilons']:
        items=[i for r in results for a in r['attributes'] for i in a['interventions'] if i['epsilon']==eps and not i['zero_readout']]
        stats.append({'epsilon':eps,'norm_min':min(i['effective_delta_norm'] for i in items),'norm_max':max(i['effective_delta_norm'] for i in items),
          'budget_max_relative_error':max(i['effective_budget_relative_error'] for i in items),
          'prediction_median_relative_error':statistics.median(i['relative_prediction_error'] for i in items),
          'prediction_max_relative_error':max(i['relative_prediction_error'] for i in items),
          'central_median_relative_error':statistics.median(i['central_relative_error'] for i in items),
          'central_max_relative_error':max(i['central_relative_error'] for i in items),'positive_F_change':sum(i['F_change']>0 for i in items)})
    assert all(digest(Path(p))==h for p,h in preserved.items())
    report={'semantic_directions':direction_rows,'fixed_direction_count':len(direction_rows),'preserved_baselines_exact':True,'trained_this_round':False,'condition':MODE,'source_sha256':digest(__file__),'classifier_frozen':True,'no_parameter_gradients':True,'caption_count':9,'attribute_count':45,'gallery_count':34054,'layer':8,'rank':64,'precision':'fp32','tau':.02,'epsilons':prior['epsilons'],
      'training_sha256':digest(BASEOUT/'training.json'),'v4_sha256':digest(v4path),'results':results,'distributions':dist,'geometry':ge,'intervention_summary':stats,
      'attention_alpha_exact_regression_count':117,'v4_weight_raw_gradient_exact_regression_count':45,'D_subspace_exact_regression_count':45,
      'IRRA_frozen':True,'reader_frozen':True,'projection_frozen':True,'zero_intervention_exact':True,'reader_projection_state_exact_unchanged':True,
      'special_token_delta_zero':True,'common_direction_space':512}
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SHARED_OBJECT',json.dumps({k:{f:v for f,v in d.items() if f!='sorted_values'} for k,d in dist['shared_object'].items()}),flush=True)
    print('BUDGET',json.dumps(stats),flush=True)
if __name__=='__main__':run()
