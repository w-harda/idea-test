"""v3.1: change only fixed-query representation from EOT to value-span states."""
import sys,os,json,math,statistics,importlib
from pathlib import Path
from functools import partial
sys.dont_write_bytecode=True
os.environ.setdefault('TORCH_HOME','/home/lzf/ldx/cache/torch')
os.environ.setdefault('HF_HOME','/home/lzf/ldx/cache/huggingface')
REPO=Path('/home/lzf/ldx/projects/idea-TBPS-test1')
sys.path.insert(0,str(REPO/'prototypes/attribute_unit_v3_query_20261002'))
from query_reader import BASE,CODE,T_QUERY,query_text,read_weights,mass_and_top,ranking,rank_string,cosine,distribution,vector_info,uniform_unit_gradient,unit_delta,AttributeUnitV2Probe,FrozenIRRA,sha256,align_irra_tokens
from attributes.irra_token_alignment import _span_token_indices
import torch

def value_span_indices(slot,canonical,alignment):
    text=query_text(slot,canonical); readable=canonical.replace('_',' ')
    start=len(slot.replace('_',' '))+1
    assert text[start:]==readable and alignment['caption']==text
    span={'start':start,'end':len(text),'raw':readable}
    indices,reasons=_span_token_indices(text,span,alignment['all_content_tokens'])
    if reasons or not indices: raise ValueError(('value span mapping unresolved',reasons))
    assert all(0<i<alignment['eot_index'] for i in indices)
    return span,sorted(indices)

def pooled_value_query(states,indices):
    states=states.detach().float(); indices=sorted(set(indices))
    assert states.ndim==3 and states.shape[1]==1 and indices and torch.isfinite(states).all()
    w=torch.zeros(len(states),dtype=torch.float32,device=states.device)
    w[indices]=1/math.sqrt(len(indices))
    raw=(w[:,None]*states[:,0]).sum(0)
    norm=raw.norm()
    if not norm: raise ValueError('zero pooled query')
    # read_weights performs this same L2 normalization inside cosine, once.
    unit=raw/norm
    return raw,unit,w

def write_summary(report,out):
    fmt=lambda x:f'{x:.6g}'
    lines=['# Attribute Unit v3.1 Value-token Query（2026-10-03）','',
      '唯一修改query representation：同一固定模板、同一block8，q_EOT改为canonical value span的contextual states按1/sqrt(k_value)聚合。read_weights直接复用v3；cosine内部完成q的L2 normalization，不额外重复归一化。',
      '', 'frozen IRRA、fp32、原9条caption/45属性、34054图clean train gallery、paired soft rank τ=.02、T_q=.1、ε=.01/.05/.1均不变。query模板不改，value span按构造字符位置映射官方BPE，slot prefix/SOT/EOT/pad不进入value pooling。不训练、不搜索、不改原caption/Stage01/provenance/ontology。',
      '', '## 原18对shared-object方向分布','', '| 方法 | mean | median | min | max |', '| --- | ---: | ---: | ---: | ---: |']
    for method,s in report['distributions']['shared_object'].items():
        lines.append(f"| {method} | {fmt(s['mean'])} | {fmt(s['median'])} | {fmt(s['min'])} | {fmt(s['max'])} |")
    lines+=['','Full仅沿用历史分布，未重跑新方法；主要对照为Evidence-only/EOT Query/Value-token Query。',
      '', '## 等预算与一阶预测','', '| ε | 实际norm min/max | budget最大相对误差 | 预测误差median/max | 中心误差median/max | 正ΔF |',
      '| ---: | --- | ---: | --- | --- | ---: |']
    for s in report['intervention_summary']:
        lines.append(f"| {s['epsilon']} | {s['norm_min']:.12g} / {s['norm_max']:.12g} | {fmt(s['budget_max_relative_error'])} | {fmt(s['prediction_median_relative_error'])} / {fmt(s['prediction_max_relative_error'])} | {fmt(s['central_median_relative_error'])} / {fmt(s['central_max_relative_error'])} | {s['positive_F_change']}/45 |")
    lines+=['','以上误差为比例，不是百分数；实际fp32加法有舍入。原始JSON保存135次paired cosine/F before/after、实际delta及内积预测。reader权重在baseline固定，干预时不重算attention。',
      '', '## 固定query/value span','', '| slot=value | 固定query | value span | value token indices/pieces |', '| --- | --- | --- | --- |']
    for q in report['queries']:
        tok=', '.join(f"{t['token_index']}:{t['bpe_piece']}" for t in q['value_tokens'])
        lines.append(f"| {q['slot']}={q['canonical']} | {q['query_text']} | {q['value_span']} | {tok} |")
    lines+=['','## 全部同caption方向对照','',
      '| caption | left | right | Evidence cosine | EOT cosine | Value cosine | Value w cosine | Value w L2差 | shared object | identical evidence |',
      '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |']
    for r in report['results']:
        for p in r['pairs']:
            lines.append(f"| {r['row_id']} | {p['left']} | {p['right']} | {fmt(p['cos_e'])} | {fmt(p['cos_query'])} | {fmt(p['cos_value'])} | {fmt(p['value_weight_cosine'])} | {fmt(p['value_weight_l2_difference'])} | {p['shared_object_indices']} | {p['identical_evidence']} |")
    lines+=['','## EOT vs Value read pattern（全部45属性）','',
      'mass基于alpha概率，不是L2权重w。context是evidence/object并集以外content；hair缺失linked_object时hair相关词位于context，不补造绑定。top5保留标点/subword；不设cosine阈值，不作显著性判断。',
      '', '| caption | query | 方法 | evidence mass | object mass | context mass | top5 index:piece:alpha |',
      '| --- | --- | --- | ---: | ---: | ---: | --- |']
    for r in report['results']:
        for a in r['attributes']:
            for method,p in [('EOT',a['eot_read_pattern']),('Value',a['value_read_pattern'])]:
                top=', '.join(f"{t['token_index']}:{t['bpe_piece']}:{t['alpha']:.4f}" for t in p['top5'])
                lines.append(f"| {r['row_id']} | {a['query_text']} | {method} | {fmt(p['evidence_mass'])} | {fmt(p['object_mass'])} | {fmt(p['context_mass'])} | {top} |")
    for r in report['results']:
        lines+=['',f"## {r['row_id']}",'',r['caption'],'']
        for method in ('S_e','S_EOT','S_value'): lines+=['排序 '+method+'：'+rank_string(r['rankings'][method]),'']
        lines+=['| slot=value | S_e | S_EOT | S_value |','| --- | ---: | ---: | ---: |']
        for a in r['attributes']: lines.append(f"| {a['slot']}={a['canonical']} | {fmt(a['S_e'])} | {fmt(a['S_EOT'])} | {fmt(a['S_value'])} |")
    lines+=['','## 回归与产物','',
       '原始caption baseline F/paired cosine逐值一致；g_full/g_e与旧产物一致；query EOT hidden、全部45个EOT alpha/w/r/g/S/read_pattern逐值复现。zero intervention逐值一致，模型parameters requires_grad=False/grad=None；query/reader无训练参数，特殊token delta=0。',
       '', f"服务器目录：`{out.parent}`。原始JSON保存q_value raw/unit、value pooling weights、alpha/w、g/d/h_unit、92对比较与135次干预；Windows仅同步研究摘要。正式pipeline未改、未commit/push。",
       '', f"旧EOT结果SHA：`{report['eot_sha256']}`；旧Evidence SHA：`{report['evidence_sha256']}`；新源码SHA：`{report['code_sha256']}`。"]
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def run():
    root=Path(__file__).resolve().parent; out=root/'outputs'; out.mkdir(exist_ok=True)
    output=out/'value.json'
    if output.exists(): raise FileExistsError(output)
    eot_path=REPO/'prototypes/attribute_unit_v3_query_20261002/outputs/query.json'
    e_path=REPO/'prototypes/attribute_unit_v21_evidence_20261002/outputs/evidence.json'
    full_path=BASE/'prototype.json'
    assert sha256(eot_path)=='b97e9f86dd228558b3a0c9df1c8f6e29102f8d3984e67afc348faa38522268e1'
    assert sha256(e_path)=='c77307003057df8f0514f85312434f1e5756f2fc1b0692b3693a60642ae91f1f'
    assert sha256(full_path)=='df6b2b2032dde33fed19bf45e9ad2fbb6301676bc5514ed81303ed3e241e42fc'
    prior=json.loads(eot_path.read_text()); ev=json.loads(e_path.read_text()); full=json.loads(full_path.read_text())
    assert prior['T_query']==T_QUERY==.1 and prior['epsilons']==[.01,.05,.1]
    assert [r['row_id'] for r in prior['results']]==[r['row_id'] for r in ev['results']]==full['row_ids']
    old_q={(q['slot'],q['canonical']):q for q in prior['queries']}
    ev_by={r['row_id']:r for r in ev['results']}; full_by={r['row_id']:r for r in full['results']}
    cache=Path('/home/lzf/ldx/outputs/idea-TBPS-test1/stage03-v2-smoke-20261001/clean-train-gallery.pt')
    assert sha256(cache)==prior['gallery_sha256']
    payload=torch.load(cache,map_location='cpu',weights_only=True); spec=payload['spec']
    assert spec==full['gallery_spec'] and len(payload['features'])==34054 and str(torch.__version__)==spec['torch_version']
    irra=Path('/home/lzf/Attack/IRRA-main/IRRA-main'); checkpoint=Path('/home/lzf/Attack/IRRA/logs/CUHK-PEDES/best.pth'); config=CODE/'configs/irra_cuhk_official.yaml'
    assert sha256(checkpoint)==spec['checkpoint_sha256'] and sha256(config)==spec['config_sha256']
    for name,digest in spec['source_sha256'].items(): assert sha256(irra/name)==digest
    assert sha256(CODE/'ontology/upar_attribute_space.yaml')==prior['ontology_sha256']
    torch.manual_seed(42); torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False
    sys.path.insert(0,str(irra)); official=importlib.import_module('model.build')
    official.build_CLIP_from_openai_pretrained=partial(official.build_CLIP_from_openai_pretrained,download_root='/home/lzf/ldx/cache/clip')
    victim=FrozenIRRA(irra,checkpoint,config,device='cuda').eval(); victim.model.float()
    gallery=payload['features'].to('cuda'); qcache={}; qrecords=[]; results=[]
    for old in prior['results']:
        probe=AttributeUnitV2Probe(victim,gallery,full_by[old['row_id']]['paired_gallery_index'],tau=.02)
        for a in old['attributes']:
            key=(a['slot'],a['canonical'])
            if key in qcache: continue
            text=query_text(*key); assert text==old_q[key]['query_text']
            alignment=align_irra_tokens(text,victim.tokenizer,victim.text_length)
            assert not alignment['truncated'] and alignment['token_ids']==old_q[key]['token_ids']
            span,value_ids=value_span_indices(*key,alignment)
            with torch.no_grad(): _,qH,z=probe.encode(text,8)
            assert not z.count_nonzero() and qH.shape[-1]==512 and qH.dtype==torch.float32
            qEOT=qH[alignment['eot_index'],0].detach().clone()
            assert torch.equal(qEOT.cpu(),torch.tensor(old_q[key]['q_eot_hidden'],dtype=torch.float32))
            qValue,qUnit,qw=pooled_value_query(qH,value_ids)
            qcache[key]=(qEOT,qValue)
            lookup={t['token_index']:t for t in alignment['content_tokens']}
            qrecords.append({'slot':key[0],'canonical':key[1],'query_text':text,'value_span':span,
              'value_indices':value_ids,'value_tokens':[lookup[i] for i in value_ids],
              'q_value_raw':qValue.cpu().tolist(),'q_value_unit':qUnit.cpu().tolist(),
              'query_value_pool_weights':qw.cpu().tolist(),'cos_EOT_value':cosine(qEOT.cpu(),qValue.cpu())})
        with torch.no_grad(): original=victim.encode_texts([old['caption']])
        feature,H,z=probe.encode(old['caption'],8)
        assert torch.equal(feature.detach(),original) and not z.count_nonzero() and H.dtype==torch.float32
        with torch.no_grad(): zerof,_,actualz=probe.encode(old['caption'],8,torch.zeros_like(H))
        assert torch.equal(zerof,original) and not actualz.count_nonzero()
        F,cos0=probe.objective(feature); G,=torch.autograd.grad(F,H)
        baseline=float(F); cosine0=float(cos0)
        assert baseline==old['F'] and cosine0==old['paired_cosine']
        e_attrs={a['slot']:a for a in ev_by[old['row_id']]['attributes']}; f_attrs={a['slot']:a for a in full_by[old['row_id']]['attribute_results']}
        tokens=old['token_alignment']['content_tokens']; ids=[t['token_index'] for t in tokens]; rows=[]
        for a in old['attributes']:
            _,gf,_,sf=uniform_unit_gradient(G,a['support_token_indices']); _,ge,_,se=uniform_unit_gradient(G,a['evidence_token_indices'])
            assert sf==a['S_full'] and se==a['S_e']
            assert torch.equal(gf.cpu(),torch.tensor(f_attrs[a['slot']]['attribute_gradient'],dtype=torch.float32))
            assert torch.equal(ge.cpu(),torch.tensor(e_attrs[a['slot']]['g_e'],dtype=torch.float32))
            qe,qv=qcache[(a['slot'],a['canonical'])]
            we,ae,re=read_weights(H[:,0],qe,ids)
            gEOT=(we[:,None]*G[:,0]).sum(0); sEOT=float(gEOT.double().norm())
            for actual,saved in [(we,a['w']),(ae,a['alpha']),(re,a['content_cosines']),(gEOT,a['g_query'])]:
                assert torch.equal(actual.cpu(),torch.tensor(saved,dtype=torch.float32))
            e_pattern=mass_and_top(ae,a,tokens)
            assert sEOT==a['S_query'] and e_pattern==a['read_pattern']
            w,alpha,similarity=read_weights(H[:,0],qv,ids)
            assert not w.requires_grad and not alpha.requires_grad and abs(float(w.double().norm())-1)<1e-6
            g=(w[:,None]*G[:,0]).sum(0); score=float(g.double().norm()); assert score>0
            row={k:a[k] for k in ['slot','canonical','query_text','semantic_spans','mentions','linked_objects','evidence_token_indices','object_token_indices','support_token_indices']}
            row.update(S_e=se,S_EOT=sEOT,S_value=score,g_value=g.cpu().tolist(),d_value=vector_info(g)['direction'],
              w_value=w.cpu().tolist(),alpha_value=alpha.cpu().tolist(),content_cosines_value=similarity.cpu().tolist(),
              h_value_unit=((w[:,None]*H.detach()[:,0]).sum(0)).cpu().tolist(),eot_read_pattern=e_pattern,
              value_read_pattern=mass_and_top(alpha,a,tokens),interventions=[])
            for eps in prior['epsilons']:
                delta=unit_delta(w,g,eps)
                with torch.no_grad():
                    fp,_,actual=probe.encode(old['caption'],8,delta); fm,_,negdelta=probe.encode(old['caption'],8,-delta)
                    Fp,cp=probe.objective(fp); Fm,cm=probe.objective(fm)
                outside=actual.clone(); outside[ids]=0
                assert not outside.count_nonzero()
                norm=float(actual.norm()); predicted=eps*score; change=float(Fp)-baseline; negative=float(Fm)-baseline
                rec={'epsilon':eps,'constructed_delta_norm':float(delta.double().norm()),'effective_delta_norm':norm,
                  'effective_budget_relative_error':abs(norm-eps)/eps,'minus_effective_delta_norm':float(negdelta.norm()),
                  'F_before':baseline,'F_after':float(Fp),'F_change':change,'F_minus_change':negative,
                  'first_order_prediction':predicted,'first_order_effective_prediction':float((G.double()*actual).sum()),
                  'relative_prediction_error':abs(change-predicted)/predicted,'central_change':(change-negative)/2,
                  'central_relative_error':abs((change-negative)/2-predicted)/predicted,
                  'paired_cosine_before':cosine0,'paired_cosine_after':float(cp),'paired_cosine_minus':float(cm)}
                assert rec['effective_budget_relative_error']<1e-4
                row['interventions'].append(rec)
            rows.append(row)
        lookup={a['slot']:a for a in rows}; pairs=[]
        for p in old['pairs']:
            l=lookup[p['left']]; r=lookup[p['right']]
            wl=torch.tensor(l['w_value'],dtype=torch.float64); wr=torch.tensor(r['w_value'],dtype=torch.float64)
            gl=torch.tensor(l['g_value'],dtype=torch.float64); gr=torch.tensor(r['g_value'],dtype=torch.float64)
            pairs.append({**p,'cos_value':cosine(gl,gr),'value_weight_cosine':cosine(wl,wr),
              'value_weight_l2_difference':float((wl-wr).norm()),'value_g_l2_difference':float((gl-gr).norm()),
              'value_g_relative_difference':float((gl-gr).norm()/gl.norm()),
              'value_S_difference':r['S_value']-l['S_value'],'value_weights_exact_equal':torch.equal(wl,wr),
              'value_gradients_exact_equal':torch.equal(gl,gr)})
        probe.assert_frozen()
        results.append({'row_id':old['row_id'],'caption':old['caption'],'F':baseline,'paired_cosine':cosine0,
          'attributes':rows,'pairs':pairs,'provenance':old['provenance'],'token_alignment':old['token_alignment'],
          'rankings':{method:ranking(rows,method) for method in ('S_e','S_EOT','S_value')}})
        print(old['row_id'],'value ranking:',rank_string(results[-1]['rankings']['S_value']),flush=True)
        for p in pairs:
            if p['identical_evidence'] or (p['left']=='hair_length' and p['right']=='upper_clothing_length'):
                print('KEY_PAIR',old['row_id'],p,flush=True)
    assert len(results)==9 and sum(len(r['attributes']) for r in results)==45 and len(qrecords)==len(prior['queries'])
    pairs=[p for r in results for p in r['pairs']]; distributions={}
    for name,selected in [('all',pairs),('shared_object',[p for p in pairs if p['shared_object_indices']]),('identical_evidence',[p for p in pairs if p['identical_evidence']])]:
        distributions[name]={method:distribution([p[key] for p in selected]) for method,key in [('Full','cos_full'),('Evidence','cos_e'),('EOT','cos_query'),('Value','cos_value')]}
    stats=[]
    for eps in prior['epsilons']:
        items=[i for r in results for a in r['attributes'] for i in a['interventions'] if i['epsilon']==eps]
        stats.append({'epsilon':eps,'norm_min':min(i['effective_delta_norm'] for i in items),'norm_max':max(i['effective_delta_norm'] for i in items),
          'budget_max_relative_error':max(i['effective_budget_relative_error'] for i in items),
          'prediction_median_relative_error':statistics.median(i['relative_prediction_error'] for i in items),
          'prediction_max_relative_error':max(i['relative_prediction_error'] for i in items),
          'central_median_relative_error':statistics.median(i['central_relative_error'] for i in items),
          'central_max_relative_error':max(i['central_relative_error'] for i in items),'positive_F_change':sum(i['F_change']>0 for i in items)})
    report={'caption_count':9,'attribute_count':45,'gallery_count':34054,'layer':8,'precision':'fp32','tau':.02,'T_query':T_QUERY,
      'epsilons':prior['epsilons'],'only_representation_changed':True,'read_weights_reused':True,
      'query_template':prior['query_template'],'query_normalization':'L2 once inside reused cosine reader',
      'eot_sha256':sha256(eot_path),'evidence_sha256':sha256(e_path),'full_sha256':sha256(full_path),
      'code_sha256':sha256(Path(__file__)),'reused_reader_sha256':sha256(REPO/'prototypes/attribute_unit_v3_query_20261002/query_reader.py'),
      'gallery_sha256':sha256(cache),'queries':qrecords,'results':results,'distributions':distributions,'intervention_summary':stats,
      'eot_exact_regression':True,'zero_intervention_exact':True,'parameters_frozen':True,'special_token_delta_zero':True}
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); write_summary(report,out)
    assert sha256(eot_path)==report['eot_sha256'] and sha256(e_path)==report['evidence_sha256']
    print('SHARED_OBJECT',json.dumps(distributions['shared_object']),flush=True); print('BUDGET_PREDICTION',json.dumps(stats),flush=True)
if __name__=='__main__': run()