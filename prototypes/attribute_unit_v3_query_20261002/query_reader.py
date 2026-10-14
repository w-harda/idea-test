"""Zero-training fixed slot/value query reader; prototype only."""
import os,sys,json,math,statistics,importlib
from pathlib import Path
from functools import partial
sys.dont_write_bytecode=True
os.environ.setdefault('TORCH_HOME','/home/lzf/ldx/cache/torch')
os.environ.setdefault('HF_HOME','/home/lzf/ldx/cache/huggingface')
REPO=Path('/home/lzf/ldx/projects/idea-TBPS-test1')
BASE=Path('/home/lzf/ldx/outputs/idea-TBPS-test1/stage03-attribute-unit-v2-20261002')
CODE=BASE/'code'
sys.path.insert(0,str(REPO/'prototypes/attribute_unit_v21_evidence_20261002'))
from evidence import ranking,rank_string
from diagnose import cosine,distribution,vector_info
from attributes.irra_attribute_unit_v2_prototype import AttributeUnitV2Probe,uniform_unit_gradient,unit_delta
from attributes.irra_token_alignment import align_irra_tokens
from attributes.irra_victim import FrozenIRRA
from attributes.ap_gallery_baseline import sha256
import torch
import yaml
T_QUERY=.1

def query_text(slot,value):
    if not isinstance(slot,str) or not isinstance(value,str) or value=='null': raise ValueError('non-null identity required')
    return slot.replace('_',' ')+' '+value.replace('_',' ')

def read_weights(H,q,indices,temperature=T_QUERY):
    if not math.isfinite(temperature) or temperature<=0: raise ValueError('positive temperature')
    H=H.detach().float(); q=q.detach().float()
    assert H.ndim==2 and q.ndim==1 and H.shape[1]==q.numel()
    ids=sorted(set(indices))
    assert ids and min(ids)>=0 and max(ids)<len(H) and torch.isfinite(H).all() and torch.isfinite(q).all()
    content=H[ids]; norms=content.norm(dim=1); qnorm=q.norm()
    if not qnorm or (norms==0).any(): raise ValueError('undefined hidden-state cosine')
    r=(content/norms[:,None])@(q/qnorm)
    alpha=torch.softmax(r/temperature,dim=0)
    w=torch.zeros(len(H),device=H.device,dtype=torch.float32)
    w[ids]=alpha/alpha.norm()
    full_alpha=torch.zeros_like(w); full_alpha[ids]=alpha
    return w,full_alpha,r

def mass_and_top(alpha,row,tokens,top_k=5):
    evidence=set(row['evidence_token_indices']); objects=set(row['object_token_indices'])
    lookup={t['token_index']:t for t in tokens}; content=set(lookup)
    assert evidence|objects<=content
    sm=lambda ids:float(alpha[sorted(ids)].double().sum()) if ids else 0.
    attended=sorted(tokens,key=lambda t:-float(alpha[t['token_index']]))[:top_k]
    top=[dict(token_index=t['token_index'],raw=t['raw'],bpe_piece=t['bpe_piece'],
              word=t.get('word'),alpha=float(alpha[t['token_index']])) for t in attended]
    return {'evidence_mass':sm(evidence),'object_mass':sm(objects),'evidence_object_overlap_mass':sm(evidence&objects),
        'object_exclusive_mass':sm(objects-evidence),'context_mass':sm(content-(evidence|objects)),
        'evidence_uniform_reference_mass':len(evidence)/len(content),
        'object_uniform_reference_mass':len(objects)/len(content),'content_token_count':len(content),
        'attention_entropy':float(-(alpha[sorted(content)].double()*alpha[sorted(content)].double().log()).sum()),
        'top5':top}

def summarize(report,out):
    f=lambda x:f'{x:.6g}'
    lines=['# Attribute Unit v3 Fixed Query Reader（2026-10-02）','',
     '零训练；固定原9条caption/45属性、frozen IRRA第8 text block、fp32、34054图clean train cache、paired soft rank τ=.02、ε=.01/.05/.1。仅新增第三种query方法，Full/Evidence旧产物原样保留。',
     '', '模板预先固定为slot key和canonical value的下划线各替换为空格，再用一个空格拼接；保留组合类别全部词，不选单个alias，不新增语义。T_q=.1预先固定，不搜索或ensemble。query单独经同一frozen IRRA，在第8 block读取contextual EOT状态q；query的最终retrieval feature不使用。',
     '', '对原caption所有保留content tokens（含标点/BPE；不含SOT/EOT/pad）计算r=cos(q,H_j)，alpha=softmax(r/.1)，w=alpha/||alpha||2，g=wᵀG，S_query=||g||2，delta=εw(g/S_query)ᵀ。H、q、alpha、w均在原caption baseline上detach并固定；一阶验证不重算reader、不对w(H)求导。delta进入原IRRA剩余forward，不构造外部retrieval feature或query攻击方向。',
     '', '## 共享对象（原18对）','', '| 方法 | mean | median | min | max |', '| --- | ---: | ---: | ---: | ---: |']
    for method,s in report['distributions']['shared_object'].items():
        lines.append(f"| {method} | {f(s['mean'])} | {f(s['median'])} | {f(s['min'])} | {f(s['max'])} |")
    lines+=['','## 等预算与一阶响应','', '| ε | 实际norm min/max | budget最大相对误差 | 预测误差median/max | 中心误差median/max | 正ΔF |', '| ---: | --- | ---: | --- | --- | ---: |']
    for s in report['intervention_summary']:
        lines.append(f"| {s['epsilon']} | {s['norm_min']:.12g} / {s['norm_max']:.12g} | {f(s['budget_max_relative_error'])} | {f(s['prediction_median_relative_error'])} / {f(s['prediction_max_relative_error'])} | {f(s['central_median_relative_error'])} / {f(s['central_max_relative_error'])} | {s['positive_F_change']}/45 |")
    lines+=['','以上误差为比例，不是百分数。理论budget严格等ε，fp32实际add存在舍入。paired cosine before/after、F变化、effective delta内积预测均保存在服务器JSON。',
     '', '## 固定query表','', '| slot=value | 唯一query文本 | EOT位置 |', '| --- | --- | ---: |']
    for q in report['queries']:
        lines.append(f"| {q['slot']}={q['canonical']} | {q['query_text']} | {q['eot_index']} |")
    lines+=['','## 所有同caption方向对照','', '| caption | left | right | full cosine | evidence cosine | query cosine | w cosine | w L2差 | shared object | identical evidence |',
      '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |']
    for r in report['results']:
        for p in r['pairs']:
            lines.append(f"| {r['row_id']} | {p['left']} | {p['right']} | {f(p['cos_full'])} | {f(p['cos_e'])} | {f(p['cos_query'])} | {f(p['weight_cosine'])} | {f(p['weight_l2_difference'])} | {p['shared_object_indices']} | {p['identical_evidence']} |")
    lines+=['','## 所有query的read pattern','',
     'mass基于概率alpha（不是L2权重w）。context为evidence/object并集之外的全部content；overlap单独存储，原45个属性的evidence与object没有交集。top5不人工过滤词、标点或subword。uniform reference仅诊断，不参与reader或score。',
     '', '| caption | query | evidence mass | object mass | context mass | top5 token:alpha |', '| --- | --- | ---: | ---: | ---: | --- |']
    for r in report['results']:
        for a in r['attributes']:
            p=a['read_pattern']; top=', '.join(f"{t['token_index']}:{t['bpe_piece']}:{t['alpha']:.4f}" for t in p['top5'])
            lines.append(f"| {r['row_id']} | {a['query_text']} | {f(p['evidence_mass'])} | {f(p['object_mass'])} | {f(p['context_mass'])} | {top} |")
    for r in report['results']:
        lines+=['',f"## {r['row_id']}",'',r['caption'],'']
        for method in ('S_full','S_e','S_query'): lines+=['排序 '+method+'：'+rank_string(r['rankings'][method]),'']
        lines+=['| slot=value | S_full | S_e | S_query |','| --- | ---: | ---: | ---: |']
        for a in r['attributes']: lines.append(f"| {a['slot']}={a['canonical']} | {f(a['S_full'])} | {f(a['S_e'])} | {f(a['S_query'])} |")
    lines+=['','## 验证与追溯','',
      'baseline F/paired cosine逐值一致；g_full/g_e与旧产物逐值一致；zero intervention逐值一致；模型参数requires_grad=False/grad=None。所有query reader及w都不训练、无梯度；原caption不修改，SOT/EOT/pad实际delta全0。',
      '', f"服务器目录：`{out.parent}`。原始JSON含所有q、alpha、w、g/d/h_unit、92对方向比较和135次干预。源码、tests、原始outputs仅服务器；Windows仅同步摘要。正式模块未改，未commit/push。",
      '', f"Full SHA：`{report['full_sha256']}`；Evidence SHA：`{report['evidence_sha256']}`；新源码SHA：`{report['code_sha256']}`；ontology SHA：`{report['ontology_sha256']}`。"]
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def run():
    root=Path(__file__).resolve().parent; out=root/'outputs'; out.mkdir(exist_ok=True)
    target=out/'query.json'
    if target.exists(): raise FileExistsError(target)
    full_path=BASE/'prototype.json'; e_path=REPO/'prototypes/attribute_unit_v21_evidence_20261002/outputs/evidence.json'
    assert sha256(full_path)=='df6b2b2032dde33fed19bf45e9ad2fbb6301676bc5514ed81303ed3e241e42fc'
    assert sha256(e_path)=='c77307003057df8f0514f85312434f1e5756f2fc1b0692b3693a60642ae91f1f'
    prior=json.loads(full_path.read_text()); ev=json.loads(e_path.read_text()); e_rows={r['row_id']:r for r in ev['results']}
    assert prior['row_ids']==[r['row_id'] for r in ev['results']] and prior['epsilons']==[.01,.05,.1]
    ontology_path=CODE/'ontology/upar_attribute_space.yaml'
    ontology=yaml.safe_load(ontology_path.read_text())
    permitted={s['key']:{v['key'] for v in s['values']} for s in ontology['slots']}
    assert len(permitted)==13
    cache=Path('/home/lzf/ldx/outputs/idea-TBPS-test1/stage03-v2-smoke-20261001/clean-train-gallery.pt')
    assert sha256(cache)==prior['gallery_cache_sha256']
    payload=torch.load(cache,map_location='cpu',weights_only=True); spec=payload['spec']
    assert spec==prior['gallery_spec'] and len(payload['features'])==34054 and str(torch.__version__)==spec['torch_version']
    irra=Path('/home/lzf/Attack/IRRA-main/IRRA-main'); checkpoint=Path('/home/lzf/Attack/IRRA/logs/CUHK-PEDES/best.pth'); config=CODE/'configs/irra_cuhk_official.yaml'
    assert sha256(checkpoint)==spec['checkpoint_sha256'] and sha256(config)==spec['config_sha256']
    for name,digest in spec['source_sha256'].items(): assert sha256(irra/name)==digest
    torch.manual_seed(42); torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False
    sys.path.insert(0,str(irra)); official=importlib.import_module('model.build')
    official.build_CLIP_from_openai_pretrained=partial(official.build_CLIP_from_openai_pretrained,download_root='/home/lzf/ldx/cache/clip')
    victim=FrozenIRRA(irra,checkpoint,config,device='cuda').eval(); victim.model.float()
    gallery=payload['features'].to('cuda'); query_cache={}; query_records=[]; results=[]
    for old in prior['results']:
        probe=AttributeUnitV2Probe(victim,gallery,old['paired_gallery_index'],tau=.02)
        # One immutable q per slot/value identity, reused across caption contexts.
        for a in old['attribute_results']:
            key=(a['slot'],a['canonical'])
            assert key[1] in permitted[key[0]] and key[1]!='null'
            if key in query_cache: continue
            text=query_text(*key); align=align_irra_tokens(text,victim.tokenizer,victim.text_length)
            assert not align['truncated'] and align['tokenizer_encode_matches_replay']
            with torch.no_grad(): _,qH,qzero=probe.encode(text,8)
            assert qH.shape[-1]==512 and qH.dtype==torch.float32 and not qzero.count_nonzero()
            q=qH[align['eot_index'],0].detach().clone()
            query_cache[key]=q
            query_records.append({'slot':key[0],'canonical':key[1],'query_text':text,'eot_index':align['eot_index'],
                'token_ids':align['token_ids'],'q_eot_hidden':q.cpu().tolist()})
        with torch.no_grad(): original=victim.encode_texts([old['caption']])
        feature,H,zero=probe.encode(old['caption'],8)
        assert H.dtype==torch.float32 and H.shape[-1]==512 and torch.equal(feature.detach(),original) and not zero.count_nonzero()
        with torch.no_grad(): zfeat,_,actual_zero=probe.encode(old['caption'],8,torch.zeros_like(H))
        assert torch.equal(zfeat,original) and not actual_zero.count_nonzero()
        F,c0=probe.objective(feature); G,=torch.autograd.grad(F,H)
        baseline=float(F); cosine0=float(c0)
        assert baseline==old['F'] and cosine0==old['paired_cosine']
        tokens=old['token_alignment']['content_tokens']; ids=[t['token_index'] for t in tokens]
        previous={a['slot']:a for a in e_rows[old['row_id']]['attributes']}; rows=[]
        for a in old['attribute_results']:
            wf,gf,_,sf=uniform_unit_gradient(G,a['support_token_indices'])
            we,ge,_,se=uniform_unit_gradient(G,a['evidence_token_indices'])
            assert sf==a['S_new'] and se==previous[a['slot']]['S_e']
            assert torch.equal(gf.cpu(),torch.tensor(a['attribute_gradient'],dtype=torch.float32))
            assert torch.equal(ge.cpu(),torch.tensor(previous[a['slot']]['g_e'],dtype=torch.float32))
            q=query_cache[(a['slot'],a['canonical'])]
            w,alpha,r=read_weights(H[:,0],q,ids)
            assert not w.requires_grad and not alpha.requires_grad and abs(float(w.double().norm())-1)<1e-6
            g=(w[:,None]*G[:,0].float()).sum(dim=0); score=float(g.double().norm()); assert score>0
            pattern=mass_and_top(alpha,a,tokens)
            assert abs(pattern['evidence_mass']+pattern['object_exclusive_mass']+pattern['context_mass']-1)<1e-6
            row={k:a[k] for k in ['slot','canonical','semantic_spans','mentions','linked_objects','evidence_token_indices','object_token_indices','support_token_indices']}
            row.update(query_text=query_text(a['slot'],a['canonical']),S_full=sf,S_e=se,S_query=score,
             g_query=g.cpu().tolist(),d_query=vector_info(g)['direction'],w=w.cpu().tolist(),alpha=alpha.cpu().tolist(),
             content_cosines=r.cpu().tolist(),weight_norm=float(w.double().norm()),
             h_query_unit=((w[:,None]*H.detach()[:,0]).sum(dim=0)).cpu().tolist(),read_pattern=pattern,interventions=[])
            for eps in prior['epsilons']:
                delta=unit_delta(w,g,eps)
                with torch.no_grad():
                    fp,_,effective=probe.encode(old['caption'],8,delta)
                    fm,_,minus=probe.encode(old['caption'],8,-delta)
                    Fp,cp=probe.objective(fp); Fm,cm=probe.objective(fm)
                outside=effective.clone(); outside[ids]=0
                assert not outside.count_nonzero()
                norm=float(effective.norm()); pred=eps*score; change=float(Fp)-baseline; neg=float(Fm)-baseline
                record={'epsilon':eps,'constructed_delta_norm':float(delta.double().norm()),'effective_delta_norm':norm,
                 'effective_budget_relative_error':abs(norm-eps)/eps,'minus_effective_delta_norm':float(minus.norm()),
                 'F_before':baseline,'F_after':float(Fp),'F_change':change,'F_minus_change':neg,
                 'first_order_prediction':pred,'first_order_effective_prediction':float((G.double()*effective).sum()),
                 'relative_prediction_error':abs(change-pred)/pred,'central_change':(change-neg)/2,
                 'central_relative_error':abs((change-neg)/2-pred)/pred,
                 'paired_cosine_before':cosine0,'paired_cosine_after':float(cp),'paired_cosine_minus':float(cm)}
                assert record['effective_budget_relative_error']<1e-4
                row['interventions'].append(record)
            rows.append(row)
        lookup={a['slot']:a for a in rows}; pairs=[]
        for p in e_rows[old['row_id']]['pairs']:
            l=lookup[p['left']]; r=lookup[p['right']]
            gl=torch.tensor(l['g_query'],dtype=torch.float64); gr=torch.tensor(r['g_query'],dtype=torch.float64)
            wl=torch.tensor(l['w'],dtype=torch.float64); wr=torch.tensor(r['w'],dtype=torch.float64)
            pairs.append({**p,'cos_query':cosine(gl,gr),'weight_cosine':cosine(wl,wr),
                'weight_l2_difference':float((wl-wr).norm()),'g_l2_difference':float((gl-gr).norm()),
                'g_relative_difference':float((gl-gr).norm()/gl.norm()),'S_query_difference':r['S_query']-l['S_query'],
                'weights_exactly_equal':torch.equal(wl,wr),'gradients_exactly_equal':torch.equal(gl,gr)})
        probe.assert_frozen()
        results.append({'row_id':old['row_id'],'caption':old['caption'],'F':baseline,'paired_cosine':cosine0,
          'attributes':rows,'pairs':pairs,'provenance':old['provenance'],'token_alignment':old['token_alignment'],
          'rankings':{method:ranking(rows,method) for method in ('S_full','S_e','S_query')}})
        print(old['row_id'],'query ranking:',rank_string(results[-1]['rankings']['S_query']),flush=True)
        for p in pairs:
            if p['identical_evidence']: print('COLLISION',old['row_id'],p,flush=True)
    assert len(results)==9 and sum(len(r['attributes']) for r in results)==45
    assert len({q['query_text'] for q in query_records})==len(query_records)
    pairs=[p for r in results for p in r['pairs']]; distributions={}
    for name,selected in [('all',pairs),('shared_object',[p for p in pairs if p['shared_object_indices']]),('identical_evidence',[p for p in pairs if p['identical_evidence']])]:
        distributions[name]={method:distribution([p[key] for p in selected]) for method,key in [('full','cos_full'),('evidence','cos_e'),('query','cos_query')]}
    stats=[]
    for eps in prior['epsilons']:
        items=[i for r in results for a in r['attributes'] for i in a['interventions'] if i['epsilon']==eps]
        stats.append({'epsilon':eps,'norm_min':min(i['effective_delta_norm'] for i in items),'norm_max':max(i['effective_delta_norm'] for i in items),
         'budget_max_relative_error':max(i['effective_budget_relative_error'] for i in items),
         'prediction_median_relative_error':statistics.median(i['relative_prediction_error'] for i in items),
         'prediction_max_relative_error':max(i['relative_prediction_error'] for i in items),
         'central_median_relative_error':statistics.median(i['central_relative_error'] for i in items),
         'central_max_relative_error':max(i['central_relative_error'] for i in items),'positive_F_change':sum(i['F_change']>0 for i in items)})
    report={'query_count':9,'attribute_count':45,'gallery_count':34054,'layer':8,'tau':.02,'precision':'fp32','T_query':T_QUERY,
     'query_template':'slot.replace(_,space) + space + canonical.replace(_,space), fixed before running',
     'reader_frozen_at_baseline':True,'query_used_as_retrieval_feature':False,'learned_parameters':0,
     'epsilons':prior['epsilons'],'full_sha256':sha256(full_path),'evidence_sha256':sha256(e_path),
     'code_sha256':sha256(Path(__file__)),'ontology_sha256':sha256(ontology_path),'gallery_sha256':sha256(cache),
     'queries':query_records,'results':results,'distributions':distributions,'intervention_summary':stats,
     'zero_intervention_exact':True,'parameters_frozen':True,'special_token_delta_zero':True}
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); summarize(report,out)
    assert sha256(full_path)==report['full_sha256'] and sha256(e_path)==report['evidence_sha256']
    print('SHARED_OBJECT',json.dumps(distributions['shared_object']),flush=True); print('BUDGET_PREDICTION',json.dumps(stats),flush=True)
if __name__=='__main__': run()