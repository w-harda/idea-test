"""Evidence-only v2.1 diagnostic; fixed v2 samples, objective and budget."""
import os,sys,json,math,itertools,importlib,statistics
from pathlib import Path
from functools import partial
sys.dont_write_bytecode=True
os.environ.setdefault('TORCH_HOME','/home/lzf/ldx/cache/torch')
os.environ.setdefault('HF_HOME','/home/lzf/ldx/cache/huggingface')
REPO=Path('/home/lzf/ldx/projects/idea-TBPS-test1')
BASE=Path('/home/lzf/ldx/outputs/idea-TBPS-test1/stage03-attribute-unit-v2-20261002')
CODE=BASE/'code'
sys.path.insert(0,str(REPO/'prototypes/attribute_unit_v2_direction_20261002'))
from diagnose import cosine,distribution,vector_info
from attributes.irra_attribute_unit_v2_prototype import AttributeUnitV2Probe,uniform_unit_gradient,unit_delta
from attributes.irra_victim import FrozenIRRA
from attributes.ap_gallery_baseline import sha256
import torch

def evidence_unit(gradient,row):
    # Use recorded evidence unchanged, not full minus object: overlap remains evidence.
    indices=row['evidence_token_indices']
    return uniform_unit_gradient(gradient,indices)

def ranking(rows,field):
    ordered=sorted(rows,key=lambda a:-a[field])
    groups=[]
    for a in ordered:
        if groups and groups[-1]['score']==a[field]: groups[-1]['slots'].append(a['slot'])
        else: groups.append({'score':a[field],'slots':[a['slot']]})
    return groups

def rank_string(groups):
    return ' > '.join(' = '.join(g['slots']) for g in groups)

def summary(report,out):
    fmt=lambda x:f'{x:.6g}'
    lines=['# Attribute Unit v2.1 Evidence-only 对照（2026-10-02）','',
      '复用原9条caption/45属性、frozen IRRA block8、fp32、34054图固定clean train gallery、paired soft rank τ=.02、ε=.01/.05/.1。保留旧Full-support JSON原样，仅新建独立Evidence-only验证。',
      '', '原始caption及Stage01 provenance/binding/ontology不改。support_e使用已有evidence token，w_e=1/sqrt(k_e)，g_e=w_eᵀG，S_e=||g_e||2，ΔH_e=εw_e(g_e/S_e)ᵀ。linked_object仍作为解释和绑定记录保留；即使不注入object位置，contextual evidence仍由完整原文编码产生。',
      '', '## Shared-object cosine（同18对）','', '| 方法 | n | mean | median | min | max |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for method in ('full','evidence'):
        s=report['pair_distributions']['shared_object'][method]
        lines.append(f"| {method} | {s['n']} | {fmt(s['mean'])} | {fmt(s['median'])} | {fmt(s['min'])} | {fmt(s['max'])} |")
    lines+=['', 'shared-object分类完全复用上一轮原文对象位置定义，未按结果重新分组、未设cosine阈值。',
      '', '## 等预算和一阶响应','', '| ε | 实际范数min/max | budget最大相对误差 | 预测误差median/max | 中心差分误差median/max | 正ΔF |',
      '| ---: | --- | ---: | --- | --- | ---: |']
    for s in report['intervention_summary']:
        lines.append(f"| {s['epsilon']} | {s['norm_min']:.12g} / {s['norm_max']:.12g} | {s['budget_max_relative_error']:.6g} | {s['prediction_median_relative_error']:.6g} / {s['prediction_max_relative_error']:.6g} | {s['central_median_relative_error']:.6g} / {s['central_max_relative_error']:.6g} | {s['positive_F_change']}/45 |")
    lines+=['', '理论预算等于ε，实际fp32加法有舍入；以上误差为比例，非百分数。逐次paired cosine before/after、F before/after及实际delta内积预测全部保存在服务器JSON。',
      '', '## 所有同caption pair 对照','', '| caption | left | right | full cosine | evidence cosine | shared object | identical evidence |',
      '| --- | --- | --- | ---: | ---: | --- | --- |']
    for r in report['results']:
        for p in r['pairs']:
            lines.append(f"| {r['row_id']} | {p['left']} | {p['right']} | {fmt(p['cos_full'])} | {fmt(p['cos_e'])} | {p['shared_object_indices']} | {p['identical_evidence']} |")
    for r in report['results']:
        lines+=['', f"## {r['row_id']}",'',r['caption'],'',
          'Full排序：'+rank_string(r['rankings']['S_full']),'',
          'Evidence-only排序：'+rank_string(r['rankings']['S_e']),'',
          '| slot=value | full support | evidence support | k_e | S_full | S_e |',
          '| --- | --- | --- | ---: | ---: | ---: |']
        for a in r['attributes']:
            lines.append(f"| {a['slot']}={a['canonical']} | {a['full_support']} | {a['evidence_support']} | {a['k_e']} | {fmt(a['S_full'])} | {fmt(a['S_e'])} |")
    lines+=['', '## 验证与追溯','',
       '同精度zero intervention与原encoder逐值一致，参数requires_grad=False/grad=None。baseline F和paired cosine逐值回归一致；所有g_full与旧v2逐值一致，所有g_e与上一轮诊断逐值一致。28个无linked_object属性的全部干预与旧Full baseline逐字段逐值回归一致。',
       '', '同evidence的slot保持完全相同的g_e、u_e与delta，明确保留alias/tie，不按slot名人为打破。',
       '', f"服务器目录：`{out.parent}`。新代码/tests和原始outputs仅在服务器；Windows仅同步本文档。未修改正式Stage1–3/Top-K/generator、未训练模块、未搜索layer/τ/权重，未commit/push。",
       '', f"旧v2结果SHA：`{report['reference_sha256']}`；旧方向诊断SHA：`{report['diagnostic_sha256']}`；新源码SHA：`{report['code_sha256']}`。"]
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def run():
    root=Path(__file__).resolve().parent; out=root/'outputs'; out.mkdir(exist_ok=True)
    output=out/'evidence.json'
    if output.exists(): raise FileExistsError(output)
    ref=BASE/'prototype.json'; diag_path=REPO/'prototypes/attribute_unit_v2_direction_20261002/outputs/diagnostics.json'
    assert sha256(ref)=='df6b2b2032dde33fed19bf45e9ad2fbb6301676bc5514ed81303ed3e241e42fc'
    assert sha256(diag_path)=='7b658629df2cf0bb990084b13d78cbdda7dc6f351da5172d616f81f4ebc697f3'
    prior=json.loads(ref.read_text()); diag=json.loads(diag_path.read_text())
    by_id={r['row_id']:r for r in diag['results']}
    assert prior['row_ids']==[r['row_id'] for r in diag['results']] and prior['epsilons']==[.01,.05,.1]
    cache=Path('/home/lzf/ldx/outputs/idea-TBPS-test1/stage03-v2-smoke-20261001/clean-train-gallery.pt')
    assert sha256(cache)==prior['gallery_cache_sha256']
    payload=torch.load(cache,map_location='cpu',weights_only=True)
    assert payload['spec']==prior['gallery_spec'] and len(payload['features'])==34054
    spec=payload['spec']; irra=Path('/home/lzf/Attack/IRRA-main/IRRA-main')
    checkpoint=Path('/home/lzf/Attack/IRRA/logs/CUHK-PEDES/best.pth'); config=CODE/'configs/irra_cuhk_official.yaml'
    assert sha256(checkpoint)==spec['checkpoint_sha256'] and sha256(config)==spec['config_sha256']
    assert str(torch.__version__)==spec['torch_version']
    for name,digest in spec['source_sha256'].items(): assert sha256(irra/name)==digest
    torch.manual_seed(42); torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False
    sys.path.insert(0,str(irra)); official=importlib.import_module('model.build')
    official.build_CLIP_from_openai_pretrained=partial(official.build_CLIP_from_openai_pretrained,download_root='/home/lzf/ldx/cache/clip')
    victim=FrozenIRRA(irra,checkpoint,config,device='cuda').eval(); victim.model.float()
    gallery=payload['features'].to('cuda'); results=[]; identical_regression=0
    for old in prior['results']:
        probe=AttributeUnitV2Probe(victim,gallery,old['paired_gallery_index'],tau=.02)
        with torch.no_grad(): original=victim.encode_texts([old['caption']])
        feature,H,zero=probe.encode(old['caption'],8)
        assert torch.equal(feature.detach(),original) and not zero.count_nonzero() and H.dtype==torch.float32
        with torch.no_grad(): fzero,_,z=probe.encode(old['caption'],8,torch.zeros_like(H))
        assert torch.equal(fzero,original) and not z.count_nonzero()
        F,cos0=probe.objective(feature); G,=torch.autograd.grad(F,H)
        baseline=float(F); cosine0=float(cos0)
        assert baseline==old['F'] and cosine0==old['paired_cosine']
        previous={a['slot']:a for a in by_id[old['row_id']]['attributes']}; rows=[]
        for a in old['attribute_results']:
            assert a['support_status']=='valid'
            wf,gf,_,sf=uniform_unit_gradient(G,a['support_token_indices'])
            assert torch.equal(gf.cpu(),torch.tensor(a['attribute_gradient'],dtype=torch.float32)) and sf==a['S_new']
            we,ge,_,se=evidence_unit(G,a)
            assert torch.equal(ge.cpu(),torch.tensor(previous[a['slot']]['diagnostic']['evidence']['gradient'],dtype=torch.float32))
            assert se>0
            row={k:a[k] for k in ['slot','canonical','linked_objects','semantic_spans','mentions']}
            row.update(full_support=a['support_token_indices'],evidence_support=a['evidence_token_indices'],
                object_token_indices=a['object_token_indices'],k_e=len(a['evidence_token_indices']),S_full=sf,S_e=se,
                g_full=gf.cpu().tolist(),g_e=ge.cpu().tolist(),d_e=vector_info(ge)['direction'],
                h_e=((we[:,None]*H.detach()[:,0]).sum(dim=0)).cpu().tolist(),interventions=[])
            for prev in a['interventions']:
                eps=prev['epsilon']; delta=unit_delta(we,ge,eps)
                with torch.no_grad():
                    fp,_,actual=probe.encode(old['caption'],8,delta)
                    fm,_,actual_minus=probe.encode(old['caption'],8,-delta)
                    Fp,cp=probe.objective(fp); Fm,cm=probe.objective(fm)
                outside=actual.clone(); outside[a['evidence_token_indices']]=0
                assert not outside.count_nonzero()
                norm=float(actual.norm()); pred=eps*se; change=float(Fp)-baseline; neg=float(Fm)-baseline
                r={'epsilon':eps,'constructed_delta_norm':float(delta.double().norm()),'effective_delta_norm':norm,
                  'effective_budget_relative_error':abs(norm-eps)/eps,'minus_effective_delta_norm':float(actual_minus.norm()),
                  'F_before':baseline,'F_after':float(Fp),'F_change':change,'F_minus_change':neg,
                  'first_order_prediction':pred,'first_order_effective_prediction':float((G.double()*actual).sum()),
                  'relative_prediction_error':abs(change-pred)/pred,'central_change':(change-neg)/2,
                  'central_relative_error':abs((change-neg)/2-pred)/pred,
                  'paired_cosine_before':cosine0,'paired_cosine_after':float(cp),'paired_cosine_minus':float(cm)}
                assert r['effective_budget_relative_error']<1e-4
                if a['evidence_token_indices']==a['support_token_indices']:
                    for key,value in r.items(): assert value==prev[key],(old['row_id'],a['slot'],key,value,prev[key])
                row['interventions'].append(r)
            if a['evidence_token_indices']==a['support_token_indices']: identical_regression+=1
            rows.append(row)
        lookup={a['slot']:a for a in rows}; pairs=[]
        for p in by_id[old['row_id']]['pairs']:
            l=lookup[p['left']]; r=lookup[p['right']]
            cf=cosine(l['g_full'],r['g_full']); ce=cosine(l['g_e'],r['g_e'])
            assert cf==p['cosine']
            identical=l['evidence_support']==r['evidence_support']
            if identical:
                assert l['g_e']==r['g_e'] and l['d_e']==r['d_e'] and l['interventions']==r['interventions']
            pairs.append({**p,'cos_full':cf,'cos_e':ce,'identical_evidence':identical,
                'evidence_overlap':sorted(set(l['evidence_support'])&set(r['evidence_support']))})
        probe.assert_frozen()
        results.append({'row_id':old['row_id'],'caption':old['caption'],'F':baseline,'paired_cosine':cosine0,
          'provenance':old['provenance'],'original_attributes':old['attributes'],'attributes':rows,'pairs':pairs,
          'rankings':{field:ranking(rows,field) for field in ('S_full','S_e')}})
        print(old['row_id'],'full:',rank_string(results[-1]['rankings']['S_full']),'evidence:',rank_string(results[-1]['rankings']['S_e']),flush=True)
    assert len(results)==9 and sum(len(r['attributes']) for r in results)==45 and identical_regression==28
    pairs=[p for r in results for p in r['pairs']]
    distributions={}
    for name,selected in [('all',pairs),('shared_object',[p for p in pairs if p['shared_object_indices']]),
          ('same_surface',[p for p in pairs if p['same_surface_evidence_words']]),('identical_evidence',[p for p in pairs if p['identical_evidence']])]:
        distributions[name]={method:distribution([p[key] for p in selected]) for method,key in [('full','cos_full'),('evidence','cos_e')]}
    stats=[]
    for eps in prior['epsilons']:
        items=[i for r in results for a in r['attributes'] for i in a['interventions'] if i['epsilon']==eps]
        stats.append({'epsilon':eps,'norm_min':min(i['effective_delta_norm'] for i in items),'norm_max':max(i['effective_delta_norm'] for i in items),
          'budget_max_relative_error':max(i['effective_budget_relative_error'] for i in items),
          'prediction_median_relative_error':statistics.median(i['relative_prediction_error'] for i in items),
          'prediction_max_relative_error':max(i['relative_prediction_error'] for i in items),
          'central_median_relative_error':statistics.median(i['central_relative_error'] for i in items),
          'central_max_relative_error':max(i['central_relative_error'] for i in items),
          'positive_F_change':sum(i['F_change']>0 for i in items)})
    report={'query_count':9,'attribute_count':45,'gallery_count':34054,'layer':8,'tau':.02,'precision':'fp32','epsilons':prior['epsilons'],
       'reference_sha256':sha256(ref),'diagnostic_sha256':sha256(diag_path),'gallery_sha256':sha256(cache),
       'code_sha256':sha256(Path(__file__)),'zero_exact':True,'parameters_frozen':True,'no_object_intervention':True,
       'unchanged_support_regression_count':identical_regression,'pair_distributions':distributions,'intervention_summary':stats,'results':results}
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); summary(report,out)
    assert sha256(ref)==report['reference_sha256'] and sha256(diag_path)==report['diagnostic_sha256']
    print('PAIR_DISTRIBUTIONS',json.dumps(distributions),flush=True); print('INTERVENTION_SUMMARY',json.dumps(stats),flush=True)
if __name__=='__main__': run()