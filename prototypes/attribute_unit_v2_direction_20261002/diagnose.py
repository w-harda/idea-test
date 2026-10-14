"""Attribute Unit v2 direction diagnostics; formulas and sample fixed."""
import os,sys,json,math,itertools,statistics,importlib,hashlib
from pathlib import Path
from functools import partial
sys.dont_write_bytecode=True
os.environ.setdefault('TORCH_HOME','/home/lzf/ldx/cache/torch')
os.environ.setdefault('HF_HOME','/home/lzf/ldx/cache/huggingface')
BASE=Path('/home/lzf/ldx/outputs/idea-TBPS-test1/stage03-attribute-unit-v2-20261002')
CODE=BASE/'code'
sys.path.insert(0,str(CODE/'src'))
import torch
from attributes.irra_attribute_unit_v2_prototype import AttributeUnitV2Probe,uniform_unit_gradient
from attributes.irra_victim import FrozenIRRA
from attributes.ap_gallery_baseline import sha256

def vector_info(g):
    g=g.detach().double().cpu()
    norm=float(g.norm())
    return {'gradient':g.tolist(),'norm':norm,'direction':(g/norm).tolist() if norm else None}

def cosine(a,b):
    a=torch.as_tensor(a,dtype=torch.float64); b=torch.as_tensor(b,dtype=torch.float64)
    if not float(a.norm()) or not float(b.norm()): return None
    return float(torch.dot(a,b)/(a.norm()*b.norm()))

def component(gradient,indices):
    if not indices: return None
    _,g,_,_=uniform_unit_gradient(gradient,indices)
    return vector_info(g)

def decomposition(gradient,row):
    e=set(row['evidence_token_indices']); o=set(row['object_token_indices']); f=e|o
    assert sorted(f)==row['support_token_indices']
    ev,ob,full=component(gradient,e),component(gradient,o),component(gradient,f)
    return {'evidence_indices':sorted(e),'object_indices':sorted(o),'full_indices':sorted(f),
        'overlap_indices':sorted(e&o),'evidence':ev,'object':ob,'full':full,
        'cos_evidence_object':cosine(ev['gradient'],ob['gradient']) if ob else None,
        'cos_full_evidence':cosine(full['gradient'],ev['gradient']),
        'cos_full_object':cosine(full['gradient'],ob['gradient']) if ob else None}

def word_map(caption,alignment):
    groups={}
    for t in alignment['content_tokens']:
        groups.setdefault(t['word_index'],[]).append(t)
    return {key:{'word':caption[ts[0]['word_start']:ts[0]['word_end']].lower(),
                'indices':[t['token_index'] for t in ts],'tokens':ts} for key,ts in groups.items()}

def attribute_pairs(rows,words):
    output=[]
    for a,b in itertools.combinations(rows,2):
        # Type provenance names the object itself; color/length binds to it.
        anchors_a=set(a['object_token_indices']) | (set(a['evidence_token_indices']) if a['slot'].endswith('_type') else set())
        anchors_b=set(b['object_token_indices']) | (set(b['evidence_token_indices']) if b['slot'].endswith('_type') else set())
        shared_object=sorted(anchors_a&anchors_b)
        ea=set(a['evidence_token_indices']); eb=set(b['evidence_token_indices'])
        wa=[(k,w) for k,w in words.items() if ea&set(w['indices'])]
        wb=[(k,w) for k,w in words.items() if eb&set(w['indices'])]
        same=[{'word':x['word'],'left_word_index':i,'right_word_index':j,'same_occurrence':i==j}
              for i,x in wa for j,y in wb if x['word']==y['word'] and x['word'].isalpha()]
        output.append({'left':a['slot'],'right':b['slot'],'left_value':a['canonical'],'right_value':b['canonical'],
            'cosine':cosine(a['diagnostic']['full']['gradient'],b['diagnostic']['full']['gradient']),
            'support_overlap':sorted(set(a['support_token_indices'])&set(b['support_token_indices'])),
            'shared_object_indices':shared_object,'same_surface_evidence_words':same})
    return output

def distribution(values):
    vals=[v for v in values if v is not None]
    if not vals: return {'n':0,'undefined':len(values)}
    return {'n':len(vals),'undefined':len(values)-len(vals),'mean':statistics.mean(vals),
            'median':statistics.median(vals),'min':min(vals),'max':max(vals),
            'sorted_values':sorted(vals)}

def write_summary(report,out):
    fmt=lambda v:'undefined' if v is None else f'{v:.6f}'
    lines=['# Attribute Unit v2 方向诊断（2026-10-02）','',
      '固定第8 block、fp32、9条原caption/45属性、34054 clean train gallery、paired soft rank τ=.02。公式、support、样本不变；每句重取一次baseline G，不运行新扰动。',
      '', 'g_evidence/g_object各自采用1/sqrt(k)聚合；g_full仍为原二值并集的v2定义。重叠token在full中只计一次。cosine用float64审计，不设阈值，不做显著性检验。',
      '', 'shared-object按原文对象token位置判断：color/length的recorded linked_object，与另一属性的linked_object或type自身evidence有交集。仅同槽组/同对象词字符串不算共享对象。non-overlap指完整support token集合不相交。类别可能交叉。',
      '', '## 分布', '', '| 类别 | n | 均值 | 中位数 | 最小 | 最大 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,s in report['distributions'].items():
        lines.append(f"| {name} | {s['n']} | {fmt(s.get('mean'))} | {fmt(s.get('median'))} | {fmt(s.get('min'))} | {fmt(s.get('max'))} |")
    lines+=['','## 所有同caption属性对','', '| caption | left | right | cosine | shared object | support overlap | same surface evidence |', '| --- | --- | --- | ---: | --- | --- | --- |']
    for r in report['results']:
        for p in r['pairs']:
            same=', '.join(f"{x['word']}({x['left_word_index']},{x['right_word_index']})" for x in p['same_surface_evidence_words'])
            lines.append(f"| {r['row_id']} | {p['left']} | {p['right']} | {fmt(p['cosine'])} | {p['shared_object_indices']} | {p['support_overlap']} | {same} |")
    lines+=['','## Evidence / object 拆分','',
     '| caption | slot | evidence / object / full tokens | norm evidence | norm object | norm full | cos e,o | cos full,e | cos full,o |',
     '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in report['results']:
        for a in r['attributes']:
            d=a['diagnostic']
            if d['object'] is None: continue
            lines.append(f"| {r['row_id']} | {a['slot']} | {d['evidence_indices']} / {d['object_indices']} / {d['full_indices']} | {fmt(d['evidence']['norm'])} | {fmt(d['object']['norm'])} | {fmt(d['full']['norm'])} | {fmt(d['cos_evidence_object'])} | {fmt(d['cos_full_evidence'])} | {fmt(d['cos_full_object'])} |")
    lines+=['','无linked_object的属性只保留evidence，不补造对象；type通过自身evidence参与shared-object配对。',
     '', '## 同一 lexical regex word 的 BPE pairs','',
     '仅官方tokenizer同word_index且原文为字母词的多个BPE pieces；不把t-shirt的t、连字符、shirt三个regex单元合成一个词。',
     '', '| caption | word | token pair | piece pair | norm left/right | cosine |', '| --- | --- | --- | --- | --- | ---: |']
    for r in report['results']:
        for p in r['bpe_pairs']:
            lines.append(f"| {r['row_id']} | {p['word']} | {p['indices']} | {p['pieces']} | {fmt(p['norms'][0])}/{fmt(p['norms'][1])} | {fmt(p['cosine'])} |")
    lines+=['','## 验证与产物','',
       'baseline F、paired cosine、属性/provenance/support、所有45个聚合g与旧v2产物一致；zero intervention逐值一致，IRRA参数冻结无梯度。原始JSON保存所有归一化方向d、各分量g和每个content token的G。',
       '',f"服务器目录：`{out}`。代码、测试和原始outputs均只在服务器。未修改正式Stage1–3/Top-K/generator；未commit/push。",
       '', f"旧v2结果SHA：`{report['reference_sha256']}`；本次code SHA：`{report['code_sha256']}`。"]
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def run():
    root=Path(__file__).resolve().parent
    out=root/'outputs'; out.mkdir(exist_ok=True)
    target=out/'diagnostics.json'
    if target.exists(): raise FileExistsError(target)
    ref=BASE/'prototype.json'
    assert sha256(ref)=='df6b2b2032dde33fed19bf45e9ad2fbb6301676bc5514ed81303ed3e241e42fc'
    prior=json.loads(ref.read_text())
    cache=Path('/home/lzf/ldx/outputs/idea-TBPS-test1/stage03-v2-smoke-20261001/clean-train-gallery.pt')
    assert sha256(cache)==prior['gallery_cache_sha256']
    payload=torch.load(cache,map_location='cpu',weights_only=True)
    assert payload['spec']==prior['gallery_spec'] and len(payload['features'])==34054
    irra=Path('/home/lzf/Attack/IRRA-main/IRRA-main')
    checkpoint=Path('/home/lzf/Attack/IRRA/logs/CUHK-PEDES/best.pth')
    config=CODE/'configs/irra_cuhk_official.yaml'
    assert sha256(checkpoint)==payload['spec']['checkpoint_sha256']
    assert sha256(config)==payload['spec']['config_sha256']
    for name,digest in payload['spec']['source_sha256'].items(): assert sha256(irra/name)==digest
    torch.manual_seed(42); torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False
    sys.path.insert(0,str(irra))
    official=importlib.import_module('model.build')
    official.build_CLIP_from_openai_pretrained=partial(official.build_CLIP_from_openai_pretrained,download_root='/home/lzf/ldx/cache/clip')
    victim=FrozenIRRA(irra,checkpoint,config,device='cuda').eval(); victim.model.float()
    gallery=payload['features'].to('cuda')
    results=[]
    for old in prior['results']:
        probe=AttributeUnitV2Probe(victim,gallery,old['paired_gallery_index'],tau=.02)
        with torch.no_grad(): baseline_feature=victim.encode_texts([old['caption']])
        feature,hidden,zero=probe.encode(old['caption'],8)
        assert hidden.dtype==torch.float32 and torch.equal(feature.detach(),baseline_feature) and not zero.count_nonzero()
        F,paired=probe.objective(feature)
        assert float(F)==old['F'] and float(paired)==old['paired_cosine']
        gradient,=torch.autograd.grad(F,hidden)
        rows=[]
        for a in old['attribute_results']:
            if a['support_status']!='valid': continue
            d=decomposition(gradient,a)
            saved=torch.tensor(a['attribute_gradient'],dtype=torch.float32)
            current=torch.tensor(d['full']['gradient'],dtype=torch.float32)
            assert torch.equal(current,saved)
            assert math.isclose(d['full']['norm'],a['S_new'],rel_tol=1e-12,abs_tol=1e-12)
            rows.append({k:a[k] for k in ['slot','canonical','evidence_token_indices','object_token_indices','support_token_indices','linked_objects','semantic_spans']})
            rows[-1]['diagnostic']=d
        words=word_map(old['caption'],old['token_alignment'])
        pairs=attribute_pairs(rows,words)
        bpes=[]
        for w in words.values():
            if not w['word'].isalpha() or len(w['tokens'])<2: continue
            for l,r in itertools.combinations(w['tokens'],2):
                gl=gradient[l['token_index'],0]; gr=gradient[r['token_index'],0]
                bpes.append({'word':w['word'],'indices':[l['token_index'],r['token_index']],
                 'pieces':[l['bpe_piece'],r['bpe_piece']],'norms':[float(gl.double().norm()),float(gr.double().norm())],
                 'cosine':cosine(gl.cpu(),gr.cpu())})
        probe.assert_frozen()
        results.append({'row_id':old['row_id'],'caption':old['caption'],'F':float(F),
          'paired_cosine':float(paired),'attributes':rows,'pairs':pairs,'bpe_pairs':bpes,
          'token_alignment':old['token_alignment'],'content_token_gradients':[
            dict(token_index=t['token_index'],**vector_info(gradient[t['token_index'],0]))
            for t in old['token_alignment']['content_tokens']]})
        print(old['row_id'],'pairs',len(pairs),'bpe_pairs',len(bpes),flush=True)
    assert len(results)==9 and sum(len(r['attributes']) for r in results)==45
    allpairs=[p for r in results for p in r['pairs']]
    splits=[a['diagnostic'] for r in results for a in r['attributes'] if a['diagnostic']['object']]
    distributions={
      'all_same_caption':distribution([p['cosine'] for p in allpairs]),
      'shared_object':distribution([p['cosine'] for p in allpairs if p['shared_object_indices']]),
      'non_overlap_support':distribution([p['cosine'] for p in allpairs if not p['support_overlap']]),
      'same_surface_different_slot':distribution([p['cosine'] for p in allpairs if p['same_surface_evidence_words']]),
      'same_surface_distinct_occurrences':distribution([p['cosine'] for p in allpairs if any(not x['same_occurrence'] for x in p['same_surface_evidence_words'])]),
      'same_word_bpe_pairs':distribution([p['cosine'] for r in results for p in r['bpe_pairs']]),
      'cos_evidence_object':distribution([d['cos_evidence_object'] for d in splits]),
      'cos_full_evidence':distribution([d['cos_full_evidence'] for d in splits]),
      'cos_full_object':distribution([d['cos_full_object'] for d in splits])}
    report={'layer':8,'precision':'fp32','tau':.02,'gallery_count':34054,'query_count':9,'attribute_count':45,
        'reference_sha256':sha256(ref),'gallery_sha256':sha256(cache),'code_sha256':sha256(Path(__file__)),
        'formula_unchanged':True,'zero_exact':True,'parameters_frozen':True,'results':results,'distributions':distributions}
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    write_summary(report,out)
    print(json.dumps(distributions,ensure_ascii=False),flush=True)
if __name__=='__main__': run()