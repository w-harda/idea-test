"""v5.1：确定性validation配对和Euclidean cached-delta诊断；无模型forward/训练。"""
import os,sys,json,hashlib,time,subprocess
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','8');os.environ.setdefault('OMP_NUM_THREADS','8')
os.environ.setdefault('XDG_CACHE_HOME','/home/lzf/ldx/cache/xdg');sys.dont_write_bytecode=True
import numpy as np
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent.parent;OUT=ROOT/'outputs'
V5=REPO/'prototypes/attribute_unit_v5_contextual_subspace_20261003';FULL=V5/'outputs/full'
V4=REPO/'prototypes/attribute_unit_v4_learned_reader_20261003'
ZFILE=REPO/'prototypes/attribute_unit_v41_low_rank_20261003/cache/frozen_v4_z.fp32.npy'
TARGETS=['upper_clothing_color','upper_clothing_type','upper_clothing_length','lower_clothing_color','lower_clothing_type']
REGIONS={'upper':['upper_clothing_color','upper_clothing_type','upper_clothing_length'],'lower':['lower_clothing_color','lower_clothing_type']}
def load(path):return json.loads(Path(path).read_text())
def save(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()
def key(record):return tuple(int(x) for x in record['row_id'].split(':')[1:])
def region(target):return REGIONS['upper' if target.startswith('upper_') else 'lower']
def verify():
    p=load(FULL/'protocol.json');a=load(FULL/'audit.json');done=load(FULL/'completed.json');old=load(FULL/'provenance.json')
    assert p['mode']=='full' and done['full_experiment'] and not p['training'] and p['features']=='pooled'
    assert (p['train_used'],p['validation_used'])==(54246,13700) and a['provenance_sha_unchanged']
    assert a['zero_intervention_exact'] and a['old_G_w_alpha_exact'] and a['parameters_frozen']
    hashes={}
    for f,expected in old.items():
        actual=digest(f);assert actual==expected,(f,'baseline hash mismatch');hashes[f]=actual
    for name,expected in p['code_sha256'].items():
        f=V5/name;actual=digest(f);assert actual==expected,(name,'v5 source mismatch');hashes[str(f)]=actual
    for n in ('protocol.json','corpus.json','provenance.json','audit.json','completed.json'):hashes[str(FULL/n)]=digest(FULL/n)
    split=load(V4/'outputs/split.json');assert p['train_indices']==split['train_indices'] and p['validation_indices']==split['validation_indices']
    val=split['validation_indices'];train=set(split['train_indices']);assert not set(val)&train
    assert not set(split['validation_pids'])&set(split['train_pids']) and not set(split['validation_pids'])&set(split['held_out_pids'])
    assert all(split['records'][i]['pid'] in split['validation_pids'] for i in val)
    return split,hashes

def select_pairs(split,target):
    slots=split['slots'];values=split['values'];s=slots.index(target);siblings=[slots.index(x) for x in region(target) if x!=target]
    null=np.array([vs.index('null') for vs in values]);records=split['records']
    eligible=sorted([i for i in split['validation_indices'] if records[i]['labels'][s]!=null[s]],key=lambda i:key(records[i]))
    # 同标签signature的candidate共享Hamming score，最早row_id即固定tie winner；不改变逐caption nearest规则。
    signatures={}
    for i in eligible:signatures.setdefault(tuple(records[i]['labels']),i)
    items=sorted(signatures.items(),key=lambda x:key(records[x[1]]));Y=np.array([x[0] for x in items],dtype=np.int16);ids=[x[1] for x in items]
    choices={};chosen={};anchor_counts={'strict':0,'relaxed':0,'no_candidate':0}
    for sig,i in items:
        aa=np.array(sig);different=Y[:,s]!=aa[s]
        strict=different.copy()
        for t in siblings:strict&=(aa[t]!=null[t])&(Y[:,t]!=null[t])&(Y[:,t]==aa[t])
        candidates=np.flatnonzero(strict);tier='strict' if len(candidates) else 'relaxed'
        if not len(candidates):candidates=np.flatnonzero(different)
        if not len(candidates):choices[sig]=None;continue
        yy=Y[candidates];distance=(yy!=aa).sum(1)-1
        if tier=='strict':best=int(candidates[np.argmin(distance)])
        else:
            rh=(yy[:,siblings]!=aa[siblings]).sum(1)
            unknown=((yy[:,siblings]==null[siblings])|(aa[siblings]==null[siblings])).sum(1)
            # lexsort最后一个key为主key，candidate已按numeric row_id升序。
            order=np.lexsort((candidates,distance,unknown,rh));best=int(candidates[order[0]])
        choices[sig]=(ids[best],tier)
    for i in eligible:
        choice=choices[tuple(records[i]['labels'])]
        if choice is None:anchor_counts['no_candidate']+=1;continue
        j,tier=choice;anchor_counts[tier]+=1
        a,b=sorted((i,j),key=lambda ix:key(records[ix]));pairkey=(a,b)
        if pairkey in chosen:chosen[pairkey]['selected_by'].append(records[i]['row_id']);continue
        ya,yb=records[a]['labels'],records[b]['labels'];diff=[slots[t] for t in range(len(slots)) if t!=s and ya[t]!=yb[t]]
        unknown=[slots[t] for t in siblings if ya[t]==null[t] or yb[t]==null[t]]
        chosen[pairkey]={'pair_id':f'{target}|{records[a]["row_id"]}|{records[b]["row_id"]}','target':target,'tier':tier,'a':a,'b':b,'row_a':records[a]['row_id'],'row_b':records[b]['row_id'],'value_a':values[s][ya[s]],'value_b':values[s][yb[s]],'selected_by':[records[i]['row_id']],'non_target_hamming':len(diff),'different_non_target':diff,'equal_non_target':[slots[t] for t in range(len(slots)) if t!=s and ya[t]==yb[t]],'both_null_non_target':[slots[t] for t in range(len(slots)) if t!=s and ya[t]==yb[t]==null[t]],'regional_hamming':sum(ya[t]!=yb[t] for t in siblings),'unknown_regional_controls':unknown,'same_pid':records[a]['pid']==records[b]['pid'],'truncated_a':records[a].get('truncated',False),'truncated_b':records[b].get('truncated',False)}
    pairs=sorted(chosen.values(),key=lambda x:(key(records[x['a']]),key(records[x['b']])))
    return {'target':target,'eligible_anchors':len(eligible),'unique_label_signatures':len(items),'anchor_counts':anchor_counts,'pairs':pairs}

def energy(delta,bases):
    delta=np.asarray(delta,dtype=np.float64);den=np.einsum('bd,bd->b',delta,delta);valid=den>0;result={}
    for name,Q in bases.items():
        Q=np.asarray(Q,dtype=np.float64);assert Q.shape[0]==delta.shape[1]
        projected=delta@Q;num=np.einsum('br,br->b',projected,projected);x=np.full(len(delta),np.nan);x[valid]=num[valid]/den[valid]
        result[name]=x
    return result,valid,den

def metrics(target,siblings,semantic,random,valid,mask=None):
    if mask is not None:valid=valid&mask
    ids=np.flatnonzero(valid)
    if not len(ids):return {'count':0}
    def one(E):
        et=E[target][ids];leak=np.maximum.reduce([E[t][ids] for t in siblings]);margin=et-leak
        stats=lambda x:{'mean':float(x.mean()),'median':float(np.median(x)),'min':float(x.min()),'max':float(x.max())}
        return {'E_target':stats(et),'E_leak':stats(leak),'margin':stats(margin),'hit_rate':float((margin>0).mean()),'sibling_hit_rates':{t:float((et>E[t][ids]).mean()) for t in siblings},'E_siblings':{t:stats(E[t][ids]) for t in siblings}}
    ss=one(semantic);rr=one(random)
    return {'count':len(ids),'semantic':ss,'random':rr,'target_gt_same_rank_random_rate':float((semantic[target][ids]>random[target][ids]).mean()),'target_minus_random':{'median':float(np.median(semantic[target][ids]-random[target][ids])),'mean':float((semantic[target][ids]-random[target][ids]).mean())}}

def pair_summary(group,tier,results):
    ps=[p for p in group['pairs'] if p['tier']==tier];rr=[results[p['pair_id']] for p in ps]
    if not ps:return {'pair_count':0,'metrics':{'count':0}}
    siblings=[t for t in region(group['target']) if t!=group['target']];target=group['target']
    se={t:np.array([x['semantic'][t] if x['valid'] else np.nan for x in rr]) for t in region(target)}
    re={t:np.array([x['random'][t] if x['valid'] else np.nan for x in rr]) for t in region(target)};valid=np.array([x['valid'] for x in rr])
    h=np.array([p['non_target_hamming'] for p in ps]);counts={str(i):int((h==i).sum()) for i in np.unique(h)}
    reuse={}
    for p in ps:
        for k in ('row_a','row_b'):reuse[p[k]]=reuse.get(p[k],0)+1
    return {'pair_count':len(ps),'undefined_zero_delta_count':int((~valid).sum()),'regional_mismatch_count':sum(p['regional_hamming']>0 for p in ps),'unknown_regional_count':sum(bool(p['unknown_regional_controls']) for p in ps),'non_target_hamming_counts':counts,'same_pid_count':sum(p['same_pid'] for p in ps),'unique_endpoint_count':len(reuse),'maximum_endpoint_reuse':max(reuse.values()),'metrics':metrics(target,siblings,se,re,valid),'zero_non_target_hamming_subset':metrics(target,siblings,se,re,valid,h==0)}

def audit_markdown(split,groups):
    slots=split['slots'];values=split['values'];records=split['records'];lines=['# v5.1 固定顺序人工可读审计','','在energy计算前固定每槽前10个strict pair；不按实验结果删除pair。strict仅保证同region sibling已知且相等，其它字段是最小Hamming；null是未知，不代表对应真实属性相同。','']
    chosen=[]
    for g in groups:
        ps=[p for p in g['pairs'] if p['tier']=='strict'][:10]
        for p in ps:
            chosen.append(p);a,b=records[p['a']],records[p['b']]
            lines += [f"## {p['pair_id']}",'',f"A ({a['pid']}): {a['caption']}",'',f"B ({b['pid']}): {b['caption']}",'',f"target: {p['value_a']} → {p['value_b']}; non-target Hamming={p['non_target_hamming']}; same PID={p['same_pid']}",'','| slot | A | B |','|---|---|---|']
            for s,slot in enumerate(slots):lines.append(f"| {slot} | {values[s][a['labels'][s]]} | {values[s][b['labels'][s]]} |")
            lines += ['',f"相同non-target：{p['equal_non_target']}",f"不同non-target：{p['different_non_target']}",f"双方null（未受控）：{p['both_null_non_target']}",'']
    (OUT/'manual_audit.md').write_text('\n'.join(lines),encoding='utf-8');save(OUT/'audit_sample.json',chosen)


def prepare():
    assert not (OUT/'pairs.json').exists(),'refuse replace frozen pairing'
    t0=time.time();split,hashes=verify();groups=[select_pairs(split,t) for t in TARGETS]
    protocol={'validation_captions':13700,'validation_pid_count':split['validation_pid_count'],'targets':TARGETS,'features':'verified existing frozen v4 raw512 pooled Z, preprojection','bases':'only v5 full Euclidean and same-rank random, no tangent','pair_rule':'one nearest different-nonnull-value partner per eligible validation caption; deduplicate unordered pairs','strict_rule':'all same-region siblings nonnull and equal; minimize other non-target Hamming; numeric row_id tie','relaxed_rule':'only anchors without strict candidate; minimize lexicographic(region Hamming, unknown region controls, total non-target Hamming, numeric row_id)','null_policy':'null is unknown; null==null not a known region control; null/non-null mismatch counts one Hamming','ties':'numeric annotation/caption row_id ascending','pair_orientation':'numeric row_a < row_b; reversing delta does not change energies','same_PID':'permitted, counted; no extra PID selection constraint','audit_rule':'first10 strict unordered pairs per target by numeric endpoints, frozen before energy','strict_H0_subset':'descriptive subgroup non-target Hamming=0, not selected by energy; null controls still unknown','margin':'target energy minus maximum sibling energy; all siblings must be beaten for hit','random':'reuse original seed42 v5 random basis for each slot; report both target energy and matched random sibling margin/hit','energy':'float64 delta of cached fp32 z_s; ||Q_t^T delta||²/||delta||²; no sum normalization; zero delta undefined','training':False,'IRRA_forward_calls':0,'retrieval_objective_calls':0,'source_sha256':{p.name:digest(p) for p in rglob_sources()},'baseline_sha256':hashes,'prepare_seconds':time.time()-t0}
    save(OUT/'protocol.json',protocol);save(OUT/'pairs.json',groups);audit_markdown(split,groups)
    save(OUT/'pairing_frozen.json',{'pairs_sha256':digest(OUT/'pairs.json'),'audit_sample_sha256':digest(OUT/'audit_sample.json'),'protocol_sha256':digest(OUT/'protocol.json'),'counts':{g['target']:{'strict':sum(p['tier']=='strict' for p in g['pairs']),'relaxed':sum(p['tier']=='relaxed' for p in g['pairs']),'anchor_counts':g['anchor_counts']} for g in groups}})
    for g in groups:print('PAIR_COUNT',g['target'],'anchors',g['eligible_anchors'],'unique strict',sum(p['tier']=='strict' for p in g['pairs']),'relaxed',sum(p['tier']=='relaxed' for p in g['pairs']),'signature_count',g['unique_label_signatures'],flush=True)
    print('PREPARED',OUT,flush=True)
def rglob_sources():return sorted(ROOT.glob('*.py'))

def measure():
    assert not (OUT/'energy.json').exists(),'refuse overwrite energy'
    t0=time.time();frozen=load(OUT/'pairing_frozen.json');assert digest(OUT/'pairs.json')==frozen['pairs_sha256'] and digest(OUT/'audit_sample.json')==frozen['audit_sample_sha256']
    protocol=load(OUT/'protocol.json');assert digest(OUT/'protocol.json')==frozen['protocol_sha256']
    for name,h in protocol['source_sha256'].items():assert digest(ROOT/name)==h,'source changed after pair freeze'
    split=load(V4/'outputs/split.json');assert digest(V4/'outputs/split.json')==protocol['baseline_sha256'][str(V4/'outputs/split.json')]
    assert digest(ZFILE)==protocol['baseline_sha256'][str(ZFILE)] and digest(FULL/'corpus.json')==protocol['baseline_sha256'][str(FULL/'corpus.json')]
    corpus={x['slot']:x for x in load(FULL/'corpus.json')};groups=load(OUT/'pairs.json');z=np.load(ZFILE,mmap_mode='r');assert z.shape==(len(split['records']),13,512) and z.dtype==np.float32
    bases={};geometry={}
    for slot in set(t for target in TARGETS for t in region(target)):
        b=corpus[slot]['geometry']['bases'];E=np.array(b['euclidean'],dtype=np.float64);R=np.array(b['random'],dtype=np.float64)
        assert E.shape==R.shape and E.shape[0]==512
        for Q in (E,R):assert np.max(np.abs(Q.T@Q-np.eye(Q.shape[1])))<1e-12
        bases[slot]=(E,R);geometry[slot]={'rank':E.shape[1],'euclidean_Gram_max_error':float(np.max(np.abs(E.T@E-np.eye(E.shape[1])))),'random_Gram_max_error':float(np.max(np.abs(R.T@R-np.eye(R.shape[1]))))}
    results={};summaries={}
    for g in groups:
        target=g['target'];s=split['slots'].index(target);ps=g['pairs'];a=[p['a'] for p in ps];b=[p['b'] for p in ps]
        delta=np.asarray(z[b,s,:],dtype=np.float64)-np.asarray(z[a,s,:],dtype=np.float64)
        semantic,valid,den=energy(delta,{t:bases[t][0] for t in region(target)});random,valid_r,_=energy(delta,{t:bases[t][1] for t in region(target)});assert np.array_equal(valid,valid_r)
        siblings=[t for t in region(target) if t!=target]
        for i,p in enumerate(ps):
            x={'pair_id':p['pair_id'],'target':target,'tier':p['tier'],'valid':bool(valid[i]),'delta_z_norm':float(den[i]**.5),'semantic':{},'random':{}}
            if valid[i]:
                for method,E in [('semantic',semantic),('random',random)]:
                    x[method]={t:float(E[t][i]) for t in region(target)};assert all(-1e-12<=e<=1+1e-12 for e in x[method].values())
                    leak=max(x[method][t] for t in siblings);x[method+'_leak']=leak;x[method+'_margin']=x[method][target]-leak;x[method+'_hit']=x[method+'_margin']>0
            else:x['undefined_reason']='identical cached target reader representations; no denominator smoothing'
            results[p['pair_id']]=x
        summaries[target]={tier:pair_summary(g,tier,results) for tier in ('strict','relaxed')}
        print('MEASURED',target,{tier:{'n':summaries[target][tier]['pair_count'],'target_median':summaries[target][tier]['metrics'].get('semantic',{}).get('E_target',{}).get('median'),'hit':summaries[target][tier]['metrics'].get('semantic',{}).get('hit_rate')} for tier in ('strict','relaxed')},flush=True)
    save(OUT/'energy.json',results);save(OUT/'metrics.json',summaries)
    assert digest(OUT/'pairs.json')==frozen['pairs_sha256']
    unchanged={f:digest(f)==h for f,h in protocol['baseline_sha256'].items()};assert all(unchanged.values())
    audit={'pair_manifest_exact_unchanged':True,'audit_sample_exact_unchanged':digest(OUT/'audit_sample.json')==frozen['audit_sample_sha256'],'baselines_exact_unchanged':all(unchanged.values()),'baseline_count':len(unchanged),'training':False,'IRRA_forward_calls':0,'retrieval_objective_calls':0,'tangent_used':False,'geometry':geometry,'seconds':time.time()-t0,'tracked_diff':subprocess.check_output(['git','diff','--stat'],cwd=REPO,text=True),'staged_diff':subprocess.check_output(['git','diff','--cached','--stat'],cwd=REPO,text=True)}
    assert audit['tracked_diff']==audit['staged_diff']=='';save(OUT/'audit.json',audit);print('MEASUREMENT_COMPLETE',OUT,flush=True)

def report():
    protocol=load(OUT/'protocol.json');groups=load(OUT/'pairs.json');met=load(OUT/'metrics.json');results=load(OUT/'energy.json');audit=load(OUT/'audit.json')
    reviews=load(OUT/'manual_review.json') if (OUT/'manual_review.json').exists() else {}
    lines=['# Stage03 v5.1 Controlled Semantic Selectivity Diagnostic','','固定已有validation 13700 captions与v4 raw512 pooled cache；只用v5 full Euclidean及其同rank random。无训练、IRRA forward、retrieval objective/intervention，未接线正式pipeline。','','## A. Pair construction','','每个eligible caption选择一个partner，数值row_id破同分、unordered去重。strict要求同region siblings全部已知且相等，其它non-target labels的Hamming最小；没有strict candidate的anchor才进入relaxed。relaxed依次最小化region Hamming、未知region字段数、non-target Hamming、numeric row_id。null相等不是已知语义控制。strict不是全caption只改变一个属性的因果反事实。','', '| target | eligible anchors | strict unique | relaxed unique | strict H0 | relaxed H0 |','|---|---:|---:|---:|---:|---:|']
    for g in groups:
        m=met[g['target']];lines.append(f"| {g['target']} | {g['eligible_anchors']} | {m['strict']['pair_count']} | {m['relaxed']['pair_count']} | {m['strict'].get('zero_non_target_hamming_subset',{}).get('count',0)} | {m['relaxed'].get('zero_non_target_hamming_subset',{}).get('count',0)} |")
    lines+=['','每槽strict固定前10组的完整原文、13槽属性及相同/不同字段见manual_audit.md；audit_sample.json与pairs.json在energy前冻结。人工review只标记，全部pair继续保留。','', '## B. Target selectivity','','数值为energy ratio，非概率，各span不正交，能量和可以超过1。hit定义target同时严格大于全部siblings；另外记录每个sibling hit率。','', '| target | tier | n | target med/mean | leak med/mean | margin med/mean | hit | random target med/mean | random hit | target > random |','|---|---|---:|---|---|---|---:|---|---:|---:|']
    fmt=lambda d:f"{d['median']:.6f}/{d['mean']:.6f}"
    for g in groups:
        for tier in ('strict','relaxed'):
            m=met[g['target']][tier]['metrics']
            if m['count']==0:continue
            s=m['semantic'];rr=m['random'];lines.append(f"| {g['target']} | {tier} | {m['count']} | {fmt(s['E_target'])} | {fmt(s['E_leak'])} | {fmt(s['margin'])} | {s['hit_rate']:.4%} | {fmt(rr['E_target'])} | {rr['hit_rate']:.4%} | {m['target_gt_same_rank_random_rate']:.4%} |")
    lines+=['','rank差异使raw hit也具有维数效应；必须联合每槽同rank target-random能量及random sibling hit解释。pairs复用endpoint，不作独立样本统计显著性结论。','','## C/D. Sibling energies与H0子组','','```json',json.dumps({t:{tier:{'E_siblings':m[tier]['metrics'].get('semantic',{}).get('E_siblings'), 'random_margin':m[tier]['metrics'].get('random',{}).get('margin'),'strict_control_details':{k:v for k,v in m[tier].items() if k!='metrics'}} for tier in ('strict','relaxed')} for t,m in met.items()},ensure_ascii=False,indent=2),'```','','## E. 人工审计标记','','标记在energy测量前录入；不按标记或结果剔除pair。未独立人工核验全部自动pair。','']
    for pid,rev in reviews.items():lines += [f"- {pid}: {rev['status']}；{rev['note']}"]
    lines+=['','## E. 固定成功/失败示例','','只从预先固定audit_sample中，按numeric pair顺序分别取首个hit/非hit；这些例子不参与pair选择或筛除。','']
    for g in groups:
        selected=[p for p in load(OUT/'audit_sample.json') if p['target']==g['target']]
        for hit in (True,False):
            candidates=[p for p in selected if results[p['pair_id']]['valid'] and results[p['pair_id']]['semantic_hit']==hit]
            if not candidates:lines.append(f"- {g['target']} {'成功' if hit else '失败'}：固定前10审计样本中无此类，不另挑pair。");continue
            p=candidates[0];x=results[p['pair_id']];lines.append(f"- {p['pair_id']} {'成功' if hit else '失败'}：E={x['semantic']}；M={x['semantic_margin']:.6f}；random E={x['random']}；人工标记={reviews.get(p['pair_id'])}")
    lines+=['','## F. 解释边界','','caption观察配对仅控制已有pseudo labels；文本措辞、未记录/为null的属性、人物身份、对象层次、可见性和reader定位均可变化。该诊断建立选择性现象，不单独证明纯因果属性方向。人工标记未改变主指标；结论须逐slot结合target/sibling/random与H0子组。','','## Provenance / source audit','','```json',json.dumps(audit,ensure_ascii=False,indent=2),'```','',f"服务器独立目录：`{ROOT}`。原pairs SHA={digest(OUT/'pairs.json')}；source SHA及29项baseline链在protocol.json。本轮无暂存/commit/push。"]
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');save(OUT/'completed.json',{'pairs':sum(len(g['pairs']) for g in groups),'audited_sample_count':len(load(OUT/'audit_sample.json')),'human_review_entries':len(reviews),'source_sha256':{p.name:digest(p) for p in rglob_sources()},'summary_sha256':digest(OUT/'summary.md'),'no_training':True,'IRRA_forward_calls':0,'retrieval_calls':0})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','measure','report'],required=True);args=p.parse_args()
    {'prepare':prepare,'measure':measure,'report':report}[args.phase]()
