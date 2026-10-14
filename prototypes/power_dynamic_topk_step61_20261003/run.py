"""固定25句纯selection诊断，不导入torch/IRRA/reader/scorer。"""
import sys,json,math,hashlib,statistics,collections,subprocess,time
from pathlib import Path
from selection import GAMMAS,BETA,K_MAX,candidates_from_record,select
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def load(p):return json.loads(Path(p).read_text())
def stats(xs):return {'mean':statistics.mean(xs) if xs else None,'median':statistics.median(xs) if xs else None,'min':min(xs) if xs else None,'max':max(xs) if xs else None}
def ids(xs):return [x['instance_id'] for x in xs]


def summarize(payloads,source):
    by_gamma={str(g):[x for x in payloads if x['gamma']==g] for g in GAMMAS};baseline={x['caption_id']:x for x in by_gamma['1.0']};by_source={x['row_id']:x for x in source};out={}
    for gamma,xs in by_gamma.items():
        eligible=[x for x in xs if x['candidate_count']>0];multi=[x for x in xs if x['candidate_count']>=2]
        natural=[min(math.ceil(x['N_eff']),x['candidate_count']) if x['candidate_count'] else 0 for x in xs]
        change=lambda pred:[x['caption_id'] for x in xs if pred(x,baseline[x['caption_id']])]
        full=[x['caption_id'] for x in eligible if x['k_star']==x['candidate_count']]
        caps=[x['caption_id'] for x,n in zip(xs,natural) if n>K_MAX]
        out[gamma]={'caption_count':len(xs),'nonempty_caption_count':len(eligible),'multi_candidate_caption_count':len(multi),'candidate_count_distribution':dict(sorted(collections.Counter(x['candidate_count'] for x in xs).items())),'N_eff':stats([x['N_eff'] for x in xs]),'N_eff_nonempty':stats([x['N_eff'] for x in eligible]),'k_star':stats([x['k_star'] for x in xs]),'k_over_m_nonempty':stats([x['k_star']/x['candidate_count'] for x in eligible]),'k_distribution':dict(sorted(collections.Counter(x['k_star'] for x in xs).items())),'natural_k_distribution_before_cap':dict(sorted(collections.Counter(natural).items())),'natural_2_or_3_count':sum(n in (2,3) for n in natural),'cap_active_count':len(caps),'cap_active_captions':caps,'full_selection_count':len(full),'full_selection_fraction_nonempty':len(full)/len(eligible),'full_selection_fraction_all25':len(full)/len(xs),'full_selection_multi_count':sum(x['k_star']==x['candidate_count'] for x in multi),'full_selection_fraction_multi':sum(x['k_star']==x['candidate_count'] for x in multi)/len(multi),'full_selection_captions':full,'k_changed_vs_gamma1':change(lambda x,y:x['k_star']!=y['k_star']),'selected_set_changed_vs_gamma1':change(lambda x,y:set(ids(x['selected_attributes']))!=set(ids(y['selected_attributes']))),'top1_changed_vs_gamma1':change(lambda x,y:ids(x['selected_attributes'])[:1]!=ids(y['selected_attributes'])[:1]),'ranking_changed_vs_gamma1':change(lambda x,y:ids(x['ranked_attributes'])!=ids(y['ranked_attributes'])),'k_changed_vs_original_step51_uncapped':[x['caption_id'] for x in xs if x['k_star']!=by_source[x['caption_id']]['topk']['S_raw']['k_star']]}
    return out


def write_markdown(out,summary,examples,audit):
    f=lambda v:'—' if v is None else f'{v:.6g}'
    lines=['# Step 6.1 Power-Sharpened Dynamic Top-K','', '固定原25句已保存S_raw，γ=1/1.5/2/3，β=1，kmax=4。纯selection-only；原AttributeInstance/reader/basis/score不改；没有模型导入、forward/backward、intervention、feature extraction、训练或attack。','', 'p=softmax(γ log S)，H=-sum p log p，N_eff=exp(H)，k=min(ceil(N_eff),m,4)。排序仅用原S_raw；γ不定义新importance。m=0约定H=N_eff=k=0；仅正S进入candidate，本数据118 scored全部正。','', 'mean/median N_eff和k包含25句（1句m=0）；k/m与全选比例仅以24句非空为分母，另列23句m>=2结果。','','## 四个gamma汇总','','| γ | N_eff mean/median | k mean/median | k/m mean/median | k=0/1/2/3/4 | 全选/24 | 全选(m≥2)/23 | 自然k=2–3/25 | cap生效 |','|---:|---|---|---|---|---|---|---|---|']
    for g,x in summary.items():
        counts='/'.join(str(x['k_distribution'].get(k,0)) for k in range(5));pair=lambda v:f(v['mean'])+' / '+f(v['median'])
        lines.append(f"| {g} | {pair(x['N_eff'])} | {pair(x['k_star'])} | {pair(x['k_over_m_nonempty'])} | {counts} | {x['full_selection_count']} ({x['full_selection_fraction_nonempty']:.2%}) | {x['full_selection_multi_count']} ({x['full_selection_fraction_multi']:.2%}) | {x['natural_2_or_3_count']} | {x['cap_active_count']} |")
    lines+=['', '| γ | k改变 vsγ1 | 集合改变 vsγ1 | Top1改变 | 排名改变 |','|---:|---:|---:|---:|---:|']
    for g,x in summary.items():lines.append(f"| {g} | {len(x['k_changed_vs_gamma1'])} | {len(x['selected_set_changed_vs_gamma1'])} | {len(x['top1_changed_vs_gamma1'])} | {len(x['ranking_changed_vs_gamma1'])} |")
    lines+=['', '注意γ=1也已加kmax=4，不能把它与旧Step5.1无cap版本的差异归因sharpening。完整uncapped natural k及cap生效caption在summary.json。','', '## 逐caption m / N_eff / k','','| caption | m | γ1 N/k | γ1.5 N/k | γ2 N/k | γ3 N/k |','|---|---:|---|---|---|---|']
    payloads=load(out/'selection_payloads.json');group=collections.defaultdict(dict)
    for x in payloads:group[x['caption_id']][x['gamma']]=x
    for rid,g in group.items():lines.append('| '+rid+' | '+str(g[1.]['candidate_count'])+' | '+' | '.join(f(g[v]['N_eff'])+'/'+str(g[v]['k_star']) for v in GAMMAS)+' |')
    lines+=['', '## 预先固定代表案例','','35:1尖锐、0:0较均匀、61:0最大m=7、99:1及27:1用户指定；在gamma计算前写入protocol.lock.json，未按最终效果选样。','']
    for e in examples:
        lines += ['### '+e['caption_id'],'',e['caption'],'', '| 属性（原S_raw排序） | S_raw | p(1) | p(1.5) | p(2) | p(3) |','|---|---:|---:|---:|---:|---:|']
        first=e['payloads']['1.0'];label=lambda a:a['region']+':'+a['semantic_axis']+'='+str(a['canonical_value'])+'@'+str(a['char_start'])
        for a in first['ranked_attributes']:lines.append('| '+label(a)+' | '+f(a['S_raw'])+' | '+' | '.join(f(e['payloads'][str(g)]['probabilities'][a['instance_id']]) for g in GAMMAS)+' |')
        lines+=['', '| γ | H | N_eff | k | selected attributes |','|---:|---:|---:|---:|---|']
        for g,x in e['payloads'].items():lines.append(f"| {g} | {f(x['entropy'])} | {f(x['N_eff'])} | {x['k_star']} | "+', '.join(label(a) for a in x['selected_attributes'])+' |')
    lines+=['', '## 检查','','```json',json.dumps(audit,ensure_ascii=False,indent=2),'```','','完整100个payload均保留原实例字段/semantic representation/source span/provenance及原S_raw，仅附selection概率/entropy/K。原模型/score结果未触碰。','']
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    out=ROOT/'outputs/diagnostic'
    if out.exists():raise FileExistsError('refuse overwrite: '+str(out))
    t0=time.time();lock=load(ROOT/'protocol.lock.json');src=Path(lock['source_root']);before={}
    for name,expected in lock['input_sha256'].items():
        before[name]=digest(src/name)
        if before[name]!=expected:raise ValueError('source SHA mismatch: '+name)
    source=load(src/'results.json');manifest=load(src/'selection.json');assert len(source)==25 and [x['row_id'] for x in source]==[x['row_id'] for x in manifest]
    protocol=dict(lock,new_source_sha256={p.name:digest(p) for p in ROOT.glob('*.py')},lock_sha256=digest(ROOT/'protocol.lock.json'),model_import_policy='stdlib only; no IRRA/torch/reader/score imports',original_score='frozen S_raw, not S_norm or S^gamma',probabilities_interface='instance_id -> probability; metadata unchanged',zero_policy='status=scored and S_raw==0 explicitly excluded; source count reported',empty_policy='m=0 => entropy=N_eff=k=0',summary_denominators='N/k mean include25; k/m and full fraction24 nonempty; multi subset23')
    out.mkdir();save(out/'protocol.json',protocol)
    payloads=[];originals={};eligible_total=0;zero_count=0;cap_only=[]
    for row in source:
        candidates=candidates_from_record(row);eligible_total+=len(candidates);originals[row['row_id']]={a['instance_id']:a for a in candidates};zero_count+=sum(a['status']=='scored' and a['S_raw']==0 for a in row['attributes'])
        assert [a['instance_id'] for a in sorted(candidates,key=lambda a:-a['S_raw'])]==row['topk']['S_raw']['order'],'frozen baseline ranking regression'
        for g in GAMMAS:
            x=select(row['row_id'],candidates,g);payloads.append(x)
            assert abs(math.fsum(x['probabilities'].values())-(1 if candidates else 0))<2e-14
            assert 0<=x['k_star']<=min(len(candidates),4)
            for a in x['ranked_attributes']:
                assert a==originals[row['row_id']][a['instance_id']]
            assert x['selected_attributes']==x['ranked_attributes'][:x['k_star']]
            if g==1:
                old=row['topk']['S_raw'];assert math.isclose(x['N_eff'],old['n_eff'],rel_tol=1e-12,abs_tol=1e-12)
                assert x['k_star']==min(old['k_star'],4)
                if old['k_star']>4:cap_only.append(row['row_id'])
    assert len(payloads)==100 and eligible_total==118 and zero_count==0
    summary=summarize(payloads,source);assert all(not x['top1_changed_vs_gamma1'] and not x['ranking_changed_vs_gamma1'] for x in summary.values())
    for row in source:
        xs=[next(x for x in payloads if x['caption_id']==row['row_id'] and x['gamma']==g) for g in GAMMAS]
        for a,b in zip(xs,xs[1:]):
            assert b['N_eff']<=a['N_eff']+1e-12 and b['k_star']<=a['k_star'];assert ids(b['selected_attributes'])==ids(a['selected_attributes'])[:b['k_star']]
    examples=[{'caption_id':rid,'caption':next(x['caption'] for x in source if x['row_id']==rid),'reason':reason,'payloads':{str(g):next(x for x in payloads if x['caption_id']==rid and x['gamma']==g) for g in GAMMAS}} for rid,reason in lock['cases_before_selection'].items()]
    imported=[m for m in sys.modules if m=='torch' or m.startswith(('torch.','numpy','attributes.','learned_reader','importance','model.'))];assert not imported
    after={name:digest(src/name) for name in before};assert before==after
    audit={'caption_count':25,'selection_payload_count':100,'positive_scored_candidates':eligible_total,'source_status_counts':dict(collections.Counter(a['status'] for row in source for a in row['attributes'])),'zero_scored_excluded':zero_count,'frozen_source_sha_unchanged':before==after,'frozen_files_checked':len(before),'all_instance_metadata_and_S_raw_exact':True,'original_raw_ranking_exact':True,'gamma1_entropy_vs_step51_exact_within_fp64_tolerance':True,'gamma1_k_equals_capped_original_k':True,'cap_only_changed_vs_original_step51_count':len(cap_only),'cap_only_changed_vs_original_step51_captions':cap_only,'entropy_effective_count_monotonic':True,'k_monotonic':True,'all_gamma_top1_unchanged':True,'all_gamma_rankings_unchanged':True,'selected_sets_nested_prefixes':True,'forbidden_model_modules_imported':imported,'model_forward_backward_or_training_calls':0,'new_features_or_scores_computed':False,'attack_or_generator_calls':0,'tracked_diff':subprocess.check_output(['git','diff','--stat'],cwd=ROOT.parent.parent,text=True),'staged_diff':subprocess.check_output(['git','diff','--cached','--stat'],cwd=ROOT.parent.parent,text=True)}
    save(out/'selection_payloads.json',payloads);save(out/'summary.json',summary);save(out/'examples.json',examples);save(out/'audit.json',audit);write_markdown(out,summary,examples,audit)
    save(out/'completed.json',{'seconds':time.time()-t0,'selection_only':True,'caption_count':25});print(json.dumps(summary,ensure_ascii=False,indent=2));print('COMPLETE',out,flush=True)

if __name__=='__main__':main()
