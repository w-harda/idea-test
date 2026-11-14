"""独立CPU流式审计；重算KL/SCST与全部三视图预算，不执行神经前向或JVP。"""
from contextlib import ExitStack
import math,time,resource,torch
from unittest.mock import patch
from .common import *
from tbps.attraction.objectives import ProfileKL,Settings,scalars
from tbps.augmented_attraction.views import recipe,ViewSettings,apply_views
from tbps.augmented_attraction.audit import validate_action
def strict(line):
    def pairs(items):
        out={}
        for k,v in items:
            need(k not in out,'duplicate ledger key');out[k]=v
        return out
    def constant(s):raise ArtifactError('nonfinite ledger JSON')
    return json.loads(line,object_pairs_hook=pairs,parse_constant=constant)
def close(a,b):
    need(math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(float(a),float(b),abs_tol=5e-5,rel_tol=2e-4),'independent numeric mismatch')
def action_check(action,bundle):
    if action.get('fixed_NOOP'):
        need(action['original']==action['adversarial'] and not action['edits'],'fixed NOOP');return
    b=bundle.bindings;contracts=b.get('contracts');constraints=b.get('constraints')
    validate_action(action,contracts,constraints)
    example=contracts.from_text_spans(action['original'],action['attributes'])
    scores=action['predicted_importance']
    available=[bool(constraints.candidates(example,example.caption,i,set(),bundle.probe)) for i in range(len(example.attributes))]
    need(b.get('policy').natural_k([s for s,v in zip(scores,available) if v])==action['budget_k'],'original Natural-K recomputation')
    current=example.caption;used=set()
    for edit in action['edits']:
        aid=edit['attribute_id'];idx=next(i for i,a in enumerate(example.attributes) if a.attribute_id==aid)
        candidates=constraints.candidates(example,current,idx,used,bundle.probe)
        need(edit['position'] in candidates and edit['replacement_char'] in candidates[edit['position']],'original legal Unicode candidate')
        current=current[:edit['position']]+edit['replacement_char']+current[edit['position']+1:]
        used.add(contracts.word_at(example.caption,edit['position']))
def audit_g(g,profile,anchor,index,bundle):
    need(g['K']+1==g['optimizer_steps']==len(g['records']),'K+1 G update count')
    action_check(g['H_action'],bundle)
    need(g['current_H_caption']==g['H_action']['adversarial'],'current H partner caption')
    need(all(x['zero_exact'] and x['FD_relative_error']<=.02 for x in g['JVP_checks']),'strict native JVP receipt')
    previous=None
    jvps=torch.tensor(g['attribute_jvps'],dtype=torch.float64).reshape(g['K'],-1) if g['K'] else torch.empty(0,anchor.shape[1],dtype=torch.float64)
    aggregate=torch.tensor(g['v_local'],dtype=torch.float64)
    partner=torch.tensor(g['current_H_feature'],dtype=torch.float64)[None].expand(3,-1)
    for i,record in enumerate(g['records']):
        need(record['finite_gradients'] and record['cumulative_linf']<=8/255,'G gradient/budget')
        md=record['attraction'];feat=torch.tensor(md['view_features'],dtype=torch.float64);clean=torch.tensor(md['clean_view_features'],dtype=torch.float64)
        need(torch.equal(clean[0].float(),anchor[index]),'native paired clean identity')
        terms=profile.terms(feat,clean,partner)
        for key,value in scalars(terms).items():close(md[key],value)
        close(record['global'],terms['loss'].mean())
        local=aggregate if record['phase']=='joint' else jvps[i]
        score=(feat[0]*local).sum();close(record['local'],score)
        retain=0 if previous is None or record['phase']=='joint' else torch.relu((feat[0]*jvps[i-1]).sum()-previous)
        close(record['retain'],retain);close(record['total'],record['global']+record['local']+record['retain'])
        if record['phase']!='joint':
            previous=score;close(record['next_reference_score'],score)
        need(len(md['view_linf'])==3 and all(0<=v<=8/255 for v in md['view_linf']),'all G views budgets')
def audit_h(h,profile,bank,index,bundle):
    if h.get('fixed_NOOP'):
        need(h['optimizer_steps_this_step']==0,'NOOP not updated');return
    for k in ('sampled','greedy'):action_check(h[k],bundle)
    need(h['optimizer_steps_this_step']==1,'single H SCST update')
    md=h['native_alignment'];text=torch.tensor(md['text_features'],dtype=torch.float64);partner=torch.tensor(md['partner_views'],dtype=torch.float64)
    need(torch.equal(partner.float(),bank[index]),'H uses current G three-view bank')
    rewards=[]
    for j,key in enumerate(('sampled','greedy'),1):
        vals=scalars(profile.terms(text[j:j+1].expand(3,-1),text[:1].expand(3,-1),partner))
        for k,v in vals.items():close(md['values'][j-1][k],v)
        close(h[key+'_reward']['total'],vals['reward']);rewards.append(vals['reward'])
    close(h['advantage'],rewards[0]-rewards[1]);close(h['SCST_loss'],-h['advantage']*h['sequence_log_prob'])
def bank_audit(store,entries,round_index,expected_G):
    from tbps.alternating.native_backend import assert_pixels
    values=[];maximum=0
    for i,e in enumerate(entries):
        store.verify(f'image_{i:04d}')
        x=torch.load(store.root/f'image_{i:04d}/data.pt',map_location='cpu',weights_only=True)
        md=read_json(store.root/f'image_{i:04d}/metadata.json')
        need(md['row']==e['identity']['row_id'] and md['index']==i and md['round']==round_index and md['G']==expected_G,'bank source identities')
        need(torch.equal(x['original'],apply_views(x['original'][:1],md['view_recipe'])),'exact original/local/global views')
        for j in range(3):
            assert_pixels(x['original'][j:j+1],x['adversarial'][j:j+1],x['effective_delta'][j:j+1])
        maximum=max(maximum,float(x['effective_delta'].double().abs().max()))
        values.append(x['features'])
    return torch.stack(values),maximum
def run(c,name,round_index):
    from .training import native_bundle
    from tbps.alternating.native_sources import native_session
    torch.set_num_threads(2);need(not torch.cuda.is_initialized(),'audit CPU-only')
    start=time.monotonic();base=OUTPUT/name;n,bundle=native_bundle(c,name)
    entries=read_json(Path(n['prototype']['history_root'])/'qualification/train.json')['entries']
    initial_store=ArtifactStore(base,new_base=OUTPUT);initial_store.verify('initial_state')
    initial=torch.load(base/'initial_state/state.pt',map_location='cpu',weights_only=False)
    proof=read_json(base/'initial_state/proof.json')
    need(not initial['GA']['state'] and not initial['HA']['state'] and tree_sha(initial['G'])==proof['initial_G'] and tree_sha(initial['H'])==proof['initial_H'],'fresh G/H and empty Adam proof')
    prepared=ArtifactStore(base/'preparation',new_base=OUTPUT);prepared.verify('completed')
    prep=read_json(base/'preparation/completed/report.json')
    anchors=torch.load(base/'preparation/completed/clean.pt',map_location='cpu',weights_only=True)
    profile=ProfileKL(anchors.double(),Settings(**c['attraction']))
    def forbidden(*a,**k):raise ArtifactError('independent audit forbids neural forward/CUDA/autograd')
    with native_session(n,reuse=bundle.modules),patch.object(torch.nn.Module,'_call_impl',forbidden),patch.object(torch.cuda,'_lazy_init',forbidden),patch.object(torch.autograd,'grad',forbidden),patch.object(torch.autograd,'backward',forbidden),torch.no_grad():
        if round_index==0:
            root=base/'smoke';store=ArtifactStore(root,new_base=OUTPUT);store.verify('completed')
            log=read_json(root/'completed/log.json')
            bank,maximum=bank_audit(store,entries[:2],1,log['G_after'])
            for i,g in enumerate(log['G']):audit_g(g,profile,anchors,i,bundle)
            for i,h in enumerate(log['H']):audit_h(h,profile,bank,i,bundle)
            report={'status':'NATIVE_AUG_SMOKE_CPU_AUDIT_PASS','source_model':name,'max_Linf':maximum,'G_updates':sum(x['optimizer_steps'] for x in log['G']),'H_updates':sum(x['optimizer_steps_this_step'] for x in log['H'])}
        else:
            root=base/'train'/f'round_{round_index:02d}';store=ArtifactStore(root,new_base=OUTPUT);store.verify('completed')
            summary=read_json(root/'completed/summary.json');checked(FileRef(**summary['ledger']))
            need(summary['configuration_sha256']==sha_file(c['_config_path']) and summary['source_model']==name,'round configuration identity')
            if round_index==1:need(summary['G_before']==proof['initial_G'] and summary['H_before']==proof['initial_H'],'round1 starts from seed42 fresh weights')
            prev={'counters':{'g_updates':0,'h_visits':0,'h_updates':0},'H_after':summary['H_before']}
            if round_index>1:
                previous=root.parent/f'round_{round_index-1:02d}';ArtifactStore(previous/'cpu_audit',new_base=OUTPUT).verify('completed')
                prev=read_json(previous/'completed/summary.json')
                need(summary['previous_round']==sha_file(previous/'completed/summary.json') and prev['G_after']==summary['G_before'] and prev['H_after']==summary['H_before'],'complete continuous round chain')
            bs=ArtifactStore(root/'bank',new_base=OUTPUT);bs.verify('completed')
            bank,maximum=bank_audit(bs,entries,round_index,summary['G_after'])
            saved=torch.load(bs.root/'completed/features.pt',map_location='cpu',weights_only=True)
            need(torch.equal(bank,saved),'all ordered bank features')
            plans=read_json(base/'preparation/completed/selection.json');ng=nh=visits=0;prior_g=summary['G_before'];prior_h=None;prior_h_tree=summary['H_before']
            with Path(summary['ledger']['path']).open(encoding='utf-8') as stream:
                for line in stream:
                    need(len(line)<=4*1024*1024,'bounded one-row strict ledger')
                    row=strict(line)
                    if row['phase']=='G':
                        need(visits==0 and row['row']==entries[ng]['identity']['row_id'],'G row order')
                        need(row['G_before']==prior_g and row['H_state']==summary['H_before'],'G/H state chain')
                        need(row['K']==len(plans[row['row']]),'native fixed selection K')
                        audit_g(row,profile,anchors,ng,bundle);prior_g=row['G_after'];ng+=1
                    else:
                        need(ng==2048 and row['row_id']==entries[visits]['identity']['row_id'],'H row order')
                        need(row['G_state']==summary['G_after'] and row['bank_sha256']==tree_sha(bank),'H current G/bank')
                        if not row.get('fixed_NOOP'):
                            if prior_h:need(prior_h==row['H_state_sha256_before'],'H internal state chain')
                            prior_h=row['H_state_sha256_after']
                            need(row['H_tree_before']==prior_h_tree,'current H tree continuity')
                            prior_h_tree=row['H_tree_after']
                        audit_h(row,profile,bank,visits,bundle);nh+=row['optimizer_steps_this_step'];visits+=1
            need(ng==visits==2048 and nh==sum(e['eligible'] for e in entries) and prior_g==summary['G_after'] and prior_h_tree==summary['H_after'],'complete G/H coverage')
            expect={'g_updates':prev['counters']['g_updates']+prep['expected_G_updates_per_round'],'h_visits':prev['counters']['h_visits']+2048,'h_updates':prev['counters']['h_updates']+nh}
            need(summary['counters']==expect,'all cumulative counters')
            if round_index in ROUNDS:
                cp=base/'checkpoints'/f'round_{round_index:02d}'
                ArtifactStore(cp.parent,new_base=OUTPUT).verify(cp.name)
                full=torch.load(cp/'state.pt',map_location='cpu',weights_only=False)
                need(full['source_model']==name and full['round']==round_index and full['counters']==expect,'checkpoint execution identity')
                need(tree_sha(full['G'])==summary['G_after'] and tree_sha(full['H'])==summary['H_after'],'paired checkpoint state identity')
                need(all(int(v['step'])==expect['g_updates'] for v in full['G_Adam']['state'].values()),'all G Adam steps')
                for side in ('G','H'):
                    v=torch.load(cp/(side+'.pt'),map_location='cpu',weights_only=True)
                    need(v['source_model']==name and v['round']==round_index and tree_sha(v['state_dict'])==summary[side+'_after'],'paired inference checkpoint')
            report={'status':'NATIVE_AUG_ROUND_CPU_AUDIT_PASS','source_model':name,'round':round_index,'covered_rounds':round_index,
                'covered_train_rows':2048*round_index,'counters':expect,'max_Linf':maximum,'state_continuity':True,
                'native_source_space':prep['source_space'],'configuration_sha256':sha_file(c['_config_path'])}
        report.update(elapsed_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      neural_forward=0,backward=0,optimizer=0,JVP=0,CUDA=False)
        publish(root/'cpu_audit','completed',{'report.json':json_bytes(report)},report,sources=(ref(root/'completed/manifest.json'),ref(c['_config_path'])))
    print(report,flush=True);return report
