"""原生代理的60轮G→当前三视图库→H；每10轮独立权重与六节点评测。"""
from dataclasses import dataclass,fields,asdict
from types import SimpleNamespace
import copy,random, numpy as np, torch
from .common import *
from .native import loaded,localization,jvp
from .condition_schema import build
from .prepare import prepare,pixels
from tbps.attraction.objectives import ProfileKL,Settings,scalars
from tbps.augmented_attraction.views import ViewSettings,recipe,apply_views
from tbps.alternating.native_sources import native_session
from tbps.alternating.fresh_sources import fresh_bundle
from tbps.alternating.state import capture_rng,restore_rng
from tbps.alternating.native_backend import assert_pixels
def condition(source,reader,split,bases,selected,entry,caption,bundle):
    cls=bundle.bindings.get('attribute_interface').AttributeInstance
    aa=[cls(**{k:v for k,v in x['attribute'].items() if k in cls.__dataclass_fields__}) for x in selected]
    located=localization(source,reader,split,bases,caption,aa)
    need(set(located)=={a.instance_id for a in aa},'all selected native attributes must remain supported')
    jvps=[];checks=[]
    for a in aa:
        value,check=jvp(source,caption,located[a.instance_id]);jvps.append(value);checks.append({'instance_id':a.instance_id,**check})
    with torch.no_grad():z=source.text([caption])[0].cpu().clone()
    ss=[x['S_raw'] for x in selected];total=sum(ss)
    value=build(entry['identity'],[a.instance_id for a in aa],ss,[s/total for s in ss],
                torch.stack(jvps) if jvps else torch.empty(0,source.feature_dim),z)
    return value,checks
@dataclass(frozen=True)
class Reward:
    total:float
    pull_kl:float
    push_kl:float
    baseline_pull_kl:float
    pull_gain:float
class HReward:
    def __init__(self,source,profile,partner,example,k,constraints):
        self.source,self.profile,self.partner,self.example,self.k,self.constraints=source,profile,partner,example,k,constraints
        self.runtime=SimpleNamespace(assert_frozen=source.check_frozen)
    def evaluate_many(self,original,variants):
        need(original==self.example.caption and len(variants)==2,'SCST original/sample/greedy contract')
        for t in variants:need(type(t)is str,'text-only variants')
        with torch.no_grad():self.features=self.source.text([original,*variants]).cpu().clone()
        self.values=[];rewards=[]
        for z in self.features[1:]:
            terms=self.profile.terms(z[None].expand(3,-1),self.features[:1].expand(3,-1),self.partner)
            v=scalars(terms);self.values.append(v)
            rewards.append(Reward(v['reward'],v['pull_kl'],v['push_kl'],v['baseline_pull_kl'],v['pull_gain']))
        return tuple(rewards)
    def diagnostics(self):
        return {'native_alignment':{'text_features':self.features.tolist(),'partner_views':self.partner.tolist(),
            'values':self.values,'objective':'pull_current_adversarial_partner_push_clean_self','source_model':self.source.name}}
def native_bundle(c,name=None):
    if "torchvision" not in __import__("sys").modules:
        from .startup import bootstrap
        bootstrap()
    n=native_config(c)
    if name:n['prototype']['source_space_sha256']=digest({'purpose':'initialization only, no embeddings','source_model':name,'source':c['targets'][name]})
    b=fresh_bundle(c,n)
    if name:
        from .guard import bind
        b=bind(c,b,name)
    return n,b
def run(c,name):
    job=f'train_{name}';base=OUTPUT/name
    need(not base.exists(),'source Attempt consumed; no automatic retry')
    base.mkdir(parents=True)
    publish(base,'claim',{'configuration.json':json_bytes({'configuration':asdict(ref(c['_config_path'])),'source':name})},sources=(ref(c['_config_path']),))
    with gpu(job):
        from tbps.models.irra import fp32_runtime
        torch.set_num_threads(2)
        n,bundle=native_bundle(c,name)
        from .compat import source_session
        with fp32_runtime(),loaded(c,name) as source,source_session(n,bundle,c,name):
            reader,split,bases,plans,clean,entries=prepare(c,n,bundle,source,name,job)
            b=bundle.bindings;g,h=bundle.g.cuda(),bundle.h
            ga=bundle.g_adam;ha=bundle.h_adam
            profile_cpu=ProfileKL(clean,Settings(**c['attraction']));profile=profile_cpu.to('cuda:0')
            views=ViewSettings(**c['augmentation']);from .g_step import train_single_sample
            constraints=b.get('constraints');phase=b.get('phase4_training')
            initial=clone_tree({'G':g.state_dict(),'H':h.state_dict(),'GA':ga.state_dict(),'HA':ha.state_dict()})
            rngs={s:clone_tree(bundle.state[s+'_RNG']) for s in ('G','H')}
            torch.cuda.manual_seed_all(42)
            for s in rngs:rngs[s]['cuda']=clone_tree(torch.cuda.get_rng_state_all())
            publish(base,'initial_state',{'state.pt':tensor_bytes(initial),'proof.json':json_bytes(bundle.report)},bundle.report,sources=(ref(c['_config_path']),ref(base/'preparation/completed/manifest.json')))
            original_cache={}
            def get_condition(e,text):
                row=e['identity']['row_id']
                if text==e['caption'] and row in original_cache:return original_cache[row]
                out=condition(source,reader,split,bases,plans[row],e,text,bundle)
                if text==e['caption']:original_cache[row]=out
                return out
            def update_g(e,index,round_index):
                row=e['identity']['row_id'];example=b.get('formal_training.data').example_from(e)
                h.eval().requires_grad_(False);h.zero_grad(set_to_none=True)
                with torch.no_grad():
                    if e['eligible']:action=h.attack(example)
                    else:
                        record={'fixed_NOOP':True,'original':example.caption,'adversarial':example.caption,'edits':[],'budget_k':0}
                        action=SimpleNamespace(adversarial=example.caption,edits=(),budget_k=0,record=lambda:record)
                need(constraints.replay(example,action.edits,action.budget_k)==action.adversarial,'current H Unicode replay')
                cond,checks=get_condition(e,action.adversarial)
                image=pixels(bundle,Path(n['prototype']['image_root'])/e['identity']['image_rel_path']).cuda()
                plan=recipe(views,row,round_index,*image.shape[-2:]);vv=apply_views(image.cpu(),plan).cuda()
                with torch.no_grad():clean_views=source(vv).detach()
                clean_views[0]=clean[index].cuda()
                g.requires_grad_(True);before=tree_sha(g.state_dict())
                result=train_single_sample(g,source,ga,vv,cond,profile=profile,clean_features=clean_views)
                result={k:v for k,v in result.items() if k not in ('serial_adversarial','joint_adversarial')}
                result.update(row=row,G_before=before,G_after=tree_sha(g.state_dict()),H_state=tree_sha(h.state_dict()),
                    current_H_caption=action.adversarial,H_action=action.record(),original_caption=e['caption'],
                    current_H_feature=cond['z_t'].tolist(),attribute_jvps=cond['attribute_jvps'].tolist(),v_local=cond['v_local'].tolist(),JVP_checks=checks,view_recipe=plan)
                return result
            def bank_sample(e,index,round_index,store):
                row=e['identity']['row_id'];im=pixels(bundle,Path(n['prototype']['image_root'])/e['identity']['image_rel_path']).cuda()
                plan=recipe(views,row,round_index,*im.shape[-2:]);vv=apply_views(im.cpu(),plan).cuda()
                g.eval().requires_grad_(False);g.zero_grad(set_to_none=True)
                with torch.no_grad():
                    generated=[b.get('inference_once').generate_once(g,vv[j:j+1]) for j in range(3)]
                    aa=torch.cat([v['adversarial'] for v in generated]);dd=aa-vv
                    for j in range(3):assert_pixels(vv[j:j+1],aa[j:j+1],dd[j:j+1])
                    features=source(aa).cpu().clone()
                payload={'original':vv.cpu().clone(),'adversarial':aa.cpu().clone(),'effective_delta':dd.cpu().clone(),'features':features}
                metadata={'row':row,'index':index,'round':round_index,'view_recipe':plan,'G':tree_sha(g.state_dict()),
                    'linf':[float(dd[j].double().abs().max()) for j in range(3)]}
                store.publish(f'image_{index:04d}',{'data.pt':tensor_bytes(payload),'metadata.json':json_bytes(metadata)},metadata)
                return features
            def update_h(e,index,partner):
                example=b.get('formal_training.data').example_from(e)
                if not e['eligible']:return {'row_id':e['identity']['row_id'],'fixed_NOOP':True,'optimizer_steps_this_step':0}
                h.requires_grad_(True)
                def reward(k):return HReward(source,profile_cpu,partner,example,k,constraints)
                before=tree_sha(h.state_dict())
                log,_=phase.train_step(h,ha,example,bundle.sampler,reward,index+1,e['identity']['row_id'])
                log.update(H_tree_before=before,H_tree_after=tree_sha(h.state_dict()))
                return log
            # 独立真实短Smoke：新G/H更新、原生输入梯度、JVP/Unicode与三视图预算，随后恢复最初空Adam/RNG。
            smoke=ArtifactStore(base/'smoke',new_base=OUTPUT)
            restore_rng(rngs['G'],bundle.sampler,allow_cuda=True);smoke_g=[];smoke_h=[]
            for i,e in enumerate(entries[:2]):
                smoke_g.append(update_g(e,i,1))
            gsha=tree_sha(g.state_dict())
            for i,e in enumerate(entries[:2]):
                partner=bank_sample(e,i,1,smoke);smoke_h.append(update_h(e,i,partner))
            source.check_frozen(final=True)
            smoke.publish('completed',{'log.json':json_bytes({'G':smoke_g,'H':smoke_h,'source':name,'source_frozen':True,'G_after':gsha})},sources=(ref(c['_config_path']),))
            subprocess.run(command('audit',name,0),check=True,cwd=ROOT)
            g.load_state_dict(initial['G'],strict=True);h.load_state_dict(initial['H'],strict=True)
            ga.load_state_dict(initial['GA']);ha.load_state_dict(initial['HA'])
            need(not ga.state and not ha.state,'formal G/H empty Adam after independent Smoke')
            g.zero_grad(set_to_none=True);h.zero_grad(set_to_none=True)
            counters={'g_updates':0,'h_visits':0,'h_updates':0};previous='fresh'
            for round_index in range(1,61):
                round_root=base/'train'/f'round_{round_index:02d}';round_root.mkdir(parents=True)
                ledger=round_root/'ledger.jsonl'
                g_before=tree_sha(g.state_dict());h_before=tree_sha(h.state_dict());inactive_ha=tree_sha(ha.state_dict())
                restore_rng(rngs['G'],bundle.sampler,allow_cuda=True)
                with ledger.open('x',encoding='utf-8') as out:
                    for i,e in enumerate(entries):
                        result=update_g(e,i,round_index);out.write(json.dumps({'phase':'G',**result},allow_nan=False,separators=(',',':'))+'\n')
                        counters['g_updates']+=result['optimizer_steps']
                        state(job,'TRAIN_G',round=round_index,done=i+1,total=2048,**counters)
                    out.flush();os.fsync(out.fileno())
                    rngs['G']=capture_rng(bundle.sampler)
                    need(h_before==tree_sha(h.state_dict()) and inactive_ha==tree_sha(ha.state_dict()),'G phase changed H/model optimizer')
                    current_g=tree_sha(g.state_dict());bank_store=ArtifactStore(round_root/'bank',new_base=OUTPUT);features=[]
                    for i,e in enumerate(entries):
                        features.append(bank_sample(e,i,round_index,bank_store))
                        state(job,'REFRESH_CURRENT_G_VIEW_BANK',round=round_index,done=i+1,total=2048,**counters)
                    bank=torch.stack(features)
                    bank_store.publish('completed',{'features.pt':tensor_bytes(bank),'identity.json':json_bytes({'G':current_g,'rows':[e['identity']['row_id'] for e in entries],'source_model':name,'round':round_index,'feature_sha256':tree_sha(bank)})})
                    inactive_ga=tree_sha(ga.state_dict())
                    restore_rng(rngs['H'],bundle.sampler,allow_cuda=True);g.requires_grad_(False);g.zero_grad(set_to_none=True)
                    for i,e in enumerate(entries):
                        log=update_h(e,i,bank[i]);log.update(phase='H',G_state=current_g,bank_sha256=tree_sha(bank))
                        out.write(json.dumps(log,allow_nan=False,separators=(',',':'))+'\n')
                        counters['h_visits']+=1;counters['h_updates']+=log['optimizer_steps_this_step']
                        state(job,'TRAIN_H',round=round_index,done=i+1,total=2048,**counters)
                    out.flush();os.fsync(out.fileno());rngs['H']=capture_rng(bundle.sampler)
                need(current_g==tree_sha(g.state_dict()) and inactive_ga==tree_sha(ga.state_dict()),'H phase changed G/model optimizer');source.check_frozen(final=True)
                summary={'source_model':name,'round':round_index,'G_before':g_before,'G_after':current_g,'H_before':h_before,
                    'H_after':tree_sha(h.state_dict()),'counters':dict(counters),'ledger':asdict(ref(ledger)),
                    'bank_completed':asdict(ref(round_root/'bank/completed/manifest.json')),'previous_round':previous,
                    'source_frozen':True,'inactive_H_Adam_sha256':inactive_ha,'inactive_G_Adam_sha256':inactive_ga,'configuration_sha256':sha_file(c['_config_path'])}
                publish(round_root,'completed',{'summary.json':json_bytes(summary)},summary,sources=(ref(ledger),ref(round_root/'bank/completed/manifest.json')))
                if round_index in ROUNDS:
                    cp=base/'checkpoints';state_value={'G':g.state_dict(),'H':h.state_dict(),'G_Adam':ga.state_dict(),'H_Adam':ha.state_dict(),
                        'G_RNG':rngs['G'],'H_RNG':rngs['H'],'counters':counters,'round':round_index,'source_model':name}
                    publish(cp,f'round_{round_index:02d}',{'state.pt':tensor_bytes(state_value),'G.pt':tensor_bytes({'state_dict':g.state_dict(),'round':round_index,'source_model':name}),
                        'H.pt':tensor_bytes({'state_dict':h.state_dict(),'round':round_index,'source_model':name}),'identity.json':json_bytes(summary)},summary,
                        sources=(ref(round_root/'completed/manifest.json'),ref(c['_config_path']),ref(base/'preparation/completed/manifest.json')))
                state(job,'CPU_AUDIT_ROUND',round=round_index,**counters)
                subprocess.run(command('audit',name,round_index),check=True,cwd=ROOT)
                previous=sha_file(round_root/'completed/summary.json')
                if round_index==10:spawn_eval(name)
            state(job,'TRAINING_COMPLETE',round=60,**counters)
