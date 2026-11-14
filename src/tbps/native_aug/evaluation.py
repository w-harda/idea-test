"""每个source×target独立常驻进程；六节点Joint Full，三目标共享同一来源攻击输入。"""
from contextlib import contextmanager
from dataclasses import asdict
import torch,time,resource
from .common import *
from tbps.alternating.native_sources import native_session
from tbps.alternating.formal_evaluation import materialize,utf8
from tbps.models.base import InputBinding
from tbps.engines.provenance import AttackOrigin,payload_digest
from tbps.contracts import AttackBudget,ImageBudget,TextBudget
def manifest(c):
    from tbps.datasets.cuhk_pedes import CuhkPedes
    n=c['native_template']['prototype']
    return CuhkPedes(ref(n['annotation']),n['image_root']).manifest('test',expected_counts=(3074,6156))
def attack_root(source,round_index):return OUTPUT/source/'attacks'/f'round_{round_index:02d}'
def checkpoint(source,round_index):return OUTPUT/source/'checkpoints'/f'round_{round_index:02d}'
def make_attack(c,source,round_index,job):
    from .training import native_bundle
    from tbps.models.irra import fp32_runtime
    cp=checkpoint(source,round_index);ArtifactStore(cp.parent,new_base=OUTPUT).verify(cp.name)
    n,bundle=native_bundle(c,source);m=manifest(c);root=attack_root(source,round_index)
    for side in ('G','H'):
        value=torch.load(cp/(side+'.pt'),map_location='cpu',weights_only=True)
        need(value['source_model']==source and value['round']==round_index,'attack paired checkpoint identities')
        getattr(bundle,side.lower()).load_state_dict(value['state_dict'],strict=True)
    with gpu(job),fp32_runtime(),native_session(n,reuse=bundle.modules):
        bundle.g=bundle.g.cuda().eval().requires_grad_(False);bundle.h.eval().requires_grad_(False)
        state(job,'GENERATE_SHARED_ATTACK',source=source,round=round_index,gallery=3074,query=6156)
        materialize(m,bundle,n,root)
    subprocess.run(command('audit-inputs',source,round_index),check=True,cwd=ROOT)
def bindings(c,source,round_index,m,space):
    attack=ArtifactStore(attack_root(source,round_index),new_base=OUTPUT);attack.verify('completed')
    identity=m.evaluation_identity();mh=digest(asdict(m))
    images=[attack.verify(f'image_{i:04d}')['provenance']['adversarial_rgb'] for i in range(len(m.gallery))]
    texts=[attack.verify(f'text_{i:04d}')['provenance']['adversarial_utf8'] for i in range(len(m.queries))]
    bound={'adv_gallery':InputBinding(space.id,mh,tuple(images),'image',identity.gallery_ids),
           'adv_text':InputBinding(space.id,mh,tuple(texts),'text',identity.query_ids)}
    cp=checkpoint(source,round_index)
    origins={'adv_gallery':AttackOrigin('native-aug-'+source+'-G',ref(cp/'G.pt'),ref(c['_config_path']),
                    AttackBudget(image=ImageBudget('linf',8/255,'RGB [0,1]')),sha_file(c['_config_path'])),
             'adv_text':AttackOrigin('native-aug-'+source+'-H',ref(cp/'H.pt'),ref(c['_config_path']),
                    AttackBudget(text=TextBudget('unicode_character',None,'original Natural-K gamma3/beta1','original H-v1 constraints',1,1)),sha_file(c['_config_path']))}
    originals={'image':tuple(g.image_sha256 for g in m.gallery),'text':tuple(utf8(q.caption) for q in m.queries)}
    return bound,origins,originals
@contextmanager
def target_loaded(c,target):
    if target=='irra':
        from tbps.models.irra import IRRAAdapter
        from tbps.models.irra_source import IRRASource
        source=IRRASource.from_dict(read_json(c['native_template']['sources']['dataset_config']['path'])['irra'])
        adapter=IRRAAdapter.from_source(source,device='cuda:0',allow_model_load=True)
        yield adapter,source,None
    else:
        from importlib import import_module
        mod=import_module('tbps.models.'+target)
        source=(mod.RasaSource if target=='rasa' else mod.APTMSource)(c['targets'][target]['path'])
        loader=mod.load_rasa if target=='rasa' else mod.load_aptm
        with loader(source) as adapter:yield adapter,source,mod
def evaluate_one(c,source,target,round_index,job):
    from tbps.models.irra import fp32_runtime
    from tbps.engines.materialized_cache import FeatureCache,CachedIRRA
    from tbps.engines.native_cache import NativeFeatureCache
    from tbps.engines.native_score_cache import ChunkCachedScorer
    from tbps.engines.aptm_score_cache import APTMSetScorer
    from tbps.engines.evaluation import EvaluationEngine,publish_result
    from tbps.alternating.parents import no_optimizer_steps
    m=manifest(c);root=OUTPUT/source/'evaluation'/target/f'round_{round_index:02d}'
    need(not root.exists(),'evaluation Attempt consumed')
    store=ArtifactStore(root,new_base=OUTPUT);store.publish('claim',{'identity.json':json_bytes({'source':source,'target':target,'round':round_index})},{'source':source,'target':target,'round':round_index,'configuration_sha256':sha_file(c['_config_path'])},
        sources=(ref(c['_config_path']),ref(checkpoint(source,round_index)/'manifest.json'),ref(attack_root(source,round_index)/'cpu_audit/completed/manifest.json')))
    started=time.monotonic()
    with gpu(job),fp32_runtime(),no_optimizer_steps(),target_loaded(c,target) as loaded:
        adapter,target_source,mod=loaded;model=adapter._backend if target=='irra' else adapter.backend
        frozen=tree_sha(model.state_dict());space=adapter.space
        b,origins,originals=bindings(c,source,round_index,m,space)
        protocol={'schema':'tbps-materialized-irra-cache-1' if target=='irra' else 'native-aug-target-cache-1','model_space':asdict(space),
            'G_H_generation':False,'ranking_feedback_to_attack':False,'source':source,'target':target,'round':round_index,
            'manifest':asdict(m),'configuration':asdict(ref(c['_config_path'])),'attack':asdict(ref(attack_root(source,round_index)/'completed/manifest.json')),
            'preprocess':mod.PREPROCESS if mod else None,'scoring':mod.SCORING if mod else 'native normalized dot product','scope':'official_test'}
        store.publish('protocol',{'protocol.json':json_bytes(protocol)},{'space':space.id},sources=(target_source.checkpoint,))
        cache=FeatureCache(store,protocol,space,chunk=16) if target=='irra' else NativeFeatureCache(store,protocol,space,mod.validate_payload,chunk=16)
        view=adapter.inference() if target=='irra' else adapter
        with torch.no_grad():
            for family,binding in b.items():
                for start in range(0,len(binding.ordered_ids),16):
                    end=min(start+16,len(binding.ordered_ids));rows=[]
                    for i in range(start,end):
                        if family=='adv_gallery':
                            x=torch.load(attack_root(source,round_index)/f'image_{i:04d}/data.pt',map_location='cpu',weights_only=True)['adversarial']
                            value=view.encode_images(x.cuda())
                            value=value.payload if target=='irra' else value
                        else:
                            action=read_json(attack_root(source,round_index)/f'text_{i:04d}/action.json')
                            value=view.encode_texts((action['adversarial'],))
                            value=value[0].payload if target=='irra' else value
                        rows.append(value)
                    payload=torch.cat([v.cpu().clone() for v in rows]) if target=='irra' else {k:torch.cat([v[k] for v in rows]) for k in rows[0]}
                    cache.publish(family,binding,start,end,payload)
                    state(job,'ENCODE_NATIVE_TARGET',source=source,target=target,round=round_index,family=family,done=end,total=len(binding.ordered_ids))
        values={family:cache.gather(family,binding,origin=origins[family],originals=originals[binding.modality]) for family,binding in b.items()}
        if target=='irra':scorer=CachedIRRA(space,target_source.checkpoint,cache_protocol=ref(root/'protocol/manifest.json'));execution='readonly_cached'
        else:
            scorer=(ChunkCachedScorer if target=='rasa' else APTMSetScorer)(adapter,values,store,cache.protocol_ref);execution='materialized_native'
        engine=EvaluationEngine(m,scorer,direction='higher_better',execution=execution,model_sources=(target_source.checkpoint,))
        def progress():state(job,'NATIVE_FULL_GALLERY_RANKING',source=source,target=target,round=round_index,gallery=3074,query=6156)
        ep,result,scores=engine.run(values,modes=('Joint',),query_chunk=128,check_stop=progress)
        publish_result(store,'result',ep,result,scores,sources=(ref(c['_config_path']),ref(attack_root(source,round_index)/'cpu_audit/completed/manifest.json')))
        need(tree_sha(model.state_dict())==frozen,'target model changed')
        summary={'source':source,'target':target,'round':round_index,'scope':'official_test','gallery':3074,'query':6156,
            'metrics':{k:result['groups']['Joint'][k] for k in ('R@1','R@5','R@10','mAP')},'model_frozen':True,'Test_feedback':False,
            'configuration_sha256':sha_file(c['_config_path']),'elapsed_seconds':time.monotonic()-started}
        store.publish('completed',{'summary.json':json_bytes(summary)},summary,sources=(ref(root/'result/manifest.json'),ref(root/'protocol/manifest.json')))
    subprocess.run(command('audit-evaluation',source,target,round_index),check=True,cwd=ROOT)
    return summary
def worker(c,source,target):
    job=f'eval_{source}_to_{target}';need(source in ('rasa','aptm') and target in MODELS,'source/target')
    for r in ROUNDS:
        root=OUTPUT/source/'train'/f'round_{r:02d}'/'cpu_audit/completed/report.json'
        wait_file(root,job,source=source,target=target,round=r)
        need(read_json(root)['status']=='NATIVE_AUG_ROUND_CPU_AUDIT_PASS','training audit gate')
        if source==target:make_attack(c,source,r,job)
        else:
            attack_proof=attack_root(source,r)/'cpu_audit/completed/report.json'
            wait_file(attack_proof,job,source=source,target=target,round=r)
            need(read_json(attack_proof)['status']=='NATIVE_AUG_INPUT_CPU_AUDIT_PASS','shared input audit')
        evaluate_one(c,source,target,r,job)
    state(job,'SUCCESS',source=source,target=target,completed_rounds=list(ROUNDS))
