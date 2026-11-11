"""固定六节点Joint-only原生IRRA；每节点独立生成/编码，不反馈训练。"""
from pathlib import Path
from dataclasses import asdict,replace
import json
from tbps.artifacts import ArtifactStore,FileRef,checked,sha_file,read_json,json_bytes,digest
from tbps.alternating.contracts import require
from .sources import preflight,load
from .run import no_historical_weights
from tbps.alternating.state import tree_sha
from tbps.alternating.formal_evaluation import materialize,utf8
from tbps.alternating.native_sources import native_session

METRICS=('R@1','R@5','R@10','mAP')
def ref(p):return FileRef(str(p),sha_file(p))

def manifest_for(c,smoke):
    from tbps.datasets.cuhk_pedes import CuhkPedes
    m=CuhkPedes(ref(c['prototype']['annotation']),c['prototype']['image_root']).manifest('test',expected_counts=(3074,6156))
    return replace(m,gallery=m.gallery[:10],queries=m.queries[:10]) if smoke else m

def joint_score(manifest,feature_store,protocol,space,checkpoint,bindings,origins,originals,result_store,*,fixture=False):
    from tbps.engines.materialized_cache import FeatureCache,CachedIRRA
    from tbps.engines.evaluation import EvaluationEngine,publish_result
    cache=FeatureCache(feature_store,protocol,space)
    inputs={family:cache.gather(family,b,origin=origins[family],originals=originals[b.modality]) for family,b in bindings.items()}
    victim=CachedIRRA(space,checkpoint,fixture=fixture,cache_protocol=ref(feature_store.root/'protocol/manifest.json'))
    engine=EvaluationEngine(manifest,victim,direction='higher_better',execution='cpu_fixture' if fixture else 'readonly_cached',model_sources=(checkpoint,))
    ep,report,scores=engine.run(inputs,modes=('Joint',),query_chunk=128)
    publish_result(result_store,'engine',ep,report,scores)
    return report

def evaluate(path,round_index,*,smoke=False,allow=False):
    require(allow,'explicit Test generation capability required')
    f,c,bundle,gate,pre=preflight(path,'test')
    require(round_index==1 if smoke else round_index in f['test_rounds'],'predeclared paired endpoint')
    training=Path(f['outputs']['smoke' if smoke else 'train'])
    proof=read_json(Path(f['audits']['smoke' if smoke else 'train'])/('completed/report.json' if smoke else f'round_{round_index:02d}/completed/report.json'))
    require(proof['status']=='GH_AUG_'+('SMOKE' if smoke else 'TRAIN')+'_CPU_AUDIT_PASS'
        and proof['configuration_sha256']==sha_file(path),'completed matching training audit required')
    cp=ArtifactStore(training/'checkpoints',new_base=f['output_base']);cp.verify(f'round_{round_index:02d}')
    import torch,random,numpy as np
    torch.set_num_threads(2);random.seed(42);np.random.seed(42);torch.manual_seed(42)
    gref=ref(cp.root/f'round_{round_index:02d}/G.pt');href=ref(cp.root/f'round_{round_index:02d}/H.pt')
    for side,r in (('G',gref),('H',href)):
        v=torch.load(r.path,map_location='cpu',weights_only=False)
        require(v['round']==round_index and v['model']==side and v['configuration_sha256']==sha_file(path),'paired v2 weights only')
        getattr(bundle,side.lower()).load_state_dict(v['state_dict'],strict=True)
    manifest=manifest_for(c,smoke)
    root=Path(f['test_output'])/('smoke' if smoke else f'round_{round_index:02d}')
    require(not root.exists(),'Test Attempt consumed; no re-generation/retry')
    store=ArtifactStore(root,new_base=f['output_base']);store.publish('claim',{'protocol.json':json_bytes({
        'round':round_index,'G':asdict(gref),'H':asdict(href),'manifest_sha256':digest(asdict(manifest)),
        'precision':'fp32/math/no_tf32','modes':['Joint'],'image_batch':1,'text_batch':1,'query_chunk':128,
        'seed':42,'Test_feedback':False,'from_scratch_run':True})},{'configuration':sha_file(path)})
    try:
        from tbps.alternating.configured_cli import resources
        require(resources(f,'train')['resource_eligible'],'RAM/disk/inode insufficient')
        gate.initialize_cuda()
        from tbps.models.irra import IRRAAdapter,fp32_runtime
        from tbps.models.irra_source import IRRASource
        from tbps.models.base import InputBinding
        from tbps.engines.materialized_cache import FeatureCache
        from tbps.engines.provenance import AttackOrigin
        from tbps.contracts import AttackBudget,ImageBudget,TextBudget
        from .audit import audit_test
        with no_historical_weights(f['output_base']),fp32_runtime(),native_session(c,reuse=bundle.modules):
            bundle.g=bundle.g.to('cuda:0').eval().requires_grad_(False);bundle.h.eval().requires_grad_(False)
            attack=materialize(manifest,bundle,c,root/'attack')
        # 攻击全部冻结后才加载受害者编码；不将分数提供给G/H。
        with fp32_runtime():
            source=IRRASource.from_dict(read_json(c['sources']['dataset_config']['path'])['irra'])
            adapter=IRRAAdapter.from_source(source,device='cuda:0',allow_model_load=True)
            frozen=tree_sha(adapter._backend.state_dict());view=adapter.inference();space=adapter.space
            features=ArtifactStore(root/'features',new_base=f['output_base'])
            protocol={'schema':'tbps-materialized-irra-cache-1','model_space':asdict(space),'G_H_generation':False,
                'ranking_feedback_to_attack':False,'manifest':asdict(manifest),'attack_completion':asdict(ref(attack.root/'completed/manifest.json'))}
            features.publish('protocol',{'protocol.json':json_bytes(protocol)},{'model_space':space.id},sources=(source.checkpoint,))
            originals={'image':tuple(g.image_sha256 for g in manifest.gallery),'text':tuple(utf8(q.caption) for q in manifest.queries)}
            im=[attack.verify(f'image_{i:04d}')['provenance']['adversarial_rgb'] for i in range(len(manifest.gallery))]
            tx=[attack.verify(f'text_{i:04d}')['provenance']['adversarial_utf8'] for i in range(len(manifest.queries))]
            identity=manifest.evaluation_identity();mh=digest(asdict(manifest))
            bindings={'adv_gallery':InputBinding(space.id,mh,tuple(im),'image',identity.gallery_ids),
                'adv_text':InputBinding(space.id,mh,tuple(tx),'text',identity.query_ids)}
            cache=FeatureCache(features,protocol,space)
            with torch.no_grad():
                for family,b in bindings.items():
                    for start in range(0,len(b.ordered_ids),64):
                        rows=[];end=min(start+64,len(b.ordered_ids))
                        for i in range(start,end):
                            if family=='adv_gallery':
                                value=torch.load(attack.root/f'image_{i:04d}/data.pt',map_location='cpu',weights_only=True)
                                x=view.encode_images(value['adversarial'].to('cuda:0')).payload
                            else:
                                action=read_json(attack.root/f'text_{i:04d}/action.json')
                                value,_=view.encode_texts((action['adversarial'],));x=value.payload
                            rows.append(x.detach().cpu().clone())
                        cache.publish(family,b,start,end,torch.cat(rows));print('GH_AUG_TEST_ENCODE',round_index,family,end,flush=True)
            require(frozen==tree_sha(adapter._backend.state_dict()),'frozen Target parameters/buffers')
            origins={'adv_gallery':AttackOrigin('gh-pushpull-aug-g',gref,ref(path),AttackBudget(image=ImageBudget('linf',8/255,'RGB [0,1]')),digest(c['sources'])),
                'adv_text':AttackOrigin('gh-pushpull-aug-h',href,ref(path),AttackBudget(text=TextBudget('unicode_character',None,
                    'original Natural-K gamma3/beta1','original H-v1 constraints',1,1)),digest(c['sources']))}
            result=joint_score(manifest,features,protocol,space,source.checkpoint,bindings,origins,originals,
                ArtifactStore(root/'results',new_base=f['output_base']))
            public={'round':round_index,'G':asdict(gref),'H':asdict(href),'gallery':len(manifest.gallery),'query':len(manifest.queries),
                'metrics':{k:result['groups']['Joint'][k] for k in METRICS},'frozen_Target':True,'scope':'smoke' if smoke else 'official_test',
                'configuration_sha256':sha_file(path),'Test_feedback':False}
            store.publish('completed',{'summary.json':json_bytes(public)},{'configuration':sha_file(path),'round':round_index},
                sources=(ref(root/'results/engine/manifest.json'),gref,href))
        return public
    except BaseException:
        import traceback
        with (root/'FAIL').open('x') as out:out.write(traceback.format_exc())
        raise
