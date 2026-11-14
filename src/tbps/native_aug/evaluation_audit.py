"""独立CPU核对全部生成输入、来源、原生评分分块和完整Gallery排名。"""
import time,resource,torch
from unittest.mock import patch
from .common import *
from .audit import action_check
from .evaluation import manifest,attack_root,checkpoint,bindings
def audit_inputs(c,source,round_index):
    from .training import native_bundle
    from .prepare import pixels
    from tbps.alternating.native_sources import native_session
    from tbps.alternating.native_backend import assert_pixels
    start=time.monotonic();need(not torch.cuda.is_initialized(),'input audit CPU-only')
    n,bundle=native_bundle(c,source);m=manifest(c);root=attack_root(source,round_index);store=ArtifactStore(root,new_base=OUTPUT)
    store.verify('completed');cp=checkpoint(source,round_index);ArtifactStore(cp.parent,new_base=OUTPUT).verify(cp.name)
    maximum=0.;edits=0
    with native_session(n,reuse=bundle.modules),torch.no_grad():
        for i,g in enumerate(m.gallery):
            store.verify(f'image_{i:04d}');v=torch.load(root/f'image_{i:04d}/data.pt',map_location='cpu',weights_only=True)
            checked(FileRef(str(Path(n['prototype']['image_root'])/g.image_rel_path),g.image_sha256))
            original=pixels(bundle,Path(n['prototype']['image_root'])/g.image_rel_path)
            need(torch.equal(original,v['original']),'original RGB source exact')
            assert_pixels(v['original'],v['adversarial'],v['effective_delta']);maximum=max(maximum,float(v['effective_delta'].double().abs().max()))
        for i,q in enumerate(m.queries):
            store.verify(f'text_{i:04d}');a=read_json(root/f'text_{i:04d}/action.json')
            need(a['original']==q.caption and a['native_action']['original']==q.caption and a['native_action']['adversarial']==a['adversarial'],'original query identity')
            action_check(a['native_action'],bundle);edits+=len(a['native_action']['edits'])
    report={'status':'NATIVE_AUG_INPUT_CPU_AUDIT_PASS','source':source,'round':round_index,'gallery':3074,'query':6156,'max_Linf':maximum,
        'Unicode_edits':edits,'G':asdict(ref(cp/'G.pt')),'H':asdict(ref(cp/'H.pt')),'elapsed_seconds':time.monotonic()-start,'CUDA':False}
    publish(root/'cpu_audit','completed',{'report.json':json_bytes(report)},report,sources=(ref(root/'completed/manifest.json'),ref(cp/'manifest.json')))
    return report
def audit_evaluation(c,source,target,round_index):
    from tbps.engines.evaluation import load_result
    from tbps.engines.metrics import ranking
    from tbps.engines.materialized_cache import FeatureCache
    from tbps.engines.native_cache import NativeFeatureCache
    from tbps.engines.provenance import payload_digest
    start=time.monotonic();need(not torch.cuda.is_initialized(),'evaluation audit CPU-only');torch.set_num_threads(2)
    root=OUTPUT/source/'evaluation'/target/f'round_{round_index:02d}';store=ArtifactStore(root,new_base=OUTPUT)
    store.verify('completed');store.verify('protocol');store.verify('result')
    summary=read_json(root/'completed/summary.json');p=read_json(root/'protocol/protocol.json');m=manifest(c)
    need(summary['source']==source and summary['target']==target and summary['round']==round_index and summary['model_frozen'] and summary['scope']=='official_test'
        and summary['gallery']==3074 and summary['query']==6156 and summary['Test_feedback'] is False,'formal evaluation receipt')
    need(p['configuration']==asdict(ref(c['_config_path'])) and p['source']==source and p['target']==target and p['round']==round_index,'config/model/round identities')
    if target=='irra':
        from tbps.models.irra_source import IRRASource
        model_source=IRRASource.from_dict(read_json(c['native_template']['sources']['dataset_config']['path'])['irra'])
        model_source.verify()
        from tbps.models.irra_io import official_tokenizer,PREPROCESS,METRICS
        from tbps.models.base import ModelSpace
        from importlib.metadata import version
        import tbps.models.irra as module
        tokenizer=official_tokenizer(model_source)
        preprocess=digest({"contract":PREPROCESS,"runtime":{k:version(k) for k in ("torch","torchvision","Pillow")}})
        producers={name:sha_file(Path(module.__file__).parent/name) for name in ("irra.py","irra_io.py","irra_source.py","base.py")}
        source_identity=digest({"official":model_source.fingerprint,"adapter_producers":producers})
        space=ModelSpace("irra",model_source.checkpoint.sha256,source_identity,preprocess,tokenizer.fingerprint,
            digest({"contract":METRICS,"official_rank_sha256":model_source.ref("utils/metrics.py").sha256}),"fp32/math/no_tf32")
        mod=None
    else:
        from importlib import import_module
        mod=import_module('tbps.models.'+target)
        model_source=(mod.RasaSource if target=='rasa' else mod.APTMSource)(c['targets'][target]['path']);model_source.verify();space=model_source.space
    plain=lambda x:json.loads(json_bytes(x))
    need(p['model_space']==plain(asdict(space)),'target feature space identity')
    b,origins,originals=bindings(c,source,round_index,m,space)
    cache=FeatureCache(store,p,space,chunk=16) if target=='irra' else NativeFeatureCache(store,p,space,mod.validate_payload,chunk=16)
    def forbidden(*a,**k):raise ArtifactError('CPU result audit forbids neural/gradient/CUDA')
    with patch.object(torch.nn.Module,'_call_impl',forbidden),patch.object(torch.cuda,'_lazy_init',forbidden),patch.object(torch.autograd,'grad',forbidden),patch.object(torch.autograd,'backward',forbidden),torch.no_grad():
        for family,binding in b.items():
            for start_index in range(0,len(binding.ordered_ids),16):
                x=cache.load(family,binding,start_index,min(start_index+16,len(binding.ordered_ids)));del x
        ep=read_json(root/'result/protocol.json')
        ep,result,scores=load_result(store,'result',expected_protocol_sha256=digest(ep))
        need(ep['modes']==['Joint'] and ep['query_chunk']==128 and ep['model_space']==plain(asdict(space)),'Joint native full protocol')
        for family,binding in b.items():
            d=ep['inputs'][family]
            need(d['inputs']==plain(asdict(binding)) and d['origin']==plain(asdict(origins[family])) and d['original_sha256']==list(originals[binding.modality]),'complete weights/input binding')
        q,g='adv_text','adv_gallery';nq,ng=6156,3074
        if target=='irra':
            values={f:cache.gather(f,x,origin=origins[f],originals=originals[x.modality]) for f,x in b.items()}
            actual=values[q].batch.payload@values[g].batch.payload.T
        elif target=='rasa':
            parts=[]
            for start_index in range(0,nq,128):
                key=f'score-{q}--{g}-{start_index:06d}';md=store.verify(key)
                need(md['provenance']=={'space_id':space.id,'query':digest(ep['inputs'][q]),'gallery':digest(ep['inputs'][g]),
                    'start':start_index,'end':min(start_index+128,nq),'protocol_sha256':cache.protocol_ref.sha256},'RaSa complete score block binding')
                x=torch.load(root/key/'score.pt',map_location='cpu',weights_only=True)
                need(payload_digest(x)==read_json(root/key/'details.json')['score_sha256'],'RaSa score content SHA')
                parts.append(x)
            actual=torch.cat(parts)
        else:
            parts=[]
            for start_index in range(0,nq,128):
                key=f'aptm-raw-{q}--{g}-{start_index:06d}';md=store.verify(key)
                need(md['provenance']=={'model_space':space.id,'query':digest(ep['inputs'][q]),'gallery':digest(ep['inputs'][g]),
                    'protocol_sha256':cache.protocol_ref.sha256,'normalization':'complete bound Query/Gallery matrix','start':start_index,'end':min(start_index+128,nq)},'APTM complete raw block binding')
                x=torch.load(root/key/'score.pt',map_location='cpu',weights_only=True)
                need({k:payload_digest(v) for k,v in x.items()}==read_json(root/key/'details.json'),'APTM raw SHA')
                parts.append(x)
            raw={k:torch.cat([x[k] for x in parts]) for k in ('itm','global')};actual=mod.finalize_scores(raw)
            key=f'aptm-final-{q}--{g}';store.verify(key)
            need(torch.equal(actual,torch.load(root/key/'score.pt',map_location='cpu',weights_only=True)['final']),'APTM whole-matrix normalization')
        need(torch.equal(actual,scores['Joint']),'all native persisted scores exact')
        metrics,_=ranking(actual,m.evaluation_identity());need(metrics==result['groups']['Joint'],'independent full native ranking')
        need(all(summary['metrics'][k]==metrics[k] for k in summary['metrics']),'summary metrics exact')
        report={'status':'NATIVE_AUG_TEST_CPU_AUDIT_PASS','source':source,'target':target,'round':round_index,'gallery':ng,'query':nq,'metrics':summary['metrics'],
            'all_native_score_blocks_exact':True,'native_full_ranking_exact':True,'Test_feedback':False,'elapsed_seconds':time.monotonic()-start,
            'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'neural_forward':0,'backward':0,'optimizer':0,'JVP':0,'CUDA':False}
        publish(root/'cpu_audit','completed',{'report.json':json_bytes(report)},report,sources=(ref(root/'completed/manifest.json'),ref(attack_root(source,round_index)/'cpu_audit/completed/manifest.json')))
    return report
