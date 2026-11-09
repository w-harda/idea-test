"""读取固定AnyAttack图像，复用三模型原生Adapter、融合评分与完整Gallery排名。"""
from contextlib import contextmanager
from dataclasses import asdict
import time, hashlib, resource
import torch
from tbps.models.irra import IRRAAdapter, fp32_runtime
from tbps.models.irra_source import IRRASource
from tbps.models.base import InputBinding
from tbps.engines.materialized_cache import FeatureCache, CachedIRRA
from tbps.engines.native_cache import NativeFeatureCache
from tbps.engines.native_score_cache import ChunkCachedScorer
from tbps.engines.aptm_score_cache import APTMSetScorer
from tbps.engines.provenance import AttackOrigin
from tbps.engines.evaluation import EvaluationEngine, publish_result, load_result
from tbps.engines.metrics import ranking
from tbps.contracts import AttackBudget,ImageBudget
from tbps.smoke_irra import model_snapshot
from .core import *

@contextmanager
def victim(ctx,name):
    if name=='irra':
        src=IRRASource.from_dict(ctx['d']['irra']);src.verify()
        adapter=IRRAAdapter.from_source(src,device='cuda:0',allow_model_load=True)
        yield adapter,src.checkpoint,None,None
    else:
        if name=='rasa':
            from tbps.models.rasa import RasaSource,load_rasa,validate_payload,SCORING,PREPROCESS
            src=RasaSource(ctx['c']['rasa_source']['path']);loader=load_rasa
        elif name=='aptm':
            from tbps.models.aptm import APTMSource,load_aptm,validate_payload,SCORING,PREPROCESS
            src=APTMSource(ctx['c']['aptm_source']['path']);loader=load_aptm
        else:raise ArtifactError('unknown victim')
        src.verify()
        with loader(src,device='cuda:0',smoke=ctx['scope']=='smoke') as adapter:
            yield adapter,src.checkpoint,validate_payload,{'source':asdict(src.ref),'scoring':SCORING,'preprocess':PREPROCESS}

def bindings(ctx,space,hashes):
    m=ctx['manifest'];mh=digest(asdict(m));gids=tuple(g.image_id for g in m.gallery)
    original=tuple(g.image_sha256 for g in m.gallery)
    return {'clean_gallery':InputBinding(space.id,mh,original,'image',gids),
            'adv_gallery':InputBinding(space.id,mh,tuple(hashes),'image',gids),
            'clean_text':InputBinding(space.id,mh,tuple(hashlib.sha256(q.caption.encode('utf8')).hexdigest() for q in m.queries),
                                     'text',tuple(q.query_id for q in m.queries))}

def run(ctx,name):
    t=time.monotonic();s=store_at(ctx,name);need(not s.root.exists(),'single victim Attempt; existing output stops')
    generation=store_at(ctx,'generation');generation.verify('completed')
    audit=store_at(ctx,'generation_cpu_audit');audit.verify('result')
    need(read_json(audit.root/'result/report.json')['status']=='PASS','independent source audit gate')
    done=read_json(generation.root/'completed/report.json')
    s.publish('started',{'identity.json':json_bytes({'config':asdict(ref(ctx['path'])),'model':name,'scope':ctx['scope']})},
        {'kind':'native_victim_attempt'},sources=(ref(audit.root/'result/manifest.json'),ref(ctx['path'])))
    report={'status':'RUNNING','model':name,'scope':ctx['scope'],'backward':0,'optimizer':0,'JVP':0,'attack_forward':0}
    try:
        torch.set_num_threads(2);torch.manual_seed(42)
        with no_training(),fp32_runtime(),victim(ctx,name) as (adapter,checkpoint,validator,native):
            backend=adapter._backend if name=='irra' else adapter.backend
            before=model_snapshot(backend);space=adapter.space
            bound=bindings(ctx,space,done['adversarial_sha256'])
            protocol={'schema':'tbps-materialized-irra-cache-1' if name=='irra' else 'tbps-native-materialized-cache-1',
                'config':asdict(ref(ctx['path'])),'model_space':asdict(space),'manifest':asdict(ctx['manifest']),
                'source_generation':asdict(ref(generation.root/'completed/manifest.json')),
                'inputs':{k:asdict(v) for k,v in bound.items()},'G_H_generation':False,'ranking_feedback_to_attack':False,
                'AnyAttack_generation':False,'native_protocol':native,'scope':ctx['scope']}
            s.publish('protocol',{'protocol.json':json_bytes(protocol)},{'kind':'native_encoding'},
                sources=(checkpoint,ref(ctx['path']),ref(generation.root/'completed/manifest.json')))
            cache=FeatureCache(s,protocol,space,chunk=16) if name=='irra' else NativeFeatureCache(s,protocol,space,validator,chunk=16)
            view=adapter.inference() if name=='irra' else adapter
            counts={'clean_gallery':0,'adv_gallery':0,'clean_text':0}
            for family,b in bound.items():
                for start in range(0,len(b.ordered_ids),16):
                    end=min(start+16,len(b.ordered_ids));rows=[]
                    for i in range(start,end):
                        if family.endswith('gallery'):
                            x,_=load_image(ctx,i)
                            value=view.encode_images(x['original' if family=='clean_gallery' else 'adversarial'].to('cuda:0') if name=='irra'
                                                     else x['original' if family=='clean_gallery' else 'adversarial'])
                            del x
                        else:value=view.encode_texts((ctx['manifest'].queries[i].caption,))
                        if name=='irra':
                            if family=='clean_text':value=value[0]
                            value=value.payload.detach().cpu().clone()
                        rows.append(value);counts[family]+=1
                    value=torch.cat(rows) if name=='irra' else {k:torch.cat([r[k] for r in rows]) for k in rows[0]}
                    cache.publish(family,b,start,end,value);del value,rows
                    print('ANYATTACK_NATIVE_ENCODING',name,family,end,'/',len(b.ordered_ids),flush=True)
            origin=AttackOrigin('anyattack_pretrained_eps8',FileRef(**ctx['c']['decoder']),ref(ctx['path']),
                AttackBudget(image=ImageBudget('linf',EPS,'RGB [0,1]; strict8/255 inward projection; zero audit tolerance')),
                digest({'official_files':ctx['c']['official_files'],'clip':ctx['c']['clip']}))
            inputs={family:cache.gather(family,b,origin=origin if family=='adv_gallery' else None,
                    originals=tuple(g.image_sha256 for g in ctx['manifest'].gallery) if family=='adv_gallery' else ())
                    for family,b in bound.items()}
            scorer=CachedIRRA(space,checkpoint,cache_protocol=cache.protocol_ref) if name=='irra' else (
                APTMSetScorer(adapter,inputs,s,cache.protocol_ref) if name=='aptm' else ChunkCachedScorer(adapter,inputs,s,cache.protocol_ref))
            engine=EvaluationEngine(ctx['manifest'],scorer,direction='higher_better',
                execution='readonly_cached' if name=='irra' else 'materialized_native',model_sources=(checkpoint,))
            p,result,scores=engine.run(inputs,modes=('Clean','Image-only'),query_chunk=128)
            publish_result(s,'result',p,result,scores,sources=(cache.protocol_ref,ref(ctx['path'])));del scores,inputs
            after=model_snapshot(backend);need(before==after,'victim parameters/buffers changed')
            need(all(not v.requires_grad and v.grad is None for v in backend.parameters()),'victim parameter gradients')
            if name!='irra':
                need(all(adapter.counts[k]==adapter.neural_counts[k] for k in adapter.neural_counts),'neural hook counts')
                report.update(native_counts=adapter.counts,native_neural_hook_counts=adapter.neural_counts)
            report.update(status='PASS',model_before=before,model_after=after,frozen=True,counts=counts,
                          metrics=result['groups'],model_space=asdict(space))
    except BaseException as exc:
        import traceback
        report.update(status='FAIL',error=str(exc),traceback=traceback.format_exc());raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-t,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        if torch.cuda.is_initialized():report.update(gpu_peak_allocated=torch.cuda.max_memory_allocated(),gpu_peak_reserved=torch.cuda.max_memory_reserved())
        s.publish('completed' if report['status']=='PASS' else 'FAIL',{'report.json':json_bytes(report)},
            {'kind':'native_evaluation_execution'},sources=(ref(s.root/'started/manifest.json'),))
    print('ANYATTACK_NATIVE_PASS',name,ctx['scope'],report['metrics'],flush=True)

def audit(ctx,name):
    t=time.monotonic();s=store_at(ctx,name)
    need(not s.pending(),'pending victim artifacts')
    for p in sorted(s.root.iterdir()):
        if p.is_dir():s.verify(p.name)
    execution=read_json(s.root/'completed/report.json')
    need(execution['status']=='PASS' and execution['model_before']==execution['model_after'] and execution['frozen'],'victim frozen execution')
    p=read_json(s.root/'result/protocol.json');_,report,scores=load_result(s,'result',expected_protocol_sha256=digest(p))
    need(p['dataset_manifest']==read_json_bytes(asdict(ctx['manifest'])) and p['modes']==['Clean','Image-only'] and
         p['score_feedback_to_attacks'] is False,'complete manifest/modes/no feedback')
    need(execution['counts']=={'clean_gallery':len(ctx['manifest'].gallery),'adv_gallery':len(ctx['manifest'].gallery),
                              'clean_text':len(ctx['manifest'].queries)},'complete native encoding counts')
    source_report=read_json(store_at(ctx,'generation').root/'completed/report.json')
    ib=p['inputs'];need(ib['adv_gallery']['inputs']['ordered_sha256']==source_report['adversarial_sha256'],'same fixed attack collection across victims')
    need(ib['clean_text']['origin'] is None and ib['clean_text']['inputs']['ordered_sha256']==[
        hashlib.sha256(q.caption.encode('utf8')).hexdigest() for q in ctx['manifest'].queries],'unchanged complete original captions')
    for mode,x in scores.items():
        values,_=ranking(x,ctx['manifest'].evaluation_identity())
        need(values==report['groups'][mode]==execution['metrics'][mode],'independent CPU full-matrix ranking exact')
    result={'status':'PASS','model':name,'scope':ctx['scope'],'gallery':len(ctx['manifest'].gallery),
        'query':len(ctx['manifest'].queries),'metrics':report['groups'],'frozen_model':True,'CUDA_initialized':torch.cuda.is_initialized(),
        'source_collection_sha256':ref(store_at(ctx,'generation').root/'completed/manifest.json').sha256,
        'elapsed_seconds':time.monotonic()-t,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    out=store_at(ctx,name+'_cpu_audit');out.publish('result',{'report.json':json_bytes(result)},
        {'kind':'independent_native_score_cpu_audit'},sources=(ref(s.root/'result/manifest.json'),ref(s.root/'completed/manifest.json')))
    print('ANYATTACK_NATIVE_CPU_AUDIT_PASS',result,flush=True)
