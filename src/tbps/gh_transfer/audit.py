"""独立CPU读回原生特征/融合分块/全矩阵/预算来源与所有排名。"""
from pathlib import Path
from dataclasses import asdict
import time,resource
from .common import *
from .evaluation import preflight,bindings_for

def protection(c,method,*,rehash=False):
    coll,s=collection(c,method);snapshot=read_json(s.root/'completed/protection.json')
    for name,v in snapshot.items():
        st=Path(name).stat()
        need((st.st_size,st.st_mtime_ns,st.st_ino)==(v['size'],v['mtime_ns'],v['inode']),'historical source stat changed: '+name)
        if rehash:checked(FileRef(name,v['sha256']))
    return len(snapshot)

def audit(path,method,model,scope):
    import torch
    from tbps.alternating.parents import cpu_restore_only
    from tbps.engines.native_cache import NativeFeatureCache
    from tbps.engines.evaluation import load_result
    from tbps.engines.metrics import ranking
    from tbps.engines.provenance import payload_digest
    torch.set_num_threads(2);start=time.monotonic()
    c,mod,source=preflight(path,model);coll,input_store=collection(c,method);m=manifest(c,scope)
    store=ArtifactStore(Path(c['output_base'])/'evaluation'/method/model/scope,new_base=c['output_base'])
    need(not (store.root/'FAIL').exists() and not store.pending(),'failed/pending evaluation cannot pass')
    store.verify('completed');store.verify('protocol');summary=read_json(store.root/'completed/summary.json')
    protocol=read_json(store.root/'protocol/protocol.json')
    need(protocol['configuration']==asdict(ref(path)) and protocol['scope']==scope and protocol['method']==method
         and protocol['model']==model and protocol['round']==60 and protocol['manifest']==plain(asdict(m))
         and protocol['model_space']==plain(asdict(source.space)) and protocol['scoring']==mod.SCORING
         and protocol['preprocess']==mod.PREPROCESS and protocol['effective_k_test']==(10 if scope=='smoke' else 128),'native protocol/source/scope identity')
    need(summary['status']=='GH_TRANSFER_EVALUATION_COMPLETE' and summary['model_frozen'] and summary['no_target_feedback']
         and all(summary[k]==0 for k in ('G_H_loads','G_H_forwards','IRRA_loads','backward','optimizer','JVP')),'frozen/no-training execution')
    allowed=[z for z in store.root.rglob('*.pt')]
    with cpu_restore_only(allowed),torch.no_grad():
        bound,original,origins=bindings_for(c,method,m,coll,source.space)
        cache=NativeFeatureCache(store,protocol,source.space,mod.validate_payload,chunk=16)
        for family,b in bound.items():
            for start_index in range(0,len(b.ordered_ids),16):
                end=min(start_index+16,len(b.ordered_ids));x=cache.load(family,b,start_index,end);del x
        ep=read_json(store.root/'result/protocol.json')
        need(ep['modes']==['Clean','Joint'] and ep['query_chunk']==128 and ep['model_space']==plain(asdict(source.space)),'formal complete native modes')
        for family,b in bound.items():
            d=ep['inputs'][family]
            need(d['inputs']==plain(asdict(b)) and d['origin']==(plain(asdict(origins[family])) if family in origins else None)
                 and d['original_sha256']==(list(original[b.modality]) if family in origins else []),'result complete input/weight binding')
        _,r,scores=load_result(store,'result',expected_protocol_sha256=digest(ep))
        need(summary['groups']==r['groups'],'summary/result agreement')
        nq,ng=len(m.queries),len(m.gallery)
        need(summary['neural_hook_counts']=={'image_samples':2*ng,'text_samples':2*nq,'fusion_pairs':2*nq*(10 if scope=='smoke' else 128)},'native full work counts')
        for mode,q,g in [('Clean','clean_text','clean_gallery'),('Joint','adv_text','adv_gallery')]:
            if model=='rasa':
                parts=[]
                for start_index in range(0,nq,128):
                    key=f'score-{q}--{g}-{start_index:06d}';md=store.verify(key);end=min(start_index+128,nq)
                    need(md['provenance']=={'space_id':source.space.id,'query':digest(ep['inputs'][q]),'gallery':digest(ep['inputs'][g]),
                         'start':start_index,'end':end,'protocol_sha256':cache.protocol_ref.sha256},'RaSa score binding')
                    x=torch.load(store.root/key/'score.pt',map_location='cpu',weights_only=True)
                    need(payload_digest(x)==read_json(store.root/key/'details.json')['score_sha256'],'RaSa score payload')
                    parts.append(x)
                actual=torch.cat(parts,0);del parts
            else:
                parts=[]
                for start_index in range(0,nq,128):
                    key=f'aptm-raw-{q}--{g}-{start_index:06d}';md=store.verify(key)
                    need(md['provenance']=={'model_space':source.space.id,'query':digest(ep['inputs'][q]),'gallery':digest(ep['inputs'][g]),
                         'protocol_sha256':cache.protocol_ref.sha256,'normalization':'complete bound Query/Gallery matrix',
                         'start':start_index,'end':min(start_index+128,nq)},'APTM raw score binding')
                    x=torch.load(store.root/key/'score.pt',map_location='cpu',weights_only=True)
                    need({k:payload_digest(v) for k,v in x.items()}==read_json(store.root/key/'details.json'),'APTM raw payload digest');parts.append(x)
                raw={k:torch.cat([v[k] for v in parts],0) for k in ('itm','global')};del parts
                a,b=raw['itm'],raw['global']
                need(torch.isfinite(a).all() and torch.isfinite(b).all() and a.max()>a.min() and b.max()>b.min(),'valid complete normalization')
                actual=(a-a.min())/(a.max()-a.min())+.002*((b-b.min())/(b.max()-b.min()));del raw,a,b
                key=f'aptm-final-{q}--{g}';store.verify(key)
                final=torch.load(store.root/key/'score.pt',map_location='cpu',weights_only=True)
                need(torch.equal(actual,final['final']),'APTM complete normalization exact');del final
            need(torch.equal(actual,scores[mode]),'persisted score equals every native block')
            metrics,_=ranking(actual,m.evaluation_identity())
            need(metrics==r['groups'][mode],'independent full CPU ranking exact');del actual
    count=protection(c,method)
    report={'status':'GH_TRANSFER_CPU_AUDIT_PASS','method':method,'model':model,'scope':scope,'round':60,
            'configuration_sha256':sha_file(path),'gallery':len(m.gallery),'query':len(m.queries),'groups':r['groups'],
            'native_ranking_exact':True,'all_native_score_blocks_exact':True,'historical_stat_preserved':count,
            'input_audit':asdict(ref(input_store.root/'completed/manifest.json')),
            'elapsed_seconds':time.monotonic()-start,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'CUDA':False}
    dest=ArtifactStore(store.root/'cpu_audit',new_base=c['output_base'])
    dest.publish('completed',{'report.json':json_bytes(report)},{'method':method,'model':model,'scope':scope},sources=(ref(path),ref(store.root/'completed/manifest.json')))
    print('TRANSFER_CPU_AUDIT_PASS',method,model,scope,flush=True);return report
