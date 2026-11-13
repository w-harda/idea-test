"""原生RaSa/APTM编码与Clean/Joint评分，禁止训练或目标反馈。"""
from dataclasses import asdict
from pathlib import Path
import time,resource
from unittest.mock import patch
from .common import *

def target(c,model):
    need(model in MODELS,'known target')
    from importlib import import_module
    module=import_module('tbps.models.'+model)
    cls=module.RasaSource if model=='rasa' else module.APTMSource
    source=cls(c['targets'][model]['path']);source.verify()
    return module,source

def preflight(path,model):
    import torch
    from .startup import bootstrap
    bootstrap()
    from tbps.alternating.parents import cpu_restore_only
    c=config(path)
    with cpu_restore_only(()):
        mod,s=target(c,model)
        with mod.native_modules(s) as (_,tok,utils):
            tokenizer=mod.official_tokenizer(tok,s)
            n=50 if model=='rasa' else 56
            batch=tokenizer([utils.pre_caption('A man wearing a blυe shirt.',n)],padding='max_length',truncation=True,max_length=n,return_tensors='pt')
            need(batch.input_ids.shape==(1,n),'native tokenizer closure')
    need(not torch.cuda.is_initialized(),'CPU preflight initialized CUDA')
    print('TRANSFER_TARGET_PREFLIGHT_PASS',model,flush=True)
    return c,mod,s

def bindings_for(c,method,m,coll,space):
    from tbps.models.base import InputBinding
    from tbps.engines.provenance import AttackOrigin
    from tbps.contracts import AttackBudget,ImageBudget,TextBudget
    h=c['historical'][method];mh=digest(asdict(m));identity=m.evaluation_identity()
    ims=coll['images'][:len(m.gallery)];txs=coll['texts'][:len(m.queries)]
    need([v['id'] for v in ims]==list(identity.gallery_ids) and [v['id'] for v in txs]==list(identity.query_ids),'collection ordered membership')
    original={'image':tuple(v['original_sha256'] for v in ims),'text':tuple(v['original_sha256'] for v in txs)}
    bound={}
    for family,items,modality,ids in [('clean_gallery',ims,'image',identity.gallery_ids),('adv_gallery',ims,'image',identity.gallery_ids),
                                     ('clean_text',txs,'text',identity.query_ids),('adv_text',txs,'text',identity.query_ids)]:
        shas=tuple(v['adversarial_sha256' if family.startswith('adv') else 'original_sha256'] for v in items)
        bound[family]=InputBinding(space.id,mh,shas,modality,ids)
    origins={
      'adv_gallery':AttackOrigin(method+'-g-round60',FileRef(**h['G']),FileRef(**h['configuration']),
                    AttackBudget(image=ImageBudget('linf',8/255,'RGB [0,1]')),h['configuration']['sha256']),
      'adv_text':AttackOrigin(method+'-h-round60',FileRef(**h['H']),FileRef(**h['configuration']),
                    AttackBudget(text=TextBudget('unicode_character',None,'original Natural-K gamma3/beta1','original H-v1 constraints',1,1)),h['configuration']['sha256'])}
    return bound,original,origins

def bound_inputs(cache,bound,original,origins):
    return {f:cache.gather(f,b,origin=origins.get(f),originals=original[b.modality] if f in origins else ()) for f,b in bound.items()}

def run(path,method,model,scope,*,allow=False):
    need(allow,'explicit native GPU evaluation authorization')
    import torch
    from tbps.materialization.irra_startup import CPUStartupGate
    from tbps.engines.native_cache import NativeFeatureCache
    from tbps.engines.native_score_cache import ChunkCachedScorer
    from tbps.engines.aptm_score_cache import APTMSetScorer
    from tbps.engines.evaluation import EvaluationEngine,publish_result
    from tbps.models.irra import fp32_runtime
    from tbps.alternating.parents import no_optimizer_steps
    from tbps.smoke_irra import model_snapshot
    torch.set_num_threads(2);torch.manual_seed(42)
    from .startup import prepare
    c,mod,source,coll,inputs_store,m,gate,startup=prepare(path,method,model,scope)
    need(__import__('os').environ.get('CUDA_VISIBLE_DEVICES')=='0','explicit physical GPU0')
    store=ArtifactStore(Path(c['output_base'])/'evaluation'/method/model/scope,new_base=c['output_base'])
    need(not store.root.exists(),'evaluation Attempt consumed; no resume/retry')
    store.publish('claim',{'configuration.json':json_bytes(asdict(ref(path)))},{'method':method,'model':model,'scope':scope},
                  sources=(ref(path),ref(inputs_store.root/'completed/manifest.json'),source.ref,source.checkpoint))
    start=time.monotonic()
    report={'method':method,'model':model,'scope':scope,'G_H_loads':0,'G_H_forwards':0,'IRRA_loads':0,'backward':0,'optimizer':0,'JVP':0,'no_target_feedback':True,'startup':startup}
    def forbidden(*a,**k):raise ArtifactError('cross-model evaluation forbids derivative')
    try:
        gate.initialize_cuda()
        loader=mod.load_rasa if model=='rasa' else mod.load_aptm
        with fp32_runtime(),no_optimizer_steps(),patch.object(torch.autograd,'grad',forbidden),patch.object(torch.autograd,'backward',forbidden),loader(source,smoke=scope=='smoke') as adapter:
            before=model_snapshot(adapter.backend)
            bound,original,origins=bindings_for(c,method,m,coll,adapter.space)
            protocol={'schema':'gh-transfer-native-cache-1','configuration':asdict(ref(path)),'method':method,'model':model,
                      'round':60,'scope':scope,'manifest':asdict(m),'model_space':asdict(adapter.space),
                      'input_audit':asdict(ref(inputs_store.root/'completed/manifest.json')),'preprocess':mod.PREPROCESS,'scoring':mod.SCORING,
                      'attack_feedback':False,'new_attack_generation':False,'effective_k_test':10 if scope=='smoke' else 128}
            store.publish('protocol',{'protocol.json':json_bytes(protocol)},{'model_space':adapter.space.id},sources=(ref(path),source.ref,source.checkpoint,ref(inputs_store.root/'completed/manifest.json')))
            cache=NativeFeatureCache(store,protocol,adapter.space,mod.validate_payload,chunk=16)
            token_changed=0;raw_changed=0
            for family,b in bound.items():
                items=coll['images'] if family.endswith('gallery') else coll['texts']
                for start_index in range(0,len(b.ordered_ids),16):
                    end=min(start_index+16,len(b.ordered_ids));rows=[]
                    for item in items[start_index:end]:
                        if family.endswith('gallery'):
                            raw=image_value(item);pixels=raw['adversarial' if family.startswith('adv') else 'original']
                            rows.append(adapter.encode_images(pixels));del raw,pixels
                        else:
                            a=text_value(item);name='adversarial' if family.startswith('adv') else 'original'
                            if name=='adversarial':
                                t=adapter.tokenize((a['original'],a['adversarial'])).input_ids
                                token_changed+=int(not torch.equal(t[0],t[1]));raw_changed+=int(a['original']!=a['adversarial'])
                            rows.append(adapter.encode_texts((a[name],)))
                    x={k:torch.cat([r[k] for r in rows],0) for k in rows[0]}
                    cache.publish(family,b,start_index,end,x);del rows,x
                    print('TRANSFER_FEATURE_PROGRESS',method,model,scope,family,end,len(b.ordered_ids),flush=True)
            values=bound_inputs(cache,bound,original,origins)
            scorer=(ChunkCachedScorer if model=='rasa' else APTMSetScorer)(adapter,values,store,cache.protocol_ref)
            engine=EvaluationEngine(m,scorer,direction='higher_better',execution='materialized_native',model_sources=(source.checkpoint,source.ref))
            ep,result,scores=engine.run(values,modes=('Clean','Joint'),query_chunk=128)
            publish_result(store,'result',ep,result,scores,sources=(cache.protocol_ref,ref(inputs_store.root/'completed/manifest.json')))
            need(before==model_snapshot(adapter.backend),'target parameters/buffers changed')
            need(all(adapter.counts[k]==adapter.neural_counts[k] for k in adapter.neural_counts),'actual neural hooks mismatch')
            nq,ng=len(m.queries),len(m.gallery)
            need(adapter.neural_counts=={'image_samples':2*ng,'text_samples':2*nq,'fusion_pairs':2*nq*(10 if scope=='smoke' else 128)},'full native Top128 work accounting')
            report.update(status='GH_TRANSFER_EVALUATION_COMPLETE',model_frozen=True,counts=dict(adapter.counts),neural_hook_counts=dict(adapter.neural_counts),
                          raw_changed_queries=raw_changed,target_token_changed_queries=token_changed,
                          model_before=before,model_after=before,gallery=ng,query=nq,groups=result['groups'],
                          elapsed_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                          GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            store.publish('completed',{'summary.json':json_bytes(report)}, {'method':method,'model':model,'scope':scope},
                          sources=(ref(store.root/'result/manifest.json'),ref(store.root/'protocol/manifest.json')))
        print('TRANSFER_EVALUATION_COMPLETE',method,model,scope,flush=True);return report
    except BaseException:
        import traceback
        (store.root/'FAIL').write_text(traceback.format_exc());raise
