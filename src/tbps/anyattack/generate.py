"""官方冻结CLIP/Decoder一次推理，不提供训练或下游微调入口。"""
import importlib, sys, time, resource
from dataclasses import asdict
import torch
from tbps.artifacts import json_bytes, digest, FileRef
from tbps.models.irra import fp32_runtime
from tbps.smoke_irra import model_snapshot
from tbps.engines.materialized_irra import gpu_available
from .core import *

def run(ctx):
    s=store_at(ctx,'generation');need(not s.root.exists(),'single Attempt generation; existing output stops')
    c=ctx['c'];m=ctx['manifest'];start=time.monotonic()
    s.publish('protocol',{'protocol.json':json_bytes(source_protocol(ctx))},{'kind':'anyattack_source'},
        sources=(ref(ctx['path']),FileRef(**c['decoder']),FileRef(**c['clip'])))
    print('GPU_BEFORE',gpu_available({'physical_gpu':0}),flush=True)
    sys.path.insert(0,c['runtime_root'])
    module=importlib.import_module('models.model')
    need(str(Path(module.__file__).resolve())==c['runtime_root']+'/models/model.py','official runtime module provenance')
    torch.set_num_threads(2);torch.manual_seed(c['seed'])
    report={'status':'RUNNING','training':False,'fine_tuning':False,'backward':0,'optimizer':0,'JVP':0}
    try:
        with no_training(),fp32_runtime():
            encoder=module.CLIPEncoder(model=c['clip']['path']).float().eval().requires_grad_(False)
            decoder=module.Decoder(embed_dim=512).float().eval().requires_grad_(False)
            checkpoint=torch.load(c['decoder']['path'],map_location='cpu',weights_only=True)
            need(type(checkpoint)is dict and 'decoder_state_dict' in checkpoint,'official Decoder checkpoint payload')
            state=checkpoint['decoder_state_dict'];keys=list(state)
            prefixed=all(k.startswith('module.') for k in keys)
            need(prefixed or not any(k.startswith('module.') for k in keys),'mixed Decoder DataParallel prefix')
            if prefixed:state={k[7:]:v for k,v in state.items()}
            decoder.load_state_dict(state,strict=True);del state,checkpoint
            before={'CLIP':model_snapshot(encoder),'Decoder':model_snapshot(decoder)}
            encoder.to('cuda:0');decoder.to('cuda:0')
            counts={'encoder':0,'decoder':0}
            def inc(name):
                def hook(*a):counts[name]+=1
                return hook
            hooks=[encoder.model.visual.register_forward_hook(inc('encoder')),decoder.register_forward_hook(inc('decoder'))]
            hashes=[];maximum=0;protocol_sha=digest(source_protocol(ctx))
            for i,g in enumerate(m.gallery):
                target=ctx['full'].gallery[ctx['targets'][i]]
                original=pixels(ctx['ds'].image_path(g.image_rel_path),(384,128))
                reference=pixels(ctx['ds'].image_path(target.image_rel_path),(224,224))
                embedding=encoder.encode_img(reference.to('cuda:0'))
                raw=decoder(embedding).float().cpu().clone();x=bridge(original,raw)
                x['embedding']=embedding.float().cpu().clone()
                linf=float((x['adversarial'].double()-original.double()).abs().max());maximum=max(maximum,linf)
                need(linf<=EPS+torch.finfo(torch.float32).eps,'FP32 budget exceeded')
                md={'carrier':asdict(g),'reference':asdict(target),'linf_double':linf,
                    'payload_sha256':{k:payload_digest(v) for k,v in x.items()}}
                s.publish(key(i),{'image.pt':tensor_bytes(x),'details.json':json_bytes(md)},
                    {'protocol_sha256':protocol_sha,'index':i},sources=(ref(s.root/'protocol/manifest.json'),
                     FileRef(str(ctx['ds'].image_path(g.image_rel_path)),g.image_sha256),
                     FileRef(str(ctx['ds'].image_path(target.image_rel_path)),target.image_sha256)))
                hashes.append(md['payload_sha256']['adversarial'])
                del x,embedding,raw,reference,original
                if (i+1)%25==0 or i+1==len(m.gallery):print('ANYATTACK_GENERATION',i+1,'/',len(m.gallery),flush=True)
            for h in hooks:h.remove()
            after={'CLIP':model_snapshot(encoder),'Decoder':model_snapshot(decoder)}
            need(before==after and counts=={'encoder':len(m.gallery),'decoder':len(m.gallery)},'source frozen state/counts')
            need(all(not p.requires_grad and p.grad is None for model in (encoder,decoder) for p in model.parameters()),'no source parameter gradients')
            report.update(status='PASS',model_before=before,model_after=after,frozen=True,
                encoder_forwards=counts['encoder'],decoder_forwards=counts['decoder'],adversarial_sha256=hashes,
                decoder_prefix_stripped=prefixed,max_linf_double=maximum)
    except BaseException as exc:
        import traceback
        report.update(status='FAIL',error=str(exc),traceback=traceback.format_exc());raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        if torch.cuda.is_initialized():report.update(gpu_peak_allocated=torch.cuda.max_memory_allocated(),gpu_peak_reserved=torch.cuda.max_memory_reserved())
        s.publish('completed' if report['status']=='PASS' else 'FAIL',{'report.json':json_bytes(report)},
            {'kind':'anyattack_generation_execution'},sources=(ref(s.root/'protocol/manifest.json'),))
    print('ANYATTACK_GENERATION_PASS',ctx['scope'],report['elapsed_seconds'],flush=True)
