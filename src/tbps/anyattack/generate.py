"""从已封存Decoder原始输出CPU派生严格8/255图像；不加载或调用任何模型。"""
import time,resource,traceback
from dataclasses import asdict
from .core import *

def run(ctx):
    s=store_at(ctx,'generation');need(not s.root.exists(),'single eps8 derivation Attempt; existing output stops')
    t=time.monotonic();parent,old,old_audit=historical(ctx);parent_ref=ref(parent.root/'completed/manifest.json')
    protocol=source_protocol(ctx);p_sha=digest(protocol)
    s.publish('protocol',{'protocol.json':json_bytes(protocol)},{'kind':'strict_eps8_raw_noise_derivation'},
        sources=(ref(ctx['path']),FileRef(**ctx['c']['raw_parent_config']),parent_ref,old_audit))
    report={'status':'RUNNING','training':False,'fine_tuning':False,'backward':0,'optimizer':0,'JVP':0,
            'model_loads':0,'encoder_forwards':0,'decoder_forwards':0,'parent_generation':asdict(parent_ref),
            'generation_mode':'reuse_sealed_raw_noise','max_text_replacements_per_query':1,
            'text_budget_unit':'word','actual_text_replacements_per_query':0}
    try:
        with no_training(cpu=True):
            hashes=[];maximum=0
            for i,g in enumerate(ctx['manifest'].gallery):
                source,md,parent_image=historical_image(parent,i)
                need(md['carrier']==asdict(g) and md['reference']==asdict(ctx['full'].gallery[ctx['targets'][i]]),
                     'historical carrier/reference exact')
                x=bridge(source['original'],source['raw_noise']);x['embedding']=source['embedding'].clone()
                actual=float((x['adversarial'].double()-x['original'].double()).abs().max());maximum=max(maximum,actual)
                need(actual<=EPS,'strict8/255; no tolerance')
                details={'carrier':md['carrier'],'reference':md['reference'],'parent_manifest':asdict(parent_image),
                         'linf_double':actual,'payload_sha256':{k:payload_digest(v) for k,v in x.items()}}
                s.publish(key(i),{'image.pt':tensor_bytes(x),'details.json':json_bytes(details)},
                    {'protocol_sha256':p_sha,'index':i},sources=(ref(s.root/'protocol/manifest.json'),parent_image))
                hashes.append(details['payload_sha256']['adversarial']);del x,source
                if (i+1)%25==0 or i+1==len(ctx['manifest'].gallery):
                    print('ANYATTACK_EPS8_CPU_DERIVATION',i+1,'/',len(ctx['manifest'].gallery),flush=True)
            report.update(status='PASS',model_before=old['model_before'],model_after=old['model_after'],frozen=True,
                original_encoder_forwards_reused=len(hashes),original_decoder_forwards_reused=len(hashes),
                adversarial_sha256=hashes,max_linf_double=maximum,budget_tolerance=0,CUDA_initialized=torch.cuda.is_initialized())
    except BaseException as exc:
        report.update(status='FAIL',error=str(exc),traceback=traceback.format_exc());raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-t,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        s.publish('completed' if report['status']=='PASS' else 'FAIL',{'report.json':json_bytes(report)},
            {'kind':'eps8_cpu_derivation_execution'},sources=(ref(s.root/'protocol/manifest.json'),parent_ref))
    print('ANYATTACK_EPS8_DERIVATION_PASS',ctx['scope'],report['elapsed_seconds'],flush=True)
