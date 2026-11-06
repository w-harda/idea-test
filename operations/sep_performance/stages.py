"""对首条固定真实Query作独立阶段/跨设备API计数，不计入主A/B速度。"""
from contextlib import ExitStack
from unittest.mock import patch
from pathlib import Path
import argparse,json,time
import torch,nltk
from operations.sep_text_only.runtime import config,source_inputs,store,source_guard
from operations.sep_source_only.common import train_manifest
from operations.sep_text_only.word_budget import resources
from tbps.baselines.cpgc_sep import sep_source
from tbps.baselines.cpgc_sep.sep_bridge import PIDPlan
from tbps.baselines.cpgc_sep.sep_packed_pid import PackedPIDPlan,enable_packed_pid
from tbps.baselines.cpgc_sep.cache import NativeScorer
from tbps.baselines.cpgc_sep.victims import load_victim
from .ab import OUT,CTL,run_query,canonical,exact

def main(warmup=False,attempt="stages"):
    torch.set_num_threads(2)
    with (OUT/(attempt+'_started.json')).open('x') as f:json.dump({'time':time.time()},f)
    item=json.loads((OUT/'preregister.json').read_text())['samples'][0]
    c,_=config();bundle=source_inputs(c);ref=store(c,'source/reference');ref.verify('completed')
    gallery=torch.load(ref.root/'completed/features.pt',map_location='cpu',weights_only=True)
    manifest=train_manifest(bundle['train_gallery']);cfg=bundle['config']
    nltk.data.path[:]=[cfg['nltk_root']];vocabulary=resources(cfg)
    c={**c,'output':str(OUT),'control':str(CTL)};source_guard(c,bundle)
    results={};rows=[]
    with load_victim(cfg,'irra','cuhk',smoke=False) as (adapter,model,src):
        for name in ('original','packed'):
            predictor=sep_source.classifier(adapter,model,'irra',sep_source.EphemeralQuery(gallery),manifest)
            if name=='packed':enable_packed_pid(predictor)
            if warmup:
                predictor([item['caption']]);predictor.cache.clear()
            stats={'stages':{},'cross_device_calls':{},'cross_device_bytes':{}}
            def timed(label,fn):
                def call(*a,**kw):
                    start=time.perf_counter()
                    try:return fn(*a,**kw)
                    finally:
                        record=stats['stages'].setdefault(label,{'calls':0,'seconds':0.})
                        record['calls']+=1;record['seconds']+=time.perf_counter()-start
                return call
            def transport(fn):
                def call(self,*a,**kw):
                    result=fn(self,*a,**kw)
                    if self.device.type!=result.device.type:
                        key=self.device.type+'->'+result.device.type
                        stats['cross_device_calls'][key]=stats['cross_device_calls'].get(key,0)+1
                        stats['cross_device_bytes'][key]=stats['cross_device_bytes'].get(key,0)+self.numel()*self.element_size()
                    return result
                return call
            with ExitStack() as stack:
                stack.enter_context(patch.object(sep_source,'encode_texts',timed('encode_texts',sep_source.encode_texts)))
                stack.enter_context(patch.object(NativeScorer,'_compute',timed('native_matrix',NativeScorer._compute)))
                plan_type=PIDPlan if name=='original' else PackedPIDPlan
                stack.enter_context(patch.object(plan_type,'aggregate',timed('pid_aggregate',plan_type.aggregate)))
                for method in ('to','cpu','cuda'):
                    stack.enter_context(patch.object(torch.Tensor,method,transport(getattr(torch.Tensor,method))))
                record=run_query(cfg,item,predictor,vocabulary,lambda *_a,**_k:None)
            stats.update(query_seconds=record['row']['elapsed_seconds'],forwards=record['row']['text_forwards'],
                         mutation_checks=record['row']['mutation_checks'],instrumentation_included=True)
            results[name]=stats;rows.append(record)
    assert canonical(rows[0]['row'])==canonical(rows[1]['row'])
    assert exact(rows[0]['logits'],rows[1]['logits']) and rows[0]['rng_after']==rows[1]['rng_after']
    (CTL/(attempt+'_result.json')).write_text(json.dumps({'status':'PASS','warmup':warmup,'results':results,'scope':'first preregistered CUHK caption; Python-level cross-device API transitions, not all C++ internal memcpy'},indent=2))
    print(json.dumps(results,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--warmup',action='store_true');parser.add_argument('--attempt',default='stages')
    args=parser.parse_args()
    assert args.attempt.replace('_','').isalnum()
    try:main(args.warmup,args.attempt)
    except BaseException:
        import traceback
        (CTL/(args.attempt+'_failure.txt')).write_text(traceback.format_exc())
        raise
