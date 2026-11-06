"""固定六条真实Caption的原入口/显式PackedPIDPlan A/B，不接管Full。"""
from pathlib import Path
from threading import Event,Thread
from unittest.mock import patch
import hashlib,json,os,random,sys,time
import numpy as np
import torch,nltk
from operations.sep_text_only.runtime import config,source_inputs,store,source_guard
from operations.sep_source_only.common import train_manifest
from operations.sep_text_only.word_budget import resources,attack_caption,budget_check
from tbps.baselines.cpgc_sep.sep_source import classifier,EphemeralQuery
from tbps.baselines.cpgc_sep.sep_bridge import PIDPlan
from tbps.baselines.cpgc_sep.sep_packed_pid import PackedPIDPlan,enable_packed_pid
from tbps.baselines.cpgc_sep.victims import load_victim,encode_texts,frozen,snapshot
from tbps.baselines.cpgc_sep.cache import NativeScorer
from tbps.models.base import RepresentationBatch
from .monitor import proc,host,gpu,summarize

OUT=Path('/home/lzf/ldx/outputs/tbps-sep-performance-20261106')
CTL=Path('/home/lzf/ldx/tmp/sep_performance_20261106')
def exact(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
def rng():
    n=np.random.get_state()
    b=repr(random.getstate()).encode()+repr((n[0],n[2:])).encode()+n[1].tobytes()+torch.get_rng_state().numpy().tobytes()
    for x in torch.cuda.get_rng_state_all():b+=x.cpu().numpy().tobytes()
    return hashlib.sha256(b).hexdigest()
def canonical(row):
    return {k:v for k,v in row.items() if k not in ('elapsed_seconds','timings')}
class Sampling:
    def __init__(self):self.samples=[];self.end=Event()
    def run(self):
        while not self.end.is_set():
            t=time.monotonic()
            self.samples.append({'time':t,'wall_time':time.time(),'process':proc(os.getpid()),'host':host(),'gpu':gpu()})
            self.end.wait(max(0,.2-(time.monotonic()-t)))
    def __enter__(self):
        self.thread=Thread(target=self.run,daemon=True);self.thread.start();return self
    def __exit__(self,*args):self.end.set();self.thread.join()
def run_query(cfg,item,predictor,vocabulary,log):
    seed=int(item['key'][:8],16);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    before=rng()
    row,logits=attack_caption(cfg,item['caption'],predictor,'cuda:0',vocabulary,log)
    after=rng()
    for request in row['requests']:
        for text in request['texts']:budget_check(item['caption'],text)
    for text in [row['adversarial']]+[x['text'] for x in row['candidates']]:budget_check(item['caption'],text)
    record={'item':item,'seed':seed,'row':row,'logits':logits,'rng_before':before,'rng_after':after,
            'cache':dict(predictor.cache),'cache_keys':list(predictor.cache)}
    predictor.cache.clear()
    return record
def arm_summary(records,samples,seconds):
    durations=[r['row']['elapsed_seconds'] for r in records]
    return {'mean_query_seconds':float(np.mean(durations)),'p50_seconds':float(np.percentile(durations,50)),
            'p95_seconds':float(np.percentile(durations,95)),'query_seconds':durations,
            'query_per_second':len(records)/sum(durations),'arm_wall_seconds':seconds,
            'model_forwards':sum(r['row']['text_forwards'] for r in records),
            'calls':sum(r['row']['calls'] for r in records),'hits':sum(r['row']['cache_hits'] for r in records),
            'violations':0,'monitor':summarize(samples)}
def main():
    torch.set_num_threads(2)
    with (OUT/'ab_started.json').open('x') as f:json.dump({'pid':os.getpid(),'time':time.time()},f)
    samples=json.loads((OUT/'preregister.json').read_text())['samples']
    c,_=config();bundle=source_inputs(c)
    ref=store(c,'source/reference');ref.verify('completed')
    gallery=torch.load(ref.root/'completed/features.pt',map_location='cpu',weights_only=True)
    manifest=train_manifest(bundle['train_gallery']);cfg=bundle['config']
    nltk.data.path[:]=[cfg['nltk_root']];vocabulary=resources(cfg)
    c={**c,'output':str(OUT),'control':str(CTL)};source_guard(c,bundle)
    with load_victim(cfg,'irra','cuhk',smoke=False) as (adapter,model,src):
        model_before=snapshot(model)
        bank=EphemeralQuery(gallery)
        predictor=classifier(adapter,model,'irra',bank,manifest)
        # 预热不属于两臂攻击，之后清空cache并逐Query重置相同seed。
        predictor([samples[0]['caption']]);predictor.cache.clear()
        records={};summaries={}
        for name in ('original','packed'):
            if name=='packed':enable_packed_pid(predictor)
            torch.cuda.reset_peak_memory_stats()
            start=time.monotonic()
            log_path=CTL/(name+'_events.jsonl')
            with log_path.open('x') as events:
                def report(event,**fields):
                    events.write(json.dumps({'event':event,**fields})+'\n');events.flush()
                with Sampling() as sampling:
                    current=[]
                    for item in samples:
                        current.append(run_query(cfg,item,predictor,vocabulary,report))
                        print('QUERY_DONE',name,item['dataset'],item['key'],current[-1]['row']['elapsed_seconds'],flush=True)
            seconds=time.monotonic()-start
            records[name]=current
            summaries[name]=arm_summary(current,sampling.samples,seconds)
            summaries[name]['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
            summaries[name]['peak_cuda_reserved_bytes']=torch.cuda.max_memory_reserved()
            torch.save(current,OUT/'source'/(name+'_records.pt'))
            (CTL/(name+'_samples.jsonl')).write_text(''.join(json.dumps(s)+'\n' for s in sampling.samples))
        checks=[]
        for a,b in zip(records['original'],records['packed']):
            check={'dataset':a['item']['dataset'],'key':a['item']['key'],
                   'row_exact_excluding_time':canonical(a['row'])==canonical(b['row']),
                   'logits_bitwise':exact(a['logits'],b['logits']),'cache_key_order':a['cache_keys']==b['cache_keys'],
                   'cache_values_bitwise':a['cache_keys']==b['cache_keys'] and all(exact(a['cache'][k],b['cache'][k]) for k in a['cache_keys']),
                   'rng_before':a['rng_before']==b['rng_before'],'rng_after':a['rng_after']==b['rng_after']}
            checks.append(check)
            assert all(v for k,v in check.items() if k not in ('dataset','key')),check
        # 同一真实输入各一次Profiler，不计入A/B时间。
        profiles={}
        for name in ('original','packed'):
            predictor.pid_plan=PIDPlan.build(tuple(g.pid for g in manifest.gallery))
            if name=='packed':enable_packed_pid(predictor)
            predictor.cache.clear()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
                with torch.profiler.record_function('REAL_CLASSIFIER_SINGLE_CALL'):
                    predictor([samples[0]['caption']])
            p.export_chrome_trace(str(CTL/(name+'_single_call.json')))
            (CTL/(name+'_torch_profile.txt')).write_text(p.key_averages().table(sort_by='cpu_time_total',row_limit=40))
            profiles[name]=[{'key':e.key,'count':e.count,'cpu_us':e.cpu_time_total,'self_cpu_us':e.self_cpu_time_total,'device_us':e.device_time_total}
                            for e in p.key_averages() if e.key in ('aten::max','aten::index','aten::to','aten::_to_copy','cudaMemcpyAsync','cudaStreamSynchronize')]
        # 仅独立原型评估Batch8风险；不用于上面的Packed A/B攻击。
        texts=records['original'][0]['cache_keys'][:8]
        sequential=torch.cat([encode_texts(adapter,[t],'irra') for t in texts])
        batched=encode_texts(adapter,texts,'irra')
        qs=[]
        for features in (sequential,batched):
            qb=EphemeralQuery(features);scorer=NativeScorer(adapter,'irra',{'q':qb,'g':bank})
            q=RepresentationBatch(qb.globals,adapter.space.id,'text');g=RepresentationBatch(bank.globals,adapter.space.id,'image')
            qs.append(scorer.score(scorer.slice_query(q,0,len(texts)),g).payload)
        batching={'batch':len(texts),'encoder_bitwise':exact(sequential,batched),
                  'encoder_max_abs':float((sequential-batched).abs().max()),
                  'native_matrix_bitwise':exact(qs[0],qs[1]),'native_matrix_max_abs':float((qs[0]-qs[1]).abs().max())}
        original_plan=PIDPlan.build(tuple(g.pid for g in manifest.gallery));packed=PackedPIDPlan(original_plan)
        frozen_scores=qs[0][:1].contiguous()
        pa=original_plan.aggregate(frozen_scores)[0]
        pb=packed.aggregate(frozen_scores)[0]
        pc=packed.aggregate(frozen_scores.to('cuda:0'))[0].cpu()
        aggregate_checks={'real_cpu_bitwise':exact(pa,pb),'real_cuda_bitwise':exact(pa,pc),
                          'shape':list(frozen_scores.shape),'packed_width':int(packed.indices.shape[1])}
        assert all(aggregate_checks[k] for k in ('real_cpu_bitwise','real_cuda_bitwise'))
        bench={}
        for name,plan in [('original',original_plan),('packed',packed)]:
            values=[]
            for i in range(12):
                start=time.perf_counter();plan.aggregate(frozen_scores);values.append(time.perf_counter()-start)
            bench[name]={'median_seconds':float(np.median(values)),'samples':values}
        # 预算Trace回调显式计时，仅作独立诊断，不改变预算验证。
        predictor.pid_plan=PackedPIDPlan(original_plan);predictor.cache.clear()
        trace_stats={'callbacks':0,'callback_seconds':0.}
        real_settrace=sys.settrace
        def install(fn):
            if fn is None:return real_settrace(None)
            def observed(frame,event,arg):
                start=time.perf_counter()
                result=fn(frame,event,arg)
                trace_stats['callbacks']+=1;trace_stats['callback_seconds']+=time.perf_counter()-start
                return observed if result is fn else result
            return real_settrace(observed)
        with patch.object(sys,'settrace',install):
            timed=run_query(cfg,samples[0],predictor,vocabulary,lambda *_a,**_k:None)
        trace_stats['query_seconds']=timed['row']['elapsed_seconds']
        trace_stats['mutation_checks']=timed['row']['mutation_checks']
        trace_stats['instrumentation_included']=True
        frozen(model);assert snapshot(model)==model_before,'model changed'
        result={'status':'PASS','arms':summaries,'exact_checks':checks,'profiles':profiles,'batching_prototype':batching,
                'real_pid_checks':aggregate_checks,'pid_microbenchmark':bench,'trace_callback_diagnostic':trace_stats,
                'checkpoint':{'path':src.checkpoint.path,'sha256':src.checkpoint.sha256},
                'speedup':summaries['original']['mean_query_seconds']/summaries['packed']['mean_query_seconds'],
                'model_unchanged':True,'GPU_OOM':False}
        (CTL/'ab_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        (CTL/'ab_failure.txt').write_text(traceback.format_exc())
        raise
