"""独立真实Source调用链诊断；正式Full保持只读。"""
from pathlib import Path
import cProfile,hashlib,io,json,os,pstats,sys,time
import torch,nltk
from operations.sep_text_only.runtime import config,source_inputs,store,selected_source,source_guard
from operations.sep_source_only.common import train_manifest
from operations.sep_text_only.word_budget import resources,attack_caption
from tbps.baselines.cpgc_sep.sep_source import classifier,EphemeralQuery
from tbps.baselines.cpgc_sep.victims import load_victim
from tbps.artifacts import sha_file

OUT=Path('/home/lzf/ldx/outputs/tbps-sep-performance-20261106')
CTL=Path('/home/lzf/ldx/tmp/sep_performance_20261106')

def main():
    torch.set_num_threads(2)
    c,_=config();bundle=source_inputs(c)
    samples=[dict(x,dataset=ds) for ds in ('cuhk','rstp','icfg') for x in selected_source(c,ds,'full')[:2]]
    assert all(set(x)=={'key','caption','dataset'} for x in samples)
    with (OUT/'preregister.json').open('x') as f:json.dump({'selection':'first two official caption-only rows of each dataset; no attack-feedback selection','samples':samples},f,ensure_ascii=False,indent=2)
    ref=store(c,'source/reference');ref.verify('completed')
    gallery=torch.load(ref.root/'completed/features.pt',map_location='cpu',weights_only=True)
    assert list(gallery.shape)==[34054,512]
    manifest=train_manifest(bundle['train_gallery'])
    cfg=bundle['config'];nltk.data.path[:]=[cfg['nltk_root']]
    vocabulary=resources(cfg)
    c={**c,'output':str(OUT),'control':str(CTL)}
    source_guard(c,bundle)
    with load_victim(cfg,'irra','cuhk',smoke=False) as (adapter,model,src):
        predictor=classifier(adapter,model,'irra',EphemeralQuery(gallery),manifest)
        item=samples[0]
        import random,numpy as np
        seed=int(item['key'][:8],16)
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        events=[]
        profiler=cProfile.Profile()
        profiler.enable()
        row,logits=attack_caption(cfg,item['caption'],predictor,'cuda:0',vocabulary,
                                 lambda event,**fields:events.append({'event':event,**fields}))
        profiler.disable()
        profiler.dump_stats(str(CTL/'baseline_query.cprof'))
        stream=io.StringIO();pstats.Stats(profiler,stream=stream).sort_stats('cumulative').print_stats(65)
        (CTL/'baseline_query_profile.txt').write_text(stream.getvalue())
        (CTL/'baseline_diagnostic_row.json').write_text(json.dumps(row,ensure_ascii=False))
        torch.save({'logits':logits,'candidate_cache':dict(predictor.cache)},OUT/'source/diagnostic_frozen_scores.pt')
        predictor.cache.clear()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
            with torch.profiler.record_function('REAL_SOURCE_CLASSIFIER_SINGLE_CALL'):
                value=predictor([item['caption']]).logits
        p.export_chrome_trace(str(CTL/'baseline_single_call_trace.json'))
        (CTL/'baseline_torch_profile.txt').write_text(p.key_averages().table(sort_by='cpu_time_total',row_limit=50))
        result={'status':'PASS','checkpoint':{'path':src.checkpoint.path,'sha256':src.checkpoint.sha256},
                'gallery':list(gallery.shape),'pids':len(predictor.pid_plan.order),
                'query_elapsed_cprofile_seconds':row['elapsed_seconds'],'calls':row['calls'],'forwards':row['text_forwards'],
                'cache_hits':row['cache_hits'],'mutation_checks':row['mutation_checks'],
                'trace_callback_stats':[{'file':k[0],'line':k[1],'function':k[2],'calls':v[1],'own_seconds':v[2],'cumulative_seconds':v[3]}
                                          for k,v in pstats.Stats(profiler).stats.items() if k[2] in ('trace','aggregate','encode_texts','_compute','budget_check','observed','attack_caption')],
                'torch_profile':[{'key':e.key,'count':e.count,'cpu_us':e.cpu_time_total,'self_cpu_us':e.self_cpu_time_total,
                                  'device_us':e.device_time_total} for e in p.key_averages() if e.key in ('aten::to','aten::_to_copy','aten::copy_','aten::max','aten::mm','aten::index','cudaMemcpyAsync','cudaStreamSynchronize')]}
        (CTL/'diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
