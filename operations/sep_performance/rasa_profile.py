"""独立复用已审计Smoke特征分析原生RaSa Top128/ITM，不发布实验指标。"""
from pathlib import Path
from dataclasses import asdict
import json,time
import torch
from operations.sep_text_only.runtime import config,target_manifest,store
from tbps.baselines.cpgc_sep.victims import load_victim,snapshot,frozen
from tbps.baselines.cpgc_sep.cache import FeatureBank,NativeScorer
from tbps.models.base import RepresentationBatch
from tbps.artifacts import digest,sha_file
from .ab import exact,CTL,OUT

def main():
    torch.set_num_threads(2)
    with (OUT/'rasa_profile_started.json').open('x') as f:json.dump({'time':time.time()},f)
    c,base=config();manifest=target_manifest(c,'cuhk','smoke');old=store(c,'smoke/target/cuhk/rasa')
    old.verify('completed')
    summary=json.loads((old.root/'completed/summary.json').read_text())
    reference=torch.load(old.root/'completed/scores.pt',map_location='cpu',weights_only=True)
    cfg={**base,'sha256':sha_file(Path(__file__).resolve().parents[2]/'configs/framework/sep_text_only_full.server.json'),
         'evaluation':{**base['evaluation'],'fusion_batch':c['fusion_batch']},'output_base':c['output']}
    with load_victim(cfg,'rasa','cuhk',smoke=True) as (adapter,model,src):
        before=snapshot(model)
        g=FeatureBank(cfg,old,'clean_gallery',adapter,manifest,tuple(x.image_id for x in manifest.gallery),
                      tuple(x.image_sha256 for x in manifest.gallery),modality='image',chunk=8)
        q=FeatureBank(cfg,old,'adv_text',adapter,manifest,tuple(x.query_id for x in manifest.queries),
                      tuple(summary['frozen_text_sha256']),modality='text',chunk=8)
        g.read_globals();q.read_globals()
        scorer=NativeScorer(adapter,'rasa',{'q':q,'g':g})
        qb=RepresentationBatch(q.globals,adapter.space.id,'text');gb=RepresentationBatch(g.globals,adapter.space.id,'image')
        fusion={'calls':0,'seconds':0.}
        original=adapter.score
        def timed(*args,**kwargs):
            t=time.perf_counter();value=original(*args,**kwargs)
            fusion['calls']+=1;fusion['seconds']+=time.perf_counter()-t
            return value
        adapter.score=timed
        start=time.perf_counter()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
            with torch.profiler.record_function('RASA_NATIVE_TOP128_SMOKE_8_QUERY'):
                value=scorer.score(scorer.slice_query(qb,0,len(q.ids)),gb).payload
        elapsed=time.perf_counter()-start
        p.export_chrome_trace(str(CTL/'rasa_trace.json'))
        (CTL/'rasa_torch_profile.txt').write_text(p.key_averages().table(sort_by='cpu_time_total',row_limit=45))
        frozen(model);assert snapshot(model)==before
        same=exact(value,reference)
        result={'status':'PASS' if same else 'NONEXACT_DIAGNOSTIC','scope':'independent previously audited Smoke features; not a new Full',
                'queries':len(q.ids),'gallery':len(g.ids),'fusion_batch':adapter.fusion_batch,
                'native_ITM_pairs':scorer.native_pairs,'expected_ITM_pairs':len(q.ids)*128,
                'scores_bitwise_against_frozen_smoke':same,'max_abs_diff':float((value-reference).abs().max()),
                'profiler_inclusive_seconds':elapsed,'fusion_inclusive':fusion,
                'checkpoint':asdict(src.checkpoint),'counts':adapter.counts,
                'token_cache':adapter.token_block_cache.statistics()}
        assert scorer.native_pairs==len(q.ids)*128
        (CTL/'rasa_result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        (CTL/'rasa_failure.txt').write_text(traceback.format_exc())
        raise
