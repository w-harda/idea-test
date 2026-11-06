"""独立逐入口核验真实编码特征、完整原生矩阵及后续PID概率；不改模型。"""
from pathlib import Path
from unittest.mock import patch
import json
import torch,nltk
from operations.sep_text_only.runtime import config,source_inputs,store,source_guard
from operations.sep_source_only.common import train_manifest
from operations.sep_text_only.word_budget import resources
from tbps.baselines.cpgc_sep import sep_source
from tbps.baselines.cpgc_sep.cache import NativeScorer
from tbps.baselines.cpgc_sep.victims import load_victim
from tbps.baselines.cpgc_sep.sep_packed_pid import enable_packed_pid
from .ab import exact,rng,OUT,CTL

def main():
    torch.set_num_threads(2)
    with (OUT/'native_verify_started.json').open('x') as f:json.dump({'status':'STARTED'},f)
    items=json.loads((OUT/'preregister.json').read_text())['samples']
    c,_=config();bundle=source_inputs(c);ref=store(c,'source/reference');ref.verify('completed')
    gallery=torch.load(ref.root/'completed/features.pt',map_location='cpu',weights_only=True)
    manifest=train_manifest(bundle['train_gallery']);cfg=bundle['config']
    c={**c,'output':str(OUT),'control':str(CTL)};source_guard(c,bundle)
    checks=[];baseline={}
    with load_victim(cfg,'irra','cuhk',smoke=False) as (adapter,model,src):
        for name in ('original','packed'):
            predictor=sep_source.classifier(adapter,model,'irra',sep_source.EphemeralQuery(gallery),manifest)
            if name=='packed':enable_packed_pid(predictor)
            for item in items:
                captured={}
                encode_original=sep_source.encode_texts;score_original=NativeScorer.score
                def encode(*a,**kw):
                    value=encode_original(*a,**kw);captured['features']=value.clone();return value
                def score(*a,**kw):
                    value=score_original(*a,**kw);captured['matrix']=value.payload.clone();return value
                before=rng()
                with patch.object(sep_source,'encode_texts',encode),patch.object(NativeScorer,'score',score):
                    logits=predictor([item['caption']]).logits
                captured['logits']=logits.detach().cpu()
                probabilities=logits.softmax(-1)
                label=int(probabilities[0].argmax())
                onehot=torch.zeros_like(probabilities);onehot[:,label]=probabilities[:,label]
                captured['probabilities']=probabilities.cpu()
                captured['margin']=(probabilities[:,label]-(probabilities-onehot).max(-1).values).cpu()
                captured['rng_unchanged']=before==rng()
                predictor.cache.clear()
                if name=='original':baseline[item['key']]=captured
                else:
                    previous=baseline[item['key']]
                    check={'dataset':item['dataset'],'key':item['key'],**{key:exact(previous[key],captured[key]) for key in ('features','matrix','logits','probabilities','margin')},
                           'rng_unchanged':previous['rng_unchanged'] and captured['rng_unchanged']}
                    checks.append(check);assert all(v for k,v in check.items() if k not in ('dataset','key')),check
    result={'status':'PASS','full_native_matrix_shape':[1,34054],'pid_shape':[1,11003],'checks':checks}
    (CTL/'native_exact.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        (CTL/'native_verify_failure.txt').write_text(traceback.format_exc())
        raise
