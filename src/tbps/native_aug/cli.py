"""显式授权作业入口：两个训练进程，首节点后各自派生三个评测进程。"""
import argparse,sys
from .common import *
def preflight(c):
    import torch,types
    from importlib import import_module
    from transformers import BertTokenizerFast
    from .training import native_bundle
    from .native import Source
    n,bundle=native_bundle(c)
    q=read_json(Path(n['prototype']['history_root'])/'qualification/train.json')['entries']
    need(len(q)==2048 and [e['identity']['row_id'] for e in q]==n['prototype']['train_ids'],'Train 2048 membership/order')
    raw=json.loads(Path(n['prototype']['annotation']).read_text())
    for e in q:
        row=e['identity'];item=raw[int(row['row_id'].split(':')[1])]
        need(item['split']==row['split']=='train' and row['role']=='train' and item['id']==row['pid'] and item['captions'][row['caption_index']]==e['caption'],'official Train identity')
        checked(FileRef(str(Path(n['prototype']['image_root'])/row['image_rel_path']),row['image_sha256']))
    report={'configuration_sha256':sha_file(c['_config_path']),'status':'NATIVE_AUG_CPU_PREFLIGHT_PASS','models':{},'Train':2048,'fresh_initialization':bundle.report,'CUDA':False}
    for name in ('rasa','aptm'):
        mod=import_module('tbps.models.'+name)
        source=(mod.RasaSource if name=='rasa' else mod.APTMSource)(c['targets'][name]['path']);source.verify()
        with mod.native_modules(source) as (_,tok,utils):
            tokenizer=mod.official_tokenizer(tok,source)
            adapter=types.SimpleNamespace(tokenizer=tokenizer,pre_caption=utils.pre_caption)
            limit=50 if name=='rasa' else 56
            adapter.tokenize=lambda texts,ad=adapter,nn=name,ll=limit:ad.tokenizer([ad.pre_caption(t,ll) if nn=='rasa' else ad.pre_caption(t,ll,icfg_rstp=False,is_eda=False) for t in texts],padding='max_length',truncation=True,max_length=ll,return_tensors='pt')
            stub=types.SimpleNamespace(adapter=adapter,name=name,max_tokens=limit,fast=BertTokenizerFast.from_pretrained(source.data['bert'],local_files_only=True))
            stub.normalized=types.MethodType(Source.normalized,stub)
            for e in q:Source.aligned(stub,e['caption'])
            # 官方命名空间与旧H/G命名空间必须能同时装载，不构造受害者模型。
            from tbps.alternating.native_sources import native_session
            from .compat import source_session
            with source_session(n,bundle,c,name):pass
        report['models'][name]={'native_tokenizer_exact_rows':2048,'source':asdict(source.ref),'weight':asdict(source.checkpoint),'legacy_namespace_closure':True}
    need(not torch.cuda.is_initialized(),'CPU preflight touched CUDA')
    d=CONTROL/'preflight';d.mkdir(parents=True,exist_ok=True);(d/'report.json').write_bytes(json_bytes(report));print(json.dumps(report,ensure_ascii=False),flush=True)
    return report
def launch(c):
    need(read_json(CONTROL/'preflight/report.json')['status']=='NATIVE_AUG_CPU_PREFLIGHT_PASS' and read_json(CONTROL/'preflight/report.json')['configuration_sha256']==sha_file(c['_config_path']),'matching CPU preflight gate')
    need(not (CONTROL/'launch.json').exists(),'launch identity consumed; no duplicate')
    for name in ('rasa','aptm'):
        proof=read_json(CONTROL/'cpu_probes'/name/'report.json')
        need(proof['status']=='NATIVE_AUG_REAL_CPU_BACKEND_PROBE_PASS' and proof['source_configuration']==c['targets'][name],'real native CPU backend gate')
        for tested in proof['tested_sources']:checked(FileRef(**tested))
    CONTROL.mkdir(parents=True,exist_ok=True)
    jobs={}
    for name in ('rasa','aptm'):
        if name=='aptm':
            # 等待前一进程完成真实模型装载，随后仅检查是否还有余量；不以预估GiB拦截。
            started=time.monotonic();first=CONTROL/'train_rasa'
            while not (first/'model_loaded.json').exists() and not (first/'FAIL').exists() and time.monotonic()-started<60:
                time.sleep(.2)
            observed=gpu_snapshot('launcher','AFTER_RASA_LOAD_BEFORE_APTM_START')
            if observed['free_mib']<=0:
                state('train_aptm','NOT_STARTED_NO_FREE_GPU_MEMORY',observed=observed)
                jobs[name]={'job':'train_aptm','phase':'NOT_STARTED_NO_FREE_GPU_MEMORY'}
                continue
        job=f'train_{name}';d=CONTROL/job;d.mkdir(exist_ok=True)
        with (d/'started.json').open('x') as f:json.dump({'source':name,'configuration_sha256':sha_file(c['_config_path'])},f)
        log=(d/'launcher.log').open('ab',buffering=0)
        process=subprocess.Popen(command('train',name),cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','PYTHONUNBUFFERED':'1','TOKENIZERS_PARALLELISM':'false'})
        log.close();jobs[name]={'pid':process.pid,'job':job,'log':str(d/'launcher.log')}
    (CONTROL/'launch.json').write_bytes(json_bytes({'jobs':jobs,'configuration_sha256':sha_file(c['_config_path']),'evaluations':'three independent target workers per source automatically after round10 CPU audit'}))
    print(json.dumps(jobs,ensure_ascii=False),flush=True)
def status():
    for source in ('rasa','aptm'):
        for job in [f'train_{source}',*[f'eval_{source}_to_{t}' for t in MODELS]]:
            path=CONTROL/job/'status.json'
            if path.exists():
                data=read_json(path)
                try:os.kill(data['pid'],0);data['process_alive']=True
                except ProcessLookupError:data['process_alive']=False
                print(json.dumps(data,ensure_ascii=False))
            else:print(json.dumps({'job':job,'phase':'NOT_STARTED; source round10 audit pending'},ensure_ascii=False))
    for source in ('rasa','aptm'):
        for p in sorted((OUTPUT/source/'evaluation').glob('*/round_*/cpu_audit/completed/report.json')):
            print('AUDITED_RESULT',json.dumps(read_json(p),ensure_ascii=False))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['preflight','launch','train','audit','evaluate','audit-inputs','audit-evaluation','probe-cpu','status'])
    ap.add_argument('args',nargs='*');a=ap.parse_args()
    if a.action=='status':return status()
    c=config();job=a.action+'_'+'_'.join(a.args)
    if a.action=='train':job='train_'+a.args[0]
    if a.action=='evaluate':job='eval_'+a.args[0]+'_to_'+a.args[1]
    try:
        if a.action=='preflight':return preflight(c)
        if a.action=='launch':return launch(c)
        if a.action=='probe-cpu':
            from .probe import run
            return run(c,a.args[0])
        if a.action=='train':
            from .training import run
            return run(c,a.args[0])
        if a.action=='audit':
            from .audit import run
            return run(c,a.args[0],int(a.args[1]))
        if a.action=='evaluate':
            from .evaluation import worker
            return worker(c,*a.args)
        if a.action=='audit-inputs':
            from .evaluation_audit import audit_inputs
            return audit_inputs(c,a.args[0],int(a.args[1]))
        if a.action=='audit-evaluation':
            from .evaluation_audit import audit_evaluation
            return audit_evaluation(c,a.args[0],a.args[1],int(a.args[2]))
    except BaseException:
        failed(job);raise
