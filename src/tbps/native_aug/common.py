"""新任务独立身份、冻结来源、进度和GPU准入；不修改历史产物。"""
from pathlib import Path
from contextlib import contextmanager
from dataclasses import asdict
import os,json,time,hashlib,subprocess,fcntl,sys,traceback,io
from tbps.artifacts import ArtifactStore,FileRef,sha_file,checked,read_json,json_bytes,digest,ArtifactError
from tbps.alternating.state import tree_sha,clone_tree
ROOT=Path(__file__).resolve().parents[3]
CONTROL=Path('/home/lzf/ldx/tmp/gh_native_aug_repair_20261114')
OUTPUT=Path('/home/lzf/ldx/outputs/tbps-framework-gh-native-aug-repair-20261114')
ROUNDS=(10,20,30,40,50,60)
MODELS=('irra','rasa','aptm')
def need(ok,msg):
    if not ok:raise ArtifactError(msg)
def ref(p):return FileRef(str(p),sha_file(p))
def config():
    c=read_json(ROOT/'configs/framework/gh_native_aug.server.json','gh-native-aug-v1')
    for p,h in c['producers'].items():checked(FileRef(p,h))
    for p,h in c['runtime_pins'].items():checked(FileRef(p,h))
    for r in c['fixed_resources'].values():checked(FileRef(**r))
    for r in c['targets'].values():checked(FileRef(**r))
    need(c['rounds']==60 and c['test_rounds']==list(ROUNDS),'frozen 60/six checkpoints')
    c['_config_path']=str(ROOT/'configs/framework/gh_native_aug.server.json')
    return c
def native_config(c):
    n=clone_tree(c['native_template'])
    n.update(producers=c['producers'],runtime_pins=c['runtime_pins'],output_base=str(OUTPUT))
    n['managed_roots']=[str(ROOT/'src') if x.endswith('/src') and 'idea-test-tbps-gh-' in x else x for x in n['managed_roots']]
    return n
def state(job,phase,**kw):
    p=CONTROL/job;p.mkdir(parents=True,exist_ok=True)
    value={'job':job,'phase':phase,'pid':os.getpid(),'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw}
    tmp=p/'status.tmp';tmp.write_bytes(json_bytes(value));os.replace(tmp,p/'status.json')
def tensor_bytes(x):
    import torch
    b=io.BytesIO();torch.save(clone_tree(x),b);return b.getvalue()
def publish(root,key,files,metadata=None,sources=()):
    store=ArtifactStore(root,new_base=OUTPUT);return store.publish(key,files,metadata or {},sources=sources)
def gpu_snapshot(job,event):
    # 仅观测实际余量与本进程分配，不设准入阈值、不sleep、不占GPU调度锁。
    values=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,memory.total,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()[0]
    free,total,used=[int(v.strip()) for v in values.split(',')]
    import torch
    record={'event':event,'pid':os.getpid(),'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'free_mib':free,'total_mib':total,'used_mib':used,'fixed_VRAM_floor':None}
    if torch.cuda.is_initialized():
        record.update(process_allocated_mib=torch.cuda.memory_allocated()/2**20,process_reserved_mib=torch.cuda.memory_reserved()/2**20)
    path=CONTROL/job;path.mkdir(parents=True,exist_ok=True)
    with (path/'gpu_observations.jsonl').open('ab') as f:f.write(json_bytes(record)+b'\n')
    return record
def model_ready(job):
    record=gpu_snapshot(job,'MODEL_LOADED_ACTUAL_MEMORY')
    (CONTROL/job/'model_loaded.json').write_bytes(json_bytes(record))
    return record
@contextmanager
def gpu(job):
    gpu_snapshot(job,'BEFORE_ACTUAL_LOAD')
    state(job,'STARTING_GPU_JOB',GPU_policy='shared_observe_only_no_VRAM_floor')
    try:yield
    finally:gpu_snapshot(job,'AFTER_GPU_WORK')

def failed(job):
    p=CONTROL/job;p.mkdir(parents=True,exist_ok=True)
    with (p/('FAIL' if not (p/'FAIL').exists() else 'FAIL_'+str(time.time_ns()))).open('x') as f:f.write(traceback.format_exc())
    state(job,'FAIL_STOPPED_NO_RETRY')
def command(*args):return [sys.executable,'-B',str(ROOT/'operations/gh_native_aug.py'),*map(str,args)]
def spawn_eval(source):
    path=CONTROL/f'train_{source}'/'evaluators_started.json'
    need(not path.exists(),'evaluators already started; no duplicate')
    jobs={}
    for target in MODELS:
        name=f'eval_{source}_to_{target}';d=CONTROL/name;d.mkdir(parents=True,exist_ok=True)
        log=(d/'launcher.log').open('ab',buffering=0)
        p=subprocess.Popen(command('evaluate',source,target),cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                           start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','PYTHONUNBUFFERED':'1'})
        log.close();jobs[target]={'pid':p.pid,'job':name}
    path.write_bytes(json_bytes(jobs));return jobs
def wait_file(path,job,**details):
    p=Path(path)
    while not p.exists():
        source=details.get('source')
        need(not source or not (CONTROL/f'train_{source}'/'FAIL').exists(),'source training failed')
        need(not source or not (CONTROL/f'eval_{source}_to_{source}'/'FAIL').exists(),'shared input generator failed')
        state(job,'WAITING_FOR_CHECKPOINT',**details);time.sleep(30)
    return p
