"""新任务独立身份、冻结来源、进度和GPU准入；不修改历史产物。"""
from pathlib import Path
from contextlib import contextmanager
from dataclasses import asdict
import os,json,time,hashlib,subprocess,fcntl,sys,traceback,io
from tbps.artifacts import ArtifactStore,FileRef,sha_file,checked,read_json,json_bytes,digest,ArtifactError
from tbps.alternating.state import tree_sha,clone_tree
ROOT=Path(__file__).resolve().parents[3]
CONTROL=Path('/home/lzf/ldx/tmp/gh_native_aug_20261114')
OUTPUT=Path('/home/lzf/ldx/outputs/tbps-framework-gh-native-aug-20261114')
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
@contextmanager
def gpu(job,minimum=6500):
    """独立进程保留；共享锁只保护显存准入瞬间，绝不结束其他任务。"""
    CONTROL.mkdir(parents=True,exist_ok=True)
    lock=(CONTROL/'gpu_admission.lock').open('a')
    while True:
        fcntl.flock(lock,fcntl.LOCK_EX)
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
        # 留出正在构造模型的短暂窗口，防止多个进程同时看到同一份空闲显存。
        stamp=CONTROL/'last_gpu_admission.json';previous=read_json(stamp) if stamp.exists() else {}
        if free>=minimum and time.time()-previous.get('timestamp',0)>=30:
            stamp.write_bytes(json_bytes({'timestamp':time.time(),'job':job,'pid':os.getpid()}))
            fcntl.flock(lock,fcntl.LOCK_UN);break
        fcntl.flock(lock,fcntl.LOCK_UN)
        state(job,'WAITING_FOR_GPU',free_mib=free,required_mib=minimum)
        time.sleep(30)
    try:yield
    finally:lock.close()
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
