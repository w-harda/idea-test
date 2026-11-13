"""唯一后处理Attempt：CPU核验→依赖退出→双目标Smoke→六组Full→发布。"""
from pathlib import Path
import os,sys,json,time,subprocess,fcntl,argparse,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from tbps.gh_transfer.common import config,need,sha_file,read_json,json_bytes,METHODS,MODELS

def atomic(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_bytes(json_bytes(value));os.replace(tmp,path)

def process_identity(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:return None

def plan():
    steps=[]
    for method in METHODS:steps.append(('verify_inputs_'+method,['--action','verify-input','--method',method]))
    for method in METHODS:
        for model in MODELS:steps.append(('startup_'+method+'_'+model,['--action','startup-check','--method',method,'--model',model,'--scope','full']))
    for scope in ('smoke','full'):
        for method in METHODS:
            for model in MODELS:
                args=['--method',method,'--model',model,'--scope',scope]
                steps.append((method+'_'+model+'_'+scope,['--action','run','--allow-gpu']+args))
                steps.append((method+'_'+model+'_'+scope+'_audit',['--action','audit']+args))
    steps.append(('report',['--action','report']));return steps

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--config',default=str(ROOT/'configs/framework/gh_cross_model_v2.server.json'))
    parser.add_argument('--allow-authorized-pipeline',action='store_true');a=parser.parse_args(argv)
    need(a.allow_authorized_pipeline,'explicit six-group pipeline authorization')
    c=config(a.config);ctl=Path(c['control']);ctl.mkdir(parents=True,exist_ok=True)
    with (ctl/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (ctl/'started.json').open('x') as f:f.write(json.dumps({'configuration_sha256':sha_file(a.config),'pid':os.getpid(),'time':time.time()}))
        atomic(ctl/'identity.json',{'pid':os.getpid(),'ppid':os.getppid(),'pgid':os.getpgid(0),'sid':os.getsid(0),'tty':os.ttyname(0) if os.isatty(0) else None,
                                  'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'configuration_sha256':sha_file(a.config)})
        (ctl/'tmp').mkdir(exist_ok=True)
        env=dict(os.environ,TMPDIR=str(ctl/'tmp'),TORCH_HOME='/home/lzf/ldx/cache/torch',HF_HOME='/home/lzf/ldx/cache/huggingface',XDG_CACHE_HOME='/home/lzf/ldx/cache',CUDA_CACHE_PATH='/home/lzf/ldx/cache/cuda',PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        try:
            waiting=False
            for label,args in plan():
                if '--allow-gpu' in args and not waiting:
                    dep=c['gpu_dependency'];pid=dep['pid']
                    while dep['start_ticks'] is not None and process_identity(pid)==dep['start_ticks']:
                        atomic(ctl/'status.json',{'phase':'WAITING_FOR_EXISTING_AUGMENTATION','dependency':dep,'next':label,'time':time.time()});time.sleep(30)
                    waiting=True
                atomic(ctl/'status.json',{'phase':label,'time':time.time()})
                with (ctl/(label+'.log')).open('xb') as log:
                    child=subprocess.Popen([sys.executable,'-u','-B',str(ROOT/'operations/gh_cross_model_v2.py'),'--config',a.config]+args,
                                           cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    atomic(ctl/'child.json',{'pid':child.pid,'label':label,'time':time.time()});code=child.wait()
                atomic(ctl/(label+'.exit.json'),{'exit_code':code,'time':time.time()})
                need(code==0,label+' exit '+str(code)+'; stopped without retry')
            # 仅发布本次生成的小型报告；没有训练/评测重入。
            paths=['docs/framework/results/gh_cross_model_v2','docs/PROJECT_STATE.md']
            subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
            subprocess.run(['git','add','--']+paths,cwd=ROOT,check=True)
            subprocess.run(['git','-c','user.name=w-harda','-c','user.email=wharda@163.com','commit','-m','docs: 发布Cooperative和Push-pull六组跨模型正式评测'],cwd=ROOT,check=True)
            subprocess.run(['git','push','origin','HEAD:codex/gh-cross-model-v2'],cwd=ROOT,check=True)
            head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
            remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/gh-cross-model-v2'],cwd=ROOT,text=True).split()[0]
            need(head==remote,'published remote SHA mismatch')
            atomic(ctl/'status.json',{'phase':'SUCCESS_SIX_FULL_TRANSFER_TESTS_PUBLISHED','commit':head,'time':time.time()})
        except BaseException as exc:
            (ctl/'FAIL').write_text(traceback.format_exc())
            atomic(ctl/'status.json',{'phase':'FAIL_STOPPED_NO_RETRY','error':repr(exc),'time':time.time()});raise
if __name__=='__main__':main()
