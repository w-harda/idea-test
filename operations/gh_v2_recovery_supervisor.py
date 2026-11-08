"""原GH-v2完成训练的单次后处理监督器；不调用训练或原Supervisor。"""
from pathlib import Path
import argparse,fcntl,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'operations')]
from gh_v2_recovery import CTL,REC,CFG,BASE,OLDCTL,tools,load
from tbps.artifacts import read_json,checked,FileRef,sha_file
PY='/home/lzf/ldx/envs/tbps-image/bin/python'
VERSION=ROOT/'configs/framework/gh_v2_posttraining_recovery_20261108.json'

def verify_version():
    v=read_json(VERSION,'gh-v2-posttraining-recovery-1')
    if v['configuration_sha256']!=sha_file(CFG) or v['tools']!=tools():raise ValueError('explicit recovery source identity mismatch')
    for p,h in v['tools'].items():checked(FileRef(p,h))
    load(CFG,'test');return v

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--allow-posttraining-recovery',action='store_true');a=ap.parse_args()
    if not a.allow_posttraining_recovery:raise ValueError('explicit posttraining-only authorization required')
    verify_version();CTL.mkdir(parents=True,exist_ok=True)
    with (CTL/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env={**os.environ,'PYTHONPATH':str(ROOT/'src')+':'+str(ROOT/'operations'),'PYTHONDONTWRITEBYTECODE':'1',
            'MPLCONFIGDIR':str(CTL/'matplotlib'),'XDG_CACHE_HOME':str(CTL/'cache'),'CUDA_CACHE_PATH':str(CTL/'cuda_cache')}
        def status(phase,**extra):
            value={'phase':phase,'time':time.time(),**extra};p=CTL/'status.pending';p.write_text(json.dumps(value,ensure_ascii=False,indent=2));os.replace(p,CTL/'status.json')
        def child(label,action,n=None):
            verify_version();status(label)
            args=[PY,'-u','-B',str(ROOT/'operations/gh_v2_recovery.py'),'--action',action]
            if n is not None:args+=['--round',str(n)]
            with (CTL/(label+'.log')).open('x') as out:
                p=subprocess.Popen(args,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
                (CTL/'child.json').write_text(json.dumps({'pid':p.pid,'label':label,'time':time.time()}))
                rc=p.wait()
            (CTL/(label+'.exit')).write_text(str(rc)+'\n')
            if rc:raise RuntimeError(label+' exit '+str(rc)+'; stopped without retry')
        with (CTL/'started.json').open('x') as out:json.dump({'pid':os.getpid(),'ppid':os.getppid(),'pgid':os.getpgid(0),'sid':os.getsid(0),
            'tty':os.ttyname(0) if os.isatty(0) else None,'time':time.time(),'training_attempt':1,'new_training':False,
            'original_control':str(OLDCTL),'original_outputs':str(BASE),'recovery_version_sha256':sha_file(VERSION),'tools':tools()},out)
        try:
            child('01_train_cpu_audit','audit')
            for n in (10,20,30,40,50,60):
                child('joint_test_'+str(n),'test',n);child('joint_cpu_audit_'+str(n),'test-audit',n)
            child('final_report','report');child('historical_preservation','verify')
            files=['docs/PROJECT_STATE.md','docs/framework/results/gh_alternating_v2_recovery']
            subprocess.run(['git','add',*files],cwd=ROOT,check=True);subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
            subprocess.run(['git','-c','user.name=w-harda','-c','user.email=wharda@163.com','commit','-m','docs: 发布GH-v2审计恢复与六组Joint正式Test'],cwd=ROOT,check=True)
            subprocess.run(['git','push','origin','codex/gh-alternating-v2'],cwd=ROOT,check=True)
            head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
            remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/gh-alternating-v2'],cwd=ROOT,text=True).split()[0]
            if head!=remote:raise RuntimeError('remote SHA mismatch')
            status('SUCCESS_POSTTRAINING_SIX_JOINT_PUBLISHED',commit=head)
        except BaseException as exc:
            status('FAIL_STOPPED_NO_RETRY',error=repr(exc))
            with (CTL/'FAIL.json').open('x') as out:json.dump({'error':repr(exc),'traceback':traceback.format_exc(),'time':time.time()},out)
            raise
if __name__=='__main__':main()
