"""一次性Smoke→60轮→六Joint Test；任何真实错误停止，无重试/外部进程干预。"""
from pathlib import Path
import argparse,fcntl,json,os,subprocess,time
ROOT=Path(__file__).resolve().parents[1];PY='/home/lzf/ldx/envs/tbps-image/bin/python'
CFG=ROOT/'configs/framework/gh_alternating_v2.server.json';CTL=Path('/home/lzf/ldx/tmp/gh_alternating_v2_supervisor_20261107')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--allow-authorized-pipeline',action='store_true');a=ap.parse_args()
    if not a.allow_authorized_pipeline:raise ValueError('explicit complete v2 scope authorization required')
    CTL.mkdir(parents=True,exist_ok=True)
    with (CTL/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'PYTHONDONTWRITEBYTECODE':'1',
            'MPLCONFIGDIR':str(CTL/'matplotlib'),'XDG_CACHE_HOME':str(CTL/'cache'),'CUDA_CACHE_PATH':str(CTL/'cuda_cache')}
        def status(phase,**extra):
            v={'phase':phase,'time':time.time(),**extra};tmp=CTL/'status.pending';tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2));os.replace(tmp,CTL/'status.json')
            with (CTL/'status.log').open('a') as f:f.write(time.strftime('%Y-%m-%d %H:%M:%S')+' '+phase+'\n')
        def child(label,args):
            status(label)
            with (CTL/(label+'.log')).open('x') as out:
                p=subprocess.Popen([PY,'-u','-B',str(ROOT/'operations/gh_alternating_v2.py'),'--config',str(CFG),*args],
                    cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
                (CTL/'child.json').write_text(json.dumps({'pid':p.pid,'label':label,'time':time.time()}))
                rc=p.wait()
            (CTL/(label+'.exit')).write_text(str(rc)+'\n')
            if rc==3:status('STOPPED_COMPLETE_ROUND_BOUNDARY',stage=label);raise SystemExit(3)
            if rc:raise RuntimeError(label+' exit '+str(rc)+'; stopped without retry')
        with (CTL/'started.json').open('x') as f:json.dump({'pid':os.getpid(),'ppid':os.getppid(),'pgid':os.getpgid(0),
            'sid':os.getsid(0),'tty':os.ttyname(0) if os.isatty(0) else None,'time':time.time()},f)
        try:
            child('01_smoke',['--action','smoke','--allow-gpu'])
            child('02_smoke_audit',['--action','audit','--scope','smoke'])
            child('03_joint_smoke',['--action','test-smoke','--round','1','--allow-gpu'])
            child('04_joint_smoke_audit',['--action','test-smoke-audit','--round','1'])
            child('05_train60',['--action','train','--allow-gpu'])
            child('06_train_audit',['--action','audit','--scope','train'])
            for n in (10,20,30,40,50,60):
                child('test_'+str(n),['--action','test','--round',str(n),'--allow-gpu'])
                child('audit_test_'+str(n),['--action','test-audit','--round',str(n)])
            child('final_report',['--action','report'])
            # 只发布本次小型报告/曲线；源码已经在启动前封存。
            files=['docs/PROJECT_STATE.md','docs/framework/results/gh_alternating_v2']
            subprocess.run(['git','add',*files],cwd=ROOT,check=True)
            subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
            subprocess.run(['git','-c','user.name=w-harda','-c','user.email=wharda@163.com','commit','-m','docs: 发布v2从零60轮六节点Joint正式结果'],cwd=ROOT,check=True)
            subprocess.run(['git','push','origin','codex/gh-alternating-v2'],cwd=ROOT,check=True)
            head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
            remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/gh-alternating-v2'],cwd=ROOT,text=True).split()[0]
            if head!=remote:raise RuntimeError('remote SHA mismatch')
            status('SUCCESS_60_ROUNDS_SIX_JOINT_TESTS_PUBLISHED',commit=head)
        except SystemExit:raise
        except BaseException as e:status('FAIL_STOPPED_NO_RETRY',error=repr(e));raise
if __name__=='__main__':main()
