"""单次后台监督器；只执行两轮Pilot，计划Round1正常边界退出后显式恢复。"""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PY='/home/lzf/ldx/envs/tbps-image/bin/python'
CFG=ROOT/'configs/framework/gh_alternating_v1.server.json'
CTL=Path('/home/lzf/ldx/tmp/gh_alternating_v1_pilot_20261106')

def main():
    a=argparse.ArgumentParser();a.add_argument('--scope',choices=('pilot','train'),default='pilot')
    a.add_argument('--allow-gpu-pilot',action='store_true');a.add_argument('--allow-training',action='store_true')
    a.add_argument('--resume-round',type=int);v=a.parse_args()
    if not (v.allow_gpu_pilot if v.scope=='pilot' else v.allow_training):raise ValueError('explicit scope capability required')
    global CTL
    suffix='' if v.resume_round is None else '_resume_'+str(v.resume_round)
    CTL=Path('/home/lzf/ldx/tmp/gh_alternating_v1_'+v.scope+'_20261106'+suffix)
    CTL.mkdir(parents=True,exist_ok=True)
    with (CTL/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'PYTHONDONTWRITEBYTECODE':'1'}
        def status(phase,**fields):
            (CTL/'supervisor_status.json').write_text(json.dumps({'phase':phase,'time':time.time(),**fields},ensure_ascii=False,indent=2))
            with (CTL/'status.log').open('a') as f:f.write(time.strftime('%Y-%m-%d %H:%M:%S')+' '+phase+'\n')
        def child(label,args,expected=0):
            status(label)
            with (CTL/(label+'.log')).open('x') as log:
                p=subprocess.Popen([PY,'-u','-B',str(ROOT/'operations/gh_alternating_v1.py'),'--config',str(CFG),*args],
                                   cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                (CTL/'child.json').write_text(json.dumps({'pid':p.pid,'label':label}))
                rc=p.wait()
            (CTL/(label+'.exit')).write_text(str(rc)+'\n')
            acceptable=expected if isinstance(expected,tuple) else (expected,)
            if rc not in acceptable:raise RuntimeError(label+' exit '+str(rc)+'; no retry')
            return rc
        try:
            # 唯一启动标记在所有GPU操作前消费；没有Watcher、循环重试或Full训练。
            with (CTL/'started.json').open('x') as f:json.dump({'pid':os.getpid(),'ppid':os.getppid(),'pgid':os.getpgid(0),
                'sid':os.getsid(0),'tty':os.ttyname(0) if os.isatty(0) else None,'time':time.time()},f)
            if v.scope=='pilot':
                if v.resume_round is not None:raise ValueError('Pilot only uses its predefined Round1 boundary')
                child('01_round1',['--action','pilot','--allow-gpu-pilot','--stop-after-round','1'],expected=3)
                child('02_resume_round2',['--action','pilot','--allow-gpu-pilot','--resume-round','1'])
            else:
                args=['--action','train','--allow-training']
                if v.resume_round is not None:args+=['--resume-round',str(v.resume_round)]
                rc=child('01_training',args,expected=(0,3))
                if rc==3:
                    status('STOPPED_COMMITTED_BOUNDARY');return
            child('03_cpu_audit',['--action','audit','--scope',v.scope])
            status('SUCCESS_'+v.scope.upper()+'_AND_CPU_AUDIT',formal_training='NOT_RUN' if v.scope=='pilot' else 'COMPLETE')
        except BaseException as e:
            status('FAIL_STOPPED_NO_RETRY',error=repr(e));raise
if __name__=='__main__':main()
