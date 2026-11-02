#!/usr/bin/env python3
"""单次独立Supervisor：两目标Smoke及独立审计后，串行Full；失败保留，不重试。"""
from pathlib import Path
import argparse,fcntl,json,os,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'configs/framework/gv2_cross_dataset.server.json'
def write_status(control,phase,**fields):
 row={'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'phase':phase,**fields}
 p=control/'status.tmp';p.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n');p.replace(control/'status.json')
 with (control/'status.log').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 print(json.dumps(row,ensure_ascii=False),flush=True)
def pipeline():
 steps=[('CPU_PREFLIGHT','preflight',None,'smoke')]
 for ds in ('rstp','icfg'):
  steps += [('SMOKE','generate',ds,'smoke'),('SMOKE','evaluate',ds,'smoke'),('SMOKE_AUDIT','audit',ds,'smoke')]
 for ds in ('rstp','icfg'):
  phase='RUNNING_'+ds.upper()+'_FULL'
  steps += [(phase,'generate',ds,'full'),(phase,'evaluate',ds,'full'),(ds.upper()+'_AUDIT','audit',ds,'full')]
 return steps
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default=str(DEFAULT));p.add_argument('--allow-smoke-and-auto-full',action='store_true');p.add_argument('--status',action='store_true');a=p.parse_args()
 c=json.loads(Path(a.config).read_text());control=Path(c['control'])
 if a.status:
  r=json.loads((control/'status.json').read_text());r['supervisor']={k:(control/('supervisor.'+k)).read_text().strip() for k in ('pid','ppid','pgid','sid')}
  r['progress']=[json.loads(x.read_text()) for x in sorted(control.glob('progress_*.json'))]
  r['receipts']={}
  for scope in ('smoke','full'):
   for ds in ('rstp','icfg'):
    case=Path(c['output_base'])/scope/ds
    r['receipts'][scope+'_'+ds]={key:(json.loads((case/rel).read_text()).get('status','PENDING') if (case/rel).exists() else 'PENDING') for key,rel in [('generation','generation_completed/report.json'),('scoring','completed/summary.json'),('cpu_audit','cpu_audit/report.json')]}
  print(json.dumps(r,ensure_ascii=False,indent=2));return
 if not a.allow_smoke_and_auto_full:p.error('explicit real Smoke and automatic Full authorization required')
 control.mkdir(parents=True,exist_ok=True)
 lock=(control/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (control/'started.json').exists():raise RuntimeError('Attempt consumed; no retry')
 if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise RuntimeError('execution worktree dirty')
 if (control/'cpu_tests.exit').read_text().strip()!='0':raise RuntimeError('CPU tests not PASS')
 if Path(c['output_base']).exists():raise RuntimeError('fresh namespace required')
 started={'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'authorization':'real two-target Smoke+CPU audits, then automatic two Full; shared GPU without VRAM floor','time':time.time()}
 with (control/'started.json').open('x') as f:json.dump(started,f,indent=2);f.flush();os.fsync(f.fileno())
 for key,val in (('pid',os.getpid()),('ppid',os.getppid()),('pgid',os.getpgid(0)),('sid',os.getsid(0))):(control/('supervisor.'+key)).write_text(str(val))
 completed=0
 try:
  for index,(phase,action,ds,scope) in enumerate(pipeline()):
   name=f'{index:02d}_{scope}_{ds or "all"}_{action}';log=control/(name+'.log')
   write_status(control,phase,dataset=ds,scope=scope,action=action,completed_groups=completed,total_groups=2,log=str(log),cpu_audit='RUNNING' if action=='audit' else 'PENDING')
   env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'0' if action in ('generate','evaluate') else '',
        'TMPDIR':c['temp_dir'],'XDG_CACHE_HOME':c['cache_dir'],'TORCH_HOME':c['cache_dir'],'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
   args=[c['python'],'-B','-u','-m','tbps.transfer.gv2_execute','--config',a.config,'--action',action,'--scope',scope]
   if ds:args+=['--dataset',ds]
   if action in ('generate','evaluate'):args+=['--allow-gpu']
   with log.open('x') as f:
    child=subprocess.Popen(args,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    (control/(name+'.pid')).write_text(str(child.pid));rc=child.wait()
   (control/(name+'.exit')).write_text(str(rc))
   if rc:raise RuntimeError(name+f' exit={rc}; preserve and stop; see '+str(log))
   if scope=='full' and action=='audit':completed+=1
  write_status(control,'PUBLISHING_REPORT',completed_groups=completed,total_groups=2,cpu_audit='PASS')
  with (control/'publish_report.log').open('x') as f:
   rc=subprocess.run([c['python'],'-B',str(ROOT/'operations/publish_gv2_cross_dataset.py'),'--config',a.config],cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'},stdout=f,stderr=subprocess.STDOUT).returncode
  if rc:raise RuntimeError('final report publishing failed; no retry')
  write_status(control,'SUCCESS',dataset=None,completed_groups=completed,total_groups=2,cpu_audit='PASS',code_commit=started['code_commit'])
 except BaseException as e:
  with (control/'FAIL.json').open('x') as f:json.dump({'error':repr(e),'traceback':traceback.format_exc()},f,ensure_ascii=False,indent=2)
  write_status(control,'FAIL',error=repr(e),completed_groups=completed,total_groups=2);raise
if __name__=='__main__':main()
