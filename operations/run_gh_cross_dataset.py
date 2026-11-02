#!/usr/bin/env python3
"""单次串行Supervisor：两目标Smoke+四审计后自动Full；源图库未审计则只等待。"""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'configs/framework/gh_cross_dataset.server.json'

def write_status(control,phase,**fields):
 row={'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'phase':phase,**fields};p=control/'status.tmp';p.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n');p.replace(control/'status.json')
 with (control/'status.log').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 print(json.dumps(row,ensure_ascii=False),flush=True)

def pipeline():
 steps=[('CPU_PREFLIGHT','preflight',None,'smoke',None)]
 for scope in ('smoke','full'):
  for ds in ('rstp','icfg'):
   phase=scope.upper()+'_'+ds.upper();steps.append((phase,'generate',ds,scope,None))
   for mode in ('Text-only','Joint'):steps.extend(((phase,'evaluate',ds,scope,mode),(phase+'_AUDIT','audit',ds,scope,mode)))
 return steps

def status(c):
 control=Path(c['control']);r=json.loads((control/'status.json').read_text());r['supervisor']={k:(control/('supervisor.'+k)).read_text().strip() for k in ('pid','ppid','pgid','sid')};r['progress']=[json.loads(p.read_text()) for p in sorted(control.glob('progress_*.json'))];r['audits']={}
 for scope in ('smoke','full'):
  for ds in ('rstp','icfg'):
   for mode in ('Text-only','Joint'):
    p=Path(c['output_base'])/scope/ds/'evaluation'/mode/'cpu_audit/report.json';r['audits'][scope+'_'+ds+'_'+mode]=json.loads(p.read_text())['status'] if p.exists() else 'PENDING'
 print(json.dumps(r,ensure_ascii=False,indent=2))

def wait_image(c,control,ds,scope,completed,env):
 # 无Torch/缓存读取；cpu_audit目录原子发布。已发布后由执行器严格复核SHA和身份。
 root=Path(c['image_output'])/scope/ds
 while not (root/'cpu_audit/completed.json').is_file():
  if (root/'failure').exists():raise RuntimeError('image source failed; no retry')
  p=Path(c['image_control'])/'status.json'
  if p.exists() and json.loads(p.read_text()).get('phase')=='FAIL':raise RuntimeError('original image Supervisor failed')
  write_status(control,'WAITING_FOR_AUDITED_IMAGE_SOURCE',dataset=ds,scope=scope,completed_groups=completed,total_groups=4,supervisor_pid=os.getpid());time.sleep(30)

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default=str(DEFAULT));p.add_argument('--allow-smoke-and-auto-full',action='store_true');p.add_argument('--status',action='store_true');a=p.parse_args();c=json.loads(Path(a.config).read_text());control=Path(c['control'])
 if a.status:status(c);return
 if not a.allow_smoke_and_auto_full:p.error('explicit frozen Smoke/automatic Full authorization required')
 control.mkdir(parents=True,exist_ok=True);lock=(control/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (control/'started.json').exists():raise RuntimeError('Attempt consumed; no automatic retry')
 if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise RuntimeError('dirty execution worktree')
 if (control/'cpu_tests.exit').read_text().strip()!='0':raise RuntimeError('CPU tests must PASS')
 tests=json.loads((control/'cpu_tests.json').read_text())
 if tests['config_sha256']!=hashlib.sha256(Path(a.config).read_bytes()).hexdigest() or any(x['exit'] for x in tests['suites']):raise RuntimeError('CPU tests/source configuration identity')
 if Path(c['output_base']).exists():raise RuntimeError('fresh output namespace required')
 started={'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'authorization':'CUHK frozen G60/H7; native targets; Smoke+independent audits then automatic Full; no target feedback/training; shared GPU no VRAM floor','time':time.time()}
 with (control/'started.json').open('x') as f:json.dump(started,f,indent=2);f.flush();os.fsync(f.fileno())
 for k,v in (('pid',os.getpid()),('ppid',os.getppid()),('pgid',os.getpgid(0)),('sid',os.getsid(0))):(control/('supervisor.'+k)).write_text(str(v))
 env=os.environ.copy();env.update(PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',TMPDIR=c['temp_dir'],XDG_CACHE_HOME=c['cache_dir'],TORCH_HOME=c['cache_dir']+'/torch',MPLCONFIGDIR=c['cache_dir']+'/matplotlib',HF_HOME=c['cache_dir']+'/huggingface',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
 for path in (c['temp_dir'],c['cache_dir']):Path(path).mkdir(parents=True,exist_ok=True)
 completed=0
 try:
  for phase,action,ds,scope,mode in pipeline():
   if action=='evaluate':wait_image(c,control,ds,scope,completed,env)
   write_status(control,phase,action=action,dataset=ds,scope=scope,mode=mode,completed_groups=completed,total_groups=4,supervisor_pid=os.getpid(),code_commit=started['code_commit'])
   cmd=[c['python'],'-B','-u','-m','tbps.transfer.gh_execute','--config',a.config,'--action',action,'--scope',scope]
   if ds:cmd+=['--dataset',ds]
   if mode:cmd+=['--mode',mode]
   if action in ('generate','evaluate'):cmd+=['--allow-inference']
   log=control/(scope+'_'+str(ds)+'_'+action+'_'+str(mode)+'.log')
   with log.open('x') as f:
    child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT);(control/'child.pid').write_text(str(child.pid));code=child.wait()
   if code:raise RuntimeError(str(cmd)+' failed exit='+str(code)+'; preserve evidence '+str(log))
   if scope=='full' and action=='audit':completed+=1
  write_status(control,'FINAL_AUDIT_AND_PUBLICATION',completed_groups=completed,total_groups=4,supervisor_pid=os.getpid())
  cmd=[c['python'],'-B','-u','-c',"from tbps.transfer.gh_core import config;from tbps.transfer.gh_results import finalize,publish;c=config("+repr(a.config)+");finalize(c);publish(c)"]
  with (control/'final_publication.log').open('x') as f:subprocess.run(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,check=True)
  write_status(control,'SUCCESS',completed_groups=completed,total_groups=4,supervisor_pid=os.getpid(),publication=json.loads((control/'publication.json').read_text()))
 except BaseException as e:
  with (control/'FAIL.json').open('x') as f:json.dump({'error':repr(e),'traceback':traceback.format_exc()},f,ensure_ascii=False,indent=2)
  write_status(control,'FAIL',error=repr(e),completed_groups=completed,total_groups=4,supervisor_pid=os.getpid());raise
if __name__=='__main__':main()
