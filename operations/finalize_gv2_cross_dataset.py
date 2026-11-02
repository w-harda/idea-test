#!/usr/bin/env python3
"""实验完成回执触发的小型报告Git发布器；不启动/恢复/重跑任何实验。"""
from pathlib import Path
import argparse,csv,fcntl,json,os,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[1]
REPORT='docs/framework/results/gv2_cross_dataset_v2'
def decision(row):
 if row.get('phase')=='FAIL':return 'STOP'
 if row.get('phase')=='SUCCESS':return 'PUBLISH'
 return 'WAIT'
def verify_rows(rows,audits):
 expected=[(label,mode,ds,key) for ds,label in [('rstp','RSTPReid'),('icfg','ICFG-PEDES')] for mode,key in [('Clean','Clean'),('Image-only (G-v2 E60)','Image-only')]]
 if len(rows)!=4:raise ValueError('exactly four audited Full rows required')
 for row,(label,mode,ds,key) in zip(rows,expected):
  if (row['Dataset'],row['Victim'],row['Mode'])!=(label,'IRRA',mode):raise ValueError('Full row identity mismatch')
  if any(float(row[k])!=audits[ds]['groups'][key][k] for k in ('R@1','R@5','R@10','mAP')):raise ValueError('Full row metric mismatch')
def publish(c,control):
 from tbps.transfer.gv2_core import load_config,store_for,read_json,require,verify_history
 from tbps.artifacts import digest
 c=load_config(ROOT/'configs/framework/gv2_cross_dataset.server.json');audits={}
 require(read_json(control/'status.json')['phase']=='SUCCESS','main Supervisor SUCCESS receipt required')
 for scope in ('smoke','full'):
  for ds in ('rstp','icfg'):
   s=store_for(c,ds,scope);s.verify('cpu_audit');a=read_json(s.root/'cpu_audit/report.json')
   require(a['status']=='PASS' and a['scope']==scope and a['dataset']==ds and a['config_sha256']==c['sha256'],'all four independent audits required')
   if scope=='full':
    require((a['Gallery'],a['Query'])==({'rstp':(1000,2000),'icfg':(19848,19848)}[ds]),'complete official Full size required');audits[ds]=a
 evidence=read_json(ROOT/REPORT/'SOURCE_EVIDENCE.json');started=read_json(control/'started.json')
 require(evidence['status']=='SUCCESS' and evidence['execution_commit']==started['code_commit'] and evidence['config_sha256']==c['sha256'],'completed report execution identity')
 require(evidence['original_failed_attempt_SHA_and_stat_exact_files']==264,'v1 FAIL protection')
 verify_history(c)
 with (ROOT/REPORT/'METRICS.csv').open() as f:rows=list(csv.DictReader(f))
 verify_rows(rows,audits)
 for rel,prefix,body in [('docs/PROJECT_STATE.md','## G-v2跨数据集v2',
   '## G-v2跨数据集v2 Full全部通过并发布（2026-11-02）\n\n141项审计修复CPU回归通过；两目标真实Smoke及独立CPU审计、RSTP1000×2000和ICFG19848×19848完整Full/独立重排名审计全部PASS。冻结CUHK G E60、原生两域IRRA、原始Caption和完整PID正例，FP32/8/255/Batch不变。旧v1 FAIL及264文件SHA/stat保持。[正式表格与来源](framework/results/gv2_cross_dataset_v2/REPORT.md)。\n'),
  ('docs/EXPERIMENT_REGISTRY.md','## G-v2跨数据集v2',
   '## G-v2跨数据集v2：SUCCESS（2026-11-02）\n\n两个完整Full及独立CPU审计PASS，4行正式Clean/Image-only指标发布于framework/results/gv2_cross_dataset_v2/。执行来源为618cc19，结果收尾只在两Smoke/两Full审计PASS及主监督器SUCCESS后运行，不启动任何实验。原v1失败现场永久保留，无目标域训练/微调或额外H。\n')]:
  p=ROOT/rel;text=p.read_text();start=text.index(prefix);end=text.find('\n## ',start+3)
  p.write_text(text[:start]+body+(text[end:] if end>=0 else ''),encoding='utf-8')
 git=lambda *a:subprocess.check_output(['git',*a],cwd=ROOT,text=True,timeout=120).strip()
 branch=git('branch','--show-current');require(branch=='codex/gv2-cross-dataset-v2','publication branch identity')
 allowed={REPORT+'/'+n for n in ('REPORT.md','METRICS.csv','METRICS_2DP.csv','SOURCE_EVIDENCE.json') }|{'docs/PROJECT_STATE.md','docs/EXPERIMENT_REGISTRY.md'}
 paths={line[3:] for line in subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=ROOT,text=True,timeout=120).splitlines()}
 require(paths and paths<=allowed,'unexpected dirty files; preserve, no automatic retry')
 git('diff','--check');git('add',*sorted(paths));git('diff','--cached','--check')
 name=git('log','-1','--format=%an');email=git('log','-1','--format=%ae')
 git('-c','user.name='+name,'-c','user.email='+email,'commit','-m','docs: 发布G-v2跨数据集Full指标与独立审计证据')
 git('push','origin',branch);head=git('rev-parse','HEAD');remote=git('ls-remote','origin','refs/heads/'+branch).split()[0]
 require(head==remote and not git('status','--porcelain'),'published HEAD and clean tree required')
 return {'phase':'PUBLISHED','commit':head,'rows':rows,'report':'https://github.com/w-harda/idea-test/blob/'+head+'/'+REPORT+'/REPORT.md'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'configs/framework/gv2_cross_dataset.server.json'));p.add_argument('--allow-report-publication',action='store_true');a=p.parse_args()
 if not a.allow_report_publication:p.error('explicit result-publication authorization required')
 c=json.loads(Path(a.config).read_text());control=Path(c['control'])
 lock=(control/'publication.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 claim=control/'publication.started.json'
 with claim.open('x') as f:json.dump({'pid':os.getpid(),'time':time.time(),'no_experiment_calls':True},f);f.flush();os.fsync(f.fileno())
 def state(row):
  row['time']=time.time();t=control/'publication_status.tmp';t.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n');t.replace(control/'publication_status.json')
 try:
  state({'phase':'WAITING_FOR_AUDITED_FULL'})
  while True:
   d=decision(json.loads((control/'status.json').read_text()))
   if d=='STOP':state({'phase':'STOPPED_MAIN_FAILURE','automatic_retry':False});return
   if d=='PUBLISH':break
   time.sleep(15)
  state({'phase':'PUBLISHING'});state(publish(c,control))
 except BaseException as e:
  state({'phase':'PUBLICATION_FAIL','error':repr(e),'traceback':traceback.format_exc(),'automatic_retry':False});raise
if __name__=='__main__':main()
