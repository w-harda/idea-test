"""严格8/255结果发布器；只读等待审计完成，不启动、重试或恢复实验。"""
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
from zoneinfo import ZoneInfo
import os,sys,json,time,csv,io,subprocess,traceback
ROOT=Path('/home/lzf/ldx/projects/idea-test-tbps-anyattack-eps8-20261109')
CONTROL=Path('/home/lzf/ldx/tmp/anyattack_eps8_20261109')
sys.path.insert(0,str(ROOT/'src'))
from tbps.artifacts import ArtifactStore,read_json,json_bytes,checked,FileRef
from tbps.engines.materialized_cache import ref

def status(value):
 p=CONTROL/'publication_status.json';tmp=p.with_suffix('.tmp');tmp.write_bytes(json_bytes(value));os.replace(tmp,p)

def main():
 identity={'kind':'report-only; no experiment start/retry/resume','pid':os.getpid(),
           'scientific_commit':os.environ['TBPS_SCIENTIFIC_COMMIT'],'start_unix':time.time()}
 status(dict(identity,status='WAITING_FOR_ALL_THREE_FULL_CPU_AUDITS'))
 while True:
  s=read_json(CONTROL/'status.json')
  if s['status']=='FAIL_STOPPED_NO_RETRY':
   status(dict(identity,status='STOPPED_SCIENTIFIC_FAILURE_NO_RETRY',stage=s['stage'],error=s['error']));return
  if s['status']=='SUCCESS_ALL_THREE_FULL_CPU_AUDITED':break
  time.sleep(30)
 cfg=ROOT/'configs/framework/anyattack_eps8.server.json';c=read_json(cfg)
 assert c['eps']==8/255 and c['text_budget_unit']=='word' and c['max_text_replacements_per_query']==1 and c['actual_text_replacements_per_query']==0
 for path,sha in c['producer_sources'].items():checked(FileRef(path,sha))
 base=Path(c['output_base']);reports={};sources={};generation={}
 for scope in ('smoke','full'):
  for name in ('generation','generation_cpu_audit','irra','irra_cpu_audit','rasa','rasa_cpu_audit','aptm','aptm_cpu_audit'):
   store=ArtifactStore(base/scope/name,new_base=base);key='result' if name.endswith('audit') else 'completed'
   store.verify(key);r=read_json(store.root/key/'report.json');assert r['status']=='PASS'
   if name in ('irra','rasa','aptm'):store.verify('result')
   sources[scope+'/'+name]=asdict(ref(store.root/key/'manifest.json'))
   if name=='generation':
    assert r['model_loads']==r['encoder_forwards']==r['decoder_forwards']==0;generation[scope]=r
   if name=='generation_cpu_audit':assert r['max_linf_double']<=8/255 and r['budget_tolerance']==0 and not r['CUDA_initialized'] and r['raw_noise_unchanged']
   if name.endswith('audit') and name!='generation_cpu_audit':
    assert (r['gallery'],r['query'])==((10,10) if scope=='smoke' else (3074,6156)) and not r['CUDA_initialized']
    reports[scope+'/'+name[:-10]]=r
 names=('irra','rasa','aptm');keys=('R@1','R@5','R@10','mAP');rows=[]
 assert len({reports['full/'+n]['source_collection_sha256'] for n in names})==1
 for n in names:
  for mode in ('Clean','Image-only'):rows.append({'Model':'RaSa' if n=='rasa' else n.upper(),'Mode':mode,**reports['full/'+n]['metrics'][mode]})
 out=ROOT/'docs/framework/results/anyattack_eps8';out.mkdir(parents=True,exist_ok=False)
 text=io.StringIO(newline='');writer=csv.DictWriter(text,fieldnames=['Model','Mode',*keys,'mINP'],lineterminator='\n');writer.writeheader();writer.writerows(rows)
 (out/'METRICS.csv').write_text(text.getvalue())
 end=datetime.fromtimestamp(s['finished_unix'],ZoneInfo('Asia/Shanghai')).isoformat()
 evidence={'schema':'tbps-anyattack-eps8-evidence-1','scientific_commit':identity['scientific_commit'],'config':asdict(ref(cfg)),
  'decoder':c['decoder'],'CLIP':c['clip'],'raw_parent_config':c['raw_parent_config'],'sources':sources,'supervisor':s,
  'strict_image_budget':8/255,'budget_tolerance':0,'text_budget_unit':'word','max_text_replacements_per_query':1,
  'actual_text_replacements_per_query':0,'generation_model_loads':0,'generation_neural_forwards':0,'training':False,
  'fine_tuning':False,'completed_asia_shanghai':end,'old16_results':'supplemental; not unified-budget comparison'}
 (out/'SOURCE_EVIDENCE.json').write_bytes(json_bytes(evidence))
 ga=read_json(base/'full/generation_cpu_audit/result/report.json')
 lines=['# AnyAttack统一预算8/255：CUHK三模型正式Full','',
  '三组3074 Gallery / 6156原Query Full及各自CPU审计全部PASS。完成时间：'+end+'。','',
  '图像实际L∞≤8/255，FP32向内投影，审计容差0。文本上限每条Query1词；用户确认AnyAttack没有文本分支就不另加，本组实际0替换，Image-only。','',
  '| Victim | 输入 | R@1 (%) | R@5 (%) | R@10 (%) | mAP (%) |','|---|---|---:|---:|---:|---:|']
 for r in rows:lines.append('| '+r['Model']+' | '+r['Mode']+' | '+' | '.join(f'{r[k]:.6f}' for k in keys)+' |')
 lines+=['',f"实际double测量最大L∞={ga['max_linf_double']:.15f}，上限8/255={8/255:.15f}，审计容差0。",'',
  '原官方coco_cos.pt、CLIP ViT-B/32、原始Decoder噪声和embedding保持；从旧封存raw_noise重新按8/255裁剪和严格投影。新增Decoder/CLIP加载与前向0，训练/微调/Backward/Optimizer/JVP均0。','',
  '原16/255数据完整保留为补充实验，不填入本表。三模型共用同一8/255攻击集合，固定异PID参考映射的Test PID访问条件仍需披露。原生RaSa Top128融合及APTM完整Query/Gallery归一化保持；所有模型权重/buffers前后exact。','',
  '[完整协议](../../ANYATTACK_EPS8.md)。执行源码Commit `'+identity['scientific_commit']+'`。状态：SUCCESS_ALL_THREE_FULL_CPU_AUDITED。']
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
 for name in ('docs/PROJECT_STATE.md','docs/EXPERIMENT_REGISTRY.md'):
  f=ROOT/name;f.write_text(f.read_text()+'\n\n## AnyAttack统一8/255 Full完成\n\n三模型完整3074/6156及独立CPU审计PASS，严格8/255、文本上限1词且实际0，无训练、无新Decoder前向。详见[结果](framework/results/anyattack_eps8/REPORT.md)。\n')
 allowed={'docs/PROJECT_STATE.md','docs/EXPERIMENT_REGISTRY.md','docs/framework/results/anyattack_eps8/'}
 dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).splitlines()
 assert all(any(row[3:]==a or row[3:].startswith(a) for a in allowed) for row in dirty),dirty
 subprocess.run(['git','add',*sorted(allowed)],cwd=ROOT,check=True);subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
 subprocess.run(['git','-c','user.name=w-harda','-c','user.email=wharda@163.com','commit','-m','docs(anyattack): 发布统一8/255三模型Full与独立审计'],cwd=ROOT,check=True)
 subprocess.run(['git','push','origin','codex/anyattack-tbps-eps8'],cwd=ROOT,check=True)
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/anyattack-tbps-eps8'],cwd=ROOT,text=True).split()[0];assert head==remote
 status(dict(identity,status='SUCCESS_REPORT_PUBLISHED',commit=head,remote_sha=remote,finished_unix=time.time()))

if __name__=='__main__':
 try:main()
 except BaseException as exc:
  status({'status':'FAIL_REPORT_ONLY_NO_SCIENTIFIC_RETRY','error':str(exc),'traceback':traceback.format_exc()});raise
