"""仅全部独立CPU审计PASS后汇总及发布；不运行神经模型。"""
from .gh_core import *
import csv,subprocess
MODES4=('Clean','Image-only','Text-only','Joint')

def collect(c):
 root=ArtifactStore(c['output_base'],new_base=c['output_base']);root.verify('preflight');evidence={};values={}
 for ds in DATASETS:
  old,m,image_audit,_=dependency(c,ds,'full');old_summary=read_json(old.root/'completed/summary.json')
  groups={mode:old_summary['groups'][mode] for mode in MODES4[:2]}
  refs={'image_audit':asdict(ref(old.root/'cpu_audit/manifest.json')),'image_summary':asdict(ref(old.root/'completed/manifest.json'))}
  for mode in MODES:
   s=result_store(c,ds,'full',mode);s.verify('cpu_audit');s.verify('completed');a=read_json(s.root/'cpu_audit/report.json')
   require(a['status']=='PASS' and a['dataset']==ds and a['scope']=='full' and a['config_sha256']==c['sha256'],'no unaudited Full publication')
   summary=read_json(s.root/'completed/summary.json');require(a['groups'][mode]==summary['groups'][mode],'audit metric identity')
   groups[mode]=a['groups'][mode];refs[mode]={'audit':asdict(ref(s.root/'cpu_audit/manifest.json')),'summary':asdict(ref(s.root/'completed/manifest.json'))}
  values[ds]=groups;evidence[ds]={'Gallery':len(m.gallery),'Query':len(m.queries),'sources':refs}
 check_protection(c)
 return values,evidence

def table_bytes(values,rounded=False):
 f=io.StringIO();w=csv.writer(f,lineterminator='\n');w.writerow(('Dataset','Victim','Mode',*KEYS))
 for ds in DATASETS:
  for mode in MODES4:w.writerow((SPECS[ds][0],'IRRA',mode,*(format(values[ds][mode][k],'.2f') if rounded else values[ds][mode][k] for k in KEYS)))
 return f.getvalue().encode()

def finalize(c):
 require(not torch.cuda.is_initialized(),'publication CPU only')
 with image.no_backward(cpu=True):
  values,evidence=collect(c)
  report={'schema':'gh-cross-dataset-final-1','status':'PASS','groups':values,'evidence':evidence,'G_checkpoint':c['G_checkpoint'],'H_checkpoint':c['H_checkpoint'],'config_sha256':c['sha256'],'attack_training':'CUHK-PEDES only; frozen G60/H7','target_feedback':False,'modes':{k:v for k,v in zip(MODES4,('clean_gallery + clean_text','adv_gallery + clean_text','clean_gallery + adv_text','adv_gallery + adv_text'))}}
  lines=['# CUHK G-v2 E60 + H-v1 E7跨数据集冻结迁移结果','','FROZEN：所有Full和独立CPU审计均已PASS；无重新训练或微调。Clean/Image-only只读引用v2原结果；Text-only/Joint共用各目标新H文本和特征。','','| Dataset | Victim | Mode | R@1 | R@5 | R@10 | mAP |','|---|---|---|---:|---:|---:|---:|']
  for ds in DATASETS:
   for mode in MODES4:lines.append('| '+' | '.join((SPECS[ds][0],'IRRA',mode,*(format(values[ds][mode][k],'.2f') for k in KEYS)))+' |')
  lines += ['',f"G SHA：`{G_SHA}`；H SHA：`{H_SHA}`。",'',f"原始完整精度CSV、来源与审计绑定：`{c['output_base']}/final/`。",'','目标Test PID仅用于正式评分/独立CPU排名审计；H只接收原Caption，无目标反馈、配对图或标签输入。合法NOOP保留；未重跑G、图像IRRA或历史Clean/Image-only评分。']
  md=('\n'.join(lines)+'\n').encode();store=ArtifactStore(c['output_base'],new_base=c['output_base'])
  files={'REPORT.md':md,'SOURCE_EVIDENCE.json':json_bytes(report),'METRICS.csv':table_bytes(values),'METRICS_2DP.csv':table_bytes(values,True)}
  store.publish('final',files,{'kind':'all-eight-Full-modes-audited'},sources=(ref(c['path']),))
  dest=ROOT/'docs/framework/results/gh_cross_dataset';dest.mkdir(parents=True,exist_ok=True)
  for name,data in files.items():(dest/name).write_bytes(data)
  p=ROOT/'docs/PROJECT_STATE.md';p.write_text('## FROZEN：G60 + H7跨数据集8组Full与独立CPU审计全部PASS\n\n正式CSV及来源见[结果报告](framework/results/gh_cross_dataset/REPORT.md)。未重新训练/微调；原v2与失败现场保持只读。\n\n'+p.read_text())
 return report

def publish(c):
 git=lambda *args:subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
 allowed={'docs/PROJECT_STATE.md'}|{str(p.relative_to(ROOT)) for p in (ROOT/'docs/framework/results/gh_cross_dataset').glob('*')}
 dirty=set(git('diff','--name-only').splitlines())|set(git('ls-files','--others','--exclude-standard').splitlines())
 require(dirty and dirty<=allowed,'result publication limited to small report/CSV/source files')
 git('diff','--check');git('add','--',*sorted(dirty));name=git('log','-1','--format=%an');email=git('log','-1','--format=%ae')
 git('-c','user.name='+name,'-c','user.email='+email,'commit','-m','docs: publish audited G60 H7 cross-dataset Full results')
 commit=git('rev-parse','HEAD');branch=git('branch','--show-current');require(branch=='codex/gh-cross-dataset','isolated publication branch')
 git('push','-u','origin','HEAD:refs/heads/'+branch);require(git('ls-remote','origin','refs/heads/'+branch).split()[0]==commit,'published remote SHA')
 receipt={'status':'PASS','commit':commit,'remote':branch,'time':time.time()};p=Path(c['control'])/'publication.json'
 with p.open('x') as f:f.write(json_bytes(receipt).decode())
 return receipt
