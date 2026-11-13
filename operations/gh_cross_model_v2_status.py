"""只读一次进度与审计结果，不执行或恢复实验。"""
from pathlib import Path
import json,datetime,os
ROOT=Path(__file__).resolve().parents[1]
c=json.loads((ROOT/'configs/framework/gh_cross_model_v2.server.json').read_text());ctl=Path(c['control'])
print('检查时间',datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
for name in ('status.json','identity.json','child.json'):
    p=ctl/name
    if p.exists():print(name,p.read_text())
if (ctl/'child.json').exists():
    d=json.loads((ctl/'child.json').read_text());log=ctl/(d['label']+'.log')
    if log.exists():
        with log.open('rb') as f:f.seek(max(0,log.stat().st_size-3500));print('日志末尾',f.read().decode('utf-8',errors='replace'))
for method in c['methods']:
    for model in c['models']:
        for scope in ('smoke','full'):
            p=Path(c['output_base'])/'evaluation'/method/model/scope/'cpu_audit/completed/report.json'
            if p.exists():
                a=json.loads(p.read_text());print(method,model,scope,a['status'],json.dumps(a['groups'],ensure_ascii=False))
