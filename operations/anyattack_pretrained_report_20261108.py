"""仅报告发布：等待本次三Full及独立审计，不执行/恢复/重试实验。"""
from pathlib import Path
from dataclasses import asdict
import os,sys,time,json,csv,io,subprocess,hashlib,traceback

ROOT=Path('/home/lzf/ldx/projects/idea-test-tbps-anyattack-pretrained-20261108')
CONTROL=Path('/home/lzf/ldx/tmp/anyattack_pretrained_20261108')
sys.path.insert(0,str(ROOT/'src'))
from tbps.artifacts import ArtifactStore,read_json,json_bytes,digest,checked,FileRef
from tbps.engines.materialized_cache import ref

def status(value):
    p=CONTROL/'publication_status.json';tmp=p.with_suffix('.tmp');tmp.write_bytes(json_bytes(value));os.replace(tmp,p)

def main():
    identity={'kind':'report-only; no experiment start/retry/resume','pid':os.getpid(),
              'scientific_commit':'536d54a77fec8ee2e7f60f57439e2c50c2652f72','start_unix':time.time()}
    status(dict(identity,status='WAITING_FOR_ALL_THREE_FULL_CPU_AUDITS'))
    while True:
        s=read_json(CONTROL/'status.json')
        if s['status']=='FAIL_STOPPED_NO_RETRY':
            status(dict(identity,status='STOPPED_SCIENTIFIC_FAILURE_NO_RETRY',stage=s['stage'],error=s['error']));return
        if s['status']=='SUCCESS_ALL_THREE_FULL_CPU_AUDITED':break
        time.sleep(30)
    cfg=ROOT/'configs/framework/anyattack_pretrained.server.json';c=read_json(cfg)
    base=Path(c['output_base']);reports={};sources={};generations={}
    for scope in ('smoke','full'):
        for name in ('generation','generation_cpu_audit','irra','irra_cpu_audit','rasa','rasa_cpu_audit','aptm','aptm_cpu_audit'):
            store=ArtifactStore(base/scope/name,new_base=base)
            key='result' if name.endswith('audit') else 'completed';store.verify(key)
            r=read_json(store.root/key/'report.json');assert r['status']=='PASS'
            sources[f'{scope}/{name}']=asdict(ref(store.root/key/'manifest.json'))
            if name=='generation':generations[scope]=r
            if name.endswith('audit') and name!='generation_cpu_audit':
                assert r['gallery']==(10 if scope=='smoke' else 3074) and r['query']==(10 if scope=='smoke' else 6156)
                assert r['CUDA_initialized'] is False
                reports[f'{scope}/{name[:-10]}']=r
    names=('irra','rasa','aptm');keys=('R@1','R@5','R@10','mAP');rows=[]
    # key suffix '_cpu_audit'长度10；所有模型绑定同一Formal生成集合。
    assert len({reports['full/'+n]['source_collection_sha256'] for n in names})==1
    for n in names:
        for mode in ('Clean','Image-only'):rows.append({'Model':n.upper() if n!='rasa' else 'RaSa','Mode':mode,**reports['full/'+n]['metrics'][mode]})
    out=ROOT/'docs/framework/results/anyattack_pretrained';out.mkdir(parents=True,exist_ok=False)
    text=io.StringIO(newline='');writer=csv.DictWriter(text,fieldnames=['Model','Mode',*keys,'mINP']);writer.writeheader();writer.writerows(rows)
    (out/'METRICS.csv').write_text(text.getvalue())
    evidence={'schema':'tbps-anyattack-published-evidence-1','scientific_commit':identity['scientific_commit'],
        'config':asdict(ref(cfg)),'decoder':c['decoder'],'CLIP':c['clip'],'official_commit':c['official_commit'],
        'HF_revision':c['HF_revision'],'sources':sources,'supervisor':s,
        'source_training_or_fine_tuning':False,'victim_feedback':False,'test_PID_access':'fixed wrong-PID reference bijection only',
        'nominal_eps':c['eps'],'scope':'CUHK-PEDES three native victims; official COCO Decoder zero-shot transfer',
        'R2TUA_exact_adaptation':'UNKNOWN; no disclosed adaptation code/checkpoint identified'}
    (out/'SOURCE_EVIDENCE.json').write_bytes(json_bytes(evidence))
    lines=['# AnyAttack官方预训练Decoder：CUHK-PEDES三模型Full结果','',
        '三组正式Full及各自独立CPU审计全部PASS；3074 Gallery / 6156原Query。官方coco_cos.pt + CLIP ViT-B/32固定权重，无TBPS训练或微调，无反向/Optimizer/JVP，无Victim反馈。三模型使用同一图像攻击集合。','',
        '| Victim | 输入 | R@1 (%) | R@5 (%) | R@10 (%) | mAP (%) |','|---|---|---:|---:|---:|---:|']
    for r in rows:lines.append('| '+r['Model']+' | '+r['Mode']+' | '+' | '.join(f'{r[k]:.6f}' for k in keys)+' |')
    lines+=['','名义预算16/255，原始文本不变；参考图由固定异PID映射选定，Test PID仅用于该映射。保留官方CLIP原始特征及Decoder，通过CPU可重放的噪声尺寸桥接适配384×128图像。该信息条件和预算不同于Idea/SGA，不作等预算优劣结论。','',
        '## 审计与时间','',
        '两组生成（Smoke、Full）与六组模型评分分别在独立CPU进程审计。全部Manifest/文件SHA、来源、模型参数/buffers不变、完整原始文本、异PID参考一一映射、像素桥接/预算、完整分数重排名均核验。RaSa保留Top128融合；APTM保留全Query/Gallery归一化；没有以Smoke代Full。','',
        '| 阶段 | 范围 | 秒 | peak RSS MiB |','|---|---|---:|---:|']
    for receipt in s['completed']:
        stage=receipt['stage'];parts=stage.split('_');scope=parts[1];action=parts[2]
        if action not in ('generate','evaluate','audit-generation','audit-evaluation'):continue
        name=parts[3] if len(parts)>3 else 'generation'
        if action.startswith('audit-'):name+='_cpu_audit'
        key='result' if name.endswith('audit') else 'completed'
        r=read_json(base/scope/name/key/'report.json')
        lines.append(f"| {action} {name} | {scope} | {r['elapsed_seconds']:.3f} | {r['peak_RSS_bytes']/1024**2:.2f} |")
    ga=read_json(base/'full/generation_cpu_audit/result/report.json')
    lines+=['',f"Formal生成耗时{generations['full']['elapsed_seconds']:.3f}秒，实际double测量最大L∞={ga['max_linf_double']:.12f}，名义16/255={c['eps']:.12f}，超出名义值的浮点端点差={ga['max_roundoff_above_nominal']:.12g}。未把FP32端点舍入伪称为严格实数零误差。",'',
        '## 与R²TUA表1的关系','',
        'R²TUA公开表1的AnyAttack（CUHK）R@1/mAP为RaSa8.48/6.49、APTM10.15/6.92、IRRA33.56/26.26。其表5还记录AnyAttack训练16h、攻击10.8ms，故3090环境不能证明没有训练。本次遵守用户免训练要求，直接迁移官方COCO Decoder；R²TUA精确Decoder身份/微调/适配代码尚未在公开来源中核实，不能称本次是其表1精确复现。三数据集/模型分数不同本身不能证明分别重训。','',
        '[完整协议及论文/官方代码来源](../../ANYATTACK_PRETRAINED.md)。运行源码冻结Commit `'+identity['scientific_commit']+'`；后续报告发布仅增加文档/独立报告辅助，未改数值源码。','',
        '控制目录：`'+str(CONTROL)+'`；`status.json=SUCCESS_ALL_THREE_FULL_CPU_AUDITED`。原G/H、SGA及其他历史资产保持。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    for f in ('docs/PROJECT_STATE.md','docs/EXPERIMENT_REGISTRY.md'):
        p=ROOT/f;p.write_text(p.read_text()+'\n\n## AnyAttack免训练正式Full完成（2026-11-08）\n\nCUHK-PEDES × IRRA/RaSa/APTM三组Full与独立CPU审计全部PASS，原3074图库/6156文本，官方coco_cos.pt直接迁移，无TBPS训练或微调。名义16/255、异PID参考映射及与R²TUA协议差异见[结果报告](framework/results/anyattack_pretrained/REPORT.md)。\n')
    allowed={'docs/PROJECT_STATE.md','docs/EXPERIMENT_REGISTRY.md','docs/framework/results/anyattack_pretrained/'}
    dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).splitlines()
    assert all(any(row[3:]==a or row[3:].startswith(a) for a in allowed) for row in dirty),dirty
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','add',*sorted(allowed)],cwd=ROOT,check=True)
    subprocess.run(['git','-c','user.name=w-harda','-c','user.email=wharda@163.com','commit','-m','docs(anyattack): 发布免训练迁移三模型Full及独立审计结果'],cwd=ROOT,check=True)
    subprocess.run(['git','push','origin','codex/anyattack-tbps-pretrained'],cwd=ROOT,check=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/anyattack-tbps-pretrained'],cwd=ROOT,text=True).split()[0]
    assert head==remote
    status(dict(identity,status='SUCCESS_REPORT_PUBLISHED',commit=head,remote_sha=remote,finished_unix=time.time()))

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        status({'status':'FAIL_REPORT_ONLY_NO_SCIENTIFIC_RETRY','error':str(exc),'traceback':traceback.format_exc()});raise
