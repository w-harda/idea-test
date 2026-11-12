"""四组正式独立审计完成后发布迁移表格及真实来源。"""
from pathlib import Path
import csv,io
from .common import *
from .audit import protection

def publish(path):
    c=config(path);rows=[];evidence=[]
    for method in METHODS:
        protection(c,method,rehash=True)
        for model in MODELS:
            root=Path(c['output_base'])/'evaluation'/method/model/'full'
            s=ArtifactStore(root,new_base=c['output_base']);s.verify('completed')
            a=ArtifactStore(root/'cpu_audit',new_base=c['output_base']);a.verify('completed')
            audit=read_json(a.root/'completed/report.json');result=read_json(root/'completed/summary.json')
            need(audit['status']=='GH_TRANSFER_CPU_AUDIT_PASS' and audit['scope']=='full' and audit['gallery']==3074 and audit['query']==6156
                 and audit['groups']==result['groups'] and audit['configuration_sha256']==sha_file(path),'four complete audited formal groups')
            for mode in ('Clean','Joint'):rows.append({'method':method,'training_surrogate':'IRRA','round':60,'victim':model,'mode':mode,**{k:result['groups'][mode][k] for k in METRICS}})
            evidence.append({'method':method,'victim':model,'summary':asdict(ref(root/'completed/summary.json')),
                             'audit':asdict(ref(a.root/'completed/report.json')),'G':c['historical'][method]['G'],'H':c['historical'][method]['H']})
    out=Path(__file__).resolve().parents[3]/'docs/framework/results/gh_cross_model_v1';out.mkdir(parents=True,exist_ok=True)
    stream=io.StringIO(newline='');w=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    (out/'METRICS.csv').write_text(stream.getvalue())
    (out/'SOURCE_EVIDENCE.json').write_bytes(json_bytes({'configuration':asdict(ref(path)),'groups':evidence,'rows':rows,'no_training':True,'target_feedback':False}))
    lines=['# Cooperative与Push-pull：IRRA训练、RaSa/APTM迁移评测','',
           '四组CUHK-PEDES完整3074 Gallery／6156 Query正式评测与独立CPU审计通过。固定各版Round60配对权重，复用原IRRA正式Test封存的FP32攻击RGB与Unicode文本。','',
           '图像预算相对384×128的原始IRRA输入严格L∞≤8/255；目标模型随后执行自己的Tensor预处理，不将插值后的数值另称相同像素空间预算。文本沿用原Natural-K，每个原词最多替换1个Unicode字符。未对目标模型训练、微调或获取攻击梯度。','',
           'RaSa保留Top-128原生ITM融合；APTM保留Top-128融合与完整Query/Gallery矩阵归一化。Clean与Joint均独立编码和评分。','',
           '| 方法 | 目标模型 | 模式 | R@1 | R@5 | R@10 | mAP |','|---|---|---|---:|---:|---:|---:|']
    for r in rows:lines.append('| '+r['method']+' | '+r['victim']+' | '+r['mode']+' | '+' | '.join(f'{r[k]:.2f}' for k in METRICS)+' |')
    lines+=['','数值单位%，Joint越低表示攻击越强。Round60在运行前固定，未按目标Test效果选Checkpoint。原Cooperative报告阶段FAIL保留，本迁移实验不重试该流水线。','',
            '原精度及来源见METRICS.csv与SOURCE_EVIDENCE.json。权重、原始攻击产物、缓存及日志只保留在服务器。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return {'status':'FOUR_TRANSFER_RESULTS_WRITTEN','rows':rows}
