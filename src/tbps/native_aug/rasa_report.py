"""从独立审计回执汇总 RaSa 源模型的正式结果。"""
import csv
import io
from .common import OUTPUT, ROUNDS, ArtifactStore, json_bytes, need, read_json, ref

def summarize(config):
    targets = config['evaluation_targets']
    rows, sources = [], []
    for r in ROUNDS:
        for target in targets:
            root = OUTPUT / 'rasa' / 'evaluation' / target / f'round_{r:02d}' / 'cpu_audit'
            ArtifactStore(root, new_base=OUTPUT).verify('completed')
            path = root / 'completed' / 'report.json'
            audit = read_json(path)
            need(audit['status']=='NATIVE_AUG_TEST_CPU_AUDIT_PASS'
                 and audit['source']=='rasa' and audit['target']==target and audit['round']==r,
                 'formal RaSa-source audit identity')
            rows.append({'Round':r,'Source':'RaSa','Target':target,
                         **{key:audit['metrics'][key] for key in ('R@1','R@5','R@10','mAP')}})
            sources.append(ref(path))
    csv_data=io.StringIO()
    writer=csv.DictWriter(csv_data,fieldnames=tuple(rows[0]),lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    text=['# RaSa 源 Push-pull + 图像增强 Joint','',
          'CUHK-PEDES 全量测试，图库 3074 张，查询 6156 条。所有指标单位为 %。','',
          '| 轮次 | 目标 | R@1 | R@5 | R@10 | mAP |','|---:|---|---:|---:|---:|---:|']
    for row in rows:
        text.append('| {} | {} | {} |'.format(row['Round'],row['Target'],
                     ' | '.join(f'{row[k]:.2f}' for k in ('R@1','R@5','R@10','mAP'))))
    payload={'METRICS.csv':csv_data.getvalue().encode(),'REPORT.md':('\n'.join(text)+'\n').encode(),
             'results.json':json_bytes({'source':'rasa','rows':rows,'audit_pass':True})}
    ArtifactStore(OUTPUT/'rasa'/'summary',new_base=OUTPUT).publish('completed',payload,
        {'source':'rasa','scope':'official_test','configuration':ref(config['_config_path']).sha256},
        sources=tuple(sources))
