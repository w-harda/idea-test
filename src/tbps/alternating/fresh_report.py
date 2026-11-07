"""六节点全部审计后发布表格和两张曲线；不依结果调整训练或选点。"""
from pathlib import Path
import csv,io,json,re
from tbps.artifacts import read_json,sha_file,FileRef,checked
from .fresh_sources import load
from .contracts import require

def publish(path):
    f,c=load(path,'test');base=Path(f['output_base']);root=Path(__file__).resolve().parents[3]
    train=read_json(Path(f['audits']['train'])/'completed/report.json')
    require(train['status']=='GH_V2_TRAIN_CPU_AUDIT_PASS','completed training audit required')
    rows=[];sources=[]
    for n in f['test_rounds']:
        d=Path(f['test_output'])/f'round_{n:02d}'
        a=read_json(d/'cpu_audit/completed/report.json');s=read_json(d/'completed/summary.json')
        require(a['status']=='GH_V2_JOINT_TEST_CPU_AUDIT_PASS' and a['scope']=='official_test'
            and a['metrics']==s['metrics'] and a['round']==n and a['gallery']==3074 and a['query']==6156,'six exact formal results')
        rows.append({'Round':n,'G':'G'+str(n),'H':'H'+str(n),**s['metrics']})
        for p in (d/'cpu_audit/completed/report.json',d/'completed/summary.json'):
            sources.append({'path':str(p),'sha256':sha_file(p)})
    out=root/'docs/framework/results/gh_alternating_v2';out.mkdir(parents=True,exist_ok=True)
    stream=io.StringIO(newline='');w=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    (out/'METRICS.csv').write_text(stream.getvalue())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric,name,color in (('R@1','R1_by_round','#2662bb'),('mAP','mAP_by_round','#c54b50')):
        fig,ax=plt.subplots(figsize=(6.4,4));ax.plot([x['Round'] for x in rows],[x[metric] for x in rows],marker='o',color=color)
        ax.set(xlabel='Alternating training round',ylabel=metric+' (%)',xticks=f['test_rounds']);ax.grid(alpha=.25)
        fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.svg'));plt.close(fig)
    monotonic={k:all(a[k]>=b[k] for a,b in zip(rows,rows[1:])) for k in ('R@1','mAP')}
    evidence={'schema':'gh-v2-six-joint-tests-1','configuration_sha256':sha_file(path),'initialization':train['fresh_initialization'],
        'modes':['Joint'],'dataset':'CUHK-PEDES','victim':'IRRA Source/same-model','gallery':3074,'query':6156,
        'points_predeclared':f['test_rounds'],'metrics':rows,'monotonic_nonincreasing':monotonic,'Test_feedback':False,
        'sources':sources,'training_audit':{'path':str(Path(f['audits']['train'])/'completed/report.json'),
            'sha256':sha_file(Path(f['audits']['train'])/'completed/report.json')}}
    (out/'SOURCE_EVIDENCE.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    lines=['# GH-Alternating-v2：从零60轮与六节点Joint正式Test','',
        '**60轮训练、独立CPU审计与六组CUHK-PEDES × IRRA Joint Test全部PASS。**','',
        'G/H均从原始seed42初始化，双Adam状态从空开始；未加载历史G10/G60、H7/H10/H60或旧动态Bank。IRRA、Reader、属性空间及固定clean资源沿用。原G/H训练无Scheduler，本版本状态为None。','',
        '恢复原H Natural Dynamic-K、原动作与SCST，每个合法原词位最多1个Unicode字符；没有WordBudget1。G保留原K+1、Global/Local/Retain及8/255，H奖励R_sim+R_rank不改。每轮固定2048 Train顺序，先G后完整当前G Bank再H，无Validation优化/评测。','',
        '第10/20/30/40/50/60轮分别导出本次G/H；它们不是历史同名Checkpoint。所有训练完成后才执行六组Test，3074 Gallery/6156原Query、全部同PID正例、FP32 RGB、图像/文本各逐样本原生编码、Query chunk128。各组独立生成自己的图像和文本，仅Joint评分，没有额外模态/模型/数据集实验。','',
        '| 训练轮次 | G | H | R@1 | R@5 | R@10 | mAP |','|---|---|---|---:|---:|---:|---:|']
    for row in rows:lines.append('| '+str(row['Round'])+' | '+row['G']+' | '+row['H']+' | '+' | '.join(f'{row[k]:.2f}' for k in ('R@1','R@5','R@10','mAP'))+' |')
    lines+=['','数值单位%。越低攻击越有效；原精度见METRICS.csv。', '',
        '六个预设节点R@1'+('单调不升' if monotonic['R@1'] else '并非单调不升')+'，mAP'+('单调不升' if monotonic['mAP'] else '并非单调不升')+'。仅一条seed42轨迹，不据此声称统计显著，也不额外用Test选点改变方法。','',
        '![R@1](R1_by_round.png)','','![mAP](mAP_by_round.png)','',
        '每10轮的G/H权重不可变；每轮保存最新完整双模型/Adam/RNG/Bank身份边界并原子切换，旧新产物按退休回执清理，保留全部小型账本。真实失败不自动重试。','',
        '来源、预算和原生独立排名审计见SOURCE_EVIDENCE.json及服务器各组cpu_audit。源码/运行说明见[GH_ALTERNATING_V2.md](../../GH_ALTERNATING_V2.md)。权重、Tensor、原始数据和日志不发布Git。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    p=root/'docs/PROJECT_STATE.md';s=p.read_text();block='<!-- GH_V2_CURRENT_BEGIN -->\n## 当前阶段：GH-Alternating-v2从零60轮及六Joint Test完成\n\n训练与六组正式Test/独立CPU审计全部PASS。[结果报告](framework/results/gh_alternating_v2/REPORT.md)。旧v1和所有历史结果保持。\n<!-- GH_V2_CURRENT_END -->'
    s=re.sub(r'<!-- GH_V2_CURRENT_BEGIN -->.*?<!-- GH_V2_CURRENT_END -->',block,s,flags=re.S);p.write_text(s)
    return {'status':'SIX_JOINT_RESULTS_WRITTEN','rows':rows,'report':str(out/'REPORT.md')}
