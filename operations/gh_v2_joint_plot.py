"""仅从固定六行已验证指标绘图；不导入模型、训练或评测。"""
import argparse,csv,hashlib,json,math
from pathlib import Path

def verified_rows(path,expected):
    p=Path(path)
    if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError('metrics SHA mismatch')
    with p.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['Round','R@1','R@5','R@10','mAP']:raise ValueError('exact metric columns required')
        rows=[]
        for r in reader:
            if set(r)!=set(reader.fieldnames):raise ValueError('exact CSV row width')
            rows.append({'Round':int(r['Round']),**{k:float(r[k]) for k in ('R@1','R@5','R@10','mAP')}})
    if [r['Round'] for r in rows]!=[10,20,30,40,50,60]:raise ValueError('six predetermined endpoints required')
    if not all(math.isfinite(v) and 0<=v<=100 for r in rows for k,v in r.items() if k!='Round'):raise ValueError('finite percent metrics required')
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--metrics',required=True);ap.add_argument('--sha256',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    rows=verified_rows(a.metrics,a.sha256);out=Path(a.output).resolve()
    expected=Path('/home/lzf/ldx/projects/idea-test-tbps-gh-alternating-v2-20261107/docs/framework/results/gh_alternating_v2_recovery')
    if out!=expected or out.is_symlink():raise ValueError('only dedicated publication directory')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric,name,color in (('R@1','R1_by_round','#2662bb'),('mAP','mAP_by_round','#c54b50')):
        fig,ax=plt.subplots(figsize=(6.4,4));ax.plot([r['Round'] for r in rows],[r[metric] for r in rows],marker='o',color=color)
        ax.set(xlabel='Alternating training round',ylabel=metric+' (%)',xticks=[10,20,30,40,50,60]);ax.grid(alpha=.25)
        fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.svg'));plt.close(fig)
    if rows!=verified_rows(a.metrics,a.sha256):raise ValueError('metrics changed during plot')
    print(json.dumps({'status':'VERIFIED_SIX_POINT_PLOTS_WRITTEN','matplotlib':matplotlib.__version__,'metrics_sha256':a.sha256}))
if __name__=='__main__':main()
