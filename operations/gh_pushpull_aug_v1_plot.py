"""仅报告绘图：独立解释器，无Torch、模型、GPU、训练或评测。"""
from pathlib import Path
import argparse,csv,hashlib,os

def main():
    p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--output',required=True)
    p.add_argument('--csv-sha256',required=True);p.add_argument('--matplotlib-version',required=True);a=p.parse_args()
    source=Path(a.csv);out=Path(a.output).resolve()
    if not out.is_relative_to(Path('/home/lzf/ldx')):raise ValueError('report output must stay under /home/lzf/ldx')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=a.csv_sha256:raise ValueError('metrics SHA mismatch')
    os.environ.setdefault('MPLCONFIGDIR','/home/lzf/ldx/tmp/gh_pushpull_aug_v1_report_cache')
    import matplotlib
    if matplotlib.__version__!=a.matplotlib_version:raise ValueError('report dependency version mismatch')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=list(csv.DictReader(source.open()));rounds=[int(r['Round']) for r in rows]
    if rounds!=[10,20,30,40,50,60]:raise ValueError('all six exact formal endpoints required')
    for metric,name,color in (('R@1','R1_by_round','#2662bb'),('mAP','mAP_by_round','#c54b50')):
        fig,ax=plt.subplots(figsize=(6.4,4));ax.plot(rounds,[float(r[metric]) for r in rows],marker='o',color=color)
        ax.set(xlabel='Alternating training round',ylabel=metric+' (%)',xticks=rounds);ax.grid(alpha=.25)
        fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.svg'));plt.close(fig)
    print('SIX_FORMAL_ENDPOINT_CURVES_WRITTEN',flush=True)
if __name__=='__main__':main()
