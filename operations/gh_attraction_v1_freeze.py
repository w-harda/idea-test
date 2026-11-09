"""仅为未消费的新方法冻结来源；本轮不授权真实GPU运行。"""
from pathlib import Path
import sys,copy,argparse,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from tbps.artifacts import read_json,json_bytes,sha_file

def freeze(rebind=False):
    old=Path('/home/lzf/ldx/projects/idea-test-tbps-gh-alternating-v2-20261107/configs/framework/gh_alternating_v2.server.json')
    f=copy.deepcopy(read_json(old));out='/home/lzf/ldx/outputs/tbps-framework-gh-attraction-v1-20261109'
    if Path(out).exists():raise ValueError('Attempt consumed; no source rebinding')
    cfg=ROOT/'configs/framework/gh_attraction_v1.server.json'
    if cfg.exists():
        if not rebind:raise ValueError('explicit unconsumed rebind required')
        a=Path('/home/lzf/ldx/tmp/gh_attraction_v1_cpu_20261109/config_revisions');a.mkdir(parents=True,exist_ok=True)
        shutil.copy2(cfg,a/(sha_file(cfg)+'.json'))
    f.update(name='GH-Adversarial-Attraction-v1',method='GH-Adversarial-Attraction-v1',
        attraction={'temperature':.07,'pull_weight':1.,'push_weight':1.},output_base=out,test_output=out+'/test',
        outputs={k:out+'/'+k for k in ('smoke','train')},
        controls={k:'/home/lzf/ldx/tmp/gh_attraction_v1_'+k+'_20261109' for k in ('smoke','train')},
        audits={k:out+'/'+k+'_cpu_audit' for k in ('smoke','train')},
        producers={str(p):sha_file(p) for p in (ROOT/'src/tbps').rglob('*.py')},
        tools={str(p):sha_file(p) for p in (ROOT/'operations').glob('gh_attraction_v1*.py')},
        authorization={'implementation_and_CPU_verification':True,'GPU_experiment':False,'automatic_training':False,'new_Test':False})
    f['native_template']['prototype']['schema']='gh-attraction-v1-protocol-1'
    cfg.write_bytes(json_bytes(f));print(cfg,sha_file(cfg))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--rebind-unconsumed',action='store_true');a=ap.parse_args();freeze(a.rebind_unconsumed)
