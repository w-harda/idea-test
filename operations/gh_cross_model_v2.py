"""跨模型入口；默认只运行CPU预检。"""
from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'configs/framework/gh_cross_model_v2.server.json'))
    p.add_argument('--action',choices=('preflight','input-audit','run','audit','report','startup-check','verify-input'),default='preflight')
    p.add_argument('--method',choices=('cooperative','attraction','augmentation'));p.add_argument('--model',choices=('rasa','aptm'))
    p.add_argument('--scope',choices=('smoke','full'),default='smoke');p.add_argument('--allow-gpu',action='store_true');a=p.parse_args()
    if a.action=='startup-check':
        from tbps.gh_transfer.startup import prepare
        from tbps.artifacts import json_bytes
        print(json_bytes(prepare(a.config,a.method,a.model,a.scope)[-1]).decode());return
    if a.action=='verify-input':
        from tbps.gh_transfer.common import config,collection
        from tbps.gh_transfer.audit import protection
        c=config(a.config);collection(c,a.method);print('VERIFIED_INPUT_COLLECTION',a.method,protection(c,a.method,rehash=True));return
    if a.action=='preflight':
        from tbps.gh_transfer.evaluation import preflight
        preflight(a.config,a.model);return
    if a.action=='input-audit':
        from tbps.gh_transfer.inputs import audit_inputs
        audit_inputs(a.config,a.method);return
    if a.action=='run':
        from tbps.gh_transfer.evaluation import run
        run(a.config,a.method,a.model,a.scope,allow=a.allow_gpu);return
    if a.action=='audit':
        from tbps.gh_transfer.audit import audit
        audit(a.config,a.method,a.model,a.scope);return
    from tbps.gh_transfer.report import publish
    print(publish(a.config))
if __name__=='__main__':main()
