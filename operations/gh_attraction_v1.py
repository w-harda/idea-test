"""只读CPU来源预检；真实训练适配器见tbps.attraction.backend。"""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default=str(ROOT/'configs/framework/gh_attraction_v1.server.json'))
    ap.add_argument('--scope',choices=('smoke','train'),default='smoke');a=ap.parse_args()
    from tbps.attraction.sources import preflight
    f,c,b,g,report=preflight(a.config,a.scope)
    print(json.dumps({k:v for k,v in report.items() if k!='imports'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
