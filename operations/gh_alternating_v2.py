"""v2独立可审查入口。"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from tbps.alternating.fresh_run import main
if __name__=='__main__':main()
