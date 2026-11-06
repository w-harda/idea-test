"""独立进程入口；不绕过legacy导入保护。"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tbps.alternating.configured_cli import main
if __name__=='__main__':main()
