#!/usr/bin/env python3
"""协同训练/审计/原生评测统一入口。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tbps.cooperative.run import main
if __name__=='__main__':main()
