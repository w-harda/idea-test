"""原生RaSa/APTM增强Push-pull实验入口。"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from tbps.native_aug.cli import main
if __name__=="__main__":main()
