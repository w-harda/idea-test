"""对抗图文相互吸引版本：来源预检、授权训练、独立审计及原生Joint评测。"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from tbps.augmented_attraction.run import main
if __name__=='__main__':main()
