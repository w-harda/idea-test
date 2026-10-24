"""独立进程入口，避免冻结legacy导入会话与src下__main__身份冲突。"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
if __name__=="__main__":
    if "--evaluate" in sys.argv:
        sys.argv.remove("--evaluate")
        from tbps.alternating.formal_evaluation import main
    else:
        from tbps.alternating.formal_cli import main
    main()
