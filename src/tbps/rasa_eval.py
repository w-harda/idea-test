"""生产入口：只读冻结攻击/物化命名空间，使用当前独立模型与评测命名空间。"""
from pathlib import Path
import tbps.attacks
import tbps.materialization

FROZEN=Path("/home/lzf/ldx/projects/idea-test-tbps-framework-stage4d1-20261017/src/tbps")
def main():
    # 在首次导入子模块前固定原生产者路径；不改旧文件、不重算/重签MaterializationPlan。
    for package,name in ((tbps.attacks,"attacks"),(tbps.materialization,"materialization")):
        package.__path__[:]=[str(FROZEN/name)]
    from tbps.engines.materialized_rasa import main as run
    return run()
if __name__=="__main__":raise SystemExit(main())
