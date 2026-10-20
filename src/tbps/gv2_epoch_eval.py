"""六节点入口：只读固定Stage4D1攻击/物化命名空间，目标评分使用现有原生Adapter。"""
def main():
    from pathlib import Path
    import tbps.attacks,tbps.materialization
    frozen=Path("/home/lzf/ldx/projects/idea-test-tbps-framework-stage4d1-20261017/src/tbps")
    tbps.attacks.__path__[:]=[str(frozen/"attacks")]
    tbps.materialization.__path__[:]=[str(frozen/"materialization")]
    from tbps.ablations.gv2_sweep import main as run
    return run()
if __name__=="__main__":main()
