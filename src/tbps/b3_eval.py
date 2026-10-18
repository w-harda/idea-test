"""独立B3入口：保持旧G/H物化命名空间及原Plan不变。"""
def main():
    from pathlib import Path
    import tbps.attacks
    import tbps.materialization
    frozen=Path("/home/lzf/ldx/projects/idea-test-tbps-framework-stage4d1-20261017/src/tbps")
    tbps.attacks.__path__[:]=[str(frozen/"attacks")]
    tbps.materialization.__path__[:]=[str(frozen/"materialization")]
    from tbps.baselines.comparison import main as run
    return run()
if __name__=="__main__":raise SystemExit(main())
