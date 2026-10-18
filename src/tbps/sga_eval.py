"""SGA入口；冻结旧输入命名空间，SGA算法在独立baselines模块。"""
def main():
    from pathlib import Path
    from torchvision import transforms
    import tbps.attacks,tbps.materialization
    frozen=Path("/home/lzf/ldx/projects/idea-test-tbps-framework-stage4d1-20261017/src/tbps")
    tbps.attacks.__path__[:]=[str(frozen/"attacks")]
    tbps.materialization.__path__[:]=[str(frozen/"materialization")]
    from tbps.baselines.sga_whitebox import main as run
    return run()
if __name__=="__main__":raise SystemExit(main())
