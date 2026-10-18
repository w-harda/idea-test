"""先导入历史像素工具的CPU依赖，再建立严格启动快照；不初始化CUDA。"""
def main():
    # torchvision首次导入会增加sys.path；必须发生在CPUStartupGate之前。
    from torchvision import transforms
    from tbps.b3_eval import main as run
    return run()
if __name__=="__main__":raise SystemExit(main())
