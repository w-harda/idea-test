"""复用冻结的派生报告生成器，仅将新文档输出定向到修复分支。"""
from pathlib import Path
import importlib.util,json
ROOT=Path(__file__).resolve().parents[3]
OLD=Path("/home/lzf/ldx/projects/idea-test-tbps-hv1-g60-adaptation-20261021")
def main():
    spec=importlib.util.spec_from_file_location("original_h60_results",OLD/"src/tbps/ablations/hv1_g60_results.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # 原函数以ROOT决定文档目录和只读配置位置；新树配置原样保留原113 producer。
    module.ROOT=ROOT
    module.main()
    out=ROOT/"docs/framework/results/hv1_g60_adaptation_epoch60"
    p=out/"SOURCE_EVIDENCE.json";data=json.loads(p.read_text())
    base=Path("/home/lzf/ldx/outputs/tbps-framework-hv1-g60-adaptation-epoch60")
    data["versioned_CPU_audit"]=json.loads((base/"audit_repair_v1/training/report.json").read_text())
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")
if __name__=="__main__":main()
