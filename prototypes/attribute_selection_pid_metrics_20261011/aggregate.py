"""读取同协议、同攻击集合的多victim summary；不执行模型推理。"""
import argparse
from pathlib import Path
from inputs import load, write_json
from pid_metrics import aggregate_victims, MODES

def markdown(report):
    a = report
    lines = ["# PID-mAP诊断汇总", "",
        "协议：" + a["aggregation_contract"]["protocol_kind"], "",
        "victim数：" + str(a["victim_count"]) + "；单模型时aAP=mAP，mDR=单模型下降率。", "",
        "mDR采用(aAP_clean-aAP_adv)/aAP_clean；逐victim下降率均值单列为辅助字段。", "",
        "| 方法 | Clean aAP (%) | Adv aAP (%) | mDR (%) |",
        "|---|---:|---:|---:|"]
    for m in MODES:
        row = a["modes"][m]
        dr = "undefined" if row["mDR"] is None else f'{row["mDR"]:.6f}'
        lines.append(f'| {m} | {a["aAP_clean"]:.6f} | {row["aAP_adversarial"]:.6f} | {dr} |')
    return "\n".join(lines) + "\n"

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = aggregate_victims([load(f) for f in args.inputs])
    args.output = args.output.resolve()
    if not args.output.is_relative_to(Path("/home/lzf/ldx").resolve()):
        raise ValueError("output must remain under /home/lzf/ldx")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "summary.json", result)
    (args.output / "summary.md").write_text(markdown(result), encoding="utf-8")
    print(markdown(result))
if __name__ == "__main__":
    main()
