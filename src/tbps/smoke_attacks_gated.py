"""Stage 3B-2一次受控入口：严格CPU闭包成功后才可调用原Smoke，不改变比较逻辑。"""
from pathlib import Path
import argparse
import json
from . import smoke_attacks
from .artifacts import ArtifactError, FileRef, checked, sha_file
from .runtime_closure import check_derived

OUTPUT = str(Path(smoke_attacks.BASE) / "equivalence_v2")

def prepare(config, derivation):
    captured = []
    def capture(path):
        value = smoke_attacks.preflight(path)
        captured.append(value)
        return value
    report = check_derived(config, derivation, capture)
    if report.get("status") != "PASS" or report.get("runtime_source_count") != 75:
        raise ArtifactError("strict CPU closure must PASS with 75 fixed sources")
    if len(captured) != 1:
        raise ArtifactError("strict preflight must execute once")
    c, sources, images, texts, refs, before, pre = captured[0]
    if before != {r.path:checked(r) for r in refs}:
        raise ArtifactError("sources changed after strict preflight")
    extra = [FileRef(p, v["sha256"]) for p, v in report["control_sources"].items()]
    extra.append(FileRef(str(Path(__file__).resolve()), sha_file(__file__)))
    refs = tuple({r.path:r for r in (*refs,*extra)}.values())
    before = {r.path:checked(r) for r in refs}
    pre = {**pre, "strict_runtime_preflight": report,
           "entry_gate": "Stage3B-2 strict CPU closure required before original run_once"}
    return (c,sources,images,texts,refs,before,pre)

def launch(config, derivation, *, allow_gpu_smoke=False, output=OUTPUT):
    if str(Path(output).resolve()) != OUTPUT:
        raise ArtifactError("only new immutable equivalence_v2 output is authorized")
    if allow_gpu_smoke and Path(output).exists():
        raise ArtifactError("equivalence_v2 already used; no retry or recovery")
    prepared = prepare(config, derivation)
    print("GH_RUNTIME_PREFLIGHT_PASS",flush=True)
    if not allow_gpu_smoke:
        print(json.dumps(prepared[-1],ensure_ascii=False,indent=2))
        return 0
    # 不另写/复制数值对照、模型恢复、Ledger或Joint路径；复用原一次性Smoke。
    return smoke_attacks.run_once(*prepared,output=output)

def main(argv=None):
    p=argparse.ArgumentParser(description="Default strict CPU; one explicitly authorized Stage3B-2 G/H attempt")
    p.add_argument("--config",required=True)
    p.add_argument("--derivation",required=True)
    p.add_argument("--output",default=OUTPUT)
    p.add_argument("--allow-gpu-smoke",action="store_true")
    a=p.parse_args(argv)
    try:
        return launch(a.config,a.derivation,allow_gpu_smoke=a.allow_gpu_smoke,output=a.output)
    except (ValueError,TypeError,OSError,KeyError,StopIteration,AssertionError,ImportError) as exc:
        p.exit(1,"GH_GATED_SMOKE_REJECTED "+str(exc)+"\n")

if __name__=="__main__":raise SystemExit(main())
