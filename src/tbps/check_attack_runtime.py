"""严格CPU加载闭包检查；无GPU选项。原配置仍拒绝，派生配置须绑定修复证据。"""
import argparse
import json
from .smoke_attacks import preflight
from .artifacts import ArtifactError
from .runtime_closure import check_derived, cpu_only, inspect_imports, source_execution_audit

def check(config, derivation=None):
    if derivation is not None:
        return check_derived(config, derivation, preflight)
    # 保留原FAIL配置的明确拒绝行为，不自动补清单或绕过来源锁。
    with cpu_only():
        result = preflight(config)
        sources = result[1]
        with source_execution_audit(sources):
            inspect_imports(sources)
        raise ArtifactError("passing CPU closure requires explicit additive derivation manifest")

def main(argv=None):
    p=argparse.ArgumentParser(description="Strict CPU-only G/H loading closure; no GPU/run option")
    p.add_argument("--config",required=True)
    p.add_argument("--derivation",help="Stage 3B-R additive source derivation manifest")
    args=p.parse_args(argv)
    try:
        report=check(args.config,args.derivation)
    except (ValueError,TypeError,OSError,KeyError,StopIteration,AssertionError,ImportError) as exc:
        p.exit(1,"GH_RUNTIME_PREFLIGHT_REJECTED "+str(exc)+"\n")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print("GH_RUNTIME_PREFLIGHT_PASS")

if __name__=="__main__":main()
