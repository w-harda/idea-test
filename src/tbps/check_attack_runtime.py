"""严格CPU运行闭包检查；没有GPU选项，当前Stage 3B来源缺项会明确拒绝。"""
import argparse
from .smoke_attacks import preflight
from .attacks.legacy_runtime import legacy_imports

def check(config):
    result=preflight(config)
    sources=result[1]
    import torch
    from unittest.mock import patch
    with patch.object(torch,"load",side_effect=AssertionError("CPU closure check forbids weight restore")), \
         patch.object(torch.nn.Module,"_call_impl",side_effect=AssertionError("CPU closure check forbids neural forward")):
        with legacy_imports(sources,allow_import=True) as old:
            # 覆盖原预检遗漏的实际G像素读取链及H加载后状态检查链。
            old("irra_image_adapter")
            old("formal_training.validation")
    if torch.cuda.is_initialized():raise ValueError("CPU closure check initialized CUDA")
    return result[-1]

def main(argv=None):
    p=argparse.ArgumentParser(description="Strict CPU-only attack loading closure; no GPU/run option")
    p.add_argument("--config",required=True);args=p.parse_args(argv)
    try:check(args.config)
    except (ValueError,TypeError,OSError,KeyError,StopIteration) as exc:
        p.exit(1,"GH_RUNTIME_PREFLIGHT_REJECTED "+str(exc)+"\n")
    print("GH_RUNTIME_PREFLIGHT_PASS")

if __name__=="__main__":main()
