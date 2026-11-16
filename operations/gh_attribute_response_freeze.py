"""冻结独立来源；默认仅实现/CPU授权，显式授权字段才允许后续GPU执行。"""
from pathlib import Path
import argparse
import copy
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from tbps.artifacts import read_json, json_bytes, sha_file
from tbps.attribute_response.settings import Settings
from tbps.attribute_response import METHOD

def freeze(*, authorize_gpu=False, rebind=False):
    template = ROOT/"configs/framework/gh_pushpull_aug_v1.server.json"
    value = copy.deepcopy(read_json(template))
    output = Path("/home/lzf/ldx/outputs/tbps-framework-gh-attribute-response-v1-20261116")
    control = Path("/home/lzf/ldx/tmp/gh_attribute_response_20261116")
    if output.exists() or (control/"started.json").exists():
        raise ValueError("Attempt consumed; no source rebinding")
    path = ROOT/"configs/framework/gh_attribute_response.server.json"
    if path.exists() and not rebind:
        raise ValueError("explicit unconsumed source rebind required")
    if path.exists():
        archive = Path("/home/lzf/ldx/tmp/gh_attribute_response_cpu_20261116/config_revisions")
        archive.mkdir(parents=True, exist_ok=True)
        saved = archive/(sha_file(path)+".json")
        if not saved.exists():
            saved.write_bytes(path.read_bytes())
    value.update(name=METHOD, method=METHOD, attribute_response=Settings().record(),
        output_base=str(output), test_output=str(output/"test"),
        outputs={k: str(output/k) for k in ("smoke", "train")},
        controls={k: str(control/k) for k in ("smoke", "train")},
        audits={k: str(output/(k+"_cpu_audit")) for k in ("smoke", "train")},
        producers={str(p): sha_file(p) for p in (ROOT/"src/tbps").rglob("*.py")},
        tools={str(p): sha_file(p) for p in (ROOT/"operations").glob("gh_attribute_response*.py")},
        authorization={"implementation_and_CPU_verification": True, "GPU_experiment": authorize_gpu,
                       "automatic_training": authorize_gpu, "new_Test": authorize_gpu,
                       "fresh_sixty_rounds": authorize_gpu, "six_Joint_Tests_after_prefix_audit": authorize_gpu,
                       "retry_real_failure": False})
    value["native_template"]["prototype"]["schema"] = "gh-attribute-response-protocol-1"
    value["fixed_resources"]["augmentation_upstreams"] = {
        "path": str(ROOT/"configs/framework/gh_pushpull_aug_v1.upstreams.json"),
        "sha256": sha_file(ROOT/"configs/framework/gh_pushpull_aug_v1.upstreams.json")}
    value["report_runtime"]["plotter"] = {
        "path": str(ROOT/"operations/gh_attribute_response_plot.py"),
        "sha256": sha_file(ROOT/"operations/gh_attribute_response_plot.py")}
    value["resources_note"] = "GPU observes actual allocation; no free-VRAM admission floor; OOM stops without retry"
    # 旧调度器的GPU条目仅用于观察；不添加新的显存准入门槛。
    limits = value["native_template"]["prototype"].get("proposed_gpu_limits", {})
    if "minimum_free_mib" in limits:
        limits["minimum_free_mib"] = 0
    path.write_bytes(json_bytes(value))
    print(json.dumps({"path": str(path), "sha256": sha_file(path), "GPU_authorized": authorize_gpu}))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorize-gpu", action="store_true")
    parser.add_argument("--rebind-unconsumed", action="store_true")
    args = parser.parse_args()
    freeze(authorize_gpu=args.authorize_gpu, rebind=args.rebind_unconsumed)
