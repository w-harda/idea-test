"""CPU-only 来源核验，发布互相隔离的 Source / Target 输入。"""
from dataclasses import asdict
from pathlib import Path
import json
import os
import subprocess
import sys
import time

from tbps.artifacts import digest, json_bytes, sha_file
from tbps.baselines.cpgc_sep.dataset import require
from tbps.baselines.cpgc_sep.official import checked_tree, sep_functions
from tbps.baselines.cpgc_sep.runtime import ref
from tbps.baselines.cpgc_sep.sep_bridge import prepare_caption
from tbps.baselines.cpgc_sep.victims import VictimSource
from tbps.models.aptm import APTMSource
from tbps.models.irra_io import official_tokenizer
from . import isolation
from .attack import resources, trace_markers
from .common import (CONFIG, PROTOCOL, ROOT, TARGETS, atomic, checked_runtime, code_commit, image_path,
                     pinned, preregister, progress, read, settings, source_config, store_at)
from .phases import cpu, guard, input_bundle, token_receipt

OLD_CONTROL = Path("/home/lzf/ldx/tmp/cpgc_sep_no_gpu_floor_v4_20261101")
OLD_SOURCE = Path("/home/lzf/ldx/outputs/tbps-framework-cpgc-sep-no-gpu-floor-v4/full/sep_source/cuhk/irra")
SUITES = ("test_cpgc_sep.py", "test_cpgc_sep_throughput.py", "test_sep_cpu_optimization_review.py",
          "test_sep_pid_integration.py", "test_sep_source_only_smoke.py")


def protected():
    folders = [OLD_SOURCE/f"text-{i:06d}" for i in range(44)]
    files = [OLD_CONTROL/name for name in ("FAIL.json", "status.json", "0006_sep_full_cuhk_irra_source.log",
                                          "0006_sep_full_cuhk_irra_source.log.exit")]
    files += [OLD_SOURCE/"text-000044.Claim.json", OLD_SOURCE/"protocol/manifest.json"]
    require(not (OLD_SOURCE/"text-000044").exists() and all(p.is_dir() for p in folders), "old 44+Claim closure")
    files += [p for d in folders for p in d.iterdir() if p.is_file()]
    return [{"path":str(p), "sha256":sha_file(p), "mtime_ns":p.stat().st_mtime_ns} for p in sorted(files)]


def cpu_tests(control):
    import xml.etree.ElementTree as ET
    log = control/"cpu-tests.log"
    xml = control/"cpu-tests.xml"
    cmd = [sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider", "-q", f"--junitxml={xml}",
           *(str(ROOT/"tests/framework"/name) for name in SUITES)]
    with log.open("xb") as output:
        subprocess.run(cmd, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=True)
    cases = ET.parse(xml).findall(".//testcase")
    counts = {name: 0 for name in SUITES}
    for case in cases:
        require(not list(case), "unexpected failure or skipped CPU case")
        stem = case.attrib["classname"].split(".")[-1] + ".py"
        counts[stem] += 1
    require(sum(counts[name] for name in SUITES[:-1]) == 176 and counts[SUITES[-1]] > 0, "full CPU regression count")
    return {"per_suite": counts, "total": len(cases), "log_sha256":sha_file(log), "xml_sha256":sha_file(xml)}


def probe(out, control, denied):
    cpu()
    bundle, path = input_bundle(out)
    _, abi = guard(bundle, path, out, control)
    isolation.must_deny(denied)
    config = bundle["config"]
    VictimSource(config, "irra", "cuhk").verify()
    tokenizer = official_tokenizer(VictimSource(config, "irra", "cuhk"))
    tokenizer.inspect(bundle["captions"][0]["caption"])
    from torchvision.transforms import ToTensor
    from PIL import Image
    train = bundle["train_gallery"][0]
    with Image.open(image_path(bundle["train_root"], train["image_rel_path"])) as img:
        require(ToTensor()(img.convert("RGB")).shape[0] == 3, "allowed Source train image read")
    print(json.dumps({"landlock_abi":abi, "denied":len(denied)+1, "source_tokenizer_and_image":True}))


def run():
    import nltk
    import torch
    torch.set_num_threads(2)
    cpu()
    c, base, aptm = settings()
    require(c["scope"] == "smoke" and c["protocol"] == PROTOCOL, "fixed Smoke scope")
    output, control = Path(c["output"]), Path(c["control"])
    require(not output.exists() and not control.exists(), "new preflight output and control required")
    output.mkdir()
    control.mkdir()
    start = time.monotonic()
    historical = protected()
    atomic(control/"historical-before.json", historical)
    progress(control, "PREFLIGHT_TESTS")
    test = cpu_tests(control)
    progress(control, "PREFLIGHT_ASSET_SHA", tests=test["total"])
    cfg = source_config(base)
    checked_runtime(cfg)
    official = cfg["official"]["sep"]
    checked_tree(official["root"], official["commit"], official["files"])
    for asset in cfg["resources"].values():
        pinned({k:asset[k] for k in ("path","sha256")})
    nltk_files = {str(p):sha_file(p) for p in sorted(Path(cfg["nltk_root"]).rglob("*")) if p.is_file()}
    require(bool(nltk_files), "real SEP POS assets")
    source = VictimSource(cfg, "irra", "cuhk")
    source.verify()
    tokenizer = official_tokenizer(source)
    probe_paths = [x["annotation"]["path"] for x in base["datasets"].values()]
    target_refs = {}
    for key, (ds, name) in TARGETS.items():
        model = APTMSource(aptm) if name == "aptm" else VictimSource(base, name, ds)
        model.verify()
        target_refs[key] = asdict(model.checkpoint)
        require(model.checkpoint.sha256 != source.checkpoint.sha256, "distinct Target weight")
        probe_paths.append(model.checkpoint.path)
    progress(control, "PREFLIGHT_OFFICIAL_MANIFESTS")
    gallery, captions, eval_sets = preregister(c, base)
    train_sha = digest(gallery)
    require(not c.get("source_train_manifest_sha256") or c["source_train_manifest_sha256"] == train_sha,
            "preregistered Source Train manifest SHA mismatch")
    nltk.data.path[:] = [cfg["nltk_root"]]
    attack = sep_functions(cfg)["attack"]
    pos = attack.__globals__["criteria"].get_pos
    markers = trace_markers(cfg)
    words, _, _ = resources(cfg)
    for item in captions:
        text = prepare_caption(cfg, item["caption"])
        eligible = [w for w, p in zip(text.split(), pos(text.split())) if len(w) > 2 and p in ("ADJ","ADV","VERB","NOUN")]
        require(len(eligible) > 1 and any(w in words for w in eligible) and not tokenizer.inspect(text).truncated, "Smoke POS/77-token preregistration")
        token_receipt(tokenizer, [text])
    for ds, manifest in eval_sets.items():
        probe_paths.append(str(image_path(base["datasets"][ds]["root"], manifest["gallery"][0]["image_rel_path"])))
    store = store_at(output, "preflight")
    bundle = {"config":cfg, "train_root":base["datasets"]["cuhk"]["root"], "train_gallery":gallery,
              "captions":captions, "nltk_files":nltk_files}
    store.publish("source-input", {"bundle.json":json_bytes(bundle)}, {"kind":"Source-only capability"},
                  sources=(ref(CONFIG),))
    store.publish("evaluation-input", {"manifests.json":json_bytes(eval_sets)}, {"kind":"private evaluation identity"},
                  sources=(ref(CONFIG),))
    result = subprocess.check_output([sys.executable, "-B", "-m", __package__+".preflight", "probe",
                                     str(output), str(control), *probe_paths], cwd=ROOT, text=True)
    closure = {"status":"PASS", "scope":"smoke", "commit":code_commit(), "config_sha256":sha_file(CONFIG),
               "source_checkpoint":asdict(source.checkpoint), "target_checkpoints":target_refs,
               "source_train_manifest_sha256":train_sha, "source_train_images":len(gallery), "source_train_pids":11003,
               "smoke_gallery_per_dataset":128, "smoke_captions_per_dataset":2, "official_attack_markers":markers,
               "tests":test, "isolation":json.loads(result), "historical_files":len(historical),
               "seconds":time.monotonic()-start, "CUDA_initialized":torch.cuda.is_initialized()}
    require(protected() == historical, "old stopped artifacts changed")
    atomic(control/"preflight.json", closure)
    progress(control, "PREFLIGHT_PASS", tests=test["total"], train_sha=train_sha)
    print(json.dumps(closure, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "probe":
        probe(*sys.argv[2:4], sys.argv[4:])
    else:
        run()
