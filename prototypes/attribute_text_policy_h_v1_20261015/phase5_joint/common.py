"""Phase5专用路径、冻结源码与G-v2已验证评测模块的只读复用。"""
from pathlib import Path
import importlib
import sys
from phase2_sources import ROOT as HROOT, G2, digest, sha, checked, read_json
from formal_training.common import immutable_json, run_lock, Stop, cpu_only
HERE=Path(__file__).resolve().parent
OUTPUT=HROOT/"outputs/phase5_joint/full"
FORMAL=HROOT/"outputs/formal_v1"
OLD_FULL=G2/"outputs/evaluation_phase29a/full"
NAMES=("common.py","protocol.py","storage.py","backend.py","evaluate.py","config.json")
GROUPS=("Clean","Image-only","Text-only","Joint")
def legacy(name):
    directory=G2/"evaluation_phase29a"
    sys.path.insert(0,str(directory))
    module=importlib.import_module(name)
    if Path(module.__file__).resolve()!=directory/(name+".py"):
        raise ValueError("legacy evaluation module path collision")
    return module
def safe_output(path):
    path=Path(path).resolve()
    allowed=(HROOT/"outputs").resolve()/"phase5_joint/full"
    if OUTPUT.resolve()!=allowed or path!=allowed:
        raise ValueError("Phase5 output fixed and isolated from active Phase4")
    return path
def source_pins():
    config=read_json(HERE/"config.json")
    pins=dict(config["frozen_sources"])
    for name in NAMES:pins[str(HERE/name)]=sha(HERE/name)
    for p,h in pins.items():checked(p,h)
    return pins
def verify_protocol(protocol):
    if protocol["schema"]!="hv1-joint-full-1":raise ValueError("unknown Phase5 protocol")
    for p,h in protocol["source_sha256"].items():checked(p,h)
    for p,h in protocol["evidence_sha256"].items():checked(p,h)
    if protocol["model_sha256"]["H"]!=protocol["H_selection"]["checkpoint_sha256"]:
        raise ValueError("selected H SHA differs from model contract")
    if protocol["model_sha256"]["G"]!=read_json(HERE/"config.json")["G_sha256"]:
        raise ValueError("wrong frozen image generator")
    if protocol["model_sha256"]["IRRA"]!=read_json(HERE/"config.json")["IRRA_sha256"]:
        raise ValueError("wrong frozen IRRA")
