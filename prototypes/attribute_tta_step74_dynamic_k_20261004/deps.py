"""只读复用历史prototype；所有新写入限本prototype。"""
import os,sys,importlib.util
from pathlib import Path
sys.dont_write_bytecode=True
for key,sub in [("TORCH_HOME","torch"),("HF_HOME","huggingface"),("XDG_CACHE_HOME","xdg"),("CUDA_CACHE_PATH","cuda")]:
    os.environ.setdefault(key,"/home/lzf/ldx/cache/"+sub)
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent
P51=REPO/"prototypes/attribute_importance_step51_20261003"
P732=REPO/"prototypes/attribute_tta_step732_20261004"
V55=REPO/"prototypes/attribute_unit_v55_interface_20261003"
V4=REPO/"prototypes/attribute_unit_v4_learned_reader_20261003"
for path in [P732,P51,V55,REPO/"src"]:
    if str(path) not in sys.path:sys.path.append(str(path))
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj)
    return obj
def scoring_runner():
    return module("step74_frozen_step51_runner",P51/"run.py")
def metrics_runner():
    return module("step74_frozen_step73_runner",REPO/"prototypes/attribute_tta_step73_20261004/run.py")
def rows(path):
    import json
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():yield json.loads(line)
def own_hashes():
    from selection_audit import digest
    return {str(p):digest(p) for p in ROOT.glob("*.py")}
def frozen_hashes():
    from selection_audit import load,verify
    hashes=load(ROOT/"provenance.lock.json")["frozen_sha256"]
    verify(hashes)
    return hashes
def assert_formal_clean():
    import subprocess
    for args in [["git","diff","--stat"],["git","diff","--cached","--stat"]]:
        if subprocess.check_output(args,cwd=REPO,text=True):
            raise ValueError("formal tracked/staged diff changed")
