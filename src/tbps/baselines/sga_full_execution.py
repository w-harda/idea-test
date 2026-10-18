"""Full执行管理薄入口：只增加原生编码标量进度，不改变攻击/评分。"""
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch
import argparse
from tbps.baselines import sga_irra as runner
from tbps.baselines import comparison
from tbps.artifacts import ArtifactStore, FileRef, checked, read_json, sha_file
from tbps.models.sga_irra_source import require

FROZEN=Path("/home/lzf/ldx/projects/idea-test-tbps-sga-irra-source-20261019")
CONFIG=FROZEN/"configs/framework/sga_irra_source.full.server.json"
CAPACITY=Path("/home/lzf/ldx/outputs/tbps-framework-sga-irra-capacity/batch8_v1/source")

def launch_preflight():
    c=runner.configuration(CONFIG)
    root=Path(__file__).resolve().parents[2]
    for path,h in c["producers"].items():
        actual=root/Path(path).relative_to(FROZEN/"src")
        checked(FileRef(str(actual),h))
    require(sha_file(runner.__file__)==sha_file(FROZEN/"src/tbps/baselines/sga_irra.py"),"unchanged frozen full runner")
    store=ArtifactStore(CAPACITY,new_base=CAPACITY.parents[1])
    for key in ("protocol","execution","completed","batch-000000","claim-batch-000000"):store.verify(key)
    report=read_json(CAPACITY/"execution/report.json")
    collection=read_json(CAPACITY/"completed/collection.json")
    require(report["status"]=="PASS" and collection["Gallery"]==8 and collection["Query"]==16,"actual Batch8 capacity PASS")
    require(report["model_before"]==report["model_after"] and report["MLM_before"]==report["MLM_after"],"capacity models frozen")
    runner.preflight(CONFIG,"full")
    print("SGA_OFFICIAL_FULL_LAUNCH_PREFLIGHT_PASS",flush=True)
    return c

@contextmanager
def encoding_progress(load_model,model,*args,**kwargs):
    with load_model(*args,model=model,**kwargs) as (adapter,backend,checkpoint,counts):
        target=backend if model=="irra" else adapter
        done={"image":0,"text":0};totals={"image":6148,"text":12312}
        def observe(fn,kind):
            def call(x,*a,**kw):
                result=fn(x,*a,**kw)
                before=done[kind];done[kind]+=len(x)
                if before//128!=done[kind]//128 or done[kind]==totals[kind]:
                    print("SGA_TARGET_ENCODING_PROGRESS",model,kind,done[kind],"/",totals[kind],flush=True)
                return result
            return call
        image="encode_image" if model=="irra" else "encode_images"
        text="encode_text" if model=="irra" else "encode_texts"
        try:
            with patch.object(target,image,observe(getattr(target,image),"image")),patch.object(target,text,observe(getattr(target,text),"text")):
                yield adapter,backend,checkpoint,counts
        finally:print("SGA_TARGET_ENCODING_FINAL",model,done,flush=True)

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--action",choices=("preflight","evaluate"),default="preflight")
    p.add_argument("--model",choices=("irra","rasa","aptm"));p.add_argument("--allow-full",action="store_true")
    a=p.parse_args(argv)
    if a.action=="preflight":launch_preflight();return 0
    require(a.allow_full and a.model is not None,"explicit Full target")
    c=runner.configuration(CONFIG);c["path"]=str(CONFIG)
    original=comparison.load_model
    @contextmanager
    def observed(config,model,scope):
        with encoding_progress(original,model,config,scope=scope) as values:yield values
    with patch.object(comparison,"load_model",observed):
        return runner.encode_target(c,"full",a.model)
if __name__=="__main__":raise SystemExit(main())
