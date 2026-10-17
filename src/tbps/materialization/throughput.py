"""CPU Fixture存储吞吐，无真实模型/策略/编码；保留结果文件用于审计。"""
from dataclasses import asdict
from pathlib import Path
import argparse
import time
from ..artifacts import ArtifactStore,FileRef,sha_file,digest,json_bytes
from ..contracts import DatasetManifest,GalleryItem,QueryItem,PersonID
from ..attacks.provenance import AttackProvenance
from ..attacks.base import output,tensor_sha
from ..attacks.hv1 import BUDGET as H_BUDGET
from ..attacks.gv2 import BUDGET as G_BUDGET
from ..engines.replay_control import rss_bytes
from .contracts import MaterializationPlan
from .store import MaterializedStore

def new_base(root):
    return next(p for p in (Path(root),*Path(root).parents) if p.name.startswith("tbps-framework"))

def fixture(root,*,queries=1024,gallery=16,sources=128,checkpoint_bytes=8*1024**2):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);images=root/"images";images.mkdir()
    annotation=root/"annotation.json";annotation.write_bytes(b'{"fixture":true}')
    refs=[FileRef(str(annotation),sha_file(annotation))]
    for i in range(sources):
        p=root/("source-"+str(i));p.write_bytes(str(i).encode());refs.append(FileRef(str(p),sha_file(p)))
    prov=[]
    for name in ("gv2","hv1"):
        p=root/(name+".fixture");p.write_bytes(bytes(checkpoint_bytes));ref=FileRef(str(p),sha_file(p));refs.append(ref)
        sha=digest("cpu fixture")
        prov.append(AttackProvenance(name,ref,sha,sha,sha,"cuhk_pedes:attack_train:2048","cpu_fixture",sha if name=="hv1" else None))
    gs=[];qs=[]
    for i in range(gallery):
        p=images/(str(i)+".bin");p.write_bytes(b"fixture image")
        gs.append(GalleryItem("g"+str(i),p.name,sha_file(p),PersonID("fixture",str(i)),"test"))
    for i in range(queries):
        g=gs[i%gallery]
        qs.append(QueryItem("q"+str(i),"red shirt",i//gallery,g.image_id,g.pid,"test"))
    manifest=DatasetManifest("fixture","test",refs[0],tuple(gs),tuple(qs))
    plan=MaterializationPlan(manifest,*prov,tuple(refs),str(images),fixture=True)
    store=MaterializedStore(ArtifactStore(root/"artifacts",new_base=new_base(root)),plan);store.publish_plan()
    return store

def text_result(store,index):
    req=store.text_request("q"+str(index))
    action={"original":"red shirt","adversarial":"rеd shirt","edits":[{"position":1}]}
    trace={"stage":"online_generation","action":action,"optimizer_steps":0,"surrogate_forwards":0,
        "target_feedback_used":False,"target_tokenizer_used":False,"state_sha256_before":"fixture","state_sha256_after":"fixture"}
    return req,output("hv1",req,H_BUDGET,action["adversarial"],None,trace,store.plan.h.as_dict())

def validate(caption,action):
    if caption!=action["original"]:raise ValueError("fixture identity")
    return action["adversarial"]

def measure(root,*,count=24,images=8,store_factory=fixture,optimized=False):
    from unittest.mock import patch
    import tbps.artifacts as artifact_module
    import torch
    store=store_factory(root);counter={"checked_calls":0,"checked_declared_bytes":0}
    from contextlib import nullcontext
    from .bulk_session import StreamingMaterializedStore,BulkSession
    if optimized:store=StreamingMaterializedStore(store.store,store.plan)
    original=artifact_module.checked
    def checked(ref):
        counter["checked_calls"]+=1;counter["checked_declared_bytes"]+=Path(ref.path).stat().st_size
        return original(ref)
    times={};started=time.perf_counter()
    with patch.object(artifact_module,"checked",new=checked),(BulkSession(store) if optimized else nullcontext()) as session:
        tick=time.perf_counter();refs=[]
        for i in range(count):
            request,result=text_result(store,i);ref=store.publish_text(request,result,validate)
            store.load(ref,validate=validate);refs.append(ref)
        times["text_publish_readback_seconds"]=time.perf_counter()-tick
        tick=time.perf_counter()
        for i in range(images):
            request=store.image_request("g"+str(i))
            clean=torch.full((1,3,384,128),.5);adv=clean+.01;delta=adv-clean
            trace={"input_pixels_sha256":tensor_sha(clean),"adversarial_pixels_sha256":tensor_sha(adv),
                "effective_delta_sha256":tensor_sha(delta),"G_forward_samples":1,"optimizer_steps":0,
                "surrogate_forwards":0,"condition_cache_read":False}
            result=output("gv2",request,G_BUDGET,adv,delta,trace,store.plan.g.as_dict())
            ref=store.publish_image(request,clean,result);arrays,_=store.load(ref)
            assert torch.equal(arrays["adversarial"],adv)
            assert all(x.untyped_storage().nbytes()==x.numel()*4 for x in arrays.values())
        times["image_publish_readback_seconds"]=time.perf_counter()-tick
    files=[p for p in store.store.root.rglob("*") if p.is_file()]
    return {"schema":"tbps-materialization-throughput-1","fixture":True,"text_units":count,"image_units":images,
        "manifest_queries":len(store.plan.manifest.queries),"sources":len(store.plan.sources),
        "elapsed_seconds":time.perf_counter()-started,"timings":times,"peak_RSS_bytes":rss_bytes(),
        "output_bytes":sum(p.stat().st_size for p in files),"output_files":len(files),
        "sha_calls":counter,"optimized":optimized,"bulk_source_stats":session.cache.stats if optimized else None,"real_model_loads":0,"real_forwards":0,"CUDA_initialized":torch.cuda.is_initialized()}

def main(argv=None):
    p=argparse.ArgumentParser(description="CPU Fixture IO benchmark only")
    p.add_argument("--output",required=True);p.add_argument("--optimized",action="store_true");a=p.parse_args(argv)
    if Path(a.output).exists():p.error("immutable fixture output already exists")
    report=measure(a.output,optimized=a.optimized)
    ArtifactStore(a.output,new_base=new_base(a.output)).publish("benchmark",{"report.json":json_bytes(report)},
        {"kind":"CPU fixture throughput"},sources=tuple(FileRef(str(Path(__file__).with_name(n)),sha_file(Path(__file__).with_name(n))) for n in ("throughput.py","bulk_session.py")))
    print(__import__("json").dumps(report,indent=2))
if __name__=="__main__":main()
