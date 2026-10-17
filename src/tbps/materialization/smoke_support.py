"""Stage4C2固定小样本来源、CPU门禁和数值验收；无默认模型加载。"""
from dataclasses import asdict
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch
import importlib.machinery
import sys
from ..artifacts import ArtifactStore, FileRef, checked, read_json, sha_file, digest, json_bytes
from ..contracts import DatasetManifest, GalleryItem, QueryItem, PersonID
from ..attacks.provenance import LegacyAttackSources, AttackProvenance
from ..runtime_closure import cpu_only
from ..models.irra_source import IRRASource, official_model_modules
from ..models.irra_io import official_tokenizer
from ..engines.historical_sources import JointFullSources
from ..engines.historical_loader import native_action_validator, JointFullLoader, feature_only_cpu
from .contracts import MaterializationPlan, need
from .store import MaterializedStore

BASE="/home/lzf/ldx/outputs/tbps-framework-stage4c2"
OUTPUT=BASE+"/materialized_smoke_v1"
GALLERY=("cuhk:37132:0","cuhk:37147:0")
QUERY=("cuhk:37132:0","cuhk:37147:1")
LIMITS={"G":2,"H":0,"IRRA_image":4,"IRRA_text":4}
THRESHOLD=1e-6
RSS=2*1024**3
SECONDS=300

def config(path):
    c=read_json(path,"tbps-materialized-smoke-1")
    need(set(c)=={"schema","gallery","query","gallery_indices","limits","threshold","rss_bytes","seconds",
                 "seed","precision","physical_gpu","minimum_free_mib","maximum_utilization_percent",
                 "replay_config","dataset_sources","gh_config","derivation","c1_result","b2_result"},"Smoke fields")
    need(c["gallery"]==list(GALLERY) and c["query"]==list(QUERY) and c["gallery_indices"]==[0,15],"fixed Test identities")
    need(c["limits"]==LIMITS and c["threshold"]==THRESHOLD and c["rss_bytes"]==RSS and c["seconds"]==SECONDS,
         "fixed numerical/resource budgets")
    need(c["seed"]==42 and c["precision"]=="fp32/math/no_tf32" and c["physical_gpu"]==0
         and c["minimum_free_mib"]==8192 and c["maximum_utilization_percent"]==10,"fixed runtime")
    for k in ("replay_config","dataset_sources","gh_config","derivation","c1_result","b2_result"):
        checked(FileRef(**c[k]))
    return c

def original_plan(c):
    ref=FileRef(**c["c1_result"]); folder=Path(ref.path).parent
    artifact=ArtifactStore(folder.parent,new_base="/home/lzf/ldx/outputs/tbps-framework-stage4c1")
    artifact.verify(folder.name)
    d=read_json(folder/"plan.json")
    m=d["manifest"]
    def member(cls,row):
        return cls(**{**row,"pid":PersonID(**row["pid"])})
    manifest=DatasetManifest(m["dataset_id"],m["official_split"],FileRef(**m["annotation"]),
        tuple(member(GalleryItem,r) for r in m["gallery"]),
        tuple(member(QueryItem,r) for r in m["queries"]),m["relevance"])
    legacy=LegacyAttackSources(d["source_config"]["path"])
    # 新树的公共代码路径参与新plan身份；保留旧plan作为只读父证据，不改旧来源。
    refs={r["path"]:FileRef(**r) for r in d["sources"]}
    for r in (*legacy.runtime_refs,FileRef(**c["c1_result"]),FileRef(**c["b2_result"]),
              *(FileRef(p,h) for p,h in legacy.framework_sources.items())):
        refs[r.path]=r
    return MaterializationPlan(manifest,legacy.provenance("gv2"),legacy.provenance("hv1"),
        tuple(refs.values()),d["image_root"],source_config=legacy.config_ref),legacy

def materialized(c,output):
    plan,legacy=original_plan(c)
    return MaterializedStore(ArtifactStore(Path(output)/"artifacts",new_base=BASE),plan),legacy

def official_closure(source):
    """实际官方build与Tokenizer导入，不执行构造/反序列化；逐执行文件核验。"""
    pins={r.path:r for _,r in source.code}
    observed={}
    original=importlib.machinery.SourceFileLoader.get_code
    before_path=list(sys.path);before=dict(sys.modules)
    def audited(loader,name):
        path=str(Path(loader.path).resolve())
        if Path(path).is_relative_to(Path(source.root)):
            need(path in pins,"unpinned official runtime import: "+path)
            observed[path]=checked(pins[path])
        return original(loader,name)
    try:
        with cpu_only(),patch.object(importlib.machinery.SourceFileLoader,"get_code",new=audited):
            with official_model_modules(source) as (build,clip):
                need(callable(build.build_model) and callable(clip.build_CLIP_from_openai_pretrained),"official factory symbols")
            official_tokenizer(source)
    finally:
        for name,module in tuple(sys.modules.items()):
            p=getattr(module,"__file__",None)
            if name not in before and p and Path(p).resolve().is_relative_to(Path(source.root)):
                sys.modules.pop(name,None)
    need(sys.path==before_path and all(sys.modules.get(k)is v for k,v in before.items()),"official namespace restoration")
    need(observed,"actual official closure empty")
    return {"executed_sources":observed,"namespace_restored":True,"model_loads":0,"CUDA_initialized":False}

def preflight(path):
    from ..check_attack_runtime import check
    c=config(path)
    gh=check(c["gh_config"]["path"],c["derivation"]["path"])
    need(gh["status"]=="PASS" and gh["runtime_source_count"]==75,"75-source GH CPU closure")
    with cpu_only():
        plan,legacy=original_plan(c)
        sources=JointFullSources(c["replay_config"]["path"])
        need(plan.manifest==sources.manifest,"C1/B2 dataset order mismatch")
        need([plan.manifest.gallery[i].image_id for i in c["gallery_indices"]]==list(GALLERY),"fixed Gallery indices")
        for group in (GALLERY,QUERY):
            need(len(set(group))==2,"unique fixed samples")
        qidx=[next(i for i,q in enumerate(plan.manifest.queries) if q.query_id==key) for key in QUERY]
        source=IRRASource.from_dict(read_json(c["dataset_sources"]["path"])["irra"])
        source.verify();closure=official_closure(source)
        b2=Path(c["b2_result"]["path"]).parent
        ArtifactStore(b2.parent,new_base="/home/lzf/ldx/outputs/tbps-framework-stage4b2").verify(b2.name)
        b2_report=read_json(b2/"report.json")
        need(b2_report["status"]=="PASS" and b2_report["metrics_replay_status"]=="METRICS_REPLAY_VERIFIED","B2 verified receipt required")
        with native_action_validator(legacy.config_ref) as validate:
            from .legacy_text import HistoricalTextSource
            selected={}
            for request,result in HistoricalTextSource(sources,plan.h,validate).iter_outputs():
                if request.binding.sample_id in QUERY:
                    selected[request.binding.sample_id]=result.trace()["action"]
                if len(selected)==2:break
        need(selected[QUERY[0]]["edits"] and not selected[QUERY[1]]["edits"]
             and selected[QUERY[1]]["original"]==selected[QUERY[1]]["adversarial"],"fixed edit/NOOP")
    history={str(sources.root/p):checked(FileRef(str(sources.root/p),h)) for p,h in sources.declared.items()}
    return {"status":"MATERIALIZED_SMOKE_CPU_PREFLIGHT_PASS","GH":gh,"IRRA":closure,"historical_feature_snapshot":history,
        "plan_id":plan.id,"manifest_sha256":plan.manifest_sha256,"gallery":list(GALLERY),"query":list(QUERY),
        "gallery_indices":[0,15],"query_indices":qidx,"actions":selected,"limits":LIMITS,
        "resources":{"rss_bytes":RSS,"seconds":SECONDS},"model_loads":0,"neural_forwards":0}

class Ledger:
    def __init__(self,limits=None):
        self.limits=LIMITS if limits is None else limits
        self.started=dict.fromkeys(self.limits,0);self.completed=dict(self.started);self.calls=[]
    def wrap(self,kind,fn):
        def run(x,*args,**kw):
            import torch
            n=len(x)
            need(kind in self.limits and 0<n<=2 and self.started[kind]+n<=self.limits[kind],"sample forward budget")
            need(torch.is_inference_mode_enabled() and not torch.is_grad_enabled(),"inference only")
            self.started[kind]+=n;call={"kind":kind,"samples":n,"completed":False};self.calls.append(call)
            value=fn(x,*args,**kw);self.completed[kind]+=n;call["completed"]=True
            return value
        return run
    def report(self):return {"started":dict(self.started),"completed":dict(self.completed),"calls":list(self.calls)}

def compare(a,b,*,exact=False):
    import torch
    need(isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor) and a.shape==b.shape
         and a.dtype==b.dtype==torch.float32 and torch.isfinite(a).all() and torch.isfinite(b).all(),"FP32 finite shape")
    a,b=a.detach().cpu(),b.detach().cpu()
    error=float((a-b).abs().max())
    result={"shape":list(a.shape),"dtype":str(a.dtype),"max_abs_error":error,"bitwise_exact":torch.equal(a,b)}
    need(result["bitwise_exact"] if exact else error<=THRESHOLD,"numerical equivalence failed: "+str(result))
    return result

def selected_features(c):
    import torch
    sources=JointFullSources(c["replay_config"]["path"])
    indices={"clean_gallery":[0,15],"image_gallery":[0,15],
        "clean_text":[next(i for i,q in enumerate(sources.manifest.queries) if q.query_id==key) for key in QUERY]}
    indices["adv_text"]=indices["clean_text"]
    allowed={str(sources.root/f"features/{family}/{i//(64 if family.endswith('gallery') else sources.protocol['text_batch']):06d}/features.pt")
             for family,rows in indices.items() for i in rows}
    values={};refs={}
    with native_action_validator(FileRef(**sources.config["attack_sources"])) as validate,feature_only_cpu(allowed):
        loader=JointFullLoader(sources,validate)
        for family,wanted in indices.items():
            selected={}
            for index,rows in loader.parts(family):
                size=64 if family.endswith("gallery") else sources.protocol["text_batch"]
                picks=[i for i in wanted if i//size==index]
                if not picks:continue
                x,_,batch_refs=loader.load_batch(family,index,rows)
                for ref in batch_refs:refs[ref.path]=ref
                for i in picks:selected[i]=x[i-index*size].detach().clone()
                del x
            values[family]=torch.stack([selected[i] for i in wanted]).detach().clone()
    return values,tuple(refs.values())
