"""只读Joint Full Loader：CLI仅验证，不计算真实Score或指标。"""
from contextlib import contextmanager,ExitStack
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import time
from ..artifacts import ArtifactError,FileRef,checked,digest,read_json,json_bytes,sha_file
from ..models.base import RepresentationBatch
from ..models.irra_cached import normalized
from .provenance import payload_digest
from .historical_contracts import (FAMILIES,HistoricalFeatureBinding,HistoricalRepresentation)
from .historical_sources import JointFullSources,safe_file,need

@contextmanager
def feature_only_cpu(allowed):
    """只允许weights_only读取白名单feature文件；构造/前向/导数/CUDA均硬性拒绝。"""
    import torch
    from unittest.mock import patch
    need(not torch.cuda.is_initialized(),"CPU feature verification must not initialize CUDA")
    original=torch.load;attempts={}
    def stop(name):
        def fail(*args,**kwargs):
            attempts[name]=attempts.get(name,0)+1
            raise ArtifactError("feature verification forbids "+name)
        return fail
    def load(path,*args,**kwargs):
        p=str(Path(path).resolve())
        need(p in allowed and not args and kwargs=={"map_location":"cpu","weights_only":True},
             "only authorized feature Tensor loading is allowed")
        return original(path,**kwargs)
    with ExitStack() as stack:
        stack.enter_context(patch.object(torch,"load",new=load))
        for owner,name,label in ((torch.nn.Module,"__init__","neural construction"),(torch.nn.Module,"_call_impl","neural forward"),
          (torch.cuda,"init","CUDA init"),(torch.cuda,"_lazy_init","CUDA lazy init"),
          (torch.autograd,"grad","autograd.grad"),(torch.autograd,"backward","backward"),
          (torch.jit,"load","jit model load")):
            stack.enter_context(patch.object(owner,name,new=stop(label)))
        if hasattr(torch._C,"_cuda_init"):stack.enter_context(patch.object(torch._C,"_cuda_init",new=stop("native CUDA init")))
        yield attempts
        need(not attempts and not torch.cuda.is_initialized(),"forbidden operation attempted")

@contextmanager
def native_action_validator(attack_sources):
    """复用冻结Caption-only规则与Tokenizer；不创建H/IRRA/G，不调用attack。"""
    from ..attacks.provenance import LegacyAttackSources
    from ..attacks.legacy_runtime import legacy_imports
    from ..runtime_closure import source_execution_audit,cpu_only
    checked(attack_sources);sources=LegacyAttackSources(attack_sources.path)
    with source_execution_audit(sources),legacy_imports(sources,allow_import=True) as old:
        with cpu_only():
            module=old("phase5_joint.backend");frontend=old("text_frontend").CaptionFrontend()
            probe=old("tokenizer_probe").official_probe()
            qualify=old("formal_training.data").qualify_caption
        def validate(caption,record):
            module.validate_action(caption,record,probe,frontend)
            q=qualify(caption,frontend,probe)
            for key in ("eligible","reasons","attributes"):
                need(json_bytes(record[key])==json_bytes(q[key]),"historical H qualification/Canonical mismatch: "+key)
            need(record["H_forward"]==int(record["eligible"]),"H forward accounting")
            if record["eligible"]:
                import math
                need(len(record["predicted_importance"])==len(record["attributes"]) and
                     all(math.isfinite(v) and v>=0 for v in record["predicted_importance"]),"H importance shape/finite")
            else:need(not record["predicted_importance"] and record["budget_k"]==0,"ineligible NOOP")
            for d in record["decisions"]:need(d["index"] in d["legal_indices"],"illegal saved action mask")
            return record["adversarial"]
        yield validate
    sources.verify()

class JointFullLoader:
    def __init__(self,sources,action_validator):
        need(type(sources)is JointFullSources and callable(action_validator),"verified sources and action validation")
        self.sources=sources;self.validate_action=action_validator
        self.batch_counts={k:0 for k in FAMILIES}
    def parts(self,family):
        need(family in FAMILIES,"unknown feature family")
        p=self.sources.protocol;rows=p["gallery"] if family.endswith("gallery") else p["queries"]
        size=64 if family.endswith("gallery") else p["text_batch"]
        for start in range(0,len(rows),size):yield start//size,rows[start:start+size]
    def load_batch(self,family,index,rows,*,check_stop=None):
        callback=check_stop or (lambda:None);callback()
        s=self.sources;unit=safe_file(s.root,f"features/{family}/{index:06d}")
        refs=tuple(FileRef(str(unit/n),s.declared[f"features/{family}/{index:06d}/{n}"])
                   for n in ("features.pt","metadata.json","intent.json","receipt.json"))
        before={r.path:checked(r) for r in refs}
        md=read_json(unit/"metadata.json");intent=read_json(unit/"intent.json");receipt=read_json(unit/"receipt.json")
        expected={"protocol_sha256":s.protocol_id,"family":family,"index":index,"rows":rows}
        need(intent==expected and all(md[k]==v for k,v in expected.items()),"feature rows/order/intent mismatch")
        need(md["feature_sha256"]==refs[0].sha256 and receipt=={"protocol_sha256":s.protocol_id,
             "metadata_sha256":refs[1].sha256,"feature_sha256":refs[0].sha256},"feature receipt/SHA mismatch")
        import torch
        # 单文件只读取一次，clone之后释放原Tensor大Storage；不使用contiguous替代clone。
        original=torch.load(unit/"features.pt",map_location="cpu",weights_only=True)
        need(isinstance(original,torch.Tensor),"historical feature file must contain a Tensor")
        x=original.detach().clone();del original
        normalized(x,len(rows));stats=md["stats"]
        need(stats["G_new_samples"]==stats["IRRA_image_new_samples"]==0,"historical Joint image-forward stats")
        texts=[]
        if family=="adv_text":
            actions=stats["actions"];need(len(actions)==len(rows),"historical action count")
            for row,action in zip(rows,actions):
                callback();texts.append(self.validate_action(row["caption"],action))
            need(stats["IRRA_text_samples"]==sum(a["adversarial"]!=r["caption"] for a,r in zip(actions,rows)) and
                 stats["H_forward_samples"]==sum(a["H_forward"] for a in actions),"historical text stats")
        else:
            need(stats["IRRA_text_samples"]==0 and stats["reference_reused"] is True,"historical reference-only stats")
            if family=="clean_text":texts=[r["caption"] for r in rows]
        callback()
        need(before=={r.path:checked(r) for r in refs},"cache SHA/mtime changed during read")
        self.batch_counts[family]+=1
        return x,tuple(texts),refs
    def verify_features(self,*,retain=False,check_stop=None,progress=None):
        need(type(retain)is bool,"explicit feature retention")
        callback=check_stop or (lambda:None);s=self.sources
        s.verify_unchanged();all_refs=[];loaded={};bindings={};file_before={}
        self.batch_counts={k:0 for k in FAMILIES}
        start=time.perf_counter()
        for family in FAMILIES:
            blocks=[];texts=[];ids=[];originals=[];row_hashes=[];family_refs=[];count=0
            for index,rows in self.parts(family):
                callback();x,tx,refs=self.load_batch(family,index,rows,check_stop=callback)
                for ref in refs:file_before[ref.path]=checked(ref)
                count+=len(rows);texts.extend(tx);family_refs.extend(refs)
                row_hashes.extend(payload_digest(row) for row in x)
                ids.extend(r["row_id"] for r in rows)
                originals.extend(r["image_sha256"] if family.endswith("gallery") else hashlib.sha256(r["caption"].encode()).hexdigest() for r in rows)
                if retain:blocks.append(x)
                del x
                if progress:progress(family,index+1,count)
            all_refs.extend(family_refs)
            binding=HistoricalFeatureBinding(s.identity.id,s.space.id,family,tuple(ids),tuple(originals),
                tuple(row_hashes),tuple(hashlib.sha256(t.encode()).hexdigest() for t in texts))
            bindings[family]=binding
            if retain:
                import torch
                value=torch.cat(blocks,0).detach().clone();del blocks
                loaded[family]=HistoricalRepresentation(RepresentationBatch(value,s.space.id,
                    "image" if family.endswith("gallery") else "text"),binding,s.identity,payload_digest(value),tuple(family_refs))
        callback();s.verify_unchanged()
        need(file_before=={p:checked(FileRef(p,v["sha256"])) for p,v in file_before.items()},"historical cache changed during verification")
        report={**s.report(),"status":"FEATURES_VERIFIED","feature_batches":dict(self.batch_counts),
                "feature_rows":{k:len(v.ordered_ids) for k,v in bindings.items()},
                "feature_files_verified":len(all_refs),"verified_feature_files_digest":digest(file_before),
                "features_loaded":sum(self.batch_counts.values()),"feature_tables_retained":retain,
                "original_sources_unchanged":True,"feature_files_sha_mtime_unchanged":True,
                "elapsed_seconds":time.perf_counter()-start,"model_loads":0,"neural_forwards":0,
                "scores_computed":0,"metrics_replay_status":"PENDING"}
        return report,loaded

def main(argv=None):
    p=argparse.ArgumentParser(description="CPU historical Joint verification only; no scoring/ranking option")
    p.add_argument("--config",required=True);p.add_argument("--verify-features",action="store_true")
    p.add_argument("--output",required=True)
    a=p.parse_args(argv)
    from ..artifacts import ArtifactStore
    out=Path(a.output).resolve()
    store=ArtifactStore(out,new_base="/home/lzf/ldx/outputs/tbps-framework-stage4b1")
    need(not out.exists(),"immutable verification output already exists")
    producer_paths=[Path(__file__),Path(__file__).with_name("historical_sources.py"),
        Path(__file__).with_name("historical_contracts.py"),Path(__file__).with_name("replay_acceptance.py"),
        Path(__file__).with_name("evaluation.py"),Path(__file__).with_name("metrics.py"),
        Path(__file__).parent.parent/"models/irra_cached.py",Path(__file__).parent.parent/"runtime_closure.py"]
    producer_refs=tuple(FileRef(str(x),sha_file(x)) for x in producer_paths)
    producer_before={ref.path:checked(ref) for ref in producer_refs}
    start=time.perf_counter();report={"status":"FAILED","scores_computed":0,"metrics_replay_status":"PENDING"}
    try:
        sources=JointFullSources(a.config);report=sources.report()
        if a.verify_features:
            allowed={str(safe_file(sources.root,n).resolve()) for n in sources.declared if n.endswith("/features.pt")}
            with feature_only_cpu(allowed) as attempts,native_action_validator(FileRef(**sources.config["attack_sources"])) as validate:
                loader=JointFullLoader(sources,validate)
                def progress(family,batches,rows):
                    if batches%32==0:print("FEATURE_VERIFY_PROGRESS",family,batches,rows,flush=True)
                report,_=loader.verify_features(progress=progress)
                report["forbidden_attempts"]=dict(attempts)
        attack_ref=FileRef(**sources.config["attack_sources"]);checked(attack_ref)
        attack=read_json(attack_ref.path)
        extra=[attack_ref,*producer_refs]
        for name,h in attack.get("runtime_sources",{}).items():extra.append(FileRef(name,h))
        for ref in extra:checked(ref)
        need(producer_before=={ref.path:checked(ref) for ref in producer_refs},"verifier producer changed")
        report["verification_producers"]={ref.path:ref.sha256 for ref in producer_refs}
        if a.verify_features:
            extra.extend(FileRef(str(safe_file(sources.root,name)),h) for name,h in sources.declared.items())
        report["mapped_model_space"]=asdict(sources.space)
        source_refs=tuple({ref.path:ref for ref in (*sources.refs.values(),*extra)}.values())
        if a.verify_features:
            import torch,resource
            report["runtime"]={"torch":str(torch.__version__),"cpu_threads":torch.get_num_threads(),
                               "peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    except BaseException as exc:
        import traceback
        report={"status":"FAILED","error":repr(exc),"traceback":traceback.format_exc(),
                "elapsed_seconds":time.perf_counter()-start,"scores_computed":0,"metrics_replay_status":"PENDING"}
        source_refs=()
    store.publish("result",{"report.json":json_bytes(report)},{"purpose":"historical CPU verification only"},sources=source_refs)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    print("HISTORICAL_"+report["status"],flush=True)
    return 0 if report["status"] in ("METADATA_VERIFIED","FEATURES_VERIFIED") else 1

if __name__=="__main__":raise SystemExit(main())
