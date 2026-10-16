"""默认CPU preflight；正式GPU缓存/训练/Validation必须由用户显式确认。"""
import argparse
from dataclasses import asdict
from pathlib import Path
import time
import traceback
import torch
from PIL import Image
from phase2_sources import digest, sha, read_json
from phase3_inputs import import_gv2
from training_inputs import load_training_catalog
from text_frontend import CaptionFrontend
from tokenizer_probe import official_probe
from .common import (BASE, safe_root, immutable_json, run_lock, Stop, verify_sources, cpu_only)
from .data import (qualification, read_qualification, base_protocol, checked_image,
                   preflight_models, members)
from .storage import GalleryStore
from .runtime import load_runtime, RewardBank
from . import training
from . import checkpoint as ck
from . import validation

def gallery(root,catalog,role,create=False):
    protocol=base_protocol(catalog,role,"gallery")
    verify_sources(protocol)
    return GalleryStore(root/("gallery_"+role),catalog,protocol,create)

def prepare(root,catalog,role,probe):
    read_qualification(root,catalog,role)
    store=gallery(root,catalog,role,True)
    partial=store.audit(require_complete=False)  # pending/损坏拒绝，不自动重算
    images=len(store.samples)-partial["image"]
    texts=len(store.samples)-partial["original_text"] if role=="validation" else 0
    if not images and not texts:
        result=store.audit()
        immutable_json(store.root/"completed.json",result)
        print("FORMAL_GALLERY_COMPLETE",role,len(store.samples),"reused_only",flush=True)
        return result
    runtime=load_runtime(store.protocol,probe,images,texts)
    started=time.monotonic()
    try:
        _,adapter=import_gv2()
        with Stop() as stop:
            for i,s in enumerate(store.samples):
                if stop.requested: break
                checked_image(s)
                def produce():
                    with Image.open(checked_image(s)) as image:pixels=adapter.pixels_from_pil(image)[None]
                    data,info=runtime.image_pair(pixels)
                    return data,{"linf":info["linf"],"execution":{"G":1,"clean":1,"adv":1,"text":0}}
                store.prepare(s,produce)
                if (i+1)%32==0:print("FORMAL_GALLERY_PROGRESS",role,i+1,len(store.samples),flush=True)
            if role=="validation" and not stop.requested:
                for i,s in enumerate(store.samples):
                    if stop.requested: break
                    def produce_text():
                        return {"text":runtime.encode_original((s.caption,))},{
                            "execution":{"G":0,"clean":0,"adv":0,"text":1},
                            "original_truncated":probe.inspect(s.caption).truncated,
                            "token_sha256":digest(probe.inspect(s.caption).model_tokens)}
                    store.prepare(s,produce_text,"original_text")
            stats=runtime.finish()
            result=store.audit(require_complete=not stop.requested)
            result.update(runtime=stats,seconds_this_attempt=time.monotonic()-started,
                GPU_peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                GPU_peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20)
            immutable_json(store.root/"attempts"/(str(time.time_ns())+".json"),result)
            if not stop.requested:
                immutable_json(store.root/"completed.json",store.audit())
            print("FORMAL_GALLERY_SAFE_STOP" if stop.requested else "FORMAL_GALLERY_COMPLETE",
                  role,result["image"],result["original_text"],flush=True)
            return result
    except BaseException:
        immutable_json(store.root/"attempts"/(str(time.time_ns())+"_failure.json"),
                       {"traceback":traceback.format_exc(),"samples":runtime.samples})
        raise
    finally:runtime.close()

def training_protocol(root,catalog,store,args):
    run=root/"training"
    if args.resume or (run/"protocol.json").exists():
        saved=read_json(run/"protocol.json")
        cfg=saved["config"]
        if args.epochs is not None and args.epochs!=cfg["epochs"]:
            raise ValueError("resume cannot change frozen epoch count")
        if args.checkpoint_every is not None and args.checkpoint_every!=cfg["checkpoint_interval_updates"]:
            raise ValueError("resume cannot change checkpoint interval")
    else:
        if args.epochs is None: raise ValueError("--epochs required; candidate10 is not automatic authorization")
        cfg=training.config(args.epochs,args.checkpoint_every or 256)
    if cfg["checkpoint_interval_updates"]<32:
        raise ValueError("formal checkpoints require interval>=32; per-step full checkpoints forbidden")
    ledger=read_qualification(root,catalog,"train")
    protocol=training.protocol_for(catalog,store,ledger,cfg)
    if (run/"protocol.json").exists() and read_json(run/"protocol.json")!=protocol:
        raise ValueError("formal training source/config/cache protocol changed")
    return protocol,ledger

def readonly(root,catalog,section):
    preflight_models()
    result={}
    if section in ("qualification","all"):
        result["qualification"]={r:read_qualification(root,catalog,r)["summary"] for r in ("train","validation")}
    if section in ("gallery","all"):
        result["gallery"]={r:gallery(root,catalog,r).audit() for r in ("train","validation")}
    run=root/"training"
    if section in ("training","validation","all"):
        protocol=read_json(run/"protocol.json");verify_sources(protocol)
        store=gallery(root,catalog,"train")
        ledger=read_qualification(root,catalog,"train")
        expected=training.protocol_for(catalog,store,ledger,protocol["config"])
        if expected!=protocol: raise ValueError("training cache/source/config mismatch")
        payload,evidence=ck.audit(run,protocol)
        result["training"]={**evidence,"epoch_statistics":training.log_statistics(run,protocol)}
        done=run/"completed.json"
        if done.exists():
            record=read_json(done)
            if record!={"protocol_sha256":digest(protocol),"checkpoint_sha256":evidence["published"][-1]["sha256"],
                         "visits":payload["cursor"],"updates":payload["updates"]} or payload["cursor"]!=len(protocol["query_order"]):
                raise ValueError("training completion mismatch")
        if section in ("validation","all"):
            valstore=gallery(root,catalog,"validation")
            valledger=read_qualification(root,catalog,"validation")
            valid_sha={x["sha256"] for x in evidence["published"]}
            values=[]
            for p in sorted((root/"validation").glob("*/completed.json")):
                s=validation.audit_result(p.parent,valstore,valledger,protocol)
                if s["checkpoint_sha256"] not in valid_sha: raise ValueError("validation references foreign checkpoint")
                values.append(s)
            result["validation"]={"completed":len(values),"selection":validation.select_best(values) if values else None}
            for path in (root/"selections").glob("*.json"):
                selection=read_json(path)
                available={v["checkpoint_sha256"]:v for v in values}
                selected=[available[x["checkpoint_sha256"]] for x in selection["candidates"]]
                if validation.select_best(selected)!=selection: raise ValueError("selection evidence mismatch")
    cpu_only()
    print("FORMAL_CPU_AUDIT",section,"PASS",flush=True)
    return result

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--action",choices=("preflight","qualify","prepare","train","validate","audit","select"),default="preflight")
    p.add_argument("--root",default=str(BASE))
    p.add_argument("--role",choices=("train","validation"))
    p.add_argument("--epochs",type=int)
    p.add_argument("--checkpoint-every",type=int)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--rollback-uncommitted",action="store_true")
    p.add_argument("--confirm-gallery-generation",action="store_true")
    p.add_argument("--confirm-training",action="store_true")
    p.add_argument("--confirm-validation",action="store_true")
    g=p.add_mutually_exclusive_group()
    g.add_argument("--checkpoint",type=Path)
    g.add_argument("--all-epoch-checkpoints",action="store_true")
    p.add_argument("--section",choices=("qualification","gallery","training","validation","all"),default="all")
    return p

def main(argv=None):
    args=parser().parse_args(argv)
    # 确认在任何来源读取、模型加载或产物写入之前检查。
    if args.action=="prepare" and (not args.confirm_gallery_generation or args.role is None):
        raise ValueError("user --confirm-gallery-generation and --role required")
    if args.action=="train" and not args.confirm_training: raise ValueError("user --confirm-training required")
    if args.action=="validate" and (not args.confirm_validation or not (args.checkpoint or args.all_epoch_checkpoints)):
        raise ValueError("user --confirm-validation and explicit checkpoint selection required")
    if args.rollback_uncommitted and (args.action!="train" or not args.resume):
        raise ValueError("rollback acknowledgement requires train --resume")
    root=safe_root(args.root)
    catalog=load_training_catalog()
    if args.action=="preflight":
        preflight_models()
        for role in ("train","validation"):
            base_protocol(catalog,role,"data")
            for s in members(catalog,role): checked_image(s)
        cpu_only();print("FORMAL_CPU_PREFLIGHT PASS 2048 128; candidate_epochs=10; GPU_NOT_STARTED")
        return
    if args.action=="qualify":
        with run_lock(root/"qualification"):result=qualification(root)
        print("FORMAL_QUALIFICATION_COMPLETE",result,flush=True);return
    if args.action=="audit":
        result=readonly(root,catalog,args.section)
        print({k:({q:v for q,v in value.items() if q!="files"} if isinstance(value,dict) else value)
               for k,value in result.items()});return
    if args.action=="prepare":
        with run_lock(root/("gallery_"+args.role)):prepare(root,catalog,args.role,official_probe())
        return
    trainstore=gallery(root,catalog,"train")
    if args.action=="train":
        with run_lock(root/"training"):
            protocol,ledger=training_protocol(root,catalog,trainstore,args)
            if args.resume:
                payload,evidence=ck.audit(root/"training",protocol)
                if payload["cursor"]==len(protocol["query_order"]) and not evidence["uncommitted_tail_exists"]:
                    print("FORMAL_TRAIN_ALREADY_COMPLETE",payload["cursor"],payload["updates"]);return
                if evidence["uncommitted_tail_exists"] and not args.rollback_uncommitted:
                    raise ValueError("uncommitted tail; explicitly acknowledge --rollback-uncommitted")
            elif (root/"training"/"protocol.json").exists():
                raise ValueError("existing formal run; use --resume")
            probe=official_probe()
            bank=RewardBank(catalog,trainstore,CaptionFrontend())
            runtime=load_runtime(protocol,probe,0,3*len(protocol["query_order"]))
            try:
                with Stop() as stop:
                    result=training.execute(root/"training",protocol,catalog,ledger,trainstore,probe,
                        bank,runtime,stop,args.resume,args.rollback_uncommitted)
                if result["complete"]:
                    immutable_json(root/"training"/"completed.json",{
                        "protocol_sha256":digest(protocol),"checkpoint_sha256":result["checkpoint_sha256"],
                        "visits":result["committed_visits"],"updates":result["committed_updates"]})
            except BaseException:
                immutable_json(root/"training"/"attempts"/(str(time.time_ns())+"_failure.json"),
                        {"traceback":traceback.format_exc(),"samples":runtime.samples})
                raise
            finally:runtime.close()
        return
    protocol=read_json(root/"training"/"protocol.json")
    verify_sources(protocol)
    expected=training.protocol_for(catalog,trainstore,read_qualification(root,catalog,"train"),protocol["config"])
    if protocol!=expected: raise ValueError("training source/cache mismatch")
    _,evidence=ck.audit(root/"training",protocol)
    valstore=gallery(root,catalog,"validation");valstore.audit()
    ledger=read_qualification(root,catalog,"validation")
    if args.action=="select":
        result=readonly(root,catalog,"validation")["validation"]["selection"]
        if result is None:raise ValueError("no completed Validation")
        immutable_json(root/"selections"/(digest(result)+".json"),result)
        print("VALIDATION_SELECTION_COMPLETE",result);return
    targets=[x for x in evidence["published"] if x["reason"]=="epoch_complete"] if args.all_epoch_checkpoints else [
        x for x in evidence["published"] if (Path(x["path"])/"checkpoint.pt").resolve()==args.checkpoint.resolve()]
    if not targets: raise ValueError("no matching immutable checkpoint")
    # 已完成验证只读返回，避免无意义加载GPU。
    if all((root/"validation"/x["sha256"]/"completed.json").exists() for x in targets):
        for x in targets:validation.audit_result(root/"validation"/x["sha256"],valstore,ledger,protocol)
        print("VALIDATION_COMPLETE_REUSED",len(targets));return
    with run_lock(root/"validation"):
        probe=official_probe()
        runtime=load_runtime(protocol,probe,0,len(valstore.samples)*len(targets))
        try:
            with Stop() as stop:
                for item in targets:
                    payload=torch.load(Path(item["path"])/"checkpoint.pt",map_location="cpu",weights_only=False)
                    policy=training.fresh_policy(probe,protocol["config"])
                    policy.load_state_dict(payload["policy"],strict=True)
                    validation.evaluate(root/"validation",protocol,payload,item["sha256"],valstore,ledger,policy,runtime,probe,stop)
                    if stop.requested:break
            immutable_json(root/"validation"/"attempts"/(str(time.time_ns())+".json"),runtime.finish())
        except BaseException:
            immutable_json(root/"validation"/"attempts"/(str(time.time_ns())+"_failure.json"),
                           {"traceback":traceback.format_exc(),"samples":runtime.samples})
            raise
        finally:runtime.close()

if __name__=="__main__":main()
