"""G首版共享图库评测；默认只CPU preflight，full必须用户显式启动。"""
import argparse
import csv
import json
import time
from pathlib import Path
from unittest.mock import patch
import torch
from PIL import Image
from torchvision import transforms
import common as c
import checkpoint_io as ck
import g_test_core as g
from generator_model import generate_once

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("preflight","prepare","generate","evaluate","all"),default="preflight")
    p.add_argument("--scope",choices=("smoke","full"),default="smoke")
    p.add_argument("--allow-full",action="store_true");p.add_argument("--output",type=Path)
    return p

def image_pixels(row):
    path=(c.IMAGE_ROOT/row["image"]).resolve()
    if not path.is_relative_to(c.IMAGE_ROOT.resolve()):raise ValueError("unsafe source image")
    with Image.open(path) as im:return transforms.ToTensor()(transforms.Resize((384,128))(im.convert("RGB"))).unsqueeze(0)

def old_entry(manifest,index):
    entry=manifest["rows"][index]
    if entry["index"]!=index:raise ValueError("reference score index")
    return entry

def prepare(out,m,reference,protocol):
    if (out/"prepare_completed.json").exists():
        g.checked_conditions(out,protocol);print("TEST_CONDITIONS_ALREADY_COMPLETE",flush=True);return
    import integration as it
    victim=it.frozen.setup();versions={n:p._version for n,p in victim.named_parameters()}
    bases=c.load(it.frozen.score.BASIS)
    entries=[];grad_calls=0;checks=[];new=0;start=time.time()
    # 完整query feature只读取，smoke也始终使用原3074图评分而非两图重评分。
    cache=torch.load(g.REFERENCE/"clean_features.pt",map_location="cpu",weights_only=True)
    g.normalized(cache["image"],3074);g.normalized(cache["text"],6156)
    qlookup={q["row_id"]:i for i,q in enumerate(reference["queries"])}
    for row in m["gallery"]:
        source=old_entry(reference,row["index"]);saved=c.load(source["score_path"])
        meta=out/"conditions/records"/(row["row_id"].replace(":","_")+".json")
        tensor=out/"conditions/tensors"/(row["row_id"].replace(":","_")+".pt")
        if c.sha(c.IMAGE_ROOT/row["image"])!=source["image_sha256"]:raise ValueError("test image changed")
        if meta.exists():
            data=c.load(meta)
            if data["protocol_sha256"]!=c.identity(protocol) or data["score_sha256"]!=source["score_sha256"]:raise ValueError("condition resume provenance")
            c.verify({str(tensor):data["tensor_sha256"]})
        else:
            with patch.object(it,"fresh_score",side_effect=AssertionError("no test rescoring")),patch.object(it.frozen,"score_one",side_effect=AssertionError("no test rescoring")),patch("torch.autograd.grad",wraps=torch.autograd.grad) as calls:
                with torch.enable_grad():condition,details=it.prepare_condition(victim,row,saved["record"],bases)
            grad_calls+=calls.call_count
            delta=float((condition["z_t"]-cache["text"][qlookup[row["row_id"]]]).abs().max())
            if delta>2e-5:raise ValueError("fresh singleton caption vs official query cache mismatch")
            if details["selected_instance_ids"]!=source["selection_ids"]["S_TOPK"] or details["K"]!=source["natural_k"]:
                raise ValueError("frozen test selection changed")
            c.tensor_save(tensor,condition)
            data={**row,"image_sha256":source["image_sha256"],"score_path":source["score_path"],"score_sha256":source["score_sha256"],
                "protocol_sha256":c.identity(protocol),"tensor_sha256":c.sha(tensor),
                **{k:v for k,v in details.items() if k!="payload"},
                "new_score_backward":0,"offline_JVP_autograd_grad_calls":calls.call_count,
                "z_vs_official_query_max_abs":delta,"source_scope":"official test score cache"}
            c.save(meta,data);new+=1
        entry={**row,"image_sha256":source["image_sha256"],"score_sha256":source["score_sha256"],"K":data["K"],
               "tensor_path":str(tensor),"tensor_sha256":data["tensor_sha256"],
               "metadata_path":str(meta),"metadata_sha256":c.sha(meta)}
        c.read_condition(entry);entries.append(entry);checks.extend(data["JVP_checks"])
        print("TEST_CONDITION",row["index"],row["row_id"],"K",entry["K"],flush=True)
    it.frozen.assert_frozen(victim,versions);c.verify(protocol["source_sha256"])
    manifest={"schema":"G-official-test-conditions-1","scope":protocol["scope"],"entries":entries,
              "protocol_sha256":c.identity(protocol),"checkpoint_sha256":protocol["checkpoint_sha256"]}
    c.save(out/"conditions/manifest.json",manifest)
    c.save(out/"prepare_completed.json",{"method_audit_passed":True,"manifest_sha256":c.sha(out/"conditions/manifest.json"),
        "new_condition_count":new,"new_score_backward":0,"score_reference_count":3074,
        "offline_JVP_autograd_grad_calls_this_run":grad_calls,"IRRA_frozen":True,
        "zero_exact":all(x["zero_exact"] for x in checks),
        "max_JVP_FD_relative_error":max((x["central_fd_relative_error"] for x in checks),default=0),
        "seconds":time.time()-start})
    print("TEST_CONDITIONS_COMPLETE",flush=True)

def check_generation_record(out,entry,protocol):
    file=out/"generated"/f'{entry["index"]:06d}.json';meta=c.load(file)
    if meta["index"]!=entry["index"] or meta["row_id"]!=entry["row_id"] or meta["pid"]!=entry["pid"]:
        raise ValueError("generation identity")
    if meta["condition_sha256"]!=entry["tensor_sha256"] or meta["protocol_sha256"]!=c.identity(protocol):
        raise ValueError("generation condition/protocol")
    path=out/"gallery/G_ADV"/f'{entry["index"]:06d}.pt'
    if meta["archive_path"]!=str(path) or c.sha(path)!=meta["archive_sha256"]:raise ValueError("generation archive SHA")
    if meta["G_forward_calls"]!=1 or meta["backward_calls"]!=0 or meta["optimization_steps"]!=0:raise ValueError("generation must be one forward")
    value=torch.load(path,map_location="cpu",weights_only=True)
    clean=image_pixels(entry)
    budget=g.validate_archive(value,clean,entry)
    if budget!=meta["linf"]:raise ValueError("saved budget differs")
    return meta,value,file

def generate(out,m,protocol):
    conditions=g.checked_conditions(out,protocol)
    if (out/"generate_completed.json").exists():
        for e in conditions["entries"]:check_generation_record(out,e,protocol)
        print("G_GENERATION_ALREADY_COMPLETE",flush=True);return
    import integration as it
    victim=it.frozen.setup();versions={n:p._version for n,p in victim.named_parameters()}
    model=g.load_generator(protocol,"cuda");g_versions={n:p._version for n,p in model.named_parameters()}
    index=[];max_budget=0.;new=0;start=time.time();fresh_clean_errors=[]
    reference=torch.load(g.REFERENCE/"clean_features.pt",map_location="cpu",weights_only=True)
    for entry in conditions["entries"]:
        record=out/"generated"/f'{entry["index"]:06d}.json'
        if c.sha(c.IMAGE_ROOT/entry["image"])!=entry["image_sha256"]:raise ValueError("source image SHA")
        if record.exists():meta,value,file=check_generation_record(out,entry,protocol)
        else:
            cond=c.read_condition(entry);clean=image_pixels(entry).cuda()
            calls=[];handle=model.register_forward_hook(lambda *args:calls.append(1))
            torch.cuda.synchronize();t=time.time()
            try:
                with patch("torch.autograd.grad",side_effect=AssertionError("generation backward forbidden")),torch.inference_mode():
                    result=generate_once(model,clean,cond["z_t"][None].cuda(),cond["c_attr"][None].cuda())
                    feature=victim.encode_images(result["adversarial"]).cpu()
                    if protocol["scope"]=="smoke":
                        fresh=victim.encode_images(clean).cpu()
                        err=float((fresh-reference["image"][entry["index"]:entry["index"]+1]).abs().max())
                        if err>2e-5:raise ValueError("fresh clean image differs from official clean cache")
                        fresh_clean_errors.append(err)
            finally:handle.remove()
            torch.cuda.synchronize();elapsed=time.time()-t
            if len(calls)!=1 or any(x.requires_grad for x in result.values()):raise ValueError("G forward contract")
            value={"pixels":result["adversarial"].cpu(),"delta":result["delta"].cpu(),
                   "effective_delta":result["effective_delta"].cpu(),"IRRA_feature":feature}
            budget=g.validate_archive(value,clean.cpu(),entry)
            path=out/"gallery/G_ADV"/f'{entry["index"]:06d}.pt';c.tensor_save(path,value)
            meta={"index":entry["index"],"row_id":entry["row_id"],"pid":entry["pid"],"image":entry["image"],
                  "K":entry["K"],"condition_sha256":entry["tensor_sha256"],"protocol_sha256":c.identity(protocol),
                  "archive_path":str(path),"archive_sha256":c.sha(path),"linf":budget,"G_forward_calls":1,
                  "backward_calls":0,"optimization_steps":0,"JVP_computations":0,"seconds":elapsed}
            c.save(record,meta);new+=1
            meta,value,file=check_generation_record(out,entry,protocol)
        max_budget=max(max_budget,meta["linf"])
        index.append({"index":entry["index"],"record_path":str(file),"record_sha256":c.sha(file)})
        print("G_GENERATED",entry["index"],entry["row_id"],"linf",meta["linf"],flush=True)
    it.frozen.assert_frozen(victim,versions)
    if any(p._version!=g_versions[n] or p.requires_grad or p.grad is not None for n,p in model.named_parameters()):raise ValueError("G changed")
    c.verify(protocol["source_sha256"])
    c.save(out/"gallery_index.json",{"rows":index,"shared_adversarial_gallery":True,"count":len(index)})
    c.save(out/"generate_completed.json",{"method_audit_passed":True,"index_sha256":c.sha(out/"gallery_index.json"),
        "gallery_count":len(index),"G_forward_calls_total":len(index),"new_G_forward_calls_this_run":new,
        "backward_calls":0,"score_backward":0,"JVP_computations":0,"optimization_steps":0,
        "IRRA_G_frozen":True,"maximum_linf":max_budget,"fresh_clean_max_abs":max(fresh_clean_errors,default=None),
        "seconds":time.time()-start})
    print("G_GENERATION_COMPLETE",flush=True)

def evaluate(out,m,protocol):
    # 只读取缓存：此阶段不导入integration，不运行任何模型。
    conditions=g.checked_conditions(out,protocol)
    done=c.load(out/"generate_completed.json")
    if not done["method_audit_passed"] or done["gallery_count"]!=len(m["gallery"]) or done["index_sha256"]!=c.sha(out/"gallery_index.json"):
        raise ValueError("incomplete shared gallery")
    index=c.load(out/"gallery_index.json")["rows"]
    if [x["index"] for x in index]!=[e["index"] for e in conditions["entries"]]:raise ValueError("gallery index order")
    c.verify({x["record_path"]:x["record_sha256"] for x in index})
    adv=[]
    for e in conditions["entries"]:
        _,v,_=check_generation_record(out,e,protocol);adv.append(v["IRRA_feature"])
    features=torch.load(g.REFERENCE/"clean_features.pt",map_location="cpu",weights_only=True)
    qlookup={q["row_id"]:i for i,q in enumerate(c.load(g.REFERENCE/"selection_manifest.json")["queries"])}
    query=features["text"][[qlookup[q["row_id"]] for q in m["queries"]]]
    clean=features["image"][[row["index"] for row in m["gallery"]]]
    torch.set_num_threads(8)
    cm,cr=g.metrics_for(query,clean,m["queries"],m["gallery"])
    am,ar=g.metrics_for(query,torch.cat(adv),m["queries"],m["gallery"])
    comparison={"scope":"smoke subset; no full clean metric recomputation"}
    if protocol["scope"]=="full":
        expected=c.load(g.REFERENCE/"summary.json")["clean"]
        try:comparison={"clean_metric_abs_error":g.assert_clean_match(cm,expected),"clean_cache_exact_reuse":True}
        except ValueError as exc:
            c.save(out/"clean_mismatch.json",{"actual":cm,"expected":expected,"reason":str(exc),
              "features":"same SHA-pinned cache; investigate metric source/CPU reduction/tie behavior before reporting"})
            raise
    dr=g.drop_rate(cm["mAP"],am["mAP"])
    summary={"scope":protocol["scope"],"formal_test_evaluation":protocol["scope"]=="full",
             "shared_adversarial_gallery":True,"gallery_count":len(m["gallery"]),"query_count":len(m["queries"]),
             "victim_count":1,"victim":"IRRA-official-CUHK-fp32","metric_unit":"percent",
             "clean":cm,"G_ADV":am,"aAP_clean":cm["mAP"],"aAP_adversarial":am["mAP"],"mDR":dr,
             "mDR_definition":"(clean_mAP-adversarial_mAP)/clean_mAP*100; single victim",
             "maximum_linf":done["maximum_linf"],"G_forwards_per_image":1,
             "score_backward_during_generation":0,"backward_during_generation":0,"test_time_optimization_steps":0,
             "condition_preparation":"exact prior official-test S/Top-K plus original JVP; offline gradients only",
             "clean_comparison":comparison,"checkpoint_sha256":protocol["checkpoint_sha256"]}
    c.save(out/"summary.json",summary)
    with c.safe(out/"per_query_metrics.jsonl").open("w",encoding="utf-8") as f:
        for q,a,b in zip(m["queries"],cr,ar):f.write(json.dumps({**q,"clean":a,"G_ADV":b},ensure_ascii=False)+"\n")
    with c.safe(out/"metrics.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["mode","mAP","R@1","mINP","mDR"])
        w.writerow(["CLEAN",cm["mAP"],cm["R@1"],cm["mINP"],0])
        w.writerow(["G_ADV",am["mAP"],am["R@1"],am["mINP"],dr])
    c.verify(protocol["source_sha256"])
    c.save(out/"completed.json",{"method_audit_passed":True,**summary,"source_SHA_unchanged":True})
    print("G_TEST_COMPLETE",json.dumps(summary,ensure_ascii=False),flush=True)

def main(argv=None):
    args=parser().parse_args(argv)
    if args.scope=="full" and args.stage!="preflight" and not args.allow_full:
        raise SystemExit("full evaluation is manual only; supply --allow-full")
    report,m,reference,locks=g.audit()
    print("TEST_PREFLIGHT",json.dumps(report,ensure_ascii=False),flush=True)
    if args.stage=="preflight":return
    m=g.scoped(m,args.scope);protocol=g.make_protocol(report,m,locks,args.scope)
    if args.scope=="full":
        smoke=c.ROOT/"outputs/test_smoke"
        if not c.load(smoke/"completed.json")["method_audit_passed"]:raise ValueError("test smoke required")
        old=c.load(smoke/"protocol.json")
        if old["source_sha256"]!=locks or old["checkpoint_sha256"]!=protocol["checkpoint_sha256"]:
            raise ValueError("source/checkpoint changed after test smoke")
        c.verify(old["source_sha256"])
    out=c.safe(args.output or c.ROOT/"outputs"/("test_"+args.scope))
    with ck.training_lock(out):
        if (out/"protocol.json").exists() and c.load(out/"protocol.json")!=protocol:raise ValueError("output protocol changed")
        c.save(out/"protocol.json",protocol)
        if args.stage in ("prepare","all"):prepare(out,m,reference,protocol)
        if args.stage in ("generate","all"):generate(out,m,protocol)
        if args.stage in ("evaluate","all"):evaluate(out,m,protocol)

if __name__=="__main__":main()
