"""G-v2完整缓存入口；默认CPU审计，显式full或small，不评分。"""
import argparse,copy,sys
from pathlib import Path
import torch
import cache_io as io
from training_data import CACHE_FILES,file_contract,publish_manifest,checked_cache,identity,check_pair,RAW,SPLIT
from condition_schema import validate

SOURCE=Path("/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_image_generator_20261014/outputs/conditions")
OLD_SMOKE=Path("/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_image_generator_v2_20261014/outputs/smoke")
SMALL_IDS=("cuhk:26530:0","cuhk:26747:0","cuhk:9955:0","cuhk:26196:0","cuhk:17663:0")

def source_rows():
    m=io.load(SOURCE/"manifest.json"); done=io.load(SOURCE/"completed.json")
    io.verify({str(SOURCE/"manifest.json"):done["manifest_sha256"],m["protocol_path"]:m["protocol_sha256"]})
    if not done["method_audit_passed"] or m["counts"]!={"train":2048,"validation":128} or m["test_used"]:
        raise ValueError("original completed source required")
    raw=io.load(RAW);split=io.load(SPLIT)
    rows=[];upstream={str(SOURCE/"manifest.json"):io.sha(SOURCE/"manifest.json"),
        str(SOURCE/"completed.json"):io.sha(SOURCE/"completed.json"),str(RAW):io.sha(RAW),str(SPLIT):io.sha(SPLIT)}
    for role in ("train","validation"):
        for e in m[role]:
            io.verify({e["tensor_path"]:e["tensor_sha256"],e["metadata_path"]:e["metadata_sha256"]})
            md=io.load(e["metadata_path"]); p=md["payload"];record=md["score_record"]
            if (record["row_id"],record["pid"],record["caption"],record["paired_image"])!=(e["row_id"],e["pid"],e["caption"],e["image"]):
                raise ValueError("frozen score identity mismatch")
            if record["retrieval_backward_count"]!=1 or not record["zero_intervention_exact"] or not md["S_raw_unchanged"]:
                raise ValueError("original clean score audit failed")
            if p["gamma"]!=3 or p["beta"]!=1 or p["caption_id"]!=e["row_id"]: raise ValueError("frozen selection protocol")
            scores=[a["S_raw"] for a in p["selected_attributes"]]
            weights=[s/sum(scores) for s in scores]
            # 仅核验公式；保留原CUDA float64权重，不以CPU求和末位替换。
            if not torch.allclose(torch.tensor(md["score_normalized_weights"],dtype=torch.float64),
                                  torch.tensor(weights,dtype=torch.float64),atol=1e-15,rtol=1e-14):
                raise ValueError("frozen score weights changed")
            if md["role"]!=role or p["natural_k"]!=e["K"] or p["selected_attributes"]!=p["ranked_attributes"][:e["K"]]:
                raise ValueError("frozen raw-ranked selection mismatch")
            if md["selected_instance_ids"]!=[a["instance_id"] for a in p["selected_attributes"]]: raise ValueError("attribute order changed")
            lookup={a["instance_id"]:a for a in md["score_record"]["attributes"]}
            for a in p["selected_attributes"]:
                if a!={**lookup[a["instance_id"]]["instance"],"S_raw":lookup[a["instance_id"]]["S_raw"]}: raise ValueError("S/instance changed")
            if (md["row_id"],md["pid"],md["caption"],md["image"])!=(e["row_id"],e["pid"],e["caption"],e["image"]):
                raise ValueError("source identity mismatch")
            ident={"row_id":e["row_id"],"image_rel_path":e["image"],"caption_index":int(e["row_id"].split(":")[-1]),
                   "split":"train","role":role,"pid":e["pid"],"caption":e["caption"],"K":e["K"]}
            # 使用原G-v2 schema验证前先核对原annotation/split/image SHA。
            _,ri,j=e["row_id"].split(":"); item=raw[int(ri)]
            if (item["split"],item["id"],item["file_path"],item["captions"][int(j)])!=("train",e["pid"],e["image"],e["caption"]):
                raise ValueError("annotation pairing mismatch")
            members={split["records"][i]["row_id"] for i in split["train_indices" if role=="train" else "validation_indices"]}
            if e["row_id"] not in members: raise ValueError("role mismatch")
            from training_data import IMAGES
            io.verify({str(IMAGES/e["image"]):e["image_sha256"]})
            rows.append((e,md,ident))
            upstream[e["tensor_path"]]=e["tensor_sha256"];upstream[e["metadata_path"]]=e["metadata_sha256"]
    if len({i["row_id"] for _,_,i in rows})!=2176: raise ValueError("source coverage")
    if {i["pid"] for _,_,i in rows if i["role"]=="train"}&{i["pid"] for _,_,i in rows if i["role"]=="validation"}:
        raise ValueError("PID leakage")
    # 锁定实际冻结JVP/basis/IRRA依赖；不纳入G-v1 Python或整个G-v2目录。
    for p,h in m["frozen_sha256"].items():
        if "attribute_image_generator_20261014" not in p:
            upstream[p]=h
    io.verify(upstream)
    return rows,upstream

def prepare(output,rows,upstream,scope,producer=None,reuse=None):
    protocol={"schema":"gv2-full-prepare-protocol-2","scope":scope,
        "samples":[i for _,_,i in rows],"target_counts":{"train":2048,"validation":128},
        "producer_sha256":file_contract(CACHE_FILES),"upstream_sha256":upstream,
        "dataset_sha256":{str(p):upstream[str(p)] for p in (RAW,SPLIT) if str(p) in upstream},
        "new_scoring_backwards":0,"z_t_usage":"Global supervision only; G(I) input is image"}
    with io.output_lock(output) as out:
        if (out/"protocol.json").exists() and io.digest(io.load(out/"protocol.json"))!=io.digest(protocol):
            raise ValueError("preparation protocol changed")
        if (out/"completed.json").exists(): return checked_cache(out)
        io.save_json(out/"protocol.json",protocol)
        entries=[];new_jvps=0;reused=0;grad_calls=0
        for e,md,ident in rows:
            target=out/"samples"/ident["row_id"].replace(":","_")
            if target.exists():
                entry=io.entry_at(target);v,saved,_=io.read_entry(entry,io.digest(protocol))
                if saved["source_fingerprint"]!=io.digest([e,md]): raise ValueError("resume source mismatch")
            else:
                cached=reuse(ident["row_id"]) if reuse is not None else None
                if cached is not None:
                    v,saved=cached; saved=copy.deepcopy(saved);reused+=1
                else:
                    if producer is None: raise ValueError("missing offline JVP producer")
                    v,saved=producer(e,md);new_jvps+=saved["new_JVPs"];grad_calls+=saved["new_JVP_autograd_grad_calls"]
                validate(v,ident)
                saved.update({"protocol_hash":io.digest(protocol),"source_fingerprint":io.digest([e,md])})
                entry=io.write_entry(out,v,saved)
            entries.append(entry)
            print("CACHE_READY",ident["row_id"],"K",ident["K"],flush=True)
        io.verify(upstream);io.verify(protocol["producer_sha256"])
        io.save_json(out/"summary.json",{"new_scoring_backwards_this_run":0,"new_JVPs_this_run":new_jvps,
            "new_JVP_autograd_grad_calls_this_run":grad_calls,"text_supervision_recomputed":False,
            "z_t_usage":"Global Loss supervision only; never G input","reused_smoke_this_run":reused,"count":len(entries),"full_preparation":scope=="full-2176"})
        return publish_manifest(out,entries,protocol)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group()
    group.add_argument("--allow-full-preparation",action="store_true")
    group.add_argument("--allow-small",action="store_true")
    p.add_argument("--output",type=Path,default=io.OUTPUTS/"conditions_full")
    args=p.parse_args(argv)
    rows,upstream=source_rows()
    if not args.allow_full_preparation and not args.allow_small:
        print("PREFLIGHT_ONLY 2048+128 source audited; no GPU/preparation");return
    if args.allow_small:
        order={s:i for i,s in enumerate(SMALL_IDS)}
        rows=sorted((r for r in rows if r[2]["row_id"] in order),key=lambda r:order[r[2]["row_id"]])
        if args.output==io.OUTPUTS/"conditions_full": args.output=io.OUTPUTS/"phase24/conditions_small"
    # 检查旧smoke的固定历史锁；新训练缓存不继承整个目录锁。
    old=io.checked_cache(OLD_SMOKE)
    reused={e["row_id"]:io.read_entry(e)[:2] for e in old["entries"]}
    producer_box=[]
    def producer(e,md):
        if not producer_box:
            from irra_image_adapter import load_official
            from prep_text_frontend import Frontend
            producer_box.append(Frontend(load_official()))
        return producer_box[0](e,md)
    def reuse(row_id):
        # small验证K1真实重算；full优先复用5条已有独立JVP。
        if args.allow_small and row_id=="cuhk:26747:0": return None
        return reused.get(row_id)
    prepare(args.output,rows,upstream,"small-five" if args.allow_small else "full-2176",producer,reuse)
if __name__=="__main__":main()
