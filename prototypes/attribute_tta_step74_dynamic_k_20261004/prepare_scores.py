"""用户手动clean score preparation；默认仅3句smoke，绝不启动全量攻击。"""
import argparse,copy,json,time,collections
from pathlib import Path
from unittest.mock import patch
import deps
from selection_audit import *
def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=["smoke","prepare"],default="smoke")
    p.add_argument("--allow-score-preparation",action="store_true")
    p.add_argument("--output",type=Path)
    return p

def compact(record):
    x={k:copy.deepcopy(v) for k,v in record.items() if k!="attributes" and k!="topk"}
    x["attributes"]=[{k:copy.deepcopy(a[k]) for k in ("instance","instance_id","status","rank","S_raw","w") if k in a}
                     for a in record["attributes"]]
    return x

def smoke_ids(split,known):
    validation={split["records"][i]["row_id"] for i in split["validation_indices"]}
    counts=collections.Counter()
    # 只看冻结接口门禁与候选数，不看新S或攻击结果。
    for a in deps.rows(deps.V55/"outputs/full/instances.jsonl"):
        if (not a["unresolved"] and a["representation_status"]=="ready"
            and a.get("representation",{}).get("basis_rank",0)>0):
            counts[a["row_id"]]+=1
    fresh=sorted((cid for cid in validation if cid not in known),
                 key=lambda cid:(-counts[cid],tuple(map(int,cid.split(":")[1:]))))
    if "cuhk:34:0" not in validation:raise ValueError("fixed k1 source not validation")
    if "cuhk:26:0" not in validation:raise ValueError("fixed natural k5 source not validation")
    return ["cuhk:34:0","cuhk:26:0"]+fresh[:1]

def main(argv=None):
    args=parser().parse_args(argv)
    if args.mode=="prepare" and not args.allow_score_preparation:
        raise SystemExit("full clean score preparation is user-run only; --allow-score-preparation required")
    hashes=deps.frozen_hashes();own=deps.own_hashes();old=deps.scoring_runner()
    split=load(deps.V4/"outputs/split.json")
    records=sorted((split["records"][i] for i in split["validation_indices"]),key=old.numeric_row_id)
    validation={r["row_id"] for r in records}
    known={r["row_id"]:r for r in load(deps.P51/"outputs/diagnostic/results.json")}
    ids=smoke_ids(split,known) if args.mode=="smoke" else [r["row_id"] for r in records]
    if args.mode=="smoke" and len(ids)>3:raise ValueError("smoke boundary")
    wanted=set(ids);records=[next(r for r in records if r["row_id"]==cid) for cid in ids] if args.mode=="smoke" else records
    out=new_output(args.output or deps.ROOT/"outputs"/("score_smoke" if args.mode=="smoke" else "clean_scores"))
    protocol={"schema":"step74-clean-score-cache-1","mode":args.mode,"row_ids":ids,
        "validation_only":True,"split_sha256":digest(deps.V4/"outputs/split.json"),
        "scoring":"unchanged Step5.1 baseline_gradient + FrozenLocalization.attach + score_instances; S_raw=||Q^T w^T G||2",
        "layer":8,"precision":"fp32","tau":.02,"gallery_count":34054,"gamma":3,"beta":1,
        "each_new_caption_one_clean_retrieval_backward":True,"reuse_verified_step51_validation_scores":True,
        "attack":False,"training":False,"IRRA_full_feature_reextraction":False,
        "smoke_rule":"validation 34:0 (existing k1), 26:0 (saved natural k5), one uncached validation row with most ready rank>0 instances, numeric row_id tie; fixed before new scoring",
        "frozen_sha256":hashes,"source_sha256":own,"material_F_threshold":MATERIAL_F_THRESHOLD}
    save(out/"protocol.json",protocol)
    labels={}
    for row in deps.rows(old.v4.LABELS):
        if row["row_id"] in wanted:labels[row["row_id"]]=row
    if set(labels)!=wanted:raise ValueError("missing Stage01 source")
    bases=load(old.BASIS);raw=load(old.v4.RAW)
    old.v4.setup();victim=old.v4.load_victim()
    reader=old.v4.SlotReader([len(v) for v in split["values"]]).cuda()
    cp=old.torch.load(deps.V4/"outputs/reader_final.pt",map_location="cuda",weights_only=False)
    if cp["epoch"]!=20 or cp["slots"]!=split["slots"] or cp["values"]!=split["values"]:raise ValueError("reader contract")
    reader.load_state_dict(cp["state_dict"]);reader.eval().requires_grad_(False)
    reader_state={k:v.detach().cpu().clone() for k,v in reader.state_dict().items()}
    versions={n:p._version for n,p in victim.named_parameters()}
    gallery=old.torch.load(old.GALLERY,map_location="cpu",weights_only=False)
    baseline=load(old.BASE)
    if gallery["spec"]!=baseline["gallery_spec"] or len(gallery["features"])!=34054:
        raise ValueError("frozen clean train gallery mismatch")
    if str(old.torch.__version__)!=gallery["spec"]["torch_version"]:raise ValueError("torch/gallery provenance")
    features=gallery["features"].cuda();path_index={x:i for i,x in enumerate(gallery["spec"]["gallery_paths"])}
    cached=old.CachedReader(split,bases);localizer=old.FrozenLocalization(reader,split,bases,cached)
    lookup={r["row_id"]:i for i,r in enumerate(split["records"])}
    started=time.time();reused=0;new_backwards=0;status=collections.Counter()
    cache_file=out/"score_cache.jsonl"
    with cache_file.open("x",encoding="utf-8") as stream:
        for rec in records:
            cid=rec["row_id"];label=labels[cid];idx=lookup[cid]
            anno=raw[int(cid.split(":")[1])]
            if (rec["caption"]!=label["caption"] or anno["split"]!="train"
                or anno["id"]!=rec["pid"] or anno["file_path"]!=label["image"]):
                raise ValueError("PID/source/paired image mismatch")
            if cid in known:
                row=compact(known[cid])
                if (row["caption"]!=rec["caption"] or row["pid"]!=rec["pid"] or
                    row["paired_image"]!=label["image"] or row["paired_gallery_index"]!=path_index[label["image"]]):
                    raise ValueError("saved score source mismatch")
                row["score_origin"]="verified_frozen_Step51_cache";reused+=1
            else:
                probe=old.AttributeUnitV2Probe(victim,features,path_index[label["image"]],tau=.02)
                with patch("torch.autograd.grad",wraps=old.torch.autograd.grad) as calls:
                    feature,H,G,F,cosine,zero=old.baseline_gradient(probe,rec["caption"])
                    if H.dtype!=old.torch.float32 or G.dtype!=old.torch.float32 or zero.any():
                        raise ValueError("fp32 block8 score contract")
                    with old.torch.no_grad():
                        reference=victim.encode_texts([rec["caption"]])
                        explicit,_,effzero=probe.encode(rec["caption"],8,old.torch.zeros_like(H))
                    if not old.torch.equal(feature,reference) or not old.torch.equal(explicit,reference) or effzero.any():
                        raise ValueError("zero intervention not exact")
                    attrs=old.convert_record(label)
                    attrs,weights,alignment,regression=localizer.attach(attrs,rec["caption"],H,idx)
                    cached_H=old.torch.from_numpy(old.np.array(cached.H[idx])).to(H.device)
                    herror=float((cached_H-H[:,0]).abs().max())
                    if not old.torch.allclose(cached_H,H[:,0],atol=2e-4,rtol=2e-5):
                        raise ValueError("fresh/cached block8 regression")
                    scored=old.score_instances(attrs,rec["caption"],G,weights,bases,random_bases=None)
                    if calls.call_count!=1:raise ValueError("not exactly one clean score backward")
                probe.assert_frozen();new_backwards+=1
                row=compact({"row_id":cid,"pid":rec["pid"],"caption":rec["caption"],
                    "paired_image":label["image"],"paired_gallery_index":path_index[label["image"]],
                    "F":F,"paired_cosine":cosine,"retrieval_backward_count":1,"zero_intervention_exact":True,
                    "cache_H_max_abs_error":herror,"cache_representation_regression":regression,"attributes":scored})
                row["score_origin"]="once_clean_Step51_frozen_functions"
                del probe,H,G,feature,scored,weights,attrs,cached_H
            selection(row) # 在攻击前核查span/resolved/rank/positive finite score门禁。
            status.update(a["status"] for a in row["attributes"])
            stream.write(json.dumps(row,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n")
            stream.flush()
            print("SCORE_DONE",cid,"origin",row["score_origin"],"natural_k",selection(row)["natural_k"],flush=True)
    if any(p.requires_grad or p.grad is not None or p._version!=versions[n] for n,p in victim.named_parameters()):
        raise ValueError("IRRA frozen state changed")
    if any(p.requires_grad or p.grad is not None for p in reader.parameters()):
        raise ValueError("reader not frozen")
    if not all(old.torch.equal(v.detach().cpu(),reader_state[k]) for k,v in reader.state_dict().items()):
        raise ValueError("reader state changed")
    verify(hashes);verify(own);deps.assert_formal_clean()
    target=3 if args.mode=="smoke" else 100
    manifest=stratify(deps.rows(cache_file),validation,target)
    manifest.update(score_cache_path=str(cache_file.resolve()),score_cache_sha256=digest(cache_file),
                    score_protocol_path=str((out/"protocol.json").resolve()),score_protocol_sha256=digest(out/"protocol.json"),
                    split_sha256=protocol["split_sha256"],full_preparation=args.mode=="prepare")
    save(out/"selection_manifest.json",manifest)
    audit={"caption_count":len(records),"new_clean_retrieval_backwards":new_backwards,"reused_verified_scores":reused,
           "each_new_caption_one_clean_backward":True,"score_cache_sha256":digest(cache_file),
           "instance_status_counts":dict(status),"IRRA_reader_frozen":True,"zero_exact":True,
           "source_cache_SHA_unchanged":True,"tracked_staged_diff_empty":True,"seconds":time.time()-started,
           "attack_performed":False,"selection_case_count":manifest["case_count"],
           "natural_k_distribution":manifest["natural_k_distribution"],"bin_selected":manifest["bin_selected"]}
    save(out/"audit.json",audit);save(out/"completed.json",audit)
    print("SCORE_PREPARATION_COMPLETE",json.dumps(audit),flush=True)
if __name__=="__main__":main()
