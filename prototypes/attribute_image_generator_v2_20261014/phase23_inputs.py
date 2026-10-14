"""固定5条真实监督来源审计；CPU-only，无前端重算。"""
from pathlib import Path
import torch
import cache_io as io
from condition_schema import IDENTITY
from irra_image_adapter import OFFICIAL,CONFIG,CHECKPOINT,CLIP_CACHE,CHECKPOINT_SHA

CACHE=Path("/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_image_generator_v2_20261014/outputs/smoke")
RAW=Path("/home/lzf/TBPS/Datasets/CUHK-PEDES/reid_raw.json")
IMAGES=RAW.parent/"imgs"
SPLIT=Path("/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_unit_v4_learned_reader_20261003/outputs/split.json")
GALLERY=Path("/home/lzf/ldx/outputs/idea-TBPS-test1/stage03-v2-smoke-20261001/clean-train-gallery.pt")
EXPECTED=[("cuhk:26530:0","train",0),("cuhk:26747:0","train",1),("cuhk:9955:0","train",3),
          ("cuhk:26196:0","train",7),("cuhk:17663:0","validation",4)]

def check_sample(entry,value,meta,protocol,raw,split,gallery):
    if meta["protocol_hash"]!=io.digest(protocol): raise ValueError("sample cache protocol mismatch")
    if any(entry[k]!=value[k] or meta[k]!=value[k] for k in IDENTITY): raise ValueError("complete identity mismatch")
    _,row,caption_index=entry["row_id"].split(":")
    item=raw[int(row)]
    actual=(item["split"],item["id"],item["file_path"],item["captions"][int(caption_index)])
    if actual!=(entry["split"],entry["pid"],entry["image_rel_path"],entry["caption"]) or entry["caption_index"]!=int(caption_index):
        raise ValueError("raw image/caption/PID mismatch")
    member={split["records"][i]["row_id"] for i in split["train_indices" if entry["role"]=="train" else "validation_indices"]}
    if entry["row_id"] not in member: raise ValueError("internal role membership mismatch")
    source=meta["source_entry"]; record=meta["source_score_record"]
    if (record["row_id"],record["pid"],record["caption"],record["paired_image"])!=(
        entry["row_id"],entry["pid"],entry["caption"],entry["image_rel_path"]):
        raise ValueError("original score identity mismatch")
    if (source["row_id"],source["pid"],source["image"],source["caption"],source["K"])!=(
        entry["row_id"],entry["pid"],entry["image_rel_path"],entry["caption"],entry["K"]):
        raise ValueError("source entry identity mismatch")
    if meta["source_image_sha256"]!=source["image_sha256"]: raise ValueError("image SHA provenance mismatch")
    image=(IMAGES/entry["image_rel_path"]).resolve()
    if not image.is_relative_to(IMAGES.resolve()): raise ValueError("image path escapes dataset")
    io.verify({str(image):meta["source_image_sha256"]})
    index=record["paired_gallery_index"]
    if gallery["spec"]["gallery_paths"][index]!=entry["image_rel_path"]: raise ValueError("paired gallery index mismatch")
    if not record["zero_intervention_exact"] or record["retrieval_backward_count"]!=1:
        raise ValueError("original score provenance mismatch")
    return image,index

def audit_inputs():
    manifest=io.checked_cache(CACHE)
    protocol=io.load(manifest["protocol_path"])
    if protocol["scope"]!="smoke-five-samples" or protocol["test_used"] or protocol["source_counts"]!={"train":2048,"validation":128}:
        raise ValueError("wrong Phase 1 cache scope")
    if [(e["row_id"],e["role"],e["K"]) for e in manifest["entries"]]!=EXPECTED:
        raise ValueError("exact five original smoke samples required")
    raw=io.load(RAW); split=io.load(SPLIT)
    # gallery SHA已包含在checked_cache来源锁；只读取，不重编码完整图库。
    gallery=torch.load(GALLERY,map_location="cpu",weights_only=False)
    spec=gallery["spec"]
    if (spec["checkpoint_sha256"]!=CHECKPOINT_SHA or spec["config_sha256"]!=io.sha(CONFIG)
        or spec["annotation_sha256"]!=io.sha(RAW) or spec["torch_version"]!=str(torch.__version__)
        or spec["image_size"]!=[384,128] or gallery["features"].shape!=(34054,512)):
        raise ValueError("official retrieval space provenance mismatch")
    rows=[]
    for entry in manifest["entries"]:
        value,meta,_=io.read_entry(entry,io.digest(protocol))
        image,index=check_sample(entry,value,meta,protocol,raw,split,gallery)
        rows.append({"entry":entry,"condition":value,"image_path":str(image),
                     "clean_gallery_feature":gallery["features"][index:index+1].clone()})
    locks=dict(protocol["source_sha256"])
    locks.update({str(p):io.sha(p) for p in [*OFFICIAL.rglob("*.py"),CONFIG,CHECKPOINT,CLIP_CACHE/"ViT-B-16.pt",
        CACHE/"manifest.json",CACHE/"completed.json",CACHE/"protocol.json"]})
    return rows,locks,{"schema":"gv2-phase23-input-audit-1","count":5,"samples":EXPECTED,
        "source_sha_count":len(locks),"official_checkpoint_sha256":CHECKPOINT_SHA,
        "historical_gallery_precision":spec["precision"],"historical_gallery_batch_size":spec["image_batch_size"],
        "z_t_usage":"Global supervision only; no G input","new_score_TopK_JVP":0}
