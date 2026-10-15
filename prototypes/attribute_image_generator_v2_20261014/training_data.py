"""G-v2新训练缓存读取与样本核验；不导入/读取G-v1缓存或前端。"""
from pathlib import Path
import torch
import cache_io as io
from condition_schema import IDENTITY,validate

RAW=Path("/home/lzf/TBPS/Datasets/CUHK-PEDES/reid_raw.json")
IMAGES=RAW.parent/"imgs"
SPLIT=Path("/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_unit_v4_learned_reader_20261003/outputs/split.json")
CACHE_FILES=("prepare_full.py","prep_text_frontend.py","training_data.py","condition_schema.py",
             "cache_io.py","irra_image_adapter.py","budget_projection.py")

def file_contract(names,root=None):
    root=io.ROOT if root is None else Path(root)
    return {str(root/name):io.sha(root/name) for name in names}

def identity(entry):
    return {k:entry[k] for k in (*IDENTITY,"K")}

def publish_manifest(output,entries,protocol):
    out=io.safe(output)
    if [identity(e) for e in entries]!=protocol["samples"]: raise ValueError("cache coverage mismatch")
    counts={r:sum(e["role"]==r for e in entries) for r in ("train","validation")}
    m={"schema":"gv2-training-cache-2","scope":protocol["scope"],"counts":counts,
       "entries":entries,"protocol_path":str(out/"protocol.json"),"protocol_sha256":io.sha(out/"protocol.json")}
    io.save_json(out/"manifest.json",m)
    io.save_json(out/"completed.json",{"method_audit_passed":True,"manifest_sha256":io.sha(out/"manifest.json")})
    return checked_cache(out)

def checked_cache(path,require_full=False):
    out=Path(path); done=io.load(out/"completed.json")
    io.verify({str(out/"manifest.json"):done["manifest_sha256"]})
    m=io.load(out/"manifest.json")
    io.verify({m["protocol_path"]:m["protocol_sha256"]})
    p=io.load(m["protocol_path"])
    if not done["method_audit_passed"] or m["schema"]!="gv2-training-cache-2": raise ValueError("invalid completed cache")
    # 上游G-v1出处SHA仅留作历史provenance；训练只验证本G-v2缓存和相关producer代码。
    if p["schema"]!="gv2-full-prepare-protocol-2" or p["target_counts"]!={"train":2048,"validation":128}:
        raise ValueError("unsupported cache protocol")
    io.verify(p["producer_sha256"]);io.verify(p["dataset_sha256"])
    if m["scope"]!=p["scope"] or [identity(e) for e in m["entries"]]!=p["samples"]:
        raise ValueError("manifest/protocol mismatch")
    counts={r:sum(e["role"]==r for e in m["entries"]) for r in ("train","validation")}
    if counts!=m["counts"] or len({e["row_id"] for e in m["entries"]})!=len(m["entries"]):
        raise ValueError("cache counts/unique identity")
    pids={r:{e["pid"] for e in m["entries"] if e["role"]==r} for r in counts}
    if pids["train"]&pids["validation"]: raise ValueError("PID leakage")
    if require_full and (m["scope"]!="full-2176" or counts!={"train":2048,"validation":128}):
        raise ValueError("formal training requires completed original 2048+128 cache")
    for e in m["entries"]: io.read_entry(e,io.digest(p))
    m["protocol"]=p
    return m

def check_pair(entry,condition,meta,raw,split):
    validate(condition,entry)
    prefix,row,j=entry["row_id"].split(":")
    if prefix!="cuhk" or int(j)!=entry["caption_index"]: raise ValueError("caption index mismatch")
    source=raw[int(row)]
    if (source["split"],source["id"],source["file_path"],source["captions"][int(j)])!=(
        entry["split"],entry["pid"],entry["image_rel_path"],entry["caption"]):
        raise ValueError("raw image/caption/PID pairing mismatch")
    members={split["records"][i]["row_id"] for i in split["train_indices" if entry["role"]=="train" else "validation_indices"]}
    if entry["row_id"] not in members: raise ValueError("internal role mismatch")
    path=(IMAGES/entry["image_rel_path"]).resolve()
    if not path.is_relative_to(IMAGES.resolve()): raise ValueError("image path escape")
    io.verify({str(path):meta["source_image_sha256"]})
    return path

def make_loader(manifest,device="cuda"):
    from PIL import Image
    from irra_image_adapter import pixels_from_pil
    raw=io.load(RAW); split=io.load(SPLIT)
    def load(entry):
        c,meta,_=io.read_entry(entry,io.digest(manifest["protocol"]))
        path=check_pair(entry,c,meta,raw,split)
        with Image.open(path) as im: image=pixels_from_pil(im)[None].to(device)
        return image,c
    return load
