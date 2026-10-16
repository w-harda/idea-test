"""紧凑Storage、同目录原子回执；Train/Validation特征绝不混用。"""
from dataclasses import asdict
from pathlib import Path
import os
import torch
from phase2_sources import digest, sha, read_json
from rewards import frozen_features
from real_feature_cache import fsync_dir
from .common import immutable_json
from .data import members

class GalleryStore:
    def __init__(self, root, catalog, protocol, create=False):
        self.root, self.catalog, self.protocol = Path(root), catalog, protocol
        self.hash = digest(protocol)
        self.role = protocol["role"]
        self.samples = members(catalog, self.role)
        if (protocol["schema"]!="hv1-formal-gallery-1"
                or protocol["execution_backend"] not in ("official_irra_real","cpu_test_fixture")
                or protocol["catalog_sha256"]!=catalog.fingerprint
                or protocol["identities"]!=[asdict(s) for s in self.samples]):
            raise ValueError("formal gallery identity/backend/role mismatch")
        self.allowed = {s.row_id: asdict(s) for s in self.samples}
        if create: immutable_json(self.root/"protocol.json", protocol)
        if read_json(self.root/"protocol.json") != protocol:
            raise ValueError("gallery protocol/source mismatch")

    def path(self, sample, kind="image"):
        if self.allowed.get(sample.row_id)!=asdict(sample):
            raise ValueError("sample belongs to another role/source")
        if kind not in ("image","original_text") or (kind=="original_text" and self.role!="validation"):
            raise ValueError("invalid formal cache family")
        return self.root/kind/digest(asdict(sample))

    def load(self, sample, kind="image"):
        unit = self.path(sample,kind)
        md,receipt = read_json(unit/"metadata.json"),read_json(unit/"receipt.json")
        if (md["protocol_sha256"]!=self.hash or md["identity"]!=asdict(sample) or md["kind"]!=kind
                or md["features_sha256"]!=sha(unit/"features.pt")
                or receipt!={"metadata_sha256":sha(unit/"metadata.json"),"protocol_sha256":self.hash,
                             "features_sha256":sha(unit/"features.pt")}):
            raise ValueError("feature/identity/receipt SHA mismatch")
        data = torch.load(unit/"features.pt",map_location="cpu",weights_only=True)
        keys = {"clean","adversarial"} if kind=="image" else {"text"}
        if set(data)!=keys: raise ValueError("feature schema mismatch")
        for key,x in data.items():
            frozen_features(x,torch.Size((1,512)))
            data[key]=x.clone()  # 历史storage再大也不持有其view
        if kind=="image":
            if not 0<=md["linf"]<=8/255 or md["execution"]!={"G":1,"clean":1,"adv":1,"text":0}:
                raise ValueError("image budget/forward receipt mismatch")
        elif md["execution"]!={"G":0,"clean":0,"adv":0,"text":1}:
            raise ValueError("validation original text receipt mismatch")
        return data,md

    def prepare(self, sample, produce, kind="image"):
        unit = self.path(sample,kind)
        if unit.exists(): return self.load(sample,kind)
        work = unit.with_name(unit.name+".pending")
        work.mkdir(parents=True,exist_ok=False)  # 残留意图禁止自动重复G
        immutable_json(work/"intent.json",{"protocol_sha256":self.hash,"identity":asdict(sample),"kind":kind})
        fsync_dir(work);fsync_dir(work.parent)
        data,info = produce()
        keys = {"clean","adversarial"} if kind=="image" else {"text"}
        if set(data)!=keys: raise ValueError("producer feature keys mismatch")
        for x in data.values(): frozen_features(x,torch.Size((1,512)))
        torch.save({k:v.detach().cpu().clone() for k,v in data.items()},work/"features.pt")
        with (work/"features.pt").open("rb") as f: os.fsync(f.fileno())
        immutable_json(work/"metadata.json",{"protocol_sha256":self.hash,"identity":asdict(sample),
                       "kind":kind,"features_sha256":sha(work/"features.pt"),**info})
        immutable_json(work/"receipt.json",{"protocol_sha256":self.hash,
                       "features_sha256":sha(work/"features.pt"),"metadata_sha256":sha(work/"metadata.json")})
        # 在发布前执行同样的内容检查；临时目录不参与任何奖励。
        if kind=="image" and (not 0<=info["linf"]<=8/255
                or info["execution"]!={"G":1,"clean":1,"adv":1,"text":0}):
            raise ValueError("producer budget/count mismatch")
        if kind=="original_text" and info["execution"]!={"G":0,"clean":0,"adv":0,"text":1}:
            raise ValueError("producer original text count mismatch")
        fsync_dir(work)
        os.rename(work,unit);fsync_dir(unit.parent)  # features/metadata/receipt同时发布
        return self.load(sample,kind)

    def audit(self, require_complete=True):
        pending = sorted(str(p) for p in self.root.rglob("*.pending"))
        if pending: raise ValueError("uncommitted gallery intent; preserve evidence: "+str(pending))
        result = {"role":self.role,"protocol_sha256":self.hash,"image":0,"original_text":0,"files":{}}
        for kind in ("image","original_text") if self.role=="validation" else ("image",):
            expected = {self.path(s,kind).name for s in self.samples}
            family = self.root/kind
            actual = {p.name for p in family.iterdir() if p.is_dir()} if family.exists() else set()
            if actual-expected: raise ValueError("unexpected cache identities")
            for sample in self.samples:
                unit = self.path(sample,kind)
                if unit.exists():
                    self.load(sample,kind)
                    result[kind]+=1
                    for name in ("features.pt","metadata.json","receipt.json","intent.json"):
                        result["files"][str(unit/name)] = sha(unit/name)
                elif require_complete: raise ValueError("missing cache sample: "+sample.row_id+" "+kind)
        result["content_sha256"]=digest(result["files"])
        if (self.root/"completed.json").exists() and read_json(self.root/"completed.json")!=result:
            raise ValueError("gallery completion receipt mismatch")
        return result

    def gather(self):
        data=[self.load(s)[0] for s in self.samples]
        return {k:torch.cat([u[k] for u in data]).clone() for k in ("clean","adversarial")}

    def originals(self):
        if self.role!="validation": raise ValueError("validation-only original text cache")
        return torch.cat([self.load(s,"original_text")[0]["text"] for s in self.samples]).clone()
