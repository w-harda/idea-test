"""Phase5紧凑特征与动作回执；已提交不可变，显式保留未提交尝试。"""
from pathlib import Path
import os
import uuid
import torch
from phase2_sources import digest,sha,read_json
from real_feature_cache import fsync_dir
from .common import immutable_json,legacy
FAMILIES=("clean_gallery","image_gallery","clean_text","adv_text")
class Store:
    def __init__(self,root,protocol,create=False):
        self.root=Path(root);self.protocol=protocol;self.hash=digest(protocol)
        if create:immutable_json(self.root/"protocol.json",protocol)
        if read_json(self.root/"protocol.json")!=protocol:raise ValueError("immutable Phase5 protocol mismatch")
    def parts(self,family):
        if family not in FAMILIES:raise ValueError("unknown feature family")
        rows=self.protocol["gallery"] if family.endswith("gallery") else self.protocol["queries"]
        size=64 if family.endswith("gallery") else self.protocol["text_batch"]
        return [(i,rows[s:s+size]) for i,s in enumerate(range(0,len(rows),size))]
    def path(self,family,index):
        if family not in FAMILIES or type(index)is not int or index<0:raise ValueError("unsafe unit")
        return self.root/"features"/family/f"{index:06d}"
    def load(self,family,index,rows):
        unit=self.path(family,index)
        if not unit.exists():return None
        md,r=read_json(unit/"metadata.json"),read_json(unit/"receipt.json")
        if read_json(unit/"intent.json")!={"protocol_sha256":self.hash,"family":family,"index":index,"rows":rows}:
            raise ValueError("published computation intent mismatch")
        if (md["protocol_sha256"]!=self.hash or md["family"]!=family or md["index"]!=index or md["rows"]!=rows
            or md["feature_sha256"]!=sha(unit/"features.pt")
            or r!={"protocol_sha256":self.hash,"metadata_sha256":sha(unit/"metadata.json"),
                   "feature_sha256":md["feature_sha256"]}):
            raise ValueError("feature/identity/receipt SHA mismatch")
        x=torch.load(unit/"features.pt",map_location="cpu",weights_only=True).clone()
        legacy("eval_metrics").normalized(x,len(rows))
        if md["stats"]["G_new_samples"]!=0 or md["stats"]["IRRA_image_new_samples"]!=0:
            raise ValueError("Phase5 cannot regenerate image attacks")
        stats=md["stats"]
        if family=="adv_text":
            actions=stats["actions"]
            if (len(actions)!=len(rows) or stats["IRRA_text_samples"]!=sum(a["adversarial"]!=r["caption"] for a,r in zip(actions,rows))
                    or stats["H_forward_samples"]!=sum(a["H_forward"] for a in actions)
                    or any(a["H_forward"] not in (0,1) for a in actions)):
                raise ValueError("committed text action/forward accounting mismatch")
        elif stats["IRRA_text_samples"]!=0 or stats.get("reference_reused") is not True:
            raise ValueError("reference batches cannot have fresh model forwards")
        return x,md
    def prepare(self,family,index,rows,produce):
        saved=self.load(family,index,rows)
        if saved is not None:return saved
        unit=self.path(family,index);work=unit.with_name(unit.name+".pending")
        work.mkdir(parents=True,exist_ok=False)
        intent={"protocol_sha256":self.hash,"family":family,"index":index,"rows":rows}
        immutable_json(work/"intent.json",intent);fsync_dir(work);fsync_dir(work.parent)
        x,stats=produce()
        legacy("eval_metrics").normalized(x,len(rows))
        if stats["G_new_samples"] or stats["IRRA_image_new_samples"]:raise ValueError("image re-generation forbidden")
        torch.save(x.detach().cpu().clone(),work/"features.pt")
        with (work/"features.pt").open("rb") as stream:os.fsync(stream.fileno())
        immutable_json(work/"metadata.json",{**intent,"feature_sha256":sha(work/"features.pt"),"stats":stats})
        immutable_json(work/"receipt.json",{"protocol_sha256":self.hash,"metadata_sha256":sha(work/"metadata.json"),
                                           "feature_sha256":sha(work/"features.pt")})
        fsync_dir(work);os.rename(work,unit);fsync_dir(unit.parent)
        return self.load(family,index,rows)
    def check_pending(self,recover=False):
        expected={self.path(k,i).with_name(f"{i:06d}.pending"):(k,i,rows)
                  for k in FAMILIES for i,rows in self.parts(k)}
        for work in sorted((self.root/"features").rglob("*.pending")):
            if work not in expected:raise ValueError("unknown pending directory")
            family,index,rows=expected[work]
            if read_json(work/"intent.json")!={"protocol_sha256":self.hash,"family":family,"index":index,"rows":rows}:
                raise ValueError("unknown pending source/identity")
            if self.path(family,index).exists():raise ValueError("pending conflicts with committed unit")
            if not recover:raise ValueError("uncommitted batch retained; user may explicitly --recover-pending")
            proof={str(p.relative_to(work)):sha(p) for p in work.rglob("*") if p.is_file()}
            target=self.root/"abandoned"/uuid.uuid4().hex;target.parent.mkdir(exist_ok=True)
            os.rename(work,target);fsync_dir(work.parent);fsync_dir(target.parent)
            immutable_json(target/"recovery.json",{"protocol_sha256":self.hash,"files_sha256":proof,
                 "reason":"explicit stateless text/reference recomputation; committed batches never repeated; G/image=0"})
    def gather(self,family):
        xs=[];stats=[];proof={}
        for index,rows in self.parts(family):
            saved=self.load(family,index,rows)
            if saved is None:raise ValueError("missing feature unit: "+family+str(index))
            x,md=saved;xs.append(x);stats.append(md["stats"])
            for p in self.path(family,index).iterdir():
                if p.is_file():proof[str(p.relative_to(self.root))]=sha(p)
        return torch.cat(xs).clone(),stats,proof
    def audit(self,complete=True):
        self.check_pending()
        actual={}
        for family in FAMILIES:
            expected={self.path(family,i).name for i,_ in self.parts(family)}
            folder=self.root/"features"/family
            units={p.name for p in folder.iterdir() if p.is_dir()} if folder.exists() else set()
            if units-expected:raise ValueError("unrecognized feature batch")
            actual[family]=len(units)
            for index,rows in self.parts(family):
                saved=self.load(family,index,rows)
                if saved is None and complete:raise ValueError("incomplete "+family)
        return actual
