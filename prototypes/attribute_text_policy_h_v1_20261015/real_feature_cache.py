"""独立Real训练特征缓存；不读取Mock/Test缓存，失败单位拒绝自动重复计算。"""
from dataclasses import asdict
from pathlib import Path
import os
import torch
from PIL import Image
from phase2_sources import ROOT,digest,sha,read_json,new_json
from rewards import frozen_features
from feedback_backend import BoundReward,validate_variant
from phase3_inputs import safe_output,import_gv2

def fsync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def feature_stats(features):
    return {k:{"shape":list(x.shape),"norm_min":float(x.norm(dim=1).min()),
               "norm_max":float(x.norm(dim=1).max()),"finite":bool(torch.isfinite(x).all()),
               "detached":not x.requires_grad}
            for k,x in features.items()}

class RealFeatureCache:
    def __init__(self,root,catalog,protocol,frontend,probe):
        self.root=Path(root)
        self.catalog,self.protocol,self.frontend,self.probe=catalog,protocol,frontend,probe
        if protocol.get("execution_backend") not in ("official_irra_real","cpu_test_fixture"):
            raise ValueError("real cache cannot use Phase 2 Mock/Test backend")
        if protocol["catalog_sha256"]!=catalog.fingerprint:
            raise ValueError("real cache training catalog mismatch")
        if protocol["execution_backend"]=="official_irra_real":
            safe_output(self.root)
        self.hash=digest(protocol)
        self.allowed={kind:{s["row_id"]:s for s in protocol[key]}
                      for kind,key in (("image","gallery"),("text","queries"))}
        self.root.mkdir(parents=True,exist_ok=True)
        p=self.root/"protocol.json"
        if p.exists():
            if digest(read_json(p))!=self.hash:
                raise ValueError("real cache protocol/source mismatch")
        else:new_json(p,protocol)

    def identity(self,kind,sample):
        self.catalog.require_train(sample)
        if self.allowed[kind].get(sample.row_id)!=asdict(sample):
            raise ValueError("identity is outside fixed train Smoke selection")
        return asdict(sample)

    def path(self,kind,identity,texts=None):
        key=digest({"identity":identity,"texts":texts})
        family=self.root/kind
        family.mkdir(exist_ok=True)
        return family/key

    def begin(self,unit,identity,extra):
        pending=unit.with_name(unit.name+".pending")
        pending.mkdir(exist_ok=False)
        new_json(pending/"intent.json",{"protocol_sha256":self.hash,"identity":identity,**extra})
        fsync_dir(pending);fsync_dir(pending.parent)
        return pending

    def publish(self,unit,pending,identity,data,extra):
        for x in data.values():frozen_features(x)
        torch.save({k:x.detach().cpu().clone() for k,x in data.items()},pending/"features.pt")
        with (pending/"features.pt").open("rb") as stream:os.fsync(stream.fileno())
        new_json(pending/"metadata.json",{"protocol_sha256":self.hash,"identity":identity,
                    "features_sha256":sha(pending/"features.pt"),"feature_stats":feature_stats(data),**extra})
        fsync_dir(pending)
        os.rename(pending,unit);fsync_dir(unit.parent)
        new_json(unit.with_name(unit.name+".receipt.json"),{"protocol_sha256":self.hash,
                 "metadata_sha256":sha(unit/"metadata.json"),"features_sha256":sha(unit/"features.pt")})
        fsync_dir(unit.parent)

    def load(self,unit,identity,extra,shape,keys):
        receipt=unit.with_name(unit.name+".receipt.json")
        if not unit.is_dir() or not receipt.exists():
            raise ValueError("missing/uncommitted real feature unit; preserve receipts and stop")
        md=read_json(unit/"metadata.json")
        r=read_json(receipt)
        if (md["identity"]!=identity or md["protocol_sha256"]!=self.hash
                or any(md.get(k)!=v for k,v in extra.items())
                or r!={"protocol_sha256":self.hash,"metadata_sha256":sha(unit/"metadata.json"),
                       "features_sha256":sha(unit/"features.pt")}
                or md["features_sha256"]!=r["features_sha256"]):
            raise ValueError("real cache identity/source/SHA mismatch")
        data=torch.load(unit/"features.pt",map_location="cpu",weights_only=True)
        if set(data)!=set(keys):raise ValueError("real feature schema mismatch")
        for key in data:
            frozen_features(data[key],torch.Size(shape))
            data[key]=data[key].clone()
        if md["feature_stats"]!=feature_stats(data):
            raise ValueError("cached feature diagnostics changed")
        return data,md

    def image(self,sample,runtime=None,loader=None):
        identity=self.identity("image",sample)
        unit=self.path("image",identity)
        extra={"kind":"image","execution_backend":self.protocol["execution_backend"],
               "execution":{"G_samples":1,"IRRA_clean_samples":1,"IRRA_adv_samples":1,"IRRA_text_samples":0}}
        if unit.exists() or unit.with_name(unit.name+".receipt.json").exists() or runtime is None:
            data,md=self.load(unit,identity,extra,(1,512),("clean","adversarial"))
        else:
            if runtime.kind!=self.protocol["execution_backend"]:
                raise ValueError("runtime/cache backend mismatch")
            path=self.catalog.verify_image(sample)
            if runtime.kind=="official_irra_real" and loader is not None:
                raise ValueError("real image preprocessing cannot be overridden")
            pending=self.begin(unit,identity,extra)
            if loader is None:
                _,adapter=import_gv2()
                with Image.open(path) as image:pixels=adapter.pixels_from_pil(image)[None]
            else:pixels=loader(path)
            data,info=runtime.image_pair(pixels)
            if {k:info[k] for k in extra["execution"]}!=extra["execution"] or not 0<=info["linf"]<=8/255:
                raise ValueError("single image real forward/budget receipt mismatch")
            self.publish(unit,pending,identity,data,{**extra,"linf":info["linf"]})
            data,md=self.load(unit,identity,extra,(1,512),("clean","adversarial"))
        if not 0<=md["linf"]<=8/255:raise ValueError("cached image budget violation")
        return data,md

    def text(self,query,texts,runtime=None):
        identity=self.identity("text",query)
        texts=tuple(texts)
        if not 1<=len(texts)<=3 or texts[0]!=query.caption:
            raise ValueError("original and <=2 H variants per verified training query required")
        example=self.frontend(query.caption)
        for variant in texts[1:]:
            # 这里检查Span/白名单上限；具体H预测K由未修改BoundReward再次严格检查。
            validate_variant(example,variant,len(example.attributes),self.probe)
        reports=[self.probe.inspect(t) for t in texts]
        if any(r.truncated for r in reports):raise ValueError("cached text truncates")
        extra={"kind":"text","execution_backend":self.protocol["execution_backend"],
               "texts":list(texts),"token_sha256":digest([r.model_tokens for r in reports]),
               "execution":{"G_samples":0,"IRRA_clean_samples":0,"IRRA_adv_samples":0,"IRRA_text_samples":len(texts)}}
        unit=self.path("text",identity,list(texts))
        if not unit.exists() and not unit.with_name(unit.name+".receipt.json").exists() and runtime is not None:
            if runtime.kind!=self.protocol["execution_backend"]:
                raise ValueError("runtime/cache backend mismatch")
            pending=self.begin(unit,identity,extra)
            feature=runtime.encode_text(texts)
            self.publish(unit,pending,identity,{"text":feature},extra)
        data,md=self.load(unit,identity,extra,(len(texts),512),("text",))
        return data["text"],md

    def gather(self,samples):
        units=[self.image(s)[0] for s in samples]
        return {k:torch.cat([x[k] for x in units]).clone() for k in ("clean","adversarial")}

    def bind_reward(self,query,samples,runtime,budget_k):
        owner=self
        class TextContext:
            probe=owner.probe
            def assert_frozen(self):runtime.assert_frozen()
            def encode_text(self,texts):return owner.text(query,texts,runtime)[0]
        return BoundReward(self.catalog,query,samples,self.gather(samples),TextContext(),self.frontend,budget_k)

    def evidence(self):
        return {str(p):sha(p) for p in self.root.rglob("*") if p.is_file()}
