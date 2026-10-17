"""原生结构化表示缓存：Tensor/局部Token按模型验证，复用不可变ArtifactStore。"""
from dataclasses import asdict
import io
from tbps.artifacts import json_bytes,digest,read_json,ArtifactError
from tbps.models.base import RepresentationBatch
from tbps.engines.provenance import BoundRepresentation,payload_digest
from tbps.engines.materialized_cache import ref,FAMILIES

def compact(x):
    if isinstance(x,dict):return {k:compact(v) for k,v in x.items()}
    return x.detach().cpu().clone()

class NativeFeatureCache:
    def __init__(self,store,protocol,space,validate,*,chunk=16):
        self.store,self.protocol,self.space,self.validate,self.chunk=store,protocol,space,validate,chunk
        self.protocol_ref=ref(store.root/"protocol/manifest.json")
    def key(self,family,start):
        if family not in FAMILIES or start%self.chunk:raise ArtifactError("native cache family/offset")
        return family+"-"+str(start//self.chunk).zfill(6)
    def provenance(self,binding,start,end):
        return {"kind":"native_materialized_representation","protocol_sha256":digest(self.protocol),
                "binding":asdict(binding),"start":start,"end":end}
    def publish(self,family,binding,start,end,x):
        import torch
        x=compact(x);self.validate(x,end-start,binding.modality)
        stream=io.BytesIO();torch.save(x,stream)
        self.store.publish(self.key(family,start),{"features.pt":stream.getvalue(),
            "details.json":json_bytes({"payload_sha256":payload_digest(x)})},
            self.provenance(binding,start,end),sources=(self.protocol_ref,))
    def load(self,family,binding,start,end):
        import torch
        key=self.key(family,start);self.store.verify(key,expected_provenance=self.provenance(binding,start,end))
        raw=torch.load(self.store.root/key/"features.pt",map_location="cpu",weights_only=True)
        x=compact(raw);del raw;self.validate(x,end-start,binding.modality)
        if payload_digest(x)!=read_json(self.store.root/key/"details.json")["payload_sha256"]:
            raise ArtifactError("native cache content SHA")
        return x
    def gather(self,family,binding,*,origin=None,originals=()):
        import torch
        parts=[];refs=[]
        for start in range(0,len(binding.ordered_ids),self.chunk):
            parts.append(self.load(family,binding,start,min(start+self.chunk,len(binding.ordered_ids))))
            refs.append(ref(self.store.root/self.key(family,start)/"manifest.json"))
        payload={k:torch.cat([p[k] for p in parts],0) for k in parts[0]}
        del parts
        return BoundRepresentation(RepresentationBatch(payload,self.space.id,binding.modality),binding,
            payload_digest(payload),tuple(refs),origin,originals)
