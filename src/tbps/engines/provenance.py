"""评测输入、攻击来源和内存payload绑定；不持有策略/反馈/身份标签接口。"""
from dataclasses import dataclass, asdict
import hashlib
from ..artifacts import ArtifactError, checked, digest, json_bytes
from ..contracts import AttackBudget, FileRef, sha256, sequence
from ..models.base import InputBinding, RepresentationBatch

def payload_digest(value):
    """支持Tensor及原生结构；不假设global embedding、维度或Tokenizer。"""
    if isinstance(value,dict):
        if any(type(k)is not str for k in value):raise ArtifactError("native payload keys must be strings")
        return digest({"dict":{k:payload_digest(v) for k,v in sorted(value.items())}})
    if type(value) in (tuple,list):return digest({type(value).__name__:[payload_digest(v) for v in value]})
    if value is None or type(value) in (str,int,float,bool):return digest({"scalar":value})
    import torch
    if not isinstance(value,torch.Tensor) or value.device.type!="cpu" or value.requires_grad:
        raise ArtifactError("unsupported native payload or non-CPU/gradient Tensor")
    x=value.detach().contiguous()
    h=hashlib.sha256(json_bytes({"dtype":str(x.dtype),"shape":list(x.shape)}))
    h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

@dataclass(frozen=True,slots=True)
class AttackOrigin:
    attack_id: str
    checkpoint: FileRef
    protocol: FileRef
    budget: AttackBudget
    surrogate_sources_sha256: str
    def __post_init__(self):
        if self.attack_id not in ("gv2","hv1"):raise ArtifactError("unsupported attack component")
        if type(self.checkpoint)is not FileRef or type(self.protocol)is not FileRef or type(self.budget)is not AttackBudget:
            raise ArtifactError("typed attack checkpoint/protocol/budget required")
        sha256(self.surrogate_sources_sha256)
        if (self.attack_id=="gv2" and (self.budget.image is None or self.budget.text is not None)) or (
            self.attack_id=="hv1" and (self.budget.text is None or self.budget.image is not None)):
            raise ArtifactError("attack modality/budget conflict")
    def verify(self):checked(self.checkpoint);checked(self.protocol)

@dataclass(frozen=True,slots=True)
class BoundRepresentation:
    batch: RepresentationBatch
    inputs: InputBinding
    payload_sha256: str
    cache_sources: tuple[FileRef,...]
    origin: AttackOrigin | None = None
    original_sha256: tuple[str,...] = ()
    def __post_init__(self):
        if type(self.batch)is not RepresentationBatch or type(self.inputs)is not InputBinding:
            raise ArtifactError("typed native representation/input binding required")
        sha256(self.payload_sha256);sequence(self.cache_sources,FileRef,"representation cache sources")
        if not self.cache_sources:raise ArtifactError("explicit feature/cache provenance required")
        if self.batch.space_id!=self.inputs.model_space_id or self.batch.modality!=self.inputs.modality:
            raise ArtifactError("representation space/modality mismatch")
        if self.origin is not None:
            if type(self.origin)is not AttackOrigin:raise ArtifactError("typed attack origin")
            if len(self.original_sha256)!=len(self.inputs.ordered_ids):raise ArtifactError("original input binding required")
            for h in self.original_sha256:sha256(h)
        elif self.original_sha256:raise ArtifactError("clean input cannot claim attack originals")
        self.verify()
    def verify(self):
        if payload_digest(self.batch.payload)!=self.payload_sha256:raise ArtifactError("representation payload tampered")
        for ref in self.cache_sources:checked(ref)
        if self.origin:self.origin.verify()
    def descriptor(self):
        return {"inputs":asdict(self.inputs),"payload_sha256":self.payload_sha256,
                       "input_content_domain":("caption_utf8" if self.inputs.modality=="text" else
                                                "adversarial_rgb_tensor" if self.origin else "original_image_file"),
                       "cache_sources":[asdict(r) for r in self.cache_sources],
                       "origin":asdict(self.origin) if self.origin else None,
                       "original_sha256":self.original_sha256}
    @property
    def id(self):return digest(self.descriptor())
