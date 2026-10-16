"""公开输入、独立身份绑定和攻击产物；没有victim/奖励/Optimizer通道。"""
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
import hashlib
import json
from ..artifacts import ArtifactError, digest, json_bytes
from ..contracts import AttackSpec, AttackBudget, FileRef, PublicImageInput, PublicTextInput, sha256, ACCESS

def need(ok, message):
    if not ok: raise ArtifactError(message)

@dataclass(frozen=True, slots=True)
class PluginInfo:
    spec: AttackSpec
    offline_preparation_access: frozenset[str]
    online_access: frozenset[str]
    surrogate_for_generation: bool = False
    gradients_for_generation: bool = False
    rewards_for_generation: bool = False
    offline_preparation_executable: bool = False
    training_executable: bool = False
    equivalence_status: str = "CPU_FIXTURE_ONLY_REAL_PENDING"
    def __post_init__(self):
        need(type(self.spec) is AttackSpec, "typed AttackSpec required")
        need(type(self.offline_preparation_access) is frozenset and type(self.online_access) is frozenset,
             "immutable phase access required")
        need(self.online_access == self.spec.attack_requirements, "online declaration/spec mismatch")
        need(self.offline_preparation_access<=ACCESS, "unknown offline preparation access")
        need(all(type(getattr(self,key)) is bool for key in
                 ("surrogate_for_generation","gradients_for_generation","rewards_for_generation",
                  "offline_preparation_executable","training_executable")), "explicit capability booleans")
    def as_dict(self):
        from dataclasses import asdict
        value = asdict(self)
        for key in ("offline_preparation_access","online_access"): value[key] = sorted(value[key])
        for key in ("modalities","train_requirements","attack_requirements"):
            value["spec"][key] = sorted(value["spec"][key])
        return value

@dataclass(frozen=True, slots=True)
class AttackBinding:
    """身份只在产物层；不传给H、G或Caption前端。"""
    dataset_manifest_sha256: str
    sample_id: str
    modality: str
    input_sha256: str
    def __post_init__(self):
        sha256(self.dataset_manifest_sha256); sha256(self.input_sha256)
        need(type(self.sample_id) is str and bool(self.sample_id), "sample identity required")
        need(self.modality in ("image","text"), "binding modality")
    @classmethod
    def from_manifest(cls, manifest, sample_id, modality):
        from dataclasses import asdict
        from ..contracts import DatasetManifest
        need(type(manifest) is DatasetManifest, "typed dataset manifest required")
        need(modality in ("image","text"), "binding modality")
        records=manifest.gallery if modality=="image" else manifest.queries
        key="image_id" if modality=="image" else "query_id"
        found=[r for r in records if getattr(r,key)==sample_id]
        need(len(found)==1, "sample not a unique manifest member")
        item=found[0]
        sha=item.image_sha256 if modality=="image" else hashlib.sha256(item.caption.encode("utf-8")).hexdigest()
        return cls(digest(asdict(manifest)),sample_id,modality,sha)
    @property
    def id(self):
        from dataclasses import asdict
        return digest(asdict(self))

@dataclass(frozen=True, slots=True)
class ImageRequest:
    public: PublicImageInput
    binding: AttackBinding
    def __post_init__(self):
        need(type(self.public) is PublicImageInput and type(self.binding) is AttackBinding,
             "public image + label-free binding required")
        need(self.binding.modality == "image" and self.binding.input_sha256 == self.public.content.sha256,
             "image content binding mismatch")
        need(self.public.pixel_space == "rgb_0_1", "raw RGB budget space required")

@dataclass(frozen=True, slots=True)
class TextRequest:
    public: PublicTextInput
    binding: AttackBinding
    def __post_init__(self):
        need(type(self.public) is PublicTextInput and type(self.binding) is AttackBinding,
             "public Caption + label-free binding required")
        need(self.binding.modality == "text" and self.binding.input_sha256 ==
             hashlib.sha256(self.public.caption.encode("utf-8")).hexdigest(), "Caption binding mismatch")
        # H必须独立复算Canonical；拒绝外部预选/评分属性。
        need(not self.public.attributes, "legacy H extracts its own Caption-only attributes")

@dataclass(frozen=True, slots=True)
class AttackOutput:
    attack_id: str
    binding: AttackBinding
    budget: AttackBudget
    payload: Any
    effective_delta: Any
    trace_json: bytes
    provenance_json: bytes
    def trace(self): return json.loads(self.trace_json)
    def provenance(self): return json.loads(self.provenance_json)
    @property
    def artifact_id(self):
        return digest({"attack":self.attack_id,"binding":self.binding.id,
                       "trace":self.trace(),"provenance":self.provenance()})

@runtime_checkable
class AttackPlugin(Protocol):
    info: PluginInfo
    budget: AttackBudget
    def generate(self, request): ...
    # 不要求train()、prepare()或Optimizer。

def tensor_sha(tensor):
    """显式结果审计，惰性Torch；不保留底层Storage或计算图。"""
    from ..artifacts import json_bytes
    x = tensor.detach().cpu().contiguous()
    h = hashlib.sha256(json_bytes([str(x.dtype),list(x.shape)]))
    h.update(x.reshape(-1).view(__import__("torch").uint8).numpy().tobytes())
    return h.hexdigest()

def frozen_state(module):
    import torch
    need(isinstance(module,torch.nn.Module), "explicit model module required")
    need(all(not m.training for m in module.modules()), "attack model must remain eval")
    need(all(not p.requires_grad and p.grad is None for p in module.parameters()),
         "attack parameters/gradients must stay frozen")
    return digest({name:tensor_sha(x) for name,x in sorted(module.state_dict().items())})

def output(attack_id, request, budget, payload, delta, trace, provenance):
    return AttackOutput(attack_id,request.binding,budget,payload,delta,json_bytes(trace),json_bytes(provenance))
