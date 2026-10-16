"""Model Adapter结构契约；不强制双塔、Tensor维度或余弦评分。"""
from dataclasses import dataclass, asdict
from typing import Protocol, runtime_checkable
from ..contracts import ModelSpec, RepresentationArtifact, ScoreArtifact, sha256, text
from ..artifacts import digest

@dataclass(frozen=True, slots=True)
class ModelSpace:
    model_id: str
    checkpoint_sha256: str
    sources_sha256: str
    preprocess_sha256: str
    tokenizer_sha256: str
    scoring_protocol_sha256: str
    precision: str
    def __post_init__(self):
        text(self.model_id,"model");text(self.precision,"precision")
        for name in ("checkpoint_sha256","sources_sha256","preprocess_sha256","tokenizer_sha256","scoring_protocol_sha256"):sha256(getattr(self,name))
    @property
    def id(self):return digest(asdict(self))
    @property
    def score_id(self):return digest({"model_space":self.id,"native_scoring_protocol":self.scoring_protocol_sha256})

@dataclass(frozen=True, slots=True)
class RepresentationBatch:
    payload: object
    space_id: str
    modality: str
    def __post_init__(self):
        sha256(self.space_id)
        if self.modality not in ("image","text","native"):raise ValueError("representation modality")

@dataclass(frozen=True, slots=True)
class ScoreBatch:
    payload: object
    space_id: str
    scoring_protocol_sha256: str
    def __post_init__(self):
        sha256(self.space_id);sha256(self.scoring_protocol_sha256)

@runtime_checkable
class ModelAdapter(Protocol):
    @property
    def spec(self) -> ModelSpec: ...
    @property
    def space(self) -> ModelSpace: ...
    def score(self, query: object, gallery: object) -> ScoreBatch:
        """实现决定接收公开输入、local tokens或global features；不强制先encode。"""
        ...

@dataclass(frozen=True, slots=True)
class InputBinding:
    """持久化缓存身份；ordered_sha256按实际原图/对抗像素或真实Caption绑定，不含PID。"""
    model_space_id: str
    dataset_manifest_sha256: str
    ordered_sha256: tuple[str,...]
    modality: str
    ordered_ids: tuple[str,...]
    def __post_init__(self):
        sha256(self.model_space_id);sha256(self.dataset_manifest_sha256)
        if type(self.ordered_sha256) is not tuple or not self.ordered_sha256:raise ValueError("immutable ordered input SHA required")
        if type(self.ordered_ids) is not tuple or len(self.ordered_ids)!=len(self.ordered_sha256) or len(set(self.ordered_ids))!=len(self.ordered_ids):raise ValueError("unique ordered member IDs required")
        for i in self.ordered_ids:text(i,"input member ID")
        for h in self.ordered_sha256:sha256(h)
        if self.modality not in ("image","text","native"):raise ValueError("input modality")
    @property
    def cache_id(self):return digest(asdict(self))

@dataclass(frozen=True, slots=True)
class RepresentationRecord:
    artifact: RepresentationArtifact
    inputs: InputBinding
    def __post_init__(self):
        if type(self.artifact) is not RepresentationArtifact or type(self.inputs) is not InputBinding or self.artifact.space_id!=self.inputs.model_space_id:
            raise ValueError("representation source/input space conflict")
        if self.artifact.shape[0]!=len(self.inputs.ordered_sha256):raise ValueError("input count/representation mismatch")
    @property
    def cache_id(self):return self.inputs.cache_id

@dataclass(frozen=True, slots=True)
class ScoreRecord:
    artifact: ScoreArtifact
    query_inputs: InputBinding
    gallery_inputs: InputBinding
    def __post_init__(self):
        if type(self.artifact) is not ScoreArtifact or type(self.query_inputs) is not InputBinding or type(self.gallery_inputs) is not InputBinding:
            raise ValueError("typed score source bindings required")
        if self.query_inputs.model_space_id!=self.gallery_inputs.model_space_id:raise ValueError("cross-model score bindings")
        if self.query_inputs.ordered_ids!=self.artifact.query_order or self.gallery_inputs.ordered_ids!=self.artifact.gallery_order:
            raise ValueError("score input/order counts")
    @property
    def cache_id(self):
        return digest({"scoring_protocol":self.artifact.scoring_protocol_sha256,
                       "query_inputs":self.query_inputs.cache_id,"gallery_inputs":self.gallery_inputs.cache_id})
