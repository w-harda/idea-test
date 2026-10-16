"""Stage 1不可变契约；无模型、数据集实现或Tensor依赖。"""
from dataclasses import dataclass
from pathlib import PurePosixPath
from math import isfinite
import re

SHA = re.compile(r"[0-9a-f]{64}\Z")
ROLES = frozenset({"model_train", "attack_train", "attack_validation", "gallery", "query", "test"})
ACCESS = frozenset({"public_images", "captions", "caption_attributes", "train_pairs",
                    "train_pids", "source_gradients", "source_features", "source_rewards",
                    "source_scores", "paired_inputs", "victim_gradients", "victim_scores", "test_labels"})

def text(value, name):
    if type(value) is not str or not value.strip():
        raise ValueError(name + " must be a nonempty string")

def sha256(value):
    if type(value) is not str or not SHA.fullmatch(value):
        raise ValueError("lowercase SHA-256 required")

def count(value, name, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(name + " invalid integer")

def sequence(value, cls, name):
    if type(value) is not tuple or any(type(x) is not cls for x in value):
        raise TypeError(name + " must be immutable typed tuple")

def capabilities(value, allowed):
    if type(value) is not frozenset or not value <= allowed:
        raise ValueError("unknown/mutable capability declaration")

@dataclass(frozen=True, slots=True)
class FileRef:
    path: str
    sha256: str
    def __post_init__(self):
        text(self.path, "path"); sha256(self.sha256)

@dataclass(frozen=True, slots=True)
class PersonID:
    dataset: str
    value: str
    def __post_init__(self):
        text(self.dataset, "PID namespace"); text(self.value, "PID value")

@dataclass(frozen=True, slots=True)
class GalleryItem:
    image_id: str
    image_rel_path: str
    image_sha256: str
    pid: PersonID
    split: str
    role: str = "gallery"
    def __post_init__(self):
        text(self.image_id, "image id"); text(self.split, "split")
        text(self.image_rel_path, "image relative path"); sha256(self.image_sha256)
        p = PurePosixPath(self.image_rel_path)
        if p.is_absolute() or ".." in p.parts or "\\" in self.image_rel_path or ":" in self.image_rel_path:
            raise ValueError("unsafe image relative path")
        if type(self.pid) is not PersonID or self.role not in ROLES:
            raise ValueError("PID or role invalid")

@dataclass(frozen=True, slots=True)
class QueryItem:
    query_id: str
    caption: str
    caption_index: int
    paired_image_id: str
    pid: PersonID
    split: str
    role: str = "query"
    def __post_init__(self):
        for v,n in ((self.query_id,"query id"),(self.caption,"caption"),(self.paired_image_id,"paired image"),(self.split,"split")):
            text(v,n)
        count(self.caption_index,"caption_index",True)
        if type(self.pid) is not PersonID or self.role not in ROLES: raise ValueError("PID or role invalid")

@dataclass(frozen=True, slots=True)
class EvaluationIdentity:
    query_ids: tuple[str, ...]
    gallery_ids: tuple[str, ...]
    query_pids: tuple[PersonID, ...]
    gallery_pids: tuple[PersonID, ...]
    def __post_init__(self):
        for v,c,n in ((self.query_ids,str,"queries"),(self.gallery_ids,str,"gallery"),
                      (self.query_pids,PersonID,"query PIDs"),(self.gallery_pids,PersonID,"gallery PIDs")):
            sequence(v,c,n)
        for item in (*self.query_ids,*self.gallery_ids): text(item,"ordered identity")
        if not self.query_ids or not self.gallery_ids or len(self.query_ids)!=len(self.query_pids) or len(self.gallery_ids)!=len(self.gallery_pids):
            raise ValueError("identity counts mismatch")
        if len(set(self.query_ids))!=len(self.query_ids) or len(set(self.gallery_ids))!=len(self.gallery_ids):
            raise ValueError("duplicate ordered identity")
        if not set(self.query_pids)<=set(self.gallery_pids): raise ValueError("query has no same-namespace positive")
    def positives(self, index):
        return tuple(i for i,p in enumerate(self.gallery_pids) if p==self.query_pids[index])

@dataclass(frozen=True, slots=True)
class DatasetManifest:
    dataset_id: str
    official_split: str
    annotation: FileRef
    gallery: tuple[GalleryItem, ...]
    queries: tuple[QueryItem, ...]
    relevance: str = "all_same_pid"
    def __post_init__(self):
        text(self.dataset_id,"dataset"); text(self.official_split,"official split")
        if type(self.annotation) is not FileRef: raise TypeError("annotation FileRef required")
        sequence(self.gallery,GalleryItem,"gallery"); sequence(self.queries,QueryItem,"queries")
        if self.relevance!="all_same_pid": raise ValueError("Stage 1 only describes all_same_pid relevance")
        self.evaluation_identity()
        items={g.image_id:g for g in self.gallery}
        pairs=set()
        for record in (*self.gallery,*self.queries):
            if record.split=="test" and record.role in ("model_train","attack_train","attack_validation"):
                raise ValueError("official Test cannot claim training/selection role")
            if record.pid.dataset!=self.dataset_id or record.split!=self.official_split:
                raise ValueError("dataset namespace/split mismatch")
        for q in self.queries:
            if q.paired_image_id not in items or items[q.paired_image_id].pid!=q.pid: raise ValueError("incorrect caption/image pairing")
            pair=q.paired_image_id,q.caption_index
            if pair in pairs: raise ValueError("duplicate image/caption_index")
            pairs.add(pair)
    def evaluation_identity(self):
        return EvaluationIdentity(tuple(q.query_id for q in self.queries),tuple(g.image_id for g in self.gallery),
                                  tuple(q.pid for q in self.queries),tuple(g.pid for g in self.gallery))

@dataclass(frozen=True, slots=True)
class AttributeSpan:
    semantic_axis: str
    canonical_value: str
    raw_text: str
    start: int
    end: int
    def __post_init__(self):
        for v,n in ((self.semantic_axis,"axis"),(self.canonical_value,"canonical"),(self.raw_text,"raw")): text(v,n)
        count(self.start,"start",True);count(self.end,"end")
        if self.end<=self.start: raise ValueError("empty span")

@dataclass(frozen=True, slots=True)
class PublicTextInput:
    caption: str
    attributes: tuple[AttributeSpan, ...] = ()
    def __post_init__(self):
        text(self.caption,"caption");sequence(self.attributes,AttributeSpan,"attributes")
        if any(a.end>len(self.caption) or self.caption[a.start:a.end]!=a.raw_text for a in self.attributes):
            raise ValueError("Span must exactly match original Caption")
        if len(set(self.attributes))!=len(self.attributes): raise ValueError("duplicate exact span")
    # 没有PID、Gallery、paired_index或reward字段。

@dataclass(frozen=True, slots=True)
class PublicImageInput:
    content: FileRef
    pixel_space: str
    def __post_init__(self):
        if type(self.content) is not FileRef: raise TypeError("image content FileRef required")
        text(self.pixel_space,"pixel space")

@dataclass(frozen=True, slots=True)
class TrainRewardContext:
    query_pid: PersonID
    gallery_pids: tuple[PersonID, ...]
    paired_index: int
    split: str
    role: str
    gallery_roles: tuple[str, ...]
    def __post_init__(self):
        if type(self.query_pid) is not PersonID: raise TypeError("PID required")
        sequence(self.gallery_pids,PersonID,"training gallery")
        sequence(self.gallery_roles,str,"training roles");count(self.paired_index,"paired_index",True)
        if self.split!="train" or self.role!="attack_train" or len(self.gallery_roles)!=len(self.gallery_pids) or any(r!="attack_train" for r in self.gallery_roles):
            raise ValueError("Validation/Test cannot enter training feedback")
        if any(p.dataset!=self.query_pid.dataset for p in self.gallery_pids): raise ValueError("cross-dataset reward binding")
        positive=tuple(p==self.query_pid for p in self.gallery_pids)
        if not any(positive) or all(positive) or self.paired_index>=len(positive) or not positive[self.paired_index]:
            raise ValueError("verified paired positive and negatives required")

@dataclass(frozen=True, slots=True)
class TokenizerSpec:
    name: str
    source_sha256: str
    max_length: int | None
    unit: str
    truncation: str
    def __post_init__(self):
        text(self.name,"tokenizer");sha256(self.source_sha256);text(self.unit,"token unit")
        if self.max_length is not None: count(self.max_length,"max_length")
        if self.truncation not in ("reject","retain_and_report","none"): raise ValueError("unknown truncation policy")

@dataclass(frozen=True, slots=True)
class ModelSpec:
    id: str
    version: str
    image_size_hw: tuple[int,int] | None
    tokenizer: TokenizerSpec | None
    representation_dim: int | None
    scoring: str
    normalization: str
    capabilities: frozenset[str]
    def __post_init__(self):
        text(self.id,"model");text(self.version,"version");text(self.normalization,"normalization")
        if self.image_size_hw is not None:
            sequence(self.image_size_hw,int,"image size")
            if len(self.image_size_hw)!=2: raise ValueError("size must be H,W")
            for n in self.image_size_hw:count(n,"size")
        if self.representation_dim is not None: count(self.representation_dim,"feature dim")
        if self.tokenizer is not None and type(self.tokenizer) is not TokenizerSpec: raise TypeError("TokenizerSpec required")
        if self.scoring not in ("cosine","dot","fusion","native"): raise ValueError("unknown scoring")
        capabilities(self.capabilities,frozenset({"image_encoding","text_encoding","tokenize","score","differentiable_images","differentiable_text","local_features","fusion"}))
        if self.scoring=="fusion" and "fusion" not in self.capabilities: raise ValueError("fusion capability required")
        if "tokenize" in self.capabilities and self.tokenizer is None: raise ValueError("tokenizer specification missing")

@dataclass(frozen=True, slots=True)
class AttackSpec:
    id: str
    version: str
    modalities: frozenset[str]
    artifact_kind: str
    training_kind: str | None
    attack_time_optimization: bool
    train_requirements: frozenset[str]
    attack_requirements: frozenset[str]
    checkpoint_boundary: str | None
    inference_state_policy: str = "read_only"
    def __post_init__(self):
        text(self.id,"attack");text(self.version,"version")
        capabilities(self.modalities,frozenset({"image","text"}))
        if not self.modalities:raise ValueError("attack modality required")
        if self.artifact_kind not in ("generator","policy","uap","iterative","composite"): raise ValueError("artifact kind")
        if self.training_kind not in (None,"offline","scst","meta","multi_stage"):raise ValueError("training kind")
        if type(self.attack_time_optimization) is not bool:raise TypeError("explicit bool required")
        capabilities(self.train_requirements,ACCESS);capabilities(self.attack_requirements,ACCESS)
        if self.checkpoint_boundary is not None:text(self.checkpoint_boundary,"checkpoint boundary")
        if self.inference_state_policy not in ("read_only","declared_stateful"):raise ValueError("state policy")

@dataclass(frozen=True, slots=True)
class ImageBudget:
    norm: str
    value: float
    pixel_space: str
    def __post_init__(self):
        if self.norm not in ("linf","l2","l1"):raise ValueError("unsupported image norm")
        if type(self.value) not in (int,float) or not isfinite(self.value) or self.value<=0:raise ValueError("positive finite budget")
        text(self.pixel_space,"budget pixel space")

@dataclass(frozen=True, slots=True)
class TextBudget:
    unit: str
    max_edits: int | None
    selection_rule: str | None
    constraints_id: str
    per_attribute: int | None = None
    per_shared_word: int | None = None
    def __post_init__(self):
        if self.unit not in ("unicode_character","word","token","byte"):raise ValueError("text budget unit")
        text(self.constraints_id,"constraints")
        if (self.max_edits is None)==(self.selection_rule is None):raise ValueError("fixed limit XOR dynamic rule required")
        if self.max_edits is not None:count(self.max_edits,"max_edits",True)
        if self.selection_rule is not None:text(self.selection_rule,"selection rule")
        for n in (self.per_attribute,self.per_shared_word):
            if n is not None:count(n,"per-word/attribute limit")

@dataclass(frozen=True, slots=True)
class AttackBudget:
    image: ImageBudget | None = None
    text: TextBudget | None = None
    def __post_init__(self):
        if self.image is None and self.text is None:raise ValueError("explicit attack budget required")
        if self.image is not None and type(self.image) is not ImageBudget:raise TypeError("ImageBudget required")
        if self.text is not None and type(self.text) is not TextBudget:raise TypeError("TextBudget required")

@dataclass(frozen=True, slots=True)
class AttackArtifact:
    artifact_id: str
    attack_id: str
    checkpoint_sha256: str
    inputs_sha256: tuple[str,...]
    content: FileRef
    budget: AttackBudget
    def __post_init__(self):
        text(self.artifact_id,"artifact");text(self.attack_id,"attack");sha256(self.checkpoint_sha256)
        sequence(self.inputs_sha256,str,"inputs")
        if not self.inputs_sha256:raise ValueError("input provenance required")
        for h in self.inputs_sha256:sha256(h)
        if type(self.content) is not FileRef or type(self.budget) is not AttackBudget:raise TypeError("typed artifact payload/budget")

@dataclass(frozen=True, slots=True)
class RepresentationArtifact:
    artifact_id: str
    model_checkpoint_sha256: str
    space_id: str
    kind: str
    shape: tuple[int,...]
    dtype: str
    normalization: str
    preprocess_sha256: str
    tokenizer_sha256: str | None
    content: FileRef
    def __post_init__(self):
        for v,n in ((self.artifact_id,"artifact"),(self.space_id,"space"),(self.dtype,"dtype"),(self.normalization,"normalization")):text(v,n)
        for h in (self.model_checkpoint_sha256,self.preprocess_sha256):sha256(h)
        if self.tokenizer_sha256 is not None:sha256(self.tokenizer_sha256)
        if self.kind not in ("global_embedding","local_tokens","native"):raise ValueError("representation kind")
        sequence(self.shape,int,"shape")
        if not self.shape:raise ValueError("nonempty shape")
        for n in self.shape:count(n,"shape",True)
        if type(self.content) is not FileRef:raise TypeError("content FileRef required")

@dataclass(frozen=True, slots=True)
class ScoreArtifact:
    artifact_id: str
    model_checkpoint_sha256: str
    scoring_protocol_sha256: str
    query_order: tuple[str,...]
    gallery_order: tuple[str,...]
    direction: str
    content: FileRef
    def __post_init__(self):
        text(self.artifact_id,"artifact")
        sha256(self.model_checkpoint_sha256);sha256(self.scoring_protocol_sha256)
        for v in (self.query_order,self.gallery_order):
            sequence(v,str,"score order")
            if not v or len(set(v))!=len(v) or any(not s for s in v):raise ValueError("invalid ordered score IDs")
        if self.direction not in ("higher_better","lower_better"):raise ValueError("score direction")
        if type(self.content) is not FileRef:raise TypeError("content FileRef required")
