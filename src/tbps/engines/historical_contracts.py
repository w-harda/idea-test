"""历史特征重放身份；不继承InputBinding，不声称拥有对抗RGB内容。"""
from dataclasses import dataclass,asdict
from ..artifacts import ArtifactError,checked,digest
from ..contracts import FileRef,sha256
from ..models.base import RepresentationBatch
from .provenance import payload_digest

FAMILIES=("clean_gallery","image_gallery","clean_text","adv_text")
ENGINE_FAMILIES={"clean_gallery":"clean_gallery","adv_gallery":"image_gallery",
                 "clean_text":"clean_text","adv_text":"adv_text"}
MODEL_SHAS={"G":"f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e",
"H":"487f3061b6cba2ddabccdd49745a7b99c8c549f4541781b8814b299ef5f279df",
"IRRA":"9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009"}
STATUSES=("METADATA_VERIFIED","FEATURES_VERIFIED","METRICS_REPLAY_VERIFIED")

@dataclass(frozen=True,slots=True)
class HistoricalReplayIdentity:
    manifest_sha256: str
    protocol: FileRef
    summary: FileRef
    completed: FileRef
    selection: FileRef
    models: tuple[tuple[str,FileRef],...]
    legacy_space_sha256: str
    feature_index_sha256: str
    fixture: bool=False
    victim_model: str="irra"
    dataset: str="cuhk_pedes"
    official_split: str="test"
    adversarial_pixel_sha256: str="UNKNOWN"
    artifact_kind: str="historical_feature_replay"
    def __post_init__(self):
        for h in (self.manifest_sha256,self.legacy_space_sha256,self.feature_index_sha256):sha256(h)
        if self.victim_model!="irra" or self.dataset!="cuhk_pedes" or self.official_split!="test":
            raise ArtifactError("historical features restricted to IRRA/CUHK official Test")
        if self.adversarial_pixel_sha256!="UNKNOWN" or self.artifact_kind!="historical_feature_replay":
            raise ArtifactError("feature SHA must never stand in for adversarial RGB SHA")
        if type(self.fixture)is not bool or type(self.models)is not tuple or len(self.models)!=3:
            raise ArtifactError("explicit replay profile and immutable model refs")
        if len({k for k,_ in self.models})!=3 or set(dict(self.models))!=set(MODEL_SHAS):
            raise ArtifactError("G/H/IRRA checkpoint sources required")
        for ref in self.refs:
            if type(ref)is not FileRef:raise ArtifactError("typed replay source FileRef")
        if not self.fixture and {k:v.sha256 for k,v in self.models}!=MODEL_SHAS:
            raise ArtifactError("frozen G10/H7/IRRA model SHA mismatch")
    @property
    def refs(self):return (self.protocol,self.summary,self.completed,self.selection,*(v for _,v in self.models))
    @property
    def id(self):return digest(asdict(self))
    def verify(self):
        for ref in self.refs:checked(ref)

@dataclass(frozen=True,slots=True)
class HistoricalFeatureBinding:
    replay_id: str
    model_space_id: str
    family: str
    ordered_ids: tuple[str,...]
    original_content_sha256: tuple[str,...]
    row_feature_sha256: tuple[str,...]
    text_content_sha256: tuple[str,...]=()
    def __post_init__(self):
        sha256(self.replay_id);sha256(self.model_space_id)
        if self.family not in FAMILIES or type(self.ordered_ids)is not tuple or not self.ordered_ids or (
            len(set(self.ordered_ids))!=len(self.ordered_ids)):
            raise ArtifactError("replay family/order")
        n=len(self.ordered_ids)
        for values in (self.original_content_sha256,self.row_feature_sha256):
            if type(values)is not tuple or len(values)!=n:raise ArtifactError("replay row binding size")
            for h in values:sha256(h)
        expected=0 if self.family.endswith("gallery") else n
        if type(self.text_content_sha256)is not tuple or len(self.text_content_sha256)!=expected:
            raise ArtifactError("real cached Caption SHA required for text; no fabricated image SHA")
        for h in self.text_content_sha256:sha256(h)

@dataclass(frozen=True,slots=True)
class HistoricalRepresentation:
    batch: RepresentationBatch
    binding: HistoricalFeatureBinding
    identity: HistoricalReplayIdentity
    payload_sha256: str
    cache_sources: tuple[FileRef,...]
    verification_status: str="FEATURES_VERIFIED"
    def __post_init__(self):
        if type(self.batch)is not RepresentationBatch or type(self.binding)is not HistoricalFeatureBinding or (
            type(self.identity)is not HistoricalReplayIdentity):
            raise ArtifactError("typed historical replay representation")
        if self.verification_status!="FEATURES_VERIFIED":raise ArtifactError("unverified historical features cannot score")
        if self.binding.replay_id!=self.identity.id or self.batch.space_id!=self.binding.model_space_id:
            raise ArtifactError("replay identity/model mismatch")
        if self.batch.modality!=("image" if self.binding.family.endswith("gallery") else "text"):
            raise ArtifactError("replay modality")
        if type(self.cache_sources)is not tuple or not self.cache_sources:raise ArtifactError("feature source refs required")
        sha256(self.payload_sha256);self.verify()
    def verify(self):
        from ..models.irra_cached import normalized
        normalized(self.batch.payload,len(self.binding.ordered_ids))
        if payload_digest(self.batch.payload)!=self.payload_sha256:raise ArtifactError("replay features mutated")
        if tuple(payload_digest(row) for row in self.batch.payload)!=self.binding.row_feature_sha256:
            raise ArtifactError("replay row feature order mutated")
        for ref in self.cache_sources:checked(ref)
    def descriptor(self):
        return {"kind":"historical_feature_replay","identity":asdict(self.identity),"binding":asdict(self.binding),
                "payload_sha256":self.payload_sha256,"verification_status":self.verification_status,
                "adversarial_pixel_sha256":"UNKNOWN","origin":None,
                "cache_sources":[asdict(r) for r in self.cache_sources],
                "reuse_scope":"same IRRA historical retrieval only; not materialized RGB/new attack"}
    @property
    def id(self):return digest(self.descriptor())

def validate_replay_inputs(engine,inputs,modes):
    from dataclasses import asdict
    import hashlib
    from .evaluation import MODES
    required={k for mode in modes for k in MODES[mode]}
    if set(inputs)!=required:raise ArtifactError("replay family set mismatch")
    replay_ids=set()
    for key,b in inputs.items():
        if type(b)is not HistoricalRepresentation:raise ArtifactError("replay cannot mix materialized BoundRepresentation")
        b.verify();replay_ids.add(b.identity.id)
        expected=ENGINE_FAMILIES[key]
        members=engine.manifest.gallery if expected.endswith("gallery") else engine.manifest.queries
        ids=tuple(g.image_id for g in members) if expected.endswith("gallery") else tuple(q.query_id for q in members)
        original=tuple(g.image_sha256 for g in members) if expected.endswith("gallery") else tuple(
            hashlib.sha256(q.caption.encode("utf-8")).hexdigest() for q in members)
        if b.binding.family!=expected or b.binding.ordered_ids!=ids or b.binding.original_content_sha256!=original:
            raise ArtifactError("replay family/member/order/original content mismatch")
        if b.identity.manifest_sha256!=engine.manifest_sha256 or b.binding.model_space_id!=engine.space.id:
            raise ArtifactError("replay manifest/victim space mismatch")
        if b.identity.fixture!=(engine.execution=="historical_fixture_replay"):
            raise ArtifactError("fixture vs real replay identity conflict")
        if expected=="clean_text" and b.binding.text_content_sha256!=original:
            raise ArtifactError("original text SHA mismatch")
    if len(replay_ids)!=1 or replay_ids!={engine.adapter.replay_identity.id}:
        raise ArtifactError("mixed historical Full sources")
    engine.adapter.replay_identity.verify()
