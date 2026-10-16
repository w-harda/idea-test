"""旧Joint Full元数据只读索引；不聚合特征，不重标新ModelSpace，不提供排名入口。"""
from dataclasses import dataclass, asdict
from pathlib import Path
from ..artifacts import ArtifactError, checked, read_json, digest
from ..contracts import DatasetManifest, FileRef, sha256

@dataclass(frozen=True,slots=True)
class LegacySpace:
    model_id: str
    checkpoint_sha256: str
    preprocess_sha256: str
    tokenizer_source: FileRef
    bpe: FileRef
    ranking_source: FileRef
    def __post_init__(self):
        if self.model_id!="irra":raise ArtifactError("historical IRRA features cannot serve another victim")
        for h in (self.checkpoint_sha256,self.preprocess_sha256):sha256(h)
        for r in (self.tokenizer_source,self.bpe,self.ranking_source):
            if type(r)is not FileRef:raise ArtifactError("explicit official I/O source")

def describe_joint_full(protocol,summary,completed,manifest,space):
    if type(manifest)is not DatasetManifest or type(space)is not LegacySpace:
        raise ArtifactError("typed official manifest and legacy space required")
    if manifest.dataset_id!="cuhk_pedes" or manifest.official_split!="test":
        raise ArtifactError("only official CUHK-PEDES Test legacy bridge")
    refs=(protocol,summary,completed,manifest.annotation,space.tokenizer_source,space.bpe,space.ranking_source)
    before={r.path:checked(r) for r in refs}
    p=read_json(protocol.path,"hv1-joint-full-1");s=read_json(summary.path,"hv1-joint-full-result-1");d=read_json(completed.path)
    if not d.get("method_audit_passed") or d.get("summary_sha256")!=summary.sha256 or (
        d.get("protocol_sha256")!=digest(p) or s.get("protocol_sha256")!=digest(p)):
        raise ArtifactError("legacy completion/protocol/summary mismatch")
    if p["model_sha256"]!=s["model_sha256"] or p["model_sha256"]["IRRA"]!=space.checkpoint_sha256 or (
        digest(p["preprocess"])!=space.preprocess_sha256):
        raise ArtifactError("legacy model/preprocess space mismatch")
    for ref in (space.tokenizer_source,space.bpe,space.ranking_source):
        if p["evidence_sha256"].get(ref.path)!=ref.sha256:raise ArtifactError("legacy native tokenizer/ranking source mismatch")
    if p["gallery_count"]!=len(manifest.gallery) or p["query_count"]!=len(manifest.queries) or (
        s["gallery_count"]!=len(manifest.gallery) or s["query_count"]!=len(manifest.queries)):
        raise ArtifactError("legacy member counts")
    if not p["all_queries_retained"] or p["Test_feedback_for_selection"] is not False or s["all_queries_retained"] is not True:
        raise ArtifactError("legacy query retention/selection isolation")
    for i,(a,b) in enumerate(zip(p["gallery"],manifest.gallery)):
        if (a["index"],a["row_id"],a["image"],a["image_sha256"],str(a["pid"]))!=(
            i,b.image_id,b.image_rel_path,b.image_sha256,b.pid.value):
            raise ArtifactError("legacy Gallery order/identity mismatch")
    for i,(a,b) in enumerate(zip(p["queries"],manifest.queries)):
        if (a["index"],a["row_id"],a["caption"],a["caption_index"],str(a["pid"]))!=(
            i,b.query_id,b.caption,b.caption_index,b.pid.value):
            raise ArtifactError("legacy Query order/Caption mismatch")
        if type(a["paired_index"])is not int or not 0<=a["paired_index"]<len(manifest.gallery) or (
            manifest.gallery[a["paired_index"]].image_id!=b.paired_image_id):
            raise ArtifactError("legacy paired source mismatch")
    if len(p["gallery"])!=len(manifest.gallery) or len(p["queries"])!=len(manifest.queries):
        raise ArtifactError("legacy omitted/extra rows")
    declared=s["feature_files_sha256"]
    for name,h in declared.items():
        path=Path(name)
        if path.is_absolute() or ".." in path.parts or not name.startswith("features/"):
            raise ArtifactError("unsafe legacy feature index")
        sha256(h)
    if not declared:raise ArtifactError("missing declared feature SHA index")
    if before!={r.path:checked(r) for r in refs}:raise ArtifactError("legacy evidence changed")
    return {"schema":"tbps-legacy-full-index-1","status":"METADATA_VERIFIED_FEATURES_PENDING",
            "manifest_sha256":digest(asdict(manifest)),"legacy_protocol_sha256":digest(p),
            "legacy_space_sha256":digest(asdict(space)),"legacy_space":asdict(space),
            "evidence":[asdict(r) for r in refs],"declared_feature_files":len(declared),
            "declared_features_digest":digest(declared),"feature_bytes_verified":False,"adversarial_pixel_content_sha256_available":False,
            "missing_evidence":"historical Store does not record per-image adversarial RGB content SHA",
            "features_loaded":0,"ranking_executed":False,"replay_executable":False,
            "reuse_scope":"only this IRRA/CUHK protocol; never RaSa/APTM or automatic new ModelSpace"}

def require_replay(index):
    raise ArtifactError("Stage 4B required: metadata index never authorizes feature loading or ranking")
