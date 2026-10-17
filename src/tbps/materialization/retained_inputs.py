"""失败Attempt内已提交G/H产物的显式只读引用；原Plan/单元/FAIL回执不改签。"""
from dataclasses import dataclass,asdict
from pathlib import Path
from ..artifacts import ArtifactStore,FileRef,checked,read_json,sha_file,digest
from ..contracts import DatasetManifest,GalleryItem,QueryItem,PersonID
from ..attacks.base import AttackBinding
from ..attacks.provenance import AttackProvenance,LegacyAttackSources
from .contracts import MaterializedUnit,need
from .store import MaterializedStore,utf8_sha

ORIGINAL="/home/lzf/ldx/outputs/tbps-framework-stage4c2/materialized_smoke_v1"
ORIGINAL_RESULT_SHA="c1922bdf32935623dc9e85b1aeffe8a817cef2ab42b3ba767f62eda90e45155b"
ORIGINAL_ATTEMPT_SHA="ff0c62a387da4f0f9de11ceedf9c6bfae0ad10ca7eee28a09326cc1e4b762c1a"
PLAN_FIELDS={"manifest","g","h","sources","image_root","pixel_size","fixture","source_config","schema"}

@dataclass(frozen=True)
class FrozenPlanReference:
    """引用已SHA验证的原始计划；不调用新树MaterializationPlan或生成新id。"""
    id: str
    manifest_sha256: str
    manifest: DatasetManifest
    g: AttackProvenance
    h: AttackProvenance
    sources: tuple
    image_root: str
    pixel_size: tuple
    def binding(self,key,kind):
        rows=self.manifest.gallery if kind=="image" else self.manifest.queries
        found=[x for x in rows if (x.image_id if kind=="image" else x.query_id)==key]
        need(kind in ("image","text") and len(found)==1,"fixed original member")
        sha=found[0].image_sha256 if kind=="image" else utf8_sha(found[0].caption)
        return AttackBinding(self.manifest_sha256,key,kind,sha)

def unit_ref(d):
    return MaterializedUnit(d["unit_id"],d["plan_id"],AttackBinding(**d["binding"]),d["modality"],
        d["content_sha256"],*(FileRef(**d[k]) for k in ("data","metadata","manifest")))

class RetainedInputs:
    def __init__(self,root=ORIGINAL,*,fixture=False):
        self.root=Path(root).resolve()
        need(fixture or str(self.root)==ORIGINAL,"only exact retained real Attempt")
        if not fixture:
            checked(FileRef(str(self.root/"result/manifest.json"),ORIGINAL_RESULT_SHA))
            checked(FileRef(str(self.root/"attempt/manifest.json"),ORIGINAL_ATTEMPT_SHA))
        self.records=ArtifactStore(self.root,new_base=self.root.parent)
        self.store=ArtifactStore(self.root/"artifacts",new_base=self.root.parent)
        need(not self.records.pending() and not self.store.pending(),"pending retained inputs")
        for name in ("attempt","result","stage-g","stage-text","stage-irra"):self.records.verify(name)
        report=read_json(self.root/"result/report.json")
        g=read_json(self.root/"stage-g/report.json");h=read_json(self.root/"stage-text/report.json")
        irra=read_json(self.root/"stage-irra/report.json")
        need(report["status"]=="FAIL" and g["status"]==h["status"]=="PASS" and irra["status"]=="FAIL",
             "retain original whole FAIL and G/H partial PASS")
        need(report["stages"]=={"g":g,"text":h,"irra":irra},"stage completion evidence mismatch")
        need(g["G_loads"]==1 and h["H_loads"]==0 and irra["IRRA_loads"]==0 and g["model_before"]==g["model_after"],
             "original frozen model/load counts")
        self.store.verify("protocol")
        d=read_json(self.root/"artifacts/protocol/protocol.json")
        body={k:d[k] for k in PLAN_FIELDS}
        need(set(d)==PLAN_FIELDS|{"attack_budgets","victim_model","target_feedback","pixel_space","pixel_hash","text_hash"}
             and d["fixture"]==fixture and d["schema"]=="tbps-materialization-plan-1"
             and d["victim_model"]is None and d["target_feedback"]is False,"original materialization protocol")
        m=d["manifest"]
        member=lambda cls,row:cls(**{**row,"pid":PersonID(**row["pid"])})
        manifest=DatasetManifest(m["dataset_id"],m["official_split"],FileRef(**m["annotation"]),
            tuple(member(GalleryItem,row) for row in m["gallery"]),
            tuple(member(QueryItem,row) for row in m["queries"]),m["relevance"])
        prov=lambda row:AttackProvenance(**{**row,"checkpoint":FileRef(**row["checkpoint"])})
        self.plan=FrozenPlanReference(digest(body),digest(m),manifest,prov(d["g"]),prov(d["h"]),
             tuple(FileRef(**r) for r in d["sources"]),d["image_root"],tuple(d["pixel_size"]))
        self.protocol=d
        self.units=tuple(unit_ref(x["unit"]) for x in (*g["images"],*h["texts"]))
        need(len(self.units)==4 and len({x.binding.id for x in self.units})==4,"two image/two text units required")
        from .smoke_support import GALLERY,QUERY
        if not fixture:
            need(tuple(x.binding.sample_id for x in self.units)==(*GALLERY,*QUERY),"fixed ordered retained samples")
            legacy=LegacyAttackSources(d["source_config"]["path"]);checked(FileRef(**d["source_config"]))
            # 原fingerprint绑定原工作树producer路径；只验证原路径字节，不替换为新树路径。
            a=self.records.verify("attempt")
            original_code=next(FileRef(**r) for r in a["sources"] if r["path"].endswith("/materialization/smoke.py"))
            old_root=Path(original_code.path).parents[3]
            current_root=Path(__file__).parents[3]
            producers={str(old_root/Path(p).relative_to(current_root)):sha for p,sha in legacy.framework_sources.items()}
            for p,sha in producers.items():need(FileRef(p,sha) in self.plan.sources,"original producer pin absent");checked(FileRef(p,sha))
            original_fingerprint=digest({"config":legacy.config_ref.sha256,"catalog":legacy.catalog_ref.sha256,
                "descriptions":legacy.descriptions,"plugin_producers":producers})
            for name,expected in (("gv2",self.plan.g),("hv1",self.plan.h)):
                actual={**legacy.provenance(name).as_dict(),"sources_sha256":original_fingerprint}
                need(actual==asdict(expected),"original frozen G/H provenance does not match")
            self.legacy=legacy
        self.references={x.binding.id:x for x in self.units}
        for u in self.units:
            self.reference(u.binding)
            if u.modality=="image":
                claim="generation-claim-"+u.unit_id;proof="generation-proof-"+u.unit_id
                self.store.verify(claim);self.store.verify(proof)
                intent=read_json(self.root/"artifacts"/claim/"generation.json")
                done=read_json(self.root/"artifacts"/proof/"generation.json")
                need(intent["unit_id"]==u.unit_id and intent["plan_id"]==self.plan.id and
                     intent["binding"]==asdict(u.binding) and done["unit"]==asdict(u) and
                     done["claim_manifest"]==asdict(FileRef(str(self.root/"artifacts"/claim/"manifest.json"),
                                                sha_file(self.root/"artifacts"/claim/"manifest.json"))),
                     "original generation claim/proof mismatch")
        self.verify_plan(force=True)
    def reference(self,binding):
        need(binding.id in self.references and binding==self.plan.binding(binding.sample_id,binding.modality),"foreign retained member")
        ref=self.references[binding.id]
        need(ref.plan_id==self.plan.id and ref.unit_id==self.unit_id(binding),"retained unit/Plan identity mismatch")
        need(Path(ref.manifest.path)==self.root/"artifacts"/ref.unit_id/"manifest.json"
             and Path(ref.metadata.path)==Path(ref.manifest.path).with_name("metadata.json")
             and Path(ref.data.path)==Path(ref.manifest.path).with_name("pixels.pt" if ref.modality=="image" else "adversarial.txt"),
             "retained file path mismatch")
        self.store.verify(ref.unit_id,expected_provenance={"kind":"materialized_attack","plan_id":self.plan.id,"binding_id":binding.id})
        for r in (ref.data,ref.metadata,ref.manifest):checked(r)
        return ref
    def verify_plan(self,*,force=False):
        self.store.verify("protocol",expected_provenance={"kind":"materialization_protocol","plan_id":self.plan.id})
        need(read_json(self.root/"artifacts/protocol/protocol.json")==self.protocol,"original protocol changed")
        return FileRef(str(self.root/"artifacts/protocol/protocol.json"),sha_file(self.root/"artifacts/protocol/protocol.json"))
    # 复用原存储读取与像素/Unicode合法性规则；本类没有publish/generate，不能重签或覆盖。
    unit_id=MaterializedStore.unit_id
    image_request=MaterializedStore.image_request
    text_request=MaterializedStore.text_request
    _historical_action=MaterializedStore._historical_action
    load=MaterializedStore.load
    def descriptor(self):
        return {"schema":"tbps-retained-smoke-input-reference-1","input_kind":"readonly_materialized_smoke_reference",
            "original_attempt_status":"FAIL","original_plan_id":self.plan.id,
            "original_protocol":asdict(self.verify_plan()),"original_result":asdict(FileRef(
                str(self.root/"result/manifest.json"),sha_file(self.root/"result/manifest.json"))),
            "units":[asdict(x) for x in self.units],"new_G_forwards":0,"new_H_forwards":0,"republished":False}
