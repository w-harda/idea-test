"""物化计划及单位引用；训练来源与目标ModelSpace分离。"""
from dataclasses import dataclass,asdict,field
from ..artifacts import FileRef,digest,checked,ArtifactError
from ..contracts import DatasetManifest,sha256
from ..attacks.base import AttackBinding
from ..attacks.provenance import AttackProvenance

def need(ok,message):
    if not ok:raise ArtifactError(message)

@dataclass(frozen=True,slots=True)
class MaterializationPlan:
    manifest: DatasetManifest
    g: AttackProvenance
    h: AttackProvenance
    sources: tuple[FileRef,...]
    image_root: str
    pixel_size: tuple[int,int]=(384,128)
    fixture: bool=False
    source_config: FileRef | None=None
    schema: str="tbps-materialization-plan-1"
    _manifest_digest: str=field(init=False,repr=False,compare=False)
    _plan_digest: str=field(init=False,repr=False,compare=False)
    def __post_init__(self):
        need(type(self.manifest)is DatasetManifest and type(self.g)is AttackProvenance and type(self.h)is AttackProvenance,"typed materialization sources")
        need(self.g.attack_id=="gv2" and self.h.attack_id=="hv1","fixed G/H components")
        need(type(self.fixture)is bool and self.g.execution_kind==self.h.execution_kind==
             ("cpu_fixture" if self.fixture else "legacy_unverified"),"fixture/real provenance isolation")
        need(self.g.surrogate_sha256==self.h.surrogate_sha256 and self.g.training_dataset==self.h.training_dataset,"shared frozen training source")
        need(self.g.sources_sha256==self.h.sources_sha256,"shared attack source bundle")
        need(type(self.sources)is tuple and self.sources and all(type(x)is FileRef for x in self.sources),"explicit source FileRefs")
        need(len({x.path for x in self.sources})==len(self.sources),"unique materialization sources")
        need(type(self.pixel_size)is tuple and len(self.pixel_size)==2 and all(type(n)is int and n>0 for n in self.pixel_size),"RGB work size")
        need(self.fixture or (self.manifest.dataset_id=="cuhk_pedes" and self.manifest.official_split=="test" and self.pixel_size==(384,128)),
             "real materialization fixed CUHK official Test/RGB workspace")
        need(self.schema=="tbps-materialization-plan-1","plan schema")
        from pathlib import Path
        need(Path(self.image_root).is_absolute(),"absolute original image root")
        need(self.manifest.annotation in self.sources,"annotation source must be bound")
        if not self.fixture:
            from ..attacks.provenance import LegacyAttackSources
            need(type(self.source_config)is FileRef and self.source_config in self.sources,"verified legacy source config required")
            origin=LegacyAttackSources(self.source_config.path)
            need(origin.config_ref==self.source_config and origin.provenance("gv2")==self.g and origin.provenance("hv1")==self.h,
                 "real G/H source/protocol mismatch")
            need(all(ref in self.sources for ref in (*origin.runtime_refs,origin.ref("gv2","protocol"),
                 origin.ref("hv1","protocol"),origin.ref("hv1","selection"))),"complete fixed runtime/training/Selection sources required")
        for p in (self.g,self.h):need(p.checkpoint in self.sources,"Checkpoint must be bound to plan")
        for x in self.sources:checked(x)
        object.__setattr__(self,"_manifest_digest",digest(asdict(self.manifest)))
        object.__setattr__(self,"_plan_digest","")
        object.__setattr__(self,"_plan_digest",digest(self._body()))
    def _body(self):return {k:v for k,v in asdict(self).items() if not k.startswith("_")}
    @property
    def manifest_sha256(self):return self._manifest_digest
    @property
    def id(self):return self._plan_digest
    def binding(self,sample_id,modality):
        from .store import utf8_sha
        need(modality in ("image","text"),"materialization modality")
        rows=self.manifest.gallery if modality=="image" else self.manifest.queries
        found=[x for x in rows if (x.image_id if modality=="image" else x.query_id)==sample_id]
        need(len(found)==1,"unique materialization member")
        content=found[0].image_sha256 if modality=="image" else utf8_sha(found[0].caption)
        return AttackBinding(self.manifest_sha256,sample_id,modality,content)
    def descriptor(self):
        from ..attacks.gv2 import BUDGET as g
        from ..attacks.hv1 import BUDGET as h
        return {**self._body(),"attack_budgets":{"gv2":asdict(g),"hv1":asdict(h)},"victim_model":None,"target_feedback":False,"pixel_space":"rgb_0_1",
                "pixel_hash":"tbps-attack-tensor-sha-v1(dtype,shape,contiguous bytes)","text_hash":"sha256 UTF-8 bytes"}

@dataclass(frozen=True,slots=True)
class MaterializedUnit:
    unit_id: str
    plan_id: str
    binding: AttackBinding
    modality: str
    content_sha256: str
    data: FileRef
    metadata: FileRef
    manifest: FileRef
    def __post_init__(self):
        for v in (self.unit_id,self.plan_id,self.content_sha256):sha256(v)
        need(type(self.binding)is AttackBinding and self.modality==self.binding.modality,"unit input binding")
        for x in (self.data,self.metadata,self.manifest):need(type(x)is FileRef,"typed saved unit reference")
