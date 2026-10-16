"""历史G/H实际成员绑定；只读协议/metadata/图像字节，不加载condition Tensor。"""
from dataclasses import dataclass, asdict, InitVar, field
from pathlib import Path
from ..artifacts import ArtifactError, checked, read_json, digest
from ..contracts import FileRef, DatasetManifest, GalleryItem, QueryItem, TrainRewardContext

IDENTITY=("row_id","image_rel_path","caption_index","split","role","pid","caption")
_MEMBERSHIP_VERIFIED=object()
SOURCES=("g_manifest","g_protocol","g_completed","internal_split","h_training","h_train_qualification",
         "h_validation_qualification","h_qualification_completed","h_gallery_train","h_gallery_validation","h_validation")

def need(ok,message):
    if not ok:raise ArtifactError(message)

@dataclass(frozen=True, slots=True)
class RewardBinding:
    """仅奖励层使用；不能传给Policy。membership摘要不是Test策略输入。"""
    context: TrainRewardContext
    query_id: str
    gallery_ids: tuple[str,...]
    membership_sha256: str

@dataclass(frozen=True, slots=True)
class AttackPartitions:
    train: DatasetManifest
    validation: DatasetManifest
    sources: tuple[FileRef,...]
    metadata_refs: tuple[FileRef,...]
    _train_members: frozenset = field(init=False,repr=False,compare=False)
    _validation_members: frozenset = field(init=False,repr=False,compare=False)
    _fingerprint: str = field(init=False,repr=False,compare=False)
    _verification: InitVar[object] = None
    def __post_init__(self,_verification):
        need(_verification is _MEMBERSHIP_VERIFIED,"use load_historical_partitions; caller-declared roles are not source verification")
        need(self.train.official_split==self.validation.official_split=="train","internal roles must come from official train")
        need(not {g.pid for g in self.train.gallery}&{g.pid for g in self.validation.gallery},"train/validation PID leakage")
        need(not {g.image_id for g in self.train.gallery}&{g.image_id for g in self.validation.gallery},"train/validation image leakage")
        for m,role in ((self.train,"attack_train"),(self.validation,"attack_validation")):
            need(all(g.role==role for g in m.gallery) and all(q.role==role for q in m.queries),"internal role mismatch")
        object.__setattr__(self,"_train_members",frozenset((*self.train.gallery,*self.train.queries)))
        object.__setattr__(self,"_validation_members",frozenset((*self.validation.gallery,*self.validation.queries)))
        object.__setattr__(self,"_fingerprint",digest({"train":asdict(self.train),"validation":asdict(self.validation),"sources":[asdict(r) for r in self.sources]}))
    @property
    def fingerprint(self):return self._fingerprint
    def require_member(self,record,role):
        need(role in ("attack_train","attack_validation"),"explicit internal role required")
        m=self.train if role=="attack_train" else self.validation
        need(type(record) in (GalleryItem,QueryItem),"typed source member required")
        members=self._train_members if role=="attack_train" else self._validation_members
        need(record in members,"identity/Caption/source not an actual selected member")
        return record
    def reward_binding(self,query,gallery):
        self.require_member(query,"attack_train")
        need(type(gallery) is tuple and gallery,"immutable training Gallery required")
        for g in gallery:self.require_member(g,"attack_train")
        need(len({g.image_id for g in gallery})==len(gallery),"duplicate reward Gallery")
        indices=[i for i,g in enumerate(gallery) if g.image_id==query.paired_image_id]
        need(len(indices)==1,"paired training image missing")
        ctx=TrainRewardContext(query.pid,tuple(g.pid for g in gallery),indices[0],"train","attack_train",
                               tuple(g.role for g in gallery))
        return RewardBinding(ctx,query.query_id,tuple(g.image_id for g in gallery),self.fingerprint)

def _load_historical_partitions(dataset, refs, *,expected_counts=(2048,128)):
    """refs来自Git固定来源目录；fixture可用自己的SHA，不默认信任role字段。"""
    need(type(refs) is dict and set(refs)==set(SOURCES) and all(type(v) is FileRef for v in refs.values()),"pinned historical source set required")
    before={k:checked(r) for k,r in refs.items()};data={k:read_json(r.path) for k,r in refs.items()}
    gm,gp,gc=data["g_manifest"],data["g_protocol"],data["g_completed"]
    need(gm["schema"]=="gv2-training-cache-2" and gp["schema"]=="gv2-full-prepare-protocol-2","G source schema")
    counts={"train":expected_counts[0],"validation":expected_counts[1]}
    need(gm["counts"]==gp["target_counts"]==counts and gc["method_audit_passed"] is True,"G source completion/counts")
    need(gc["manifest_sha256"]==refs["g_manifest"].sha256 and gm["protocol_sha256"]==refs["g_protocol"].sha256,"G source SHA linkage")
    need(Path(gm["protocol_path"]).resolve()==Path(refs["g_protocol"].path).resolve(),"G protocol path mismatch")
    need(gp["dataset_sha256"].get(dataset.annotation.path)==dataset.annotation.sha256 and
         gp["dataset_sha256"].get(refs["internal_split"].path)==refs["internal_split"].sha256,"G annotation/internal split provenance mismatch")
    internal=data["internal_split"];membership={}
    for role in counts:
        for i in internal[role+"_indices"]:
            need(type(i) is int and 0<=i<len(internal["records"]),"internal split index")
            r=internal["records"][i]
            need(r["row_id"] not in membership,"duplicate internal split row")
            membership[r["row_id"]]=(role,r["pid"],r["caption"])
    need(not set(internal["train_pids"])&set(internal["validation_pids"]),"source split PID leakage")
    gp_digest=digest(gp)
    planned=gp["samples"];entries=gm["entries"]
    need(len(entries)==len(planned)==sum(counts.values()),"G members missing")
    qualification={}
    qr=data["h_qualification_completed"]
    for role in counts:
        key="h_"+role+"_qualification";q=data[key]
        need(qr[role+"_sha256"]==refs[key].sha256 and q["protocol_sha256"]==digest(q["protocol"]),"H qualification SHA")
        need(q["protocol"]["role"]==role and q["protocol"]["schema"]=="hv1-formal-data-1","H qualification role/schema")
        need(q["protocol"]["training_sources"]["manifest_sha256"]==refs["g_manifest"].sha256 and
             q["protocol"]["training_sources"]["raw_annotation_sha256"]==dataset.annotation.sha256 and
             q["protocol"]["training_sources"]["internal_split_sha256"]==refs["internal_split"].sha256,"H training sources")
        identities=q["protocol"]["identities"]
        need([e["identity"] for e in q["entries"]]==identities,"H qualification member order")
        for e in q["entries"]:
            need(e["caption"]==e["identity"]["caption"],"H Caption altered")
            for a in e["attributes"]:
                need(type(a["char_start"]) is int and type(a["char_end"]) is int and 0<=a["char_start"]<a["char_end"]<=len(e["caption"]) and
                     e["caption"][a["char_start"]:a["char_end"]]==a["raw_text"],"H original Span mismatch")
        qualification[role]=identities
    selected={"train":[],"validation":[]};metadata=[];seen_rows=set();seen_images=set()
    for e,plan in zip(entries,planned):
        need(all(e[k]==plan[k] for k in (*IDENTITY,"K")),"G planned identity/order mismatch")
        need(type(e["pid"]) is int and type(e["caption_index"]) is int,"native PID/caption index must be integers")
        role=e["role"];need(role in counts and e["split"]=="train","official Test/val cannot enter internal attack split")
        row,j=dataset.row(e["row_id"])
        need((row.split,int(row.pid.value),row.image_rel_path,row.captions[j],j)==
             (e["split"],e["pid"],e["image_rel_path"],e["caption"],e["caption_index"]),"annotation/Caption/PID pairing mismatch")
        need(membership.get(e["row_id"])==(role,e["pid"],e["caption"]),"not an actual source split member")
        need(e["pid"] in internal[role+"_pids"],"PID outside original split")
        need(e["row_id"] not in seen_rows and e["image_rel_path"] not in seen_images,"duplicate selected identity/image")
        seen_rows.add(e["row_id"]);seen_images.add(e["image_rel_path"])
        mdref=FileRef(e["metadata_path"],e["metadata_sha256"])
        need(Path(mdref.path).resolve().is_relative_to(Path(refs["g_manifest"].path).resolve().parent),"metadata outside G cache")
        checked(mdref);md=read_json(mdref.path);metadata.append(mdref)
        need(md["protocol_hash"]==gp_digest and md["tensor_sha256"]==e["tensor_sha256"] and
             all(md[k]==e[k] for k in (*IDENTITY,"K")),"G metadata provenance/identity mismatch")
        identity={k:e[k] for k in IDENTITY}
        identity.update(image_sha256=md["source_image_sha256"],metadata_sha256=e["metadata_sha256"],condition_sha256=e["tensor_sha256"])
        selected[role].append(identity)
    manifests={}
    for role in counts:
        identities=selected[role]
        need(len(identities)==counts[role] and identities==qualification[role],"G/H actual members/order differ")
        gallery_protocol=data["h_gallery_"+role]
        need(gallery_protocol["schema"]=="hv1-formal-gallery-1" and gallery_protocol["role"]==role and
             gallery_protocol["identities"]==identities,"H reward/Validation Gallery member mismatch")
        public_role="attack_train" if role=="train" else "attack_validation"
        gs=[];qs=[]
        for x in identities:
            row,j=dataset.row(x["row_id"])
            checked(FileRef(str(dataset.image_path(row.image_rel_path)),x["image_sha256"]))
            gs.append(GalleryItem(row.image_id,row.image_rel_path,x["image_sha256"],row.pid,"train",public_role))
            qs.append(QueryItem(row.query_id(j),row.captions[j],j,row.image_id,row.pid,"train",public_role))
        manifests[role]=DatasetManifest("cuhk_pedes","train",dataset.annotation,tuple(gs),tuple(qs))
    hp=data["h_training"];vp=data["h_validation"]
    need(hp["schema"]=="hv1-formal-SCST-training-1" and hp["validation_feedback"] is False and
         hp["train_identities"]==selected["train"] and hp["epoch_size"]==counts["train"],"H SCST training member/source mismatch")
    train_ids={x["row_id"] for x in selected["train"]}
    epochs=hp["config"]["epochs"]
    need(type(epochs) is int and epochs>0 and type(hp["query_order"]) is list and set(hp["query_order"])<=train_ids and len(hp["query_order"])==counts["train"]*epochs,"H scheduler includes nontraining member or wrong visits")
    for i in range(epochs):
        need(set(hp["query_order"][i*counts["train"]:(i+1)*counts["train"]])==train_ids,"H epoch has missing/duplicate training members")
    need(hp["eligible_by_row"]=={e["identity"]["row_id"]:e["eligible"] for e in data["h_train_qualification"]["entries"]},"H eligibility source members mismatch")
    need(hp["qualification_sha256"]==digest(data["h_train_qualification"]) and
         hp["gallery_protocol_sha256"]==digest(data["h_gallery_train"]),"H training qualification/Gallery binding")
    need(vp["schema"]=="hv1-formal-validation-1" and vp["identities"]==selected["validation"] and
         vp["training_protocol_sha256"]==digest(hp) and
         vp["qualification_sha256"]==digest(data["h_validation_qualification"]) and
         vp["validation_gallery_protocol_sha256"]==digest(data["h_gallery_validation"]),"H independent Validation binding")
    need(before=={k:checked(r) for k,r in refs.items()},"historical sources changed during read")
    return AttackPartitions(manifests["train"],manifests["validation"],tuple(refs.values()),tuple(metadata),_verification=_MEMBERSHIP_VERIFIED)

def load_historical_partitions(dataset, refs, *,expected_counts=(2048,128)):
    """来源/成员校验失败显式拒绝；不以调用方role作为权限证明。"""
    try:return _load_historical_partitions(dataset,refs,expected_counts=expected_counts)
    except (KeyError,IndexError,TypeError,AttributeError) as exc:
        raise ArtifactError("missing/invalid historical membership schema") from exc
