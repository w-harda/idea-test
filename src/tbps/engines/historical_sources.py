"""Joint Full冻结来源/Selection/成员校验；仅SHA/JSON，不调用旧恢复或排名。"""
from dataclasses import asdict
from pathlib import Path
from ..artifacts import ArtifactError,FileRef,checked,digest,read_json
from ..contracts import DatasetManifest,GalleryItem,QueryItem
from ..datasets.cuhk_pedes import CuhkPedes
from ..models.base import ModelSpace
from .legacy import LegacySpace,describe_joint_full
from .historical_contracts import HistoricalReplayIdentity,MODEL_SHAS

def need(value,message):
    if not value:raise ArtifactError(message)

def safe_file(root,name):
    p=Path(name)
    need(not p.is_absolute() and ".." not in p.parts,"unsafe historical path")
    target=root/p
    need(target.resolve().is_relative_to(root.resolve()) and not any(
        x.is_symlink() for x in (target,*target.parents) if x.is_relative_to(root)),"symlink/escape in historical store")
    return target

class JointFullSources:
    def __init__(self,config_path):
        self.config_ref=FileRef(str(Path(config_path).resolve()),__import__("tbps.artifacts",fromlist=["sha_file"]).sha_file(config_path))
        c=read_json(config_path,"tbps-historical-replay-sources-1")
        need(set(c)=={"schema","profile","root","stage4a_sources","attack_sources","expected_counts"},"replay source config fields")
        need(c["profile"] in ("real","cpu_fixture"),"explicit replay source profile")
        self.fixture=c["profile"]=="cpu_fixture";self.config=c;self.root=Path(c["root"]).resolve()
        self.base_ref=FileRef(**c["stage4a_sources"]);checked(self.base_ref)
        self.base=read_json(self.base_ref.path,"tbps-stage4a-legacy-metadata-1")
        self.headers={k:FileRef(**v) for k,v in self.base["joint"].items()}
        for ref in self.headers.values():
            need(Path(ref.path).resolve().parent==self.root,"joint header root mismatch");checked(ref)
        self.protocol=read_json(self.headers["protocol"].path,"hv1-joint-full-1")
        self.summary=read_json(self.headers["summary"].path,"hv1-joint-full-result-1")
        self.completed=read_json(self.headers["completed"].path)
        self.protocol_id=digest(self.protocol)
        p=self.protocol;s=self.summary;d=self.completed
        n=(p["gallery_count"],p["query_count"])
        need(n==(c["expected_counts"]["gallery"],c["expected_counts"]["query"]),"replay expected counts")
        need((self.fixture and max(n)<=64) or (not self.fixture and n==(3074,6156)),"fixed Full or small fixture only")
        if not self.fixture:
            need(p["model_sha256"]==MODEL_SHAS,"frozen G10/H7/official IRRA sources")
            need(p["runtime"]["torch"]=="2.5.1+cu124" and p["runtime"]["cpu_threads"]==2,"historical runtime")
            need(p["epsilon"]==8/255 and p["text_batch"]==16,"original epsilon/text batch")
        # 只读原始annotation，原图实际字节SHA在evidence_sha256校验；不解码图片。
        self.annotation=FileRef(**self.base["annotation"])
        dataset=CuhkPedes(self.annotation,self.base["image_root"])
        rows=tuple(r for r in dataset.rows if r.split=="test")
        need(len(rows)==n[0] and sum(len(r.captions) for r in rows)==n[1],"official Test annotation counts")
        gs=tuple(GalleryItem(r.image_id,r.image_rel_path,g["image_sha256"],r.pid,r.split)
                 for r,g in zip(rows,p["gallery"]))
        qs=tuple(QueryItem(r.query_id(i),t,i,r.image_id,r.pid,r.split) for r in rows for i,t in enumerate(r.captions))
        self.manifest=DatasetManifest("cuhk_pedes","test",self.annotation,gs,qs)
        raw_space=self.base["legacy_space"]
        self.legacy_space=LegacySpace(**{**raw_space,**{k:FileRef(**raw_space[k]) for k in ("tokenizer_source","bpe","ranking_source")}})
        describe_joint_full(self.headers["protocol"],self.headers["summary"],self.headers["completed"],self.manifest,self.legacy_space)
        self.declared=s["feature_files_sha256"]
        self.refs={}
        def pin(ref):
            if ref.path in self.refs:need(self.refs[ref.path]==ref,"conflicting source SHA")
            self.refs[ref.path]=ref
        for ref in (self.config_ref,self.base_ref,self.annotation,*self.headers.values()):pin(ref)
        # 所有声明来源、模型、Selection证明均只校验文件SHA，不反序列化。
        for mapping in (p["source_sha256"],p["evidence_sha256"],p["H_selection"]["proof"]):
            for name,h in mapping.items():pin(FileRef(name,h))
        selected=p["H_selection"]
        self.selection=FileRef(selected["path"],selected["sha256"]);pin(self.selection)
        record=read_json(self.selection.path)
        need(self.summary.get("H_selection_sha256",self.selection.sha256)==self.selection.sha256,"summary Selection source")
        need(Path(self.selection.path).stem==digest(record),"selection filename/content identity")
        need(record==selected["selection"] and record["uses_official_Test"] is False,"immutable Selection/Test isolation")
        candidates=record["candidates"]
        need(candidates and len({r["checkpoint_sha256"] for r in candidates})==len(candidates),"selection candidate identity")
        best=min(candidates,key=lambda x:(x["groups"]["Joint"]["mAP"],x["cursor"]))
        need(best["checkpoint_sha256"]==record["selected_checkpoint_sha256"]==selected["checkpoint_sha256"]==p["model_sha256"]["H"],
             "Validation selection Checkpoint mismatch")
        need(best["cursor"]==record["cursor"] and (self.fixture or record["cursor"]==14336),"H selection visit")
        for candidate in candidates:
            path=Path(selected["checkpoint"]).parents[3]/"validation"/candidate["checkpoint_sha256"]
            refpath=str(path/"summary.json")
            need(refpath in self.refs,"candidate Validation proof missing")
            saved=read_json(refpath)
            for key in ("groups","cursor","Joint_minus_ImageOnly_mAP_pp","checkpoint_sha256"):
                need(saved[key]==candidate[key],"selection candidate data changed")
        models={}
        for name,h in p["model_sha256"].items():
            possible=[r for r in self.refs.values() if r.sha256==h and r.path.endswith((".pt",".pth"))]
            need(possible,"missing frozen model file: "+name);models[name]=possible[0]
        self.models=tuple(sorted(models.items()))
        for ref in (self.legacy_space.tokenizer_source,self.legacy_space.bpe,self.legacy_space.ranking_source):pin(ref)
        # 完成态CSV/baseline/execution/所有attempt证据均严格关联。
        need(d["summary_sha256"]==self.headers["summary"].sha256 and d["method_audit_passed"] is True,"completion SHA")
        for field,name in (("metrics_csv_sha256","metrics.csv"),("baseline_verified_sha256","baseline_verified.json")):
            ref=FileRef(str(safe_file(self.root,name)),d[field]);pin(ref)
        baseline=read_json(safe_file(self.root,"baseline_verified.json"))
        need(baseline["protocol_sha256"]==self.protocol_id and baseline["passed"] is True,"baseline verification")
        pin(FileRef(str(safe_file(self.root,d["execution_path"])),d["execution_sha256"]))
        self.attempt_files=d["attempt_files_sha256"]
        for name,h in self.attempt_files.items():pin(FileRef(str(safe_file(self.root,name)),h))
        self.identity=HistoricalReplayIdentity(digest(asdict(self.manifest)),self.headers["protocol"],self.headers["summary"],
            self.headers["completed"],self.selection,self.models,digest(asdict(self.legacy_space)),digest(self.declared),self.fixture)
        from ..artifacts import sha_file
        producers={str(Path(__file__).with_name(k)):sha_file(Path(__file__).with_name(k))
                   for k in ("historical_sources.py","historical_contracts.py","historical_loader.py")}
        for relative in ("models/irra_cached.py","models/base.py","engines/evaluation.py"):
            path=Path(__file__).parent.parent/relative;producers[str(path)]=sha_file(path)
        self.space=ModelSpace("irra_fixture" if self.fixture else "irra",dict(self.models)["IRRA"].sha256,
            digest({"historical_identity":self.identity.id,"loader_producers":producers}),
            self.legacy_space.preprocess_sha256,digest({"tokenizer":asdict(self.legacy_space.tokenizer_source),"bpe":asdict(self.legacy_space.bpe)}),
            digest({"official_rank":asdict(self.legacy_space.ranking_source),"scoring":"CPU FP32 normalized q@g.T"}),"fp32")
        for name,h in producers.items():pin(FileRef(name,h))
        self.metadata_snapshot=self.verify_metadata()
    def pending(self):
        return sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*") if
                      "pending" in p.name or p.name.startswith(".pending"))
    def verify_metadata(self):
        need(not self.pending(),"uncommitted/pending historical batch; preserve and stop")
        expected=self.attempt_files
        actual={str(p.relative_to(self.root)) for p in (self.root/"attempts").rglob("*") if p.is_file()}
        need(actual==set(expected),"execution/attempt evidence missing or extra")
        inventory={str(p.relative_to(self.root)) for p in (self.root/"features").rglob("*") if p.is_file()}
        need(inventory==set(self.declared),"declared feature inventory missing/extra")
        return {p:checked(ref) for p,ref in self.refs.items()}
    def verify_unchanged(self):
        need(self.verify_metadata()==self.metadata_snapshot,"historical source SHA/mtime changed")
    def report(self):
        return {"schema":"tbps-historical-replay-verification-1","status":"METADATA_VERIFIED","identity":asdict(self.identity),
                "source_files_verified":len(self.refs),"declared_feature_files":len(self.declared),
                "gallery":len(self.manifest.gallery),"queries":len(self.manifest.queries),
                "features_loaded":0,"scores_computed":0,"metrics_replay_status":"PENDING"}
