"""冻结G/H来源与训练条件元数据只读描述；不反序列化权重或条件Tensor。"""
from dataclasses import dataclass
from pathlib import Path
import argparse
import json
from ..artifacts import checked, digest, json_bytes, read_json, sha_file
from ..contracts import FileRef
from ..compat.legacy_assets import describe_asset
from .base import need

G_SHA="f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e"
H_SHA="487f3061b6cba2ddabccdd49745a7b99c8c549f4541781b8814b299ef5f279df"
IRRA_SHA="9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009"

@dataclass(frozen=True, slots=True)
class AttackProvenance:
    attack_id: str
    checkpoint: FileRef
    training_protocol_sha256: str
    sources_sha256: str
    surrogate_sha256: str
    training_dataset: str
    execution_kind: str
    selection_sha256: str | None = None
    def __post_init__(self):
        from ..contracts import sha256
        need(self.attack_id in ("gv2","hv1") and type(self.checkpoint) is FileRef, "typed G/H provenance")
        for value in (self.training_protocol_sha256,self.sources_sha256,self.surrogate_sha256): sha256(value)
        need(self.training_dataset=="cuhk_pedes:attack_train:2048", "locked attack training source")
        need(self.execution_kind in ("cpu_fixture","legacy_unverified"), "explicit backend kind")
        if self.selection_sha256 is not None: sha256(self.selection_sha256)
        if self.execution_kind=="legacy_unverified":
            need(self.checkpoint.sha256==(G_SHA if self.attack_id=="gv2" else H_SHA),
                 "only frozen G10/H7 checkpoints supported")
            need(self.surrogate_sha256==IRRA_SHA, "wrong training surrogate")
            need((self.selection_sha256 is not None)==(self.attack_id=="hv1"), "H Validation Selection required")
    def as_dict(self):
        from dataclasses import asdict
        return asdict(self)

class LegacyAttackSources:
    def __init__(self, config_path):
        path=Path(config_path).resolve()
        self.config_ref=FileRef(str(path),sha_file(path))
        self.config=read_json(path,"tbps-legacy-attack-plugins-1")
        need(set(self.config)=={"schema","legacy_root","catalog","assets","runtime_sources",
             "surrogate_model","surrogate_sha256","image_preprocess","text_constraint_tokenizer",
             "execution_status","attack_budgets"}, "legacy attack source config fields")
        need(self.config["assets"]=={"gv2":"gv2_epoch10","hv1":"hv1_epoch7_selection"},
             "locked component asset indexes")
        need(Path(self.config["legacy_root"]).is_absolute(), "absolute legacy repository root required")
        self.root=Path(self.config["legacy_root"]).resolve()
        need(self.root.is_absolute() and "prototypes" not in self.root.parts, "legacy repository root required")
        need(self.config["surrogate_model"]=="irra" and self.config["surrogate_sha256"]==IRRA_SHA,
             "frozen IRRA training surrogate required")
        self.catalog_ref=FileRef(**self.config["catalog"])
        from dataclasses import asdict
        from .gv2 import BUDGET as g
        from .hv1 import BUDGET as h
        from ..contracts import AttackBudget
        need(self.config["attack_budgets"]=={"gv2":asdict(g),"hv1":asdict(h),
             "joint":asdict(AttackBudget(g.image,h.text))}, "fixed plugin budgets required")
        self.framework_sources={str(Path(__file__).parent/name):sha_file(Path(__file__).parent/name)
            for name in ("__init__.py","base.py","registry.py","gv2.py","hv1.py","joint.py",
                         "provenance.py","legacy_runtime.py","configuration.py","storage.py")}
        for name in ("config.py","contracts.py","artifacts.py","compat/legacy_assets.py"):
            path=Path(__file__).parent.parent/name
            self.framework_sources[str(path)]=sha_file(path)
        self.verify()
    def verify(self):
        before=checked(self.config_ref)
        for name,sha in self.framework_sources.items(): checked(FileRef(name,sha))
        checked(self.catalog_ref)
        catalog=read_json(self.catalog_ref.path,"tbps-legacy-catalog-1")
        self.assets={name:next(a for a in catalog["assets"] if a["id"]==asset_id)
                     for name,asset_id in self.config["assets"].items()}
        self.descriptions={k:describe_asset(v).as_dict() for k,v in self.assets.items()}
        need(all(d["status"]=="METADATA_VERIFIED" for d in self.descriptions.values()),
             "legacy assets NOT_READY; missing paths cannot be executable")
        need(self.assets["gv2"]["models"]["G"]==G_SHA and self.assets["hv1"]["models"]["H"]==H_SHA and
             self.assets["hv1"]["models"]["G"]==G_SHA and
             all(a["models"]["IRRA"]==IRRA_SHA for a in self.assets.values()), "component model origins mismatch")
        self.runtime_refs=tuple(FileRef(p,h) for p,h in self.config["runtime_sources"].items())
        need(self.runtime_refs, "explicit legacy runtime dependency closure required")
        for ref in self.runtime_refs: checked(ref)
        # Config must explicitly bind the original 77-token constraint tokenizer, never target tokenizer.
        t=self.config["text_constraint_tokenizer"]
        need(set(t)=={"source","bpe","context"} and t["context"]==77, "fixed H constraint tokenizer")
        from ..models.irra_source import OFFICIAL_IRRA_SHA
        need(OFFICIAL_IRRA_SHA==IRRA_SHA, "surrogate descriptor mismatch")
        for key in ("source","bpe"):
            ref=FileRef(**t[key]);checked(ref)
            need(ref in self.runtime_refs, "Tokenizer not in pinned runtime closure")
        require_paths=("prototypes/attribute_image_generator_v2_20261014/inference_once.py",
                       "prototypes/attribute_image_generator_v2_20261014/image_only_generator.py",
                       "prototypes/attribute_image_generator_v2_20261014/budget_projection.py",
                       "prototypes/attribute_text_policy_h_v1_20261015/phase5_joint/backend.py",
                       "prototypes/attribute_text_policy_h_v1_20261015/policy.py",
                       "prototypes/attribute_text_policy_h_v1_20261015/constraints.py",
                       "prototypes/attribute_text_policy_h_v1_20261015/text_frontend.py")
        need(all(str(self.root/p) in self.config["runtime_sources"] for p in require_paths),
             "runtime roots/source closure mismatch")
        need(checked(self.config_ref)==before, "source config changed")
        return self
    @property
    def fingerprint(self):
        return digest({"config":self.config_ref.sha256,"catalog":self.catalog_ref.sha256,
                       "descriptions":self.descriptions,"plugin_producers":self.framework_sources})
    def ref(self, component, name):
        data=self.assets[component]["files"][name]
        return FileRef(data["path"],data["sha256"])
    def protocol(self, component):
        return read_json(self.ref(component,"protocol").path)
    def provenance(self, component):
        return AttackProvenance(component,self.ref(component,"checkpoint"),
            self.descriptions[component]["protocol_digest"],self.fingerprint,IRRA_SHA,
            "cuhk_pedes:attack_train:2048","legacy_unverified",
            self.ref("hv1","selection").sha256 if component=="hv1" else None)
    def condition_record(self, row_id):
        """只读一条原始训练监督索引，不生成、选择或加载JVP张量。"""
        manifest_ref=self.ref("gv2","cache_manifest");checked(manifest_ref)
        manifest=read_json(manifest_ref.path,"gv2-training-cache-2")
        cpref=self.ref("gv2","cache_protocol");checked(cpref)
        cp=read_json(cpref.path,"gv2-full-prepare-protocol-2")
        entries=[e for e in manifest["entries"] if e["row_id"]==row_id]
        need(len(entries)==1, "unknown/duplicate condition identity")
        e=entries[0]
        mdref=FileRef(e["metadata_path"],e["metadata_sha256"])
        need(Path(mdref.path).resolve().is_relative_to(Path(manifest_ref.path).parent), "condition metadata escapes cache")
        checked(mdref);md=read_json(mdref.path)
        need(md["protocol_hash"]==digest(cp) and md["tensor_sha256"]==e["tensor_sha256"], "condition protocol/SHA")
        for key in ("row_id","image_rel_path","caption_index","split","role","pid","caption","K"):
            need(md[key]==e[key], "condition source identity mismatch")
        selected=md["source_selection"]["selected_attributes"]
        ids=[a["instance_id"] for a in selected];scores=[a["S_raw"] for a in selected]
        need(ids==md["selected_instance_ids"] and len(set(ids))==len(ids) and len(ids)==e["K"] and
             scores==sorted(scores,reverse=True) and md["source_selection"]["natural_k"]==e["K"],
             "original Top-K order mismatch")
        need(selected==md["source_selection"]["ranked_attributes"][:e["K"]], "selected prefix changed")
        selection=md["source_selection"];record=md["source_score_record"]
        need(selection["gamma"]==3 and selection["beta"]==1, "original Dynamic-K parameters changed")
        need(e["split"]=="train" and e["role"] in ("train","validation"), "training supervision only")
        need((record["row_id"],record["pid"],record["caption"],record["paired_image"])==
             (e["row_id"],e["pid"],e["caption"],e["image_rel_path"]), "score/image/Caption pairing changed")
        score_by_id={a["instance_id"]:a["S_raw"] for a in record["attributes"]}
        need(all(score_by_id[i]==s for i,s in zip(ids,scores)), "raw score provenance mismatch")
        return {"row_id":row_id,"image_rel_path":e["image_rel_path"],"source_image_sha256":md["source_image_sha256"],
                "caption_index":e["caption_index"],"caption":e["caption"],"split":e["split"],"role":e["role"],
                "K":e["K"],"selected_instance_ids":ids,"S_raw":scores,"attributes":selected,
                "metadata":mdref.__dict__ if hasattr(mdref,"__dict__") else {"path":mdref.path,"sha256":mdref.sha256},
                "tensor":{"path":e["tensor_path"],"sha256":e["tensor_sha256"]},
                "score_record_digest":digest(md["source_score_record"]),
                "historical_scoring_backwards_per_caption":md["source_score_record"]["retrieval_backward_count"],
                "this_run_scoring_backwards":0,"this_run_JVPs":0,
                "scope":"metadata only; tensor SHA is declared, tensor contents not re-audited"}
    def as_dict(self):
        return {"schema":"tbps-attack-plugin-source-report-1","source_sha256":self.fingerprint,
                "components":{k:self.provenance(k).as_dict() for k in self.assets},
                "runtime_source_files":len(self.runtime_refs),"plugin_producers":self.framework_sources,"model_loaded":False,"model_forwards":0,
                "preparation_executable":False,"equivalence":"real G/H output pending Stage 3B"}

def main(argv=None):
    p=argparse.ArgumentParser(description="CPU read-only G/H plugin source audit; no run/model option")
    p.add_argument("--config",required=True);p.add_argument("--condition-row",action="append",default=[])
    args=p.parse_args(argv)
    try:
        sources=LegacyAttackSources(args.config);report=sources.as_dict()
        report["condition_records"]=[sources.condition_record(r) for r in args.condition_row]
        print(json.dumps(report,ensure_ascii=False,indent=2))
        print("STAGE3A_PLUGIN_SOURCE_AUDIT_PASS")
    except (ValueError,TypeError,OSError,KeyError,StopIteration) as exc:
        p.exit(1,"PLUGIN_SOURCE_REJECTED "+str(exc)+"\n")

if __name__=="__main__": main()
