"""历史资产只读桥：只用stdlib读字节/JSON，不导入历史模块，不加载Tensor或重算排名。"""
from dataclasses import dataclass
from pathlib import Path
import argparse
import json
from ..artifacts import ArtifactError, ArtifactStore, checked, digest, json_bytes, read_json, sha_file
from ..contracts import FileRef, sha256, text

KINDS = {
    "gv2_checkpoint":{"checkpoint","sidecar","protocol","cache_manifest","cache_protocol"},
    "hv1_selection":{"checkpoint","sidecar","protocol","selection","intent","sample_log"},
    "gv2_full":{"protocol","summary","completed","metrics","clean_verified","compatibility"},
    "joint_full":{"protocol","summary","completed","metrics","baseline_verified","selection"},
    "ablation_full":{"protocol","summary","completed","metrics","aggregate","audit_log"},
    "ap_metadata":{"metadata","summary"},
}
SCHEMAS={"gv2_full":"gv2-official-evaluation-1","joint_full":"hv1-joint-full-1",
         "ablation_full":"hv1-joint-ablation-1"}
FORMATS=frozenset({"json","opaque","log","csv"})

def require(condition, message):
    if not condition:raise ArtifactError(message)

def pointer(obj, path):
    if type(path) is not str or (path and not path.startswith("/")):raise ArtifactError("JSON pointer required")
    for part in path.split("/")[1:] if path else ():
        part=part.replace("~1","/").replace("~0","~")
        try:obj=obj[int(part)] if type(obj) is list else obj[part]
        except (KeyError,IndexError,ValueError,TypeError) as exc:raise ArtifactError("missing JSON pointer: "+path) from exc
    return obj

@dataclass(frozen=True, slots=True)
class BridgeDescription:
    canonical: bytes
    def as_dict(self):return json.loads(self.canonical)
    def require_feature_reuse(self, model_checkpoint_sha256, protocol_sha256):
        d=self.as_dict()
        if model_checkpoint_sha256!=d["models"].get("IRRA") or protocol_sha256!=d.get("protocol_digest"):
            raise ArtifactError("cross-model/protocol representation reuse forbidden")
        raise ArtifactError("Stage 1 verifies metadata only; feature reuse requires a separately audited Model Adapter")

def _catalog_asset(asset):
    require(type(asset) is dict and set(asset)=={"id","kind","files","models","datasets","checks","reuse_scope"},"asset fields")
    text(asset["id"],"legacy asset ID");require(asset["kind"] in KINDS,"unknown legacy kind")
    files=asset["files"];require(type(files) is dict and KINDS[asset["kind"]]<=set(files),"required historical evidence missing from catalog")
    for name,ref in files.items():
        text(name,"file key")
        require(type(ref) is dict and set(ref)=={"path","sha256","format"} and ref["format"] in FORMATS,"file ref fields/format")
        FileRef(ref["path"],ref["sha256"])
    require(type(asset["models"]) is dict and asset["models"],"model provenance required")
    for key,h in asset["models"].items():text(key,"model name");sha256(h)
    require(type(asset["datasets"]) is dict and asset["datasets"],"dataset provenance required")
    for k,v in asset["datasets"].items():text(k,"dataset role");text(v,"dataset source")
    require(asset["reuse_scope"]=="read_only_reporting","Stage 1 cannot authorize executable feature reuse")
    require(type(asset["checks"]) is list,"explicit evidence checks required")
    for check in asset["checks"]:
        require(type(check) is dict and set(check) in (
            {"file","pointer","equals"},{"file","pointer","equals_digest"},{"file","pointer","equals_file_sha"},
            {"file","pointer","equals_pointer"}),"check fields")
        require(check["file"] in files and files[check["file"]]["format"]=="json","check must address JSON file")
        text(check["pointer"],"check pointer")
        if "equals_pointer" in check:
            other=check["equals_pointer"]
            require(type(other) is dict and set(other)=={"file","pointer"} and other["file"] in files and
                    files[other["file"]]["format"]=="json","pointer relation requires JSON file")
            text(other["pointer"],"relation pointer")
        for k in ("equals_digest","equals_file_sha"):
            if k in check:require(check[k] in files,"unknown relation file")
            if k=="equals_digest" and k in check:require(files[check[k]]["format"]=="json","digest requires JSON")

def _identities(protocol):
    gallery=protocol["gallery"];queries=protocol["queries"]
    require(type(gallery) is list and type(queries) is list and gallery and queries,"ordered identity arrays required")
    require(len(gallery)==protocol["gallery_count"] and len(queries)==protocol["query_count"],"identity counts mismatch")
    require([g["index"] for g in gallery]==list(range(len(gallery))),"gallery order mismatch")
    require([q["index"] for q in queries]==list(range(len(queries))),"query order mismatch")
    require(len({g["image"] for g in gallery})==len(gallery),"duplicate Gallery image")
    require(len({q["row_id"] for q in queries})==len(queries),"duplicate Query identity")
    for g in gallery:sha256(g["image_sha256"])
    for q in queries:
        i=q["paired_index"]
        require(type(i) is int and 0<=i<len(gallery) and gallery[i]["pid"]==q["pid"],"wrong Query/Gallery pairing")
        text(q["caption"],"original Caption")
    return digest({"gallery":gallery,"queries":queries})

def _semantic(asset, data, refs):
    kind=asset["kind"];models=asset["models"]
    if kind=="gv2_checkpoint":
        p,s=data["protocol"],data["sidecar"]
        require(p["config"]["schema"]=="gv2-epoch-config-1","G training protocol schema")
        require(s["epoch"]==10 and s["sha256"]==refs["checkpoint"]["sha256"],"G Epoch10/checkpoint mismatch")
        require(s["contract_sha256"]==digest(p),"G checkpoint contract mismatch")
        require(p["cache_manifest_sha256"]==refs["cache_manifest"]["sha256"] and
                p["cache_protocol_sha256"]==refs["cache_protocol"]["sha256"],"G training cache provenance mismatch")
        require(models["G"]==s["sha256"] and models["IRRA"]==p["config"]["surrogate"]["checkpoint_sha256"],"G model provenance mismatch")
    elif kind=="hv1_selection":
        p,s,selection=data["protocol"],data["sidecar"],data["selection"]
        require(p["schema"]=="hv1-formal-SCST-training-1" and s["schema"]=="hv1-formal-periodic-checkpoint-1","H schema")
        require(selection["uses_official_Test"] is False and p["validation_feedback"] is False,"H Test/Validation reward leakage")
        require(s["protocol_sha256"]==digest(p) and s["checkpoint_sha256"]==refs["checkpoint"]["sha256"]==models["H"],"H checkpoint provenance mismatch")
        require(s["intent_sha256"]==refs["intent"]["sha256"] and s["sample_log_sha256"]==refs["sample_log"]["sha256"],"H checkpoint log/intent mismatch")
        candidates=selection["candidates"]
        require(type(candidates) is list and candidates,"selection candidate metadata required")
        best=min(candidates,key=lambda c:(c["groups"]["Joint"]["mAP"],c["cursor"]))
        require(best["checkpoint_sha256"]==selection["selected_checkpoint_sha256"]==models["H"],"Selection is not stored Validation argmin")
        require(best["cursor"]==selection["cursor"]==s["cursor"] and best["groups"]["Joint"]["mAP"]==selection["Joint_mAP"],"Selection score/cursor mismatch")
    elif kind in SCHEMAS:
        p,s,d=data["protocol"],data["summary"],data["completed"]
        require(p["schema"]==SCHEMAS[kind] and d["method_audit_passed"] is True,"Full schema/completion")
        require(d["protocol_sha256"]==digest(p) and d["summary_sha256"]==refs["summary"]["sha256"],"Full completion binding")
        order_digest=_identities(p)
        if kind!="gv2_full":
            require(s["protocol_sha256"]==digest(p) and s["model_sha256"]==p["model_sha256"]==models,"Full model/protocol mismatch")
            require(d["metrics_csv_sha256"]==refs["metrics"]["sha256"],"CSV receipt mismatch")
        if kind=="gv2_full":
            require(s["checkpoint_sha256"]==p["checkpoint_sha256"] and models["G"]==p["checkpoint_sha256"]["10"],"G Full checkpoints mismatch")
            require(s["Text_Query_unchanged"] is True and s["text_features_shared"] is True,"G Full text protocol mismatch")
            require(data["clean_verified"]["passed"] is True,"Clean verification failed")
            require(data["compatibility"]["schema"]=="gv2-storage-fix-compatibility-1" and
                    data["compatibility"]["legacy_protocol_sha256"]==digest(p) and
                    data["compatibility"]["legacy_protocol_file_sha256"]==refs["protocol"]["sha256"],"storage compatibility mismatch")
        elif kind=="joint_full":
            require(p["Test_feedback_for_selection"] is False and s["selection_uses_Test"] is False and
                    p["all_queries_retained"] is True and s["all_queries_retained"] is True,"Joint selection/query protocol")
            require(data["baseline_verified"]["passed"] is True and d["baseline_verified_sha256"]==refs["baseline_verified"]["sha256"],"Joint baseline binding")
            require(p["H_selection"]["sha256"]==refs["selection"]["sha256"] and
                    data["selection"]["selected_checkpoint_sha256"]==models["H"],"Joint Selection provenance")
        else:
            require(p["Test_selection"] is False and s["Test_model_selection"] is False,"Ablation cannot select on Test")
            require(d["aggregate_csv_sha256"]==refs["aggregate"]["sha256"],"aggregate receipt mismatch")
            require(len(s["trials"])==len(p["trials"])==11,"Ablation trial coverage mismatch")
            log=Path(refs["audit_log"]["path"]).read_text(encoding="utf-8")
            require("ABLATION_CPU_AUDIT PASS" in log.splitlines() and "ABLATION_PIPELINE_SUCCESS" in log.splitlines(),"missing historical audit/pipeline success markers")
        if kind!="ablation_full":
            require(s["gallery_count"]==p["gallery_count"] and s["query_count"]==p["query_count"],"Full result identity counts")
        return order_digest
    elif kind=="ap_metadata":
        m=data["metadata"]
        require(m["victim"]["irra_checkpoint_sha256"]==models["IRRA"],"AP victim provenance mismatch")
        require(data["summary"]["metadata"]==m and data["summary"]["scope"]=="full" and data["summary"]["smoke_only"] is False,"AP summary provenance mismatch")
        # AP只是metadata级来源索引；不把缺少完成回执的旧Baseline升级成新审计通过。
    return None

def describe_asset(asset):
    _catalog_asset(asset)
    files=asset["files"]
    missing=[r["path"] for r in files.values() if not Path(r["path"]).is_file()]
    common={"id":asset["id"],"kind":asset["kind"],"models":asset["models"],"datasets":asset["datasets"],
            "reuse_scope":"read_only_reporting","executable":False,
            "limitations":["No model/Tensor loading, ranking, optimizer/RNG restore or exhaustive feature/source audit.",
                           "IRRA features are not transferable to RaSa/APTM or another protocol."]}
    if missing:
        return BridgeDescription(json_bytes({**common,"status":"NOT_READY","missing":missing,
                                            "files":files,"protocol_digest":None}))
    before={key:checked(FileRef(ref["path"],ref["sha256"])) for key,ref in files.items()}
    data={key:read_json(ref["path"]) for key,ref in files.items() if ref["format"]=="json"}
    try:
        for c in asset["checks"]:
            actual=pointer(data[c["file"]],c["pointer"])
            expected=c.get("equals")
            if "equals_digest" in c:expected=digest(data[c["equals_digest"]])
            if "equals_file_sha" in c:expected=files[c["equals_file_sha"]]["sha256"]
            if "equals_pointer" in c:
                other=c["equals_pointer"];expected=pointer(data[other["file"]],other["pointer"])
            require(json_bytes(actual)==json_bytes(expected),"catalog assertion failed: "+c["file"]+c["pointer"])
        order_digest=_semantic(asset,data,files)
    except (KeyError,TypeError,IndexError) as exc:raise ArtifactError("missing/invalid legacy schema field") from exc
    after={key:checked(FileRef(ref["path"],ref["sha256"])) for key,ref in files.items()}
    require(before==after,"historical evidence changed during read-only inspection")
    return BridgeDescription(json_bytes({**common,"status":"METADATA_VERIFIED","files":after,
        "protocol_digest":digest(data["protocol"]) if "protocol" in data else None,
        "ordered_identity_digest":order_digest,"verification_scope":"Pinned bytes + declared JSON relations; historical audit log claim where supplied.",
        "full_feature_audit_performed":False}))

def bridge_manifest(catalog):
    require(type(catalog) is dict and set(catalog)=={"schema","baseline_commit","assets"} and
            catalog["schema"]=="tbps-legacy-catalog-1","catalog schema/fields")
    require(type(catalog["baseline_commit"]) is str and len(catalog["baseline_commit"])==40 and
            all(c in "0123456789abcdef" for c in catalog["baseline_commit"]),"Git baseline SHA")
    assets=catalog["assets"];require(type(assets) is list and assets,"asset list required")
    for asset in assets:_catalog_asset(asset)
    require(len({a["id"] for a in assets})==len(assets),"duplicate asset ID")
    return {"schema":"tbps-legacy-bridge-1","baseline_commit":catalog["baseline_commit"],
            "catalog_sha256":digest(catalog),"assets":[describe_asset(a).as_dict() for a in assets]}

def main(argv=None):
    parser=argparse.ArgumentParser(description="CPU read-only historical descriptor; no model/metric execution")
    parser.add_argument("--catalog",required=True)
    parser.add_argument("--output-base")
    parser.add_argument("--run-id")
    args=parser.parse_args(argv)
    if bool(args.output_base)!=bool(args.run_id):parser.error("--output-base and --run-id must be supplied together")
    try:
        catalog_path=Path(args.catalog).resolve()
        catalog_ref=FileRef(str(catalog_path),sha_file(catalog_path))
        result=bridge_manifest(read_json(catalog_path))
        checked(catalog_ref)
        if args.output_base:
            store=ArtifactStore(Path(args.output_base)/"bridges",new_base=args.output_base)
            package=Path(__file__).resolve().parents[1]
            producers=tuple(FileRef(str(p),sha_file(p)) for p in
                            (Path(__file__).resolve(),package/"contracts.py",package/"artifacts.py"))
            store.publish(args.run_id,{"bridge_manifest.json":json_bytes(result)},
                          {"schema":"tbps-bridge-publication-1","catalog_sha256":result["catalog_sha256"],
                           "producer_sha256":{p.path:p.sha256 for p in producers}},
                          sources=(catalog_ref,*producers))
    except (ValueError,TypeError,OSError) as exc:parser.exit(1,"BRIDGE_REJECTED "+str(exc)+"\n")
    print(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2))
    if any(a["status"]=="NOT_READY" for a in result["assets"]):parser.exit(2,"BRIDGE_NOT_READY\n")
    print("BRIDGE_METADATA_VERIFIED_NO_MODEL_EXECUTION")

if __name__=="__main__":main()
