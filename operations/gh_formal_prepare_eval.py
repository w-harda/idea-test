"""正式B训练及CPU审计完成后，冻结独立比较配置；不改训练配置，不运行评分或CUDA。"""
import argparse,copy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from tbps.artifacts import ArtifactStore,FileRef,read_json,json_bytes,sha_file
from tbps.alternating.contracts import require
from tbps.alternating.formal_sources import load_formal,derived_native,ref

def prepare(path,output):
    f=load_formal(path);training=Path(f["outputs"]["train"])
    require((training/"generated/completed.json").is_file(),"B training not complete; no evaluation protocol created")
    ArtifactStore(training,new_base=f["output_base"]).verify("generated",
        expected_provenance={"formal_config":sha_file(path),"scope":"train"})
    audit=ArtifactStore(f["audits"]["train"],new_base=f["output_base"])
    audit.verify("completed",expected_provenance={"formal_configuration":sha_file(path),"scope":"train"})
    a=read_json(audit.root/"completed/report.json");g=read_json(training/"generated/report.json")
    require(a["status"]=="GH_B_FORMAL_CPU_AUDIT_PASS" and g["complete"]
        and g["cursor"]["round"]==f["train_rounds"]==a["rounds"][-1]["cursor"]["round"],"final B training/audit")
    name=f"round_{f['train_rounds']:04d}"
    ArtifactStore(training/"rounds",new_base=f["output_base"]).verify(name)
    ck=training/"rounds"/name/"state.pt"
    target=Path(output).resolve()
    children={s:str(target.with_name(target.stem+"."+s+".native.json")) for s in ("pilot","train")}
    require(not target.exists() and not any(Path(v).exists() for v in children.values()),"evaluation protocol immutable")
    require(not Path(f["evaluation"]["output_base"]).exists(),"evaluation already consumed")
    e=copy.deepcopy(f);e.pop("_config_path")
    e["evaluation"]["authorized"]=True;e["native_configs"]=children
    e["references"].update(comparison_B_checkpoint=ref(ck),original_training_configuration=ref(path),
        comparison_training_audit=ref(audit.root/"completed/report.json"),
        comparison_training_complete=ref(training/"generated/manifest.json"))
    e["tools"][str(Path(__file__).resolve())]=sha_file(__file__)
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("xb") as out:out.write(json_bytes(e))
    e["_config_path"]=str(target)
    for scope,p in children.items():
        with Path(p).open("xb") as out:out.write(json_bytes(derived_native(e,scope)))
    return {"status":"UNIFIED_EVALUATION_PROTOCOL_PREPARED_NOT_EXECUTED","configuration":ref(target),
        "B_checkpoint":ref(ck),"baseline_H60":e["references"]["H60"],"GPU":False,"scoring":False}
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--training-config",required=True);ap.add_argument("--output",required=True)
    a=ap.parse_args();print(prepare(a.training_config,a.output))
