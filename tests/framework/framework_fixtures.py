"""合成Fixture；不导入历史模块，不包含真实模型/数据。"""
from pathlib import Path
from tbps.artifacts import json_bytes, sha_file, digest
H="1"*64
I="2"*64
G="3"*64
C="4"*64

def write(root,name,value,raw=False):
    p=Path(root)/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_bytes(value if raw else json_bytes(value))
    return p

def ref(path):return {"path":str(path),"sha256":sha_file(path)}

def config(root):
    cp={"path":str(Path(root)/"not-ready.pt"),"sha256":I,"model_id":"irra","model_training_dataset":"CUHK-PEDES official train"}
    manifest=ref(write(root,"manifest.json",{"schema":"fixture"}))
    return {"schema":"tbps-experiment-1","attack_method":"gv2","surrogate_model":"irra","victim_model":"irra",
        "attack_train_dataset":{"id":"cuhk_pedes","split":"train","role":"attack_train","manifest":manifest},
        "attack_validation_dataset":{"id":"cuhk_pedes","split":"train","role":"attack_validation","manifest":manifest},
        "attack_test_dataset":{"id":"cuhk_pedes","split":"test","role":"test","manifest":manifest},
        "attack_checkpoint":{"path":str(Path(root)/"g.pt"),"sha256":G,"attack_training_dataset":"cuhk_pedes",
           "surrogate_model":"irra","surrogate_checkpoint_sha256":I,"selection_dataset":"cuhk_pedes","uses_test_selection":False},
        "surrogate_checkpoint":dict(cp),"victim_checkpoint":dict(cp),"attack_mode":"white_box",
        "attack_budget":{"image":{"norm":"linf","value":8/255,"pixel_space":"rgb_0_1"},"text":None},"seed":42,
        "permissions":{"train":["train_pairs","source_gradients"],"attack_time":["public_images"]}}

def asset(root,kind="gv2_full"):
    a={"id":kind,"kind":kind,"files":{},"models":{"G":G,"IRRA":I},"datasets":{"evaluation":"fixture-only"},"checks":[],"reuse_scope":"read_only_reporting"}
    def add(key,value,fmt="json"):
        p=write(root,key+(".json" if fmt=="json" else ".bin"),value,fmt!="json")
        a["files"][key]={**ref(p),"format":fmt}
        return value
    def update(key,value):return add(key,value)
    a["_add"]=add
    return a

def finish(a):
    a.pop("_add",None);return a

def gallery_protocol(schema):
    return {"schema":schema,"gallery_count":2,"query_count":2,
        "gallery":[{"index":0,"image":"a.jpg","pid":1,"image_sha256":H},{"index":1,"image":"b.jpg","pid":2,"image_sha256":H}],
        "queries":[{"index":0,"row_id":"q0","paired_index":0,"pid":1,"caption":"red shirt"},
                   {"index":1,"row_id":"q1","paired_index":1,"pid":2,"caption":"blue shirt"}]}

def full(root,kind="gv2_full"):
    a=asset(root,kind);add=a["_add"]
    schema={"gv2_full":"gv2-official-evaluation-1","joint_full":"hv1-joint-full-1","ablation_full":"hv1-joint-ablation-1"}[kind]
    p=gallery_protocol(schema)
    s={"gallery_count":2,"query_count":2,"protocol_sha256":None}
    if kind=="gv2_full":
        p["checkpoint_sha256"]={"10":G}
        s.update(checkpoint_sha256={"10":G},Text_Query_unchanged=True,text_features_shared=True)
        add("clean_verified",{"passed":True})
    else:
        a["models"]["H"]=H;p["model_sha256"]=a["models"];s["model_sha256"]=a["models"]
    if kind=="joint_full":
        sel=add("selection",{"selected_checkpoint_sha256":H})
        p.update(Test_feedback_for_selection=False,all_queries_retained=True,H_selection={"sha256":a["files"]["selection"]["sha256"]})
        s.update(selection_uses_Test=False,all_queries_retained=True)
        add("baseline_verified",{"passed":True})
    if kind=="ablation_full":
        p.update(Test_selection=False,trials=[{"key":str(i)} for i in range(11)])
        s.update(Test_model_selection=False,trials={str(i):{} for i in range(11)})
        add("aggregate",b"fixture", "csv")
        add("audit_log",b"ABLATION_CPU_AUDIT PASS\nABLATION_PIPELINE_SUCCESS\n","log")
    add("protocol",p)
    if kind=="gv2_full":
        add("compatibility",{"schema":"gv2-storage-fix-compatibility-1","legacy_protocol_sha256":digest(p),
                             "legacy_protocol_file_sha256":a["files"]["protocol"]["sha256"]})
    s["protocol_sha256"]=digest(p);add("summary",s);add("metrics",b"group,mAP\nclean,100\n","csv")
    d={"method_audit_passed":True,"protocol_sha256":digest(p),"summary_sha256":a["files"]["summary"]["sha256"]}
    if kind!="gv2_full":d["metrics_csv_sha256"]=a["files"]["metrics"]["sha256"]
    if kind=="joint_full":d["baseline_verified_sha256"]=a["files"]["baseline_verified"]["sha256"]
    if kind=="ablation_full":d["aggregate_csv_sha256"]=a["files"]["aggregate"]["sha256"]
    add("completed",d)
    return finish(a)

def checkpoint(root,kind="gv2_checkpoint"):
    a=asset(root,kind);add=a["_add"]
    add("checkpoint",b"not-a-real-checkpoint","opaque")
    h=a["files"]["checkpoint"]["sha256"]
    if kind=="gv2_checkpoint":
        a["models"]["G"]=h
        add("cache_manifest",{"fixture":1});add("cache_protocol",{"fixture":2})
        p=add("protocol",{"config":{"schema":"gv2-epoch-config-1","surrogate":{"checkpoint_sha256":I}},
            "cache_manifest_sha256":a["files"]["cache_manifest"]["sha256"],
            "cache_protocol_sha256":a["files"]["cache_protocol"]["sha256"]})
        add("sidecar",{"epoch":10,"sha256":h,"contract_sha256":digest(p)})
    else:
        a["models"]["H"]=h
        p=add("protocol",{"schema":"hv1-formal-SCST-training-1","validation_feedback":False})
        add("intent",{"fixture":"intent"});add("sample_log",b"log","opaque")
        add("sidecar",{"schema":"hv1-formal-periodic-checkpoint-1","checkpoint_sha256":h,
             "protocol_sha256":digest(p),"intent_sha256":a["files"]["intent"]["sha256"],
             "sample_log_sha256":a["files"]["sample_log"]["sha256"],"cursor":14336})
        add("selection",{"uses_official_Test":False,"selected_checkpoint_sha256":h,"cursor":14336,"Joint_mAP":50,
                         "candidates":[{"checkpoint_sha256":h,"cursor":14336,"groups":{"Joint":{"mAP":50}}},
                                       {"checkpoint_sha256":H,"cursor":2048,"groups":{"Joint":{"mAP":60}}}]})
    return finish(a)

def mutate(a,key,fn):
    import json
    path=Path(a["files"][key]["path"]);d=json.loads(path.read_text());fn(d);path.write_bytes(json_bytes(d))
    a["files"][key]["sha256"]=sha_file(path)
    return a
