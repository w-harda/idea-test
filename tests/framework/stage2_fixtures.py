"""Stage 2小型合成official annotation/内部来源闭包；无真实模型。"""
from pathlib import Path
from tbps.artifacts import json_bytes, sha_file, digest, read_json
from tbps.contracts import FileRef
from tbps.datasets.cuhk_pedes import CuhkPedes

def write(root,name,value,raw=False):
    p=Path(root)/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_bytes(value if raw else json_bytes(value));return FileRef(str(p),sha_file(p))

def dataset(root):
    rows=[
        {"split":"train","id":1,"file_path":"train/a.jpg","captions":["red red shirt","  red shirt  "]},
        {"split":"train","id":2,"file_path":"train/b.jpg","captions":["blue shirt"]},
        {"split":"train","id":3,"file_path":"train/c.jpg","captions":["green shirt"]},
        {"split":"val","id":4,"file_path":"val/a.jpg","captions":["val shirt"]},
        {"split":"test","id":5,"file_path":"test/a.jpg","captions":["red shirt","red coat"]},
        {"split":"test","id":5,"file_path":"test/b.jpg","captions":["  red shirt  "]},
        {"split":"test","id":6,"file_path":"test/c.jpg","captions":["blue shirt"]}]
    images=Path(root)/"images"
    for r in rows:write(images,r["file_path"],r["file_path"].encode(),True)
    ann=write(root,"annotation.json",rows)
    return CuhkPedes(ann,images),rows

def historical(root,ds):
    root=Path(root);refs={};identities={"train":[],"validation":[]}
    def put(key,data):refs[key]=write(root,key+".json",data);return data
    records=[]
    for r in ds.rows[:3]:
        for j,caption in enumerate(r.captions):
            records.append({"row_id":r.query_id(j),"pid":int(r.pid.value),"caption":caption})
    internal=put("internal_split",{"records":records,"train_indices":[0,1,2],"validation_indices":[3],"train_pids":[1,2],"validation_pids":[3]})
    plans=[]
    for i,role in ((0,"train"),(1,"train"),(2,"validation")):
        r=ds.rows[i]
        plans.append({"row_id":r.query_id(0),"image_rel_path":r.image_rel_path,"caption_index":0,"split":"train",
                      "role":role,"pid":int(r.pid.value),"caption":r.captions[0],"K":1})
    gp=put("g_protocol",{"schema":"gv2-full-prepare-protocol-2","samples":plans,"target_counts":{"train":2,"validation":1},
                        "dataset_sha256":{ds.annotation.path:ds.annotation.sha256,refs["internal_split"].path:refs["internal_split"].sha256}})
    entries=[]
    for plan in plans:
        image_hash=sha_file(ds.image_path(plan["image_rel_path"]))
        md=write(root,"samples/"+plan["row_id"].replace(":","_")+"/metadata.json",
            {**plan,"source_image_sha256":image_hash,"protocol_hash":digest(gp),"tensor_sha256":"5"*64})
        e={**plan,"metadata_path":md.path,"metadata_sha256":md.sha256,"tensor_sha256":"5"*64}
        entries.append(e)
        identities[plan["role"]].append({k:v for k,v in plan.items() if k!="K"}|
            {"image_sha256":image_hash,"metadata_sha256":md.sha256,"condition_sha256":"5"*64})
    put("g_manifest",{"schema":"gv2-training-cache-2","counts":{"train":2,"validation":1},"entries":entries,
                     "protocol_path":refs["g_protocol"].path,"protocol_sha256":refs["g_protocol"].sha256})
    put("g_completed",{"method_audit_passed":True,"manifest_sha256":refs["g_manifest"].sha256})
    receipt={}
    for role,ids in identities.items():
        p={"schema":"hv1-formal-data-1","role":role,"identities":ids,"training_sources":{
            "manifest_sha256":refs["g_manifest"].sha256,"raw_annotation_sha256":ds.annotation.sha256,
            "internal_split_sha256":refs["internal_split"].sha256}}
        q=put("h_"+role+"_qualification",{"protocol":p,"protocol_sha256":digest(p),
            "entries":[{"identity":i,"caption":i["caption"],"eligible":True,"attributes":[{"char_start":0,"char_end":3 if role=="train" and i["pid"]==1 else 4 if i["pid"]==2 else 5,"raw_text":i["caption"].split()[0]}]} for i in ids]})
        receipt[role+"_sha256"]=refs["h_"+role+"_qualification"].sha256
        put("h_gallery_"+role,{"schema":"hv1-formal-gallery-1","role":role,"identities":ids})
    put("h_qualification_completed",receipt)
    hp=put("h_training",{"schema":"hv1-formal-SCST-training-1","validation_feedback":False,"train_identities":identities["train"],
        "epoch_size":2,"config":{"epochs":1},"query_order":[i["row_id"] for i in identities["train"]],
        "eligible_by_row":{i["row_id"]:True for i in identities["train"]},"qualification_sha256":digest(read_json(refs["h_train_qualification"].path)),
        "gallery_protocol_sha256":digest(read_json(refs["h_gallery_train"].path))})
    put("h_validation",{"schema":"hv1-formal-validation-1","identities":identities["validation"],
         "training_protocol_sha256":digest(hp),"qualification_sha256":digest(read_json(refs["h_validation_qualification"].path)),
         "validation_gallery_protocol_sha256":digest(read_json(refs["h_gallery_validation"].path))})
    return refs

def mutate(refs,key,fn):
    d=read_json(refs[key].path);fn(d);Path(refs[key].path).write_bytes(json_bytes(d))
    refs[key]=FileRef(refs[key].path,sha_file(refs[key].path))
