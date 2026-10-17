"""独立合成历史Store：真实Tensor格式/回执，但没有模型或正式Test数据。"""
from dataclasses import asdict
from pathlib import Path
import hashlib
import torch
from tbps.artifacts import FileRef,sha_file,json_bytes,digest,read_json
from tbps.engines.historical_sources import JointFullSources
from tbps.engines.historical_loader import JointFullLoader
from tbps.models.irra_cached import IRRACachedScorer

class ReplayFixture:
    def __init__(self,root):
        self.root=root;self.store=root/"joint";self.store.mkdir()
        self.ref=lambda p:FileRef(str(p),sha_file(p))
        def write(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(json_bytes(obj));return self.ref(path)
        self.write=write
        annotation=[]
        for i in range(12):
            image=root/"images"/f"{i}.jpg";image.parent.mkdir(exist_ok=True);image.write_bytes(str(i).encode())
            annotation.append({"split":"test","id":i//2+1,"file_path":f"{i}.jpg","captions":[f"A man {i} in black.",f"A man {i} with a hat."]})
        ann=write(root/"annotation.json",annotation)
        gallery=[{"index":i,"annotation_index":i,"row_id":f"cuhk:{i}:0","pid":i//2+1,"image":f"{i}.jpg",
                  "image_sha256":sha_file(root/"images"/f"{i}.jpg")} for i in range(12)]
        queries=[{"index":i*2+j,"row_id":f"cuhk:{i}:{j}","caption_index":j,"pid":i//2+1,"paired_index":i,"caption":caption}
                 for i,r in enumerate(annotation) for j,caption in enumerate(r["captions"])]
        models={}
        for name in ("G","H","IRRA"):
            path=root/"formal/training/checkpoints/visit_00014336/checkpoint.pt" if name=="H" else root/(name+".pt")
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(name.encode());models[name]=self.ref(path)
        groups={"Joint":{"mAP":50.0}}
        candidate={"checkpoint_sha256":models["H"].sha256,"cursor":14336,"groups":groups,"Joint_minus_ImageOnly_mAP_pp":-1.0}
        selection={"uses_official_Test":False,"candidates":[candidate],"selected_checkpoint_sha256":models["H"].sha256,
                   "cursor":14336,"groups":groups}
        sel=write(root/"formal/selections"/(digest(selection)+".json"),selection)
        val=write(root/"formal/validation"/models["H"].sha256/"summary.json",candidate)
        native=write(root/"native.json",{"fixture":"tokenizer/BPE/ranking source"})
        pre={"pixels":"fixture RGB","size":[384,128]}
        evidence={r.path:r.sha256 for r in (*models.values(),ann,native,sel,val)}
        evidence.update({str(root/"images"/f"{i}.jpg"):gallery[i]["image_sha256"] for i in range(12)})
        self.protocol={"schema":"hv1-joint-full-1","gallery":gallery,"queries":queries,"gallery_count":12,"query_count":24,
            "text_batch":16,"model_sha256":{k:v.sha256 for k,v in models.items()},"preprocess":pre,"epsilon":8/255,
            "all_queries_retained":True,"Test_feedback_for_selection":False,"source_sha256":{},
            "evidence_sha256":evidence,"runtime":{"torch":"2.5.1+cu124","cpu_threads":2},
            "H_selection":{"path":sel.path,"sha256":sel.sha256,"checkpoint":models["H"].path,
                           "checkpoint_sha256":models["H"].sha256,"selection":selection,
                           "proof":{r.path:r.sha256 for r in (sel,val,*models.values())}}}
        pr=write(self.store/"protocol.json",self.protocol);pid=digest(self.protocol)
        self.values={};declared={}
        for family in ("clean_gallery","image_gallery","clean_text","adv_text"):
            rows=gallery if family.endswith("gallery") else queries
            x=torch.zeros(len(rows),512)
            for i,row in enumerate(rows):x[i,(row["pid"]+(1 if family in ("image_gallery","adv_text") else 0))%512]=1.
            self.values[family]=x.clone()
            size=64 if family.endswith("gallery") else 16
            for start in range(0,len(rows),size):
                index=start//size;part=rows[start:start+size];unit=self.store/"features"/family/f"{index:06d}";unit.mkdir(parents=True)
                path=unit/"features.pt";torch.save(x[start:start+len(part)].clone(),path)
                intent={"protocol_sha256":pid,"family":family,"index":index,"rows":part}
                write(unit/"intent.json",intent)
                stats={"G_new_samples":0,"IRRA_image_new_samples":0,"IRRA_text_samples":0,"reference_reused":True}
                if family=="adv_text":
                    actions=[]
                    for row in part:
                        original=row["caption"];edited=original.replace("man","mаn")
                        actions.append({"original":original,"adversarial":edited,"H_forward":1,"eligible":True})
                    stats={"G_new_samples":0,"IRRA_image_new_samples":0,"IRRA_text_samples":len(actions),
                           "H_forward_samples":len(actions),"actions":actions}
                md=write(unit/"metadata.json",{**intent,"feature_sha256":sha_file(path),"stats":stats})
                write(unit/"receipt.json",{"protocol_sha256":pid,"metadata_sha256":md.sha256,"feature_sha256":sha_file(path)})
                for file in unit.iterdir():declared[str(file.relative_to(self.store))]=sha_file(file)
        self.summary={"schema":"hv1-joint-full-result-1","protocol_sha256":pid,"model_sha256":self.protocol["model_sha256"],
                      "gallery_count":12,"query_count":24,"all_queries_retained":True,"feature_files_sha256":declared}
        sr=write(self.store/"summary.json",self.summary)
        cv=write(self.store/"baseline_verified.json",{"protocol_sha256":pid,"passed":True})
        csv=self.store/"metrics.csv";csv.write_text("fixture only\n")
        ex=write(self.store/"attempts/1/execution.json",{"fixture":True})
        self.completed={"protocol_sha256":pid,"summary_sha256":sr.sha256,"method_audit_passed":True,
            "metrics_csv_sha256":sha_file(csv),"baseline_verified_sha256":cv.sha256,
            "execution_path":"attempts/1/execution.json","execution_sha256":ex.sha256,
            "attempt_files_sha256":{"attempts/1/execution.json":ex.sha256}}
        dr=write(self.store/"completed.json",self.completed)
        base={"schema":"tbps-stage4a-legacy-metadata-1","joint":{k:asdict(v) for k,v in zip(("protocol","summary","completed"),(pr,sr,dr))},
              "annotation":asdict(ann),"image_root":str(root/"images"),
              "legacy_space":{"model_id":"irra","checkpoint_sha256":models["IRRA"].sha256,"preprocess_sha256":digest(pre),
                "tokenizer_source":asdict(native),"bpe":asdict(native),"ranking_source":asdict(native)}}
        self.base=base;self.base_path=root/"base.json";br=write(self.base_path,base)
        self.config_path=root/"config.json"
        write(self.config_path,{"schema":"tbps-historical-replay-sources-1","profile":"cpu_fixture","root":str(self.store),
            "stage4a_sources":asdict(br),"attack_sources":asdict(native),"expected_counts":{"gallery":12,"query":24}})
    def sources(self):return JointFullSources(self.config_path)
    @staticmethod
    def validate(caption,record):
        if record["original"]!=caption or record["adversarial"]!=caption.replace("man","mаn"):
            raise ValueError("fixture Unicode/Caption mismatch")
        return record["adversarial"]
    def load(self):
        sources=self.sources();loader=JointFullLoader(sources,self.validate)
        report,values=loader.verify_features(retain=True)
        return sources,report,values
    def resign(self):
        self.base["joint"]["summary"]=asdict(self.write(self.store/"summary.json",self.summary))
        self.completed["summary_sha256"]=self.base["joint"]["summary"]["sha256"]
        self.base["joint"]["completed"]=asdict(self.write(self.store/"completed.json",self.completed))
        br=self.write(self.base_path,self.base)
        config=read_json(self.config_path);config["stage4a_sources"]=asdict(br);self.write(self.config_path,config)
