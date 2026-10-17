"""原Joint Full Unicode动作只读桥；不加载H、不读取features.pt、不重新选择字符。"""
from dataclasses import asdict
from ..artifacts import FileRef,checked,read_json,json_bytes
from ..contracts import PublicTextInput
from ..attacks.base import TextRequest,AttackBinding,output
from ..attacks.hv1 import BUDGET
from .contracts import need
from .store import utf8_sha
from ..engines.historical_sources import JointFullSources,safe_file

class HistoricalTextSource:
    def __init__(self,sources,provenance,validate):
        need(type(sources)is JointFullSources and callable(validate),"verified original Joint source and native validator")
        need(provenance.attack_id=="hv1" and provenance.checkpoint==dict(sources.models)["H"],"frozen selected H mismatch")
        need(sources.fixture==(provenance.execution_kind=="cpu_fixture"),"real/mock text source separation")
        if not sources.fixture:
            need(provenance.selection_sha256==sources.selection.sha256 and
                 provenance.training_protocol_sha256==read_json(provenance.checkpoint.path.replace("checkpoint.pt","checkpoint.json"))["protocol_sha256"],
                 "original H Selection/training protocol binding")
        self.sources,self.provenance,self.validate=sources,provenance,validate
        self.file_snapshot={};self.count=0;self.batches=0;self.complete=False
    def iter_outputs(self):
        s=self.sources;s.verify_unchanged();self.file_snapshot={};self.count=0;self.batches=0;self.complete=False
        rows=s.protocol["queries"];batch=s.protocol["text_batch"]
        for start in range(0,len(rows),batch):
            index=start//batch;block=rows[start:start+batch];base=f"features/adv_text/{index:06d}"
            refs={n:FileRef(str(safe_file(s.root,base+"/"+n)),s.declared[base+"/"+n])
                  for n in ("metadata.json","intent.json","receipt.json")}
            for ref in refs.values():self.file_snapshot[ref.path]=checked(ref)
            md=read_json(refs["metadata.json"].path);intent=read_json(refs["intent.json"].path);receipt=read_json(refs["receipt.json"].path)
            expected={"protocol_sha256":s.protocol_id,"family":"adv_text","index":index,"rows":block}
            need(intent==expected and all(md[k]==v for k,v in expected.items()),"historical text batch identity/order")
            need(md["feature_sha256"]==s.declared[base+"/features.pt"] and receipt==
                 {"protocol_sha256":s.protocol_id,"metadata_sha256":refs["metadata.json"].sha256,"feature_sha256":md["feature_sha256"]},
                 "historical text receipt/SHA")
            actions=md["stats"]["actions"];need(len(actions)==len(block),"all historical Query actions required")
            for row,action in zip(block,actions):
                adv=self.validate(row["caption"],action)
                request=TextRequest(PublicTextInput(row["caption"]),AttackBinding(s.identity.manifest_sha256,row["row_id"],"text",utf8_sha(row["caption"])))
                trace={"stage":"historical_text_reuse","action":action,
                       "original_utf8_sha256":utf8_sha(row["caption"]),"adversarial_utf8_sha256":utf8_sha(adv),
                       "historical_metadata":asdict(refs["metadata.json"]),"historical_batch_index":index,
                       "historical_query_index":row["index"],"historical_protocol":asdict(s.headers["protocol"]),
                       "historical_completed":asdict(s.headers["completed"]),"historical_Selection":asdict(s.selection),
                       "H_forward_samples_this_run":0,"historical_H_forward_samples":action["H_forward"],
                       "optimizer_steps":0,"surrogate_forwards":0,"target_feedback_used":False,"target_tokenizer_used":False}
                self.count+=1
                yield request,output("hv1",request,BUDGET,adv,None,trace,self.provenance.as_dict())
            self.batches+=1
        need(self.count==len(s.manifest.queries),"historical text coverage")
        need(self.file_snapshot=={p:checked(FileRef(p,v["sha256"])) for p,v in self.file_snapshot.items()},"historical text SHA/mtime changed")
        s.verify_unchanged();self.complete=True
    def report(self):
        return {"status":"TEXT_SOURCE_VERIFIED" if self.complete else "TEXT_SOURCE_PENDING",
                "queries":self.count,"batches":self.batches,"metadata_files_verified":len(self.file_snapshot),
                "H_forwards":0,"IRRA_forwards":0,"feature_tensors_loaded":0,"new_text_units_published":0}
