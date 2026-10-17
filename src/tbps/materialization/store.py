"""复用ArtifactStore；真实FP32像素与UTF-8文本独立不可变保存，不存victim特征。"""
from dataclasses import asdict
from pathlib import Path
import hashlib
import io
from ..artifacts import ArtifactStore,FileRef,checked,read_json,sha_file,json_bytes,digest,PendingPublication
from ..contracts import AttackArtifact
from ..attacks.base import AttackOutput,ImageRequest,TextRequest,tensor_sha
from ..attacks.gv2 import BUDGET as G_BUDGET
from ..attacks.hv1 import BUDGET as H_BUDGET
from .contracts import MaterializationPlan,MaterializedUnit,need

def utf8_sha(value):
    need(type(value)is str,"Unicode Caption required")
    return hashlib.sha256(value.encode("utf-8",errors="strict")).hexdigest()

def image_checks(clean,adv,delta,size):
    import torch
    for x in (clean,adv,delta):
        need(isinstance(x,torch.Tensor) and x.device.type=="cpu" and x.dtype==torch.float32 and
             x.shape==(1,3,*size) and not x.requires_grad and bool(torch.isfinite(x).all()),"FP32 RGB Tensor contract")
    need(bool(((clean>=0)&(clean<=1)).all()) and bool(((adv>=0)&(adv<=1)).all()),"RGB [0,1] required")
    need(torch.equal(delta,adv-clean),"effective_delta must equal adv-original exactly")
    linf=float((adv.double()-clean.double()).abs().max())
    need(linf<=8/255,"strict original-relative L-infinity budget exceeded")
    return {"original_pixels_sha256":tensor_sha(clean),"adversarial_pixels_sha256":tensor_sha(adv),
            "effective_delta_sha256":tensor_sha(delta),"linf":linf,"shape":list(adv.shape),"dtype":"torch.float32",
            "pixel_space":"rgb_0_1","hash_schema":"tbps-attack-tensor-sha-v1","preview":False}

class MaterializedStore:
    def __init__(self,store,plan):
        need(type(store)is ArtifactStore and type(plan)is MaterializationPlan,"typed new Store/plan")
        self.store,self.plan=store,plan
        self._plan_ref=None;self._plan_snapshot={}
    def publish_plan(self):
        names=("contracts.py","store.py","pipeline.py","legacy_text.py")
        producers=tuple(FileRef(str(Path(__file__).with_name(n)),sha_file(Path(__file__).with_name(n))) for n in names)
        return self.store.publish("protocol",{"protocol.json":json_bytes(self.plan.descriptor())},
            {"kind":"materialization_protocol","plan_id":self.plan.id},sources=(*self.plan.sources,*producers))
    def verify_plan(self,*,force=False):
        def stat(path):
            v=Path(path).stat();return (v.st_ino,v.st_size,v.st_mtime_ns,v.st_ctime_ns)
        if self._plan_ref is not None and not force:
            need(all(stat(p)==v for p,v in self._plan_snapshot.items()),"materialization plan/source changed after SHA verification")
            return self._plan_ref
        manifest=self.store.verify("protocol",expected_provenance={"kind":"materialization_protocol","plan_id":self.plan.id})
        p=self.store.root/"protocol/protocol.json"
        need(read_json(p)==__import__("json").loads(json_bytes(self.plan.descriptor())),"plan identity/schema changed")
        paths=[self.store.root/"protocol"/name for name in (*manifest["files_sha256"],"manifest.json","completed.json")]
        paths.extend(Path(v["path"]) for v in manifest["sources"])
        self._plan_snapshot={str(p):stat(p) for p in paths}
        self._plan_ref=FileRef(str(p),sha_file(p))
        return self._plan_ref
    def unit_id(self,binding):
        need(binding==self.plan.binding(binding.sample_id,binding.modality),"not a materialization member")
        return digest({"plan":self.plan.id,"input":binding.id,"modality":binding.modality})
    def available(self,binding):
        unit=self.unit_id(binding)
        for pending in self.store.pending():
            intent=self.store.root/pending/"intent.json"
            if not intent.exists() or read_json(intent).get("artifact_id")==unit:
                raise PendingPublication("uncommitted materialization unit; do not generate again")
        return (self.store.root/unit).exists()
    def image_request(self,image_id):
        from ..contracts import PublicImageInput
        binding=self.plan.binding(image_id,"image")
        row=next(g for g in self.plan.manifest.gallery if g.image_id==image_id)
        path=(Path(self.plan.image_root)/row.image_rel_path).resolve()
        need(path.is_relative_to(Path(self.plan.image_root).resolve()),"source image escapes root")
        return ImageRequest(PublicImageInput(FileRef(str(path),row.image_sha256),"rgb_0_1"),binding)
    def text_request(self,query_id):
        from ..contracts import PublicTextInput
        row=next(q for q in self.plan.manifest.queries if q.query_id==query_id)
        return TextRequest(PublicTextInput(row.caption),self.plan.binding(query_id,"text"))
    def _result(self,request,result,kind):
        need(type(result)is AttackOutput and result.binding==request.binding and result.attack_id==kind,"output identity/modality mismatch")
        source=self.plan.g if kind=="gv2" else self.plan.h
        need(result.provenance()==source.as_dict(),"wrong frozen attack source")
        need(result.budget==(G_BUDGET if kind=="gv2" else H_BUDGET),"frozen attack budget mismatch")
        return source
    def _publish(self,request,result,metadata,files,content_sha,source,*,fault=None):
        plan_ref=self.verify_plan();unit=self.unit_id(request.binding)
        need(not self.available(request.binding),"immutable unit already exists")
        metadata.update(schema="tbps-materialized-unit-1",plan_id=self.plan.id,binding=asdict(request.binding),
            modality=request.binding.modality,attack_id=result.attack_id,attack_provenance=source.as_dict(),
            budget=asdict(result.budget),content_sha256=content_sha,trace=result.trace(),victim_model=None)
        refs=[plan_ref,source.checkpoint]
        if type(request)is ImageRequest:refs.append(request.public.content)
        self.store.publish(unit,{**files,"metadata.json":json_bytes(metadata)},
            {"kind":"materialized_attack","plan_id":self.plan.id,"binding_id":request.binding.id},
            sources=tuple(refs),fault=fault)
        data_name="pixels.pt" if type(request)is ImageRequest else "adversarial.txt"
        base=self.store.root/unit
        return MaterializedUnit(unit,self.plan.id,request.binding,request.binding.modality,content_sha,
            FileRef(str(base/data_name),sha_file(base/data_name)),FileRef(str(base/"metadata.json"),sha_file(base/"metadata.json")),
            FileRef(str(base/"manifest.json"),sha_file(base/"manifest.json")))
    def publish_image(self,request,clean,result,*,fault=None):
        need(type(request)is ImageRequest and request==self.image_request(request.binding.sample_id),"wrong original image request")
        source=self._result(request,result,"gv2");checked(request.public.content)
        import torch
        # 非Tensor不接受；CPU转移只在显式发布API，载入后同样真正clone。
        need(all(isinstance(x,torch.Tensor) for x in (clean,result.payload,result.effective_delta)),"actual RGB required; embeddings cannot materialize")
        arrays={k:x.detach().cpu().clone() for k,x in (("original",clean),("adversarial",result.payload),("effective_delta",result.effective_delta))}
        stats=image_checks(*arrays.values(),self.plan.pixel_size)
        trace=result.trace()
        need(trace["input_pixels_sha256"]==stats["original_pixels_sha256"] and
             trace["adversarial_pixels_sha256"]==stats["adversarial_pixels_sha256"] and
             trace["effective_delta_sha256"]==stats["effective_delta_sha256"],"G trace pixel content changed")
        need(trace["G_forward_samples"]==1 and trace["optimizer_steps"]==trace["surrogate_forwards"]==0 and
             trace["condition_cache_read"]is False,"G single image-only forward contract")
        stream=io.BytesIO();torch.save(arrays,stream)
        return self._publish(request,result,{"pixels":stats,"original_file":asdict(request.public.content)},
             {"pixels.pt":stream.getvalue()},stats["adversarial_pixels_sha256"],source,fault=fault)
    def _historical_action(self,request,trace):
        need(trace["stage"] in ("online_generation","historical_text_reuse"),"explicit text generation/reuse origin")
        if trace["stage"]=="online_generation":
            need(trace["optimizer_steps"]==trace["surrogate_forwards"]==0 and trace["target_feedback_used"]is False and
                 trace["target_tokenizer_used"]is False and trace["state_sha256_before"]==trace["state_sha256_after"],
                 "frozen Caption-only generation trace")
            return
        for key in ("historical_metadata","historical_protocol","historical_completed","historical_Selection"):
            ref=FileRef(**trace[key]);need(ref in self.plan.sources,"historical text proof not bound to plan");checked(ref)
        md=read_json(trace["historical_metadata"]["path"])
        slot=trace["historical_query_index"]-md["rows"][0]["index"]
        need(0<=slot<len(md["rows"]) and md["rows"][slot]["row_id"]==request.binding.sample_id and
             md["rows"][slot]["caption"]==request.public.caption and md["stats"]["actions"][slot]==trace["action"],
             "historical text action pointer/identity mismatch")
        need(trace["H_forward_samples_this_run"]==0 and trace["target_feedback_used"]is False,
             "historical reuse must not claim a new H forward")
    def publish_text(self,request,result,validate,*,fault=None):
        need(type(request)is TextRequest and request==self.text_request(request.binding.sample_id),"wrong original Caption request")
        source=self._result(request,result,"hv1");self._historical_action(request,result.trace());action=result.trace()["action"]
        need(callable(validate),"frozen Caption-only action validator required")
        need(validate(request.public.caption,action)==result.payload,"Unicode action/output mismatch")
        before=utf8_sha(request.public.caption);after=utf8_sha(result.payload)
        need(before==request.binding.input_sha256 and action["original"]==request.public.caption and
             action["adversarial"]==result.payload,"actual original/adversarial Caption binding")
        return self._publish(request,result,{"original_utf8_sha256":before,"adversarial_utf8_sha256":after,
             "original_caption":request.public.caption,"adversarial_caption":result.payload,"action":action},
             {"original.txt":request.public.caption.encode(),"adversarial.txt":result.payload.encode()},
             after,source,fault=fault)
    def reference(self,binding):
        unit=self.unit_id(binding);self.verify_plan()
        self.store.verify(unit,expected_provenance={"kind":"materialized_attack","plan_id":self.plan.id,"binding_id":binding.id})
        base=self.store.root/unit;md=read_json(base/"metadata.json")
        name="pixels.pt" if binding.modality=="image" else "adversarial.txt"
        return MaterializedUnit(unit,self.plan.id,binding,binding.modality,md["content_sha256"],
             FileRef(str(base/name),sha_file(base/name)),FileRef(str(base/"metadata.json"),sha_file(base/"metadata.json")),
             FileRef(str(base/"manifest.json"),sha_file(base/"manifest.json")))
    def load(self,ref,*,validate=None):
        need(type(ref)is MaterializedUnit and ref.plan_id==self.plan.id and ref.unit_id==self.unit_id(ref.binding),"foreign materialized unit")
        need(ref==self.reference(ref.binding),"unit file reference/content identity changed")
        md=read_json(ref.metadata.path)
        expected=self.plan.g if ref.modality=="image" else self.plan.h
        need(md["schema"]=="tbps-materialized-unit-1" and md["plan_id"]==self.plan.id and
             md["binding"]==asdict(ref.binding) and md["attack_provenance"]==expected.as_dict() and
             md["content_sha256"]==ref.content_sha256 and md["victim_model"]is None and
             md["budget"]==asdict(G_BUDGET if ref.modality=="image" else H_BUDGET),"saved unit schema/source mismatch")
        if ref.modality=="image":
            import torch
            old=torch.load(ref.data.path,map_location="cpu",weights_only=True)
            need(type(old)is dict and set(old)=={"original","adversarial","effective_delta"},"FP32 RGB payload keys")
            arrays={k:x.detach().clone() for k,x in old.items()};del old
            stats=image_checks(arrays["original"],arrays["adversarial"],arrays["effective_delta"],self.plan.pixel_size)
            need(stats==md["pixels"] and stats["adversarial_pixels_sha256"]==ref.content_sha256 and
                 md["original_file"]==asdict(self.image_request(ref.binding.sample_id).public.content) and
                 md["trace"]["input_pixels_sha256"]==stats["original_pixels_sha256"] and
                 md["trace"]["adversarial_pixels_sha256"]==stats["adversarial_pixels_sha256"] and
                 md["trace"]["effective_delta_sha256"]==stats["effective_delta_sha256"],"pixel SHA/budget mismatch")
            return arrays,md
        need(callable(validate),"frozen text action validator required")
        self._historical_action(self.text_request(ref.binding.sample_id),md["trace"])
        original=Path(ref.metadata.path).with_name("original.txt").read_bytes().decode("utf-8",errors="strict")
        adversarial=Path(ref.data.path).read_bytes().decode("utf-8",errors="strict")
        need(utf8_sha(original)==ref.binding.input_sha256==md["original_utf8_sha256"] and
             utf8_sha(adversarial)==ref.content_sha256==md["adversarial_utf8_sha256"],"UTF-8 content SHA mismatch")
        need(original==md["original_caption"] and adversarial==md["adversarial_caption"] and
             md["trace"]["action"]==md["action"] and
             validate(original,md["action"])==adversarial,"saved Unicode action mismatch")
        return adversarial,md
    def attack_artifact(self,ref,*,validate=None):
        self.load(ref,validate=validate)
        p=self.plan.g if ref.modality=="image" else self.plan.h
        return AttackArtifact(ref.unit_id,p.attack_id,p.checkpoint.sha256,(ref.binding.input_sha256,),ref.data,
                              G_BUDGET if ref.modality=="image" else H_BUDGET)
