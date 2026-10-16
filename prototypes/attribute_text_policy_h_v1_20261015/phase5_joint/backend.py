"""H仅接受字符串Caption；GPU只编码Unicode文本，图像特征全部只读复用。"""
from collections import Counter
from dataclasses import asdict
import torch
from torch.nn import functional as F
from phase2_sources import digest,checked
from policy import CharacterPolicy
from contracts import from_text_spans
from constraints import EditRecord,replay
from text_frontend import CaptionFrontend
from tokenizer_probe import official_probe
from feedback_backend import state_sha
from formal_training.data import qualify_caption,example_from
from phase3_inputs import import_gv2
from .common import legacy

def validate_action(caption,record,probe,frontend=None):
    if record["original"]!=caption:raise ValueError("wrong Query Caption")
    ex=from_text_spans(caption,record["attributes"])
    if frontend is not None and ex!=frontend(caption):raise ValueError("Canonical source mismatch")
    edits=[EditRecord(**{**e,"source_span":tuple(e["source_span"])}) for e in record["edits"]]
    if replay(ex,edits,record["budget_k"])!=record["adversarial"]:
        raise ValueError("Unicode action trace mismatch")
    selected=record["selected_attribute_ids"]
    available={a.attribute_id for a in ex.attributes}
    if (len(set(selected))!=len(selected) or not set(selected)<=available
            or len(selected)>record["budget_k"] or any(e.attribute_id not in selected for e in edits)
            or record["budget_k"]>len(ex.attributes)):
        raise ValueError("attribute selection/edit budget mismatch")
    pair=probe.pair_report(caption,record["adversarial"])
    if digest(pair)!=digest(record["tokenizer_audit"]):raise ValueError("Tokenizer evidence mismatch")
    if edits and (not pair["model_tokens_changed"] or pair["before"]["truncated"] or pair["after"]["truncated"]):
        raise ValueError("illegal neutral/truncated Unicode action")
    if not record["eligible"] and (edits or record["adversarial"]!=caption or record["H_forward"]):
        raise ValueError("ineligible Query must be retained NOOP")
    return record

class TextAttacker:
    def __init__(self,policy,frontend,probe):
        self.policy=policy.eval().requires_grad_(False);self.frontend,self.probe=frontend,probe
        self.before=state_sha(policy);self.calls=0
    @torch.inference_mode()
    def attack(self,caption):
        if type(caption)is not str:raise TypeError("H input must be Caption string only")
        entry=qualify_caption(caption,self.frontend,self.probe)
        if entry["eligible"]:
            result=self.policy.attack(example_from(entry)).record();self.calls+=1
        else:
            result={"original":caption,"adversarial":caption,"budget_k":0,
                    "attributes":entry["attributes"],"predicted_importance":[],
                    "selected_attribute_ids":[],"edits":[],"decisions":[],
                    "tokenizer_audit":self.probe.pair_report(caption,caption)}
        result.update(eligible=entry["eligible"],reasons=entry["reasons"],H_forward=int(entry["eligible"]))
        validate_action(caption,result,self.probe)
        return result
    def finish(self):
        if (state_sha(self.policy)!=self.before or any(m.training for m in self.policy.modules())
                or any(p.requires_grad or p.grad is not None for p in self.policy.parameters())):
            raise ValueError("H parameters/buffers/flags changed")
        return {"H_state_sha256_before":self.before,"H_state_sha256_after":self.before,
                "H_unchanged":True,"H_forward_samples":self.calls}

class OfficialTextEncoder:
    def __init__(self,probe,max_samples):
        _,adapter=import_gv2()
        with torch.inference_mode(False):self.encoder=adapter.load_official("cuda").eval().requires_grad_(False)
        self.probe,self.limit,self.samples,self.calls=probe,max_samples,0,0
        self.before=state_sha(self.encoder)
        self.versions=tuple((n,x._version) for n,x in self.encoder.state_dict(keep_vars=True).items())
        # 普通函数计数；禁止调用IRRA图像前向，无MagicMock历史Tensor。
        self.original_image=self.encoder.model.encode_image
        def forbidden(*_):raise AssertionError("Phase5 image forward forbidden; reuse verified Gallery")
        self.encoder.model.encode_image=forbidden
    def frozen(self):
        if (any(m.training for m in self.encoder.modules())
                or any(p.requires_grad or p.grad is not None for p in self.encoder.parameters())
                or tuple((n,x._version) for n,x in self.encoder.state_dict(keep_vars=True).items())!=self.versions):
            raise ValueError("IRRA freeze contract changed")
    @torch.inference_mode()
    def encode(self,texts):
        self.frozen()
        if not texts or self.samples+len(texts)>self.limit:raise ValueError("IRRA text budget exceeded")
        reports=[self.probe.inspect(t) for t in texts]
        if any(r.truncated for r in reports):raise ValueError("new attacked text truncation forbidden")
        self.samples+=len(texts);self.calls+=1
        tokens=torch.tensor([r.model_tokens for r in reports],dtype=torch.long,device="cuda")
        x=F.normalize(self.encoder.model.encode_text(tokens).float(),dim=1).detach().cpu().clone()
        legacy("eval_metrics").normalized(x,len(texts));self.frozen()
        return x
    def finish(self):
        self.frozen()
        if state_sha(self.encoder)!=self.before:raise ValueError("IRRA parameters/buffers changed")
        self.encoder.model.encode_image=self.original_image
        return {"IRRA_unchanged":True,"IRRA_state_sha256_before":self.before,"IRRA_state_sha256_after":self.before,
                "IRRA_text_samples_this_attempt":self.samples,"IRRA_text_calls_this_attempt":self.calls,
                "G_samples_this_attempt":0,"IRRA_image_samples_this_attempt":0,
                "GPU_peak_allocated_MiB":torch.cuda.max_memory_allocated()/2**20,
                "GPU_peak_reserved_MiB":torch.cuda.max_memory_reserved()/2**20}

class Backend:
    def __init__(self,protocol,encoder_factory=OfficialTextEncoder):
        # 与Phase4 Validation的官方adapter相同；必须早于首批H forward，
        # 不能等到首次有编辑后才由懒加载IRRA切换全局MHA路径。
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.backends.mha.set_fastpath_enabled(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        self.protocol=protocol;self.probe=official_probe();self.frontend=CaptionFrontend()
        selection=protocol["H_selection"];checked(selection["checkpoint"],selection["checkpoint_sha256"])
        payload=torch.load(selection["checkpoint"],map_location="cpu",weights_only=False)
        if (payload["schema"]!="hv1-formal-periodic-checkpoint-1"
                or payload["protocol_sha256"]!=selection["training_protocol_sha256"]):
            raise ValueError("selected H training checkpoint schema/source mismatch")
        with torch.inference_mode(False):
            policy=CharacterPolicy(self.probe,**selection["architecture"])
            policy.load_state_dict(payload["policy"],strict=True)
        self.attacker=TextAttacker(policy,self.frontend,self.probe)
        from formal_training.validation import state_sha_from
        if self.attacker.before!=state_sha_from(payload["policy"]):
            raise ValueError("H checkpoint values/dtype changed on loading")
        self.encoder_factory=encoder_factory;self.encoder=None
    @torch.inference_mode()
    def text(self,captions,original_features):
        if type(captions)is not tuple or any(type(t)is not str for t in captions):
            raise TypeError("backend accepts only tuple of Caption strings; no Test identities")
        legacy("eval_metrics").normalized(original_features,len(captions))
        actions=[self.attacker.attack(t) for t in captions]
        changed=[i for i,a in enumerate(actions) if a["adversarial"]!=captions[i]]
        out=original_features.detach().clone()
        if changed:
            if self.encoder is None:self.encoder=self.encoder_factory(self.probe,self.protocol["query_count"])
            out[changed]=self.encoder.encode(tuple(actions[i]["adversarial"] for i in changed))
        return out,{"G_new_samples":0,"IRRA_image_new_samples":0,"IRRA_text_samples":len(changed),
                    "H_forward_samples":sum(a["H_forward"] for a in actions),"actions":actions}
    def finish(self):
        result=self.attacker.finish()
        result.update(self.encoder.finish() if self.encoder is not None else {
            "no_GPU_model_needed":True,"IRRA_text_samples_this_attempt":0,"G_samples_this_attempt":0,
            "IRRA_image_samples_this_attempt":0,"GPU_peak_allocated_MiB":0.,"GPU_peak_reserved_MiB":0.})
        return result

def action_statistics(actions):
    result={"query_count":len(actions),"K_distribution":dict(Counter(str(a["budget_k"]) for a in actions)),
            "NOOP_queries":sum(not a["edits"] for a in actions),
            "ineligible_queries":sum(not a["eligible"] for a in actions),
            "edited_queries":sum(bool(a["edits"]) for a in actions),
            "edits":sum(len(a["edits"]) for a in actions),
            "H_forward_samples":sum(a["H_forward"] for a in actions),
            "attribute_selections":sum(len(a["selected_attribute_ids"]) for a in actions)}
    axes=Counter();chars=Counter();stops=0
    for a in actions:
        ex=from_text_spans(a["original"],a["attributes"]);mapping={x.attribute_id:x.semantic_axis for x in ex.attributes}
        axes.update(mapping[x] for x in a["selected_attribute_ids"])
        chars.update(e["original_codepoint"]+"->"+e["replacement_codepoint"] for e in a["edits"])
        stops+=sum(d["head"]=="attribute" and d["index"]==len(ex.attributes) for d in a["decisions"])
    result.update(selected_attributes_by_axis=dict(axes),unicode_replacements=dict(chars),STOP_decisions=stops,
        NOOP_ratio=result["NOOP_queries"]/len(actions) if actions else 0.,
        tokenizer_changed_queries=sum(a["tokenizer_audit"]["model_tokens_changed"] for a in actions),
        original_truncated_queries=sum(a["tokenizer_audit"]["before"]["truncated"] for a in actions),
        new_truncations=sum(a["tokenizer_audit"]["new_truncation"] for a in actions))
    return result
