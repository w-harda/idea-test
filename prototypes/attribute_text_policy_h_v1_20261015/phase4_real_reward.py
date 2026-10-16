"""Phase 4只编码当前H的文本；G与IRRA图像前向预算固定0。"""
import hashlib
import torch
from feedback_backend import BoundReward
from real_feedback import InferenceRuntime
from phase3_inputs import import_gv2
from phase4_inputs import verify_protocol
from real_feature_cache import feature_stats
from phase2_sources import digest

def load_runtime(protocol,probe,remaining_steps):
    if not 1<=remaining_steps<=32:
        raise ValueError("real training step budget exceeded")
    verify_protocol(protocol)
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise ValueError("RTX 4090 unavailable")
    inference,adapter=import_gv2()
    with torch.inference_mode(False):
        # G只载入CPU进行冻结状态核验，不执行图像生成。
        generator=inference.load_generator(protocol["phase3_protocol"]["checkpoint"],"cpu").eval().requires_grad_(False)
        encoder=adapter.load_official("cuda").eval().requires_grad_(False)
    return InferenceRuntime(generator,encoder,inference.generate_once,probe,"cuda",
            {"G":0,"clean":0,"adv":0,"text":3*remaining_steps},"official_irra_real")

class FreshReward:
    def __init__(self,catalog,query,gallery,features,runtime,frontend,budget):
        self.runtime=runtime
        self.calls_before=dict(runtime.calls)
        self.samples_before=dict(runtime.samples)
        owner=self
        class TextContext:
            probe=runtime.probe
            def assert_frozen(self):runtime.assert_frozen()
            def encode_text(self,texts):
                if getattr(owner,"features",None) is not None:
                    raise ValueError("one fresh triplet per SCST step")
                owner.texts=tuple(texts)
                owner.features=runtime.encode_text(texts)
                return owner.features
        self.bound=BoundReward(catalog,query,gallery,features,TextContext(),frontend,budget)

    def evaluate_many(self,original,variants):
        if len(variants)!=2:
            raise ValueError("sampled and greedy pair required")
        rewards=self.bound.evaluate_many(original,variants)
        if (self.runtime.calls["text"]!=self.calls_before["text"]+1
                or self.runtime.samples["text"]!=self.samples_before["text"]+3
                or any(self.runtime.samples[k]!=0 for k in ("G","clean","adv"))):
            raise ValueError("fresh text-only GPU budget mismatch")
        return rewards

    def diagnostics(self):
        raw=self.features.contiguous().numpy().tobytes()
        return {"IRRA_text_calls":1,"IRRA_text_samples":3,"G_calls":0,"IRRA_image_calls":0,
                "text_feature_numeric_sha256":hashlib.sha256(raw).hexdigest(),
                "token_sha256":digest([self.runtime.probe.inspect(t).model_tokens for t in self.texts]),
                "features":feature_stats({"text":self.features})}
