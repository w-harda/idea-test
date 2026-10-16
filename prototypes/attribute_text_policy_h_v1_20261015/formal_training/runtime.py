"""真实加载只在用户确认CLI分支调用；训练复用原SCST奖励公式。"""
from dataclasses import asdict
import hashlib
import math
import torch
from torch.nn import functional as F
from real_feedback import InferenceRuntime
from phase3_inputs import import_gv2, CHECKPOINT
from phase2_sources import digest
from feedback_backend import validate_variant
from rewards import TrainGallery, RewardEvaluator
from real_feature_cache import feature_stats
from .data import preflight_models, members
from .common import verify_sources

class FormalRuntime(InferenceRuntime):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # 旧Smoke执行器用MagicMock记录调用，会累计持有图像/token Tensor。
        # 长程只用普通闭包累计整数，保留原预算/冻结检查，不保存调用参数。
        from contextlib import ExitStack
        self.stack.close()
        self.stack=ExitStack()
        original_image=self.encoder.model.encode_image
        original_text=self.encoder.model.encode_text
        def image_hook(pixels):
            self.consume(self.image_family,len(pixels))
            return original_image(pixels)
        def text_hook(tokens):
            self.consume("text",len(tokens))
            return original_text(tokens)
        self.encoder.model.encode_image=image_hook
        self.encoder.model.encode_text=text_hook
        self.stack.callback(setattr,self.encoder.model,"encode_image",original_image)
        self.stack.callback(setattr,self.encoder.model,"encode_text",original_text)
        def g_hook(module,args):
            if len(args)!=1 or args[0].requires_grad:
                raise ValueError("image-only G inference argument required")
            self.consume("G",len(args[0]))
        handle=self.generator.register_forward_pre_hook(g_hook)
        self.stack.callback(handle.remove)

    @torch.inference_mode()
    def encode_original(self, texts):
        """仅Validation原文允许官方77-token截断；保持全部Query，不据此筛选。"""
        self.assert_frozen()
        if not texts or any(type(t) is not str for t in texts): raise ValueError("raw captions required")
        tokens=torch.tensor([self.probe.inspect(t).model_tokens for t in texts],
                            dtype=torch.long,device=self.device)
        out=self.checked_features(F.normalize(self.encoder.model.encode_text(tokens).float(),dim=1),len(texts))
        self.assert_frozen()
        return out

def load_runtime(protocol, probe, image_limit=0, text_limit=0):
    verify_sources(protocol); preflight_models()
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise ValueError("RTX 4090 unavailable")
    inference,adapter=import_gv2()
    with torch.inference_mode(False):
        g=inference.load_generator(CHECKPOINT,"cuda" if image_limit else "cpu")
        encoder=adapter.load_official("cuda")
    return FormalRuntime(g,encoder,inference.generate_once,probe,"cuda",
            {"G":image_limit,"clean":image_limit,"adv":image_limit,"text":text_limit},"official_irra_real")

class RewardBank:
    """一次核验整个Train Gallery；PID仅用于奖励上下文，不进入H。"""
    def __init__(self,catalog,store,frontend):
        if store.role!="train": raise ValueError("validation gallery cannot supply SCST feedback")
        self.catalog,self.samples,self.frontend=catalog,members(catalog,"train"),frontend
        self.features=store.gather()
        self.by_id={s.row_id:i for i,s in enumerate(self.samples)}
        self.contexts={}

    def bind(self,query,runtime,example,budget):
        self.catalog.require_train(query)
        if query.row_id not in self.by_id or self.samples[self.by_id[query.row_id]]!=query:
            raise ValueError("paired Train identity mismatch")
        if example != self.frontend(query.caption):
            raise ValueError("caption-only example source mismatch")
        if not 0<=budget<=len(example.attributes): raise ValueError("invalid Dynamic-K budget")
        # 相同query的静态ctx复用；不缓存任何历史H动作、文本特征或reward。
        if query.row_id not in self.contexts:
            ctx=TrainGallery(self.features["clean"],self.features["adversarial"],
                torch.tensor([s.pid==query.pid for s in self.samples]),self.by_id[query.row_id],
                "train",("train",)*len(self.samples),"train",("train",)*len(self.samples))
            self.contexts[query.row_id]=ctx
        return FreshReward(self.contexts[query.row_id],query,example,budget,runtime)

class FreshReward:
    def __init__(self,context,query,example,budget,runtime):
        self.runtime,self.query,self.example,self.budget=runtime,query,example,budget
        self.before=dict(runtime.samples)
        self.evaluator=RewardEvaluator(context,self.encode,1.,1.)

    def encode(self,texts):
        if hasattr(self,"features"): raise ValueError("fresh triplet may only encode once")
        self.texts=tuple(texts)
        self.features=self.runtime.encode_text(texts)
        return self.features

    def evaluate_many(self,original,variants):
        if original!=self.query.caption or len(variants)!=2:
            raise ValueError("current sampled/greedy caption triplet required")
        for text in variants: validate_variant(self.example,text,self.budget,self.runtime.probe)
        self.runtime.assert_frozen()
        result=self.evaluator.evaluate_many(original,variants)
        if self.runtime.samples["text"]!=self.before["text"]+3 or any(
                self.runtime.samples[k]!=0 for k in ("G","clean","adv")):
            raise ValueError("SCST text-only fresh forward budget mismatch")
        for reward in result:
            if not all(math.isfinite(v) for v in asdict(reward).values()):
                raise ValueError("nonfinite reward")
            if reward.total!=reward.R_sim+reward.R_rank or reward.synergy_increment!=reward.adv_sim_drop-reward.clean_sim_drop:
                raise ValueError("frozen reward formula mismatch")
        self.runtime.assert_frozen()
        return result

    def diagnostics(self):
        return {"IRRA_text_calls":1,"IRRA_text_samples":3,"G_calls":0,"IRRA_image_calls":0,
                "text_feature_numeric_sha256":hashlib.sha256(self.features.numpy().tobytes()).hexdigest(),
                "token_sha256":digest([self.runtime.probe.inspect(t).model_tokens for t in self.texts]),
                "features":feature_stats({"text":self.features})}
