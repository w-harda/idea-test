"""冻结G-v2/官方IRRA真实奖励执行器；H始终只拿TextExample。"""
from contextlib import ExitStack
from unittest.mock import patch
import torch
from torch.nn import functional as F
from rewards import frozen_features
from feedback_backend import state_sha
from phase3_inputs import import_gv2, verify_sources

class InferenceRuntime:
    """CPU fixture可测试同一前向/冻结检查核心；真实加载器只能使用CUDA和锁定官方模型。"""
    def __init__(self,generator,encoder,generate_once,probe,device,limits,backend_kind):
        if backend_kind not in ("official_irra_real","cpu_test_fixture"):
            raise ValueError("explicit real/CPU-test backend required")
        self.device=torch.device(device)
        if backend_kind=="official_irra_real" and self.device.type!="cuda":
            raise ValueError("real runtime requires CUDA")
        if backend_kind=="cpu_test_fixture" and self.device.type!="cpu":
            raise ValueError("fixture runtime must remain CPU")
        self.kind=backend_kind
        self.generator=generator.eval().requires_grad_(False)
        self.encoder=encoder.eval().requires_grad_(False)
        self.generate_once,self.probe=generate_once,probe
        self.models=(self.generator,self.encoder)
        self.before={k:state_sha(m) for k,m in zip(("G","IRRA"),self.models)}
        self.versions=tuple(tuple((n,x._version) for n,x in m.state_dict(keep_vars=True).items()) for m in self.models)
        self.samples={k:0 for k in ("G","clean","adv","text")}
        self.calls=dict(self.samples)
        self.limits=dict(limits)
        self.image_family=None
        self.stack=ExitStack()
        original_image=self.encoder.model.encode_image
        original_text=self.encoder.model.encode_text
        def image_hook(pixels):
            self.consume(self.image_family,len(pixels))
            return original_image(pixels)
        def text_hook(tokens):
            self.consume("text",len(tokens))
            return original_text(tokens)
        self.stack.enter_context(patch.object(self.encoder.model,"encode_image",side_effect=image_hook))
        self.stack.enter_context(patch.object(self.encoder.model,"encode_text",side_effect=text_hook))
        def generator_hook(module,args):
            if len(args)!=1 or args[0].requires_grad:
                raise ValueError("G input must be a single image-only inference argument")
            self.consume("G",len(args[0]))
        handle=self.generator.register_forward_pre_hook(generator_hook)
        self.stack.callback(handle.remove)
        self.closed=False

    def consume(self,key,n):
        self.assert_frozen()
        if (torch.is_grad_enabled() or not torch.is_inference_mode_enabled()
                or key not in self.limits or type(n) is not int or n<1
                or self.samples[key]+n>self.limits[key]):
            raise ValueError("real inference forward budget/context violation")
        self.samples[key]+=n
        self.calls[key]+=1

    def assert_frozen(self):
        if self.closed:
            raise ValueError("closed inference runtime")
        for i,m in enumerate(self.models):
            if (any(child.training for child in m.modules())
                    or any(p.requires_grad or p.grad is not None for p in m.parameters())
                    or tuple((n,x._version) for n,x in m.state_dict(keep_vars=True).items())!=self.versions[i]):
                raise ValueError("frozen model flags/parameters/buffers changed")

    @staticmethod
    def checked_features(x,n):
        if not isinstance(x,torch.Tensor) or x.shape!=(n,512) or not torch.isfinite(x).all():
            raise ValueError("finite retrieval [B,512] required")
        result=x.float().detach().cpu().clone()
        frozen_features(result,torch.Size((n,512)))
        return result

    @torch.inference_mode()
    def image_pair(self,pixels):
        self.assert_frozen()
        if (not isinstance(pixels,torch.Tensor) or pixels.ndim!=4 or pixels.shape[0]!=1
                or pixels.shape[1]!=3 or pixels.dtype!=torch.float32
                or not torch.isfinite(pixels).all() or pixels.min()<0 or pixels.max()>1):
            raise ValueError("one finite [0,1] RGB image required")
        image=pixels.detach().to(self.device).clone()
        self.image_family="clean"
        clean=self.checked_features(self.encoder(image),1)
        before=self.samples["G"]
        generated=self.generate_once(self.generator,image)
        adv=generated["adversarial"]
        if self.samples["G"]!=before+1:
            raise ValueError("exactly one G forward per image required")
        if adv.shape!=image.shape or not torch.isfinite(adv).all() or adv.min()<0 or adv.max()>1:
            raise ValueError("invalid adversarial image")
        linf=float((adv.double()-image.double()).abs().max())
        if linf>8/255:
            raise ValueError("strict cumulative L-infinity violation")
        self.image_family="adv"
        adversarial=self.checked_features(self.encoder(adv),1)
        self.assert_frozen()
        return {"clean":clean,"adversarial":adversarial},{"linf":linf,"G_samples":1,
                "IRRA_clean_samples":1,"IRRA_adv_samples":1,"IRRA_text_samples":0}

    @torch.inference_mode()
    def encode_text(self,texts):
        self.assert_frozen()
        if not texts or any(type(t) is not str for t in texts):
            raise ValueError("raw Unicode Caption strings required")
        reports=[self.probe.inspect(t) for t in texts]
        if any(r.truncated for r in reports):
            raise ValueError("IRRA tokenizer truncation rejected")
        tokens=torch.tensor([r.model_tokens for r in reports],dtype=torch.long,device=self.device)
        result=self.checked_features(F.normalize(self.encoder.model.encode_text(tokens).float(),dim=1),len(texts))
        self.assert_frozen()
        return result

    def finish(self):
        if self.closed:
            return self.final
        self.assert_frozen()
        after={k:state_sha(m) for k,m in zip(("G","IRRA"),self.models)}
        if after!=self.before:
            raise ValueError("full model state SHA changed")
        self.final={"state_sha256_before":self.before,"state_sha256_after":after,
                    "parameters_buffers_unchanged":True,"gradients_absent":True,
                    "samples":dict(self.samples),"calls":dict(self.calls)}
        self.stack.close()
        self.closed=True
        return self.final

    def close(self):
        self.stack.close()
        self.closed=True

def load_real_runtime(protocol,probe):
    verify_sources(protocol)
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise ValueError("authorized RTX 4090 unavailable")
    inference,adapter=import_gv2()
    # 参数在普通Tensor模式构造，以便每轮检查version；所有前向仍严格inference_mode。
    with torch.inference_mode(False):
        generator=inference.load_generator(protocol["checkpoint"],"cuda").eval().requires_grad_(False)
        encoder=adapter.load_official("cuda").eval().requires_grad_(False)
    return InferenceRuntime(generator,encoder,inference.generate_once,probe,"cuda",
             {"G":len(protocol["gallery"]),"clean":len(protocol["gallery"]),
              "adv":len(protocol["gallery"]),"text":3*len(protocol["queries"])+2},"official_irra_real")
