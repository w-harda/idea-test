"""G-v2薄插件：调用注入的原generate_once，不实现网络或投影算法。"""
from dataclasses import asdict
from ..artifacts import checked, json_bytes
from ..config import ATTACKS
from ..contracts import AttackBudget, ImageBudget
from .base import PluginInfo, ImageRequest, need, frozen_state, tensor_sha, output

BUDGET = AttackBudget(image=ImageBudget("linf",8/255,"rgb_0_1"))
INFO = PluginInfo(ATTACKS["gv2"],frozenset({"train_pairs","source_gradients","source_features"}),
                  frozenset({"public_images"}))

class GV2Plugin:
    info = INFO
    def __init__(self, *, backend, provenance, budget=BUDGET):
        need(budget == BUDGET, "frozen G-v2 requires exact 8/255 budget")
        need(provenance.attack_id == "gv2", "wrong G checkpoint provenance")
        need(backend.execution_kind == provenance.execution_kind, "fixture/legacy provenance mismatch")
        self.backend,self.provenance,self.budget = backend,provenance,budget
    def generate(self, request):
        import torch
        need(type(request) is ImageRequest, "G takes only public image request")
        before_file = checked(request.public.content)
        before_state = frozen_state(self.backend.model)
        with torch.inference_mode():
            pixels = self.backend.load_pixels(request.public.content.path)
            need(isinstance(pixels,torch.Tensor) and pixels.dtype == torch.float32 and
                 pixels.ndim == 4 and pixels.shape[0:2] == (1,3) and
                 bool(torch.isfinite(pixels).all()) and bool(((pixels>=0)&(pixels<=1)).all()),
                 "original FP32 RGB pixels required")
            input_hash = tensor_sha(pixels)
            # 对原模块实际forward计数，不能仅信任后端返回的统计。
            calls = []
            def observe(module, args):
                need(len(args)==1 and args[0] is pixels, "G input is original pixels only")
                calls.append(len(args[0]))
                need(sum(calls)<=1, "one G forward per image required")
            handle = self.backend.model.register_forward_pre_hook(observe)
            try: value = self.backend.generate_once(self.backend.model,pixels)
            finally: handle.remove()
            need(calls == [1], "G must run exactly once")
            adv,delta = value["adversarial"],value["effective_delta"]
            need(adv.shape == pixels.shape and adv.dtype == pixels.dtype and adv.device == pixels.device,
                 "G must output native RGB pixels, not embeddings")
            need(bool(torch.isfinite(adv).all()) and bool(((adv>=0)&(adv<=1)).all()), "invalid attacked pixels")
            need(torch.equal(delta,adv-pixels), "effective perturbation mismatch")
            linf = float((adv.double()-pixels.double()).abs().max())
            need(linf<=8/255, "G cumulative budget exceeded")
            need(tensor_sha(pixels)==input_hash, "backend changed original pixels")
            payload,effective = adv.detach().clone(),delta.detach().clone()
        need(frozen_state(self.backend.model)==before_state, "G state changed during generation")
        need(checked(request.public.content)==before_file, "original image source changed")
        trace={"stage":"online_generation","input_file":asdict(request.public.content),
               "image_preprocess":self.backend.preprocess_id,"input_pixels_sha256":input_hash,
               "adversarial_pixels_sha256":tensor_sha(payload),"effective_delta_sha256":tensor_sha(effective),
               "shape":list(payload.shape),"dtype":str(payload.dtype),"pixel_space":"rgb_0_1",
               "linf":linf,"G_forward_samples":1,"optimizer_steps":0,"surrogate_forwards":0,
               "condition_cache_read":False,"state_sha256_before":before_state,"state_sha256_after":before_state,
               "budget":asdict(self.budget)}
        return output("gv2",request,self.budget,payload,effective,trace,self.provenance.as_dict())
