"""共用冻结IRRA文本编码；随机策略及初始H均不见检索身份。"""
import torch
from tokenizer_probe import official_probe
from text_frontend import CaptionFrontend
from phase5_joint.backend import OfficialTextEncoder
from .strategies import RandomStrategy,UntrainedStrategy,inference_settings

class Engine:
    def __init__(self,protocol,encoder_factory=OfficialTextEncoder):
        self.protocol=protocol;self.probe=official_probe();self.frontend=CaptionFrontend()
        self.encoder=None;self.factory=encoder_factory
        inference_settings()
    def strategy(self,trial):
        if trial["kind"]=="untrained":
            return UntrainedStrategy(self.protocol["architecture"],self.probe,self.frontend)
        return RandomStrategy(trial["kind"],trial["seed"],self.probe,self.frontend,
                              self.protocol["config"]["search_max_nodes"])
    @torch.inference_mode()
    def batch(self,strategy,captions,learned,original_features):
        if (type(captions)is not tuple or any(type(x)is not str for x in captions)
                or len(captions)!=len(learned) or len(captions)!=len(original_features)):
            raise ValueError("Caption-only tuples and verified action references required")
        actions=[strategy.attack(caption,ref) for caption,ref in zip(captions,learned)]
        changed=[i for i,a in enumerate(actions) if a["adversarial"]!=captions[i]]
        features=original_features.detach().clone()
        if changed:
            if self.encoder is None:
                self.encoder=self.factory(self.probe,len(self.protocol["queries"])*len(self.protocol["trials"]))
            features[changed]=self.encoder.encode(tuple(actions[i]["adversarial"] for i in changed))
        return features,{"G_new_samples":0,"IRRA_image_new_samples":0,"IRRA_text_samples":len(changed),
                         "H_forward_samples":sum(a["H_forward"] for a in actions),"actions":actions}
    def samples(self):return self.encoder.samples if self.encoder is not None else 0
    def finish(self):
        return self.encoder.finish() if self.encoder is not None else {
            "no_new_IRRA_model":True,"IRRA_text_samples_this_attempt":0,
            "G_samples_this_attempt":0,"IRRA_image_samples_this_attempt":0}

def peak_memory():
    return {"GPU_peak_allocated_MiB":torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_initialized() else 0.,
            "GPU_peak_reserved_MiB":torch.cuda.max_memory_reserved()/2**20 if torch.cuda.is_initialized() else 0.}
