"""只读冻结受害者的可微SGA目标；最终评分仍走原Adapter，禁止把融合称为cosine。"""
import torch
import torch.nn.functional as F
from tbps.baselines.sga_attack import need

OBJECTIVES={
 "irra":"negative sum of native normalized image/text dot products",
 "rasa":"negative sum of native positive ITM fusion logits over original paired Captions",
 "aptm":"negative sum of native positive ITM fusion logits over original paired Captions; local surrogate only",
}

class SGASurrogate:
    def __init__(self,adapter,model,captions):
        self.adapter,self.model,self.captions=adapter,model,captions
        self.backend=adapter._backend if model=="irra" else adapter.backend
        need(model in OBJECTIVES,"supported native SGA victim")
        self.backend.eval().requires_grad_(False)
        self.counts={"image_samples":0,"fusion_pairs":0,"input_gradient_scales":0}
    def _pixels(self,pixels):
        need(pixels.dtype==torch.float32 and pixels.ndim==4 and len(pixels)==1
             and torch.isfinite(pixels).all(),"finite official SGA augmented RGB")
        # 官方BICUBIC reverse可轻微超出[0,1]；只允许内部增强，不新增clamp改变官方增强。
        if self.model=="irra":
            from tbps.models.irra_io import MEAN,STD
            return (pixels-pixels.new_tensor(MEAN).view(1,3,1,1))/pixels.new_tensor(STD).view(1,3,1,1)
        from torchvision.transforms.functional import resize,normalize
        from torchvision.transforms import InterpolationMode
        from tbps.models.rasa import PREPROCESS as R
        from tbps.models.aptm import PREPROCESS as A
        p=R if self.model=="rasa" else A
        return normalize(resize(pixels,p["size"],interpolation=InterpolationMode.BICUBIC,antialias=True),p["mean"],p["std"])
    def __call__(self,pixels):
        need(torch.is_grad_enabled() and pixels.requires_grad,"SGA surrogate must preserve input gradients")
        need(all(not p.requires_grad and p.grad is None for p in self.backend.parameters()) and not self.backend.training,
             "frozen eval victim only")
        x=self._pixels(pixels);c=self.captions
        self.counts["image_samples"]+=1;self.counts["input_gradient_scales"]+=1
        if self.model=="irra":
            image=F.normalize(self.backend.encode_image(x).float(),dim=-1)
            text=c.to(x.device)
            return -(image@text.T).sum()
        if self.model=="rasa":tokens=self.backend.visual_encoder(x)
        else:tokens,_=self.backend.get_vision_embeds(x)
        # CPU历史文本只作为常量，绝不反传到文本或参数。
        loss=x.new_zeros(())
        for i in range(len(c["global"])):
            text=c["tokens"][i:i+1].to(x.device).detach()
            att=c["attention"][i:i+1].to(x.device).detach()
            image_att=torch.ones(tokens.shape[:-1],dtype=torch.long,device=x.device)
            if self.model=="rasa":
                out=self.backend.text_encoder.bert(encoder_embeds=text,attention_mask=att,
                    encoder_hidden_states=tokens,encoder_attention_mask=image_att,return_dict=True,mode="fusion").last_hidden_state
            else:out=self.backend.get_cross_embeds(tokens,image_att,text_embeds=text,text_atts=att)
            loss=loss-self.backend.itm_head(out[:,0,:])[:,1].sum()
            self.counts["fusion_pairs"]+=1
        return loss
