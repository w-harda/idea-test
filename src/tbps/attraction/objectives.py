"""固定Train候选分布上的KL；明确区分靠近对方和远离自身。"""
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F
from tbps.artifacts import ArtifactError

def require(ok,message):
    if not ok:raise ArtifactError(message)

@dataclass(frozen=True)
class Settings:
    temperature:float=.07
    pull_weight:float=1.
    push_weight:float=1.
    def __post_init__(self):
        require(all(type(v) in (int,float) and math.isfinite(v) for v in (self.temperature,self.pull_weight,self.push_weight)),'finite attraction settings')
        require(self.temperature>0 and self.pull_weight>=0 and self.push_weight>=0 and self.pull_weight+self.push_weight>0,'valid attraction weights/temperature')

class ProfileKL:
    """候选ID顺序由绑定层保证；锚点和参考分布不接收梯度。"""
    def __init__(self,anchors,settings):
        require(isinstance(settings,Settings),'explicit Settings required')
        require(anchors.ndim==2 and len(anchors)>=2 and anchors.shape[1]>0 and anchors.dtype in (torch.float32,torch.float64),'floating [M,D] anchors')
        require(torch.isfinite(anchors).all() and (anchors.norm(dim=1)>0).all(),'finite nonzero anchors')
        self.settings=settings;self.anchors=F.normalize(anchors.detach().clone(),dim=1)
    def to(self,device):return ProfileKL(self.anchors.to(device),self.settings)
    def log_probs(self,features):
        require(features.ndim==2 and features.shape[1]==self.anchors.shape[1] and features.device==self.anchors.device and features.dtype==self.anchors.dtype,'same candidate feature space/dtype/device')
        require(torch.isfinite(features).all() and (features.detach().norm(dim=1)>0).all(),'finite nonzero feature rows')
        return F.log_softmax(F.normalize(features,dim=1)@self.anchors.T/self.settings.temperature,dim=1)
    def terms(self,current,clean,partner):
        require(current.shape==clean.shape==partner.shape,'paired current/clean/opposite-adversarial features')
        log_current=self.log_probs(current)
        with torch.no_grad():
            log_clean=self.log_probs(clean.detach());log_partner=self.log_probs(partner.detach())
            baseline=F.kl_div(log_clean,log_partner,log_target=True,reduction='none').sum(1)
        pull=F.kl_div(log_current,log_partner,log_target=True,reduction='none').sum(1)
        push=F.kl_div(log_current,log_clean,log_target=True,reduction='none').sum(1)
        gain=baseline-pull
        reward=self.settings.pull_weight*gain+self.settings.push_weight*push
        loss=-reward
        with torch.no_grad():
            partner_cos=F.cosine_similarity(current.detach(),partner.detach(),dim=1)
            clean_cos=F.cosine_similarity(current.detach(),clean.detach(),dim=1)
        return {'loss':loss,'reward':reward,'pull_kl':pull,'push_kl':push,'baseline_pull_kl':baseline,'pull_gain':gain,'partner_cosine':partner_cos,'clean_cosine':clean_cos}

def scalars(terms):return {k:float(v.detach().mean().cpu()) for k,v in terms.items()}
