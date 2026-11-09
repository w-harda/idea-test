"""原三头SCST不变；当前G攻击图像提供吸引目标，原文提供排斥目标。"""
from dataclasses import dataclass,asdict
from .objectives import scalars,require

@dataclass(frozen=True)
class AlignmentReward:
    total:float
    pull_kl:float
    push_kl:float
    baseline_pull_kl:float
    pull_gain:float

class BoundAlignmentReward:
    def __init__(self,base,profile,partner_image):
        self.base,self.runtime,self.profile=base,base.runtime,profile
        self.partner=partner_image.detach().clone()
    def evaluate_many(self,original,variants):
        # 原绑定器继续核验Train身份、当前Caption、Unicode规则和三条新文本的前向预算。
        raw=self.base.evaluate_many(original,variants)
        self.features=self.base.features.detach().clone()
        require(len(self.features)==3 and len(raw)==2,'original/sample/greedy only')
        require(self.partner.shape==self.features[:1].shape,'current paired G image')
        self.raw_rewards=[asdict(v) for v in raw];self.alignment=[];rewards=[]
        for current in self.features[1:]:
            values=scalars(self.profile.terms(current[None],self.features[:1],self.partner))
            self.alignment.append(values)
            rewards.append(AlignmentReward(values['reward'],values['pull_kl'],values['push_kl'],values['baseline_pull_kl'],values['pull_gain']))
        return tuple(rewards)
    def diagnostics(self):
        return {**self.base.diagnostics(),'attraction':{'sampled':self.alignment[0],'greedy':self.alignment[1],
            'old_rewards_diagnostic_only':self.raw_rewards,'R_sim_optimized':False,'R_rank_optimized':False,
            'objective':'pull_current_G_image_push_original_text','candidate_count':len(self.profile.anchors),'text_features':self.features.tolist(),'partner_image_feature':self.partner.reshape(-1).tolist()}}

def train_step(phase,policy,optimizer,example,sampler,bind_reward,step,row_id,profile,partner_image):
    def reward(k):return BoundAlignmentReward(bind_reward(k),profile,partner_image)
    return phase.train_step(policy,optimizer,example,sampler,reward,step,row_id)
