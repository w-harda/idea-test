"""显式启用的PID归约优化；不改变当前Full默认入口或SEP搜索。

按原PID/成员顺序组织索引，以组内首成员补齐尾部。原始首个max
（含并列、signed zero和NaN）仍先于所有补齐项；没有-inf哨兵。
只减少Python和小Tensor调用，不批处理模型、不改变GEMM或缓存。
"""
from .sep_bridge import PIDPlan
from .dataset import require

class PackedPIDPlan:
    def __init__(self, original):
        import torch
        require(type(original) is PIDPlan, "fixed original PIDPlan required")
        self.original=original
        self.order,self.members=original.order,original.members
        self.gallery_count,self.fast_domain=original.gallery_count,original.fast_domain
        width=max((len(g) for g in self.members),default=0)
        self.fallback=(not self.members or any(not g for g in self.members)
                       or len(self.members)*width>8*max(1,self.gallery_count))
        self.indices=None if self.fallback else torch.tensor(
            [list(g)+[g[0]]*(width-len(g)) for g in self.members],dtype=torch.long)
        self._device_indices={}

    def aggregate(self,scores):
        require(scores.ndim==2 and scores.shape[1]==self.gallery_count,"complete native Gallery scores")
        if self.fallback:return self.original.aggregate(scores)
        device=scores.device
        if device not in self._device_indices:self._device_indices[device]=self.indices.to(device)
        values=scores[:,self._device_indices[device]].max(dim=-1).values
        return values,self.order

def enable_packed_pid(classifier):
    """仅由未来显式选择的接线调用；原classifier/cache/forward原样复用。"""
    require(type(classifier.pid_plan) is PIDPlan,"PIDPlan already replaced or unknown")
    classifier.pid_plan=PackedPIDPlan(classifier.pid_plan)
    return classifier
