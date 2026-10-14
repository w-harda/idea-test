"""只聚合已经冻结的Top-K；不改变S、排序、Q或AttributeInstance。"""
import torch
from torch.nn import functional as F
def aggregate_condition(z_t,plan,vectors):
 z=z_t.detach().reshape(-1)
 if z.shape!=(512,) or not torch.isfinite(z).all() or abs(float(z.norm())-1)>2e-4:raise ValueError("caption retrieval feature")
 if not plan:
  return {"z_t":z.cpu().float(),"c_attr":torch.zeros(512),"v_local":torch.zeros(512),"has_local":False},[]
 scores=torch.tensor([a.metadata["S_raw"] for a in plan],dtype=torch.float64,device=z.device)
 if not torch.isfinite(scores).all() or not (scores>0).all():raise ValueError("positive frozen scores required")
 weights=scores/scores.sum()
 e=torch.tensor([a.metadata["semantic_representation"]["projected_context"] for a in plan],dtype=torch.float32,device=z.device)
 if e.shape!=(len(plan),512) or not torch.isfinite(e).all() or (e.norm(dim=1)==0).any():raise ValueError("invalid semantic representation")
 vv=torch.cat([vectors[a.instance_id].detach().to(z.device) for a in plan],dim=0)
 if vv.shape!=e.shape or not torch.isfinite(vv).all():raise ValueError("JVP shapes")
 w=weights.float()
 pooled=(w[:,None]*F.normalize(e,dim=1)).sum(0)
 local=(w[:,None]*vv).sum(0) # 保留原JVP尺度，不再次归一化聚合向量。
 cond={"z_t":z.cpu().float(),"c_attr":pooled.detach().cpu(),"v_local":local.detach().cpu(),"has_local":True}
 return cond,weights.cpu().tolist()
