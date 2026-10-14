"""冻结S_raw加权；仅修正本轮新增属性。输入为实际TTA IG贡献。"""
import math
import torch
MODES=("PREFIX_BASELINE","SCORE_WEIGHTED_PREFIX","SCORE_WEIGHTED_PRIORITY_PROJECTION")
def normalized_weights(scores,mode):
    if mode not in MODES or not scores or any(not math.isfinite(s) or s<=0 for s in scores):
        raise ValueError("positive frozen scores and known mode required")
    if any(a<b for a,b in zip(scores,scores[1:])):raise ValueError("frozen priority order changed")
    total=sum(scores)
    return [1/len(scores)]*len(scores) if mode==MODES[0] else [s/total for s in scores]
def combine(gradients,scores,mode,original_ig=None):
    """返回实际momentum输入；无冲突时projection与weighted是同一计算路径。"""
    weights=normalized_weights(scores,mode)
    if len(gradients)!=len(scores) or any(g.shape!=gradients[0].shape or not torch.isfinite(g).all() for g in gradients):
        raise ValueError("invalid real IG components")
    versions=[g._version for g in gradients]
    # 与原prefix聚合相同的单项特例；共享round1不重新执行此路径。
    if len(gradients)==1:
        base=gradients[0];history=torch.zeros_like(base)
    else:
        ws=torch.tensor(weights,dtype=gradients[0].dtype,device=gradients[0].device)
        base=(torch.stack(gradients)*ws.reshape((-1,)+(1,)*gradients[0].ndim)).sum(0)
        hw=torch.tensor([s/sum(scores[:-1]) for s in scores[:-1]],dtype=base.dtype,device=base.device)
        history=(torch.stack(gradients[:-1])*hw.reshape((-1,)+(1,)*base.ndim)).sum(0)
    new=gradients[-1]
    # 只用fp64 dot/norm作数值运算，稳定项严格为该dtype的machine epsilon。
    h64=history.double();n64=new.double();eps=torch.finfo(h64.dtype).eps
    h2=h64.square().sum();dot=(h64*n64).sum()
    hn=float(h2.sqrt());nn=float(n64.norm());conflict=bool(dot<0)
    repaired=new
    intervention=mode==MODES[2] and conflict
    if intervention:
        repaired=(n64-dot/(h2+eps)*h64).to(new.dtype)
        # 仅替换新属性；历史原tensor不修改、不对称投影。
        adjusted=list(gradients);adjusted[-1]=repaired
        ws=torch.tensor(weights,dtype=base.dtype,device=base.device)
        result=(torch.stack(adjusted)*ws.reshape((-1,)+(1,)*base.ndim)).sum(0)
    else:result=base
    if mode==MODES[0]:
        if original_ig is None:raise ValueError("baseline must receive unmodified official IG")
        result=original_ig
    removed=(repaired.double()-n64).norm();after_dot=float((repaired.double()*h64).sum())
    after_n=float(repaired.double().norm())
    original_norm=float(original_ig.double().norm()) if original_ig is not None else None
    recon=float((original_ig.double()-base.double()).norm()) if original_ig is not None else None
    if [g._version for g in gradients]!=versions:raise ValueError("historical/new input components mutated")
    record={
        "normalized_weights":weights,"S_raw":list(scores),
        "history_definition":"S_raw weighted mean of previous real IG components",
        "history_norm":hn,"new_norm_before":nn,"dot_before":float(dot),
        "cosine_before":float(dot)/(hn*nn) if hn and nn else None,
        "conflict":conflict,"projection_intervention":intervention,
        "new_norm_after":after_n,"dot_after":after_dot,
        "cosine_after":after_dot/(hn*after_n) if hn and after_n else None,
        "removed_conflict_component_norm":float(removed),
        "projection_modification_ratio":float(removed)/(nn+eps),
        "aggregate_gradient_norm":float(result.double().norm()),
        "weighted_unprojected_gradient_norm":float(base.double().norm()),
        "history_components_unchanged":True,"only_new_attribute_modified":True,
        "no_conflict_projection_exact":not conflict and torch.equal(result,base) if mode!=MODES[0] else None,
        "baseline_original_tensor_identity":result is original_ig if mode==MODES[0] else None,
        "IG_reconstruction_absolute_error":recon,
        "IG_reconstruction_relative_error":recon/original_norm if original_norm else None,
        "projection_numeric_dtype":"float64","projection_epsilon":eps,
        "attack_direction":"ascent of unchanged negative image/JVP similarity loss",
    }
    return result,record
