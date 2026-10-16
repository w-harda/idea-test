"""官方IRRA all-same-PID定义；仅CPU FP32分数，不调用任何模型。"""
from ..artifacts import ArtifactError, digest
from ..contracts import EvaluationIdentity

METRICS = ("mAP","R@1","R@5","R@10","mINP")
RANKING = {"name":"irra-all-same-pid-v1","unit":"percent","camera_filter":False,
           "sort":"torch.argsort(dim=1,descending=True,stable=False)",
           "AP":"mean precision at every positive rank",
           "INP":"number of positives / last positive rank","max_rank":10,
           "aggregation":"FP32 full-query mean; no mean of chunk means"}
RANKING_ID = digest(RANKING)

def ranking(scores, identity, *, direction="higher_better"):
    import torch
    if type(identity) is not EvaluationIdentity:raise ArtifactError("typed retrieval identity required")
    if direction not in ("higher_better","lower_better"):raise ArtifactError("explicit score direction required")
    if not isinstance(scores,torch.Tensor) or scores.device.type!="cpu" or scores.dtype!=torch.float32:
        raise ArtifactError("finite CPU FP32 score matrix required")
    if scores.requires_grad or scores.shape!=(len(identity.query_ids),len(identity.gallery_ids)) or not torch.isfinite(scores).all():
        raise ArtifactError("score shape/finite/gradient mismatch")
    if scores.shape[1]<10:raise ArtifactError("R@10 requires at least 10 Gallery members")
    # 不使用stable=True或额外ID排序；与官方当前CPU/Torch版本的tie行为一致。
    order=torch.argsort(scores if direction=="higher_better" else -scores,dim=1,descending=True)
    labels={pid:i for i,pid in enumerate(dict.fromkeys(identity.gallery_pids))}
    gids=torch.tensor([labels[p] for p in identity.gallery_pids])
    qids=torch.tensor([labels[p] for p in identity.query_pids])
    matches=gids[order].eq(qids[:,None])
    cmc=matches[:,:10].cumsum(1)
    cmc[cmc>1]=1
    cmc=cmc.float().mean(0)*100
    cumulative=matches.cumsum(1)
    counts=matches.sum(1)
    # 列向量化精确保留官方逐列除法和FP32最终reduce；避免Python列列表。
    positions=torch.arange(1,scores.shape[1]+1,dtype=torch.float32)
    ap=((cumulative/positions)*matches).sum(1)/counts
    last=torch.where(matches,torch.arange(1,scores.shape[1]+1),0).max(1).values
    inp=counts/last.float()
    values={"mAP":float(ap.mean()*100),"R@1":float(cmc[0]),"R@5":float(cmc[4]),
            "R@10":float(cmc[9]),"mINP":float(inp.mean()*100)}
    return values,order

def drops(clean, attacked):
    if set(clean)!=set(METRICS) or set(attacked)!=set(METRICS):raise ArtifactError("five metrics required")
    from math import isfinite
    if any(type(v) not in (int,float) or not isfinite(v) or not 0<=v<=100 for v in (*clean.values(),*attacked.values())):
        raise ArtifactError("metrics must be finite percentages")
    return {k:(100*(clean[k]-attacked[k])/clean[k] if clean[k] else None) for k in METRICS}
