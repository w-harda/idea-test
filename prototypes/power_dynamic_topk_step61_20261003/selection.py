"""Selection-only；只用已保存S_raw决定K，不导入任何模型/旧pipeline。"""
from __future__ import annotations
import math,copy
GAMMAS=(1.,1.5,2.,3.)
BETA=1.
K_MAX=4


def candidates_from_record(record):
    out=[];seen=set()
    for a in record['attributes']:
        if a['status']!='scored':continue
        inst=a['instance'];score=a['S_raw'];aid=a['instance_id']
        if inst['unresolved'] or inst.get('semantic_representation') is None or a['rank']<=0:raise ValueError('scored record violates frozen eligibility')
        if aid!=inst['instance_id'] or aid in seen:raise ValueError('mismatched/duplicate instance ID')
        seen.add(aid)
        if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or score<0:raise ValueError('finite nonnegative saved S_raw required')
        if score==0:continue  # 用户本轮限定S_i>0；明确排除，不造probability/fallback。
        if 'S_raw' in inst:raise ValueError('unexpected scoring field in frozen instance')
        item=copy.deepcopy(inst);item['S_raw']=score;out.append(item)
    return out


def select(caption_id,candidates,gamma):
    if isinstance(gamma,bool) or gamma not in GAMMAS:raise ValueError('only fixed gamma 1/1.5/2/3')
    if len({a['instance_id'] for a in candidates})!=len(candidates):raise ValueError('duplicate instance ID')
    for a in candidates:
        s=a['S_raw']
        if isinstance(s,bool) or not isinstance(s,(int,float)) or not math.isfinite(s) or s<=0:raise ValueError('positive finite scores required')
        if a['unresolved']:raise ValueError('unresolved not eligible')
    # 排序只读取原S_raw；ties沿原Stage01实例输入顺序。
    ranked=sorted(candidates,key=lambda a:-a['S_raw']);m=len(ranked)
    if not m:
        return dict(caption_id=caption_id,candidate_count=0,gamma=gamma,probabilities={},entropy=0.,N_eff=0.,k_star=0,ranked_attributes=[],selected_attributes=[])
    # Log-space幂归一，等价S^gamma/sum；避免溢出。仅决定K。
    peak=math.log(ranked[0]['S_raw']);u=[math.exp(gamma*(math.log(a['S_raw'])-peak)) for a in ranked];total=math.fsum(u)
    ps=[v/total for v in u];entropy=-math.fsum(v*math.log(v) for v in ps if v>0);neff=math.exp(entropy);k=min(math.ceil(BETA*neff),m,K_MAX)
    return dict(caption_id=caption_id,candidate_count=m,gamma=gamma,probabilities={a['instance_id']:p for a,p in zip(ranked,ps)},entropy=entropy,N_eff=neff,k_star=k,ranked_attributes=ranked,selected_attributes=ranked[:k])
