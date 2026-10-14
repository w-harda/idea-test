"""Step7.4：冻结clean score上的自然K、确定性分层与结果统计；无模型导入。"""
from __future__ import annotations
import copy, hashlib, json, math, statistics
from collections import Counter
from pathlib import Path
BINS = ("1", "2", "3", "4", "5", "6", "7+")
MODES = ("CAPPED_4", "NATURAL_DYNAMIC_K")
METRICS = ("paired_cosine", "paired_soft_rank_F", "positive_image_rank")
GAMMA, BETA, CAP = 3.0, 1.0, 4
# 在读取任何新攻击结果前固定；绝对100个soft-rank单位，不按结果调整。
MATERIAL_F_THRESHOLD = 100.0

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024), b""): h.update(block)
    return h.hexdigest()

def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def verify(hashes):
    for path, expected in hashes.items():
        if digest(path)!=expected: raise ValueError("SHA mismatch: "+path)

def new_output(path):
    path=Path(path).resolve()
    if not path.is_relative_to(Path("/home/lzf/ldx").resolve()) or path.exists():
        raise ValueError("new output under /home/lzf/ldx required; no overwrite")
    path.mkdir(parents=True)
    return path

def bin_for(k):
    if k<1: raise ValueError("nonempty natural K required")
    return str(k) if k<7 else "7+"

def span_gate(caption, a):
    span=a.get("source_span")
    return (isinstance(span,dict) and type(span.get("start")) is int and type(span.get("end")) is int
            and 0<=span["start"]<span["end"]<=len(caption)
            and caption[span["start"]:span["end"]]==span.get("raw")==a.get("raw_text")
            and span["start"]==a.get("char_start") and span["end"]==a.get("char_end"))

def selection(record):
    eligible=[]
    if len({x["instance"]["instance_id"] for x in record["attributes"]})!=len(record["attributes"]):
        raise ValueError("duplicate AttributeInstance")
    for row in record["attributes"]:
        a=row["instance"];score=row.get("S_raw")
        if row["status"]!="scored": continue
        e=a.get("semantic_representation")
        if a.get("unresolved") or not span_gate(record["caption"],a) or not e or e["basis_rank"]<=0:
            raise ValueError("scored instance violates frozen gate")
        if not isinstance(score,(int,float)) or not math.isfinite(score) or score<0:
            raise ValueError("invalid frozen S_raw")
        if score>0: eligible.append({**copy.deepcopy(a),"S_raw":score})
    # 与旧实现相同：tie保留原AttributeInstance输入顺序，不改importance。
    ranked=sorted(eligible,key=lambda a:-a["S_raw"])
    logs=[GAMMA*math.log(a["S_raw"]) for a in ranked]
    if logs:
        largest=max(logs);values=[math.exp(x-largest) for x in logs]
        total=sum(values);probs=[x/total for x in values]
        entropy=-sum(p*math.log(p) for p in probs if p)
        neff=math.exp(entropy);natural=min(math.ceil(BETA*neff),len(ranked))
    else: probs=[];entropy=0.0;neff=0.0;natural=0
    return {"caption_id":record["row_id"],"candidate_count":len(ranked),"gamma":GAMMA,"beta":BETA,
            "probabilities":probs,"entropy":entropy,"N_eff":neff,"natural_k":natural,
            "capped_k":min(natural,CAP),"truncated":natural>CAP,
            "ranked_attributes":ranked,"selected_attributes":ranked[:natural],"k_star":natural}

def stratify(records, validation_ids, target=100):
    if target<1: raise ValueError("positive target required")
    seen=set();available=Counter()
    pools={b:[] for b in BINS};excluded=Counter()
    key=lambda row:(hashlib.sha256(("42|"+row["caption_id"]).encode()).hexdigest(),row["caption_id"])
    for r in records:
        if r["row_id"] in seen: raise ValueError("duplicate caption")
        seen.add(r["row_id"])
        if r["row_id"] not in validation_ids: excluded["not_validation"]+=1;continue
        if r.get("retrieval_backward_count")!=1 or not r.get("zero_intervention_exact"):
            raise ValueError("missing frozen clean score contract")
        p=selection(r)
        if not p["natural_k"]:excluded["no_positive_supported_candidate"]+=1;continue
        b=bin_for(p["natural_k"]);available[b]+=1
        pools[b].append({"caption_id":r["row_id"],"pid":r["pid"],**p})
        pools[b].sort(key=key)
        if len(pools[b])>target:pools[b].pop()
    for pool in pools.values():pool.sort(key=key)
    chosen=[];counts=Counter()
    # 固定bin顺序轮流分配，稀缺bin全取；余量自动分给尚有剩余的bin。
    while len(chosen)<target:
        progress=False
        for b in BINS:
            if len(chosen)==target:break
            if counts[b]<len(pools[b]):
                chosen.append(pools[b][counts[b]]);counts[b]+=1;progress=True
        if not progress:break
    return {"schema":"step74-stratified-natural-k-1","target_count":target,"case_count":len(chosen),
            "sampling":"round-robin 1/2/3/4/5/6/7+; each bin sorted SHA256('42|caption_id'), then caption_id; no attack results",
            "bin_available":{b:available[b] for b in BINS},"bin_selected":{b:counts[b] for b in BINS},
            "natural_k_distribution":dict(sorted(Counter(str(x["natural_k"]) for x in chosen).items(),key=lambda x:int(x[0]))),
            "candidate_m_distribution":dict(sorted(Counter(str(x["candidate_count"]) for x in chosen).items(),key=lambda x:int(x[0]))),
            "truncated_count":sum(x["truncated"] for x in chosen),"excluded":dict(excluded),
            "distinct_pid_count":len({x["pid"] for x in chosen}),
            "maximum_captions_per_pid":max(Counter(x["pid"] for x in chosen).values(),default=0),
            "material_rule":{"metric":"F_natural_final - F_round4","threshold":MATERIAL_F_THRESHOLD,
                             "improved":">=100","worse":"<=-100","nearly_unchanged":"-100 < delta < 100"},
            "sampling_caveat":"stratified diagnostic, not validation-prevalence-weighted population estimate",
            "rows":chosen}

def stats(xs):
    xs=list(xs)
    return {"count":len(xs),"mean":statistics.mean(xs) if xs else None,
            "median":statistics.median(xs) if xs else None,"min":min(xs) if xs else None,"max":max(xs) if xs else None}

def best_prefix(data):
    states=data["states"];idx=max(range(len(states)),key=lambda i:states[i]["paired_soft_rank_F"])
    return {"best_round":idx,"final_equals_best":idx==len(states)-1 or states[idx]["paired_soft_rank_F"]==states[-1]["paired_soft_rank_F"],
            "F_best_minus_final":states[idx]["paired_soft_rank_F"]-states[-1]["paired_soft_rank_F"],
            "rank_best_minus_final":states[idx]["positive_image_rank"]-states[-1]["positive_image_rank"]}

def comparisons(cases):
    out={}
    for key in METRICS:
        diffs=[c["modes"][MODES[1]]["states"][-1][key]-c["modes"][MODES[0]]["states"][-1][key] for c in cases]
        signs=[-x if key=="paired_cosine" else x for x in diffs]
        out[key]={"natural_wins":sum(x>0 for x in signs),"capped_wins":sum(x<0 for x in signs),
                  "ties":sum(x==0 for x in signs),"natural_minus_capped":stats(diffs)}
    return out

def marginal(events):
    ds=[e["metric_delta"]["paired_soft_rank_F"] for e in events]
    return {"round_count":len(events),"delta_F":stats(ds),
            "delta_cosine":stats(e["metric_delta"]["paired_cosine"] for e in events),
            "delta_rank":stats(e["metric_delta"]["positive_image_rank"] for e in events),
            "worsening_count":sum(x>0 for x in ds),"recovery_count":sum(x<0 for x in ds),
            "tie_count":sum(x==0 for x in ds),"recovery_ratio":sum(x<0 for x in ds)/len(ds) if ds else None,
            "post_projection_update_l2":stats(e["post_projection_actual_change"]["l2"] for e in events),
            "clipping_ratio":stats(e["projection_clipping_ratio_mean_over_steps"] for e in events),
            "update_cosine":stats(e["actual_post_projection_update_cosine_previous_round"] for e in events if e["actual_post_projection_update_cosine_previous_round"] is not None),
            "aggregate_gradient_norm":stats(e["image_gradient_l2_at_round_start"] for e in events)}

def summarize(cases):
    extended=[c for c in cases if c["natural_k"]>4]
    out={"case_count":len(cases),"truncated_count":len(extended),"comparisons_all":comparisons(cases),
         "comparisons_truncated":comparisons(extended),"by_natural_k_bin":{},"modes":{}}
    for b in BINS:
        cs=[c for c in cases if bin_for(c["natural_k"])==b]
        out["by_natural_k_bin"][b]={"count":len(cs),"comparisons":comparisons(cs)}
    for mode in MODES:
        data=[c["modes"][mode] for c in cases]
        def retention(ds):
            bs=[best_prefix(d) for d in ds]
            return {"count":len(bs),"final_equals_best_count":sum(b["final_equals_best"] for b in bs),
                    "final_equals_best_ratio":sum(b["final_equals_best"] for b in bs)/len(bs) if bs else None,
                    "F_best_minus_final":stats(b["F_best_minus_final"] for b in bs),
                    "rank_best_minus_final":stats(b["rank_best_minus_final"] for b in bs)}
        out["modes"][mode]={"final":{key:stats(d["states"][-1][key] for d in data) for key in METRICS},
                            "clean_to_final":{key:stats(d["clean_to_final"][key] for d in data) for key in METRICS},
                            "retention_all":retention(data),"retention_truncated":retention([c["modes"][mode] for c in extended])}
    after=[];category=Counter()
    for c in extended:
        data=c["modes"][MODES[1]];delta=data["states"][-1]["paired_soft_rank_F"]-data["states"][4]["paired_soft_rank_F"]
        label="material_improvement" if delta>=MATERIAL_F_THRESHOLD else "material_worse" if delta<=-MATERIAL_F_THRESHOLD else "nearly_unchanged"
        category[label]+=1;after.append({"caption_id":c["caption_id"],"delta_F_after4":delta,"category":label,
            "delta_rank_after4":data["states"][-1]["positive_image_rank"]-data["states"][4]["positive_image_rank"]})
    out["after_round4"]={"rule":"fixed absolute F threshold +/-100, registered before attacks","counts":dict(category),"per_caption":after}
    maxk=max((c["natural_k"] for c in cases),default=0)
    rounds={str(i):marginal([c["modes"][MODES[1]]["rounds"][i-1] for c in cases if c["natural_k"]>=i]) for i in range(1,maxk+1)}
    rounds["7+"] = marginal([e for c in cases for e in c["modes"][MODES[1]]["rounds"][6:]])
    rounds["5+"] = marginal([e for c in cases for e in c["modes"][MODES[1]]["rounds"][4:]])
    return out,rounds
