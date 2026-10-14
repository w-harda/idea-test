"""冻结分数上的集合选择；只在manifest准备时调用既有Natural-K公式。"""
import copy,hashlib,random
from collections import Counter,defaultdict
import deps
from selection_audit import selection,bin_for,BINS,load,verify,digest,save
MODES=("S_TOPK","RANDOM_K","S_BOTTOMK")
SEED=42
def seed_for(cid):
    return int.from_bytes(hashlib.sha256(("random-k|42|"+cid).encode()).digest()[:8],"big")
def choose(payload):
    ranked=payload["ranked_attributes"];k=payload["natural_k"];m=len(ranked)
    if not 0<k<m or m!=payload["candidate_count"]:raise ValueError("comparison requires 0<K<m")
    ids=[a["instance_id"] for a in ranked]
    if len(set(ids))!=m:raise ValueError("duplicate candidate ID")
    if any(a["S_raw"]<b["S_raw"] for a,b in zip(ranked,ranked[1:])):raise ValueError("frozen rank changed")
    indices=sorted(random.Random(seed_for(payload["caption_id"])).sample(range(m),k))
    selected={MODES[0]:ids[:k],MODES[1]:[ids[i] for i in indices],MODES[2]:ids[-k:]}
    overlap={}
    for l,r in [(MODES[0],MODES[1]),(MODES[0],MODES[2]),(MODES[1],MODES[2])]:
        a,b=set(selected[l]),set(selected[r])
        overlap[l+"_vs_"+r]={"intersection_count":len(a&b),"jaccard":len(a&b)/len(a|b),"identical_set":a==b}
    return {"selection_ids":selected,"random_seed_hex":hex(seed_for(payload["caption_id"])),"set_overlap":overlap}
def mode_payload(row,mode):
    if mode not in MODES:raise ValueError("unknown mode")
    lookup={a["instance_id"]:a for a in row["ranked_attributes"]}
    selected=[lookup[i] for i in row["selection_ids"][mode]]
    # prepared_plan validates every source/basis/score. Its local ranking is this chosen sorted subset.
    return {**row,"ranked_attributes":selected,"selected_attributes":selected,
            "natural_k":len(selected),"k_star":len(selected),"capped_k":min(len(selected),4)}
def verify_row(row,record):
    candidates=[{**copy.deepcopy(x["instance"]),"S_raw":x["S_raw"]} for x in record["attributes"]
                if x["status"]=="scored" and x["S_raw"]>0]
    candidates.sort(key=lambda a:-a["S_raw"])
    if candidates!=row["ranked_attributes"]:raise ValueError("candidate source/metadata/score changed")
    if row["selected_attributes"]!=candidates[:row["natural_k"]]:raise ValueError("frozen Top-K changed")
    for key,value in choose(row).items():
        if row.get(key)!=value:raise ValueError("selection manifest modified: "+key)
def cache_sources():
    reference=load(deps.P74/"outputs/clean_scores/selection_manifest.json")
    if not reference["full_preparation"]:raise ValueError("full score preparation required")
    verify({reference["score_cache_path"]:reference["score_cache_sha256"],
            reference["score_protocol_path"]:reference["score_protocol_sha256"]})
    protocol=load(reference["score_protocol_path"])
    expected={"schema":"step74-clean-score-cache-1","gamma":3,"beta":1,"tau":.02,"layer":8}
    if any(protocol[k]!=v for k,v in expected.items()):raise ValueError("frozen score settings changed")
    verify(protocol["frozen_sha256"])
    verify({p:h for p,h in protocol["source_sha256"].items()
            if deps.Path(p).name in ("prepare_scores.py","deps.py","selection_audit.py")})
    if digest(deps.V4/"outputs/split.json")!=reference["split_sha256"]:raise ValueError("split SHA")
    return reference,protocol
def stratum(p):
    axes=tuple(sorted({a["semantic_axis"] for a in p["ranked_attributes"]}))
    return (p["candidate_count"],axes)
def build(records,validation,old,target=100):
    pools={b:defaultdict(list) for b in BINS};available=Counter();excluded=Counter()
    old_by={p["caption_id"]:p for p in old["rows"]};eligible_old=0;seen=set();count=0
    key=lambda p:(hashlib.sha256(("selection-value|42|"+p["caption_id"]).encode()).hexdigest(),p["caption_id"])
    for record in records:
        cid=record["row_id"];count+=1
        if cid in seen:raise ValueError("duplicate score row")
        seen.add(cid)
        if cid not in validation or record["caption"]!=validation[cid]["caption"] or record["pid"]!=validation[cid]["pid"]:
            raise ValueError("validation PID/source mismatch")
        if record["retrieval_backward_count"]!=1 or not record["zero_intervention_exact"]:raise ValueError("clean cache contract")
        p=selection(record)
        if cid in old_by and any(p[k]!=v for k,v in old_by[cid].items() if k!="pid"):
            raise ValueError("original100 Natural K/ranking regression")
        if not 0<p["natural_k"]<p["candidate_count"]:
            excluded["m_not_greater_than_K"]+=1;continue
        eligible_old+=cid in old_by
        p.update(pid=record["pid"])
        b=bin_for(p["natural_k"]);available[b]+=1
        group=pools[b][stratum(p)];group.append(p);group.sort(key=key)
        if len(group)>target:group.pop()
    if seen!=set(validation):raise ValueError("full13700 cache coverage mismatch")
    # Within each K bin, deterministic round-robin over (m, axis signature), each ordered by hash.
    orders={}
    for b in BINS:
        strata=sorted(pools[b],key=lambda x:hashlib.sha256(("stratum|42|"+b+"|"+repr(x)).encode()).hexdigest())
        order=[];i=0
        while len(order)<target:
            batch=[pools[b][s][i] for s in strata if i<len(pools[b][s])]
            if not batch:break
            order.extend(batch);i+=1
        orders[b]=order[:target]
    chosen=[];used=Counter()
    while len(chosen)<target:
        progress=False
        for b in BINS:
            if len(chosen)>=target:break
            if used[b]<len(orders[b]):
                chosen.append(orders[b][used[b]]);used[b]+=1;progress=True
        if not progress:break
    if len(chosen)!=target:raise ValueError("insufficient eligible diagnostic rows")
    for p in chosen:p.update(choose(p))
    return {"schema":"frozen-score-selection-value-1","case_count":len(chosen),"score_cache_caption_count":count,
            "seed":SEED,"sampling":"m>K only; K bins round-robin; within bin (m,axis signature) round-robin; SHA256 order",
            "old100_eligible_count":eligible_old,"old100_reused_as_manifest":False,
            "eligible_K_distribution":dict(available),"excluded":dict(excluded),
            "K_distribution":dict(Counter(str(p["natural_k"]) for p in chosen)),
            "m_distribution":dict(Counter(str(p["candidate_count"]) for p in chosen)),
            "candidate_axis_counts":dict(Counter(a["semantic_axis"] for p in chosen for a in p["ranked_attributes"])),
            "distinct_pid_count":len({p["pid"] for p in chosen}),
            "logical_rounds_per_mode":sum(p["natural_k"] for p in chosen),
            "selection_only_no_model":True,"score_backwards":0,
            "caveats":["K1 with m>K may have no examples: ceil(exp(entropy))>=2 for nondegenerate positive scores",
                       "stratified diagnostic, not validation population estimate","single reproducible random draw, no seed search",
                       "overlapping sets and repeated canonical instances retained"],
            "rows":chosen}
