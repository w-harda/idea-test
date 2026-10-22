"""G/H属性对齐只读CPU审计：已有JSON动作/分数派生CSV，无模型、Tokenizer执行、训练或GPU入口。"""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,ast,csv,hashlib,json,math,os,re,resource,statistics,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OLD=Path("/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014")
LEGACY=Path("/home/lzf/ldx/projects/idea-TBPS-test1")
G=OLD/"prototypes/attribute_image_generator_v2_20261014"
H=OLD/"prototypes/attribute_text_policy_h_v1_20261015"
HIST=H/"outputs/formal_v1"
BASE=Path("/home/lzf/ldx/outputs/tbps-framework-hv1-g60-adaptation-epoch60")
EXEC=Path("/home/lzf/ldx/projects/idea-test-tbps-hv1-g60-adaptation-20261021")
PREVIOUS=ROOT/"docs/framework/results/hv1_g60_posthoc_analysis"
OUT=ROOT/"docs/framework/results/gh_attribute_alignment_audit"
FIELDS=("semantic_axis","region","canonical_value","char_start","char_end","raw_text")
EPOCHS=(9,20,60)
WHITELIST=None
def need(x,m):
    if not x:raise ValueError(m)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()
def st(p):
    a=Path(p).stat();return [a.st_size,a.st_mtime_ns,a.st_ino]
def canon(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode()
def digest(x):return hashlib.sha256(canon(x)).hexdigest()
def attr_id(a):
    x={k:a[k] for k in FIELDS}
    return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:24]
def key(a,caption):
    s,e=a.get("char_start"),a.get("char_end")
    need(type(s) is int and type(e) is int and 0<=s<e<=len(caption) and caption[s:e]==a["raw_text"],"invalid original span")
    return tuple(a[k] for k in FIELDS)
def groups(attrs,caption):
    out=defaultdict(list)
    for i,a in enumerate(attrs):out[key(a,caption)].append((i,a))
    return out
def natural(scores):
    need(all(math.isfinite(x) and x>=0 for x in scores),"nonnegative finite scores")
    logs=[3*math.log(x) for x in scores if x>0]
    if not logs:return dict(k=0,entropy=0.,neff=0.,positive=0)
    v=[math.exp(x-max(logs)) for x in logs];p=[x/sum(v) for x in v]
    ent=-sum(x*math.log(x) for x in p if x)
    return dict(k=min(len(logs),math.ceil(math.exp(ent))),entropy=ent,neff=math.exp(ent),positive=len(logs))
def ranks(x):
    a=np.asarray(x,float);order=np.argsort(a,kind="stable");r=np.empty(len(a));i=0
    while i<len(a):
        j=i+1
        while j<len(a) and a[order[j]]==a[order[i]]:j+=1
        r[order[i:j]]=(i+j-1)/2+1;i=j
    return r
def corr(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    keep=np.isfinite(x)&np.isfinite(y);x=x[keep];y=y[keep]
    out=dict(n=len(x),pearson=None,spearman=None)
    if len(x)>=3 and np.std(x)>0 and np.std(y)>0:
        out.update(pearson=float(np.corrcoef(x,y)[0,1]),spearman=float(np.corrcoef(ranks(x),ranks(y))[0,1]))
    return out
def dist(x):
    x=list(x)
    if not x:return dict(n=0,mean=None,median=None,min=None,max=None)
    a=np.asarray(x,float)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),min=float(a.min()),max=float(a.max()),
                p05=float(np.quantile(a,.05)),p95=float(np.quantile(a,.95)))
def writej(name,data):
    with (OUT/name).open("x",encoding="utf8") as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
def writec(name,rows):
    need(rows,"empty output "+name);cols=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open("x",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,cols,lineterminator="\n");w.writeheader();w.writerows(rows)
class Inputs:
    def __init__(self,previous=None):self.files={};self.previous=previous or {}
    def add(self,p,expected=None):
        p=Path(p).resolve();n=str(p);s=st(p)
        if n not in self.files:
            h=sha(p);need(st(p)==s,"input mutated while reading")
            self.files[n]=dict(path=n,sha256=h,metadata=s)
        v=self.files[n]
        if expected:need(v["sha256"]==expected,"SHA mismatch "+n)
        if n in self.previous:need(v["sha256"]==self.previous[n],"prior analysis source changed "+n)
        return v["sha256"]
    def read(self,p,expected=None):
        self.add(p,expected);return json.loads(Path(p).read_text())
    def finish(self):
        for p,v in self.files.items():need(st(p)==v["metadata"],"input metadata mutated "+p)
I=Inputs()
def breakdown(a,q):
    attrs=a.get("attributes",q["attributes"]);caption=q["caption"];N=len(attrs)
    importance=a.get("predicted_importance")
    ids=[attr_id(x) for x in attrs];need(len(ids)==len(set(ids)),"duplicate H attribute")
    ds=a.get("decisions",[]);ad=[d for d in ds if d["head"]=="attribute"]
    pd=[d for d in ds if d["head"]=="position"];rd=[d for d in ds if d["head"]=="replacement"]
    selected=[d["index"] for d in ad if d["index"]!=N]
    stopped=bool(ad and ad[-1]["index"]==N)
    need(a.get("selected_attribute_ids",[])==[ids[i] for i in selected],"H selected ID/decision mismatch")
    need(len(pd)==len(selected),"position decision count")
    noop=sum(d["index"]==attrs[ai]["char_end"]-attrs[ai]["char_start"] for ai,d in zip(selected,pd))
    edits=a.get("edits",[]);K=a.get("budget_k",0)
    need(len(edits)==len(rd)==len(pd)-noop and len(selected)<=K,"action accounting")
    unused=K-len(edits);stopslots=K-len(selected)
    need(unused==noop+stopslots and (not stopslots or stopped or not q["eligible"]),"K gap decomposition")
    replay=list(caption);used=set();usedwords=set();editedids=[]
    for e in edits:
        ai=ids.index(e["attribute_id"]);attr=attrs[ai];p=e["position"]
        need(attr["char_start"]<=p<attr["char_end"] and caption[p]==e["original_char"],"edit original position")
        if WHITELIST is not None:need(e["replacement_char"] in WHITELIST.get(e["original_char"],()),"illegal Unicode replacement")
        need(e["source_span"]==[attr["char_start"],attr["char_end"]] and e["source_raw"]==attr["raw_text"],"edit span")
        need(e["original_codepoint"]==f"U+{ord(e['original_char']):04X}" and e["replacement_codepoint"]==f"U+{ord(e['replacement_char']):04X}","codepoint")
        word=next((m.span() for m in re.finditer(r"\w+",caption) if m.start()<=p<m.end()),None)
        need(p not in used and word is not None and word not in usedwords,"shared word/position rule")
        replay[p]=e["replacement_char"];used.add(p);usedwords.add(word);editedids.append(e["attribute_id"])
    adv=a.get("adversarial",caption);need("".join(replay)==adv and len(adv)==len(caption),"exact Unicode replay")
    legal=q["legal_candidate_counts"]
    initial={i for i,x in enumerate(legal) if x>0} if q["eligible"] else set()
    if importance is not None and q["eligible"]:
        need(len(importance)==N and natural([importance[i] for i in sorted(initial)])["k"]==K,"recorded natural K")
    elif not q["eligible"]:need(K==0 and not edits,"ineligible fixed handling")
    masks=[];already=set()
    for d in ad:
        remaining=initial-already;actual=set(d["legal_indices"])-{N}
        need(actual<=remaining,"mask added illegal attr")
        masks.append(dict(removed=sorted(remaining-actual),selected_before=sorted(already)))
        if d["index"]!=N:already.add(d["index"])
    return dict(candidate_count=N,legal_count=len(initial),qualification_legal_count=sum(x>0 for x in legal),
        K=K,selected_count=len(selected),edit_count=len(edits),position_NOOP=noop,STOP=int(stopped),
        unused_K=unused,STOP_unused_slots=stopslots,initial_legal=sorted(initial),selected_indices=selected,
        edited_indices=[ids.index(x) for x in editedids],mask_removed_events=sum(len(x["removed"]) for x in masks),
        mask_removals=masks,has_importance=int(importance is not None),eligible=int(q["eligible"]),
        char_length_delta=len(adv)-len(caption),utf8_length_delta=len(adv.encode())-len(caption.encode()),
        tokenizer_content_delta=(a.get("tokenizer_audit") or {}).get("content_length_delta"))
def strict_alignment(gattrs,hattrs,caption):
    gg=groups(gattrs,caption);hh=groups(hattrs,caption);allkeys=gg.keys()|hh.keys();rows=[]
    for k in sorted(allkeys):
        gs=gg.get(k,[]);hs=hh.get(k,[])
        status="AMBIGUOUS" if len(gs)>1 or len(hs)>1 else "EXACT" if gs and hs else "G_ONLY" if gs else "H_ONLY"
        rows.append((k,status,gs,hs))
    return rows
def overlap(a,b):
    return dict(intersection=len(a&b),union=len(a|b),jaccard=len(a&b)/len(a|b) if a|b else None,
                G_only=len(a-b),H_only=len(b-a))
def process(scope,epoch,row_id,g,q,a,query=None):
    cap=q["caption"];need(g["caption"]==cap and g["pid"]==q["identity"]["pid"] and g["role"]==q["identity"]["role"],"strict sample identity")
    ga=g["ranked"];ha=a.get("attributes",q["attributes"]);br=breakdown(a,q);al=strict_alignment(ga,ha,cap)
    keyg={k:i for k,s,gs,hs in al if s!="AMBIGUOUS" for i,_ in gs}
    keyh={k:i for k,s,gs,hs in al if s!="AMBIGUOUS" for i,_ in hs}
    exact={k for k,s,gs,hs in al if s=="EXACT"}
    gs={key(x,cap) for x in g["selected"]};hs={key(ha[i],cap) for i in br["selected_indices"]};he={key(ha[i],cap) for i in br["edited_indices"]}
    valid=not any(s=="AMBIGUOUS" for k,s,gs0,hs0 in al)
    imp=a.get("predicted_importance");legal=set(br["initial_legal"])
    comparable=valid and bool(imp) and len(ga)>=2 and len(exact)==len(ga)==len(ha)
    comparison_keys=[k for k in keyg if k in keyh and keyh[k] in legal] if imp is not None else []
    applicable=comparable and len(comparison_keys)>=2
    rank=corr([ga[keyg[k]]["S_raw"] for k in comparison_keys],[imp[keyh[k]] for k in comparison_keys]) if applicable else dict(n=0,spearman=None,pearson=None)
    nonambiguous_keys=set(keyg)|set(keyh)
    go=overlap(gs,hs);ge=overlap(gs,he)
    gtop={key(x,cap) for x in ga if x["S_raw"]==ga[0]["S_raw"]} if ga else set()
    hmax=max((imp[i] for i in legal),default=None) if imp is not None else None
    htop={key(ha[i],cap) for i in legal if imp[i]==hmax} if hmax is not None else set()
    r=dict(scope=scope,H_epoch=epoch,row_id=row_id,caption_sha256=hashlib.sha256(cap.encode()).hexdigest(),
        G_K=g["K"],H_K=br["K"],K_H_minus_G=br["K"]-g["K"],**{k:v for k,v in br.items() if k not in ("mask_removals","initial_legal","selected_indices","edited_indices","K")},
        G_candidates=len(g["all"]),G_scored=g["scored"],G_positive_candidates=len(ga),
        candidate_exact=len(exact),G_candidate_only=len(set(keyg)-exact),H_candidate_only=len(set(keyh)-exact),
        ambiguous_attributes=sum(s=="AMBIGUOUS" for k,s,gg,hh in al),candidate_sets_equal=int(valid and len(exact)==len(ga)==len(ha)),
        rank_applicable=int(applicable),rank_n=rank["n"],rank_spearman=rank["spearman"],
        G_H_top1_applicable=int(valid and bool(gtop) and bool(htop)),
        G_H_top1_exact_any=int(bool(gtop&htop)) if valid and gtop and htop else None,
        G_top1_edited=int(bool(gtop&he)) if valid and gtop else None,
        G_top1_selected=int(bool(gtop&hs)) if valid and gtop else None,
        G_selected_H_selected_jaccard=go["jaccard"] if valid else None,
        G_selected_H_edited_jaccard=ge["jaccard"] if valid else None,
        G_selected_H_edited_intersection=ge["intersection"] if valid else None,
        G_selected_H_edited_union=ge["union"] if valid else None,
        G_selected_not_H_edited=ge["G_only"] if valid else None,H_edited_not_G_selected=ge["H_only"] if valid else None,
        G_selected_H_selected_intersection=go["intersection"] if valid else None,
        G_selected_H_selected_union=go["union"] if valid else None,
        G_Sraw_mass_edited=sum(x["S_raw"] for x in ga if key(x,cap) in he)/sum(x["S_raw"] for x in ga) if ga else None,
        source_G_metadata_sha256=g["metadata_sha256"],H_action_sha256=digest(a),
        sampled_greedy_same_text=None,gradient_norm=None)
    if query:
        for k in ("AP","rank1","AP_delta_vs_image_only_pp","best_positive_rank","margin"):r[k]=float(query[k])
    attrs=[]
    for k,status,gg,hh in al:
        gi,gv=gg[0] if len(gg)==1 else (None,None);hi,hv=hh[0] if len(hh)==1 else (None,None)
        attrs.append(dict(scope=scope,H_epoch=epoch,row_id=row_id,alignment_status=status,
            **dict(zip(FIELDS,k)),G_id=gv["instance_id"] if gv else None,H_id=attr_id(hv) if hv else None,
            G_S_raw=gv["S_raw"] if gv else None,H_importance=imp[hi] if imp is not None and hi is not None else None,
            G_order=gi+1 if gi is not None else None,H_legal=int(hi in legal) if hi is not None else None,
            G_selected=int(k in gs),H_selected=int(k in hs),H_edited=int(k in he),H_epoch_frozen=int(scope=="Validation")))
    maps=[]
    adv=a.get("adversarial",cap);edits=a.get("edits",[])
    for i,hv in enumerate(ha):
        sp=(hv["char_start"],hv["char_end"]);affected=[e for e in edits if sp[0]<=e["position"]<sp[1]]
        maps.append(dict(scope=scope,H_epoch=epoch,row_id=row_id,H_id=attr_id(hv),**{k:hv[k] for k in FIELDS},
            edited_raw=adv[sp[0]:sp[1]],span_coordinates_preserved=int(len(cap)==len(adv)),
            UTF8_start_original=len(cap[:sp[0]].encode()),UTF8_start_edited=len(adv[:sp[0]].encode()),
            UTF8_end_original=len(cap[:sp[1]].encode()),UTF8_end_edited=len(adv[:sp[1]].encode()),
            H_selected=int(i in br["selected_indices"]),H_edited=int(i in br["edited_indices"]),
            G_selected=int(key(hv,cap) in gs),edits_affecting_span=len(affected),
            stable_original_anchor=1,attacked_literal_still_matches=int(cap[sp[0]:sp[1]]==adv[sp[0]:sp[1]]),
            original_caption_sha256=hashlib.sha256(cap.encode()).hexdigest(),adversarial_caption_sha256=hashlib.sha256(adv.encode()).hexdigest(),
            actions_json=json.dumps(affected,ensure_ascii=False,separators=(",",":"))))
    return r,attrs,maps
def summarize(rows):
    result={}
    for scope in ("Train","Validation"):
        for ep in EPOCHS:
            xs=[x for x in rows if x["scope"]==scope and x["H_epoch"]==ep]
            if not xs:continue
            z=dict(samples=len(xs),eligible=sum(x["eligible"] for x in xs),importance_available=sum(x["has_importance"] for x in xs),
                candidate_sets_equal=sum(x["candidate_sets_equal"] for x in xs),ambiguous_samples=sum(x["ambiguous_attributes"]>0 for x in xs))
            for k in ("G_K","H_K","candidate_count","legal_count","selected_count","edit_count","unused_K","position_NOOP",
                      "STOP_unused_slots","mask_removed_events","G_selected_H_edited_jaccard","G_selected_H_selected_jaccard",
                      "G_selected_not_H_edited","H_edited_not_G_selected","rank_spearman","G_Sraw_mass_edited"):
                z[k]=dist(x[k] for x in xs if x[k] is not None)
            for k in ("H_K","edit_count","K_H_minus_G","unused_K"):
                z[k+"_counts"]=dict(sorted(Counter(str(x[k]) for x in xs).items()))
            for k in ("G_H_top1_exact_any","G_top1_edited","G_top1_selected"):
                vals=[x[k] for x in xs if x[k] is not None];z[k]=dict(n=len(vals),count=sum(vals),fraction=sum(vals)/len(vals) if vals else None)
            z["STOP_rollouts"]=sum(x["STOP"] for x in xs);z["NOOP_outputs"]=sum(x["edit_count"]==0 for x in xs)
            z["K_sum"]=sum(x["H_K"] for x in xs);z["edits_sum"]=sum(x["edit_count"] for x in xs)
            z["position_NOOP_sum"]=sum(x["position_NOOP"] for x in xs);z["STOP_unused_slots_sum"]=sum(x["STOP_unused_slots"] for x in xs)
            z["legal_many_edits_few"]=sum(x["legal_count"]>=5 and x["edit_count"]<=1 for x in xs)
            z["legal_many_denominator"]=sum(x["legal_count"]>=5 for x in xs)
            z["correlations"]={}
            pairs=[("candidate_count","unused_K"),("legal_count","unused_K"),("H_K","unused_K")]
            if scope=="Validation":pairs += [(a,b) for a in ("G_selected_H_edited_jaccard","unused_K","edit_count","H_K","G_Sraw_mass_edited") for b in ("AP","AP_delta_vs_image_only_pp")]
            for a,b in pairs:
                vv=[x for x in xs if x.get(a) is not None and x.get(b) is not None]
                z["correlations"][a+"__"+b]=corr([x[a] for x in vv],[x[b] for x in vv])
            result[scope+"_H"+str(ep)]=z
    return result
def main():
    global WHITELIST
    start=time.monotonic();need(not OUT.exists(),"independent new output must not exist")
    OUT.mkdir(parents=True)
    oldmanifest=json.loads((PREVIOUS/"INPUT_MANIFEST.json").read_text())
    I.previous={x["path"]:x["sha256"] for x in oldmanifest};I.add(PREVIOUS/"INPUT_MANIFEST.json")
    protected=I.read("/home/lzf/ldx/tmp/hv1_g60_adaptation_development_20261021/protected_before.json")
    for p,v in protected.items():need(st(p)==v,"historical protection before")
    priorpreserve_dirs=[BASE,Path("/home/lzf/ldx/tmp/hv1_g60_adaptation_epoch60_v1"),
        Path("/home/lzf/ldx/projects/idea-test-tbps-hv1-g60-audit-fix-20261022/docs/framework/results/hv1_g60_adaptation_epoch60")]
    protected2={str(p):st(p) for d in priorpreserve_dirs for p in d.rglob("*") if p.is_file()}
    conf=I.read(EXEC/"configs/framework/hv1_g60_adaptation.server.json")
    for p,h in conf["producers"].items():I.add(p,h)
    for node in ast.parse((H/"constraints.py").read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="CONFUSABLES" for t in node.targets):WHITELIST=ast.literal_eval(node.value)
    need(WHITELIST is not None,"verified whitelist absent")
    gm=I.read(G/"outputs/conditions_full/manifest.json")
    gd=I.read(G/"outputs/conditions_full/completed.json")
    need(sha(G/"outputs/conditions_full/manifest.json")==gd["manifest_sha256"],"G cache manifest")
    gp=I.read(gm["protocol_path"],gm["protocol_sha256"])
    for p,h in gp["producer_sha256"].items():I.add(p,h)
    # 原条件的历史来源逐字节SHA，不反序列化任何Tensor。
    for p,h in gp["upstream_sha256"].items():I.add(p,h)
    ann=I.read("/home/lzf/TBPS/Datasets/CUHK-PEDES/reid_raw.json")
    gby={};grows=[];gattrs=[];gdist={}
    for e in gm["entries"]:
        md=I.read(e["metadata_path"],e["metadata_sha256"]);I.add(e["tensor_path"],e["tensor_sha256"])
        src=md["source_score_record"];sel=md["source_selection"]
        need(src["row_id"]==e["row_id"] and src["caption"]==e["caption"],"G identity")
        source=ann[int(e["row_id"].split(":")[1])]
        need(source["split"]=="train" and source["id"]==e["pid"] and source["file_path"]==e["image_rel_path"] and source["captions"][e["caption_index"]]==e["caption"],"annotation identity")
        ranked=sel["ranked_attributes"];need(ranked[:e["K"]]==sel["selected_attributes"],"G selected prefix")
        ns=natural([a["S_raw"] for a in ranked])
        need(ns["k"]==e["K"]==sel["natural_k"]==sel["k_star"] and sel["gamma"]==3 and sel["beta"]==1,"G uncapped K")
        need(md["selected_instance_ids"]==[x["instance_id"] for x in sel["selected_attributes"]],"G selection ids")
        record=dict(caption=e["caption"],pid=e["pid"],role=e["role"],ranked=ranked,selected=sel["selected_attributes"],K=e["K"],
            all=src["attributes"],scored=sum(a["status"]=="scored" for a in src["attributes"]),metadata_sha256=e["metadata_sha256"])
        gby[e["row_id"]]=record
        grows.append(dict(row_id=e["row_id"],role=e["role"],candidate_count=len(src["attributes"]),scored_count=record["scored"],
            positive_candidate_count=len(ranked),K=e["K"],optimizer_updates=(e["K"]+1 if e["role"]=="train" else 0),theoretical_Kplus1=e["K"]+1,entropy=ns["entropy"],neff=ns["neff"],
            unsupported_count=sum(a["status"]!="scored" for a in src["attributes"]),zero_score_count=sum(a.get("S_raw")==0 for a in src["attributes"]),
            metadata_sha256=e["metadata_sha256"]))
        chosen=set(md["selected_instance_ids"])
        for a in src["attributes"]:
            v=a["instance"]
            if not v["unresolved"]:key(v,e["caption"])
            gattrs.append(dict(row_id=e["row_id"],role=e["role"],**{k:v.get(k) for k in FIELDS},instance_id=a["instance_id"],
                status=a["status"],S_raw=a.get("S_raw"),G_selected=int(a["instance_id"] in chosen),
                basis_rank=a.get("rank"),metadata_sha256=e["metadata_sha256"]))
    for role,n in (("train",2048),("validation",128)):
        xs=[x for x in grows if x["role"]==role];need(len(xs)==n,"G dataset coverage")
        gdist[role]=dict(samples=n,candidates=dist(x["candidate_count"] for x in xs),scored=dist(x["scored_count"] for x in xs),
            positive=dist(x["positive_candidate_count"] for x in xs),K=dist(x["K"] for x in xs),K_counts=dict(Counter(str(x["K"]) for x in xs)),
            attributes_total=sum(x["K"] for x in xs),updates=sum(x["optimizer_updates"] for x in xs),
            scores_all=dist(x["S_raw"] for x in gattrs if x["role"]==role and x["S_raw"] is not None),
            scores_selected=dist(x["S_raw"] for x in gattrs if x["role"]==role and x["G_selected"] and x["S_raw"] is not None))
    q={role:{x["identity"]["row_id"]:x for x in I.read(HIST/"qualification"/(role+".json"))["entries"]} for role in ("train","validation")}
    need(set(q["train"])=={k for k,x in gby.items() if x["role"]=="train"} and set(q["validation"])=={k for k,x in gby.items() if x["role"]=="validation"},"G/H strict member sets")
    need(not {x["identity"]["pid"] for x in q["train"].values()}&{x["identity"]["pid"] for x in q["validation"].values()},"PID isolation")
    prior=I.read(conf["sources"]["history_evidence"]["path"],conf["sources"]["history_evidence"]["sha256"])
    perq=list(csv.DictReader((PREVIOUS/"per_query_analysis.csv").open()));I.add(PREVIOUS/"per_query_analysis.csv")
    qp={(int(x["candidate"][1:]),x["row_id"]):x for x in perq if x["group"]=="Joint"}
    rows=[];aligned=[];mapping=[];examples=[];actions_val={}
    def consume(scope,epoch,rid,a,log=None):
        role="train" if scope=="Train" else "validation";qual=q[role][rid]
        r,aa,mm=process(scope,epoch,rid,gby[rid],qual,a,qp.get((epoch,rid)) if scope=="Validation" else None)
        if log:
            r["sampled_greedy_same_text"]=int(log["sampled"]["adversarial"]==log["greedy"]["adversarial"]) if qual["eligible"] else None
            r["gradient_norm"]=log.get("gradient_norm")
        rows.append(r);aligned.extend(aa);mapping.extend(mm)
        if scope=="Validation":actions_val[(epoch,rid)]=a
        if r["legal_count"]>=5 and r["edit_count"]<=1 and len(examples)<12:
            examples.append(dict(scope=scope,H_epoch=epoch,row_id=rid,original=qual["caption"],action=a,budget=r))
    for epoch in EPOCHS:
        vroot=HIST/"training" if epoch==9 else BASE/"training";seen=set()
        for version in sorted((vroot/"checkpoints").glob("visit_*")):
            md=I.read(version/"checkpoint.json")
            if md["cursor"]<=(epoch-1)*2048 or md["start_cursor"]>=epoch*2048:continue
            p=version/"steps.jsonl";I.add(p,md["sample_log_sha256"])
            with p.open() as f:
                for line in f:
                    d=json.loads(line)
                    if (d["visit"]-1)//2048+1!=epoch:continue
                    rid=d["row_id"];need(rid not in seen,"duplicate H train sample");seen.add(rid)
                    qual=q["train"][rid];a=d.get("greedy")
                    if a is None:
                        need(not qual["eligible"],"missing eligible H action")
                        a=dict(original=qual["caption"],adversarial=qual["caption"],attributes=qual["attributes"],budget_k=0,selected_attribute_ids=[],decisions=[],edits=[])
                    consume("Train",epoch,rid,a,d if qual["eligible"] else None)
        need(seen==set(q["train"]),"complete H Train epoch")
        print("TRAIN_ACTIONS_PARSED",epoch,len(seen),flush=True)
        if epoch==9:
            c=next(x for x in prior["candidates"] if x["epoch"]==epoch);v=Path(c["validation_summary_path"]).parent
            I.read(v/"summary.json",c["validation_summary_sha256"]);vp=I.read(v/"protocol.json")
            observed={}
            for unit in (v/"queries").iterdir():
                rc=I.read(unit/"receipt.json");a=I.read(unit/"metadata.json",rc["metadata_sha256"])
                I.add(unit/"text.pt",rc["feature_sha256"])
                need(a["protocol_sha256"]==digest(vp),"H9 validation protocol")
                rid=a["identity"]["row_id"];need(a["identity"]==q["validation"][rid]["identity"],"H9 identity")
                observed[rid]=a["action"]
            need(set(observed)==set(q["validation"]),"H9 complete validation")
            for rid,a in observed.items():consume("Validation",epoch,rid,a)
        else:
            v=BASE/"validation"/f"epoch_{epoch:04d}";s=I.read(v/"completed/summary.json")
            for i,rid in enumerate(q["validation"]):
                p=v/f"query-{i:03d}";m=I.read(p/"manifest.json");done=I.read(p/"completed.json")
                need(sha(p/"manifest.json")==done["manifest_sha256"],"new validation manifest")
                for file,h in m["files_sha256"].items():I.add(p/file,h)
                rec=I.read(p/"record.json");need(rec["identity"]==q["validation"][rid]["identity"],"H identity")
                consume("Validation",epoch,rid,rec["action"])
    summary=summarize(rows)
    # 只使用已保存逐Query派生指标，不重算分数或排名。
    a={r["row_id"]:r for r in rows if r["scope"]=="Validation" and r["H_epoch"]==20}
    b={r["row_id"]:r for r in rows if r["scope"]=="Validation" and r["H_epoch"]==60}
    changes=[dict(row_id=k,delta_unused_K=b[k]["unused_K"]-a[k]["unused_K"],delta_AP_pp=100*(b[k]["AP"]-a[k]["AP"]),
        delta_edits=b[k]["edit_count"]-a[k]["edit_count"],delta_overlap=b[k]["G_selected_H_edited_jaccard"]-a[k]["G_selected_H_edited_jaccard"]
        if b[k]["G_selected_H_edited_jaccard"] is not None and a[k]["G_selected_H_edited_jaccard"] is not None else None) for k in a]
    delta_stats={k:corr([x[k] for x in changes if x[k] is not None],[x["delta_AP_pp"] for x in changes if x[k] is not None])
        for k in ("delta_unused_K","delta_edits","delta_overlap")}
    categories=[]
    for role in ("train","validation"):
        gs=[x for x in gattrs if x["role"]==role];counts=Counter((x["semantic_axis"],x["region"]) for x in gs if x["G_selected"])
        for (axis,region),n in counts.items():categories.append(dict(scope="G_"+role,H_epoch=None,axis=axis,region=region,count=n,denominator=sum(counts.values()),fraction=n/sum(counts.values())))
    for scope in ("Train","Validation"):
        for epoch in EPOCHS:
            aa=[x for x in aligned if x["scope"]==scope and x["H_epoch"]==epoch]
            for group in ("G_selected","H_selected","H_edited"):
                cnt=Counter((x["semantic_axis"],x["region"]) for x in aa if x[group]);total=sum(cnt.values())
                for (axis,region),n in cnt.items():categories.append(dict(scope=scope+"_"+group,H_epoch=epoch,axis=axis,region=region,count=n,denominator=total,fraction=n/total))
    for p,v in protected.items():need(st(p)==v,"historical protection after")
    for p,v in protected2.items():need(st(p)==v,"training/failed report changed")
    I.finish()
    writec("g_attribute_statistics.csv",grows);writec("g_attribute_scores.csv",gattrs)
    writec("h_attribute_statistics.csv",rows);writec("k_budget_comparison.csv",rows)
    writec("attribute_alignment.csv",aligned);writec("attribute_span_mapping.csv",mapping)
    writec("attribute_category_statistics.csv",categories);writec("validation_H20_H60_changes.csv",changes)
    overlaprows=[]
    for k,x in summary.items():
        row=dict(cohort=k,samples=x["samples"],importance_available=x["importance_available"],candidate_sets_equal=x["candidate_sets_equal"],ambiguous_samples=x["ambiguous_samples"])
        for n in ("G_H_top1_exact_any","G_top1_edited","G_top1_selected"):row[n+"_denominator"]=x[n]["n"];row[n+"_count"]=x[n]["count"];row[n+"_fraction"]=x[n]["fraction"]
        for n in ("G_selected_H_edited_jaccard","G_selected_H_selected_jaccard","rank_spearman","G_selected_not_H_edited","H_edited_not_G_selected"):
            row[n+"_denominator"]=x[n]["n"];row[n+"_mean"]=x[n]["mean"];row[n+"_median"]=x[n]["median"]
        overlaprows.append(row)
    writec("attribute_overlap_statistics.csv",overlaprows)
    writej("statistics.json",dict(G=gdist,H=summary,paired_changes=delta_stats,examples=examples,
        coverage=dict(G_Train=2048,G_Validation=128,H_Train_actions=6144,H_Validation_actions=384,attribute_rows=len(aligned),mapping_rows=len(mapping)),
        constraints=dict(no_GPU=True,no_neural_forward=True,no_Tokenizer_execution=True,no_new_text=True,no_new_scores=True,no_new_rankings=True),
        historical_protected_files=len(protected),additional_preserved_files=len(protected2),
        analysis_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
    writej("INPUT_MANIFEST.json",list(I.files.values()))
    writej("SOURCE_EVIDENCE.json",dict(schema="gh-attribute-alignment-audit-1",status="CPU_ATTRIBUTE_ALIGNMENT_PASS",
        base_commit="23fd02daebdcd28bc2e77eb18de1e6fd04432518",input_files=len(I.files),input_manifest_sha256=sha(OUT/"INPUT_MANIFEST.json"),
        original_producers_verified=len(conf["producers"]),G_producers_verified=len(gp["producer_sha256"]),
        historical_protected_files=len(protected),additional_preserved_files=len(protected2),analysis_script_sha256=sha(__file__),
        limitations=["Train actions are online per-visit snapshots, not frozen epoch-end full Train inference",
        "H9 trained under G10, H20/H60 adapted under G60; source and duration confounded",
        "G records are original clean-caption sensitivity, not observed attacked image regions",
        "No causal K intervention, no new model/Tokenizer encoding or new ranking"]))
    print("GH_ATTRIBUTE_ALIGNMENT_CPU_PASS",json.dumps(dict(G=gdist,H=summary)),flush=True)
if __name__=="__main__":main()
