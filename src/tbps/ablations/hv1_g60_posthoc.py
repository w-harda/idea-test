"""H60事后CPU分析：只读日志/已存128×128分数，无模型、训练、编码或审计入口调用。"""
from pathlib import Path
from collections import Counter,defaultdict
from unittest.mock import patch
import ast,argparse,csv,hashlib,json,math,os,resource,sys,time,platform
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
BASE=Path("/home/lzf/ldx/outputs/tbps-framework-hv1-g60-adaptation-epoch60")
EXEC=Path("/home/lzf/ldx/projects/idea-test-tbps-hv1-g60-adaptation-20261021")
REPAIR=Path("/home/lzf/ldx/projects/idea-test-tbps-hv1-g60-audit-fix-20261022")
CONTROL=Path("/home/lzf/ldx/tmp/hv1_g60_adaptation_epoch60_v1")
OUT=ROOT/"docs/framework/results/hv1_g60_posthoc_analysis"
KEYS=("R@1","R@5","R@10","mAP")
def need(x,msg):
    if not x:raise ValueError(msg)
def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def stat(p):
    s=Path(p).stat();return [s.st_size,s.st_mtime_ns,s.st_ino]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(2**20),b""):h.update(b)
    return h.hexdigest()
class Inputs:
    def __init__(self):self.files={};self.artifacts=set()
    def add(self,p,expected=None):
        p=Path(p).resolve();name=str(p)
        if name not in self.files:
            before=stat(p);h=sha(p);need(stat(p)==before,"source changed during reading: "+name)
            self.files[name]=dict(path=name,sha256=h,size=before[0],mtime_ns=before[1],inode=before[2])
        if expected:need(self.files[name]["sha256"]==expected,"source SHA mismatch: "+name)
        return self.files[name]
    def streamed(self,p,before,h,expected):
        need(stat(p)==before and h==expected,"log SHA/metadata mismatch: "+str(p))
        self.files[str(Path(p).resolve())]=dict(path=str(Path(p).resolve()),sha256=h,size=before[0],mtime_ns=before[1],inode=before[2])
    def read(self,p,expected=None):
        self.add(p,expected);return json.loads(Path(p).read_text())
    def artifact(self,p):
        p=Path(p).resolve()
        if str(p) in self.artifacts:return self.read(p/"manifest.json")
        done=self.read(p/"completed.json");m=self.read(p/"manifest.json",done["manifest_sha256"])
        need(m["artifact_id"]==p.name,"artifact identity")
        actual={str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()}
        need(actual==set(m["files_sha256"])|{"manifest.json","completed.json"},"artifact file set")
        for n,h in m["files_sha256"].items():self.add(p/n,h)
        for ref in m["sources"]:self.add(ref["path"],ref["sha256"])
        self.artifacts.add(str(p));return m
    def finish(self):
        for p,v in self.files.items():need(stat(p)==[v["size"],v["mtime_ns"],v["inode"]],"input metadata changed: "+p)
I=Inputs()
def write_json(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def write_csv(name,rows):
    need(bool(rows),"empty CSV "+name)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open("x",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,keys,lineterminator="\n");w.writeheader()
        for r in rows:w.writerow(r)
def dist(values):
    a=np.asarray(values,dtype=np.float64);need(a.size and np.isfinite(a).all(),"finite nonempty metric")
    out=dict(n=len(a),mean=float(a.mean()),std=float(a.std()),min=float(a.min()),max=float(a.max()),
             zero_count=int((a==0).sum()),positive_count=int((a>0).sum()),negative_count=int((a<0).sum()))
    for key,q in [("p05",.05),("p25",.25),("median",.5),("p75",.75),("p95",.95)]:out[key]=float(np.quantile(a,q))
    return out

ACTION_HIST=[];ACTION_SUM=[];TYPE_ROWS=[];DIST_ROWS=[]
class Actions:
    def __init__(self,scope,epoch,mode):
        self.scope,self.epoch,self.mode=scope,epoch,mode
        self.n=0;self.s=Counter();self.h=defaultdict(Counter)
    def add(self,a,original,qualification,recorded=None):
        edits=a.get("edits",[]);k=a.get("budget_k",0);attrs=a.get("attributes",qualification.get("attributes",[]))
        need(len(edits)<=k,"edit budget")
        chars=list(original);positions=set();spans=set()
        for e in edits:
            pos=e["position"];span=e["source_span"]
            need(pos not in positions and tuple(span) not in spans,"one edit per word/attribute span")
            need(original[pos]==e["original_char"] and original[span[0]:span[1]]==e["source_raw"],"edit trace")
            need(e["original_codepoint"]==f"U+{ord(original[pos]):04X}" and
                 e["replacement_codepoint"]==f"U+{ord(e['replacement_char']):04X}","Unicode codepoint")
            need(any(x["semantic_axis"]==e["semantic_axis"] and x["region"]==e["region"] and
                     [x["char_start"],x["char_end"]]==span for x in attrs),"attribute/span binding")
            chars[pos]=e["replacement_char"];positions.add(pos);spans.add(tuple(span))
            for key,value in [("axis",e["semantic_axis"]),("axis_region",e["semantic_axis"]+"/"+e["region"]),
                              ("canonical",e["semantic_axis"]+":"+e["canonical_value"]),
                              ("character",e["original_codepoint"]+"→"+e["replacement_codepoint"]),
                              ("replacement",e["replacement_codepoint"]),("position_in_attribute",e["position_in_attribute"])]:
                self.h[key][str(value)]+=1
        adversarial=a.get("adversarial",original)
        need("".join(chars)==adversarial,"Unicode exact replay")
        decisions=a.get("decisions",[])
        ad=[x for x in decisions if x["head"]=="attribute"];pd=[x for x in decisions if x["head"]=="position"]
        stops=sum(x["index"]==len(attrs) for x in ad)
        pnoops=sum(x["index"]==max(x["legal_indices"]) for x in pd)
        if recorded:
            need(recorded["edits"]==len(edits) and recorded["STOP_count"]==stops and
                 recorded["NOOP_output"]==int(not edits) and recorded["position_NOOP_count"]==pnoops,"recorded action stats")
        self.n+=1;self.s.update(dict(edits=len(edits),budget=k,attributes=len(attrs),
            legal_attributes=qualification.get("legal_attribute_count",0),NOOP=int(not edits),
            STOP_rollouts=int(stops>0),STOP_decisions=stops,attribute_decisions=len(ad),
            position_NOOP_decisions=pnoops,position_decisions=len(pd),selected=len(a.get("selected_attribute_ids",[]))))
        for key,value in [("edits",len(edits)),("K",k),("selected_attributes",len(a.get("selected_attribute_ids",[]))),
                          ("attributes",len(attrs)),("legal_attributes",qualification.get("legal_attribute_count",0))]:
            self.h[key][str(value)]+=1
        t=a.get("tokenizer_audit")
        if t:
            need(not t.get("new_truncation",False),"new BPE truncation")
            self.s["tokenizer_audited"]+=1
            self.s["tokens_changed"]+=int(t["model_tokens_changed"])
            self.s["raw_changed"]+=int(t["raw_changed"])
            self.s["token_delta_sum"]+=t["content_length_delta"]
            self.h["token_length_delta"][str(t["content_length_delta"])]+=1
            if "model_tokens" in t.get("before",{}) and "model_tokens" in t.get("after",{}):
                a0=t["before"]["model_tokens"];a1=t["after"]["model_tokens"]
                need(len(a0)==len(a1)==77,"recorded BPE77")
                self.h["token_hamming"][str(sum(x!=y for x,y in zip(a0,a1)))]+=1
        return dict(edits=len(edits),budget=k,STOP=int(stops>0),NOOP=int(not edits))
    def finish(self):
        n=self.n;s=self.s;row=dict(scope=self.scope,H_epoch=self.epoch,rollout=self.mode,actions=n,**s)
        for k in ("edits","budget","attributes","legal_attributes","selected"):row["mean_"+k]=s[k]/n
        row["NOOP_rate"]=s["NOOP"]/n;row["STOP_rollout_rate"]=s["STOP_rollouts"]/n
        row["STOP_decision_rate"]=s["STOP_decisions"]/s["attribute_decisions"] if s["attribute_decisions"] else 0.
        row["position_NOOP_rate"]=s["position_NOOP_decisions"]/s["position_decisions"] if s["position_decisions"] else 0.
        row["budget_utilization"]=s["edits"]/s["budget"] if s["budget"] else 0.
        row["mean_token_length_delta"]=s["token_delta_sum"]/s["tokenizer_audited"] if s["tokenizer_audited"] else None
        ACTION_SUM.append(row)
        for key,counts in self.h.items():
            total=sum(counts.values())
            for value,count in sorted(counts.items()):
                ACTION_HIST.append(dict(scope=self.scope,H_epoch=self.epoch,rollout=self.mode,dimension=key,
                                        value=value,count=count,denominator=total,fraction=count/total))
        return row

def native_rank():
    p=Path("/home/lzf/Attack/IRRA-main/IRRA-main/utils/metrics.py")
    I.add(p)
    tree=ast.parse(p.read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=="rank")
    ns={"torch":torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),"exec"),ns)
    return ns["rank"]
def scores(path):
    path=Path(path);need(path.name=="scores.pt","only stored score tensors may be deserialized")
    I.add(path);x=torch.load(path,map_location="cpu",weights_only=True)
    if isinstance(x,dict):need(set(x)=={"scores"},"score payload");x=x["scores"]
    x=x.detach().clone()
    need(x.shape==(128,128) and x.dtype==torch.float32 and torch.isfinite(x).all(),"finite FP32 128x128 scores")
    return x

def query_metrics(s,pids,official=None):
    need(s.ndim==2 and s.shape[0]==len(pids) and s.shape[1]==len(pids),"query/gallery order")
    rank=torch.argsort(s,dim=1,descending=True);m=pids[rank]==pids[:,None]
    need(m.any(1).all() and (~m).any(1).all(),"positives and negatives")
    cumulative=m.cumsum(1)
    precision=torch.stack([cumulative[:,i]/(i+1.) for i in range(s.shape[1])],1)
    AP=(precision*m).sum(1)/m.sum(1)
    positive=(pids[:,None]==pids[None,:])
    best=s.masked_fill(~positive,-torch.inf).max(1).values
    neg=s.masked_fill(positive,-torch.inf).max(1).values
    first=m.int().argmax(1)+1
    out=dict(AP=AP.numpy().copy(),rank1=m[:,0].numpy().copy(),best_positive_rank=first.numpy().copy(),
        best_positive_score=best.numpy().copy(),hard_negative_score=neg.numpy().copy(),
        margin=(best-neg).numpy().copy(),paired_score=s.diag().numpy().copy(),positives=positive.sum(1).numpy().copy())
    summary={"R@1":float(m[:,:1].any(1).float().mean()*100),
             "R@5":float(m[:,:5].any(1).float().mean()*100),
             "R@10":float(m[:,:10].any(1).float().mean()*100),"mAP":float(AP.mean()*100)}
    if official:
        cmc,map0,_,rank0=official(s,pids,pids)
        need(torch.equal(rank,rank0) and summary["mAP"]==float(map0)
             and [summary[k] for k in ("R@1","R@5","R@10")]==[float(cmc[i]) for i in (0,4,9)],"independent official rank agreement")
    return out,summary

def stream_training(root,protocol,start_cursor,start_updates,phase,qualification,ledger=None):
    root=Path(root);parent=None if start_cursor==0 else protocol["parent_checkpoint_sha256"]
    cursor=start_cursor;updates=start_updates;rows=[];inventory=[]
    metric=defaultdict(list);counts=Counter();acts=None;epoch=None;members=set()
    paths=sorted((root/"checkpoints").glob("visit_*"))
    for version in paths:
        md=I.read(version/"checkpoint.json");I.add(version/"checkpoint.pt",md["checkpoint_sha256"])
        I.add(version/"intent.json",md["intent_sha256"])
        need(md["parent_checkpoint_sha256"]==parent and md["start_cursor"]==cursor and
             md["start_updates"]==updates and md["protocol_sha256"]==digest(protocol),"checkpoint parent/protocol chain")
        h=hashlib.sha256();before=stat(version/"steps.jsonl")
        with (version/"steps.jsonl").open("rb") as f:
            for line in f:
                h.update(line);need(line.endswith(b"\n"),"committed complete log line")
                d=json.loads(line);cursor+=1;epoch_now=(cursor-1)//2048+1
                if epoch is None:
                    epoch=epoch_now;acts={m:Actions(phase,epoch,m) for m in ("sampled","greedy")}
                need(epoch_now==epoch and d["visit"]==cursor and d["row_id"]==protocol["query_order"][cursor-1],"absolute visit/order")
                q=qualification[d["row_id"]];need(d["row_id"] not in members,"duplicate epoch member");members.add(d["row_id"])
                n=d["optimizer_steps_this_step"]
                need(n==int(q["eligible"])==int(protocol["eligible_by_row"][d["row_id"]]),"fixed eligibility")
                counts["visits"]+=1;counts["updates"]+=n;updates+=n
                need(d["committed_update_candidate"]==updates,"Adam accounting")
                if n:
                    for key in ("SCST_loss","advantage","gradient_norm","parameter_delta_l2","parameter_delta_max_abs","sequence_log_prob"):
                        metric[key].append(d[key])
                    for key,value in d.get("group_gradient_norm",{}).items():metric["gradient_"+key].append(value)
                    for mode in ("sampled","greedy"):
                        reward=d[mode+"_reward"];need(reward["total"]==reward["R_sim"]+reward["R_rank"],"original reward weights")
                        need(reward["synergy_increment"]==reward["adv_sim_drop"]-reward["clean_sim_drop"],"synergy diagnostic")
                        for key,value in reward.items():metric[mode+"_"+key].append(value)
                        acts[mode].add(d[mode],q["caption"],q,d.get(mode+"_action_stats"))
                    counts["same_text"]+=int(d["sampled"]["adversarial"]==d["greedy"]["adversarial"])
                    counts["same_actions"]+=int(d["sampled"]["edits"]==d["greedy"]["edits"] and
                      d["sampled"]["decisions"]==d["greedy"]["decisions"] and
                      d["sampled"]["selected_attribute_ids"]==d["greedy"]["selected_attribute_ids"])
                    counts["zero_gradient_parameter_changed"]+=int(d["gradient_norm"]==0 and d["parameter_delta_l2"]>0)
                    need(d["IRRA_text_samples"]==3,"real SCST triplet")
                else:counts["skips"]+=1
                if cursor%2048==0:
                    need(counts["visits"]==2048 and counts["updates"]==2037 and counts["skips"]==11 and len(members)==2048,"complete epoch")
                    row=dict(epoch=epoch,reward_gallery=phase,**counts)
                    for key,values in metric.items():
                        st=dist(values)
                        DIST_ROWS.append(dict(epoch=epoch,reward_gallery=phase,metric=key,**st))
                        for k in ("mean","std","min","p05","p25","median","p75","p95","max","zero_count","positive_count","negative_count"):
                            row[key+"_"+k]=st[k]
                    row["same_text_rate"]=counts["same_text"]/2037;row["same_actions_rate"]=counts["same_actions"]/2037
                    row["zero_gradient_rate"]=row["gradient_norm_zero_count"]/2037
                    row["nonzero_gradient_rate"]=1-row["zero_gradient_rate"]
                    row["zero_advantage_rate"]=row["advantage_zero_count"]/2037
                    for mode in acts:
                        a=acts[mode].finish()
                        for key in ("mean_edits","mean_budget","NOOP_rate","STOP_rollout_rate","STOP_decision_rate","budget_utilization","mean_token_length_delta"):
                            row[mode+"_"+key]=a[key]
                    rows.append(row)
                    print("EPOCH_PARSED",phase,epoch,"reward",row["sampled_total_mean"],"zero_grad",row["zero_gradient_rate"],flush=True)
                    metric=defaultdict(list);counts=Counter();members=set();acts=None;epoch=None
        I.streamed(version/"steps.jsonl",before,h.hexdigest(),md["sample_log_sha256"])
        need(cursor==md["cursor"] and updates==md["updates"],"checkpoint ledger counts")
        parent=md["checkpoint_sha256"];inventory.append(dict(path=str(version),**md))
    need(not counts and not (root/"pending").exists(),"completed trajectory/no pending")
    if ledger:
        need(len(inventory)==len(ledger["published"]) and cursor==ledger["cursor"] and updates==ledger["updates"] and parent==ledger["parent"],"completed ledger")
        for item,md in zip(ledger["published"],inventory):
            need(item["path"]==md["path"] and item["sha256"]==md["checkpoint_sha256"],"ledger paths/SHA")
    return rows,inventory

def paired_bootstrap(a,b,pids,seed=20261022,n=20000):
    ids=np.unique(pids);delta=np.asarray(b,dtype=np.float64)-np.asarray(a,dtype=np.float64)
    sums=np.array([delta[pids==p].sum() for p in ids]);counts=np.array([(pids==p).sum() for p in ids])
    rng=np.random.default_rng(seed);sample=rng.integers(0,len(ids),size=(n,len(ids)))
    means=100*sums[sample].sum(1)/counts[sample].sum(1)
    lo,hi=np.quantile(means,[.025,.975])
    return dict(difference_b_minus_a_pp=float(delta.mean()*100),CI95_lower_pp=float(lo),CI95_upper_pp=float(hi),
                replicates=n,seed=seed,clusters=len(ids),queries=len(pids),
                includes_zero=bool(lo<=0<=hi),method="paired PID-cluster percentile bootstrap; exploratory, selection bias remains")

def compare_pair(a,b,perquery,identities,base,actions):
    qa,qb=perquery[a],perquery[b];d=100*(qb["AP"].astype(float)-qa["AP"].astype(float))
    rdelta=qb["rank1"].astype(int)-qa["rank1"].astype(int)
    pos=np.maximum(d,0);neg=np.minimum(d,0);absolute=np.abs(d);order=np.argsort(-absolute)
    rows=[]
    for i,x in enumerate(identities):
        rows.append(dict(candidate_a=a,candidate_b=b,query_index=i,row_id=x["row_id"],pid=x["pid"],
            AP_a=float(qa["AP"][i]),AP_b=float(qb["AP"][i]),delta_AP_pp=float(d[i]),
            rank1_a=int(qa["rank1"][i]),rank1_b=int(qb["rank1"][i]),rank1_delta=int(rdelta[i]),
            best_rank_a=int(qa["best_positive_rank"][i]),best_rank_b=int(qb["best_positive_rank"][i]),
            margin_a=float(qa["margin"][i]),margin_b=float(qb["margin"][i]),
            edits_a=len(actions[a][i].get("edits",[])),edits_b=len(actions[b][i].get("edits",[]))))
    stat0=dict(a=a,b=b,mean_AP_delta_pp=float(d.mean()),AP_improved_retrieval=int((d>0).sum()),
       AP_worsened_retrieval=int((d<0).sum()),AP_unchanged=int((d==0).sum()),
       rank1_a_correct_b_wrong=int((rdelta<0).sum()),rank1_a_wrong_b_correct=int((rdelta>0).sum()),
       rank1_net_correct_change=int(rdelta.sum()),
       positive_AP_contribution_pp=float(pos.sum()/128),negative_AP_contribution_pp=float(neg.sum()/128),
       top1_absolute_share=float(absolute[order[:1]].sum()/absolute.sum()) if absolute.sum() else 0.,
       top5_absolute_share=float(absolute[order[:5]].sum()/absolute.sum()) if absolute.sum() else 0.,
       top10_absolute_share=float(absolute[order[:10]].sum()/absolute.sum()) if absolute.sum() else 0.,
       largest_absolute_changes=[rows[int(i)] for i in order[:10]],
       rank1_flips=[row for row in rows if row["rank1_delta"]])
    success_a=base["rank1"]&~qa["rank1"];success_b=base["rank1"]&~qb["rank1"]
    union=success_a|success_b
    stat0["new_attack_success"]={"definition":"Image-only rank1 correct -> Joint rank1 wrong",
       "a":int(success_a.sum()),"b":int(success_b.sum()),"intersection":int((success_a&success_b).sum()),
       "union":int(union.sum()),"jaccard":float((success_a&success_b).sum()/union.sum()) if union.any() else None}
    pid=np.array([x["pid"] for x in identities])
    stat0["bootstrap_AP"]=paired_bootstrap(qa["AP"],qb["AP"],pid)
    stat0["bootstrap_R1"]=paired_bootstrap(qa["rank1"],qb["rank1"],pid)
    return stat0,rows

def main():
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU-only analysis required")
    need(not OUT.exists(),"independent analysis output must be new")
    OUT.mkdir(parents=True);torch.set_num_threads(2);begin=time.monotonic()
    config=I.read(EXEC/"configs/framework/hv1_g60_adaptation.server.json")
    history=Path(config["history_root"]);gval=Path(config["validation_root"])
    prior=I.read(config["sources"]["history_evidence"]["path"],config["sources"]["history_evidence"]["sha256"])
    for p,h in config["producers"].items():I.add(p,h)
    need(len(config["producers"])==113,"original producer count")
    for ref0 in (*config["sources"].values(),config["parent"],config["G60"],config["G10"]):
        I.add(ref0["path"],ref0["sha256"])
    protected=I.read(config["sources"]["protection_snapshot"]["path"],config["sources"]["protection_snapshot"]["sha256"])
    for p,s in protected.items():need(stat(p)==s,"historical protection before analysis")
    # 保护训练/Validation/回执/控制及原失败报告现场的所有既有文件。
    preserve_dirs=[BASE,CONTROL,REPAIR/"docs/framework/results/hv1_g60_adaptation_epoch60"]
    preserved={str(p):stat(p) for d in preserve_dirs for p in d.rglob("*") if p.is_file()}
    I.artifact(BASE/"training_audit/result");I.artifact(BASE/"audit_repair_v1/training")
    audit=I.read(BASE/"audit_repair_v1/training/report.json")
    need(audit["status"]=="HV1_G60_TRAINING_CPU_AUDIT_PASS" and audit["cursor"]==122880 and audit["updates"]==122220,"completed independent CPU audit")
    for p in (CONTROL/"phase",CONTROL/"status.log",CONTROL/"cpu_audit.log",CONTROL/"report.log",CONTROL/"training.log"):I.add(p)
    need("hashlib" in (CONTROL/"report.log").read_text(),"preserved publication failure")
    I.artifact(BASE/"gallery_audit/result");I.artifact(BASE/"gallery_train/completed")
    done=I.read(BASE/"training/completed.json");protocol=I.read(BASE/"training/protocol.json")
    need(done["status"]=="HV1_G60_TRAINING_PASS" and done["new_visits"]==102400 and done["new_updates"]==101850,"formal training complete")
    need(protocol["original_training_protocol"]["config"]["epochs"]==10 and protocol["config"]["epochs"]==60,"separate extension contract")
    original=I.read(history/"training/protocol.json");need(original==protocol["original_training_protocol"],"parent contract exact")
    restore=I.read(BASE/"training/restore_20480.json")
    qtrain=I.read(history/"qualification/train.json");qval=I.read(history/"qualification/validation.json")
    trainq={x["identity"]["row_id"]:x for x in qtrain["entries"]}
    valq={x["identity"]["row_id"]:x for x in qval["entries"]}
    trainp={q["identity"]["pid"] for q in trainq.values()}
    valp={q["identity"]["pid"] for q in valq.values()}
    need(not trainp&valp,"Train/Validation PID isolation")
    annotation=I.read(config["sources"]["annotation"]["path"])
    testp={x["id"] for x in annotation if x["split"]=="test"}
    need(not trainp&testp and not valp&testp,"official Test PID isolation")
    del annotation
    oldrows,oldinventory=stream_training(history/"training",original,0,0,"G10",trainq)
    newrows,inventory=stream_training(BASE/"training",protocol,20480,20370,"G60 adaptation",trainq,done["ledger"])
    need(len(newrows)==50 and sum(x["updates"] for x in newrows)==101850,"complete 50-epoch statistics")
    write_csv("training_epoch_statistics.csv",newrows);write_csv("historical_training_epoch_statistics.csv",oldrows)
    write_csv("training_metric_distributions.csv",DIST_ROWS)
    write_csv("checkpoint_inventory.csv",[{**x,"phase":phase} for phase,seq in [("G10",oldinventory),("G60 adaptation",inventory)] for x in seq])
    pref=I.read(gval/"protocol/preflight.json");identities=pref["identities"]
    need(len(identities)==128 and [x["row_id"] for x in identities]==list(valq),"128 original order")
    pids=torch.tensor([x["pid"] for x in identities]);official=native_rank()
    matrices={};qm={};summaries={};action_by_H={};valrows=[];allrows=[];queryrows=[];valactionrows=[]
    matrices["Image-only"]=scores(gval/"score-01/scores.pt");I.artifact(gval/"score-01")
    baseq,basem=query_metrics(matrices["Image-only"],pids,official);qm["Image-only"]=baseq
    gsummary=I.read(gval/"completed/summary.json")
    need(basem==gsummary["groups"]["G60 Image-only"],"Image-only summary exact")
    for candidate in prior["candidates"]:
        epoch=candidate["epoch"];key="H"+str(epoch);v=Path(candidate["validation_summary_path"]).parent
        s=I.read(v/"summary.json",candidate["validation_summary_sha256"])
        vp=I.read(v/"protocol.json");vd=I.read(v/"completed.json")
        need(vp["identities"]==identities and vd["summary_sha256"]==sha(v/"summary.json"),"old H caption/PID/order")
        I.add(candidate["checkpoint_path"],candidate["checkpoint_sha256"])
        by_id={}
        for unit in (v/"queries").iterdir():
            md=I.read(unit/"metadata.json");rc=I.read(unit/"receipt.json")
            I.add(unit/"metadata.json",rc["metadata_sha256"]);I.add(unit/"text.pt",rc["feature_sha256"])
            need(md["protocol_sha256"]==digest(vp),"historical query protocol")
            by_id[md["identity"]["row_id"]]=md
        acts=[];agg=Actions("Validation",epoch,"greedy")
        for i,identity in enumerate(identities):
            md=by_id[identity["row_id"]];need(md["identity"]==identity,"historical query identity")
            action=md["action"];need(md["adversarial"]==action.get("adversarial",identity["caption"]),"old adv text")
            agg.add(action,identity["caption"],valq[identity["row_id"]]);acts.append(action)
        agg.finish();action_by_H[key]=acts
        unit=gval/f"score-{epoch+1:02d}";I.artifact(unit);matrices[key]=scores(unit/"scores.pt")
        qm[key],actual=query_metrics(matrices[key],pids,official)
        need(actual==gsummary["groups"][f"G60 + H{epoch} Joint"],"historical G60 Joint exact")
        summaries[key]={"Text-only":{k:s["groups"]["Text-only"][k] for k in KEYS},"Joint":actual,
                        "G10 Joint":{k:s["groups"]["Joint"][k] for k in KEYS}}
    for epoch in (20,30,40,50,60):
        key="H"+str(epoch);v=BASE/"validation"/f"epoch_{epoch:04d}"
        I.artifact(v/"completed");s=I.read(v/"completed/summary.json")
        need(s["query"]==s["gallery"]==128 and s["all_queries_retained"] and s["G60_image_feature_sha256"]==
             "ae7bce28e383745487812f9fbf0af6bd44cd44312936c905cb27f5df2871a23d","fixed G60 Validation")
        md=next(x for x in inventory if x["cursor"]==epoch*2048)
        need(md["checkpoint_sha256"]==s["checkpoint_sha256"],"candidate checkpoint chain")
        agg=Actions("Validation",epoch,"greedy");acts=[]
        for i,identity in enumerate(identities):
            unit=v/f"query-{i:03d}";I.artifact(unit);rec=I.read(unit/"record.json")
            need(rec["identity"]==identity and rec["checkpoint_sha256"]==s["checkpoint_sha256"],"new caption/PID identity")
            action=rec["action"];agg.add(action,identity["caption"],valq[identity["row_id"]]);acts.append(action)
            need(rec["adversarial"]==action.get("adversarial",identity["caption"]),"new adv text")
        agg.finish();action_by_H[key]=acts
        for name,tag in [("joint",key),("text-only",key+" Text-only")]:
            I.artifact(v/name);matrices[tag]=scores(v/name/"scores.pt")
            qm[tag],actual=query_metrics(matrices[tag],pids,official)
            need(actual==s["groups"]["Joint" if name=="joint" else "Text-only"],"new summary official rank exact")
        summaries[key]={"Text-only":s["groups"]["Text-only"],"Joint":s["groups"]["Joint"]}
    for key,s in summaries.items():
        epoch=int(key[1:]);t=s["Text-only"];j=s["Joint"]
        valrows.append(dict(H_epoch=epoch,training_guide="G10" if epoch<=10 else "G60 adaptation after H10",
          **{"Text-only "+k:t[k] for k in KEYS},**{"Joint "+k:j[k] for k in KEYS},
          Joint_extra_drop_pp=basem["mAP"]-j["mAP"],G60_Image_only_mAP=basem["mAP"],
          **({"G10 Joint "+k:s["G10 Joint"][k] for k in KEYS} if epoch<=10 else {})))
        for group,m in [("Text-only",t),("Joint",j)]:
            allrows.append(dict(H_epoch=epoch,training_guide="G10" if epoch<=10 else "G60 adaptation after H10",group=group,**m))
        # 仅已保存G60 Joint与Image-only矩阵可直接构造Adv奖励诊断，不重新编码。
        r_sim=(matrices["Image-only"].diag()-matrices[key].diag()).numpy()
        r_rank=qm["Image-only"]["margin"]-qm[key]["margin"]
        query_reward=r_sim+r_rank
        for i,identity in enumerate(identities):
            action=action_by_H[key][i];q=qm[key];base=qm["Image-only"]
            row=dict(candidate=key,group="Joint",query_index=i,row_id=identity["row_id"],pid=identity["pid"],
             **{k:(int(v[i]) if k in ("rank1","best_positive_rank","positives") else float(v[i])) for k,v in q.items()},
             AP_delta_vs_image_only_pp=float(100*(float(q["AP"][i])-float(base["AP"][i]))),
             R_sim_posthoc=float(r_sim[i]),R_rank_posthoc=float(r_rank[i]),R_total_posthoc=float(query_reward[i]),
             actual_edits=len(action.get("edits",[])),budget_k=action.get("budget_k",0))
            queryrows.append(row)
        for qrow in queryrows[-128:]:
            valactionrows.append({k:qrow[k] for k in ("candidate","query_index","row_id","pid","actual_edits","budget_k")})
    # 基准行完整保留，旧Text-only没有保存分数矩阵，不凭总体指标编造逐QueryAP。
    for i,identity in enumerate(identities):
        queryrows.append(dict(candidate="Image-only",group="Image-only",query_index=i,row_id=identity["row_id"],pid=identity["pid"],
           **{k:(int(v[i]) if k in ("rank1","best_positive_rank","positives") else float(v[i])) for k,v in baseq.items()}))
    for epoch in (20,30,40,50,60):
        q=qm[f"H{epoch} Text-only"]
        for i,x in enumerate(identities):
            queryrows.append(dict(candidate=f"H{epoch}",group="Text-only",query_index=i,row_id=x["row_id"],pid=x["pid"],
                **{k:(int(v[i]) if k in ("rank1","best_positive_rank","positives") else float(v[i])) for k,v in q.items()}))
    pairs={};pairrows=[]
    for a,b in [("H9","H20"),("H20","H60"),("H7","H9"),("H10","H20"),("H20","H40")]:
        st,rr=compare_pair(a,b,qm,identities,baseq,action_by_H);pairs[a+"_vs_"+b]=st;pairrows+=rr
    write_csv("validation_comparison.csv",valrows);write_csv("validation_metrics_raw.csv",allrows)
    write_csv("per_query_analysis.csv",queryrows);write_csv("per_query_comparisons.csv",pairrows)
    write_csv("validation_action_by_query.csv",valactionrows)
    write_csv("action_distribution.csv",ACTION_HIST);write_csv("action_epoch_statistics.csv",ACTION_SUM)
    # 代表样例由最大AP变化/Rank1翻转选取，仅描述，不影响策略或Checkpoint选择。
    examples=[]
    for pair in ("H9_vs_H20","H20_vs_H60"):
        st=pairs[pair];chosen=[]
        for row in st["rank1_flips"]+st["largest_absolute_changes"]:
            i=row["query_index"]
            if i in chosen:continue
            chosen.append(i)
            examples.append(dict(pair=pair,identity={k:identities[i][k] for k in ("row_id","pid")},
                 original=identities[i]["caption"],effect=row,
                 actions={k:action_by_H[k][i] for k in (st["a"],st["b"])}))
            if len(chosen)==5:break
    # Full背景只读引用已审计的E10/E60结果，不读取或重排正式Full分数。
    full=list(csv.DictReader((ROOT/"docs/framework/results/gv2_epoch_sweep_full_analysis/all_metrics_raw.csv").open()))
    I.add(ROOT/"docs/framework/results/gv2_epoch_sweep_full_analysis/all_metrics_raw.csv")
    full0=[x for x in full if x["epoch"] in ("10","60") or x.get("method") in ("Clean","H-v1 Text-only")]
    for x in full0:
        p=x.get("source")
        if p and Path(p).exists():I.add(p)
    write_csv("historical_full_context.csv",full0)
    for path in [EXEC/"src/tbps/ablations/hv1_g60_adaptation.py",EXEC/"src/tbps/ablations/hv1_g60_state.py",
        REPAIR/"src/tbps/ablations/hv1_g60_audit.py",REPAIR/"src/tbps/ablations/hv1_g60_audit_repair.py",
        Path(config["sources"]["history_evidence"]["path"]).parents[1]/"../REPORT.md"]:
        if path.exists():I.add(path)
    proto=history.parents[1]
    for f in ("rewards.py","scst.py","policy.py","constraints.py","contracts.py","text_frontend.py","tokenizer_probe.py","phase4_training.py","formal_training/runtime.py"):
        I.add(proto/f)
    g=proto.parent/"attribute_image_generator_v2_20261014"
    for f in ("image_only_generator.py","serial_training.py","epoch_training.py","losses.py"):I.add(g/f)
    for f in ("docs/framework/HV1_EPOCH7_TRAINING_SELECTION_AUDIT.md",
              "docs/framework/results/g60_h1_h10_validation/REPORT.md",
              "docs/framework/results/g60_h1_h10_validation/comparison_raw.csv",
              "docs/framework/results/gv2_epoch_sweep_full_analysis/SOURCE_EVIDENCE.json",
              "docs/framework/results/gv2_epoch_sweep_full_analysis/REPORT.md"):
        I.add(ROOT/f)
    I.add(ROOT/"src/tbps/ablations/hv1_g60_posthoc.py")
    for p,s in preserved.items():need(stat(p)==s,"existing experiment or failed report changed")
    for p,s in protected.items():need(stat(p)==s,"historical protection after analysis")
    I.finish()
    statistics=dict(training_completed=done,CPU_audit=audit,restore=restore,
       training_epochs=oldrows+newrows,validation=valrows,controls=dict(Clean_mAP=100.,G60_Image_only=basem),
       query_comparisons=pairs,examples=examples,
       validation_pid_count=len(set(x["pid"] for x in identities)),
       positives_distribution=dict(Counter(map(str,baseq["positives"]))),
       missing=["Historical H1-H10 Text-only score matrices not saved; no per-query Text-only AP inferred",
                "No policy entropy/STOP probabilities in logs; event frequencies are not entropy",
                "No fresh independent Validation, multiple training seeds, or G10->H60 control branch",
                "No new H20-H60 official Test or cross-victim results"],
       analysis_seconds=time.monotonic()-begin,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
       guards=dict(CUDA_initialized=torch.cuda.is_initialized(),model_forward=0,training=0,neural_encoding=0,
                   formal_Full_ranking=0,old_audit_entry_calls=0,only_saved_128x128_score_ranking=True),
       old_failure_preserved=True,historical_protected_files=len(protected),additional_preserved_files=len(preserved))
    write_json("statistics.json",statistics)
    write_json("INPUT_MANIFEST.json",list(I.files.values()))
    source=dict(schema="hv1-g60-posthoc-analysis-evidence-1",status="POSTHOC_SOURCE_AND_STATISTICS_VERIFIED",
       baseline_commit="cb51cd3c0f11041fadb072c2f0853b222a649421",execution_commit="9765a536a11a9a0e9111f2f834230c53d41bf2d8",
       repair_code_commit="135df869b81d728261a625ca6e4a2c6b8b7f2ae5",training_status=done["status"],
       independent_CPU_audit_status=audit["status"],controller_status=(CONTROL/"phase").read_text().strip(),
       publication_failure="Python3.10 hashlib.file_digest unavailable; original failure unchanged",
       original_producers_verified=113,historical_protected_files=len(protected),additional_preserved_files=len(preserved),
       input_files=len(I.files),input_bytes=sum(x["size"] for x in I.files.values()),artifacts_verified=len(I.artifacts),
       input_manifest_sha256=sha(OUT/"INPUT_MANIFEST.json"),analysis_script_sha256=sha(__file__),
       Python=platform.python_version(),Torch=torch.__version__,NumPy=np.__version__,CPU_threads=torch.get_num_threads(),
       bootstrap=dict(seed=20261022,replicates=20000,cluster="PID",exploratory=True),
       guards=statistics["guards"],missing=statistics["missing"])
    write_json("SOURCE_EVIDENCE.json",source)
    print("H60_POSTHOC_DATA_EXTRACTION_PASS",json.dumps(source),flush=True)

if __name__=="__main__":
    def forbidden(*a,**k):raise RuntimeError("CPU posthoc forbids neural forward/GPU/gradient")
    with patch.object(torch.cuda,"_lazy_init",forbidden),patch.object(torch.nn.Module,"_call_impl",forbidden), \
         patch.object(torch.Tensor,"backward",forbidden),patch.object(torch.autograd,"grad",forbidden):
        main()
