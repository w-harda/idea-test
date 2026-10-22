"""已有属性/动作的补充CPU对照；只做算术预算分解和冻结规则重提取，不生成攻击。"""
from pathlib import Path
from collections import Counter,defaultdict
from unittest.mock import patch
import ast,csv,hashlib,importlib.util,json,math,os,sys,time
import numpy as np
from tbps.ablations import gh_attribute_audit as a
D=a.OUT
def readcsv(name):return list(csv.DictReader((D/name).open()))
def writec(name,rows):a.writec(name,rows)
def number(x):return float(x) if x not in ("",None) else None
def safe_key(row):
    x={k:row[k] for k in a.FIELDS}
    try:
        x["char_start"]=int(x["char_start"]);x["char_end"]=int(x["char_end"])
    except (TypeError,ValueError):return None
    if not x["semantic_axis"] or not x["region"] or not x["canonical_value"]:return None
    return x
def rankrho(x,y):
    x=a.ranks(x);y=a.ranks(y)
    return float(np.corrcoef(x,y)[0,1]) if len(x)>=2 and np.std(x)>0 and np.std(y)>0 else None
def main():
    I=a.Inputs();src=I.read(D/"SOURCE_EVIDENCE.json")
    for f in ("attribute_alignment.csv","h_attribute_statistics.csv","g_attribute_scores.csv","statistics.json"):I.add(D/f)
    stats=I.read(D/"statistics.json");rows=readcsv("h_attribute_statistics.csv");attrs=readcsv("attribute_alignment.csv")
    by=defaultdict(list)
    for x in attrs:by[(x["scope"],int(x["H_epoch"]),x["row_id"])].append(x)
    qs={role:{x["identity"]["row_id"]:x for x in I.read(a.HIST/"qualification"/(role+".json"))["entries"]} for role in ("train","validation")}
    rawgroups=defaultdict(list)
    for x in readcsv("g_attribute_scores.csv"):rawgroups[x["row_id"]].append(x)
    rawrows=[]
    for role in ("train","validation"):
        for rid,q in qs[role].items():
            gg=[];invalid=[]
            for x in rawgroups[rid]:
                v=safe_key(x)
                if v is None or q["caption"][v["char_start"]:v["char_end"]]!=v["raw_text"]:invalid.append(x)
                else:gg.append(v)
            align=a.strict_alignment(gg,q["attributes"],q["caption"])
            counts=Counter(x[1] for x in align)
            rawrows.append(dict(role=role,row_id=rid,G_original_candidates=len(rawgroups[rid]),G_span_semantic_valid=len(gg),
                G_unresolved_or_invalid=len(invalid),H_resolved=len(q["attributes"]),**counts,
                complete_raw_candidate_match=int(counts["EXACT"]==len(gg)==len(q["attributes"]) and not counts["AMBIGUOUS"])))
    writec("raw_candidate_coverage.csv",rawrows)
    decomposition=[];ranking=[]
    for r in rows:
        key=(r["scope"],int(r["H_epoch"]),r["row_id"]);xs=by[key]
        if not int(r["has_importance"]):continue
        common=[x for x in xs if x["alignment_status"]=="EXACT" and x["H_legal"]=="1"]
        gc=a.natural([number(x["G_S_raw"]) for x in common])["k"]
        hc=a.natural([number(x["H_importance"]) for x in common])["k"]
        kg=int(r["G_K"]);kh=int(r["H_K"])
        decomposition.append(dict(scope=r["scope"],H_epoch=int(r["H_epoch"]),row_id=r["row_id"],
            G_K=kg,H_K=kh,common_legal_candidates=len(common),G_K_common=gc,H_K_common=hc,
            candidate_component_G_to_common=kg-gc,importance_component_common=gc-hc,
            candidate_component_common_to_H=hc-kh,total_G_minus_H=kg-kh,
            exact_telescope=int((kg-gc)+(gc-hc)+(hc-kh)==kg-kh)))
        if int(r["candidate_sets_equal"]) and len(common)>=2:
            sx=[number(x["G_S_raw"]) for x in common];sy=[number(x["H_importance"]) for x in common]
            rho=rankrho(sx,sy)
            ranking.append(dict(scope=r["scope"],H_epoch=int(r["H_epoch"]),row_id=r["row_id"],candidates=len(common),
                spearman=rho,G_tie=int(len(set(sx))<len(sx)),H_tie=int(len(set(sy))<len(sy)),
                same_candidate_set=1,positive_same_legal_subset=1,
                top1_agreement=int(bool(set(i for i,x in enumerate(sx) if x==max(sx))&set(i for i,x in enumerate(sy) if x==max(sy))))))
    writec("k_decomposition.csv",decomposition);writec("strict_rank_comparison.csv",ranking)
    summ={}
    for scope in ("Train","Validation"):
        for ep in a.EPOCHS:
            kk=f"{scope}_H{ep}";dd=[x for x in decomposition if x["scope"]==scope and x["H_epoch"]==ep]
            rr=[x for x in ranking if x["scope"]==scope and x["H_epoch"]==ep]
            summ[kk]=dict(decomposition_samples=len(dd),**{k:a.dist(x[k] for x in dd) for k in ("candidate_component_G_to_common","importance_component_common","candidate_component_common_to_H","total_G_minus_H")},
                ranking_samples=len(rr),spearman=a.dist(x["spearman"] for x in rr if x["spearman"] is not None),
                G_ties=sum(x["G_tie"] for x in rr),H_ties=sum(x["H_tie"] for x in rr),
                two_candidate_samples=sum(x["candidates"]==2 for x in rr),
                rank_comparable_top1=dict(count=sum(x["top1_agreement"] for x in rr),denominator=len(rr)),
                missing_importance=sum(x["scope"]==scope and int(x["H_epoch"])==ep and not int(x["has_importance"]) for x in rows))
    # 唯一规则重提取范围：128内部Validation的既有原文与H9/H20/H60既有攻击文本。
    # 不调用Tokenizer，不调用H.forward，不生成新攻击；源算法文件均只读锁定。
    cf=I.read(a.H/"config_phase2.json")
    for rel,h in cf["frontend_sources"].items():I.add(a.OLD/rel,h)
    for name in ("text_frontend.py","phase2_sources.py","contracts.py"):I.add(a.H/name)
    original_path=list(sys.path);before_modules=set(sys.modules);sys.dont_write_bytecode=True
    sys.path.insert(0,str(a.H))
    import torch
    def forbidden(*x,**kw):raise RuntimeError("CPU rule analysis forbids GPU/model/gradient")
    reextract=[];advcounts=[];mapping_lookup={}
    for x in readcsv("attribute_span_mapping.csv"):
        if x["scope"]=="Validation":mapping_lookup[(int(x["H_epoch"]),x["row_id"],x["H_id"])]=x
    conf=I.read(a.EXEC/"configs/framework/hv1_g60_adaptation.server.json")
    pri=I.read(conf["sources"]["history_evidence"]["path"],conf["sources"]["history_evidence"]["sha256"])
    allacts={}
    c=next(x for x in pri["candidates"] if x["epoch"]==9)
    v=Path(c["validation_summary_path"]).parent
    for unit in (v/"queries").iterdir():
        rc=I.read(unit/"receipt.json");md=I.read(unit/"metadata.json",rc["metadata_sha256"])
        allacts[(9,md["identity"]["row_id"])]=md["action"]
    for ep in (20,60):
        for i,rid in enumerate(qs["validation"]):
            p=a.BASE/"validation"/f"epoch_{ep:04d}"/f"query-{i:03d}"
            md=I.read(p/"manifest.json");rec=I.read(p/"record.json",md["files_sha256"]["record.json"])
            allacts[(ep,rid)]=rec["action"]
    try:
        with patch.object(torch.cuda,"_lazy_init",forbidden),patch.object(torch.nn.Module,"_call_impl",forbidden),patch.object(torch.Tensor,"backward",forbidden),patch.object(torch.autograd,"grad",forbidden):
            from text_frontend import CaptionFrontend
            frontend=CaptionFrontend()
            for rid,q in qs["validation"].items():
                example,audit=frontend.convert(q["caption"])
                from dataclasses import asdict
                actual=[asdict(x) for x in example.attributes]
                a.need(actual==q["attributes"],"original canonical rule/qualification mismatch")
                for ep in a.EPOCHS:
                    act=allacts[(ep,rid)];adv=act.get("adversarial",q["caption"])
                    converted,_=frontend.convert(adv)
                    aa=[asdict(x) for x in converted.attributes]
                    def semantic(x):return tuple(x[k] for k in a.FIELDS if k!="raw_text")
                    ag=defaultdict(list)
                    for x in aa:ag[semantic(x)].append(x)
                    same=lost=amb=0
                    for x in q["attributes"]:
                        hid=a.attr_id(x);matches=ag.get(semantic(x),[])
                        status="AMBIGUOUS" if len(matches)>1 else "SEMANTIC_ANCHOR_REFOUND" if matches else "NOT_REFOUND"
                        same+=status=="SEMANTIC_ANCHOR_REFOUND";lost+=status=="NOT_REFOUND";amb+=status=="AMBIGUOUS"
                        mp=mapping_lookup[(ep,rid,hid)]
                        reextract.append(dict(H_epoch=ep,row_id=rid,H_id=hid,**x,status=status,
                            original_literal=mp["raw_text"],edited_literal=mp["edited_raw"],own_attribute_edited=int(mp["H_edited"]),
                            span_affected=int(mp["edits_affecting_span"])>0,G_selected=int(mp["G_selected"]),
                            original_span_preserved=1,UTF8_start_shift=int(mp["UTF8_start_edited"])-int(mp["UTF8_start_original"])))
                    advcounts.append(dict(H_epoch=ep,row_id=rid,original_attributes=len(actual),reextracted_attributes=len(aa),
                        semantic_refound=same,semantic_not_refound=lost,ambiguous=amb,edits=len(act.get("edits",[])),
                        original_utf8_bytes=len(q["caption"].encode()),adversarial_utf8_bytes=len(adv.encode())))
            a.need(not torch.cuda.is_initialized(),"CUDA initialized")
    finally:
        sys.path[:]=original_path
        for n in set(sys.modules)-before_modules:
            f=getattr(sys.modules.get(n),"__file__",None)
            if f and (str(a.H) in f or str(a.OLD) in f):sys.modules.pop(n,None)
    writec("canonical_reextraction.csv",reextract);writec("canonical_reextraction_by_query.csv",advcounts)
    rex={}
    for ep in a.EPOCHS:
        rr=[x for x in reextract if x["H_epoch"]==ep]
        cc=[x for x in advcounts if x["H_epoch"]==ep]
        rex[str(ep)]=dict(samples=len(cc),original_attributes=len(rr),reextracted_attributes=sum(x["reextracted_attributes"] for x in cc),
            semantic_refound=sum(x["status"]=="SEMANTIC_ANCHOR_REFOUND" for x in rr),
            not_refound=sum(x["status"]=="NOT_REFOUND" for x in rr),
            ambiguous=sum(x["status"]=="AMBIGUOUS" for x in rr),
            own_edit_not_refound=sum(x["own_attribute_edited"] and x["status"]=="NOT_REFOUND" for x in rr),
            unedited_not_refound=sum(not x["own_attribute_edited"] and x["status"]=="NOT_REFOUND" for x in rr),
            UTF8_start_shifted_attributes=sum(x["UTF8_start_shift"]!=0 for x in rr),
            edits=sum(x["edits"] for x in cc))
    rawsumm={}
    for role in ("train","validation"):
        rr=[x for x in rawrows if x["role"]==role]
        rawsumm[role]=dict(samples=len(rr),complete_raw_candidate_match=sum(x["complete_raw_candidate_match"] for x in rr),
            G_original_candidates=sum(x["G_original_candidates"] for x in rr),G_span_semantic_valid=sum(x["G_span_semantic_valid"] for x in rr),
            G_unresolved_or_invalid=sum(x["G_unresolved_or_invalid"] for x in rr),
            H_resolved=sum(x["H_resolved"] for x in rr),G_only=sum(x.get("G_ONLY",0) for x in rr),
            H_only=sum(x.get("H_ONLY",0) for x in rr),ambiguous=sum(x.get("AMBIGUOUS",0) for x in rr))
    I.finish()
    out=dict(raw_coverage=rawsumm,budget_decomposition_and_rank=summ,canonical_reextraction=rex,
        rule_only_conversions=128*4,neural_forward=0,Tokenizer_calls=0,GPU_calls=0,
        input_files=list(I.files.values()),note="预算分解仅代数重算，不是改变K后的攻击实验。Spearman含n=2，独立列出其数量；无常量排名伪造相关。")
    a.writej("supplementary_statistics.json",out)
    print("GH_ATTRIBUTE_RULE_AND_BUDGET_SUPPLEMENT_PASS",json.dumps({k:v for k,v in out.items() if k!="input_files"}),flush=True)
if __name__=="__main__":main()
