"""新分析交付的CPU只读来源/账本核验；不调用旧实验审计入口。"""
import csv,json,ast
from pathlib import Path
from tbps.ablations import gh_attribute_audit as a
D=a.OUT
def rows(n):return list(csv.DictReader((D/n).open()))
def main():
    I=a.Inputs();src=I.read(D/"SOURCE_EVIDENCE.json");s=I.read(D/"statistics.json");u=I.read(D/"supplementary_statistics.json")
    maininputs=I.read(D/"INPUT_MANIFEST.json")
    for x in maininputs:
        a.need(a.st(x["path"])==x["metadata"],"input stat changed")
        I.add(x["path"],x["sha256"])
    for x in u["input_files"]:
        # G派生统计的Validation更新字段后经显式澄清；输入旧实验从未修改。
        p=x["path"]
        if p.startswith(str(D)):continue
        a.need(a.st(p)==x["metadata"],"supplement source stat changed");I.add(p,x["sha256"])
    config=I.read(a.EXEC/"configs/framework/hv1_g60_adaptation.server.json")
    for v in (*config["sources"].values(),config["parent"],config["G60"],config["G10"]):I.add(v["path"],v["sha256"])
    for p,h in config["producers"].items():I.add(p,h)
    oldsource=a.LEGACY/"prototypes/attribute_image_generator_20261014/outputs/conditions"
    manifest=I.read(oldsource/"manifest.json");I.read(manifest["protocol_path"],manifest["protocol_sha256"])
    locks=manifest["frozen_sha256"]
    for p,h in locks.items():I.add(p,h)
    cps=[]
    prior=I.read(config["sources"]["history_evidence"]["path"],config["sources"]["history_evidence"]["sha256"])
    for ep in a.EPOCHS:
        if ep==9:
            c=next(x for x in prior["candidates"] if x["epoch"]==ep);p=Path(c["checkpoint_path"]);h=c["checkpoint_sha256"]
        else:
            p=a.BASE/"training/checkpoints"/f"visit_{ep*2048:08d}"/"checkpoint.pt"
            md=I.read(p.with_name("checkpoint.json"));h=md["checkpoint_sha256"]
        I.add(p,h);cps.append(dict(H_epoch=ep,path=str(p),sha256=h))
    selection=a.HIST/"selections/30f573098d28b6af6afe146330823986f724378fc7bd4df9d257fdd2c6f20dd1.json"
    I.add(selection,"584e4ba2afc7b17fea3bdfb8501716498b84731fa216558daa2e8e4bb353f757")
    for x in rows("code_evidence.csv"):I.add(x["path"],x["sha256"])
    h=rows("h_attribute_statistics.csv");g=rows("g_attribute_statistics.csv");al=rows("attribute_alignment.csv")
    a.need(len(h)==6528 and len(g)==2176,"whole cohorts")
    for x in h:
        a.need(int(x["H_K"])-int(x["edit_count"])==int(x["position_NOOP"])+int(x["STOP_unused_slots"]),"K slot accounting")
    for x in g:
        a.need(int(x["optimizer_updates"])==(int(x["K"])+1 if x["role"]=="train" else 0),"actual G training-only updates")
    a.need(sum(int(x["optimizer_updates"]) for x in g if x["role"]=="train")==8068,"G update ledger")
    a.need(sum(int(x["G_selected"]) for x in rows("g_attribute_scores.csv") if x["role"]=="train")==6020,"G attribute ledger")
    a.need(all(x["exact_telescope"]=="1" for x in rows("k_decomposition.csv")),"budget algebra")
    a.need(u["raw_coverage"]["train"]["ambiguous"]==1 and u["raw_coverage"]["validation"]["ambiguous"]==0,"ambiguity retained")
    re=rows("canonical_reextraction.csv")
    a.need(len(re)==1836,"all original validation attribute anchors")
    for x in rows("attribute_span_mapping.csv"):a.need(x["span_coordinates_preserved"]=="1","character position mapping")
    a.need(len(list((D/"figures").glob("*.png")))==len(list((D/"figures").glob("*.svg")))==8,"figures complete")
    for log in ("analysis","supplement","cpu_tests_final","plots","report"):
        p=Path("/home/lzf/ldx/tmp/gh_attribute_alignment_audit_20261022")/(log+"_exit.json")
        a.need(I.read(p)["exit_code"]==0,"CPU analysis/test failed")
        I.add(p.with_name(log+".log"))
    protected=I.read(config["sources"]["protection_snapshot"]["path"],config["sources"]["protection_snapshot"]["sha256"])
    for p,ss in protected.items():a.need(a.st(p)==ss,"historical protection after finalization")
    preserve={}
    for root in (a.BASE,Path("/home/lzf/ldx/tmp/hv1_g60_adaptation_epoch60_v1"),Path("/home/lzf/ldx/projects/idea-test-tbps-hv1-g60-audit-fix-20261022/docs/framework/results/hv1_g60_adaptation_epoch60")):
        for p in root.rglob("*"):
            if p.is_file():preserve[str(p)]=a.st(p)
    a.need(len(preserve)==s["additional_preserved_files"],"unexpected old experiment file set")
    for name in ("gh_attribute_audit.py","gh_attribute_supplement.py","plot_gh_attribute_audit.py","report_gh_attribute_audit.py","validate_gh_attribute_audit.py"):
        I.add(a.ROOT/"src/tbps/ablations"/name)
    I.add(a.ROOT/"tests/framework/test_gh_attribute_audit.py")
    # 按“首个有位置NOOP且仍编辑”的既有Validation动作选择演示，不看攻击指标。
    quals=I.read(a.HIST/"qualification/validation.json")["entries"]
    selected={x["row_id"] for x in h if x["scope"]=="Validation" and x["H_epoch"]=="20" and int(x["position_NOOP"])>0 and int(x["edit_count"])>0}
    idx,q=next((i,q) for i,q in enumerate(quals) if q["identity"]["row_id"] in selected)
    p=a.BASE/"validation/epoch_0020"/f"query-{idx:03d}"/"record.json"
    rec=I.read(p);act=rec["action"];rid=q["identity"]["row_id"]
    aa=[x for x in al if x["scope"]=="Validation" and x["H_epoch"]=="20" and x["row_id"]==rid]
    trace=dict(selection_rule="first fixed Validation row with position NOOP and >=1 existing edit; no score filtering",
        row_id=rid,query_index=idx,record_source=dict(path=str(p),sha256=a.sha(p)),caption=q["caption"],action=act,attributes=aa)
    a.writej("TRACE_EXAMPLE.json",trace)
    # 保留原候选的一对多歧义，明确给出旧ID/linked_object，不建立强制映射。
    gm=I.read(a.G/"outputs/conditions_full/manifest.json")
    e=next(x for x in gm["entries"] if x["row_id"]=="cuhk:8643:0");md=I.read(e["metadata_path"],e["metadata_sha256"])
    qq=next(x for x in I.read(a.HIST/"qualification/train.json")["entries"] if x["identity"]["row_id"]==e["row_id"])
    gg=a.groups([x["instance"] for x in md["source_score_record"]["attributes"]],qq["caption"])
    hh=a.groups(qq["attributes"],qq["caption"]);amb=[]
    for k,vs in gg.items():
        if len(vs)>1 or len(hh.get(k,[]))>1:
            amb.append(dict(key=dict(zip(a.FIELDS,k)),G_candidates=[dict(instance_id=v["instance_id"],linked_object=v.get("linked_object"),provenance=v["provenance"]) for i,v in vs],
                H_candidates=[v for i,v in hh.get(k,[])],status="AMBIGUOUS"))
    a.need(len(amb)==1,"raw ambiguity evidence")
    a.writej("AMBIGUOUS_EXAMPLES.json",dict(row_id=e["row_id"],source=dict(path=e["metadata_path"],sha256=e["metadata_sha256"]),groups=amb))
    text="\n## 附录：一次真实H20动作及原G属性对照\n\n"
    text+="固定规则：按原Validation顺序，首个既有Position NOOP且实际编辑至少1字符的样本；不读取指标选例。\n\n"
    text+="row_id="+rid+"，原始记录："+str(p)+"，SHA="+a.sha(p)+"。\n\n"
    text+="原文："+json.dumps(q["caption"],ensure_ascii=False)+"\n\n攻击文本："+json.dumps(act["adversarial"],ensure_ascii=False)+"\n\n"
    text+="| 原词/Span | axis/region | G S_raw | H importance | G选中 | H选中 | H编辑 |\n| --- | --- | --- | --- | --- | --- | --- |\n"
    for x in aa:text+="| "+"/".join([x["raw_text"],x["char_start"]+":"+x["char_end"]])+" | "+x["semantic_axis"]+"/"+x["region"]+" | "+(x["G_S_raw"] or "N/A")+" | "+(x["H_importance"] or "N/A")+" | "+x["G_selected"]+" | "+x["H_selected"]+" | "+x["H_edited"]+" |\n"
    text+="\nK="+str(act["budget_k"])+"，属性选择顺序="+json.dumps(act["selected_attribute_ids"],ensure_ascii=False)+"。\n\n"
    text+="完整decisions与掩码、Unicode编辑/码位、Tokenizer旧审计见TRACE_EXAMPLE.json；此处未调用H。\n"
    text+="原候选歧义cuhk:8643:0的两G来源/linked_object及唯一H锚点见AMBIGUOUS_EXAMPLES.json，未强配。\n"
    with (D/"REPORT.md").open("a") as f:f.write(text)

    I.finish()
    src.update(status="GH_ATTRIBUTE_ALIGNMENT_AND_SOURCE_AUDIT_PASS",final_input_source_count=len(I.files),
        original_G1_frozen_sources_verified=len(locks),checkpoint_sources=cps,H7_selection_unchanged=True,
        final_source_evidence=list(I.files.values()),CPU_tests=24,rule_only_conversions=512,
        no_GPU=True,no_model_forward=True,no_new_Tokenizer_encoding=True,no_new_score_or_ranking=True,
        no_original_audit_rerun=True,PNG=8,SVG=8)
    (D/"SOURCE_EVIDENCE.json").write_text(json.dumps(src,ensure_ascii=False,indent=2)+"\n")
    a.writej("DELIVERABLE_CHECK.json",dict(status="PASS",G_samples=2176,H_Train_actions=6144,H_Validation_actions=384,
        ranking_comparable_Train=1736,ranking_comparable_Validation=105,raw_ambiguities_Train=1,raw_ambiguities_Validation=0,
        original_validation_anchors=1836,G_Train_attribute_updates=6020,G_Train_all_updates=8068,G_Validation_updates=0,
        protected_files=len(protected),additional_preserved_files=len(preserve),CPU_tests=24,no_GPU=True,no_model_forward=True))
    print("GH_ATTRIBUTE_DELIVERABLE_CPU_CHECK_PASS",len(I.files),len(protected),len(preserve),flush=True)
if __name__=="__main__":main()
