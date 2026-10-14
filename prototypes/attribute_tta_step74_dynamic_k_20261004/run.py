"""Step7.4 PREFIX+CAPPED-4 vs PREFIX+NATURAL；默认3句短smoke，100句必须用户显式运行。"""
import argparse,copy,json,time
from pathlib import Path
import deps
from selection_audit import *
def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=["smoke","diagnostic"],default="smoke")
    p.add_argument("--allow-diagnostic",action="store_true")
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path)
    return p

def verify_fixed_payload(payload,record):
    if payload["gamma"]!=3 or payload["beta"]!=1:raise ValueError("selection settings")
    eligible={}
    order=[]
    for row in record["attributes"]:
        instance=row["instance"];score=row.get("S_raw")
        if row["status"]=="scored" and score>0:
            e=instance.get("semantic_representation")
            if instance["unresolved"] or not span_gate(record["caption"],instance) or not e or e["basis_rank"]<=0:
                raise ValueError("invalid frozen candidate")
            eligible[instance["instance_id"]]={**instance,"S_raw":score}
            order.append(instance["instance_id"])
    ranked=payload["ranked_attributes"];ids=[a["instance_id"] for a in ranked]
    if len(ids)!=len(set(ids)) or set(ids)!=set(eligible) or payload["candidate_count"]!=len(ids):
        raise ValueError("frozen candidate IDs changed")
    if any(a!=eligible[a["instance_id"]] for a in ranked):raise ValueError("instance/score mutated")
    if any(x["S_raw"]<y["S_raw"] for x,y in zip(ranked,ranked[1:])):raise ValueError("frozen ranking changed")
    original_positions={aid:i for i,aid in enumerate(order)}
    if any(x["S_raw"]==y["S_raw"] and original_positions[x["instance_id"]]>original_positions[y["instance_id"]]
           for x,y in zip(ranked,ranked[1:])):raise ValueError("tie order changed")
    k=payload["natural_k"]
    if (type(k) is not int or not 1<=k<=len(ids) or payload["k_star"]!=k
        or payload["capped_k"]!=min(k,4) or payload["selected_attributes"]!=ranked[:k]):
        raise ValueError("frozen prefix changed")
    # N_eff/probabilities只作为已冻结记录保留；此处不调用selection/Top-K。

def checked_manifest(path,mode):
    manifest=load(path)
    verify({manifest["score_cache_path"]:manifest["score_cache_sha256"],
            manifest["score_protocol_path"]:manifest["score_protocol_sha256"]})
    protocol=load(manifest["score_protocol_path"])
    if protocol["schema"]!="step74-clean-score-cache-1" or protocol["gamma"]!=3 or protocol["beta"]!=1 or protocol["tau"]!=.02 or protocol["layer"]!=8:
        raise ValueError("score cache settings changed")
    verify(protocol["frozen_sha256"])
    # score缓存只依赖准备/选择代码；攻击runner/tests不参与S_raw计算。
    verify({p:h for p,h in protocol["source_sha256"].items()
            if Path(p).name in ("prepare_scores.py","deps.py","selection_audit.py")})
    if manifest["split_sha256"]!=digest(deps.V4/"outputs/split.json") or protocol["split_sha256"]!=manifest["split_sha256"]:
        raise ValueError("score/split provenance mismatch")
    if mode=="diagnostic" and (not manifest["full_preparation"] or manifest["case_count"]!=100):
        raise ValueError("full diagnostic requires completed score preparation and exactly100 frozen selected rows")
    if mode=="smoke" and manifest["case_count"]>3:raise ValueError("smoke max3 cases")
    if manifest["material_rule"]["threshold"]!=MATERIAL_F_THRESHOLD:raise ValueError("material threshold changed")
    expected={p["caption_id"] for p in manifest["rows"]}
    if len(expected)!=manifest["case_count"]:raise ValueError("duplicate/missing manifest rows")
    records={r["row_id"]:r for r in deps.rows(manifest["score_cache_path"]) if r["row_id"] in expected}
    if set(records)!=expected:raise ValueError("missing frozen score")
    split=load(deps.V4/"outputs/split.json")
    val={split["records"][i]["row_id"]:split["records"][i] for i in split["validation_indices"]}
    for payload in manifest["rows"]:
        r=records[payload["caption_id"]]
        if r["row_id"] not in val or r["caption"]!=val[r["row_id"]]["caption"] or r["pid"]!=val[r["row_id"]]["pid"]:
            raise ValueError("non-validation/source mismatch")
        verify_fixed_payload(payload,r) # 只核验已有实例/分数/排名/前缀，不重算entropy或K。
    return manifest,protocol,records

def main(argv=None):
    args=parser().parse_args(argv)
    if args.mode=="diagnostic" and not args.allow_diagnostic:
        raise SystemExit("100-case attack is user-run only; --allow-diagnostic required")
    hashes=deps.frozen_hashes();own=deps.own_hashes()
    manifest,score_protocol,records=checked_manifest(args.manifest,args.mode)
    out=new_output(args.output or deps.ROOT/"outputs"/args.mode)
    import torch
    from PIL import Image
    import prefix_tta
    from controlled_tta import tensor_hash
    from attribute_tta import TextTransport,IMAGE_EPSILON
    from paired_prefix import prepare_uncapped,execute_pair
    from attributes.irra_surrogate import IRRASurrogate
    old=deps.metrics_runner()
    import sys
    sys.path.insert(0,str(deps.V4));import learned_reader as v4
    v4.setup()
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    victim=v4.load_victim();source=IRRASurrogate(victim)
    versions={n:p._version for n,p in victim.named_parameters()}
    features,path_index,gallery_audit=old.load_gallery(hashes)
    bases=load(deps.V55/"outputs/full/semantic_bases.json")
    steps,transforms,scales=(2,1,None) if args.mode=="smoke" else (10,6,(.5,.75,1.25,1.5))
    locked={**hashes,**own,str(args.manifest.resolve()):digest(args.manifest),
            manifest["score_cache_path"]:manifest["score_cache_sha256"],
            manifest["score_protocol_path"]:manifest["score_protocol_sha256"]}
    protocol={"mode":args.mode,"modes":list(MODES),"case_count":manifest["case_count"],
        "scheduler":"unchanged Step7.3.2 PREFIX_NEW; unweighted prefix mean, no aggregate normalization",
        "gamma":3,"beta":1,"kmax":{"CAPPED_4":4,"NATURAL_DYNAMIC_K":None},
        "steps_per_round":steps,"transforms_per_scale":transforms,"scales":scales,"seed":42,
        "epsilon":IMAGE_EPSILON,"step_size":2/255,"momentum":.9,"tau":.02,"gallery_count":34054,
        "gallery":gallery_audit,"score_preparation":manifest["score_protocol_path"],
        "score_backwards_during_attack":0,"selection_backwards":0,"score":"read-only frozen S_raw",
        "JVP":"unchanged Step72 TextTransport.read once/selected instance before image optimization",
        "random_control":"one common actual PREFIX trajectory; cap checkpoint at min(k,4), same full image/momentum/RNG/transform/noise history; only natural suffix proceeds",
        "adaptive_weights":"original official loss-dependent probability updates remain",
        "material_F_threshold":MATERIAL_F_THRESHOLD,"full_training":False,
        "formal_shared_adversarial_gallery":False,"frozen_sha256":locked}
    save(out/"protocol.json",protocol);save(out/"selection_manifest.json",manifest)
    cases=[];started=time.time()
    for payload in manifest["rows"]:
        cid=payload["caption_id"];record=records[cid];t0=time.time()
        plan=prepare_uncapped(payload,record,bases,"cuda")
        original_metadata=copy.deepcopy([a.metadata for a in plan])
        tv=[(a.w._version,a.Q._version) for a in plan]
        image_path=(Path("/home/lzf/TBPS/Datasets/CUHK-PEDES/imgs")/record["paired_image"]).resolve()
        if not image_path.is_relative_to(Path("/home/lzf/TBPS/Datasets/CUHK-PEDES/imgs")):raise ValueError("image path")
        if path_index[record["paired_image"]]!=record["paired_gallery_index"]:raise ValueError("paired gallery mapping")
        with Image.open(image_path) as im:image=victim.pixels(im).unsqueeze(0).cuda()
        transport=TextTransport(victim,record["caption"]);vectors={};mapping=[]
        for a in plan:
            v,report=transport.read(a)
            if report["central_fd_relative_error"]>.02:raise ValueError("unchanged JVP FD gate failed")
            vectors[a.instance_id]=v;mapping.append(report)
        vector_sha={key:tensor_hash(v) for key,v in vectors.items()}
        evaluator=old.RetrievalMetrics(source,transport.clean,features,record["paired_gallery_index"],tau=.02)
        data,checks=execute_pair(source,image,plan,vectors,evaluator,
            steps=steps,transforms=transforms,scales=scales)
        if original_metadata!=[a.metadata for a in plan] or any((a.w._version,a.Q._version)!=n for a,n in zip(plan,tv)):
            raise ValueError("AttributeInstance/basis/localization mutated")
        if vector_sha!={key:tensor_hash(v) for key,v in vectors.items()}:raise ValueError("JVP changed")
        if any(p.requires_grad or p.grad is not None or p._version!=versions[n] for n,p in victim.named_parameters()):
            raise ValueError("IRRA not frozen")
        checks.update(zero_exact=True,source_span_roundtrip=True,AttributeInstance_S_JVP_unchanged=True,IRRA_frozen=True)
        case={"caption_id":cid,"pid":record["pid"],"caption":record["caption"],
            "paired_image":record["paired_image"],"candidate_count":payload["candidate_count"],
            "natural_k":len(plan),"capped_k":min(len(plan),4),"truncated":len(plan)>4,
            "selected_attributes":original_metadata,"clean_scores":[a["S_raw"] for a in payload["ranked_attributes"]],
            "JVP_mapping":mapping,"JVP_sha256":vector_sha,"JVP_computations":len(plan),
            "modes":data,"checks":checks,"seconds":time.time()-t0}
        cases.append(case);save(out/"per_caption_results.json",cases)
        print("CASE_DONE",cid,"natural_k",len(plan),"capped_k",min(len(plan),4),"seconds",case["seconds"],flush=True)
    verify(locked);deps.assert_formal_clean()
    summary,marginals=summarize(cases)
    save(out/"summary.json",summary);save(out/"round_marginal_effects.json",marginals)
    audit={"case_count":len(cases),"modes":list(MODES),"steps_per_round":steps,
        "logical_attribute_rounds":{"CAPPED_4":sum(c["capped_k"] for c in cases),"NATURAL_DYNAMIC_K":sum(c["natural_k"] for c in cases)},
        "physical_attribute_rounds":sum(c["natural_k"] for c in cases),
        "common_prefix_exact":True,"k_le4_final_exact":True,"noise_transform_rng_common_prefix_exact":True,
        "score_backwards_during_attack":0,"score_cache_SHA_unchanged":True,"source_SHA_unchanged":True,
        "IRRA_frozen":True,"AttributeInstance_S_TopK_JVP_unchanged":True,"caption_loss_terms":0,
        "maximum_linf":max(c["modes"][MODES[1]]["final_linf"] for c in cases),
        "no_training":True,"diagnostic_run":args.mode=="diagnostic","seconds":time.time()-started}
    save(out/"audit.json",audit)
    lines=["# Step7.4 CAPPED-4 vs NATURAL-DYNAMIC-K","",
        "固定PREFIX，无新scheduler；分层样本不能当作validation总体均值。smoke不能决定是否去掉cap。",
        "",json.dumps(summary["comparisons_truncated"],ensure_ascii=False,indent=2),
        "","逐caption完整轨迹见per_caption_results.json；round5/6/7+见round_marginal_effects.json。"]
    (out/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    save(out/"completed.json",audit);print("COMPLETE",json.dumps(audit),flush=True)
if __name__=="__main__":main()
