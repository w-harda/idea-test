"""CPU只读复核：保留严格NOOP失败，结构验证与反馈统计分层，不修改生产审计。"""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from contextlib import ExitStack
from unittest.mock import patch

FROZEN=Path("/home/lzf/ldx/projects/idea-test-tbps-gh-probe-shared-20261023")
OPS=Path("/home/lzf/ldx/projects/idea-test-tbps-gh-rng-smoke-20261023")
SMOKE=Path("/home/lzf/ldx/outputs/tbps-framework-gh-bidirectional-backend/smoke_v2")
CTL=Path("/home/lzf/ldx/tmp/gh_smoke_wait_v1_20261023")
OUTPUT=Path("/home/lzf/ldx/outputs/tbps-framework-gh-smoke-review/readonly_v3")
CONFIG=FROZEN/"configs/framework/gh_bidirectional_backend.probe_v2.shared.server.json"

def noop_checks(a,b):
    return {"same_G_before":a["G_before"]==b["G_before"],
        "same_G_after":a["G_after"]==b["G_after"],
        "same_gradient_observations":a["gradient_observations"]==b["gradient_observations"],
        "same_gradient_SHA":a["gradient_observations"][0]["gradient_sha256"]==b["gradient_observations"][0]["gradient_sha256"]}
def h_summary(rows):
    h=[x for x in rows if not x.get("fixed_NOOP",False)]
    deltas=[v for x in h for v in x["reward_delta_same_text"]]
    ad=[x["reward_delta_same_text"][0]-x["reward_delta_same_text"][1] for x in h]
    return {"visits":len(rows),"fixed_NOOP":len(rows)-len(h),"Adam_updates":sum(x.get("optimizer_steps_this_step",0) for x in rows),
        "nonzero_advantage":sum(x["advantage"]!=0 for x in h),
        "nonzero_gradient":sum(x["gradient_norm"]!=0 for x in h),
        "H_state_changed":sum(x["H_state_sha256_before"]!=x["H_state_sha256_after"] for x in h),
        "zero_gradient_but_state_changed":sum(x["gradient_norm"]==0 and x["H_state_sha256_before"]!=x["H_state_sha256_after"] for x in h),
        "same_text_Reward_changed":sum(any(v!=0 for v in x["reward_delta_same_text"]) for x in h),
        "same_text_Advantage_changed":sum(v!=0 for v in ad),
        "reward_delta_min":min(deltas),"reward_delta_max":max(deltas),
        "advantage_delta_min":min(ad),"advantage_delta_max":max(ad),
        "mean_advantage":sum(x["advantage"] for x in h)/len(h)}
def feedback(a,b,all_h):
    raw={"first_nonNOOP_gradient_diff":a[1]["gradient_observations"][0]["gradient_sha256"]!=b[1]["gradient_observations"][0]["gradient_sha256"],
        "first_nonNOOP_G_update_diff":a[1]["G_after"]!=b[1]["G_after"],
        "same_text_Reward_changed":any(any(v!=0 for v in h.get("reward_delta_same_text",[])) for h in all_h),
        "same_text_Advantage_changed":any(x["reward_delta_same_text"][0]!=x["reward_delta_same_text"][1] for x in all_h if "reward_delta_same_text" in x)}
    raw["bidirectional_update_evidence_verified"]=all(raw.values())
    control=noop_checks(a[0],b[0])
    exact=control["same_G_before"] and control["same_G_after"] and control["same_gradient_observations"]
    return {"original_audit_feedback_effects":"NOT_EMITTED: original audit raised before constructing it",
        "derived_original_expression":raw,"NOOP_exact_control":control,
        "first_nonNOOP_same_G_before":a[1]["G_before"]==b[1]["G_before"],
        "bidirectional_update_evidence_verified":bool(raw["bidirectional_update_evidence_verified"] and exact and a[1]["G_before"]==b[1]["G_before"]),
        "strict_H_to_G_causal_attribution_in_this_Smoke":"NOT_VERIFIED" if not exact else "control exact"}

def tensor_difference(a,b):
    """只比较已保存state Tensor，不从SHA推导距离。"""
    import torch
    assert set(a)==set(b)
    max_abs=0.;squared=0.;changed=0;total=0
    for k in a:
        x,y=a[k],b[k]
        assert x.shape==y.shape and x.dtype==y.dtype
        d=x.to(torch.float64)-y.to(torch.float64)
        max_abs=max(max_abs,float(d.abs().max()) if d.numel() else 0.)
        squared+=float(d.square().sum());changed+=int((x!=y).sum());total+=x.numel()
    return {"max_abs":max_abs,"L2":math.sqrt(squared),"changed_elements":changed,"elements":total}

def legacy_h_hasher(path,expected_sha):
    """只执行已锁定原H state_sha函数；不导入旧模型/训练器，不替换生产函数。"""
    import ast
    from types import SimpleNamespace
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected_sha,"legacy H hash source SHA"
    tree=ast.parse(Path(path).read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="state_sha")
    scope={"hashlib":hashlib}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),"exec"),scope)
    def invoke(state):return scope["state_sha"](SimpleNamespace(state_dict=lambda:state))
    return invoke

def condition_key_from_json(value):
    """JSON数组无损还原原契约的tuple；顺序、分数与内容不变。"""
    from tbps.alternating.contracts import ConditionIdentity
    return ConditionIdentity(**{**value,"selected_g_ids":tuple(value["selected_g_ids"]),
        "scores":tuple(value["scores"])}).key

def state_metadata(value):
    from tbps.alternating.state import tree_sha
    def steps(opt):return sorted({int(v["step"].item()) if hasattr(v["step"],"item") else int(v["step"]) for v in opt["state"].values()})
    def rng(v):
        import torch
        assert set(v)=={"python","numpy","torch_cpu","sampler","cuda"} and len(v["cuda"])==1
        assert v["torch_cpu"].dtype==v["sampler"].dtype==torch.uint8
        assert all(t.dtype==torch.uint8 and t.device.type=="cpu" for t in v["cuda"])
        return {"state_sha256":tree_sha(v),"components":{k:tree_sha(x) for k,x in v.items()},
            "CUDA_devices":len(v["cuda"]),"CUDA_bytes":[x.numel() for x in v["cuda"]]}
    return {"G":tree_sha(value["G"]),"H":tree_sha(value["H"]),
        "G_Adam":tree_sha(value["G_Adam"]),"H_Adam":tree_sha(value["H_Adam"]),
        "G_Adam_steps":steps(value["G_Adam"]),"H_Adam_steps":steps(value["H_Adam"]),
        "G_optimizer_groups":value["G_Adam"]["param_groups"],"H_optimizer_groups":value["H_Adam"]["param_groups"],
        "G_RNG":rng(value["G_RNG"]),"H_RNG":rng(value["H_RNG"]),
        "cursor":value["cursor"],"parents":value["parents"]}

def inventory(root):
    result={}
    for p in root.rglob("*"):
        if p.is_file():
            s=p.stat();result[str(p)]=[s.st_size,s.st_mtime_ns,s.st_ino]
    return result

def execute():
    sys.path[:0]=[str(FROZEN/"src"),str(OPS/"operations")]
    from gh_smoke_checks import check_sources,rng_receipt
    from gh_cuda_rng_validation import verify_history
    from tbps.artifacts import ArtifactStore,FileRef,sha_file,json_bytes,digest
    from tbps.alternating.native_sources import source_key
    c=check_sources();rng_receipt(c)
    assert not OUTPUT.exists(),"independent review identity already consumed"
    before=inventory(SMOKE);control_before=inventory(CTL)
    assert not [p for p in SMOKE.rglob("*") if p.name.startswith(".pending-") or p.name=="FAIL"]
    history=verify_history()
    store=ArtifactStore(SMOKE,new_base=c["output_base"])
    assert not (SMOKE/"FAIL").exists() and not store.pending()
    native=store.verify("completed",expected_provenance={"source":source_key(c),"protocol":digest(c["prototype"])})
    report=json.loads((SMOKE/"completed/report.json").read_text())
    assert report["status"]=="GH_NATIVE_TWO_ROUND_SMOKE_GENERATED" and report["rounds"]==4
    assert report["counts"]=={"G":8300,"H":84,"IRRA_image":8300,"IRRA_text":226,"JVP":38,"reader":14}
    assert report["frozen_sources"] and (CTL/"smoke.exit").read_text().strip()=="0"
    assert (CTL/"audit-smoke.exit").read_text().strip()=="1"
    assert "first NOOP control differs" in (CTL/"cpu_audit.log").read_text()
    producer=FileRef(str(Path(__file__).resolve()),sha_file(Path(__file__)))
    identity=digest({"review_producer":producer.sha256,"input_source_key":source_key(c),
        "smoke_completed_manifest":sha_file(SMOKE/"completed/manifest.json")})
    target=ArtifactStore(OUTPUT,new_base=OUTPUT.parent)
    target.publish("claim",{"input.json":json_bytes({"input_source":source_key(c),"smoke":str(SMOKE),"review_identity":identity})},
        {"review_identity":identity},sources=(producer,FileRef(str(SMOKE/"completed/manifest.json"),sha_file(SMOKE/"completed/manifest.json"))))
    start=time.monotonic()
    try:
        import torch
        from tbps.alternating.parents import no_optimizer_steps
        from tbps.alternating.state import tree_sha
        from tbps.alternating.native_backend import assert_pixels
        from tbps.alternating.contracts import ConditionIdentity
        from tbps.ablations.gv2_epoch_state import finite_state
        torch.set_num_threads(2)
        assert not torch.cuda.is_initialized()
        def forbidden(*a,**k):raise RuntimeError("CPU review forbids CUDA, neural forward and derivatives")
        records={};values={};units=[]
        with no_optimizer_steps(),ExitStack() as guards:
            for owner,name in [(torch.cuda,"init"),(torch.cuda,"_lazy_init"),(torch.nn.Module,"_call_impl"),
                (torch.autograd,"grad"),(torch.autograd,"backward"),(torch.Tensor,"backward")]:
                guards.enter_context(patch.object(owner,name,forbidden))
            def unit(s,n,expected=None):
                m=s.verify(n,expected_provenance=expected)
                units.append({"root":str(s.root),"unit":n,"manifest":sha_file(s.root/n/"manifest.json")})
                return m
            unit(store,"claim",{"source":source_key(c),"action":"smoke"})
            for idx,entry in enumerate(report["conditions"]):
                unit(store,f"condition_{idx:04d}");unit(store,f"rollout_{idx:04d}")
                value=torch.load(SMOKE/f"condition_{idx:04d}/condition.pt",map_location="cpu",weights_only=True)
                assert tree_sha(value)==entry["payload_content_sha256"]
            rndstore=ArtifactStore(SMOKE/"rounds",new_base=c["output_base"])
            previous={}
            parent=json.loads((Path("/home/lzf/ldx/outputs/tbps-framework-gh-bidirectional-backend/cuda_rng_v1")/"completed/report.json").read_text())["CPU_preflight"]
            from tbps.alternating.parents import normalized_rng
            from tbps.ablations.gv2_sweep import generator_state
            gp=torch.load(c["sources"]["G60"]["path"],map_location="cpu",weights_only=False)
            hp=torch.load(c["sources"]["H10"]["path"],map_location="cpu",weights_only=False)
            original_contract=json.loads(Path(c["sources"]["original_G_contract"]["path"]).read_text())
            gw=generator_state(gp,60,original_contract,gp["contract"]["parent_checkpoint"])
            gr,hr=normalized_rng(gp["rng"],hp["rng"])
            assert tree_sha(gw)==parent["G_state"] and tree_sha(hp["policy"])==parent["H_state"]
            assert tree_sha(gp["optimizer"])==parent["G_Adam"] and tree_sha(hp["optimizer"])==parent["H_Adam"]
            assert tree_sha(gr)==parent["G_RNG"] and tree_sha(hr)==parent["H_RNG"]
            parent_state={"G":gw,"H":hp["policy"],"G_Adam":gp["optimizer"],"H_Adam":hp["optimizer"],
                "G_RNG":gr,"H_RNG":hr,"cursor":{"round":0,"g_updates":484080,"h_visits":20480,"h_updates":20370},
                "parents":{"G":c["sources"]["G60"]["sha256"],"H":c["sources"]["H10"]["sha256"],"protocol":digest(c["prototype"])}}
            parent_complete_sha=tree_sha(parent_state)
            h_hash_path=Path(c["sources"]["H10"]["path"]).parents[5]/"feedback_backend.py"
            assert str(h_hash_path) in c["runtime_pins"]
            native_h_sha=legacy_h_hasher(h_hash_path,c["runtime_pins"][str(h_hash_path)])
            parent_native_h_sha=native_h_sha(hp["policy"])
            parent_momentum={"H_parameters_with_nonzero_exp_avg":sum(bool(torch.count_nonzero(v["exp_avg"])) for v in hp["optimizer"]["state"].values()),
                "H_parameters_with_nonzero_exp_avg_sq":sum(bool(torch.count_nonzero(v["exp_avg_sq"])) for v in hp["optimizer"]["state"].values()),
                "H_optimizer_parameter_states":len(hp["optimizer"]["state"])}
            for arm in ("A1","B"):
                for index in (1,2):
                    rid=f"{arm}-r{index}"
                    unit(rndstore,rid+"-claim");m=unit(rndstore,rid)
                    claim=json.loads((rndstore.root/(rid+"-claim")/"claim.json").read_text())
                    assert claim=={"protocol":digest(c["prototype"]),"arm":arm,"round":index,
                        "parent_state":parent_complete_sha if index==1 else tree_sha(values[(arm,1)])}
                    value=torch.load(SMOKE/"rounds"/rid/"state.pt",map_location="cpu",weights_only=False)
                    assert tree_sha(value)==m["provenance"]["state_sha256"]
                    expected_parents={"G":c["sources"]["G60"]["sha256"],"H":c["sources"]["H10"]["sha256"],"protocol":digest(c["prototype"])}
                    assert value["parents"]==expected_parents
                    assert value["cursor"]=={"round":index,"g_updates":484080+27*index,"h_visits":20480+8*index,"h_updates":20370+7*index}
                    meta=state_metadata(value)
                    assert meta["G_Adam_steps"]==[484080+27*index]
                    assert max(meta["H_Adam_steps"])==20370+7*index
                    assert all(g["lr"]==2e-4 and tuple(g["betas"])==(.5,.999) for g in value["G_Adam"]["param_groups"])
                    for key in ["G","H","G_Adam","H_Adam"]:finite_state(value[key])
                    assert value["G_Adam"]["param_groups"]==gp["optimizer"]["param_groups"]
                    assert value["H_Adam"]["param_groups"]==hp["optimizer"]["param_groups"]
                    ledger=json.loads((SMOKE/"rounds"/rid/"ledger.json").read_text())
                    assert ledger["all_stages_verified"] and ledger["arm"]==arm and ledger["round"]==index
                    assert len(ledger["G"])==len(ledger["H"])==8
                    assert sum(g["optimizer_steps"] for g in ledger["G"])==27
                    assert sum(h.get("optimizer_steps_this_step",0) for h in ledger["H"])==7
                    initial_g=parent["G_state"] if index==1 else previous[arm]["G"]
                    initial_h=parent_native_h_sha if index==1 else previous[arm]["H_native_ledger_sha256"]
                    current_g,current_h=initial_g,initial_h
                    for position,(g,h,row) in enumerate(zip(ledger["G"],ledger["H"],c["prototype"]["samples"])):
                        assert g["G_before"]==current_g
                        assert g["optimizer_steps"]==g["K"]+1==len(g["records"])==len(g["gradient_observations"])
                        assert g["weights"]=={"lambda_global":1.,"lambda_local":1.,"lambda_retain":1.}
                        for record,obs in zip(g["records"],g["gradient_observations"]):
                            assert record==obs["record"] and record["finite_gradients"] and record["cumulative_linf"]<=8/255
                        current_g=g["G_after"]
                        assert h["row_id"]==row
                        if not h.get("fixed_NOOP",False):
                            assert h["H_state_sha256_before"]==current_h
                            current_h=h["H_state_sha256_after"]
                            assert h["optimizer_steps_this_step"]==1
                            assert h["reward_delta_same_text"]==[h[k]["total"]-old["total"] for k,old in zip(
                                ["sampled_reward","greedy_reward"],h["old_bank_same_text_rewards"])]
                    meta["H_native_ledger_sha256"]=native_h_sha(value["H"])
                    assert current_g==meta["G"] and current_h==meta["H_native_ledger_sha256"]
                    offset=((0 if arm=="A1" else 2)+index-1)*8
                    conditions=report["conditions"][offset:offset+8]
                    assert ledger["conditions"]==[condition_key_from_json(e["identity"]) for e in conditions]
                    assert [e["row"] for e in conditions]==c["prototype"]["samples"]
                    bs=ArtifactStore(SMOKE/arm/f"bank_{index}",new_base=c["output_base"])
                    gm=unit(bs,"G_state");bm=unit(bs,"completed")
                    g_saved=torch.load(bs.root/"G_state/state.pt",map_location="cpu",weights_only=True)
                    assert tree_sha(g_saved)==meta["G"]==gm["provenance"]["G"]
                    bid=json.loads((bs.root/"completed/identity.json").read_text())
                    assert bid==ledger["bank"]
                    assert bid["ordered_train_ids"]==c["prototype"]["train_ids"] and len(bid["ordered_train_ids"])==2048
                    assert bid["g_state_sha256"]==meta["G"] and bid["arm"]==arm and bid["round"]==index
                    assert bid["source_space_sha256"]==c["prototype"]["source_space_sha256"] and bid["clean_sha256"]==c["prototype"]["clean_gallery_sha256"]
                    bank_features=[];max_budget=0.;low=1.;high=0.
                    for position,row in enumerate(c["prototype"]["train_ids"]):
                        unit(bs,f"claim_{position:04d}");im=unit(bs,f"image_{position:04d}")
                        claim=json.loads((bs.root/f"claim_{position:04d}/claim.json").read_text())
                        assert claim=={"row":row,"index":position,"G":meta["G"]}
                        v=torch.load(bs.root/f"image_{position:04d}/data.pt",map_location="cpu",weights_only=True)
                        assert set(v)=={"original","adversarial","effective_delta","feature"}
                        for name in ("original","adversarial","effective_delta"):
                            assert v[name].shape==(1,3,384,128) and v[name].dtype==torch.float32
                        assert_pixels(v["original"],v["adversarial"],v["effective_delta"])
                        feature=v["feature"]
                        assert feature.shape==(1,512) and feature.dtype==torch.float32 and torch.isfinite(feature).all()
                        assert abs(float(feature.norm())-1)<=1e-6
                        md=im["provenance"]
                        assert json.loads((bs.root/f"image_{position:04d}/metadata.json").read_text())==md
                        assert md["row"]==row and md["index"]==position and md["G"]==meta["G"] and md["content_sha256"]==tree_sha(v)
                        bank_features.append(feature.detach().clone())
                        max_budget=max(max_budget,float(v["effective_delta"].abs().max()))
                        low=min(low,float(v["adversarial"].min()));high=max(high,float(v["adversarial"].max()))
                        if (position+1)%512==0:print("READONLY_BANK_VERIFIED",rid,position+1,flush=True)
                    feature=torch.cat(bank_features).clone()
                    assert tree_sha(feature)==bid["adversarial_sha256"]
                    saved=torch.load(bs.root/"completed/features.pt",map_location="cpu",weights_only=True)
                    assert tree_sha(saved)==tree_sha(feature)
                    assert not bs.pending()
                    for h in ledger["H"]:
                        if not h.get("fixed_NOOP",False):assert h["bank_sha256"]==bid["adversarial_sha256"]
                    hstat=h_summary(ledger["H"])
                    gnorm=[x["G_gradient_l2"] for g in ledger["G"] for x in g["records"]]
                    records[rid]={"arm":arm,"round":index,"source_paths":{"ledger":str(SMOKE/"rounds"/rid/"ledger.json"),
                        "state":str(SMOKE/"rounds"/rid/"state.pt"),"bank":str(bs.root)},
                        "round_manifest":sha_file(rndstore.root/rid/"manifest.json"),"state_file_sha256":sha_file(rndstore.root/rid/"state.pt"),
                        "ledger_sha256":sha_file(rndstore.root/rid/"ledger.json"),"Round_claim":claim,"state":meta,
                        "G_steps":27,"H":hstat,"G_gradient_L2_min":min(gnorm),"G_gradient_L2_max":max(gnorm),
                        "bank":{"images":2048,"feature_content_sha256":bid["adversarial_sha256"],
                        "g_state_sha256":bid["g_state_sha256"],"bank_manifest_sha256":sha_file(bs.root/"completed/manifest.json"),
                        "ordered_ids_sha256":digest(bid["ordered_train_ids"]),"maximum_linf":max_budget,
                        "adversarial_min":low,"adversarial_max":high,"all_claims_SHA_shapes_FP32_finite_L2_budget_verified":True},
                        "G_logs":ledger["G"],"H_logs":ledger["H"],"conditions":conditions}
                    values[(arm,index)]=value
                    previous[arm]=meta
            for i in (1,2):
                records[f"A1-r{i}"]["versus_B_state_tensor_difference"]={
                    k:tensor_difference(values[("A1",i)][k],values[("B",i)][k]) for k in ["G","H"]}
            a,b=records["A1-r1"]["G_logs"],records["B-r1"]["G_logs"]
            effects=feedback(a,b,[h for r in records.values() for h in r["H_logs"]])
            noop_condition_equal=report["conditions"][0]["payload_content_sha256"]==report["conditions"][16]["payload_content_sha256"]
            assert noop_condition_equal
            assert not (SMOKE/"audit").exists(),"original audit unexpectedly wrote output"
            assert not torch.cuda.is_initialized()
        assert inventory(SMOKE)==before and inventory(CTL)==control_before,"original Smoke/control metadata changed"
        protected=verify_history()
        # 每个ArtifactStore.verify已实际重读所有payload文件SHA；此索引汇总其完整Manifest内容。
        unit_index={"units":len(units),"ordered_unit_manifest_digest":digest(units),"original_files":len(before)}
        result={"status":"READONLY_REVIEW_COMPLETE_PROTOCOL_FAIL","input_source_key":source_key(c),
            "review_identity":identity,"review_producer_sha256":producer.sha256,
            "GPU_generation":"PASS","artifact_structure_integrity":"PASS","original_CPU_audit":"FAIL",
            "original_error":{"type":"ValueError","message":"first NOOP control differs","path":str(FROZEN/"src/tbps/alternating/native_cli.py"),"line":178},
            "NOOP_condition_content_exact":noop_condition_equal,"complete_parent_state_sha256":parent_complete_sha,"parent_momentum":parent_momentum,"H_hash_schema":{"framework_tree_sha":parent["H_state"],"native_H_ledger_sha":parent_native_h_sha,"source_path":str(h_hash_path),"source_sha256":c["runtime_pins"][str(h_hash_path)]},"feedback_effects":effects,
            "rounds":records,"native_report_without_conditions":{k:v for k,v in report.items() if k!="conditions"},
            "totals":{"G_Adam_updates":108,"H_visits":32,"H_Adam_updates":28,"bank_refreshes":4,"bank_images":8192,
                "nonzero_H_advantage_steps":sum(r["H"]["nonzero_advantage"] for r in records.values()),
                "nonzero_H_gradient_steps":sum(r["H"]["nonzero_gradient"] for r in records.values()),
                "H_state_changed_steps":sum(r["H"]["H_state_changed"] for r in records.values()),
                "zero_gradient_but_H_state_changed_steps":sum(r["H"]["zero_gradient_but_state_changed"] for r in records.values()),
                "same_text_Reward_changed_steps":sum(r["H"]["same_text_Reward_changed"] for r in records.values()),
                "same_text_Advantage_changed_steps":sum(r["H"]["same_text_Advantage_changed"] for r in records.values())},
            "unit_SHA_index":unit_index,"original_Smoke_and_control_unchanged":True,"history":protected,
            "readonly_review_seconds":time.monotonic()-start,"CUDA_initialized":False,"model_forwards":0,
            "new_gradients_or_rewards_computed":0,"original_audit_modified":False,
            "strict_protocol_audit_pass":False,"cause_of_NOOP_gradient_nondeterminism":"NOT_VERIFIED",
            "first_NOOP_parameter_tensor_distance":"NOT_VERIFIED: only per-step SHA retained; no first-step tensors"}
        # 独立新回执如实记录协议失败；不发布原审计的PASS标志。
        target.publish("completed",{"findings.json":json_bytes(result)},{"review_identity":identity,"strict_protocol_PASS":False},sources=(producer,))
        print("READONLY_STRUCTURE_PASS_STRICT_CONTROL_FAIL",flush=True)
        return result
    except BaseException:
        import traceback
        with (OUTPUT/"FAIL").open("x") as f:f.write(traceback.format_exc())
        raise

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--allow-readonly-review",action="store_true");a=ap.parse_args()
    if not a.allow_readonly_review:raise SystemExit("explicit CPU read-only review required")
    value=execute()
    raise SystemExit(0 if value["strict_protocol_audit_pass"] else 2)
