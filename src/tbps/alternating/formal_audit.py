"""B新实验独立CPU审计；不调用GPU，不改判旧Smoke的exact审计。"""
from pathlib import Path
from dataclasses import asdict
from tbps.artifacts import ArtifactStore,FileRef,json_bytes,read_json,sha_file
from .contracts import require
from .formal_sources import source_preflight
from .formal_cli import history
from .state import tree_sha
from .native_sources import source_key

def audit(path,scope):
    f,c,pre=source_preflight(path,scope)
    import torch
    require(not torch.cuda.is_initialized(),"CPU audit only")
    from .native_backend import assert_pixels
    from .parents import no_optimizer_steps
    root=Path(f["outputs"][scope]);store=ArtifactStore(root,new_base=f["output_base"])
    preserved={}
    rsources=ArtifactStore(root/"rounds",new_base=f["output_base"])
    for d in (root/"rounds").glob("resume_*"):
        rsources.verify(d.name)
        for v in read_json(d/"resume.json").get("preserved_failures",[]):preserved[v["path"]]=v["sha256"]
    require(all(preserved.get(str(p))==sha_file(p) for p in root.glob("FAIL*"))
        and not store.pending(),"unacknowledged failure/pending output retained")
    store.verify("generated",expected_provenance={"formal_config":sha_file(path),"scope":scope})
    report=read_json(root/"generated/report.json")
    require(report["complete"] and report["frozen_sources"] and report["scope"]==scope
        and report["native_source"]==source_key(c) and report["status"]==
        ("GH_B_FORMAL_PILOT_GENERATED" if scope=="pilot" else "GH_B_FORMAL_GENERATED"),"GPU generated/freeze/source receipt")
    rounds=c["prototype"]["rounds"];previous=None;records=[]
    with no_optimizer_steps():
        for i in range(1,rounds+1):
            rs=ArtifactStore(root/"rounds",new_base=f["output_base"]);m=rs.verify(f"round_{i:04d}")
            rs.verify(f"round_{i:04d}_claim",expected_provenance={"config":sha_file(path),"round":i})
            require(m["provenance"]["config"]==sha_file(path) and m["provenance"]["round"]==i,"round protocol binding")
            d=root/"rounds"/f"round_{i:04d}"
            state=torch.load(d/"state.pt",map_location="cpu",weights_only=False)
            ledger=read_json(d/"ledger.json")
            require(set(state)=={"G","H","G_Adam","H_Adam","G_RNG","H_RNG","cursor","parents"},"complete dual state")
            from tbps.ablations.gv2_epoch_state import adam_steps,finite_state
            for name in ("G","H","G_Adam","H_Adam"):finite_state(state[name])
            require(adam_steps(state["G_Adam"])==[state["cursor"]["g_updates"]]
                and max(adam_steps(state["H_Adam"]))==state["cursor"]["h_updates"],"dual Adam absolute steps")
            for name in ("G_RNG","H_RNG"):
                rng=state[name]
                require(set(rng)=={"python","numpy","torch_cpu","cuda","sampler"}
                    and isinstance(rng["cuda"],list) and len(rng["cuda"])==1,"complete one-device phase RNG")
            require(tree_sha(state)==m["provenance"]["state"] and ledger["end_cursor"]==state["cursor"],"round state content")
            require(tree_sha(state["G"])==ledger["G_after"] and tree_sha(state["H"])==ledger["H_after"],"model binding")
            require(state["parents"]=={"G":c["sources"]["G60"]["sha256"],"H":c["sources"]["H10"]["sha256"],
                "protocol":__import__("tbps.artifacts",fromlist=["digest"]).digest(c["prototype"])},"full parent chain")
            if previous:
                require(ledger["G_before"]==previous["G_after"] and ledger["H_before"]==previous["H_after"]
                    and ledger["start_cursor"]==previous["end_cursor"],"Round reinitialized or discontinuous")
            require(ledger["condition_H_state"]==ledger["H_before"],"current H condition binding")
            before=ledger["start_cursor"];after=ledger["end_cursor"]
            gu=sum(x["optimizer_steps"] for x in ledger["G"])
            hu=sum(x.get("optimizer_steps_this_step",1) for x in ledger["H"] if not x.get("fixed_NOOP"))
            require(after=={"round":i,"g_updates":before["g_updates"]+gu,
                "h_visits":before["h_visits"]+len(ledger["H"]),"h_updates":before["h_updates"]+hu},"cursor/update accounting")
            require(len(ledger["g_order"])==len(ledger["h_order"])==len(c["prototype"]["samples"])
                and set(ledger["g_order"])==set(ledger["h_order"])==set(c["prototype"]["samples"]),"training member completeness")
            require(len(ledger["conditions"])==len(ledger["g_order"]),"condition member coverage")
            io=ArtifactStore(root/"round_io"/f"round_{i:04d}",new_base=f["output_base"])
            for position,condition in enumerate(ledger["conditions"]):
                require(condition["row"]==ledger["g_order"][position],"condition order")
                from tbps.artifacts import digest
                io.verify(f"condition_{position:04d}",expected_provenance={
                    "source":source_key(c),"condition":digest(condition["identity"])})
                io.verify(f"rollout_{position:04d}")
                require(read_json(io.root/f"condition_{position:04d}/evidence.json")==condition
                    and read_json(io.root/f"rollout_{position:04d}/manifest.json")["provenance"]["source"]==source_key(c),
                    "condition/rollout source evidence")
                payload=torch.load(io.root/f"condition_{position:04d}/condition.pt",map_location="cpu",weights_only=True)
                require(tree_sha(payload)==condition["payload_content_sha256"],"condition Tensor content")
                ev=condition["evidence"]
                require(ev["new_JVPs"]>=0,"condition accounting")
                # 非NOOP不能以旧缓存冒充新语境；prepare_condition本身还校验Token、FD和Canonical。
                if condition["identity"]["original_caption_sha256"]!=condition["identity"]["context_caption_sha256"]:
                    require(ev["new_text_encodings"]>0 and ev["new_JVPs"]==len(condition["identity"]["selected_g_ids"])
                        and ev["fresh_localization"] and ev["old_JVP_relabelled"] is False,"missing real context encoding")
                    require(all(q["zero_exact"] and q["central_fd_relative_error"]<=.02 for q in ev["JVP_checks"]),
                            "fresh JVP zero/FD")
                else:require(ev["new_JVPs"]==ev["new_text_encodings"]==0,"NOOP should reuse exact original condition")
            require(not io.pending(),"pending context/rollout")
            bankroot=root/"B"/f"bank_{i}";bs=ArtifactStore(bankroot,new_base=f["output_base"])
            bs.verify("G_state");bs.verify("completed")
            bid=read_json(bankroot/"completed/identity.json")
            require(bid==ledger["bank"] and bid["g_state_sha256"]==ledger["G_after"]
                and bid["ordered_train_ids"]==c["prototype"]["train_ids"],"bank source/order")
            features=[];maximum=0.
            for j,row in enumerate(c["prototype"]["train_ids"]):
                bs.verify(f"claim_{j:04d}");md=bs.verify(f"image_{j:04d}")["provenance"]
                v=torch.load(bankroot/f"image_{j:04d}/data.pt",map_location="cpu",weights_only=True)
                assert_pixels(v["original"],v["adversarial"],v["effective_delta"])
                require(md["row"]==row and md["index"]==j and md["G"]==ledger["G_after"]
                    and md["content_sha256"]==tree_sha(v),"bank pixel/feature SHA")
                require(v["feature"].shape==(1,512) and v["feature"].dtype==torch.float32
                    and torch.isfinite(v["feature"]).all()
                    and abs(float(v["feature"].norm())-1.)<=1e-5,"native bank feature FP32/L2")
                features.append(v["feature"]);maximum=max(maximum,float(v["effective_delta"].abs().max()))
            require(tree_sha(torch.cat(features))==bid["adversarial_sha256"] and not bs.pending(),"full bank content")
            saved=torch.load(bankroot/"completed/features.pt",map_location="cpu",weights_only=True)
            require(tree_sha(saved)==bid["adversarial_sha256"],"compact bank features")
            delta=[x for h in ledger["H"] for x in h.get("reward_delta_same_text",[])]
            records.append({"round":i,"cursor":state["cursor"],"bank":bid,"G_changed":ledger["G_changed"],
                "H_changed":ledger["H_changed"],"G_updates":gu,"H_updates":hu,
                "max_Linf":maximum,"same_text_reward_delta_nonzero":any(x!=0 for x in delta),
                "same_text_advantage_delta_nonzero":any(len(h.get("reward_delta_same_text",[]))==2
                    and h["reward_delta_same_text"][0]!=h["reward_delta_same_text"][1] for h in ledger["H"]),
                "state_manifest":asdict(FileRef(str(d/"manifest.json"),sha_file(d/"manifest.json")))})
            previous=ledger
    require(report["counts"]["G"]==report["counts"]["IRRA_image"]
        ==sum(x["G_updates"]+len(c["prototype"]["train_ids"]) for x in records),"actual full image forward count")
    if scope=="pilot":
        require(any(x["G_changed"] for x in records) and any(x["H_changed"] for x in records),"Pilot parameter updates not demonstrated")
    result={"status":"GH_B_FORMAL_PILOT_CPU_AUDIT_PASS" if scope=="pilot" else "GH_B_FORMAL_CPU_AUDIT_PASS",
        "scope":scope,"rounds":records,"GPU_report":report,"history":history(f),"source":pre,
        "performance_improvement":"NOT_EVALUATED","old_Smoke_exact_FAIL_unchanged":True}
    out=ArtifactStore(Path(f["audits"][scope]),new_base=f["output_base"])
    out.publish("completed",{"report.json":json_bytes(result)},{"formal_configuration":sha_file(path),"scope":scope})
    if scope=="pilot":
        from tbps.artifacts import digest
        ready=ArtifactStore(f["output_base"],new_base=f["output_base"])
        proof=FileRef(str(out.root/"completed/report.json"),sha_file(out.root/"completed/report.json"))
        ready.publish("readiness",{"ready.json":json_bytes({
            "status":"READY_FOR_USER_AUTHORIZATION","formal_configuration_sha256":sha_file(path),
            "producer_sha256":digest(f["producers"]),"pilot_audit":asdict(proof),
            "automatically_start_training":False})},{"formal_configuration":sha_file(path)},sources=(proof,))
    return result
