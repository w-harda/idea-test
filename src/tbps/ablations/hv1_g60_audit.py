"""独立CPU审计新H适配轨迹；不启动旧审计、模型前向或GPU。"""
from pathlib import Path
from dataclasses import asdict
import json,math
import torch
from tbps.artifacts import ArtifactStore,json_bytes,read_json,checked,FileRef
from tbps.attacks.legacy_runtime import legacy_imports
from .hv1_g60_state import *
from .g60_h_bridge import load_tensors,tensor_sha,metrics_csv
from .g60_h_validation import official_metrics

def audit_training(ctx,smoke=False):
    need(not torch.cuda.is_initialized(),"independent training audit CPU only")
    root=BASE/("smoke_v1" if smoke else "training")
    result=read_json(root/"completed.json");protocol=read_json(root/"protocol.json")
    need(result["status"]==("HV1_G60_SMOKE_PASS" if smoke else "HV1_G60_TRAINING_PASS")
         and result["parent_H10"]==ctx["c"]["parent"] and result["model_updates_only_H"]
         and result["H_changed"] and result["H_Adam_RNG_reload_exact"],"completed new H-only training evidence")
    need(result["runtime"]["IRRA_state_before"]==result["runtime"]["IRRA_state_after"]
         and result["runtime"]["parameters_buffers_unchanged"]
         and all(result["runtime"]["samples"][k]==0 for k in ("G","clean","adv")),"frozen IRRA/image budget")
    with legacy_imports(ctx["legacy"],allow_import=True) as old:
        payload,ledger=audit_versions(root,protocol,old)
        need(ledger["cursor"]==result["cursor"] and ledger["updates"]==result["updates"]
             and ledger["parent"]==result["last_checkpoint_sha256"],"new checkpoint chain exact")
        per_epoch={};query_order=protocol["query_order"]
        for item in ledger["published"]:
            for record in old("formal_training.checkpoint").logs(Path(item["path"])/"steps.jsonl"):
                epoch=(record["visit"]-1)//2048+1
                need(record["epoch"]==epoch and record["epoch_position"]==(record["visit"]-1)%2048+1,"absolute epoch")
                stat=per_epoch.setdefault(str(epoch),{"visits":0,"updates":0,"skipped":0})
                stat["visits"]+=1;stat["updates"]+=record["optimizer_steps_this_step"]
                if not record["optimizer_steps_this_step"]:stat["skipped"]+=1;continue
                need(record["row_id"]==query_order[record["visit"]-1] and record["IRRA_text_samples"]==3,"real fresh triplet binding")
                for mode in ("sampled","greedy"):
                    reward=record[mode+"_reward"];action=record[mode]
                    ex=old("contracts").from_text_spans(action["original"],action["attributes"])
                    edits=[old("constraints").EditRecord(**{**x,"source_span":tuple(x["source_span"])}) for x in action["edits"]]
                    need(old("constraints").replay(ex,edits,action["budget_k"])==action["adversarial"],"all Unicode actions replay")
                    need(all(math.isfinite(v) for v in reward.values())
                         and reward["total"]==reward["R_sim"]+reward["R_rank"]
                         and reward["synergy_increment"]==reward["adv_sim_drop"]-reward["clean_sim_drop"],"reward formula unchanged")
                need(math.isfinite(record["SCST_loss"]) and math.isfinite(record["gradient_norm"])
                     and record["parameter_delta_l2"]>=0,"H gradient/update finite")
        if smoke:
            need(result["new_visits"]==ctx["c"]["smoke_visits"] and result["updates"]>20370
                 and result["cursor"]==20480+ctx["c"]["smoke_visits"] and result["final_epoch"]==10,"smoke is partial Epoch11")
        else:
            need(list(per_epoch)==[str(i) for i in range(11,61)] and all(v=={"visits":2048,"updates":2037,"skipped":11}
                  for v in per_epoch.values()),"50 full new Epoch counts")
            need(ledger["cursor"]==122880 and ledger["updates"]==122220,"H60 totals")
            need(sum(x["reason"]=="epoch_complete" for x in ledger["published"])==50,"50 immutable Epoch Checkpoints")
        cfg=protocol["config"];probe=old("tokenizer_probe").official_probe()
        policy=old("policy").CharacterPolicy(probe,**cfg["architecture"])
        opt=old("phase4_training").optimizer_for(policy,cfg);sampler=torch.Generator()
        old("formal_training.checkpoint").restore(payload,policy,opt,sampler,allow_cuda=False)
        need(tree_hash(policy.state_dict())==tree_hash(payload["policy"])
             and tree_hash(opt.state_dict())==tree_hash(payload["optimizer"]),"independent final CPU restore")
    if not smoke:audit_validation(ctx)
    unchanged(ctx["report"])
    protected=read_json(ctx["c"]["sources"]["protection_snapshot"]["path"])
    for p,expected in protected.items():need(stat(p)==expected,"protected historical file changed: "+p)
    audit={"status":"HV1_G60_SMOKE_CPU_AUDIT_PASS" if smoke else "HV1_G60_TRAINING_CPU_AUDIT_PASS",
           "cursor":ledger["cursor"],"updates":ledger["updates"],"epochs":per_epoch,
           "new_checkpoints":len(ledger["published"]),"parent_H10":ctx["c"]["parent"],
           "G60":ctx["c"]["G60"],"new_visits":result["new_visits"],"new_updates":result["new_updates"],
           "full_RNG_saved_CPU_restoration_exact":True,"CUDA_RNG_not_applied_in_CPU_audit":True,
           "protected_historical_files":len(protected),"source_history_unchanged":True,
           "all_Unicode_actions_replayed":True,"old_Selection_untouched":True}
    out=ArtifactStore(BASE/("smoke_audit" if smoke else "training_audit"),new_base=BASE)
    out.publish("result",{"report.json":json_bytes(audit)},{"kind":"independent-new-adaptation-CPU-audit"},
                sources=(ref(root/"completed.json"),ref(root/"protocol.json")))
    print(audit["status"],json.dumps(audit),flush=True)

def audit_validation(ctx):
    oldroot=Path(ctx["c"]["validation_root"]);original=read_old_features(ctx,"validation","text")
    clean=read_old_features(ctx,"validation","clean")
    adv=load_tensors(oldroot/"features/gallery.pt")["adversarial"]
    identities=ctx["report"]["validation_identities"];pids=torch.tensor([x["pid"] for x in identities])
    for epoch in (20,30,40,50,60):
        root=BASE/"validation"/f"epoch_{epoch:04d}";store=ArtifactStore(root,new_base=BASE)
        expected={f"query-{i:03d}" for i in range(128)}|{"text-only","joint","completed"}
        need(not store.pending() and {p.name for p in root.iterdir() if p.is_dir()}==expected,"complete Validation set")
        for name in expected:store.verify(name)
        summary=read_json(root/"completed/summary.json");features=[]
        for i,row in enumerate(identities):
            record=read_json(root/f"query-{i:03d}/record.json")
            need(record["identity"]==row and record["checkpoint_sha256"]==summary["checkpoint_sha256"],"Validation identity/H")
            x=load_tensors(root/f"query-{i:03d}/text.pt")["text"];normalized(x,1);features.append(x)
        edited=torch.cat(features)
        for name,image in (("Text-only",clean),("Joint",adv)):
            scores=load_tensors(root/name.lower()/"scores.pt")["scores"]
            need(torch.equal(scores,edited@image.T),"native Validation score exact")
            need(official_metrics(scores,pids,ctx["source"].ref("utils/metrics.py").path)==summary["groups"][name],
                 "official Validation metrics exact")
        rows=[{"Group":k,**summary["groups"][k]} for k in ("Clean","Image-only","Text-only","Joint")]
        need((root/"completed/metrics_raw.csv").read_bytes()==metrics_csv(rows)
             and (root/"completed/metrics_2dp.csv").read_bytes()==metrics_csv(rows,True),"Validation CSV exact")
