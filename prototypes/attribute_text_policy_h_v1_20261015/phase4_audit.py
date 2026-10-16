"""Phase4独立CPU审计：只读训练资产，无模型前向/更新/检索指标。"""
import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
from unittest.mock import patch
import torch
from phase2_sources import ROOT,read_json,new_json,sha,digest
from phase4_inputs import preflight,safe_output
from train_phase4 import readonly_verify,policy_factory
from phase4_training import optimizer_for
from phase4_checkpoint import restore,capture_rng,same
from rewards import TrainGallery,RewardEvaluator
from feedback_backend import state_sha

def verify_rooted_logs(output,summary):
    """完成summary的SHA由completed锁定；不只信任各Checkpoint可改写的sidecar。"""
    output=Path(output).resolve()
    expected={str(p/"sample.json") for p in (output/"checkpoints").glob("step_*")}
    locks=summary["step_log_sha256"]
    if set(locks)!=expected:
        raise ValueError("rooted log SHA coverage mismatch")
    for name,expected_sha in locks.items():
        if not Path(name).resolve().is_relative_to(output) or sha(name)!=expected_sha:
            raise ValueError("completion-rooted sample log SHA mismatch")

def forbidden(*a,**k):
    raise AssertionError("no new forward/update/CUDA during CPU audit")

def audit(output):
    output=safe_output(output)
    with patch.object(torch.cuda,"_lazy_init",forbidden), \
         patch.object(torch.nn.Module,"_call_impl",forbidden), \
         patch.object(torch.optim.Adam,"step",forbidden), \
         patch.object(torch.Tensor,"backward",forbidden):
        catalog,gallery,queries,features,probe,protocol=preflight()
        path,md,logs=readonly_verify(output,protocol)
        summary=read_json(output/"summary.json")
        verify_rooted_logs(output,summary)
        h=policy_factory(probe,protocol["config"])
        optimizer=optimizer_for(h,protocol["config"])
        sampler=torch.Generator(device="cpu")
        value=restore(path,h,optimizer,sampler,protocol,allow_cuda=False)
        if (state_sha(h)!=summary["final_H_state_sha256"]
                or not same(optimizer.state_dict(),value["optimizer"])
                or value["step"]!=32 or value["next_step"]!=33 or value["sample_cursor"]!=32):
            raise ValueError("final CPU H/Adam/progress restore mismatch")
        actual=capture_rng(sampler)
        if not all(same(actual[k],value["rng"][k]) for k in actual if k!="torch_cuda"):
            raise ValueError("CPU RNG restore mismatch")
        maximum_error=0.
        lookup={q.row_id:q for q in queries}
        for log in logs:
            q=lookup[log["row_id"]]
            checkpoint=output/"checkpoints"/f"step_{log['step']:04d}"
            text=torch.load(checkpoint/"text_features.pt",map_location="cpu",weights_only=True).clone()
            if hashlib.sha256(text.contiguous().numpy().tobytes()).hexdigest()!=log["text_feature_numeric_sha256"]:
                raise ValueError("real text feature numeric SHA mismatch")
            texts=(q.caption,log["sampled"]["adversarial"],log["greedy"]["adversarial"])
            if (log["sampled"]["original"]!=q.caption or log["greedy"]["original"]!=q.caption
                    or digest([probe.inspect(t).model_tokens for t in texts])!=log["token_sha256"]):
                raise ValueError("current-policy text/source/token mismatch")
            ctx=TrainGallery(features["clean"],features["adversarial"],
                torch.tensor([s.pid==q.pid for s in gallery]),gallery.index(q),
                "train",("train",)*10,"train",("train",)*10)
            a,b=RewardEvaluator(ctx,lambda _:text).evaluate_many(q.caption,texts[1:])
            for mode,reward in (("sampled",a),("greedy",b)):
                error=max(abs(asdict(reward)[k]-log[mode+"_reward"][k]) for k in asdict(reward))
                maximum_error=max(maximum_error,error)
        if maximum_error!=0.:
            raise ValueError("cached training reward differs from published step log")
        return {"passed":True,"step":32,"checkpoint_sha256":md["checkpoint_sha256"],
            "H_exact":True,"Adam_exact":True,"CPU_RNG_exact":True,
            "CUDA_RNG_saved_count":len(value["rng"]["torch_cuda"]),
            "CUDA_RNG_actual_restore_exact_at_steps":[e["step"] for e in summary["execution"]["restore_checks"] if e["RNG_exact"]],
            "query_order_exact":True,"rooted_step_logs_verified":33,
            "reward_reconstruction_max_abs_error":maximum_error,"reward_logs_verified":32,
            "new_model_forwards":0,"new_optimizer_steps":0,"cuda_initialized":torch.cuda.is_initialized(),
            "summary_sha256":sha(output/"summary.json"),"completed_sha256":sha(output/"completed.json")}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,default=ROOT/"outputs/phase4/scst32")
    p.add_argument("--output",type=Path,help="可选新审计报告；拒绝覆盖")
    a=p.parse_args(argv)
    result=audit(a.input)
    if a.output:new_json(safe_output(a.output),result)
    print("HV1_PHASE4_CPU_AUDIT_PASS",result,flush=True)
if __name__=="__main__":main()
