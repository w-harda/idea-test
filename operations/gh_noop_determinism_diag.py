"""独立NOOP数值诊断：只观察冻结原训练器，四次父状态首步，不重跑Smoke。"""
import argparse
from contextlib import contextmanager, ExitStack
import fcntl
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

FROZEN=Path("/home/lzf/ldx/projects/idea-test-tbps-gh-probe-shared-20261023")
CONFIG=FROZEN/"configs/framework/gh_bidirectional_backend.probe_v2.shared.server.json"
CONFIG_SHA="77ae41af56189ece991ee7cf21fd0ea0f3be5645202bd1cb7cf7e0b2ae735649"
SOURCE_KEY="06b5c6a15061e635e49a9e13deb14022956fdbc272ad0ed378676e57ca5f826b"
PROBE_SHA="d6e0b960e20a1ba5128f9e0b4edf497e7ea4b098fbf5cd3125799d2ad1136764"
OUTPUT=Path("/home/lzf/ldx/outputs/tbps-framework-gh-noop-determinism/noop_determinism_diag_v1")
SMOKE=Path("/home/lzf/ldx/outputs/tbps-framework-gh-bidirectional-backend/smoke_v2")
CONTROL=Path("/home/lzf/ldx/tmp/gh_smoke_wait_v1_20261023")
ROW="cuhk:26530:0"
RUNS=(("A1_repeat_1","A1"),("A1_repeat_2","A1"),("A1_cross","A1"),("B_cross","B"))

def need(ok,message):
    if not ok:raise ValueError(message)

def flatten(value,prefix=""):
    import torch
    if isinstance(value,torch.Tensor):return {prefix:value}
    result={}
    if isinstance(value,dict):
        for k,v in value.items():result.update(flatten(v,prefix+"/"+str(k)))
    elif isinstance(value,(list,tuple)):
        for k,v in enumerate(value):result.update(flatten(v,prefix+"/"+str(k)))
    return result

def compare(a,b):
    import torch
    x,y=flatten(a),flatten(b)
    need(set(x)==set(y),"different tensor members")
    rows=[];squared=0.;reference=0.;maximum=0.;changed=0;elements=0
    for key in sorted(x):
        p,q=x[key],y[key]
        need(p.shape==q.shape and p.dtype==q.dtype,"shape/dtype mismatch: "+key)
        d=p.double()-q.double();ss=float(d.square().sum());rr=float(p.double().square().sum())
        m=float(d.abs().max()) if p.numel() else 0.
        count=int((p!=q).sum())
        rows.append({"tensor":key,"shape":list(p.shape),"dtype":str(p.dtype),"exact":bool(torch.equal(p,q)),
            "max_abs":m,"L2":math.sqrt(ss),"relative_L2":math.sqrt(ss)/max(math.sqrt(rr),1e-300),
            "reference_L2":math.sqrt(rr),"changed_elements":count,"elements":p.numel()})
        squared+=ss;reference+=rr;maximum=max(maximum,m);changed+=count;elements+=p.numel()
    return {"exact":all(r["exact"] for r in rows),"max_abs":maximum,"L2":math.sqrt(squared),
        "relative_L2":math.sqrt(squared)/max(math.sqrt(reference),1e-300),
        "changed_elements":changed,"elements":elements,"per_tensor":rows}

def first_divergence(stages):
    return next((name for name,value in stages.items() if not value["exact"]),"NONE")

def metadata_tree(root):
    return {str(p):[p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ino]
        for p in sorted(root.rglob("*")) if p.is_file()}

def runtime_settings():
    import torch
    c=torch.backends.cuda
    return {"torch":torch.__version__,"CUDA_build":torch.version.cuda,"cuDNN":torch.backends.cudnn.version(),
        "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only":torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic":torch.backends.cudnn.deterministic,"cudnn_benchmark":torch.backends.cudnn.benchmark,
        "cudnn_tf32":torch.backends.cudnn.allow_tf32,"matmul_tf32":c.matmul.allow_tf32,
        "math_sdp":c.math_sdp_enabled(),"flash_sdp":c.flash_sdp_enabled(),
        "memory_efficient_sdp":c.mem_efficient_sdp_enabled(),"threads":torch.get_num_threads(),
        "CUBLAS_WORKSPACE_CONFIG":os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_build":torch.__config__.show(),
        "CUDA_VISIBLE_DEVICES":os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device":torch.cuda.get_device_name(0) if torch.cuda.is_initialized() else None}

def modes(module):
    return {"training":{n:m.training for n,m in module.named_modules()},
        "requires_grad":{n:p.requires_grad for n,p in module.named_parameters()},
        "parameter_layout":{n:{"shape":list(p.shape),"stride":list(p.stride()),"dtype":str(p.dtype),
             "device":str(p.device)} for n,p in module.named_parameters()}}

@contextmanager
def observations(g,encoder):
    """只读forward/Tensor backward hook；不改返回值，不改Loss、grad或Optimizer。"""
    from tbps.alternating.state import clone_tree
    data={"G_layers":{},"G_layer_input_gradients":{},"G_layer_output_gradients":{}}
    handles=[]
    def watch_tensor(value,key,target):
        for path,t in flatten(value).items():
            if t.requires_grad:
                handles.append(t.register_hook(lambda grad,k=key+path,to=target: to.__setitem__(k,clone_tree(grad))))
    def g_output(module,args,output):
        data["G_input"]=clone_tree(args[0]);data["G_forward"]=clone_tree(output)
        watch_tensor(output,"",data.setdefault("G_forward_gradient",{}))
        data["effective_G_modes"]=modes(g)
    def feature(module,args,output):
        data["projected_RGB"]=clone_tree(args[0])
        data["IRRA_feature"]=clone_tree(output)
        watch_tensor(args[0],"",data.setdefault("projected_RGB_gradient",{}))
        watch_tensor(output,"",data.setdefault("IRRA_feature_gradient",{}))
    handles.append(g.register_forward_hook(g_output))
    handles.append(encoder.register_forward_hook(feature))
    for name,mod in g.named_modules():
        if name and not list(mod.children()):
            def layer(module,args,output,n=name):
                data["G_layers"][n]={"input":clone_tree(args),"output":clone_tree(output),
                    "class":type(module).__name__}
                watch_tensor(args,n,data["G_layer_input_gradients"])
                watch_tensor(output,n,data["G_layer_output_gradients"])
            handles.append(mod.register_forward_hook(layer))
    try:yield data
    finally:
        for h in reversed(handles):h.remove()

def tensor_blob(value):
    import torch
    from tbps.alternating.state import clone_tree
    b=io.BytesIO();torch.save(clone_tree(value),b);return b.getvalue()

def stage_comparison(a,b):
    # 只比较真实保存的Tensor；Loss转FP64仅用于比较原记录的Python浮点值。
    import torch
    stage={}
    for name,key in (("G_BEFORE","G_before"),("ADAM_BEFORE","Adam_before"),
        ("RNG_BEFORE","rng_before"),("INPUT","image"),("CONDITION","condition"),
        ("G_FORWARD","G_forward"),("PROJECTED_RGB","projected_RGB"),("IRRA_FEATURE","IRRA_feature")):
        stage[name]=compare(a[key],b[key])
    stage["LOSS"]=compare({k:torch.tensor(a["loss_record"][k],dtype=torch.float64) for k in ("global","local","retain","total")},
                          {k:torch.tensor(b["loss_record"][k],dtype=torch.float64) for k in ("global","local","retain","total")})
    for name,key in (("LOSS_TO_FEATURE_GRAD","IRRA_feature_gradient"),
        ("IRRA_INPUT_GRAD","projected_RGB_gradient"),("G_OUTPUT_GRAD","G_forward_gradient"),
        ("G_PARAMETER_GRAD","parameter_gradients"),("G_AFTER","G_after"),("ADAM_AFTER","Adam_after")):
        stage[name]=compare(a[key],b[key])
    return {"stages":stage,"first_divergence_stage":first_divergence(stage),
        "G_layer_forward":compare(a["G_layers"],b["G_layers"]),
        "G_layer_input_gradients":compare(a["G_layer_input_gradients"],b["G_layer_input_gradients"]),
        "G_layer_output_gradients":compare(a["G_layer_output_gradients"],b["G_layer_output_gradients"]),
        "full_rng_before_exact":a["rng_before_sha256"]==b["rng_before_sha256"],
        "condition_full_exact":a["condition_sha256"]==b["condition_sha256"],
        "effective_G_modes_exact":a["effective_G_modes"]==b["effective_G_modes"],
        "IRRA_modes_exact":a["IRRA_modes"]==b["IRRA_modes"]}

def check_history():
    # 复用已发布独立RNG工具的只读保护函数，冻结生产代码无变化。
    import importlib.util
    p=Path(__file__).with_name("gh_cuda_rng_validation.py")
    spec=importlib.util.spec_from_file_location("diag_protection",p);m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.verify_history()

def diagnostic_preflight_protocol(c):
    # 仅将唯一Attempt检查定位到本次独立诊断；所有来源、算法与预算字段逐值不变。
    return {**c["prototype"],"output_base":str(OUTPUT)}

def preflight():
    from tbps.artifacts import sha_file,read_json,ArtifactStore
    from tbps.alternating.native_sources import verify_config,strict_import_preflight
    from tbps.alternating.preflight import preflight as original_source_preflight
    from tbps.alternating.native_sources import source_key
    from tbps.alternating.native_gate import verify_probe
    need(not OUTPUT.exists(),"diagnostic Attempt consumed")
    need(sha_file(CONFIG)==CONFIG_SHA,"frozen configuration SHA")
    c=verify_config(CONFIG)
    protocol=diagnostic_preflight_protocol(c)
    sources=original_source_preflight(protocol)
    import torch
    from tbps.materialization.irra_startup import CPUStartupGate
    from tbps.alternating.parents import restore_cpu_parents
    events=[];gate=CPUStartupGate(events.append)
    imports=strict_import_preflight(c);bundle=restore_cpu_parents(c)
    gate.checked();gate.exit_checked()
    need(not torch.cuda.is_initialized(),"CPU diagnostic initialized CUDA")
    pre={"sources":sources,"imports":imports,"parent_restore":bundle.report,"startup_events":events,
        "preflight_derivation":{"allowed_changes":["output_base"],"parent_output":c["prototype"]["output_base"],
            "diagnostic_output":str(OUTPUT),"parent_protocol_sha256":__import__("tbps.artifacts",fromlist=["digest"]).digest(c["prototype"]),
            "diagnostic_preflight_protocol_sha256":__import__("tbps.artifacts",fromlist=["digest"]).digest(protocol)}}
    need(source_key(c)==SOURCE_KEY and verify_probe(c)==PROBE_SHA,"frozen source/probe mismatch")
    store=ArtifactStore(SMOKE,new_base=c["output_base"]);store.verify("completed")
    need(read_json(SMOKE/"completed/report.json")["status"]=="GH_NATIVE_TWO_ROUND_SMOKE_GENERATED","native Smoke receipt")
    need("first NOOP control differs" in (CONTROL/"cpu_audit.log").read_text(),"original failure must remain")
    need(subprocess.check_output(["git","-C",str(FROZEN),"rev-parse","HEAD"],text=True).strip()=="aafc7715ddafa78f9a30410b6a6ed8852e81780a","frozen HEAD")
    need(not subprocess.check_output(["git","-C",str(FROZEN),"status","--porcelain"],text=True).strip(),"frozen worktree changed")
    pre.update(history=check_history(),smoke_metadata=metadata_tree(SMOKE),control_metadata=metadata_tree(CONTROL))
    return c,bundle,gate,pre

def execute(allow):
    need(allow,"explicit GPU authorization --allow-gpu-diagnostic required")
    from tbps.artifacts import ArtifactStore,FileRef,json_bytes,sha_file,digest
    from tbps.alternating.preflight import resource_snapshot
    from tbps.alternating.native_sources import source_key,verify_config
    # 外部锁目录不是Attempt；CPU/资源拒绝不消费唯一身份。
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    with (OUTPUT.parent/"noop_diagnostic.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        c,bundle,gate,pre=preflight()
        resource_protocol={**c["prototype"],"proposed_gpu_limits":{
            **c["prototype"]["proposed_gpu_limits"],"minimum_free_mib":6144}}
        resources=resource_snapshot(resource_protocol)
        need(resources["resource_eligible"],"RESOURCE_BUSY: no Attempt/CUDA created")
        plan={"schema":"gh-noop-determinism-diag-1","parent_config_sha256":CONFIG_SHA,"source_key":SOURCE_KEY,
            "row":ROW,"runs":RUNS,"max_G_updates":4,"max_G_forwards":4,"max_IRRA_images":4,
            "H_forwards":0,"JVP":0,"IRRA_text":0,"gallery_refresh":0,
            "environment":"original fp32_runtime; deterministic flags are observed, NOT changed",
            "minimum_free_mib":6144,"script_sha256":sha_file(__file__),
            "preflight_derivation":pre["preflight_derivation"],
            "protection_script_sha256":sha_file(Path(__file__).with_name("gh_cuda_rng_validation.py")),
            "git_commit":subprocess.check_output(["git","-C",str(Path(__file__).parents[1]),"rev-parse","HEAD"],text=True).strip()}
        store=ArtifactStore(OUTPUT,new_base=OUTPUT.parent)
        store.publish("claim",{"plan.json":json_bytes(plan),"preflight.json":json_bytes(pre),
            "resources.json":json_bytes(resources)},{"source":SOURCE_KEY,"diagnostic":digest(plan)},
            sources=(FileRef(str(CONFIG),CONFIG_SHA),FileRef(str(Path(__file__).resolve()),plan["script_sha256"]),
                     FileRef(str(Path(__file__).with_name("gh_cuda_rng_validation.py")),plan["protection_script_sha256"])))
        started=time.monotonic()
        try:
            import torch,resource
            from tbps.models.irra import fp32_runtime
            from tbps.alternating.native_backend import loaded_backend
            from tbps.alternating.native_gate import ExecutionGrant
            from tbps.alternating.state import clone_tree,tree_sha,capture_rng
            torch.set_num_threads(2);gate.initialize_cuda()
            grant=ExecutionGrant("smoke",SOURCE_KEY,digest(c["prototype"]),PROBE_SHA)
            runs={};counts={"G":0,"encoder":0,"Adam":0}
            with fp32_runtime(),loaded_backend(c,bundle,grant) as b:
                # 外部诊断只有NOOP rollout/condition落盘；绝不写原Smoke root或启动Prototype。
                b.store=ArtifactStore(OUTPUT/"condition_evidence",new_base=OUTPUT.parent)
                def count(name):
                    def hook(*args):
                        counts[name]+=1;need(counts[name]<=4,"diagnostic budget "+name)
                    return hook
                def forbidden(*args,**kwargs):raise ValueError("diagnostic forbids H/reader/text neural calls")
                with ExitStack() as stack:
                    for h in (b.g.register_forward_pre_hook(count("G")),
                        b.encoder.register_forward_pre_hook(count("encoder")),
                        b.ga.register_step_pre_hook(count("Adam")),b.h.register_forward_pre_hook(forbidden),
                        b.reader.register_forward_pre_hook(forbidden),b.ha.register_step_pre_hook(forbidden)):
                        stack.callback(h.remove)
                    original_text=b.irra.encode_text;b.irra.encode_text=forbidden
                    stack.callback(setattr,b.irra,"encode_text",original_text)
                    frozen={"H":tree_sha(b.h.state_dict()),"H_Adam":tree_sha(b.ha.state_dict())}
                    for label,arm in RUNS:
                        b.restore_complete(clone_tree(bundle.state))
                        b.enter_phase_rng("H")
                        trace=b.h_rollout(b.public_text(ROW));b.leave_phase_rng("H")
                        need(trace.original==trace.adversarial and not trace.edits,"locked NOOP only")
                        condition=b.condition(arm,ROW,trace)
                        need(condition.payload["K"]==0 and condition.evidence["new_JVPs"]==0,"single NOOP K0 update")
                        b.enter_phase_rng("G")
                        saved={"G_before":clone_tree(b.g.state_dict()),"Adam_before":clone_tree(b.ga.state_dict()),
                            "rng_before":capture_rng(b.sampler),"condition":clone_tree(condition.payload),
                            "condition_evidence":condition.evidence,"G_modes_before_train":modes(b.g),
                            "IRRA_modes":modes(b.irra),"H_modes":modes(b.h),"runtime":runtime_settings(),
                            "G_pending_gradients_before":{n:p.grad is not None for n,p in b.g.named_parameters()}}
                        saved["rng_before_sha256"]=tree_sha(saved["rng_before"])
                        saved["condition_sha256"]=tree_sha(saved["condition"])
                        need(tree_sha(saved["G_before"])==tree_sha(bundle.state["G"])
                            and tree_sha(saved["Adam_before"])==tree_sha(bundle.state["G_Adam"]),"not same parent")
                        with observations(b.g,b.encoder) as observed:
                            # 实际NativeBackend.g_update -> LegacySteps -> 冻结train_single_sample。
                            log=b.g_update(ROW,condition)
                        b.leave_phase_rng("G")
                        saved.update(observed)
                        saved.update(parameter_gradients={n:clone_tree(p.grad) for n,p in b.g.named_parameters()},
                            G_after=clone_tree(b.g.state_dict()),Adam_after=clone_tree(b.ga.state_dict()),
                            rng_after=capture_rng(b.sampler),loss_record=log["records"][0],native_log=log)
                        # 根G hook捕获实际训练输入；不增加任何前向。
                        saved["image"]=observed["G_input"]
                        need(log["optimizer_steps"]==1,"more than one NOOP update")
                        need(tree_sha(b.h.state_dict())==frozen["H"] and tree_sha(b.ha.state_dict())==frozen["H_Adam"],"H changed")
                        b.final_frozen()
                        runs[label]=saved
                        store.publish(label,{"tensors.pt":tensor_blob(saved),"log.json":json_bytes({
                            "arm":arm,"native_log":log,"runtime":saved["runtime"],
                            "before_G":tree_sha(saved["G_before"]),"before_Adam":tree_sha(saved["Adam_before"]),
                            "after_G":tree_sha(saved["G_after"]),"after_Adam":tree_sha(saved["Adam_after"]),
                            "before_rng":saved["rng_before_sha256"],"after_rng":tree_sha(saved["rng_after"])})},
                            {"diagnostic":digest(plan),"label":label})
                        print("NOOP_DIAGNOSTIC_UPDATE_COMPLETE",label,counts,flush=True)
                comparisons={"same_arm":stage_comparison(runs["A1_repeat_1"],runs["A1_repeat_2"]),
                    "cross_arm":stage_comparison(runs["A1_cross"],runs["B_cross"])}
                report={"status":"NOOP_DIAGNOSTIC_COMPLETE","comparisons":comparisons,"counts":counts,
                    "native_counts":b.counts,"frozen_sources":b.final_frozen(),
                    "SAME_ARM_REPEAT_EXACT":all(s["exact"] for s in comparisons["same_arm"]["stages"].values()),
                    "SAME_NOOP_GRADIENT_EXACT":comparisons["cross_arm"]["stages"]["G_PARAMETER_GRAD"]["exact"],
                    "SAME_NOOP_UPDATE_EXACT":comparisons["cross_arm"]["stages"]["G_AFTER"]["exact"],
                    "NONDETERMINISTIC_KERNEL_CONFIRMED":False,"STATE_LEAK_CONFIRMED":False,
                    "kernel_note":"no profiler/kernel isolation; module-boundary observation only",
                    "elapsed_seconds":time.monotonic()-started,
                    "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes":torch.cuda.max_memory_reserved(),
                    "peak_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
            verify_config(CONFIG)
            need(metadata_tree(SMOKE)==pre["smoke_metadata"] and metadata_tree(CONTROL)==pre["control_metadata"],"original Smoke/log changed")
            report["history_after"]=check_history()
            report["original_smoke_control_metadata_unchanged"]=True
            store.publish("completed",{"result.json":json_bytes(report)},{"diagnostic":digest(plan),"source":SOURCE_KEY})
            print(json.dumps({k:v for k,v in report.items() if k!="comparisons"},ensure_ascii=False,indent=2),flush=True)
        except BaseException:
            (OUTPUT/"FAIL").write_text(traceback.format_exc())
            raise

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--allow-gpu-diagnostic",action="store_true")
    ap.add_argument("--preflight",action="store_true");args=ap.parse_args()
    sys.path.insert(0,str(FROZEN/"src"))
    if args.preflight:
        c,b,g,r=preflight()
        print(json.dumps({"status":"NOOP_DIAGNOSTIC_CPU_PREFLIGHT_PASS","source_key":SOURCE_KEY,
            "parent_restore":r["parent_restore"],"history":r["history"],"startup_events":r["startup_events"],
            "original_smoke_files":len(r["smoke_metadata"]),"original_control_files":len(r["control_metadata"]),
            "preflight_derivation":r["preflight_derivation"]},ensure_ascii=False,indent=2),flush=True)
    else:execute(args.allow_gpu_diagnostic)
