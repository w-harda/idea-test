"""独立CUDA RNG验证；调用冻结后端原RNG方法，不进行神经前向或更新。"""
import argparse
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

FROZEN_ROOT = Path("/home/lzf/ldx/projects/idea-test-tbps-gh-probe-shared-20261023")
CONFIG = FROZEN_ROOT/"configs/framework/gh_bidirectional_backend.probe_v2.shared.server.json"
CONFIG_SHA = "77ae41af56189ece991ee7cf21fd0ea0f3be5645202bd1cb7cf7e0b2ae735649"
SOURCE_KEY = "06b5c6a15061e635e49a9e13deb14022956fdbc272ad0ed378676e57ca5f826b"
PROBE_MANIFEST = "d6e0b960e20a1ba5128f9e0b4edf497e7ea4b098fbf5cd3125799d2ad1136764"
RNG_OUTPUT = Path("/home/lzf/ldx/outputs/tbps-framework-gh-bidirectional-backend/cuda_rng_v1")

def check_sequences(ctx, draw, persist):
    """CPU Fixture与真实CUDA运行同一阶段切换流程。"""
    from tbps.alternating.state import restore_rng, capture_rng, clone_tree, tree_sha
    from tbps.alternating.native_backend import NativeBackend
    parent=clone_tree(ctx.rng)
    def restore(name):
        restore_rng(parent[name],ctx.sampler,allow_cuda=True)
        assert tree_sha(capture_rng(ctx.sampler))==tree_sha(parent[name])
    reference={}
    for name in ("G","H"):
        restore(name);first=draw();second=draw()
        restore(name);assert tree_sha(draw())==tree_sha(first)
        assert tree_sha(draw())==tree_sha(second)
        reference[name]=(first,second)
    for first,other in (("G","H"),("H","G")):
        ctx.rng=clone_tree(parent)
        other_before=tree_sha(ctx.rng[other])
        NativeBackend.enter_phase_rng(ctx,first)
        assert tree_sha(draw())==tree_sha(reference[first][0])
        NativeBackend.leave_phase_rng(ctx,first)
        first_after=tree_sha(ctx.rng[first])
        assert tree_sha(ctx.rng[other])==other_before
        NativeBackend.enter_phase_rng(ctx,other)
        assert tree_sha(draw())==tree_sha(reference[other][0])
        NativeBackend.leave_phase_rng(ctx,other)
        assert tree_sha(ctx.rng[first])==first_after
        NativeBackend.enter_phase_rng(ctx,first)
        assert tree_sha(draw())==tree_sha(reference[first][1])
        NativeBackend.leave_phase_rng(ctx,first)
    restored=persist(clone_tree(ctx.rng))
    assert tree_sha(restored)==tree_sha(ctx.rng)
    for name in ("G","H"):
        restore_rng(restored[name],ctx.sampler,allow_cuda=True)
        assert tree_sha(capture_rng(ctx.sampler))==tree_sha(restored[name])
        a=draw()
        restore_rng(restored[name],ctx.sampler,allow_cuda=True)
        assert tree_sha(draw())==tree_sha(a)
    return {"G_parent_RNG_sha256":tree_sha(parent["G"]),"H_parent_RNG_sha256":tree_sha(parent["H"]),
        "persisted_state_sha256":tree_sha(restored),"same_sequence_exact":True,
        "G_H_G_continuation_exact":True,"H_G_H_continuation_exact":True,
        "phase_state_isolation":True,"persist_read_restore_exact":True}

def verify_history():
    from tbps.artifacts import sha_file
    from tbps.alternating.preflight import metadata
    old=FROZEN_ROOT/"docs/framework/results/gh_probe_shared/CPU_RECEIPT_AUDIT.json"
    audit=json.loads(old.read_text())
    path=Path(audit["history_snapshot"]["path"])
    assert sha_file(path)==audit["history_snapshot"]["sha256"]
    records=json.loads(path.read_text())
    assert len(records)==151378
    bad=[p for p,m in records.items() if not Path(p).is_file() or metadata(p)!=m]
    assert not bad,str(bad[:5])
    index=Path(audit["old_FAIL_Watcher_index"]["path"])
    assert sha_file(index)==audit["old_FAIL_Watcher_index"]["sha256"]
    refs=json.loads(index.read_text())
    for p,ref in refs.items():
        assert metadata(p)==ref["metadata"] and sha_file(p)==ref["sha256"],p
    for name,ref in audit["parent_checkpoints"].items():
        assert sha_file(ref["path"])==ref["sha256"],name
    for p,ref in audit["artifact_files"].items():
        assert metadata(p)==ref["metadata"] and sha_file(p)==ref["sha256"],p
    return {"history_metadata_files":len(records),"old_FAIL_Watcher_SHA_metadata_files":len(refs),
        "probe_v2_SHA_metadata_unchanged":True,"parents_SHA_unchanged":True}

def execute():
    # 来源标准库预检必须先于torch导入。
    from tbps.alternating.native_cli import prepare_cpu_gate
    from tbps.alternating.native_sources import source_key,resource_protocol
    from tbps.alternating.native_gate import verify_probe
    from tbps.artifacts import ArtifactStore, FileRef, json_bytes,sha_file
    from tbps.alternating.preflight import resource_snapshot
    c,bundle,gate,pre=prepare_cpu_gate(CONFIG)
    assert sha_file(CONFIG)==CONFIG_SHA and source_key(c)==SOURCE_KEY
    assert verify_probe(c)==PROBE_MANIFEST
    assert not Path(c["smoke_output"]).exists(),"Smoke already consumed"
    history=verify_history()
    resource=resource_snapshot(resource_protocol(c,"smoke"))
    assert resource["resource_eligible"],"RESOURCE_BUSY"
    assert not RNG_OUTPUT.exists(),"CUDA RNG Attempt already consumed"
    store=ArtifactStore(RNG_OUTPUT,new_base=c["output_base"])
    operation_ref=FileRef(str(Path(__file__).resolve()),sha_file(Path(__file__)))
    store.publish("claim",{"preflight.json":json_bytes(pre),"resources.json":json_bytes(resource),
        "history_before.json":json_bytes(history)},{"source":SOURCE_KEY,"probe":PROBE_MANIFEST},sources=(operation_ref,))
    try:
        import random,time,resource as process_resource
        import numpy as np
        import torch
        from contextlib import ExitStack
        from tbps.alternating.parents import no_optimizer_steps
        from tbps.alternating.state import tree_sha,clone_tree,capture_rng,restore_rng
        before=tree_sha({k:bundle.state[k] for k in ("G","H","G_Adam","H_Adam","G_RNG","H_RNG")})
        for name in ("G_RNG","H_RNG"):
            assert len(bundle.state[name]["cuda"])==torch.cuda.device_count()==1
        assert not torch.cuda.is_initialized(),"CPU gate isolation not restored"
        start=time.monotonic()
        gate.initialize_cuda()
        torch.cuda.reset_peak_memory_stats()
        caller=capture_rng(bundle.sampler)
        counts={"CUDA_random_tensor_calls":0,"model_forwards":0,"JVPs":0,"optimizer_steps":0}
        ctx=SimpleNamespace(g=bundle.g,h=bundle.h,sampler=bundle.sampler,
            rng={"G":clone_tree(bundle.state["G_RNG"]),"H":clone_tree(bundle.state["H_RNG"])})
        def draw():
            counts["CUDA_random_tensor_calls"]+=1
            return {"python":[random.random() for _ in range(3)],"numpy":np.random.random(4),
                "torch_cpu":torch.rand(8),"sampler":torch.rand(8,generator=ctx.sampler),
                "cuda":torch.rand(8,device="cuda:0").cpu()}
        def persist(value):
            data=io.BytesIO();torch.save(value,data)
            store.publish("state_capture",{"rng.pt":data.getvalue()},
                {"source":SOURCE_KEY,"RNG_content_sha256":tree_sha(value)},sources=(operation_ref,))
            return torch.load(RNG_OUTPUT/"state_capture/rng.pt",map_location="cpu",weights_only=False)
        def forbidden(*a,**k):raise AssertionError("RNG validator forbids neural forward/derivatives")
        try:
            with no_optimizer_steps(),ExitStack() as stack:
                for owner,name in ((torch.nn.Module,"_call_impl"),(torch.autograd,"grad"),
                    (torch.autograd,"backward"),(torch.Tensor,"backward")):
                    stack.enter_context(patch.object(owner,name,forbidden))
                checks=check_sequences(ctx,draw,persist)
        finally:
            restore_rng(caller,bundle.sampler,allow_cuda=True)
        assert tree_sha({k:bundle.state[k] for k in ("G","H","G_Adam","H_Adam","G_RNG","H_RNG")})==before
        assert tree_sha(bundle.g.state_dict())==tree_sha(bundle.state["G"])
        assert tree_sha(bundle.h.state_dict())==tree_sha(bundle.state["H"])
        assert tree_sha(bundle.g_adam.state_dict())==tree_sha(bundle.state["G_Adam"])
        assert tree_sha(bundle.h_adam.state_dict())==tree_sha(bundle.state["H_Adam"])
        after=verify_history()
        assert verify_probe(c)==PROBE_MANIFEST
        result={"status":"CUDA_RNG_PASS","operation_source_sha256":operation_ref.sha256,"source":SOURCE_KEY,"config_sha256":CONFIG_SHA,
            "probe_manifest_sha256":PROBE_MANIFEST,"checks":checks,"counts":counts,
            "devices":torch.cuda.device_count(),"parent_CUDA_state_counts":{"G":1,"H":1},
            "G_H_Adam_unchanged":True,"caller_RNG_restored":True,
            "CPU_preflight":pre["parent_restore"],"history_after":after,
            "elapsed_seconds":time.monotonic()-start,
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes":torch.cuda.max_memory_reserved(),
            "peak_RSS_bytes":process_resource.getrusage(process_resource.RUSAGE_SELF).ru_maxrss*1024}
        store.publish("completed",{"report.json":json_bytes(result)},
            {"source":SOURCE_KEY,"probe":PROBE_MANIFEST},sources=(operation_ref,))
        print(json.dumps(result,ensure_ascii=False,indent=2));print("CUDA_RNG_PASS",flush=True)
    except BaseException:
        import traceback
        with (RNG_OUTPUT/"FAIL").open("x") as f:f.write(traceback.format_exc())
        raise

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--allow-cuda-rng-validation",action="store_true")
    args=ap.parse_args()
    if not args.allow_cuda_rng_validation:raise SystemExit("explicit RNG validation authorization required")
    execute()
