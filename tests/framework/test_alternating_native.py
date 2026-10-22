"""生产接线的CPU控制流测试；合成模型绝不冒充真实GPU数值验证。"""
import copy
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from tbps.artifacts import ArtifactStore, digest, json_bytes, sha_file
from tbps.alternating.contracts import require
from tbps.alternating.native_gate import ExecutionGrant, authorize, verify_probe
from tbps.alternating.native_sources import native_session, source_key
from tbps.alternating.parents import no_optimizer_steps, cpu_restore_only, normalized_rng
from tbps.alternating.state import tree_sha, capture_rng
from tbps.alternating.native_numeric import compare_gradients
from tbps.alternating.native_backend import assert_pixels, TextRuntime

@pytest.fixture
def c(tmp_path):
    root=tmp_path/"tbps-framework-native-fixture"
    return {"output_base":str(root),"probe_output":str(root/"probe_v1"),
            "smoke_output":str(root/"smoke_v1"),"prototype":{"fixture":True},
            "probe":{"noop_gradient_atol":1e-6,"max_IRRA_texts":6}}

def publish_probe(c,**overrides):
    result={"status":"GH_NATIVE_NUMERIC_PROBE_PASS","optimizer_steps":0,"full_gallery_generations":0,
            "frozen_sources":True,"nonnoop_gradient_max_abs":.1,"noop_gradient_max_abs":0,
            "counts":{"G":4,"IRRA_image":4,"JVP":1,"reader":1,"H":1,"IRRA_text":6},
            "original_text_cache_max_abs":0}
    result.update(overrides)
    ArtifactStore(c["probe_output"],new_base=c["output_base"]).publish("completed",
        {"report.json":json_bytes(result)},{"source":source_key(c),"protocol":digest(c["prototype"])})

@pytest.mark.parametrize("action,flags",[("probe",{}),("smoke",{}),("probe",{"allow_smoke":True}),
                                         ("smoke",{"allow_probe":True}),
                                         ("probe",{"allow_probe":True,"allow_smoke":True})])
def test_no_implicit_gpu_grant(c,action,flags):
    with pytest.raises(ValueError):authorize(c,action,**flags)
    assert not Path(c["output_base"]).exists()

def test_probe_grant_no_output_creation(c):
    grant=authorize(c,"probe",allow_probe=True)
    assert grant.validate(c,"probe") is grant
    assert not Path(c["output_base"]).exists()

def test_smoke_needs_committed_probe(c):
    with pytest.raises(ValueError):authorize(c,"smoke",allow_smoke=True)
    publish_probe(c)
    assert authorize(c,"smoke",allow_smoke=True).validate(c,"smoke")

@pytest.mark.parametrize("field,value",[("status","FAIL"),("optimizer_steps",1),
    ("full_gallery_generations",1),("frozen_sources",False),
    ("nonnoop_gradient_max_abs",0),("noop_gradient_max_abs",.01)])
def test_inadequate_probe_rejected(c,field,value):
    publish_probe(c,**{field:value})
    with pytest.raises(ValueError):verify_probe(c)

def test_probe_tamper_and_foreign_sources(c):
    publish_probe(c)
    p=Path(c["probe_output"])/"completed/report.json"
    p.write_text("{}")
    with pytest.raises(ValueError):verify_probe(c)

def test_changed_config_rejects_probe(c):
    publish_probe(c)
    c["prototype"]["fixture"]=False
    with pytest.raises(ValueError):verify_probe(c)

@pytest.mark.parametrize("action",["probe","smoke"])
def test_consumed_attempt_rejected(c,action):
    Path(c[action+"_output"]).mkdir(parents=True)
    with pytest.raises(ValueError):authorize(c,action,**{"allow_"+action:True})

def test_probe_fail_evidence_blocks_smoke(c):
    publish_probe(c);(Path(c["probe_output"])/"FAIL").write_text("retained")
    with pytest.raises(ValueError):verify_probe(c)

@pytest.mark.parametrize("optimizer_cls",[torch.optim.Adam,torch.optim.SGD,torch.optim.AdamW])
def test_probe_forbids_all_optimizer_steps(optimizer_cls):
    p=torch.nn.Parameter(torch.ones(1));opt=optimizer_cls([p],lr=.1)
    old=optimizer_cls.step
    with no_optimizer_steps():
        with pytest.raises(ValueError):opt.step()
    assert optimizer_cls.step is old
    assert torch.equal(p,torch.ones(1))

@pytest.mark.parametrize("operation",["forward","load","backward","grad","cuda"])
def test_cpu_restore_forbids_operations(operation,tmp_path):
    original=torch.nn.Module._call_impl
    with cpu_restore_only(()) as evidence:
        with pytest.raises(ValueError):
            if operation=="forward":torch.nn.Linear(1,1)(torch.ones(1))
            elif operation=="load":torch.load(tmp_path/"not-a-parent")
            elif operation=="backward":torch.ones(1,requires_grad=True).backward()
            elif operation=="grad":torch.autograd.grad(torch.ones(1,requires_grad=True),[])
            else:torch.cuda.init()
        assert evidence["loads"]==[]
    assert torch.nn.Module._call_impl is original and not torch.cuda.is_initialized()

def test_cpu_restore_exact_whitelist(tmp_path):
    p=tmp_path/"parent.pt";torch.save({"w":torch.arange(3)},p)
    with cpu_restore_only([p]) as evidence:
        assert torch.equal(torch.load(p,map_location="cpu",weights_only=True)["w"],torch.arange(3))
    assert evidence["loads"]==[str(p)]

def native_fixture(tmp_path,body="VALUE=1"):
    root=tmp_path/"runtime";root.mkdir()
    p=root/"gh_native_fixture_module.py";p.write_text(body)
    return {"managed_roots":[str(root)],"runtime_pins":{str(p):sha_file(p)},
            "producers":{},"import_paths":[str(root)]},p

def test_import_restore_and_same_class_reuse(tmp_path):
    c,p=native_fixture(tmp_path,"class Example: pass")
    before=list(sys.path)
    with native_session(c) as session:
        cls=session.import_module("gh_native_fixture_module").Example
    assert list(sys.path)==before and "gh_native_fixture_module" not in sys.modules
    with native_session(c,reuse=session.modules) as again:
        assert again.import_module("gh_native_fixture_module").Example is cls
    assert "gh_native_fixture_module" not in sys.modules

@pytest.mark.parametrize("failure",["tamper","missing","exception","nested"])
def test_import_failures_restore_namespace_and_lock(tmp_path,failure):
    c,p=native_fixture(tmp_path)
    before=list(sys.path)
    if failure=="tamper":p.write_text("VALUE=2")
    if failure=="missing":c["runtime_pins"]={}
    with pytest.raises(ValueError):
        with native_session(c) as s:
            if failure=="exception":raise ValueError("test")
            if failure=="nested":
                with native_session(c):pass
            s.import_module("gh_native_fixture_module")
    assert list(sys.path)==before and "gh_native_fixture_module" not in sys.modules
    # 失败后锁必须释放；未触发导入无需消费损坏来源。
    with native_session(c):pass

class TinyG(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.001))
    def forward(self,x):return self.weight*torch.ones_like(x)

def numeric_fixture():
    def attr(feature,z,local):
        return {"global":(feature*z).sum(),"local":(feature*local).sum(),"retain":torch.tensor(0.)}
    losses=SimpleNamespace(attribute_losses=attr)
    losses.joint_losses=lambda f,z,v,has_local:{"global":(f*z).sum(),"local":(f*v).sum()}
    serial=SimpleNamespace(LossWeights=lambda:None,
                           _weighted=lambda terms,w,joint:sum(terms.values()))
    modules={"budget_projection":SimpleNamespace(add_projected_delta=lambda o,x,d:x+d),
             "losses":losses,"serial_training":serial}
    return SimpleNamespace(get=modules.__getitem__)

def test_real_probe_control_flow_no_step_and_gradient_controls():
    g=TinyG()
    encoder=lambda x:torch.stack((x.mean(),x.square().mean())).reshape(1,2)
    def payload(z):return {"K":1,"z_t":torch.tensor(z),"attribute_jvps":torch.tensor([[1.,.2]]),
                           "v_local":torch.zeros(2),"has_local":True}
    a=SimpleNamespace(payload=payload([1.,0.]),evidence={})
    b=SimpleNamespace(payload=payload([0.,1.]),evidence={"fresh":True})
    rows=[{"noop":True,"original":a,"image":torch.full((1,3,2,2),.4)},
          {"noop":False,"original":a,"image":torch.full((1,3,2,2),.4)}]
    before=tree_sha(g.state_dict())
    result=compare_gradients(g,encoder,rows,numeric_fixture(),lambda row:a if row["noop"] else b)
    assert result["noop"]["max_abs"]==0 and result["nonnoop"]["max_abs"]>0
    assert tree_sha(g.state_dict())==before and g.weight.grad is None

def test_probe_rejects_fake_noop_condition():
    g=TinyG();a=SimpleNamespace(payload={"K":1,"z_t":torch.ones(2)},evidence={})
    b=SimpleNamespace(payload={"K":1,"z_t":torch.zeros(2)},evidence={})
    rows=[{"noop":True,"original":a,"image":torch.zeros(1)},{"noop":False,"original":a,"image":torch.zeros(1)}]
    with pytest.raises(ValueError):compare_gradients(g,lambda x:x,rows,numeric_fixture(),lambda r:b)

@pytest.mark.parametrize("fault",["range","budget","delta","dtype","shape","nan"])
def test_real_bank_pixel_guard(fault):
    image=torch.full((1,3,384,128),.5);adv=image.clone();delta=adv-image
    if fault=="range":adv.fill_(1.1);delta=adv-image
    if fault=="budget":adv+=.04;delta=adv-image
    if fault=="delta":delta+=.01
    if fault=="dtype":adv=adv.double()
    if fault=="shape":adv=adv[:,:,:10]
    if fault=="nan":adv[0,0,0,0]=float("nan")
    with pytest.raises(ValueError):assert_pixels(image,adv,delta)

def test_valid_bank_pixel_guard():
    image=torch.full((1,3,384,128),.5);adv=image+.01
    assert_pixels(image,adv,adv-image)

def test_reward_runtime_separates_jvp_image_counters():
    class Backend:
        def __init__(self):
            self.bundle=SimpleNamespace(probe="probe")
            self.victim=SimpleNamespace(encode_texts=lambda texts:torch.nn.functional.normalize(torch.ones(len(texts),512),dim=1))
        def assert_frozen_sources(self):pass
    runtime=TextRuntime(Backend())
    assert runtime.encode_text(["original","sampled","greedy"]).shape==(3,512)
    assert runtime.samples=={"G":0,"clean":0,"adv":0,"text":3}

def test_complete_parent_rng_normalization():
    r=capture_rng(torch.Generator())
    saved=torch.arange(10,dtype=torch.uint8)
    g={"python":r["python"],"numpy":r["numpy"],"torch":r["torch_cpu"],"cuda":[saved]}
    h={"python":r["python"],"numpy":r["numpy"],"torch_cpu":r["torch_cpu"],
       "sample_generator":r["sampler"],"torch_cuda":[saved]}
    gr,hr=normalized_rng(g,h)
    assert tree_sha(gr)==tree_sha(hr)
    assert gr["cuda"][0].data_ptr()!=saved.data_ptr() and not torch.cuda.is_initialized()

@pytest.mark.parametrize("side",["G","H"])
def test_missing_parent_rng_rejected(side):
    with pytest.raises(ValueError):normalized_rng({},{})

def test_no_import_gpu_side_effects():
    assert not torch.cuda.is_initialized()
    import tbps.alternating.native_cli
    import tbps.alternating.native_backend
    import tbps.alternating.native_numeric
    assert not torch.cuda.is_initialized()


def test_byte_bridge_rejects_non_byte_mapping_problem():
    from tbps.alternating.token_bridge import ByteSpanBridge
    tokenizer=object()
    native=SimpleNamespace(align_irra_tokens=lambda c,t:{"tokenizer_encode_matches_replay":True,
        "truncated":False,"reasons":["cleaning_ambiguous"]})
    with pytest.raises(ValueError):ByteSpanBridge(native,tokenizer).align("blue",tokenizer)


def test_device_binding_preserves_adam_and_owns_replaced_parameters():
    from tbps.alternating.parents import bind_g_device
    g=torch.nn.Linear(1,1);a=torch.optim.Adam(g.parameters(),lr=2e-4,betas=(.5,.999))
    g(torch.ones(1)).sum().backward();a.step();a.zero_grad(set_to_none=True)
    from tbps.alternating.state import clone_tree
    state={"G":clone_tree(g.state_dict()),"G_Adam":clone_tree(a.state_dict())}
    # 在Fixture中模拟设备迁移替换Parameter；绝不触碰真实CUDA。
    g.weight=torch.nn.Parameter(g.weight.detach().clone())
    g.bias=torch.nn.Parameter(g.bias.detach().clone())
    bundle=SimpleNamespace(g=g,state=state)
    restored,adam=bind_g_device(bundle,"cpu")
    assert tree_sha(adam.state_dict())==tree_sha(state["G_Adam"])
    assert [id(p) for group in adam.param_groups for p in group["params"]]==[id(p) for p in restored.parameters()]
    assert [id(p) for group in a.param_groups for p in group["params"]]!=[id(p) for p in restored.parameters()]


def test_native_image_normalization_buffers_follow_model_device_without_cuda():
    from tbps.alternating.native_backend import native_image_view
    model=torch.nn.Module();model.register_parameter("weight",torch.nn.Parameter(torch.empty(1,device="meta")))
    class Wrapper(torch.nn.Module):
        def __init__(self,model):
            super().__init__();self.model=model
            self.register_buffer("mean",torch.ones(3));self.register_buffer("std",torch.ones(3))
        def forward(self,x):raise AssertionError("no neural forward in device-contract check")
    native=SimpleNamespace(get=lambda name:SimpleNamespace(FrozenIRRAImageEncoder=Wrapper))
    wrapper=native_image_view(native,model)
    assert wrapper.mean.device==wrapper.std.device==model.weight.device==torch.device("meta")
    assert not torch.cuda.is_initialized()
