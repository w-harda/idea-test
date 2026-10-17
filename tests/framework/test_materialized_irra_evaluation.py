"""物化IRRA接线CPU测试；真实权重、CUDA、G/H推理均不执行。"""
from pathlib import Path
from dataclasses import asdict,replace
from types import SimpleNamespace
from unittest.mock import patch,Mock
from contextlib import ExitStack
import hashlib,json,io,os
import pytest
import torch
from tbps.artifacts import ArtifactStore,ArtifactError,FileRef,digest,json_bytes,sha_file,PendingPublication
from tbps.models.base import ModelSpace,InputBinding,RepresentationBatch
from tbps.engines.provenance import payload_digest
from tbps.engines.evaluation import EvaluationEngine
from tbps.engines import materialized_irra as runner
from tbps.engines.materialized_cache import FeatureCache,CachedIRRA
from tbps.materialization.bulk_session import StreamingMaterializedStore
from tbps.materialization.generation import generate_image_once,generate_text_once,seal_generated_collection
import test_stage4c_materialization as fixtures

ROOT=Path(__file__).resolve().parents[2]
def ref(p):return FileRef(str(p),sha_file(p))

@pytest.fixture
def world():
    w=fixtures.MaterializationTests();w.setUp()
    try:yield w
    finally:w.tearDown()

def cache_fixture(tmp_path):
    root=tmp_path/"tbps-framework-cpu-features";root.mkdir()
    ck=root/"weight.bin";ck.write_bytes(b"CPU fixture only")
    space=ModelSpace("irra_fixture",sha_file(ck),*(digest(i) for i in range(4)),"fp32/math/no_tf32")
    store=ArtifactStore(root/"features",new_base=root)
    protocol={"fixture":True};store.publish("protocol",{"protocol.json":json_bytes(protocol)},{"fixture":True})
    cache=FeatureCache(store,protocol,space,chunk=4)
    binding=InputBinding(space.id,digest("manifest"),tuple(digest(i) for i in range(10)),"text",tuple("q"+str(i) for i in range(10)))
    return cache,binding,space,ref(ck)

def features(n=10):
    return torch.nn.functional.normalize(torch.arange(1,n*512+1,dtype=torch.float32).reshape(n,512),dim=1)

def test_save_sliced_feature_storage_is_compact_and_values_exact(tmp_path):
    cache,b,space,ck=cache_fixture(tmp_path)
    x=features(50)
    cache.publish("clean_text",b,0,4,x[:4])
    value=cache.load("clean_text",b,0,4)
    assert torch.equal(value,x[:4])
    assert value.untyped_storage().nbytes()==4*512*4
    saved=torch.load(cache.store.root/"clean_text-000000/features.pt",map_location="cpu",weights_only=True)
    assert saved.untyped_storage().nbytes()==4*512*4

def test_chunk_gather_preserves_order_and_content_binding(tmp_path):
    cache,b,space,ck=cache_fixture(tmp_path);x=features()
    for start in range(0,10,4):cache.publish("clean_text",b,start,min(start+4,10),x[start:start+4])
    bound=cache.gather("clean_text",b)
    assert torch.equal(bound.batch.payload,x)
    assert bound.inputs==b and bound.payload_sha256==payload_digest(x)

@pytest.mark.parametrize("kind",["order","content","space","modality"])
def test_foreign_binding_rejected(tmp_path,kind):
    cache,b,space,ck=cache_fixture(tmp_path);cache.publish("clean_text",b,0,4,features(4))
    bad=replace(b,**({"ordered_ids":tuple(reversed(b.ordered_ids))} if kind=="order" else
        {"ordered_sha256":tuple(reversed(b.ordered_sha256))} if kind=="content" else
        {"model_space_id":digest("other Tokenizer")} if kind=="space" else {"modality":"image"}))
    with pytest.raises(ArtifactError):cache.load("clean_text",bad,0,4)

def test_tampered_feature_rejected(tmp_path):
    cache,b,space,ck=cache_fixture(tmp_path);cache.publish("clean_text",b,0,4,features(4))
    (cache.store.root/"clean_text-000000/features.pt").write_bytes(b"damaged")
    with pytest.raises(ArtifactError):cache.load("clean_text",b,0,4)

def test_incomplete_chunk_preserved_and_no_overwrite(tmp_path):
    cache,b,space,ck=cache_fixture(tmp_path)
    def fault(stage):
        if stage=="before_receipt":raise InterruptedError("CPU fixture interruption")
    with pytest.raises(InterruptedError):
        cache.store.publish(cache.key("clean_text",0),{"features.pt":b"incomplete"},
                            cache.provenance(b,0,4),fault=fault)
    assert cache.store.pending()
    with pytest.raises(PendingPublication):cache.publish("clean_text",b,0,4,features(4))

@pytest.mark.parametrize("kind",["nan","dtype","norm","dimension"])
def test_invalid_features_rejected(tmp_path,kind):
    cache,b,space,ck=cache_fixture(tmp_path);x=features(4)
    if kind=="nan":x[0,0]=float("nan")
    elif kind=="dtype":x=x.double()
    elif kind=="norm":x=x*2
    else:x=x[:,:511]
    with pytest.raises(ArtifactError):cache.publish("clean_text",b,0,4,x)

def test_native_scorer_and_chunking_exact(tmp_path):
    cache,b,space,ck=cache_fixture(tmp_path);scorer=CachedIRRA(space,ck,fixture=True)
    q=RepresentationBatch(features(3),space.id,"text");g=RepresentationBatch(features(12),space.id,"image")
    with torch.inference_mode():
        actual=scorer.score(q,g).payload
        expected=q.payload@g.payload.T
    assert torch.equal(actual,expected)
    with pytest.raises(ArtifactError):scorer.score(q,replace(g,space_id=digest("RaSa")))
    with pytest.raises(ArtifactError):CachedIRRA(space,ck)

def test_smoke_fixed_positive_coverage_and_all_queries_full(world):
    full=world.manifest;smoke=runner.select(full,"smoke")
    assert len(smoke.gallery)==len(smoke.queries)==10
    assert all(q.paired_image_id in {g.image_id for g in smoke.gallery} for q in smoke.queries)
    assert runner.select(full,"full") is full
    assert len(full.gallery)!=len(full.queries)

def test_full_authorization_and_default_CPU_are_fail_closed():
    with patch.object(runner,"preflight",return_value={"report":{"status":"CPU fixture"}}) as pre, \
         patch.object(runner,"run",side_effect=AssertionError("GPU forbidden")):
        assert runner.main(["--config","fixture"])==0
        pre.assert_called_once_with("fixture","full")
        for flags in [[],["--allow-smoke"],["--allow-full-evaluation","--allow-smoke"]]:
            with pytest.raises(ArtifactError):runner.main(["--config","fixture","--action","run",*flags])
    with patch.object(runner,"preflight",side_effect=AssertionError("must reject before preflight")):
        with pytest.raises(ArtifactError):runner.run("fixture","full")

def test_clean_check_uses_fixed_historical_tolerance_without_metric_fallback():
    clean={k:50. for k in ("mAP","mINP","R@1","R@5","R@10")}
    assert runner.clean_check(clean,clean)==dict.fromkeys(clean,0.)
    with pytest.raises(ArtifactError):runner.clean_check({**clean,"mAP":49.998},clean)
    with pytest.raises(ArtifactError):runner.clean_check({**clean,"mAP":float("nan")},clean)

def test_full_config_SHA_scope_and_no_real_model_load():
    p=ROOT/"configs/framework/materialized_irra.server.json"
    with patch.object(torch,"load",side_effect=AssertionError("real torch.load forbidden")), \
         patch.object(torch.cuda,"init",side_effect=AssertionError("CUDA forbidden")), \
         patch.object(torch.cuda,"_lazy_init",side_effect=AssertionError("CUDA forbidden")), \
         patch.object(torch.nn.Module,"_call_impl",side_effect=AssertionError("neural forward forbidden")):
        c=runner.config(p)
    assert (c["gallery"],c["query"],c["image_batch"],c["text_batch"])==(3074,6156,1,1)
    assert c["plan_id"]==runner.PLAN and c["clean_metric_atol_pp"]==.001
    assert "generation_v3" in c["generation_result"]["path"]

class Backend(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.ones(1),requires_grad=False)
        self.eval()
    def encode_image(self,pixels):
        assert pixels.device.type=="cpu" and float(pixels.min())>=0 and float(pixels.max())<=1
        x=torch.arange(1,513,dtype=torch.float32)[None,:].repeat(len(pixels),1)
        x[:,0]+=pixels.flatten(1).mean(1)
        return torch.nn.functional.normalize(x,dim=1)
    def encode_text(self,texts):
        assert all(type(t)is str for t in texts) # PID、配对与奖励不能进入文本编码/攻击输出读取。
        x=torch.arange(1,513,dtype=torch.float32)[None,:].repeat(len(texts),1)
        x[:,0]+=torch.tensor([sum(map(ord,t))%100 for t in texts])
        return torch.nn.functional.normalize(x,dim=1)

class View:
    def __init__(self,b,space):self.b,self.space=b,space
    def encode_images(self,x):return RepresentationBatch(self.b.encode_image(x),self.space.id,"image")
    def encode_texts(self,x):return RepresentationBatch(self.b.encode_text(x),self.space.id,"text"),()

def assembled(world):
    w=world;images=[];texts=[];pairs={}
    for i in range(len(w.manifest.gallery)):
        images.append(generate_image_once(w.store,"g"+str(i),w.gp,lambda _:w.gb.pixels.clone()))
    for i,q in enumerate(w.manifest.queries):
        u=generate_text_once(w.store,q.query_id,w.hp,w.validate);texts.append(u)
        adv,md=w.store.load(u,validate=w.validate);pairs[q.query_id]=(q.caption,adv)
    core=seal_generated_collection(w.store,"joint",images=tuple(images),texts=tuple(texts),validate=w.validate)
    w.store.store.publish("joint-full",{"joint_collection.json":json_bytes({
        "status":"JOINT_FULL_MATERIALIZED","plan_id":w.plan.id,"victim_model":None,"collection":asdict(core)})},{"fixture":True})
    manifest=runner.select(w.manifest,"smoke")
    stream=StreamingMaterializedStore(w.store.store,w.plan)
    ck=w.root/"IRRA-fixture";ck.write_bytes(b"not real weights")
    space=ModelSpace("irra_fixture",sha_file(ck),*(digest(i) for i in range(4)),"fp32/math/no_tf32")
    ctx=dict(c={"output_base":str(w.root/"eval"/"tbps-framework-eval"),"cache_chunk":4},
        store=stream,manifest=manifest,plan=w.plan,legacy=None,
        images={u.binding.sample_id:u for u in images},text_units={u.binding.sample_id:u for u in texts},
        texts=pairs,core=core,gate=Mock(),source=SimpleNamespace(checkpoint=ref(ck)),
        report={"scope":"smoke","status":"CPU fixture"})
    cfile=w.root/"config.json";cfile.write_bytes(b'{"fixture":true}')
    gen=w.root/"sealed.json";gen.write_bytes(b"sealed CPU fixture")
    ctx["c"].update(generation_result=asdict(ref(gen)),materialization_audit=asdict(ref(ck)),entry_sources={})
    return ctx,space,cfile

def test_collection_full_order_and_missing_member_rejection(world):
    ctx,_,_=assembled(world)
    images,texts,core=runner.collection(ctx["store"])
    assert len(images)==12 and len(texts)==24
    old=ctx["store"].plan
    ctx["store"].plan=replace(old,manifest=replace(old.manifest,gallery=tuple(reversed(old.manifest.gallery))))
    with pytest.raises(ArtifactError):runner.collection(ctx["store"])

def test_actual_runner_CPU_mock_four_modes_counters_NOOP_and_readback(world):
    ctx,space,cfile=assembled(world);backend=Backend()
    adapter=SimpleNamespace(space=space,_backend=backend,inference=lambda:View(backend,space))
    original_to=torch.Tensor.to;engine=EvaluationEngine
    def to_cpu(value,*a,**kw):
        return value if a and a[0]=="cuda:0" else original_to(value,*a,**kw)
    def fixture_engine(manifest,scorer,**kw):
        kw["execution"]="cpu_fixture";return engine(manifest,scorer,**kw)
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":"0"}))
        stack.enter_context(patch.object(runner,"preflight",return_value=ctx))
        stack.enter_context(patch.object(runner,"gpu_available",return_value={"CPU fixture":True}))
        stack.enter_context(patch.object(runner.IRRAAdapter,"from_source",return_value=adapter))
        stack.enter_context(patch.object(runner,"CachedIRRA",side_effect=lambda s,c,**kw:CachedIRRA(s,c,fixture=True)))
        stack.enter_context(patch.object(runner,"EvaluationEngine",side_effect=fixture_engine))
        stack.enter_context(patch.object(torch.Tensor,"to",new=to_cpu))
        stack.enter_context(patch.object(torch.cuda,"max_memory_allocated",return_value=0))
        stack.enter_context(patch.object(torch.cuda,"max_memory_reserved",return_value=0))
        stack.enter_context(patch.object(torch.cuda,"empty_cache"))
        for name in ("init","_lazy_init"):
            stack.enter_context(patch.object(torch.cuda,name,side_effect=AssertionError("real CUDA forbidden")))
        assert runner.run(str(cfile),"smoke",authorized=True)==0
        # 已发布Result可只读复用，不重新评分、不重新调用模型或写旧缓存。
        from tbps.artifacts import read_json
        root=Path(ctx["c"]["output_base"])/"smoke"
        protocol=read_json(root/"protocol/protocol.json")
        cache=FeatureCache(ArtifactStore(root,new_base=ctx["c"]["output_base"]),protocol,space,chunk=4)
        with patch.object(EvaluationEngine,"run",side_effect=AssertionError("must not rerank committed result")):
            reused,calls=runner.finish(cache,ctx,space,ctx["source"].checkpoint)
            assert calls==0 and set(reused["groups"])==set(runner.MODES)
        with patch.object(runner,"config",return_value=ctx["c"]), \
             patch.object(EvaluationEngine,"run",side_effect=AssertionError("audit cannot rank")):
            audited=runner.audit_outputs(str(cfile),"smoke")
            assert audited==reused
    root=Path(ctx["c"]["output_base"])/"smoke"
    report=json.loads((root/"execution-000001/report.json").read_text())
    assert report["counts"]["IRRA_image_completed"]==20
    assert report["counts"]["IRRA_text_completed"]==20
    assert report["counts"]["G_forwards"]==report["counts"]["H_forwards"]==0
    assert report["model_before"]==report["model_after"]
    summary=json.loads((root/"completed/summary.json").read_text())
    assert set(summary["groups"])==set(runner.MODES)
    assert summary["gallery_count"]==summary["query_count"]==10
    # 独立加载官方rank，而非对照Engine自身；仅10x10 CPU Fixture。
    import importlib.util
    path="/home/lzf/Attack/IRRA-main/IRRA-main/utils/metrics.py"
    spec=importlib.util.spec_from_file_location("_materialized_fixture_official_rank",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    m=ctx["manifest"];gids=torch.tensor([int(g.pid.value) for g in m.gallery])
    qids=torch.tensor([int(q.pid.value) for q in m.queries])
    for mode in runner.MODES:
        scores=torch.load(root/"result/scores"/(mode+".pt"),map_location="cpu",weights_only=True)
        cmc,ap,inp,order=module.rank(scores,qids,gids,max_rank=10,get_mAP=True)
        expected={"mAP":float(ap),"mINP":float(inp),"R@1":float(cmc[0]),"R@5":float(cmc[4]),"R@10":float(cmc[9])}
        assert summary["groups"][mode]==expected
    assert not torch.cuda.is_initialized()
    assert ctx["gate"].initialize_cuda.call_count==1


def test_shared_GPU_state_records_Ollama_without_exclusive_or_free_memory_gate():
    device=Mock(stdout="NVIDIA GeForce RTX 4090, 24564, 3865, 20226, 0\n")
    processes=Mock(stdout="2436556, /usr/local/lib/ollama/llama-server, 20178\n")
    with patch.object(runner.subprocess,"run",side_effect=[device,processes]) as probe:
        result=runner.gpu_available({"physical_gpu":0})
    assert result["free_mib"]==3865 and "2436556" in result["compute_processes"]
    assert result["policy"]=="shared_observe_only" and probe.call_count==2

def test_shared_GPU_low_free_memory_is_observed_not_preemptively_refused():
    with patch.object(runner.subprocess,"run",side_effect=[
        Mock(stdout="NVIDIA GeForce RTX 4090, 24564, 1, 24563, 95\n"),
        Mock(stdout="2436556, llama-server, 20178\n")]):
        assert runner.gpu_available({"physical_gpu":0})["free_mib"]==1

def test_CUDA_OOM_saved_printed_and_not_retried(world,capsys):
    ctx,_,cfile=assembled(world)
    with patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":"0"}), \
         patch.object(runner,"preflight",return_value=ctx), \
         patch.object(runner,"gpu_available",return_value={"fixture":True}), \
         patch.object(runner.IRRAAdapter,"from_source",side_effect=torch.cuda.OutOfMemoryError("synthetic CPU OOM test")) as load:
        assert runner.run(str(cfile),"smoke",authorized=True)==1
    root=Path(ctx["c"]["output_base"])/"smoke"
    report=json.loads((root/"execution-000001/report.json").read_text())
    assert load.call_count==1 and report["CUDA_OOM"] is True
    assert report["exception_type"]=="OutOfMemoryError" and "synthetic CPU OOM test" in report["traceback"]
    assert report["counts"]["IRRA_load_attempts"]==1 and report["counts"]["IRRA_loads"]==0
    assert report["counts"]["IRRA_image_started"]==report["counts"]["IRRA_text_started"]==0
    assert not (root/"completed").exists()
    assert "synthetic CPU OOM test" in capsys.readouterr().err
