"""仅CPU：B3像素桥、来源、原子发布、原生评分复用与权限门禁。"""
from pathlib import Path
from dataclasses import asdict,replace
from types import SimpleNamespace
from unittest.mock import patch,Mock
import json,io
import pytest
import torch
from PIL import Image
from tbps.baselines.ap_b3 import validate_native,B3Pixels,HistoricalB3
from tbps.baselines import comparison as c
from tbps.artifacts import ArtifactError,ArtifactStore,FileRef,sha_file,digest,json_bytes
from tbps.engines.materialized_cache import ref
from tbps.engines.materialized_irra import select
from tbps.materialization.store import image_checks
import test_stage4c_materialization as fixtures
from test_materialized_irra_evaluation import cache_fixture,features

@pytest.fixture
def world():
    w=fixtures.MaterializationTests();w.setUp()
    try:yield w
    finally:w.tearDown()

def native():
    rows=[SimpleNamespace(image_rel_path="a.jpg",image_sha256=digest("raw"))]
    return dict(start=0,paths=["a.jpg"],raw_sha256=[rows[0].image_sha256],
                adv_pixels=torch.full((1,3,256,128),.5)),dict(start=0,count=1),rows

def test_native_positive():
    validate_native(*native())

@pytest.mark.parametrize("kind",["start","paths","raw","shape","dtype","range","nan","extra"])
def test_native_reject(kind):
    p,part,rows=native()
    if kind=="start":p["start"]=1
    elif kind=="paths":p["paths"]=["b.jpg"]
    elif kind=="raw":p["raw_sha256"]=[digest("other")]
    elif kind=="shape":p["adv_pixels"]=p["adv_pixels"][:,:,:128]
    elif kind=="dtype":p["adv_pixels"]=p["adv_pixels"].double()
    elif kind=="range":p["adv_pixels"][0,0,0,0]=1.1
    elif kind=="nan":p["adv_pixels"][0,0,0,0]=float("nan")
    else:p["pid"]=0
    with pytest.raises(ValueError):validate_native(p,part,rows)

def setup_pixels(tmp_path,row):
    source=tmp_path/"native.bin";source.write_bytes(b"CPU native fixture")
    root=tmp_path/"tbps-framework-b3"/"smoke"/"pixels"
    store=B3Pixels(root,{"source":asdict(ref(source))});store.publish_protocol((ref(source),))
    clean=torch.full((1,3,384,128),.5);adv=clean+.01
    return store,dict(original=clean,adversarial=adv,effective_delta=adv-clean),ref(source)

def test_roundtrip_compact_exact_immutable(tmp_path,world):
    row=world.plan.manifest.gallery[0];store,x,source=setup_pixels(tmp_path,row)
    store.publish(row,x,source)
    got,md=store.load(row)
    for k in x:
        assert torch.equal(got[k],x[k])
        assert got[k].untyped_storage().nbytes()==got[k].numel()*4
    assert md["B3_new_forwards"]==0 and md["target_feedback"] is False
    with pytest.raises(ArtifactError):store.publish(row,x,source)

@pytest.mark.parametrize("kind",["budget","delta","range","dtype","shape"])
def test_pixel_contract_rejects(tmp_path,world,kind):
    row=world.plan.manifest.gallery[0];store,x,source=setup_pixels(tmp_path,row)
    if kind=="budget":x["adversarial"]+=.05;x["effective_delta"]=x["adversarial"]-x["original"]
    elif kind=="delta":x["effective_delta"]*=0
    elif kind=="range":x["adversarial"]+=1;x["effective_delta"]=x["adversarial"]-x["original"]
    elif kind=="dtype":x["adversarial"]=x["adversarial"].double()
    else:x["adversarial"]=x["adversarial"][:,:,:12]
    with pytest.raises(ArtifactError):store.publish(row,x,source)

def test_tampered_native_source_refused(tmp_path,world):
    row=world.plan.manifest.gallery[0];store,x,source=setup_pixels(tmp_path,row)
    store.publish(row,x,source);Path(source.path).write_bytes(b"tampered")
    with pytest.raises(ArtifactError):store.load(row)

def test_wrong_row_refused(tmp_path,world):
    row=world.plan.manifest.gallery[0];store,x,source=setup_pixels(tmp_path,row);store.publish(row,x,source)
    with pytest.raises(ValueError):store.load(replace(row,image_sha256=digest("wrong")))

def test_interrupted_publication_preserves_pending(tmp_path,world):
    row=world.plan.manifest.gallery[0];store,x,source=setup_pixels(tmp_path,row)
    def fault(stage):
        if stage=="before_receipt":raise InterruptedError
    with pytest.raises(InterruptedError):
        store.store.publish(store.key(row.image_id),{"pixels.pt":b"partial"},{"fixture":True},fault=fault)
    assert store.store.pending()
    with pytest.raises(ArtifactError):store.publish(row,x,source)

def test_structured_row_selection_clone():
    old={"global":torch.rand(100,8),"tokens":torch.rand(100,7,5)}
    got=c.payload_select(old,[1,7])
    for k in got:
        assert torch.equal(got[k],old[k][[1,7]])
        assert got[k].untyped_storage().nbytes()==got[k].numel()*4
    back=c.payload_cat([c.payload_select(old,[i]) for i in [1,7]])
    assert all(torch.equal(got[k],back[k]) for k in got)

def test_reference_bridge_uses_cache_order_and_sha(tmp_path,world):
    cache,b,space,ck=cache_fixture(tmp_path)
    for start in range(0,10,4):cache.publish("clean_text",b,start,min(start+4,10),features()[start:start+4])
    bridge=c.FullReference.__new__(c.FullReference)
    bridge.bound={"clean_text":b};bridge.space=space;bridge.cache=cache;bridge.root=cache.store.root
    queries=tuple(SimpleNamespace(query_id=i) for i in b.ordered_ids[:3])
    manifest=SimpleNamespace(queries=queries)
    with patch.object(c,"asdict",return_value={"fixture":True}):
        result=bridge.read("clean_text",{"manifest":manifest})
    assert torch.equal(result.batch.payload,features()[:3])
    assert result.inputs.ordered_ids==b.ordered_ids[:3]

def test_four_columns_order_precision_only():
    d={k:33.3333333 for k in c.KEYS}
    groups={method:d for method in c.METHODS}
    output=c.table_bytes(groups,True).decode()
    assert output.splitlines()[0]=="Attack Method,R@1,R@5,R@10,mAP"
    assert [line.split(",")[0] for line in output.splitlines()[1:]]==list(c.METHODS)
    assert "33.33" in output and "DR" not in output and "mINP" not in output
    assert c.four({**d,"mINP":12})==d
    with pytest.raises(ValueError):c.table_bytes(dict(reversed(list(groups.items()))))

@pytest.mark.parametrize("kind",["missing","nan","negative","overflow"])
def test_bad_metrics(kind):
    d={k:50. for k in c.KEYS}
    if kind=="missing":del d["mAP"]
    else:d["mAP"]={"nan":float("nan"),"negative":-1.,"overflow":101.}[kind]
    with pytest.raises(ValueError):c.four(d)

@pytest.mark.parametrize("scope",["smoke","full"])
def test_unapproved_run_stops_before_preflight(scope):
    with patch.object(c,"preflight") as before,pytest.raises(ValueError):
        c.run("missing",scope,"irra")
    before.assert_not_called()

def test_full_cli_needs_explicit_authorization():
    with patch.object(c,"preflight") as before,pytest.raises(ValueError):
        c.main(["--config","missing","--action","run","--scope","full","--model","irra","--allow-smoke"])
    before.assert_not_called()

def test_preflight_flags_rejected_before_loading():
    with patch.object(c,"preflight") as before,pytest.raises(ValueError):
        c.main(["--config","missing","--action","preflight","--allow-smoke"])
    before.assert_not_called()

def test_official_transport_reused_without_model_or_cuda():
    import importlib.util
    original=Path(__file__).resolve().parents[2]/"src/attributes/ap_gallery_baseline.py"
    spec=importlib.util.spec_from_file_location("_b3_cpu_test_pixels",original)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    im=Image.new("RGB",(15,21),(100,120,130))
    with patch.object(torch,"load",side_effect=AssertionError("weights forbidden")),patch.object(torch.cuda,"_lazy_init",side_effect=AssertionError("CUDA forbidden")):
        clean=module.native_pixels(im).unsqueeze(0)
        assert clean.shape==(1,3,256,128)
        base=torch.full((1,3,384,128),.5)
        mapped,linf=module.transport_delta(clean,(clean+.01).clamp(0,1),base)
        image_checks(base,mapped,mapped-base,(384,128))
        assert abs(linf[0]-.01)<1e-6

def test_runtime_has_no_B1_B2_entrypoints():
    import inspect
    text=inspect.getsource(c)+inspect.getsource(HistoricalB3)
    assert "run_ap_duke" not in text and "FrozenAPGenerator(" not in text
    assert "load_latest(" not in text
