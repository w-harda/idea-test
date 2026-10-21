"""新Validation桥/单次生成/独立审计的CPU回归；不读取真实模型或初始化CUDA。"""
from pathlib import Path
from unittest.mock import patch
import json, io
import pytest
import torch
from tbps.artifacts import ArtifactStore,ArtifactError,FileRef,sha_file
from tbps.ablations.g60_h_bridge import *
from tbps.ablations.g60_h_validation import stream,official_metrics,frozen
from tbps.materialization.gpu_safety import no_optimization

torch.set_num_threads(2)

@pytest.fixture
def tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("tbps-framework-fixture")

def test_import_cpu():
    assert not torch.cuda.is_initialized()

@pytest.mark.parametrize("n",[1,128])
def test_storage_clone(tmp_path,n):
    backing=torch.randn(1000,512);small=backing[:n]
    p=tmp_path/"x.pt";p.write_bytes(tensor_bytes({"feature":small}))
    x=load_tensors(p)["feature"]
    assert torch.equal(x,small)
    assert x.untyped_storage().nbytes()==n*512*4

def test_metrics_csv_stable_order():
    a={"Group":"x","R@1":2.,"R@5":4.,"R@10":6.,"mAP":3.12345}
    b=json.loads(json.dumps(a,sort_keys=True))
    assert metrics_csv([a])==metrics_csv([b])
    assert b"3.12" in metrics_csv([a],True)

@pytest.mark.parametrize("bad",["dtype","shape","nan","norm"])
def test_feature_rejection(bad):
    x=torch.nn.functional.normalize(torch.ones(1,512),dim=1)
    if bad=="dtype":x=x.double()
    if bad=="shape":x=x[:,:20]
    if bad=="nan":x[0,0]=float("nan")
    if bad=="norm":x=x*2
    with pytest.raises(ValueError):normalized(x,1)

def fake_members():
    ids=[];ann=[];entries=[]
    for i in range(128):
        row={"row_id":f"cuhk:{i}:0","image_rel_path":f"{i}.jpg","pid":i+1,
             "role":"validation","split":"train","caption_index":0,"caption":f"caption {i}",
             "metadata_sha256":"a"*64}
        ids.append(row);entries.append(dict(row))
        ann.append({"id":i+1,"split":"train","file_path":f"{i}.jpg","captions":[f"caption {i}"]})
    return ids,[{"pid":999}],ann,{"entries":entries}

def test_membership_actual_identity():
    args=fake_members();assert membership(*args)["validation"]==128

@pytest.mark.parametrize("bad",["role","test","overlap","order","caption","duplicate"])
def test_membership_reject(bad):
    ids,train,ann,cache=fake_members()
    if bad=="role":ids[0]["role"]="train"
    if bad=="test":ann[0]["split"]="test"
    if bad=="overlap":train[0]["pid"]=ids[0]["pid"]
    if bad=="order":cache["entries"].reverse()
    if bad=="caption":ann[0]["captions"][0]="changed"
    if bad=="duplicate":ids[1]=dict(ids[0])
    with pytest.raises(ValueError):membership(ids,train,ann,cache)

def test_noop_explicit():
    md={"identity":{"caption":"empty"},"adversarial":"empty",
        "action":{"NOOP":True,"reasons":["no_attrs"],"edits":[],"budget_k":0}}
    assert legacy_text_record(md,{"eligible":False,"reasons":["no_attrs"]}) is None
    md["adversarial"]="changed"
    with pytest.raises(ValueError):legacy_text_record(md,{"eligible":False,"reasons":["no_attrs"]})

def run_fixture(tmp_path,n=2,fail=False):
    store=ArtifactStore(tmp_path/"new",new_base=tmp_path)
    store.publish("protocol",{"p.json":b"{}"},{"test":True})
    refs=[]
    for i in range(n):
        p=tmp_path/f"input{i}";p.write_text(str(i));refs.append(FileRef(str(p),sha_file(p)))
    ctx={"identities":[{"row_id":str(i)} for i in range(n)],"c":{"G60":{"sha256":"b"*64}},"image_refs":refs}
    counts={"G_started":0,"G_completed":0,"IRRA_images":0};sequence=[]
    def gen(x):
        i=counts["G_started"]
        assert (store.root/f"claim-{i:03d}").exists()
        counts["G_started"]+=1;sequence.append(i)
        if fail:raise RuntimeError("fixture generation fail")
        counts["G_completed"]+=1
        return {"adversarial":x,"effective_delta":x-x}
    def encode(x):
        counts["IRRA_images"]+=1
        return torch.nn.functional.normalize(torch.ones(1,512),dim=1)
    if n==2:
        result=stream(ctx,store,gen,encode,lambda ref:torch.full((1,3,384,128),.5),counts,limit=n)
    else:
        # 完整128成员的控制流用小Tensor，真实RGB合法性另外独立验证。
        import tbps.ablations.g60_h_validation as runner
        values={}
        def pack(d):values["last"]={k:v.clone() for k,v in d.items()};return b"fixture"
        with patch.object(runner,"image_checks",return_value={"linf":0}),patch.object(runner,"tensor_bytes",side_effect=pack),patch.object(runner,"load_tensors",side_effect=lambda p:values["last"]),patch.object(runner,"prefix_cpu_audit",return_value={"status":"CPU_FIXTURE_ONLY"}):
            result=stream(ctx,store,gen,encode,lambda ref:torch.ones(1,1),counts,limit=n)
    return ctx,store,counts,sequence,result

def test_smoke_prefix_retained_and_counts(tmp_path):
    ctx,store,counts,sequence,result=run_fixture(tmp_path)
    assert counts=={"G_started":2,"G_completed":2,"IRRA_images":2}
    assert sequence==[0,1] and result[0].shape==(2,512)
    report=read_json(store.root/"prefix-smoke/report.json")
    assert report["retained_as_part_of_128"] and report["indices"]==[0,1]

def test_complete_order128(tmp_path):
    _,store,counts,sequence,_=run_fixture(tmp_path,n=128)
    assert sequence==list(range(128)) and set(counts.values())=={128}
    assert len(list(store.root.glob("image-*")))==128

def test_claim_failure_preserved_no_retry(tmp_path):
    with pytest.raises(RuntimeError):run_fixture(tmp_path,fail=True)
    store=ArtifactStore(tmp_path/"new",new_base=tmp_path)
    assert (store.root/"claim-000").exists() and not (store.root/"image-000").exists()
    with pytest.raises(ArtifactError):store.publish("claim-000",{"x":b"x"},{"test":True})

def test_budget_and_dtype():
    from tbps.materialization.store import image_checks
    x=torch.full((1,3,384,128),.5);y=x+.03
    assert image_checks(x,y,y-x,(384,128))["linf"]<=8/255
    y=x+.04
    with pytest.raises(ValueError):image_checks(x,y,y-x,(384,128))

def test_official_metrics_multiple_positives(tmp_path):
    # 独立CPU Fixture只验证wrapper原样委托rank，而非另写指标。
    p=tmp_path/"rank.py"
    p.write_text("import torch\ndef rank(scores,qids,gids,max_rank,get_mAP):\n assert max_rank==10 and get_mAP\n assert (qids==gids).all()\n return torch.arange(10).float(),torch.tensor(42.5),None,None\n")
    scores=torch.eye(12);pids=torch.tensor([0,0]+list(range(1,11)))
    assert official_metrics(scores,pids,str(p))=={"R@1":0.,"R@5":4.,"R@10":9.,"mAP":42.5}

@pytest.mark.parametrize("bad",["shape","nan","dtype"])
def test_score_rejection(bad):
    pids=torch.arange(12);x=torch.eye(12)
    if bad=="shape":x=x[:10]
    if bad=="nan":x[0,0]=float("nan")
    if bad=="dtype":x=x.double()
    with pytest.raises(ValueError):official_metrics(x,pids,"unused.py")

def test_model_frozen():
    m=torch.nn.Linear(2,2).eval().requires_grad_(False)
    a=frozen(m);assert frozen(m)==a
    m.train()
    with pytest.raises(ValueError):frozen(m)

@pytest.mark.parametrize("operation",["backward","optimizer","jvp"])
def test_optimization_forbidden(operation):
    with pytest.raises(ArtifactError):
        with no_optimization():
            if operation=="backward":torch.ones(1,requires_grad=True).backward()
            if operation=="optimizer":torch.optim.Adam([torch.nn.Parameter(torch.ones(1))])
            if operation=="jvp":torch.func.jvp(lambda x:x,(torch.ones(1),),(torch.ones(1),))

def test_sha_and_atomic_rejection(tmp_path):
    source=tmp_path/"src";source.write_text("original")
    store=ArtifactStore(tmp_path/"out",new_base=tmp_path);pin=FileRef(str(source),sha_file(source))
    store.publish("test",{"v":b"v"},{"test":True},sources=(pin,))
    source.write_text("tampered")
    with pytest.raises(ArtifactError):store.verify("test")
    def stop(_):raise RuntimeError("interrupt")
    with pytest.raises(RuntimeError):store.publish("new",{"v":b"v"},{"x":1},fault=stop)
    assert store.pending() and not (store.root/"new").exists()


def test_missing_new_runtime_pin_rejected(tmp_path):
    import tbps.ablations.g60_h_bridge as bridge
    root=Path(bridge.__file__).resolve().parents[3]
    c=json.loads((root/"configs/framework/g60_h_validation.server.json").read_text())
    c["producers"].pop(str(Path(bridge.__file__).resolve()))
    target=tmp_path/"config.json";target.write_text(json.dumps(c))
    with pytest.raises(ValueError,match="complete fixed"):configuration(target)

def test_changed_new_source_sha_rejected(tmp_path):
    import tbps.ablations.g60_h_bridge as bridge
    root=Path(bridge.__file__).resolve().parents[3]
    c=json.loads((root/"configs/framework/g60_h_validation.server.json").read_text())
    c["G60"]["sha256"]="0"*64
    target=tmp_path/"config.json";target.write_text(json.dumps(c))
    with pytest.raises(ValueError,match="locked G"):configuration(target)

def test_prefix_audit_failure_stops_before_third(tmp_path):
    import tbps.ablations.g60_h_validation as runner
    # 原stream先完成两图，然后审计异常；没有第三个Claim/前向或Smoke PASS。
    with patch.object(runner,"prefix_cpu_audit",side_effect=ValueError("prefix audit fail")):
        with pytest.raises(ValueError,match="prefix audit"):run_fixture(tmp_path,n=2)
    base=tmp_path/"new"
    assert (base/"image-000").exists() and (base/"image-001").exists()
    assert not (base/"claim-002").exists() and not (base/"prefix-smoke").exists()
