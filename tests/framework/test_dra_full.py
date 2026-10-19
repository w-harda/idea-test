"""Full只测CPU调度/存储/官方接口，不执行真实模型或CUDA。"""
import ast
from contextlib import contextmanager,redirect_stdout
from dataclasses import replace,asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib,io,json,random
import numpy as np
import pytest
import torch
from tbps.artifacts import ArtifactStore,ArtifactError,FileRef,read_json
from tbps.baselines import dra_full as full
from tbps.baselines import dra_full_target as target
from tbps.baselines.dra_official import ordered_dataset
from tbps.engines.metrics import ranking
ROOT=Path(__file__).parents[2]
CONFIG=ROOT/"configs/framework/dra_irra_full.server.json"

@pytest.fixture(scope="module")
def data():
    c=full.configuration(CONFIG)
    ds,raw,m=ordered_dataset(c,"full")
    return c,ds,raw,m

def test_full_mapping(data):
    c,ds,raw,m=data;batches=full.schedule(m)
    assert len(batches)==385 and len(batches[-1][0])==2
    assert sum(len(gs) for gs,_,_ in batches)==3074
    assert sum(len(qs) for _,qs,_ in batches)==6156
    ten=[(gs,qs,mp) for gs,qs,mp in batches if len(qs)>16]
    assert len(ten)==1 and len(ten[0][1])==24
    assert max(ten[0][2])<8

def test_all_original_pairs_not_pid_expansion(data):
    m=data[3]
    for gs,qs,mp in full.schedule(m):
        assert all(q.paired_image_id==gs[i].image_id for q,i in zip(qs,mp))
        assert len(set(q.query_id for q in qs))==len(qs)

def test_rng_read_does_not_consume():
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    a=full.rng_state();b=full.rng_state()
    assert full.rng_digest(a)==full.rng_digest(b)
    raw=full.rng_bytes(a)
    state=torch.load(io.BytesIO(raw),weights_only=False)
    assert full.rng_digest(a)==full.rng_digest(state)
    random.random()
    assert full.rng_digest(a)!=full.rng_digest(full.rng_state())

def test_save_compact_bitwise_and_immutable(tmp_path):
    base=tmp_path/"tbps-framework-fixture";store=ArtifactStore(base/"source",new_base=base)
    big=torch.arange(1000.).reshape(100,10)
    x=big[:1];values={"original":x,"adversarial":x.clone(),"effective_delta":torch.zeros_like(x)}
    full.save_pixels(store,"batch-000000",values,{"fixture":True},())
    got=torch.load(store.root/"batch-000000/pixels.pt",weights_only=True)
    assert got["original"].untyped_storage().nbytes()==x.numel()*4
    with pytest.raises(ArtifactError):full.save_pixels(store,"batch-000000",values,{},())

def test_full_metrics_independent_official(data):
    m=replace(data[3],gallery=data[3].gallery[:16],
        queries=tuple(q for q in data[3].queries if q.paired_image_id in {g.image_id for g in data[3].gallery[:16]}))
    gen=torch.Generator().manual_seed(3);x=torch.randn(len(m.queries),16,generator=gen)
    a,order=target.full_metrics(x,m,target.rank_function(data[0]))
    b,other=ranking(x,m.evaluation_identity())
    assert a=={k:b[k] for k in full.METRICS} and torch.equal(order,other)

@pytest.mark.parametrize("bad",["shape","double","nan","grad"])
def test_bad_scores_rejected(data,bad):
    m=replace(data[3],gallery=data[3].gallery[:16],queries=data[3].queries[:4])
    x=torch.zeros(4,16)
    if bad=="shape":x=x[:3]
    if bad=="double":x=x.double()
    if bad=="nan":x[0,0]=float("nan")
    if bad=="grad":x.requires_grad_(True)
    with pytest.raises(ArtifactError):target.full_metrics(x,m,target.rank_function(data[0]))

def test_no_gpu_without_explicit_permission(data):
    with patch.object(full,"configuration",return_value=data[0]),patch.object(full,"source_run") as source:
        with pytest.raises(ArtifactError):full.main(["--config",str(CONFIG),"--action","source"])
        source.assert_not_called()

def test_official_call_and_initialization_unchanged():
    old=ast.parse((ROOT/"src/tbps/baselines/dra_irra.py").read_text())
    new=ast.parse((ROOT/"src/tbps/baselines/dra_full.py").read_text())
    def calls(tree,name):
        return [ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)==name]
    for name in ("attacker.attack","official.ImageAttacker","official.TextAttacker","official.Attacker",
                 "torch.manual_seed","np.random.seed","random.seed"):
        assert calls(old,name)==calls(new,name)
    fn=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=="source_run")
    loop=next(n for n in ast.walk(fn) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and isinstance(n.target.elts[0],ast.Name) and n.target.elts[0].id=="number")
    assert not any(isinstance(n,ast.Call) and ast.unparse(n.func) in
                   ("torch.manual_seed","np.random.seed","random.seed") for n in ast.walk(loop))

def test_native_full_scoring_and_four_metrics():
    s=(ROOT/"src/tbps/baselines/dra_full_target.py").read_text()
    assert 'load_model(loader,model,"full")' in s
    assert 'engine.run(inputs,modes=MODES,query_chunk=128)' in s
    assert "EightGalleryEngine" not in s
    assert full.METRICS==("R@1","R@5","R@10","mAP")

@pytest.mark.parametrize("fail_batch",[None,1])
def test_real_source_control_385_batches_cpu_mock(data,tmp_path,monkeypatch,fail_batch):
    # 执行实际source_run控制流程；只替换模型/官方攻击对象/磁盘大像素，禁止真实CUDA。
    c=dict(data[0]);c["output_base"]=str(tmp_path/"tbps-framework-source")
    m=data[3];state={"calls":0,"models":0,"MLM":0,"saved":[],"rng":[]}
    class Store:
        def __init__(self,root,**kw):self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        def publish(self,key,files,*a,**kw):
            p=self.root/key
            assert not p.exists();p.mkdir()
            for name,value in files.items():(p/name).write_bytes(value)
    class Source(torch.nn.Module):
        def __init__(self,backend,*a,**kw):
            super().__init__();self.backend=backend.requires_grad_(True);self.counts={};self.eval()
    class Attacker:
        def __init__(self,source,image,text):self.source=source
        def attack(self,pixels,texts,mapping,**kw):
            state["calls"]+=1
            if fail_batch is not None and state["calls"]==fail_batch+1:raise RuntimeError("injected official failure")
            state["rng"].append(full.rng_digest(full.rng_state()))
            random.random();np.random.rand();torch.rand(1)
            self.source.backend.weight.grad=torch.ones_like(self.source.backend.weight)
            return pixels.clone(),list(texts),.01
    class Image:
        def __init__(self,*a,**kw):pass
        def txt_guided_attack(self):pass
    class Text:
        def __init__(self,*a,**kw):pass
        def img_guided_attack(self):pass
    class Trace:
        def __init__(self,*a):pass
        @contextmanager
        def active(self):yield self
        def report(self):
            texts=state["texts"]
            return {"events":["text1","image","text2"],"counts":{"call:rand3Num":45,"backward":100},
                "selected_ratios":[{}]*9,"all_image_gradients_finite":True,"T1":texts,
                "text_actions":[{"stage":2,"query_local_index":i,"original_word_slots":x.split(" "),
                    "final_word_slots":x.split(" "),"official_change_count":0} for i,x in enumerate(texts)]}
    def members(manifest):
        for gs,qs,mp in full.schedule(manifest):
            state["texts"]=[q.caption for q in qs]
            yield gs,qs,mp
    def adapter(*a,**kw):
        state["models"]+=1
        return SimpleNamespace(_backend=torch.nn.Linear(1,1).eval().requires_grad_(False),_tokenizer=None)
    def mlm(*a,**kw):
        state["MLM"]+=1;return torch.nn.Linear(1,1)
    def save(store,key,tensors,md,sources):
        state["saved"].append((key,md["gallery_ids"],md["query_ids"],md["rng_before_state"],md["rng_after_state"]))
    original_schedule=full.schedule
    def batch_iterator(manifest):
        for gs,qs,mp in original_schedule(manifest):
            state["texts"]=[q.caption for q in qs];yield gs,qs,mp
    monkeypatch.setattr(full,"preflight",lambda path:(c,data[1],data[2],m,{}))
    monkeypatch.setattr(full,"version",lambda k:c["source_versions"][k])
    monkeypatch.setattr(full,"resources",lambda c:{})
    monkeypatch.setattr(full,"ArtifactStore",Store)
    monkeypatch.setattr(full,"ref",lambda p:FileRef(str(Path(p).resolve()),"0"*64))
    monkeypatch.setattr(full.IRRAAdapter,"from_source",adapter)
    import transformers
    monkeypatch.setattr(transformers.BertForMaskedLM,"from_pretrained",mlm)
    monkeypatch.setattr(full,"modules",lambda c:(SimpleNamespace(ImageAttacker=Image,TextAttacker=Text,Attacker=Attacker),
        SimpleNamespace(BertTokenizer=SimpleNamespace(from_pretrained=lambda *a,**kw:None)),None))
    monkeypatch.setattr(full,"IRRADRASource",Source)
    monkeypatch.setattr(full,"PassiveTrace",Trace)
    monkeypatch.setattr(full,"source_pixels",lambda p:torch.zeros(3,2,2))
    monkeypatch.setattr(full,"save_pixels",save)
    monkeypatch.setattr(full,"validate_source",lambda *a:None)
    monkeypatch.setattr(full,"verify_official",lambda c:{})
    monkeypatch.setattr(full,"schedule",batch_iterator)
    original_to=torch.Tensor.to
    def cpu_to(self,*a,**kw):
        if a and str(a[0]).startswith("cuda"):return self
        return original_to(self,*a,**kw)
    monkeypatch.setattr(torch.Tensor,"to",cpu_to)
    monkeypatch.setattr(torch.cuda,"_lazy_init",lambda:(_ for _ in ()).throw(AssertionError("CUDA forbidden")))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","0")
    with redirect_stdout(io.StringIO()):
        rc=full.source_run(CONFIG)
    if fail_batch is not None:
        assert rc==1 and state["calls"]==fail_batch+1 and len(state["saved"])==fail_batch
        assert (Path(c["output_base"])/"full/source"/("claim-batch-"+str(fail_batch).zfill(6))).exists()
        assert not (Path(c["output_base"])/"full/source/completed").exists()
        return
    assert rc==0 and state["calls"]==385 and state["models"]==state["MLM"]==1
    assert len(state["saved"])==385 and len(state["saved"][-1][1])==2
    assert len({x for row in state["saved"] for x in row[2]})==6156
    assert all(a[4]==b[3] for a,b in zip(state["saved"],state["saved"][1:]))
    assert len(set(state["rng"]))==385
    report=read_json(Path(c["output_base"])/"full/source/execution/report.json")
    assert report["status"]=="PASS" and report["parameter_updates"]==0

@pytest.mark.parametrize("fail_source",[False,True])
def test_real_bash_supervisor_sequence_and_failure(tmp_path,fail_source):
    import os,subprocess
    root=tmp_path/"root";control=tmp_path/"control";out=tmp_path/"full"
    (root/"src/tbps/baselines").mkdir(parents=True)
    (root/"configs/framework").mkdir(parents=True)
    (root/"src/tbps/baselines/dra_full.py").write_text("# fixture")
    (root/"configs/framework/dra_irra_full.server.json").write_text("{}")
    subprocess.run(["git","init","-q"],cwd=root,check=True)
    subprocess.run(["git","-c","user.name=fixture","-c","user.email=f@f","commit","-qm","fixture","--allow-empty"],cwd=root,check=True)
    fake=tmp_path/"python";calls=tmp_path/"calls"
    fake.write_text("""#!/bin/bash
echo "$*" >> '"""+str(calls)+"""'
case "$*" in
  *"--action resources"*) echo DRA_FULL_RESOURCES_PASS ;;
  *"--action source"*) """+("exit 7" if fail_source else "echo DRA_OFFICIAL_IRRA_SOURCE_PASS full")+""" ;;
  *"--model irra"*) echo DRA_NATIVE_TARGET_PASS full irra ;;
  *"--model rasa"*) echo DRA_NATIVE_TARGET_PASS full rasa ;;
  *"--model aptm"*) echo DRA_NATIVE_TARGET_PASS full aptm ;;
  *"--action audit"*) echo DRA_INDEPENDENT_CPU_AUDIT_PASS full ;;
  *) echo DRA_FULL_CPU_PREFLIGHT_PASS ;;
esac
""");fake.chmod(0o755)
    binpath=tmp_path/"bin";binpath.mkdir()
    gpu=binpath/"nvidia-smi";gpu.write_text("#!/bin/bash\necho fixture\n");gpu.chmod(0o755)
    text=(ROOT/"src/tbps/baselines/run_dra_full.sh").read_text()
    text=text.replace("/home/lzf/ldx/projects/idea-test-tbps-vlpt-full-20261019",str(root))
    text=text.replace("/home/lzf/ldx/tmp/vlpt_official_irra_full_v1",str(control))
    text=text.replace("/home/lzf/ldx/outputs/tbps-framework-vlpt-official-irra/full",str(out))
    text=text.replace("/home/lzf/ldx/envs/dra-irra-source/bin/python",str(fake))
    text=text.replace("/home/lzf/ldx/envs/tbps-image/bin/python",str(fake))
    script=tmp_path/"run.sh";script.write_text(text)
    env=dict(os.environ,PATH=str(binpath)+":"+os.environ["PATH"])
    p=subprocess.run(["bash",str(script),"--allow-full"],env=env,capture_output=True,text=True)
    assert p.returncode==(7 if fail_source else 0),p.stdout+p.stderr
    assert (control/"phase").read_text().strip()==("FAIL" if fail_source else "SUCCESS")
    count=len(calls.read_text().splitlines())
    assert count==(3 if fail_source else 7)
    again=subprocess.run(["bash",str(script),"--allow-full"],env=env,capture_output=True)
    assert again.returncode!=0 and len(calls.read_text().splitlines())==count
