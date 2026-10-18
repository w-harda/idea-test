"""只用CPU Fixture/官方CPU oracle，不加载真实权重。"""
from pathlib import Path
from types import SimpleNamespace as NS
from dataclasses import replace
import io,json
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from tbps.artifacts import ArtifactError,ArtifactStore,FileRef,sha_file
from tbps.contracts import PersonID,GalleryItem,QueryItem,DatasetManifest
from tbps.baselines.sga_attack import (official_image_attacker,attack_image,project,sample_seed,caption_groups,SCALES,EPSILON)
from tbps.models.sga_surrogate import SGASurrogate

OFFICIAL=Path("/home/lzf/ldx/projects/SGA-official/attacker.py")
def official():
    return official_image_attacker({"path":str(OFFICIAL),"sha256":sha_file(OFFICIAL)})
def fixture():
    pid=PersonID("fixture","1")
    gallery=tuple(GalleryItem("g"+str(i),"i"+str(i)+".jpg","0"*64,pid,"test") for i in range(2))
    q=tuple(QueryItem("q"+str(i),"caption "+str(i),i%2,"g"+str(i//2),pid,"test") for i in range(4))
    return DatasetManifest("fixture","test",FileRef("/synthetic/annotation","1"*64),gallery,q)
class LinearNative(nn.Module):
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.tensor([.3,.7,1.1]))
    def encode_image(self,x):
        return x.mean(dim=(2,3))*self.weight
    def inference_image(self,x):return {"image_feat":F.normalize(self.encode_image(x),dim=-1)}
def test_official_augmentation_exact():
    x=torch.linspace(.01,.99,3*12*8).reshape(1,3,12,8).requires_grad_()
    a=official();np.random.seed(8);one=a.get_scaled_imgs(x,list(SCALES),"cpu")
    np.random.seed(8);two=official().get_scaled_imgs(x,list(SCALES),"cpu")
    assert torch.equal(one,two) and len(one)==5
    assert torch.autograd.grad(one.sum(),x)[0].abs().sum()>0

@pytest.mark.parametrize("seed",[1,42,153])
def test_image_loop_independent_official_oracle(seed):
    torch.set_num_threads(2);x=torch.linspace(.1,.9,3*12*8).reshape(1,3,12,8)
    text=F.normalize(torch.tensor([[.7,.3,.6],[.1,.9,.2]]),dim=-1)
    net=LinearNative().eval().requires_grad_(False);a=official();a.steps=3
    np.random.seed(seed);expected=a.txt_guided_attack(net,x,[0,0],"cpu",scales=list(SCALES),txt_embeds=text)
    got,rounds=attack_image(x,lambda z:-(net.inference_image(z)["image_feat"]@text.T).sum(),official(),seed=seed,steps=3)
    assert torch.allclose(got,expected,atol=1e-6,rtol=0)
    assert len(rounds)==3 and all(r["gradient_l2"]>0 for r in rounds)
    assert all(p.grad is None for p in net.parameters())

@pytest.mark.parametrize("value",[-20.,0.,.3,.5,1.,20.])
def test_strict_budget_and_pixel_range(value):
    clean=torch.tensor([0.,.01,.5,.99,1.]).reshape(1,1,1,5)
    adv=project(clean,torch.full_like(clean,value))
    assert (adv.double()-clean.double()).abs().max()<=EPSILON and adv.min()>=0 and adv.max()<=1

def test_projection_gradient_saturation():
    clean=torch.full((1,3,2,2),.5)
    x=clean.clone().requires_grad_();project(clean,x).sum().backward();assert torch.equal(x.grad,torch.ones_like(x))
    x=(clean+1).requires_grad_();project(clean,x).sum().backward();assert x.grad.abs().sum()==0

def test_rng_restored_and_per_image_deterministic():
    x=torch.full((1,3,8,8),.5);a=official();np.random.seed(19);before=np.random.get_state()
    one,_=attack_image(x,lambda z:z.sum(),a,seed=sample_seed("g1"),steps=2)
    after=np.random.get_state()
    assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]
    two,_=attack_image(x,lambda z:z.sum(),a,seed=sample_seed("g1"),steps=2)
    assert torch.equal(one,two) and sample_seed("g1")!=sample_seed("g2")

def test_all_paired_captions_preserve_order_not_same_pid():
    m=fixture();g=caption_groups(m)
    assert [q.query_id for q in g["g0"]]==["q0","q1"]
    assert [q.query_id for q in g["g1"]]==["q2","q3"]
    assert all(q.caption==m.queries[i].caption for i,q in enumerate(g["g0"]+g["g1"]))

def test_missing_caption_rejected():
    m=fixture();m=replace(m,queries=m.queries[:2])
    with pytest.raises(ArtifactError,match="Caption set"):caption_groups(m)

@pytest.mark.parametrize("bad",["zero","nan"])
def test_bad_gradient_or_objective_stops(bad):
    x=torch.full((1,3,8,8),.5)
    def loss(z):return z.sum()*(float("nan") if bad=="nan" else 0)
    with pytest.raises(ArtifactError):attack_image(x,loss,official(),seed=2,steps=1)

def test_bad_original_rejected():
    with pytest.raises(ArtifactError):attack_image(torch.ones(1,3,8,8)*2,lambda z:z.sum(),official(),seed=2)

class FusionNative(nn.Module):
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.tensor(1.3))
        self.itm_head=nn.Linear(3,2,bias=False)
        with torch.no_grad():self.itm_head.weight.copy_(torch.tensor([[0.,0.,0.],[.3,.6,.9]]))
        self.text_encoder=NS(bert=self.fusion)
    def visual_encoder(self,x):return x.mean((2,3))[:,None,:]*self.weight
    def get_vision_embeds(self,x):return self.visual_encoder(x),None
    def fusion(self,**kw):
        return NS(last_hidden_state=kw["encoder_hidden_states"]+kw["encoder_embeds"])
    def get_cross_embeds(self,tokens,att,**kw):
        return self.fusion(encoder_hidden_states=tokens,encoder_embeds=kw["text_embeds"]).last_hidden_state

@pytest.mark.parametrize("model",["irra","rasa","aptm"])
def test_native_targets_nonzero_input_gradient_frozen(model):
    x=torch.linspace(.1,.9,3*16*8).reshape(1,3,16,8).requires_grad_()
    if model=="irra":
        backend=LinearNative().eval().requires_grad_(False)
        adapter=NS(_backend=backend);supervision=F.normalize(torch.tensor([[.3,.7,.5],[.6,.2,.7]]),dim=-1)
    else:
        backend=FusionNative().eval().requires_grad_(False);adapter=NS(backend=backend)
        supervision={"global":torch.ones(2,3),"tokens":torch.zeros(2,1,3),"attention":torch.ones(2,1,dtype=torch.long)}
    a=SGASurrogate(adapter,model,supervision)
    # 使用真实原生预处理函数，IRRA尺寸放宽仅Fixture内直接调用目标。
    if model=="irra":assert a._pixels(x).shape==x.shape
    loss=a(x);grad,=torch.autograd.grad(loss,x)
    assert torch.isfinite(grad).all() and grad.abs().sum()>0 and loss.ndim==0
    assert all(not p.requires_grad and p.grad is None for p in backend.parameters())
    assert a.counts["image_samples"]==1
    if model!="irra":
        assert a.counts["fusion_pairs"]==2
        expected=-backend.itm_head(backend.visual_encoder(a._pixels(x))[:,0,:])[:,1].sum()*2
        assert torch.equal(loss,expected)

def test_fusion_target_rejects_inference_mode():
    backend=FusionNative().eval().requires_grad_(False)
    c={"global":torch.ones(1,3),"tokens":torch.zeros(1,1,3),"attention":torch.ones(1,1,dtype=torch.long)}
    with torch.no_grad(),pytest.raises(ArtifactError):
        SGASurrogate(NS(backend=backend),"aptm",c)(torch.ones(1,3,4,4))

def test_claim_pixel_roundtrip_and_orphan_refusal(tmp_path):
    from tbps.baselines.sga_whitebox import SGAPixels
    base=tmp_path/"tbps-framework-sga";root=base/"sga";root.mkdir(parents=True)
    store=ArtifactStore(root,new_base=base);store.publish("protocol",{"protocol.json":b"{}"},{"kind":"fixture"})
    pixels=SGAPixels(store,FileRef(str(root/"protocol/manifest.json"),sha_file(root/"protocol/manifest.json")))
    m=fixture();row=m.gallery[0];queries=caption_groups(m)[row.image_id]
    claim=pixels.claim(row,queries,42)
    with pytest.raises(ArtifactError,match="Claim"):pixels.claim(row,queries,42)
    clean=torch.full((1,3,384,128),.5);adv=project(clean,clean+.1)
    rounds=[{"iteration":i+1,"objective":-.3,"gradient_l2":1.,"linf":float((adv.double()-clean.double()).abs().max())} for i in range(10)]
    md=pixels.publish(row,clean,adv,rounds,claim,queries)
    got,_=pixels.load(row)
    assert torch.equal(got["adversarial"],adv)
    assert got["adversarial"].untyped_storage().nbytes()==got["adversarial"].numel()*4
    assert md["text_modified"] is False and md["query_ids"]==["q0","q1"]
    with pytest.raises(Exception):pixels.publish(row,clean,adv,rounds,claim,queries)
    assert (root/("claim-"+__import__("tbps.artifacts",fromlist=["digest"]).digest(row.image_id))).exists()

def test_official_source_tamper_rejected(tmp_path):
    p=tmp_path/"attacker.py";p.write_text("raise RuntimeError('must not execute')")
    with pytest.raises(ArtifactError):
        official_image_attacker({"path":str(p),"sha256":"0"*64})

def test_cli_default_cpu_no_model(monkeypatch):
    from tbps.baselines import sga_whitebox as s
    seen=[]
    monkeypatch.setattr(s,"preflight",lambda *args:seen.append(args))
    monkeypatch.setattr(s,"run",lambda *args:pytest.fail("GPU run forbidden"))
    assert s.main(["--config","fixture","--model","irra"])==0 and len(seen)==1

@pytest.mark.parametrize("scope,flags",[("full",[]),("smoke",["--allow-full"]),("full",["--allow-smoke"]),("smoke",["--allow-full","--allow-smoke"])])
def test_cli_gpu_requires_matching_permission(scope,flags):
    from tbps.baselines.sga_whitebox import main
    with pytest.raises(ArtifactError):
        main(["--config","fixture","--model","irra","--action","run","--scope",scope]+flags)

def test_real_configuration_and_no_gpu(monkeypatch):
    from tbps.baselines.sga_whitebox import configuration
    def forbidden(*a,**k):pytest.fail("model loading or CUDA in configuration")
    monkeypatch.setattr(torch,"load",forbidden);monkeypatch.setattr(torch.cuda,"_lazy_init",forbidden)
    p=Path(__file__).resolve().parents[2]/"configs/framework/sga_whitebox.server.json"
    c=configuration(p)
    assert c["official_commit"]=="f49e6926cd13148e49bdbc2c3e537660e096230c" and c["epsilon"]==8/255
    assert "b3" not in c and set(c["objectives"])=={"irra","rasa","aptm"}

@pytest.mark.parametrize("kind",["source","objective","epsilon","caption"])
def test_config_tamper_rejected(tmp_path,kind):
    from tbps.baselines.sga_whitebox import configuration
    p=Path(__file__).resolve().parents[2]/"configs/framework/sga_whitebox.server.json"
    c=json.loads(p.read_text())
    if kind=="source":c["producers"].pop(next(iter(c["producers"])))
    elif kind=="objective":c["objectives"]["aptm"]="pretend cosine native fusion"
    elif kind=="epsilon":c["epsilon"]=16/255
    else:c["caption_policy"]="first_caption_only"
    q=tmp_path/"config.json";q.write_text(json.dumps(c))
    with pytest.raises(ArtifactError):configuration(q)

def test_internal_official_overshoot_not_silently_clamped():
    backend=LinearNative().eval().requires_grad_(False);x=torch.tensor([-0.01,1.01,.5]).view(1,3,1,1).requires_grad_()
    a=SGASurrogate(NS(_backend=backend),"irra",torch.ones(1,3))
    normalized=a._pixels(x)
    from tbps.models.irra_io import MEAN,STD
    expected=(x-x.new_tensor(MEAN).view(1,3,1,1))/x.new_tensor(STD).view(1,3,1,1)
    assert torch.equal(normalized,expected)

def test_generation_real_cpu_control_flow_claim_before_forward(monkeypatch,tmp_path):
    """实际generate/Claim/SGA循环/保存读回/集合顺序；仅把CUDA设备传递映射到CPU。"""
    from contextlib import nullcontext
    from tbps.baselines import sga_whitebox as s
    m=fixture();base=tmp_path/"tbps-framework-sga";root=base/"smoke"/"irra"
    store=ArtifactStore(root,new_base=base)
    store.publish("protocol",{"protocol.json":b"{}"},{"kind":"fixture"})
    clean=torch.linspace(.1,.9,3*384*128).reshape(1,3,384,128)
    input_store=NS(load=lambda u:({"original":clean.clone()},{}))
    ctx={"c":{"output_base":str(base),"seed":42,"objectives":{"irra":"fixture"}},
         "manifest":m,"attack_manifest":m,"groups":caption_groups(m),"official":official(),
         "store":input_store,"images":{g.image_id:g.image_id for g in m.gallery}}
    backend=LinearNative().eval().requires_grad_(False);adapter=NS(_backend=backend)
    events=[];original=backend.encode_image
    def observed(x):
        current=m.gallery[len(events)//50].image_id
        assert (root/"pixels"/("claim-"+s.digest(current))).exists()
        events.append(current);return original(x)
    monkeypatch.setattr(backend,"encode_image",observed)
    original_to=torch.Tensor.to
    def cpu_to(self,*args,**kw):
        if args and isinstance(args[0],str) and args[0].startswith("cuda"):args=("cpu",)+args[1:]
        return original_to(self,*args,**kw)
    monkeypatch.setattr(torch.Tensor,"to",cpu_to)
    monkeypatch.setattr(s,"BulkSession",lambda *args:nullcontext())
    text=F.normalize(torch.tensor([[.2,.4,.7],[.7,.5,.2],[.3,.9,.1],[.6,.4,.2]]),dim=-1)
    condition=NS(batch=NS(payload=text))
    checkpoint=FileRef(str(root/"protocol/manifest.json"),sha_file(root/"protocol/manifest.json"))
    pixels,counts=s.generate(ctx,adapter,"irra",root,checkpoint,condition)
    assert counts["image_samples"]==100 and counts["fusion_pairs"]==0
    assert events==["g0"]*50+["g1"]*50
    collection=json.loads((root/"pixels/collection/collection.json").read_text())
    assert collection["ordered_ids"]==["g0","g1"] and collection["count"]==2
    assert all(p.grad is None for p in backend.parameters())
    for row in m.gallery:
        arrays,md=pixels.load(row)
        assert len(md["rounds"])==10 and md["query_ids"]==[q.query_id for q in ctx["groups"][row.image_id]]

def test_3074_gallery_many_caption_grouping_complete_order():
    pid=PersonID("fixture","1")
    gallery=tuple(GalleryItem("g"+str(i),"i"+str(i)+".jpg","0"*64,pid,"test") for i in range(3074))
    queries=tuple(QueryItem("q"+str(i)+":"+str(j),"clean caption",j,gallery[i].image_id,pid,"test")
                  for i in range(3074) for j in range(2+(i<8)))
    manifest=DatasetManifest("fixture","test",FileRef("/synthetic/annotation","1"*64),gallery,queries)
    assert len(queries)==6156
    groups=caption_groups(manifest)
    assert list(groups)==[g.image_id for g in gallery] and sum(map(len,groups.values()))==6156
    assert all(len(v)==(3 if i<8 else 2) for i,v in enumerate(groups.values()))
