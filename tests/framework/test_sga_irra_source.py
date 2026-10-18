"""官方算法本体CPU对照及BERT→IRRA接口保真；不恢复真实神经权重。"""
from pathlib import Path
from types import SimpleNamespace
import io, json
import numpy as np
import pytest
import torch
from transformers.modeling_outputs import MaskedLMOutput
from tbps.artifacts import ArtifactError, ArtifactStore
from tbps.baselines.sga_official_io import official_modules, dataset, official_projection_check
from tbps.baselines.sga_irra import OfficialObservation, tensor_archive, manifest_from_dict, main
from tbps.models.sga_irra_source import AuditedBert, IRRASGASource, decode_bert
from tbps.models.irra_io import official_tokenizer
from tbps.models.irra_source import IRRASource

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/"configs/framework/sga_irra_source.server.json"

@pytest.fixture(scope="module")
def resources():
    c=json.loads(CONFIG.read_text())
    mod,tok,_=official_modules(c)
    bert=AuditedBert(tok.BertTokenizer.from_pretrained(c["bert"],local_files_only=True))
    return c,mod,bert,official_tokenizer(IRRASource.from_dict(c["irra"]))

@pytest.mark.parametrize("text",["a man wears blue jeans","a man wears [UNK] jeans","a [CLS] woman","black jacket .","a person's bag","[MASK] dress","white shirt [SEP]"])
def test_real_tokenizers_roundtrip(resources,text):
    _,_,bert,bpe=resources
    batch=bert([text],padding="max_length",truncation=True,max_length=30,return_tensors="pt")
    assert set(batch)=={"input_ids","token_type_ids","attention_mask"}
    trace=decode_bert(batch,bert)[0]
    assert trace["wordpiece_roundtrip_exact"]
    assert bert.encode(trace["decoded_text"],add_special_tokens=False)==batch.input_ids[0][batch.attention_mask[0].bool()].tolist()[1:]
    assert trace["decoded_text"].endswith(bert.decode(bert.encode(text,add_special_tokens=False),skip_special_tokens=False).strip().split()[-1])
    ids,reports=bpe.batch([trace["decoded_text"]],truncate=False)
    assert ids.shape==(1,77) and not reports[0].truncated

def test_original_truncation_logged(resources):
    _,_,bert,_=resources
    x=bert(["jacket "*60],padding="max_length",truncation=True,max_length=30,return_tensors="pt")
    r=decode_bert(x,bert)[0]
    assert r["bert_truncated"] and len(r["bert_visible_ids"])==30
    assert len(bert.encode(r["decoded_text"],add_special_tokens=False))==29

def test_byte_bpe_extra_truncation_rejected(resources):
    _,_,_,bpe=resources
    with pytest.raises(ArtifactError,match="77-token"):bpe.batch(["verylongwords "*80],truncate=False)

class Backend(torch.nn.Module):
    def __init__(self):
        super().__init__();self.image=torch.nn.Linear(3,512);self.text=torch.nn.Embedding(997,512)
    def encode_image(self,x):return self.image(x.mean((-1,-2)))
    def encode_text(self,x):return self.text(x.remainder(997)).mean(1)

class MLM(torch.nn.Module):
    def __init__(self,bert):super().__init__();self.p=torch.nn.Parameter(torch.zeros(()),requires_grad=False);self.ids=[bert.convert_tokens_to_ids(x) for x in ["blue","red","black","white","green","pants","shirt","coat","hat","shoes"]]
    @property
    def device(self):return self.p.device
    def forward(self,x,attention_mask=None):
        logits=torch.full((*x.shape,30522),-9.)
        for n,i in enumerate(self.ids):logits[:,:,i]=12.-n*.5
        return MaskedLMOutput(logits=logits)

def test_official_complete_algorithm_mock(resources):
    _,mod,bert,bpe=resources
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
    backend=Backend();before={k:v.clone() for k,v in backend.state_dict().items()}
    source=IRRASGASource(backend,bpe,bert)
    image=mod.ImageAttacker(lambda x:x,eps=2/255,steps=10,step_size=.5/255)
    text=mod.TextAttacker(MLM(bert),bert,cls=False,max_length=30,number_perturbation=1,topk=10,threshold_pred_score=.3)
    attack=mod.SGAttacker(source,image,text);events=[]
    obs=OfficialObservation(attack,source,events.append)
    originals=["a man blue shirt","a person red coat","a woman black pants","a person white hat"]
    obs.reset(originals);clean=torch.rand(2,3,12,8)
    adv,words=attack.attack(clean,originals,[0,0,1,1],device="cpu",max_length=30,scales=[.5,.75,1.25,1.5])
    assert obs.events==["text1","image","text2"] and len(words)==4
    assert source.counts["image_samples"]==104 # 2 original + 10 * 5 * 2 + 2 adversarial
    assert all(torch.equal(v,before[k]) for k,v in backend.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in backend.parameters())
    assert official_projection_check(clean,adv)["extra_projection"] is False
    assert any(e["event"]=="candidate_choice" for e in events)
    assert [e["input_texts"] for e in events if e["event"] in ("text1","text2")]==[originals,originals]
    assert any(x!=y for x,y in zip(clean.flatten(),adv.flatten()))

def test_losses_independent(resources):
    _,mod,_,_=resources
    i=torch.tensor([[1.,2.],[3.,4.]]);t=torch.tensor([[2.,-1.],[1.,3.],[4.,2.]])
    mapping=[0,0,1]
    actual=mod.ImageAttacker(None).loss_func(i,t,mapping)
    expected=-((i[0]*t[0]).sum()+(i[0]*t[1]).sum()+(i[1]*t[2]).sum())/2
    assert torch.equal(actual,expected)
    text=mod.TextAttacker(None,None,cls=False)
    assert torch.equal(text.loss_func(t,i,1),-(t*i[1]).sum(-1))

@pytest.mark.parametrize("bad",["nan","overshoot","dtype"])
def test_budget_rejects(bad):
    x=torch.full((1,3,2,2),.5);y=x.clone()
    if bad=="nan":y[0,0,0,0]=float("nan")
    if bad=="overshoot":y+=.02
    if bad=="dtype":y=y.double()
    with pytest.raises(ArtifactError):official_projection_check(x,y)

def test_compact_immutable_readback(tmp_path):
    base=tmp_path/"tbps-framework-fixture";store=ArtifactStore(base/"new",new_base=base)
    huge=torch.arange(300.).view(30,10);view=huge[:1]
    with tensor_archive(store,"batch",{"x":view},{},()):pass
    got=torch.load(store.root/"batch/pixels.pt",weights_only=True)["x"]
    assert torch.equal(got,view) and got.untyped_storage().nbytes()==got.numel()*4
    with pytest.raises(ArtifactError):
        with tensor_archive(store,"batch",{"x":view},{},()):pass

def test_dataset_real_locked_order(resources):
    c,_,_,_=resources
    ds,raw,matched=dataset(c,"smoke")
    assert [g.image_id for g in raw.gallery]==["cuhk:37132:0","cuhk:37147:0"]
    assert len(raw.queries)==4 and len({g.pid for g in raw.gallery})==2
    assert [q.query_id for q in raw.queries]==[q.query_id for q in matched.queries]
    assert manifest_from_dict(__import__("dataclasses").asdict(matched))==matched

def test_default_no_gpu_authorization(resources):
    with pytest.raises(ArtifactError,match="explicit GPU scope"):
        main(["--config",str(CONFIG),"--action","pipeline"])
    assert not torch.cuda.is_initialized()

def test_full_caption_members_without_encoding(resources):
    c,_,_,_=resources
    _,raw,matched=dataset(c,"full")
    assert len(raw.gallery)==3074 and len(raw.queries)==6156
    counts={g.image_id:0 for g in raw.gallery}
    for q in raw.queries:counts[q.paired_image_id]+=1
    assert list(counts.values()).count(10)==1 and list(counts.values()).count(2)==3073
    assert [q.query_id for q in raw.queries]==[q.query_id for q in matched.queries]
    assert all(a.pid==b.pid and a.paired_image_id==b.paired_image_id for a,b in zip(raw.queries,matched.queries))
