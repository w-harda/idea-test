"""APTM CPU合成/真实来源测试；不加载真实模型、不执行CUDA。"""
from dataclasses import replace,asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
import ast,json,random,sys
import pytest
import torch
import torch.nn.functional as F
from tbps.artifacts import ArtifactError,ArtifactStore,FileRef,sha_file,digest,json_bytes
from tbps.models.base import ModelSpace,RepresentationBatch,InputBinding
from tbps.models.aptm import APTMAdapter,APTMSource,native_modules,official_tokenizer,validate_payload,finalize_scores,load_aptm
from tbps.engines.aptm_score_cache import APTMSetScorer
from tbps.engines.native_cache import NativeFeatureCache
from tbps.engines.evaluation import EvaluationEngine
from tbps.engines import materialized_aptm as runner
import test_rasa_adapter as shared

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"configs/framework/aptm_sources.server.json"
OFFICIAL=Path("/home/lzf/ldx/external/APTM")

def space():return ModelSpace("aptm",digest("w"),*(digest(i) for i in range(4)),"fp32/math/no_tf32")
class Backend(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_proj=torch.nn.Linear(1024,256,bias=False)
        self.text_proj=torch.nn.Linear(768,256,bias=False)
        self.itm_head=torch.nn.Linear(768,2,bias=False)
        with torch.no_grad():
            torch.manual_seed(42)
            self.vision_proj.weight.copy_(torch.randn_like(self.vision_proj.weight)*.01)
            self.text_proj.weight.copy_(torch.randn_like(self.text_proj.weight)*.01)
            self.itm_head.weight[1].fill_(.001)
        self.eval().requires_grad_(False)
    def get_vision_embeds(self,x):
        t=torch.sin(x.mean((1,2,3))[:,None,None]*torch.arange(1,1025)[None,None,:]).expand(-1,49,-1).clone()
        return t,torch.ones(t.shape[:-1],dtype=torch.long)
    def get_text_embeds(self,ids,atts):return torch.sin(ids.float()[:,:,None]*torch.arange(1,769)[None,None,:]*.01)
    def get_cross_embeds(self,images,atts,*,text_embeds,text_atts):
        return text_embeds+images.mean((1,2))[:,None,None]
def adapter():return APTMAdapter(Backend(),None,None,space(),"cpu",smoke=True)
def payload(n,mod):
    torch.manual_seed(n)
    x={"global":F.normalize(torch.randn(n,256),dim=-1),
       "tokens":torch.arange(n,dtype=torch.float32)[:,None,None].expand(-1,56 if mod=="text" else 49,768 if mod=="text" else 1024).clone()/100}
    if mod=="text":x["attention"]=torch.ones(n,56,dtype=torch.long)
    return x
def batch(x,mod):return RepresentationBatch(x,space().id,mod)

def test_native_contract_not_rasa_or_irra():
    a=adapter();assert a.spec.image_size_hw==(384,128)
    assert a.spec.tokenizer.max_length==56 and a.spec.representation_dim is None
    assert a.query_independent_scores is False and a.input_gradients is False

@pytest.mark.parametrize("mod",["image","text"])
def test_native_payload(mod):validate_payload(payload(2,mod),2,mod)

@pytest.mark.parametrize("kind",["norm","dtype","nan","shape","attention","extra"])
def test_payload_errors(kind):
    x=payload(2,"text")
    if kind=="norm":x["global"]*=2
    if kind=="dtype":x["tokens"]=x["tokens"].double()
    if kind=="nan":x["tokens"][0,0,0]=float("nan")
    if kind=="shape":x["tokens"]=x["tokens"][:,:55]
    if kind=="attention":x["attention"][0,0]=2
    if kind=="extra":x["pid"]=torch.ones(2)
    with pytest.raises(ArtifactError):validate_payload(x,2,"text")

def test_pixel_preprocess_exact_and_no_quantization():
    from torchvision.transforms.functional import normalize,resize
    from torchvision.transforms import InterpolationMode
    from tbps.models.aptm import PREPROCESS
    x=torch.linspace(0,1,3*384*128).reshape(1,3,384,128)
    y=adapter().prepare_pixels(x)
    assert torch.equal(y,normalize(resize(x,[384,128],interpolation=InterpolationMode.BICUBIC,antialias=True),PREPROCESS["mean"],PREPROCESS["std"]))
    assert not torch.equal(y,adapter().prepare_pixels((x*255).round()/255))

def native_evaluation_function():
    tree=ast.parse((OFFICIAL/"reTools.py").read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="evaluation")
    fn.decorator_list=[]
    class Log:
        def __init__(self,**kw):pass
        def log_every(self,values,*a):return values
    import time,datetime
    ns={"torch":torch,"F":F,"utils":SimpleNamespace(MetricLogger=Log,get_world_size=lambda:1,get_rank=lambda:0),
        "time":time,"datetime":datetime}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),"<official APTM reTools.evaluation>","exec"),ns)
    return ns["evaluation"]

def test_independent_official_complete_evaluation_score_exact():
    # 运行官方原函数（合成模型）；包含特征、TopK、填充、全矩阵归一化，不是自比较。
    class Tokenizer:
        def __call__(self,texts,**kw):
            class Tokens(SimpleNamespace):
                def to(self,device):return self
            ids=torch.tensor([[int(t)+i for i in range(56)] for t in texts])
            return Tokens(input_ids=ids,attention_mask=torch.ones_like(ids))
    a=adapter();a.tokenizer=Tokenizer();a.pre_caption=lambda t,*args,**kw:t
    images=torch.rand(10,3,384,128)
    prepared=a.prepare_pixels(images)
    texts=["1","3","5"]
    class Loader:
        dataset=SimpleNamespace(text=texts,image=list(range(10)))
        def __iter__(self):yield prepared,torch.arange(10)
    config={"batch_size_test_text":1,"max_tokens":56,"k_test":10}
    expected=native_evaluation_function()(a.backend,Loader(),a.tokenizer,"cpu",config,SimpleNamespace(distributed=False))
    q=a.encode_texts(texts);g=a.encode_images(images)
    value=a.score(batch(q,"text"),batch(g,"image")).payload
    assert torch.allclose(value,torch.from_numpy(expected),atol=1e-6,rtol=0)
    assert a.counts["fusion_pairs"]==30

def test_full_top128_sentinel_and_no_rasa_logic():
    a=adapter();a.smoke=False
    q,g=payload(2,"text"),payload(130,"image")
    raw=a.raw_score(batch(q,"text"),batch(g,"image"))
    assert a.counts["fusion_pairs"]==256
    for i,row in enumerate(q["global"]@g["global"].T):
        selected=row.topk(128).indices
        outside=torch.ones(130,dtype=torch.bool);outside[selected]=False
        assert torch.equal(raw["itm"][i,outside],raw["itm"][i].min().expand(2))
    with pytest.raises(ArtifactError):a.raw_score(batch(q,"text"),batch(payload(10,"image"),"image"))

def test_wrong_space_rejected():
    with pytest.raises(ArtifactError):adapter().raw_score(replace(batch(payload(1,"text"),"text"),space_id=digest("rasa")),batch(payload(10,"image"),"image"))

@pytest.mark.parametrize("bad",["constant_itm","constant_global","nan","dtype","shape"])
def test_final_normalization_errors(bad):
    raw={"itm":torch.tensor([[1.,2.],[3.,7.]]),"global":torch.tensor([[.1,.2],[.8,.9]])}
    if bad=="constant_itm":raw["itm"].fill_(1)
    if bad=="constant_global":raw["global"].fill_(1)
    if bad=="nan":raw["itm"][0,0]=float("nan")
    if bad=="dtype":raw["itm"]=raw["itm"].double()
    if bad=="shape":raw["global"]=raw["global"][:1]
    with pytest.raises(ArtifactError):finalize_scores(raw)

def test_whole_query_normalization_cannot_be_done_per_chunk():
    raw={"itm":torch.tensor([[1.,2.],[100.,110.]]),"global":torch.tensor([[.1,.3],[.5,.9]])}
    whole=finalize_scores(raw)
    chunks=torch.cat([finalize_scores({k:v[i:i+1] for k,v in raw.items()}) for i in range(2)])
    assert not torch.equal(whole,chunks)

def test_native_cache_clone_and_readonly(tmp_path):
    base=tmp_path/"tbps-framework-aptm";s=ArtifactStore(base/"cache",new_base=base)
    s.publish("protocol",{"protocol.json":b"{}"},{"fixture":True})
    cache=NativeFeatureCache(s,{},space(),validate_payload,chunk=2)
    b=InputBinding(space().id,digest("m"),(digest(0),digest(1)),"text",("q0","q1"))
    original=payload(8,"text");cache.publish("clean_text",b,0,2,{k:v[:2] for k,v in original.items()})
    value=cache.load("clean_text",b,0,2)
    for k,v in value.items():
        assert torch.equal(v,original[k][:2]) and v.untyped_storage().nbytes()==v.numel()*v.element_size()
    (s.root/"clean_text-000000/features.pt").write_bytes(b"damage")
    with pytest.raises(ArtifactError):cache.load("clean_text",b,0,2)

def test_bound_set_scorer_four_modes_resume_and_global_context(tmp_path):
    f,old,m,inputs=shared.native_fixture(tmp_path)
    m=replace(m,gallery=m.gallery[:10])
    a=adapter();a.space=replace(a.space,checkpoint_sha256=f.ref.sha256)
    for key,v in tuple(inputs.items()):
        mod=v.inputs.modality;n=10 if mod=="image" else len(v.inputs.ordered_ids)
        x=payload(n,mod)
        binding=replace(v.inputs,model_space_id=a.space.id,dataset_manifest_sha256=digest(asdict(m)),
                        ordered_ids=v.inputs.ordered_ids[:n],ordered_sha256=v.inputs.ordered_sha256[:n])
        from tbps.engines.provenance import payload_digest
        inputs[key]=replace(v,inputs=binding,batch=RepresentationBatch(x,a.space.id,mod),payload_sha256=payload_digest(x),original_sha256=v.original_sha256[:n])
    # fixture Gallery <10, legal only as Smoke
    base=tmp_path/"tbps-framework-aptm";s=ArtifactStore(base/"run",new_base=base)
    s.publish("protocol",{"protocol.json":b"{}"},{"fixture":True})
    ref=FileRef(str(s.root/"protocol/manifest.json"),sha_file(s.root/"protocol/manifest.json"))
    wrapped=APTMSetScorer(a,inputs,s,ref)
    engine=EvaluationEngine(m,wrapped,direction="higher_better",execution="materialized_native",model_sources=(f.ref,))
    p,r,scores=engine.run(inputs,query_chunk=2)
    assert wrapped.new==4
    for mode,(q,g) in __import__("tbps.engines.evaluation",fromlist=["MODES"]).MODES.items():
        expected=a.score(inputs[q].batch,inputs[g].batch).payload
        assert torch.equal(scores[mode],expected)
    before=dict(a.counts);_,again,_=engine.run(inputs,query_chunk=2)
    assert a.counts==before and again==r
    key=next(x for x in s.root.iterdir() if x.name.startswith("aptm-final-"))
    (key/"score.pt").write_bytes(b"tamper")
    with pytest.raises(ArtifactError):engine.run(inputs,query_chunk=2)

@pytest.mark.parametrize("kind",["missing","tamper","runtime","checkpoint","wrong_checkpoint"])
def test_actual_source_rejects_conflicts(tmp_path,kind):
    data=json.loads(SOURCE.read_text());key=data["root"]+"/models/aptm.py"
    if kind=="missing":data["files"].pop(key)
    elif kind=="tamper":data["files"][key]="0"*64
    elif kind=="runtime":data["runtime_versions"]["torch"]="0"
    elif kind=="checkpoint":data["checkpoint"]=None
    else:data["checkpoint"]["sha256"]="0"*64
    p=tmp_path/"source.json";p.write_text(json.dumps(data))
    with pytest.raises(ArtifactError):APTMSource(p).verify()

def test_actual_import_tokenizer_namespace_rng_restore_no_load():
    source=APTMSource(SOURCE);paths=list(sys.path);rng=random.getstate()
    modules={k:v for k,v in sys.modules.items() if k=="models" or k.startswith("models.")}
    def no(*a,**k):raise AssertionError("no weights/neural/CUDA")
    with patch.object(torch,"load",no),patch.object(torch.cuda,"_lazy_init",no),patch.object(torch.nn.Module,"__call__",no):
        with native_modules(source) as (m,t,u):
            tok=official_tokenizer(t,source)
            assert m.APTM_Retrieval.__module__=="models.model_retrieval"
            assert u.pre_caption("A BLUE, shirt!",56)=="A BLUE, shirt!"
            values=tok(["A blue shirt.","A blυe shirt."],max_length=56,padding="max_length",truncation=True,return_tensors="pt")
            assert values.input_ids.shape==(2,56) and not torch.equal(*values.input_ids)
    assert sys.path==paths and random.getstate()==rng
    assert all(sys.modules.get(k)is v for k,v in modules.items())
    assert not torch.cuda.is_initialized()

def test_default_cpu_and_explicit_gpu_gate():
    with patch.object(runner,"preflight",return_value={}) as p,patch.object(runner,"run",side_effect=AssertionError("no GPU")):
        assert runner.main(["--config","fixture"])==0;p.assert_called_once_with("fixture","smoke")
    with patch.object(runner,"preflight",side_effect=AssertionError("permission first")):
        with pytest.raises(ValueError):runner.run("fixture","full",authorized=False)

def test_bootstrap_frozen_plan_no_rasa_mutation():
    from tbps import aptm_eval
    import tbps.attacks,tbps.materialization
    paths=[list(p.__path__) for p in (tbps.attacks,tbps.materialization)]
    try:
        with patch("tbps.engines.materialized_aptm.main",return_value=0):assert aptm_eval.main()==0
        assert list(tbps.materialization.__path__)==[str(aptm_eval.FROZEN/"materialization")]
    finally:
        for p,old in zip((tbps.attacks,tbps.materialization),paths):p.__path__[:]=old


def test_official_APTM_metrics_all_five_match_shared_engine():
    from tbps.engines.metrics import ranking
    from tbps.contracts import EvaluationIdentity,PersonID
    from prettytable import PrettyTable
    tree=ast.parse((OFFICIAL/"reTools.py").read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="mAP")
    fn.decorator_list=[]
    ns={"torch":torch,"PrettyTable":PrettyTable}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),"<official APTM mAP>","exec"),ns)
    g=[0,0,1,1,2,2,3,3,4,4];q=[0,1]
    x=torch.tensor([[10,8,9,7,6,5,4,3,2,1],[1,2,10,7,9,8,6,5,4,3]],dtype=torch.float32)
    native=ns["mAP"](x.numpy(),g,q)
    identity=EvaluationIdentity(("q0","q1"),tuple("g"+str(i) for i in range(10)),
        tuple(PersonID("fixture",str(i)) for i in q),tuple(PersonID("fixture",str(i)) for i in g))
    shared_,_=ranking(x,identity)
    for k,v in shared_.items():assert native[k.replace("@","")]==v

def test_raw_128_chunks_finalize_once_over_129_queries_and_resume(tmp_path):
    from tbps.models.base import ScoreBatch
    a=adapter();calls=[]
    def raw(query,gallery):
        calls.append(len(query.payload["global"]))
        ids=query.payload["global"][:,0,None]
        # Deliberately different query ranges so chunk-wise minmax would change scores.
        return {"itm":ids+torch.tensor([[1.,3.]]),
                "global":ids*.01+torch.tensor([[.2,.7]])}
    a.raw_score=raw
    q=RepresentationBatch({"global":torch.arange(129,dtype=torch.float32)[:,None]},a.space.id,"text")
    g=RepresentationBatch({"global":torch.zeros(2,1)},a.space.id,"image")
    inputs={"clean_text":SimpleNamespace(batch=q,id=digest("q")),
            "clean_gallery":SimpleNamespace(batch=g,id=digest("g"))}
    base=tmp_path/"tbps-framework-aptm";s=ArtifactStore(base/"set",new_base=base)
    s.publish("protocol",{"protocol.json":b"{}"},{"fixture":True})
    ref=FileRef(str(s.root/"protocol/manifest.json"),sha_file(s.root/"protocol/manifest.json"))
    scorer=APTMSetScorer(a,inputs,s,ref);pieces=[]
    for start,end in ((0,128),(128,129)):
        pieces.append(scorer.score(scorer.slice_query(q,start,end),g).payload)
    expected=finalize_scores(raw(q,g))
    assert torch.equal(torch.cat(pieces),expected) and calls==[128,1,129]
    scorer2=APTMSetScorer(a,inputs,s,ref)
    with patch.object(a,"raw_score",side_effect=AssertionError("must reuse completed set")):
        assert torch.equal(scorer2.score(scorer2.slice_query(q,128,129),g).payload,expected[128:])
    rawkey=next(p for p in s.root.iterdir() if p.name.startswith("aptm-raw-"))
    (rawkey/"manifest.json").write_bytes(b"bad")
    scorer3=APTMSetScorer(a,inputs,s,ref)
    with pytest.raises(ArtifactError):scorer3.score(scorer3.slice_query(q,0,1),g)


@pytest.mark.parametrize("kind",["missing","tamper","batch","precision"])
def test_final_APTM_config_rejects_changes(tmp_path,kind):
    data=json.loads((ROOT/"configs/framework/materialized_aptm.server.json").read_text())
    first=next(iter(data["producer_sources"]))
    if kind=="missing":data["producer_sources"].pop(first)
    elif kind=="tamper":data["producer_sources"][first]="0"*64
    elif kind=="batch":data["image_batch"]=2
    else:data["precision"]="fp16"
    p=tmp_path/"config.json";p.write_text(json.dumps(data))
    with pytest.raises((ArtifactError,ValueError)):runner.config(p)

def test_final_APTM_config_dry_no_model_or_cuda():
    def no(*a,**kw):raise AssertionError("no model/CUDA")
    with patch.object(torch,"load",no),patch.object(torch.cuda,"_lazy_init",no),patch.object(torch.nn.Module,"_call_impl",no):
        c=runner.config(ROOT/"configs/framework/materialized_aptm.server.json")
    assert c["output_base"]=="/home/lzf/ldx/outputs/tbps-framework-aptm-materialized-evaluation"
    assert len(c["producer_sources"])==13
