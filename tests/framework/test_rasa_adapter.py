"""合成CPU验证RaSa原生融合、来源、结构化缓存；不执行GPU或真实模型前向。"""
from dataclasses import replace,asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
import ast,hashlib,json,sys
import pytest
import torch
import torch.nn.functional as F
from tbps.artifacts import ArtifactError,ArtifactStore,FileRef,sha_file,digest,json_bytes
from tbps.models.base import ModelSpace,RepresentationBatch,InputBinding
from tbps.models.rasa import RasaAdapter,validate_payload,RasaSource,native_modules,official_tokenizer
from tbps.engines.native_cache import NativeFeatureCache
from tbps.engines.native_score_cache import ChunkCachedScorer
from tbps.engines.provenance import payload_digest,AttackOrigin
from tbps.engines.evaluation import EvaluationEngine,MODES,publish_result,load_result
from tbps.engines import materialized_rasa as runner
import test_stage4a_engine as fixtures

def space(checkpoint=None):
    return ModelSpace("rasa",checkpoint or digest("weight"),*(digest(i) for i in range(4)),"fp32/math/no_tf32")

class Text(torch.nn.Module):
    def __init__(self):super().__init__();self.bert=self.call
    def call(self,input_ids=None,attention_mask=None,encoder_embeds=None,encoder_hidden_states=None,**kw):
        if encoder_embeds is None:
            x=input_ids.float()[:,:,None].repeat(1,1,768)/1000
        else:
            x=encoder_embeds+encoder_hidden_states.mean((1,2))[:,None,None]
        return SimpleNamespace(last_hidden_state=x)
class Backend(torch.nn.Module):
    def __init__(self):
        super().__init__();self.text_encoder=Text()
        self.vision_proj=torch.nn.Linear(768,256,bias=False);self.text_proj=torch.nn.Linear(768,256,bias=False)
        self.itm_head=torch.nn.Linear(768,2,bias=False)
        with torch.no_grad():
            self.vision_proj.weight.fill_(.001);self.text_proj.weight.fill_(.002)
            self.itm_head.weight[0].fill_(-.001);self.itm_head.weight[1].fill_(.001)
        self.eval().requires_grad_(False)
    def visual_encoder(self,x):return x.mean((1,2,3))[:,None,None].expand(-1,577,768).clone()
def payload(n,mod):
    g=F.normalize(torch.arange(1,n*256+1,dtype=torch.float32).reshape(n,256),dim=-1)
    tokens=torch.arange(n,dtype=torch.float32)[:,None,None].expand(-1,50 if mod=="text" else 577,768).clone()/100
    x={"global":g,"tokens":tokens}
    if mod=="text":x["attention"]=torch.ones(n,50,dtype=torch.long)
    return x
def adapter():return RasaAdapter(Backend(),None,None,space(),"cpu",smoke=True)

def test_model_contract_is_fusion_not_512_cosine():
    a=adapter()
    assert a.spec.scoring=="fusion" and a.spec.representation_dim is None
    assert a.spec.image_size_hw==(384,384) and a.spec.tokenizer.max_length==50
    assert a.input_gradients is False

@pytest.mark.parametrize("mod",["image","text"])
def test_native_payload_shapes_norm_and_dtype(mod):
    validate_payload(payload(2,mod),2,mod)

@pytest.mark.parametrize("kind",["nan","dtype","norm","tokens","mask"])
def test_native_payload_invalid_rejected(kind):
    x=payload(2,"text")
    if kind=="nan":x["tokens"][0,0,0]=float("nan")
    if kind=="dtype":x["global"]=x["global"].double()
    if kind=="norm":x["global"]*=2
    if kind=="tokens":x["tokens"]=x["tokens"][:,:49]
    if kind=="mask":x["attention"][0,0]=2
    with pytest.raises(ArtifactError):validate_payload(x,2,"text")

def test_native_pixels_preserve_FP32_no_quantization():
    from torchvision.transforms.functional import resize,normalize
    from torchvision.transforms import InterpolationMode
    from tbps.models.rasa import PREPROCESS
    x=torch.linspace(0,1,3*384*128).reshape(1,3,384,128)
    actual=adapter().prepare_pixels(x)
    expected=normalize(resize(x,[384,384],interpolation=InterpolationMode.BICUBIC,antialias=True),
                       PREPROCESS["mean"],PREPROCESS["std"])
    assert torch.equal(actual,expected)
    assert not torch.equal(actual,adapter().prepare_pixels((x*255).round()/255))

def test_native_score_independent_official_fusion_expression_exact():
    a=adapter();q=payload(2,"text");g=payload(10,"image")
    actual=a.score(RepresentationBatch(q,a.space.id,"text"),RepresentationBatch(g,a.space.id,"image")).payload
    # 独立按官方Retrieval.py逐Query同时融合全部候选；Adapter按pair microbatch。
    sims=q["global"]@g["global"].T;expected=torch.full((2,10),-100.)
    with torch.inference_mode():
        for i,row in enumerate(sims):
            indices=row.topk(10).indices;image=g["tokens"][indices]
            output=a.backend.text_encoder.bert(encoder_embeds=q["tokens"][i:i+1].repeat(10,1,1),
                attention_mask=q["attention"][i:i+1].repeat(10,1),
                encoder_hidden_states=image,encoder_attention_mask=torch.ones(image.shape[:-1],dtype=torch.long),
                return_dict=True,mode="fusion")
            expected[i,indices]=a.backend.itm_head(output.last_hidden_state[:,0,:])[:,1]
    assert torch.allclose(actual,expected,atol=1e-6,rtol=0)
    assert a.counts["fusion_pairs"]==20

def test_full_does_not_silently_reduce_top128():
    a=adapter();a.smoke=False
    with pytest.raises(ArtifactError):a.score(RepresentationBatch(payload(1,"text"),a.space.id,"text"),
                                            RepresentationBatch(payload(10,"image"),a.space.id,"image"))

@pytest.mark.parametrize("kind",["space","modality"])
def test_foreign_IRRA_or_wrong_modality_rejected(kind):
    a=adapter();q=RepresentationBatch(payload(1,"text"),a.space.id,"text");g=RepresentationBatch(payload(2,"image"),a.space.id,"image")
    if kind=="space":g=replace(g,space_id=digest("IRRA"))
    else:g=replace(g,modality="text")
    with pytest.raises(ArtifactError):a.score(q,g)

def test_structured_cache_clone_storage_and_tamper(tmp_path):
    base=tmp_path/"tbps-framework-rasa";store=ArtifactStore(base/"cache",new_base=base)
    store.publish("protocol",{"protocol.json":json_bytes({"fixture":True})},{"fixture":True})
    a=adapter();cache=NativeFeatureCache(store,{"fixture":True},a.space,validate_payload,chunk=2)
    b=InputBinding(a.space.id,digest("m"),tuple(digest(i) for i in range(4)),"text",tuple("q"+str(i) for i in range(4)))
    original=payload(10,"text")
    for start in [0,2]:cache.publish("clean_text",b,start,start+2,{k:v[start:start+2] for k,v in original.items()})
    value=cache.load("clean_text",b,0,2)
    for k,x in value.items():
        assert torch.equal(x,original[k][:2]) and x.untyped_storage().nbytes()==x.numel()*x.element_size()
    assert torch.equal(cache.gather("clean_text",b).batch.payload["tokens"],original["tokens"][:4])
    with pytest.raises(ArtifactError):cache.load("clean_text",replace(b,model_space_id=digest("other")),0,2)
    (store.root/"clean_text-000000/features.pt").write_bytes(b"damaged")
    with pytest.raises(ArtifactError):cache.load("clean_text",b,0,2)

def native_fixture(tmp_path):
    f=fixtures.Fixture(tmp_path,dim=3)
    ref=f.ref
    manifest=replace(f.manifest,dataset_id="cuhk_pedes",
        gallery=tuple(replace(x,pid=replace(x.pid,dataset="cuhk_pedes")) for x in f.manifest.gallery),
        queries=tuple(replace(x,pid=replace(x.pid,dataset="cuhk_pedes")) for x in f.manifest.queries))
    a=f.adapter;a.execution_kind="materialized_native";a.source_verified=True
    a.space=replace(a.space,checkpoint_sha256=ref.sha256)
    inputs={}
    for k,b in f.inputs.items():
        binding=replace(b.inputs,model_space_id=a.space.id,dataset_manifest_sha256=digest(asdict(manifest)))
        batch=replace(b.batch,space_id=a.space.id)
        inputs[k]=replace(b,batch=batch,inputs=binding)
    return f,a,manifest,inputs

def test_new_native_execution_four_modes_and_score_resume(tmp_path):
    f,a,m,inputs=native_fixture(tmp_path)
    base=tmp_path/"tbps-framework-rasa";store=ArtifactStore(base/"native",new_base=base)
    store.publish("protocol",{"protocol.json":b"{}"},{"fixture":True})
    ref=FileRef(str(store.root/"protocol/manifest.json"),sha_file(store.root/"protocol/manifest.json"))
    wrapped=ChunkCachedScorer(a,inputs,store,ref)
    engine=EvaluationEngine(m,wrapped,direction="higher_better",execution="materialized_native",model_sources=(f.ref,))
    p,r,s=engine.run(inputs,query_chunk=2)
    assert wrapped.new==8 and wrapped.hits==0
    expected=EvaluationEngine(m,a,direction="higher_better",execution="materialized_native",model_sources=(f.ref,)).run(inputs,query_chunk=2)[1]
    assert r["groups"]==expected["groups"]
    calls=len(a.calls);_,again,_=engine.run(inputs,query_chunk=2)
    assert wrapped.hits==8 and len(a.calls)==calls and again["groups"]==r["groups"]
    publish_result(store,"result",p,r,s,sources=(ref,))
    assert load_result(store,"result",expected_protocol_sha256=digest(p))[1]==r

def test_native_gate_rejects_unverified_or_missing_checkpoint(tmp_path):
    f,a,m,inputs=native_fixture(tmp_path)
    with pytest.raises(ArtifactError):EvaluationEngine(m,a,direction="higher_better",execution="materialized_native")
    a.source_verified=False
    with pytest.raises(ArtifactError):EvaluationEngine(m,a,direction="higher_better",execution="materialized_native",model_sources=(f.ref,))

def test_other_attack_origin_supported_without_adapter_changes(tmp_path):
    f=fixtures.Fixture(tmp_path)
    b=f.inputs["adv_gallery"]
    origin=replace(b.origin,attack_id="ap_attack")
    f.inputs["adv_gallery"]=replace(b,origin=origin)
    assert f.engine().run(f.inputs)[1]["groups"]
    assert not hasattr(adapter(),"attack") and not hasattr(adapter(),"pid")

def test_GPU_permission_before_preflight_or_loading():
    with patch.object(runner,"preflight",side_effect=AssertionError("must not preflight")):
        with pytest.raises(ValueError):runner.run("fixture","full",authorized=False)

def test_default_entry_is_CPU_only():
    with patch.object(runner,"preflight",return_value={}) as pre,patch.object(runner,"run",side_effect=AssertionError("GPU forbidden")):
        assert runner.main(["--config","fixture"])==0
        pre.assert_called_once_with("fixture","smoke")

def test_official_RaSa_itm_eval_vs_shared_metrics_multipoitive_exact():
    from tbps.engines.metrics import ranking
    from tbps.contracts import EvaluationIdentity,PersonID
    source=Path("/home/lzf/ldx/external/RaSa/Retrieval.py").read_text()
    tree=ast.parse(source)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="itm_eval")
    fn.decorator_list=[]
    namespace={"torch":torch}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),"<official RaSa itm_eval fixture>","exec"),namespace)
    gallery=[0,0,1,1,2,2,3,3,4,4];query=[0,1]
    scores=torch.tensor([[10,8,9,7,6,5,4,3,2,1],[1,2,10,7,9,8,6,5,4,3]],dtype=torch.float32)
    native=namespace["itm_eval"](scores,gallery,query,True)
    identity=EvaluationIdentity(("q0","q1"),tuple("g"+str(i) for i in range(10)),
        tuple(PersonID("fixture",str(i)) for i in query),tuple(PersonID("fixture",str(i)) for i in gallery))
    shared,_=ranking(scores,identity)
    assert native["mAP"]==shared["mAP"]
    for n in [1,5,10]:assert native["r"+str(n)]==shared["R@"+str(n)]
