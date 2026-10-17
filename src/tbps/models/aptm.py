"""官方APTM Adapter；固定RGB/Unicode输入与攻击算法无关。"""
from contextlib import contextmanager
from pathlib import Path
import importlib, inspect, sys, random
from unittest.mock import patch
from tbps.artifacts import FileRef, checked, read_json, digest, sha_file, ArtifactError
from tbps.contracts import ModelSpec, TokenizerSpec
from tbps.models.base import ModelSpace, RepresentationBatch, ScoreBatch

CHECKPOINT_SHA="521c9bf784e7e821ade471deaafa2f25e2501950591e574dc60ca177e9be5a2c"
OFFICIAL_COMMIT="b20025445471eb4b743164a6faec29a9eccfb293"
PREPROCESS={"input":"materialized FP32 RGB [0,1]; no PIL quantization","size":[384,128],
 "resize":"Tensor BICUBIC antialias=True; original high-resolution PIL equivalence not claimed",
 "mean":[.38901278,.3651612,.34836376],"std":[.24344306,.23738699,.23368555]}
SCORING={"global":"normalized projected CLS dot product","full_k_test":128,
 "raw":"ITM positive logit; unselected=per-query minimum of sentinel1000 matrix",
 "final":"whole-query/gallery minmax(ITM)+0.002*whole-query/gallery minmax(global)",
 "precision":"fp32/math/no_tf32","fusion_batch":1,"distributed":False}
def need(ok,message):
    if not ok:raise ArtifactError(message)

class APTMSource:
    def __init__(self,path):
        self.ref=FileRef(str(Path(path).resolve()),sha_file(path))
        self.data=read_json(path,"tbps-aptm-source-1")
        value=self.data["checkpoint"]
        self.checkpoint=FileRef(**value) if value else None
    def verify(self,*,require_checkpoint=True):
        checked(self.ref)
        import importlib.metadata as m, subprocess
        need(all(m.version(k)==v for k,v in self.data["runtime_versions"].items()),"APTM runtime mismatch")
        need(self.data["official_commit"]==OFFICIAL_COMMIT,"official APTM commit mismatch")
        root=Path(self.data["root"])
        expected={str(p) for p in root.rglob("*") if p.is_file() and
                  (p.suffix==".py" or (p.parent.name=="configs" and p.suffix in (".json",".yaml")))}
        expected|={str(Path(self.data["bert"])/n) for n in
                   ("config.json","model.safetensors","tokenizer_config.json","tokenizer.json","vocab.txt")}
        need(set(self.data["files"])==expected,"APTM complete sources/config/BERT required")
        for path,sha in self.data["files"].items():checked(FileRef(path,sha))
        need(subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()==OFFICIAL_COMMIT
             and not subprocess.check_output(["git","status","--porcelain"],cwd=root,text=True),"official APTM checkout changed")
        if require_checkpoint:
            need(self.checkpoint is not None,"APTM official CUHK checkpoint missing; bind verified SHA first")
            need(self.checkpoint.sha256==CHECKPOINT_SHA,"locked official CUHK APTM checkpoint")
            checked(self.checkpoint)
        elif self.checkpoint is not None:
            need(self.checkpoint.sha256==CHECKPOINT_SHA,"locked official CUHK APTM checkpoint")
            checked(self.checkpoint)
    @property
    def space(self):
        need(self.checkpoint is not None,"checkpoint required for APTM ModelSpace")
        return ModelSpace("aptm",self.checkpoint.sha256,digest(self.data),digest(PREPROCESS),
            digest({"vocab":self.data["files"][self.data["bert"]+"/vocab.txt"],
                    "pre_caption":self.data["files"][self.data["root"]+"/dataset/utils.py"],
                    "max_words":56,"max_tokens":56}),digest(SCORING),"fp32/math/no_tf32")

@contextmanager
def native_modules(source,*,require_checkpoint=True):
    """仅边界兼容旧Transformers文档/Tokenizer API，恢复命名空间与EDA import的RNG副作用。"""
    source.verify(require_checkpoint=require_checkpoint)
    names=lambda n:any(n==p or n.startswith(p+".") for p in ("models","dataset","utils"))
    before={k:v for k,v in sys.modules.items() if names(k)}
    paths=list(sys.path);rng=random.getstate()
    import transformers.file_utils as docs
    original=docs.add_code_sample_docstrings
    def compatible(*args,**kw):
        if "tokenizer_class" not in inspect.signature(original).parameters:kw.pop("tokenizer_class",None)
        return original(*args,**kw)
    try:
        for k in before:sys.modules.pop(k,None)
        sys.path.insert(0,source.data["root"])
        with patch.object(docs,"add_code_sample_docstrings",compatible):
            native=importlib.import_module("models.model_retrieval")
            tokenizer=importlib.import_module("models.tokenization_bert")
            utils=importlib.import_module("dataset.utils")
            yield native,tokenizer,utils
    finally:
        for k in tuple(sys.modules):
            if names(k):sys.modules.pop(k,None)
        sys.modules.update(before);sys.path[:]=paths;random.setstate(rng)

def official_tokenizer(module,source):
    cls=module.BertTokenizer
    def init(self,vocab_file,*args,**kw):
        self.vocab=module.load_vocab(vocab_file)
        cls.__init__(self,vocab_file,*args,**kw)
    compatible=type("BertTokenizer",(cls,),{"__init__":init})
    return compatible.from_pretrained(source.data["bert"],local_files_only=True)

def compact(x):return {k:v.detach().cpu().clone() for k,v in x.items()}
def validate_payload(x,n,modality):
    import torch
    need(modality in ("image","text"),"APTM modality")
    need(type(x)is dict and set(x)==({"global","tokens","attention"} if modality=="text" else {"global","tokens"}),"APTM fields")
    for v in x.values():
        need(isinstance(v,torch.Tensor) and v.device.type=="cpu" and not v.requires_grad and len(v)==n
             and torch.isfinite(v).all(),"APTM finite CPU payload")
    need(x["global"].shape==(n,256) and x["global"].dtype==torch.float32,"APTM projected CLS")
    need(x["tokens"].shape==(n,56,768) if modality=="text" else x["tokens"].shape==(n,49,1024),"APTM native tokens")
    need(x["tokens"].dtype==torch.float32,"APTM token precision")
    need(torch.allclose(x["global"].norm(dim=-1),torch.ones(n),atol=1e-5,rtol=0),"APTM global normalization")
    if modality=="text":
        need(x["attention"].shape==(n,56) and x["attention"].dtype==torch.int64 and
             ((x["attention"]==0)|(x["attention"]==1)).all(),"APTM attention")

def finalize_scores(raw):
    """reTools.evaluation的全矩阵归一化；常量矩阵原生会NaN，明确拒绝而非添加epsilon。"""
    import torch
    need(type(raw)is dict and set(raw)=={"itm","global"},"raw APTM score fields")
    a,b=raw["itm"],raw["global"]
    need(a.ndim==2 and a.shape==b.shape and a.numel()>0,"raw APTM score shape")
    for x in (a,b):
        need(x.dtype==torch.float32 and x.device.type=="cpu" and not x.requires_grad and torch.isfinite(x).all(),"raw score FP32 finite")
        need((x.max()-x.min()).item()>0,"native APTM constant-score normalization undefined")
    return (a-a.min())/(a.max()-a.min())+.002*((b-b.min())/(b.max()-b.min()))

class APTMAdapter:
    score_direction="higher_better";execution_kind="materialized_native"
    query_independent_scores=False;input_gradients=False
    def __init__(self,backend,tokenizer,pre_caption,space,device,*,smoke=False):
        self.backend,self.tokenizer,self.pre_caption=backend,tokenizer,pre_caption
        self.space,self.device,self.smoke=space,device,smoke;self.source_verified=True
        self.spec=ModelSpec("aptm","official-"+OFFICIAL_COMMIT,(384,128),
            TokenizerSpec("official BERT WordPiece",space.tokenizer_sha256,56,"WordPiece token","retain_and_report"),
            None,"fusion","projected CLS l2; Swin/BERT local tokens unnormalized",
            frozenset({"image_encoding","text_encoding","tokenize","score","local_features","fusion"}))
        self.counts={"image_samples":0,"text_samples":0,"fusion_pairs":0,"score_calls":0}
    def prepare_pixels(self,x):
        import torch
        from torchvision.transforms.functional import resize,normalize
        from torchvision.transforms import InterpolationMode
        need(x.ndim==4 and x.shape[1]==3 and x.dtype==torch.float32 and torch.isfinite(x).all()
             and x.min()>=0 and x.max()<=1,"raw FP32 RGB [0,1]")
        return normalize(resize(x,[384,128],interpolation=InterpolationMode.BICUBIC,antialias=True),
                         PREPROCESS["mean"],PREPROCESS["std"])
    def tokenize(self,texts):
        need(all(type(t)is str for t in texts),"Caption-only text inputs")
        return self.tokenizer([self.pre_caption(t,56,icfg_rstp=False,is_eda=False) for t in texts],
                    padding="max_length",truncation=True,max_length=56,return_tensors="pt")
    def encode_images(self,x):
        import torch,torch.nn.functional as F
        with torch.inference_mode():
            tokens,_=self.backend.get_vision_embeds(self.prepare_pixels(x.to(self.device)))
            global_=F.normalize(self.backend.vision_proj(tokens[:,0,:]),dim=-1)
        self.counts["image_samples"]+=len(x)
        return compact({"global":global_,"tokens":tokens})
    def encode_texts(self,texts):
        import torch,torch.nn.functional as F
        batch=self.tokenize(texts).to(self.device)
        with torch.inference_mode():
            tokens=self.backend.get_text_embeds(batch.input_ids,batch.attention_mask)
            global_=F.normalize(self.backend.text_proj(tokens[:,0,:]),dim=-1)
        self.counts["text_samples"]+=len(texts)
        return compact({"global":global_,"tokens":tokens,"attention":batch.attention_mask})
    def raw_score(self,query,gallery):
        import torch
        for b,mod in ((query,"text"),(gallery,"image")):
            need(type(b)is RepresentationBatch and b.space_id==self.space.id and b.modality==mod,"APTM space/modality")
            validate_payload(b.payload,len(b.payload["global"]),mod)
        q,g=query.payload,gallery.payload;ng=len(g["global"])
        need(ng>=128 or (self.smoke and ng<=10),"APTM Full requires Top128; only Smoke may reduce")
        with torch.inference_mode():
            sims=q["global"].to(self.device)@g["global"].to(self.device).T
            itm=torch.full(sims.shape,1000.,dtype=torch.float32,device=self.device)
            for i,row in enumerate(sims):
                top=row.topk(min(128,ng),dim=0).indices
                for idx in top.cpu().tolist():
                    image=g["tokens"][idx:idx+1].to(self.device)
                    out=self.backend.get_cross_embeds(image,torch.ones(image.shape[:-1],dtype=torch.long,device=self.device),
                        text_embeds=q["tokens"][i:i+1].to(self.device),text_atts=q["attention"][i:i+1].to(self.device))
                    itm[i,idx]=self.backend.itm_head(out[:,0,:])[:,1][0]
                    self.counts["fusion_pairs"]+=1
            minima=itm.min(dim=1).values[:,None].expand_as(itm)
            itm=torch.where(itm==1000.,minima,itm)
        self.counts["score_calls"]+=1
        return compact({"itm":itm,"global":sims})
    def score(self,query,gallery):
        # 全Query调用合法；不能让Engine把此Adapter误当成Query独立评分。
        return ScoreBatch(finalize_scores(self.raw_score(query,gallery)),self.space.id,self.space.score_id)

@contextmanager
def load_aptm(source,device="cuda:0",*,smoke=False):
    import torch
    import yaml
    with native_modules(source) as (native,tok,utils):
        c=yaml.safe_load(Path(source.data["root"]+"/configs/Retrieval_cuhk.yaml").read_text())
        c.update(vision_config=source.data["root"]+"/configs/config_swinB_384.json",
                 text_config=source.data["root"]+"/configs/config_bert.json",text_encoder=source.data["bert"])
        model=native.APTM_Retrieval(c)
        # 同官方is_eval=True，不做预训练插值或键改写；严格拒绝不匹配的模型。
        envelope=torch.load(source.checkpoint.path,map_location="cpu",weights_only=True)
        state=envelope["model"] if "model" in envelope else envelope
        model.load_state_dict(state,strict=True);del state,envelope
        model.eval().requires_grad_(False).to(device)
        adapter=APTMAdapter(model,official_tokenizer(tok,source),utils.pre_caption,source.space,device,smoke=smoke)
        actual={"image_samples":0,"text_samples":0,"fusion_pairs":0}
        def image_hook(module,args):actual["image_samples"]+=len(args[0])
        def text_hook(module,args,kwargs):
            if kwargs.get("mode")=="fusion":actual["fusion_pairs"]+=len(kwargs["encoder_embeds"])
            else:actual["text_samples"]+=len(args[0])
        hooks=(model.vision_encoder.register_forward_pre_hook(image_hook),
               model.text_encoder.bert.register_forward_pre_hook(text_hook,with_kwargs=True))
        adapter.neural_counts=actual
        try:yield adapter
        finally:
            for h in hooks:h.remove()
