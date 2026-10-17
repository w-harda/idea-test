"""官方RaSa薄Adapter：原生全局候选选择+局部Token融合，不依赖任何攻击器。"""
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import importlib, inspect, sys
from unittest.mock import patch
from tbps.artifacts import FileRef,checked,read_json,digest,sha_file,ArtifactError
from tbps.contracts import ModelSpec,TokenizerSpec
from tbps.models.base import ModelSpace,RepresentationBatch,ScoreBatch

CHECKPOINT_SHA="bc85da09c2991d5de503c2c2ac4f032e20cc536fec5094f0acff45e93d0104d2"
OFFICIAL_COMMIT="bd16aa1a15f149548a90d196fcf10a27d7ab7c66"
PREPROCESS={"input":"materialized FP32 RGB [0,1]; no PIL quantization","size":[384,384],
    "resize":"torchvision Tensor BICUBIC antialias=True; PIL-equivalence not claimed",
    "mean":[.48145466,.4578275,.40821073],"std":[.26862954,.26130258,.27577711]}
SCORING={"global":"normalized projected CLS dot product","full_k_test":128,
    "final":"itm_head(fusion CLS)[:,1]; no global score addition","unselected":-100.,
    "precision":"fp32/math/no_tf32","fusion_batch":1}
def need(ok,message):
    if not ok:raise ArtifactError(message)

class RasaSource:
    def __init__(self,path):
        self.ref=FileRef(str(Path(path).resolve()),sha_file(path));self.data=read_json(path,"tbps-rasa-source-1")
        self.checkpoint=FileRef(**self.data["checkpoint"])
    def verify(self):
        checked(self.ref);checked(self.checkpoint)
        import importlib.metadata
        need(all(importlib.metadata.version(k)==v for k,v in self.data["runtime_versions"].items()),
             "RaSa runtime dependency version mismatch")
        need(self.checkpoint.sha256==CHECKPOINT_SHA and self.data["official_commit"]==OFFICIAL_COMMIT,"locked official RaSa source/weight")
        expected={str(p) for p in Path(self.data["root"]).rglob("*") if p.is_file() and
                  (p.suffix==".py" or (p.parent.name=="configs" and p.suffix in (".json",".yaml")))}
        expected|={str(Path(self.data["bert"])/name) for name in
                   ("config.json","model.safetensors","tokenizer_config.json","tokenizer.json","vocab.txt")}
        need(set(self.data["files"])==expected,"complete official runtime/config/BERT sources required")
        for path,sha in self.data["files"].items():checked(FileRef(path,sha))
        import subprocess
        need(subprocess.check_output(["git","rev-parse","HEAD"],cwd=self.data["root"],text=True).strip()==OFFICIAL_COMMIT
             and not subprocess.check_output(["git","status","--porcelain"],cwd=self.data["root"],text=True),"official RaSa checkout changed")
    @property
    def space(self):
        return ModelSpace("rasa",self.checkpoint.sha256,digest(self.data),
            digest(PREPROCESS),digest({"vocab":self.data["files"][self.data["bert"]+"/vocab.txt"],
                "pre_caption":self.data["files"][self.data["root"]+"/dataset/utils.py"],"max_words":50,"max_tokens":50}),
            digest(SCORING),"fp32/math/no_tf32")

@contextmanager
def native_modules(source):
    """隔离官方models/dataset/utils；仅兼容已删除的文档装饰器参数，不改模型计算。"""
    source.verify()
    names=lambda n:n in ("models","dataset","utils") or n.startswith(("models.","dataset."))
    before={k:v for k,v in sys.modules.items() if names(k)};paths=list(sys.path)
    for k in before:sys.modules.pop(k,None)
    import transformers.file_utils as docs
    original=docs.add_code_sample_docstrings
    def compatible(*args,**kw):
        if "tokenizer_class" not in inspect.signature(original).parameters:kw.pop("tokenizer_class",None)
        return original(*args,**kw)
    try:
        sys.path.insert(0,source.data["root"])
        with patch.object(docs,"add_code_sample_docstrings",compatible):
            model=importlib.import_module("models.model_person_search")
            tokenizer=importlib.import_module("models.tokenization_bert")
            utils=importlib.import_module("dataset.utils")
            yield model,tokenizer,utils
    finally:
        for k in tuple(sys.modules):
            if names(k):sys.modules.pop(k,None)
        sys.modules.update(before);sys.path[:]=paths

def official_tokenizer(module,source):
    # transformers>=4.44父类提前调用get_vocab；仅提前载入同一官方词表，分词实现仍来自原类。
    original=module.BertTokenizer
    def init(self,vocab_file,*args,**kwargs):
        self.vocab=module.load_vocab(vocab_file)
        original.__init__(self,vocab_file,*args,**kwargs)
    compatible=type("BertTokenizer",(original,),{"__init__":init})
    return compatible.from_pretrained(source.data["bert"],local_files_only=True)

def clone_payload(x):
    return {k:v.detach().cpu().clone() for k,v in x.items()}

def validate_payload(x,n,modality):
    import torch
    keys={"global","tokens"}|({"attention"} if modality=="text" else set())
    need(type(x)is dict and set(x)==keys,"RaSa native representation fields")
    for v in x.values():
        need(isinstance(v,torch.Tensor) and v.device.type=="cpu" and not v.requires_grad and len(v)==n and torch.isfinite(v).all(),"CPU finite native payload")
    need(x["global"].shape==(n,256) and x["global"].dtype==torch.float32,"RaSa projected global shape/dtype")
    need(x["tokens"].shape==(n,50 if modality=="text" else 577,768) and x["tokens"].dtype==torch.float32,"RaSa local Token shape/dtype")
    need(torch.allclose(x["global"].norm(dim=-1),torch.ones(n),atol=1e-5,rtol=0),"RaSa global normalization")
    if modality=="text":
        need(x["attention"].shape==(n,50) and x["attention"].dtype==torch.int64 and
             ((x["attention"]==0)|(x["attention"]==1)).all(),"BERT attention mask")

class RasaAdapter:
    score_direction="higher_better"
    execution_kind="materialized_native"
    query_independent_scores=True
    input_gradients=False
    def __init__(self,backend,tokenizer,pre_caption,space,device,*,smoke=False):
        self.backend,self.tokenizer,self.pre_caption=backend,tokenizer,pre_caption
        self.space,self.device,self.smoke=space,device,smoke;self.source_verified=True
        self.spec=ModelSpec("rasa","official-"+OFFICIAL_COMMIT,(384,384),
            TokenizerSpec("official BERT WordPiece",space.tokenizer_sha256,50,"WordPiece token","retain_and_report"),
            None,"fusion","projected CLS l2; local tokens unnormalized",
            frozenset({"image_encoding","text_encoding","tokenize","score","local_features","fusion"}))
        self.counts={"image_samples":0,"text_samples":0,"fusion_pairs":0,"score_calls":0}
    def prepare_pixels(self,pixels):
        import torch
        from torchvision.transforms.functional import resize,normalize
        from torchvision.transforms import InterpolationMode
        need(pixels.ndim==4 and pixels.shape[1]==3 and pixels.dtype==torch.float32 and torch.isfinite(pixels).all()
             and pixels.min()>=0 and pixels.max()<=1,"raw FP32 RGB [0,1]")
        return normalize(resize(pixels,[384,384],interpolation=InterpolationMode.BICUBIC,antialias=True),PREPROCESS["mean"],PREPROCESS["std"])
    def tokenize(self,texts):
        need(all(type(t)is str for t in texts),"Caption-only tokenizer inputs")
        values=[self.pre_caption(t,50) for t in texts]
        return self.tokenizer(values,padding="max_length",truncation=True,max_length=50,return_tensors="pt")
    def encode_images(self,pixels):
        import torch
        import torch.nn.functional as F
        with torch.inference_mode():
            tokens=self.backend.visual_encoder(self.prepare_pixels(pixels.to(self.device)))
            global_=F.normalize(self.backend.vision_proj(tokens[:,0,:]),dim=-1)
        self.counts["image_samples"]+=len(pixels)
        return clone_payload({"global":global_,"tokens":tokens})
    def encode_texts(self,texts):
        import torch
        import torch.nn.functional as F
        batch=self.tokenize(texts).to(self.device)
        with torch.inference_mode():
            tokens=self.backend.text_encoder.bert(batch.input_ids,attention_mask=batch.attention_mask,mode="text").last_hidden_state
            global_=F.normalize(self.backend.text_proj(tokens[:,0,:]),dim=-1)
        self.counts["text_samples"]+=len(texts)
        return clone_payload({"global":global_,"tokens":tokens,"attention":batch.attention_mask})
    def slice_query(self,batch,start,end):
        need(batch.space_id==self.space.id and batch.modality=="text","native query space")
        return RepresentationBatch({k:v[start:end].detach().clone() for k,v in batch.payload.items()},self.space.id,"text")
    def score(self,query,gallery):
        import torch
        for b,mod in [(query,"text"),(gallery,"image")]:
            need(type(b)is RepresentationBatch and b.space_id==self.space.id and b.modality==mod,"RaSa source/modality mismatch")
            validate_payload(b.payload,len(b.payload["global"]),mod)
        q,g=query.payload,gallery.payload;ng=len(g["global"])
        need(ng>=128 or (self.smoke and ng<=10),"Full requires original Top-128; small Smoke only")
        k=min(128,ng);result=torch.full((len(q["global"]),ng),-100.,dtype=torch.float32)
        with torch.inference_mode():
            sims=q["global"].to(self.device)@g["global"].to(self.device).T
            for i,row in enumerate(sims):
                indices=row.topk(k=k,dim=0).indices
                # 原生逐pair融合，microbatch1控制显存；最终分数只取ITM，不加入global相似度。
                for index in indices.cpu().tolist():
                    image=g["tokens"][index:index+1].to(self.device)
                    output=self.backend.text_encoder.bert(encoder_embeds=q["tokens"][i:i+1].to(self.device),
                        attention_mask=q["attention"][i:i+1].to(self.device),encoder_hidden_states=image,
                        encoder_attention_mask=torch.ones(image.shape[:-1],dtype=torch.long,device=self.device),
                        return_dict=True,mode="fusion")
                    result[i,index]=self.backend.itm_head(output.last_hidden_state[:,0,:])[:,1].cpu()[0]
                    self.counts["fusion_pairs"]+=1
        self.counts["score_calls"]+=1
        return ScoreBatch(result,self.space.id,self.space.score_id)

@contextmanager
def load_rasa(source,device="cuda:0",*,smoke=False):
    """官方构造器/权重恢复；全模型eval冻结，不创建Optimizer或执行训练forward。"""
    import torch,json
    with native_modules(source) as (native,tok,utils):
        config={"image_res":384,"vision_width":768,"embed_dim":256,"temp":.07,
                "mlm_probability":.15,"mrtd_mask_probability":.3,"queue_size":65536,"momentum":.995,
                "bert_config":source.data["root"]+"/configs/config_bert.json"}
        tokenizer=official_tokenizer(tok,source)
        model=native.ALBEF(config=config,text_encoder=source.data["bert"],tokenizer=tokenizer)
        state=torch.load(source.checkpoint.path,map_location="cpu",weights_only=True)
        need(set(state)=={"model"},"official RaSa checkpoint envelope")
        model.load_state_dict(state["model"],strict=True);del state
        model.eval().requires_grad_(False);model.to(device)
        adapter=RasaAdapter(model,tokenizer,utils.pre_caption,source.space,device,smoke=smoke)
        actual={"image_samples":0,"text_samples":0,"fusion_pairs":0}
        def image_hook(module,args):actual["image_samples"]+=len(args[0])
        def text_hook(module,args,kwargs):
            if kwargs.get("mode")=="fusion":actual["fusion_pairs"]+=len(kwargs["encoder_embeds"])
            else:actual["text_samples"]+=len(args[0])
        hooks=(model.visual_encoder.register_forward_pre_hook(image_hook),
               model.text_encoder.bert.register_forward_pre_hook(text_hook,with_kwargs=True))
        adapter.neural_counts=actual
        try:yield adapter
        finally:
            for hook in hooks:hook.remove()
