"""官方IRRA双塔Adapter及两种冻结View；真正加载必须显式授权。"""
from contextlib import contextmanager
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from ..artifacts import ArtifactError, digest, sha_file
from ..contracts import FileRef, RepresentationArtifact, ScoreArtifact
from ..config import IRRA
from .base import ModelSpace, RepresentationBatch, ScoreBatch, InputBinding, RepresentationRecord, ScoreRecord
from .irra_io import PREPROCESS, METRICS, normalize_pixels, official_tokenizer
from .irra_source import load_official_model

@contextmanager
def fp32_runtime():
    """复现旧IRRA fp32/math执行设置；调用结束恢复全局开关。无需初始化CUDA。"""
    import torch
    c=torch.backends.cuda
    previous=(c.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,torch.backends.cudnn.benchmark,
              torch.backends.mha.get_fastpath_enabled(),c.flash_sdp_enabled(),c.mem_efficient_sdp_enabled(),c.math_sdp_enabled())
    try:
        c.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        torch.backends.mha.set_fastpath_enabled(False)
        c.enable_flash_sdp(False);c.enable_mem_efficient_sdp(False);c.enable_math_sdp(True)
        yield
    finally:
        c.matmul.allow_tf32=previous[0];torch.backends.cudnn.allow_tf32=previous[1];torch.backends.cudnn.benchmark=previous[2]
        torch.backends.mha.set_fastpath_enabled(previous[3])
        c.enable_flash_sdp(previous[4]);c.enable_mem_efficient_sdp(previous[5]);c.enable_math_sdp(previous[6])

class IRRAAdapter:
    def __init__(self,backend,tokenizer,*,spec,space):
        """显式注入Backend供CPU测试；真实来源必须使用from_source，不把Mock标成真实来源。"""
        if spec.id!="irra_fixture":raise ArtifactError("real IRRA must use from_source; injected backend is fixture-only")
        self._initialize(backend,tokenizer,spec,space,False)
    def _initialize(self,backend,tokenizer,spec,space,verified):
        import torch
        if spec.id not in ("irra","irra_fixture") or space.model_id!=spec.id or spec.representation_dim!=512:
            raise ArtifactError("IRRA backend/spec/space mismatch")
        if not isinstance(backend,torch.nn.Module) or not callable(getattr(backend,"encode_image",None)) or not callable(getattr(backend,"encode_text",None)):
            raise ArtifactError("native IRRA backend interface required")
        if space.tokenizer_sha256!=tokenizer.fingerprint:raise ArtifactError("Tokenizer space mismatch")
        self._backend=backend.float().eval().requires_grad_(False);self._tokenizer=tokenizer
        self.spec=spec;self.space=space;self.source_verified=verified
    @classmethod
    def from_source(cls,source,*,device,allow_model_load=False):
        # 无授权时在Tokenizer/Torch/Checkpoint加载之前拒绝。
        if allow_model_load is not True:raise ArtifactError("explicit model-load authorization required")
        source.verify()
        tokenizer=official_tokenizer(source)
        preprocess=digest({"contract":PREPROCESS,"runtime":{k:version(k) for k in ("torch","torchvision","Pillow")}})
        producers={name:sha_file(Path(__file__).parent/name) for name in ("irra.py","irra_io.py","irra_source.py","base.py")}
        source_identity=digest({"official":source.fingerprint,"adapter_producers":producers})
        space=ModelSpace("irra",source.checkpoint.sha256,source_identity,preprocess,tokenizer.fingerprint,
                         digest({"contract":METRICS,"official_rank_sha256":source.ref("utils/metrics.py").sha256}),"fp32/math/no_tf32")
        backend=load_official_model(source,device=device,allow_model_load=True)
        result=cls.__new__(cls)
        result._initialize(backend,tokenizer,IRRA,space,True)
        return result
    def inference(self):return IRRAView(self,False)
    def surrogate(self):return IRRAView(self,True)
    def _frozen(self):
        if any(m.training for m in self._backend.modules()) or any(p.requires_grad for p in self._backend.parameters()):
            raise ArtifactError("IRRA must remain eval and parameter-frozen")
    def score(self,query,gallery):return self.inference().score(query,gallery)
    def representation_artifact(self,artifact_id,batch,content,*,inputs):
        if type(batch) is not RepresentationBatch or batch.space_id!=self.space.id or batch.modality not in ("image","text"):
            raise ArtifactError("foreign representation space")
        import torch
        x=batch.payload
        if not isinstance(x,torch.Tensor) or x.dtype!=torch.float32 or x.ndim!=2 or x.shape[1]!=512 or not len(x) or not torch.isfinite(x).all():
            raise ArtifactError("invalid IRRA representation artifact shape/dtype")
        if not bool(torch.allclose(x.norm(dim=1),torch.ones(len(x),device=x.device),atol=1e-5,rtol=1e-5)):
            raise ArtifactError("IRRA artifact normalization mismatch")
        if type(inputs) is not InputBinding or inputs.model_space_id!=self.space.id or inputs.modality!=batch.modality:
            raise ArtifactError("explicit dataset/order/input binding required")
        artifact=RepresentationArtifact(artifact_id,self.space.checkpoint_sha256,self.space.id,"global_embedding",
            tuple(batch.payload.shape),str(batch.payload.dtype),"l2",self.space.preprocess_sha256,
            self.space.tokenizer_sha256,content)
        return RepresentationRecord(artifact,inputs)
    def score_artifact(self,artifact_id,content,query_order,gallery_order,*,query_inputs,gallery_inputs):
        if type(query_inputs) is not InputBinding or type(gallery_inputs) is not InputBinding or query_inputs.model_space_id!=self.space.id or gallery_inputs.model_space_id!=self.space.id or query_inputs.modality!="text" or gallery_inputs.modality!="image":
            raise ArtifactError("score model/dataset/input bindings required")
        artifact=ScoreArtifact(artifact_id,self.space.checkpoint_sha256,self.space.score_id,
                             query_order,gallery_order,"higher_better",content)
        return ScoreRecord(artifact,query_inputs,gallery_inputs)

class IRRAView:
    __slots__=("_adapter","_input_gradients")
    def __init__(self,adapter,input_gradients):
        self._adapter=adapter;self._input_gradients=input_gradients
    @property
    def input_gradients(self):return self._input_gradients
    @property
    def spec(self):return self._adapter.spec
    @property
    def space(self):return self._adapter.space
    def _run(self,callback):
        import torch
        a=self._adapter;a._frozen()
        with fp32_runtime(),torch.autocast(device_type=next(a._backend.parameters()).device.type,enabled=False),torch.inference_mode(not self.input_gradients),(torch.enable_grad() if self.input_gradients else torch.no_grad()):
            output=callback()
        # inference tensor先脱离inference_mode再clone，后续可作为Surrogate常量参与反传。
        if self.input_gradients:return output
        with torch.inference_mode(False),torch.no_grad():return output.detach().clone()
    def _normalize(self,feature,n):
        import torch
        from torch.nn import functional as F
        if not isinstance(feature,torch.Tensor) or feature.shape!=(n,512) or not torch.isfinite(feature).all() or bool((feature.norm(dim=1)==0).any()):
            raise ArtifactError("invalid native IRRA global features")
        return F.normalize(feature.float(),dim=-1)
    def encode_images(self,pixels):
        a=self._adapter
        result=self._run(lambda:self._normalize(a._backend.encode_image(normalize_pixels(pixels)),len(pixels)))
        return RepresentationBatch(result,self.space.id,"image")
    def encode_texts(self,texts,*,truncate=True):
        tokens,reports=self._adapter._tokenizer.batch(texts,truncate=truncate)
        return self.encode_tokens(tokens),reports
    def encode_tokens(self,tokens):
        import torch
        if not isinstance(tokens,torch.Tensor) or tokens.dtype!=torch.long or tokens.ndim!=2 or tokens.shape[1]!=77 or not len(tokens):
            raise ArtifactError("IRRA token shape/dtype required")
        if bool((tokens<0).any()) or bool((tokens>=49408).any()) or not bool((tokens[:,0]==49406).all()) or not bool((tokens==49407).any(dim=1).all()):
            raise ArtifactError("IRRA SOT/EOT/vocabulary mismatch")
        device=next(self._adapter._backend.parameters()).device
        result=self._run(lambda:self._normalize(self._adapter._backend.encode_text(tokens.to(device)),len(tokens)))
        return RepresentationBatch(result,self.space.id,"text")
    def score(self,query,gallery):
        import torch
        for batch,modality in ((query,"text"),(gallery,"image")):
            if type(batch) is not RepresentationBatch or batch.space_id!=self.space.id or batch.modality!=modality:
                raise ArtifactError("foreign model/preprocess/Tokenizer/representation space")
            x=batch.payload
            if not isinstance(x,torch.Tensor) or x.dtype!=torch.float32 or x.ndim!=2 or x.shape[1]!=512 or not len(x) or not torch.isfinite(x).all():
                raise ArtifactError("invalid IRRA score representation")
            if not bool(torch.allclose(x.norm(dim=1),torch.ones(len(x),device=x.device),atol=1e-5,rtol=1e-5)):
                raise ArtifactError("expected native L2-normalized retrieval space")
        if query.payload.device!=gallery.payload.device:raise ArtifactError("explicit same-device representations required")
        scores=self._run(lambda:query.payload@gallery.payload.T)
        return ScoreBatch(scores,self.space.id,self.space.score_id)
