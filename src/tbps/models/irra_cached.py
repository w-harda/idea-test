"""只读IRRA原生缓存评分器，无IRRA/G/H模型、Tokenizer实例或攻击入口。"""
from dataclasses import replace
from ..artifacts import ArtifactError
from ..config import IRRA
from .base import ModelSpace,RepresentationBatch,ScoreBatch

def normalized(x,n):
    import torch
    if not isinstance(x,torch.Tensor) or x.device.type!="cpu" or x.dtype!=torch.float32 or x.shape!=(n,512) or (
        x.requires_grad or not torch.isfinite(x).all() or not torch.allclose(x.norm(dim=1),torch.ones(n),atol=2e-5,rtol=0)):
        raise ArtifactError("invalid historical CPU FP32 normalized [N,512] feature")

class IRRACachedScorer:
    score_direction="higher_better"
    query_independent_scores=True
    input_gradients=False
    def __init__(self,space,identity,*,allow_full_scoring=False):
        from ..engines.historical_contracts import HistoricalReplayIdentity
        if type(space)is not ModelSpace or type(identity)is not HistoricalReplayIdentity:
            raise ArtifactError("verified legacy identity and new mapped space required")
        expected="irra_fixture" if identity.fixture else "irra"
        if space.model_id!=expected or space.checkpoint_sha256!=dict(identity.models)["IRRA"].sha256:
            raise ArtifactError("historical IRRA victim/Checkpoint mismatch")
        if type(allow_full_scoring)is not bool:raise ArtifactError("explicit replay execution flag")
        identity.verify()
        self.spec=replace(IRRA,id=expected)
        self.space=space;self.replay_identity=identity;self.source_verified=True
        self.execution_kind="historical_fixture_replay" if identity.fixture else "historical_feature_replay"
        self.allow_full_scoring=allow_full_scoring;self.score_calls=0
    def slice_query(self,batch,start,end):
        if type(batch)is not RepresentationBatch or batch.space_id!=self.space.id or batch.modality!="text":
            raise ArtifactError("foreign cached query")
        return RepresentationBatch(batch.payload[start:end].detach().clone(),self.space.id,"text")
    def score(self,query,gallery):
        if not self.replay_identity.fixture and not self.allow_full_scoring:
            raise ArtifactError("real scoring disabled: Stage 4B-2 authorization required")
        if self.replay_identity.fixture and (len(query.payload)>64 or len(gallery.payload)>64):
            raise ArtifactError("small synthetic replay only")
        for b,kind in ((query,"text"),(gallery,"image")):
            if type(b)is not RepresentationBatch or b.space_id!=self.space.id or b.modality!=kind:
                raise ArtifactError("foreign historical feature space/modality")
            normalized(b.payload,len(b.payload))
        import torch
        with torch.inference_mode():score=query.payload@gallery.payload.T
        self.score_calls+=1
        return ScoreBatch(score,self.space.id,self.space.score_id)
