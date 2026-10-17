"""受害者原生分块Score缓存；只缓存评分，不持有攻击策略或身份标签。"""
import io
import torch
from tbps.artifacts import json_bytes,digest,read_json,ArtifactError
from tbps.models.base import ScoreBatch
from tbps.engines.provenance import payload_digest

class ChunkCachedScorer:
    score_direction="higher_better"
    execution_kind="materialized_native"
    query_independent_scores=True
    input_gradients=False
    def __init__(self,adapter,inputs,store,protocol_ref):
        self.adapter,self.store,self.protocol_ref=adapter,store,protocol_ref
        self.space,self.spec,self.source_verified=adapter.space,adapter.spec,adapter.source_verified
        self.families={id(v.batch.payload):k for k,v in inputs.items()}
        self.bindings={k:v.id for k,v in inputs.items()}
        self.gallery_sizes={id(v.batch.payload):len(v.inputs.ordered_ids) for v in inputs.values()}
        self.score_direction=adapter.score_direction
        self.hits=0;self.new=0;self.slice=None
    def slice_query(self,batch,start,end):
        value=self.adapter.slice_query(batch,start,end)
        self.slice=(id(value.payload),self.families[id(batch.payload)],start,end)
        return value
    def score(self,query,gallery):
        if self.slice is None or self.slice[0]!=id(query.payload):raise ArtifactError("explicit query slice required")
        _,q,start,end=self.slice;g=self.families[id(gallery.payload)]
        key="score-"+q+"--"+g+"-"+str(start).zfill(6)
        provenance={"space_id":self.space.id,"query":self.bindings[q],"gallery":self.bindings[g],
                    "start":start,"end":end,"protocol_sha256":self.protocol_ref.sha256}
        path=self.store.root/key
        reused=path.exists()
        if reused:
            self.store.verify(key,expected_provenance=provenance)
            raw=torch.load(path/"score.pt",map_location="cpu",weights_only=True)
            x=raw.detach().clone();del raw
            if payload_digest(x)!=read_json(path/"details.json")["score_sha256"]:raise ArtifactError("cached score content SHA")
            self.hits+=1
        else:
            x=self.adapter.score(query,gallery).payload.detach().cpu().clone()
            self.validate(x,end-start,self.gallery_sizes[id(gallery.payload)])
            stream=io.BytesIO();torch.save(x,stream)
            self.store.publish(key,{"score.pt":stream.getvalue(),"details.json":json_bytes({"score_sha256":payload_digest(x)})},
                provenance,sources=(self.protocol_ref,));self.new+=1
        self.validate(x,end-start,self.gallery_sizes[id(gallery.payload)])
        print("NATIVE_SCORE_PROGRESS",q,g,end,"REUSED" if reused else "READY",flush=True)
        return ScoreBatch(x,self.space.id,self.space.score_id)
    @staticmethod
    def validate(x,n,m):
        if x.shape!=(n,m) or x.dtype!=torch.float32 or x.device.type!="cpu" or not torch.isfinite(x).all() or x.requires_grad:
            raise ArtifactError("native score shape/dtype/finite")
