"""APTM全Query原生归一化上下文；原始ITM/全局分数分块缓存后再归一化。"""
import io
import torch
from tbps.artifacts import json_bytes,read_json,ArtifactError
from tbps.models.base import RepresentationBatch,ScoreBatch
from tbps.engines.provenance import payload_digest
from tbps.models.aptm import finalize_scores
from tbps.engines.native_score_cache import ChunkCachedScorer

class APTMSetScorer:
    """仅已绑定完整Query集合上的最终Score可分块读取；原始Adapter不声明Query独立。"""
    score_direction="higher_better";execution_kind="materialized_native"
    query_independent_scores=True;input_gradients=False
    def __init__(self,adapter,inputs,store,protocol_ref):
        self.adapter,self.inputs,self.store,self.protocol_ref=adapter,inputs,store,protocol_ref
        self.space,self.spec,self.source_verified=adapter.space,adapter.spec,adapter.source_verified
        self.families={id(v.batch.payload):k for k,v in inputs.items()}
        self.bindings={k:v.id for k,v in inputs.items()}
        self.hits=0;self.new=0;self.slice=None;self._last_key=None;self._last_value=None
    def slice_query(self,batch,start,end):
        family=self.families.get(id(batch.payload))
        if family not in ("clean_text","adv_text"):raise ArtifactError("bound complete APTM query required")
        n=len(batch.payload["global"])
        if not 0<=start<end<=n:raise ArtifactError("APTM query slice bounds")
        value=RepresentationBatch({k:v[start:end].detach().clone() for k,v in batch.payload.items()},self.space.id,"text")
        self.slice=(id(value.payload),family,start,end)
        return value
    def _provenance(self,q,g):
        return {"model_space":self.space.id,"query":self.bindings[q],"gallery":self.bindings[g],
                "protocol_sha256":self.protocol_ref.sha256,"normalization":"complete bound Query/Gallery matrix"}
    def _load(self,key,provenance,fields,n,m):
        self.store.verify(key,expected_provenance=provenance)
        raw=torch.load(self.store.root/key/"score.pt",map_location="cpu",weights_only=True)
        if type(raw)is not dict or set(raw)!=set(fields):raise ArtifactError("APTM score cache fields")
        x={k:v.detach().clone() for k,v in raw.items()};del raw
        for v in x.values():ChunkCachedScorer.validate(v,n,m)
        if {k:payload_digest(v) for k,v in x.items()}!=read_json(self.store.root/key/"details.json"):
            raise ArtifactError("APTM score content mismatch")
        return x
    def _save(self,key,p,x,*,sources=()):
        stream=io.BytesIO();torch.save({k:v.detach().cpu().clone() for k,v in x.items()},stream)
        self.store.publish(key,{"score.pt":stream.getvalue(),
            "details.json":json_bytes({k:payload_digest(v) for k,v in x.items()})},p,sources=(self.protocol_ref,)+sources)
    def _complete(self,q,g):
        key="aptm-final-"+q+"--"+g
        p=self._provenance(q,g);qb=self.inputs[q].batch;gb=self.inputs[g].batch
        nq,ng=len(qb.payload["global"]),len(gb.payload["global"])
        if self._last_key==key:
            return self._last_value
        self._last_key=None;self._last_value=None
        if (self.store.root/key).exists():
            self.hits+=1
            final=self._load(key,p,("final",),nq,ng)["final"]
            self._last_key,self._last_value=key,final
            return final
        parts=[];rawrefs=[]
        for start in range(0,nq,128):
            end=min(start+128,nq);rawkey="aptm-raw-"+q+"--"+g+"-"+str(start).zfill(6)
            rp=dict(p,start=start,end=end)
            if (self.store.root/rawkey).exists():
                x=self._load(rawkey,rp,("itm","global"),end-start,ng);self.hits+=1
            else:
                b=RepresentationBatch({k:v[start:end].detach().clone() for k,v in qb.payload.items()},self.space.id,"text")
                x=self.adapter.raw_score(b,gb)
                for v in x.values():ChunkCachedScorer.validate(v,end-start,ng)
                self._save(rawkey,rp,x);self.new+=1
            parts.append(x)
            from tbps.engines.materialized_cache import ref
            rawrefs.append(ref(self.store.root/rawkey/"manifest.json"))
            print("APTM_RAW_SCORE_PROGRESS",q,g,end,"/",nq,flush=True)
        all_={k:torch.cat([x[k] for x in parts],0) for k in ("itm","global")}
        del parts
        final=finalize_scores(all_);del all_
        ChunkCachedScorer.validate(final,nq,ng)
        self._save(key,p,{"final":final},sources=tuple(rawrefs))
        self._last_key,self._last_value=key,final
        return final
    def score(self,query,gallery):
        if self.slice is None or self.slice[0]!=id(query.payload):raise ArtifactError("explicit bound query slice")
        _,q,start,end=self.slice;g=self.families.get(id(gallery.payload))
        if g not in ("clean_gallery","adv_gallery"):raise ArtifactError("bound Gallery required")
        final=self._complete(q,g)
        value=final[start:end].detach().clone();del final
        return ScoreBatch(value,self.space.id,self.space.score_id)
