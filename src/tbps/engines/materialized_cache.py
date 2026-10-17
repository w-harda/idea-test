"""物化评估的紧凑特征缓存与IRRA原生CPU评分；复用原ArtifactStore/Engine。"""
from dataclasses import asdict,replace
from pathlib import Path
import io
from tbps.artifacts import ArtifactError,FileRef,checked,read_json,json_bytes,digest,sha_file
from tbps.config import IRRA
from tbps.models.base import ModelSpace,RepresentationBatch,ScoreBatch,InputBinding
from tbps.models.irra_cached import normalized
from tbps.models.irra_source import OFFICIAL_IRRA_SHA
from tbps.engines.provenance import BoundRepresentation,payload_digest

FAMILIES=("clean_gallery","adv_gallery","clean_text","adv_text")

def ref(path):return FileRef(str(Path(path).resolve()),sha_file(path))

class CachedIRRA:
    """仅IRRA的已核验新缓存评分；不加载任何模型、不生成攻击、不采用历史重放身份。"""
    score_direction="higher_better"
    query_independent_scores=True
    input_gradients=False
    execution_kind="readonly_cached"
    def __init__(self,space,checkpoint,*,fixture=False,cache_protocol=None):
        if type(space)is not ModelSpace:raise ArtifactError("typed native ModelSpace")
        expected="irra_fixture" if fixture else "irra"
        if space.model_id!=expected or checkpoint.sha256!=space.checkpoint_sha256:
            raise ArtifactError("IRRA cached victim/source mismatch")
        if not fixture and space.checkpoint_sha256!=OFFICIAL_IRRA_SHA:
            raise ArtifactError("only locked official IRRA")
        checked(checkpoint)
        if not fixture:
            if type(cache_protocol)is not FileRef:raise ArtifactError("verified native encoding session required")
            checked(cache_protocol);m=read_json(cache_protocol.path,"tbps-artifact-1")
            p=Path(cache_protocol.path).with_name("protocol.json")
            checked(FileRef(str(p),m["files_sha256"]["protocol.json"]))
            session=read_json(p,"tbps-materialized-irra-cache-1")
            if (session["model_space"]!=asdict(space) or session["G_H_generation"] is not False
                or session["ranking_feedback_to_attack"] is not False or
                asdict(checkpoint) not in m["sources"]):
                raise ArtifactError("foreign native encoding session/model provenance")
        self.space=space;self.spec=replace(IRRA,id=expected)
        self.source_verified=True;self.score_calls=0
    def slice_query(self,batch,start,end):
        if batch.space_id!=self.space.id or batch.modality!="text":raise ArtifactError("foreign query space")
        return RepresentationBatch(batch.payload[start:end].detach().clone(),self.space.id,"text")
    def score(self,query,gallery):
        import torch
        for b,kind in ((query,"text"),(gallery,"image")):
            if type(b)is not RepresentationBatch or b.space_id!=self.space.id or b.modality!=kind:
                raise ArtifactError("foreign native IRRA family")
            normalized(b.payload,len(b.payload))
        # 与官方IRRA Evaluator及已核验IRRACachedScorer相同的原生评分表达式。
        with torch.inference_mode():x=query.payload@gallery.payload.T
        self.score_calls+=1
        return ScoreBatch(x,self.space.id,self.space.score_id)

class FeatureCache:
    def __init__(self,store,protocol,space,*,chunk=64):
        self.store,self.protocol,self.space,self.chunk=store,protocol,space,chunk
        self.protocol_ref=ref(store.root/"protocol/manifest.json")
        if type(chunk)is not int or chunk<1:raise ArtifactError("cache chunk")
    def key(self,family,start):
        if family not in FAMILIES or start%self.chunk:raise ArtifactError("cache family/offset")
        return family+"-"+str(start//self.chunk).zfill(6)
    def provenance(self,binding,start,end):
        return {"kind":"materialized_IRRA_feature","protocol_sha256":digest(self.protocol),
                "binding":asdict(binding),"start":start,"end":end}
    def load(self,family,binding,start,end):
        key=self.key(family,start);want=self.provenance(binding,start,end)
        self.store.verify(key,expected_provenance=want)
        import torch
        raw=torch.load(self.store.root/key/"features.pt",map_location="cpu",weights_only=True)
        if not isinstance(raw,torch.Tensor):raise ArtifactError("cached Tensor required")
        x=raw.detach().clone();del raw
        normalized(x,end-start)
        if payload_digest(x)!=read_json(self.store.root/key/"details.json")["feature_sha256"]:
            raise ArtifactError("feature content SHA conflict")
        return x
    def publish(self,family,binding,start,end,x,*,stats=None):
        import torch
        x=x.detach().cpu().clone();normalized(x,end-start)
        stream=io.BytesIO();torch.save(x,stream)
        return self.store.publish(self.key(family,start),
            {"features.pt":stream.getvalue(),"details.json":json_bytes({
                "feature_sha256":payload_digest(x),"stats":stats or {}})},
            self.provenance(binding,start,end),sources=(self.protocol_ref,))
    def gather(self,family,binding,*,origin=None,originals=()):
        import torch
        parts=[];refs=[]
        for start in range(0,len(binding.ordered_ids),self.chunk):
            end=min(start+self.chunk,len(binding.ordered_ids))
            parts.append(self.load(family,binding,start,end))
            refs.append(ref(self.store.root/self.key(family,start)/"manifest.json"))
        x=torch.cat(parts).detach().clone()
        return BoundRepresentation(RepresentationBatch(x,self.space.id,binding.modality),
            binding,payload_digest(x),tuple(refs),origin,originals)
