"""独立历史点积/官方rank对照；仅受控CPU入口或小合成Fixture调用。"""
from contextlib import contextmanager
from pathlib import Path
import importlib.util
from unittest.mock import patch
from ..artifacts import ArtifactError,checked
from .metrics import METRICS,drops
from .provenance import payload_digest
from .evaluation import MODES
from .historical_sources import need

def official_rank(ref):
    checked(ref)
    spec=importlib.util.spec_from_file_location("tbps_readonly_official_rank",ref.path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    checked(ref);return module.rank

def compare_metrics(actual,reference,tolerance):
    need(set(actual)==set(reference)==set(METRICS),"complete five metric comparison")
    errors={k:abs(actual[k]-reference[k]) for k in METRICS}
    need(all(v<=tolerance for v in errors.values()),"metric mismatch: "+str(errors))
    return errors

def compare_drops(actual,reference,tolerance):
    need(set(actual)==set(reference),"complete DR groups")
    errors={}
    for group,a in actual.items():
        need(set(a)==set(reference[group])==set(METRICS),"complete five DR metrics")
        errors[group]={}
        for k,v in a.items():
            expected=reference[group][k]
            need((v is None)==(expected is None),"DR zero-Clean semantics")
            delta=0 if v is None else abs(v-expected)
            need(delta<=tolerance,"DR mismatch: "+group+"/"+k)
            errors[group][k]=delta
    return errors

class RankOracle:
    def __init__(self,inputs,protocol,rank_fn,plan,*,check_stop=lambda:None):
        self.inputs=inputs;self.p=protocol;self.rank_fn=rank_fn;self.plan=plan;self.check=check_stop
        self.records={};self.calls=0;self.score_calls=0
    @contextmanager
    def instrument(self,modes):
        from . import evaluation
        original=evaluation.ranking
        def verified(matrix,identity,*,direction):
            import torch
            self.check();mode=modes[self.calls]
            q,g=(self.inputs[k].batch.payload for k in MODES[mode])
            # 独立于CachedScorer：原eval_metrics.metrics中的分块点积表达式。
            parts=[]
            for i in range(0,len(q),128):
                self.check();parts.append(q[i:i+128]@g.T);self.score_calls+=1
            reference=torch.cat(parts,0);del parts
            maximum=0.0
            for i in range(0,len(q),128):
                self.check();maximum=max(maximum,float((reference[i:i+128]-matrix[i:i+128]).abs().max()))
            self.records[mode]={"native_score_max_abs_error":maximum}
            need(maximum<=self.plan["score_max_abs_error"],"native Score mismatch: "+str(maximum))
            actual,order=original(matrix,identity,direction=direction)
            self.check()
            qids=torch.tensor([r["pid"] for r in self.p["queries"]])
            gids=torch.tensor([r["pid"] for r in self.p["gallery"]])
            cmc,ap,inp,official_order=self.rank_fn(reference,qids,gids,max_rank=10,get_mAP=True)
            expected={"mAP":float(ap),"R@1":float(cmc[0]),"R@5":float(cmc[4]),
                      "R@10":float(cmc[9]),"mINP":float(inp)}
            self.check()
            different=int((order!=official_order).sum())
            self.records[mode].update(ranking_indices_exact=different==0,different_indices=different,
                ranking_sha256=payload_digest(order),official_ranking_sha256=payload_digest(official_order),
                actual_metrics=actual,official_metrics=expected)
            if different:
                self.records[mode]["first_index_mismatches"]=[
                    {"query":int(i),"rank":int(j),"new_gallery_index":int(order[i,j]),
                     "official_gallery_index":int(official_order[i,j])}
                    for i,j in (order!=official_order).nonzero()[:10].tolist()]
            need(different==0,"official Ranking indices mismatch: "+str(different))
            errors=compare_metrics(actual,expected,self.plan["metric_atol_percentage_points"])
            self.records[mode]["official_metric_errors_pp"]=errors
            del reference,official_order,cmc,ap,inp,qids,gids
            self.calls+=1;self.check()
            print("REPLAY_MODE_VERIFIED",mode,actual,flush=True)
            return actual,order
        with patch.object(evaluation,"ranking",new=verified):yield self
        need(self.calls==len(modes),"incomplete independent official ranking comparison")

def verify_feature_values(loader,representations,check_stop=lambda:None):
    """第二条原文件读取路径逐值比较，避免仅用Loader返回值自我比较。"""
    import torch
    from .historical_sources import safe_file
    from ..artifacts import FileRef
    count=0
    for family,b in representations.items():
        offset=0
        for index,rows in loader.parts(family):
            check_stop();name=f"features/{family}/{index:06d}/features.pt"
            p=safe_file(loader.sources.root,name);ref=FileRef(str(p),loader.sources.declared[name]);checked(ref)
            old=torch.load(p,map_location="cpu",weights_only=True)
            need(isinstance(old,torch.Tensor),"reference feature Tensor required")
            exact=old.detach().clone();del old
            need(torch.equal(exact,b.batch.payload[offset:offset+len(rows)]),"feature values not exact")
            offset+=len(rows);count+=1;del exact;checked(ref)
        need(offset==len(b.binding.ordered_ids),"feature reference row count")
    return {"feature_values_exact":True,"independent_feature_batches":count}
