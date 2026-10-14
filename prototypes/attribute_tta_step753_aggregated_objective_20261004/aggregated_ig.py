"""正式定义aggregated objective：真实历史IG一次backward，完整IG之后投影。"""
import ast
import torch
from pathlib import Path
from real_ig import instrument_methods
from controlled_tta import observed_attack_type
from attribute_tta import OFFICIAL
from priority import MODES, normalized_weights, combine

class AggregatedCollector:
    def __init__(self,members,vectors,mode):
        self.members=members;self.vectors=[vectors[a.instance_id].detach() for a in members]
        self.scores=[a.metadata["S_raw"] for a in members];self.mode=mode
        self.records=[];self.last_gradient=None;self.snapshot=None
    def start_step(self):
        self.history=None;self.new=None;self.nodes={"scaled_ig":0,"transformed_ig":0}
        self.backwards=0;self.losses=[]
    def node(self,features,imgs,path,kind,official_loss):
        terms=torch.stack([-(features*v).sum() for v in self.vectors])
        weights=terms.new_tensor(normalized_weights(self.scores,self.mode))
        weighted=(terms*weights).sum()
        self.losses.append({"kind":kind,"attribute_losses":terms.detach().cpu().tolist(),
                            "weighted_loss":float(weighted.detach()),"official_loss":float(official_loss.detach())})
        def grad(loss):
            self.backwards+=1
            value,=torch.autograd.grad(loss,imgs,retain_graph=True,create_graph=False,allow_unused=True)
            return torch.zeros_like(imgs) if value is None else value
        if self.mode==MODES[2] and len(self.members)>1:
            hw=terms.new_tensor([s/sum(self.scores[:-1]) for s in self.scores[:-1]])
            h=grad((terms[:-1]*hw).sum());n=grad(terms[-1])
            hv=(h*path).detach();nv=(n*path).detach()
            self.history=hv if self.history is None else self.history+hv
            self.new=nv if self.new is None else self.new+nv
            value=(sum(self.scores[:-1])*h+self.scores[-1]*n)/sum(self.scores)
        else:
            # Baseline原loss/原backward不变；weighted仅一次完整目标backward。
            value=grad(official_loss if self.mode==MODES[0] else weighted)
        self.nodes[kind]+=1
        return value
    def finish(self,original_ig):
        if not all(self.nodes.values()):raise ValueError("missing actual IG nodes")
        if self.mode==MODES[2] and len(self.members)>1:
            gradient,record=combine([self.history,self.new],[sum(self.scores[:-1]),self.scores[-1]],self.mode,original_ig)
            h64=self.history.double();n64=self.new.double()
            dot=(h64*n64).sum();eps=torch.finfo(torch.float64).eps
            repaired=(n64-dot/(h64.square().sum()+eps)*h64).to(self.new.dtype) if dot<0 else self.new
            self.snapshot={"g_H":self.history.detach().clone(),"g_N":self.new.detach().clone(),
                           "projected_g_N":repaired.detach().clone(),"aggregate_gradient":gradient.detach().clone()}
            record["normalized_weights"]=normalized_weights(self.scores,self.mode)
            record["S_raw"]=list(self.scores)
        else:
            gradient=original_ig
            self.snapshot={"aggregate_gradient":gradient.detach().clone()}
            record={"normalized_weights":normalized_weights(self.scores,self.mode),"S_raw":list(self.scores),
                    "history_norm":None,"new_norm_before":None,"new_norm_after":None,"dot_before":None,"dot_after":None,
                    "cosine_before":None,"cosine_after":None,"conflict":None,"projection_intervention":False,
                    "removed_conflict_component_norm":0.,"projection_modification_ratio":0.,
                    "aggregate_gradient_norm":float(gradient.double().norm()),
                    "no_conflict_projection_exact":True,"baseline_original_tensor_identity":self.mode==MODES[0],
                    "history_components_unchanged":True,"only_new_attribute_modified":True,
                    "diagnostic_note":"individual historical gradient norms/conflict unmeasured; no diagnostic extra backward"}
        record.update(history_definition='IG of frozen score-weighted historical objective; no individual history backward',
            projected_new_unchanged_exact=bool(torch.equal(self.snapshot['g_N'],self.snapshot['projected_g_N'])) if 'g_N' in self.snapshot else None,
            projection_component_max_abs=float((self.snapshot['g_N']-self.snapshot['projected_g_N']).abs().max()) if 'g_N' in self.snapshot else None)
        record.update(step_index=len(self.records)+1,IG_nodes=dict(self.nodes),
            image_gradient_backward_calls=self.backwards,aggregation_backend="AGGREGATED_OBJECTIVE",
            diagnostic_point="complete actual scaled/transformed IG before unchanged momentum",
            attribute_instance_id=self.members[-1].instance_id,new_semantic_axis=self.members[-1].metadata["semantic_axis"],
            new_region=self.members[-1].metadata["region"],node_losses=self.losses,
            attribute_components=[{"attribute_instance_id":a.instance_id,"semantic_axis":a.metadata["semantic_axis"],
                "region":a.metadata["region"],"canonical_value":a.metadata["canonical_value"],"S_raw":s,
                "normalized_weight":w,"real_IG_gradient_norm":None,"weighted_contribution_norm":None}
                for a,s,w in zip(self.members,self.scores,record["normalized_weights"])])
        self.last_gradient=gradient;self.records.append(record);return gradient
    def verify_update(self,gradient):
        if gradient is not self.last_gradient:raise ValueError("not actual momentum input")
        self.records[-1]["actual_momentum_input_verified"]=True

def fast_methods(source):
    methods,counts=instrument_methods(source)
    class Replace(ast.NodeTransformer):
        def visit_Expr(self,node):
            # 移除slow额外capture；正式forward及概率目标保持。
            if isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=="self._priority.capture":return None
            return self.generic_visit(node)
        def visit_Assign(self,node):
            targets=[ast.unparse(t) for t in node.targets]
            if targets==["each_ig_grad"]:
                node.value=ast.parse("self._priority.node(scaled_img_embed, imgs, path, 'scaled_ig', loss)",mode="eval").body
            elif targets==["gk"] and ast.unparse(node.value).startswith("torch.autograd.grad("):
                node.value=ast.parse("self._priority.node(img_feats[k:k+1], imgs, path[k:k+1], 'transformed_ig', losses[k])",mode="eval").body
            return self.generic_visit(node)
    methods={k:Replace().visit(v) for k,v in methods.items()}
    return methods

def attack_type(tape):
    base=observed_attack_type(tape);scope=dict(base.img_attack.__globals__)
    for method in fast_methods((Path(OFFICIAL)/"attacker_TTA.py").read_text(encoding="utf-8")).values():
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),"<step753-objective-IG>","exec"),scope)
    return type("AggregatedObjectiveAttack",(base,),{name:scope[name] for name in ("ig","transform_ig_batch","img_attack")})
