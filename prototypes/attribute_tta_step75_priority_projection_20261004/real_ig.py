"""在原官方AST添加IG贡献捕获，合成点位于实际momentum更新之前。"""
import ast,copy
import torch
from controlled_tta import observed_attack_type
from attribute_tta import OFFICIAL
from priority import combine,MODES
from pathlib import Path
class IGCollector:
    def __init__(self,members,vectors,mode):
        self.members=members;self.vectors=[vectors[a.instance_id] for a in members]
        self.mode=mode;self.scores=[a.metadata["S_raw"] for a in members]
        self.records=[];self.last_gradient=None
    def start_step(self):
        self.components=[None]*len(self.members);self.nodes={"scaled_ig":0,"transformed_ig":0}
    def capture(self,features,imgs,path,kind):
        # 同一image状态/插值节点/transform共享forward graph；不增加forward或RNG。
        for i,vector in enumerate(self.vectors):
            loss=-(features*vector.detach()).sum()
            g,=torch.autograd.grad(loss,imgs,retain_graph=True,create_graph=False,allow_unused=True)
            if g is None:g=torch.zeros_like(imgs)
            value=(g*path).detach()
            self.components[i]=value if self.components[i] is None else self.components[i]+value
        self.nodes[kind]+=1
    def finish(self,original_ig):
        if any(g is None for g in self.components) or not all(self.nodes.values()):
            raise ValueError("actual scaled/transformed IG node missing")
        gradient,record=combine(self.components,self.scores,self.mode,original_ig)
        record.update(step_index=len(self.records)+1,IG_nodes=dict(self.nodes),
                      diagnostic_point="actual transformed/scaled IG of this optimization step",
                      attribute_instance_id=self.members[-1].instance_id,
                      new_semantic_axis=self.members[-1].metadata["semantic_axis"],
                      new_region=self.members[-1].metadata["region"],
                      attribute_components=[{
                          "attribute_instance_id":a.instance_id,"semantic_axis":a.metadata["semantic_axis"],
                          "region":a.metadata["region"],"canonical_value":a.metadata["canonical_value"],
                          "S_raw":s,"normalized_weight":w,
                          "real_IG_gradient_norm":float(g.double().norm()),
                          "weighted_contribution_norm":float(g.double().norm())*w,
                      } for a,s,w,g in zip(self.members,self.scores,record["normalized_weights"],self.components)])
        self.last_gradient=gradient;self.records.append(record)
        return gradient
    def verify_update(self,ig_grad):
        if ig_grad is not self.last_gradient:raise ValueError("projection not connected to actual momentum input")
        self.records[-1]["actual_momentum_input_verified"]=True

def instrument_methods(source):
    """保留所有官方语句，仅加capture/start/finish/verify；断言精确AST锚点。"""
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="Attack")
    methods={n.name:copy.deepcopy(n) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ("ig","transform_ig_batch","img_attack")}
    counts={"scaled":0,"transformed":0,"start":0,"finish":0,"verify":0}
    def call(code,node):return ast.copy_location(ast.parse(code).body[0],node)
    class Capture(ast.NodeTransformer):
        def visit_Assign(self,node):
            node=self.generic_visit(node);targets=[ast.unparse(t) for t in node.targets]
            if targets==["each_ig_grad"]:
                counts["scaled"]+=1
                return [node,call("self._priority.capture(scaled_img_embed, imgs, path, 'scaled_ig')",node)]
            if targets==["gk"] and ast.unparse(node.value).startswith("torch.autograd.grad("):
                counts["transformed"]+=1
                return [node,call("self._priority.capture(img_feats[k:k+1], imgs, path[k:k+1], 'transformed_ig')",node)]
            return node
    methods["ig"]=Capture().visit(methods["ig"])
    methods["transform_ig_batch"]=Capture().visit(methods["transform_ig_batch"])
    # 以Step73同一observer变换先构建原img_attack，然后插入三处新调用。
    # 复用其代码生成而非修改官方/历史文件；此处observer AST完全相同。
    class Observe(ast.NodeTransformer):
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if ast.unparse(node.func)=="np.random.uniform":
                return ast.copy_location(ast.Call(ast.parse("self._audit.capture_noise",mode="eval").body,[node],[]),node)
            return node
        def visit_Assign(self,node):
            node=self.generic_visit(node);targets=[ast.unparse(t) for t in node.targets]
            if targets==["sole_grad"]:
                counts["start"]+=1;return [call("self._priority.start_step()",node),node]
            if targets==["ig_grad"] and ast.unparse(node.value)=="sole_grad + trans_grads":
                counts["finish"]+=1;return [node,call("ig_grad = self._priority.finish(ig_grad)",node)]
            if targets==["imgs_0"] and ast.unparse(node.value)=="imgs_0.detach() + perturbation":
                counts["verify"]+=1
                return [call("self._priority.verify_update(ig_grad)",node),
                        call("self._audit.pre_step(imgs_0, perturbation, ig_grad, grad)",node),node]
            if targets==["imgs_0"] and ast.unparse(node.value)=="torch.clamp(imgs_0, 0.0, 1.0)":
                return [node,call("self._audit.after_clamp(imgs_0, imgs, origin_imgs)",node)]
            return node
    methods["img_attack"]=Observe().visit(methods["img_attack"])
    if counts!={"scaled":1,"transformed":1,"start":1,"finish":1,"verify":1}:
        raise ValueError("official IG/update AST anchors changed: "+str(counts))
    return methods,counts

def attack_type(tape):
    base=observed_attack_type(tape)
    methods,counts=instrument_methods((Path(OFFICIAL)/"attacker_TTA.py").read_text(encoding="utf-8"))
    scope=dict(base.img_attack.__globals__)
    for method in methods.values():
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),
                     "<step75-real-IG>", "exec"),scope)
    return type("PriorityObservedAttack",(base,),{name:scope[name] for name in methods})
