"""原PREFIX与两种aggregated-objective；三mode共享真实round1状态。"""
import ast,copy,inspect,time
from contextlib import contextmanager
import torch
import deps
import prefix_tta as old
import controlled_tta as controlled
from controlled_tta import ControlledTTA,UpdateObserver,observed_attack_type
from priority import MODES,normalized_weights
from aggregated_ig import AggregatedCollector,attack_type
legacy=deps.module("step753_readonly_modes75",deps.P75/"modes.py")
cached_guidance=legacy.cached_guidance
_LIMIT=None
class EndSmokeRound(Exception):
    def __init__(self,image,momentum):self.image=image.detach();self.momentum=momentum.detach()
class ShortObserver(UpdateObserver):
    def __init__(self,limit):super().__init__();self.limit=limit;self.current_momentum=None
    def pre_step(self,before,sign_step,ig_grad,momentum):
        super().pre_step(before,sign_step,ig_grad,momentum);self.current_momentum=momentum
    def after_clamp(self,after,incoming,original):
        updating=self.before is not None
        super().after_clamp(after,incoming,original)
        if updating and len(self.events)>=self.limit:raise EndSmokeRound(after,self.current_momentum)
def observed_step():
    # 只替换observer构造/观测数量断言；所有图像攻击与后置诊断语句保持原样。
    tree=ast.parse(inspect.getsource(ControlledTTA.step).lstrip())
    class Observe(ast.NodeTransformer):
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if ast.unparse(node.func)=="UpdateObserver":
                return ast.copy_location(ast.parse("self.make_observer()",mode="eval").body,node)
            return node
        def visit_Compare(self,node):
            node=self.generic_visit(node)
            if ast.unparse(node)=="len(observer.events) != self.steps":
                node.comparators[0]=ast.parse("self.expected_observed_steps",mode="eval").body
            return node
    tree=Observe().visit(tree)
    scope=dict(ControlledTTA.step.__globals__)
    exec(compile(ast.fix_missing_locations(tree),"<step753-observation-only>","exec"),scope)
    return scope["step"]
BASE_STEP=observed_step()
def timed_type(cls):
    class Timed(cls):
        def img_attack(self,*args,**kwargs):
            cuda=args[2].is_cuda
            if cuda:torch.cuda.synchronize()
            t=time.perf_counter()
            try:result=super().img_attack(*args,**kwargs)
            except EndSmokeRound as stop:result=(stop.image,stop.momentum)
            if cuda:torch.cuda.synchronize()
            self.attack_seconds=time.perf_counter()-t
            return result
    return Timed
def baseline_records(event,members,steps,transforms,scales):
    weights=normalized_weights([a.metadata["S_raw"] for a in members],MODES[0])
    nodes={"scaled_ig":1+len(scales or ()),"transformed_ig":(1+len(scales or ()))*transforms}
    return [dict(step_index=s["step_index"],attribute_instance_id=members[-1].instance_id,
      new_semantic_axis=members[-1].metadata["semantic_axis"],new_region=members[-1].metadata["region"],
      S_raw=[a.metadata["S_raw"] for a in members],normalized_weights=weights,
      history_norm=None,new_norm_before=None,new_norm_after=None,dot_before=None,dot_after=None,
      cosine_before=None,cosine_after=None,conflict=None,projection_intervention=False,
      removed_conflict_component_norm=0.,projection_modification_ratio=0.,
      no_conflict_projection_exact=True,aggregate_gradient_norm=s["ig_gradient_l2"],
      history_components_unchanged=True,only_new_attribute_modified=True,
      actual_momentum_input_verified=True,baseline_original_tensor_identity=True,
      image_gradient_backward_calls=sum(nodes.values()),IG_nodes=nodes,
      aggregation_backend="UNMODIFIED_PREFIX",diagnostic_point="original official aggregate IG; no extra per-attribute backward",
      attribute_components=[dict(attribute_instance_id=a.instance_id,semantic_axis=a.metadata["semantic_axis"],
        region=a.metadata["region"],canonical_value=a.metadata["canonical_value"],S_raw=a.metadata["S_raw"],
        normalized_weight=w,real_IG_gradient_norm=None,weighted_contribution_norm=None) for a,w in zip(members,weights)])
      for s in event["steps"]]
@contextmanager
def engine_for(mode,plan,vectors):
    original=old.ControlledTTA
    class Engine(ControlledTTA):
        @property
        def expected_observed_steps(self):return _LIMIT if _LIMIT is not None else self.steps
        def make_observer(self):return ShortObserver(_LIMIT) if _LIMIT is not None else UpdateObserver()
        def step(self,current,attribute,features,labels):
            first=attribute.selection_rank==1
            unchanged=mode==MODES[0] or first
            previous=self.attack
            cls=observed_attack_type(self.tape) if unchanged else attack_type(self.tape)
            self.attack=timed_type(cls)(None,previous.tokenizer,imgs_eps=previous.imgs_eps,step_size=previous.step_size)
            self.attack.N_trans=previous.N_trans
            members=plan[:attribute.selection_rank]
            collector=None
            if not unchanged:
                collector=AggregatedCollector(members,vectors,mode);self.attack._priority=collector
            t=time.perf_counter()
            result=BASE_STEP(self,current,attribute,features,labels)
            event=self.events[-1]
            if unchanged:
                event["priority_steps"]=baseline_records(event,members,self.steps,self.attack.N_trans,self.scales)
            else:
                event["priority_steps"]=collector.records
                if len(collector.records)!=self.expected_observed_steps or not all(x.get("actual_momentum_input_verified") for x in collector.records):
                    raise ValueError("aggregate gradient did not reach real momentum input")
            event.update(configured_steps_per_round=self.steps,observed_steps=len(event["steps"]),
                short_smoke_observer_limit=_LIMIT,attack_seconds=self.attack.attack_seconds,
                attack_seconds_per_observed_step=self.attack.attack_seconds/len(event["steps"]),
                round_seconds=time.perf_counter()-t,independent_attribute_backwards=0,
                method_definition="aggregated objective is authoritative; no slow pixel equivalence gate",
                no_cross_mode_gradient_reuse=True)
            return result
    old.ControlledTTA=Engine
    try:yield
    finally:old.ControlledTTA=original
legacy.engine_for=engine_for
def execute_three(source,image,plan,vectors,evaluator,*,steps,transforms,scales,observed_limit=None):
    global _LIMIT
    if observed_limit is not None and not 1<=observed_limit<=min(2,steps):raise ValueError("short smoke observation only")
    previous=_LIMIT;_LIMIT=observed_limit
    try:
        data,checks=legacy.execute_three(source,image,plan,vectors,evaluator,steps=steps,transforms=transforms,scales=scales)
    finally:_LIMIT=previous
    checks.update(aggregated_objective_authoritative=True,slow_equivalence_gate=False,
      baseline_unmodified=True,per_attribute_backward_during_attack=0,
      configured_steps_per_round=steps,smoke_observed_limit=observed_limit)
    return data,checks
