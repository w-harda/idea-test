"""G-v2方案A单样本K轮训练机制；仅消费缓存，不加载模型/数据或启动训练。"""
from dataclasses import dataclass, asdict
import math
import numbers
import torch
from condition_schema import validate
from budget_projection import (DEFAULT_EPSILON, check_pixels, add_projected_delta,
                               projection_bounds)
from losses import attribute_losses, joint_losses, retain_reference

@dataclass(frozen=True)
class LossWeights:
    lambda_global: float = 1.
    lambda_local: float = 1.
    lambda_retain: float = 1.

    def __post_init__(self):
        for name, value in asdict(self).items():
            if (isinstance(value, bool) or not isinstance(value, numbers.Real)
                or not math.isfinite(value) or value < 0):
                raise ValueError(name + " must be finite and nonnegative")

def _check_optimizer(generator, encoder, optimizer, image):
    trainable = [p for p in generator.parameters() if p.requires_grad]
    owned = [p for group in optimizer.param_groups for p in group["params"]]
    if not trainable or len({id(p) for p in owned}) != len(owned):
        raise ValueError("nonempty G-only optimizer with unique parameters required")
    if {id(p) for p in owned} != {id(p) for p in trainable}:
        raise ValueError("optimizer must contain exactly trainable G parameters")
    if {id(p) for p in generator.parameters()} & {id(p) for p in encoder.parameters()}:
        raise ValueError("generator and encoder parameters must be disjoint")
    for module in (generator, encoder):
        for p in module.parameters():
            if p.device != image.device or p.dtype != image.dtype or not torch.isfinite(p).all():
                raise ValueError("finite model parameters matching image dtype/device required")

def _encoder_snapshot(encoder):
    return ({n:p._version for n,p in encoder.named_parameters()},
            {n:b.detach().clone() for n,b in encoder.named_buffers()})

def _check_encoder(encoder, snapshot):
    versions, buffers = snapshot
    if encoder.training or any(p.requires_grad or p.grad is not None or p._version != versions[n]
        for n,p in encoder.named_parameters()):
        raise ValueError("frozen encoder parameter state changed")
    if any(not torch.equal(b.detach(),buffers[n]) for n,b in encoder.named_buffers()):
        raise ValueError("frozen encoder buffers changed")

def _weighted(terms, weights, joint):
    total = weights.lambda_global * terms["global"] + weights.lambda_local * terms["local"]
    if not joint: total = total + weights.lambda_retain * terms["retain"]
    return total

def _image_stats(original, current, proposed):
    lower, upper = projection_bounds(original)
    at_lower, at_upper = current == lower, current == upper
    budget_active_lower = original.detach().double() - DEFAULT_EPSILON >= 0
    budget_active_upper = original.detach().double() + DEFAULT_EPSILON <= 1
    budget_sat = (at_lower & budget_active_lower) | (at_upper & budget_active_upper)
    clipped = (proposed.detach() < lower) | (proposed.detach() > upper)
    linf = float((current.detach().double()-original.detach().double()).abs().max())
    if not torch.isfinite(current).all() or linf > DEFAULT_EPSILON:
        raise ValueError("nonfinite image or cumulative original-image budget violation")
    return {"cumulative_linf":linf,"budget_saturation_ratio":float(budget_sat.float().mean()),
            "pixel_saturation_ratio":float(((current==0)|(current==1)).float().mean()),
            "projection_endpoint_ratio":float((at_lower|at_upper).float().mean()),
            "strict_projection_clipping_ratio":float(clipped.float().mean())}

def _update(generator, encoder, optimizer, original, current, z_t, local, weights,
            *, joint, has_local, previous_jvp, previous_reference, phase_index,
            instance_id, previous_instance_id, encoder_state, observer):
    # 每次调用独立建图；返回值只含detach张量和Python标量。
    optimizer.zero_grad(set_to_none=True)
    delta = generator(current)
    adversarial = add_projected_delta(original, current, delta)
    proposed = current + delta
    feature = encoder(adversarial)
    if joint:
        terms = joint_losses(feature,z_t,local,torch.tensor([has_local],
                                                          device=feature.device,dtype=torch.bool))
        next_reference = None
    else:
        terms = attribute_losses(feature,z_t,local,previous_jvp=previous_jvp,
                                 reference_score=previous_reference)
        next_reference = retain_reference(feature,local)
    total = _weighted(terms,weights,joint)
    if not total.requires_grad: raise ValueError("encoder/G must preserve image gradient path")
    if any(not torch.isfinite(v).all() for v in (*terms.values(),total)):
        raise ValueError("nonfinite loss")
    stats = _image_stats(original,adversarial,proposed)
    total.backward() # 不使用retain_graph/create_graph。
    gradients = [p.grad for p in generator.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or any(not torch.isfinite(g).all() for g in gradients):
        raise ValueError("missing or nonfinite G gradients")
    gradient_l2 = float(sum(g.detach().double().square().sum() for g in gradients).sqrt())
    _check_encoder(encoder,encoder_state)
    record = {"phase":"joint" if joint else "attribute","update_index":phase_index,
        "instance_id":None if joint else instance_id,"previous_instance_id":previous_instance_id,
        "previous_reference_score":None if previous_reference is None else float(previous_reference[0]),
        "next_reference_score":None if next_reference is None else float(next_reference[0]),
        "global":float(terms["global"].detach()),"local":float(terms["local"].detach()),
        "retain":0. if joint else float(terms["retain"].detach()),"total":float(total.detach()),
        "G_gradient_l2":gradient_l2,"finite_gradients":True,**stats}
    optimizer.step() # 必须在下一轮G forward之前完成。
    if any(not torch.isfinite(p).all() for p in generator.parameters()):
        raise ValueError("optimizer produced nonfinite G parameters")
    _check_encoder(encoder,encoder_state)
    detached_image = adversarial.detach()
    if observer is not None:
        observer({"record":dict(record),"input_image":current.detach().clone(),
                  "adversarial":detached_image.clone(),"feature":feature.detach().clone(),
                  "previous_jvp":None if previous_jvp is None else previous_jvp.detach().clone(),
                  "previous_reference":None if previous_reference is None else previous_reference.detach().clone()})
    return detached_image, next_reference, record

def train_single_sample(generator, encoder, optimizer, image, condition, *,
                        weights=None, observer=None):
    """每个属性立即更新G，最后原图联合更新；无epoch/batch/CLI/checkpoint调度。

    encoder为可微nn.Module: image->[1,512]，由调用方提供，本函数冻结/eval。
    condition为Phase 1 CPU缓存schema。图像须为单样本[1,3,H,W]。
    返回的两个图像是各自更新前前向所得图像，均detach，不在step后再生成。
    """
    check_pixels(image)
    if image.shape[0] != 1: raise ValueError("single-sample [1,3,H,W] required")
    validate(condition)
    if weights is None: weights = LossWeights()
    if not isinstance(weights,LossWeights): raise ValueError("LossWeights required")
    _check_optimizer(generator,encoder,optimizer,image)
    if observer is not None and not callable(observer): raise ValueError("observer must be callable")
    if not torch.is_grad_enabled(): raise ValueError("training requires enabled autograd")
    generator.train()
    encoder.eval().requires_grad_(False)
    encoder.zero_grad(set_to_none=True)
    encoder_state = _encoder_snapshot(encoder)
    original = image.detach().clone()
    current = original.clone()
    z_t = condition["z_t"].to(device=image.device,dtype=image.dtype)[None]
    jvps = condition["attribute_jvps"].to(device=image.device,dtype=image.dtype)
    aggregate = condition["v_local"].to(device=image.device,dtype=image.dtype)[None]
    ids = list(condition["selected_instance_ids"])
    previous_jvp = None; previous_reference = None; previous_id = None
    records = []
    for index, instance_id in enumerate(ids,1):
        local = jvps[index-1:index]
        current, previous_reference, record = _update(
            generator,encoder,optimizer,original,current,z_t,local,weights,
            joint=False,has_local=True,previous_jvp=previous_jvp,previous_reference=previous_reference,
            phase_index=index,instance_id=instance_id,previous_instance_id=previous_id,
            encoder_state=encoder_state,observer=observer)
        records.append(record)
        previous_jvp = local.detach()
        previous_id = instance_id
    serial_adversarial = current.detach().clone()
    # K=0也执行；联合步明确重新输入原图，不把K轮图像传入G。
    joint_adversarial, _, record = _update(
        generator,encoder,optimizer,original,original,z_t,aggregate,weights,
        joint=True,has_local=condition["has_local"],previous_jvp=None,previous_reference=None,
        phase_index=len(ids)+1,instance_id=None,
        previous_instance_id=None,encoder_state=encoder_state,observer=observer)
    records.append(record)
    return {"K":len(ids),"optimizer_steps":len(records),"G_forwards":len(records),
            "encoder_forwards":len(records),"weights":asdict(weights),"records":records,
            "serial_adversarial":serial_adversarial,
            "joint_adversarial":joint_adversarial.detach().clone()}
