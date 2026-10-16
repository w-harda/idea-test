"""H-only真实SCST单步；复用既有scst_loss，不改变奖励或动作策略。"""
from dataclasses import asdict
import math
import torch
from feedback_backend import state_sha
from constraints import replay
from scst import scst_loss

def optimizer_for(policy,config):
    cfg=config["optimizer"]
    return torch.optim.Adam(policy.parameters(),lr=cfg["lr"],betas=tuple(cfg["betas"]),
                            eps=cfg["eps"],weight_decay=cfg["weight_decay"])

def check_optimizer(policy,optimizer):
    expected={id(p) for p in policy.parameters() if p.requires_grad}
    actual={id(p) for group in optimizer.param_groups for p in group["params"]}
    if (not expected or expected!=actual or sum(len(g["params"]) for g in optimizer.param_groups)!=len(expected)
            or any(p.device.type!="cpu" for p in policy.parameters())):
        raise ValueError("optimizer must contain each trainable CPU H parameter exactly once")

def action_stats(output):
    attrs=[d for d in output.decisions if d["head"]=="attribute"]
    positions=[d for d in output.decisions if d["head"]=="position"]
    stops=sum(d["index"]==len(output.attributes) for d in attrs)
    # position的NOOP索引为source span内字符数。
    position_noops=0
    selected=iter(output.selected_attribute_ids)
    mapping={__import__("contracts").AttributeSpan(**a).attribute_id:a for a in output.attributes}
    for d in positions:
        attr=mapping[next(selected)]
        position_noops+=int(d["index"]==attr["char_end"]-attr["char_start"])
    return {"edits":len(output.edits),"NOOP_output":int(not output.edits),
            "attribute_decisions":len(attrs),"STOP_count":stops,
            "STOP_ratio":stops/len(attrs) if attrs else 0.,
            "position_decisions":len(positions),"position_NOOP_count":position_noops,
            "position_NOOP_ratio":position_noops/len(positions) if positions else 0.}

def train_step(policy,optimizer,example,sampler,make_reward,step,row_id):
    check_optimizer(policy,optimizer)
    if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
        raise ValueError("H log-prob must retain ordinary autograd graph")
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    before={k:p.detach().clone() for k,p in policy.named_parameters()}
    h_before=state_sha(policy)
    with torch.no_grad():
        greedy=policy.rollout(example,"greedy")
    sampled=policy.rollout(example,"sample",generator=sampler)
    if sampled.budget_k!=greedy.budget_k or not sampled.log_prob.requires_grad:
        raise ValueError("H sampling/autograd contract mismatch")
    for out in (sampled,greedy):
        if replay(example,out.edits,out.budget_k)!=out.adversarial:
            raise ValueError("current H Unicode trace mismatch")
    evaluator=make_reward(sampled.budget_k)
    sr,gr=evaluator.evaluate_many(example.caption,(sampled.adversarial,greedy.adversarial))
    loss,advantage=scst_loss(sampled.log_prob,sr.total,gr.total)
    loss.backward()
    grads={n:p.grad for n,p in policy.named_parameters() if p.grad is not None}
    if not grads or any(not torch.isfinite(x).all() for x in grads.values()):
        raise ValueError("missing/nonfinite H gradients")
    gradient=math.sqrt(sum(float(x.double().square().sum()) for x in grads.values()))
    group_grads={}
    for prefix in ("attribute_head","position_head","replacement_head","byte_embedding","encoder"):
        group_grads[prefix]=math.sqrt(sum(float(x.double().square().sum()) for n,x in grads.items() if n.startswith(prefix+".")))
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in policy.parameters()):
        raise ValueError("nonfinite updated H parameter")
    changes=[(p.detach().double()-before[n].double()) for n,p in policy.named_parameters()]
    log={"step":step,"row_id":row_id,"H_state_sha256_before":h_before,"H_state_sha256_after":state_sha(policy),
        "sampled_reward":asdict(sr),"greedy_reward":asdict(gr),"advantage":float(advantage),
        "SCST_loss":float(loss.detach()),"sequence_log_prob":float(sampled.log_prob.detach()),
        "gradient_norm":gradient,"group_gradient_norm":group_grads,
        "parameter_delta_l2":math.sqrt(sum(float(x.square().sum()) for x in changes)),
        "parameter_delta_max_abs":max(float(x.abs().max()) for x in changes),
        "K":sampled.budget_k,"sampled":sampled.record(),"greedy":greedy.record(),
        "sampled_action_stats":action_stats(sampled),"greedy_action_stats":action_stats(greedy),
        "optimizer_steps_this_step":1,**evaluator.diagnostics()}
    evaluator.runtime.assert_frozen()
    return log,evaluator.features.detach().clone()
