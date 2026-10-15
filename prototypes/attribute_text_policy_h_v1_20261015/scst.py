"""SCST合成CPU训练接口；无真实数据/G/IRRA加载入口。"""
from dataclasses import dataclass, asdict
import math
import torch

def scst_loss(log_prob, sampled_reward, greedy_reward):
    if log_prob.ndim or not torch.isfinite(log_prob):
        raise ValueError("finite scalar sequence log-prob required")
    sample = torch.as_tensor(sampled_reward, device=log_prob.device, dtype=log_prob.dtype)
    greedy = torch.as_tensor(greedy_reward, device=log_prob.device, dtype=log_prob.dtype)
    if sample.ndim or greedy.ndim or not torch.isfinite(sample) or not torch.isfinite(greedy):
        raise ValueError("finite scalar rewards required")
    advantage = (sample-greedy).detach()
    return -advantage*log_prob, advantage

def synthetic_scst_step(policy, example, evaluator, optimizer, generator=None):
    """仅CPU调用方注入的奖励接口；独立sample/greedy，无真实训练调度。"""
    if next(policy.parameters()).device.type != "cpu":
        raise ValueError("Phase 1 SCST only permits CPU")
    expected = {id(p) for p in policy.parameters() if p.requires_grad}
    actual = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if actual != expected:
        raise ValueError("optimizer must contain H parameters only")
    with torch.no_grad():
        greedy = policy.rollout(example, "greedy")
    sampled = policy.rollout(example, "sample", generator=generator)
    sr, gr = evaluator.evaluate_many(example.caption, (sampled.adversarial, greedy.adversarial))
    loss, advantage = scst_loss(sampled.log_prob, sr.total, gr.total)
    if not torch.isfinite(loss):
        raise ValueError("nonfinite SCST loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    if any(not torch.isfinite(g).all() for g in grads):
        raise ValueError("nonfinite H gradient")
    grad_norm = math.sqrt(sum(float(g.double().square().sum()) for g in grads))
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in policy.parameters()):
        raise ValueError("nonfinite H parameters")
    return {"loss": float(loss.detach()), "advantage": float(advantage),
            "gradient_norm": grad_norm, "sampled_reward": asdict(sr),
            "greedy_reward": asdict(gr), "sampled": sampled.record(),
            "greedy": greedy.record()}
