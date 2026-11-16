"""属性权重、条件动作优势和可选同模块梯度协调。"""
import math
import torch
from .settings import require
from .memory import ACTION_DIM

def action_features(edit, caption, edits_count):
    return [edit.position / max(1, len(caption)), ord(edit.original_char) / 1114111.,
            ord(edit.replacement_char) / 1114111., min(edits_count, 32) / 32.,
            (ord(edit.replacement_char) - ord(edit.original_char)) / 1114111.,
            (edit.position - edit.source_span[0]) / max(1, edit.source_span[1] - edit.source_span[0])]

def conditional_advantage(values, probabilities, selected):
    q = torch.as_tensor(values).detach()
    p = torch.as_tensor(probabilities, dtype=q.dtype).detach()
    require(q.ndim == 1 and q.shape == p.shape and len(q) > 0 and 0 <= selected < len(q),
            "nonempty complete conditional actions")
    require(torch.isfinite(q).all() and torch.isfinite(p).all() and (p >= 0).all()
            and math.isclose(float(p.sum()), 1., abs_tol=2e-6), "normalized legal policy probabilities")
    baseline = (q * p).sum()
    return (q[selected] - baseline).detach(), baseline.detach()

def credit_loss(group_log_probs, advantages, weight):
    require(len(group_log_probs) == len(advantages), "one advantage per actual edit")
    if not group_log_probs:
        return None
    lp = torch.stack(group_log_probs)
    advantage = lp.new_tensor(advantages).detach()
    return -weight * (lp * advantage).mean()

def coordinated_backward(global_loss, attribute_loss, parameters):
    """只协调同一个G参数空间中的两个目标；不比较G/H不同参数空间。"""
    parameters = list(parameters)
    ga = torch.autograd.grad(global_loss, parameters, retain_graph=True, allow_unused=True)
    gb = torch.autograd.grad(attribute_loss, parameters, allow_unused=True)
    a = [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, ga)]
    b = [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, gb)]
    dot = sum((x * y).sum() for x, y in zip(a, b))
    norm = sum(x.square().sum() for x in a)
    conflict = float(dot.detach()) < 0 and float(norm.detach()) > 0
    coefficient = dot / norm if conflict else dot * 0
    for p, x, y in zip(parameters, a, b):
        p.grad = (x + y - coefficient * x).detach()
    return {"conflict": conflict, "dot_before": float(dot.detach()),
            "global_norm_squared": float(norm.detach()), "projection_coefficient": float(coefficient.detach())}
