"""G-v2独立累计预算投影；真实clamp导数，无STE或预算容差。"""
import math
import numbers
import torch

DEFAULT_EPSILON = 8 / 255

def checked_epsilon(epsilon):
    if (isinstance(epsilon, bool) or not isinstance(epsilon, numbers.Real)
        or not math.isfinite(epsilon) or not 0 <= epsilon <= DEFAULT_EPSILON):
        raise ValueError("epsilon must be finite and within [0, 8/255]")
    return float(epsilon)

def check_pixels(image, name="image"):
    if (not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] < 1
        or image.shape[1] != 3 or min(image.shape[2:]) < 1
        or image.dtype not in (torch.float32, torch.float64)
        or not torch.isfinite(image).all() or (image < 0).any() or (image > 1).any()):
        raise ValueError(name + " must be finite float32/float64 RGB [B,3,H,W] in [0,1]")

def check_like(value, reference, name):
    if (not isinstance(value, torch.Tensor) or value.shape != reference.shape
        or value.dtype != reference.dtype or value.device != reference.device
        or not torch.isfinite(value).all()):
        raise ValueError(name + " must match image shape/dtype/device and be finite")

def epsilon_tensor(reference, epsilon=DEFAULT_EPSILON):
    """向内舍入单轮上限，避免float32 epsilon被表示为更大的数。"""
    epsilon = checked_epsilon(epsilon)
    cap = reference.new_tensor(epsilon)
    with torch.no_grad():
        if float(cap.double()) > epsilon:
            cap = torch.nextafter(cap, torch.zeros_like(cap))
    return cap

def projection_bounds(original, epsilon=DEFAULT_EPSILON):
    """原图仅作固定参考；向内舍入可表示端点，防止fp32累计预算越界。"""
    check_pixels(original, "original")
    epsilon = checked_epsilon(epsilon)
    with torch.no_grad():
        anchor = original.detach().double()
        lower = (anchor - epsilon).clamp(0, 1).to(original.dtype)
        upper = (anchor + epsilon).clamp(0, 1).to(original.dtype)
        lower = torch.where(anchor - lower.double() > epsilon,
            torch.nextafter(lower, torch.ones_like(lower)), lower)
        upper = torch.where(upper.double() - anchor > epsilon,
            torch.nextafter(upper, torch.zeros_like(upper)), upper)
    return lower, upper

def project_image(original, proposed, epsilon=DEFAULT_EPSILON):
    """允许proposed越界；梯度只流向proposed，原图参考stop-gradient。"""
    check_pixels(original, "original")
    check_like(proposed, original, "proposed")
    lower, upper = projection_bounds(original, epsilon)
    return torch.clamp(proposed, min=lower, max=upper)

def add_projected_delta(original, current, delta, epsilon=DEFAULT_EPSILON):
    """只做一轮合成，不隐式detach current；轮间detach由未来训练循环负责。"""
    epsilon = checked_epsilon(epsilon)
    check_pixels(original, "original"); check_pixels(current, "current")
    check_like(current, original, "current"); check_like(delta, original, "delta")
    if (current.detach().double() - original.detach().double()).abs().max() > epsilon:
        raise ValueError("current exceeds cumulative budget relative to original")
    if delta.detach().double().abs().max() > epsilon:
        raise ValueError("delta exceeds single-round budget")
    return project_image(original, current + delta, epsilon)
