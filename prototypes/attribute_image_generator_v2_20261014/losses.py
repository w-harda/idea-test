"""独立监督loss：正dot最小化；无扰动正则、TV或G-v1依赖。"""
import torch

def features(value):
    if (not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] < 1
        or value.shape[1] != 512 or value.dtype not in (torch.float32, torch.float64)
        or not torch.isfinite(value).all()):
        raise ValueError("image features must be finite float32/float64 [B,512]")

def target(value, image_features, name):
    if (not isinstance(value, torch.Tensor) or value.shape != image_features.shape
        or value.device != image_features.device or value.dtype != image_features.dtype
        or value.requires_grad or not torch.isfinite(value).all()):
        raise ValueError(name + " must be matching detached supervision [B,512]")

def matching_scores(image_features, supervision):
    features(image_features); target(supervision, image_features, "supervision")
    return (image_features * supervision).sum(dim=1)

def retain_reference(previous_image_features, previous_jvp):
    """应在上一轮完成图像上取分数；返回断开图的[B]参考，不缓存训练图。"""
    return matching_scores(previous_image_features, previous_jvp).detach()

def retain_loss(image_features, previous_jvp=None, reference_score=None):
    """只保留上一轮属性：惩罚其正dot比stop-gradient参考分数升高。"""
    features(image_features)
    if previous_jvp is None and reference_score is None:
        return image_features.sum() * 0
    if previous_jvp is None or reference_score is None:
        raise ValueError("previous JVP and reference score must be supplied together")
    scores = matching_scores(image_features, previous_jvp)
    if (not isinstance(reference_score, torch.Tensor) or reference_score.shape != scores.shape
        or reference_score.device != scores.device or reference_score.dtype != scores.dtype
        or not torch.isfinite(reference_score).all()):
        raise ValueError("finite matching reference scores [B] required")
    return torch.relu(scores - reference_score.detach()).mean()

def attribute_losses(image_features, z_t, current_jvp, *, previous_jvp=None, reference_score=None):
    """一轮Global+当前Local+上一轮Retain，三个分项系数均为1；不更新参数。"""
    global_loss = matching_scores(image_features, z_t).mean()
    local_loss = matching_scores(image_features, current_jvp).mean()
    retain = retain_loss(image_features, previous_jvp, reference_score)
    return {"global":global_loss, "local":local_loss, "retain":retain,
            "total":global_loss + local_loss + retain}

def joint_losses(image_features, z_t, v_local, has_local):
    """原图单步联合目标；直接消费Phase 1聚合监督，不重加权或归一化。"""
    global_loss = matching_scores(image_features, z_t).mean()
    scores = matching_scores(image_features, v_local)
    if (not isinstance(has_local, torch.Tensor) or has_local.shape != scores.shape
        or has_local.dtype != torch.bool or has_local.device != scores.device):
        raise ValueError("has_local must be bool [B] on feature device")
    if v_local[~has_local].any():
        raise ValueError("K0 aggregated supervision must be exactly zero")
    local_loss = torch.where(has_local, scores, torch.zeros_like(scores)).mean()
    return {"global":global_loss, "local":local_loss, "total":global_loss + local_loss}
