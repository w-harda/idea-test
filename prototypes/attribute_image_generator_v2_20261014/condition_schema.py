"""G-v2逐属性训练监督缓存；不包含生成器或训练循环。"""
from pathlib import PurePosixPath
import torch
SCHEMA = "gv2-attribute-supervision-1"
IDENTITY = ("row_id", "image_rel_path", "caption_index", "split", "role", "pid", "caption")
ATOL, RTOL = 1e-9, 1e-5

def check_identity(x):
    if not isinstance(x, dict) or any(k not in x for k in IDENTITY):
        raise ValueError("missing complete sample identity")
    parts = x["row_id"].split(":") if isinstance(x["row_id"], str) else []
    if (len(parts) != 3 or parts[0] != "cuhk" or not parts[1].isdigit()
        or not parts[2].isdigit() or type(x["caption_index"]) is not int
        or x["caption_index"] != int(parts[2])):
        raise ValueError("row_id/caption_index mismatch")
    p = x["image_rel_path"]
    if (not isinstance(p, str) or not p or "\\" in p
        or PurePosixPath(p).is_absolute() or ".." in PurePosixPath(p).parts
        or ":" in p or str(PurePosixPath(p)) != p):
        raise ValueError("invalid relative image path")
    if x["split"] != "train" or x["role"] not in ("train", "validation"):
        raise ValueError("only official train / internal train-validation allowed")
    if type(x["pid"]) is not int or x["pid"] < 1 or not isinstance(x["caption"], str) or not x["caption"]:
        raise ValueError("invalid PID/caption")
    return tuple(x[k] for k in IDENTITY)

def tensor(x, shape, dtype, name):
    if (not isinstance(x, torch.Tensor) or tuple(x.shape) != shape or x.dtype != dtype
        or x.device.type != "cpu" or x.requires_grad or not torch.isfinite(x).all()):
        raise ValueError("invalid detached CPU tensor: " + name)

def reconstruct(value):
    if value["K"] == 0: return torch.zeros(512, dtype=torch.float32)
    return (value["score_weights"].float()[:, None] * value["attribute_jvps"]).sum(0)

def error(actual, expected):
    d = actual.double() - expected.double(); denom = float(expected.double().norm())
    return {"max_abs_error": float(d.abs().max()), "l2_error": float(d.norm()),
            "relative_l2_error": float(d.norm()) / denom if denom else (0.0 if not d.any() else None),
            "exact": torch.equal(actual, expected)}

def validate(value, expected_identity=None):
    if value.get("schema_version") != SCHEMA: raise ValueError("G-v2 cache schema mismatch")
    identity = check_identity(value)
    if expected_identity is not None and identity != check_identity(expected_identity):
        raise ValueError("sample identity mismatch")
    k = value.get("K")
    if type(k) is not int or k < 0: raise ValueError("invalid K")
    ids = value.get("selected_instance_ids")
    if (not isinstance(ids, list) or len(ids) != k or any(not isinstance(i, str) or not i for i in ids)
        or len(set(ids)) != k): raise ValueError("attribute ID coverage")
    for name, shape, dtype in (
        ("S_raw", (k,), torch.float64), ("score_weights", (k,), torch.float64),
        ("attribute_jvps", (k,512), torch.float32), ("z_t", (512,), torch.float32),
        ("v_local", (512,), torch.float32)):
        tensor(value.get(name), shape, dtype, name)
    if abs(float(value["z_t"].norm()) - 1) > 2e-4: raise ValueError("non-unit full-caption retrieval feature")
    if type(value.get("has_local")) is not bool or value["has_local"] != (k > 0):
        raise ValueError("has_local/K mismatch")
    if k == 0:
        if ids != [] or value["S_raw"].numel() or value["score_weights"].numel() or value["v_local"].any():
            raise ValueError("K0 protocol violation")
    else:
        s = value["S_raw"]; w = value["score_weights"]
        if not (s > 0).all() or not (w > 0).all() or (s[1:] > s[:-1]).any():
            raise ValueError("positive descending frozen scores required")
        if not torch.allclose(w, s / s.sum(), atol=1e-15, rtol=1e-14):
            raise ValueError("original raw-score weights required")
        if (value["attribute_jvps"].double().norm(dim=1) == 0).any():
            raise ValueError("zero attribute JVP; no fallback")
    rebuilt = reconstruct(value); report = error(rebuilt, value["v_local"])
    if not torch.allclose(rebuilt, value["v_local"], atol=ATOL, rtol=RTOL):
        raise ValueError("aggregate reconstruction mismatch")
    return report

def build(identity, ids, scores, weights, jvps, z_t):
    value = {"schema_version": SCHEMA, **{k:identity[k] for k in IDENTITY},
             "K": len(ids), "selected_instance_ids": list(ids),
             "S_raw": torch.tensor(scores, dtype=torch.float64),
             "score_weights": torch.tensor(weights, dtype=torch.float64),
             "attribute_jvps": jvps.detach().cpu().float().clone(),
             "z_t": z_t.detach().cpu().float().reshape(-1).clone(), "has_local": bool(ids)}
    value["v_local"] = reconstruct(value)
    validate(value)
    return value
