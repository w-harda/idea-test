"""Step7.3 受控TTA：原更新仅加观测；NEW的随机变换选择供其它模式回放。"""
from __future__ import annotations
import ast
import copy
import hashlib
import json
import pickle
from contextlib import contextmanager
from pathlib import Path
import random
import numpy as np
import torch
from torch.nn import functional as F
from attribute_tta import FixedGuidanceBridge, IMAGE_EPSILON, OFFICIAL, project_image
from attributes.tta_adapter import load_official_attack, _tokenize

MODES = ("NEW", "RAW", "NONE")
DEFINITIONS = {
    "NEW": "-<image_feature, original-text-forward JVP(attribute)>; no caption loss term",
    "RAW": "-<image_feature, frozen IRRA encode_texts(selected raw source span)>; no caption term",
    "NONE": "-<image_feature, frozen IRRA encode_texts(original full caption)>; no attribute-specific term",
    "LEGACY": "-<image_feature, full-caption feature> - <image_feature, selected raw feature>; extra caption term, not strict controlled baseline",
}

def tensor_hash(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def rng_fingerprint():
    cuda = [x.cpu().numpy().tobytes() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else []
    data = (random.getstate(), np.random.get_state(), torch.get_rng_state().numpy().tobytes(), cuda)
    return hashlib.sha256(pickle.dumps(data)).hexdigest()

def norms(x):
    y = x.detach().double()
    return {"l2": float(y.norm()), "linf": float(y.abs().max())}

def cosine(a, b):
    if a is None or b is None or not a.double().norm() > 0 or not b.double().norm() > 0:
        return None
    return float(F.cosine_similarity(a.detach().double().flatten(), b.detach().double().flatten(), dim=0))

def projection_stats(before, proposal, after):
    pre, actual, removed = proposal-before, after-before, proposal-after
    pre_l1 = float(pre.detach().double().abs().sum())
    pre_l2 = float(pre.detach().double().norm())
    return {
        "pre_projection_update": norms(pre),
        "post_projection_actual_change": norms(actual),
        "projection_removed": norms(removed),
        "projection_clipping_ratio": float((proposal != after).float().mean()),
        # 首步的随机初始化可能已越过原图预算，removed/pre可能>1；不可当作概率。
        "projection_removed_l2_over_requested_l2": float(removed.double().norm())/pre_l2 if pre_l2 else None,
        "absolute_step_magnitude_removed_fraction": float((pre.detach().double().abs()-actual.detach().double().abs()).clamp_min(0).sum())/pre_l1 if pre_l1 else None,
        "actual_change_exact_zero": bool(torch.equal(after, before)),
        "actual_change_linf_le_1e_8": norms(actual)["linf"] <= 1e-8,
    }

class CategoricalTape:
    """仅耦合抽到的变换ID；loss、概率权重及其原样梯度更新各模式独立保留。"""
    def __init__(self, records=None):
        self.replay = records is not None
        self.records = copy.deepcopy(records) if self.replay else []
        self.position = 0

    def multinomial(self, probs, num_samples, replacement=False, **kwargs):
        # 仍调用原采样器以保持RNG消耗；回放时只替换采样结果。
        sampled = torch.multinomial(probs, num_samples, replacement, **kwargs)
        signature = [list(probs.shape), int(num_samples), bool(replacement)]
        if self.replay:
            if self.position >= len(self.records):
                raise ValueError("replay exhausted")
            row = self.records[self.position]
            if row["signature"] != signature:
                raise ValueError("replay sampling topology changed")
            sampled = torch.tensor(row["ids"], dtype=sampled.dtype, device=sampled.device).reshape(sampled.shape)
        else:
            self.records.append({"signature": signature, "ids": sampled.detach().cpu().tolist()})
        self.position += 1
        return sampled

    def finish(self):
        if self.position != len(self.records):
            raise ValueError("replay not completely consumed")
        payload = json.dumps(self.records, separators=(",", ":")).encode()
        return {"categorical_draw_count": self.position, "categorical_trace_sha256": hashlib.sha256(payload).hexdigest()}

class TorchProxy:
    def __init__(self, tape):
        self.tape = tape
    def __getattr__(self, name):
        return getattr(torch, name)
    def multinomial(self, *args, **kwargs):
        return self.tape.multinomial(*args, **kwargs)

class UpdateObserver:
    def __init__(self):
        self.events = []
        self.before = None
        self.last_proposal = None
        self.last_gradient = None
        self.noise_sha256 = None
        self.initialization = None

    def capture_noise(self, noise):
        self.noise_sha256 = hashlib.sha256(noise.tobytes()).hexdigest()
        return noise

    def pre_step(self, before, sign_step, ig_grad, momentum):
        self.before = before.detach().clone()
        self.last_proposal = (before.detach()+sign_step.detach()).clone()
        self.pending = {
            "step_index": len(self.events)+1,
            "ig_gradient_l2": norms(ig_grad)["l2"],
            "momentum_l2": norms(momentum)["l2"],
            "ig_gradient_cosine_previous_step": cosine(ig_grad, self.last_gradient),
            "pre_step_linf_from_original_pending": None,
        }
        self.last_gradient = ig_grad.detach().clone()

    def after_clamp(self, after, incoming, original):
        if self.before is None:
            self.initialization = {
                "noise_sha256": self.noise_sha256,
                "initialization_actual_change": norms(after-incoming),
                "initialized_linf_from_original": norms(after-original)["linf"],
                "note": "原TTA每轮噪声只clamp[0,1]；此内部起点不保证满足累计epsilon。",
            }
            return
        item = projection_stats(self.before, self.last_proposal, after)
        item.update(self.pending)
        item["pre_step_linf_from_original"] = norms(self.before-original)["linf"]
        item["cumulative_perturbation_linf"] = norms(after-original)["linf"]
        self.events.append(item)
        self.before = None

def observed_attack_type(tape, official=OFFICIAL):
    """对已SHA锁定的img_attack AST添加只读observer，不改官方文件/数学语句。"""
    base = load_official_attack(official)
    tree = ast.parse((Path(official)/"attacker_TTA.py").read_text(encoding="utf-8"))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "Attack")
    method = copy.deepcopy(next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "img_attack"))
    counts = {"noise": 0, "clamp": 0, "pre": 0}
    class Instrument(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if ast.unparse(node.func) == "np.random.uniform":
                counts["noise"] += 1
                return ast.copy_location(ast.Call(
                    func=ast.parse("self._audit.capture_noise", mode="eval").body,
                    args=[node], keywords=[]), node)
            return node
        def visit_Assign(self, node):
            node = self.generic_visit(node)
            targets = [ast.unparse(t) for t in node.targets]
            if targets == ["imgs_0"] and ast.unparse(node.value) == "imgs_0.detach() + perturbation":
                counts["pre"] += 1
                call = ast.parse("self._audit.pre_step(imgs_0, perturbation, ig_grad, grad)").body[0]
                return [ast.copy_location(call, node), node]
            if targets == ["imgs_0"] and ast.unparse(node.value) == "torch.clamp(imgs_0, 0.0, 1.0)":
                counts["clamp"] += 1
                call = ast.parse("self._audit.after_clamp(imgs_0, imgs, origin_imgs)").body[0]
                return [node, ast.copy_location(call, node)]
            return node
    method = Instrument().visit(method)
    if counts != {"noise": 1, "clamp": 2, "pre": 1}:
        raise ValueError("official instrumentation anchors changed: "+str(counts))
    scope = dict(base.img_attack.__globals__)
    scope["torch"] = TorchProxy(tape)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "<step73-observed-img_attack>", "exec"), scope)
    # 其它原TTA helper亦共享proxy，覆盖block/operation的全部categorical sampling。
    base.img_attack.__globals__["torch"] = scope["torch"]
    return type("ObservedAttack", (base,), {"img_attack": scope["img_attack"]})

def guidance(mode, attribute, new_vectors, raw_features, clean_text):
    aid = attribute.instance_id
    if mode == "NEW":
        return new_vectors[aid], ["attribute_unit:"+aid]
    if mode == "RAW":
        return raw_features[aid], [attribute.metadata["raw_text"]]
    if mode == "NONE":
        return clean_text, ["full_caption_no_attribute"]
    if mode == "LEGACY":
        return torch.cat([clean_text, raw_features[aid]], dim=0), ["legacy_caption", attribute.metadata["raw_text"]]
    raise ValueError("unknown guidance mode")

class ControlledTTA:
    def __init__(self, source, original, mode, *, steps, transforms, scales, tape):
        self.source = source.eval()
        self.original = original.detach().clone()
        self.mode, self.steps, self.scales, self.tape = mode, steps, scales, tape
        self.attack = observed_attack_type(tape)(None, _tokenize, imgs_eps=IMAGE_EPSILON, step_size=2/255)
        self.attack.N_trans = transforms
        self.momentum = torch.zeros_like(original)
        self.previous_actual_change = None
        self.previous_raw_gradient = None
        self.events = []

    def step(self, current, attribute, features, labels):
        if norms(current-self.original)["linf"] > IMAGE_EPSILON+1e-6:
            raise ValueError("incoming budget invalid")
        bridge = FixedGuidanceBridge(self.source, features, labels)
        self.attack._audit = observer = UpdateObserver()
        with torch.enable_grad():
            proposed, momentum = self.attack.img_attack(
                bridge, list(labels), current.detach(), self.original,
                [0]*len(labels), self.steps, self.momentum, current.device, scales=self.scales)
            # 对未改变的round输入补做诊断梯度；不在官方更新前保留额外GPU图/梯度工作区。
            x = current.detach().clone().requires_grad_(True)
            loss = -(self.source.encode_images(x)*features.sum(0, keepdim=True).detach()).sum()
            gradient, = torch.autograd.grad(loss, x)
        if not torch.isfinite(proposed).all() or not torch.isfinite(momentum).all():
            raise ValueError("nonfinite official TTA output")
        final = project_image(self.original, proposed.detach()).detach()
        self.momentum = momentum.detach()
        if len(observer.events) != self.steps or observer.noise_sha256 is None:
            raise ValueError("incomplete update observation")
        actual = final-current
        round_stats = projection_stats(current, observer.last_proposal, final)
        a = attribute.metadata
        round_stats.update({
            "round_index": attribute.selection_rank, "instance_id": attribute.instance_id,
            "region": a["region"], "semantic_axis": a["semantic_axis"], "canonical_value": a["canonical_value"],
            "S_raw": a["S_raw"], "source_span": copy.deepcopy(a["source_span"]),
            "labels_passed": list(labels), "caption_loss_terms": int(self.mode in ("NONE", "LEGACY")),
            "attribute_loss_terms": int(self.mode != "NONE"), "text_encoder_calls_during_attack": 0,
            "image_gradient_l2_at_round_start": norms(gradient)["l2"],
            "image_gradient_cosine_previous_round_start": cosine(gradient, self.previous_raw_gradient),
            "actual_post_projection_update_cosine_previous_round": cosine(actual, self.previous_actual_change),
            "cumulative_perturbation_linf": norms(final-self.original)["linf"],
            "initialization": observer.initialization, "steps": observer.events,
            "projection_clipping_ratio_mean_over_steps": float(np.mean([s["projection_clipping_ratio"] for s in observer.events])),
            "outer_projection_changed_official_return": not torch.equal(final, proposed.detach()),
            "post_projection_image_sha256": tensor_hash(final),
        })
        self.previous_actual_change = actual.detach().clone()
        self.previous_raw_gradient = gradient.detach().clone()
        self.events.append(round_stats)
        return final

class RetrievalMetrics:
    """固定clean train竞争分数，仅动态评价paired image；不生成/评价正式adversarial gallery。"""
    def __init__(self, source, clean_text, gallery_features, paired_index, tau=.02):
        if len(gallery_features.shape) != 2 or clean_text.shape[0] != 1 or not 0 <= paired_index < len(gallery_features):
            raise ValueError("invalid paired evaluation inputs")
        self.source, self.text = source, clean_text.detach().clone()
        self.tau = tau
        scores = (self.text @ gallery_features.T).flatten().detach()
        self.others = torch.cat([scores[:paired_index], scores[paired_index+1:]])
        self.paired_index = paired_index
    @torch.no_grad()
    def __call__(self, image):
        feature = self.source.encode_images(image)
        paired = (self.text*feature).sum()
        # 原paired_soft_rank的同一公式：1+sum_{j!=paired} sigmoid((s_j-s_plus)/tau)
        rank = 1+torch.sigmoid((self.others-paired)/self.tau).sum()
        return {"paired_cosine": float(paired), "paired_soft_rank_F": float(rank),
                "positive_image_rank": 1+int((self.others>paired).sum()),
                "positive_image_rank_tie_count": int((self.others==paired).sum())}

def metric_delta(before, after):
    return {key: after[key]-before[key] for key in ("paired_cosine", "paired_soft_rank_F", "positive_image_rank")}
