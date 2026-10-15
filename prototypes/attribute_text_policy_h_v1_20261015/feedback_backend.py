"""真实奖励的注入式契约及CPU Mock执行器；本阶段没有真实模型加载入口。"""
from dataclasses import asdict
import hashlib
import importlib.util
import math
import torch
from torch import nn
from torch.nn import functional as F
from phase2_sources import G2, ROOT, EPOCH10_SHA, IRRA_SHA, sha
from rewards import TrainGallery, RewardEvaluator, frozen_features
from constraints import make_edit, replay
from contracts import TextExample

def state_sha(module):
    h = hashlib.sha256()
    for name, x in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(str((tuple(x.shape), x.dtype)).encode())
        h.update(x.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

class FrozenMockOps:
    """仅接收显式CPU Mock。下一阶段另加真实加载器，不能将Mock缓存冒充真实特征。"""
    def __init__(self, generator, image_encoder, text_encoder, probe):
        self.modules = (generator, image_encoder, text_encoder)
        for module in self.modules:
            if not isinstance(module, nn.Module) or getattr(module, "h_cpu_mock", False) is not True:
                raise ValueError("Phase 2 permits explicit CPU mock modules only")
            if any(x.device.type != "cpu" for x in (*module.parameters(), *module.buffers())):
                raise ValueError("CPU only")
            module.eval().requires_grad_(False)
        self.generator, self.image_encoder, self.text_encoder = self.modules
        from tokenizer_probe import TokenizerProbe, official_probe
        if (not isinstance(probe, TokenizerProbe)
                or type(probe.tokenizer).__module__ != "h_v1_official_tokenizer"):
            raise ValueError("verified official tokenizer required even for CPU mock")
        # 重验源文件/BPE并比对实际词表，不能仅靠类型名声明官方Tokenizer。
        reference = official_probe()
        if (probe.tokenizer.encoder != reference.tokenizer.encoder
                or probe.tokenizer.bpe_ranks != reference.tokenizer.bpe_ranks):
            raise ValueError("tokenizer vocabulary/BPE differs from frozen official source")
        self.probe = probe
        path = G2/"budget_projection.py"
        if sha(path) != "a4c391b5d0f4a1d4ddcac0431431615de86fd982eaaa009a0774c2cc3fbd88c0":
            raise ValueError("frozen projection source SHA mismatch")
        spec = importlib.util.spec_from_file_location("h_phase2_frozen_projection", path)
        self.projection = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.projection)
        self.initial = tuple(state_sha(m) for m in self.modules)
        self.counts = {"G_forward": 0, "IRRA_image_forward": 0, "IRRA_text_forward": 0,
                       "real_model_forward": 0}

    def assert_frozen(self):
        if (tuple(state_sha(m) for m in self.modules) != self.initial
                or any(m.training or any(p.requires_grad or p.grad is not None for p in m.parameters())
                       for m in self.modules)):
            raise ValueError("frozen mock model state/gradient changed")

    @property
    def protocol(self):
        return {"execution_backend": "cpu_mock", "mock_model_sha256": self.initial,
                "implementation_sha256": {n:sha(ROOT/n) for n in
                    ("feedback_backend.py", "feature_cache.py", "training_inputs.py",
                     "phase2_sources.py", "tokenizer_probe.py", "rewards.py", "constraints.py",
                     "config_phase2.json")},
                "projection_sha256": sha(G2/"budget_projection.py"),
                "feature_space": "mock-unit-float32-512; NOT official IRRA",
                "future_real_contract": {"G_epoch10_sha256": EPOCH10_SHA, "IRRA_sha256": IRRA_SHA,
                    "image_size_hw": [384,128], "epsilon": 8/255,
                    "pixel_space": "[0,1] before differentiable IRRA mean/std; normalized retrieval 512",
                    "G_input": "image-only", "G_forwards_per_gallery_image": 1},
                "tokenizer_sha256": __import__("tokenizer_probe").TOKENIZER_SHA,
                "bpe_sha256": __import__("tokenizer_probe").BPE_SHA}

    @staticmethod
    def feature(x, count):
        if not isinstance(x, torch.Tensor) or x.shape != (count,512) or x.device.type != "cpu":
            raise ValueError("CPU encoder output [B,512] required")
        x = F.normalize(x.detach().float(), dim=1).clone()
        frozen_features(x, torch.Size((count,512)))
        return x

    @torch.no_grad()
    def image_pair(self, pixels):
        self.assert_frozen()
        self.projection.check_pixels(pixels)
        if pixels.device.type != "cpu" or pixels.dtype != torch.float32 or pixels.shape[0] != 1:
            raise ValueError("single CPU float32 image required")
        anchor = pixels.detach().clone()
        # 只一次G前向；不读取Caption/属性/PID。
        delta = self.generator(anchor.clone())
        self.counts["G_forward"] += 1
        adv = self.projection.add_projected_delta(anchor, anchor, delta)
        clean_feature = self.feature(self.image_encoder(anchor.clone()), 1)
        adv_feature = self.feature(self.image_encoder(adv.clone()), 1)
        self.counts["IRRA_image_forward"] += 2
        self.assert_frozen()
        return {"clean": clean_feature, "adversarial": adv_feature}, {
            "linf": float((adv.double()-anchor.double()).abs().max()),
            "G_forward": 1, "IRRA_image_forward": 2, "real_model_forward": 0}

    @torch.no_grad()
    def encode_text(self, texts):
        self.assert_frozen()
        reports = [self.probe.inspect(t) for t in texts]
        if any(r.truncated for r in reports):
            raise ValueError("text truncation forbidden for reward feedback")
        tokens = torch.tensor([r.model_tokens for r in reports], dtype=torch.long)
        features = self.feature(self.text_encoder(tokens), len(texts))
        self.counts["IRRA_text_forward"] += 1
        self.assert_frozen()
        return features

def validate_variant(example, variant, budget, probe):
    """真实Unicode逐字符验证；不接受任意生成文本，允许原文NOOP。"""
    if not isinstance(example, TextExample) or type(variant) is not str or len(variant) != len(example.caption):
        raise ValueError("only one-for-one Unicode substitutions accepted")
    edits, used = [], set()
    for pos, (a,b) in enumerate(zip(example.caption, variant)):
        if a == b:
            continue
        support = [i for i,x in enumerate(example.attributes)
                   if x.char_start <= pos < x.char_end and i not in used]
        if not support:
            raise ValueError("edit outside canonical source span")
        edit = make_edit(example, support[0], pos, b)
        used.add(support[0])
        edits.append(edit)
    if replay(example, edits, budget) != variant:
        raise ValueError("invalid Unicode action replay")
    if variant != example.caption and not probe.legal_change(example.caption, variant):
        raise ValueError("tokenizer-equivalent or truncated replacement forbidden")
    if probe.inspect(example.caption).truncated:
        raise ValueError("original caption truncates")
    return tuple(edits)

class BoundReward:
    """只接受已验证训练query；PID留在绑定层，绝不进入文本策略。"""
    def __init__(self, catalog, query, gallery_samples, features, ops, frontend, budget_k):
        catalog.require_train(query)
        samples = tuple(catalog.require_train(s) for s in gallery_samples)
        if len({s.row_id for s in samples}) != len(samples):
            raise ValueError("duplicate gallery identity")
        if query not in samples:
            raise ValueError("paired training image missing")
        self.example = frontend(query.caption)
        if type(budget_k) is not int or not 0 <= budget_k <= len(self.example.attributes):
            raise ValueError("invalid H predicted editing budget")
        self.budget = budget_k
        self.ops = ops
        self.query = query
        ctx = TrainGallery(features["clean"], features["adversarial"],
            torch.tensor([s.pid == query.pid for s in samples], dtype=torch.bool),
            samples.index(query), "train", ("train",)*len(samples),
            "train", ("train",)*len(samples))
        # 固定沿用第一阶段默认1/1与best-positive Margin，不额外优化synergy。
        self.evaluator = RewardEvaluator(ctx, ops.encode_text, 1., 1.)

    def evaluate_many(self, original, variants):
        if original != self.query.caption:
            raise ValueError("query Caption does not match verified paired source")
        for text in variants:
            validate_variant(self.example, text, self.budget, self.ops.probe)
        self.ops.assert_frozen()
        result = self.evaluator.evaluate_many(original, variants)
        for reward in result:
            if (not all(math.isfinite(v) for v in asdict(reward).values())
                    or reward.synergy_increment != reward.adv_sim_drop-reward.clean_sim_drop
                    or reward.total != reward.R_sim+reward.R_rank):
                raise ValueError("reward contract/synergy inconsistency")
        self.ops.assert_frozen()
        return result
