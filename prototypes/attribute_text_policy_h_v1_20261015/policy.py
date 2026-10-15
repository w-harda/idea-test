"""轻量UTF-8 Byte Transformer；三Head与无放回属性动作，测试仅文本。"""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F
from contracts import AXES, REGIONS, TextExample, word_at
from constraints import REPLACEMENTS, candidates, make_edit, replay

def natural_k(scores):
    """沿用gamma=3/beta=1有效数公式；输入是H预测分数，绝非配对S_raw。"""
    if any(not math.isfinite(s) or s < 0 for s in scores):
        raise ValueError("finite nonnegative predicted importance required")
    logs = [3*math.log(s) for s in scores if s > 0]
    if not logs:
        return 0
    scale = max(logs)
    values = [math.exp(v-scale) for v in logs]
    total = sum(values)
    probabilities = [v/total for v in values]
    entropy = -sum(p*math.log(p) for p in probabilities if p > 0)
    return min(len(logs), math.ceil(math.exp(entropy)))

def choose(logits, mask, mode, generator=None):
    if mode not in ("sample", "greedy"):
        raise ValueError("sample or greedy required")
    if logits.ndim != 1 or mask.shape != logits.shape or mask.dtype != torch.bool:
        raise ValueError("invalid action mask shape")
    if not torch.isfinite(logits).all() or not mask.any():
        raise ValueError("nonfinite logits or empty action support")
    log_probs = F.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=0)
    index = (int(torch.multinomial(log_probs.exp(), 1, generator=generator))
             if mode == "sample" else int(log_probs.argmax()))
    return index, log_probs[index]

@dataclass
class PolicyOutput:
    original: str
    adversarial: str
    budget_k: int
    predicted_importance: tuple[float, ...]
    selected_attribute_ids: tuple[str, ...]
    edits: tuple
    decisions: tuple[dict, ...]
    log_prob: torch.Tensor
    tokenizer_audit: dict
    attributes: tuple[dict, ...]

    def record(self):
        return {"original": self.original, "adversarial": self.adversarial,
                "budget_k": self.budget_k,
                "predicted_importance": self.predicted_importance,
                "attributes": self.attributes,
                "selected_attribute_ids": self.selected_attribute_ids,
                "edits": [asdict(e) for e in self.edits],
                "decisions": self.decisions,
                "tokenizer_audit": self.tokenizer_audit}

class CharacterPolicy(nn.Module):
    def __init__(self, probe, width=64, layers=2, heads=4, max_bytes=1024):
        super().__init__()
        if (any(type(v) is not int or v < 1 for v in (width, layers, heads, max_bytes))
                or width < 4 or width % heads):
            raise ValueError("invalid lightweight Transformer configuration")
        self.probe = probe
        self.max_bytes = max_bytes
        self.byte_embedding = nn.Embedding(258, width, padding_idx=0)
        self.position_embedding = nn.Embedding(max_bytes+1, width)
        self.axis_embedding = nn.Embedding(len(AXES), width)
        self.region_embedding = nn.Embedding(len(REGIONS), width)
        layer = nn.TransformerEncoderLayer(width, heads, width*2, dropout=0.,
                                           activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.attribute_head = nn.Linear(width, 1)
        self.position_head = nn.Linear(width*2, 1)
        self.replacement_head = nn.Linear(width*2, len(REPLACEMENTS))

    def forward(self, example):
        if not isinstance(example, TextExample):
            raise TypeError("TextExample only; no image/PID/score input")
        encoded = example.caption.encode("utf-8")
        if len(encoded) > self.max_bytes:
            raise ValueError("caption exceeds H byte limit; no silent truncation")
        device = self.byte_embedding.weight.device
        if device.type != "cpu":
            raise ValueError("Phase 1 only permits CPU policy execution")
        ids = torch.tensor([1, *[b+2 for b in encoded]], device=device)
        h = self.encoder((self.byte_embedding(ids)
                         +self.position_embedding(torch.arange(len(ids), device=device)))[None])[0]
        chars, offset = [], 1
        for character in example.caption:
            size = len(character.encode("utf-8"))
            chars.append(h[offset:offset+size].mean(0))
            offset += size
        char_h = torch.stack(chars)
        attrs = []
        for attr in example.attributes:
            attrs.append(char_h[attr.char_start:attr.char_end].mean(0)
                         +self.axis_embedding.weight[AXES.index(attr.semantic_axis)]
                         +self.region_embedding.weight[REGIONS.index(attr.region)])
        attr_h = torch.stack(attrs) if attrs else h.new_empty((0, h.shape[1]))
        return h[0], char_h, attr_h

    def rollout(self, example, mode="sample", generator=None):
        if mode not in ("sample", "greedy"):
            raise ValueError("unsupported policy mode")
        pooled, chars, attrs = self(example)
        logits = self.attribute_head(attrs).flatten()
        importance = F.softplus(logits)
        raw = tuple(float(x) for x in importance.detach())
        initial = [candidates(example, example.caption, i, set(), self.probe)
                   for i in range(len(example.attributes))]
        available = [bool(c) for c in initial]
        k = natural_k([raw[i] for i, valid in enumerate(available) if valid])
        current = example.caption
        edits, selected, decisions, logps = [], [], [], []
        used_words = set()
        for step in range(k):
            supports = [candidates(example, current, i, used_words, self.probe)
                        if available[i] else {} for i in range(len(attrs))]
            # STOP始终合法；无候选时它是唯一合法动作，不贡献伪梯度。
            mask = torch.tensor([bool(s) for s in supports]+[True])
            a_logits = torch.cat((logits, self.attribute_head(pooled).reshape(1)))
            ai, lp = choose(a_logits, mask, mode, generator)
            logps.append(lp)
            decisions.append({"head": "attribute", "index": ai,
                              "legal_indices": mask.nonzero().flatten().tolist()})
            if ai == len(attrs):
                break
            available[ai] = False
            selected.append(example.attributes[ai].attribute_id)
            attr = example.attributes[ai]
            positions = list(range(attr.char_start, attr.char_end))
            pos_features = torch.stack([torch.cat((chars[p], attrs[ai])) for p in positions]
                                       +[torch.cat((pooled, attrs[ai]))])
            pos_logits = self.position_head(pos_features).flatten()
            pos_mask = torch.tensor([p in supports[ai] for p in positions]+[True])
            pi, lp = choose(pos_logits, pos_mask, mode, generator)
            logps.append(lp)
            decisions.append({"head": "position", "index": pi,
                              "legal_indices": pos_mask.nonzero().flatten().tolist()})
            if pi == len(positions):  # 本属性不替换，仍消耗一个选择槽。
                continue
            pos = positions[pi]
            r_logits = self.replacement_head(torch.cat((chars[pos], attrs[ai])))
            r_mask = torch.tensor([r in supports[ai][pos] for r in REPLACEMENTS])
            ri, lp = choose(r_logits, r_mask, mode, generator)
            logps.append(lp)
            decisions.append({"head": "replacement", "index": ri,
                              "legal_indices": r_mask.nonzero().flatten().tolist()})
            edits.append(make_edit(example, ai, pos, REPLACEMENTS[ri]))
            used_words.add(word_at(example.caption, pos))
            current = replay(example, edits, k)
        # K=0构建可backward的精确0，不发明编辑或奖励。
        log_prob = torch.stack(logps).sum() if logps else pooled.sum()*0.
        return PolicyOutput(example.caption, current, k, raw, tuple(selected),
                            tuple(edits), tuple(decisions), log_prob,
                            self.probe.pair_report(example.caption, current),
                            tuple(asdict(a) for a in example.attributes))

    @torch.inference_mode()
    def attack(self, example):
        self.eval()
        return self.rollout(example, mode="greedy")
