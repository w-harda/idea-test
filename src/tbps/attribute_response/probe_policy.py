"""固定合法轨迹探测集上的条件期望；避免仅用greedy时更新收益长期为零。"""
import hashlib
import torch
from torch.nn import functional as F
from .settings import require

def probe_actions(policy, example, *, version, count=4):
    seed = int.from_bytes(hashlib.sha256((example.caption+str(version)).encode()).digest()[:4], "little")
    sampler = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        outputs = [policy.rollout(example, "greedy")]
        outputs.extend(policy.rollout(example, "sample", generator=sampler) for _ in range(count-1))
    records = []; seen = set()
    for output in outputs:
        record = output.record()
        key = repr((record["decisions"], record["adversarial"]))
        if key not in seen:
            records.append(record); seen.add(key)
    return records

def trajectory_log_prob(policy, example, record):
    pooled, chars, attrs = policy(example)
    attr_logits = policy.attribute_head(attrs).flatten()
    logs = []; ai = pi = None
    for decision in record["decisions"]:
        head = decision["head"]
        if head == "attribute":
            logits = torch.cat((attr_logits, policy.attribute_head(pooled).reshape(1)))
            ai = decision["index"]
        elif head == "position":
            attr = example.attributes[ai]
            positions = list(range(attr.char_start, attr.char_end))
            features = torch.stack([torch.cat((chars[p], attrs[ai])) for p in positions]
                                   + [torch.cat((pooled, attrs[ai]))])
            logits = policy.position_head(features).flatten()
            pi = decision["index"]
        elif head == "replacement":
            attr = example.attributes[ai]
            position = attr.char_start + pi
            logits = policy.replacement_head(torch.cat((chars[position], attrs[ai])))
        else:
            raise ValueError("unknown original policy head")
        legal = decision["legal_indices"]
        mask = torch.zeros(len(logits), dtype=torch.bool)
        mask[legal] = True
        index = decision["index"]
        require(mask[index], "probe trajectory selected action remains in frozen legal support")
        logs.append(F.log_softmax(logits.masked_fill(~mask, -torch.inf), 0)[index])
    return torch.stack(logs).sum() if logs else pooled.sum()*0.
