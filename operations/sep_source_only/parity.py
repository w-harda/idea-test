"""真实固定 FP32 分数的 CPU Exact 对照，旧入口从基准 Git 对象取出。"""
import ast
from pathlib import Path
import random
import subprocess
from types import SimpleNamespace

from tbps.baselines.cpgc_sep import sep_bridge as bridge, sep_source as source
from tbps.baselines.cpgc_sep.dataset import require
from .common import BASE, ROOT, bits, seed, tensor_sha


def old_definition(module, name, **namespace):
    path = Path(module.__file__).relative_to(ROOT)
    text = subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT, text=True)
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {**vars(module), **namespace}
    exec(compile(ast.Module(body=[node], type_ignores=[]), f"{BASE}:{path}", "exec"), scope)
    return scope[name]


def assert_bits(stage, left, right):
    import torch
    require(left.shape == right.shape and left.dtype == right.dtype, f"{stage}: shape/dtype changed")
    if bits(left) != bits(right):
        a, b = (x.detach().cpu().contiguous().reshape(-1).view(torch.uint8) for x in (left, right))
        byte = int(torch.nonzero(a != b)[0])
        index = byte // left.element_size()
        raise ValueError(f"{stage}: first byte {byte}, element {index}, "
                         f"old={left.flatten()[index].item()!r}, new={right.flatten()[index].item()!r}, "
                         f"old SHA={tensor_sha(left)}, new SHA={tensor_sha(right)}")


def rng():
    import numpy as np
    import torch
    state = np.random.get_state()
    return random.getstate(), state[0], state[1].tobytes(), state[2:], bits(torch.get_rng_state())


def replay(factory, name, manifest, texts, scores):
    """只替换 I/O，调用生产 classifier 本体、其缓存、聚合及排序路径。"""
    import torch
    from unittest.mock import patch
    bank = SimpleNamespace(globals=torch.zeros((scores.shape[1], 1)))
    adapter = SimpleNamespace(space=SimpleNamespace(id="0" * 64))
    backend = torch.nn.Linear(1, 1).requires_grad_(False)
    # 创建占位参数不属于 replay 所测 RNG 区间。
    indices = {text: i for i, text in enumerate(texts)}
    encoded, scored = [], []

    def encode(_adapter, batch, _name):
        encoded.extend(batch)
        return torch.tensor([[indices[t]] for t in batch], dtype=torch.float32)

    class FrozenScorer:
        def __init__(self, *_):
            pass

        def slice_query(self, q, *_):
            return q

        def score(self, q, _g):
            i = int(q.payload[0, 0])
            scored.append(texts[i])
            return SimpleNamespace(payload=scores[i:i+1])

    with patch.dict(factory.__globals__, {"encode_texts": encode, "NativeScorer": FrozenScorer}):
        classifier = factory(adapter, backend, name, bank, manifest)
        stream = (*texts, texts[0], texts[-1])
        before = rng()
        logits = classifier(stream).logits
        require(rng() == before, "CPU replay RNG consumed")
    return logits, (tuple(classifier.cache), tuple(encoded), tuple(scored), classifier.calls, classifier.hits)


def compare(c, manifest, texts, matrices):
    import torch
    old_pid = old_definition(bridge, "pid_logits")
    old_classifier = old_definition(source, "classifier", pid_logits=old_pid)
    pids = tuple(g.pid for g in manifest.gallery)
    plan = bridge.PIDPlan.build(pids)
    reference_members = tuple(tuple(i for i, p in enumerate(pids) if p == pid) for pid in dict.fromkeys(pids))
    require(plan.members == reference_members, "PID membership/order")
    label = plan.order.index(manifest.queries[0].pid)
    streams = [{}, {}]
    evidence = {}
    for name, scores in matrices.items():
        require(scores.device.type == "cpu" and scores.dtype == torch.float32
                and scores.shape == (len(texts), len(pids)) and torch.isfinite(scores).all(),
                "frozen finite native FP32 scores")
        original, order = old_pid(scores, pids)
        optimized, new_order = plan.aggregate(scores)
        require(order == new_order == tuple(dict.fromkeys(pids)), "PID order")
        assert_bits(f"{name}/aggregate", original, optimized)
        a, old_cache = replay(old_classifier, name, manifest, texts, scores)
        b, new_cache = replay(source.classifier, name, manifest, texts, scores)
        require(old_cache == new_cache and old_cache[:3] == (tuple(texts),) * 3
                and old_cache[-1] == 2, f"{name}/cache keys or order")
        assert_bits(f"{name}/classifier", a, b)
        assert_bits(f"{name}/softmax", a.softmax(-1), b.softmax(-1))
        streams[0][name], streams[1][name] = a, b
        evidence[name] = {"scores_sha256": tensor_sha(scores), "pid_logits_sha256": tensor_sha(a),
                          "softmax_sha256": tensor_sha(a.softmax(-1)), "cache_keys": len(old_cache[0]),
                          "cache_hits": old_cache[-1], "shape": list(scores.shape)}

    class Recorded:
        def __init__(self, logits):
            self.logits = logits

        def eval(self):
            return self

        def __call__(self, texts):
            return SimpleNamespace(logits=self.logits[:len(texts)])

    decisions = []
    for state in streams:
        seed()
        ensemble = bridge.make_ensemble(c, {k: Recorded(v) for k, v in state.items()}, "cpu")
        ensemble.init_weights()
        ensemble.next_weights()
        before = rng()
        weighted, per_model = ensemble(texts, label, flag=True)
        _, uniform = ensemble(texts, label, unique=True)
        require(rng() == before, "CPU downstream RNG consumed")
        order = sorted(range(len(texts)), key=lambda i: float(uniform[i]))
        decisions.append((weighted, per_model, uniform, ensemble.weights_zoo, order, rng()))
    for key, a, b in zip(("margin", "per_model_margin", "uniform_margin", "DPP"),
                         decisions[0][:4], decisions[1][:4]):
        assert_bits(key, a, b)
    require(decisions[0][4:] == decisions[1][4:], "candidate ordering or RNG changed")
    return {"status": "EXACT", "models": evidence, "pid_count": len(plan.order),
            "pid_order": True, "pid_members": True, "rng": True,
            "candidate_order": decisions[0][4], "candidate_top1": decisions[0][4][0],
            "margin_sha256": tensor_sha(decisions[0][0]), "weighted_and_uniform_margins_exact": True}
