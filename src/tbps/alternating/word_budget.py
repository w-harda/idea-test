"""WordBudget1适配：原网络/SCST原样调用，仅约束候选支持集与输出。"""
from contextlib import contextmanager
import re
import threading
from .contracts import require

_LOCK = threading.RLock()

def word_spans(text):
    # 原始空白分隔位置；不规范化文本。连字符、标点、Unicode仍属于同一原词。
    return tuple((m.start(), m.end()) for m in re.finditer(r"\S+", text))

def check_words(original, adversarial, *, maximum=1):
    require(type(maximum) is int and maximum == 1, "WordBudget1 required")
    require(len(original) == len(adversarial), "word budget forbids insertion/deletion")
    spans = word_spans(original)
    changed = set()
    for p, (a, b) in enumerate(zip(original, adversarial)):
        if a == b:
            continue
        require(not a.isspace() and not b.isspace(), "word budget forbids whitespace change")
        indices = [i for i, (s, e) in enumerate(spans) if s <= p < e]
        require(len(indices) == 1, "edit lacks original word position")
        changed.add(indices[0])
    require(len(changed) <= maximum, "word budget exceeded")
    return {"maximum_changed_words": 1, "changed_word_indices": sorted(changed),
            "changed_words": len(changed), "word_unit": "original whitespace-separated positions",
            "maximum_characters_per_word": 1}

@contextmanager
def candidate_guard(module):
    """隔离legacy会话中的有界调用；不改历史源码、网络、Natural-K或动作log-prob。"""
    with _LOCK:
        native = module.candidates
        def constrained(example, current, attribute_index, used_words, probe):
            report = check_words(example.caption, current)
            # 保留原每属性/共享词最多1字符；一次实际编辑后仅STOP/NOOP可用。
            if report["changed_words"]:
                return {}
            supports = native(example, current, attribute_index, used_words, probe)
            for pos, choices in supports.items():
                for replacement in choices:
                    check_words(example.caption, current[:pos] + replacement + current[pos+1:])
            return supports
        module.candidates = constrained
        try:
            yield
        finally:
            module.candidates = native

def install(policy, native_module):
    """原对象上安装受约束rollout，保留Parameter对象及已恢复Adam动量。"""
    require(not getattr(policy, "word_budget1", False), "WordBudget1 already installed")
    base = type(policy)
    class WordBudgetPolicy(base):
        def rollout(self, example, mode="sample", generator=None):
            with candidate_guard(native_module):
                output = super().rollout(example, mode, generator)
            report = check_words(example.caption, output.adversarial)
            require(len(output.edits) <= 1, "original one-character-per-word rule")
            output.tokenizer_audit = {**output.tokenizer_audit, "word_budget": report}
            return output
    policy.__class__ = WordBudgetPolicy
    policy.word_budget1 = True
    return policy
