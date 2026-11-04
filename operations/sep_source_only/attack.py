"""只读 Source 的伪标签接线，原官方 SEP 搜索体保持冻结。"""
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
import ast
import math
import pickle
import signal
import sys
import time

from tbps.baselines.cpgc_sep.dataset import require
from tbps.baselines.cpgc_sep.official import sep_ensemble_class, sep_functions
from tbps.baselines.cpgc_sep.sep_bridge import TransportTokenizer, legal_semantic_flags, prepare_caption


def s1_ensemble(config, source, device):
    class SourceEnsemble(sep_ensemble_class(config)):
        def _parse_models(self):
            require(tuple(self.models_name) == ("irra",), "S1 source only")
            self.models = OrderedDict(irra=source)
            self.tokenizers = OrderedDict(irra=TransportTokenizer())
    return SourceEnsemble(OrderedDict(irra="irra"), device, .9)


def resources(c):
    with Path(c["resources"]["sep_vectors"]["path"]).open() as f:
        words = [line.split(" ", 1)[0] for line in f]
    with Path(c["resources"]["sep_neighbors"]["path"]).open("rb") as f:
        neighbors = pickle.load(f)
    require(len(neighbors) == len(words) == len(set(words)) == 65713
            and all(len(row) == 50 for row in neighbors), "official synonym vocabulary and top50")
    return {w: i for i, w in enumerate(words)}, dict(enumerate(words)), neighbors


def trace_markers(config):
    official = Path(config["official"]["sep"]["root"]) / "SEP_Attack.py"
    tree = ast.parse(official.read_text())
    tau = next(n.lineno for n in ast.walk(tree) if isinstance(n, ast.For) and isinstance(n.iter, ast.Subscript)
               and isinstance(n.iter.value, ast.Name) and n.iter.value.id == "log_stack_sub")
    opt = next(n.lineno for n in ast.walk(tree) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
               and isinstance(n.value.func, ast.Name) and n.value.func.id == "optimize")
    return tau, opt


@contextmanager
def deadline(seconds):
    def expired(*_):
        raise TimeoutError("SEP diagnostic per-Query deadline; attempt stops")
    previous = signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def verify_edits(clean, adv, vocabulary, pos):
    mapping, inverse, neighbors = vocabulary
    info = legal_semantic_flags(clean, adv)
    words = clean.split()
    budget = max(math.floor(len(words) * .15), 1)
    for edit in info["edits"]:
        word = edit["original"]
        require(word in mapping and pos[edit["position"]] in ("ADJ", "ADV", "VERB", "NOUN")
                and len(word) > 2, "illegal SEP edit position")
        require(edit["replacement"] in {inverse[i] for _, i in neighbors[mapping[word]]},
                "replacement outside original top50")
    require(info["word_change_count"] <= budget, "official SEP final word budget")
    return {**info, "word_budget": budget}


def attack_caption(c, raw, classifier, device, vocabulary, report):
    import torch
    clean = prepare_caption(c, raw)
    before_calls, before_hits = classifier.calls, classifier.hits
    requests, oracle_calls, timings, candidates = [], [], [], []
    phase = "PSEUDO_PID"
    start = time.monotonic()

    def observed(_module, args, kwargs):
        texts = tuple(args[0] if args else kwargs["texts"])
        requests.append({"phase": phase, "texts": texts})
        if len(requests) % 16 == 0:
            report("CANDIDATE_PROGRESS", phase=phase, requests=len(requests), calls=classifier.calls-before_calls,
                   hits=classifier.hits-before_hits, elapsed=time.monotonic()-start)

    def phase_event(name, **fields):
        nonlocal phase
        phase = name
        event = {"stage": name, "elapsed": time.monotonic() - start, **fields}
        timings.append(event)
        report("QUERY_STAGE", **event)

    hook = classifier.register_forward_pre_hook(observed, with_kwargs=True)
    old_trace = sys.gettrace()
    try:
        logits = classifier([clean]).logits.detach().cpu()
        require(logits.shape == (1, len(classifier.pid_plan.order)) and torch.isfinite(logits).all(),
                "finite complete Source PID scores")
        label = int(logits[0].argmax())
        phase_event("SOURCE_CLEAN_GATE")

        def oracle(texts, orig_label, _cache):
            require(orig_label == label, "fixed Source pseudo PID")
            result = classifier(texts=texts).logits.argmax(dim=-1) != label
            oracle_calls.append({"texts": list(texts), "source_top1_changed": result.cpu().tolist()})
            return result.squeeze()

        attack = sep_functions(c)["attack"]
        tau, opt = trace_markers(c)
        ensemble = s1_ensemble(c, classifier, device)

        def trace(frame, event, _arg):
            if frame.f_code is not attack.__code__:
                return None
            if event == "line" and frame.f_lineno == opt:
                phase_event("OFFICIAL_OPTIMIZE", iteration=frame.f_locals["t"])
            if event == "line" and frame.f_lineno == tau:
                if frame.f_locals["ms_idx"] == 0:
                    phase_event("OFFICIAL_TAU")
                if frame.f_locals["ms_idx"] == 1:
                    candidates[:] = [{"text": " ".join(words), "tau": float(score)}
                                     for words, score in frame.f_locals["log_stack_sub"]]
            return trace

        sys.settrace(trace)
        result = attack(1000000, clean.split(), label, oracle, *vocabulary, ensemble,
                        device, .9, 0, "source-self-consistency", ".", max_change_rate=.2)
        require(result[0] is not None and oracle_calls[0]["source_top1_changed"] == [False],
                "unexpected Source pseudo PID Clean gate")
        require(ensemble.weights_zoo.cpu().tolist() == [[1.]], "official S1 DPP result")
    finally:
        sys.settrace(old_trace)
        hook.remove()

    pos = attack.__globals__["criteria"].get_pos(clean.split())
    require(all(math.isfinite(x["tau"]) for x in candidates), "finite SEP tau scores")
    chosen = candidates[0]["text"] if candidates else clean
    edits = verify_edits(clean, chosen, vocabulary, pos)
    noop = edits["word_change_count"] == 0
    calls, hits = classifier.calls - before_calls, classifier.hits - before_hits
    return {"original": raw, "matched_clean": clean, "selected": chosen, "adversarial": raw if noop else chosen,
            "noop": noop, "noop_reason": "official tau has no distinct candidate" if noop else None,
            "pseudo_pid_index": label, "pseudo_pid": classifier.pid_plan.order[label].value,
            "candidates": candidates, "requests": requests, "source_oracle": oracle_calls,
            "DPP": {"N": 1, "E": 1, "draws": 100, "weights": [[1.]]},
            "original_source_statistics": result, "word_edits": edits, "pos": pos,
            "calls": calls, "cache_hits": hits, "text_forwards": calls - hits,
            "cache_hit_rate": hits/calls, "timings": timings, "elapsed_seconds": time.monotonic()-start}, logits
