"""虚拟更新克隆模型/完整Adam，退出时恢复全部RNG；不回写正式状态。"""
from contextlib import contextmanager
import copy
import torch
from tbps.alternating.state import capture_rng, restore_rng, tree_sha, clone_tree
from .settings import require

def clone_trainable(module, optimizer):
    # tokenizer/probe是无状态只读服务，不重复复制BPE大字典。
    memo = {id(module.probe): module.probe} if hasattr(module, "probe") else {}
    clone = copy.deepcopy(module, memo).requires_grad_(True)
    for m in clone.modules():
        m._forward_hooks.clear(); m._forward_pre_hooks.clear(); m._backward_hooks.clear()
    original = list(module.parameters()); copied = list(clone.parameters())
    mapping = {id(a): b for a, b in zip(original, copied)}
    groups = [{**{k: copy.deepcopy(v) for k, v in group.items() if k != "params"},
               "params": [mapping[id(p)] for p in group["params"]]}
              for group in optimizer.param_groups]
    require(all(type(optimizer).__name__ == "Adam" for _ in [0]), "virtual full Adam required")
    opt = type(optimizer)(groups, **optimizer.defaults)
    opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    require(tree_sha(opt.state_dict()) == tree_sha(optimizer.state_dict()), "virtual Adam exact restore")
    require(not {id(p) for p in copied} & {id(p) for p in original}, "disjoint virtual parameters")
    return clone, opt

@contextmanager
def sandbox(modules, optimizers, sampler):
    require(len(modules) == len(optimizers), "complete model/optimizer pairs")
    states = [(tree_sha(m.state_dict()), tree_sha(o.state_dict()),
               [p.grad.detach().clone() if p.grad is not None else None for p in m.parameters()],
               [(x.training, [p.requires_grad for p in x.parameters(recurse=False)]) for x in m.modules()])
              for m, o in zip(modules, optimizers)]
    rng = capture_rng(sampler); before = tree_sha(rng)
    report = {"rng_before": before, "formal_before": [list(x[:2]) for x in states]}
    try:
        yield report
    finally:
        restore_rng(rng, sampler, allow_cuda=torch.cuda.is_initialized())
        for (m, o), (ms, os, grads, modes) in zip(zip(modules, optimizers), states):
            require(tree_sha(m.state_dict()) == ms and tree_sha(o.state_dict()) == os,
                    "virtual update polluted live model/Adam")
            for p, g in zip(m.parameters(), grads):
                require((p.grad is None and g is None)
                        or p.grad is not None and g is not None and torch.equal(p.grad, g),
                        "virtual update polluted live gradients")
            require([(x.training, [p.requires_grad for p in x.parameters(recurse=False)])
                     for x in m.modules()] == modes, "virtual update polluted live mode/freeze")
        report.update(rng_after=tree_sha(capture_rng(sampler)), formal_after=[list(x[:2]) for x in states],
                      isolated=True)
        require(report["rng_after"] == before, "virtual RNG isolation")
