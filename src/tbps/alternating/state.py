"""完整双模型/双Adam及独立阶段RNG；仅操作调用者给定的新状态。"""
import copy
import io
import random
from pathlib import Path
from tbps.artifacts import ArtifactStore, json_bytes, read_json, sha_file, digest
from .contracts import require

def clone_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: clone_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(clone_tree(v) for v in value)
    return copy.deepcopy(value)

def tree_sha(value):
    # 复用既有张量/Adam/RNG内容摘要，不以torch.save字节非确定性作为状态同一性。
    from tbps.ablations.gv2_epoch_state import tree_hash
    return tree_hash(value)

def capture_rng(sampler):
    import numpy as np
    import torch
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(), "sampler": sampler.get_state().clone(),
        # is_initialized不触发CUDA；CPU测试禁止初始化真实CUDA。
        "cuda": clone_tree(torch.cuda.get_rng_state_all()) if torch.cuda.is_initialized() else None,
    }

def restore_rng(value, sampler, *, allow_cuda=False):
    import numpy as np
    import torch
    required = {"python", "numpy", "torch_cpu", "sampler", "cuda"}
    require(set(value) == required, "incomplete RNG")
    require(allow_cuda or value["cuda"] is None, "GPU RNG restoration requires later authorization")
    # 缺CUDA/CPU不相容时先拒绝，不能先修改部分CPU随机状态。
    if value["cuda"] is not None:
        require(torch.cuda.is_initialized(), "CUDA must already be explicitly initialized")
        require(len(value["cuda"]) == torch.cuda.device_count(), "CUDA RNG device count")
    random.setstate(value["python"]); np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"]); sampler.set_state(value["sampler"])
    if value["cuda"] is not None:
        torch.cuda.set_rng_state_all(value["cuda"])
    require(tree_sha(capture_rng(sampler)) == tree_sha(value), "RNG restore mismatch")

def snapshot(g, h, g_adam, h_adam, *, g_rng, h_rng, cursor, parents):
    require(set(cursor) == {"round", "g_updates", "h_visits", "h_updates"}, "complete joint cursor")
    require(set(parents) == {"G", "H", "protocol"}, "full parent sources")
    return clone_tree({"G": g.state_dict(), "H": h.state_dict(),
        "G_Adam": g_adam.state_dict(), "H_Adam": h_adam.state_dict(),
        "G_RNG": g_rng, "H_RNG": h_rng, "cursor": cursor, "parents": parents})

def restore(value, g, h, g_adam, h_adam, *, expected_parents):
    require(value["parents"] == expected_parents, "parent/protocol conflict")
    require(set(value) == {"G", "H", "G_Adam", "H_Adam", "G_RNG", "H_RNG",
                          "cursor", "parents"}, "incomplete joint state")
    g.load_state_dict(value["G"], strict=True); h.load_state_dict(value["H"], strict=True)
    g_adam.load_state_dict(clone_tree(value["G_Adam"]))
    h_adam.load_state_dict(clone_tree(value["H_Adam"]))
    require(all(tree_sha(x) == tree_sha(y) for x, y in [
        (g.state_dict(), value["G"]), (h.state_dict(), value["H"]),
        (g_adam.state_dict(), value["G_Adam"]), (h_adam.state_dict(), value["H_Adam"])]),
        "G/H/Adam restore mismatch")
    # 各阶段开始时再恢复对应RNG；不能把两个历史全局CUDA RNG同时当成一个。
    return clone_tree(value["G_RNG"]), clone_tree(value["H_RNG"]), copy.deepcopy(value["cursor"])

class RoundStore:
    """Claim永久保留；未提交round绝不自动重放。恢复仅显式完整边界+后续无Claim。"""
    def __init__(self, root, protocol_sha256, *, new_base=None):
        self.root = Path(root)
        self.store = ArtifactStore(self.root, new_base=new_base or self.root.parent)
        self.protocol = protocol_sha256

    def begin(self, arm, index, parent_state_sha):
        require(arm in ("A1", "B") and index in (1, 2), "fixed two rounds")
        require(not self.store.pending(), "uncommitted storage transaction")
        return self.store.publish(f"{arm}-r{index}-claim", {
            "claim.json": json_bytes({"protocol": self.protocol, "arm": arm, "round": index,
                                      "parent_state": parent_state_sha})},
            {"protocol": self.protocol, "kind": "round-claim"})

    def commit(self, arm, index, value, ledger, *, fault=None):
        self.store.verify(f"{arm}-r{index}-claim", expected_provenance={
            "protocol": self.protocol, "kind": "round-claim"})
        require(value["cursor"]["round"] == index and ledger["arm"] == arm
                and ledger["round"] == index and ledger["all_stages_verified"], "incomplete round")
        import torch
        payload = io.BytesIO(); torch.save(clone_tree(value), payload)
        return self.store.publish(f"{arm}-r{index}", {
            "state.pt": payload.getvalue(), "ledger.json": json_bytes(ledger)},
            {"protocol": self.protocol, "state_sha256": tree_sha(value)}, fault=fault)

    def load(self, arm, index, *, expected_parents):
        require(not self.store.pending(), "pending retained; no automatic recovery")
        # 任意后一轮Claim都可能已发生更新，不能静默回退或重播。
        require(not (self.root/f"{arm}-r{index+1}-claim").exists(), "uncommitted/later Claim blocks restore")
        manifest = self.store.verify(f"{arm}-r{index}")
        require(manifest["provenance"]["protocol"] == self.protocol, "protocol mismatch")
        import torch
        value = torch.load(self.root/f"{arm}-r{index}"/"state.pt",
                           map_location="cpu", weights_only=False)
        require(tree_sha(value) == manifest["provenance"]["state_sha256"]
                and value["parents"] == expected_parents, "joint checkpoint provenance")
        return clone_tree(value)
