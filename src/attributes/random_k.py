"""固定 Random-K 调度：共享 Stage 04 候选池和 k_star，不按分数排序。"""
from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping

from .attack_scheduler import plan_attack
from .attribute_scoring import SUPPORTED_SLOTS


def random_k_entry(record: Mapping, seed: int = 42) -> dict:
    """每个 row_id 独立抽取均匀有序无放回样本。"""
    plan_attack(record)
    row_id = record["row_id"]
    shared = record["shared_attributes"]
    scores = record["scores"]
    pool = [slot for slot in SUPPORTED_SLOTS if slot in shared]
    if set(pool) != set(shared) or set(scores) != set(shared):
        raise ValueError("Stage 04 candidate pool and scores differ")
    k_star = record["k_star"]
    if k_star > len(pool):
        raise ValueError("k_star exceeds the valid candidate pool")
    for slot in pool:
        target = record["provenance"].get(slot)
        if (not isinstance(target, Mapping)
                or target.get("canonical") != shared[slot]
                or not target.get("mentions")):
            raise ValueError(f"candidate lacks provenance: {slot}")
    digest = hashlib.sha256(f"{seed}:{row_id}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest, "big"))
    order = rng.sample(pool, k_star)
    selected_set = set(order)
    selected = [slot for slot in pool if slot in selected_set]
    return {
        "row_id": row_id,
        "seed": seed,
        "k_star": k_star,
        "candidate_slots": pool,
        "random_selected_slots": selected,
        "random_attack_order": order,
        "scores_by_slot": {slot: scores[slot] for slot in pool},
        "topk_selected_slots": list(record["selected_slots"]),
    }


def apply_random_k(record: Mapping, entry: Mapping, seed: int = 42) -> dict:
    """只替换 Stage 05 读取的 selected_slots/attributes，不改原记录。"""
    expected = random_k_entry(record, seed)
    if dict(entry) != expected:
        raise ValueError(f"Random-K manifest differs: {record['row_id']}")
    order = expected["random_attack_order"]
    scheduled = dict(record)
    scheduled["selected_slots"] = list(order)
    scheduled["selected_attributes"] = {
        slot: record["shared_attributes"][slot] for slot in order
    }
    plan_attack(scheduled)
    return scheduled
