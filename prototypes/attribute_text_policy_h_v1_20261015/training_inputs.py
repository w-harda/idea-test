"""G-v2原训练身份来源只读绑定；不加载condition Tensor，不向H传递配对评分。"""
from dataclasses import dataclass, asdict
from pathlib import Path
from phase2_sources import (G2, RAW, SPLIT, IMAGES, MANIFEST_SHA, PROTOCOL_SHA,
                            RAW_SHA, SPLIT_SHA, checked, read_json, digest, within, sha)

IDENTITY = ("row_id", "image_rel_path", "caption_index", "split", "role", "pid", "caption")

@dataclass(frozen=True)
class SampleIdentity:
    row_id: str
    image_rel_path: str
    caption_index: int
    split: str
    role: str
    pid: int
    caption: str
    image_sha256: str
    metadata_sha256: str
    condition_sha256: str

def validate_identities(samples, raw, split_records, train_indices, validation_indices, counts):
    """纯CPU验证核心，测试可用独立合成来源；正式入口另有固定SHA门禁。"""
    membership = {}
    for role, indices in (("train", train_indices), ("validation", validation_indices)):
        for i in indices:
            r = split_records[i]
            key = r["row_id"]
            if key in membership:
                raise ValueError("duplicate split identity")
            membership[key] = (role, r["pid"], r["caption"])
    seen_rows, seen_images = set(), set()
    roles, pids = {"train": 0, "validation": 0}, {"train": set(), "validation": set()}
    for s in samples:
        if s.split != "train" or s.role not in roles:
            raise ValueError("official Test/nontraining source forbidden")
        if type(s.pid) is not int or type(s.caption_index) is not int:
            raise ValueError("PID/caption index type mismatch")
        parts = s.row_id.split(":")
        if len(parts) != 3 or parts[0] != "cuhk" or int(parts[2]) != s.caption_index:
            raise ValueError("row_id/caption index mismatch")
        row = raw[int(parts[1])]
        if (row["split"] != "train" or row["id"] != s.pid
                or row["file_path"] != s.image_rel_path
                or row["captions"][s.caption_index] != s.caption
                or membership.get(s.row_id) != (s.role, s.pid, s.caption)):
            raise ValueError("annotation/split/Caption/PID identity mismatch")
        if s.row_id in seen_rows or s.image_rel_path in seen_images:
            raise ValueError("duplicate sample/image")
        if not s.caption.strip() or any(len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
                for v in (s.image_sha256, s.metadata_sha256, s.condition_sha256)):
            raise ValueError("missing identity SHA")
        seen_rows.add(s.row_id)
        seen_images.add(s.image_rel_path)
        roles[s.role] += 1
        pids[s.role].add(s.pid)
    if roles != counts or pids["train"] & pids["validation"]:
        raise ValueError("count mismatch or train/validation PID leakage")
    return {"counts": roles, "pid_counts": {k: len(v) for k, v in pids.items()},
            "pid_intersection": 0}

class TrainingCatalog:
    def __init__(self, samples, source_contract, image_root, audit):
        self.samples = tuple(samples)
        self.train = tuple(s for s in samples if s.role == "train")
        self.validation = tuple(s for s in samples if s.role == "validation")
        self._by_id = {s.row_id: s for s in samples}
        if len(self._by_id) != len(samples):
            raise ValueError("duplicate catalog identity")
        self.source_contract = source_contract
        self._source_digest = digest(source_contract)
        self.fingerprint = digest({"contract": source_contract, "samples": [asdict(s) for s in samples]})
        self.image_root, self.audit = Path(image_root), audit

    def require_train(self, sample):
        if digest(self.source_contract) != self._source_digest:
            raise ValueError("catalog source contract mutated")
        if (not isinstance(sample, SampleIdentity) or sample.split != "train"
                or sample.role != "train" or self._by_id.get(sample.row_id) != sample):
            raise ValueError("verified original training sample required; validation/Test forbidden")
        return sample

    def get_train(self, row_id):
        return self.require_train(self._by_id.get(row_id))

    def verify_image(self, sample):
        self.require_train(sample)
        return checked(within(self.image_root, self.image_root/sample.image_rel_path), sample.image_sha256)

def load_training_catalog():
    root = G2 / "outputs/conditions_full"
    manifest = read_json(checked(root/"manifest.json", MANIFEST_SHA))
    protocol = read_json(checked(root/"protocol.json", PROTOCOL_SHA))
    completed = read_json(root/"completed.json")
    if (not completed.get("method_audit_passed") or completed["manifest_sha256"] != MANIFEST_SHA
            or manifest["schema"] != "gv2-training-cache-2" or manifest["scope"] != "full-2176"
            or manifest["counts"] != {"train": 2048, "validation": 128}
            or manifest["protocol_sha256"] != PROTOCOL_SHA
            or Path(manifest["protocol_path"]).resolve() != (root/"protocol.json").resolve()
            or protocol["schema"] != "gv2-full-prepare-protocol-2"
            or protocol["target_counts"] != manifest["counts"]):
        raise ValueError("original completed G-v2 protocol required")
    for name, expected in protocol["producer_sha256"].items():
        checked(G2/name, expected)
    checked(RAW, RAW_SHA)
    split = read_json(checked(SPLIT, SPLIT_SHA))
    # 精确复用原cache_io.digest的紧凑JSON协议。
    protocol_hash = digest(protocol)
    planned = {x["row_id"]: x for x in protocol["samples"]}
    samples = []
    for e in manifest["entries"]:
        md = read_json(checked(within(root, e["metadata_path"]), e["metadata_sha256"]))
        checked(within(root, e["tensor_path"]), e["tensor_sha256"])
        if (md["protocol_hash"] != protocol_hash or md["tensor_sha256"] != e["tensor_sha256"]
                or e["row_id"] not in planned
                or any(md[k] != e[k] or planned[e["row_id"]][k] != e[k] for k in IDENTITY)
                or md["K"] != e["K"] or planned[e["row_id"]]["K"] != e["K"]):
            raise ValueError("cache metadata/protocol identity mismatch")
        samples.append(SampleIdentity(**{k: e[k] for k in IDENTITY},
                       image_sha256=md["source_image_sha256"],
                       metadata_sha256=e["metadata_sha256"], condition_sha256=e["tensor_sha256"]))
    audit = validate_identities(samples, read_json(RAW), split["records"],
                                split["train_indices"], split["validation_indices"], manifest["counts"])
    contract = {"manifest_sha256": MANIFEST_SHA, "protocol_sha256": PROTOCOL_SHA,
                "raw_annotation_sha256": RAW_SHA, "internal_split_sha256": SPLIT_SHA,
                "producer_sha256": protocol["producer_sha256"], "cache_units_sha_verified": len(samples),
                "image_sha_policy": "verify lazily before image use; inspection verifies only selected 16"}
    return TrainingCatalog(samples, contract, IMAGES, audit)
