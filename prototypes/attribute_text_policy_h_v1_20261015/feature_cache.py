"""训练图库特征独立缓存：来源绑定、紧凑Storage、SHA、不可变逐图发布。"""
from dataclasses import asdict
from pathlib import Path
import os
import uuid
import torch
from rewards import frozen_features
from phase2_sources import digest, sha, read_json, new_json

class GalleryFeatureCache:
    def __init__(self, root, catalog, backend_protocol):
        self.root = Path(root)
        self.catalog = catalog
        if backend_protocol.get("execution_backend") != "cpu_mock":
            raise ValueError("Phase 2 cache only permits explicitly labelled CPU mock")
        self.protocol = {"schema": "hv1-train-gallery-features-1",
                         "catalog_sha256": catalog.fingerprint,
                         "sources": catalog.source_contract, "backend": backend_protocol}
        self.protocol_hash = digest(self.protocol)
        self.root.mkdir(parents=True, exist_ok=True)
        p = self.root/"protocol.json"
        if p.exists():
            if read_json(p) != __import__("json").loads(__import__("json").dumps(self.protocol)):
                raise ValueError("gallery cache protocol/source/model mismatch")
        else:
            new_json(p, self.protocol)

    def paths(self, sample):
        self.catalog.require_train(sample)
        key = digest(asdict(sample))
        return self.root/key, self.root/(key+".receipt.json")

    def load(self, sample):
        unit, receipt = self.paths(sample)
        if not unit.is_dir() or not receipt.exists():
            raise ValueError("missing/uncommitted feature unit; do not silently regenerate")
        md = read_json(unit/"metadata.json")
        r = read_json(receipt)
        if (md["identity"] != asdict(sample) or md["protocol_hash"] != self.protocol_hash
                or r != {"protocol_hash": self.protocol_hash, "identity_hash": digest(asdict(sample)),
                         "metadata_sha256": sha(unit/"metadata.json"), "features_sha256": sha(unit/"features.pt")}
                or md["features_sha256"] != r["features_sha256"]
                or md["execution"] != {"G_forward": 1, "IRRA_image_forward": 2, "real_model_forward": 0}
                or not 0 <= md["linf"] <= 8/255):
            raise ValueError("feature identity/SHA/receipt/budget mismatch")
        data = torch.load(unit/"features.pt", map_location="cpu", weights_only=True)
        if set(data) != {"clean","adversarial"}:
            raise ValueError("feature schema mismatch")
        for key,x in data.items():
            frozen_features(x, torch.Size((1,512)))
            data[key] = x.clone()
        return data

    def prepare_mock(self, sample, pixels, ops):
        """pixels为合成CPU fixture；真实图像加载与正式资产接线留待下一阶段。"""
        if ops.protocol != self.protocol["backend"]:
            raise ValueError("producer state/source mismatch")
        unit, receipt = self.paths(sample)
        pending = self.root/(unit.name+".pending")
        if unit.exists() or receipt.exists():
            return self.load(sample)
        # exclusive目录先记录意图；失败留下pending证据，不自动重复G。
        pending.mkdir(exist_ok=False)
        new_json(pending/"intent.json", {"identity": asdict(sample), "protocol_hash": self.protocol_hash,
                                        "pixel_source": "synthetic CPU mock; not real image features"})
        data, info = ops.image_pair(pixels)
        torch.save({k:v.detach().cpu().clone() for k,v in data.items()}, pending/"features.pt")
        with (pending/"features.pt").open("rb") as f:
            os.fsync(f.fileno())
        new_json(pending/"metadata.json", {"identity": asdict(sample),
                    "protocol_hash": self.protocol_hash, "features_sha256": sha(pending/"features.pt"),
                    "execution": {k: info[k] for k in ("G_forward","IRRA_image_forward","real_model_forward")},
                    "linf": info["linf"]})
        os.rename(pending, unit)
        new_json(receipt, {"protocol_hash": self.protocol_hash, "identity_hash": digest(asdict(sample)),
                         "metadata_sha256": sha(unit/"metadata.json"), "features_sha256": sha(unit/"features.pt")})
        return self.load(sample)

    def gather(self, samples):
        # 同序返回，调用方不能用无身份Tensor自行绑定真实奖励。
        units = [self.load(s) for s in samples]
        return {k:torch.cat([u[k] for u in units]).clone() for k in ("clean","adversarial")}

    def bind_reward(self, query, samples, ops, frontend, budget_k):
        from feedback_backend import BoundReward
        return BoundReward(self.catalog, query, samples, self.gather(samples), ops, frontend, budget_k)
