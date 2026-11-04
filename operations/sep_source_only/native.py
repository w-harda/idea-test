"""沿用旧 SEP 原生评分器与冻结模型加载路径。"""
from contextlib import contextmanager
from pathlib import Path

from tbps.baselines.cpgc_sep.cache import NativeScorer
from tbps.baselines.cpgc_sep.dataset import require
from tbps.baselines.cpgc_sep.sep_source import EphemeralQuery
from tbps.baselines.cpgc_sep.victims import (encode_images, encode_texts, frozen, global_features,
                                            load_victim, native_pixels, snapshot)
from tbps.models.base import RepresentationBatch
from .common import image_path, sha_file


@contextmanager
def native(c, name, ds, aptm_config=None):
    if name == "aptm":
        from tbps.models.aptm import APTMSource, load_aptm
        source = APTMSource(aptm_config)
        with load_aptm(source, smoke=True) as adapter:
            yield adapter, adapter.backend, source
    else:
        with load_victim(c, name, ds, smoke=True) as loaded:
            yield loaded


class Gallery(EphemeralQuery):
    family = "gallery"

    def __init__(self, adapter, name, root, items, report):
        import torch
        from PIL import Image
        from torchvision.transforms.functional import to_tensor
        parts = []
        for index, item in enumerate(items, 1):
            path = image_path(root, item.image_rel_path)
            require(sha_file(path) == item.image_sha256, "frozen gallery image SHA")
            if name == "aptm":
                with Image.open(path) as im:
                    payload = adapter.encode_images(to_tensor(im.convert("RGB")).unsqueeze(0))
            else:
                payload = encode_images(adapter, native_pixels(path, name), name)
            parts.append(payload)
            if index % 128 == 0 or index == len(items):
                report("GALLERY_PROGRESS", model=name, encoded=index, gallery=len(items))
        if isinstance(parts[0], dict):
            payload = {k: torch.cat([v[k] for v in parts]) for k in parts[0]}
        else:
            payload = torch.cat(parts)
        super().__init__(payload)
        self.ids = tuple(item.image_id for item in items)


def score(adapter, name, bank, texts):
    import torch
    if name == "aptm":
        parts = [adapter.encode_texts([s]) for s in texts]
        payload = {k: torch.cat([x[k] for x in parts]) for k in parts[0]}
        q = RepresentationBatch(payload, adapter.space.id, "text")
        g = RepresentationBatch(bank.payload, adapter.space.id, "image")
        before = adapter.counts["fusion_pairs"]
        scores = adapter.score(q, g).payload
        return scores, adapter.counts["fusion_pairs"] - before
    rows = []
    pairs = 0
    for text in texts:
        qb = EphemeralQuery(encode_texts(adapter, [text], name))
        scorer = NativeScorer(adapter, name, {"query": qb, "gallery": bank})
        q = scorer.slice_query(RepresentationBatch(qb.globals, adapter.space.id, "text"), 0, 1)
        g = RepresentationBatch(bank.globals, adapter.space.id, "image")
        rows.append(scorer.score(q, g).payload)
        pairs += scorer.native_pairs
    return torch.cat(rows), pairs


def state_sha(model):
    from tbps.artifacts import digest
    frozen(model)
    return digest(snapshot(model))
