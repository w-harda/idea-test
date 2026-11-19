"""原图内容绑定缓存及有界CPU三视图预取；保留每次来源检查。"""
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch
from PIL import Image
from tbps.artifacts import checked,FileRef
from tbps.augmented_attraction.views import recipe,apply_views

class ImagePipeline:
    def __init__(self,backend,*,workers=4,window=16,pin=True):
        self.b=backend;self.rows=list(backend.p["train_ids"])
        self.positions={row:i for i,row in enumerate(self.rows)}
        self.pixels=backend.native.get("irra_image_adapter").pixels_from_pil
        self.lock=threading.RLock();self.raw={};self.pending=OrderedDict()
        self.pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix="gh-image")
        self.window=window;self.version=None;self.pin=pin
    def cpu(self,row):
        e=self.b.entries[row]["identity"]
        path=Path(self.b.p["image_root"])/e["image_rel_path"]
        checked(FileRef(str(path),e["image_sha256"]))
        with self.lock:value=self.raw.get(row)
        if value is None:
            with Image.open(path) as im:value=self.pixels(im).unsqueeze(0)
            with self.lock:value=self.raw.setdefault(row,value)
        return value.clone()
    def render(self,row,version):
        with torch.no_grad():
            image=self.cpu(row)
            plan=recipe(self.b.view_settings,row,version,*image.shape[-2:])
            result=apply_views(image,plan)
            return result.pin_memory() if self.pin else result
    def views(self,row,version):
        e=self.b.entries[row]["identity"]
        checked(FileRef(str(Path(self.b.p["image_root"])/e["image_rel_path"]),e["image_sha256"]))
        if self.version!=version:
            for f in self.pending.values():f.cancel()
            self.pending.clear();self.version=version
        if row not in self.pending:self.pending[row]=self.pool.submit(self.render,row,version)
        result=self.pending.pop(row).result()
        start=self.positions[row]+1
        for nextrow in self.rows[start:start+self.window]:
            if len(self.pending)>=self.window:break
            if nextrow not in self.pending:
                self.pending[nextrow]=self.pool.submit(self.render,nextrow,version)
        return result
    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=True)
        self.raw.clear();self.pending.clear()
