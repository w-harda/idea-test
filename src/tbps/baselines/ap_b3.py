"""B3历史FP32像素只读桥；不恢复Generator，不读取其他Baseline。"""
from dataclasses import asdict
from pathlib import Path
import importlib.util
import io
import torch
from PIL import Image
from tbps.artifacts import ArtifactStore,FileRef,checked,read_json,json_bytes,digest
from tbps.engines.materialized_cache import ref
from tbps.materialization.store import image_checks

SHA="c1c637e2e5346a3011f23006bb77bad60e4bd4bc28eb746d5f6bfcd881ba4ebb"
EPSILON=8/255

def need(ok,message):
    if not ok:raise ValueError(message)

def validate_native(payload,chunk,rows):
    need(type(payload)is dict and set(payload)=={"start","paths","raw_sha256","adv_pixels"},"B3 native payload fields")
    need(payload["start"]==chunk["start"] and payload["paths"]==[g.image_rel_path for g in rows]
         and payload["raw_sha256"]==[g.image_sha256 for g in rows],"B3 native member/order/raw image SHA")
    x=payload["adv_pixels"]
    need(isinstance(x,torch.Tensor) and x.shape==(len(rows),3,256,128) and x.dtype==torch.float32
         and x.device.type=="cpu" and not x.requires_grad and bool(torch.isfinite(x).all())
         and bool(((x>=0)&(x<=1)).all()),"B3 native FP32 RGB")

class HistoricalB3:
    def __init__(self,c,manifest):
        self.c,self.manifest=c,manifest
        for key in ("checkpoint","metadata","config","progress","gallery","module"):
            checked(FileRef(**c[key]))
        need(c["checkpoint"]["sha256"]==SHA,"only formal B3 frozen checkpoint")
        md=read_json(c["metadata"]["path"]);cfg=read_json(c["config"]["path"]);progress=read_json(c["progress"]["path"])
        need(md["checkpoint_sha256"]==progress["sha256"]==SHA and md["epoch"]==progress["epoch"]==60
             and progress["state"]=="complete" and md["mode"]=="formal"
             and md["training_dataset"]=="CUHK-PEDES/train" and md["surrogate"]=="official_IRRA_image_encoder"
             and md["training_uses_captions"] is False and md["training_uses_irra_text"] is False
             and md["generator_config"]==cfg["generator"] and md["epsilon"]==EPSILON,"B3 training provenance")
        g=cfg["generator"]
        need(g["irra_weight"]==g["semantic_weight"]==10 and g["epochs"]==60
             and md["train_images"]==34054 and md["train_identities"]==11003,"B3 ReID+Semantic recipe")
        self.gallery=read_json(c["gallery"]["path"]);s=self.gallery["spec"]
        need(self.gallery["complete"] is True and self.gallery["done"]==len(manifest.gallery)==3074
             and s["checkpoint_sha256"]==SHA and s["annotation_sha256"]==manifest.annotation.sha256
             and s["training_dataset"]==md["training_dataset"] and s["surrogate"]==md["surrogate"]
             and s["native_size"]==[256,128] and s["epsilon"]==EPSILON
             and s["cache_precision"]=="float32" and s["victim_information_used"]==s["query_information_used"]=="none",
             "B3 full cache origin/space")
        need(s["gallery_paths"]==[g.image_rel_path for g in manifest.gallery]
             and digest(s["gallery_paths"])==s["gallery_sha256"],"B3 official Gallery order")
        need(s["adapter_sha256"]==c["module"]["sha256"],"B3 original pixel reader source")
        for name,h in s["source_hashes"].items():
            checked(FileRef(str(Path(c["ap_source"])/"advers"/name),h))
        cursor=0;self.refs=[]
        for part in self.gallery["chunks"]:
            need(part["start"]==cursor and 0<part["count"]<=s["batch_size"]
                 and Path(part["file"]).name==part["file"],"B3 missing/overlapping/unsafe chunk")
            cursor+=part["count"]
            f=FileRef(str(Path(c["gallery"]["path"]).parent/part["file"]),part["sha256"])
            checked(f);self.refs.append(f)
        need(cursor==3074 and len(self.gallery["linf"])==3074
             and max(self.gallery["linf"])<=EPSILON,"B3 complete native budget evidence")
        # 只导入旧像素工具定义；不会构建模型或读取权重。
        spec=importlib.util.spec_from_file_location("_tbps_b3_original_pixels",c["module"]["path"])
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.native_pixels,self.transport=module.native_pixels,module.transport_delta
        self._chunk=None;self._payload=None

    def pixels(self,index,clean):
        need(0<=index<len(self.manifest.gallery),"B3 row index")
        part=next(p for p in self.gallery["chunks"] if p["start"]<=index<p["start"]+p["count"])
        source=FileRef(str(Path(self.c["gallery"]["path"]).parent/part["file"]),part["sha256"])
        checked(source)
        if self._chunk!=part["start"]:
            raw=torch.load(source.path,map_location="cpu",weights_only=True)
            rows=self.manifest.gallery[part["start"]:part["start"]+part["count"]]
            validate_native(raw,part,rows)
            self._payload=raw;self._chunk=part["start"]
        adv=self._payload["adv_pixels"][index-part["start"]:index-part["start"]+1].detach().clone()
        row=self.manifest.gallery[index];path=Path(self.c["image_root"])/row.image_rel_path
        checked(FileRef(str(path),row.image_sha256))
        with Image.open(path) as im:native=self.native_pixels(im).unsqueeze(0)
        need(float((adv.double()-native.double()).abs().max())<=EPSILON,"B3 native original-relative budget")
        # 保留原实现：bilinear(adv256-clean256)，叠加同一clean384；不重新量化或裁预算。
        value,_=self.transport(native,adv,clean)
        value=value.float().cpu().clone();stats=image_checks(clean,value,value-clean,(384,128))
        return {"original":clean.clone(),"adversarial":value,"effective_delta":value-clean},stats,source

class B3Pixels:
    """复用ArtifactStore和MaterializedStore像素契约；旧Plan仅允许G/H，故B3使用独立协议。"""
    def __init__(self,root,protocol):
        self.store=ArtifactStore(root,new_base=Path(root).parent.parent)
        self.protocol=protocol;self.protocol_ref=ref(self.store.root/"protocol/manifest.json") if (self.store.root/"protocol").exists() else None
    @staticmethod
    def key(image_id):return "image-"+digest(image_id)
    def publish_protocol(self,sources):
        if (self.store.root/"protocol").exists():
            self.store.verify("protocol");need(read_json(self.store.root/"protocol/protocol.json")==self.protocol,"B3 pixel protocol changed")
        else:self.store.publish("protocol",{"protocol.json":json_bytes(self.protocol)},{"kind":"historical_B3_pixel_transport"},sources=sources)
        self.protocol_ref=ref(self.store.root/"protocol/manifest.json")
    def publish(self,row,arrays,source):
        stats=image_checks(arrays["original"],arrays["adversarial"],arrays["effective_delta"],(384,128))
        stream=io.BytesIO();torch.save({k:v.detach().clone() for k,v in arrays.items()},stream)
        self.store.publish(self.key(row.image_id),{"pixels.pt":stream.getvalue(),
            "metadata.json":json_bytes({"row":asdict(row),"pixels":stats,"B3_new_model_loads":0,"B3_new_forwards":0,
                "historical_native_chunk":asdict(source),"target_feedback":False})},
            {"kind":"B3_pixels","protocol":self.protocol_ref.sha256,"image_id":row.image_id},
            sources=(self.protocol_ref,source))
        return self.load(row)
    def load(self,row):
        key=self.key(row.image_id);self.store.verify(key,expected_provenance={"kind":"B3_pixels",
            "protocol":self.protocol_ref.sha256,"image_id":row.image_id})
        raw=torch.load(self.store.root/key/"pixels.pt",map_location="cpu",weights_only=True)
        need(type(raw)is dict and set(raw)=={"original","adversarial","effective_delta"},"B3 compact pixel fields")
        x={k:v.detach().clone() for k,v in raw.items()};del raw
        md=read_json(self.store.root/key/"metadata.json")
        need(md["row"]==asdict(row) and md["pixels"]==image_checks(x["original"],x["adversarial"],x["effective_delta"],(384,128))
             and md["B3_new_model_loads"]==md["B3_new_forwards"]==0 and md["target_feedback"] is False,"B3 pixel identity/content")
        checked(FileRef(**md["historical_native_chunk"]))
        return x,md
