"""CUHK-PEDES只读官方annotation；不导入模型、不解码图片。"""
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import json
from ..artifacts import ArtifactError, checked, digest, _object
from ..contracts import (FileRef, PersonID, GalleryItem, QueryItem, DatasetManifest,
                         PublicImageInput, PublicTextInput, AttributeSpan)

@dataclass(frozen=True, slots=True)
class AnnotationRow:
    index: int
    split: str
    pid: PersonID
    image_rel_path: str
    captions: tuple[str,...]
    @property
    def image_id(self):return f"cuhk:{self.index}:0"
    def query_id(self,index):return f"cuhk:{self.index}:{index}"

class CuhkPedes:
    """构造只读JSON；manifest显式读取所选图片字节SHA，不做PIL/模型前向。"""
    def __init__(self, annotation: FileRef, image_root):
        if type(annotation) is not FileRef:raise TypeError("pinned annotation FileRef required")
        before=checked(annotation)
        try:
            raw=json.loads(Path(annotation.path).read_text(encoding="utf-8"),object_pairs_hook=_object,
                parse_constant=lambda s:(_ for _ in ()).throw(ArtifactError("nonfinite annotation")))
        except (UnicodeError,json.JSONDecodeError) as exc:raise ArtifactError("invalid annotation JSON") from exc
        if type(raw) is not list or not raw:raise ArtifactError("annotation must be nonempty official list")
        rows=[];images=set();split_pids={s:set() for s in ("train","val","test")}
        for i,r in enumerate(raw):
            if type(r) is not dict or not {"split","id","file_path","captions"}<=set(r):raise ArtifactError("annotation fields")
            if r["split"] not in split_pids or type(r["id"]) is not int or r["id"]<1:raise ArtifactError("official split/native PID")
            path=r["file_path"]
            if type(path) is not str or not path or PurePosixPath(path).is_absolute() or any(p in ("","..",".") for p in path.split("/")) or "\\" in path or ":" in path:
                raise ArtifactError("unsafe image relative path")
            if path in images:raise ArtifactError("duplicate official image")
            images.add(path)
            captions=r["captions"]
            if type(captions) is not list or not captions or any(type(t) is not str or not t.strip() for t in captions):
                raise ArtifactError("original Caption list required")
            pid=PersonID("cuhk_pedes",str(r["id"]));split_pids[r["split"]].add(pid)
            rows.append(AnnotationRow(i,r["split"],pid,path,tuple(captions)))
        if any(split_pids[a]&split_pids[b] for a,b in (("train","val"),("train","test"),("val","test"))):
            raise ArtifactError("official split PID leakage")
        if checked(annotation)!=before:raise ArtifactError("annotation changed during read")
        self.annotation=annotation;self.image_root=Path(image_root).resolve();self.rows=tuple(rows);self._manifests={}
    def counts(self):
        return {s:{"gallery":sum(r.split==s for r in self.rows),
                   "queries":sum(len(r.captions) for r in self.rows if r.split==s),
                   "pids":len({r.pid for r in self.rows if r.split==s})} for s in ("train","val","test")}
    def row(self,query_id):
        try:
            prefix,i,j=query_id.split(":");i=int(i);j=int(j)
            if prefix!="cuhk" or i<0 or j<0:raise ValueError()
            r=self.rows[i]
            if j>=len(r.captions) or query_id!=r.query_id(j):raise ValueError()
            return r,j
        except (ValueError,IndexError,AttributeError) as exc:raise ArtifactError("unknown canonical row_id") from exc
    def image_path(self,rel):
        p=(self.image_root/rel).resolve()
        if p==self.image_root or not p.is_relative_to(self.image_root):raise ArtifactError("image escapes root")
        return p
    def _gallery(self,row,role):
        path=self.image_path(row.image_rel_path)
        from ..artifacts import sha_file
        return GalleryItem(row.image_id,row.image_rel_path,sha_file(path),row.pid,row.split,role)
    def manifest(self,split,*,expected_counts=None):
        if split not in ("train","val","test"):raise ArtifactError("explicit official split required")
        checked(self.annotation)
        rows=tuple(r for r in self.rows if r.split==split)
        if expected_counts is not None and (len(rows),sum(len(r.captions) for r in rows))!=tuple(expected_counts):
            raise ArtifactError("official Gallery/Query count mismatch")
        gallery=tuple(self._gallery(r,"gallery") for r in rows)
        queries=tuple(QueryItem(r.query_id(j),caption,j,r.image_id,r.pid,r.split)
                      for r in rows for j,caption in enumerate(r.captions))
        manifest=DatasetManifest("cuhk_pedes",split,self.annotation,gallery,queries)
        self._remember(manifest)
        return manifest
    def _remember(self,manifest):
        self._manifests[id(manifest)]=(manifest,{g.image_id:g for g in manifest.gallery},{q.query_id:q for q in manifest.queries})
    def public_image(self,manifest,image_id):
        self.require_manifest(manifest)
        try:g=self._manifests[id(manifest)][1][image_id]
        except KeyError as exc:raise ArtifactError("unknown Gallery member") from exc
        ref=FileRef(str(self.image_path(g.image_rel_path)),g.image_sha256);checked(ref)
        return PublicImageInput(ref,"rgb_0_1")
    def public_text(self,manifest,query_id,attributes=()):
        self.require_manifest(manifest)
        try:q=self._manifests[id(manifest)][2][query_id]
        except KeyError as exc:raise ArtifactError("unknown Query member") from exc
        return PublicTextInput(q.caption,attributes)
    def require_manifest(self,manifest):
        if id(manifest) in self._manifests and self._manifests[id(manifest)][0] is manifest:return
        if type(manifest) is not DatasetManifest or manifest.annotation!=self.annotation or manifest.dataset_id!="cuhk_pedes":
            raise ArtifactError("foreign manifest source")
        checked(self.annotation)
        gs={g.image_id:g for g in manifest.gallery}
        for g in manifest.gallery:
            r,j=self.row(g.image_id)
            if j!=0 or (g.image_rel_path,g.pid,g.split)!=(r.image_rel_path,r.pid,r.split):raise ArtifactError("forged Gallery member")
            checked(FileRef(str(self.image_path(g.image_rel_path)),g.image_sha256))
        for q in manifest.queries:
            r,j=self.row(q.query_id)
            if (q.caption,q.caption_index,q.pid,q.split,q.paired_image_id)!=(r.captions[j],j,r.pid,r.split,r.image_id) or r.image_id not in gs:
                raise ArtifactError("forged Query member")
        self._remember(manifest)
    def match_test_protocol(self,manifest,protocol):
        """比较现有协议完整身份/顺序，不调用旧评测或排名。"""
        self.require_manifest(manifest)
        if manifest.official_split!="test":raise ArtifactError("official Test manifest required")
        if len(manifest.gallery)!=protocol["gallery_count"] or len(manifest.queries)!=protocol["query_count"] or len(protocol["gallery"])!=len(manifest.gallery) or len(protocol["queries"])!=len(manifest.queries):
            raise ArtifactError("historical count mismatch")
        for i,(g,old) in enumerate(zip(manifest.gallery,protocol["gallery"])):
            if (i,g.image_id,g.image_rel_path,int(g.pid.value),g.image_sha256)!=(old["index"],old["row_id"],old["image"],old["pid"],old["image_sha256"]):
                raise ArtifactError("historical Gallery order/source mismatch")
        positions={g.image_id:i for i,g in enumerate(manifest.gallery)}
        for i,(q,old) in enumerate(zip(manifest.queries,protocol["queries"])):
            if (i,q.query_id,q.caption,q.caption_index,int(q.pid.value),positions[q.paired_image_id])!=(old["index"],old["row_id"],old["caption"],old["caption_index"],old["pid"],old["paired_index"]):
                raise ArtifactError("historical Query order/pairing mismatch")
        return {"gallery":len(manifest.gallery),"queries":len(manifest.queries),"ordered_identity_digest":digest({"gallery":protocol["gallery"],"queries":protocol["queries"]})}
