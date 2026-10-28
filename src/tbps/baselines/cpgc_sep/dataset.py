"""三个TBPS官方Test的只读接线；不修改原标注或旧CUHK读取器。"""
from pathlib import Path,PurePosixPath
from dataclasses import dataclass,asdict
import json
from tbps.artifacts import ArtifactError,sha_file,checked,digest
from tbps.contracts import FileRef,PersonID,GalleryItem,QueryItem,DatasetManifest
SPECS={"cuhk":("CUHK-PEDES","reid_raw.json","file_path",3074,6156,1000),
 "icfg":("ICFG-PEDES","ICFG-PEDES.json","file_path",19848,19848,1000),
 "rstp":("RSTPReid","data_captions.json","img_path",1000,2000,200)}
def require(ok,message):
 if not ok:raise ArtifactError(message)
def read_test(config,*,limit=None,caption_transform=None):
 key=config["id"];folder,ann,path_key,ng,nq,npid=SPECS[key]
 root=Path(config["root"]).resolve();ref=FileRef(**config["annotation"]);checked(ref)
 rows=json.loads(Path(ref.path).read_text());selected=[];splits={}
 for i,r in enumerate(rows):
  require(r["split"] in ("train","val","test"),"official split")
  splits.setdefault(r["split"],set()).add(r["id"])
  if r["split"]=="test":selected.append((i,r))
 for a in splits:
  for b in splits:
   if a!=b:require(not(splits[a]&splits[b]),"official split PID overlap")
 require(len(selected)==ng and sum(len(r["captions"]) for _,r in selected)==nq,"official Test size")
 require(len(splits["test"])==npid,"official Test PID size")
 require(len({r[path_key] for _,r in selected})==ng,"duplicate Test image path")
 if limit is not None:selected=selected[:limit]
 gs=[];qs=[];namespace=key+"_pedes" if key!="rstp" else "rstp_reid"
 for index,row in selected:
  rel=row[path_key];pp=PurePosixPath(rel)
  require(not pp.is_absolute() and ".." not in pp.parts and "\\" not in rel,"unsafe image path")
  image=(root/"imgs"/rel).resolve();require(image.is_relative_to(root/"imgs") and image.is_file(),"image missing/escape")
  require(image.stat().st_size>0,"empty image")
  image_id=f"{key}:{index}:0";pid=PersonID(namespace,str(row["id"]))
  gs.append(GalleryItem(image_id,rel,sha_file(image),pid,"test","gallery"))
  for j,raw in enumerate(row["captions"]):
   require(isinstance(raw,str) and raw.strip(),"Caption must be original string")
   text=caption_transform(raw) if caption_transform else raw
   require(text.strip(),"empty matched Caption")
   qs.append(QueryItem(f"{key}:{index}:{j}",text,j,image_id,pid,"test"))
 return DatasetManifest(namespace,"test",ref,tuple(gs),tuple(qs))
def scoped_manifest(manifest,scope):
 if scope=="full":return manifest
 require(scope=="smoke","scope")
 # 固定前128图库及其前2个Query，最小覆盖原生RaSa Top128融合路径。
 gs=manifest.gallery[:128];ids={g.image_id for g in gs}
 qs=tuple(q for q in manifest.queries if q.paired_image_id in ids)[:2]
 return DatasetManifest(manifest.dataset_id,"test",manifest.annotation,gs,qs)
