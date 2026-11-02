"""Matched CPGC补充模式CPU契约：不初始化CUDA。"""
import json,copy,hashlib,inspect
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace,asdict
import pytest,torch
from tbps.baselines.cpgc_sep import modal_ablation as a
from tbps.artifacts import ArtifactError,FileRef,sha_file,digest,ArtifactStore
from tbps.contracts import DatasetManifest,GalleryItem,QueryItem,PersonID
from tbps.models.base import InputBinding,RepresentationBatch
from tbps.engines.provenance import BoundRepresentation,payload_digest

@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
 monkeypatch.setattr(torch.cuda,"_lazy_init",lambda:(_ for _ in ()).throw(AssertionError("CUDA in CPU test")))

def test_explicit_map():
 assert a.FAMILIES=={"Matched Clean":("matched_text","clean_gallery"),
  "Image-only":("matched_text","adv_gallery"),"Text-only":("adv_text","clean_gallery"),"Joint":("adv_text","adv_gallery")}
 assert a.NEW_MODES==("Image-only","Text-only")

@pytest.mark.parametrize("mode",list(a.FAMILIES))
def test_map_never_uses_raw(mode):
 assert "clean_text" not in a.FAMILIES[mode]

def test_historical_metadata_immutable(tmp_path):
 p=tmp_path/"old";p.mkdir();x=p/"data";x.write_bytes(b"old")
 s=a.snapshot_files([p]);a.unchanged(s)
 x.write_bytes(b"new")
 with pytest.raises(ArtifactError,match="metadata changed"):a.unchanged(s)

@pytest.mark.parametrize("kind",["pending","FAIL","Claim"])
def test_reject_partial_history(tmp_path,kind):
 base=tmp_path/"tbps-framework-fixture";root=base/"old";root.mkdir(parents=True)
 store=ArtifactStore(root,new_base=base);store.publish("row",{"data":b"old"},{"ok":True})
 if kind=="pending":(root/".pending-fixture").mkdir()
 elif kind=="FAIL":(root/"FAIL.json").write_text("{}")
 else:(root/"bad.Claim.json").write_text("{}")
 with pytest.raises(ArtifactError):a.check_store(store)

def test_good_committed_claim(tmp_path):
 base=tmp_path/"tbps-framework-fixture";root=base/"old"
 store=ArtifactStore(root,new_base=base);store.publish("row",{"data":b"old"},{"ok":True})
 (root/"row.Claim.json").write_text("{}")
 assert a.check_store(store)==1

@pytest.mark.parametrize("kind",["payload","manifest","receipt"])
def test_checksum_refuses_tamper(tmp_path,kind):
 base=tmp_path/"tbps-framework-fixture";store=ArtifactStore(base/"old",new_base=base)
 store.publish("row",{"data":b"old"},{"ok":True})
 name={"payload":"data","manifest":"manifest.json","receipt":"completed.json"}[kind]
 (store.root/"row"/name).write_bytes(b"bad")
 with pytest.raises((ArtifactError,ValueError,KeyError)):a.check_store(store)

def test_matched_bridge_preserves_features_and_caption_lineage(tmp_path):
 src=tmp_path/"source";(src/"protocol").mkdir(parents=True);(src/"protocol/manifest.json").write_text("{}")
 source=SimpleNamespace(root=src);ref=FileRef(str(src/"protocol/manifest.json"),sha_file(src/"protocol/manifest.json"))
 pid=PersonID("cuhk_pedes","1");g=GalleryItem("g","x.jpg","1"*64,pid,"test")
 q=QueryItem("q","RAW Caption!",0,"g",pid,"test")
 m=DatasetManifest("cuhk_pedes","test",ref,(g,),(q,));mm=replace(m,queries=(replace(q,caption="raw caption"),))
 config={"sha256":"2"*64,"resources":{"cpgc_noise":asdict(ref),"cpgc_embedding":asdict(ref)}}
 tensors={k:torch.tensor([[i+1.]],dtype=torch.float32)for i,k in enumerate(("clean_gallery","adv_gallery","matched_text","adv_text"))}
 class Bank:
  def __init__(self,k):self.k=k
  def bound(self,origin=None,originals=()):
   k=self.k;text=k.endswith("text")
   h=hashlib.sha256((mm.queries[0].caption if k=="matched_text" else "attacked").encode()).hexdigest() if text else "1"*64
   binding=InputBinding("3"*64,digest(asdict(mm if k=="matched_text" else m)),(h,),"text" if text else "image",("q" if text else "g",))
   batch=RepresentationBatch(tensors[k],"3"*64,"text" if text else "image")
   return BoundRepresentation(batch,binding,payload_digest(batch.payload),(ref,),origin,originals)
 banks={k:Bank(k)for k in tensors};out=a.matched_inputs(config,banks,m,mm,source,ref)
 assert torch.equal(out["clean_text"].batch.payload,tensors["matched_text"])
 assert torch.equal(out["adv_gallery"].batch.payload,tensors["adv_gallery"])
 assert torch.equal(out["adv_text"].batch.payload,tensors["adv_text"])
 assert all(v.inputs.dataset_manifest_sha256==digest(asdict(mm)) for v in out.values())
 assert out["adv_text"].original_sha256==(hashlib.sha256(b"raw caption").hexdigest(),)
 assert out["adv_gallery"].origin.budget.image.value==12/255

def test_csv_order_and_precision():
 groups={mode:{k:1.23456789 for k in a.METRICS}for mode in a.MODES}
 rows=a.csv_bytes("rasa",groups,True).decode().splitlines()
 assert [r.split(",")[1] for r in rows[1:]]==list(a.MODES)
 assert all(r.endswith(",1.23,1.23,1.23,1.23") for r in rows[1:])
 assert "1.23456789" in a.csv_bytes("irra",groups,False).decode()

def test_no_attack_generation_or_feature_encoding_in_runner():
 s=inspect.getsource(a.run)
 for x in ("encode_images(","encode_texts(","native_pixels(","optimizer.step(",".backward(","noise="):
  assert x not in s
 assert "native score replay not exact" in s and "before==after" in s
 assert 'modes=NEW_MODES' in s

def test_audit_is_independent_and_checks_full_scores():
 s=inspect.getsource(a.audit)
 assert "load_victim(" not in s and "ranking(score" in s
 assert "score chunks exact" in s and "(dimensions(c)[1],dimensions(c)[0])" in s and "history_unchanged" in s

def test_existing_attempt_refuses_restart(tmp_path):
 from tbps.baselines.cpgc_sep.runtime import attempt
 base=tmp_path/"tbps-framework-fixture";c={"output_base":str(base),"sha256":"1"*64};p=base/"run"
 p.mkdir(parents=True)
 with pytest.raises(ArtifactError,match="Attempt exists"):
  with attempt(c,p,"full"):pass
