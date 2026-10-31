"""只读实测旧ICFG Token块I/O、SHA与有界内存缓存，不调用模型。"""
from pathlib import Path
import json,time,sys,importlib.util,resource
from types import SimpleNamespace
import torch
from tbps.artifacts import ArtifactStore,sha_file
from tbps.baselines.cpgc_sep.verified_store import VerifiedArtifactStore,DigestSession
from tbps.baselines.cpgc_sep.cache import FeatureBank,BlockCache
from tbps.baselines.cpgc_sep.dataset import read_test
from tbps.models.base import ModelSpace
ROOT=Path("/home/lzf/ldx/outputs/tbps-framework-cpgc-sep-throughput-validation-v3")
OUT=ROOT/"cache_profile";OUT.mkdir(exist_ok=False)
old=Path("/home/lzf/ldx/projects/idea-test-tbps-cpgc-sep-uap-detach-20261029")
cfg=old/"configs/framework/cpgc_sep_tbps.uap_detach.server.json"
c=json.loads(cfg.read_text());c["sha256"]=sha_file(cfg)
root=Path(c["output_base"])/"full/cpgc/icfg/rasa";m=read_test(c["datasets"]["icfg"])
proto=json.loads((root/"protocol/protocol.json").read_text())
space=ModelSpace(**proto["model_space"])
spec=importlib.util.spec_from_file_location("tbps.baselines.cpgc_sep.reference_cache",old/"src/tbps/baselines/cpgc_sep/cache.py")
ref=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ref;spec.loader.exec_module(ref)
score=torch.load(root/"scorechunk-clean_text--clean_gallery-000000/scores.pt",map_location="cpu",weights_only=True)
indices=score[:4].topk(128,dim=1).indices.tolist()
ids=tuple(g.image_id for g in m.gallery);hashes=tuple(g.image_sha256 for g in m.gallery)
torch.set_num_threads(2)
oldbank=ref.FeatureBank(c,ArtifactStore(root,new_base=c["output_base"]),"clean_gallery",SimpleNamespace(space=space),m,ids,hashes,modality="image")
cache=BlockCache(16*1024**3)
newstore=VerifiedArtifactStore(root,new_base=c["output_base"])
newbank=FeatureBank(c,newstore,"clean_gallery",SimpleNamespace(space=space,token_block_cache=cache),m,ids,hashes,modality="image")
report={"queries":4,"pairs":512,"old":[],"new":[],"exact":True}
oldvalues=[]
for repeat in range(2):
 t=time.monotonic()
 for i in indices:
  v=oldbank.fetch(i)
  if repeat==0:oldvalues.append(v)
 report["old"].append(time.monotonic()-t);print("OLD",repeat,report["old"][-1],flush=True)
for repeat in range(2):
 t=time.monotonic()
 for n,i in enumerate(indices):
  v=newbank.fetch(i)
  assert all(torch.equal(v[k],oldvalues[n][k]) for k in v),"cached Tensor difference"
 report["new"].append(time.monotonic()-t);print("NEW",repeat,report["new"][-1],flush=True)
report["cache"]=cache.statistics();report["RSS"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
(OUT/"result.json").write_text(json.dumps(report,indent=2))
print("CACHE_NUMERIC_EXACT_PASS",flush=True)
# Preserve full Gallery global vectors for independent Top128 boundary/Query-chunk checks.
g=newbank.read_globals()
qb=FeatureBank(c,newstore,"clean_text",SimpleNamespace(space=space),m,tuple(q.query_id for q in m.queries),
 tuple(__import__("hashlib").sha256(q.caption.encode()).hexdigest() for q in m.queries),modality="text")
q=torch.cat([qb.block(i)["global"] for i in range(0,128,8)])
torch.save({"gallery":g,"queries":q},OUT/"full_global_reference.pt")
(OUT/"completed.json").write_text(json.dumps({"status":"PASS","result_sha256":sha_file(OUT/"result.json"),"global_sha256":sha_file(OUT/"full_global_reference.pt")}))
print("CACHE_PROFILE_COMPLETE",flush=True)
