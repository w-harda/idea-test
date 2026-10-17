"""可恢复物化的前置生成Claim：失败后禁止自动重做可能已发生的模型前向。"""
from dataclasses import asdict
from pathlib import Path
from ..artifacts import FileRef,sha_file,json_bytes,read_json,PendingPublication,digest
from .contracts import need
from .pipeline import materialize_image,materialize_text,seal_collection

def _prepare(store,request,*,validate=None,allow_real_generation=False):
    need(store.plan.fixture or allow_real_generation is True,"real generation not authorized")
    store.verify_plan();unit=store.unit_id(request.binding);claim="generation-claim-"+unit;proof="generation-proof-"+unit
    if store.available(request.binding):
        if (store.store.root/claim).exists():
            need((store.store.root/proof).exists(),"completed unit has uncommitted generation proof; inspect before resume")
            store.store.verify(proof)
        ref=store.reference(request.binding);store.load(ref,validate=validate)
        return ref,None,None
    for p in store.store.pending():
        intent=store.store.root/p/"intent.json"
        if not intent.exists() or read_json(intent).get("artifact_id") in (claim,proof):
            raise PendingPublication("uncommitted generation Claim/proof; preserve and stop")
    if (store.store.root/claim).exists():
        store.store.verify(claim)
        raise PendingPublication("generation may already have occurred; never repeat it automatically")
    code=FileRef(str(Path(__file__)),sha_file(__file__))
    store.store.publish(claim,{"generation.json":json_bytes({"schema":"tbps-generation-claim-1","unit_id":unit,
        "plan_id":store.plan.id,"binding":asdict(request.binding),"may_have_generated":True})},
        {"kind":"generation_claim","plan_id":store.plan.id,"unit_id":unit},sources=(store.verify_plan(),code))
    return None,claim,proof

def _finish(store,ref,claim,proof):
    base=store.store.root
    sources=(FileRef(str(base/claim/"manifest.json"),sha_file(base/claim/"manifest.json")),ref.manifest,
             FileRef(str(Path(__file__)),sha_file(__file__)))
    store.store.publish(proof,{"generation.json":json_bytes({"schema":"tbps-generation-proof-1","unit":asdict(ref),
        "claim_manifest":asdict(sources[0]),"new_G_H_forwards_during_publication":0})},
        {"kind":"generation_proof","plan_id":store.plan.id,"unit_id":ref.unit_id},sources=sources)
    return ref

def generate_image_once(store,image_id,plugin,clean_reader,*,allow_real_generation=False):
    request=store.image_request(image_id)
    existing,claim,proof=_prepare(store,request,allow_real_generation=allow_real_generation)
    if existing is not None:return existing
    ref=materialize_image(store,image_id,plugin,clean_reader,allow_real_generation=allow_real_generation)
    return _finish(store,ref,claim,proof)

def generate_text_once(store,query_id,plugin,validate,*,allow_real_generation=False):
    request=store.text_request(query_id)
    existing,claim,proof=_prepare(store,request,validate=validate,allow_real_generation=allow_real_generation)
    if existing is not None:return existing
    ref=materialize_text(store,query_id,plugin,validate,allow_real_generation=allow_real_generation)
    return _finish(store,ref,claim,proof)

def seal_generated_collection(store,mode,*,images=(),texts=(),validate=None):
    proofs=[]
    for ref in (*images,*texts):
        claim=store.store.root/("generation-claim-"+ref.unit_id)
        proof="generation-proof-"+ref.unit_id
        if claim.exists():
            need((store.store.root/proof).exists(),"cannot seal incomplete generation")
            store.store.verify(proof)
            p=store.store.root/proof/"manifest.json";proofs.append(FileRef(str(p),sha_file(p)))
    collection=seal_collection(store,mode,images=images,texts=texts,validate=validate)
    identity=digest(asdict(collection))
    body={"collection":asdict(collection),"generation_proofs":[asdict(p) for p in proofs],
          "mode":mode,"new_G_H_forwards":0,"historical_text_reuse_requires_no_generation_claim":True}
    store.store.publish("materialization-set-"+identity,{"index.json":json_bytes(body)},
        {"kind":"materialization_set","plan_id":store.plan.id},sources=(collection,*proofs,
         FileRef(str(Path(__file__)),sha_file(__file__))))
    p=store.store.root/("materialization-set-"+identity)/"index.json"
    return FileRef(str(p),sha_file(p))
