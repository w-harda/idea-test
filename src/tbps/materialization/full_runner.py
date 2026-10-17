"""全量物化的流式事务调度：H历史复用CPU，G仅显式外部授权与冻结工厂。"""
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
import time
from ..artifacts import FileRef,sha_file,json_bytes,read_json,digest,PendingPublication
from ..attacks.base import frozen_state
from .contracts import need
from .bulk_session import BulkSession,StreamingMaterializedStore
from .generation import generate_image_once,seal_generated_collection
from .pipeline import seal_collection

class ValidatedActions:
    """仅缓存已验证的相同Caption+完整动作，受限容量；不选择/生成动作。"""
    def __init__(self,validate,maxsize=64):
        need(callable(validate) and 0<maxsize<=256,"bounded native action validator")
        self.validate=validate;self.maxsize=maxsize;self.cache=OrderedDict();self.hits=0;self.misses=0
    def __call__(self,caption,action):
        key=(caption,json_bytes(action))
        if key in self.cache:
            self.hits+=1;self.cache.move_to_end(key);return self.cache[key]
        result=self.validate(caption,action);self.misses+=1;self.cache[key]=result
        if len(self.cache)>self.maxsize:self.cache.popitem(last=False)
        return result

def ref_file(path):return FileRef(str(path),sha_file(path))

def publish_texts(store,source,validate,*,expected=None,progress=None,limit=None):
    """limit仅供独立小批Pilot；完整模式要求全部Query且不能静默跳过。"""
    need(type(store)is StreamingMaterializedStore,"streaming Store")
    refs=[];stats={"edited":0,"NOOP":0,"edits":0};start=time.perf_counter()
    with BulkSession(store) as session:
        for request,result in source.iter_outputs():
            index=len(refs)
            need(index<len(store.plan.manifest.queries) and request==store.text_request(store.plan.manifest.queries[index].query_id),
                 "historical Query order/identity mismatch")
            if store.available(request.binding):
                ref=store.reference(request.binding)
            else:
                ref=store.publish_text(request,result,validate)
            text,metadata=store.load(ref,validate=validate)
            need(text==result.payload and metadata["action"]==result.trace()["action"],"UTF8/action readback exact")
            stats["NOOP" if text==request.public.caption else "edited"]+=1
            stats["edits"]+=len(metadata["action"]["edits"]);refs.append(ref)
            if progress and (len(refs)%64==0 or len(refs)==len(store.plan.manifest.queries)):
                progress(len(refs),time.perf_counter()-start,dict(stats))
            if limit is not None and len(refs)==limit:break
        if limit is None:
            need(len(refs)==len(store.plan.manifest.queries),"complete Query coverage required")
            need(source.report()["status"]=="TEXT_SOURCE_VERIFIED","full historical source audit incomplete")
            if expected is not None:need(stats==expected,"fixed full text counts mismatch")
            collection=seal_collection(store,"text_only",texts=tuple(refs),validate=validate)
            sealed=None
        else:
            need(len(refs)==limit,"pilot size mismatch");sealed=None
        source_stats=dict(session.cache.stats)
    store.verify_plan(force=True)
    if limit is None:
        # 原schema集合保持；补充独立语义回执，不能被误称Joint/G完整。
        files={"text_collection.json":json_bytes({"schema":"tbps-text-full-materialized-1",
            "status":"TEXT_FULL_MATERIALIZED","origin":"historical_H7_text_reuse",
            "plan_id":store.plan.id,"collection":asdict(collection),
            "ordered_query_ids":[r.binding.sample_id for r in refs],"stats":stats,
            "H_model_loads":0,"H_forwards":0,"G_model_loads":0,"G_forwards":0,
            "image_collection_status":"PENDING","joint_collection_status":"PENDING"})}
        store.store.publish("text-full",files,{"kind":"historical_H7_text_collection","plan_id":store.plan.id},
            sources=(collection,ref_file(__file__)))
        sealed=ref_file(store.store.root/"text-full/text_collection.json")
    return {"stats":stats,"units":len(refs),"elapsed_seconds":time.perf_counter()-start,
            "collection":asdict(sealed) if sealed else None,"session":source_stats,
            "H_model_loads":0,"H_forwards":0,"G_model_loads":0,"G_forwards":0},tuple(refs)

def stream_images(store,plugin_factory,*,allow_real_generation=False,resource_check=None,progress=None):
    """没有默认模型加载；real调用须由下一阶段独立授权入口提供资源检查。"""
    need(type(store)is StreamingMaterializedStore and callable(plugin_factory),"explicit stream/factory")
    need(store.plan.fixture or (allow_real_generation is True and callable(resource_check)),
         "real G stream requires separately authorized resource policy")
    if resource_check:resource_check()
    start=time.perf_counter();refs=[];generated=0;skipped=0
    # 在模型加载之前检查所有已存在单元和不完整Claim；不能加载后才发现需重复前向。
    todo=[]
    with BulkSession(store):
        need(not store.store.pending(),"pending image/Claim/proof; refuse loading G")
        for row in store.plan.manifest.gallery:
            req=store.image_request(row.image_id);unit=store.unit_id(req.binding)
            claim="generation-claim-"+unit;proof="generation-proof-"+unit
            if store.available(req.binding):
                need((store.store.root/proof).exists(),"existing generated unit lacks proof")
                store.store.verify(proof);ref=store.reference(req.binding);store.load(ref)
                todo.append((row.image_id,ref));skipped+=1
            else:
                need(not (store.store.root/claim).exists(),"uncommitted forward Claim; refuse regeneration")
                todo.append((row.image_id,None))
        if all(ref is not None for _,ref in todo):
            refs=[ref for _,ref in todo]
        else:
            # 单次模型会话；只保留当前像素，历史列表仅保存小型FileRef。
            with plugin_factory() as plugin:
                before=frozen_state(plugin.backend.model)
                for key,existing in todo:
                    if resource_check:resource_check()
                    if existing is not None:refs.append(existing);continue
                    ref=generate_image_once(store,key,plugin,lambda r:plugin.backend.load_pixels(r.path),
                                             allow_real_generation=allow_real_generation)
                    arrays,_=store.load(ref);del arrays
                    refs.append(ref);generated+=1
                    if progress:progress(len(refs),time.perf_counter()-start,{"generated":generated,"skipped":skipped})
                need(frozen_state(plugin.backend.model)==before,"G state changed during stream")
        need(tuple(r.binding.sample_id for r in refs)==tuple(g.image_id for g in store.plan.manifest.gallery),"complete Gallery order")
        if (store.store.root/"image-full").exists():
            store.store.verify("image-full")
            body=read_json(store.store.root/"image-full/image_collection.json")
            need(body["plan_id"]==store.plan.id and body["ordered_image_ids"]==[r.binding.sample_id for r in refs],
                 "completed image collection identity")
            collection=FileRef(**body["collection"]);need(sha_file(collection.path)==collection.sha256,"collection SHA")
        else:
            collection=seal_generated_collection(store,"image_only",images=tuple(refs))
    store.verify_plan(force=True)
    if not (store.store.root/"image-full").exists():
        store.store.publish("image-full",{"image_collection.json":json_bytes({"schema":"tbps-image-full-materialized-1",
            "status":"IMAGE_FULL_MATERIALIZED","plan_id":store.plan.id,"collection":asdict(collection),
            "ordered_image_ids":[r.binding.sample_id for r in refs],"H_forwards":0,"IRRA_forwards":0})},
            {"kind":"image_only_materialized_collection","plan_id":store.plan.id},sources=(collection,ref_file(__file__)))
    return {"collection":asdict(collection),"generated":generated,"skipped":skipped,
            "G_loads":int(generated>0),"H_loads":0,"IRRA_loads":0,"elapsed_seconds":time.perf_counter()-start}
