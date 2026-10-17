"""按独立Gallery/Query顺序物化；无受害者、标签反馈或自动模型加载入口。"""
from dataclasses import asdict
from ..artifacts import FileRef,sha_file,json_bytes,digest
from ..models.base import InputBinding,ModelSpace
from ..attacks.gv2 import GV2Plugin
from ..attacks.hv1 import HV1Plugin
from .contracts import need

def materialize_image(store,image_id,plugin,clean_reader,*,allow_real_generation=False):
    need(store.plan.fixture or allow_real_generation is True,"real G generation not authorized")
    store.verify_plan()
    request=store.image_request(image_id)
    if store.available(request.binding):
        ref=store.reference(request.binding);store.load(ref);return ref
    need(type(plugin)is GV2Plugin and plugin.provenance==store.plan.g and callable(clean_reader),"frozen G component/reader required")
    # 读取原RGB用于正式像素SHA/有效扰动验收；模型只收到插件自己的公开图像输入。
    clean=clean_reader(request.public.content)
    result=plugin.generate(request)
    return store.publish_image(request,clean,result)

def materialize_text(store,query_id,plugin,validate,*,allow_real_generation=False):
    need(store.plan.fixture or allow_real_generation is True,"real H generation not authorized")
    store.verify_plan()
    request=store.text_request(query_id)
    if store.available(request.binding):
        ref=store.reference(request.binding);store.load(ref,validate=validate);return ref
    need(type(plugin)is HV1Plugin and plugin.provenance==store.plan.h,"frozen H component required")
    return store.publish_text(request,plugin.generate(request),validate)

def seal_collection(store,mode,*,images=(),texts=(),validate=None):
    need(mode in ("image_only","text_only","joint"),"explicit materialized mode")
    need(type(images)is tuple and type(texts)is tuple,"immutable independent input lists")
    g=tuple(x.image_id for x in store.plan.manifest.gallery) if mode in ("image_only","joint") else ()
    q=tuple(x.query_id for x in store.plan.manifest.queries) if mode in ("text_only","joint") else ()
    need(tuple(x.binding.sample_id for x in images)==g and tuple(x.binding.sample_id for x in texts)==q,
         "complete original Gallery/Query order required; never zip or drop NOOP")
    need(all(x.modality=="image" for x in images) and all(x.modality=="text" for x in texts),"collection modality mismatch")
    plan_ref=store.verify_plan(force=True)
    refs={plan_ref.path:plan_ref}
    for ref in (*images,*texts):
        value,_=store.load(ref,validate=validate);del value
        for s in (ref.manifest,ref.data,ref.metadata):refs[s.path]=s
    store.verify_plan(force=True)
    body={"schema":"tbps-materialized-collection-1","plan_id":store.plan.id,"mode":mode,
          "images":[asdict(x) for x in images],"texts":[asdict(x) for x in texts],
          "new_G_forwards":0,"new_H_forwards":0,"victim_model":None,"joint_operation":"references only"}
    artifact_id=digest(body)
    store.store.publish(artifact_id,{"collection.json":json_bytes(body)},
        {"kind":"materialized_collection","plan_id":store.plan.id,"mode":mode},sources=tuple(refs.values()))
    p=store.store.root/artifact_id/"collection.json"
    return FileRef(str(p),sha_file(p))

def target_input_binding(store,space,modality,refs,*,validate=None):
    """同一真实输入内容按不同目标ModelSpace隔离；不返回旧Embedding，不执行目标模型。"""
    need(type(space)is ModelSpace and modality in ("image","text") and type(refs)is tuple,"explicit target identity/modality")
    members=store.plan.manifest.gallery if modality=="image" else store.plan.manifest.queries
    ids=tuple(x.image_id if modality=="image" else x.query_id for x in members)
    need(tuple(x.binding.sample_id for x in refs)==ids and all(x.modality==modality for x in refs),"target input complete ordered identity")
    for ref in refs:
        value,_=store.load(ref,validate=validate);del value
    store.verify_plan(force=True)
    return InputBinding(space.id,store.plan.manifest_sha256,tuple(x.content_sha256 for x in refs),modality,ids)
