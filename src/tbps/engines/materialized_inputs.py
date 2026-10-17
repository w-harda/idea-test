"""既有封存输入的只读接线；目标模型与攻击算法均不进入输入生成。"""
from pathlib import Path
from dataclasses import asdict
import importlib.util
from tbps.artifacts import ArtifactStore,FileRef,checked,read_json,digest
from tbps.materialization.contracts import need
from tbps.materialization.bulk_session import StreamingMaterializedStore,BulkSession
from tbps.materialization.full_runner import ValidatedActions
from tbps.materialization.irra_startup import CPUStartupGate
from tbps.engines.historical_loader import native_action_validator,feature_only_cpu
from tbps.engines.materialized_irra import collection,select,bindings
from tbps.runtime_closure import cpu_only

FROZEN="/home/lzf/ldx/projects/idea-test-tbps-framework-stage4d1-20261017"
PLAN="7ac4fefd0abaa10df085be3ec3ed56c4b38c782a21bfc44473f40b90965447d5"
def load_inputs(c,scope):
    """复用冻结full_prepare真实路径，保持原Plan；不更改旧库或读取旧Embedding。"""
    import tbps.materialization.full_prepare as frozen
    need(Path(frozen.__file__).resolve()==Path(FROZEN+"/src/tbps/materialization/full_prepare.py"),
         "use tbps/rasa_eval.py entry with original frozen producer namespaces")
    gate=CPUStartupGate()
    with cpu_only():
        for name in ("materialization_config","generation_result","materialization_audit","dataset_sources"):
            checked(FileRef(**c[name]))
        for name in ("generation_result","materialization_audit"):
            folder=Path(c[name]["path"]).parent
            ArtifactStore(folder.parent,new_base=folder.parent.parent).verify(folder.name)
        done=read_json(Path(c["generation_result"]["path"]).with_name("report.json"))
        audit=read_json(Path(c["materialization_audit"]["path"]).with_name("report.json"))
        need(done["status"]=="JOINT_FULL_MATERIALIZED" and audit["status"]=="JOINT_FULL_CPU_AUDIT_PASS"
             and done["plan_id"]==audit["plan_id"]==PLAN and audit["generated"]==3074
             and audit["text"]["units"]==6156 and not audit["pending"],"sealed/audited input required")
        _,plan,legacy,_,_=frozen.prepare(c["materialization_config"]["path"])
        need(plan.id==PLAN,"frozen producer Plan identity changed")
        from tbps.datasets.cuhk_pedes import CuhkPedes
        ds=read_json(c["dataset_sources"]["path"])
        dataset=CuhkPedes(FileRef(**ds["annotation"]),ds["image_root"])
        need(dataset.manifest("test",expected_counts=(3074,6156))==plan.manifest,"official identity/order")
        store=StreamingMaterializedStore(ArtifactStore(Path(frozen.FULL)/"artifacts",new_base=frozen.BASE),plan)
        need(not store.store.pending(),"pending materialized inputs")
        pairs={};stats={"edited":0,"NOOP":0,"edits":0}
        with BulkSession(store),native_action_validator(legacy.config_ref) as native:
            images,texts,core=collection(store);validator=ValidatedActions(native)
            for u in texts:
                adv,md=store.load(u,validate=validator);original=md["original_caption"]
                pairs[u.binding.sample_id]=(original,adv)
                stats["NOOP" if original==adv else "edited"]+=1;stats["edits"]+=len(md["action"]["edits"])
            need(stats=={"edited":6116,"NOOP":40,"edits":13537},"full historical text preservation")
        manifest=select(plan.manifest,scope)
    with feature_only_cpu({images[0].data.path,images[-1].data.path}):
        for u in (images[0],images[-1]):
            arrays,_=store.load(u);del arrays
    gate.checked();gate.exit_checked()
    return dict(c=c,plan=plan,legacy=legacy,store=store,manifest=manifest,gate=gate,core=core,
        images={u.binding.sample_id:u for u in images},text_units={u.binding.sample_id:u for u in texts},
        texts=pairs,report={"status":"MATERIALIZED_INPUT_CPU_PASS","scope":scope,"plan_id":PLAN,
            "gallery":len(manifest.gallery),"query":len(manifest.queries),"text_statistics":stats,"model_loads":0})
