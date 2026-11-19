"""来源可审查的运行时兼容桥；原213份源码及科学配置仍由原入口严格验证。"""
from pathlib import Path
import functools
from tbps.artifacts import read_json,sha_file,FileRef,checked,ArtifactStore,json_bytes
from tbps.attribute_response.settings import require
ROOT=Path(__file__).resolve().parents[3]
ORIGINAL=Path("/home/lzf/ldx/projects/idea-test-tbps-gh-attribute-response-20261116")
CONFIG=ORIGINAL/"configs/framework/gh_attribute_response.server.json"
BASE=Path("/home/lzf/ldx/outputs/tbps-framework-gh-attribute-response-v1-20261116")
MANIFEST=ROOT/"configs/framework/gh_performance_v1.json"

def verify_version():
    spec=read_json(MANIFEST)
    require(spec["schema"]=="gh-performance-runtime-1","explicit optimization schema")
    require(spec["original_root"]==str(ORIGINAL) and spec["configuration_sha256"]==sha_file(CONFIG),
            "unchanged original scientific configuration")
    expected={str(p) for p in (ROOT/"src/tbps/performance").glob("*.py")}
    expected|={str(ROOT/"operations"/n) for n in
        ("gh_performance_worker.py","gh_performance_supervisor.py","gh_performance_benchmark.py")}
    require(set(spec["tools"])==expected,"all runtime and orchestration sources pinned")
    originals={str(ORIGINAL/n) for n in (
        "src/tbps/attribute_response/backend.py","src/tbps/attribute_response/h_step.py",
        "src/tbps/attribute_response/teacher.py","src/tbps/attribute_response/store.py",
        "src/tbps/attribute_response/run.py","src/tbps/augmented_attraction/backend.py",
        "src/tbps/augmented_attraction/views.py","src/tbps/attraction/objectives.py",
        "src/tbps/alternating/native_backend.py")}
    require(set(spec["original_sources"])==originals,"explicit complete original patch targets")
    for collection in ("tools","original_sources"):
        for path,sha in spec[collection].items():checked(FileRef(path,sha))
    return spec

def install():
    import torch
    from tbps.attribute_response import backend,h_step,teacher,run
    from tbps.alternating.native_backend import NativeBackend
    from tbps.augmented_attraction.backend import AttractionBackend
    from .images import ImagePipeline
    from .fast import fast_predict
    from .h_step import train_step
    from .bank import refresh_bank
    spec=verify_version()
    identity={"schema":"gh-performance-binding-1","manifest_sha256":sha_file(MANIFEST),
              "configuration_sha256":sha_file(CONFIG),"features":spec["features"]}
    original_load=backend.AttributeResponseBackend.load_reward_inputs
    original_close=backend.AttributeResponseBackend.close
    def load(self):
        original_load(self)
        self._performance_images=ImagePipeline(self,workers=4,window=16,pin=True)
    def image(self,row):
        return self._performance_images.cpu(row).to(next(self.g.parameters()).device)
    def views(self,row):
        return self._performance_images.views(row,self.round_index).to(
            next(self.g.parameters()).device,non_blocking=True)
    def close(self):
        pipeline=getattr(self,"_performance_images",None)
        if pipeline:pipeline.close()
        return original_close(self)
    backend.AttributeResponseBackend.load_reward_inputs=load
    backend.AttributeResponseBackend.image=image
    backend.AttributeResponseBackend.make_views=views
    backend.AttributeResponseBackend.close=close
    backend.AttributeResponseBackend.refresh_bank=refresh_bank
    teacher.ResponseTeacher.predict=fast_predict
    h_step.train_step=train_step
    old=run.FreshJournal
    class PerformanceJournal(old):
        def begin(self,index,parent):
            store=ArtifactStore(self.run/"runtime_tools",new_base=self.base)
            if (store.root/"performance_v1").exists():
                store.verify("performance_v1",expected_provenance=identity)
            else:
                refs=tuple(FileRef(p,h) for k in ("tools","original_sources")
                           for p,h in spec[k].items())
                store.publish("performance_v1",{"identity.json":json_bytes(spec)},
                    identity,sources=refs)
            super().begin(index,parent)
        def commit(self,index,state,ledger,bank_ref):
            manifest=self.run/"runtime_tools/performance_v1/manifest.json"
            ledger["runtime_optimization"]={**identity,"runtime_manifest":{
                "path":str(manifest),"sha256":sha_file(manifest)}}
            super().commit(index,state,ledger,bank_ref)
    run.FreshJournal=PerformanceJournal
    return identity

def audit_binding(endpoint,control):
    spec=verify_version()
    deployment=read_json(Path(control)/"handoff_completed.json")
    start=deployment["round"]
    started=read_json(Path(control)/"started.json")
    checked(FileRef(str(Path(control)/"benchmark.json"),started["benchmark_sha256"]))
    checked(FileRef(str(Path(control)/"cpu_tests.json"),started["cpu_tests_sha256"]))
    require(deployment["manifest_sha256"]==sha_file(MANIFEST),"handoff runtime identity")
    root=BASE/"train"
    runtime=ArtifactStore(root/"runtime_tools",new_base=BASE)
    if endpoint>start:runtime.verify("performance_v1")
    expected=sha_file(MANIFEST)
    for n in range(start+1,endpoint+1):
        value=read_json(root/f"ledgers/round_{n:04d}/ledger.json")["runtime_optimization"]
        require(value["manifest_sha256"]==expected
            and value["configuration_sha256"]==sha_file(CONFIG),"per-round optimization provenance")
        checked(FileRef(**value["runtime_manifest"]))
    original=BASE/f"train_cpu_audit/round_{endpoint:02d}/completed/report.json"
    require(read_json(original)["status"]=="GH_AR_TRAIN_CPU_AUDIT_PASS","original full CPU audit required")
    result={"status":"GH_PERFORMANCE_BINDING_CPU_AUDIT_PASS","from_round":start+1,
        "through_round":endpoint,"manifest_sha256":expected,"original_audit_sha256":sha_file(original),
        "original_SHA_and_numeric_audit_unchanged":True}
    ArtifactStore(BASE/f"performance_cpu_audit/round_{endpoint:02d}",new_base=BASE).publish(
        "completed",{"report.json":json_bytes(result)},{"runtime_manifest":expected})
    return result
