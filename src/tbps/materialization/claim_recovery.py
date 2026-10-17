"""仅generation_v2零前向首图Claim的显式恢复；不改变旧生成算法或删除Claim。"""
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
import time
from tbps.artifacts import ArtifactStore, FileRef, checked, read_json, json_bytes, digest, sha_file
from tbps.materialization.contracts import need
from tbps.materialization.audit_retained import snapshot
from tbps.materialization.bulk_session import BulkSession
from tbps.materialization import generation
from tbps.materialization.full_runner import stream_images

FIRST = "cuhk:37132:0"
OLD_ATTEMPT = "/home/lzf/ldx/outputs/tbps-framework-stage4d2/generation_v2"
PLAN = "7ac4fefd0abaa10df085be3ec3ed56c4b38c782a21bfc44473f40b90965447d5"

def file_ref(path): return FileRef(str(Path(path).resolve()), sha_file(path))

class FirstClaimRecovery:
    def __init__(self, reference):
        self.reference = reference
        checked(reference)
        self.spec = read_json(reference.path, "tbps-zero-forward-first-claim-1")
        groups=[self.spec["failed_files"],self.spec["claim_files"],*self.spec["prior_failed_files"].values()]
        for group in groups:
            need(type(group) is dict and all(type(v) is dict and
                 set(v)=={"path","sha256","size","mtime_ns"} and v["path"]==path and
                 type(v["size"]) is int and type(v["mtime_ns"]) is int for path,v in group.items()),
                 "complete canonical snapshot path/SHA/size/mtime required")

    def verify(self, store, *, inventory=True):
        checked(self.reference)
        s = self.spec
        need(set(s) == {"schema","plan_id","image_id","unit_id","failed_attempt",
                       "failed_files","claim_files","baseline_files","prior_failed_files"}, "recovery schema fields")
        need(store.plan.id == s["plan_id"] and s["image_id"] == store.plan.manifest.gallery[0].image_id,
             "only original first Gallery member can recover")
        if not store.plan.fixture:
            need(s["plan_id"] == PLAN and s["image_id"] == FIRST and s["failed_attempt"] == OLD_ATTEMPT,
                 "only fixed generation_v2 / CUHK first-image recovery")
            need(set(s["prior_failed_files"])=={str(Path(OLD_ATTEMPT).with_name("generation_v1"))},
                 "retain original generation_v1 evidence")
        for path,files in s["prior_failed_files"].items():
            need(snapshot(path)==files,"prior FAIL evidence changed")
        old = Path(s["failed_attempt"])
        need(snapshot(old) == s["failed_files"], "failed Attempt bytes/mtime changed")
        records = ArtifactStore(old, new_base=next(p for p in (old,*old.parents) if p.name.startswith("tbps-framework")))
        for name in ("attempt","gpu-start","gpu-result","result"): records.verify(name)
        pre = read_json(old/"attempt/preflight.json")
        report = read_json(old/"gpu-result/report.json")
        result = read_json(old/"result/report.json")
        need(report["status"] == result["status"] == "FAIL" and report["plan_id"] == result["plan_id"] == store.plan.id,
             "original Attempt must remain FAIL with same Plan")
        need(pre["status"] == "G_FULL_CPU_PREFLIGHT_PASS" and pre["plan_id"] == store.plan.id
             and pre["CPU_contexts_restored"] and pre["GH_status"] == "PASS", "original launch provenance")
        counts = report["counts"]; actual = report["actual_G_forwards"]
        need(all(type(actual[k]) is int for k in ("started","completed"))
             and all(type(counts[k]) is int for k in ("G_load_attempts","G_loads","pixel_tensor_loads"))
             and actual == {"started":0,"completed":0,"records":[],"historical_two_image_comparisons":{}}
             and counts == {"G_load_attempts":1,"G_loads":1,"pixel_tensor_loads":0},
             "zero actual G forward / single restore evidence required")
        need(all(type(report[k]) is int and report[k] == 0 for k in
             ("H_loads","H_forwards","IRRA_loads","IRRA_forwards","backward","optimizer","JVP")),
             "other models/optimization in failed Attempt")
        need(report["error"] == "non-G neural forward forbidden"
             and "pixels_from_pil" in report["traceback"] and "clean_reader" in report["traceback"],
             "only verified preprocessing failure before G")
        request = store.image_request(s["image_id"]); uid = store.unit_id(request.binding)
        need(uid == s["unit_id"], "recovery unit identity mismatch")
        root = store.store.root; claim = root/("generation-claim-"+uid)
        store.store.verify(claim.name)
        need(snapshot(claim) == s["claim_files"] and len(s["claim_files"]) == 3, "old Claim bytes/mtime mismatch")
        expected = {"schema":"tbps-generation-claim-1","unit_id":uid,"plan_id":store.plan.id,
                    "binding":asdict(request.binding),"may_have_generated":True}
        need(read_json(claim/"generation.json") == expected, "first Claim binding/image SHA mismatch")
        checked(request.public.content)
        need(not store.available(request.binding) and not store.store.pending(), "image/pending already exists")
        need(not any(root.glob("generation-proof-*")) and not (root/"image-full").exists()
             and not (root/"joint-full").exists(), "Proof/Image/Joint already exists")
        need([p.name for p in root.glob("generation-claim-*")] == [claim.name], "exactly one first-image Claim required")
        for row in store.plan.manifest.gallery:
            need(not store.available(store.image_request(row.image_id).binding), "any image unit forbids recovery")
        baseline = FileRef(**s["baseline_files"]); checked(baseline)
        need(Path(baseline.path) == old/"attempt/old_files.json", "use original pre-Claim inventory")
        original = read_json(baseline.path)
        need(not set(original).intersection(s["claim_files"]), "Claim already in original inventory")
        if inventory:
            # 当前全Store只能比原30797文件多这3个Claim文件，不能忽略任意新增/篡改。
            full = root.parent
            need(snapshot(full) == {**original, **s["claim_files"]}, "Full inventory differs beyond exact first Claim")
        return {"status":"ZERO_FORWARD_FIRST_CLAIM_RECOVERY_VERIFIED","plan_id":store.plan.id,
                "image_id":s["image_id"],"unit_id":uid,"old_claim":asdict(file_ref(claim/"manifest.json")),
                "evidence":asdict(self.reference),"prior_G_forwards":0,"new_G_forwards":0}

    @contextmanager
    def resumed_prepare(self, store):
        """仅对已验证首图的一次原generate_image_once开放；其余调用原保护。"""
        self.verify(store)
        import threading
        from tbps.attacks.legacy_runtime import _LOCK
        original = generation._prepare
        called = [];owner=threading.get_ident()
        if not store.plan.fixture:need(_LOCK.locked(),"active frozen G session required")
        def prepare(current, request, **kwargs):
            need(threading.get_ident()==owner,"Claim recovery is single-thread scoped")
            if current is not store or request != store.image_request(self.spec["image_id"]):
                return original(current, request, **kwargs)
            need(not called, "one first-image recovery call only")
            self.verify(store, inventory=False); called.append(True)
            uid = self.spec["unit_id"]
            return None, "generation-claim-"+uid, "generation-proof-"+uid
        with patch.object(generation, "_prepare", new=prepare):
            yield
        need(called == [True], "first Claim was not consumed exactly once")

def stream_with_recovery(store, factory, recovery, attempt, *, resource_check, progress):
    eligibility = recovery.verify(store)
    need(not (attempt.root/"claim-recovery-start").exists(), "recovery attempt already consumed")
    started = time.perf_counter()
    attempt.publish("claim-recovery-start", {"recovery.json":json_bytes(eligibility)},
        {"kind":"one_zero_forward_claim_recovery"},
        sources=(recovery.reference,FileRef(**eligibility["old_claim"]),
                 file_ref(attempt.root/"gpu-start/manifest.json"),file_ref(__file__)))
    resource_check()
    with factory() as plugin:
        with BulkSession(store):
            with recovery.resumed_prepare(store):
                unit = generation.generate_image_once(store,recovery.spec["image_id"],plugin,
                    lambda r:plugin.backend.load_pixels(r.path),allow_real_generation=True)
            arrays,_=store.load(unit);del arrays
            proof=file_ref(store.store.root/("generation-proof-"+unit.unit_id)/"manifest.json")
            attempt.publish("claim-recovery-proof",{"recovery.json":json_bytes({
                **eligibility,"status":"FIRST_CLAIM_RECOVERED","new_G_forwards":1,
                "unit":asdict(unit),"normal_generation_proof":asdict(proof)})},
                {"kind":"zero_forward_first_claim_recovery_proof"},
                sources=(recovery.reference,FileRef(**eligibility["old_claim"]),proof,unit.manifest,
                         file_ref(attempt.root/"claim-recovery-start/manifest.json"),file_ref(__file__)))
        progress(1,time.perf_counter()-started,{"generated":1,"skipped":0})
        @contextmanager
        def shared(): yield plugin
        def adjusted(n,elapsed,stats):
            progress(n,time.perf_counter()-started,{"generated":stats["generated"]+1,"skipped":stats["skipped"]-1})
        result=stream_images(store,shared,allow_real_generation=True,resource_check=resource_check,progress=adjusted)
    need(result["generated"]==len(store.plan.manifest.gallery)-1 and result["skipped"]==1,
         "only this run's recovered first unit can be skipped by legacy stream")
    return {**result,"generated":result["generated"]+1,"skipped":0,"recovered_first":1,
            "legacy_stream_generated":result["generated"],"legacy_stream_skipped":1,
            "elapsed_seconds":time.perf_counter()-started,
            "recovery_proof":asdict(file_ref(attempt.root/"claim-recovery-proof/manifest.json"))}

def verify_recovery_completion(store, recovery, attempt):
    """生成结束后的独立CPU校验，不再要求恢复前的无图像状态。"""
    s=recovery.spec
    checked(recovery.reference)
    for path,files in s["prior_failed_files"].items():
        need(snapshot(path)==files,"prior FAIL evidence changed")
    need(snapshot(s["failed_attempt"])==s["failed_files"],"v2 FAIL evidence changed")
    claim=store.store.root/("generation-claim-"+s["unit_id"])
    need(snapshot(claim)==s["claim_files"],"old Claim overwritten")
    for name in ("claim-recovery-start","claim-recovery-proof"):attempt.verify(name)
    start=read_json(attempt.root/"claim-recovery-start/recovery.json")
    proof=read_json(attempt.root/"claim-recovery-proof/recovery.json")
    need(start["evidence"]==proof["evidence"]==asdict(recovery.reference)
         and start["old_claim"]==proof["old_claim"]==asdict(file_ref(claim/"manifest.json"))
         and start["prior_G_forwards"]==proof["prior_G_forwards"]==0
         and start["new_G_forwards"]==0 and proof["new_G_forwards"]==1
         and proof["status"]=="FIRST_CLAIM_RECOVERED"
         and start["image_id"]==proof["image_id"]==s["image_id"]
         and start["unit_id"]==proof["unit_id"]==s["unit_id"]
         and start["plan_id"]==proof["plan_id"]==store.plan.id,"recovery proof identity/counts")
    unit=store.reference(store.image_request(s["image_id"]).binding)
    normal=file_ref(store.store.root/("generation-proof-"+unit.unit_id)/"manifest.json")
    need(proof["unit"]==asdict(unit) and proof["normal_generation_proof"]==asdict(normal),
         "recovery must bind original normal generation proof/unit")
    store.store.verify(Path(normal.path).parent.name)
    return {"image_id":s["image_id"],"prior_G_forwards":0,"new_G_forwards":1,
            "old_claim_preserved":True,"proof":asdict(file_ref(attempt.root/"claim-recovery-proof/manifest.json"))}
