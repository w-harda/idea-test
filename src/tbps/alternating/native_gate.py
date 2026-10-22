"""默认拒绝真实执行；探针与两轮Smoke分别显式授权，回执严格绑定当前来源。"""
from dataclasses import dataclass
from pathlib import Path
from tbps.artifacts import ArtifactStore, FileRef, checked, digest, read_json, sha_file
from .contracts import require
from .native_sources import source_key

@dataclass(frozen=True)
class ExecutionGrant:
    action: str
    source_sha256: str
    protocol_sha256: str
    probe_manifest_sha256: str | None = None

    def validate(self, c, action):
        require(self.action == action and self.source_sha256 == source_key(c)
                and self.protocol_sha256 == digest(c["prototype"]), "foreign execution grant")
        if action == "smoke":
            require(self.probe_manifest_sha256 is not None, "numerical probe PASS required")
            receipt = verify_probe(c)
            require(receipt == self.probe_manifest_sha256, "probe changed after authorization")
        return self

def verify_probe(c):
    root = Path(c["probe_output"])
    store = ArtifactStore(root, new_base=c["output_base"])
    require(not store.pending() and not (root/"FAIL").exists(), "probe incomplete/failed")
    store.verify("completed", expected_provenance={"source":source_key(c),"protocol":digest(c["prototype"])})
    result = read_json(root/"completed/report.json")
    require(result["status"] == "GH_NATIVE_NUMERIC_PROBE_PASS"
            and result["optimizer_steps"] == 0 and result["full_gallery_generations"] == 0
            and result["frozen_sources"] is True
            and result["nonnoop_gradient_max_abs"] > 0
            and result["noop_gradient_max_abs"] <= c["probe"]["noop_gradient_atol"],
            "numerical probe did not verify required interfaces")
    counts=result["counts"]
    require(counts["G"]==counts["IRRA_image"]==4 and counts["JVP"]==counts["reader"]==counts["H"]==1
            and 0<counts["IRRA_text"]<=c["probe"]["max_IRRA_texts"],"probe actual forward coverage")
    require(result["original_text_cache_max_abs"]<=1e-6,"original text identity failed")
    return sha_file(root/"completed/manifest.json")

def authorize(c, action, *, allow_probe=False, allow_smoke=False):
    require(action in ("probe","smoke"), "execution action")
    require((action=="probe" and allow_probe and not allow_smoke)
            or (action=="smoke" and allow_smoke and not allow_probe), "explicit separate GPU authorization required")
    require(not Path(c[action+"_output"]).exists(), "Attempt consumed; preserve failure/no automatic retry")
    return ExecutionGrant(action, source_key(c), digest(c["prototype"]),
                          verify_probe(c) if action=="smoke" else None)
