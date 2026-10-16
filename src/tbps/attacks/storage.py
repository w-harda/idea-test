"""仅新框架的像素/文本攻击产物发布；不保存victim Embedding。"""
from dataclasses import asdict
from pathlib import Path
import io
from ..artifacts import ArtifactStore, json_bytes, sha_file
from ..contracts import AttackArtifact, FileRef
from .base import AttackOutput, need, tensor_sha

def publish_output(store, result, *, sources=()):
    need(type(store) is ArtifactStore and type(result) is AttackOutput, "typed new-framework Store/output")
    trace=result.trace();provenance=result.provenance()
    metadata={"schema":"tbps-attack-output-1","attack_id":result.attack_id,
              "binding":asdict(result.binding),"budget":asdict(result.budget),
              "trace":trace,"provenance":provenance}
    if result.attack_id=="gv2":
        import torch
        need(isinstance(result.payload,torch.Tensor) and isinstance(result.effective_delta,torch.Tensor),
             "pixel/perturbation payload required")
        need(tensor_sha(result.payload)==trace["adversarial_pixels_sha256"] and
             tensor_sha(result.effective_delta)==trace["effective_delta_sha256"], "pixel payload changed after generation")
        stream=io.BytesIO()
        torch.save({"adversarial":result.payload.detach().cpu().clone(),
                    "effective_delta":result.effective_delta.detach().cpu().clone()},stream)
        name="pixels.pt";content=stream.getvalue()
    elif result.attack_id=="hv1":
        need(type(result.payload) is str, "Unicode Caption payload required")
        need(result.payload==trace["action"]["adversarial"], "Caption payload/action mismatch")
        name="caption.json";content=json_bytes({"adversarial":result.payload})
    else: raise ValueError("unsupported attack payload serialization")
    bound={ref.path:ref for ref in sources}
    if result.attack_id=="gv2":
        ref=FileRef(**trace["input_file"]);bound[ref.path]=ref
    if provenance["execution_kind"]=="legacy_unverified":
        ref=FileRef(**provenance["checkpoint"]);bound[ref.path]=ref
    store.publish(result.artifact_id,{name:content,"action.json":json_bytes(metadata)},
                  {"schema":"tbps-attack-publication-1","attack_id":result.attack_id,
                   "input_binding":result.binding.id,"attack_sources":provenance["sources_sha256"]},
                  sources=tuple(bound.values()))
    path=store.root/result.artifact_id/name
    return AttackArtifact(result.artifact_id,result.attack_id,provenance["checkpoint"]["sha256"],
                          (result.binding.input_sha256,),FileRef(str(path),sha_file(path)),result.budget)
