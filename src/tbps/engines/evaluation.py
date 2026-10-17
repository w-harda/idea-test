"""四组只读评测：输入预先准备，评分归Adapter，身份仅进入最后的排名。"""
from dataclasses import asdict
from pathlib import Path
import io
import hashlib
from ..artifacts import ArtifactError, ArtifactStore, FileRef, checked, digest, json_bytes, sha_file
from ..contracts import DatasetManifest, ScoreArtifact
from ..models.base import ModelAdapter, ScoreBatch, RepresentationBatch, ScoreRecord, InputBinding
from .metrics import ranking, drops, RANKING, RANKING_ID
from .provenance import BoundRepresentation, payload_digest

MODES={"Clean":("clean_text","clean_gallery"),"Image-only":("clean_text","adv_gallery"),
       "Text-only":("adv_text","clean_gallery"),"Joint":("adv_text","adv_gallery")}

class EvaluationEngine:
    def __init__(self,manifest,adapter,*,direction,execution="cpu_fixture",model_sources=()):
        if type(manifest)is not DatasetManifest or not isinstance(adapter,ModelAdapter):
            raise ArtifactError("DatasetManifest/native ModelAdapter required")
        if direction not in ("higher_better","lower_better"):raise ArtifactError("explicit score direction required")
        native_direction=getattr(adapter,"score_direction","higher_better" if adapter.spec.id=="irra" else None)
        if native_direction!=direction:raise ArtifactError("native Adapter score direction conflict")
        if "score" not in adapter.spec.capabilities or adapter.space.model_id!=adapter.spec.id:
            raise ArtifactError("native model score capability/space mismatch")
        if execution=="cpu_fixture":
            if not adapter.spec.id.endswith("_fixture") or len(manifest.gallery)>64 or len(manifest.queries)>64:
                raise ArtifactError("explicit small CPU fixture model required")
            if manifest.dataset_id not in ("cuhk_pedes_fixture","fusion_dataset_fixture"):
                raise ArtifactError("unsupported fixture Dataset")
        elif execution=="readonly_cached":
            if manifest.dataset_id!="cuhk_pedes" or adapter.spec.id!="irra" or getattr(adapter,"source_verified",False)is not True:
                raise ArtifactError("unimplemented Dataset/Model or unverified cached source")
            if getattr(adapter,"execution_kind",None)!="readonly_cached":
                raise ArtifactError("explicit model-free cached scoring Adapter required")
        elif execution=="materialized_native":
            if (manifest.dataset_id!="cuhk_pedes" or manifest.official_split!="test"
                or getattr(adapter,"execution_kind",None)!="materialized_native"
                or getattr(adapter,"source_verified",False)is not True):
                raise ArtifactError("verified native Adapter and official materialized dataset required")
        elif execution in ("historical_feature_replay","historical_fixture_replay"):
            need_fixture=execution=="historical_fixture_replay"
            if (getattr(adapter,"execution_kind",None)!=execution or getattr(adapter,"source_verified",False)is not True or adapter.space.model_id!=("irra_fixture" if need_fixture else "irra")
                or manifest.dataset_id!="cuhk_pedes" or manifest.official_split!="test"):
                raise ArtifactError("invalid historical IRRA/CUHK replay execution")
            if need_fixture and max(len(manifest.gallery),len(manifest.queries))>64:
                raise ArtifactError("small historical fixture only")
            if not need_fixture and (len(manifest.gallery),len(manifest.queries))!=(3074,6156):
                raise ArtifactError("historical Full identity size")
        else:raise ArtifactError("unknown evaluation execution; no model/attack loading engine")
        if getattr(adapter,"input_gradients",False):raise ArtifactError("Surrogate/gradient view forbidden")
        self.manifest,self.adapter,self.direction,self.execution=manifest,adapter,direction,execution
        if type(model_sources)is not tuple or any(type(ref)is not FileRef for ref in model_sources):
            raise ArtifactError("immutable model source FileRefs required")
        if execution in ("readonly_cached","historical_feature_replay","materialized_native") and not any(ref.sha256==adapter.space.checkpoint_sha256 for ref in model_sources):
            raise ArtifactError("cached victim checkpoint source required")
        self.model_sources=model_sources
        for ref in self.model_sources:checked(ref)
        self.identity=manifest.evaluation_identity()
        self.manifest_sha256=digest(asdict(manifest))
        self.space=adapter.space;self.spec=adapter.spec

    def _validate(self,inputs,modes):
        if self.adapter.space!=self.space or self.adapter.spec!=self.spec or (
            getattr(self.adapter,"score_direction","higher_better" if self.spec.id=="irra" else None)!=self.direction):
            raise ArtifactError("victim ModelSpace/spec/direction changed")
        if type(inputs)is not dict or not modes or len(set(modes))!=len(modes) or any(m not in MODES for m in modes):
            raise ArtifactError("explicit unique evaluation modes and inputs")
        required={k for m in modes for k in MODES[m]}
        if set(inputs)!=required:raise ArtifactError("missing/extra representation family")
        if self.execution in ("historical_feature_replay","historical_fixture_replay"):
            from .historical_contracts import validate_replay_inputs
            validate_replay_inputs(self,inputs,modes)
            for ref in self.model_sources:checked(ref)
            return
        import hashlib
        originals={"image":tuple(g.image_sha256 for g in self.manifest.gallery),
                   "text":tuple(hashlib.sha256(q.caption.encode("utf-8")).hexdigest() for q in self.manifest.queries)}
        for family,b in inputs.items():
            if type(b)is not BoundRepresentation:raise ArtifactError("bound representation required")
            b.verify();modality="image" if family.endswith("gallery") else "text"
            ids=self.identity.gallery_ids if modality=="image" else self.identity.query_ids
            if b.inputs.model_space_id!=self.space.id or b.inputs.dataset_manifest_sha256!=self.manifest_sha256 or (
                b.inputs.ordered_ids!=ids or b.inputs.modality!=modality):
                raise ArtifactError("model/Tokenizer/manifest/order/modality cache conflict")
            if family.startswith("clean"):
                if b.origin is not None or b.inputs.ordered_sha256!=originals[modality]:
                    raise ArtifactError("clean content/source mismatch")
            else:
                if b.origin is None or b.original_sha256!=originals[modality]:
                    raise ArtifactError("adversarial attack/original input provenance conflict")
                budget=b.origin.budget
                if (modality=="image" and (budget.image is None or budget.text is not None)) or (
                    modality=="text" and (budget.text is None or budget.image is not None)):
                    raise ArtifactError("attack budget/modality provenance conflict")
        checked(self.manifest.annotation)
        for ref in self.model_sources:checked(ref)

    def run(self,inputs,*,modes=tuple(MODES),query_chunk=None,check_stop=None):
        self._validate(inputs,modes)
        if self.execution=="historical_feature_replay":
            if query_chunk!=128 or not self.adapter.allow_full_scoring:
                raise ArtifactError("real replay scoring disabled; requires Stage 4B-2 and chunk=128")
        if query_chunk is not None and (type(query_chunk)is not int or query_chunk<1):
            raise ArtifactError("positive query chunk size")
        callback=check_stop or (lambda:None)
        import torch
        results={};scores={};protocol={
            "schema":"tbps-evaluation-1","dataset_manifest":asdict(self.manifest),
            "manifest_sha256":self.manifest_sha256,"model_spec":asdict(self.adapter.spec),
            "model_space":asdict(self.space),"model_sources":[asdict(r) for r in self.model_sources],"direction":self.direction,"ranking":RANKING,
            "ranking_sha256":RANKING_ID,"execution":self.execution,"modes":list(modes),
            "query_chunk":query_chunk,"inputs":{k:v.descriptor() for k,v in inputs.items()},
            "score_feedback_to_attacks":False}
        # frozenset ModelSpec能力转为确定性JSON。
        protocol["model_spec"]["capabilities"]=sorted(self.adapter.spec.capabilities)
        for mode in modes:
            callback();q,g=(inputs[k] for k in MODES[mode]);parts=[]
            step=query_chunk or len(self.identity.query_ids)
            with torch.inference_mode():
                for start in range(0,len(self.identity.query_ids),step):
                    callback()
                    end=min(start+step,len(self.identity.query_ids))
                    if query_chunk is None:batch=q.batch
                    else:
                        if getattr(self.adapter,"query_independent_scores",False)is not True:
                            raise ArtifactError("native score has not declared query independence")
                        select=getattr(self.adapter,"slice_query",None)
                        if not callable(select):raise ArtifactError("native Adapter does not support query chunk selection")
                        batch=select(q.batch,start,end)
                        if type(batch)is not RepresentationBatch or batch.space_id!=self.space.id or batch.modality!="text":
                            raise ArtifactError("foreign native query slice")
                    value=self.adapter.score(batch,g.batch)
                    if type(value)is not ScoreBatch or value.space_id!=self.space.id or value.scoring_protocol_sha256!=self.space.score_id:
                        raise ArtifactError("foreign victim/scoring protocol")
                    x=value.payload
                    if not isinstance(x,torch.Tensor) or x.device.type!="cpu" or x.dtype!=torch.float32 or (
                        x.shape!=(end-start,len(self.identity.gallery_ids)) or x.requires_grad or not torch.isfinite(x).all()):
                        raise ArtifactError("native score shape/dtype/finite/direction protocol mismatch")
                    parts.append(x.detach().clone())
                matrix=torch.cat(parts,0)
            with torch.inference_mode(False),torch.no_grad():matrix=matrix.detach().clone()
            results[mode],order=ranking(matrix,self.identity,direction=self.direction)
            del order  # 不让上组完整int64排名留到下一组峰值内存。
            scores[mode]=matrix
        # 不把分数或身份提供给任何攻击接口；评测后再次验证输入未被原生评分器修改。
        self._validate(inputs,modes)
        if self.adapter.space!=self.space or self.adapter.spec!=self.spec:raise ArtifactError("victim model space changed")
        report={"schema":"tbps-evaluation-result-1","protocol_sha256":digest(protocol),"metric_unit":"percent",
                "groups":results,"drop_definition":"(Clean-Adv)/Clean*100; zero Clean => null",
                "drop_rate_percent":{k:drops(results["Clean"],v) for k,v in results.items()} if "Clean" in results else {},
                "scores_sha256":{k:payload_digest(v) for k,v in scores.items()},
                "identity_sha256":digest(asdict(self.identity)),"ranking_sha256":RANKING_ID,
                "gallery_count":len(self.identity.gallery_ids),"query_count":len(self.identity.query_ids)}
        if "Joint" in results and "Image-only" in results:
            report["Joint_minus_ImageOnly_mAP_pp"]=results["Joint"]["mAP"]-results["Image-only"]["mAP"]
        return protocol,report,scores

def publish_result(store,artifact_id,protocol,report,scores,*,sources=(),fault=None):
    """新框架原子发布score+指标+来源；失败pending保留，禁止覆盖或自动重算。"""
    if type(store)is not ArtifactStore:raise ArtifactError("new framework ArtifactStore required")
    if report["protocol_sha256"]!=digest(protocol) or set(scores)!=set(report["groups"]):
        raise ArtifactError("evaluation protocol/result mismatch")
    if report.get("metric_unit")!="percent" or report.get("ranking_sha256")!=RANKING_ID:
        raise ArtifactError("metric unit/ranking source conflict")
    if set(protocol["modes"])!=set(report["groups"]):raise ArtifactError("mode/result mismatch")
    for values in report["groups"].values():drops(values,values)
    if "Clean" in report["groups"]:
        expected={k:drops(report["groups"]["Clean"],v) for k,v in report["groups"].items()}
        if report["drop_rate_percent"]!=expected:raise ArtifactError("DR/result inconsistent")
    import torch
    files={"protocol.json":json_bytes(protocol),"metrics.json":json_bytes(report)}
    records={}
    for mode,value in scores.items():
        if payload_digest(value)!=report["scores_sha256"][mode]:raise ArtifactError("score payload changed")
        stream=io.BytesIO();torch.save(value.detach().cpu().clone(),stream)
        name="scores/"+mode+".pt";data=stream.getvalue();files[name]=data
        q,g=(protocol["inputs"][k] for k in MODES[mode])
        if protocol["execution"] in ("historical_feature_replay","historical_fixture_replay"):
            records[mode]={"kind":"historical_feature_replay_score","query_binding":q["binding"],
                "gallery_binding":g["binding"],"query_replay":q["identity"],"gallery_replay":g["identity"],
                "model_space":protocol["model_space"],"direction":protocol["direction"],
                "content":asdict(FileRef(str(store.root/artifact_id/name),hashlib.sha256(data).hexdigest())),
                "adversarial_pixel_sha256":"UNKNOWN","not_a_new_attack":True}
            continue
        artifact=ScoreArtifact(mode,protocol["model_space"]["checkpoint_sha256"],
            digest({"model_space":digest(protocol["model_space"]),
                    "native_scoring_protocol":protocol["model_space"]["scoring_protocol_sha256"]}),
            tuple(q["inputs"]["ordered_ids"]),tuple(g["inputs"]["ordered_ids"]),protocol["direction"],
            FileRef(str(store.root/artifact_id/name),hashlib.sha256(data).hexdigest()))
        record=ScoreRecord(artifact,InputBinding(**q["inputs"]),InputBinding(**g["inputs"]))
        records[mode]=asdict(record)
    files["score_records.json"]=json_bytes(records)
    # 生产者、annotation和特征/攻击来源全部显式绑定；不枚举整个新框架目录。
    refs={r.path:r for r in sources}
    annotation=FileRef(**protocol["dataset_manifest"]["annotation"]);refs[annotation.path]=annotation
    for v in protocol["model_sources"]:
        ref=FileRef(**v);refs[ref.path]=ref
    for bound in protocol["inputs"].values():
        for v in bound["cache_sources"]:
            ref=FileRef(**v);refs[ref.path]=ref
        if bound.get("kind")=="historical_feature_replay":
            identity=bound["identity"]
            for key in ("protocol","summary","completed","selection"):
                ref=FileRef(**identity[key]);refs[ref.path]=ref
            for _,v in identity["models"]:
                ref=FileRef(**v);refs[ref.path]=ref
        if bound["origin"]:
            for key in ("checkpoint","protocol"):
                ref=FileRef(**bound["origin"][key]);refs[ref.path]=ref
    for name in ("metrics.py","provenance.py","evaluation.py"):
        p=Path(__file__).with_name(name);refs[str(p)]=FileRef(str(p),sha_file(p))
    return store.publish(artifact_id,files,{"evaluation_protocol_sha256":digest(protocol),"ranking_sha256":RANKING_ID},
                         sources=tuple(refs.values()),fault=fault)


def load_result(store,artifact_id,*,expected_protocol_sha256):
    """只读新框架完成产物；先验证全部来源，再读取紧凑score，不重新排名。"""
    from ..artifacts import read_json
    store.verify(artifact_id,expected_provenance={"evaluation_protocol_sha256":expected_protocol_sha256,
                                                 "ranking_sha256":RANKING_ID})
    root=store.root/artifact_id
    protocol=read_json(root/"protocol.json","tbps-evaluation-1")
    report=read_json(root/"metrics.json","tbps-evaluation-result-1")
    if digest(protocol)!=expected_protocol_sha256 or report["protocol_sha256"]!=expected_protocol_sha256:
        raise ArtifactError("result protocol digest mismatch")
    import torch
    records=read_json(root/"score_records.json")
    if set(records)!=set(report["groups"]):raise ArtifactError("score record members mismatch")
    scores={}
    for mode in report["groups"]:
        if mode not in MODES:raise ArtifactError("unknown persisted mode")
        item=records[mode]
        if protocol["execution"] in ("historical_feature_replay","historical_fixture_replay"):
            q,g=(protocol["inputs"][k] for k in MODES[mode])
            if (item.get("kind")!="historical_feature_replay_score"
                or item["query_binding"]!=q["binding"] or item["gallery_binding"]!=g["binding"]
                or item["query_replay"]!=q["identity"] or item["gallery_replay"]!=g["identity"]
                or item["model_space"]!=protocol["model_space"] or item["direction"]!=protocol["direction"]
                or item["adversarial_pixel_sha256"]!="UNKNOWN" or item["not_a_new_attack"]is not True):
                raise ArtifactError("historical score source mismatch")
            content=FileRef(**item["content"]);checked(content)
            if Path(content.path)!=root/"scores"/(mode+".pt"):raise ArtifactError("historical score reference escape")
            raw=torch.load(content.path,map_location="cpu",weights_only=True)
            if not isinstance(raw,torch.Tensor):raise ArtifactError("invalid historical persisted score type")
            x=raw.detach().clone();del raw
            expected_shape=(len(protocol["dataset_manifest"]["queries"]),len(protocol["dataset_manifest"]["gallery"]))
            if (x.shape!=expected_shape or x.shape!=(report["query_count"],report["gallery_count"])
                or x.dtype!=torch.float32 or x.device.type!="cpu" or x.requires_grad or not torch.isfinite(x).all()):
                raise ArtifactError("invalid historical persisted score shape/dtype/finite")
            if payload_digest(x)!=report["scores_sha256"][mode]:raise ArtifactError("historical score payload mismatch")
            scores[mode]=x
            continue
        q,g=(protocol["inputs"][k]["inputs"] for k in MODES[mode])
        a=item["artifact"]
        if (a["query_order"]!=q["ordered_ids"] or a["gallery_order"]!=g["ordered_ids"]
            or item["query_inputs"]!=q or item["gallery_inputs"]!=g
            or a["direction"]!=protocol["direction"]
            or a["model_checkpoint_sha256"]!=protocol["model_space"]["checkpoint_sha256"]):
            raise ArtifactError("score record order/model/input binding mismatch")
        expected_score_id=digest({"model_space":digest(protocol["model_space"]),
                                  "native_scoring_protocol":protocol["model_space"]["scoring_protocol_sha256"]})
        if a["scoring_protocol_sha256"]!=expected_score_id:raise ArtifactError("foreign cached score protocol")
        content=FileRef(**a["content"]);checked(content)
        if Path(content.path)!=root/"scores"/(mode+".pt"):raise ArtifactError("score reference escapes artifact")
        x=torch.load(content.path,map_location="cpu",weights_only=True).detach().clone()
        if x.shape!=(report["query_count"],report["gallery_count"]) or x.dtype!=torch.float32 or x.device.type!="cpu" or not torch.isfinite(x).all():
            raise ArtifactError("invalid persisted score payload")
        if payload_digest(x)!=report["scores_sha256"][mode]:raise ArtifactError("persisted score digest mismatch")
        scores[mode]=x
    return protocol,report,scores
