"""G60/H60与B共用的原生IRRA四模式评估；攻击生成与评分分离，无Test训练反馈。"""
import argparse
import hashlib
from dataclasses import asdict
from pathlib import Path
from tbps.artifacts import ArtifactStore,FileRef,checked,digest,json_bytes,read_json,sha_file
from tbps.contracts import AttackBudget,ImageBudget,TextBudget
from tbps.models.base import InputBinding
from tbps.engines.materialized_cache import FeatureCache,CachedIRRA
from tbps.engines.provenance import AttackOrigin,payload_digest
from tbps.engines.evaluation import EvaluationEngine,publish_result
from .contracts import require
from .state import tree_sha
from .formal_state import tensor_bytes

MODES=("Clean","Image-only","Text-only","Joint")
METRICS=("R@1","R@5","R@10","mAP")
def ref(p):return FileRef(str(Path(p).resolve()),sha_file(p))
def utf8(text):return hashlib.sha256(text.encode("utf-8")).hexdigest()

def asr_at_r1(clean_scores,attack_scores,identity):
    """分母仅Clean Top1正确的Query；成功为同一Query攻击后Top1错误（全部同PID正例）。"""
    import torch
    require(clean_scores.shape==attack_scores.shape==(len(identity.query_ids),len(identity.gallery_ids)),"ASR matrix identity")
    require(clean_scores.dtype==attack_scores.dtype==torch.float32 and torch.isfinite(clean_scores).all()
        and torch.isfinite(attack_scores).all(),"ASR finite FP32")
    a=torch.argsort(clean_scores,dim=1,descending=True)[:,0]
    b=torch.argsort(attack_scores,dim=1,descending=True)[:,0]
    ca=torch.tensor([identity.gallery_pids[j]==identity.query_pids[i] for i,j in enumerate(a.tolist())])
    cb=torch.tensor([identity.gallery_pids[j]==identity.query_pids[i] for i,j in enumerate(b.tolist())])
    denominator=int(ca.sum());successes=int((ca&~cb).sum())
    return {"ASR@R1":100.*successes/denominator if denominator else None,
        "successes":successes,"denominator_clean_correct":denominator,"all_query_count":len(ca),
        "definition":"P(attack Top1 wrong | Clean Top1 correct); same-PID positives; not 100-R@1"}

def score_cached(manifest,store,protocol,space,checkpoint,bindings,origins,originals,output,*,chunk=128,fixture=False):
    cache=FeatureCache(store,protocol,space)
    families={k:cache.gather(k,b,origin=origins.get(k),
        originals=originals[b.modality] if k.startswith("adv") else ()) for k,b in bindings.items()}
    adapter=CachedIRRA(space,checkpoint,fixture=fixture,cache_protocol=ref(store.root/"protocol/manifest.json"))
    execution="cpu_fixture" if fixture else "readonly_cached"
    engine=EvaluationEngine(manifest,adapter,direction="higher_better",execution=execution,model_sources=(checkpoint,))
    ep,result,scores=engine.run(families,modes=MODES,query_chunk=chunk)
    publish_result(output,"engine",ep,result,scores)
    groups={mode:{k:result["groups"][mode][k] for k in METRICS} for mode in MODES}
    asr={mode:asr_at_r1(scores["Clean"],scores[mode],manifest.evaluation_identity()) for mode in MODES[1:]}
    public={"schema":"gh-b-comparison-1","groups":groups,"ASR":asr,
        "gallery":len(manifest.gallery),"query":len(manifest.queries),"Raw_Clean":True,
        "Engine_protocol":digest(ep),"Test_feedback_to_training":False,"checkpoint":asdict(checkpoint)}
    header="Attack Method,"+",".join(METRICS)+",ASR@R1\n"
    def csv(rounded):
        out=header
        for mode in MODES:
            v=[groups[mode][k] for k in METRICS]+[asr[mode]["ASR@R1"] if mode!="Clean" else None]
            out+=mode+","+",".join("" if x is None else format(x,".2f") if rounded else repr(x) for x in v)+"\n"
        return out.encode()
    output.publish("completed",{"summary.json":json_bytes(public),"metrics.csv":csv(False),
        "metrics_2dp.csv":csv(True)},{"Engine_protocol":digest(ep)},sources=(ref(output.root/"engine/manifest.json"),))
    return public

def select_weights(bundle,method,baseline_ref,round_checkpoint=None):
    """同一G(I)/H(T)推理代码，仅选择经过来源验证的模型状态。"""
    import torch
    if method=="baseline":
        checked(baseline_ref);value=torch.load(baseline_ref.path,map_location="cpu",weights_only=False)
        require(value["cursor"]==122880 and value["updates"]==122220 and value["reason"]=="epoch_complete",
                "true adapted H60 required; not H7/H10")
        require(value["protocol"]["parent_checkpoint_sha256"]==bundle.state["parents"]["H"],"H60 parent identity")
        bundle.h.load_state_dict(value["policy"],strict=True)
        return None,baseline_ref
    require(method=="B" and round_checkpoint is not None,"explicit committed B checkpoint")
    checked(round_checkpoint)
    p=Path(round_checkpoint.path);s=ArtifactStore(p.parent.parent,new_base=p.parents[3]);m=s.verify(p.parent.name)
    value=torch.load(p,map_location="cpu",weights_only=False)
    require(m["provenance"]["state"]==tree_sha(value) and value["parents"]==bundle.state["parents"]
        and value["cursor"]["round"]>=1,"B checkpoint source/content")
    bundle.g.load_state_dict(value["G"],strict=True);bundle.h.load_state_dict(value["H"],strict=True)
    return round_checkpoint,round_checkpoint

def materialize(manifest,bundle,c,root):
    """G一次前向/图；H仅接原文Canonical。绝不把Adv RGB转回PIL/JPEG。"""
    import torch
    from PIL import Image
    from .native_backend import assert_pixels
    from .parents import no_optimizer_steps
    store=ArtifactStore(root,new_base=c["output_base"])
    require(not Path(root).exists(),"immutable attack inference Attempt")
    store.publish("claim",{"dataset_manifest.json":json_bytes(asdict(manifest))},{"native_source":digest(c)})
    before=(tree_sha(bundle.g.state_dict()),tree_sha(bundle.h.state_dict()))
    with no_optimizer_steps(),torch.no_grad():
        for j,g in enumerate(manifest.gallery):
            store.publish(f"image_claim_{j:04d}",{"claim.json":json_bytes({"id":g.image_id})},{"id":g.image_id})
            path=Path(c["prototype"]["image_root"])/g.image_rel_path
            checked(FileRef(str(path),g.image_sha256))
            with Image.open(path) as im:original=bundle.bindings.get("irra_image_adapter").pixels_from_pil(im).unsqueeze(0).to("cuda:0")
            value=bundle.bindings.get("inference_once").generate_once(bundle.g,original)
            assert_pixels(original,value["adversarial"],value["effective_delta"])
            data={k:original.cpu().clone() if k=="original" else value[k].cpu().clone()
                for k in ("original","adversarial","effective_delta")}
            store.publish(f"image_{j:04d}",{"data.pt":tensor_bytes(data)},
                {"id":g.image_id,"original_file":g.image_sha256,"adversarial_rgb":payload_digest(data["adversarial"])})
            print("B_EVAL_IMAGE",j+1,len(manifest.gallery),flush=True)
        for j,q in enumerate(manifest.queries):
            store.publish(f"text_claim_{j:04d}",{"claim.json":json_bytes({"id":q.query_id})},{"id":q.query_id})
            example,audit=bundle.frontend.convert(q.caption)
            out=bundle.h.attack(example)
            require(bundle.bindings.get("constraints").replay(example,out.edits,out.budget_k)==out.adversarial,"Unicode action replay")
            store.publish(f"text_{j:04d}",{"action.json":json_bytes({"original":q.caption,"adversarial":out.adversarial,
                "native_action":out.record(),"frontend":audit})},
                {"id":q.query_id,"original_utf8":utf8(q.caption),"adversarial_utf8":utf8(out.adversarial)})
            print("B_EVAL_TEXT",j+1,len(manifest.queries),flush=True)
    require(before==(tree_sha(bundle.g.state_dict()),tree_sha(bundle.h.state_dict())),"inference changed G/H")
    store.publish("completed",{"report.json":json_bytes({"images":len(manifest.gallery),"texts":len(manifest.queries),
        "G_forwards":len(manifest.gallery),"Adam_steps":0,"paired_image_used_by_H":False,"models_frozen":True})},{"native_source":digest(c)})
    return store

def encode_and_score(manifest,c,attack,root,gref,href):
    """同一原生IRRA Adapter编码，紧凑缓存后由原EvaluationEngine全排名。"""
    import torch
    from tbps.models.irra import IRRAAdapter
    from tbps.models.irra_source import IRRASource
    source=IRRASource.from_dict(read_json(c["sources"]["dataset_config"]["path"])["irra"])
    adapter=IRRAAdapter.from_source(source,device="cuda:0",allow_model_load=True)
    before=tree_sha(adapter._backend.state_dict());space=adapter.space;view=adapter.inference()
    features=ArtifactStore(Path(root)/"features",new_base=c["output_base"])
    protocol={"schema":"tbps-materialized-irra-cache-1","model_space":asdict(space),
        "G_H_generation":False,"ranking_feedback_to_attack":False,"manifest":asdict(manifest),
        "attack_completion":asdict(ref(attack.root/"completed/manifest.json"))}
    features.publish("protocol",{"protocol.json":json_bytes(protocol)},{"model_space":space.id},sources=(source.checkpoint,))
    cache=FeatureCache(features,protocol,space)
    identity=manifest.evaluation_identity();mh=digest(asdict(manifest))
    originals={"image":tuple(g.image_sha256 for g in manifest.gallery),"text":tuple(utf8(q.caption) for q in manifest.queries)}
    imd=[attack.verify(f"image_{j:04d}")["provenance"] for j in range(len(manifest.gallery))]
    txd=[attack.verify(f"text_{j:04d}")["provenance"] for j in range(len(manifest.queries))]
    bindings={
        "clean_gallery":InputBinding(space.id,mh,originals["image"],"image",identity.gallery_ids),
        "adv_gallery":InputBinding(space.id,mh,tuple(m["adversarial_rgb"] for m in imd),"image",identity.gallery_ids),
        "clean_text":InputBinding(space.id,mh,originals["text"],"text",identity.query_ids),
        "adv_text":InputBinding(space.id,mh,tuple(m["adversarial_utf8"] for m in txd),"text",identity.query_ids)}
    with torch.no_grad():
        for family,b in bindings.items():
            for start in range(0,len(b.ordered_ids),64):
                rows=[];end=min(start+64,len(b.ordered_ids))
                for j in range(start,end):
                    if family.endswith("gallery"):
                        raw=torch.load(attack.root/f"image_{j:04d}/data.pt",map_location="cpu",weights_only=True)
                        x=raw["original" if family=="clean_gallery" else "adversarial"]
                        value=view.encode_images(x.to("cuda:0")).payload
                    else:
                        text=read_json(attack.root/f"text_{j:04d}/action.json")["original" if family=="clean_text" else "adversarial"]
                        value,_=view.encode_texts((text,));value=value.payload
                    rows.append(value.detach().cpu().clone())
                cache.publish(family,b,start,end,torch.cat(rows));print("B_IRRA_ENCODE",family,end,len(b.ordered_ids),flush=True)
    require(before==tree_sha(adapter._backend.state_dict()),"frozen IRRA changed")
    protocol_ref=ref(c["_config_path"])
    origins={"adv_gallery":AttackOrigin("gv2",gref,protocol_ref,AttackBudget(image=ImageBudget("linf",8/255,"RGB [0,1]")),digest(c["sources"])),
        "adv_text":AttackOrigin("hv1",href,protocol_ref,AttackBudget(text=TextBudget("unicode_character",None,
            "original Natural-K gamma3/beta1","original H-v1 constraints",1,1)),digest(c["sources"]))}
    return score_cached(manifest,features,protocol,space,source.checkpoint,bindings,origins,originals,
        ArtifactStore(Path(root)/"results",new_base=c["output_base"]))

def comparison_checkpoint(f,method,provided):
    if method=="baseline":
        require(provided is None,"baseline cannot substitute B checkpoint")
        return None
    require(method=="B" and "comparison_B_checkpoint" in f["references"],"precommitted final B endpoint required")
    locked=FileRef(**f["references"]["comparison_B_checkpoint"])
    require(provided is None or provided==locked,"B endpoint differs from preregistered final Round")
    return locked

def run(path,method,*,allow=False,round_checkpoint=None):
    from .formal_sources import prepare_cpu
    f,c,bundle,gate,pre=prepare_cpu(path,"train")
    require(allow and f["evaluation"]["authorized"] is True,"Test evaluation not authorized in this preparation protocol")
    round_checkpoint=comparison_checkpoint(f,method,round_checkpoint)
    from tbps.datasets.cuhk_pedes import CuhkPedes
    manifest=CuhkPedes(FileRef(c["prototype"]["annotation"],sha_file(c["prototype"]["annotation"])),
                      c["prototype"]["image_root"]).manifest("test",expected_counts=(3074,6156))
    from .preflight import resource_snapshot
    require(resource_snapshot(c["prototype"])["resource_eligible"],"RESOURCE_BUSY")
    root=Path(f["evaluation"]["output_base"])/method
    require(not root.exists(),"evaluation Attempt already consumed")
    baseline=FileRef(**f["references"]["H60"])
    gref,href=select_weights(bundle,method,baseline,round_checkpoint)
    if gref is None:gref=FileRef(**c["sources"]["G60"])
    from .native_sources import native_session
    from tbps.models.irra import fp32_runtime
    import torch
    store=ArtifactStore(root,new_base=c["output_base"])
    store.publish("claim",{"preflight.json":json_bytes(pre)},{"method":method,"config":sha_file(path)})
    try:
        gate.initialize_cuda()
        with fp32_runtime():
            with native_session(c,reuse=bundle.modules):
                bundle.g.to("cuda:0").eval().requires_grad_(False);bundle.h.eval().requires_grad_(False)
                attack=materialize(manifest,bundle,c,root/"attacks")
            report=encode_and_score(manifest,c,attack,root,gref,href)
        print(report)
    except BaseException:
        import traceback
        with (root/"FAIL").open("x") as out:out.write(traceback.format_exc())
        raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True)
    ap.add_argument("--method",choices=("baseline","B"),required=True)
    ap.add_argument("--round-checkpoint");ap.add_argument("--allow-test-evaluation",action="store_true")
    a=ap.parse_args()
    run(a.config,a.method,allow=a.allow_test_evaluation,round_checkpoint=ref(a.round_checkpoint) if a.round_checkpoint else None)
if __name__=="__main__":main()
