"""官方DRA只读来源、数据映射和被动运行追踪；不含攻击更新或候选决策。"""
from pathlib import Path
from dataclasses import replace
import ast
import importlib.util
import subprocess
import sys
from collections import Counter
from contextlib import contextmanager
from tbps.artifacts import ArtifactError, FileRef, checked, sha_file
from tbps.datasets.cuhk_pedes import CuhkPedes
from tbps.baselines.sga_official_io import load_file, source_pixels
from tbps.engines.provenance import payload_digest

COMMIT = "91f78325e7cf0c497c9bbb5be2215be652d8cd9d"
CORE_SHA = "75518cb9465d9f8e994ea2f59c86b89fd4c39e3fe74f66b7a1d0a090eec2d313"
PARAMETERS = dict(epsilon=8/255, steps=10, step_size=2/255, sample_numbers=5,
                  text_ratios=[.6,.2,.2], batch_size=8, seed=42, max_length=77,
                  pre_caption_words=30, cls=False, number_perturbation=1,
                  topk=10, threshold_pred_score=.3, scales=[.5,.75,1.25,1.5])
MODES = ("Clean", "Text-only", "Image-only", "Joint")

def need(ok, message):
    if not ok:
        raise ArtifactError(message)

def verify_official(c):
    root=Path(c["official_root"])
    need(subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()==COMMIT, "official DRA Commit")
    need(not subprocess.check_output(["git","status","--porcelain"],cwd=root,text=True), "official DRA modified")
    need(c["official_commit"]==COMMIT and c["official"]["attacker"]["sha256"]==CORE_SHA, "locked DRA source")
    for x in c["official"].values():
        checked(FileRef(**x))
    for path, value in c["official_files"].items():
        checked(FileRef(path,value))
    return {"commit":COMMIT,"RAttacker_sha256":CORE_SHA,"git_diff":""}

def modules(c):
    verify_official(c)
    return tuple(load_file(c["official"][key],"_dra_"+key) for key in ("attacker","tokenizer","dataset"))

def ordered_dataset(c, scope):
    ds=CuhkPedes(FileRef(**c["annotation"]),c["image_root"])
    full=ds.manifest("test",expected_counts=(3074,6156))
    if scope=="smoke":
        gallery=full.gallery[:8]
        ids={x.image_id for x in gallery}
        queries=tuple(q for q in full.queries if q.paired_image_id in ids)
        raw=replace(full,gallery=gallery,queries=queries)
        need([x.image_id for x in gallery]==c["smoke_gallery_ids"] and
             [q.query_id for q in queries]==c["smoke_query_ids"],"fixed first-eight and all Caption")
    elif scope=="full":
        raw=full
    else:
        raise ArtifactError("unknown DRA scope")
    # Stateless original pre_caption; no model or attacker import needed for data preparation.
    dataset=load_file(c["official"]["dataset"],"_dra_dataset")
    matched=replace(raw,queries=tuple(replace(q,caption=dataset.pre_caption(q.caption,30)) for q in raw.queries))
    return ds,raw,matched

def batch_members(manifest,start,batch_size=8):
    gallery=manifest.gallery[start:start+batch_size]
    ids=[g.image_id for g in gallery]
    queries=tuple(q for q in manifest.queries if q.paired_image_id in ids)
    mapping=[ids.index(q.paired_image_id) for q in queries]
    need(len(set(q.query_id for q in queries))==len(queries),"duplicate Query")
    return gallery,queries,mapping

def projection_check(clean,adv):
    import torch
    eps=8/255
    need(clean.shape==adv.shape and adv.dtype==torch.float32 and torch.isfinite(adv).all(),"DRA output shape/dtype/finite")
    need((adv>=(clean-eps).clamp(0,1)).all() and (adv<=(clean+eps).clamp(0,1)).all(),"official FP32 projection")
    actual=float((adv.double()-clean.double()).abs().max())
    return dict(official_FP32_bounds=True,real_linf=actual,mathematical_epsilon=eps,
                roundoff_above_real_epsilon=max(0.,actual-eps),extra_projection=False)

class PassiveTrace:
    """sys.settrace只读官方frame；不替换任何函数、返回值、Tensor或随机状态。"""
    def __init__(self,path,source=None,write=None):
        self.path=str(Path(path).resolve());self.source=source
        self.write=write or (lambda row:None)
        tree=ast.parse(Path(path).read_text())
        self.lines={}
        for node in ast.walk(tree):
            if isinstance(node,ast.FunctionDef):
                for part in ast.walk(node):
                    if isinstance(part,ast.Call) and isinstance(part.func,ast.Attribute) and part.func.attr=="backward":
                        self.lines[part.lineno]="backward"
                    if isinstance(part,ast.Assign):
                        names=[t.id for t in part.targets if isinstance(t,ast.Name)]

                        if "perturbation" in names:self.lines[part.lineno]="gradient"
        for i,line in enumerate(Path(path).read_text().splitlines(),1):
            if "ratio_list.append(" in line:self.lines[i]="selected_ratio"
            if "final_adverse.append(" in line:self.lines[i]="text_action"
        self.counts=Counter();self.events=[];self.text_stage=0;self.T1=None
        self.selected=[];self.text_actions=[];self.gradient_finite=True;self.gradient_norms=[]
    def __call__(self,frame,event,arg):
        if str(Path(frame.f_code.co_filename).resolve())!=self.path:
            return None
        name=frame.f_code.co_name
        if event=="call":
            self.counts["call:"+name]+=1
            if name=="img_guided_attack":
                self.text_stage+=1
                self.events.append("text1" if self.text_stage==1 else "text2")
                loc=frame.f_locals
                need(self.text_stage in (1,2),"unexpected official text call")
                self.write({"event":self.events[-1],"texts":list(loc["texts"]),
                            "image_supervision":{k:payload_digest(loc[k].detach().cpu()) for k in
                            ("img_embeds","adv_img_embeds","last_adv_img_embeds") if loc.get(k) is not None}})
            elif name=="txt_guided_attack":
                self.events.append("image")
                if self.source is not None:
                    import torch
                    need(self.source.last_text_requests==self.T1,"T1 was not actual image guidance")
                    need(torch.equal(self.source.last_text[1],frame.f_locals["txt_embeds"].detach().cpu()),"T1 feature binding")
                self.write({"event":"image","T1":self.T1})
        elif event=="line":
            kind=self.lines.get(frame.f_lineno)
            if kind=="text_action" and name=="img_guided_attack":
                loc=frame.f_locals
                row={"event":"text_action","stage":self.text_stage,"query_local_index":int(loc["i"]),
                     "official_change_count":int(loc["change"]),"original_word_slots":list(loc["words"]),
                     "final_word_slots":list(loc["final_words"])}
                self.text_actions.append(row);self.write(row)
            elif kind=="backward":self.counts["backward"]+=1
            elif kind=="gradient" and "grad" in frame.f_locals:
                import torch
                grad=frame.f_locals["grad"]
                self.gradient_finite &= bool(torch.isfinite(grad).all())
                self.gradient_norms.append(float(grad.detach().norm()))
                self.counts["gradient"]+=1
            elif kind=="selected_ratio" and "candidate_index" in frame.f_locals and name=="txt_guided_attack":
                loc=frame.f_locals
                row={"event":"selected_ratio","step":int(loc["step"]),
                     "samples":[list(x) for x in loc["samples"]],"losses":list(loc["loss_list"]),
                     "selected_index":int(loc["candidate_index"])}
                import torch
                if torch.cuda.is_initialized():
                    row["GPU_allocated_bytes"]=torch.cuda.memory_allocated()
                    row["GPU_reserved_bytes"]=torch.cuda.memory_reserved()
                self.selected.append(row);self.write(row)
        elif event=="return" and name=="img_guided_attack" and arg is not None:
            if self.text_stage==1:self.T1=list(arg)
            self.write({"event":"text_return","stage":self.text_stage,"texts":list(arg)})
        return self
    @contextmanager
    def active(self):
        need(sys.gettrace() is None,"existing tracer; no replacement")
        before=sha_file(self.path)
        sys.settrace(self)
        try:
            yield self
        finally:
            sys.settrace(None)
            need(sha_file(self.path)==before,"official source changed")
    def report(self):
        return {"events":self.events,"counts":dict(self.counts),"selected_ratios":self.selected,"text_actions":self.text_actions,
                "all_image_gradients_finite":self.gradient_finite,"gradient_norms":self.gradient_norms,"T1":self.T1,
                "observer":"passive sys.settrace; no monkey patch"}
