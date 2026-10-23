"""真实后端薄接线：原训练数学、原定位/JVP、原RewardBank；模块导入不加载模型。"""
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import io
from tbps.artifacts import ArtifactStore, FileRef, checked, digest, read_json, json_bytes
from .contracts import Anchor, ConditionIdentity, BankIdentity, TextTrace, require, text_sha
from .conditions import PreparedCondition, CanonicalLocator, LegacyContextKernel, LegacySteps, prepare_condition, trace_from_legacy
from .state import tree_sha, clone_tree, restore, restore_rng, capture_rng, snapshot
from .native_sources import native_session, source_key
from .token_bridge import ByteSpanBridge, token_audit

def native_image_view(native,model):
    # 原wrapper的mean/std默认创建于CPU，必须连同模型放到同一设备。
    return native.get("irra_image_adapter").FrozenIRRAImageEncoder(model).to(next(model.parameters()).device)

def tensor_bytes(value):
    import torch
    stream=io.BytesIO();torch.save(clone_tree(value),stream);return stream.getvalue()

def assert_pixels(clean,adv,delta):
    import torch
    require(clean.shape==adv.shape==delta.shape==(1,3,384,128), "native RGB shape")
    require(all(x.dtype==torch.float32 and torch.isfinite(x).all() for x in (clean,adv,delta)), "FP32 finite RGB")
    require(torch.equal(adv-clean,delta) and float(delta.double().abs().max())<=8/255
            and float(adv.min())>=0 and float(adv.max())<=1, "strict original pixel budget")

class TextVictim:
    def __init__(self,model,probe):
        self.model,self.probe=model,probe
    def encode_texts(self,texts):
        import torch
        from torch.nn import functional as F
        for text in texts:token_audit(self.probe,text)
        device=next(self.model.parameters()).device
        tokens=torch.tensor([self.probe.inspect(t).model_tokens for t in texts],dtype=torch.long,device=device)
        return F.normalize(self.model.encode_text(tokens).float(),dim=1)

class TextRuntime:
    """H原FreshReward视图；其计数不包含G图库刷新和JVP阶段的模型调用。"""
    def __init__(self,backend):
        self.backend=backend;self.probe=backend.bundle.probe
        self.samples={"G":0,"clean":0,"adv":0,"text":0}
    def assert_frozen(self):self.backend.assert_frozen_sources()
    def encode_text(self,texts):
        import torch
        with torch.inference_mode():
            x=self.backend.victim.encode_texts(texts).detach().cpu()
        x=x.clone()
        require(x.dtype==torch.float32 and x.shape==(len(texts),512) and torch.isfinite(x).all(), "H text features")
        self.last_texts=tuple(texts);self.last_features=x
        self.samples["text"]+=len(texts)
        return x

class NativeBackend:
    execution_kind="native_irra_backend"
    def __init__(self,c,bundle,grant,adapter):
        import torch
        self.c,self.p,self.bundle,self.grant,self.adapter=c,c["prototype"],bundle,grant,adapter
        grant.validate(c,grant.action)
        self.g,self.h,self.ga,self.ha=bundle.g,bundle.h,bundle.g_adam,bundle.h_adam
        self.sampler=bundle.sampler;self.native=bundle.bindings
        self.counts={"G":0,"IRRA_image":0,"IRRA_text":0,"reader":0,"H":0,"JVP":0}
        self.root=Path(c[grant.action+"_output"])
        self.store=ArtifactStore(self.root,new_base=c["output_base"])
        self.irra=adapter._backend
        self.victim=TextVictim(self.irra,bundle.probe)
        self.encoder=native_image_view(self.native,self.irra)
        checked(FileRef(**c["sources"]["reader"]))
        rv=torch.load(c["sources"]["reader"]["path"],map_location="cpu",weights_only=False)
        self.split=read_json(c["sources"]["reader_split"]["path"])
        require(rv["epoch"]==20 and rv["slots"]==self.split["slots"]
                and rv["values"]==self.split["values"],"frozen reader source")
        self.reader=self.native.get("learned_reader").SlotReader([len(values) for values in self.split["values"]])
        self.reader.load_state_dict(rv["state_dict"],strict=True)
        self.reader=self.reader.float().eval().requires_grad_(False).to("cuda")
        self.bases=read_json(c["sources"]["basis"]["path"])
        self.bridge=ByteSpanBridge(self.native.get("attributes.irra_token_alignment"),bundle.probe.tokenizer)
        cached=SimpleNamespace(tokenizer=bundle.probe.tokenizer,align=self.bridge.align,span_tokens=self.bridge.span_tokens)
        self.localizer=self.native.get("importance").FrozenLocalization(self.reader,self.split,self.bases,cached)
        self.steps=LegacySteps(self.native.get("serial_training").train_single_sample,
                               self.native.get("phase4_training").train_step)
        entries=read_json(Path(self.p["history_root"])/"qualification/train.json")["entries"]
        self.entries={x["identity"]["row_id"]:x for x in entries}
        gm=read_json(self.p["g_manifest"])["entries"];self.gentries={x["row_id"]:x for x in gm if x["role"]=="train"}
        self.original={};self.plans={};self.instances={}
        for row in self.p["samples"]:
            e=self.gentries[row]
            checked(FileRef(e["tensor_path"],e["tensor_sha256"]));checked(FileRef(e["metadata_path"],e["metadata_sha256"]))
            value=torch.load(e["tensor_path"],map_location="cpu",weights_only=True)
            self.native.get("condition_schema").validate(value)
            md=read_json(e["metadata_path"]);selected=md["source_selection"]["selected_attributes"]
            identity=ConditionIdentity(text_sha(e["caption"]),text_sha(e["caption"]),
                tuple(value["selected_instance_ids"]),tuple(value["S_raw"].tolist()),
                self.c["condition_space_sha256"],digest([asdict(Anchor(**{k:a[k] for k in Anchor.__dataclass_fields__})) for a in selected]))
            self.original[row]=PreparedCondition(identity,value,{"mode":"original_frozen","tensor_sha256":e["tensor_sha256"]})
            self.plans[row]=self.native.get("paired_prefix").prepare_uncapped(md["source_selection"],md["source_score_record"],self.bases,"cuda")
            cls=self.native.get("attribute_interface").AttributeInstance
            self.instances[row]=tuple(cls(**{k:a[k] for k in cls.__dataclass_fields__ if k in a
                and k!="semantic_representation"}) for a in selected)
        from .parents import bind_g_device
        self.g,self.ga=bind_g_device(bundle,"cuda")
        require(tree_sha(self.ga.state_dict())==tree_sha(bundle.state["G_Adam"]), "GPU Adam transfer changed state")
        self.irra_before=tree_sha(self.irra.state_dict());self.reader_before=tree_sha(self.reader.state_dict())
        self.versions=self._versions()
        self.h.requires_grad_(grant.action=="smoke")
        self.g.requires_grad_(True).eval()
        self.h.eval()
        self._hooks=[]
        def hook(key):
            def count(module,args):
                self.counts[key]+=1 if key in ("reader","H") else len(args[0])
                limits=(c["probe"] if grant.action=="probe" else c["smoke"])
                caps={"G":limits.get("max_G_forwards"),"IRRA_image":limits.get("max_IRRA_images"),
                      "IRRA_text":limits.get("max_IRRA_texts"),"H":limits.get("max_H_forwards"),
                      "reader":limits.get("max_reader_forwards")}
                require(caps[key] is None or self.counts[key]<=caps[key], "neural forward budget: "+key)
            return count
        self._hooks.append(self.g.register_forward_pre_hook(hook("G")))
        self._hooks.append(self.h.register_forward_pre_hook(hook("H")))
        self._hooks.append(self.reader.register_forward_pre_hook(hook("reader")))
        self._original_image,self._original_text=self.irra.encode_image,self.irra.encode_text
        def image(x):
            self.counts["IRRA_image"]+=len(x)
            cap=(c["probe"] if grant.action=="probe" else c["smoke"])["max_IRRA_images"]
            require(self.counts["IRRA_image"]<=cap,"IRRA image budget")
            return self._original_image(x)
        def text(x):
            self.counts["IRRA_text"]+=len(x)
            cap=(c["probe"] if grant.action=="probe" else c["smoke"]).get("max_IRRA_texts")
            require(cap is None or self.counts["IRRA_text"]<=cap,"IRRA text budget")
            return self._original_text(x)
        self.irra.encode_image=image;self.irra.encode_text=text
        self.runtime=TextRuntime(self)
        self.cursor=clone_tree(bundle.state["cursor"])
        self.rng={"G":clone_tree(bundle.state["G_RNG"]),"H":clone_tree(bundle.state["H_RNG"])}
        self.conditions_log=[];self.rollout_log=[]

    def _versions(self):
        return [(id(x),x._version) for m in (self.irra,self.reader) for x in (*m.parameters(),*m.buffers())]
    def assert_frozen_sources(self):
        require(self._versions()==self.versions,"frozen source tensor changed")
        require(all(not m.training for source in (self.irra,self.reader) for m in source.modules())
                and all(not p.requires_grad and p.grad is None for source in (self.irra,self.reader) for p in source.parameters()),
                "IRRA/reader freeze violation")
    def final_frozen(self):
        self.assert_frozen_sources()
        require(tree_sha(self.irra.state_dict())==self.irra_before and tree_sha(self.reader.state_dict())==self.reader_before,
                "IRRA/reader final state SHA changed")
        return True
    def close(self):
        self.irra.encode_image=self._original_image;self.irra.encode_text=self._original_text
        for h in self._hooks:h.remove()
    def g_fingerprint(self):return tree_sha(self.g.state_dict())
    def h_fingerprint(self):return tree_sha(self.h.state_dict())
    def restore_complete(self,state):
        self.grant.validate(self.c,"smoke")
        if state["cursor"]["round"]==0 and hasattr(self,"initial_features"):
            self.current_features=self.initial_features.clone()
        self.rng["G"],self.rng["H"],self.cursor=restore(state,self.g,self.h,self.ga,self.ha,expected_parents=self.bundle.state["parents"])
    def enter_phase_rng(self,name):
        self.g.requires_grad_(name=="G");self.h.requires_grad_(name=="H")
        self.g.zero_grad(set_to_none=True);self.h.zero_grad(set_to_none=True)
        restore_rng(self.rng[name],self.sampler,allow_cuda=True)
    def leave_phase_rng(self,name):self.rng[name]=capture_rng(self.sampler)
    def public_text(self,row):
        e=self.entries[row]
        return {"caption":e["caption"],"attributes":clone_tree(e["attributes"])}
    def h_rollout(self,public):
        require(set(public)=={"caption","attributes"},"caption-only H input")
        example=self.native.get("contracts").from_text_spans(public["caption"],public["attributes"])
        eligible=any(a for a in example.attributes) and not self.bundle.probe.inspect(example.caption).truncated
        # 固定不合格行仅根据预锁定Caption资格判断；不据图库或反馈选择。
        matching=[e for e in self.entries.values() if e["caption"]==example.caption]
        require(matching and all(e["attributes"]==public["attributes"] for e in matching),"public frontend binding")
        eligible=eligible and all(e["eligible"] for e in matching)
        if not eligible:
            trace=TextTrace(example.caption,example.caption,tuple(Anchor(**a) for a in public["attributes"]),(),0,
                            digest({"fixed_NOOP":public,"reason":"locked caption qualification"}))
        else:
            out=self.h.attack(example)
            trace=trace_from_legacy(example,out.adversarial,out.edits,out.budget_k,out.record(),
                                   replay=self.native.get("constraints").replay)
        pairs={(a,b) for a,bs in self.native.get("constraints").CONFUSABLES.items() for b in bs}
        self.rollout_log.append({"public_input":public,"original":trace.original,"adversarial":trace.adversarial,
                                 "edits":[asdict(e) for e in trace.edits],"K":trace.budget_k,
                                 "native_action":out.record() if eligible else {"fixed_NOOP":True}})
        trace.validate(pairs)
        self.store.publish(f"rollout_{len(self.rollout_log)-1:04d}",
            {"action.json":json_bytes(self.rollout_log[-1])},{"source":source_key(self.c),"action":trace.action_sha256})
        return trace
    def condition(self,arm,row,trace):
        original=self.original[row]
        anchors=tuple(Anchor(**{k:getattr(a,k) for k in Anchor.__dataclass_fields__}) for a in self.instances[row])
        locator=CanonicalLocator(trace.original,self.instances[row],self.plans[row],self.localizer,
                                 self.native.get("attribute_tta").SelectedAttribute)
        def transport(caption):return self.native.get("attribute_tta").TextTransport(self.victim,caption)
        def locate(caption,H,mappings,source):
            plan,audit=locator(caption,H,mappings,source)
            audit.update(token_audit(self.bundle.probe,caption))
            audit["context_caption_sha256"]=text_sha(caption)
            return plan,audit
        kernel=LegacyContextKernel(transport_factory=transport,locate_and_plan=locate,
            build=self.native.get("condition_schema").build,source_space_sha256=self.c["condition_space_sha256"],
            text_counter=lambda:self.counts["IRRA_text"])
        pairs={(a,b) for a,bs in self.native.get("constraints").CONFUSABLES.items() for b in bs}
        limit=(self.c["probe"] if self.grant.action=="probe" else self.c["smoke"])["max_JVPs"]
        expected=len(original.identity.selected_g_ids) if arm=="B" and trace.original!=trace.adversarial else 0
        require(self.counts["JVP"]+expected<=limit,"JVP budget before computation")
        out=prepare_condition(arm,trace,original,anchors,legal_pairs=pairs,kernel=kernel)
        self.counts["JVP"]+=out.evidence["new_JVPs"]
        limit=(self.c["probe"] if self.grant.action=="probe" else self.c["smoke"])["max_JVPs"]
        require(self.counts["JVP"]<=limit,"JVP budget")
        evidence={"row":row,"identity":asdict(out.identity),"evidence":out.evidence,
                  "payload_content_sha256":tree_sha(out.payload)}
        self.store.publish(f"condition_{len(self.conditions_log):04d}",
            {"condition.pt":tensor_bytes(out.payload),"evidence.json":json_bytes(evidence)},
            {"source":source_key(self.c),"condition":out.identity.key},
            sources=(FileRef(self.gentries[row]["tensor_path"],self.gentries[row]["tensor_sha256"]),))
        self.conditions_log.append(evidence)
        return out
    def image(self,row):
        from PIL import Image
        e=self.entries[row]["identity"]
        path=Path(self.p["image_root"])/e["image_rel_path"]
        checked(FileRef(str(path),e["image_sha256"]))
        with Image.open(path) as im:
            x=self.native.get("irra_image_adapter").pixels_from_pil(im).unsqueeze(0)
        return x.to("cuda")
    def g_update(self,row,condition):
        self.grant.validate(self.c,"smoke")
        self.g.requires_grad_(True)
        gradients=[]
        before=tree_sha(self.g.state_dict())
        def observe(event):
            # 观察原训练器已有梯度；不持有图像或计算图，不额外反向。
            gradients.append({"record":event["record"],"gradient_sha256":tree_sha(
                {n:p.grad for n,p in self.g.named_parameters() if p.grad is not None})})
        result=self.steps.update_g(self.g,self.ga,self.encoder,self.image(row),condition,observer=observe)
        result["G_before"]=before;result["G_after"]=self.g_fingerprint();result["gradient_observations"]=gradients
        self.cursor["g_updates"]+=result["optimizer_steps"]
        return result
    def load_reward_inputs(self):
        import torch
        from tbps.ablations.hv1_g60_state import read_old_features
        self.clean=read_old_features({"c":{"history_root":self.p["history_root"]}},"train","clean")
        old=Path("/home/lzf/ldx/outputs/tbps-framework-hv1-g60-adaptation-epoch60/gallery_train")
        ArtifactStore(old,new_base=old.parent).verify("completed")
        self.current_features=torch.load(old/"completed/features.pt",map_location="cpu",weights_only=True)["adversarial"].detach().clone()
        from tbps.ablations.g60_h_bridge import tensor_sha as historical_tensor_sha
        audit=read_json(self.c["sources"]["old_G60_bank_audit"]["path"])
        require(audit["status"]=="G60_TRAIN_GALLERY_CPU_AUDIT_PASS" and audit["G60"]==self.c["sources"]["G60"]
                and audit["images"]==2048 and audit["feature_content_sha256"]==historical_tensor_sha(self.current_features),
                "initial G60 bank audited content/source")
        self.initial_features=self.current_features.clone()
        require(self.current_features.shape==self.clean.shape==(2048,512),"initial reward bank dimensions")
        self.catalog=self.native.get("training_inputs").load_training_catalog()
        require([s.row_id for s in self.catalog.train]==self.p["train_ids"],"reward catalog order")
    def refresh_bank(self,ids,arm,index):
        import torch
        self.grant.validate(self.c,"smoke")
        require(list(ids)==self.p["train_ids"] and len(ids)==2048,"no partial/stale Train bank")
        root=self.root/arm/f"bank_{index}"
        require(not root.exists(),"bank immutable; no forward replay")
        store=ArtifactStore(root,new_base=self.c["output_base"])
        self.g.eval().requires_grad_(False);self.g.zero_grad(set_to_none=True)
        gsha=self.g_fingerprint()
        store.publish("G_state",{"state.pt":tensor_bytes(self.g.state_dict())},{"G":gsha,"arm":arm,"round":index})
        features=[]
        for position,row in enumerate(ids):
            store.publish(f"claim_{position:04d}",{"claim.json":json_bytes({"row":row,"index":position,"G":gsha})},
                          {"source":source_key(self.c),"G":gsha})
            image=self.image(row)
            with torch.no_grad():
                generated=self.native.get("inference_once").generate_once(self.g,image)
                adv=generated["adversarial"];delta=generated["effective_delta"]
                assert_pixels(image,adv,delta)
                feature=self.encoder(adv).detach().cpu().clone()
            payload={"original":image.detach().cpu().clone(),"adversarial":adv.detach().cpu().clone(),
                     "effective_delta":delta.detach().cpu().clone(),"feature":feature}
            data=tensor_bytes(payload)
            md={"row":row,"index":position,"G":gsha,"content_sha256":tree_sha(payload),"budget":8/255}
            store.publish(f"image_{position:04d}",{"data.pt":data,"metadata.json":json_bytes(md)},md)
            restored=torch.load(root/f"image_{position:04d}/data.pt",map_location="cpu",weights_only=True)
            require(tree_sha(restored)==tree_sha(payload),"per-image atomic readback")
            features.append(feature)
            del image,adv,delta,payload,restored,generated
            print("GH_BANK_PROGRESS",arm,index,position+1,2048,flush=True)
        tensor=torch.cat(features).clone()
        self.previous_features=self.current_features
        self.current_features=tensor
        identity=BankIdentity(gsha,tuple(ids),self.p["clean_gallery_sha256"],tree_sha(tensor),
                              self.p["source_space_sha256"],index,arm)
        store.publish("completed",{"features.pt":tensor_bytes(tensor),"identity.json":json_bytes(asdict(identity))},
                      {"source":source_key(self.c),"bank":asdict(identity)})
        require(gsha==self.g_fingerprint(),"G changed during inference bank refresh")
        self.last_bank=identity
        return identity
    def install_new_reward_bank(self,bank):
        bank.validate(self.p["train_ids"],self.g_fingerprint(),bank.round,bank.arm,
                      space=self.p["source_space_sha256"],clean=self.p["clean_gallery_sha256"])
        store=SimpleNamespace(role="train",gather=lambda:{"clean":self.clean,"adversarial":self.current_features})
        self.reward=self.native.get("formal_training.runtime").RewardBank(self.catalog,store,self.bundle.frontend)
    def h_update(self,row,bank):
        self.grant.validate(self.c,"smoke")
        require(bank==self.last_bank,"stale H reward bank")
        self.cursor["h_visits"]+=1
        e=self.entries[row]
        if not e["eligible"]:
            return {"row_id":row,"fixed_NOOP":True,"optimizer_steps_this_step":0,"IRRA_text_samples":0}
        self.h.requires_grad_(True)
        example=self.native.get("formal_training.data").example_from(e)
        query=self.catalog.get_train(row)
        def bind(k):return self.reward.bind(query,self.runtime,example,k)
        log=self.steps.update_h(self.h,self.ha,example,self.sampler,bind,self.cursor["h_visits"],row)
        self.cursor["h_updates"]+=1
        # 同一组已编码的当前H文本，在刷新前后图库分别计算原Reward；不增加模型前向，
        # 仅日志诊断，不参与H采样或梯度。实际SCST仍使用刚安装的新RewardBank。
        import torch
        runtime=self.native.get("formal_training.runtime")
        positive=torch.tensor([s.pid==query.pid for s in self.catalog.train])
        context=runtime.TrainGallery(self.clean,self.previous_features,positive,
            self.p["train_ids"].index(row),"train",("train",)*2048,"train",("train",)*2048)
        evaluator=runtime.RewardEvaluator(context,lambda texts:self.runtime.last_features,1.,1.)
        old_rewards=evaluator.evaluate_many(self.runtime.last_texts[0],self.runtime.last_texts[1:])
        log["old_bank_same_text_rewards"]=[asdict(x) for x in old_rewards]
        log["reward_delta_same_text"]=[log[key]["total"]-x.total for key,x in zip(
            ("sampled_reward","greedy_reward"),old_rewards)]
        log["bank_sha256"]=bank.adversarial_sha256
        return log
    def complete_state(self,index):
        self.cursor["round"]=index
        return snapshot(self.g,self.h,self.ga,self.ha,g_rng=self.rng["G"],h_rng=self.rng["H"],
                        cursor=self.cursor,parents=self.bundle.state["parents"])

@contextmanager
def loaded_backend(c,bundle,grant):
    """调用者必须已完成CPUStartupGate；此函数不负责/不能绕过CUDA门禁。"""
    import torch
    from tbps.models.irra import IRRAAdapter
    from tbps.models.irra_source import IRRASource
    require(torch.cuda.is_initialized(),"explicit startup gate required before loading GPU models")
    grant.validate(c,grant.action)
    source=IRRASource.from_dict(read_json(c["sources"]["dataset_config"]["path"])["irra"])
    # 原生模型加载器必须在legacy命名空间会话外执行。
    adapter=IRRAAdapter.from_source(source,device="cuda:0",allow_model_load=True)
    with native_session(c,reuse=bundle.modules):
        backend=NativeBackend(c,bundle,grant,adapter)
        try:yield backend
        finally:backend.close()
