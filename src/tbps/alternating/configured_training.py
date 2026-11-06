"""复用FormalBackend/Journal的可配置调度；不实现新G/H/Loss/SCST。"""
from contextlib import contextmanager
from dataclasses import asdict
import time
from pathlib import Path
from tbps.artifacts import FileRef, ArtifactStore, json_bytes, sha_file, digest
from .contracts import require
from .formal_backend import FormalBackend
from .formal_state import FormalJournal
from .state import clone_tree, tree_sha
from .word_budget import check_words

EXTRA = {"schedulers", "training_configuration"}

def scheduler(optimizer, spec):
    import torch
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=spec["step_size"], gamma=spec["gamma"])

def configure_optimizer(optimizer, rate):
    for group in optimizer.param_groups:
        group["lr"] = rate
        group["initial_lr"] = rate

class ConfiguredBackend(FormalBackend):
    def __init__(self, c, bundle, grant, adapter):
        super().__init__(c,bundle,grant,adapter)
        self.training=c["alternating_configuration"]
        # 只显式配置LR，保留父Adam全部动量/参数step；身份中记录这是新训练方案。
        configure_optimizer(self.ga,self.training["g_learning_rate"])
        configure_optimizer(self.ha,self.training["h_learning_rate"])
        self.gs=scheduler(self.ga,self.training["g_scheduler"])
        self.hs=scheduler(self.ha,self.training["h_scheduler"])
    def complete_state(self,index):
        value=super().complete_state(index)
        value.update(schedulers={"G":clone_tree(self.gs.state_dict()),"H":clone_tree(self.hs.state_dict())},
                     training_configuration=clone_tree(self.training))
        return value
    def restore_complete(self,value):
        require(value["training_configuration"]==self.training,"resume training configuration differs")
        require(set(value["schedulers"])=={"G","H"},"both Scheduler states required")
        super().restore_complete({k:v for k,v in value.items() if k not in EXTRA})
        self.gs.load_state_dict(value["schedulers"]["G"]);self.hs.load_state_dict(value["schedulers"]["H"])
        require(tree_sha(self.gs.state_dict())==tree_sha(value["schedulers"]["G"])
                and tree_sha(self.hs.state_dict())==tree_sha(value["schedulers"]["H"]),"Scheduler restore exact")
    def h_rollout(self,public):
        # attack()虽为inference_mode，仍显式冻结参数；不能以eval替代冻结。
        self.h.requires_grad_(False);self.h.zero_grad(set_to_none=True)
        trace=super().h_rollout(public)
        check_words(trace.original,trace.adversarial)
        require(not any(p.requires_grad or p.grad is not None for p in self.h.parameters()),"rollout H freeze")
        return trace
    def h_update(self,row,bank):
        result=super().h_update(row,bank)
        if result.get("fixed_NOOP"):
            result["word_budget"]=check_words(self.entries[row]["caption"],self.entries[row]["caption"])
            return result
        for name in ("sampled","greedy"):
            action=result[name]
            result[name]["word_budget"]=check_words(action["original"],action["adversarial"])
        # FreshReward已编码原文/sample/greedy；只取既有Tensor，绝不追加神经前向。
        features=self.runtime.last_features
        index=self.p["train_ids"].index(row)
        result["four_mode_paired_similarity"]={}
        for n,i in (("sampled",1),("greedy",2)):
            result["four_mode_paired_similarity"][n]={
                "Clean":float(features[0] @ self.clean[index]),
                "Image-only":float(features[0] @ self.current_features[index]),
                "Text-only":float(features[i] @ self.clean[index]),
                "Joint":float(features[i] @ self.current_features[index])}
        return result
    def orders(self,index,side,pass_index):
        if self.c["formal_binding"]["scope"]=="pilot":return list(self.p["samples"])
        seed=self.training["seed"]
        if side=="G":
            epoch=60+(index-1)*self.training["g_passes_per_round"]+pass_index
            m=self.native_session.import_module("epoch_training")
            return [e["row_id"] for e in m.epoch_order(list(self.gentries.values()),seed,epoch)]
        epoch=10+(index-1)*self.training["h_passes_per_round"]+pass_index
        m=self.native_session.import_module("formal_training.training")
        return m.order(self.catalog.train,epoch,seed)[-2048:]
    def frozen_phase(self,name):
        module,optimizer=(self.h,self.ha) if name=="H" else (self.g,self.ga)
        require(all(not p.requires_grad and p.grad is None for p in module.parameters()),"inactive module gradients: "+name)
        self.assert_frozen_sources()
        return tree_sha(module.state_dict()),tree_sha(optimizer.state_dict())

@contextmanager
def loaded(c,bundle,grant):
    from tbps.models.irra import IRRAAdapter
    from tbps.models.irra_source import IRRASource
    from tbps.artifacts import read_json
    from .native_sources import native_session
    source=IRRASource.from_dict(read_json(c["sources"]["dataset_config"]["path"])["irra"])
    adapter=IRRAAdapter.from_source(source,device="cuda:0",allow_model_load=True)
    with native_session(c,reuse=bundle.modules) as session:
        backend=ConfiguredBackend(c,bundle,grant,adapter);backend.native_session=session
        try:yield backend
        finally:backend.close()

def run_rounds(b,journal,parent,rounds,*,start=0,stop_after=None,stop_requested=lambda:False):
    state=clone_tree(parent);b.restore_complete(state)
    cfg=b.training
    for index in range(start+1,rounds+1):
        begin=time.monotonic();journal.begin(index,state);b.begin_round(index)
        before=clone_tree(state["cursor"]);gb=b.g_fingerprint();hb=b.h_fingerprint()
        counts_before=dict(b.counts);glog=[];hlog=[];gorders=[];horders=[]
        b.enter_phase_rng("G");inactive_h=b.frozen_phase("H");b.leave_phase_rng("G")
        for gp in range(1,cfg["g_passes_per_round"]+1):
            rows=b.orders(index,"G",gp);gorders.append(rows)
            require(len(rows)==len(set(rows)) and set(rows)==set(b.p["samples"]),"G pass full membership")
            b.progress("H_ROLLOUT",index,0,len(rows));b.enter_phase_rng("H")
            traces=[b.h_rollout(b.public_text(row)) for row in rows];b.leave_phase_rng("H")
            require(b.h_fingerprint()==hb,"rollout updated H")
            b.progress("NEW_CONTEXT_JVP",index,0,len(rows))
            conditions=[b.condition("B",row,trace) for row,trace in zip(rows,traces)]
            del traces
            b.enter_phase_rng("G");b.progress("G_TRAIN",index,0,len(rows))
            for pos,(row,condition) in enumerate(zip(rows,conditions),1):
                log=b.g_update(row,condition)
                require(log["optimizer_steps"]==condition.payload["K"]+1,"original K+1")
                log.update(row_id=row,pass_index=gp);glog.append(log)
                if pos%32==0:b.progress("G_TRAIN",index,pos,len(rows))
            b.leave_phase_rng("G")
            require(b.frozen_phase("H")==inactive_h,"G phase changed H or H Adam")
            del conditions
        inactive_h_after=b.frozen_phase("H")
        b.gs.step();g_end=time.monotonic();ga=b.g_fingerprint()
        b.progress("FULL_TRAIN_BANK",index,0,2048)
        bank=b.refresh_bank(b.p["train_ids"],"B",index)
        bank.validate(b.p["train_ids"],ga,index,"B",space=b.p["source_space_sha256"],clean=b.p["clean_gallery_sha256"])
        b.install_new_reward_bank(bank);bank_end=time.monotonic()
        b.enter_phase_rng("H");inactive_g=b.frozen_phase("G")
        for hp in range(1,cfg["h_passes_per_round"]+1):
            rows=b.orders(index,"H",hp);horders.append(rows)
            require(len(rows)==len(set(rows)) and set(rows)==set(b.p["samples"]),"H pass full membership")
            b.progress("H_SCST_TRAIN",index,0,len(rows))
            for pos,row in enumerate(rows,1):
                log=b.h_update(row,bank);log.update(row_id=row,pass_index=hp);hlog.append(log)
                if pos%32==0:b.progress("H_SCST_TRAIN",index,pos,len(rows))
            require(b.frozen_phase("G")==inactive_g,"H phase changed G or G Adam")
        b.leave_phase_rng("H");b.hs.step()
        state=b.complete_state(index);gu=sum(x["optimizer_steps"] for x in glog)
        hu=sum(x["optimizer_steps_this_step"] for x in hlog)
        require(state["cursor"]=={"round":index,"g_updates":before["g_updates"]+gu,
            "h_visits":before["h_visits"]+len(hlog),"h_updates":before["h_updates"]+hu},"absolute update accounting")
        ledger={"round":index,"G_before":gb,"G_after":ga,"H_before":hb,"H_after":b.h_fingerprint(),
            "condition_H_state":hb,"G_changed":gb!=ga,"H_changed":hb!=b.h_fingerprint(),
            "start_cursor":before,"end_cursor":state["cursor"],"G":glog,"H":hlog,
            "g_orders":gorders,"h_orders":horders,"bank":asdict(bank),"conditions":b.conditions_log,
            "forward_counts_before":counts_before,"forward_counts_cumulative":dict(b.counts),
            "isolation":{"H_and_H_Adam_unchanged_during_G":True,"G_and_G_Adam_unchanged_during_H":True,
                         "IRRA_reader_frozen":True,"inactive_gradients_none":True},
            "isolation_fingerprints":{"inactive_H_before_G":inactive_h,"inactive_H_after_G":inactive_h_after,
                "inactive_G_before_H":inactive_g,"inactive_G_after_H":b.frozen_phase("G")},
            "seconds":{"G_and_context":g_end-begin,"bank":bank_end-g_end,"H":time.monotonic()-bank_end,
                       "total":time.monotonic()-begin},"learning_rates":{"G":b.ga.param_groups[0]["lr"],"H":b.ha.param_groups[0]["lr"]}}
        journal.commit(index,state,ledger,b.bank_ref(index))
        if index%cfg["checkpoint_interval_rounds"]==0 or index==rounds:
            p=journal.root/f"round_{index:04d}"/"manifest.json"
            journal.store.publish(f"checkpoint_{index:04d}",{"checkpoint.json":json_bytes({"round":index,"boundary":True,
                "state_manifest":asdict(FileRef(str(p),sha_file(p)))})},{"configuration":journal.config},sources=(FileRef(str(p),sha_file(p)),))
        b.progress("ROUND_COMMITTED",index,len(glog),len(glog))
        print("GH_ALTERNATING_ROUND_COMPLETE",index,state["cursor"],ledger["seconds"],flush=True)
        if (stop_after is not None and index>=stop_after) or stop_requested():
            journal.stop_boundary(index);return state,False
    journal.finish(state,rounds)
    return state,True
