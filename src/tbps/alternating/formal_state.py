"""B正式训练的不可变Round边界；失败现场不清理、不自动重播。"""
import io
from pathlib import Path
from tbps.artifacts import ArtifactStore,json_bytes,read_json,sha_file,digest,FileRef
from .contracts import require
from .state import clone_tree,tree_sha

def tensor_bytes(value):
    import torch
    b=io.BytesIO();torch.save(clone_tree(value),b);return b.getvalue()

class FormalJournal:
    def __init__(self,root,config_sha256,base):
        self.root=Path(root);self.config=config_sha256
        self.store=ArtifactStore(self.root,new_base=base)
    def initialize(self,protocol,parent):
        require(not self.root.exists(),"formal output already consumed; explicit boundary resume only")
        self.store.publish("protocol",{"protocol.json":json_bytes(protocol)},
            {"config":self.config,"parent_state":tree_sha(parent)})
    def begin(self,index,parent):
        require(index>=1 and not self.store.pending(),"pending preserved")
        require(not (self.root/"completed").exists(),"training completed")
        self.store.publish(f"round_{index:04d}_claim",{"claim.json":json_bytes({
            "round":index,"parent_state":tree_sha(parent),"config":self.config})},
            {"config":self.config,"round":index})
    def commit(self,index,state,ledger,bank_ref):
        self.store.verify(f"round_{index:04d}_claim",expected_provenance={"config":self.config,"round":index})
        require(state["cursor"]["round"]==index and ledger["end_cursor"]==state["cursor"],"round cursor")
        require(ledger["bank"]["g_state_sha256"]==ledger["G_after"]
                and ledger["bank"]["round"]==index,"bank version")
        self.store.publish(f"round_{index:04d}",{"state.pt":tensor_bytes(state),"ledger.json":json_bytes(ledger)},
            {"config":self.config,"round":index,"state":tree_sha(state)},sources=(bank_ref,))
    def load_boundary(self,index,parents):
        import torch
        require(index>=1 and not self.store.pending() and not (self.root/"completed").exists(),"invalid resume state")
        self.store.verify("protocol")
        present=[int(p.name[6:10]) for p in self.root.glob("round_*") if p.is_dir()]
        require(present and max(present)==index,"later/uncommitted Round Claim blocks replay")
        m=self.store.verify(f"round_{index:04d}",expected_provenance={
            "config":self.config,"round":index,"state":read_json(self.root/f"round_{index:04d}"/"manifest.json")["provenance"]["state"]})
        value=torch.load(self.root/f"round_{index:04d}/state.pt",map_location="cpu",weights_only=False)
        require(tree_sha(value)==m["provenance"]["state"] and value["parents"]==parents,"checkpoint source/state")
        ledger=read_json(self.root/f"round_{index:04d}/ledger.json")
        require(ledger["end_cursor"]==value["cursor"],"checkpoint ledger")
        return clone_tree(value),ledger
    def record_resume(self,index,failures=()):
        count=len(list(self.root.glob("resume_*")))+1
        self.store.publish(f"resume_{count:04d}",{"resume.json":json_bytes({"from_round":index,"explicit":True,
            "preserved_failures":[{"path":v.path,"sha256":v.sha256} for v in failures]})},
            {"config":self.config,"boundary_only":True},sources=failures)
        return count
    def stop_boundary(self,index):
        key=f"boundary_stop_{index:04d}"
        self.store.publish(key,{"stop.json":json_bytes({"round":index,"reason":"signal; committed boundary"})},
                           {"config":self.config})
    def finish(self,state,rounds):
        self.store.publish("completed",{"report.json":json_bytes({
            "status":"GH_B_ROUND_SCHEDULE_COMPLETE","rounds":rounds,"cursor":state["cursor"],
            "state":tree_sha(state),"not_Test_evaluation":True})},{"config":self.config})

def run_rounds(backend,journal,parent,rounds,orders,*,start=0,stop_requested=lambda:False,on_round=None):
    """A1不混入本入口；每轮仅沿上一轮已更新状态前进，不重载原G60/H10。"""
    from dataclasses import asdict
    state=clone_tree(parent)
    backend.restore_complete(state)  # 全任务只在启动/显式边界恢复时调用一次。
    for index in range(start+1,rounds+1):
        journal.begin(index,state);backend.begin_round(index)
        g_order,h_order=orders(index)
        require(len(g_order)==len(h_order)==len(set(g_order)) and set(g_order)==set(h_order),"G/H member completeness")
        progress=getattr(backend,"progress",lambda *a,**k:None)
        progress("H_ROLLOUT",index,0,len(g_order))
        before=dict(state["cursor"]);g_before=backend.g_fingerprint();h_before=backend.h_fingerprint()
        backend.enter_phase_rng("H")
        traces=[]
        for position,row in enumerate(g_order):
            traces.append(backend.h_rollout(backend.public_text(row)))
            if (position+1)%32==0:progress("H_ROLLOUT",index,position+1,len(g_order))
        backend.leave_phase_rng("H")
        require(backend.h_fingerprint()==h_before,"rollout must not update H")
        progress("NEW_CONTEXT_JVP",index,0,len(g_order))
        conditions=[]
        for position,(row,trace) in enumerate(zip(g_order,traces)):
            conditions.append(backend.condition("B",row,trace))
            if (position+1)%32==0:progress("NEW_CONTEXT_JVP",index,position+1,len(g_order))
        del traces
        progress("G_TRAIN",index,0,len(g_order))
        backend.enter_phase_rng("G");glog=[]
        for pos,(row,c) in enumerate(zip(g_order,conditions)):
            log=backend.g_update(row,c);require(log["optimizer_steps"]==c.payload["K"]+1,"original K+1")
            glog.append(log)
            if (pos+1)%32==0:
                print("B_G_PROGRESS",index,pos+1,len(g_order),flush=True)
                progress("G_TRAIN",index,pos+1,len(g_order))
        backend.leave_phase_rng("G")
        require(backend.h_fingerprint()==h_before,"G phase changed H")
        g_after=backend.g_fingerprint()
        progress("FULL_TRAIN_BANK",index,0,len(backend.p["train_ids"]))
        bank=backend.refresh_bank(backend.p["train_ids"],"B",index)
        bank.validate(backend.p["train_ids"],g_after,index,"B",
            space=backend.p["source_space_sha256"],clean=backend.p["clean_gallery_sha256"])
        backend.install_new_reward_bank(bank)
        progress("H_SCST_TRAIN",index,0,len(h_order))
        backend.enter_phase_rng("H");hlog=[]
        for pos,row in enumerate(h_order):
            hlog.append(backend.h_update(row,bank))
            if (pos+1)%32==0:
                print("B_H_PROGRESS",index,pos+1,len(h_order),flush=True)
                progress("H_SCST_TRAIN",index,pos+1,len(h_order))
        backend.leave_phase_rng("H")
        require(g_after==backend.g_fingerprint(),"H phase changed G")
        backend.assert_frozen_sources()
        state=backend.complete_state(index)
        gu=sum(x["optimizer_steps"] for x in glog)
        hu=sum(x.get("optimizer_steps_this_step",1) for x in hlog if not x.get("fixed_NOOP"))
        require(state["cursor"]=={"round":index,"g_updates":before["g_updates"]+gu,
            "h_visits":before["h_visits"]+len(h_order),"h_updates":before["h_updates"]+hu},"update ledger")
        ledger={"round":index,"G_before":g_before,"G_after":g_after,"H_before":h_before,
            "H_after":backend.h_fingerprint(),"start_cursor":before,"end_cursor":state["cursor"],
            "g_order":g_order,"h_order":h_order,"G":glog,"H":hlog,"bank":asdict(bank),
            "conditions":backend.conditions_log,"forward_counts_cumulative":dict(backend.counts),"condition_H_state":h_before,
            "G_changed":g_before!=g_after,"H_changed":h_before!=backend.h_fingerprint()}
        journal.commit(index,state,ledger,backend.bank_ref(index))
        print("GH_B_ROUND_COMPLETE",index,state["cursor"],flush=True)
        if on_round:on_round(index,state,ledger)
        if stop_requested():
            journal.stop_boundary(index)
            return state,False
    journal.finish(state,rounds)
    return state,True
