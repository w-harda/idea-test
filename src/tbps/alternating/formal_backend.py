"""仅调度和边界恢复适配；G/H训练、JVP、投影、Reward全部调用冻结实现。"""
from contextlib import contextmanager
from pathlib import Path
from dataclasses import asdict
from tbps.artifacts import ArtifactStore,FileRef,sha_file,read_json
from .contracts import BankIdentity,require
from .native_backend import NativeBackend
from .native_sources import native_session
from .state import tree_sha

class FormalBackend(NativeBackend):
    def restore_complete(self,state):
        super().restore_complete(state)
        if state["cursor"]["round"]:
            require(getattr(self,"resume_ledger",None) is not None,"resume requires committed bank ledger")
            self.restore_bank(self.resume_ledger)
            self.counts.update(self.resume_ledger["forward_counts_cumulative"])
    def begin_round(self,index):
        self.round_index=index
        root=self.root/"round_io"/f"round_{index:04d}"
        require(not root.exists(),"round IO exists; no uncommitted replay")
        self.store=ArtifactStore(root,new_base=self.c["output_base"])
        self.conditions_log=[];self.rollout_log=[]
    def bank_ref(self,index):
        p=self.root/"B"/f"bank_{index}"/"completed/manifest.json"
        return FileRef(str(p),sha_file(p))
    def restore_bank(self,ledger):
        """边界恢复使用已提交G版本的完整图库，不能回退到初始G60奖励。"""
        import torch
        i=ledger["round"];root=self.root/"B"/f"bank_{i}"
        store=ArtifactStore(root,new_base=self.c["output_base"]);store.verify("completed")
        d=read_json(root/"completed/identity.json");d["ordered_train_ids"]=tuple(d["ordered_train_ids"])
        bank=BankIdentity(**d)
        bank.validate(self.p["train_ids"],self.g_fingerprint(),i,"B",
            space=self.p["source_space_sha256"],clean=self.p["clean_gallery_sha256"])
        x=torch.load(root/"completed/features.pt",map_location="cpu",weights_only=True).clone()
        require(tree_sha(x)==bank.adversarial_sha256,"resume bank feature SHA")
        self.current_features=x;self.previous_features=x.clone();self.last_bank=bank
        self.install_new_reward_bank(bank)
    def original_orders(self,index):
        """直接调用原训练器的绝对Epoch顺序；不改变原随机种子规则。"""
        if self.c["formal_binding"]["scope"]=="pilot":
            rows=list(self.p["samples"]);return rows,rows.copy()
        eg=self.native_session.import_module("epoch_training")
        ht=self.native_session.import_module("formal_training.training")
        selected=eg.epoch_order(list(self.gentries.values()),42,60+index)
        g=[e["row_id"] for e in selected]
        h=ht.order(self.catalog.train,10+index,42)[-len(self.catalog.train):]
        require(set(g)==set(h)==set(self.p["train_ids"]),"full epoch membership/order")
        return g,h

@contextmanager
def loaded_formal_backend(c,bundle,grant):
    import torch
    from tbps.models.irra import IRRAAdapter
    from tbps.models.irra_source import IRRASource
    require(torch.cuda.is_initialized(),"startup gate required")
    grant.validate(c,"smoke")
    source=IRRASource.from_dict(read_json(c["sources"]["dataset_config"]["path"])["irra"])
    adapter=IRRAAdapter.from_source(source,device="cuda:0",allow_model_load=True)
    with native_session(c,reuse=bundle.modules) as session:
        backend=FormalBackend(c,bundle,grant,adapter);backend.native_session=session
        try:yield backend
        finally:backend.close()
