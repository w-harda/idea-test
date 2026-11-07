"""v2仅替换初始化/预算/顺序/状态保存；G/H训练数学复用NativeBackend。"""
from contextlib import contextmanager
from pathlib import Path
from tbps.artifacts import ArtifactStore,read_json,digest,FileRef,sha_file
from .contracts import require,BankIdentity
from .native_backend import NativeBackend
from .native_sources import native_session
from .state import clone_tree,tree_sha,capture_rng

class NoScheduler:
    def step(self):pass

class FreshBackend(NativeBackend):
    def __init__(self,c,bundle,grant,adapter):
        super().__init__(c,bundle,grant,adapter)
        self.training=c['training_configuration'];self.gs=NoScheduler();self.hs=NoScheduler()
    def load_reward_inputs(self):
        from tbps.ablations.hv1_g60_state import read_old_features
        history=Path(self.p['history_root'])/'gallery_train'
        done=read_json(history/'completed.json')
        require(digest({'kind':'immutable original Clean gallery source receipt binding',
            'protocol':done['protocol_sha256'],'files':done['files']})==self.p['clean_gallery_sha256'],'fixed Clean source receipt')
        self.clean=read_old_features({'c':{'history_root':self.p['history_root']}},'train','clean')
        # 首轮旧图库仅作为原诊断参考，实际H奖励永远等G更新并完成新Bank后才安装。
        self.current_features=self.clean.clone();self.previous_features=self.clean.clone()
        self.catalog=self.native.get('training_inputs').load_training_catalog()
        require([s.row_id for s in self.catalog.train]==self.p['train_ids'],'clean/catalog order')
    def h_rollout(self,public):
        self.h.requires_grad_(False);self.h.zero_grad(set_to_none=True)
        return super().h_rollout(public)  # 原Natural-K/Unicode/共享词规则，无WordBudget1。
    def orders(self,index,side,pass_index):return list(self.p['samples'])
    def frozen_phase(self,name):
        m,o=(self.h,self.ha) if name=='H' else (self.g,self.ga)
        require(all(not p.requires_grad and p.grad is None for p in m.parameters()),'inactive gradients '+name)
        self.assert_frozen_sources();return tree_sha(m.state_dict()),tree_sha(o.state_dict())
    def begin_round(self,index):
        self.round_index=index;root=self.root/'round_io'/f'round_{index:04d}'
        require(not root.exists(),'round IO consumed; no replay');self.store=ArtifactStore(root,new_base=self.c['output_base'])
        self.conditions_log=[];self.rollout_log=[]
    def bank_ref(self,index):
        p=self.root/'B'/f'bank_{index}'/'completed/manifest.json';return FileRef(str(p),sha_file(p))
    def complete_state(self,index):
        value=super().complete_state(index)
        value.update(schedulers={'G':None,'H':None},training_configuration=clone_tree(self.training))
        return value
    def restore_complete(self,value):
        require(value['training_configuration']==self.training and value['schedulers']=={'G':None,'H':None},'original no-Scheduler state')
        super().restore_complete({k:v for k,v in value.items() if k not in ('training_configuration','schedulers')})
        if value['cursor']['round']:
            import torch
            log=self.resume_ledger;d=self.root/'B'/('bank_'+str(log['round']))
            store=ArtifactStore(d,new_base=self.c['output_base']);store.verify('completed')
            bid=read_json(d/'completed/identity.json');bid['ordered_train_ids']=tuple(bid['ordered_train_ids'])
            bank=BankIdentity(**bid);bank.validate(self.p['train_ids'],self.g_fingerprint(),log['round'],'B',
                space=self.p['source_space_sha256'],clean=self.p['clean_gallery_sha256'])
            self.current_features=torch.load(d/'completed/features.pt',map_location='cpu',weights_only=True).clone()
            require(tree_sha(self.current_features)==bank.adversarial_sha256,'latest bank feature SHA')
            self.previous_features=self.current_features.clone();self.last_bank=bank;self.install_new_reward_bank(bank)
            self.counts.update(log['forward_counts_cumulative'])

@contextmanager
def loaded(c,bundle,grant):
    import torch
    from tbps.models.irra import IRRAAdapter
    from tbps.models.irra_source import IRRASource
    # 新CUDA RNG从本次seed开始，绝不使用历史G/H RNG。
    torch.cuda.manual_seed_all(42)
    cuda=clone_tree(torch.cuda.get_rng_state_all())
    for side in ('G','H'):bundle.state[side+'_RNG']['cuda']=clone_tree(cuda)
    source=IRRASource.from_dict(read_json(c['sources']['dataset_config']['path'])['irra'])
    adapter=IRRAAdapter.from_source(source,device='cuda:0',allow_model_load=True)
    with native_session(c,reuse=bundle.modules) as session:
        backend=FreshBackend(c,bundle,grant,adapter);backend.native_session=session
        try:yield backend
        finally:backend.close()
