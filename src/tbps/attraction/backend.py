"""现有整轮调度的原生适配：G1→新Bank→H1→H1语境/JVP→G2。"""
from contextlib import contextmanager
import torch
from tbps.artifacts import read_json,sha_file
from tbps.alternating.fresh_training import FreshBackend
from tbps.alternating.native_sources import native_session
from tbps.alternating.state import clone_tree,tree_sha
from .objectives import Settings,ProfileKL,require
from .g_step import attraction_g
from .h_reward import train_step

class AttractionBackend(FreshBackend):
    def load_reward_inputs(self):
        super().load_reward_inputs()
        self.profile_cpu=ProfileKL(self.clean,Settings(**self.c['attraction']))
        self.profile_g=self.profile_cpu.to(next(self.g.parameters()).device)
        self.anchor_sha=tree_sha(self.clean)
    def g_update(self,row,condition):
        index=self.p['train_ids'].index(row);records=[]
        teacher=self.h_fingerprint()
        with attraction_g(self.native.get('serial_training'),self.profile_g,self.clean[index:index+1],
                condition.payload['z_t'].detach().reshape(1,-1),records):
            result=super().g_update(row,condition)
        require(len(records)==condition.payload['K']+1,'original G K+1 updates')
        for record,values in zip(result['records'],records):record['attraction']=values
        result['attraction']={'teacher_H_state':teacher,'condition_identity':condition.identity.key,
            'anchor_sha256':self.anchor_sha,'candidate_count':len(self.clean),'settings':dict(self.c['attraction'])}
        return result
    def h_update(self,row,bank):
        self.grant.validate(self.c,'smoke');require(bank==self.last_bank,'current G reward bank required')
        require(bank.g_state_sha256==self.g_fingerprint(),'stale G parameters invalidate H partner bank')
        self.cursor['h_visits']+=1;e=self.entries[row]
        if not e['eligible']:return {'row_id':row,'fixed_NOOP':True,'optimizer_steps_this_step':0,'IRRA_text_samples':0}
        self.h.requires_grad_(True)
        example=self.native.get('formal_training.data').example_from(e);query=self.catalog.get_train(row)
        index=self.p['train_ids'].index(row)
        def bind(k):return self.reward.bind(query,self.runtime,example,k)
        log,_=train_step(self.native.get('phase4_training'),self.h,self.ha,example,self.sampler,bind,
            self.cursor['h_visits'],row,self.profile_cpu,self.current_features[index:index+1])
        self.cursor['h_updates']+=1;log['bank_sha256']=bank.adversarial_sha256
        log['attraction'].update(teacher_G_state=bank.g_state_sha256,anchor_sha256=self.anchor_sha,settings=dict(self.c['attraction']))
        return log

@contextmanager
def loaded(c,bundle,grant):
    from tbps.models.irra import IRRAAdapter
    from tbps.models.irra_source import IRRASource
    torch.cuda.manual_seed_all(42);cuda=clone_tree(torch.cuda.get_rng_state_all())
    for side in ('G','H'):bundle.state[side+'_RNG']['cuda']=clone_tree(cuda)
    source=IRRASource.from_dict(read_json(c['sources']['dataset_config']['path'])['irra'])
    adapter=IRRAAdapter.from_source(source,device='cuda:0',allow_model_load=True)
    with native_session(c,reuse=bundle.modules) as session:
        backend=AttractionBackend(c,bundle,grant,adapter);backend.native_session=session
        try:yield backend
        finally:backend.close()
