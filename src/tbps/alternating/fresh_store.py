"""每10轮不可变G/H评估权重；仅保留最新完整边界，旧新产物显式退休。"""
from pathlib import Path
import os,shutil
from tbps.artifacts import ArtifactStore,FileRef,sha_file,read_json,json_bytes,digest,_fsync
from .contracts import require
from .state import tree_sha,clone_tree
from .formal_state import tensor_bytes

def atomic_json(path,value):
    p=Path(path);tmp=p.with_name(p.name+'.pending')
    with tmp.open('xb') as f:f.write(json_bytes(value));f.flush();os.fsync(f.fileno())
    os.replace(tmp,p);_fsync(p.parent)

def prune_owned(path,run_root):
    target=Path(path).resolve();root=Path(run_root).resolve()
    require(root.name in ('train','smoke') and root.parent.name=='tbps-framework-gh-alternating-v2-20261107','v2 owned root')
    require(target.is_relative_to(root) and target!=root and not Path(path).is_symlink(),'prune confined to owned run')
    require(target.name.startswith(('round_','bank_')),'only obsolete boundary/IO/bank directories')
    if target.exists():shutil.rmtree(target)

class Milestones:
    def __init__(self,journal):self.j=journal
    def publish(self,key,files,provenance,*,sources=()):
        # 原调度器的里程碑引用改指独立G/H导出，不引用将退休的边界状态。
        i=int(key.split('_')[-1]);p=self.j.run/'checkpoints'/f'round_{i:02d}'/'manifest.json'
        return self.j.art.publish(key,{'checkpoint.json':json_bytes({'round':i,'paired_export':{'path':str(p),'sha256':sha_file(p)}})},
            {'configuration':self.j.config},sources=(FileRef(str(p),sha_file(p)),))

class FreshJournal:
    def __init__(self,run,config,base):
        self.run=Path(run);self.root=self.run/'boundaries';self.config=config;self.base=base
        self.art=ArtifactStore(self.root,new_base=base);self.store=Milestones(self)
        self.ledgers=ArtifactStore(self.run/'ledgers',new_base=base)
    def initialize(self,protocol,parent):
        require(not self.root.exists(),'new Attempt only')
        self.art.publish('protocol',{'protocol.json':json_bytes(protocol)},{'configuration':self.config,'initial_state':tree_sha(parent)})
    def begin(self,index,parent):
        require(not self.art.pending() and not (self.root/'completed').exists(),'pending/completed boundary')
        if index>1:
            latest=read_json(self.root/'latest.json');require(latest['round']==index-1 and latest['state_sha']==tree_sha(parent),'latest exact parent')
        self.art.publish(f'claim_{index:04d}',{'claim.json':json_bytes({'round':index,'parent_state':tree_sha(parent)})},{'configuration':self.config})
    def commit(self,index,state,ledger,bank_ref):
        require(state['cursor']['round']==index and ledger['bank']['g_state_sha256']==tree_sha(state['G']),'state/bank binding')
        self.art.verify(f'claim_{index:04d}')
        self.ledgers.publish(f'round_{index:04d}',{'ledger.json':json_bytes(ledger)},
            {'configuration':self.config,'state_sha':tree_sha(state)})
        self.art.publish(f'round_{index:04d}',{'state.pt':tensor_bytes(state),'ledger.json':json_bytes(ledger)},
            {'configuration':self.config,'state_sha':tree_sha(state)},sources=(bank_ref,))
        # 每10轮及Smoke结束独立保存用于评估的两个模型；没有Optimizer/RNG混作评估权重。
        if index%10==0 or self.run.name=='smoke':
            export=ArtifactStore(self.run/'checkpoints',new_base=self.base)
            export.publish(f'round_{index:02d}',{side+'.pt':tensor_bytes({'schema':'gh-v2-eval-weight-1','round':index,
                'model':side,'state_dict':state[side],'configuration_sha256':self.config,
                'training_ledger_sha256':sha_file(self.run/'ledgers'/f'round_{index:04d}/ledger.json')}) for side in ('G','H')},
                {'round':index,'configuration':self.config,'G_state_sha':tree_sha(state['G']),'H_state_sha':tree_sha(state['H'])})
        atomic_json(self.root/'latest.json',{'round':index,'state_sha':tree_sha(state),
            'manifest':{'path':str(self.root/f'round_{index:04d}/manifest.json'),'sha256':sha_file(self.root/f'round_{index:04d}/manifest.json')}})
        if index>1:
            prev=index-1;old=self.root/f'round_{prev:04d}';m=self.art.verify(old.name)
            bank=self.run/'B'/f'bank_{prev}';oldbank=read_json(bank/'completed/manifest.json')
            self.art.publish(f'retired_{prev:04d}',{'retirement.json':json_bytes({'round':prev,
                'boundary_manifest':m,'bank_manifest':oldbank,'replaced_by':index,'evaluation_weights_retained':prev%10==0})},
                {'configuration':self.config,'explicit_rolling_retention':True})
            for target in (old,bank,self.run/'round_io'/f'round_{prev:04d}'):prune_owned(target,self.run)
    def load_latest(self,parents):
        import torch
        require(not list(self.run.glob('FAIL*')) and not self.art.pending() and not (self.root/'completed').exists(),'failure/pending/completed cannot resume')
        latest=read_json(self.root/'latest.json');i=latest['round']
        require(not any(int(p.name.split('_')[-1])>i for p in self.root.glob('claim_*')),'later claim blocks replay')
        require(not any(int(p.name.split('_')[-1])>i for p in (self.run/'round_io').glob('round_*')),'later IO blocks replay')
        m=self.art.verify(f'round_{i:04d}');value=torch.load(self.root/f'round_{i:04d}/state.pt',map_location='cpu',weights_only=False)
        require(tree_sha(value)==latest['state_sha']==m['provenance']['state_sha'] and value['parents']==parents,'complete resume state/source')
        return clone_tree(value),read_json(self.root/f'round_{i:04d}/ledger.json')
    def stop_boundary(self,index):
        self.art.publish(f'stopped_{index:04d}',{'report.json':json_bytes({'round':index,'complete_boundary':True})},{'configuration':self.config})
    def finish(self,state,rounds):
        require(state['cursor']['round']==rounds,'complete round count')
        self.art.publish('completed',{'report.json':json_bytes({'rounds':rounds,'cursor':state['cursor'],'state_sha':tree_sha(state)})},{'configuration':self.config})
