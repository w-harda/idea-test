"""每10轮不可变G/H评估权重；仅保留最新完整边界，旧新产物显式退休。"""
from pathlib import Path
import os,shutil
from tbps.artifacts import ArtifactStore,FileRef,sha_file,read_json,json_bytes,digest,_fsync
from tbps.alternating.contracts import require
from tbps.alternating.state import tree_sha,clone_tree
from tbps.alternating.formal_state import tensor_bytes

def atomic_json(path,value):
    p=Path(path);tmp=p.with_name(p.name+'.pending')
    with tmp.open('xb') as f:f.write(json_bytes(value));f.flush();os.fsync(f.fileno())
    os.replace(tmp,p);_fsync(p.parent)

def prune_owned(path,run_root):
    target=Path(path).resolve();root=Path(run_root).resolve()
    require(root.name in ('train','smoke') and root.parent.name=='tbps-framework-gh-pushpull-aug-v1-20261111','v2 owned root')
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
        io=ArtifactStore(self.run/'round_io'/f'round_{index:04d}',new_base=self.base)
        require(len(ledger['conditions'])==len(ledger['g_orders'][0]),'all G contexts preserved')
        for pos,(condition,row) in enumerate(zip(ledger['conditions'],ledger['g_orders'][0])):
            require(condition['row']==row,'context original order')
            io.verify(f'rollout_{pos:04d}')
            condition['G_rollout']=read_json(io.root/f'rollout_{pos:04d}/action.json')
        bank=ArtifactStore(self.run/'B'/f'bank_{index}',new_base=self.base)
        bank_manifest=bank.verify('completed');source=bank.root/'completed/features.pt'
        ledger['bank_features_sha256']=sha_file(source);ledger['bank_manifest_snapshot']=bank_manifest
        view_source=bank.root/'completed/view_features.pt'
        ledger['bank_view_features_sha256']=sha_file(view_source)
        ledger['bank_view_inputs']=read_json(bank.root/'completed/view_inputs.json')['entries']
        files=shard_ledger(ledger);files['bank_features.pt']=source.read_bytes();files['bank_view_features.pt']=view_source.read_bytes()
        self.ledgers.publish(f'round_{index:04d}',files,{'configuration':self.config,'state_sha':tree_sha(state)})
        lm=self.run/'ledgers'/f'round_{index:04d}'/'manifest.json'
        self.art.publish(f'round_{index:04d}',{'state.pt':tensor_bytes(state),'ledger.json':files['ledger.json']},
            {'configuration':self.config,'state_sha':tree_sha(state)},sources=(bank_ref,FileRef(str(lm),sha_file(lm))))
        # 每10轮及Smoke结束独立保存用于评估的两个模型；没有Optimizer/RNG混作评估权重。
        if index%10==0 or self.run.name=='smoke':
            export=ArtifactStore(self.run/'checkpoints',new_base=self.base)
            export.publish(f'round_{index:02d}',{side+'.pt':tensor_bytes({'schema':'gh-pushpull-aug-eval-weight-1','round':index,
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

ARRAYS=('G','H','conditions')
SHARD_LIMIT=4*1024*1024

def shard_ledger(ledger):
    files={};summary={k:v for k,v in ledger.items() if k not in ARRAYS};summary['sharded_arrays']={}
    for name in ARRAYS:
        chunks=[];items=[];size=64
        def flush():
            nonlocal items,size
            if not items:return
            key=f'{name}_{len(chunks):04d}.json';raw=json_bytes({'items':items})
            require(len(raw)<=SHARD_LIMIT,'bounded strict ledger shard')
            files[key]=raw;chunks.append({'file':key,'count':len(items)});items=[];size=64
        for item in ledger[name]:
            raw=json_bytes(item);require(len(raw)+64<=SHARD_LIMIT,'single ledger record exceeds shard budget')
            if size+len(raw)+2>SHARD_LIMIT or len(items)>=32:flush()
            items.append(item);size+=len(raw)+2
        flush();summary['sharded_arrays'][name]={'count':len(ledger[name]),'chunks':chunks}
    files['ledger.json']=json_bytes(summary);return files

class ShardedRows:
    def __init__(self,path,descriptor):self.path=Path(path);self.descriptor=descriptor
    def __len__(self):return self.descriptor['count']
    def __iter__(self):
        n=0
        for ch in self.descriptor['chunks']:
            p=self.path/ch['file'];require(p.parent==self.path and p.stat().st_size<=SHARD_LIMIT,'bounded local ledger shard')
            chunk=read_json(p);require(set(chunk)=={'items'} and len(chunk['items'])==ch['count'],'exact shard schema/count')
            for item in chunk['items']:n+=1;yield item
        require(n==len(self),'complete ledger coverage')

def read_ledger(path):
    path=Path(path);value=read_json(path);require(set(value['sharded_arrays'])==set(ARRAYS),'all ledger arrays covered')
    ArtifactStore(path.parent.parent,new_base=path.parents[3]).verify(path.parent.name)
    for key in ARRAYS:value[key]=ShardedRows(path.parent,value['sharded_arrays'][key])
    return value
