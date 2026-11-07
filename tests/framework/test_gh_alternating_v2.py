"""v2从零/原Dynamic-K/滚动边界/Joint-only定向CPU测试。"""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import pytest
import torch
from tbps.artifacts import ArtifactStore,FileRef,sha_file,json_bytes,digest
from tbps.alternating.fresh_sources import load,fresh_bundle
from tbps.alternating.fresh_run import dynamic_budget_probe,no_historical_weights
from tbps.alternating.fresh_store import FreshJournal,prune_owned
from tbps.alternating.fresh_training import NoScheduler
from tbps.alternating.state import tree_sha,clone_tree
from tbps.alternating.configured_training import run_rounds

@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    def deny(*a,**kw):raise AssertionError('CPU test attempted CUDA')
    monkeypatch.setattr(torch.cuda,'_lazy_init',deny);monkeypatch.setattr(torch.cuda,'init',deny)
    torch.set_num_threads(1)

@pytest.fixture
def source():
    cfg=Path(__file__).resolve().parents[2]/'configs/framework/gh_alternating_v2.server.json'
    return cfg,*load(cfg,'smoke')

def test_zero_initialization_deterministic_and_empty_Adam(source):
    cfg,f,c=source
    from tbps.alternating.native_sources import strict_import_preflight
    strict_import_preflight(c)
    a=fresh_bundle(f,c);b=fresh_bundle(f,c)
    assert tree_sha(a.state)==tree_sha(b.state)
    assert a.state['cursor']=={'round':0,'g_updates':0,'h_visits':0,'h_updates':0}
    assert not a.state['G_Adam']['state'] and not a.state['H_Adam']['state']
    assert a.state['G_RNG']['cuda'] is None and a.state['H_RNG']['cuda'] is None
    assert torch.equal(a.sampler.get_state(),torch.Generator().manual_seed(42).get_state())
    assert a.report['trained_checkpoint_loads']==0 and not getattr(a.h,'word_budget1',False)

def test_fresh_original_policy_can_edit_multiple_words(source):
    cfg,f,c=source
    from tbps.alternating.native_sources import strict_import_preflight
    strict_import_preflight(c);b=fresh_bundle(f,c)
    proof=dynamic_budget_probe(b,c)
    assert proof['K']>=2 and len(proof['changed_word_positions'])>=2
    assert proof['action']['budget_k']>=len(proof['action']['edits'])
    assert not getattr(b.h,'word_budget1',False)

@pytest.mark.parametrize('old',[
 '/home/lzf/ldx/outputs/tbps-framework-gv2-epoch60-ablation/train/checkpoints/epoch_0060/checkpoint.pt',
 '/home/lzf/ldx/projects/x/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoint.pt',
 '/home/lzf/ldx/projects/x/attribute_text_policy_h_v1_20261015/outputs/formal_v1/training/checkpoint.pt',
 '/home/lzf/ldx/outputs/tbps-framework-hv1-g60-adaptation-epoch60/training/checkpoint.pt',
 '/home/lzf/ldx/outputs/tbps-framework-gh-alternating-v1-20261106/pilot_v1/state.pt'])
def test_historical_weight_load_guard(old,monkeypatch):
    monkeypatch.setattr(torch,'load',lambda *a,**k:'read')
    with no_historical_weights('/home/lzf/ldx/outputs/tbps-framework-gh-alternating-v2-20261107'):
        with pytest.raises(ValueError,match='forbidden'):torch.load(old)

def test_new_own_checkpoint_allowed_without_reset(monkeypatch):
    monkeypatch.setattr(torch,'load',lambda *a,**k:'read')
    base='/home/lzf/ldx/outputs/tbps-framework-gh-alternating-v2-20261107'
    with no_historical_weights(base) as events:assert torch.load(base+'/train/checkpoints/round_10/G.pt')=='read'
    assert len(events)==1

def test_source_protocol_removes_historical_parents_and_Scheduler(source):
    cfg,f,c=source
    assert not {'G60','H10','old_G60_bank_audit'} & set(c['sources'])
    assert not {'parent_sha256','historical_config','g_checkpoint_metadata'} & set(c['prototype'])
    assert f['g_scheduler'] is f['h_scheduler'] is None
    assert f['test_rounds']==[10,20,30,40,50,60] and f['rounds']==60
    assert c['training_configuration']['g_passes_per_round']==c['training_configuration']['h_passes_per_round']==1

def fake_backend():
    spec=importlib.util.spec_from_file_location('gh_fresh_fixture',Path(__file__).with_name('test_gh_formal_b.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    class B(m.Fake):
        def __init__(self,root):
            super().__init__(root);self.training={'g_passes_per_round':1,'h_passes_per_round':1,'checkpoint_interval_rounds':10}
            self.gs=NoScheduler();self.hs=NoScheduler();self.progress=lambda *a:None
        def complete_state(self,i):
            x=super().complete_state(i);x.update(schedulers={'G':None,'H':None},training_configuration=self.training);return x
        def restore_complete(self,s):super().restore_complete({k:v for k,v in s.items() if k not in ('schedulers','training_configuration')})
        def orders(self,*a):return ['x','y']
        def frozen_phase(self,n):
            model,opt=(self.h,self.ha) if n=='H' else (self.g,self.ga)
            return tree_sha(model.state_dict()),tree_sha(opt.state_dict())
    return B

def fixture_run(tmp_path,rounds=10,stop=None):
    base=tmp_path/'tbps-framework-gh-alternating-v2-20261107';root=base/'train';b=fake_backend()(root)
    s=b.complete_state(0);j=FreshJournal(root,'a'*64,base);j.initialize({},s)
    result,done=run_rounds(b,j,s,rounds,stop_after=stop)
    return b,j,result,done

def test_rolling_boundaries_and_immutable_paired_exports(tmp_path):
    b,j,s,done=fixture_run(tmp_path,20)
    assert done and s['cursor']=={'round':20,'g_updates':40,'h_visits':40,'h_updates':40}
    assert [p.name for p in (j.run/'checkpoints').iterdir() if p.is_dir()]==['round_10','round_20']
    assert (j.root/'round_0020/state.pt').is_file() and not (j.root/'round_0019').exists()
    assert (j.root/'retired_0019/retirement.json').is_file()
    assert not (j.run/'B/bank_19').exists() and (j.run/'B/bank_20/completed/features.pt').is_file()
    for i in (10,20):
        cp=ArtifactStore(j.run/'checkpoints',new_base=j.base);cp.verify(f'round_{i:02d}')
        for side in ('G','H'):
            x=torch.load(j.run/'checkpoints'/f'round_{i:02d}'/(side+'.pt'),map_location='cpu',weights_only=False)
            assert x['round']==i and x['model']==side and x['configuration_sha256']=='a'*64

def test_boundary_resume_preserves_full_state_and_counts(tmp_path):
    a,j,s,done=fixture_run(tmp_path/'split',10,stop=1);assert not done
    value,ledger=j.load_latest(a.parents)
    b=fake_backend()(a.root);b.resume_ledger=ledger
    # Fixture计数不参与模型更新；真实FreshBackend从ledger恢复计数/当前Bank。
    b.counts.update(ledger['forward_counts_cumulative'])
    resumed,done=run_rounds(b,j,value,10,start=1)
    _,_,expected,_=fixture_run(tmp_path/'whole',10)
    assert done and tree_sha(resumed)==tree_sha(expected)

def test_later_claim_blocks_replay(tmp_path):
    b,j,s,_=fixture_run(tmp_path,10,stop=1);j.begin(2,s)
    with pytest.raises(ValueError,match='claim'):j.load_latest(b.parents)

def test_real_failure_blocks_resume(tmp_path):
    b,j,s,_=fixture_run(tmp_path,10,stop=1);(j.run/'FAIL_initial').write_text('kept')
    with pytest.raises(ValueError,match='failure'):j.load_latest(b.parents)

def test_corrupt_latest_state_rejected(tmp_path):
    b,j,s,_=fixture_run(tmp_path,10,stop=1);(j.root/'round_0001/state.pt').write_bytes(b'bad')
    with pytest.raises(ValueError):j.load_latest(b.parents)

def test_prune_cannot_touch_history_or_run_root(tmp_path):
    base=tmp_path/'tbps-framework-gh-alternating-v2-20261107';root=base/'train';root.mkdir(parents=True)
    history=tmp_path/'history';history.mkdir();(history/'keep').write_text('safe')
    for target in (history,root):
        with pytest.raises(ValueError):prune_owned(target,root)
    assert (history/'keep').read_text()=='safe'

def test_Joint_only_native_engine_fixture(tmp_path):
    from dataclasses import asdict
    from tbps.contracts import DatasetManifest,GalleryItem,QueryItem,PersonID,AttackBudget,ImageBudget,TextBudget
    from tbps.models.base import ModelSpace,InputBinding
    from tbps.engines.materialized_cache import FeatureCache
    from tbps.engines.provenance import AttackOrigin
    from tbps.alternating.fresh_evaluation import joint_score
    from tbps.alternating.formal_evaluation import utf8
    base=tmp_path/'tbps-framework-joint';base.mkdir();a=base/'annotation';a.write_text('fixture');ck=base/'fixed_model';ck.write_text('no model')
    rf=lambda p:FileRef(str(p),sha_file(p))
    gs=tuple(GalleryItem('g'+str(i),str(i)+'.png',digest(['orig',i]),PersonID('cuhk_pedes_fixture',str(i//2)),'test') for i in range(10))
    qs=tuple(QueryItem('q'+str(i),'caption '+str(i),0,'g'+str(i),gs[i].pid,'test') for i in range(10))
    m=DatasetManifest('cuhk_pedes_fixture','test',rf(a),gs,qs);space=ModelSpace('irra_fixture',sha_file(ck),*(digest(i) for i in range(4)),'fp32/math/no_tf32')
    st=ArtifactStore(base/'features',new_base=base);p={'fixture':True};st.publish('protocol',{'protocol.json':json_bytes(p)},{})
    cache=FeatureCache(st,p,space);bindings={};origins={};originals={'image':tuple(g.image_sha256 for g in gs),'text':tuple(utf8(q.caption) for q in qs)}
    for family,modality,ids in [('adv_gallery','image',tuple(g.image_id for g in gs)),('adv_text','text',tuple(q.query_id for q in qs))]:
        b=InputBinding(space.id,digest(asdict(m)),tuple(digest([family,i]) for i in range(10)),modality,ids);bindings[family]=b
        cache.publish(family,b,0,10,torch.eye(10,512))
        budget=AttackBudget(image=ImageBudget('linf',8/255,'RGB [0,1]')) if modality=='image' else AttackBudget(text=TextBudget('unicode_character',None,'Natural-K','original',1,1))
        origins[family]=AttackOrigin('fixture-'+family,rf(ck),rf(a),budget,digest('source'))
    result=joint_score(m,st,p,space,rf(ck),bindings,origins,originals,ArtifactStore(base/'result',new_base=base),fixture=True)
    assert list(result['groups'])==['Joint'] and result['groups']['Joint']['R@1']==100
    assert result['drop_rate_percent']=={}


def test_sixty_round_forward_caps_include_all_banks(source):
    cfg,f,c=source
    _,train=load(cfg,'train')
    assert train['smoke']['max_G_forwards']==train['smoke']['max_IRRA_images']==484080+122880==606960
    assert train['smoke']['max_H_forwards']==60*2037*3
    assert train['smoke']['max_JVPs']==60*6020
    assert c['smoke']['max_G_forwards']==27+2048
