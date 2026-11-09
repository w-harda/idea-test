"""新目标符号、真实原生G/H CPU适配及G1→H1→G2来源链。"""
import copy,hashlib,importlib.util,math
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch.nn import functional as F
from tbps.artifacts import ArtifactStore,ArtifactError,read_json,sha_file
from tbps.attraction.objectives import Settings,ProfileKL
from tbps.attraction.g_step import attraction_g
from tbps.attraction.backend import AttractionBackend
from tbps.alternating.state import tree_sha
ROOT=Path(__file__).resolve().parents[2]
CFG=ROOT/'configs/framework/gh_attraction_v1.server.json'

@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    def deny(*a,**kw):raise AssertionError('CPU test attempted CUDA')
    monkeypatch.setattr(torch.cuda,'init',deny);monkeypatch.setattr(torch.cuda,'_lazy_init',deny)
    torch.set_num_threads(2)
    yield
    assert not torch.cuda.is_initialized()

@pytest.mark.parametrize('pull,push',[(1.,0.),(0.,1.),(1.,1.)])
def test_gradient_pulls_toward_partner_and_pushes_from_clean(pull,push):
    p=ProfileKL(torch.tensor([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]],dtype=torch.float64),Settings(.5,pull,push))
    clean=torch.tensor([[1.,0.]],dtype=torch.float64,requires_grad=True)
    partner=torch.tensor([[0.,1.]],dtype=torch.float64,requires_grad=True)
    current=torch.tensor([[.9,.1]],dtype=torch.float64,requires_grad=True)
    before=p.terms(current,clean,partner);before['loss'].mean().backward()
    after=p.terms(current.detach()-.001*current.grad,clean,partner)
    if pull:assert after['pull_kl'].item()<before['pull_kl'].item()
    if push:assert after['push_kl'].item()>before['push_kl'].item()
    assert clean.grad is partner.grad is None and not p.anchors.requires_grad

def test_noop_exact_zero_and_KL_direction():
    p=ProfileKL(torch.eye(3),Settings(.5));clean=torch.tensor([[.8,.2,.1]]);partner=torch.tensor([[.1,.8,.2]])
    values=p.terms(clean,clean,partner)
    assert values['reward'].item()==values['loss'].item()==values['push_kl'].item()==0.
    lp=p.log_probs(clean);lq=p.log_probs(partner)
    assert torch.allclose(values['pull_kl'],(lq.exp()*(lq-lp)).sum(1))

@pytest.mark.parametrize('cfg',[{'temperature':0.},{'temperature':float('nan')},{'pull_weight':-1.},{'pull_weight':0.,'push_weight':0.}])
def test_invalid_settings_rejected(cfg):
    with pytest.raises(ArtifactError):Settings(**cfg)

def test_invalid_anchor_and_pair_rejected():
    with pytest.raises(ArtifactError):ProfileKL(torch.zeros(3,4),Settings())
    p=ProfileKL(torch.eye(3),Settings())
    with pytest.raises(ArtifactError):p.terms(torch.ones(1,2),torch.ones(1,2),torch.ones(1,2))
    with pytest.raises(ArtifactError):p.terms(torch.ones(1,3),torch.ones(2,3),torch.ones(1,3))

def test_candidate_profiles_invariant_to_common_rotation():
    anchors=torch.tensor([[1.,0.],[0.,1.],[-1.,0.]],dtype=torch.float64);c=math.sqrt(.5);q=torch.tensor([[c,-c],[c,c]],dtype=torch.float64)
    x=torch.tensor([[.8,.2]],dtype=torch.float64)
    a=ProfileKL(anchors,Settings());b=ProfileKL(anchors@q.T,Settings())
    assert torch.allclose(a.log_probs(x),b.log_probs(x@q.T),atol=1e-12)

@pytest.fixture(scope='module')
def native():
    from tbps.attraction.sources import load
    from tbps.alternating.fresh_sources import fresh_bundle
    from tbps.alternating.native_sources import native_session,strict_import_preflight
    f,c=load(CFG,'smoke');strict_import_preflight(c);bundle=fresh_bundle(f,c)
    with native_session(c,reuse=bundle.modules) as session:yield f,c,bundle,session

class TinyG(torch.nn.Module):
    def __init__(self):super().__init__();self.v=torch.nn.Parameter(torch.tensor([.001,-.001,.0005])[None,:,None,None])
    def forward(self,x):return self.v.tanh().expand_as(x)
class TinyImageEncoder(torch.nn.Module):
    def __init__(self):super().__init__();self.register_buffer('w',torch.randn(3,512))
    def forward(self,x):return F.normalize(x.mean((2,3))@self.w,dim=1)

@pytest.mark.parametrize('k',[0,1,3,7])
def test_native_G_adapter_preserves_K_plus_one_budget_and_teacher_freeze(native,tmp_path,k):
    f,c,bundle,session=native;bindings=bundle.bindings
    entries=read_json(c['prototype']['g_manifest'])['entries'];entry=next(e for e in entries if e['role']=='train' and e['K']==k)
    row=entry['row_id'];condition=torch.load(entry['tensor_path'],map_location='cpu',weights_only=True)
    b=AttractionBackend.__new__(AttractionBackend);b.c=c;b.p=c['prototype'];b.bundle=bundle;b.native=bindings
    b.g=TinyG();b.ga=torch.optim.Adam(b.g.parameters(),lr=.0002);b.h=bundle.h.eval().requires_grad_(False)
    b.encoder=TinyImageEncoder().eval();before_encoder=tree_sha(b.encoder.state_dict());before_h=tree_sha(b.h.state_dict())
    b.grant=SimpleNamespace(validate=lambda *a:None);b.cursor={'g_updates':0}
    b.clean=F.normalize(torch.randn(2048,512),dim=1);b.anchor_sha=tree_sha(b.clean)
    b.profile_cpu=ProfileKL(b.clean,Settings(**c['attraction']));b.profile_g=b.profile_cpu
    b.image=lambda unused:torch.ones(1,3,8,8)*.5
    from tbps.alternating.conditions import LegacySteps
    serial=bindings.get('serial_training');before=(serial.attribute_losses,serial.joint_losses)
    b.steps=LegacySteps(serial.train_single_sample,None)
    from tbps.artifacts import digest
    ident={'original_caption_sha256':'a'*64,'context_caption_sha256':'a'*64,
        'selected_g_ids':list(condition['selected_instance_ids']),'scores':condition['S_raw'].tolist(),
        'source_space_sha256':'b'*64,'anchor_mapping_sha256':'c'*64}
    cond=SimpleNamespace(payload=condition,identity=SimpleNamespace(key=digest(ident)))
    log=b.g_update(row,cond)
    assert log['optimizer_steps']==k+1==b.cursor['g_updates']
    assert tree_sha(b.h.state_dict())==before_h and tree_sha(b.encoder.state_dict())==before_encoder
    assert log['attraction']['teacher_H_state']==b.h_fingerprint()
    assert all(not p.requires_grad and p.grad is None for p in b.h.parameters())
    for rec in log['records']:
        terms=rec['attraction'];assert rec['cumulative_linf']<=8/255
        assert math.isclose(rec['global'],terms['loss'],rel_tol=1e-6,abs_tol=1e-6)
        assert math.isclose(rec['total'],rec['global']+rec['local']+rec['retain'],rel_tol=1e-6,abs_tol=1e-6)
    assert (serial.attribute_losses,serial.joint_losses)==before
    from tbps.attraction.audit_math import audit_attraction
    log['row_id']=row
    ledger={'G':[log],'H':[],'conditions':[{'row':row,'identity':ident}],'g_orders':[[row]],'h_orders':[[]],
        'G_before':log['G_before'],'G_after':log['G_after'],'condition_H_state':b.h_fingerprint(),
        'bank':{'ordered_train_ids':b.p['train_ids']}}
    report=audit_attraction(ledger,c['attraction'],None,None,b.clean,b.clean)
    assert report['G_steps_independently_recomputed']==k+1

def test_G_binding_restores_after_exception(native):
    _,_,bundle,_=native;s=bundle.bindings.get('serial_training');old=s.joint_losses
    with pytest.raises(RuntimeError):
        with attraction_g(s,None,None,None,[]):raise RuntimeError('injected')
    assert s.joint_losses is old

def test_native_H_three_heads_single_SCST_update_and_current_G_binding(native):
    f,c,bundle,session=native;bindings=bundle.bindings;h=bundle.h;h.requires_grad_(True)
    entries=read_json(Path(c['prototype']['history_root'])/'qualification/train.json')['entries'];found=None
    for e in entries[:64]:
        if not e['eligible']:continue
        ex=bindings.get('formal_training.data').example_from(e)
        with torch.no_grad():greedy=h.rollout(ex,'greedy').adversarial
        for seed in range(1000,1064):
            with torch.no_grad():out=h.rollout(ex,'sample',torch.Generator().manual_seed(seed))
            if out.edits and out.adversarial!=greedy:found=(e,ex,seed);break
        if found:break
    assert found is not None;e,ex,seed=found;rtmod=bindings.get('formal_training.runtime')
    class Runtime:
        def __init__(self):self.probe=bundle.probe;self.samples={'G':0,'clean':0,'adv':0,'text':0}
        def assert_frozen(self):pass
        def encode_text(self,texts):
            self.samples['text']+=len(texts);rows=[]
            for text in texts:
                seed=int.from_bytes(hashlib.sha256(text.encode()).digest()[:4],'little')
                rows.append(torch.randn(512,generator=torch.Generator().manual_seed(seed)))
            self.last_features=F.normalize(torch.stack(rows),dim=1)
            return self.last_features
    rt=Runtime();clean=F.normalize(torch.randn(4,512),dim=1);adv=F.normalize(clean+.03*torch.randn(4,512),dim=1)
    context=rtmod.TrainGallery(clean,adv,torch.tensor([True,False,False,False]),0,'train',('train',)*4,'train',('train',)*4)
    row=e['identity']['row_id'];query=SimpleNamespace(caption=ex.caption)
    b=AttractionBackend.__new__(AttractionBackend);b.c=c;b.p={'train_ids':[row,'fixture-1','fixture-2','fixture-3']};b.native=bindings
    b.grant=SimpleNamespace(validate=lambda *a:None);b.cursor={'h_visits':0,'h_updates':0};b.entries={row:e}
    b.h=h;b.ha=torch.optim.Adam(h.parameters(),lr=.0001);b.sampler=torch.Generator().manual_seed(seed)
    b.catalog=SimpleNamespace(get_train=lambda r:query);b.current_features=adv;b.clean=clean;b.anchor_sha=tree_sha(clean)
    b.profile_cpu=ProfileKL(clean,Settings(**c['attraction']))
    b.g=TinyG().requires_grad_(False);g_before=tree_sha(b.g.state_dict())
    bank=SimpleNamespace(g_state_sha256=b.g_fingerprint(),adversarial_sha256=tree_sha(adv));b.last_bank=bank
    b.runtime=rt;b.reward=SimpleNamespace(bind=lambda q,r,example,k:rtmod.FreshReward(context,q,example,k,r))
    keys=set(h.state_dict());log=b.h_update(row,bank)
    assert set(h.state_dict())==keys and all(hasattr(h,name) for name in ('attribute_head','position_head','replacement_head'))
    assert log['optimizer_steps_this_step']==1 and b.cursor=={'h_visits':1,'h_updates':1} and rt.samples['text']==3
    assert log['attraction']['teacher_G_state']==bank.g_state_sha256 and log['bank_sha256']==tree_sha(adv)
    assert log['attraction']['R_sim_optimized'] is log['attraction']['R_rank_optimized'] is False
    for mode in ('sampled','greedy'):
        reward=log[mode+'_reward'];assert math.isclose(reward['total'],reward['pull_gain']+reward['push_kl'],abs_tol=1e-6)
        action=log[mode];contracts=bindings.get('contracts');words=[contracts.word_at(ex.caption,v['position']) for v in action['edits']]
        assert len(words)==len(set(words)) and len(words)<=action['budget_k']
    assert math.isclose(log['SCST_loss'],-log['advantage']*log['sequence_log_prob'],rel_tol=2e-5,abs_tol=2e-6)
    assert tree_sha(b.g.state_dict())==g_before and all(v.grad is None for v in b.g.parameters())
    expected=b.profile_cpu.terms(rt.last_features[1:2],rt.last_features[:1],adv[:1])['reward'].item()
    assert math.isclose(log['sampled_reward']['total'],expected,rel_tol=1e-6,abs_tol=1e-6)
    assert log['gradient_norm']>0 and all(int(v['step'])==1 for v in b.ha.state.values())
    from tbps.attraction.audit_math import audit_attraction
    ledger={'G':[],'H':[log],'conditions':[],'g_orders':[[]],'h_orders':[[row]],
        'G_before':b.g_fingerprint(),'G_after':b.g_fingerprint(),'bank':{'ordered_train_ids':b.p['train_ids']}}
    report=audit_attraction(ledger,c['attraction'],None,None,clean,adv)
    assert report['H_steps_independently_recomputed']==1
    with pytest.raises(ArtifactError):b.h_update(row,SimpleNamespace(g_state_sha256='stale',adversarial_sha256='stale'))
    with torch.no_grad():b.g.v.add_(.001)
    with pytest.raises(ArtifactError,match='stale G parameters'):b.h_update(row,bank)


def test_two_round_schedule_G1_bank_to_H1_and_H1_to_G2(tmp_path):
    # 原调度器的控制流fixture；真实G K+1和离散H SCST分别由上面的原生适配测试覆盖。
    spec=importlib.util.spec_from_file_location('attraction_schedule_fixture',ROOT/'tests/framework/test_gh_alternating_v2.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture);Base=fixture.fake_backend()
    from tbps.alternating.configured_training import run_rounds
    from tbps.cooperative.store import FreshJournal
    class Backend(Base):
        def __init__(self,root):
            super().__init__(root);self.profile=ProfileKL(torch.tensor([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]]),Settings(.5))
            self.h_teacher_records=[]
        def g_update(self,row,condition):
            self.ga.zero_grad(set_to_none=True)
            current=torch.stack((self.g.weight.sum(),self.g.weight.new_tensor(1.)))[None]
            clean=current.new_tensor([[.1,1.]]);partner=current.new_tensor([[condition.payload['signal'],1.]])
            self.profile.terms(current,clean,partner)['loss'].mean().backward();self.ga.step()
            self.cursor['g_updates']+=1;self.counts['G']+=1
            return {'optimizer_steps':1}
        def h_update(self,row,bank):
            assert bank.g_state_sha256==self.g_fingerprint()
            self.h_teacher_records.append((self.i,bank.g_state_sha256,float(self.g.weight.detach().sum())))
            self.ha.zero_grad(set_to_none=True)
            current=torch.stack((self.h.weight.sum(),self.h.weight.new_tensor(1.)))[None]
            clean=current.new_tensor([[.2,1.]]);partner=torch.stack((self.g.weight.detach().sum(),self.g.weight.new_tensor(1.)))[None]
            self.profile.terms(current,clean,partner)['loss'].mean().backward();self.ha.step()
            self.cursor['h_visits']+=1;self.cursor['h_updates']+=1
            return {'optimizer_steps_this_step':1}
    base=tmp_path/'tbps-framework-gh-cooperative-v1-20261109';b=Backend(base/'train');parent=b.complete_state(0)
    journal=FreshJournal(base/'train','f'*64,base);journal.initialize({'fixture':True},parent)
    state,done=run_rounds(b,journal,parent,60,stop_after=2);assert not done and state['cursor']['round']==2
    from tbps.cooperative.store import read_ledger
    first=read_ledger(base/'train/ledgers/round_0001/ledger.json');second=read_ledger(base/'train/ledgers/round_0002/ledger.json')
    assert first['G_after']==second['G_before'] and first['H_after']==second['H_before']
    assert first['H_changed'] and first['G_changed']
    assert b.seen[0][2]!=b.seen[2][2]  # G2收到更新后的H1语境，而非H0。
    assert b.h_teacher_records[0][1]==first['G_after']
    assert b.h_teacher_records[2][1]==second['G_after']
