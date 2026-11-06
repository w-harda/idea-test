"""仅新预算/配置/调度/恢复定向CPU验证；不执行真实模型或CUDA。"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
import torch
from tbps.alternating.word_budget import check_words,word_spans,candidate_guard,install
from tbps.alternating.state import tree_sha,clone_tree
from tbps.alternating.configured_training import scheduler,configure_optimizer,run_rounds,EXTRA
from tbps.alternating.formal_state import FormalJournal

@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def deny(*a,**kw):raise AssertionError('CPU test attempted CUDA')
    monkeypatch.setattr(torch.cuda,'_lazy_init',deny);monkeypatch.setattr(torch.cuda,'init',deny)
    torch.set_num_threads(1)

@pytest.mark.parametrize('text', ['red red','blue-shirt, coat','a  b\tc','é coat',''])
def test_noop_budget(text):assert check_words(text,text)['changed_words']==0

def test_original_positions_not_regex_or_subwords():
    a='coat-shirt,  coat';b='cоat-shirt,  coat'
    assert word_spans(a)==((0,11),(13,17))
    assert check_words(a,b)['changed_word_indices']==[0]
    with pytest.raises(ValueError,match='exceeded'):check_words(a,'cоat-shirt,  cоat')

@pytest.mark.parametrize('adv',['coat shirt x','coatshirt','coat\tshirt','coat shіrт'])
def test_insert_delete_separator_or_two_words_rejected(adv):
    # 最后一例是两字符同词，check_words只计词；原Unicode replay另外限制每词一次。
    if adv=='coat shіrт':assert check_words('coat shirt',adv)['changed_words']==1
    else:
        with pytest.raises(ValueError):check_words('coat shirt',adv)

def test_candidate_guard_prevents_second_edit_and_restores_on_error():
    mod=SimpleNamespace(candidates=lambda *a:{1:('о',)})
    original=mod.candidates;ex=SimpleNamespace(caption='coat shirt')
    with pytest.raises(RuntimeError):
        with candidate_guard(mod):
            assert mod.candidates(ex,ex.caption,0,set(),None)=={1:('о',)}
            assert mod.candidates(ex,'cоat shirt',1,set(),None)=={}
            raise RuntimeError('fixture')
    assert mod.candidates is original

@pytest.fixture
def native(monkeypatch):
    legacy=Path('/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015')
    # 测试在独立进程内只加载原H的纯CPU模块；无真实IRRA/权重。
    monkeypatch.syspath_prepend(str(legacy))
    import policy,contracts,tokenizer_probe
    class ByteTokenizer:
        encoder={'<|startoftext|>':1001,'<|endoftext|>':1002}
        def encode(self,text):return [b+1 for b in text.encode()]
    torch.manual_seed(42)
    h=policy.CharacterPolicy(tokenizer_probe.TokenizerProbe(ByteTokenizer()),width=16,layers=1,heads=4)
    ex=contracts.TextExample('coat shirt',(contracts.AttributeSpan('garment_type','coat','upper','coat',0,4),
        contracts.AttributeSpan('garment_type','shirt','upper','shirt',5,10)))
    return h,ex,policy

def test_native_policy_sample_greedy_natural_k_and_optimizer_identity(native):
    h,ex,mod=native;opt=torch.optim.Adam(h.parameters(),lr=.01)
    ids=[id(p) for p in h.parameters()];before=tree_sha(h.state_dict());old_candidates=mod.candidates
    install(h,mod)
    assert ids==[id(p) for p in h.parameters()] and before==tree_sha(h.state_dict())
    assert {id(p) for g in opt.param_groups for p in g['params']}==set(ids)
    for seed in range(20):
        for mode in ('sample','greedy'):
            out=h.rollout(ex,mode,torch.Generator().manual_seed(seed))
            assert out.budget_k==2 and len(out.edits)<=1
            check_words(ex.caption,out.adversarial)
            assert out.log_prob.requires_grad
    assert mod.candidates is old_candidates

def test_original_scst_all_three_heads_backpropagate(native):
    h,ex,mod=native;install(h,mod)
    for seed in range(100):
        out=h.rollout(ex,'sample',torch.Generator().manual_seed(seed))
        if out.edits:break
    assert out.edits
    import scst
    loss,adv=scst.scst_loss(out.log_prob,2.,.5);loss.backward()
    assert float(adv)==1.5
    for head in ('attribute_head','position_head','replacement_head'):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in getattr(h,head).parameters())

def test_changing_H_state_changes_legal_attack(native):
    h,ex,mod=native;install(h,mod)
    # 原三Head均确定性选择首个合法动作，只干预已有replacement Head。
    with torch.no_grad():
        for head in (h.attribute_head,h.position_head,h.replacement_head):
            head.weight.zero_();head.bias.zero_()
    a=h.attack(ex)
    assert a.edits
    first=a.edits[0].replacement_char
    options=mod.CONFUSABLES if hasattr(mod,'CONFUSABLES') else __import__('constraints').CONFUSABLES
    other=next(x for x in options[a.edits[0].original_char] if x!=first)
    with torch.no_grad():h.replacement_head.bias[mod.REPLACEMENTS.index(other)]=10.
    b=h.attack(ex)
    assert a.adversarial!=b.adversarial and check_words(ex.caption,b.adversarial)['changed_words']==1

def test_Scheduler_restore_and_configured_LR_preserve_Adam():
    p=torch.nn.Parameter(torch.tensor(1.));o=torch.optim.Adam([p],lr=.1)
    p.square().backward();o.step();mom=clone_tree(o.state_dict()['state'])
    configure_optimizer(o,.02);assert tree_sha(mom)==tree_sha(o.state_dict()['state'])
    s=scheduler(o,{'kind':'step','step_size':1,'gamma':.5});o.zero_grad();p.square().backward();o.step();s.step()
    t=scheduler(o,{'kind':'step','step_size':1,'gamma':.5});t.load_state_dict(s.state_dict())
    assert tree_sha(s.state_dict())==tree_sha(t.state_dict())

def fake_class():
    spec=importlib.util.spec_from_file_location('gh_formal_cpu_fixture',Path(__file__).with_name('test_gh_formal_b.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    class B(mod.Fake):
        def __init__(self,root,gp=1,hp=1):
            super().__init__(root);self.training={'g_passes_per_round':gp,'h_passes_per_round':hp,'checkpoint_interval_rounds':2}
            self.gs=scheduler(self.ga,{'kind':'constant','step_size':1,'gamma':1.})
            self.hs=scheduler(self.ha,{'kind':'constant','step_size':1,'gamma':1.})
            self.progress=lambda *a:None
        def complete_state(self,i):
            x=super().complete_state(i);x.update(schedulers={'G':clone_tree(self.gs.state_dict()),'H':clone_tree(self.hs.state_dict())},training_configuration=clone_tree(self.training));return x
        def restore_complete(self,s):
            super().restore_complete({k:v for k,v in s.items() if k not in EXTRA})
            self.gs.load_state_dict(s['schedulers']['G']);self.hs.load_state_dict(s['schedulers']['H'])
        def orders(self,i,side,p):return ['x','y'] if side=='G' else ['y','x']
        def frozen_phase(self,n):
            m,o=(self.h,self.ha) if n=='H' else (self.g,self.ga)
            return tree_sha(m.state_dict()),tree_sha(o.state_dict())
        def h_update(self,row,b):
            out=super().h_update(row,b);out['row_id']=row;return out
    return B

def launch(base,gp=1,hp=1,stop=None):
    B=fake_class();base=base/'tbps-framework-v1';b=B(base/'banks',gp,hp);s=b.complete_state(0)
    j=FormalJournal(base/'rounds','a'*64,base);j.initialize({},s)
    result,done=run_rounds(b,j,s,2,stop_after=stop);return b,j,result,done

def test_configurable_unequal_pass_counts_and_state_carry(tmp_path):
    b,j,s,done=launch(tmp_path,2,3)
    assert done and s['cursor']=={'round':2,'g_updates':8,'h_visits':12,'h_updates':12}
    assert b.seen[0][2]!=b.seen[4][2] and b.restore_calls==1
    assert s['schedulers']['G']['last_epoch']==s['schedulers']['H']['last_epoch']==2
    assert (j.root/'checkpoint_0002/checkpoint.json').exists()

def test_continuous_equals_complete_boundary_resume_CPU(tmp_path):
    a,j,s,done=launch(tmp_path/'split',stop=1);assert not done
    value,ledger=j.load_boundary(1,a.parents);j.record_resume(1)
    b=fake_class()(a.root);result,done=run_rounds(b,j,value,2,start=1)
    _,_,expected,_=launch(tmp_path/'continuous')
    assert done and tree_sha(result)==tree_sha(expected)

def test_later_claim_prevents_replay(tmp_path):
    b,j,s,_=launch(tmp_path,stop=1);j.begin(2,s)
    with pytest.raises(ValueError,match='Claim'):j.load_boundary(1,b.parents)

def test_GPU_free_memory_is_observation_only(monkeypatch,tmp_path):
    from tbps.alternating.configured_cli import resources
    import subprocess
    def output(args,**k):return 'GPU, 0, 24576, 24576, 100' if '--id=0' in args else '123, external, 24576'
    monkeypatch.setattr(subprocess,'check_output',output)
    f={'resource_probe_directory':str(tmp_path),'resources':{'pilot':{'minimum_ram_bytes':0,'minimum_disk_bytes':0,'minimum_free_inodes':0}}}
    r=resources(f,'pilot');assert r['resource_eligible'] and 'GPU' not in r['checks']


@pytest.mark.parametrize('damage',[None,'word','bank','state'])
def test_new_independent_audit_full_flow_and_tamper_refusal(tmp_path,monkeypatch,damage):
    import json
    from tbps.artifacts import ArtifactStore,json_bytes,FileRef,sha_file,digest
    from tbps.alternating.formal_state import tensor_bytes
    from tbps.alternating.native_sources import source_key
    from tbps.alternating import configured_audit as au
    from contextlib import contextmanager
    B=fake_class();base=tmp_path/'tbps-framework-audit-v1';root=base/'pilot'
    b=B(root)
    # CPU Fixture仅构造可审计的CUDA RNG字段，不初始化/恢复CUDA。
    for value in b.rng.values():value['cuda']=[torch.zeros(8,dtype=torch.uint8)]
    b.c={'output_base':str(base),'prototype':b.p,'alternating_configuration':b.training}
    c=b.c
    original_condition=b.condition
    def condition(arm,row,trace):
        result=original_condition(arm,row,trace);md=b.conditions_log[-1]
        from tbps.alternating.contracts import text_sha
        md['identity']['original_caption_sha256']=md['identity']['context_caption_sha256']=text_sha('coat')
        j=len(b.conditions_log)-1
        io=ArtifactStore(root/'round_io'/f'round_{b.i:04d}',new_base=base)
        act={'original':'coat','adversarial':'coat','attributes':[],'edits':[],'budget_k':0}
        io.publish(f'rollout_{j:04d}',{'action.json':json_bytes({'original':'coat','adversarial':'coat','edits':[],
            'native_action':act})},{'source':source_key(c)})
        io.publish(f'condition_{j:04d}',{'condition.pt':tensor_bytes({'K':0,'fixture':torch.ones(1)}),
            'evidence.json':json_bytes(md)},{'source':source_key(c),'condition':digest(md['identity'])})
        return result
    b.condition=condition
    old_h=b.h_update
    def h_update(row,bank):
        out=old_h(row,bank);action={'original':'coat','adversarial':'coat','attributes':[],'edits':[],'budget_k':0}
        out.update(bank_sha256=bank.adversarial_sha256,sampled=action.copy(),greedy=action.copy(),
            four_mode_paired_similarity={'sampled':{},'greedy':{}})
        return out
    b.h_update=h_update
    parent=b.complete_state(0);j=FormalJournal(root/'rounds','x',base);j.initialize({},parent)
    store=ArtifactStore(root,new_base=base);store.publish('initial_state',{'state.pt':tensor_bytes(parent)},{})
    final,done=run_rounds(b,j,parent,2)
    cfg=base/'config.json';cfg.write_text('{}')
    store.publish('generated',{'report.json':json_bytes({'complete':True,'frozen_sources':True,
        'configuration_sha256':sha_file(cfg),'counts':{'G':b.counts['G'],'IRRA_image':b.counts['G']}})},{})
    f={'outputs':{'pilot':str(root)},'audits':{'pilot':str(base/'audit')},'output_base':str(base),
        'g_passes_per_round':1,'h_passes_per_round':1}
    monkeypatch.setattr(au,'load',lambda *a:(f,c))
    # 使用原H纯数据/回放函数，无神经执行；test_native fixture之后可能已导入它们。
    import sys,importlib
    legacy=Path('/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015')
    monkeypatch.syspath_prepend(str(legacy))
    @contextmanager
    def native_session(c):
        yield SimpleNamespace(import_module=importlib.import_module)
    monkeypatch.setattr('tbps.alternating.native_sources.native_session',native_session)
    if damage=='word':
        p=root/'round_io/round_0001/rollout_0000/action.json';p.write_text('{}')
    elif damage=='bank':
        p=root/'B/bank_1/completed/features.pt';p.write_bytes(b'corrupt')
    elif damage=='state':
        p=root/'rounds/round_0001/state.pt';p.write_bytes(b'corrupt')
    if damage:
        with pytest.raises((ValueError,Exception)):au.audit(cfg,'pilot')
    else:
        # READY发布需配置中给出固定路径。
        f['ready_receipt']=str(base/'readiness/ready.json')
        out=au.audit(cfg,'pilot')
        assert out['status']=='GH_ALTERNATING_V1_PILOT_CPU_AUDIT_PASS'
        assert out['word_budget']['checked_outputs']==12
