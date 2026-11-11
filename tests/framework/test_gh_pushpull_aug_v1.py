"""三视图输入/真实G K+1/H三头SCST、独立数值和预算审计。"""
import copy,hashlib,math,importlib.util,json
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
import pytest
import torch
from torch.nn import functional as F
from tbps.artifacts import ArtifactError,ArtifactStore,FileRef,read_json,sha_file,json_bytes,digest
from tbps.alternating.state import tree_sha
from tbps.augmented_attraction.objectives import Settings,ProfileKL,scalars
from tbps.augmented_attraction.views import ViewSettings,recipe,apply_views,transform,OPS
from tbps.augmented_attraction.backend import AttractionBackend
from tbps.augmented_attraction.audit_math import audit_attraction
ROOT=Path(__file__).resolve().parents[2];CFG=ROOT/'configs/framework/gh_pushpull_aug_v1.server.json'

@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    def deny(*a,**k):raise AssertionError('CPU verification attempted CUDA')
    monkeypatch.setattr(torch.cuda,'_lazy_init',deny);torch.set_num_threads(2)
    yield
    assert not torch.cuda.is_initialized()

def bindings(ids,bank,round_index=1):
    return [{'row':row,'input_sha256':'e'*64,'recipe_sha256':digest(recipe(ViewSettings(),row,round_index,384,128)),
        'feature_sha256':tree_sha(bank[i]),'view_linf':[.01,.01,.01]} for i,row in enumerate(ids)]

def test_views_replay_shape_original_exact_rng_isolation_and_differentiability():
    torch.manual_seed(17);x=torch.rand(1,3,384,128,requires_grad=True);rng=torch.get_rng_state().clone()
    p=recipe(ViewSettings(),'train-row',1,384,128);v=apply_views(x,p)
    assert v.shape==(3,3,384,128) and torch.equal(x,v[:1]) and torch.equal(v,apply_views(x,p))
    assert torch.equal(torch.get_rng_state(),rng) and p!=recipe(ViewSettings(),'train-row',2,384,128)
    assert all(set(row['permutation'])=={0,1} and row['row']==i for i,row in enumerate(p['local']))
    assert not torch.equal(v[1:],v[:1].expand(2,-1,-1,-1))
    v.square().mean().backward();assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.norm()>0

@pytest.mark.parametrize('kind',OPS)
def test_each_transform_preserves_RGB_domain_and_shape(kind):
    x=torch.rand(1,3,96,64)
    op={'kind':kind,'sigma':.7,'degrees':3.,'dx':.03,'dy':-.02,'size':[85,56],'padding':[4,4,5,6]}
    v=transform(x,op);assert v.shape==x.shape and torch.isfinite(v).all() and v.min()>=0 and v.max()<=1

def test_recipe_tamper_and_invalid_configuration_rejected():
    p=recipe(ViewSettings(),'train-row',1,384,128);p['global'][0]['kind']='vflip'
    with pytest.raises(ArtifactError):apply_views(torch.rand(1,3,384,128),p)
    with pytest.raises(ArtifactError):ViewSettings(views=20)

@pytest.fixture(scope='module')
def native():
    from tbps.augmented_attraction.sources import load
    from tbps.alternating.fresh_sources import fresh_bundle
    from tbps.alternating.native_sources import native_session,strict_import_preflight
    f,c=load(CFG,'smoke');strict_import_preflight(c);b=fresh_bundle(f,c)
    with native_session(c,reuse=b.modules) as session:yield f,c,b,session

class TinyG(torch.nn.Module):
    def __init__(self):super().__init__();self.v=torch.nn.Parameter(torch.tensor([.001,-.001,.0005])[None,:,None,None])
    def forward(self,x):
        out=self.v.tanh().expand_as(x)
        if out.requires_grad:out.retain_grad()
        self.last=out
        return out
class TinyEncoder(torch.nn.Module):
    def __init__(self):super().__init__();self.register_buffer('w',torch.randn(3,512))
    def forward(self,x):return F.normalize(x.mean((2,3))@self.w,dim=1)

@pytest.mark.parametrize('k',[0,1,3,7])
def test_real_G_K_plus_one_all_three_views_receive_gradient_and_frozen_teachers(native,k):
    f,c,bundle,session=native;entries=read_json(c['prototype']['g_manifest'])['entries'];entry=next(e for e in entries if e['role']=='train' and e['K']==k)
    row=entry['row_id'];condition=torch.load(entry['tensor_path'],map_location='cpu',weights_only=True)
    b=AttractionBackend.__new__(AttractionBackend);b.c=c;b.p=c['prototype'];b.bundle=bundle;b.native=bundle.bindings
    b.g=TinyG();b.ga=torch.optim.Adam(b.g.parameters(),lr=.0002);b.h=bundle.h.eval().requires_grad_(False)
    b.encoder=TinyEncoder().eval();enc=tree_sha(b.encoder.state_dict());teacher=tree_sha(b.h.state_dict())
    b.grant=SimpleNamespace(validate=lambda *a:None);b.cursor={'g_updates':0};b.round_index=1;b.view_settings=ViewSettings()
    b.clean=F.normalize(torch.randn(2048,512),dim=1);b.anchor_sha=tree_sha(b.clean);b.profile_g=ProfileKL(b.clean,Settings())
    x=.2+.6*torch.rand(1,3,384,128);b.image=lambda _:x
    ident={'original_caption_sha256':'a'*64,'context_caption_sha256':'b'*64,'selected_g_ids':list(condition['selected_instance_ids']),'scores':condition['S_raw'].tolist(),'source_space_sha256':'c'*64,'anchor_mapping_sha256':'d'*64}
    cond=SimpleNamespace(payload=condition,identity=SimpleNamespace(key=digest(ident)));log=b.g_update(row,cond);log['row_id']=row
    assert log['optimizer_steps']==k+1==b.cursor['g_updates'] and log['G_input_view_samples']==3*(k+1)
    assert tree_sha(b.h.state_dict())==teacher and tree_sha(b.encoder.state_dict())==enc
    assert all(p.grad is None and not p.requires_grad for p in b.h.parameters())
    assert all(b.g.last.grad[j].norm()>0 for j in range(3))
    for rec in log['records']:
        v=rec['attraction'];assert len(v['view_linf'])==3 and max(v['view_linf'])<=8/255
        assert math.isclose(rec['global'],sum(x['loss'] for x in v['per_view'])/3,abs_tol=1e-5)
        assert math.isclose(rec['total'],rec['global']+rec['local']+rec['retain'],abs_tol=1e-5)
    bank=b.clean[:,None].expand(-1,3,-1).clone();ins=bindings(b.p['train_ids'],bank);ins[b.p['train_ids'].index(row)]['input_sha256']=log['attraction']['clean_view_sha256']
    ledger={'round':1,'G':[log],'H':[],'conditions':[{'row':row,'identity':ident}],'g_orders':[[row]],'h_orders':[[]],
        'G_before':log['G_before'],'G_after':log['G_after'],'condition_H_state':b.h_fingerprint(),'bank':{'ordered_train_ids':b.p['train_ids']},'bank_view_inputs':ins}
    report=audit_attraction(ledger,c['attraction'],None,None,b.clean,b.clean,bank,c['augmentation'])
    assert report['G_view_terms_independently_recomputed']==3*(k+1)


def test_native_H_three_heads_mean_three_current_G_partners_and_scst_once(native):
    f,c,bundle,session=native;h=copy.deepcopy(bundle.h).requires_grad_(True);m=bundle.bindings
    entries=read_json(Path(c['prototype']['history_root'])/'qualification/train.json')['entries'];found=None
    for e in entries[:64]:
        if not e['eligible']:continue
        ex=m.get('formal_training.data').example_from(e)
        with torch.no_grad():greedy=h.rollout(ex,'greedy').adversarial
        for seed in range(1000,1064):
            with torch.no_grad():out=h.rollout(ex,'sample',torch.Generator().manual_seed(seed))
            if out.edits and out.adversarial!=greedy:found=(e,ex,seed);break
        if found:break
    assert found is not None;e,ex,seed=found;rtmod=m.get('formal_training.runtime')
    class Runtime:
        def __init__(self):self.probe=bundle.probe;self.samples={'G':0,'clean':0,'adv':0,'text':0}
        def assert_frozen(self):pass
        def encode_text(self,texts):
            self.samples['text']+=len(texts);rows=[]
            for text in texts:
                seed=int.from_bytes(hashlib.sha256(text.encode()).digest()[:4],'little');rows.append(torch.randn(512,generator=torch.Generator().manual_seed(seed)))
            self.last_features=F.normalize(torch.stack(rows),dim=1);return self.last_features
    rt=Runtime();clean=F.normalize(torch.randn(4,512),dim=1);views=F.normalize(torch.randn(4,3,512),dim=-1);adv=views[:,0].clone()
    context=rtmod.TrainGallery(clean,adv,torch.tensor([True,False,False,False]),0,'train',('train',)*4,'train',('train',)*4)
    row=e['identity']['row_id'];query=SimpleNamespace(caption=ex.caption);b=AttractionBackend.__new__(AttractionBackend)
    b.c=c;b.p={'train_ids':[row,'fixture-1','fixture-2','fixture-3']};b.native=m;b.grant=SimpleNamespace(validate=lambda *a:None);b.cursor={'h_visits':0,'h_updates':0};b.entries={row:e}
    b.h=h;b.ha=torch.optim.Adam(h.parameters(),lr=.0001);b.sampler=torch.Generator().manual_seed(seed);b.catalog=SimpleNamespace(get_train=lambda r:query)
    b.current_features=adv;b.current_view_features=views;b.view_bank_sha=tree_sha(views);b.clean=clean;b.anchor_sha=tree_sha(clean);b.profile_cpu=ProfileKL(clean,Settings())
    b.g=TinyG().requires_grad_(False);before=tree_sha(b.g.state_dict());bank=SimpleNamespace(g_state_sha256=b.g_fingerprint(),adversarial_sha256=tree_sha(adv));b.last_bank=bank
    b.runtime=rt;b.reward=SimpleNamespace(bind=lambda q,r,example,k:rtmod.FreshReward(context,q,example,k,r))
    log=b.h_update(row,bank)
    assert all(hasattr(h,name) for name in ('attribute_head','position_head','replacement_head'))
    assert log['optimizer_steps_this_step']==1 and b.cursor=={'h_visits':1,'h_updates':1} and rt.samples['text']==3
    assert tree_sha(b.g.state_dict())==before and all(p.grad is None for p in b.g.parameters())
    expected=b.profile_cpu.terms(rt.last_features[1:2].expand(3,-1),rt.last_features[:1].expand(3,-1),views[0])['reward'].mean().item()
    assert math.isclose(log['sampled_reward']['total'],expected,abs_tol=1e-6) and log['gradient_norm']>0
    for name in ('sampled','greedy'):
        action=log[name];words=[m.get('contracts').word_at(ex.caption,x['position']) for x in action['edits']];assert len(words)==len(set(words)) and len(words)<=action['budget_k']
    ledger={'round':1,'G':[],'H':[log],'conditions':[],'g_orders':[[]],'h_orders':[[row]],'G_before':before,'G_after':before,'bank':{'ordered_train_ids':b.p['train_ids']},'bank_view_inputs':bindings(b.p['train_ids'],views)}
    report=audit_attraction(ledger,c['attraction'],None,None,clean,adv,views,c['augmentation']);assert report['H_view_terms_independently_recomputed']==6
    with pytest.raises(ArtifactError):b.h_update(row,SimpleNamespace(g_state_sha256='stale',adversarial_sha256='stale'))
    with torch.no_grad():b.g.v.add_(.001)
    with pytest.raises(ArtifactError,match='stale G parameters'):b.h_update(row,bank)


def fixture_ledger():
    torch.manual_seed(47);anchors=F.normalize(torch.randn(8,4),dim=1);views=F.normalize(torch.randn(8,3,4),dim=-1);bank=views[:,0].clone();texts=F.normalize(torch.randn(3,4),dim=1)
    cfg=asdict(Settings());profile=ProfileKL(anchors,Settings());clean=anchors[:1].expand(3,-1).clone();features=views[0]
    def values(a,b,p):
        terms=profile.terms(a,b,p);return {**scalars(terms),'per_view':[scalars({k:v[j:j+1] for k,v in terms.items()}) for j in range(3)]}
    gv=values(features,clean,texts[1:2].expand(3,-1));rec={'phase':'joint','global':gv['loss'],'local':0.,'retain':0.,'total':gv['loss'],'finite_gradients':True,'G_gradient_l2':.5,'cumulative_linf':.01,
        'attraction':{'phase':'joint','image_feature':features[0].tolist(),'view_features':features.tolist(),'clean_view_features':clean.tolist(),'view_linf':[.01,.01,.01],**gv}}
    ident={'original_caption_sha256':'a'*64,'context_caption_sha256':'b'*64,'selected_g_ids':[],'scores':[],'source_space_sha256':'c'*64,'anchor_mapping_sha256':'d'*64}
    g={'row_id':'r','G_before':'G0','G_after':'G1','K':0,'optimizer_steps':1,'records':[rec],'weights':{'lambda_global':1.,'lambda_local':1.,'lambda_retain':1.},
        'gradient_observations':[{'record':{k:v for k,v in rec.items() if k!='attraction'},'gradient_sha256':'f'*64}],
        'attraction':{'teacher_H_state':'H0','anchor_sha256':tree_sha(anchors),'candidate_count':8,'settings':cfg,'condition_identity':digest(ident),'partner_text_feature':texts[1].tolist(),'view_recipe':recipe(ViewSettings(),'r',1,384,128),'clean_view_sha256':'e'*64}}
    hv=[values(x[None].expand(3,-1),texts[:1].expand(3,-1),features) for x in texts[1:]]
    h={'row_id':'r','optimizer_steps_this_step':1,'IRRA_text_samples':3,'gradient_norm':.1,'H_state_sha256_before':'H0-legacy','H_state_sha256_after':'H1-legacy','sequence_log_prob':-3.,
        'attraction':{'teacher_G_state':'G1','settings':cfg,'anchor_sha256':tree_sha(anchors),'candidate_count':8,'R_sim_optimized':False,'R_rank_optimized':False,'text_features':texts.tolist(),'partner_image_feature':features[0].tolist(),'partner_view_features':features.tolist(),'view_bank_sha256':tree_sha(views),'sampled':hv[0],'greedy':hv[1]}}
    for name,v in zip(('sampled','greedy'),hv):h[name+'_reward']={'total':v['reward'],**{k:v[k] for k in ('pull_kl','push_kl','baseline_pull_kl','pull_gain')}}
    h['advantage']=hv[0]['reward']-hv[1]['reward'];h['SCST_loss']=-h['advantage']*h['sequence_log_prob']
    ids=['r']+['other-'+str(i) for i in range(7)]
    return {'round':1,'G_before':'G0','G_after':'G1','condition_H_state':'H0','G':[g],'H':[h],'conditions':[{'row':'r','identity':ident}],'g_orders':[['r']],'h_orders':[['r']],'bank':{'ordered_train_ids':ids},'bank_view_inputs':bindings(ids,views)},cfg,anchors,bank,views

def test_independent_three_view_FP64_audit():
    log,cfg,a,b,v=fixture_ledger();report=audit_attraction(log,cfg,None,None,a,b,v,asdict(ViewSettings()));assert report['G_view_terms_independently_recomputed']==3 and report['H_view_terms_independently_recomputed']==6

@pytest.mark.parametrize('case',['G_loss','G_feature','G_clean','G_teacher','H_reward','H_partner','H_teacher','H_SCST','view_budget','recipe','view_bank_binding'])
def test_three_view_audit_rejects_corruption(case):
    log,cfg,a,b,v=fixture_ledger();g=log['G'][0];h=log['H'][0]
    if case=='G_loss':g['records'][0]['attraction']['per_view'][2]['loss']+=.1
    elif case=='G_feature':g['records'][0]['attraction']['view_features'][1][0]+=.2
    elif case=='G_clean':g['records'][0]['attraction']['clean_view_features'][2][0]+=.2
    elif case=='G_teacher':g['attraction']['teacher_H_state']='stale'
    elif case=='H_reward':h['sampled_reward']['total']+=.1
    elif case=='H_partner':h['attraction']['partner_view_features'][2][0]+=.2
    elif case=='H_teacher':h['attraction']['teacher_G_state']='stale'
    elif case=='H_SCST':h['SCST_loss']+=.1
    elif case=='view_budget':g['records'][0]['attraction']['view_linf'][2]=8/255+1e-9
    elif case=='recipe':g['attraction']['view_recipe']['global'][0]['kind']='vflip'
    elif case=='view_bank_binding':log['bank_view_inputs'][0]['feature_sha256']='stale'
    with pytest.raises(ArtifactError):audit_attraction(log,cfg,None,None,a,b,v,asdict(ViewSettings()))

def test_attraction_Joint_engine_fixture(tmp_path):
    # 复用既有原生Engine数值fixture，注入本版本的评分入口。
    spec=importlib.util.spec_from_file_location('attraction_native_joint_fixture',ROOT/'tests/framework/test_gh_alternating_v2.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    from tbps.alternating import fresh_evaluation
    from tbps.augmented_attraction.evaluation import joint_score
    from unittest.mock import patch
    with patch.object(fresh_evaluation,'joint_score',joint_score):fixture.test_Joint_only_native_engine_fixture(tmp_path)

def test_full_plan_contains_six_predeclared_pairs_and_no_retry():
    import ast
    source=(ROOT/'operations/gh_pushpull_aug_v1_supervisor.py').read_text();ast.parse(source)
    assert 'for n in (10,20,30,40,50,60)' in source
    assert "expected=3 if n<60 else 0" in source and "FAIL_STOPPED_NO_RETRY" in source
    assert "audit_train_" in source and "audit_test_" in source


def test_final_report_six_results_SVG_and_current_state(tmp_path,monkeypatch):
    from tbps.augmented_attraction import report
    root=tmp_path/'repo';base=tmp_path/'tbps-framework-report-fixture';cfg=tmp_path/'configuration.json';cfg.write_text('{}')
    (root/'docs').mkdir(parents=True);(root/'docs/PROJECT_STATE.md').write_text('<!-- GH_AUG_CURRENT_BEGIN -->pending<!-- GH_AUG_CURRENT_END -->')
    f={'test_rounds':[10,20,30,40,50,60],'output_base':str(base),'test_output':str(base/'test'),
        'audits':{'train':str(base/'audit')},'attraction':asdict(Settings()),'augmentation':asdict(ViewSettings()),
        'report_runtime':json.loads((ROOT/'configs/framework/gh_pushpull_aug_v1.server.json').read_text())['report_runtime']}
    train=base/'audit/round_60/completed';train.mkdir(parents=True)
    (train/'report.json').write_bytes(json_bytes({'status':'GH_AUG_TRAIN_CPU_AUDIT_PASS','fresh_initialization':{'CPU_fixture':True}}))
    for n in f['test_rounds']:
        d=base/'test'/f'round_{n:02d}';(d/'cpu_audit/completed').mkdir(parents=True);(d/'completed').mkdir()
        metrics={'R@1':float(80-n),'R@5':float(90-n),'R@10':float(95-n),'mAP':float(70-n)}
        (d/'completed/summary.json').write_bytes(json_bytes({'metrics':metrics}))
        (d/'cpu_audit/completed/report.json').write_bytes(json_bytes({'status':'GH_AUG_JOINT_TEST_CPU_AUDIT_PASS',
            'scope':'official_test','metrics':metrics,'round':n,'gallery':3074,'query':6156,'CPU_fixture':True}))
    monkeypatch.setattr(report,'load',lambda *a:(f,{}));monkeypatch.setattr(report,'__file__',str(root/'src/tbps/attraction/report.py'))
    monkeypatch.setenv('MPLCONFIGDIR',str(tmp_path/'mpl'))
    result=report.publish(cfg);out=root/'docs/framework/results/gh_pushpull_aug_v1'
    assert len(result['rows'])==6 and len(list(out.glob('*.svg')))==2 and len(list(out.glob('*.png')))==2
    for svg in out.glob('*.svg'):assert all(line==line.rstrip() for line in svg.read_text().splitlines())
    assert 'GH_PUSHPULL_AUG_V1.md' in (out/'REPORT.md').read_text()
    assert '六Joint Test完成' in (root/'docs/PROJECT_STATE.md').read_text()


def test_rolling_journal_retains_all_view_bank_bytes_before_retirement(tmp_path):
    from tbps.augmented_attraction.store import FreshJournal,read_ledger
    from tbps.alternating.formal_state import tensor_bytes
    base=tmp_path/'tbps-framework-gh-pushpull-aug-v1-20261111';run=base/'train';j=FreshJournal(run,'f'*64,base)
    state={'G':{'v':torch.ones(1)},'H':{'v':torch.ones(1)},'parents':{'source':'fixed'},'cursor':{'round':0}};j.initialize({'fixture':True},state)
    for i in (1,2):
        j.begin(i,state);state=copy.deepcopy(state);state['cursor']['round']=i;state['G']['v'].add_(1)
        bank=ArtifactStore(run/'B'/f'bank_{i}',new_base=base);views=torch.randn(2,3,4)
        bank.publish('completed',{'features.pt':tensor_bytes(views[:,0].clone()),'view_features.pt':tensor_bytes(views),
            'view_inputs.json':json_bytes({'entries':[{'row':'r','fixture':True}]}),'identity.json':json_bytes({'round':i})},{'G':tree_sha(state['G'])})
        io=ArtifactStore(run/'round_io'/f'round_{i:04d}',new_base=base);action={'original':'caption','adversarial':'caption','K':0,'edits':[],'native_action':{'fixed_NOOP':True}}
        io.publish('rollout_0000',{'action.json':json_bytes(action)},{})
        ledger={'round':i,'G':[],'H':[],'conditions':[{'row':'r'}],'g_orders':[['r']],'bank':{'g_state_sha256':tree_sha(state['G'])}}
        ref=bank.root/'completed/manifest.json';j.commit(i,state,ledger,FileRef(str(ref),sha_file(ref)))
    first=read_ledger(run/'ledgers/round_0001/ledger.json')
    assert not (run/'B/bank_1').exists() and list(first['conditions'])[0]['G_rollout']==action
    assert sha_file(run/'ledgers/round_0001/bank_view_features.pt')==first['bank_view_features_sha256']==first['bank_manifest_snapshot']['files_sha256']['view_features.pt']
    assert first['bank_view_inputs']==[{'row':'r','fixture':True}]
    restored,_=j.load_latest(state['parents']);assert tree_sha(restored)==tree_sha(state)


def test_latest_three_view_budget_recipe_replay_under_CPU_no_model_guard(tmp_path):
    from tbps.augmented_attraction.audit_math import audit_view_bank
    from tbps.alternating.formal_state import tensor_bytes
    from tbps.alternating.parents import cpu_restore_only,no_optimizer_steps
    base=tmp_path/'tbps-framework-fixture';root=base/'train';bank=ArtifactStore(root/'B/bank_1',new_base=base);features=[];inputs=[]
    for i,row in enumerate(('r1','r2')):
        original=torch.rand(1,3,384,128);plan=recipe(ViewSettings(),row,1,384,128);clean=apply_views(original,plan);v=F.normalize(torch.randn(3,4),dim=1)
        payload={'original':original,'adversarial':original.clone(),'effective_delta':torch.zeros_like(original),'feature':v[:1].clone(),
            'augmented_original':clean[1:].clone(),'augmented_adversarial':clean[1:].clone(),'augmented_effective_delta':torch.zeros_like(clean[1:]),'view_features':v}
        bank.publish(f'image_{i:04d}',{'data.pt':tensor_bytes(payload),'metadata.json':json_bytes({'view_recipe':plan})},{})
        features.append(v);inputs.append({'row':row,'input_sha256':tree_sha(clean),'recipe_sha256':digest(plan),'feature_sha256':tree_sha(v),'view_linf':[0.,0.,0.]})
    bank.publish('completed',{'view_features.pt':tensor_bytes(torch.stack(features))},{})
    allowed=list(bank.root.glob('image_*/data.pt'))+[bank.root/'completed/view_features.pt']
    with cpu_restore_only(allowed),no_optimizer_steps(),torch.no_grad():
        report=audit_view_bank(root,{'round':1,'bank_view_inputs':inputs},{'prototype':{'train_ids':['r1','r2']},'output_base':str(base)})
    assert report['total_view_images']==6 and report['exact_recipe_pixel_replay'] and report['image_budget_tolerance']==0
